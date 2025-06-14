import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as C
from modeling.language_model import LanguageModel


@torch.no_grad()
def run_validation(
    model: LanguageModel,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Performs end-to-end validation of the NoProp-LM model on a given validation set.

    This function evaluates the model by running the complete denoising chain.
    For each batch in the validation loader:
    1. Clean embeddings for input and label sequences are obtained.
       Padding tokens in labels are mapped to a valid token (e.g., 0) before embedding
       as they are masked out during loss calculation anyway.
    2. An initial noise tensor is created, matching the shape of the target label embeddings.
    3. This noise is iteratively passed through all denoising blocks of the model.
       The input to each block is the sum of the clean input embeddings (context) and
       the (partially) denoised embeddings from the previous block.
    4. The final output from the last denoising block is taken as the predicted embeddings.
    5. The loss (typically MSE) is calculated between these predicted embeddings and the
       clean label embeddings.
    6. The loss is masked to exclude positions corresponding to padding tokens in the labels.
    7. The average loss across all batches is computed and returned.

    The model is moved to the specified `device` for validation and set to `eval()` mode.
    After validation, it's moved back to "cpu" (except for the embedding table, which
    remains on `device`) and set back to `train()` mode.

    Args:
        model (LanguageModel): The NoProp-LM model to be validated.
        val_loader (DataLoader): DataLoader providing the validation data. Each batch
            should be a dictionary containing 'input_ids' and 'labels'.
        criterion (nn.Module): The loss function (e.g., `nn.MSELoss(reduction='none')`)
            used to compare predicted embeddings with target embeddings.
        device (torch.device): The device (e.g., 'cuda', 'cpu') on which to perform
            validation.

    Returns:
        float: The average validation loss over the entire validation dataset.
               Returns 0.0 if the validation loader is empty.
    """
    print("\n--- Running End-to-End Validation ---")
    model.to(device)  # Move the entire model (including all blocks) to the device
    model.eval()      # Set the model to evaluation mode

    total_loss = 0.0  # Accumulator for the total loss over all batches

    # Initialize tqdm progress bar for visual feedback
    progress_bar = tqdm(val_loader, desc="Validating", leave=False, dynamic_ncols=True)

    for batch in progress_bar:
        # Move input_ids and labels to the specified device
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # Skip batch if it's empty (e.g., if last batch has fewer items than workers)
        if input_ids.size(0) == 0:
            continue

        # Get clean (noise-free) embeddings for inputs and labels
        # No gradients are needed here as we are in eval mode.
        with torch.no_grad():
            clean_input_embeddings = model.get_clean_embedding(input_ids)
            # For label embeddings, replace IGNORE_INDEX with a valid token (e.g., 0)
            # before embedding, as these positions will be masked out in the loss later.
            # This prevents errors if IGNORE_INDEX is out of vocab range.
            clean_label_embeddings = model.get_clean_embedding(
                labels.masked_fill(labels == C.IGNORE_INDEX, 0) # Use 0 or any valid token_id
            )

        # Initialize the starting point for the denoising chain: pure Gaussian noise.
        # This tensor has the same shape as the target (clean_label_embeddings).
        current_denoised_embeddings = torch.randn_like(clean_label_embeddings)

        # Run the full denoising chain: pass through each denoising block sequentially.
        # The blocks are ordered from most noisy (largest sigma) to least noisy (smallest sigma).
        for block_idx, block in enumerate(model.denoising_blocks):
            # The input to each denoising block is the sum of:
            #   1. The clean embeddings of the input sequence (context).
            #   2. The (partially) denoised embeddings from the output of the previous block.
            # This sum represents the noisy input at the current step of the denoising process.
            block_input_embeddings = (
                clean_input_embeddings + current_denoised_embeddings
            )
            # Pass the noisy input through the current denoising block.
            # The block predicts a "cleaner" version of the embeddings.
            current_denoised_embeddings = block(block_input_embeddings)

        # After iterating through all blocks, `current_denoised_embeddings` holds the
        # final predicted embeddings for the label sequence.
        final_predicted_embeddings = current_denoised_embeddings

        # Calculate the loss (e.g., MSE) between the final predicted embeddings
        # and the clean target label embeddings. `reduction='none'` keeps per-element losses.
        loss = criterion(final_predicted_embeddings, clean_label_embeddings)

        # Create a mask to ignore padded positions in the loss calculation.
        # `labels != C.IGNORE_INDEX` creates a boolean mask (True for non-padding).
        # `unsqueeze(-1)` expands the mask to match the embedding dimension.
        loss_mask = (labels != C.IGNORE_INDEX).unsqueeze(-1).float() # Ensure float for multiplication

        # Apply the mask: multiply element-wise loss by the mask.
        # Padded positions will have their loss component zeroed out.
        masked_loss = loss * loss_mask

        # Calculate the average loss for the current batch:
        # Sum of masked losses divided by the number of non-padded elements.
        # Add a small epsilon (1e-8) to prevent division by zero if a batch is all padding.
        batch_loss = masked_loss.sum() / (loss_mask.sum() + 1e-8)
        total_loss += batch_loss.item() # Accumulate batch loss

        # Update the progress bar with the current batch's validation loss
        progress_bar.set_postfix(Val_Loss=f"{batch_loss.item():.4f}")

    # After validation, restore model state for training:
    # Move denoising blocks back to CPU (standard training setup for this model)
    model.to("cpu")
    # Ensure the embedding table remains on the designated device (e.g., GPU)
    model.embedding_table.to(device)
    model.train()  # Set the model back to training mode (enables dropout, etc.)

    # Calculate the average validation loss over all batches.
    # Handle case where val_loader might be empty to prevent division by zero.
    return total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
