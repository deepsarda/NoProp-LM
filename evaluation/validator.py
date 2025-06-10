import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

import config as C
from modeling.language_model import LanguageModel
from utilities.helpers import get_ddpm_schedule


@torch.no_grad()
def run_validation(
    model: LanguageModel,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    tokenizer: PreTrainedTokenizerBase,
    config: C,
) -> dict:
    """
    Performs end-to-end validation of the NoProp-LM model.

    This function evaluates the model by running the complete denoising chain.
    For each batch in the validation loader:
    1. Clean embeddings for inputs (src) and labels (tgt) are obtained.
    2. An initial noise tensor is created, matching the shape of the target embeddings.
    3. Padding masks are created for both the src and tgt sequences.
    4. The noisy tensor is iteratively passed through all denoising blocks. In each step,
        the block's encoder processes the clean `src` embeddings, and its decoder
        processes the current (partially) denoised `tgt` embeddings.
    5. The final output from the last block is taken as the predicted embeddings.
    6. A masked MSE loss is calculated between predicted and clean label embeddings.
    7. The average loss across all batches is computed and returned.

    Args:
        model (LanguageModel): The NoProp-LM model to be validated.
        val_loader (DataLoader): DataLoader providing the validation data.
        criterion (nn.Module): The loss function (e.g., `nn.MSELoss(reduction='none')`).
        device (torch.device): The device on which to perform validation.
        tokenizer (PreTrainedTokenizerBase): The tokenizer, required to get the
            `pad_token_id` for creating the source padding mask.
        config (C): The global configuration object.

    Returns:
        dict: A dictionary containing 'overall_loss' (float) for the average
              validation loss, and 'block_losses' (list of floats) where each
              element is the average loss after that specific block in the chain.
    """
    print("\n--- Running End-to-End Validation ---")
    model.to(device)  # Move the entire model (including all blocks) to the device
    model.eval()  # Set the model to evaluation mode

    # Get the pre-computed noise schedule
    T = len(model.denoising_blocks)
    schedule = get_ddpm_schedule(T, config, device)
    posterior_mean_coef1 = schedule["posterior_mean_coef1"]
    posterior_mean_coef2 = schedule["posterior_mean_coef2"]
    posterior_variance = schedule["posterior_variance"]

    total_loss = 0.0  # Accumulator for the total overall loss over all batches
    # Accumulators for loss after each block
    block_losses_accumulators = [0.0] * len(model.denoising_blocks)

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
                labels.masked_fill(
                    labels == C.IGNORE_INDEX, 0
                )  # Use 0 or any valid token_id
            )

            src_padding_mask = input_ids == tokenizer.pad_token_id
            tgt_padding_mask = labels == C.IGNORE_INDEX

        # Initialize the starting point for the denoising chain: pure Gaussian noise.
        # This tensor has the same shape as the target (clean_label_embeddings).
        current_embeddings = torch.randn_like(clean_label_embeddings)
        final_predicted_embeddings = None
        # Run the full denoising chain: pass through each denoising block sequentially.
        # The blocks are ordered from most noisy (largest sigma) to least noisy (smallest sigma).
        for block_idx, block in enumerate(model.denoising_blocks):
            # Pass clean context to the encoder (src) and the current state of the
            # denoised embeddings to the decoder (tgt).
            # The block predicts a "cleaner" version of the embeddings.
            predicted_clean_embeddings = block(
                src_embeds=clean_input_embeddings,
                tgt_embeds=current_embeddings,
                src_padding_mask=src_padding_mask,
                tgt_padding_mask=tgt_padding_mask,
            )

            # Calculate loss after each block and accumulate it
            loss_per_block = criterion(
                predicted_clean_embeddings, clean_label_embeddings
            )
            loss_mask = (labels != C.IGNORE_INDEX).unsqueeze(-1).float()
            masked_loss_per_block = loss_per_block * loss_mask
            batch_loss_per_block = masked_loss_per_block.sum() / (
                loss_mask.sum() + 1e-8
            )
            block_losses_accumulators[block_idx] += batch_loss_per_block.item()

            if block_idx < len(model.denoising_blocks) - 1:
                posterior_mean = (
                    posterior_mean_coef1[block_idx].view(-1, 1, 1) * current_embeddings
                    + posterior_mean_coef2[block_idx].view(-1, 1, 1)
                    * predicted_clean_embeddings
                )
                noise = torch.randn_like(current_embeddings)
                current_embeddings = (
                    posterior_mean
                    + torch.sqrt(posterior_variance[block_idx]).view(-1, 1, 1) * noise
                )
            else:
                final_predicted_embeddings = predicted_clean_embeddings

        # Calculate the overall loss (e.g., MSE) between the final predicted embeddings
        # and the clean target label embeddings. `reduction='none'` keeps per-element losses.
        loss = criterion(final_predicted_embeddings, clean_label_embeddings)

        # Create a mask to ignore padded positions in the loss calculation.
        # `labels != C.IGNORE_INDEX` creates a boolean mask (True for non-padding).
        # `unsqueeze(-1)` expands the mask to match the embedding dimension.
        loss_mask = (
            (labels != C.IGNORE_INDEX).unsqueeze(-1).float()
        )  # Ensure float for multiplication

        # Apply the mask: multiply element-wise loss by the mask.
        # Padded positions will have their loss component zeroed out.
        masked_loss = loss * loss_mask

        # Calculate the average loss for the current batch:
        # Sum of masked losses divided by the number of non-padded elements.
        # Add a small epsilon (1e-8) to prevent division by zero if a batch is all padding.
        batch_loss = masked_loss.sum() / (loss_mask.sum() + 1e-8)
        total_loss += batch_loss.item()  # Accumulate overall batch loss

        # Update the progress bar with the current batch's overall validation loss
        progress_bar.set_postfix(Val_Loss=f"{batch_loss.item():.4f}")

    # After validation, restore model state for training:
    # Move denoising blocks back to CPU (standard training setup for this model)
    model.to("cpu")
    # Ensure the embedding table remains on the designated device (e.g., GPU)
    model.embedding_table.to(device)
    model.train()  # Set the model back to training mode (enables dropout, etc.)

    # Calculate the average validation loss over all batches.
    num_batches = (
        len(val_loader) if len(val_loader) > 0 else 1
    )  # Avoid division by zero
    avg_overall_loss = total_loss / num_batches

    # Calculate average loss for each block
    avg_block_losses = [l / num_batches for l in block_losses_accumulators]

    return {"overall_loss": avg_overall_loss, "block_losses": avg_block_losses}
