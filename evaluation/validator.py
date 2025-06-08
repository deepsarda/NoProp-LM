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
    Performs end-to-end validation on the entire denoising chain.

    Args:
        model: The NoProp LanguageModel.
        val_loader: The DataLoader for the validation set.
        criterion: The loss function (MSELoss with reduction='none').
        device: The device to run validation on.

    Returns:
        The average validation loss.
    """
    print("\n--- Running End-to-End Validation ---")
    model.to(device)  # Move the entire model to the device for validation
    model.eval()

    total_loss = 0.0

    progress_bar = tqdm(val_loader, desc="Validating", leave=False, dynamic_ncols=True)

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        if input_ids.size(0) == 0:
            continue

        with torch.no_grad():
            clean_input_embeddings = model.get_clean_embedding(input_ids)
            clean_label_embeddings = model.get_clean_embedding(
                labels.masked_fill(labels == C.IGNORE_INDEX, 0)
            )

        # Start with pure Gaussian noise of the same shape as the target labels
        current_denoised_embeddings = torch.randn_like(clean_label_embeddings)

        # Run the full denoising chain from the most noisy to the least noisy block
        for block in model.denoising_blocks:
            # The input to each block is the clean context + the output from the previous block
            block_input_embeddings = (
                clean_input_embeddings + current_denoised_embeddings
            )
            current_denoised_embeddings = block(block_input_embeddings)

        # After all blocks, `current_denoised_embeddings` is the final prediction
        final_predicted_embeddings = current_denoised_embeddings

        # Calculate the final, unweighted MSE loss
        loss = criterion(final_predicted_embeddings, clean_label_embeddings)

        # Mask out padded positions
        loss_mask = (labels != C.IGNORE_INDEX).unsqueeze(-1)
        masked_loss = loss * loss_mask

        # Average the loss for the batch and add to the total
        batch_loss = masked_loss.sum() / (loss_mask.sum() + 1e-8)
        total_loss += batch_loss.item()

        progress_bar.set_postfix(Val_Loss=f"{batch_loss.item():.4f}")

    # After validation, move the model back to the training configuration (blocks on CPU)
    model.to("cpu")
    model.embedding_table.to(device)
    model.train()  # Set model back to training mode

    return total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
