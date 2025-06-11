import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

import config as C
from modeling.language_model import LanguageModel
from utilities.helpers import get_alpha_squared_from_cosine_schedule


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
    Performs end-to-end validation of the NoProp-LM model using a
    DDIM-style sampler.

    This function evaluates the model by running the complete denoising chain.
    For each batch in the validation loader:
    1. Clean, normalized embeddings for inputs (src) and labels (tgt) are obtained.
    2. An initial noise tensor is created, matching the shape of the target embeddings.
    3. The noisy tensor is iteratively passed through all denoising blocks. In each step:
        a. Karras et al. pre-conditioning factors (c_in, c_out, c_skip) are calculated.
        b. The current noisy embedding is scaled by c_in and passed to the block.
        c. The block's output is combined with a skip connection to predict the clean embedding.
        d. This high-quality prediction is used to deterministically sample the next,
           less noisy state using the DDIM update rule.
    4. The final output from the last block is taken as the predicted embeddings.
    5. A masked MSE loss is calculated between predicted and clean label embeddings.
    6. The average loss across all batches is computed and returned.
    """
    print("\n--- Running End-to-End Validation ---")
    model.to(device)  # Move the entire model (including all blocks) to the device
    model.eval()  # Set the model to evaluation mode

    # Pre-compute the entire alpha_squared schedule for the DDIM sampler
    T = len(model.denoising_blocks)
    alphas_squared_schedule = torch.tensor(
        [get_alpha_squared_from_cosine_schedule(i, config) for i in range(T)],
        device=device,
        dtype=torch.float32,
    ).view(
        -1, 1, 1
    )  # Reshape for broadcasting

    total_loss = 0.0  # Accumulator for the total overall loss over all batches
    block_losses_accumulators = [0.0] * T

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
            src_padding_mask = input_ids == tokenizer.pad_token_id
            tgt_padding_mask = labels == C.IGNORE_INDEX

        # Start the denoising chain with pure Gaussian noise
        current_embeddings = torch.randn_like(clean_label_embeddings)
        final_predicted_embeddings = None

        # Run the full denoising chain from T-1 (max noise) down to 0 (min noise)
        for block_idx, block in enumerate(model.denoising_blocks):
            # Get the noise level parameters for the current step (t)
            alpha_sq_t = alphas_squared_schedule[block_idx]
            sigma_t = torch.sqrt(1.0 - alpha_sq_t)

            # --- Karras Pre-conditioning (Identical to Training) ---
            sigma_sq = sigma_t**2
            c_skip = 1.0 / (sigma_sq + 1.0)
            c_out = sigma_t * torch.sqrt(c_skip)
            c_in = torch.sqrt(c_skip)

            # Scale the input and pass it through the denoising block
            network_input = current_embeddings * c_in
            network_output = block(
                src_embeds=clean_input_embeddings,
                tgt_embeds=network_input,
                src_padding_mask=src_padding_mask,
                tgt_padding_mask=tgt_padding_mask,
            )
            # Combine the skip connection and network output to predict the clean embedding
            predicted_clean_embeddings = (current_embeddings * c_skip) + (
                network_output * c_out
            )
            # Clamp for stability
            predicted_clean_embeddings.clamp_(-1.0, 1.0)

            # Loss calculation for this block
            loss_per_block = criterion(
                predicted_clean_embeddings, clean_label_embeddings
            )
            loss_mask = (labels != C.IGNORE_INDEX).unsqueeze(-1).float()
            masked_loss_per_block = loss_per_block * loss_mask
            batch_loss_per_block = masked_loss_per_block.sum() / (
                loss_mask.sum() + 1e-8
            )
            block_losses_accumulators[block_idx] += batch_loss_per_block.item()

            # If not the last block, calculate the next, less noisy state.
            if block_idx < T - 1:
                # Get the noise level for the *next* step (t-1)
                alpha_sq_t_prev = alphas_squared_schedule[block_idx + 1]

                # 1. Estimate the noise direction from our x0 prediction
                pred_noise_direction = (
                    current_embeddings
                    - torch.sqrt(alpha_sq_t) * predicted_clean_embeddings
                ) / torch.sqrt(1.0 - alpha_sq_t)

                # 2. Use the DDIM formula to step to the next state
                current_embeddings = (
                    torch.sqrt(alpha_sq_t_prev) * predicted_clean_embeddings
                    + torch.sqrt(1.0 - alpha_sq_t_prev) * pred_noise_direction
                )
            else:
                # This is the final prediction from the last block
                final_predicted_embeddings = predicted_clean_embeddings

        # Overall Loss calculation for the batch
        loss = criterion(final_predicted_embeddings, clean_label_embeddings)
        loss_mask = (labels != C.IGNORE_INDEX).unsqueeze(-1).float()
        masked_loss = loss * loss_mask
        batch_loss = masked_loss.sum() / (loss_mask.sum() + 1e-8)
        total_loss += batch_loss.item()

        progress_bar.set_postfix(Val_Loss=f"{batch_loss.item():.4f}")

    # After validation, restore model state for training
    model.to("cpu")
    model.embedding_table.to(device)
    model.train()

    num_batches = len(val_loader) if len(val_loader) > 0 else 1
    avg_overall_loss = total_loss / num_batches
    avg_block_losses = [l / num_batches for l in block_losses_accumulators]

    return {"overall_loss": avg_overall_loss, "block_losses": avg_block_losses}
