import math
from typing import Optional

import torch
import torch.nn as nn
import wandb
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as C
from modeling.language_model import LanguageModel


def get_alpha_squared_from_cosine_schedule(block_idx: int, config: C) -> float:
    """
    Calculates the alpha_squared value for a given block index based on a
    cosine noise schedule. alpha_squared represents the proportion of signal
    (clean embedding) in the noisy embedding at a particular denoising step.

    The schedule maps block indices (from 0 to NUM_DENOISING_BLOCKS-1)
    to a value between nearly 1 (mostly signal, for later blocks/less noise)
    and nearly 0 (mostly noise, for earlier blocks/more noise).

    If block_idx < 0 (e.g., for the initial clean state), alpha_squared is 1.0.

    Args:
        block_idx (int): The index of the current DenoisingBlock being trained.
                         Ranges from 0 to NUM_DENOISING_BLOCKS - 1.
        config (C): The global configuration object, used to get NUM_DENOISING_BLOCKS.

    Returns:
        float: The calculated alpha_squared value for the given block index.
               This value is between 0 and 1.
    """
    if (
        block_idx < 0
    ):  # Corresponds to the initial clean state (t=0 in paper's notation)
        return 1.0

    total_blocks = config.NUM_DENOISING_BLOCKS
    # `t_paper` corresponds to t in the original paper's formulation (1 to T)
    # Our block_idx is 0 to T-1. So, t_paper = block_idx + 1.
    t_paper = block_idx + 1

    # Ratio of current step t to total steps T
    t_ratio = t_paper / total_blocks

    # Cosine schedule as defined in some diffusion model papers: cos^2(t/T * pi/2)
    # This results in alpha_squared decreasing from 1 (at t=0, though here t_paper starts at 1)
    # towards 0 (at t=T).
    return math.cos(t_ratio * math.pi / 2) ** 2


def run_training_loop(
    config: C,
    model: LanguageModel,
    tokenizer,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],  # val_loader can be None
    device: torch.device,
    generator_fn,
    validation_fn,
    wandb_run,
):
    """
    Runs the main NoProp-LM training loop (Phase 2).

    This function iterates over epochs and, within each epoch, trains each
    DenoisingBlock of the `model` sequentially. For each block:
    1. An optimizer is set up for its parameters.
    2. The noise level (alpha_squared) is calculated to determine the noise injection amount.
    3. For each batch in `train_loader`:
        a. Clean embeddings for inputs (src) and labels (tgt) are obtained.
        b. Noise is injected into the clean label embeddings to create the decoder's input.
        c. Padding masks are created for both src and tgt sequences.
        d. The current DenoisingBlock's encoder processes the clean `src` embeddings, and its
           decoder processes the noisy `tgt` embeddings, using cross-attention to condition
           on the encoder's output.
        e. A masked Mean Squared Error (MSE) loss is calculated between the block's
           predictions and the original clean label embeddings.
        f. Gradients are computed, and the block's parameters are updated.
    4. After training, the block is moved to the CPU to free GPU memory.
    5. Periodically, text samples are generated and validation is performed.

    Args:
        config (C): Global configuration object.
        model (LanguageModel): The NoProp-LM model instance.
        tokenizer: The tokenizer for encoding/decoding text.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (Optional[DataLoader]): DataLoader for the validation dataset. Can be None.
        device (torch.device): The primary device for training (e.g., 'cuda').
        generator_fn: Function to call for generating text samples.
        validation_fn: Function to call for running validation.
        wandb_run: Initialized Weights & Biases run object for logging.
    """
    print("\n--- PHASE 2: Starting NoProp Training Loop ---")
    # Criterion: Mean Squared Error loss, calculated per element (reduction='none').
    # This allows for custom masking and weighting later.
    criterion = nn.MSELoss(reduction="none")
    # Gradient scaler for mixed-precision training (FP16), if enabled by config.
    scaler = torch.amp.GradScaler(enabled=config.FP16_ENABLED)
    global_step = 0  # Counter for total training steps across all epochs and blocks.
    # W&B Table for logging generated text samples.
    gen_table = wandb.Table(columns=["Step", "Prompt", "Generated Text"])

    for epoch in range(config.EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{config.EPOCHS} ---")
        epoch_log = (
            {}
        )  # Dictionary to store metrics for logging at the end of the epoch.

        # Iterate through each DenoisingBlock in the model.
        # Each block is trained sequentially for one pass over the training data.
        for block_idx in range(config.NUM_DENOISING_BLOCKS):
            # Get the current DenoisingBlock and move it to the training device.
            current_block = model.denoising_blocks[block_idx].to(device)
            current_block.train()  # Set the block to training mode.

            # Optimizer for the current block's parameters.
            # Each block has its own optimizer, as they are trained independently.
            optimizer = AdamW(
                current_block.parameters(),
                lr=config.LEARNING_RATE,
                betas=config.ADAM_BETAS,
                eps=config.ADAM_EPS,
            )
            # Progress bar for the current block's training.
            progress_bar = tqdm(
                train_loader,
                desc=f"E{epoch+1} | Training Block {block_idx+1}/{config.NUM_DENOISING_BLOCKS}",
                leave=False,  # Remove progress bar when done
                dynamic_ncols=True,  # Adjust to terminal width
            )
            total_loss_block = 0.0  # Accumulator for the current block's average loss.

            # Determine noise level and loss weighting for the current block.
            # `alpha_sq_current` is the signal variance (cosine schedule).
            alpha_sq_current = get_alpha_squared_from_cosine_schedule(block_idx, config)

            for batch_num, batch in enumerate(progress_bar):
                # Move batch data (input_ids, labels) to the training device.
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

                # Skip empty batches (can happen with multi-worker DataLoader).
                if input_ids.size(0) == 0:
                    continue

                optimizer.zero_grad()  # Clear gradients from previous step.

                # --- Prepare embeddings and inject noise (all within torch.no_grad()) ---
                with torch.no_grad():  # No gradients needed for these operations.
                    # Get clean (original) embeddings for input_ids from the frozen embedding table.
                    clean_input_embeddings = model.get_clean_embedding(input_ids)

                    # Prepare labels for embedding:
                    # Clone labels to avoid modifying the original tensor.
                    # Replace IGNORE_INDEX (-100) with a valid token ID (e.g., pad_token_id)
                    # because nn.Embedding cannot handle negative indices.
                    # These positions will be masked out in the loss calculation anyway.
                    valid_labels = labels.clone()
                    valid_labels[labels == config.IGNORE_INDEX] = tokenizer.pad_token_id
                    clean_label_embeddings = model.get_clean_embedding(valid_labels)

                    # Calculate signal and noise components for noising the label embeddings.
                    signal_strength = math.sqrt(alpha_sq_current)
                    noise_strength = math.sqrt(1.0 - alpha_sq_current)
                    # Generate Gaussian noise with the same shape as clean_label_embeddings.
                    noise = torch.randn_like(clean_label_embeddings)
                    # Create noisy label embeddings: `sqrt(alpha^2)*clean + sqrt(1-alpha^2)*noise`.
                    noisy_label_embeddings = (
                        clean_label_embeddings * signal_strength
                    ) + (noise * noise_strength)

                    # Source padding mask for the encoder. True where input_ids is a pad token.
                    src_padding_mask = input_ids == tokenizer.pad_token_id
                    # Target padding mask for the decoder. True where labels is the ignore_index.
                    tgt_padding_mask = labels == config.IGNORE_INDEX

                # --- Forward pass, loss calculation, and backward pass ---
                # Use Automatic Mixed Precision (AMP) if enabled.
                with torch.amp.autocast(
                    device_type=device.type, enabled=config.FP16_ENABLED
                ):
                    # Forward pass through the current DenoisingBlock.
                    # It predicts the denoised version of the label embeddings.
                    predicted_denoised_embeddings = current_block(
                        src_embeds=clean_input_embeddings,
                        tgt_embeds=noisy_label_embeddings,
                        src_padding_mask=src_padding_mask,
                        tgt_padding_mask=tgt_padding_mask,
                    )

                    # Calculate unweighted MSE loss between prediction and original clean labels.
                    unweighted_loss = criterion(
                        predicted_denoised_embeddings, clean_label_embeddings
                    )

                    # Create a mask to ignore padded positions in the loss.
                    # `labels != config.IGNORE_INDEX` is True for non-padded tokens.
                    # `unsqueeze(-1)` makes it broadcastable with the embedding dimension.
                    loss_mask = (labels != config.IGNORE_INDEX).unsqueeze(-1).float()
                    # Apply mask: zero out loss for padded positions.
                    masked_loss = unweighted_loss * loss_mask
                    # Calculate final batch loss: sum of weighted losses divided by number of non-padded tokens.
                    # Add epsilon to denominator to prevent division by zero.
                    loss = masked_loss.sum() / (loss_mask.sum() + 1e-8)

                # Backpropagation using the gradient scaler.
                scaler.scale(loss).backward()
                # Gradient clipping, if configured.
                if config.GRAD_CLIP_VALUE > 0:
                    scaler.unscale_(optimizer)  # Unscale gradients before clipping.
                    torch.nn.utils.clip_grad_norm_(
                        current_block.parameters(), config.GRAD_CLIP_VALUE
                    )
                # Optimizer step.
                scaler.step(optimizer)
                # Update scaler for next iteration.
                scaler.update()

                # Log training loss to W&B.
                if wandb_run:
                    wandb_run.log(
                        {
                            "train/loss": loss.item(),
                            "train/block_idx": block_idx,
                            "train/epoch": epoch,
                            "global_step": global_step,
                        }
                    )
                total_loss_block += loss.item()  # Accumulate loss for averaging.
                progress_bar.set_postfix(
                    Loss=f"{loss.item():.4f}", BLOCK_IDX=f"{block_idx}"
                )

                global_step += 1  # Increment global training step count.

                # --- Periodic Text Generation ---
                if (
                    wandb_run
                    and config.LOG_GENERATION_EVERY_N_STEPS > 0
                    and global_step % config.LOG_GENERATION_EVERY_N_STEPS == 0
                ):
                    print(
                        f"\n--- Generating Samples at Step {global_step} (Block {block_idx}) ---"
                    )
                    # Call the provided generator_fn.
                    # Model is passed as is; generator_fn handles moving blocks to device.
                    generator_fn(
                        model=model,  # Pass the main LanguageModel
                        tokenizer=tokenizer,
                        device=device,  # Device for generation
                        config=config,
                        prompts=config.TEST_PROMPTS_FOR_LOGGING[
                            : config.NUM_GENERATION_EXAMPLES_TO_LOG  # Select configured number of prompts
                        ],
                        step=global_step,
                        wandb_table=gen_table,  # Log to this W&B table
                    )
                    # Log the W&B table with generated samples.
                    # Note: Logging a growing table repeatedly can be resource-intensive for W&B.
                    # Consider alternative logging strategies for very long runs.
                    wandb_run.log({f"generations/samples": gen_table})
                    # Ensure the current_block is back on the training device and in train mode
                    # as generator_fn might change its state or device.
                    current_block = current_block.to(device)
                    current_block.train()

            # Calculate average loss for the current block over the epoch.
            avg_loss_block = (
                total_loss_block / len(train_loader) if len(train_loader) > 0 else 0
            )
            print(
                f"  Block {block_idx+1} trained. Avg Weighted Loss: {avg_loss_block:.6f}"
            )
            epoch_log[f"train_loss/block_{block_idx}"] = (
                avg_loss_block  # Log per-block average loss.
            )

            # Move the trained block to CPU to save GPU memory for the next block.
            current_block.to("cpu")
            del optimizer  # Delete optimizer for this block
            if device.type == "cuda":
                torch.cuda.empty_cache()  # Clear CUDA cache.

        # --- End of Epoch Validation ---
        if val_loader:
            # Perform end-to-end validation using the provided validation_fn.
            # The validation_fn is responsible for moving model parts to the device.
            avg_val_loss = validation_fn(
                model, val_loader, criterion, device, tokenizer
            )  # Note: criterion here is MSELoss
            print(
                f"\nEpoch {epoch + 1} | End-to-End Validation MSE Loss: {avg_val_loss:.6f}"
            )
            epoch_log["validation/epoch_loss"] = (
                avg_val_loss  # Log overall validation loss for the epoch.
            )

        # Log all accumulated epoch metrics to W&B.
        if wandb_run:
            wandb_run.log(epoch_log)
