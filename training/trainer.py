import math

import torch
import torch.nn as nn
import wandb
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as C
from modeling.language_model import LanguageModel


def get_alpha_squared_from_cosine_schedule(block_idx: int, config: C) -> float:
    if block_idx < 0:
        return 1.0
    total_blocks = config.NUM_DENOISING_BLOCKS
    t_paper = block_idx + 1
    t_ratio = t_paper / total_blocks
    return math.cos(t_ratio * math.pi / 2) ** 2


def get_snr(alpha_squared: float) -> float:
    return alpha_squared / (1.0 - alpha_squared + 1e-8)


def run_training_loop(
    config: C,
    model: LanguageModel,
    tokenizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    generator_fn,
    validation_fn,
    wandb_run,
):
    print("\n--- PHASE 2: Starting NoProp Training Loop ---")
    criterion = nn.MSELoss(reduction="none")
    scaler = torch.amp.GradScaler(enabled=config.FP16_ENABLED)
    global_step = 0
    gen_table = wandb.Table(columns=["Step", "Prompt", "Generated Text"])
    for epoch in range(config.EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{config.EPOCHS} ---")
        epoch_log = {}
        for block_idx in range(config.NUM_DENOISING_BLOCKS):
            current_block = model.denoising_blocks[block_idx].to(device)
            current_block.train()

            optimizer = AdamW(
                current_block.parameters(),
                lr=config.LEARNING_RATE,
                betas=config.ADAM_BETAS,
                eps=config.ADAM_EPS,
            )
            progress_bar = tqdm(
                train_loader,
                desc=f"E{epoch+1} | Training Block {block_idx}",
                leave=False,
                dynamic_ncols=True,
            )
            total_loss = 0.0

            alpha_sq_current = get_alpha_squared_from_cosine_schedule(block_idx, config)
            snr_current = get_snr(alpha_sq_current)
            loss_weight = 1.0 + (1.0 / snr_current)

            for batch in progress_bar:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                if input_ids.size(0) == 0:
                    continue
                optimizer.zero_grad()

                with torch.no_grad():
                    clean_input_embeddings = model.get_clean_embedding(input_ids)
                    valid_labels = labels.clone()
                    valid_labels[labels == config.IGNORE_INDEX] = tokenizer.pad_token_id
                    clean_label_embeddings = model.get_clean_embedding(valid_labels)

                signal_strength = math.sqrt(alpha_sq_current)
                noise_strength = math.sqrt(1.0 - alpha_sq_current)
                noise = torch.randn_like(clean_label_embeddings)
                noisy_label_embeddings = (clean_label_embeddings * signal_strength) + (
                    noise * noise_strength
                )

                block_input_embeddings = clean_input_embeddings + noisy_label_embeddings

                with torch.amp.autocast(
                    device_type=device.type, enabled=config.FP16_ENABLED
                ):
                    predicted_denoised_embeddings = current_block(
                        block_input_embeddings
                    )
                    unweighted_loss = criterion(
                        predicted_denoised_embeddings, clean_label_embeddings
                    )

                    loss_mask = (labels != config.IGNORE_INDEX).unsqueeze(-1)
                    masked_loss = unweighted_loss * loss_mask

                    weighted_loss = masked_loss * loss_weight
                    loss = weighted_loss.sum() / (loss_mask.sum() + 1e-8)

                scaler.scale(loss).backward()
                if config.GRAD_CLIP_VALUE > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        current_block.parameters(), config.GRAD_CLIP_VALUE
                    )
                scaler.step(optimizer)
                scaler.update()

                if wandb_run:
                    wandb_run.log(
                        {"train/loss": loss.item(), "global_step": global_step}
                    )
                total_loss += loss.item()
                progress_bar.set_postfix(Loss=f"{loss.item():.4f}")

                global_step += 1

                if (
                    wandb_run
                    and config.LOG_GENERATION_EVERY_N_STEPS > 0
                    and global_step % config.LOG_GENERATION_EVERY_N_STEPS == 0
                ):
                    print(f"\n--- Generating Samples at Step {global_step} ---")

                    generator_fn(
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        config=config,
                        prompts=config.TEST_PROMPTS_FOR_LOGGING[
                            : config.NUM_GENERATION_EXAMPLES_TO_LOG
                        ],
                        step=global_step,
                        wandb_table=gen_table,
                    )
                    wandb_run.log(
                        {f"generations/samples_step_{global_step}": gen_table}
                    )
                    current_block = current_block.to(device)
                    current_block.train()

            avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
            print(f"  Block {block_idx} trained. Avg Weighted Loss: {avg_loss:.6f}")
            epoch_log[f"train_loss/block_{block_idx}"] = avg_loss

            current_block.to("cpu")
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if val_loader:
            avg_val_loss = validation_fn(model, val_loader, criterion, device)
            print(
                f"\nEpoch {epoch + 1} | End-to-End Validation MSE Loss: {avg_val_loss:.6f}"
            )
            epoch_log["validation/loss"] = avg_val_loss

        if wandb_run:
            wandb_run.log(epoch_log)
