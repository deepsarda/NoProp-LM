"""
This file contains the logic for the first phase of training: pretraining
the global token embedding table using a Masked Language Model (MLM) objective.
The `MLMPretrainModel` is used for this phase. After pretraining, the learned
embedding weights are saved to be loaded by the main NoProp-LM.
"""
import gc

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerBase

import config as C
from modeling.pretrain_model import MLMPretrainModel
from utilities.wandb_utils import init_wandb_run, log_model_parameters


def run_embedding_pretraining(
    config: C,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: torch.utils.data.Dataset,
    device: torch.device,
) -> str:
    """
    Runs the embedding pretraining phase using a Masked Language Model (MLM) objective.

    This function initializes an `MLMPretrainModel`, sets up a DataLoader with a
    specialized MLM data collator, and trains the model to predict masked tokens.
    The primary goal is to learn high-quality token embeddings.

    After training, the weights of the model's token embedding layer are extracted,
    saved to a file specified by `config.PRETRAINED_EMBEDDING_FILE`, and this
    filepath is returned. Weights & Biases logging is used if configured.

    Args:
        config (C): The global configuration object, providing parameters for
            pretraining such as epochs, learning rate, batch size, W&B project names,
            and the save path for embeddings.
        tokenizer (PreTrainedTokenizerBase): The tokenizer used for preparing data
            for MLM (e.g., identifying mask token ID).
        train_dataset (torch.utils.data.Dataset): The training dataset, expected to
            contain tokenized sequences.
        device (torch.device): The device (e.g., 'cuda', 'cpu') on which to perform
            pretraining.

    Returns:
        str: The file path where the pretrained embedding weights have been saved.
             This is typically `config.PRETRAINED_EMBEDDING_FILE`.
    """
    print("\n--- PHASE 1: Pretraining Global Embedding Table via MLM ---")

    # Initialize Weights & Biases run for pretraining, if configured
    pretrain_run = init_wandb_run(
        config, C.WANDB_PRETRAIN_PROJECT_NAME, C.WANDB_PRETRAIN_RUN_NAME, "pretraining"
    )

    vocab_size = len(tokenizer)
    # Initialize the MLMPretrainModel and move it to the specified device
    model = MLMPretrainModel(vocab_size, config).to(device)
    if pretrain_run: # Log model parameters to W&B
        log_model_parameters(model, pretrain_run)

    # Data collator for MLM: automatically creates batches with masked tokens.
    # `mlm_probability` defines how many tokens are masked in each sequence.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=config.PRETRAINING_MASK_PROB
    )
    # DataLoader for the pretraining phase
    pretrain_loader = DataLoader(
        train_dataset,
        batch_size=config.PRETRAINING_BATCH_SIZE, # Batch size for pretraining
        collate_fn=data_collator, # Use the MLM data collator
        shuffle=True, # Shuffle data at each epoch
        num_workers=C.DATALOADER_NUM_WORKERS_TRAIN, # Number of worker processes
        pin_memory=True, # For faster data transfer to GPU, if applicable
        prefetch_factor=2, # Number of batches to prefetch
    )

    # Optimizer (AdamW) for training the MLM model
    optimizer = AdamW(model.parameters(), lr=config.PRETRAINING_LR)
    # Loss function: CrossEntropyLoss is standard for MLM.
    criterion = nn.CrossEntropyLoss()
    # Gradient scaler for mixed-precision training (FP16), if enabled
    scaler = torch.amp.GradScaler(enabled=config.FP16_ENABLED)

    global_step = 0
    print(f"Starting MLM pretraining for {config.PRETRAINING_EPOCHS} epochs...")
    for epoch in range(config.PRETRAINING_EPOCHS):
        model.train() # Set the model to training mode
        # Progress bar for iterating over batches
        progress_bar = tqdm(
            pretrain_loader, desc=f"Pretrain E{epoch+1}/{config.PRETRAINING_EPOCHS}", dynamic_ncols=True
        )

        for batch in progress_bar:
            # Move batch data to the target device
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            optimizer.zero_grad() # Clear previous gradients

            # Automatic mixed precision context, if enabled
            with torch.amp.autocast(
                device_type=device.type, enabled=config.FP16_ENABLED
            ):
                # Forward pass: get model predictions (logits) for masked tokens
                outputs = model(batch["input_ids"])
                # Calculate loss: compare predicted logits with actual token IDs (labels)
                # Reshape outputs to (batch_size * seq_len, vocab_size) and labels to (batch_size * seq_len)
                loss = criterion(outputs.view(-1, vocab_size), batch["labels"].view(-1))

            # Perform backpropagation with gradient scaling
            scaler.scale(loss).backward()
            # Update model parameters
            scaler.step(optimizer)
            # Update the scaler for the next iteration
            scaler.update()

            # Update progress bar and log loss to W&B
            progress_bar.set_postfix(Loss=f"{loss.item():.4f}")
            if pretrain_run:
                pretrain_run.log({"pretrain/loss": loss.item()}, step=global_step)
            global_step += 1

    print("MLM Pretraining complete.")
    if pretrain_run:
        pretrain_run.finish() # Close the W&B run

    # Save the learned embedding weights
    model.to("cpu") # Move model to CPU before extracting weights
    embedding_weights = model.token_embedder.weight.detach() # Get embedding weights
    torch.save(embedding_weights, config.PRETRAINED_EMBEDDING_FILE) # Save weights to file
    print(f"Saved pretrained embedding weights to '{config.PRETRAINED_EMBEDDING_FILE}'")

    # Clean up memory
    del model, embedding_weights, optimizer, scaler, pretrain_loader, data_collator, train_dataset
    gc.collect() # Python garbage collection
    if device.type == "cuda":
        torch.cuda.empty_cache() # Clear PyTorch's CUDA cache

    return config.PRETRAINED_EMBEDDING_FILE
