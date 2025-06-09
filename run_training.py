import torch
import wandb
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer

import config as C
from data_handling.collate import Collate
from data_handling.dataset_loader import load_tokenized_dataset
from evaluation.generator import generate_text
from evaluation.validator import run_validation
from modeling.language_model import LanguageModel
from training.pretrainer import run_embedding_pretraining
from training.trainer import run_training_loop
from utilities.helpers import set_seed
from utilities.wandb_utils import init_wandb_run, log_model_parameters


def main():
    """
    Main function to orchestrate the NoProp-LM training process.

    Steps:
    1. Set up seed and device.
    2. Load tokenizer and datasets.
    3. Optionally run embedding pretraining (Phase 1).
    4. Initialize WandB for NoProp training.
    5. Create DataLoaders.
    6. Initialize NoProp-LM model, load pretrained embeddings if available.
    7. Run the main NoProp training loop (Phase 2).
    8. Save the final model.
    9. Finish WandB run.
    """
    print("--- Starting NoProp-LM Training Program ---")
    # Set random seed for reproducibility
    set_seed(C.SEED)
    # Determine and set the training device (GPU if available, otherwise CPU)
    device = torch.device(C.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(C.TOKENIZER_NAME)
    if tokenizer.pad_token is None: # Ensure tokenizer has a pad token
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    vocab_size = len(tokenizer)

    # --- Load Training and Validation Datasets ---
    print("\n--- Loading Datasets ---")
    raw_train_dataset = load_tokenized_dataset(tokenizer=tokenizer, split="train")
    if not raw_train_dataset:
        raise ValueError("Training data could not be loaded.")

    raw_val_dataset = load_tokenized_dataset(tokenizer=tokenizer, split="validation")
    if not raw_val_dataset:
        print("Warning: Validation data could not be loaded. Skipping validation.")

    # --- Phase 1: Optional Embedding Pretraining ---
    embedding_path = None # Path to saved pretrained embeddings
    if C.DO_PRETRAINING:
        print("\n--- Starting PHASE 1: Embedding Pretraining ---")
        embedding_path = run_embedding_pretraining(
            config=C,
            tokenizer=tokenizer,
            train_dataset=raw_train_dataset,
            device=device,
        )
    else:
        print("\n--- SKIPPING PHASE 1: Embedding Pretraining ---")

    # Initialize WandB for the main NoProp training phase
    noprop_run = init_wandb_run(
        C, C.WANDB_PROJECT_NAME, C.WANDB_RUN_NAME, "noprop-training"
    )

    # Create DataLoader for training data
    collate_fn = Collate(tokenizer) # Custom collate function for batching
    train_loader = DataLoader(
        raw_train_dataset,
        sampler=RandomSampler(raw_train_dataset), # Use RandomSampler for training
        batch_size=C.BATCH_SIZE,
        num_workers=C.DATALOADER_NUM_WORKERS_TRAIN,
        collate_fn=collate_fn,
    )

    # Create DataLoader for validation data if available
    val_loader = None
    if raw_val_dataset:
        val_loader = DataLoader(
            raw_val_dataset,
            sampler=SequentialSampler(raw_val_dataset), # Use SequentialSampler for validation
            batch_size=C.BATCH_SIZE,
            num_workers=C.DATALOADER_NUM_WORKERS_TRAIN,
            collate_fn=collate_fn,
        )

    # --- Phase 2: NoProp-LM Model Training ---
    print("\n--- Initializing NoProp-LM Model for PHASE 2: NoProp Training ---")
    # Initialize the LanguageModel
    model = LanguageModel(vocab_size, C)
    # Load and freeze embeddings (either pretrained or randomly initialized)
    model.load_and_freeze_embeddings(embedding_weights_path=embedding_path)

    # Move the embedding table to the designated device.
    # Denoising blocks will be moved to the device dynamically during training.
    model.embedding_table.to(device)
    print(f"Moved embedding_table to {device}. Denoising blocks remain on CPU initially.")

    # Log model parameters to WandB if enabled
    if noprop_run:
        log_model_parameters(model, noprop_run)

    # Start the main training loop
    print("\n--- Starting PHASE 2: NoProp Training Loop ---")
    run_training_loop(
        config=C,
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        generator_fn=generate_text, # Function to generate text samples during training
        validation_fn=run_validation, # Function to run validation
        wandb_run=noprop_run, # WandB run object for logging
    )

    # --- Save Final Model ---
    if C.FINAL_MODEL_SAVE_PATH:
        print(
            f"\n--- Saving final model state dictionary to '{C.FINAL_MODEL_SAVE_PATH}' ---"
        )
        try:
            # Ensure the entire model is on the CPU before saving for consistency.
            # The training loop might leave parts of the model on different devices.
            model.to("cpu")
            torch.save(model.state_dict(), C.FINAL_MODEL_SAVE_PATH)
            print("Model saved successfully.")
            # If using W&B, save the model as a W&B artifact
            if noprop_run:
                artifact = wandb.Artifact(name=C.WANDB_RUN_NAME, type="model")
                artifact.add_file(C.FINAL_MODEL_SAVE_PATH)
                noprop_run.log_artifact(artifact)
                print("Model also saved as a W&B artifact.")

        except Exception as e:
            print(f"Error saving model: {e}")

    # Finish WandB run if initialized
    if noprop_run:
        noprop_run.finish()
    print("\n--- NoProp-LM Training Program Completed ---")


if __name__ == "__main__":
    # Entry point when the script is executed
    main()
