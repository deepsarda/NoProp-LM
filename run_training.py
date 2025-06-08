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
    print("--- Starting NoProp-LM Training Program ---")
    set_seed(C.SEED)
    device = torch.device(C.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(C.TOKENIZER_NAME)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    vocab_size = len(tokenizer)

    # --- Load both Training and Validation Datasets ---
    raw_train_dataset = load_tokenized_dataset(tokenizer=tokenizer, split="train")
    if not raw_train_dataset:
        raise ValueError("Training data could not be loaded.")

    raw_val_dataset = load_tokenized_dataset(tokenizer=tokenizer, split="validation")
    if not raw_val_dataset:
        print("Warning: Validation data could not be loaded. Skipping validation.")

    embedding_path = None
    if C.DO_PRETRAINING:
        embedding_path = run_embedding_pretraining(
            config=C,
            tokenizer=tokenizer,
            train_dataset=raw_train_dataset,
            device=device,
        )
    else:
        print("\n--- SKIPPING PHASE 1: Pretraining ---")

    noprop_run = init_wandb_run(
        C, C.WANDB_PROJECT_NAME, C.WANDB_RUN_NAME, "noprop-training"
    )

    collate_fn = Collate(tokenizer)
    train_loader = DataLoader(
        raw_train_dataset,
        sampler=RandomSampler(raw_train_dataset),
        batch_size=C.BATCH_SIZE,
        num_workers=C.DATALOADER_NUM_WORKERS_TRAIN,
        collate_fn=collate_fn,
    )

    val_loader = None
    if raw_val_dataset:
        val_loader = DataLoader(
            raw_val_dataset,
            sampler=SequentialSampler(raw_val_dataset),
            batch_size=C.BATCH_SIZE,
            num_workers=C.DATALOADER_NUM_WORKERS_TRAIN,
            collate_fn=collate_fn,
        )

    print("\n--- Initializing NoProp-LM Model for Phase 2 ---")
    model = LanguageModel(vocab_size, C)
    model.load_and_freeze_embeddings(embedding_weights_path=embedding_path)

    model.embedding_table.to(device)
    print(f"Moved embedding_table to {device}. Denoising blocks remain on CPU.")

    if noprop_run:
        log_model_parameters(model, noprop_run)

    run_training_loop(
        config=C,
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        generator_fn=generate_text,
        validation_fn=run_validation,
        wandb_run=noprop_run,
    )

    if C.FINAL_MODEL_SAVE_PATH:
        print(
            f"\n--- Saving final model state dictionary to '{C.FINAL_MODEL_SAVE_PATH}' ---"
        )
        try:
            # Ensure the entire model is on the CPU before saving for consistency.
            # The training loop already moves blocks to CPU, but this is an extra safeguard.
            model.to("cpu")
            torch.save(model.state_dict(), C.FINAL_MODEL_SAVE_PATH)
            print("Model saved successfully.")
            # If using W&B, let's save the model as an artifact
            if noprop_run:
                artifact = wandb.Artifact(name=C.WANDB_RUN_NAME, type="model")
                artifact.add_file(C.FINAL_MODEL_SAVE_PATH)
                noprop_run.log_artifact(artifact)
                print("Model also saved as a W&B artifact.")

        except Exception as e:
            print(f"Error saving model: {e}")

    if noprop_run:
        noprop_run.finish()
    print("\n--- NoProp-LM Training Program Completed ---")


if __name__ == "__main__":
    main()
