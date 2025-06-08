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
    config: C, tokenizer: PreTrainedTokenizerBase, train_dataset, device: torch.device
) -> str:
    print("\n--- PHASE 1: Pretraining Global Embedding Table via MLM ---")

    pretrain_run = init_wandb_run(
        config, C.WANDB_PRETRAIN_PROJECT_NAME, C.WANDB_PRETRAIN_RUN_NAME, "pretraining"
    )

    vocab_size = len(tokenizer)
    model = MLMPretrainModel(vocab_size, config).to(device)
    if pretrain_run:
        log_model_parameters(model, pretrain_run)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=config.PRETRAINING_MASK_PROB
    )
    pretrain_loader = pretrain_loader = DataLoader(
        train_dataset,
        batch_size=config.PRETRAINING_BATCH_SIZE,
        collate_fn=data_collator,
        shuffle=True,
        num_workers=C.DATALOADER_NUM_WORKERS_TRAIN,
        pin_memory=True,
        prefetch_factor=2,
    )

    optimizer = AdamW(model.parameters(), lr=config.PRETRAINING_LR)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=config.FP16_ENABLED)

    global_step = 0
    for epoch in range(config.PRETRAINING_EPOCHS):
        model.train()
        progress_bar = tqdm(
            pretrain_loader, desc=f"Pretrain E{epoch+1}", dynamic_ncols=True
        )

        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            with torch.amp.autocast(
                device_type=device.type, enabled=config.FP16_ENABLED
            ):
                outputs = model(batch["input_ids"])
                loss = criterion(outputs.view(-1, vocab_size), batch["labels"].view(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            progress_bar.set_postfix(Loss=f"{loss.item():.4f}")
            if pretrain_run:
                pretrain_run.log({"pretrain/loss": loss.item()}, step=global_step)
            global_step += 1

    print("Pretraining complete.")
    if pretrain_run:
        pretrain_run.finish()
    model.to("cpu")
    embedding_weights = model.token_embedder.weight.detach()
    torch.save(embedding_weights, config.PRETRAINED_EMBEDDING_FILE)
    print(f"Saved pretrained embedding weights to '{config.PRETRAINED_EMBEDDING_FILE}'")
    del model, embedding_weights, optimizer, scaler, pretrain_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return config.PRETRAINED_EMBEDDING_FILE
