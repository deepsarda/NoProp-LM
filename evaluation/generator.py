from typing import Optional

import torch
import torch.nn.functional as F
import wandb
from transformers import PreTrainedTokenizerBase

import config as C
from modeling.language_model import LanguageModel


def find_nearest_token(
    predicted_embedding: torch.Tensor, embedding_table: torch.nn.Embedding
) -> torch.Tensor:
    pred_norm = F.normalize(predicted_embedding, p=2, dim=1)
    table_norm = F.normalize(embedding_table.weight, p=2, dim=1)
    similarity_scores = torch.matmul(pred_norm, table_norm.T)
    _, best_token_ids = torch.max(similarity_scores, dim=1)
    return best_token_ids


@torch.no_grad()
def generate_text(
    model: LanguageModel,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
    config: C,
    prompts: list[str],
    step: int,
    wandb_table: Optional[wandb.Table] = None,
):
    model.to(device)
    model.eval()
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        context_ids = tokenizer.encode(
            prompt, return_tensors="pt", add_special_tokens=False
        ).to(device)

        for _ in range(config.GENERATION_MAX_OUTPUT_LENGTH):
            # The context for the next token is the entire sequence generated so far.
            current_context_ids = context_ids[:, -config.MAX_SEQ_LENGTH :]

            # --- Denoising chain for the single next token ---
            clean_context_embeddings = model.get_clean_embedding(current_context_ids)
            initial_noise = torch.randn(
                1, 1, config.EMBEDDING_DIM, device=device
            )  # Noise for one token

            # Denoise from most noisy to least noisy block
            current_denoised_embedding = initial_noise
            for block in model.denoising_blocks:
                # The block input is the clean context followed by the noisy target-to-be
                block_input_sequence = torch.cat(
                    [clean_context_embeddings, current_denoised_embedding], dim=1
                )

                # The block processes the full sequence
                full_denoised_sequence = block(block_input_sequence)

                # We only care about the last output, which corresponds to our target token
                current_denoised_embedding = full_denoised_sequence[:, -1:, :]

            final_embedding = current_denoised_embedding.squeeze(1)
            next_token_id = find_nearest_token(final_embedding, model.embedding_table)

            if next_token_id.item() == tokenizer.eos_token_id:
                break

            # Append the predicted token to the context for the next step
            context_ids = torch.cat([context_ids, next_token_id.unsqueeze(0)], dim=1)

        generated_text = tokenizer.decode(context_ids.squeeze(0).tolist())
        print(f"Generated: {generated_text}")
        if wandb_table is not None:
            wandb_table.add_data(step, prompt, generated_text)

    model.to("cpu")
    model.embedding_table.to(device)
