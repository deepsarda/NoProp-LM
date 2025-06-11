from typing import Optional

import torch
import torch.nn.functional as F
import wandb
from transformers import PreTrainedTokenizerBase

import config as C
from modeling.language_model import LanguageModel
from utilities.helpers import get_alpha_squared_from_cosine_schedule


def find_nearest_token(
    predicted_embedding: torch.Tensor, embedding_table: torch.nn.Embedding
) -> torch.Tensor:
    """
    Finds the token ID in the embedding_table whose embedding is most similar
    to the predicted_embedding. Similarity is calculated based on cosine similarity.

    Args:
        predicted_embedding (torch.Tensor): A tensor of shape (batch_size, embedding_dim)
            representing the predicted embedding(s) from the model.
        embedding_table (torch.nn.Embedding): The model's embedding layer, which contains
            the vocabulary of token embeddings.

    Returns:
        torch.Tensor: A tensor of shape (batch_size,) containing the token IDs
            that are most similar to the predicted_embedding(s).
    """
    # The find_nearest_token function works by finding the highest cosine similarity.
    # Because our model's clean embeddings are already normalized, we only need to
    # normalize the predicted embedding before the matrix multiplication.
    pred_norm = F.normalize(predicted_embedding, p=2, dim=1)

    # The embedding table weights are NOT normalized in memory,
    # so we normalize them here for the similarity calculation.
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
    """
    Generates text sequences using the NoProp-LM model with a DDIM sampler.
    """
    model.to(device)
    model.eval()

    # Pre-compute the entire alpha_squared schedule for the DDIM sampler
    T = len(model.denoising_blocks)
    alphas_squared_schedule = torch.tensor(
        [get_alpha_squared_from_cosine_schedule(i, config) for i in range(T)],
        device=device,
        dtype=torch.float32,
    ).view(-1, 1, 1)

    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        context_ids = tokenizer.encode(
            prompt, return_tensors="pt", add_special_tokens=False
        ).to(device)

        for _ in range(config.GENERATION_MAX_OUTPUT_LENGTH):
            # Prepare context for the current generation step
            current_context_ids = context_ids[:, -config.MAX_SEQ_LENGTH :]

            # Since get_clean_embedding now normalizes, this is the clean, normalized context
            clean_context_embeddings = model.get_clean_embedding(current_context_ids)
            src_padding_mask = current_context_ids == tokenizer.pad_token_id

            # The target is a single token, so no padding mask is needed for it
            tgt_padding_mask = torch.zeros((1, 1), dtype=torch.bool, device=device)

            # Start with pure Gaussian noise for the single token to be generated
            current_embeddings = torch.randn(1, 1, config.EMBEDDING_DIM, device=device)
            final_embedding = None

            for block_idx, block in enumerate(model.denoising_blocks):
                # Get the noise level parameters for the current step (t)
                alpha_sq_t = alphas_squared_schedule[block_idx]
                sigma_t = torch.sqrt(1.0 - alpha_sq_t)

                # Pre-conditioning
                sigma_sq = sigma_t**2
                c_skip = 1.0 / (sigma_sq + 1.0)
                c_out = sigma_t * torch.sqrt(c_skip)
                c_in = torch.sqrt(c_skip)

                network_input = current_embeddings * c_in
                network_output = block(
                    src_embeds=clean_context_embeddings,
                    tgt_embeds=network_input,
                    src_padding_mask=src_padding_mask,
                    tgt_padding_mask=tgt_padding_mask,
                )
                predicted_clean_embeddings = (current_embeddings * c_skip) + (
                    network_output * c_out
                )
                predicted_clean_embeddings.clamp_(-1.0, 1.0)

                # --- DDIM Update Step ---
                if block_idx < T - 1:
                    alpha_sq_t_prev = alphas_squared_schedule[block_idx + 1]
                    pred_noise_direction = (
                        current_embeddings
                        - torch.sqrt(alpha_sq_t) * predicted_clean_embeddings
                    ) / torch.sqrt(
                        1.0 - alpha_sq_t + 1e-8
                    )  # Add epsilon for stability

                    current_embeddings = (
                        torch.sqrt(alpha_sq_t_prev) * predicted_clean_embeddings
                        + torch.sqrt(1.0 - alpha_sq_t_prev) * pred_noise_direction
                    )
                else:
                    final_embedding = predicted_clean_embeddings.squeeze(1)

            # Find the closest token in the vocabulary to the final denoised embedding
            next_token_id = find_nearest_token(final_embedding, model.embedding_table)

            if next_token_id.item() == tokenizer.eos_token_id:
                break

            context_ids = torch.cat([context_ids, next_token_id.unsqueeze(0)], dim=1)

        generated_text = tokenizer.decode(
            context_ids.squeeze(0).tolist(), skip_special_tokens=True
        )
        print(f"Generated: {generated_text}")

        if wandb_table is not None:
            wandb_table.add_data(step, prompt, generated_text)

    # Restore model state
    model.to("cpu")
    model.embedding_table.to(device)
