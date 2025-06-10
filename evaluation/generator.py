from typing import Optional

import torch
import torch.nn.functional as F
import wandb
from transformers import PreTrainedTokenizerBase

import config as C
from modeling.language_model import LanguageModel
from utilities.helpers import get_ddpm_schedule


def find_nearest_token(
    predicted_embedding: torch.Tensor, embedding_table: torch.nn.Embedding
) -> torch.Tensor:
    """
    Finds the token ID in the embedding_table whose embedding is most similar
    to the predicted_embedding. Similarity is calculated based on cosine similarity
    after L2 normalization.

    Args:
        predicted_embedding (torch.Tensor): A tensor of shape (batch_size, embedding_dim)
            representing the predicted embedding(s) from the model.
        embedding_table (torch.nn.Embedding): The model's embedding layer, which contains
            the vocabulary of token embeddings.

    Returns:
        torch.Tensor: A tensor of shape (batch_size,) containing the token IDs
            that are most similar to the predicted_embedding(s).
    """
    # Normalize the predicted embedding and the embedding table weights for cosine similarity
    pred_norm = F.normalize(predicted_embedding, p=2, dim=1)
    table_norm = F.normalize(embedding_table.weight, p=2, dim=1)

    # Calculate cosine similarity scores between predicted embedding and all table embeddings
    similarity_scores = torch.matmul(pred_norm, table_norm.T)

    # Find the token ID with the highest similarity score
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
    Generates text sequences given a list of starting prompts using the NoProp-LM model.

    For each prompt, text is generated token by token up to `config.GENERATION_MAX_OUTPUT_LENGTH`.
    Each new token is predicted by:
    1. Taking the current context (prompt + already generated tokens).
    2. Creating a clean embedding of this context along with the padding masks.
    3. Initializing a noisy embedding for the token to be predicted.
    4. Iteratively denoising this noisy embedding through all denoising blocks of the model,
       using the clean context embedding as conditioning.
    5. Finding the token in the vocabulary whose embedding is closest to the final denoised embedding.

    The generated text is printed to the console. If a `wandb.Table` is provided,
    the generation step, prompt, and generated text are logged to it.

    Args:
        model (LanguageModel): The trained NoProp-LM model.
        tokenizer (PreTrainedTokenizerBase): The tokenizer used for encoding prompts and
            decoding generated token IDs.
        device (torch.device): The device (e.g., 'cuda', 'cpu') to run generation on.
            The model is moved to this device.
        config (C): The project's configuration object, providing parameters like
            `GENERATION_MAX_OUTPUT_LENGTH`, `MAX_SEQ_LENGTH`, and `EMBEDDING_DIM`.
        prompts (list[str]): A list of initial text prompts to start generation from.
        step (int): The current training step, used for logging to W&B.
        wandb_table (Optional[wandb.Table], optional): A Weights & Biases Table object.
            If provided, generation results are logged to this table. Defaults to None.

    Side effects:
        - Prints prompts and generated text to standard output.
        - Logs data to `wandb_table` if provided.
        - Moves the model to `device` for generation and then back to "cpu",
          keeping the embedding table on `device`.
    """
    model.to(device)  # Move the entire model to the specified device
    model.eval()  # Set the model to evaluation mode (disables dropout, etc.)

    # Get the pre-computed noise schedule
    T = len(model.denoising_blocks)
    schedule = get_ddpm_schedule(T, config, device)
    posterior_mean_coef1 = schedule["posterior_mean_coef1"]
    posterior_mean_coef2 = schedule["posterior_mean_coef2"]
    posterior_variance = schedule["posterior_variance"]

    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        # Encode the initial prompt into token IDs
        context_ids = tokenizer.encode(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,  # `pt` for PyTorch tensors
        ).to(device)

        # Generate tokens one by one up to the max output length
        for _ in range(config.GENERATION_MAX_OUTPUT_LENGTH):
            # Take the most recent tokens as context
            current_context_ids = context_ids[:, -config.MAX_SEQ_LENGTH :]
            clean_context_embeddings = model.get_clean_embedding(current_context_ids)
            src_padding_mask = current_context_ids == tokenizer.pad_token_id
            # The target to be generated is a single token, so no padding mask needed
            tgt_padding_mask = torch.zeros((1, 1), dtype=torch.bool, device=device)

            # Denoising chain for predicting the single next token
            # 1. Initialize with pure Gaussian noise (z_T)
            current_embedding = torch.randn(1, 1, config.EMBEDDING_DIM, device=device)
            final_embedding = None

            # 2. Iterate FORWARDS through blocks (backwards in diffusion time)
            for block_idx, block in enumerate(model.denoising_blocks):
                # Predict the original clean embedding from the current noisy one
                predicted_clean_embedding = block(
                    src_embeds=clean_context_embeddings,
                    tgt_embeds=current_embedding,
                    src_padding_mask=src_padding_mask,
                    tgt_padding_mask=tgt_padding_mask,
                )

                # 3. Sample the next, less noisy embedding `z_{t-1}`
                if block_idx < T - 1:
                    posterior_mean = (
                        posterior_mean_coef1[block_idx] * current_embedding
                        + posterior_mean_coef2[block_idx] * predicted_clean_embedding
                    )
                    noise = torch.randn_like(current_embedding)
                    current_embedding = (
                        posterior_mean
                        + torch.sqrt(posterior_variance[block_idx]) * noise
                    )
                else:
                    # 4. This is the final block, its prediction is our result
                    final_embedding = predicted_clean_embedding.squeeze(1)

            # 5. Find the closest token in the vocabulary.
            next_token_id = find_nearest_token(final_embedding, model.embedding_table)

            # Stop generation if the End-Of-Sequence token is predicted.
            if next_token_id.item() == tokenizer.eos_token_id:
                break

            # Append the predicted token ID to the context for the next generation step.
            # `unsqueeze(0)` adds batch dimension as `next_token_id` is scalar.
            context_ids = torch.cat([context_ids, next_token_id.unsqueeze(0)], dim=1)

        # Decode all collected token IDs (prompt + generated) back to text.
        generated_text = tokenizer.decode(
            context_ids.squeeze(0).tolist(), skip_special_tokens=True
        )
        print(f"Generated: {generated_text}")

        # Log to Weights & Biases table if provided
        if wandb_table is not None:
            wandb_table.add_data(step, prompt, generated_text)

    # After generation, move the main model parts (denoising blocks) back to CPU
    # to free up GPU memory, but keep the embedding table on the (potentially GPU) device
    # as it might be shared or used by other parts of the training loop.
    model.to("cpu")
    model.embedding_table.to(
        device
    )  # Ensure embedding table stays on the designated device
