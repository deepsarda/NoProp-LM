"""
This file contains functions for generating text sequences using a trained NoProp-LM model.
It includes a core `generate_text` function that iteratively predicts tokens
by running a denoising chain for each new token, and a helper function
`find_nearest_token` to map predicted embeddings back to token IDs.
"""
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
    2. Creating a clean embedding of this context.
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
    model.to(device) # Move the entire model to the specified device
    model.eval()     # Set the model to evaluation mode (disables dropout, etc.)

    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        # Encode the initial prompt into token IDs
        context_ids = tokenizer.encode(
            prompt, return_tensors="pt", add_special_tokens=False # `pt` for PyTorch tensors
        ).to(device)

        # Generate tokens one by one up to the maximum output length
        for _ in range(config.GENERATION_MAX_OUTPUT_LENGTH):
            # Ensure the context for the next token prediction does not exceed MAX_SEQ_LENGTH
            # Takes the most recent `config.MAX_SEQ_LENGTH` tokens as context.
            current_context_ids = context_ids[:, -config.MAX_SEQ_LENGTH :]

            # --- Denoising chain for predicting the single next token ---
            # 1. Get clean (noise-free) embeddings for the current context tokens.
            # These embeddings act as conditioning for the denoising process.
            clean_context_embeddings = model.get_clean_embedding(current_context_ids)

            # 2. Initialize a random noise vector for the token we want to predict.
            # This represents the initial "fully noisy" state of the target token's embedding.
            # Shape: (batch_size=1, num_tokens_to_predict=1, embedding_dim)
            initial_noise = torch.randn(
                1, 1, config.EMBEDDING_DIM, device=device
            )

            # 3. Iteratively denoise the `initial_noise` using the model's denoising blocks.
            current_denoised_embedding = initial_noise
            for block_idx, block in enumerate(model.denoising_blocks):
                # The input to each denoising block is the concatenation of:
                #   a) The clean embeddings of the context tokens.
                #   b) The currently (partially) denoised embedding of the target token.
                # Shape: (batch_size, context_len + 1, embedding_dim)
                block_input_sequence = torch.cat(
                    [clean_context_embeddings, current_denoised_embedding], dim=1
                )

                # Pass the sequence through the current denoising block.
                # The block processes the entire sequence (context + target).
                full_denoised_sequence = block(block_input_sequence)

                # We are only interested in the output corresponding to the target token,
                # which is the last embedding in the output sequence.
                # This becomes the input for the next denoising block (or the final prediction).
                current_denoised_embedding = full_denoised_sequence[:, -1:, :] # Keep seq_len dim

            # 4. The `current_denoised_embedding` is now the final predicted embedding for the next token.
            # Squeeze to remove the sequence length dimension (it's 1).
            final_embedding = current_denoised_embedding.squeeze(1) # Shape: (batch_size, embedding_dim)

            # 5. Find the token in the vocabulary whose embedding is closest to our predicted embedding.
            next_token_id = find_nearest_token(final_embedding, model.embedding_table)

            # Stop generation if the End-Of-Sequence token is predicted.
            if next_token_id.item() == tokenizer.eos_token_id:
                break

            # Append the predicted token ID to the context for the next generation step.
            # `unsqueeze(0)` adds batch dimension if `next_token_id` is scalar.
            context_ids = torch.cat([context_ids, next_token_id.unsqueeze(0)], dim=1)

        # Decode all collected token IDs (prompt + generated) back to text.
        generated_text = tokenizer.decode(context_ids.squeeze(0).tolist())
        print(f"Generated: {generated_text}")

        # Log to Weights & Biases table if provided
        if wandb_table is not None:
            wandb_table.add_data(step, prompt, generated_text)

    # After generation, move the main model parts (denoising blocks) back to CPU
    # to free up GPU memory, but keep the embedding table on the (potentially GPU) device
    # as it might be shared or used by other parts of the training loop.
    model.to("cpu")
    model.embedding_table.to(device) # Ensure embedding table stays on the designated device
