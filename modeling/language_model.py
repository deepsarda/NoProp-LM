from typing import List

import torch
import torch.nn as nn

import config as C
from modeling.denoising_block import DenoisingBlock


class LanguageModel(nn.Module):
    """
    The NoProp Language Model.
    This class is now a lightweight container. It holds the embedding table
    and the list of denoising blocks, which are managed by the trainer.
    """

    def __init__(self, vocab_size: int, config: C):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size

        # Initialized on CPU by default. Device placement is handled by the main script.
        self.embedding_table = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=config.EMBEDDING_DIM
        )

        # Initialized on CPU by default.
        self.denoising_blocks = nn.ModuleList(
            [
                DenoisingBlock(vocab_size, config)
                for _ in range(config.NUM_DENOISING_BLOCKS)
            ]
        )
        print(
            f"Initialized {len(self.denoising_blocks)} independent Denoising Blocks on CPU."
        )

    def load_and_freeze_embeddings(self, embedding_weights_path: str = None):
        """
        Loads pretrained weights into the embedding table and freezes it.
        To be called after model initialization.
        """
        if embedding_weights_path:
            try:
                print(
                    f"Loading pretrained embedding weights from '{embedding_weights_path}'..."
                )
                loaded_weights = torch.load(embedding_weights_path, map_location="cpu")
                self.embedding_table.weight.data.copy_(loaded_weights)
            except Exception as e:
                print(
                    f"Warning: Could not load pretrained weights: {e}. Using random initialization."
                )
        else:
            print("Initializing embedding table with random weights.")

        self.embedding_table.weight.requires_grad = False
        print("Embedding table frozen.")

    def get_clean_embedding(self, token_ids: torch.Tensor) -> torch.Tensor:
        # This will work as long as self.embedding_table and token_ids are on the same device.
        return self.embedding_table(token_ids)

    def get_parameters_for_block(self, block_idx: int) -> List[torch.nn.Parameter]:
        return list(self.denoising_blocks[block_idx].parameters())

    def full_denoising_chain(
        self, initial_noisy_embedding: torch.Tensor, context_ids: torch.Tensor
    ) -> torch.Tensor:
        # This method is used by the generator, which moves the whole model to the device first.
        current_embedding = initial_noisy_embedding

        # Embed the context once before the denoising chain begins.
        with torch.no_grad():
            context_embeds = self.get_clean_embedding(context_ids)

        for block in self.denoising_blocks:
            current_embedding = block(
                noisy_target_embedding=current_embedding, context_embeds=context_embeds
            )
        return current_embedding
