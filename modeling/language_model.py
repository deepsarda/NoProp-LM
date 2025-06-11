from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C
from modeling.denoising_block import DenoisingBlock


class LanguageModel(nn.Module):
    """
    The main NoProp Language Model (NoProp-LM).

    This class serves as a container for the core components of the NoProp-LM:
    1.  `embedding_table`: An `nn.Embedding` layer that stores token embeddings.
        This table is typically initialized with pretrained embeddings and then frozen.
    2.  `denoising_blocks`: An `nn.ModuleList` containing multiple instances of
        `DenoisingBlock`. Each block is an independent Encoder-Decoder Transformer
        responsible for one step in the iterative denoising process.

    The `LanguageModel` itself is a lightweight container. Critical aspects like
    device placement and the training logic are managed externally by the training script.
    """

    def __init__(self, vocab_size: int, config: C):
        """
        Initializes the LanguageModel.

        Args:
            vocab_size (int): The total number of unique tokens in the vocabulary.
            config (C): The global configuration object, providing parameters like
                embedding dimension and number of denoising blocks.
        """
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size

        # Embedding table for token representations.
        self.embedding_table = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=config.EMBEDDING_DIM
        )

        # A list of independent DenoisingBlock modules.
        self.denoising_blocks = nn.ModuleList(
            [
                DenoisingBlock(vocab_size, config)
                for _ in range(config.NUM_DENOISING_BLOCKS)
            ]
        )
        print(
            f"Initialized LanguageModel with {len(self.denoising_blocks)} independent Denoising Blocks on CPU."
        )

    def load_and_freeze_embeddings(self, embedding_weights_path: Optional[str] = None):
        """
        Loads pretrained weights into the model's embedding table and freezes it.

        Args:
            embedding_weights_path (Optional[str], optional): Path to a .pt file
                containing pretrained embedding weights. Defaults to None (random init).
        """
        if embedding_weights_path:
            try:
                print(
                    f"Attempting to load pretrained embedding weights from '{embedding_weights_path}'..."
                )
                loaded_weights = torch.load(embedding_weights_path, map_location="cpu")
                self.embedding_table.weight.data.copy_(loaded_weights)
                print("Successfully loaded pretrained embedding weights.")
            except FileNotFoundError:
                print(
                    f"Warning: Pretrained embedding file not found at '{embedding_weights_path}'. Using random initialization."
                )
            except Exception as e:
                print(
                    f"Warning: Could not load pretrained weights due to an error: {e}. Using random initialization."
                )
        else:
            print("No pretrained embedding path provided. Using random initialization.")

        self.embedding_table.weight.requires_grad = False
        print("Embedding table is now frozen (requires_grad=False).")

    def get_clean_embedding(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Retrieves the clean (noise-free) embeddings for a given tensor of token IDs.

        Args:
            token_ids (torch.Tensor): A tensor containing token IDs.

        Returns:
            torch.Tensor: A tensor containing the corresponding embeddings.
        """
        raw_embeddings = self.embedding_table(token_ids)
        # L2-normalize the embeddings along the feature dimension
        normalized_embeddings = F.normalize(raw_embeddings, p=2, dim=-1)
        return normalized_embeddings

    def get_parameters_for_block(self, block_idx: int) -> List[torch.nn.Parameter]:
        """
        Retrieves the list of learnable parameters for a specific DenoisingBlock.

        Args:
            block_idx (int): The index of the DenoisingBlock.

        Returns:
            List[torch.nn.Parameter]: A list of parameters for the specified block.
        """
        if not 0 <= block_idx < len(self.denoising_blocks):
            raise IndexError(f"Block index {block_idx} is out of range.")
        return list(self.denoising_blocks[block_idx].parameters())
