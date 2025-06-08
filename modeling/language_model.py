"""
This file defines the main LanguageModel class for the NoProp-LM.
The LanguageModel acts as a container for the token embedding table and
a list of DenoisingBlock modules. It manages loading pretrained embeddings
and provides methods for accessing embeddings and model parameters.
"""
from typing import List

import torch
import torch.nn as nn

import config as C
from modeling.denoising_block import DenoisingBlock


class LanguageModel(nn.Module):
    """
    The main NoProp Language Model (NoProp-LM).

    This class serves as a container for the core components of the NoProp-LM:
    1.  `embedding_table`: An `nn.Embedding` layer that stores token embeddings.
        This table is typically initialized with pretrained embeddings and then frozen.
    2.  `denoising_blocks`: An `nn.ModuleList` containing multiple instances of
        `DenoisingBlock`. Each block is an independent Transformer-based module
        responsible for one step in the iterative denoising process.

    The `LanguageModel` itself is a lightweight container. Critical aspects like
    device placement (moving embeddings or blocks to GPU/CPU) and the training
    logic (how blocks are trained, how noise is scheduled) are managed externally
    by the training script (`run_training.py`).

    Denoising blocks and the embedding table are initialized on the CPU by default.
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
        # Initialized on CPU by default. Device placement (e.g., to GPU)
        # is handled by the main training script.
        self.embedding_table = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=config.EMBEDDING_DIM
        )

        # A list of independent DenoisingBlock modules.
        # These are also initialized on CPU by default. The training script
        # will manage moving individual blocks to the active device during training.
        self.denoising_blocks = nn.ModuleList(
            [
                DenoisingBlock(vocab_size, config) # Pass vocab_size and config to each block
                for _ in range(config.NUM_DENOISING_BLOCKS)
            ]
        )
        print(
            f"Initialized LanguageModel with {len(self.denoising_blocks)} independent Denoising Blocks on CPU."
        )

    def load_and_freeze_embeddings(self, embedding_weights_path: Optional[str] = None):
        """
        Loads pretrained weights into the model's embedding table and freezes the
        table to prevent further updates during training.

        If `embedding_weights_path` is provided and valid, weights are loaded from
        the specified file. If the path is invalid, a warning is printed, and
        the table remains randomly initialized. If no path is provided, random
        initialization is used by default.

        Args:
            embedding_weights_path (Optional[str], optional): Path to a PyTorch file
                (.pt) containing the pretrained embedding weights. Defaults to None,
                in which case random initialization is used.
        """
        if embedding_weights_path:
            try:
                print(
                    f"Attempting to load pretrained embedding weights from '{embedding_weights_path}'..."
                )
                # Load weights, ensuring they are mapped to CPU to avoid device mismatches.
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
            print("No pretrained embedding path provided. Initializing embedding table with random weights.")

        # Freeze the embedding table by setting requires_grad to False for its weights.
        self.embedding_table.weight.requires_grad = False
        print("Embedding table is now frozen (requires_grad=False).")

    def get_clean_embedding(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Retrieves the clean (noise-free) embeddings for a given tensor of token IDs.

        This method assumes that `self.embedding_table` and `token_ids` are on the
        same device. Device management is handled externally.

        Args:
            token_ids (torch.Tensor): A tensor containing token IDs.

        Returns:
            torch.Tensor: A tensor containing the corresponding embeddings from the
                frozen embedding table.
        """
        return self.embedding_table(token_ids)

    def get_parameters_for_block(self, block_idx: int) -> List[torch.nn.Parameter]:
        """
        Retrieves the list of learnable parameters for a specific DenoisingBlock.

        This is useful for setting up optimizers that train individual blocks.

        Args:
            block_idx (int): The index of the DenoisingBlock in the `denoising_blocks` list.

        Returns:
            List[torch.nn.Parameter]: A list of parameters belonging to the specified block.

        Raises:
            IndexError: If `block_idx` is out of range.
        """
        if not 0 <= block_idx < len(self.denoising_blocks):
            raise IndexError(f"Block index {block_idx} is out of range.")
        return list(self.denoising_blocks[block_idx].parameters())

    def full_denoising_chain(
        self, initial_noisy_embedding: torch.Tensor, context_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Performs a full denoising chain starting from an initial noisy embedding,
        conditioned on context IDs. This method is primarily used for text generation.

        Important: This method assumes that the entire model (all denoising blocks
        and the embedding table) has been moved to the target device *before* calling it.
        This is typically handled by the generation script.

        Args:
            initial_noisy_embedding (torch.Tensor): The starting noisy embedding for the
                token(s) to be denoised. Shape: (batch_size, num_targets, embedding_dim).
            context_ids (torch.Tensor): Token IDs of the preceding context.
                Shape: (batch_size, context_sequence_length).

        Returns:
            torch.Tensor: The final denoised embedding after passing through all
                denoising blocks. Shape: (batch_size, num_targets, embedding_dim).
        """
        current_embedding = initial_noisy_embedding

        # Embed the context IDs once, without tracking gradients as embeddings are frozen.
        with torch.no_grad():
            context_embeds = self.get_clean_embedding(context_ids) # (B, SeqLen_ctx, EmbDim)

        # Sequentially pass the embedding through each DenoisingBlock.
        # Each block refines the `current_embedding`.
        for block in self.denoising_blocks:
            # The input to a DenoisingBlock during generation is typically
            # a concatenation of the clean context embeddings and the current noisy target embedding.
            # Assuming DenoisingBlock's forward method is: forward(self, block_input_embeddings)
            # where block_input_embeddings = torch.cat([context_embeds, current_embedding], dim=1)
            # The DenoisingBlock should then internally handle extracting the relevant part.
            # For this example, let's adjust based on the DenoisingBlock.forward signature
            # which expects the combined sequence.

            # Let's assume current_embedding is (B, 1, EmbDim) for a single target token
            # and context_embeds is (B, SeqLen_ctx, EmbDim)
            if context_embeds.size(0) != current_embedding.size(0):
                 # If context_embeds has batch but current_embedding doesn't (e.g. from generation loop)
                 # or vice-versa, this needs careful broadcasting or repeating.
                 # For typical generation, batch size is 1.
                 # If context_embeds is (1, S, D) and current_embedding is (1, 1, D)
                 # then cat([context_embeds, current_embedding], dim=1) is (1, S+1, D)
                 pass # Assuming batch sizes are compatible or handled by caller / DenoisingBlock

            block_input = torch.cat([context_embeds, current_embedding], dim=1)

            # The block processes the full sequence and we expect it to return
            # the full processed sequence. We then extract the part corresponding to the target.
            processed_sequence = block(block_input)
            current_embedding = processed_sequence[:, context_embeds.size(1):, :] # Extract target part

        return current_embedding
