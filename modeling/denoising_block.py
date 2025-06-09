import torch
import torch.nn as nn

import config as C


class DenoisingBlock(nn.Module):
    """
    A single, independent Denoising Block, which forms the fundamental processing
    unit of the NoProp-LM architecture.

    Each DenoisingBlock is implemented as a stack of Transformer Encoder layers.
    Its primary role is to take a sequence of potentially noisy embeddings and
    predict a "denoised" or cleaner version of those embeddings. In the context
    of NoProp-LM, this block processes a sequence comprising clean context token
    embeddings followed by one or more noisy target token embeddings.

    The block operates with causal attention, meaning that the prediction for a
    token at a given position can only depend on tokens at preceding positions
    and the input at the current position. This is crucial for autoregressive
    generation and for maintaining the flow of information during training.

    Attributes:
        config (C): The global configuration object.
        vocab_size (int): The size of the vocabulary.
        transformer_encoder (nn.TransformerEncoder): The stack of Transformer Encoder
            layers that performs the main computation.
    """

    def __init__(self, vocab_size: int, config: C):
        """
        Initializes the DenoisingBlock.

        Args:
            vocab_size (int): The size of the vocabulary. While not directly used in
                this block's architecture (as it operates on embeddings), it's
                passed for consistency with other model components and future-proofing.
            config (C): The global configuration object, which provides hyperparameters
                for the Transformer Encoder layers, such as embedding dimension,
                number of heads, feedforward dimension, dropout rate, and number of layers.
        """
        super().__init__()

        self.config = config
        self.vocab_size = vocab_size # Stored for reference, not directly used in this block

        # Configure a single Transformer Encoder Layer.
        # - d_model: The dimensionality of the input/output embeddings (BLOCK_DIM).
        # - nhead: The number of attention heads in the multi-head attention mechanism.
        # - dim_feedforward: The dimension of the feed-forward network model.
        # - dropout: The dropout rate.
        # - activation: The activation function (GELU).
        # - batch_first=True: Input and output tensors are (batch, seq, feature).
        # - layer_norm_eps: Epsilon for layer normalization.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.BLOCK_DIM,
            nhead=config.BLOCK_NHEAD,
            dim_feedforward=config.BLOCK_DIM_FEEDFORWARD,
            dropout=config.BLOCK_DROPOUT_RATE,
            activation="gelu",
            batch_first=True,  # Expects (batch, seq, feature)
            layer_norm_eps=config.LAYER_NORM_EPS,
        )
        # Stack multiple encoder layers to form the full DenoisingBlock.
        # The number of layers is defined by BLOCK_NUM_ENCODER_LAYERS.
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.BLOCK_NUM_ENCODER_LAYERS
        )

    def forward(
        self,
        block_input_embeddings: torch.Tensor,  # Shape: (B, SeqLen, EmbDim)
    ) -> torch.Tensor:
        """
        Performs one denoising step on the input sequence of embeddings.

        The input `block_input_embeddings` typically consists of a concatenation of
        clean context embeddings and noisy target embeddings. The Transformer Encoder
        processes this sequence, applying causal self-attention.

        Args:
            block_input_embeddings (torch.Tensor): A tensor of shape
                (batch_size, sequence_length, embedding_dimension) representing
                the input sequence to be denoised.

        Returns:
            torch.Tensor: A tensor of the same shape as the input, representing
                the block's prediction for the cleaner (denoised) embeddings
                for each position in the sequence.
        """

        # The TransformerEncoder processes the sequence.
        # `is_causal=True` ensures that attention is masked appropriately for
        # autoregressive behavior, meaning a position can only attend to itself
        # and previous positions.
        denoised_sequence_embeddings = self.transformer_encoder(
            src=block_input_embeddings,
            mask=None, # Not needed when is_causal=True for self-attention
            src_key_padding_mask=None, # Assuming no padding mask is explicitly needed here
            is_causal=True
        )

        return denoised_sequence_embeddings
