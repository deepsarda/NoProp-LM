import torch
import torch.nn as nn

import config as C


class DenoisingBlock(nn.Module):
    """
    A single, independent Denoising Block, which is the fundamental unit of NoProp-LM.
    This module takes a sequence of embeddings (context + noisy targets) and
    denoises all target positions in parallel.
    """

    def __init__(self, vocab_size: int, config: C):
        super().__init__()

        self.config = config
        self.vocab_size = vocab_size

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.BLOCK_DIM,
            nhead=config.BLOCK_NHEAD,
            dim_feedforward=config.BLOCK_DIM_FEEDFORWARD,
            dropout=config.BLOCK_DROPOUT_RATE,
            activation="gelu",
            batch_first=True,
            layer_norm_eps=config.LAYER_NORM_EPS,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.BLOCK_NUM_ENCODER_LAYERS
        )

        self.register_buffer("causal_mask", torch.empty(0))

    def forward(
        self,
        block_input_embeddings: torch.Tensor,  # Shape: (B, SeqLen, EmbDim)
    ) -> torch.Tensor:
        """
        Performs one parallel denoising step on a sequence.

        Args:
            block_input_embeddings: The combined embeddings of clean context and noisy targets.

        Returns:
            denoised_sequence_embeddings: The block's prediction for the clean embeddings.
                                          Shape: (B, SeqLen, EmbDim)
        """

        # The TransformerEncoderLayer processes the sequence in parallel,
        # respecting the causal mask to only look at past positions.
        denoised_sequence_embeddings = self.transformer_encoder(
            src=block_input_embeddings, is_causal=True
        )

        return denoised_sequence_embeddings
