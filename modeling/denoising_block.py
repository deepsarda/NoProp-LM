import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C


class DenoisingBlock(nn.Module):
    """
    A single, independent Denoising Block, architected as a self-contained
    Decoder Transformer. This is the fundamental processing unit of the
    NoProp-LM.

    For each token in the target sequence, this block computes a unique,
    causally-correct context from the source sequence using cross-attention.
    Stacking multiple layers allows for more complex feature extraction and
    transformation at each denoising step.

    This block's role is to perform one step of a denoising process in parallel
    across an entire sequence. It takes two inputs:
    1.  `src_embeds`: A sequence of clean context embeddings, processed by the encoder.
    2.  `tgt_embeds`: A corresponding sequence of noisy target embeddings, processed
        by the decoder.
    """

    def __init__(self, vocab_size: int, config: C):
        """
        Initializes the DenoisingBlock.

        Args:
            vocab_size (int): The size of the vocabulary.
            config (C): The global configuration object, which specifies
                        BLOCK_NUM_DECODER_LAYERS among other parameters.
        """
        super().__init__()
        self.config = config

        decoder_layer_template = nn.TransformerDecoderLayer(
            d_model=config.BLOCK_DIM,
            nhead=config.BLOCK_NHEAD,
            dim_feedforward=config.BLOCK_DIM_FEEDFORWARD,
            dropout=config.BLOCK_DROPOUT_RATE,
            activation="gelu",
            batch_first=True,
            layer_norm_eps=config.LAYER_NORM_EPS,
        )

        self.decoder_stack = nn.TransformerDecoder(
            decoder_layer=decoder_layer_template,
            num_layers=config.BLOCK_NUM_DECODER_LAYERS,
        )

    def forward(
        self,
        src_embeds: torch.Tensor,
        tgt_embeds: torch.Tensor,
        src_padding_mask: torch.Tensor,
        tgt_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Performs one parallel denoising step using the stack of decoder layers.

        Args:
            src_embeds (torch.Tensor): Clean context embeddings (the "memory").
                Shape: (batch_size, source_seq_len, embedding_dim).
            tgt_embeds (torch.Tensor): Noisy target embeddings to be denoised.
                Shape: (batch_size, target_seq_len, embedding_dim).
            src_padding_mask (torch.Tensor): Mask for padding in the source.
                Shape: (batch_size, source_seq_len).
            tgt_padding_mask (torch.Tensor): Mask for padding in the target.
                Shape: (batch_size, target_seq_len).

        Returns:
            torch.Tensor: The predicted noise for the target embeddings.
                Shape: (batch_size, target_seq_len, embedding_dim).
        """
        tgt_len = tgt_embeds.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(
            tgt_embeds.device
        )

        predicted_noise = self.decoder_stack(
            tgt=tgt_embeds,
            memory=src_embeds,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )

        return predicted_noise
