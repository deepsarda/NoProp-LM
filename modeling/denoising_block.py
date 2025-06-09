import torch
import torch.nn as nn

import config as C


class DenoisingBlock(nn.Module):
    """
    A single, independent Denoising Block, architected as a self-contained
    Encoder-Decoder Transformer. This is the fundamental processing unit of the
    NoProp-LM.

    This block's role is to perform one step of a denoising process in parallel
    across an entire sequence. It takes two inputs:
    1.  `src_embeds`: A sequence of clean context embeddings, processed by the encoder.
    2.  `tgt_embeds`: A corresponding sequence of noisy target embeddings, processed
        by the decoder.

    The decoder uses its own noisy input and the rich context from the encoder's
    output (via cross-attention) to predict a "cleaner" version of the target
    embeddings.

    Attributes:
        config (C): The global configuration object.
        vocab_size (int): The size of the vocabulary.
        transformer (nn.Transformer): The core PyTorch Transformer module that
            contains both the encoder and decoder layers.
    """

    def __init__(self, vocab_size: int, config: C):
        """
        Initializes the DenoisingBlock.

        Args:
            vocab_size (int): The size of the vocabulary. Passed for consistency
                with other model components.
            config (C): The global configuration object, providing hyperparameters
                for the Transformer module, such as embedding dimension, number of
                heads, feedforward dimension, dropout rate, and number of layers.
        """
        super().__init__()

        self.config = config
        self.vocab_size = vocab_size

        self.transformer = nn.Transformer(
            d_model=config.BLOCK_DIM,
            nhead=config.BLOCK_NHEAD,
            num_encoder_layers=config.BLOCK_NUM_ENCODER_LAYERS,
            num_decoder_layers=config.BLOCK_NUM_DECODER_LAYERS,
            dim_feedforward=config.BLOCK_DIM_FEEDFORWARD,
            dropout=config.BLOCK_DROPOUT_RATE,
            activation="gelu",
            batch_first=True,
            layer_norm_eps=config.LAYER_NORM_EPS,
        )

    def forward(
        self,
        src_embeds: torch.Tensor,
        tgt_embeds: torch.Tensor,
        src_padding_mask: torch.Tensor,
        tgt_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Performs one parallel denoising step on the target sequence, conditioned
        on the source sequence.

        Args:
            src_embeds (torch.Tensor): The clean context embeddings for the encoder.
                Shape: (batch_size, source_seq_len, embedding_dim).
            tgt_embeds (torch.Tensor): The noisy target embeddings for the decoder.
                Shape: (batch_size, target_seq_len, embedding_dim).
            src_padding_mask (torch.Tensor): A boolean mask for the source sequence,
                where `True` indicates a padding token.
                Shape: (batch_size, source_seq_len).
            tgt_padding_mask (torch.Tensor): A boolean mask for the target sequence,
                where `True` indicates a padding token.
                Shape: (batch_size, target_seq_len).

        Returns:
            torch.Tensor: The predicted cleaner embeddings from the decoder.
                Shape: (batch_size, target_seq_len, embedding_dim).
        """
        # 1. Generate a causal (subsequent) mask for the decoder's self-attention.
        # This prevents positions from attending to future positions, which is essential
        # for autoregressive tasks and maintaining the proper flow of information.
        # The mask needs to be on the same device as the input tensors.
        tgt_len = tgt_embeds.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(
            tgt_embeds.device
        )

        # 2. Pass all inputs to the nn.Transformer module.
        #    - src: The context for the encoder.
        #    - tgt: The noisy sequence for the decoder to denoise.
        #    - tgt_mask: The causal mask for the decoder.
        #    - src_key_padding_mask: Prevents attention over padded tokens in the encoder.
        #    - tgt_key_padding_mask: Prevents attention over padded tokens in the decoder.
        #    - memory_key_padding_mask: This is crucial. It prevents the decoder's
        #      cross-attention from attending to padded parts of the encoder's output (memory).
        #      It must be the same as the source padding mask.
        denoised_sequence_embeddings = self.transformer(
            src=src_embeds,
            tgt=tgt_embeds,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )

        return denoised_sequence_embeddings
