import torch
import torch.nn as nn

import config as C


class MLMPretrainModel(nn.Module):
    """
    A standard Transformer Encoder model for Masked Language Model (MLM) pretraining.
    Its sole purpose is to learn high-quality token embeddings. After training,
    its embedding layer weights will be extracted and frozen for the main NoProp-LM.
    """

    def __init__(self, vocab_size: int, config: C):
        super().__init__()
        self.config = config

        self.token_embedder = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=config.BLOCK_DIM
        )

        self.position_embedder = nn.Embedding(
            num_embeddings=config.MAX_SEQ_LENGTH,
            embedding_dim=config.BLOCK_DIM,
        )
        self.register_buffer(
            "position_ids", torch.arange(config.MAX_SEQ_LENGTH).expand((1, -1))
        )

        self.layernorm = nn.LayerNorm(config.BLOCK_DIM, eps=config.LAYER_NORM_EPS)
        self.dropout = nn.Dropout(config.BLOCK_DROPOUT_RATE)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.BLOCK_DIM,
            nhead=config.BLOCK_NHEAD,
            dim_feedforward=config.BLOCK_DIM_FEEDFORWARD,
            dropout=config.BLOCK_DROPOUT_RATE,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.BLOCK_NUM_ENCODER_LAYERS
        )

        self.lm_head = nn.Linear(config.BLOCK_DIM, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for MLM.
        Args:
            input_ids: Token IDs, where some tokens have been replaced by [MASK].
                       Shape: (B, SeqLen)
        Returns:
            logits: Logits for each token in the vocabulary. Shape: (B, SeqLen, VocabSize)
        """
        seq_len = input_ids.size(1)

        token_embeds = self.token_embedder(input_ids)
        pos_embeds = self.position_embedder(self.position_ids[:, :seq_len])

        embeddings = self.layernorm(token_embeds + pos_embeds)
        embeddings = self.dropout(embeddings)

        encoder_output = self.transformer_encoder(embeddings)
        logits = self.lm_head(encoder_output)

        return logits
