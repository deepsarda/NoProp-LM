import torch
import torch.nn as nn

import config as C


class MLMPretrainModel(nn.Module):
    """
    A Transformer Encoder model tailored for Masked Language Model (MLM) pretraining.

    This model is not the main NoProp-LM itself but serves a crucial preliminary step:
    learning robust token embeddings. It follows a standard BERT-like architecture
    where input sequences have some tokens masked, and the model tries to predict
    these masked tokens.

    The architecture consists of:
    1.  Token Embedding Layer: Maps input token IDs to dense vector representations.
    2.  Positional Embedding Layer: Adds information about token positions in the sequence.
    3.  Layer Normalization and Dropout: Applied to the sum of token and positional embeddings.
    4.  Transformer Encoder: A stack of Transformer Encoder layers that process the
        sequence of embeddings, allowing tokens to attend to each other.
    5.  LM Head: A linear layer that projects the Transformer's output back to the
        vocabulary space, producing logits for each token prediction.

    After this model is trained on an MLM task, the weights of its `token_embedder`
    layer are typically extracted and used as the pretrained, frozen embeddings for
    the main `LanguageModel` (NoProp-LM).
    """

    def __init__(self, vocab_size: int, config: C):
        """
        Initializes the MLMPretrainModel.

        Args:
            vocab_size (int): The total number of unique tokens in the vocabulary.
            config (C): The global configuration object, providing hyperparameters
                such as embedding dimension (`BLOCK_DIM`), maximum sequence length
                (`MAX_SEQ_LENGTH`), number of attention heads (`BLOCK_NHEAD`),
                feedforward dimension (`BLOCK_DIM_FEEDFORWARD`), dropout rate
                (`BLOCK_DROPOUT_RATE`), and number of encoder layers
                (`BLOCK_NUM_ENCODER_LAYERS`).
        """
        super().__init__()
        self.config = config

        # Token embedding layer: maps token IDs to dense vectors.
        # The dimension of these embeddings is `config.BLOCK_DIM`.
        self.token_embedder = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=config.BLOCK_DIM
        )

        # Positional embedding layer: provides positional information for each token.
        # It has `config.MAX_SEQ_LENGTH` possible positions, each with `config.BLOCK_DIM`.
        self.position_embedder = nn.Embedding(
            num_embeddings=config.MAX_SEQ_LENGTH,
            embedding_dim=config.BLOCK_DIM,
        )
        # Register `position_ids` as a buffer. This creates a tensor `(0, 1, ..., MAX_SEQ_LENGTH-1)`
        # that is part of the model's state but not a learnable parameter.
        # Shape: (1, MAX_SEQ_LENGTH) for easy broadcasting.
        self.register_buffer(
            "position_ids", torch.arange(config.MAX_SEQ_LENGTH).expand((1, -1))
        )

        # Layer Normalization applied after summing token and positional embeddings.
        self.layernorm = nn.LayerNorm(config.BLOCK_DIM, eps=config.LAYER_NORM_EPS)
        # Dropout layer applied to the normalized embeddings.
        self.dropout = nn.Dropout(config.BLOCK_DROPOUT_RATE)

        # Configure a single Transformer Encoder Layer.
        # Parameters are sourced from the global config `C`.
        # `batch_first=True` means input/output tensors are (batch, seq, feature).
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.BLOCK_DIM,  # Dimensionality of embeddings
            nhead=config.BLOCK_NHEAD,  # Number of attention heads
            dim_feedforward=config.BLOCK_DIM_FEEDFORWARD,  # Dim of feed-forward network
            dropout=config.BLOCK_DROPOUT_RATE,  # Dropout rate
            activation="gelu",  # Activation function
            batch_first=True,  # Input format: (batch, seq_len, embedding_dim)
        )
        # Stack multiple encoder layers to form the full Transformer Encoder.
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.BLOCK_NUM_PRETRAIN_LAYERS
        )

        # Language Model Head: A linear layer that maps the Transformer's output
        # (of dimension `config.BLOCK_DIM`) back to the vocabulary size.
        # This produces logits for each token in the vocabulary.
        self.lm_head = nn.Linear(config.BLOCK_DIM, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass for MLM pretraining.

        Args:
            input_ids (torch.Tensor): A tensor of token IDs. Some of these tokens
                are expected to be [MASK] tokens that the model needs to predict.
                Shape: (batch_size, sequence_length).

        Returns:
            torch.Tensor: A tensor of logits representing the model's prediction for
                each token position over the entire vocabulary.
                Shape: (batch_size, sequence_length, vocab_size).
        """
        # Get the actual sequence length from the input_ids tensor.
        # This allows handling batches with sequences shorter than MAX_SEQ_LENGTH.
        seq_len = input_ids.size(1)

        # 1. Get token embeddings for the input IDs.
        token_embeds = self.token_embedder(input_ids)  # Shape: (B, SeqLen, EmbDim)

        # 2. Get positional embeddings for the sequence.
        # We use the pre-registered `position_ids` buffer, sliced to the current `seq_len`.
        pos_embeds = self.position_embedder(
            self.position_ids[:, :seq_len]
        )  # Shape: (B, SeqLen, EmbDim)

        # 3. Combine token and positional embeddings, then apply LayerNorm and Dropout.
        embeddings = token_embeds + pos_embeds
        embeddings = self.layernorm(embeddings)
        embeddings = self.dropout(embeddings)  # Shape: (B, SeqLen, EmbDim)

        # 4. Pass the combined embeddings through the Transformer Encoder.
        # The encoder processes the sequence, allowing tokens to interact via self-attention.
        # Causal masking is NOT used here, as MLM allows bidirectional context.
        encoder_output = self.transformer_encoder(
            embeddings
        )  # Shape: (B, SeqLen, EmbDim)

        # 5. Pass the Transformer's output through the LM head to get logits.
        # This projects the embeddings back into the vocabulary space.
        logits = self.lm_head(encoder_output)  # Shape: (B, SeqLen, VocabSize)

        return logits
