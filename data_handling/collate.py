from typing import Dict, List

import torch
import torch.nn.utils.rnn as rnn_utils
from transformers import PreTrainedTokenizerBase

import config as C


class Collate:
    """
    A custom collate function for the NoProp-LM.

    This class is responsible for processing a batch of tokenized sequences from the dataset
    and preparing them for input to the language model. It creates `input_ids`
    and corresponding `labels` for GPT-style next-token prediction.

    Key responsibilities:
    - Generating input_ids (sequence up to the second to last token).
    - Generating labels (sequence from the second token to the end).
    - Padding sequences within a batch to the same length.
    - Using a specific `ignore_index` for padding tokens in the labels.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        """
        Initializes the Collate class.

        Args:
            tokenizer (PreTrainedTokenizerBase): The tokenizer instance.
                It is used to access the `pad_token_id`.
                The tokenizer must have `pad_token_id` defined.

        Raises:
            ValueError: If the provided tokenizer does not have a `pad_token_id`.
        """
        if tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must have a pad_token_id defined.")
        self.pad_token_id = tokenizer.pad_token_id  # ID of the padding token
        self.ignore_index = C.IGNORE_INDEX  # Value to ignore in loss calculation for labels

    def __call__(
        self, batch_of_items: List[Dict[str, List[int]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Processes a list of tokenized sequences (from the dataset) and prepares
        a batch of `input_ids` and `labels` tensors for language model training.

        For each sequence in the batch (e.g., `[t1, t2, t3, t4]`):
        - `input_ids` are created as `[t1, t2, t3]` (all tokens except the last).
        - `labels` are created as `[t2, t3, t4]` (all tokens except the first, shifted).

        Sequences shorter than 2 tokens are filtered out.
        The resulting `input_ids` and `labels` sequences are then padded to the
        maximum length in the batch. `input_ids` are padded with `pad_token_id`,
        and `labels` are padded with `ignore_index`.

        Args:
            batch_of_items (List[Dict[str, List[int]]]): A list of items, where each item
                is a dictionary expected to have an "input_ids" key.
                Example: `[{'input_ids': [101, 2054, 2023, ...]}, ...]`

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing two tensors:
                - "input_ids": A LongTensor of shape (batch_size, max_seq_len_in_batch - 1)
                               containing the padded input sequences.
                - "labels": A LongTensor of shape (batch_size, max_seq_len_in_batch - 1)
                            containing the padded label sequences.
                If all sequences are too short, empty tensors are returned.
        """
        input_sequences = []  # List to store processed input sequences
        label_sequences = []  # List to store processed label sequences

        for item in batch_of_items:
            # Extract the full tokenized sequence from the item
            full_sequence = torch.tensor(item["input_ids"], dtype=torch.long)

            # A sequence must have at least 2 tokens to create a valid input/label pair.
            # For example, if sequence is `[t1]`, input would be `[]` and label would be `[]`.
            if len(full_sequence) < 2:
                continue  # Skip this item if it's too short

            # Create input_ids: all tokens except the last one
            input_sequences.append(full_sequence[:-1])
            # Create labels: all tokens except the first one (shifted relative to inputs)
            label_sequences.append(full_sequence[1:])

        # If all items were filtered out (e.g., all sequences were too short)
        if not input_sequences:
            # Return empty tensors with appropriate shape (0, 0) to avoid errors downstream
            return {
                "input_ids": torch.empty(0, 0, dtype=torch.long),
                "labels": torch.empty(0, 0, dtype=torch.long),
            }

        # Pad all input sequences in the batch to the same length.
        # `batch_first=True` means the output tensor will have shape (batch_size, seq_len).
        # `padding_value` is used to fill shorter sequences.
        padded_inputs = rnn_utils.pad_sequence(
            input_sequences, batch_first=True, padding_value=self.pad_token_id
        )
        # Pad all label sequences similarly.
        # `ignore_index` is used for padding in labels so that these positions
        # do not contribute to the loss calculation.
        padded_labels = rnn_utils.pad_sequence(
            label_sequences, batch_first=True, padding_value=self.ignore_index
        )

        # Return the batch as a dictionary of tensors
        return {
            "input_ids": padded_inputs,
            "labels": padded_labels,
        }
