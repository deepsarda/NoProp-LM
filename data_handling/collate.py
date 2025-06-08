from typing import Dict, List

import torch
import torch.nn.utils.rnn as rnn_utils
from transformers import PreTrainedTokenizerBase

import config as C


class Collate:
    """
    A custom collate function for the NoProp-LM, designed for parallel, GPT-style training.
    It transforms a batch of tokenized sequences into a batch of
    (input_ids, labels) pairs suitable for parallel denoising.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        """
        Args:
            tokenizer: The tokenizer used for its padding token ID.
        """
        if tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must have a pad_token_id.")
        self.pad_token_id = tokenizer.pad_token_id
        self.ignore_index = C.IGNORE_INDEX

    def __call__(
        self, batch_of_items: List[Dict[str, List[int]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Processes a list of tokenized sequences to create input and label sequences.
        For a sequence `[t1, t2, t3, t4]`:
        - input_ids: `[t1, t2, t3]`
        - labels:    `[t2, t3, t4]`

        Args:
            batch_of_items: A list of dictionaries, like [{'input_ids': [101, 2054, ...]}, ...]

        Returns:
            A dictionary containing a batch of padded input_ids and labels.
        """
        input_sequences = []
        label_sequences = []

        for item in batch_of_items:
            # The full sequence from the dataset
            full_sequence = torch.tensor(item["input_ids"], dtype=torch.long)

            # A sequence must have at least 2 tokens to create a valid pair
            if len(full_sequence) < 2:
                continue

            # Input is everything except the last token
            input_sequences.append(full_sequence[:-1])
            # Labels are everything except the first token (shifted)
            label_sequences.append(full_sequence[1:])

        if not input_sequences:
            return {
                "input_ids": torch.empty(0, 0, dtype=torch.long),
                "labels": torch.empty(0, 0, dtype=torch.long),
            }

        # Pad all sequences in the batch to the same length
        padded_inputs = rnn_utils.pad_sequence(
            input_sequences, batch_first=True, padding_value=self.pad_token_id
        )
        padded_labels = rnn_utils.pad_sequence(
            label_sequences, batch_first=True, padding_value=self.ignore_index
        )

        return {
            "input_ids": padded_inputs,
            "labels": padded_labels,
        }
