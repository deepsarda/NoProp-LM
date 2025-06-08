from typing import Optional, Union

from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase

import config as C


def load_tokenized_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_name: str = C.DATASET_NAME,
    dataset_config_name: Optional[str] = C.DATASET_CONFIG_NAME,
    max_length: int = C.MAX_SEQ_LENGTH,
    split: str = "train",
    num_preprocessing_workers: int = 3,
    min_text_length: int = 25,
) -> Optional[Dataset]:
    """
    Loads a dataset, filters it, tokenizes it, and returns the tokenized dataset object.

    Args:
        tokenizer: The tokenizer to use.
        dataset_name: Name of the dataset from Hugging Face datasets library.
        dataset_config_name: Specific configuration of the dataset.
        max_length: Maximum sequence length for tokenization.
        split: The dataset split to load (e.g., "train", "validation", "test").
        num_preprocessing_workers: Number of processes to use for dataset mapping.
        min_text_length: Minimum character length for a text sample to be included.

    Returns:
        A Hugging Face `Dataset` object containing tokenized sequences, or None if loading fails.
    """
    print(
        f"Attempting to load dataset: {dataset_name}, config: {dataset_config_name}, split: {split}"
    )
    try:
        raw_dataset: Union[Dataset, None] = load_dataset(
            dataset_name, dataset_config_name, split=split
        )
    except Exception as e:
        print(
            f"Error loading dataset {dataset_name} with config {dataset_config_name} for split {split}: {e}"
        )
        # Fallback logic for wikitext-103-raw-v1 validation
        if (
            dataset_name == "wikitext"
            and dataset_config_name == "wikitext-103-raw-v1"
            and split == "validation"
        ):
            print(
                "Attempting to load wikitext-2 for validation as a fallback for wikitext-103-raw-v1 validation."
            )
            try:
                raw_dataset = load_dataset(
                    dataset_name, "wikitext-2-raw-v1", split="validation"
                )
            except Exception as e2:
                print(f"Fallback dataset loading failed: {e2}")
                return None
        else:
            return None

    if not raw_dataset or len(raw_dataset) == 0:
        print(
            f"Dataset {dataset_name} (config: {dataset_config_name}, split: {split}) is empty after initial load."
        )
        return None

    # Filter out very short texts
    filtered_dataset = raw_dataset.filter(
        lambda x: len(x.get("text", "").strip()) > min_text_length,
        num_proc=num_preprocessing_workers,
    )

    if len(filtered_dataset) == 0:
        print(
            f"Dataset {dataset_name} (config: {dataset_config_name}, split: {split}) is empty after filtering."
        )
        return None
    print(f"Dataset size after filtering: {len(filtered_dataset)} samples.")

    def tokenize_function(examples):
        # Tokenize without padding. The collate function will handle padding.
        tokenized_output = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            return_attention_mask=False,
        )
        return tokenized_output

    print(
        f"Tokenizing dataset with max_length: {max_length} using {num_preprocessing_workers} workers..."
    )
    tokenized_dataset = filtered_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=raw_dataset.column_names,
        num_proc=num_preprocessing_workers,
        load_from_cache_file=True,
        desc=f"Tokenizing {split} split",
    )

    if len(tokenized_dataset) == 0:
        print(
            f"Dataset {dataset_name} (config: {dataset_config_name}, split: {split}) is empty after tokenization."
        )
        return None

    # Return the memory-mapped dataset object directly.
    # The DataLoader will handle pulling items and sending them to the collate_fn.
    print(
        f"Successfully created tokenized dataset with {len(tokenized_dataset)} samples."
    )
    return tokenized_dataset
