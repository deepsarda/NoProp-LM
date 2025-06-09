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
    Loads a dataset from the Hugging Face Hub, filters it based on text length,
    and then tokenizes it.

    Args:
        tokenizer (PreTrainedTokenizerBase): The tokenizer instance to use for tokenizing
            the text.
        dataset_name (str, optional): The name of the dataset to load from the
            Hugging Face Hub. Defaults to `C.DATASET_NAME`.
        dataset_config_name (Optional[str], optional): The specific configuration of the
            dataset (e.g., 'wikitext-103-raw-v1'). Defaults to `C.DATASET_CONFIG_NAME`.
        max_length (int, optional): The maximum sequence length to truncate or pad
            sequences to during tokenization. Defaults to `C.MAX_SEQ_LENGTH`.
        split (str, optional): The dataset split to load (e.g., "train", "validation",
            "test"). Defaults to "train".
        num_preprocessing_workers (int, optional): The number of worker processes to use
            for dataset preprocessing (filtering and tokenization). Defaults to 3.
        min_text_length (int, optional): The minimum character length for a text sample
            to be included after filtering. Shorter texts are discarded. Defaults to 25.

    Returns:
        Optional[Dataset]: A Hugging Face `Dataset` object containing the tokenized
            sequences (as 'input_ids'). The original text column and other columns
            are removed. Returns `None` if the dataset cannot be loaded, is empty
            after loading, or becomes empty after filtering or tokenization.
    """
    print(
        f"Attempting to load dataset: {dataset_name}, config: {dataset_config_name}, split: {split}"
    )
    try:
        # Load the raw dataset from Hugging Face Hub
        raw_dataset: Union[Dataset, None] = load_dataset(
            dataset_name, dataset_config_name, split=split
        )
    except Exception as e:
        # Handle errors during dataset loading
        print(
            f"Error loading dataset {dataset_name} with config {dataset_config_name} for split {split}: {e}"
        )

    # Check if the dataset is empty after initial loading
    if not raw_dataset or len(raw_dataset) == 0:
        print(
            f"Dataset {dataset_name} (config: {dataset_config_name}, split: {split}) is empty after initial load."
        )
        return None

    # Filter out texts that are too short (based on character length)
    # This uses the `filter` method of the `Dataset` object, which can be parallelized.
    print(f"Filtering dataset... Original size: {len(raw_dataset)} samples.")
    filtered_dataset = raw_dataset.filter(
        lambda x: len(x.get("text", "").strip()) > min_text_length, # Filter condition
        num_proc=num_preprocessing_workers, # Number of processes for filtering
    )

    # Check if the dataset is empty after filtering
    if len(filtered_dataset) == 0:
        print(
            f"Dataset {dataset_name} (config: {dataset_config_name}, split: {split}) is empty after filtering."
        )
        return None
    print(f"Dataset size after filtering: {len(filtered_dataset)} samples.")

    # Define the tokenization function to be applied to the dataset
    def tokenize_function(examples):
        # Tokenize the 'text' field.
        # `truncation=True` ensures sequences longer than `max_length` are cut.
        # `max_length` specifies the maximum length for tokenized output.
        # `return_attention_mask=False` as it's not typically needed for GPT-style LM.
        # Padding is handled by the `Collate` function, not here.
        tokenized_output = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            return_attention_mask=False, # Attention mask is not used in this project's LM
        )
        return tokenized_output

    print(
        f"Tokenizing dataset with max_length: {max_length} using {num_preprocessing_workers} workers..."
    )
    # Apply the tokenization function to the filtered dataset.
    # `batched=True` processes multiple examples at once for efficiency.
    # `remove_columns` deletes the original columns (like 'text') after tokenization.
    # `load_from_cache_file=True` enables caching of the tokenized dataset.
    tokenized_dataset = filtered_dataset.map(
        tokenize_function,
        batched=True, 
        remove_columns=raw_dataset.column_names,
        num_proc=num_preprocessing_workers,
        load_from_cache_file=True,
        desc=f"Tokenizing {split} split",
    )

    # Check if the dataset is empty after tokenization
    if len(tokenized_dataset) == 0:
        print(
            f"Dataset {dataset_name} (config: {dataset_config_name}, split: {split}) is empty after tokenization."
        )
        return None

    # The result is a memory-mapped `Dataset` object.
    # The actual data loading into memory happens when items are accessed (e.g., by DataLoader).
    print(
        f"Successfully created tokenized dataset with {len(tokenized_dataset)} samples."
    )
    return tokenized_dataset
