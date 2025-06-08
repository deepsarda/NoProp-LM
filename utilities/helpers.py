"""
This file provides general helper functions that can be used across
various parts of the NoProp-LM project. Currently, it includes a function
for setting random seeds to ensure reproducibility.
"""
import os
import random

import torch


def set_seed(seed_value: int):
    """
    Sets the random seed for various libraries and modules to ensure
    reproducibility of experiments.

    This function affects:
    - Python's built-in `random` module.
    - The `PYTHONHASHSEED` environment variable, which influences hash-based
      operations (e.g., in dictionaries, sets) for string, bytes, and datetime objects.
    - PyTorch's random number generators for CPU (`torch.manual_seed`).
    - PyTorch's random number generators for CUDA (GPU), if available
      (`torch.cuda.manual_seed` for the current GPU, `torch.cuda.manual_seed_all`
      for all GPUs).
    - PyTorch's cuDNN backend settings:
        - `torch.backends.cudnn.deterministic = True`: Makes cuDNN use deterministic
          algorithms, which can be slower but are crucial for reproducibility.
        - `torch.backends.cudnn.benchmark = False`: Disables cuDNN's auto-tuner
          that selects the fastest algorithm for the current hardware. While often
          speeding up training, this can introduce non-determinism.

    Args:
        seed_value (int): The integer value to use as the seed.
    """
    # Set seed for Python's random module
    random.seed(seed_value)
    # Set PYTHONHASHSEED environment variable for hash randomization
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    # Set seed for PyTorch CPU operations
    torch.manual_seed(seed_value)

    # If CUDA (GPU support) is available, set seeds for CUDA operations
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value) # Seed for current GPU
        torch.cuda.manual_seed_all(seed_value)  # Seed for all GPUs if using multi-GPU setup

        # Configure cuDNN for deterministic behavior
        # Using deterministic algorithms can have a performance impact.
        torch.backends.cudnn.deterministic = True
        # Disabling benchmark mode ensures cuDNN doesn't pick potentially non-deterministic algorithms.
        torch.backends.cudnn.benchmark = False

    print(f"Global random seed set to {seed_value}")
