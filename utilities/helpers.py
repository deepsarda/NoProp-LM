import math
import os
import random

import torch

import config as C


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
        torch.cuda.manual_seed(seed_value)  # Seed for current GPU
        torch.cuda.manual_seed_all(
            seed_value
        )  # Seed for all GPUs if using multi-GPU setup

    print(f"Global random seed set to {seed_value}")


def get_alpha_squared_from_cosine_schedule(block_idx: int, config: C) -> float:
    """
    Calculates the alpha_squared value for a given block index based on a
    numerically stable and CLAMPED cosine noise schedule.
    """
    if block_idx < 0:
        return 1.0

    total_blocks = config.NUM_DENOISING_BLOCKS

    # Use the standard time mapping from 0 to T
    t = total_blocks - block_idx
    # This correctly maps t=0..T to a ratio from 0..1
    t_ratio = t / (total_blocks)
    alpha_squared = math.cos((t_ratio) * math.pi / 2) ** 2
    return alpha_squared
