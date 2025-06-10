import math
import os
import random
from typing import Dict

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
        torch.cuda.manual_seed(seed_value)  # Seed for current GPU
        torch.cuda.manual_seed_all(
            seed_value
        )  # Seed for all GPUs if using multi-GPU setup

    print(f"Global random seed set to {seed_value}")


def get_alpha_squared_from_cosine_schedule(block_idx: int, config: C) -> float:
    """
    Calculates the alpha_squared value for a given block index based on a
    cosine noise schedule.

    This schedule is **reversed** for training. Block 0, which is processed first
    during inference and sees the most noise, is trained with the highest noise
    (lowest alpha_squared). The final block is trained with the lowest noise.
    """
    if block_idx < 0:
        return 1.0

    total_blocks = config.NUM_DENOISING_BLOCKS

    # We map the block index to the noise schedule in reverse.
    # block_idx = 0  -> should have MOST noise -> corresponds to the END of the schedule
    # block_idx = N-1 -> should have LEAST noise -> corresponds to the START of the schedule
    t_paper = total_blocks - block_idx

    t_ratio = t_paper / total_blocks
    return math.cos(t_ratio * math.pi / 2) ** 2


def get_ddpm_schedule(
    T: int, config: C, device: torch.device
) -> Dict[str, torch.Tensor]:
    """
    Pre-computes the full noise schedule and coefficients required for DDPM sampling.
    """
    # Note: This schedule is built based on block_idx from 0 to T-1, following
    # the reversed schedule from get_alpha_squared_from_cosine_schedule.
    alphas_squared = torch.tensor(
        [get_alpha_squared_from_cosine_schedule(i, config) for i in range(T)],
        device=device,
        dtype=torch.float32,
    )
    alphas_squared_prev = torch.cat(
        [torch.tensor([1.0], device=device), alphas_squared[:-1]]
    )

    alphas = alphas_squared / (alphas_squared_prev + 1e-8)
    betas = 1.0 - alphas

    posterior_variance = (
        betas * (1.0 - alphas_squared_prev) / (1.0 - alphas_squared + 1e-8)
    )
    # The variance for the first block (highest noise) should be handled carefully.
    # Clamping to avoid instability from alphas_squared[0] being near zero.
    posterior_variance[0] = betas[0]

    # `posterior_mean_coef1` is used for the current noisy embedding (x_t).
    # `posterior_mean_coef2` is used for the predicted clean embedding (x_0).
    # Coefficient for x_t (current noisy embedding)
    posterior_mean_coef1 = (
        (1.0 - alphas_squared_prev) * torch.sqrt(alphas) / (1.0 - alphas_squared + 1e-8)
    )
    # Coefficient for x_0 (predicted clean embedding)
    posterior_mean_coef2 = (
        betas * torch.sqrt(alphas_squared_prev) / (1.0 - alphas_squared + 1e-8)
    )

    return {
        "alphas": alphas,
        "betas": betas,
        "alphas_squared": alphas_squared,
        "alphas_squared_prev": alphas_squared_prev,
        "posterior_variance": posterior_variance,
        "posterior_mean_coef1": posterior_mean_coef1,
        "posterior_mean_coef2": posterior_mean_coef2,
    }
