from typing import Any, Dict, Optional

import torch.nn as nn
import wandb

from modeling.language_model import LanguageModel


def get_config_dict(config_module: Any) -> Dict[str, Any]:
    """
    Converts a Python module or class instance (used as a configuration object)
    into a dictionary. This dictionary can then be passed to `wandb.init()`
    to log the configuration parameters.

    It iterates over the attributes of the `config_module`, excluding dunder
    methods (like `__name__`) and callable attributes (methods).

    Args:
        config_module (Any): The configuration module or object (e.g., `config.py`
                             after being imported, or an instance of a config class).

    Returns:
        Dict[str, Any]: A dictionary where keys are the attribute names from
                        `config_module` and values are their corresponding values.
    """
    config_dict = {}
    # Iterate over all attributes of the config_module
    for attr_name in dir(config_module):
        # Exclude private/magic attributes (starting with '__')
        if not attr_name.startswith("__"):
            attr_value = getattr(config_module, attr_name)
            # Exclude callable attributes (methods)
            if not callable(attr_value):
                config_dict[attr_name] = attr_value
    return config_dict


def init_wandb_run(
    config_module: Any, project_name: str, run_name: str, job_type: str
) -> Optional[wandb.sdk.wandb_run.Run]:
    """
    Initializes and starts a new Weights & Biases run.

    If `project_name` is not provided or empty, W&B logging is disabled, and
    this function returns `None`. Otherwise, it attempts to initialize a W&B run
    with the given parameters. The configuration from `config_module` is converted
    to a dictionary and logged.

    Args:
        config_module (Any): The configuration module or object (e.g., `config.py`)
                             to be logged with W&B.
        project_name (str): The name of the W&B project to log to.
        run_name (str): The display name for this specific run in W&B.
        job_type (str): Specifies the type of run (e.g., "training", "pretraining",
                        "evaluation"). This helps in organizing runs within W&B.

    Returns:
        Optional[wandb.sdk.wandb_run.Run]: The initialized W&B run object if successful,
                                           or `None` if `project_name` is missing or
                                           if initialization fails.
    """
    if not project_name:
        print("W&B project name not set. Disabling W&B logging.")
        return None
    try:
        # Initialize W&B run
        run = wandb.init(
            project=project_name,  # Target W&B project
            name=run_name,  # Name of the run
            config=get_config_dict(config_module),  # Log configuration parameters
            job_type=job_type,  # Type of job (e.g., 'train', 'eval')
            reinit=True,  # Allows reinitializing W&B in the same process (useful for notebooks)
        )
        print(
            f"W&B run '{run.name}' initialized successfully in project '{project_name}'. Job type: '{job_type}'."
        )
        return run
    except Exception as e:
        # Catch any errors during W&B initialization (e.g., network issues, API key problems)
        print(f"Could not initialize W&B run: {e}. W&B logging disabled.")
        return None


def log_model_parameters(
    model: nn.Module, wandb_run: Optional[wandb.sdk.wandb_run.Run]
):
    """
    Logs the total and trainable parameter counts of a PyTorch model to an active
    Weights & Biases run.

    If the provided `model` is an instance of `LanguageModel` (the NoProp-LM),
    it also logs the parameter count for a single DenoisingBlock and the total
    parameters across all DenoisingBlocks.

    Args:
        model (nn.Module): The PyTorch model whose parameters are to be counted.
        wandb_run (Optional[wandb.sdk.wandb_run.Run]): The active W&B run object.
            If `None`, the function does nothing.
    """
    if not wandb_run:  # Do nothing if W&B run is not initialized
        return

    # Calculate total number of parameters in the model
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model total parameters: {total_params:,}")
    # Calculate number of trainable parameters (those requiring gradients)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model trainable parameters: {trainable_params:,}")

    # Update W&B run's configuration with these parameter counts
    wandb_run.config.update(
        {"model_total_params": total_params, "model_trainable_params": trainable_params}
    )

    # If the model is the specific NoProp LanguageModel, log block-specific parameters
    if isinstance(model, LanguageModel):
        if model.denoising_blocks:  # Check if there are any denoising blocks
            # Calculate parameters for the first denoising block (assuming all are identical)
            block_params = sum(
                p.numel() for p in model.denoising_blocks[0].parameters()
            )
            print(f"Each Denoising Block has parameters: {block_params:,}")
            total_params_all_blocks = block_params * len(model.denoising_blocks)
            print(
                f"Total parameters in all Denoising Blocks: {total_params_all_blocks:,}"
            )

            wandb_run.config.update(
                {
                    "params_per_denoising_block": block_params,
                    "total_params_all_denoising_blocks": total_params_all_blocks,
                }
            )
        else:
            print("LanguageModel has no denoising blocks to log.")
