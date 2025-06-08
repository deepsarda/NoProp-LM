from typing import Any, Dict

import wandb

from modeling.language_model import LanguageModel


def get_config_dict(config_module: Any) -> Dict[str, Any]:
    config_dict = {}
    for attr in dir(config_module):
        if not attr.startswith("__") and not callable(getattr(config_module, attr)):
            config_dict[attr] = getattr(config_module, attr)
    return config_dict


def init_wandb_run(config_module: Any, project_name: str, run_name: str, job_type: str):
    if not project_name:
        print("W&B project name not set. Disabling logging.")
        return None
    try:
        run = wandb.init(
            project=project_name,
            name=run_name,
            config=get_config_dict(config_module),
            job_type=job_type,
            reinit=True,
        )
        print(f"W&B run '{run.name}' initialized in project '{project_name}'.")
        return run
    except Exception as e:
        print(f"Could not initialize W&B: {e}. Logging disabled.")
        return None


def log_model_parameters(model, wandb_run):
    if not wandb_run:
        return

    total_params = sum(p.numel() for p in model.parameters())
    print("Model has total parameters:", total_params)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model has trainable parameters:", trainable_params)
    wandb_run.config.update(
        {"total_params": total_params, "trainable_params": trainable_params}
    )

    if isinstance(model, LanguageModel):
        block_params = sum(p.numel() for p in model.denoising_blocks[0].parameters())
        print("Each Denoising Block has total parameters:", block_params)
        wandb_run.config.update(
            {
                "params_per_denoising_block": block_params,
                "total_params_all_blocks": block_params * len(model.denoising_blocks),
            }
        )
