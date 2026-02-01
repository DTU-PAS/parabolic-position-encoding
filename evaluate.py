import dataclasses

import lightning.pytorch as pl
import torch
import tyro
import wandb
from lightning.pytorch.loggers import WandbLogger

from pape.configs import Checkpoint
from pape.configs import Config
from pape.configs import Split
from pape.constants import WANDB_PROJECT
from pape.data import load_datamodule
from pape.models import load_model
from pape.paths import get_experiment_dir


def main(
    name: str,
    /,
    batch_size: int = 1,
    checkpoint: Checkpoint = Checkpoint.best,
    num_workers: int = 4,
    split: Split = Split.val,
    detection_max_det: int | None = None,
    detection_min_conf: float | None = None,
) -> None:
    torch.set_float32_matmul_precision("high")

    config, run_id, weights_path = load_checkpoint(name, checkpoint)

    overrides = {"batch_size": batch_size, "num_workers": num_workers}
    if detection_max_det is not None or detection_min_conf is not None:
        detection_overrides = {}
        if detection_max_det is not None:
            detection_overrides["max_det"] = detection_max_det
        if detection_min_conf is not None:
            detection_overrides["min_conf"] = detection_min_conf
        overrides["detection"] = dataclasses.replace(config.detection, **detection_overrides)
    config = dataclasses.replace(config, **overrides)

    datamodule = load_datamodule(config)
    lightning_model = load_model(config)

    logger = WandbLogger(
        id=run_id,
        prefix="eval",
        project=WANDB_PROJECT,
        settings=wandb.Settings(_disable_stats=True),
    )

    trainer = pl.Trainer(
        callbacks=[],
        logger=logger,
        precision=config.train.precision,
    )

    match split:
        case Split.val:
            trainer.validate(lightning_model, datamodule=datamodule, ckpt_path=weights_path)
        case Split.test:
            trainer.test(lightning_model, datamodule=datamodule, ckpt_path=weights_path)
        case _:
            raise ValueError(f"Unsupported evaluation split: {split}")


def load_checkpoint(name: str, checkpoint: Checkpoint) -> tuple[Config, str, str]:
    experiment_dir = get_experiment_dir(name)

    if not experiment_dir.exists():
        raise ValueError(f"Could not find experiment directory {experiment_dir}.")

    run_id = (experiment_dir / "run_id.txt").read_text()

    # get model parameters from wandb run with the given name
    api = wandb.Api()
    run = api.run(f"{WANDB_PROJECT}/{run_id}")

    config = Config.from_dict(run.config)

    checkpoints_dir = experiment_dir / "checkpoints"
    best_path = checkpoints_dir / f"{Checkpoint.best.value}.ckpt"
    last_path = checkpoints_dir / f"{Checkpoint.last.value}.ckpt"
    if not best_path.exists() and not last_path.exists():
        raise ValueError(f"Could not find any checkpoints in {checkpoints_dir}.")
    if checkpoint == Checkpoint.best:
        weights_path = best_path if best_path.exists() else last_path
    else:
        weights_path = last_path if last_path.exists() else best_path

    return config, run_id, weights_path


tyro.cli(main)
