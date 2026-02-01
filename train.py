import math
import random
from datetime import timedelta
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
import tyro
import wandb
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from pape.configs import Checkpoint
from pape.configs import Config
from pape.configs import Dataset
from pape.configs import EventsConfig
from pape.configs import OptimizerConfig
from pape.configs import TrainConfig
from pape.constants import WANDB_PROJECT
from pape.data import load_datamodule
from pape.models import load_model
from pape.paths import get_experiment_dir


def train(config: Config):
    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

    torch.set_float32_matmul_precision("high")

    if config.debug:
        print("Running in debug mode. Model weights will not be saved.")

    print(f"Position encoder: {config.positional.value}")

    datamodule = load_datamodule(config)
    datamodule.prepare_data()
    datamodule.setup("fit")
    train_loader = datamodule.train_dataloader()
    steps_per_epoch = math.ceil(len(train_loader) / config.train.acc_gradients)

    lightning_model = load_model(config, steps_per_epoch=steps_per_epoch)

    run_id, checkpoint_path = load_checkpoint(config.name)

    if config.debug:
        callbacks = []
        logger = False
    else:
        logger = WandbLogger(
            config=config.to_dict(),
            group=config.group,
            id=run_id,
            log_model=False,
            name=config.name,
            project=WANDB_PROJECT,
            tags=TAGS[config.dataset],
            settings=wandb.Settings(_disable_stats=True),
        )

        name = config.name
        if name is None:
            name = logger.experiment.name

        experiment_dir = get_experiment_dir(name, raise_missing=False)
        experiment_dir.mkdir(exist_ok=True)
        run_id = logger.experiment.id
        (experiment_dir / "run_id.txt").write_text(run_id)

        checkpoint_kwargs = {}
        if config.train.ckpt_n_hour is not None:
            checkpoint_kwargs["train_time_interval"] = timedelta(hours=config.train.ckpt_n_hour)

        callbacks = [
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(
                dirpath=experiment_dir / "checkpoints",
                enable_version_counter=False,
                filename=Checkpoint.best.value,
                mode=MONITOR_MODE[config.dataset],
                monitor=MONITOR_METRIC[config.dataset],
                save_last=True,
                save_top_k=1,
                save_weights_only=False,
                **checkpoint_kwargs,
            ),
        ]

    if not config.validate:
        config.train.limit_val_batches = 0

    trainer = pl.Trainer(
        accumulate_grad_batches=config.train.acc_gradients,
        callbacks=callbacks,
        check_val_every_n_epoch=config.train.check_val_every_n_epoch,
        enable_checkpointing=not config.debug,
        enable_progress_bar=config.debug,
        gradient_clip_algorithm=config.train.gradient_clip_algorithm,
        gradient_clip_val=config.train.grad_clip_value,
        limit_train_batches=config.train.limit_train_batches,
        limit_val_batches=config.train.limit_val_batches,
        logger=logger,
        log_every_n_steps=config.train.log_every_n_step,
        max_epochs=config.epochs,
        precision=config.train.precision,
    )

    trainer.fit(lightning_model, datamodule=datamodule, ckpt_path=checkpoint_path)


def load_checkpoint(name: str | None) -> tuple[str | None, Path | None]:
    run_id = None
    checkpoint_path = None

    if name is None:
        return None, None

    experiment_dir = get_experiment_dir(name, raise_missing=False)
    if experiment_dir.exists():
        run_id_path = experiment_dir / "run_id.txt"
        if run_id_path.exists():
            run_id = run_id_path.read_text()

        potential_checkpoint_path = experiment_dir / "checkpoints" / "last.ckpt"
        if potential_checkpoint_path.exists():
            checkpoint_path = potential_checkpoint_path

    return run_id, checkpoint_path


MONITOR_METRIC = {
    Dataset.coco: "val/map",
    Dataset.dvsgesture: "val/accuracy",
    Dataset.gen1: "val/map",
    Dataset.imagenet: "val/accuracy",
    Dataset.ucf101: "val/accuracy",
}

MONITOR_MODE = {
    Dataset.coco: "max",
    Dataset.dvsgesture: "max",
    Dataset.gen1: "max",
    Dataset.imagenet: "max",
    Dataset.ucf101: "max",
}


TAGS = {
    Dataset.coco: ["coco"],
    Dataset.dvsgesture: ["dvsgesture"],
    Dataset.gen1: ["gen1"],
    Dataset.imagenet: ["imagenet"],
    Dataset.ucf101: ["ucf101"],
}

CONFIGS = {
    "coco": (
        "COCO",
        Config(
            dataset=Dataset.coco,
            batch_size=64,
            compile=True,
            epochs=150,
            optimizer=OptimizerConfig(lr=6e-4, final_lr=0, warmup_ratio=0.1),
            train=TrainConfig(log_every_n_step=100),
        ),
    ),
    "dvsgesture": (
        "DvsGesture",
        Config(
            dataset=Dataset.dvsgesture,
            batch_size=32,
            events=EventsConfig(ref_ms=100),
            epochs=300,
            optimizer=OptimizerConfig(lr=5e-5, final_lr=5e-7),
            train=TrainConfig(log_every_n_step=10),
        ),
    ),
    "gen1": (
        "GEN1",
        Config(
            dataset=Dataset.gen1,
            batch_size=64,
            events=EventsConfig(ref_ms=100, time_scale=100_000),
            epochs=150,
            train=TrainConfig(log_every_n_step=100, precision="bf16-mixed"),
        ),
    ),
    "imagenet": (
        "ImageNet1k",
        Config(
            dataset=Dataset.imagenet,
            batch_size=1024,
            compile=True,
            epochs=300,
            optimizer=OptimizerConfig(lr=6e-4, final_lr=0, warmup_ratio=0.05),
            train=TrainConfig(log_every_n_step=100),
            valid_ratio=0.01,
        ),
    ),
    "ucf101": (
        "UCF101",
        Config(
            dataset=Dataset.ucf101,
            batch_size=32,
            compile=True,
            epochs=200,
            optimizer=OptimizerConfig(lr=3e-5, final_lr=0, warmup_ratio=0.1),
            train=TrainConfig(log_every_n_step=100),
        ),
    ),
}

if __name__ == "__main__":
    config = tyro.extras.overridable_config_cli(CONFIGS)
    train(config)
