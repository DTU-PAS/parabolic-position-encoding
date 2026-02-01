import os

import lightning.pytorch as pl
import torch
import tyro

from pape.configs import Config
from pape.configs import Dataset
from pape.data import load_datamodule
from pape.models import load_model


def measure_time(config: Config):
    os.environ["ENABLE_TIMING"] = "1"

    torch.set_float32_matmul_precision("high")

    print(f"Position encoder: {config.positional.value}")

    lightning_model = load_model(config)
    datamodule = load_datamodule(config)

    trainer = pl.Trainer(
        enable_checkpointing=False,
        enable_progress_bar=True,
        limit_train_batches=config.train.limit_train_batches,
        limit_val_batches=config.train.limit_val_batches,
        limit_test_batches=5000,
        logger=False,
        precision=config.train.precision,
    )

    trainer.test(lightning_model, datamodule=datamodule)


if __name__ == "__main__":
    default_config = Config(
        dataset=Dataset.imagenet,
        batch_size=1,
        compile=True,
        epochs=1,
        valid_ratio=0.01,
    )

    config = tyro.cli(Config, default=default_config)

    measure_time(config)
