import json
from pathlib import Path

import lightning.pytorch as pl
import torch

from pape.collators import EventsClassificationCollator
from pape.configs import Config
from pape.data.dvs_gesture.dataset import Dataset
from pape.data.dvs_gesture.dataset import Sample
from pape.dvs_gesture import DATASET_NAME
from pape.dvs_gesture import VALIDATION_USERS
from pape.paths import get_dataset_dir


class DVSGestureDataModule(pl.LightningDataModule):
    def __init__(self, config: Config):
        super().__init__()
        self.batch_size = config.batch_size
        self.config = config
        self.num_workers = config.num_workers
        self.pin_memory = config.pin_memory
        self.validate = config.validate

        self.dataset_dir = get_dataset_dir(f"{DATASET_NAME}-preprocessed")

    def setup(self, stage: str):
        if stage == "fit" or stage == "validate":
            samples = self.load_samples(self.dataset_dir / "train.json")
            train, val = self.train_val_split(samples)
            self.train_dataset = Dataset(self.config, train, augment=True)
            self.val_dataset = None if val is None else Dataset(self.config, val)
        elif stage == "test":
            samples = self.load_samples(self.dataset_dir / "test.json")
            self.test_dataset = Dataset(self.config, samples)
        else:
            raise ValueError(f"Unsupported stage: {stage}")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=EventsClassificationCollator(),
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=EventsClassificationCollator(),
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=EventsClassificationCollator(),
            pin_memory=self.pin_memory,
        )

    def load_samples(self, path: Path) -> list[Sample]:
        json_samples = json.loads(path.read_text())
        samples = []
        for json_sample in json_samples:
            sample = Sample(
                label=json_sample["label"],
                path=(self.dataset_dir / json_sample["filename"]).with_suffix(".h5"),
                user=json_sample["user"],
            )
            samples.append(sample)
        return samples

    def train_val_split(self, samples: list[Sample]):
        if not self.validate:
            return samples, None

        train = [sample for sample in samples if sample.user not in VALIDATION_USERS]
        val = [sample for sample in samples if sample.user in VALIDATION_USERS]
        return train, val
