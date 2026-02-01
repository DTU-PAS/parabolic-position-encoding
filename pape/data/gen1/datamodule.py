import lightning.pytorch as pl
import numpy as np
import torch

from pape.collators import EventDetectionCollator
from pape.configs import Config
from pape.configs import Split
from pape.data.gen1.dataset import Gen1Dataset
from pape.data.gen1.dataset import Sample
from pape.gen1 import DATASET_NAME
from pape.io import Sequence
from pape.io import load_sequences
from pape.paths import get_dataset_dir


class Gen1DataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: Config,
        augment: bool = True,
        pin_memory: bool = False,
    ):
        super().__init__()

        self.augment = augment
        self.batch_size = config.batch_size
        self.config = config
        self.num_workers = config.num_workers
        self.pin_memory = pin_memory
        self.prepare_data_per_node = True
        self.preprocessed_dir = get_dataset_dir(f"{DATASET_NAME}-preprocessed")

    def setup(self, stage: str):
        match stage:
            case "fit":
                self.train_dataset = self.load_dataset(Split.train, augment=self.augment)
                self.val_dataset = self.load_dataset(Split.val)
            case "validate":
                self.val_dataset = self.load_dataset(Split.val)
            case "test":
                self.test_dataset = self.load_dataset(Split.test)
            case _:
                raise ValueError(f"Unsupported stage: {stage}")

    def load_dataset(self, split: Split, augment: bool = False):
        sequences = load_sequences(self.preprocessed_dir / split.value)
        samples = self.load_samples(split, sequences)
        return Gen1Dataset(
            augment=augment,
            config=self.config,
            samples=samples,
            sequences=sequences,
        )

    def load_samples(self, split: Split, sequences: list[Sequence]) -> list[Sample]:
        samples = []
        for sequence_index in range(len(sequences)):
            sequence = sequences[sequence_index]
            path = self.preprocessed_dir / split.value / sequence.name / "bbox.npy"
            labels = np.load(path)

            labels = np.sort(labels, order="t", stable=True)
            _, unique_indices = np.unique(labels["t"], return_index=True)
            labels = np.split(labels, unique_indices)  # assumes sorted labels by time (so don't remove sort above)

            # some sequences may not have any labels due to preprocessing
            if len(labels) == 0:
                return {}

            labels = labels[1:]  # skip the first empty label group

            chunk_index = 0
            for label_group in labels:
                time = label_group[0]["t"].item()
                while chunk_index < len(sequence.chunks) and sequence.chunks[chunk_index].end_time_us < time:
                    chunk_index += 1
                samples.append(Sample(sequence=sequence_index, chunk=chunk_index, time=time, labels=label_group))

        return samples

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            batch_size=self.batch_size,
            collate_fn=EventDetectionCollator(),
            dataset=self.train_dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            batch_size=self.batch_size,
            collate_fn=EventDetectionCollator(),
            dataset=self.val_dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            batch_size=self.batch_size,
            collate_fn=EventDetectionCollator(),
            dataset=self.test_dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
