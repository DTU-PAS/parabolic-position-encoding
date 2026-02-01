import lightning
import torch
from torch.utils.data.sampler import SubsetRandomSampler

from pape.collators import ImageClassificationCollator
from pape.configs import Config
from pape.data.imagenet.dataset import ImageNetDataset


class ImageNetDataModule(lightning.LightningDataModule):
    def __init__(self, config: Config):
        super().__init__()
        self.batch_size = config.batch_size
        self.config = config
        self.num_classes = config.num_classes
        self.num_workers = config.num_workers
        self.pin_memory = config.pin_memory
        self.size = (config.height, config.width)
        self.valid_ratio = config.valid_ratio

    def setup(self, stage: str):
        if stage is None or stage == "fit" or stage == "validate":
            self.train_dataset = ImageNetDataset(split="train", size=self.size, augment=True)
            self.val_dataset = ImageNetDataset(split="train", size=self.size)
            num_val_samples = int(len(self.train_dataset) * self.valid_ratio)
            val_step = len(self.train_dataset) // num_val_samples
            val_indices = list(range(0, len(self.train_dataset), val_step))[:num_val_samples]
            train_indices = list(set(range(len(self.train_dataset))) - set(val_indices))
            self.train_sampler = SubsetRandomSampler(train_indices)
            self.val_sampler = SubsetRandomSampler(val_indices)
        elif stage == "test":
            self.test_dataset = ImageNetDataset(split="val", size=self.size)
        else:
            raise ValueError(f"Invalid stage: {stage}. Must be 'fit', 'validate', or 'test'.")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=ImageClassificationCollator(self.config, train=True),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            sampler=self.train_sampler,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=ImageClassificationCollator(self.config),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            sampler=self.val_sampler,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=ImageClassificationCollator(self.config),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
