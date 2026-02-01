import lightning
import torch

from pape.collators import ImageDetectionCollator
from pape.configs import Config
from pape.data.coco.dataset import COCODataset


class COCODataModule(lightning.LightningDataModule):
    def __init__(self, config: Config):
        super().__init__()
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.pin_memory = config.pin_memory
        self.size = (config.height, config.width)

    def setup(self, stage: str):
        if stage is None or stage == "fit":
            self.train_dataset = COCODataset("train", self.size)
            self.val_dataset = COCODataset("val", self.size)
        elif stage == "train":
            self.train_dataset = COCODataset("train", self.size)
        elif stage == "validate":
            self.val_dataset = COCODataset("val", self.size)
        elif stage == "test":
            self.test_dataset = COCODataset("test", self.size)
        else:
            raise ValueError(f"Invalid stage: {stage}. Must be 'fit', 'train', 'validate', or 'test'.")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=ImageDetectionCollator(),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=ImageDetectionCollator(),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=ImageDetectionCollator(),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
