import lightning
import torch

from pape.collators import ImageClassificationCollator
from pape.configs import Config
from pape.data.imagenetv2.dataset import ImageNetV2Dataset


class ImageNetV2DataModule(lightning.LightningDataModule):
    def __init__(self, config: Config):
        super().__init__()
        self.batch_size = config.batch_size
        self.config = config
        self.num_workers = config.num_workers
        self.pin_memory = config.pin_memory
        self.size = (config.height, config.width)

    def setup(self, stage: str):
        if stage == "test":
            self.test_dataset = ImageNetV2Dataset(size=self.size)
        else:
            raise ValueError(f"Invalid stage: {stage}. ImageNetV2 is for inference only.")

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=ImageClassificationCollator(self.config),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
