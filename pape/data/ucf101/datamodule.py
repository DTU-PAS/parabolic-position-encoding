import lightning
import torch

from pape.collators import VideoClassificationCollator
from pape.configs import Config
from pape.configs import Split
from pape.data.ucf101.dataset import UCF101Dataset
from pape.paths import get_dataset_dir
from pape.ucf101 import DATASET_NAME


class UCF101DataModule(lightning.LightningDataModule):
    def __init__(self, config: Config):
        super().__init__()
        self.batch_size = config.batch_size
        self.config = config
        self.num_workers = config.num_workers
        self.pin_memory = config.pin_memory
        self.path_to_label = self.get_path_to_label()

    def setup(self, stage: str):
        if stage is None or stage == "fit":
            self.train_dataset = UCF101Dataset(self.config, self.path_to_label, split=Split.train)
            self.val_dataset = UCF101Dataset(self.config, self.path_to_label, split=Split.val)
        elif stage == "test":
            self.test_dataset = UCF101Dataset(self.config, self.path_to_label, split=Split.test)
        else:
            raise ValueError(f"Invalid stage: {stage}. Must be 'fit' or 'test'.")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=VideoClassificationCollator(self.config),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=VideoClassificationCollator(self.config),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=VideoClassificationCollator(self.config),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def get_path_to_label(self) -> dict[str, int]:
        dataset_dir = get_dataset_dir(DATASET_NAME)
        annotation_dir = dataset_dir / "ucfTrainTestlist"
        trainlist_paths = annotation_dir.glob("trainlist*.txt")
        path_to_labels = {}
        for path in trainlist_paths:
            with open(path, "r") as f:
                lines = f.readlines()
            for line in lines:
                video_path, label = line.strip().split()
                path_to_labels[video_path] = int(label) - 1
        return path_to_labels
