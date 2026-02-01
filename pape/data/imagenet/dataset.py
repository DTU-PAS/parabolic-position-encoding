"""Code adapted from https://towardsdatascience.com/downloading-and-using-the-imagenet-dataset-with-pytorch-f0908437c4be"""

import json
from pathlib import Path

import torch
import torchvision
from torchvision.transforms import v2 as T

from pape.augmentations.images import ResizeSmall
from pape.data_types import ImageClassificationData
from pape.paths import get_dataset_dir


class ImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, size: tuple[int, int], augment: bool = False):
        if split not in ("train", "val", "test"):
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")

        source_dir = get_dataset_dir("imagenet")

        if augment:
            self.transform = T.Compose(
                [
                    T.RandomResizedCrop(size),
                    T.RandomHorizontalFlip(),
                    T.RandAugment(),
                ]
            )
        else:
            scale = 256 / 224
            smaller_size = int(min(size) * scale)
            self.transform = T.Compose(
                [
                    ResizeSmall(smaller_size),
                    T.CenterCrop(size),
                ]
            )

        self.samples: list[Path] = []
        self.targets: list[int] = []

        self.syn_to_class = {}
        with open(source_dir / "imagenet_class_index.json", "rb") as f:
            json_file = json.load(f)
            for class_id, v in json_file.items():
                self.syn_to_class[v[0]] = int(class_id)

        with open(source_dir / "ILSVRC2012_val_labels.json", "rb") as f:
            self.val_to_syn = json.load(f)

        samples_dir = source_dir / "ILSVRC/Data/CLS-LOC" / split
        if split == "train":
            for syn_folder in samples_dir.iterdir():
                syn_id = syn_folder.name
                target = self.syn_to_class[syn_id]
                for sample in syn_folder.iterdir():
                    sample_path = syn_folder / sample
                    self.samples.append(sample_path)
                    self.targets.append(target)
        elif split == "val":
            for sample_path in samples_dir.iterdir():
                entry = sample_path.name
                syn_id = self.val_to_syn[entry]
                target = self.syn_to_class[syn_id]
                self.samples.append(sample_path)
                self.targets.append(target)
        else:  # split == "test"
            for sample_path in samples_dir.iterdir():
                self.samples.append(sample_path)
                self.targets.append(0)  # Test set targets are unknown

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path: Path = self.samples[idx]
        image = torchvision.io.decode_image(path, mode=torchvision.io.ImageReadMode.RGB)
        assert image.ndim == 3, f"Image at '{path}' is not 3D: {image.shape}"

        image = self.transform(image)
        image = image.float() / 255.0  # Normalize to [0, 1]

        label = self.targets[idx]

        return ImageClassificationData(id=path.name, image=image, label=label)
