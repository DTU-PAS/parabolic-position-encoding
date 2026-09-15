from pathlib import Path

import torch
import torchvision
from torchvision.transforms import v2 as T

from pape.augmentations.images import ResizeSmall
from pape.data_types import ImageClassificationData
from pape.paths import get_dataset_dir


class ImageNetV2Dataset(torch.utils.data.Dataset):
    def __init__(self, size: tuple[int, int]):
        source_dir = get_dataset_dir("imagenetv2")

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

        for class_dir in sorted(source_dir.iterdir(), key=lambda p: int(p.name)):
            if not class_dir.is_dir():
                continue
            label = int(class_dir.name)
            for sample_path in sorted(class_dir.iterdir()):
                self.samples.append(sample_path)
                self.targets.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path: Path = self.samples[idx]
        image = torchvision.io.decode_image(path, mode=torchvision.io.ImageReadMode.RGB)
        assert image.ndim == 3, f"Image at '{path}' is not 3D: {image.shape}"

        image = self.transform(image)
        image = image.float() / 255.0

        label = self.targets[idx]

        return ImageClassificationData(id=path.name, image=image, label=label)
