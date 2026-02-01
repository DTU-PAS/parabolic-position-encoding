from typing import Literal

import torch
from torchvision import tv_tensors
from torchvision.datasets import CocoDetection
from torchvision.datasets import wrap_dataset_for_transforms_v2
from torchvision.transforms import v2
from torchvision.tv_tensors import BoundingBoxes
from torchvision.tv_tensors import BoundingBoxFormat

from pape.data_types import ImageDetectionData
from pape.paths import get_dataset_dir


class COCODataset(torch.utils.data.Dataset):
    def __init__(self, split: Literal["train", "val", "test"], size: tuple[int, int]):
        super().__init__()

        if split == "test":
            anno_file = get_dataset_dir("coco") / "annotations" / "image_info_test-dev2017.json"
        else:
            anno_file = get_dataset_dir("coco") / "annotations" / f"instances_{split}2017.json"

        self.is_training = split == "train"
        if self.is_training:
            # v2.RandAugment does not support bounding boxes, so we choose augmentations and values
            # that correspond roughly to RandAugment with a magnitude of 9
            # note that we disable rotation and shear as bounding boxes are transformed poorly
            # we also add ScaleJitter which is not part of RandAugment
            self.transforms = v2.Compose(
                [
                    v2.ToImage(),
                    v2.RandomHorizontalFlip(),
                    v2.RandomChoice(
                        [
                            v2.Identity(),
                            v2.RandomAffine(degrees=0, translate=(0.14, 0.14)),
                            v2.ColorJitter(brightness=0.32),
                            v2.ColorJitter(saturation=0.32, hue=0.16),
                            v2.ColorJitter(contrast=0.32),
                            v2.RandomAdjustSharpness(sharpness_factor=0.32, p=1.0),
                            v2.RandomPosterize(bits=7, p=1.0),
                            v2.RandomSolarize(threshold=178.5, p=1.0),
                            v2.RandomAutocontrast(p=1.0),
                            v2.RandomEqualize(p=1.0),
                            v2.ScaleJitter(size),
                        ]
                    ),
                    # replicate list to sample two augmentations
                    # this is similar to RandAugment's two augmentations per image
                    # except that we sample with replacement here
                    v2.RandomChoice(
                        [
                            v2.Identity(),
                            v2.RandomAffine(degrees=0, translate=(0.14, 0.14)),
                            v2.ColorJitter(brightness=0.32),
                            v2.ColorJitter(saturation=0.32, hue=0.16),
                            v2.ColorJitter(contrast=0.32),
                            v2.RandomAdjustSharpness(sharpness_factor=0.32, p=1.0),
                            v2.RandomPosterize(bits=7, p=1.0),
                            v2.RandomSolarize(threshold=178.5, p=1.0),
                            v2.RandomAutocontrast(p=1.0),
                            v2.RandomEqualize(p=1.0),
                            v2.ScaleJitter(size),
                        ]
                    ),
                    v2.Resize(size),
                    v2.ConvertBoundingBoxFormat(BoundingBoxFormat.CXCYWH),
                    v2.ToDtype(torch.float32, scale=True),
                ]
            )
            self.sanitizer = v2.SanitizeBoundingBoxes()
        else:
            self.transforms = v2.Compose(
                [
                    v2.ToImage(),
                    v2.Resize(size),
                    v2.ConvertBoundingBoxFormat(BoundingBoxFormat.CXCYWH),
                    v2.ToDtype(torch.float32, scale=True),
                ]
            )

        self.size = size
        self.box_normalizer = torch.tensor([size[1], size[0], size[1], size[0]], dtype=torch.float32)

        dataset = CocoDetection(
            root=get_dataset_dir("coco") / f"{split}2017",
            annFile=anno_file,
        )
        dataset = wrap_dataset_for_transforms_v2(dataset, target_keys=["labels", "boxes", "image_id"])
        self.tv_dataset = dataset

    def __len__(self):
        return len(self.tv_dataset)

    def __getitem__(self, index: int):
        image, target = self.tv_dataset[index]

        image_id = target["image_id"]
        orig_height = image.height
        orig_width = image.width

        image, target = self.transforms(image, target)

        if self.is_training and "boxes" in target:
            image, target = self.sanitizer(image, target)

        labels = target.get("labels")
        if labels is None:
            labels = torch.tensor([], dtype=torch.int64)

        boxes = target.get("boxes")
        if boxes is None:
            boxes = torch.tensor([], dtype=torch.float32).reshape(0, 4)
            boxes = BoundingBoxes(boxes, format=BoundingBoxFormat.CXCYWH, canvas_size=self.size)

        # Normalize boxes to [0, 1]
        normalized_boxes = boxes / self.box_normalizer
        boxes = tv_tensors.wrap(normalized_boxes, like=boxes)

        return ImageDetectionData(
            image=image,
            image_id=image_id,
            orig_height=orig_height,
            orig_width=orig_width,
            labels=labels,
            boxes=boxes,
        )
