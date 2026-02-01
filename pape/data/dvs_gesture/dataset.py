import dataclasses
from pathlib import Path

import torch
import torch.nn.functional as F

import pape.augmentations.events as augmentations
from pape.configs import Config
from pape.data_types import EventsClassificationData
from pape.dvs_gesture import HEIGHT
from pape.dvs_gesture import NUM_CLASSES
from pape.dvs_gesture import WIDTH
from pape.io import load_events
from pape.tokenizer import Tokenizer

HORIZONTAL_FLIP_LABEL_MAP = [
    0,  # hand clapping
    2,  # right hand wave -> left hand wave
    1,  # left hand wave -> right hand wave
    6,  # right hand clockwise -> left hand counter clockwise
    5,  # right hand counter clockwise -> left hand clockwise
    4,  # left hand clockwise -> right hand counter clockwise
    3,  # left hand counter clockwise -> right hand clockwise
    7,  # forearm roll forward/backward
    8,  # drums
    9,  # guitar
    10,  # random other gesture
]


@dataclasses.dataclass
class Sample:
    path: Path
    label: int
    user: int


class Dataset(torch.utils.data.Dataset):
    def __init__(self, config: Config, samples: list[Sample], augment: bool = False):
        super().__init__()
        self.samples = samples
        self.tokenizer = Tokenizer(config)
        self.augment = augment

        if augment:
            self.augmentation = augmentations.Compose(
                [
                    augmentations.Chance(
                        augmentations.HorizontalFlip(width=WIDTH, classification_label_mapper=self.horizontal_map_flip),
                    ),
                    augmentations.OneOf(
                        [
                            augmentations.Identity(),
                            augmentations.DropByArea(height=HEIGHT, width=WIDTH),
                            augmentations.DropByTime(),
                            augmentations.DropEvent(),
                            augmentations.HorizontalShear(width=WIDTH),
                            augmentations.Rolling(height=HEIGHT, width=WIDTH),
                            augmentations.Rotation(height=HEIGHT, width=WIDTH),
                        ]
                    ),
                ]
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        events = load_events(sample.path)

        label = torch.tensor([sample.label - 1])
        label = F.one_hot(label, num_classes=NUM_CLASSES).squeeze(0)
        label = label.to(torch.float32)
        sample = augmentations.Sample(events=events, label=augmentations.Label(classification=label))

        if self.augment:
            augmented_sample = self.augmentation(sample)
            tokens = self.tokenizer(augmented_sample.events)

            if tokens.patches.size(0) == 0:
                # The augmentation removed too many events,
                # which prevented the tokenizer from creating patches.
                # In this case, we just use the original events.
                tokens = self.tokenizer(sample.events)
            else:
                sample = augmented_sample
        else:
            tokens = self.tokenizer(sample.events)

        return EventsClassificationData(tokens=tokens, label=sample.label.classification)

    def horizontal_map_flip(self, label: torch.Tensor) -> torch.Tensor:
        return label[HORIZONTAL_FLIP_LABEL_MAP]
