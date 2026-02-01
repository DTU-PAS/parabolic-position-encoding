import dataclasses

import numpy as np
import torch

import pape.augmentations.events as augmentations
from pape.configs import Config
from pape.data_types import EventDetectionData
from pape.data_types import Events
from pape.data_types import EventTokens
from pape.gen1 import HEIGHT
from pape.gen1 import WIDTH
from pape.io import Sequence
from pape.io import load_chunks
from pape.tokenizer import Tokenizer


@dataclasses.dataclass
class Sample:
    sequence: int
    chunk: int
    time: int
    labels: np.ndarray


class Gen1Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config: Config,
        samples: list[Sample],
        sequences: list[Sequence],
        augment: bool = False,
    ):
        self.max_detection_events = config.events.max_detection_events
        self.samples = samples
        self.sequences = sequences
        self.should_augment = augment
        self.tokenizer = Tokenizer(config)
        height, width = config.height, config.width
        self.box_normalizer = torch.tensor([width, height, width, height], dtype=torch.float32)

        if self.should_augment:
            self.augmentation = augmentations.Compose(
                [
                    augmentations.Chance(
                        augmentations.HorizontalFlip(WIDTH),
                    ),
                    augmentations.OneOf(
                        [
                            augmentations.Identity(),
                            augmentations.DropByArea(HEIGHT, width=WIDTH),
                            augmentations.DropByTime(),
                            augmentations.DropEvent(),
                            augmentations.Rotation(HEIGHT, width=WIDTH, max_degree=20),
                        ]
                    ),
                ]
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        events = self.load_events(index)
        tokens, augmented_labels = self.augment_and_tokenize(events, sample.labels)
        labels, boxes = self.convert_to_yolov10_labels(augmented_labels)

        # set token timestamps relative to the end time of the chunk
        t = sample.time - tokens.t.to(torch.long)
        t = t.to(torch.uint64)
        tokens = dataclasses.replace(tokens, t=t)

        # normalize boxes to [0, 1]
        boxes = boxes / self.box_normalizer

        return EventDetectionData(
            tokens=tokens,
            labels=labels,
            boxes=boxes,
            prediction_time=sample.time,
        )

    def augment_and_tokenize(
        self, events: Events, labels: np.ndarray
    ) -> tuple[EventTokens, list[augmentations.ObjectDetectionLabel]]:
        original = augmentations.Sample(
            events=events,
            label=augmentations.Label(
                object_detection=[
                    augmentations.ObjectDetectionLabel(
                        x=box["x"].item(),
                        y=box["y"].item(),
                        width=box["w"].item(),
                        height=box["h"].item(),
                        class_id=box["class_id"].item(),
                        t=box["t"].item(),
                    )
                    for box in labels
                ]
            ),
        )

        if not self.should_augment:
            tokens = self.tokenizer(events)
            return tokens, original.label.object_detection

        augmented = self.augmentation(original)

        tokens = self.tokenizer(augmented.events)

        had_labels = len(original.label.object_detection) > 0
        has_labels = len(augmented.label.object_detection) > 0
        if tokens.x.numel() == 0 or (had_labels and not has_labels):
            # Augmentation was too strong. We revert to the original sample instead.
            tokens = self.tokenizer(events)
            return tokens, original.label.object_detection

        return tokens, augmented.label.object_detection

    def load_events(self, index: int):
        sample = self.samples[index]
        return load_chunks(
            sequence=self.sequences[sample.sequence],
            end_chunk_index=sample.chunk,
            end_time_us=sample.time,
            max_events=self.max_detection_events,
        )

    def convert_to_yolov10_labels(
        self, labels: list[augmentations.ObjectDetectionLabel]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the labels to the format expected by YOLOv10 (CXCYWH)."""
        if len(labels) == 0:
            class_labels = torch.empty((0,), dtype=torch.long)
            boxes = torch.empty((0, 4), dtype=torch.float32)
            return class_labels, boxes
        class_labels = [box.class_id for box in labels]
        class_labels = torch.tensor(class_labels, dtype=torch.long)
        boxes = list(map(self.format_yolov10_bounding_box, labels))
        boxes = torch.tensor(boxes, dtype=torch.float32)
        return class_labels, boxes

    def format_yolov10_bounding_box(self, box: augmentations.ObjectDetectionLabel):
        """Converts from a top-left corner and dimensions to a center and dimensions format."""
        center_x = box.x + box.width / 2
        center_y = box.y + box.height / 2
        width = box.width
        height = box.height
        return [center_x, center_y, width, height]
