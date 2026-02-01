import dataclasses
from random import randint

import numpy as np

from pape.augmentations.events.augmentation import Augmentation
from pape.augmentations.events.augmentation import ObjectDetectionLabel
from pape.augmentations.events.augmentation import Sample


class Rolling(Augmentation):
    def __init__(self, height: int, width: int, max_roll: int = 5):
        self.height = height
        self.width = width
        self.max_roll = max_roll

    def augment(self, params, sample: Sample, mix: Sample | None = None) -> Sample:
        roll_x, roll_y = params

        x = sample.events.x.astype(np.int32)
        y = sample.events.y.astype(np.int32)

        x = (x + roll_x) % self.width
        y = (y + roll_y) % self.height

        x = x.astype(np.uint16)
        y = y.astype(np.uint16)

        label = sample.label
        if label is not None:
            object_detection = label.object_detection
            if object_detection is not None:
                object_detection = self.roll_object_detection_labels(object_detection, roll_x, roll_y)

            if label.optical_flow is not None:
                raise ValueError("Rolling of optical flow labels is not supported.")

        rolled_events = dataclasses.replace(sample.events, x=x, y=y)

        return dataclasses.replace(sample, events=rolled_events, label=label)

    def sample_parameters(self):
        roll_x = randint(-self.max_roll, self.max_roll)
        roll_y = randint(-self.max_roll, self.max_roll)
        return roll_x, roll_y

    def roll_object_detection_labels(
        self, labels: list[ObjectDetectionLabel], roll_x: int, roll_y: int
    ) -> list[ObjectDetectionLabel]:
        rolled_labels = []
        for label in labels:
            x = label.x + roll_x
            y = label.y + roll_y

            # Remove labels that would cross the frame boundaries after the roll
            if x < 0 or x >= self.width or y < 0 or y >= self.height:
                continue

            label = dataclasses.replace(label, x=x, y=y)
            rolled_labels.append(label)

        return rolled_labels
