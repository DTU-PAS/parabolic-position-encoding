import dataclasses
import math
from random import uniform

import numpy as np

from pape.augmentations.events.augmentation import Augmentation
from pape.augmentations.events.augmentation import ObjectDetectionLabel
from pape.augmentations.events.augmentation import Sample
from pape.data_types import Events


class Rotation(Augmentation):
    def __init__(self, height: int, width: int, max_degree: int = 30):
        self.height = height
        self.width = width
        self.max_radians = max_degree * (math.pi / 180)
        self.center_x = width / 2
        self.center_y = height / 2

    def augment(self, params, sample: Sample, mix: Sample | None = None) -> Sample:
        radians = params

        # Rotate around center
        cos = math.cos(radians)
        sin = math.sin(radians)

        events = sample.events

        offset_x = events.x - self.center_x
        offset_y = events.y - self.center_y

        x = self.center_x + (offset_x * cos - offset_y * sin)
        y = self.center_y + (offset_x * sin + offset_y * cos)

        x = np.floor(x).astype(np.uint16)
        y = np.floor(y).astype(np.uint16)

        # Remove events that are outside the frame after the rotation
        mask = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
        x = x[mask]
        y = y[mask]
        t = events.t[mask]
        p = events.p[mask]

        rotated_events = Events(
            x=x,
            y=y,
            t=t,
            p=p,
        )

        label = sample.label
        if label is not None:
            object_detection = label.object_detection
            if object_detection is not None:
                object_detection = self.rotate_object_detection_labels(object_detection, cos, sin)

            optical_flow = label.optical_flow
            if optical_flow is not None:
                raise ValueError("Rotation of optical flow labels is not supported.")

            label = dataclasses.replace(label, object_detection=object_detection, optical_flow=optical_flow)

        return dataclasses.replace(sample, events=rotated_events, label=label)

    def sample_parameters(self):
        radians = uniform(-self.max_radians, self.max_radians)
        return radians

    def rotate_object_detection_labels(
        self, labels: list[ObjectDetectionLabel], cos: float, sin: float
    ) -> list[ObjectDetectionLabel]:
        rotated_labels = []
        for label in labels:
            coordinates = label.coordinates()
            x = np.array([x for x, _ in coordinates])
            y = np.array([y for _, y in coordinates])
            offset_x = x - self.center_x
            offset_y = y - self.center_y
            x = self.center_x + (offset_x * cos - offset_y * sin)
            y = self.center_y + (offset_x * sin + offset_y * cos)

            # Get enclosing box
            left = np.min(x)
            right = np.max(x)
            top = np.min(y)
            bottom = np.max(y)

            # Remove labels that are outside of the frame after the rotation
            if left < 0 or right >= self.width or top < 0 or bottom >= self.height:
                continue

            width = right - left
            height = bottom - top

            label = dataclasses.replace(label, x=left, y=top, width=width, height=height)
            rotated_labels.append(label)

        return rotated_labels
