import dataclasses
from random import uniform

import numpy as np

from pape.augmentations.events.augmentation import Augmentation
from pape.augmentations.events.augmentation import ObjectDetectionLabel
from pape.augmentations.events.augmentation import Sample
from pape.data_types import Events


class HorizontalShear(Augmentation):
    def __init__(self, width: int, max_shear_factor: float = 0.30):
        self.width = width
        self.max_shear_factor = max_shear_factor

    def augment(self, params, sample: Sample, mix: Sample | None = None) -> Sample:
        shear = params

        events = sample.events
        x = events.x + events.y * shear
        x = np.floor(x)

        # Remove events that are outside of the frame after the shear
        mask = (x >= 0) & (x < self.width)

        x = x.astype(np.uint16)
        x = x[mask]
        y = events.y[mask]
        t = events.t[mask]
        p = events.p[mask]

        sheared_events = Events(
            x=x,
            y=y,
            t=t,
            p=p,
        )

        label = sample.label
        if label is not None:
            object_detection = label.object_detection
            if object_detection is not None:
                object_detection = self.shear_object_detection_labels(object_detection, shear)
            label = dataclasses.replace(label, object_detection=object_detection)

            if label.optical_flow is not None:
                raise ValueError("Shearing of optical flow labels is not supported.")

        return dataclasses.replace(sample, events=sheared_events, label=label)

    def sample_parameters(self):
        shear = uniform(-self.max_shear_factor, self.max_shear_factor)
        return shear

    def shear_object_detection_labels(
        self, labels: list[ObjectDetectionLabel], shear: float
    ) -> list[ObjectDetectionLabel]:
        if shear < 0:
            return self.shear_object_detection_labels_left(labels, abs(shear))
        else:
            return self.shear_object_detection_labels_right(labels, shear)

    def shear_object_detection_labels_left(
        self, labels: list[ObjectDetectionLabel], shear: float
    ) -> list[ObjectDetectionLabel]:
        sheared_labels = []
        for label in labels:
            top = label.y
            bottom = label.y + label.height
            sheared_left = label.x - bottom * shear
            sheared_right = (label.x + label.width) - (top * shear)

            # Remove labels that are outside of the frame after the shear
            if sheared_left < 0:
                continue

            sheared_width = sheared_right - sheared_left
            sheared_label = dataclasses.replace(label, x=sheared_left, width=sheared_width)
            sheared_labels.append(sheared_label)
        return sheared_labels

    def shear_object_detection_labels_right(
        self, labels: list[ObjectDetectionLabel], shear: float
    ) -> list[ObjectDetectionLabel]:
        sheared_labels = []
        for label in labels:
            top = label.y
            bottom = label.y + label.height
            sheared_left = label.x + top * shear
            sheared_right = (label.x + label.width) + (bottom * shear)

            # Remove labels that are outside of the frame after the shear
            if sheared_right >= self.width:
                continue

            sheared_width = sheared_right - sheared_left
            sheared_label = dataclasses.replace(label, x=sheared_left, width=sheared_width)
            sheared_labels.append(sheared_label)
        return sheared_labels
