import dataclasses
from typing import Callable

import numpy as np

from pape.augmentations.events.augmentation import Augmentation
from pape.augmentations.events.augmentation import ObjectDetectionLabel
from pape.augmentations.events.augmentation import OpticalFlowLabel
from pape.augmentations.events.augmentation import Sample
from pape.data_types import Events


class HorizontalFlip(Augmentation):
    def __init__(self, width: int, classification_label_mapper: Callable | None = None):
        self.width = width
        self.classification_label_mapper = classification_label_mapper

    def augment(self, params, sample: Sample, mix: Sample | None = None) -> Sample:
        events = self.flip_events(sample.events)

        label = sample.label
        if label is not None:
            classification = label.classification
            if classification is not None and self.classification_label_mapper is not None:
                classification = self.classification_label_mapper(classification)

            object_detection = label.object_detection
            if object_detection is not None:
                object_detection = self.flip_object_detection_label(object_detection)

            optical_flow = label.optical_flow
            if optical_flow is not None:
                optical_flow = self.flip_optical_flow_label(optical_flow)

            label = dataclasses.replace(
                label, classification=classification, object_detection=object_detection, optical_flow=optical_flow
            )

        return Sample(events=events, label=label)

    def sample_parameters(self):
        return None

    def flip_events(self, events: Events) -> Events:
        x = (self.width - 1) - events.x
        return dataclasses.replace(events, x=x)

    def flip_object_detection_label(self, object_detection: list[ObjectDetectionLabel]) -> list[ObjectDetectionLabel]:
        flipped = []
        for label in object_detection:
            x = self.width - (label.x + label.width)
            flipped.append(dataclasses.replace(label, x=x))
        return flipped

    def flip_optical_flow_label(self, label: OpticalFlowLabel) -> OpticalFlowLabel:
        flow = np.fliplr(label.flow).copy()  # Copy to ensure contiguous array
        flow[..., 0] *= -1  # Invert x component
        valid = np.fliplr(label.valid).copy()  # Copy to ensure contiguous array
        return OpticalFlowLabel(flow=flow, valid=valid)
