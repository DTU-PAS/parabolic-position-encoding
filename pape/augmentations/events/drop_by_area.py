import dataclasses
from math import ceil
from random import randint
from random import uniform

from pape.augmentations.events.augmentation import Augmentation
from pape.augmentations.events.augmentation import ObjectDetectionLabel
from pape.augmentations.events.augmentation import OpticalFlowLabel
from pape.augmentations.events.augmentation import Sample
from pape.data_types import Events


class DropByArea(Augmentation):
    def __init__(self, height: int, width: int, min_ratio: int = 0.05, max_ratio: int = 0.3):
        if min_ratio < 0 or min_ratio > 1:
            raise ValueError("min_ratio must be in the range [0, 1]")

        if max_ratio < 0 or max_ratio > 1:
            raise ValueError("max_ratio must be in the range [0, 1]")

        if min_ratio > max_ratio:
            raise ValueError("min_ratio must be less than or equal to max_ratio")

        self.height = height
        self.width = width

        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def augment(self, params, sample: Sample, mix: Sample | None = None) -> Sample:
        ratio, center_x, center_y = params

        drop_height = ceil(self.height * ratio)
        drop_width = ceil(self.width * ratio)

        x_start = max(0, int(center_x - drop_width / 2))
        x_end = min(self.width, int(center_x + drop_width / 2))

        y_start = max(0, int(center_y - drop_height / 2))
        y_end = min(self.height, int(center_y + drop_height / 2))

        events = self.drop_events(sample.events, x_start, x_end, y_start, y_end)

        label = sample.label
        if label is not None:
            if label.object_detection is not None:
                object_detection = self.drop_object_detection_label(
                    label.object_detection, x_start, x_end, y_start, y_end
                )
                label = dataclasses.replace(label, object_detection=object_detection)

            if label.optical_flow is not None:
                optical_flow = self.drop_optical_flow_label(label.optical_flow, x_start, x_end, y_start, y_end)
                label = dataclasses.replace(label, optical_flow=optical_flow)

        return Sample(events=events, label=label)

    def sample_parameters(self):
        ratio = uniform(self.min_ratio, self.max_ratio)
        center_x = randint(0, self.width - 1)
        center_y = randint(0, self.height - 1)
        return ratio, center_x, center_y

    def drop_events(self, events: Events, x_start: int, x_end: int, y_start: int, y_end: int) -> Events:
        mask = (events.x < x_start) | (events.x > x_end) | (events.y < y_start) | (events.y > y_end)

        masked_events = events.mask(mask)
        if len(masked_events) == 0:
            return events

        return masked_events

    def drop_object_detection_label(
        self, labels: list[ObjectDetectionLabel], x_start: int, x_end: int, y_start: int, y_end: int
    ) -> list[ObjectDetectionLabel]:
        masked_labels = []
        for label in labels:
            left = label.x
            right = label.x + label.width
            top = label.y
            bottom = label.y + label.height
            if (right < x_start) or (x_end < left) or (bottom < y_start) or (y_end < top):
                masked_labels.append(label)
        return masked_labels

    def drop_optical_flow_label(
        self, label: OpticalFlowLabel, x_start: int, x_end: int, y_start: int, y_end: int
    ) -> OpticalFlowLabel:
        flow = label.flow.copy()
        valid = label.valid.copy()
        flow[y_start:y_end, x_start:x_end] = 0
        valid[y_start:y_end, x_start:x_end] = False
        return OpticalFlowLabel(flow=flow, valid=valid)
