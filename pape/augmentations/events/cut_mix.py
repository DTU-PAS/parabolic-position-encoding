import dataclasses
import math
import random

import numpy as np
import torch

from pape.augmentations.events.augmentation import Augmentation
from pape.augmentations.events.augmentation import Label
from pape.augmentations.events.augmentation import Sample
from pape.data_types import Events


class CutMix(Augmentation):
    def __init__(self, height: int, width: int):
        super().__init__()
        self.height = height
        self.width = width

    def augment(self, params, sample: Sample, mix: Sample | None = None) -> Sample:
        if mix is None:
            raise ValueError("mix argument is required for CutMix augmentation")

        if sample.label is None or mix.label is None:
            raise ValueError("sample and mix must have labels")

        if sample.label.classification is not None and mix.label.classification is not None:
            return self.augment_classification(sample, mix, *params)

        return sample

    def sample_parameters(self):
        mix_ratio = random.uniform(0.0, 1.0)

        width = math.floor(self.width * math.sqrt(1 - mix_ratio))
        height = math.floor(self.height * math.sqrt(1 - mix_ratio))
        half_width = width // 2
        half_height = height // 2
        center_x = random.randint(half_width, self.width - 1 - half_width)
        center_y = random.randint(half_height, self.height - 1 - half_height)

        return mix_ratio, center_x, center_y

    def augment_classification(
        self, sample: Sample, mix: Sample, mix_ratio: float, center_x: int, center_y: int
    ) -> Sample:
        # Cut events
        width = math.floor(self.width * math.sqrt(1 - mix_ratio))
        height = math.floor(self.height * math.sqrt(1 - mix_ratio))

        outside = self.filter_outside_box(center_x, center_y, width, height, sample.events)
        inside = self.filter_inside_box(center_x, center_y, width, height, mix.events)

        # We change the original CutMix slightly, since events have different characteristics from images.
        # A masked patch may have few or many events, so area is not a good metric of how much is being mixed.
        # Instead, we use the ratio of events in the outside box to the total number of events.
        label_mix_ratio = len(outside) / (len(outside) + len(inside))

        # Mix and sort by timestamp to ensure ascending order
        x = np.concatenate([outside.x, inside.x])
        y = np.concatenate([outside.y, inside.y])
        p = np.concatenate([outside.p, inside.p])
        t = np.concatenate([outside.t, inside.t])
        sort_indices = np.argsort(t)
        x = x[sort_indices]
        y = y[sort_indices]
        p = p[sort_indices]
        t = t[sort_indices]

        events = Events(x=x, y=y, p=p, t=t)

        # Mix labels
        classification_label = None
        if sample.label.classification is not None and mix.label.classification is not None:
            classification_label = self.mix_classification_labels(
                sample.label.classification, mix.label.classification, label_mix_ratio
            )
        label = Label(classification=classification_label)

        return dataclasses.replace(sample, events=events, label=label)

    def mix_classification_labels(self, sample: torch.Tensor, mix: torch.Tensor, mix_ratio: float):
        return sample * mix_ratio + mix * (1 - mix_ratio)

    def filter_inside_box(self, center_x: int, center_y: int, width: int, height: int, events: Events):
        x_start = center_x - width / 2
        x_end = center_x + width / 2
        y_start = center_y - height / 2
        y_end = center_y + height / 2
        mask = (events.x >= x_start) & (events.x <= x_end) & (events.y >= y_start) & (events.y <= y_end)
        return events.mask(mask)

    def filter_outside_box(self, center_x: int, center_y: int, width: int, height: int, events: Events):
        x_start = center_x - width / 2
        x_end = center_x + width / 2
        y_start = center_y - height / 2
        y_end = center_y + height / 2
        mask = (events.x < x_start) | (events.x > x_end) | (events.y < y_start) | (events.y > y_end)
        return events.mask(mask)
