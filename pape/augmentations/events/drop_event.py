import dataclasses
from random import uniform

import numpy as np

from pape.augmentations.events.augmentation import Augmentation
from pape.augmentations.events.augmentation import Sample
from pape.data_types import Events


class DropEvent(Augmentation):
    def __init__(self, min_ratio: float = 0.1, max_ratio: float = 0.75):
        if min_ratio < 0 or min_ratio > 1:
            raise ValueError("min_ratio must be in the range [0, 1]")

        if max_ratio < 0 or max_ratio > 1:
            raise ValueError("max_ratio must be in the range [0, 1]")

        if min_ratio > max_ratio:
            raise ValueError("min_ratio must be less than or equal to max_ratio")

        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def augment(self, params, sample: Sample, mix: Sample | None = None) -> Sample:
        events = self.drop_events(sample.events, params)
        return dataclasses.replace(sample, events=events)

    def sample_parameters(self):
        return uniform(self.min_ratio, self.max_ratio)

    def drop_events(self, events: Events, ratio: float) -> Events:
        mask = np.random.rand(len(events.x)) > ratio
        masked_events = events.mask(mask)

        if len(masked_events) == 0:
            return events

        return masked_events
