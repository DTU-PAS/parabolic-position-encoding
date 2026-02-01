import dataclasses
import math
from random import uniform

from pape.augmentations.events.augmentation import Augmentation
from pape.augmentations.events.augmentation import Sample
from pape.data_types import Events


class DropByTime(Augmentation):
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
        start_ratio = uniform(0.0, 1.0)
        end_ratio = uniform(self.min_ratio, self.max_ratio)
        return start_ratio, end_ratio

    def drop_events(self, events: Events, ratios: tuple[float, float]) -> Events:
        if len(events) == 0:
            return events

        start = events.t[0]
        end = events.t[-1]
        duration = end - start

        start_ratio, end_ratio = ratios
        min_time = math.floor(start + start_ratio * duration)
        max_time = math.ceil(min_time + end_ratio * duration)

        mask = (events.t < min_time) | (events.t > max_time)

        masked_events = events.mask(mask)

        if len(masked_events) == 0:
            return events

        return masked_events
