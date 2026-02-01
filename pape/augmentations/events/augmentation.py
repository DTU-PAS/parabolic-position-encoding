import abc
import functools
from dataclasses import dataclass
from typing import Any
from typing import Callable

import numpy as np

from pape.data_types import Events


@dataclass
class ObjectDetectionLabel:
    x: int | float
    y: int | float
    width: int | float
    height: int | float
    class_id: int
    t: int

    def coordinates(self) -> list[tuple[int | float, int | float]]:
        return [
            (self.x, self.y),  # Top left
            (self.x + self.width, self.y),  # Top right
            (self.x + self.width, self.y + self.height),  # Bottom right
            (self.x, self.y + self.height),  # Bottom left
        ]


@dataclass
class OpticalFlowLabel:
    flow: np.ndarray  # shape (H, W, 2), dtype float32
    valid: np.ndarray  # shape (H, W), dtype bool


@dataclass
class Label:
    classification: object | None = None
    object_detection: list[ObjectDetectionLabel] | None = None
    optical_flow: OpticalFlowLabel | None = None


@dataclass
class Sample:
    events: Events | None = None
    label: Label | None = None


class Augmentation(abc.ABC):
    @abc.abstractmethod
    def augment(self, params: Any, sample: Sample, mix: Sample | None = None) -> Sample:
        pass

    @abc.abstractmethod
    def sample_parameters(self) -> Any:
        pass

    def __call__(self, sample: Sample, mix: Sample | None = None) -> Sample:
        params = self.sample_parameters()
        return self.augment(params, sample, mix=mix)

    def sample(self) -> Callable[[Sample, Sample | None], Sample]:
        params = self.sample_parameters()
        return functools.partial(self.augment, params)
