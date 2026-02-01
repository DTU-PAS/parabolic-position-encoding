from random import random

from pape.augmentations.events.augmentation import Augmentation
from pape.augmentations.events.augmentation import Sample


class Chance(Augmentation):
    def __init__(self, augmentation: Augmentation, probability: float = 0.5):
        if probability < 0 or probability > 1:
            raise ValueError("probability must be in the range [0, 1]")

        self.augmentation = augmentation
        self.probability = probability

    def augment(self, params: float, sample: Sample, mix: Sample | None = None) -> Sample:
        if params > self.probability:
            return sample

        return self.augmentation(sample, mix=mix)

    def sample_parameters(self):
        return random()
