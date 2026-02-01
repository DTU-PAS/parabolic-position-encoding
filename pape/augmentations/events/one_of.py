import random

from pape.augmentations.events.augmentation import Augmentation
from pape.augmentations.events.augmentation import Sample


class OneOf(Augmentation):
    def __init__(self, augmentations: list[Augmentation], weights: list[float | int] | None = None):
        self.augmentations = augmentations

        self.cumulative_weights = None
        if weights is not None:
            assert len(weights) == len(augmentations)
            self.cumulative_weights = [weights[0]]
            for i in range(1, len(weights)):
                previous = self.cumulative_weights[i - 1]
                current = weights[i]
                self.cumulative_weights.append(previous + current)

    def augment(self, params, sample: Sample, mix: Sample | None = None) -> Sample:
        augmentation, augmentation_params = params
        return augmentation.augment(augmentation_params, sample, mix=mix)

    def sample_parameters(self):
        augmentation = random.choices(self.augmentations, cum_weights=self.cumulative_weights)[0]
        params = augmentation.sample_parameters()
        return augmentation, params
