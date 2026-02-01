from pape.augmentations.events.augmentation import Augmentation
from pape.augmentations.events.augmentation import Sample


class Compose(Augmentation):
    def __init__(self, augmentations: list[Augmentation]):
        self.augmentations = augmentations

    def augment(self, params, sample: Sample, mix: Sample | None = None) -> Sample:
        for augmentation, augmentation_params in zip(self.augmentations, params, strict=True):
            sample = augmentation.augment(augmentation_params, sample, mix=mix)
        return sample

    def sample_parameters(self):
        return [augmentation.sample_parameters() for augmentation in self.augmentations]
