from pape.augmentations.events.augmentation import Augmentation
from pape.augmentations.events.augmentation import Sample


class Identity(Augmentation):
    def augment(self, params, sample: Sample, mix: Sample | None = None) -> Sample:
        return sample

    def sample_parameters(self):
        return None
