import lightning.pytorch as pl

from pape.configs import Config
from pape.configs import Dataset


def load_model(config: Config, steps_per_epoch: int = -1) -> pl.LightningModule:
    match config.dataset:
        case Dataset.imagenet:
            from pape.models.image_classification import ImageClassificationModel

            return ImageClassificationModel(config, steps_per_epoch=steps_per_epoch)
        case Dataset.coco:
            from pape.models.image_detection import ImageDetectionModel

            return ImageDetectionModel(config, steps_per_epoch=steps_per_epoch)
        case Dataset.dvsgesture:
            from pape.models.event_classification import EventClassificationModel

            return EventClassificationModel(config, steps_per_epoch=steps_per_epoch)
        case Dataset.gen1:
            from pape.models.event_detection import EventDetectionModel

            return EventDetectionModel(config, steps_per_epoch=steps_per_epoch)
        case Dataset.ucf101:
            from pape.models.video_classification import VideoClassificationModel

            return VideoClassificationModel(config, steps_per_epoch=steps_per_epoch)
        case _:
            raise ValueError(f"Unsupported dataset: {config.dataset}")
