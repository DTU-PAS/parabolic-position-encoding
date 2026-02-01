import lightning.pytorch as pl

from pape.configs import Config
from pape.configs import Dataset


def load_datamodule(config: Config) -> pl.LightningDataModule:
    match config.dataset:
        case Dataset.coco:
            from pape.data.coco.datamodule import COCODataModule

            return COCODataModule(config)
        case Dataset.dvsgesture:
            from pape.data.dvs_gesture.datamodule import DVSGestureDataModule

            return DVSGestureDataModule(config)
        case Dataset.gen1:
            from pape.data.gen1.datamodule import Gen1DataModule

            return Gen1DataModule(config)
        case Dataset.imagenet:
            from pape.data.imagenet.damodule import ImageNetDataModule

            return ImageNetDataModule(config)
        case Dataset.ucf101:
            from pape.data.ucf101.datamodule import UCF101DataModule

            return UCF101DataModule(config)
        case _:
            raise ValueError(f"Dataset '{config.dataset}' is not supported.")
