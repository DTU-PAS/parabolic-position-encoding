import torch
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms import v2

from pape.configs import Config
from pape.data_types import EventDetectionBatch
from pape.data_types import EventDetectionData
from pape.data_types import EventsClassificationBatch
from pape.data_types import EventsClassificationData
from pape.data_types import EventTokens
from pape.data_types import ImageClassificationBatch
from pape.data_types import ImageClassificationData
from pape.data_types import ImageDetectionBatch
from pape.data_types import ImageDetectionData
from pape.data_types import VideoClassificationBatch
from pape.data_types import VideoClassificationData


class ImageClassificationCollator:
    def __init__(self, config: Config, train: bool = False):
        self.config = config
        self.train = train
        if train:
            cutmix = v2.CutMix(num_classes=config.num_classes, alpha=config.model.cutmix)
            mixup = v2.MixUp(num_classes=config.num_classes, alpha=config.model.mixup)
            self.cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

    def __call__(self, batch: list[ImageClassificationData]) -> ImageClassificationBatch:
        ids = [item.id for item in batch]
        images = torch.stack([item.image for item in batch])
        labels = torch.tensor([item.label for item in batch], dtype=torch.int64)
        if self.train:
            images, labels = self.cutmix_or_mixup(images, labels)
        return ImageClassificationBatch(ids=ids, images=images, labels=labels)


class ImageDetectionCollator:
    def __call__(self, batch: list[ImageDetectionData]) -> ImageDetectionBatch:
        images = torch.stack([item.image for item in batch])
        batch_idx = torch.cat(
            [
                torch.full((item.labels.size(0),), fill_value=i, device=images.device, dtype=torch.int64)
                for i, item in enumerate(batch)
            ]
        )
        labels = torch.cat([item.labels for item in batch], dim=0).unsqueeze(1)
        boxes = torch.cat([item.boxes for item in batch], dim=0)
        image_ids = [item.image_id for item in batch]
        orig_heights = [item.orig_height for item in batch]
        orig_widths = [item.orig_width for item in batch]
        return ImageDetectionBatch(
            images=images,
            batch_idx=batch_idx,
            labels=labels,
            boxes=boxes,
            image_ids=image_ids,
            orig_heights=orig_heights,
            orig_widths=orig_widths,
        )


class EventsClassificationCollator:
    def __call__(self, batch: list[EventsClassificationData]) -> EventsClassificationBatch:
        tokens = self.collate_tokens(batch)
        labels = torch.stack([item.label for item in batch], dim=0)
        padding_mask = self.make_padding_mask(batch)
        return EventsClassificationBatch(tokens=tokens, labels=labels, padding_mask=padding_mask)

    def collate_tokens(self, batch: list[EventsClassificationData]) -> EventTokens:
        tokens = [element.tokens for element in batch]
        x = self.pad_for_key(tokens, "x")
        y = self.pad_for_key(tokens, "y")
        t = self.pad_for_key(tokens, "t")
        patches = self.pad_for_key(tokens, "patches")
        return EventTokens(x=x, y=y, t=t, patches=patches)

    def make_padding_mask(self, batch: list[EventsClassificationData]) -> torch.Tensor:
        tokens = [element.tokens for element in batch]
        batch_size = len(batch)
        sequence_length = max(len(element.x) for element in tokens)
        lengths = torch.tensor([element.x.size(0) for element in tokens])
        padding_mask = torch.arange(sequence_length).expand(batch_size, -1) > lengths.unsqueeze(1)
        return padding_mask

    def pad_for_key(self, batch: list[EventTokens], key: str):
        values = [getattr(token, key) for token in batch]
        values = pad_sequence(values, batch_first=True)
        return values


class EventDetectionCollator:
    def __call__(self, batch: list[EventDetectionData]) -> EventDetectionBatch:
        tokens = self.collate_tokens(batch)
        padding_mask = self.make_padding_mask(batch)
        batch_idx = torch.cat(
            [torch.full((item.labels.size(0),), fill_value=i, dtype=torch.int64) for i, item in enumerate(batch)]
        )
        labels = torch.cat([item.labels for item in batch], dim=0).unsqueeze(1)
        boxes = torch.cat([item.boxes for item in batch], dim=0)
        prediction_times = [item.prediction_time for item in batch]
        return EventDetectionBatch(
            tokens=tokens,
            labels=labels,
            boxes=boxes,
            batch_idx=batch_idx,
            padding_mask=padding_mask,
            prediction_times=prediction_times,
        )

    def collate_tokens(self, batch: list[EventDetectionData]) -> EventTokens:
        tokens = [element.tokens for element in batch]
        x = self.pad_for_key(tokens, "x")
        y = self.pad_for_key(tokens, "y")
        t = self.pad_for_key(tokens, "t")
        patches = self.pad_for_key(tokens, "patches")
        return EventTokens(x=x, y=y, t=t, patches=patches)

    def make_padding_mask(self, batch: list[EventDetectionData]) -> torch.Tensor:
        tokens = [element.tokens for element in batch]
        batch_size = len(batch)
        sequence_length = max(len(element.x) for element in tokens)
        lengths = torch.tensor([element.x.size(0) for element in tokens])
        padding_mask = torch.arange(sequence_length).expand(batch_size, -1) > lengths.unsqueeze(1)
        return padding_mask

    def pad_for_key(self, batch: list[EventTokens], key: str):
        values = [getattr(token, key) for token in batch]
        values = pad_sequence(values, batch_first=True)
        return values


class VideoClassificationCollator:
    def __init__(self, config: Config):
        self.frame_length = config.video.frame_length

    def __call__(self, batch: list[VideoClassificationData]) -> VideoClassificationBatch:
        videos = torch.stack([item.video for item in batch])
        lengths = torch.tensor([item.length for item in batch], dtype=torch.int64)
        labels = torch.tensor([item.label for item in batch], dtype=torch.int64)
        return VideoClassificationBatch(videos=videos, lengths=lengths, labels=labels)
