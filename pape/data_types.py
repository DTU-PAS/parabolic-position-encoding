import dataclasses

import numpy as np
import torch


@dataclasses.dataclass
class Events:
    x: np.ndarray
    y: np.ndarray
    t: np.ndarray
    p: np.ndarray

    def __len__(self):
        return len(self.x)

    def mask(self, mask: np.ndarray) -> "Events":
        return Events(
            x=self.x[mask],
            y=self.y[mask],
            t=self.t[mask],
            p=self.p[mask],
        )

    def __getitem__(self, index: int) -> "Events":
        return Events(
            x=self.x[index],
            y=self.y[index],
            t=self.t[index],
            p=self.p[index],
        )


@dataclasses.dataclass
class EventTokens:
    x: torch.Tensor
    y: torch.Tensor
    t: torch.Tensor
    patches: torch.Tensor

    def to(self, device: torch.device) -> "EventTokens":
        return EventTokens(
            x=self.x.to(device),
            y=self.y.to(device),
            t=self.t.to(device),
            patches=self.patches.to(device),
        )


@dataclasses.dataclass
class ImageClassificationData:
    id: str
    image: torch.Tensor
    label: int


@dataclasses.dataclass
class ImageClassificationBatch:
    ids: list[str]
    images: torch.Tensor
    labels: torch.Tensor

    def to(self, device: torch.device) -> "ImageClassificationBatch":
        return ImageClassificationBatch(
            ids=self.ids,
            images=self.images.to(device),
            labels=self.labels.to(device),
        )


@dataclasses.dataclass
class ImageDetectionData:
    image: torch.Tensor
    labels: torch.Tensor
    boxes: torch.Tensor
    image_id: int
    orig_height: int
    orig_width: int


@dataclasses.dataclass
class ImageDetectionBatch:
    images: torch.Tensor
    batch_idx: torch.Tensor
    labels: torch.Tensor
    boxes: torch.Tensor
    image_ids: list[int]
    orig_heights: list[int]
    orig_widths: list[int]

    def to(self, device: torch.device) -> "ImageDetectionBatch":
        return ImageDetectionBatch(
            images=self.images.to(device),
            batch_idx=self.batch_idx.to(device),
            labels=self.labels.to(device),
            boxes=self.boxes.to(device),
            image_ids=self.image_ids,
            orig_heights=self.orig_heights,
            orig_widths=self.orig_widths,
        )


@dataclasses.dataclass
class EventsClassificationData:
    tokens: EventTokens
    label: torch.Tensor


@dataclasses.dataclass
class EventsClassificationBatch:
    tokens: EventTokens
    labels: torch.Tensor
    padding_mask: torch.Tensor


PROPHESEE_BBOX_DTYPE = np.dtype(
    {
        "names": ["t", "x", "y", "w", "h", "class_id", "class_confidence"],
        "formats": ["<i8", "<f4", "<f4", "<f4", "<f4", "<u4", "<f4"],
    }
)


@dataclasses.dataclass
class EventDetectionData:
    tokens: EventTokens
    labels: torch.Tensor
    boxes: torch.Tensor
    prediction_time: int


@dataclasses.dataclass
class EventDetectionBatch:
    tokens: EventTokens
    batch_idx: torch.Tensor
    labels: torch.Tensor
    boxes: torch.Tensor
    padding_mask: torch.Tensor
    prediction_times: list[int]

    def to(self, device: torch.device) -> "EventDetectionBatch":
        return EventDetectionBatch(
            tokens=self.tokens.to(device),
            batch_idx=self.batch_idx.to(device),
            labels=self.labels.to(device),
            boxes=self.boxes.to(device),
            padding_mask=self.padding_mask.to(device),
            prediction_times=self.prediction_times,
        )


@dataclasses.dataclass
class VideoClassificationData:
    video: torch.Tensor
    length: int
    label: int


@dataclasses.dataclass
class VideoClassificationBatch:
    videos: torch.Tensor
    lengths: torch.Tensor
    labels: torch.Tensor

    def to(self, device: torch.device) -> "VideoClassificationBatch":
        return VideoClassificationBatch(
            videos=self.videos.to(device),
            lengths=self.lengths.to(device),
            labels=self.labels.to(device),
        )
