import numpy as np
import torch

import spiking_patches
from pape.configs import Config
from pape.data_types import Events
from pape.data_types import EventTokens


class Tokenizer:
    """This is a Python wrapper around the Rust tokenizer."""

    def __init__(self, config: Config):
        self.buckets = config.events.buckets
        self.patch_size = config.patch_size

        self.height = config.height
        self.width = config.width

        self.tokenizer = spiking_patches.BatchTokenizer(
            height=config.height,
            patch_size=self.patch_size,
            refractory_period=config.events.ref_us,
            threshold=config.events.threshold,
            width=config.width,
        )

    def __call__(self, events: Events) -> EventTokens:
        """Tokenizes a sequence of events into tokens (spiking patches)."""

        if events.x.dtype != np.uint16 or events.y.dtype != np.uint16:
            raise ValueError("x and y must be of type uint16")

        if events.t.dtype != np.uint64:
            raise ValueError("t must be of type uint64.")

        if events.p.dtype != bool:
            raise ValueError("p must be of type bool.")

        tokens = self.tokenizer.tokenize_batch([(events.x, events.y, events.t, events.p)])[0]
        x, y, t, events_x, events_y, events_t, events_p = tokens

        x = torch.tensor(x, dtype=torch.uint16)
        y = torch.tensor(y, dtype=torch.uint16)
        t = torch.tensor(t, dtype=torch.uint64)

        batch = [np.full_like(value, i) for i, value in enumerate(events_x)]
        patches = self.batched_events_to_logspace_volume(
            batch,
            events_x,
            events_y,
            events_t,
            events_p,
            buckets=self.buckets,
            height=self.patch_size,
            width=self.patch_size,
        )
        patches = torch.tensor(patches, dtype=torch.int64)

        return EventTokens(
            x=x,
            y=y,
            t=t,
            patches=patches,
        )

    def batched_events_to_logspace_volume(
        self,
        batch: list[np.ndarray],
        x: list[np.ndarray],
        y: list[np.ndarray],
        t: list[np.ndarray],
        p: list[np.ndarray],
        buckets: int,
        height: int,
        width: int,
        time_base_us: int = 1000,
        power_base: int = 2,
    ) -> np.ndarray:
        batch_size = len(batch)
        shape = (batch_size, 2, buckets, height, width)
        if batch_size == 0:
            return np.zeros(shape, dtype=np.int64)

        batch = np.concatenate(batch)

        x = np.concatenate(x) % width
        y = np.concatenate(y) % height

        t = [values[-1] - values for values in t]
        t = np.concatenate(t)

        p = np.concatenate(p)

        bucket_ends = time_base_us * np.power(power_base, np.arange(buckets))
        t_bucket = np.digitize(t, bucket_ends)
        t_bucket = np.minimum(t_bucket, buckets - 1)

        batch = batch.astype(np.uint32)
        x = x.astype(np.uint32)
        y = y.astype(np.uint32)
        t_bucket = t_bucket.astype(np.uint32)
        p = p.astype(np.uint32)

        area_size = height * width
        volume_size = buckets * area_size
        batch_volume_size = 2 * volume_size

        positions = (batch * batch_volume_size) + (p * volume_size) + (t_bucket * area_size) + (y * width) + x

        positions = positions.astype(np.int64)
        length = np.prod(shape, dtype=np.int64)

        assert positions.min() >= 0
        assert positions.max() <= length

        volume = np.bincount(positions, minlength=length)
        volume = volume.reshape(shape)

        return volume
