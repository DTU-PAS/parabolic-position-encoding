from math import ceil
from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp
from tempfile import mkstemp
from typing import BinaryIO

import ffmpeg
import numpy as np

from pape.data_types import Events


class EventsVideo:
    """Create a video from the events.

    Blue and red pixels represent positive and negative events, respectively.
    """

    def __init__(
        self,
        events: Events,
        height: int,
        width: int,
        fps: int = 30,
        boxes: list[tuple[int, int, int, int]] | None = None,
    ):
        self.events = events
        self.height = height
        self.width = width
        self.fps = fps
        self.boxes = boxes

    def __enter__(self):
        self.frames_dir = Path(mkdtemp())
        _, video_path = mkstemp(suffix=".mp4")
        self.video_path = Path(video_path)

        self.make_video()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        rmtree(self.frames_dir)
        self.video_path.unlink()

    def read(self) -> bytes:
        """Read the video."""
        with open(self.video_path, "rb") as video:
            return video.read()

    def save(self, buffer_or_path: str | Path | BinaryIO):
        """Save the video to a buffer or a file."""
        if isinstance(buffer_or_path, (str, Path)):
            buffer_or_path = open(buffer_or_path, "wb")
        with open(self.video_path, "rb") as video:
            buffer_or_path.write(video.read())
        buffer_or_path.close()

    def make_video(self):
        red = (255, 0, 0)
        blue = (0, 0, 255)

        window_size_us = (1000 // self.fps) * (10**3)
        min_timestamp = self.events.t.min()
        max_timestamp = self.events.t.max()
        num_frames = ceil((max_timestamp - min_timestamp) / window_size_us)

        positive_frames = self.make_frames(
            self.events.mask(self.events.p == 1), min_timestamp, window_size_us, num_frames
        )
        negative_frames = self.make_frames(
            self.events.mask(self.events.p == 0), min_timestamp, window_size_us, num_frames
        )

        frames = []
        for frame_index in range(num_frames):
            positive = positive_frames[frame_index]
            negative = negative_frames[frame_index]
            diff = positive - negative

            image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            image[diff < 0] = red
            image[diff > 0] = blue

            if self.boxes is not None:
                border_colour = (0, 255, 0)
                for box in self.boxes:
                    x, y, width, height = box

                    if x < 0:
                        width += x
                        x = 0

                    if y < 0:
                        height += y
                        y = 0

                    if x + width >= self.width:
                        width = self.width - x - 1

                    if y + height >= self.height:
                        height = self.height - y - 1

                    # draw top border
                    image[y, x : x + width, :] = border_colour
                    # draw bottom border
                    image[y + height, x : x + width, :] = border_colour
                    # draw left border
                    image[y : y + height, x, :] = border_colour
                    # draw right border
                    image[y : y + height, x + width, :] = border_colour

            frames.append(image.tobytes())

        video = ffmpeg.input("pipe:", format="rawvideo", r=self.fps, s=f"{self.width}x{self.height}", pix_fmt="rgb24")
        video = video.output(str(self.video_path))
        video = video.overwrite_output()
        video = video.run_async(pipe_stdin=True, quiet=True)

        video_bytes = b"".join(frames)
        video.stdin.write(video_bytes)
        video.stdin.close()
        video.wait()

    def make_frames(self, events: Events, min_timestamp: int, window_size_us: int, num_frames: int):
        frame_bins = (events.t - min_timestamp) // window_size_us

        x = events.x.astype(np.uint32)
        y = events.y.astype(np.uint32)

        area = self.height * self.width
        positions = (frame_bins * area) + (y * self.width) + x

        shape = (num_frames, self.height, self.width)
        frames = np.bincount(positions, minlength=np.prod(shape))
        frames = frames.reshape(shape)

        return frames
