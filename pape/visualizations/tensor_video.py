from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp
from tempfile import mkstemp
from typing import BinaryIO

import ffmpeg
import torch


class TensorVideo:
    """Create a video from an input tensor of shape TxCxHxW."""

    def __init__(self, tensor: torch.Tensor, fps: int = 30):
        self.tensor = tensor
        self.fps = fps

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
        tensor = self.tensor.permute(0, 2, 3, 1).numpy()

        frames = []
        for frame_index in range(self.tensor.size(0)):
            frame = tensor[frame_index]
            frame = frame.tobytes()
            frames.append(frame)

        width = self.tensor.size(3)
        height = self.tensor.size(2)
        video = ffmpeg.input("pipe:", format="rawvideo", r=self.fps, s=f"{width}x{height}", pix_fmt="rgb24")
        video = video.output(str(self.video_path))
        video = video.overwrite_output()
        video = video.run_async(pipe_stdin=True, quiet=True)

        video_bytes = b"".join(frames)
        video.stdin.write(video_bytes)
        video.stdin.close()
        video.wait()
