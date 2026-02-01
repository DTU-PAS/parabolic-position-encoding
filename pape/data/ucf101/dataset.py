from collections import defaultdict

import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from torchcodec.decoders import VideoDecoder

from pape.configs import Config
from pape.configs import Split
from pape.data_types import VideoClassificationData
from pape.paths import get_dataset_dir


class UCF101Dataset(torch.utils.data.Dataset):
    def __init__(self, config: Config, path_to_label: dict[str, int], split: Split):
        super().__init__()

        self.fold = config.fold
        self.frame_length = config.video.frame_length
        self.frame_step = config.video.frame_step
        self.max_samples = config.video.max_samples
        self.path_to_label = path_to_label

        self.root_dir = get_dataset_dir("ucf_101")
        self.video_dir = self.root_dir / "UCF-101"
        self.paths = self.get_video_paths(split)

        if split == Split.train:
            self.transform = T.Compose(
                [
                    T.RandomHorizontalFlip(),
                    T.RandAugment(),
                    T.Resize((config.height, config.width)),
                ]
            )
        else:
            self.transform = T.Resize((config.height, config.width))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        label = self.path_to_label[path]
        video = self.read_video(path)
        video = self.transform(video)

        # pad to max_samples
        length = video.size(0)
        max_frames = self.max_samples * self.frame_length
        frame_diff = max_frames - length
        if frame_diff > 0:
            video = F.pad(video, (0, 0, 0, 0, 0, 0, 0, frame_diff))

        # TCHW -> CTWH
        video = video.permute(1, 0, 2, 3)

        return VideoClassificationData(
            label=label,
            length=length // self.frame_length,
            video=video,
        )

    def get_video_paths(self, split: Split) -> list[str]:
        prefix = "test" if split == Split.test else "train"
        split_file = self.root_dir / "ucfTrainTestlist" / f"{prefix}list0{self.fold}.txt"
        lines = split_file.read_text().strip().split("\n")
        video_paths = []
        for line in lines:
            video_path = line.strip().split()[0]
            video_paths.append(video_path)
        if split == Split.train:
            video_paths, _ = self.train_val_split(video_paths)
        elif split == Split.val:
            _, video_paths = self.train_val_split(video_paths)
        return video_paths

    def train_val_split(self, video_paths: list[str]) -> tuple[list[str], list[str]]:
        """Splits the video paths into training and validation sets.

        Uses the first group of each action class for validation.
        """
        action_to_paths = defaultdict(list)
        for path in video_paths:
            action = path.split("/")[0]
            action_to_paths[action].append(path)

        train_paths = []
        val_paths = []
        for paths in action_to_paths.values():
            # path format: <action_class>/v_<action_class>_g<group>_c<clip>.avi
            first_path = paths[0]
            group_id = first_path.split("_g")[1].split("_")[0]
            for path in paths:
                current_group_id = path.split("_g")[1].split("_")[0]
                if current_group_id == group_id:
                    val_paths.append(path)
                else:
                    train_paths.append(path)

        return train_paths, val_paths

    def read_video(self, path: str) -> torch.Tensor:
        decoder = VideoDecoder(self.video_dir / path)

        # sample frames uniformly with a fixed distance, but make sure to keep neighbouring frames
        # e.g., say we have 30 frames at distance 10 and want to keep 2 frames per sample,
        # then we get indices [0, 1, 10, 11, 20, 21]
        max_frames = self.max_samples * self.frame_step
        num_frames = min(len(decoder), max_frames)
        indices = []
        for i in range(self.frame_length):
            step_indices = list(range(i, num_frames, self.frame_step))
            indices.extend(step_indices)
        indices = sorted(indices)
        if len(indices) % self.frame_length != 0:
            end_index = -(len(indices) % self.frame_length)
            indices = indices[:end_index]

        frames = decoder.get_frames_at(indices)

        return frames.data
