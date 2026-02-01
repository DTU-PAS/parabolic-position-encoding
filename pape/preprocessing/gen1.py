import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from pape.configs import Split
from pape.data_types import Events
from pape.gen1 import DATASET_NAME
from pape.gen1 import HEIGHT
from pape.gen1 import MIN_BOX_DIAG
from pape.gen1 import MIN_BOX_SIDE
from pape.gen1 import WIDTH
from pape.io import save_events
from pape.paths import get_dataset_dir
from pape.preprocessing.events import preprocess_events
from pape.prophesee.loader import PSEELoader


class PropheseeLoader:
    def __init__(self, data_path: Path):
        super().__init__()
        self.loader = PSEELoader(str(data_path))

    def close(self):
        del self.loader

    def done(self):
        return self.loader.done

    def load_delta_t(self, delta_t: int) -> Events:
        events = self.loader.load_delta_t(delta_t)
        return Events(
            x=events["x"],
            y=events["y"],
            t=events["t"],
            p=events["p"],
        )

    def load_past(self) -> Events:
        pos = self.loader.file.tell()
        past_count = (pos - self.loader.start) // self.loader.ev_size
        events = self.loader.load_n_past_events(past_count)
        return Events(
            x=events["x"],
            y=events["y"],
            t=events["t"],
            p=events["p"],
        )

    def seek_time(self, time: int):
        self.loader.seek_time(time)


LABELS_DTYPE = np.dtype(
    [
        ("sequence", np.uint32),
        ("chunk", np.uint32),
        ("event", np.uint32),
        ("t", np.uint64),
        ("x", np.float32),
        ("y", np.float32),
        ("w", np.float32),
        ("h", np.float32),
        ("class_id", np.uint8),
    ]
)


class Gen1Preprocessor:
    """Preprocess the GEN1 object detection dataset."""

    def __init__(
        self,
        chunk_duration_ms: int,
        limit: int | None,
        test: bool,
        train: bool,
        val: bool,
    ):
        super().__init__()

        self.chunk_duration_us = chunk_duration_ms * 1000
        self.limit = limit
        self.train = train
        self.val = val
        self.test = test
        self.source_dir = get_dataset_dir(DATASET_NAME)
        self.output_dir = get_dataset_dir(f"{DATASET_NAME}-preprocessed", raise_missing=False)

    def get_split_paths(self, split: Split) -> list[Path]:
        split_dir = self.source_dir / split.value
        return list(split_dir.glob("*_bbox.npy"))

    def preprocess(self):
        # save config
        config = {
            "chunk_duration_ms": self.chunk_duration_us // 1000,
            "limit": self.limit,
        }
        config_path = self.output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        if self.train:
            self.preprocess_split(Split.train)
        if self.val:
            self.preprocess_split(Split.val)
        if self.test:
            self.preprocess_split(Split.test)

    def preprocess_split(self, split: Split):
        labels_paths = self.get_split_paths(split)
        if self.limit is not None:
            labels_paths = labels_paths[: self.limit]
        output_dir = self.output_dir / split.value
        output_dir.mkdir(exist_ok=True, parents=True)
        index = []
        sequence_id = 0
        for labels_path in tqdm(labels_paths, desc=f"Preprocessing {split.value}"):
            sequence_dir = output_dir / labels_path.stem.removesuffix("_bbox")
            sequence_dir.mkdir(exist_ok=True, parents=True)
            index.append(self.preprocess_sequence(split, sequence_dir, sequence_id, labels_path))
            sequence_id += 1
        index_path = output_dir / "index.json"
        with open(index_path, "w") as f:
            json.dump(index, f)

    def preprocess_sequence(self, split: Split, output_dir: Path, sequence_id: int, labels_path: Path) -> dict:
        sequence_name = labels_path.stem.removesuffix("_bbox")
        events_filename = f"{sequence_name}_td.dat"
        events_path = labels_path.parent / events_filename
        chunks = []
        loader = PropheseeLoader(events_path)
        loader.seek_time(self.chunk_duration_us)
        events = loader.load_past()
        events = preprocess_events(events, HEIGHT, WIDTH, 0)
        chunk_index = 0
        if len(events) > 0:
            start_time = 0
            end_time = self.chunk_duration_us - 1
            chunks.append(
                {
                    "index": chunk_index,
                    "start_time": start_time,
                    "end_time": end_time,
                    "count": len(events),
                    "t": events.t,
                }
            )
            output_path = output_dir / f"{chunk_index}.h5"
            save_events(output_path, events)
            chunk_index += 1
        time_offset = self.chunk_duration_us
        while not loader.done():
            events = loader.load_delta_t(self.chunk_duration_us)
            events = preprocess_events(events, HEIGHT, WIDTH, time_offset)
            if len(events) > 0:
                start_time = time_offset
                end_time = time_offset + self.chunk_duration_us - 1
                chunks.append(
                    {
                        "index": chunk_index,
                        "start_time": start_time,
                        "end_time": end_time,
                        "count": len(events),
                        "t": events.t,
                    }
                )
                output_path = output_dir / f"{chunk_index}.h5"
                save_events(output_path, events)
                chunk_index += 1
            time_offset += self.chunk_duration_us
        loader.close()
        labels = np.load(labels_path)
        labels = self.preprocess_labels(labels, split, sequence_id, chunks)
        labels_path = output_dir / "bbox.npy"
        np.save(labels_path, labels)
        for chunk in chunks:
            # We only needed to store timestamps for preprocessing labels, so we can remove them now
            del chunk["t"]
        return {"name": sequence_name, "chunks": chunks}

    def preprocess_labels(self, labels: np.ndarray, split: Split, sequence: int, chunks: list[dict]) -> np.ndarray:
        labels = self.crop_boxes_outside_of_image(labels)
        labels = self.remove_small_boxes(labels)
        if split == Split.train:
            labels = self.remove_faulty_huge_bbox(labels)
        labels = self.add_chunk_location(labels, sequence, chunks)
        return labels

    def crop_boxes_outside_of_image(self, labels: np.ndarray) -> np.ndarray:
        left = labels["x"]
        right = labels["x"] + labels["w"]
        top = labels["y"]
        bottom = labels["y"] + labels["h"]

        left = np.clip(left, 0, WIDTH - 1)
        right = np.clip(right, 0, WIDTH - 1)
        top = np.clip(top, 0, HEIGHT - 1)
        bottom = np.clip(bottom, 0, HEIGHT - 1)

        width = right - left
        height = bottom - top

        assert np.all(width >= 0)
        assert np.all(height >= 0)

        cropped_labels = labels.copy()
        cropped_labels["x"] = left
        cropped_labels["y"] = top
        cropped_labels["w"] = width
        cropped_labels["h"] = height

        # remove boxes with zero area
        # this may happen if the box is completely outside of the image
        mask = (width > 0) & (height > 0)
        cropped_labels = cropped_labels[mask]

        return cropped_labels

    def remove_small_boxes(self, labels: np.ndarray) -> np.ndarray:
        width = labels["w"]
        height = labels["h"]
        diagonal_mask = width**2 + height**2 >= MIN_BOX_DIAG**2
        width_mask = width >= MIN_BOX_SIDE
        height_mask = height >= MIN_BOX_SIDE
        mask = diagonal_mask & width_mask & height_mask
        return labels[mask]

    def remove_faulty_huge_bbox(self, labels: np.ndarray) -> np.ndarray:
        """There are some labels which span the frame horizontally without actually covering an object.
        Source: https://github.com/uzh-rpg/RVT/blob/master/scripts/genx/preprocess_dataset.py#L222
        """
        width = labels["w"]
        max_width = (9 * WIDTH) // 10
        mask = width <= max_width
        labels = labels[mask]
        return labels

    def add_chunk_location(self, labels: list[np.ndarray], sequence: int, chunks: list[dict]) -> np.ndarray:
        if len(labels) == 0:
            return np.empty(0, dtype=LABELS_DTYPE)
        labels = labels[np.argsort(labels["ts"])]
        _, indices = np.unique(labels["ts"], return_index=True)
        groups = np.split(labels, indices)[1:]
        labels = []
        chunk_index = 0
        for group in groups:
            timestamp = group[0]["ts"]
            while chunk_index < len(chunks) and timestamp > chunks[chunk_index]["end_time"]:
                chunk_index += 1
            chunk_index = min(chunk_index, len(chunks) - 1)
            chunk_labels = np.empty(len(group), dtype=LABELS_DTYPE)
            chunk_labels["sequence"] = sequence
            chunk_labels["chunk"] = chunk_index
            chunk_labels["event"] = np.searchsorted(chunks[chunk_index]["t"], timestamp, side="right")
            chunk_labels["t"] = group["ts"]
            chunk_labels["x"] = group["x"]
            chunk_labels["y"] = group["y"]
            chunk_labels["w"] = group["w"]
            chunk_labels["h"] = group["h"]
            chunk_labels["class_id"] = group["class_id"]
            labels.append(chunk_labels)
        labels = np.concatenate(labels)
        return labels
