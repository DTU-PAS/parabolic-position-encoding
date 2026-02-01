import dataclasses
import json
from hashlib import md5
from typing import Any

import dataclasses_json
import pandas as pd
from tqdm import tqdm

from pape.aedat import AedatReader
from pape.configs import Split
from pape.dvs_gesture import DATASET_NAME
from pape.dvs_gesture import HEIGHT
from pape.dvs_gesture import WIDTH
from pape.io import save_events
from pape.paths import get_dataset_dir
from pape.preprocessing.events import preprocess_events


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class Sample:
    action_id: str
    end: int
    filename: str
    illumination: str
    label: int
    split: str
    start: int
    user: int


class DVSGesturePreprocessor:
    """Preprocess DvsGesture."""

    def __init__(
        self,
        limit: int | None,
        test: bool,
        train: bool,
    ):
        super().__init__()
        self.limit = limit
        self.train = train
        self.test = test
        self.source_dir = get_dataset_dir(DATASET_NAME) / "DvsGesture"
        self.output_dir = get_dataset_dir(f"{DATASET_NAME}-preprocessed", raise_missing=False)

    def preprocess(self):
        if not self.train and not self.test:
            print("No splits selected for preprocessing.")
            return
        self.output_dir.mkdir(exist_ok=True, parents=True)

        if self.train:
            self.preprocess_split(Split.train)
        if self.test:
            self.preprocess_split(Split.test)

    def preprocess_split(self, split: Split):
        samples = self.get_samples(split)

        if self.limit is not None:
            samples = dict(list(samples.items())[: self.limit])

        for source_name, actions in tqdm(samples.items(), desc=split.value, unit="recording", total=len(samples)):
            self.preprocess_recording(source_name, actions)

        output_path = self.output_dir / f"{split.value}.json"
        samples = [sample.to_dict() for recording in samples.values() for sample in recording]
        with output_path.open("w") as file:
            json.dump(samples, file)

    def get_samples(self, split: Split):
        trials_path = self.source_dir / f"trials_to_{split.value}.txt"
        trials = trials_path.read_text().strip().split("\n")

        groups: dict[str, Sample] = {}
        for recording_file_name in trials:
            name = recording_file_name.split(".", maxsplit=1)[0]
            user, illumination = name.split("_", maxsplit=1)
            user = int(user[-2:])
            labels_path = self.source_dir / f"{name}_labels.csv"
            labels = pd.read_csv(labels_path)
            group = []
            for _, row in labels.iterrows():
                label = row["class"].item()
                time_hash = self.params_to_hash([("start", row["startTime_usec"]), ("end", row["endTime_usec"])])
                sample = Sample(
                    action_id=f"{user}_{illumination}_{label}",
                    end=row["endTime_usec"].item(),
                    filename=f"{name}_{time_hash}",
                    illumination=illumination,
                    label=label,
                    split=split.value,
                    start=row["startTime_usec"].item(),
                    user=user,
                )
                group.append(sample)
            groups[recording_file_name] = group

        return groups

    def preprocess_recording(self, source_name: str, actions: list[Sample]):
        source_path = self.source_dir / source_name
        with AedatReader(source_path) as aedat:
            events = aedat.read()

        for sample in actions:
            mask = (events.t >= sample.start) & (events.t <= sample.end)
            action_events = events.mask(mask)
            t = action_events.t - sample.start
            action_events = dataclasses.replace(action_events, t=t)
            action_events = preprocess_events(
                events=action_events,
                height=HEIGHT,
                min_time=0,
                width=WIDTH,
            )

            save_path = self.output_dir / f"{sample.filename}.h5"
            save_events(save_path, action_events, compress=False)

    def params_to_hash(self, params: list[tuple[str, Any]]):
        params = [f"{name}={value}" for name, value in params if value is not None]
        params = "-".join(params).encode()
        params_hash = md5(params).hexdigest()
        return params_hash[:16]
