import dataclasses
import json
from pathlib import Path

import h5py
import numpy as np

from pape.data_types import Events


@dataclasses.dataclass
class Chunk:
    id: int
    start_time_us: int
    end_time_us: int
    count: int


@dataclasses.dataclass
class Sequence:
    source_dir: Path
    name: str
    chunks: list[Chunk]


def load_sequences(source_dir: Path) -> list[Sequence]:
    index_path = source_dir / "index.json"
    sequences = json.loads(index_path.read_text())
    return [
        Sequence(
            source_dir=source_dir,
            name=sequence["name"],
            chunks=[
                Chunk(
                    id=chunk["index"],
                    start_time_us=chunk["start_time"],
                    end_time_us=chunk["end_time"],
                    count=chunk["count"],
                )
                for chunk in sequence["chunks"]
            ],
        )
        for sequence in sequences
    ]


def load_chunks(
    sequence: Sequence,
    end_chunk_index: int,
    end_time_us: int,
    max_events: int,
) -> Events:
    last_chunk = sequence.chunks[end_chunk_index]
    last_events = load_events_by_time(sequence.source_dir / sequence.name / f"{last_chunk.id}.h5", end_time_us)
    total_events = len(last_events)
    if total_events >= max_events:
        # keep most recent max_events
        events = last_events[-max_events:]
        return events

    events = [last_events]
    chunk_index = end_chunk_index
    while total_events < max_events and chunk_index > 0:
        chunk_index -= 1
        chunk = sequence.chunks[chunk_index]
        chunk_path = sequence.source_dir / sequence.name / f"{chunk.id}.h5"
        if total_events + chunk.count > max_events:
            chunk_events = load_events_by_count(chunk_path, max_events - total_events)
            total_events = max_events
        else:
            chunk_events = load_events(chunk_path)
            total_events += chunk.count
        events.append(chunk_events)

    events.reverse()
    return Events(
        x=np.concatenate([chunk_events.x for chunk_events in events]),
        y=np.concatenate([chunk_events.y for chunk_events in events]),
        t=np.concatenate([chunk_events.t for chunk_events in events]),
        p=np.concatenate([chunk_events.p for chunk_events in events]),
    )


def load_events(input_path: Path) -> Events:
    with h5py.File(input_path, "r") as file:
        x = file["x"][()]
        y = file["y"][()]
        t = file["t"][()]
        p = file["p"][()]

    t = t.astype(np.uint64)

    return Events(x=x, y=y, t=t, p=p)


def load_events_by_count(input_path: Path, count: int) -> Events:
    with h5py.File(input_path, "r") as file:
        x = file["x"][-count:]
        y = file["y"][-count:]
        t = file["t"][-count:]
        p = file["p"][-count:]

    t = t.astype(np.uint64)

    return Events(x=x, y=y, t=t, p=p)


def load_events_by_time(input_path: Path, end_time_us: int) -> Events:
    with h5py.File(input_path, "r") as file:
        t = file["t"][()]

        end_index = np.searchsorted(t, end_time_us, side="right")
        t = t[:end_index]

        x = file["x"][:end_index]
        y = file["y"][:end_index]
        p = file["p"][:end_index]

    t = t.astype(np.uint64)

    return Events(x=x, y=y, t=t, p=p)


def save_events(output_path: Path, events: Events, compress: bool = True):
    kwargs = {"compression": "gzip"} if compress else {}
    with h5py.File(output_path, "w") as out_file:
        out_file.create_dataset(
            "x",
            data=events.x,
            dtype=np.uint16,
            **kwargs,
        )
        out_file.create_dataset(
            "y",
            data=events.y,
            dtype=np.uint16,
            **kwargs,
        )
        out_file.create_dataset(
            "t",
            data=events.t,
            dtype=np.uint32,
            **kwargs,
        )
        out_file.create_dataset(
            "p",
            data=events.p,
            dtype=bool,
            **kwargs,
        )
