import struct

import numpy as np

from pape.data_types import Events

HEADER_END = b"#!END-HEADER\r\n"
LOCATION_MASK = 0x00001FFF
POLARITY_MASK = 0x00000001

HEADER_KEYS = (
    "event_type",
    "event_source",
    "event_size",
    "event_offset",
    "event_overflow",
    "event_capacity",
    "event_number",
    "event_valid",
)


def parse_aedat_v3_1(recording: bytes) -> Events:
    """Read all events from an AEDAT 3.1 file."""

    timestamps = []
    data = []

    index = find_body_start(recording)
    while index < len(recording):
        header_values = struct.unpack("<HHIIIIII", recording[index : index + 28])
        header = dict(zip(HEADER_KEYS, header_values, strict=True))
        index += 28

        data_size = header["event_size"] * header["event_capacity"]

        package_events = np.frombuffer(recording[index : index + data_size], dtype=np.uint32)
        package_events = package_events.reshape((header["event_capacity"], header["event_size"] // 4))

        timestamp = package_events[:, 1]
        timestamp = timestamp.astype(np.uint64, casting="safe")
        timestamp = (header["event_overflow"] << 31) | timestamp
        timestamps.append(timestamp)

        data.append(package_events[:, 0])

        index += data_size

    timestamps = np.concatenate(timestamps, dtype=np.uint64)

    data = np.concatenate(data)
    polarity = (data >> 1) & POLARITY_MASK
    x = (data >> 17) & LOCATION_MASK
    y = (data >> 2) & LOCATION_MASK

    x = x.astype(np.uint16)
    y = y.astype(np.uint16)
    polarity = polarity.astype(bool)

    assert len(x) == len(y) == len(timestamps) == len(polarity), (
        "Lengths of x, y, timestamps, and polarity do not match."
    )

    if not (timestamps[1:] >= timestamps[:-1]).all():
        print("Timestamps are not in ascending order. Fixing by sorting events wrt. timestamps.")
        sorted_indices = np.argsort(timestamps)
        x = x[sorted_indices]
        y = y[sorted_indices]
        timestamps = timestamps[sorted_indices]
        polarity = polarity[sorted_indices]

    return Events(
        x=x,
        y=y,
        t=timestamps,
        p=polarity,
    )


def find_body_start(recording: bytes) -> int:
    last_chars = recording[: len(HEADER_END)]
    index = 0
    while last_chars != HEADER_END:
        index += 1
        last_chars = recording[index : index + len(HEADER_END)]
    index += len(HEADER_END)
    return index
