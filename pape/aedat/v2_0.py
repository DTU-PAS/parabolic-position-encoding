import struct

import numpy as np

from pape.data_types import Events

LOCATION_MASK = 0x007F
POLARITY_MASK = 0x0001


def parse_aedat_v2_0(recording: bytes) -> Events:
    """Specification: https://docs.inivation.com/software/software-advanced-usage/file-formats/aedat-2.0.html

    This methods assumes a DVS128 chip class.
    It means that the method will yield incorrect x, y, and polarity values for other chip classes.
    Do not use this method if you are not sure about the chip class.
    """

    data = []
    ts = []
    index = find_body_start(recording)
    while index < len(recording):
        xyp = int.from_bytes(recording[index : index + 4])
        data.append(xyp)
        index += 4

        t = struct.unpack(">I", recording[index : index + 4])[0]
        ts.append(t)
        index += 4

    data = np.array(data, dtype=np.uint32)
    xs = (data >> 1) & LOCATION_MASK
    ys = (data >> 8) & LOCATION_MASK
    ps = data & POLARITY_MASK

    events = Events(
        x=np.array(xs, dtype=np.uint16),
        y=np.array(ys, dtype=np.uint16),
        t=np.array(ts, dtype=np.uint64),
        p=np.array(ps, dtype=bool),
    )

    if not (events.t[1:] >= events.t[:-1]).all():
        count = (events.t[1:] < events.t[:-1]).sum()
        print(f"Found {count:,} out of {len(events):,} events in descending order. Fixing by sorting events.")

        sorted_indices = np.argsort(events.t)
        events = events[sorted_indices]

    return events


def find_body_start(recording: bytes) -> int:
    """All header lines start with #. Finds first line that does not start with #."""
    index = 0
    header_line_start = ord(b"#")
    new_line = ord(b"\n")
    while recording[index] == header_line_start:
        while recording[index] != new_line:
            index += 1
        index += 1
    return index
