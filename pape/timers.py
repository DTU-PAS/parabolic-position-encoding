import atexit
import collections
import os
import time
from contextlib import ContextDecorator

import numpy as np
import torch
from rich.console import Console
from rich.table import Column
from rich.table import Table

counts = collections.Counter()
timers = collections.defaultdict(list)

WARMUP_STEPS = 10

ENABLED_ENV_NAME = "ENABLE_TIMING"


class Timer(ContextDecorator):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.enabled = os.environ.get(ENABLED_ENV_NAME, "0") == "1"

    def __enter__(self):
        counts[self.name] += 1
        if not self.enabled or counts[self.name] <= WARMUP_STEPS:
            return
        self.start = time.perf_counter()

    def __exit__(self, *exc):
        if not self.enabled or counts[self.name] <= WARMUP_STEPS:
            return
        elapsed = time.perf_counter() - self.start
        elapsed = elapsed * 1000  # convert to milliseconds
        timers[self.name].append(elapsed)


class CudaTimer(ContextDecorator):
    def __init__(self, name: str, enabled: bool = True):
        super().__init__()
        self.name = name
        self.enabled = os.environ.get(ENABLED_ENV_NAME, "0") == "1"

    def __enter__(self):
        counts[self.name] += 1
        if not self.enabled or counts[self.name] <= WARMUP_STEPS:
            return
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()

    def __exit__(self, *exc):
        if not self.enabled or counts[self.name] <= WARMUP_STEPS:
            return
        self.end.record()
        torch.cuda.synchronize()
        elapsed = self.start.elapsed_time(self.end)
        timers[self.name].append(elapsed)


def print_timing_statistics():
    if len(timers) == 0:
        return

    table = Table(
        "Name",
        Column(header="Mean", justify="right"),
        Column(header="Std", justify="right"),
        Column(header="Min", justify="right"),
        Column(header="25%", justify="right"),
        Column(header="50%", justify="right"),
        Column(header="75%", justify="right"),
        Column(header="Max", justify="right"),
        title="Timing Statistics",
    )
    for name, timings in timers.items():
        timings = np.array(timings)
        table.add_row(
            name,
            f"{np.mean(timings):,.1f} ms",
            f"{np.std(timings):,.1f} ms",
            f"{np.min(timings):,.1f} ms",
            f"{np.percentile(timings, 25):,.1f} ms",
            f"{np.percentile(timings, 50):,.1f} ms",
            f"{np.percentile(timings, 75):,.1f} ms",
            f"{np.max(timings):,.1f} ms",
        )
    console = Console()
    console.print(table)


atexit.register(print_timing_statistics)
