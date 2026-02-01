import numpy as np

from pape.data_types import Events


def preprocess_events(
    events: Events,
    height: int,
    width: int,
    min_time: int,
) -> Events:
    if len(events) == 0:
        return events
    events = sort_events(events, min_time)
    events = remove_out_of_bounds(events, height, width)
    return events


def sort_events(events: Events, min_time: int) -> Events:
    """Most events are already sorted by time, but occasionally they are not.

    We further guarantee a minimum time to ensure sorting across adjacent chunks.
    """
    t = np.maximum(events.t, min_time)
    indices = np.argsort(t)
    return Events(
        x=events.x[indices],
        y=events.y[indices],
        t=t[indices],
        p=events.p[indices],
    )


def remove_out_of_bounds(events: Events, height: int, width: int) -> Events:
    mask = (events.x >= 0) & (events.x < width) & (events.y >= 0) & (events.y < height)
    return events.mask(mask)
