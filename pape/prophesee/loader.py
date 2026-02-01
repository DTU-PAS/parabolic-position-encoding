# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
This class loads events from dat or npy files
"""

from __future__ import print_function

import os

import numpy as np

from . import dat_events_tools as dat
from . import npy_events_tools as npy_format


class PSEELoader(object):
    """
    PSEELoader loads a dat or npy file and stream events
    """

    def __init__(self, datfile):
        """
        ctor
        :param datfile: binary dat or npy file
        """
        self.extension = datfile.split(".")[-1]
        assert self.extension in ["dat", "npy"], "input file path = {}".format(datfile)
        if self.extension == "dat":
            self.binary_format = dat
        elif self.extension == "npy":
            self.binary_format = npy_format
        self.file = open(datfile, "rb")
        self.start, self.ev_type, self.ev_size, self.size = self.binary_format.parse_header(self.file)
        assert self.ev_size != 0
        if self.extension == "dat":
            self.dtype = self.binary_format.EV_TYPE
        elif self.extension == "npy":
            self.dtype = self.ev_type
        else:
            raise AssertionError("unsupported extension")

        self.decode_dtype = []
        for dtype in self.dtype:
            if dtype[0] == "_":
                self.decode_dtype += [("x", "u2"), ("y", "u2"), ("p", "u1")]
            else:
                self.decode_dtype.append(dtype)

        # size
        self.file.seek(0, os.SEEK_END)
        self.end = self.file.tell()
        self.ev_count = (self.end - self.start) // self.ev_size
        self.done = False
        self.file.seek(self.start)
        # If the current time is t, it means that next event that will be loaded has a
        # timestamp superior or equal to t (event with timestamp exactly t is not loaded yet)
        self.current_time = 0
        self.duration_s = self.total_time() * 1e-6

    def reset(self):
        """reset at beginning of file"""
        self.file.seek(self.start)
        self.done = False
        self.current_time = 0

    def event_count(self):
        """
        getter on event_count
        :return:
        """
        return self.ev_count

    def get_size(self):
        """ "(height, width) of the imager might be (None, None)"""
        return self.size

    def __repr__(self):
        """
        prints properties
        :return:
        """
        wrd = ""
        wrd += "PSEELoader:" + "\n"
        wrd += "-----------" + "\n"
        if self.extension == "dat":
            wrd += "Event Type: " + str(self.binary_format.EV_STRING) + "\n"
        elif self.extension == "npy":
            wrd += "Event Type: numpy array element\n"
        wrd += "Event Size: " + str(self.ev_size) + " bytes\n"
        wrd += "Event Count: " + str(self.ev_count) + "\n"
        wrd += "Duration: " + str(self.duration_s) + " s \n"
        wrd += "-----------" + "\n"
        return wrd

    def load_n_past_events(self, ev_count):
        """
        load batch of n past events
        :param ev_count: number of events that will be loaded
        :return: events

        This method does not change the current time,
        as it is intended to be used for loading past events from a known end time.
        """
        event_buffer = {
            "x": np.empty((ev_count + 1,), dtype=np.uint16),
            "y": np.empty((ev_count + 1,), dtype=np.uint16),
            "t": np.empty((ev_count + 1,), dtype=np.uint64),
            "p": np.empty((ev_count + 1,), dtype=np.uint8),
        }

        pos = self.file.tell()
        past_count = (pos - self.start) // self.ev_size
        if ev_count >= past_count:
            self.file.seek(self.start)
            ev_count = past_count
            self.binary_format.stream_td_data(self.file, event_buffer, self.dtype, ev_count)
        else:
            self.file.seek(pos - ev_count * self.ev_size)
            self.binary_format.stream_td_data(self.file, event_buffer, self.dtype, ev_count)

        return {key: value[:ev_count] for key, value in event_buffer.items()}

    def load_n_events(self, ev_count):
        """
        load batch of n events
        :param ev_count: number of events that will be loaded
        :return: events
        Note that current time will be incremented to reach the timestamp of the first event not loaded yet
        """
        event_buffer = {
            "x": np.empty((ev_count + 1,), dtype=np.uint16),
            "y": np.empty((ev_count + 1,), dtype=np.uint16),
            "t": np.empty((ev_count + 1,), dtype=np.uint64),
            "p": np.empty((ev_count + 1,), dtype=np.uint8),
        }

        pos = self.file.tell()
        count = (self.end - pos) // self.ev_size
        if ev_count >= count:
            self.done = True
            ev_count = count
            self.binary_format.stream_td_data(self.file, event_buffer, self.dtype, ev_count)
            self.current_time = event_buffer["t"][ev_count - 1] + 1
        else:
            self.binary_format.stream_td_data(self.file, event_buffer, self.dtype, ev_count + 1)
            self.current_time = event_buffer["t"][ev_count]
            self.file.seek(pos + ev_count * self.ev_size)

        return {key: value[:ev_count] for key, value in event_buffer.items()}

    def load_delta_t(self, delta_t):
        """
        loads a slice of time.
        :param delta_t: (us) slice thickness
        :return: events
        Note that current time will be incremented by delta_t.
        If an event is timestamped at exactly current_time it will not be loaded.
        """
        if delta_t < 1:
            raise ValueError("load_delta_t(): delta_t must be at least 1 micro-second: {}".format(delta_t))

        if self.done or (self.file.tell() >= self.end):
            self.done = True
            return np.empty((0,), dtype=self.decode_dtype)

        final_time = self.current_time + delta_t
        tmp_time = self.current_time
        start = self.file.tell()
        pos = start
        nevs = 0
        batch = 100000
        event_buffer = []
        # data is read by buffers until enough events are read or until the end of the file
        while tmp_time < final_time and pos < self.end:
            count = (min(self.end, pos + batch * self.ev_size) - pos) // self.ev_size
            buffer = {
                "x": np.empty((count,), dtype=np.uint16),
                "y": np.empty((count,), dtype=np.uint16),
                "t": np.empty((count,), dtype=np.uint64),
                "p": np.empty((count,), dtype=np.uint8),
            }
            self.binary_format.stream_td_data(self.file, buffer, self.dtype, count)
            tmp_time = buffer["t"][-1]
            event_buffer.append(buffer)
            nevs += count
            pos = self.file.tell()
        if tmp_time >= final_time:
            self.current_time = final_time
        else:
            self.current_time = tmp_time + 1
        assert len(event_buffer) > 0
        idx = np.searchsorted(event_buffer[-1]["t"], final_time)
        event_buffer[-1] = {k: v[:idx] for k, v in event_buffer[-1].items()}
        event_buffer = {key: np.concatenate([ev[key] for ev in event_buffer]) for key in ("x", "y", "t", "p")}
        idx = len(event_buffer["t"])
        self.file.seek(start + idx * self.ev_size)
        self.done = self.file.tell() >= self.end
        return event_buffer

    def seek_event(self, ev_count):
        """
        seek in the file by ev_count events
        :param ev_count: seek in the file after ev_count events
        Note that current time will be set to the timestamp of the next event.
        """
        if ev_count <= 0:
            self.file.seek(self.start)
            self.current_time = 0
        elif ev_count >= self.ev_count:
            # we put the cursor one event before and read the last event
            # which puts the file cursor at the right place
            # current_time is set to the last event timestamp + 1
            self.file.seek(self.start + (self.ev_count - 1) * self.ev_size)
            self.current_time = np.fromfile(self.file, dtype=self.dtype, count=1)["t"][0] + 1
        else:
            # we put the cursor at the *ev_count*nth event
            self.file.seek(self.start + (ev_count) * self.ev_size)
            # we read the timestamp of the following event (this change the position in the file)
            self.current_time = np.fromfile(self.file, dtype=self.dtype, count=1)["t"][0]
            # this is why we go back at the right position here
            self.file.seek(self.start + (ev_count) * self.ev_size)
        self.done = self.file.tell() >= self.end

    def seek_time(self, final_time, term_criterion=100000):
        """
        go to the time final_time inside the file. This is implemented using a binary search algorithm
        :param final_time: expected time
        :param term_cirterion: (nb event) binary search termination criterion
        it will load those events in a buffer and do a numpy searchsorted so the result is always exact
        """
        if final_time > self.total_time():
            self.file.seek(self.end)
            self.done = True
            self.current_time = self.total_time() + 1
            return

        if final_time <= 0:
            self.reset()
            return

        low = 0
        high = self.ev_count

        # binary search
        while high - low > term_criterion:
            middle = (low + high) // 2

            self.seek_event(middle)
            mid = np.fromfile(self.file, dtype=self.dtype, count=1)["t"][0]

            if mid > final_time:
                high = middle
            elif mid < final_time:
                low = middle + 1
            else:
                self.current_time = final_time
                self.done = self.file.tell() >= self.end
                return
        # we now know that it is between low and high
        self.seek_event(low)
        final_buffer = np.fromfile(self.file, dtype=self.dtype, count=high - low)["t"]
        final_index = np.searchsorted(final_buffer, final_time)

        self.seek_event(low + final_index)
        self.current_time = final_time
        self.done = self.file.tell() >= self.end

    def total_time(self):
        """
        get total duration of video in mus, providing there is no overflow
        :return:
        """
        if not self.ev_count:
            return 0
        # save the state of the class
        pos = self.file.tell()
        current_time = self.current_time
        done = self.done
        # read the last event's timestamp
        self.seek_event(self.ev_count - 1)
        time = np.fromfile(self.file, dtype=self.dtype, count=1)["t"][0]
        # restore the state
        self.file.seek(pos)
        self.current_time = current_time
        self.done = done

        return time

    def __del__(self):
        self.file.close()
