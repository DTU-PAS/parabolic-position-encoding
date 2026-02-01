import re
from pathlib import Path

from pape.aedat.v2_0 import parse_aedat_v2_0
from pape.aedat.v3_1 import parse_aedat_v3_1
from pape.data_types import Events

VERSION_PATTERN = re.compile(r"#!AER-DAT(\d.\d)")


class AedatReader:
    def __init__(self, path: Path):
        self.path = path

    def __enter__(self):
        self.recording = self.path.read_bytes()
        return self

    def __exit__(self, type, value, traceback):
        pass

    def read(self) -> Events:
        version = self.detect_version()
        match version:
            case "2.0":
                return parse_aedat_v2_0(self.recording)
            case "3.1":
                return parse_aedat_v3_1(self.recording)
            case _:
                raise ValueError(f"Unsupported AEDAT version: {version}")

    def detect_version(self) -> str:
        index = 0
        new_line = ord(b"\n")
        while index < len(self.recording) and self.recording[index] != new_line:
            index += 1

        first_line = self.recording[:index].decode()
        version = VERSION_PATTERN.search(first_line)
        if version is None:
            raise ValueError("Could not detect AEDAT version.")

        return version.group(1)
