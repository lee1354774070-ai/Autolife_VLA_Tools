#!/usr/bin/env python3
"""Read Autolife camera frames directly from the shared-memory files.

The robot camera service and ``hand_camera_producer.py`` expose one latest
frame through two files in ``/dev/shm``.  This module contains the small,
ROS-free reader shared by the recorder and the optional ROS bridge.  Keeping
the reader here makes direct-SHM recording and topic recording use exactly the
same validation rules.
"""

from __future__ import annotations

import time
import struct
from dataclasses import dataclass

from camera_config import SHM_METADATA_FORMAT, CameraSpec

METADATA_SIZE = struct.calcsize(SHM_METADATA_FORMAT)
EPOCH_NS_MIN = 946684800 * 1_000_000_000
EPOCH_NS_MAX = 4102444800 * 1_000_000_000


@dataclass(frozen=True)
class ShmFrame:
    """One complete frame copied out of a camera's shared-memory files."""

    timestamp_ns: int
    width: int
    height: int
    channels: int
    pixel_format: int
    byte_count: int
    data: bytes


def read_shm_metadata(spec: CameraSpec) -> tuple[int, int, int, int, int, int] | None:
    try:
        with open(spec.meta_path, "rb") as stream:
            raw = stream.read(METADATA_SIZE)
        if len(raw) != METADATA_SIZE:
            return None
        return struct.unpack(SHM_METADATA_FORMAT, raw)
    except (FileNotFoundError, OSError, struct.error):
        return None


def read_shm_frame(
    spec: CameraSpec,
    metadata: tuple[int, int, int, int, int, int] | None = None,
) -> ShmFrame | None:
    """Read one internally consistent metadata/image pair.

    Metadata and image bytes are separate files, so a producer can update the
    pair between our reads.  Reading metadata again after the image catches
    that race and prevents a frame from being built from mixed generations.
    """

    first = metadata if metadata is not None else read_shm_metadata(spec)
    if first is None:
        return None
    timestamp_ns, width, height, channels, pixel_format, buffer_size = first
    if width <= 0 or height <= 0 or channels <= 0 or buffer_size <= 0:
        return None

    bytes_per_pixel = 2 if pixel_format == 2 else 1
    expected_size = width * height * channels * bytes_per_pixel
    if expected_size <= 0 or buffer_size < expected_size:
        return None
    try:
        with open(spec.buffer_path, "rb") as stream:
            data = stream.read(expected_size)
    except (FileNotFoundError, OSError):
        return None
    if len(data) != expected_size:
        return None

    second = read_shm_metadata(spec)
    if second != first:
        return None
    return ShmFrame(timestamp_ns, width, height, channels, pixel_format, expected_size, data)


def shm_timestamp_sec(timestamp_ns: int, received_sec: float | None = None) -> float:
    """Return SHM epoch seconds, falling back for monotonic/device clocks."""

    if EPOCH_NS_MIN <= timestamp_ns <= EPOCH_NS_MAX:
        return timestamp_ns * 1e-9
    return time.time() if received_sec is None else received_sec
