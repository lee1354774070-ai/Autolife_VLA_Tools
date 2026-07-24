#!/usr/bin/env python3
"""Display three AutoLife RGB cameras forwarded from a robot over SSH."""

from __future__ import annotations

import argparse
import os
import queue
import shlex
import signal
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np


CAMERA_NAMES = ("rgbd_head_color", "hand_left", "hand_right")
CAMERA_LABELS = {
    "rgbd_head_color": "Head RGB",
    "hand_left": "Left hand",
    "hand_right": "Right hand",
}
KNOWN_ROBOT_HOSTS = {
    "283": "192.168.8.42",
    "300": "192.168.8.202",
    "306": "192.168.8.11",
}
FRAME_MAGIC = b"DTCM"
FRAME_HEADER = struct.Struct("<4sBQI")
MAX_JPEG_BYTES = 8 * 1024 * 1024


REMOTE_CAMERA_READER = r"""
import fcntl
import os
import signal
import struct
import sys
import time
import traceback

import cv2

names = sys.argv[1].split(",")
fps = float(sys.argv[2])
jpeg_quality = int(sys.argv[3])
header = struct.Struct("<4sBQI")
period = 1.0 / max(fps, 1.0)
modules = [
    "mod_camera_rgbd_head",
    "mod_camera_hand_left",
    "mod_camera_hand_right",
]
output_names = ["color", "decoded", "decoded"]

# Keep one viewer per robot to avoid duplicate SSH/JPEG bandwidth.
lock_stream = open("/tmp/autolife_digital_twin_camera.lock", "w")
try:
    fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    print("Another digital-twin camera viewer is already running.", file=sys.stderr)
    raise SystemExit(3)

# Preserve the SSH binary stream while silencing verbose SDK/GStreamer startup
# output. Initialization failures are restored to stderr with a traceback.
protocol_fd = os.dup(sys.stdout.fileno())
stderr_fd = os.dup(sys.stderr.fileno())
devnull_fd = os.open(os.devnull, os.O_WRONLY)
os.dup2(devnull_fd, sys.stdout.fileno())
os.dup2(devnull_fd, sys.stderr.fileno())
try:
    from autolife_robot_sdk.utils import (
        get_camera_shm_output,
        open_camera_shm_consumer,
    )

    consumers = [
        open_camera_shm_consumer(
            get_camera_shm_output(
                module,
                output_name,
                robot_model="autolife_s1",
                robot_version="robot_v2_2",
            ),
            name=name,
        )
        for name, module, output_name in zip(names, modules, output_names)
    ]
except Exception:
    os.dup2(stderr_fd, sys.stderr.fileno())
    traceback.print_exc()
    raise
finally:
    os.dup2(stderr_fd, sys.stderr.fileno())
    os.close(devnull_fd)
    os.close(stderr_fd)

running = True
def stop(*_args):
    global running
    running = False

signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGHUP, stop)

try:
    last_frame_ids = [None] * len(consumers)
    while running:
        started = time.monotonic()
        for index, consumer in enumerate(consumers):
            result = consumer.get_latest(nonblock=True, with_meta=True)
            if result is None:
                continue
            image, frame_id, metadata = result
            if frame_id == last_frame_ids[index]:
                continue
            last_frame_ids[index] = frame_id
            ok, encoded = cv2.imencode(
                ".jpg",
                image,
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
            )
            if not ok:
                continue
            payload = encoded.tobytes()
            timestamp_ns = int(metadata.get("publish_epoch_ns") or time.time_ns())
            os.write(
                protocol_fd,
                header.pack(b"DTCM", index, timestamp_ns, len(payload)),
            )
            os.write(protocol_fd, payload)
        delay = period - (time.monotonic() - started)
        if delay > 0:
            time.sleep(delay)
finally:
    for consumer in consumers:
        consumer.close()
    os.close(protocol_fd)
"""


@dataclass(frozen=True)
class CameraFrame:
    image: np.ndarray
    timestamp_ns: int
    received_monotonic: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-id", required=True)
    parser.add_argument("--robot-host", default=None)
    parser.add_argument("--robot-user", default="ubuntu")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--tile-width", type=int, default=400)
    parser.add_argument("--parent-pid", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def read_exact(stream, size: int) -> bytes | None:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def placeholder(width: int, text: str) -> np.ndarray:
    height = round(width * 3 / 4)
    image = np.full((height, width, 3), (35, 38, 42), dtype=np.uint8)
    cv2.putText(
        image,
        text,
        (24, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )
    return image


def render_tile(name: str, frame: CameraFrame | None, width: int) -> np.ndarray:
    height = round(width * 3 / 4)
    if frame is None:
        tile = placeholder(width, "Waiting for camera ...")
        age_text = "no frame"
    else:
        tile = cv2.resize(frame.image, (width, height), interpolation=cv2.INTER_AREA)
        age = time.monotonic() - frame.received_monotonic
        age_text = f"age {age * 1000:.0f} ms"
        if age > 1.0:
            cv2.rectangle(tile, (0, 0), (width - 1, height - 1), (0, 0, 255), 4)
    cv2.rectangle(tile, (0, 0), (width, 42), (0, 0, 0), -1)
    cv2.putText(
        tile,
        f"{CAMERA_LABELS[name]} | {age_text}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return tile


def compose_vertical_canvas(
    frames: dict[str, CameraFrame],
    tile_width: int,
) -> np.ndarray:
    """Stack head, left-hand, and right-hand camera tiles from top to bottom."""

    return np.vstack(
        [render_tile(name, frames.get(name), tile_width) for name in CAMERA_NAMES]
    )


class SSHCameraStream:
    def __init__(
        self,
        host: str,
        user: str,
        fps: float,
        jpeg_quality: int,
    ) -> None:
        remote_python = "/home/ubuntu/miniconda3/envs/robot_env/bin/python"
        remote_command = " ".join(
            (
                remote_python,
                "-u -c",
                shlex.quote(REMOTE_CAMERA_READER),
                shlex.quote(",".join(CAMERA_NAMES)),
                shlex.quote(str(fps)),
                shlex.quote(str(jpeg_quality)),
            )
        )
        self.process = subprocess.Popen(
            ["ssh", "-o", "BatchMode=yes", f"{user}@{host}", remote_command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self.frames: queue.Queue[tuple[int, int, bytes]] = queue.Queue(maxsize=12)
        self.stderr_lines: queue.Queue[str] = queue.Queue()
        self.reader = threading.Thread(target=self._read_frames, daemon=True)
        self.stderr_reader = threading.Thread(target=self._read_stderr, daemon=True)
        self.reader.start()
        self.stderr_reader.start()

    def _read_frames(self) -> None:
        assert self.process.stdout is not None
        while True:
            raw_header = read_exact(self.process.stdout, FRAME_HEADER.size)
            if raw_header is None:
                return
            magic, camera_index, timestamp_ns, payload_size = FRAME_HEADER.unpack(raw_header)
            if (
                magic != FRAME_MAGIC
                or camera_index >= len(CAMERA_NAMES)
                or payload_size <= 0
                or payload_size > MAX_JPEG_BYTES
            ):
                self.stderr_lines.put("Invalid camera stream frame header.")
                return
            payload = read_exact(self.process.stdout, payload_size)
            if payload is None:
                return
            item = (camera_index, timestamp_ns, payload)
            try:
                self.frames.put_nowait(item)
            except queue.Full:
                try:
                    self.frames.get_nowait()
                except queue.Empty:
                    pass
                self.frames.put_nowait(item)

    def _read_stderr(self) -> None:
        assert self.process.stderr is not None
        for raw in self.process.stderr:
            text = raw.decode("utf-8", errors="replace").rstrip()
            if text:
                self.stderr_lines.put(text)

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()


def main() -> int:
    args = parse_args()
    if args.fps <= 0 or args.tile_width < 160:
        raise SystemExit("--fps must be positive and --tile-width must be at least 160.")
    if not 1 <= args.jpeg_quality <= 100:
        raise SystemExit("--jpeg-quality must be between 1 and 100.")
    host = args.robot_host or KNOWN_ROBOT_HOSTS.get(str(args.robot_id))
    if not host:
        raise SystemExit("Unknown robot ID; provide --robot-host.")

    stream = SSHCameraStream(host, args.robot_user, args.fps, args.jpeg_quality)
    latest: dict[str, CameraFrame] = {}
    window_name = f"AutoLife {args.robot_id} | 3 RGB Cameras"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    print(
        f"[camera_viewer] {args.robot_user}@{host}: "
        + ", ".join(CAMERA_NAMES),
        flush=True,
    )
    print("[camera_viewer] press q or Esc to close.", flush=True)

    try:
        while True:
            if args.parent_pid is not None:
                try:
                    os.kill(args.parent_pid, 0)
                except ProcessLookupError:
                    break
            while True:
                try:
                    camera_index, timestamp_ns, payload = stream.frames.get_nowait()
                except queue.Empty:
                    break
                image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is not None:
                    latest[CAMERA_NAMES[camera_index]] = CameraFrame(
                        image=image,
                        timestamp_ns=timestamp_ns,
                        received_monotonic=time.monotonic(),
                    )
            while True:
                try:
                    print(f"[camera_viewer][ssh] {stream.stderr_lines.get_nowait()}", file=sys.stderr)
                except queue.Empty:
                    break
            if stream.process.poll() is not None:
                print(
                    f"[camera_viewer] SSH camera stream exited with code {stream.process.returncode}.",
                    file=sys.stderr,
                )
                return 1
            canvas = compose_vertical_canvas(latest, args.tile_width)
            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key in (27, ord("q")):
                break
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
    finally:
        stream.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda *_: sys.exit(130))
    raise SystemExit(main())
