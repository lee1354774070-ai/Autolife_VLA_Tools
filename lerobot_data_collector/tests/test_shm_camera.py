import struct
import tempfile
import unittest
from pathlib import Path

from camera_config import SHM_METADATA_FORMAT, CameraSpec
from shm_camera import read_shm_frame, read_shm_metadata, shm_timestamp_sec


class ShmCameraTest(unittest.TestCase):
    def test_reader_requires_a_stable_metadata_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = CameraSpec(
                name="test",
                meta_path=str(root / "meta"),
                buffer_path=str(root / "buffer"),
                topic="/test",
                frame_id="test",
            )
            metadata = struct.pack(SHM_METADATA_FORMAT, 1_700_000_000_000_000_000, 2, 1, 3, 1, 6)
            Path(spec.buffer_path).write_bytes(b"\x01\x02\x03\x04\x05\x06")
            Path(spec.meta_path).write_bytes(metadata)

            parsed = read_shm_metadata(spec)
            frame = read_shm_frame(spec, parsed)

            self.assertEqual(parsed, (1_700_000_000_000_000_000, 2, 1, 3, 1, 6))
            self.assertIsNotNone(frame)
            self.assertEqual(frame.data, b"\x01\x02\x03\x04\x05\x06")

    def test_non_epoch_timestamp_falls_back_to_receive_time(self):
        self.assertEqual(shm_timestamp_sec(123, 42.5), 42.5)


if __name__ == "__main__":
    unittest.main()
