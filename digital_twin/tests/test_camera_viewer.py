from __future__ import annotations

import io
import unittest

from digital_twin.camera_viewer import (
    CAMERA_NAMES,
    compose_vertical_canvas,
    placeholder,
    read_exact,
)


class CameraViewerTest(unittest.TestCase):
    def test_three_default_rgb_cameras(self) -> None:
        self.assertEqual(
            CAMERA_NAMES,
            ("rgbd_head_color", "hand_left", "hand_right"),
        )

    def test_read_exact_handles_partial_stream_reads(self) -> None:
        class PartialReader(io.BytesIO):
            def read(self, size: int = -1) -> bytes:
                return super().read(min(size, 2))

        self.assertEqual(read_exact(PartialReader(b"abcdef"), 6), b"abcdef")
        self.assertIsNone(read_exact(PartialReader(b"abc"), 6))

    def test_placeholder_matches_camera_aspect_ratio(self) -> None:
        image = placeholder(480, "waiting")
        self.assertEqual(image.shape, (360, 480, 3))

    def test_camera_tiles_are_stacked_vertically(self) -> None:
        canvas = compose_vertical_canvas({}, 400)
        self.assertEqual(canvas.shape, (900, 400, 3))


if __name__ == "__main__":
    unittest.main()
