from __future__ import annotations

import unittest

import numpy as np

from deploy.lingbot_va.lingbot_remote_protocol import (
    RemoteInferenceError,
    decode_jpeg,
    encode_jpeg,
    validate_action_response,
    validate_policy_metadata,
)


class LingBotRemoteProtocolTest(unittest.TestCase):
    def test_jpeg_round_trip_has_policy_shape_and_range(self) -> None:
        image = np.zeros((32, 48, 3), dtype=np.uint8)
        image[:, :, 1] = 127
        decoded = decode_jpeg(encode_jpeg(image, size=24, quality=95))
        self.assertEqual(decoded.shape, (3, 24, 24))
        self.assertEqual(decoded.dtype, np.float32)
        self.assertGreaterEqual(float(decoded.min()), 0.0)
        self.assertLessEqual(float(decoded.max()), 1.0)

    def test_action_response_requires_finite_16d_chunk(self) -> None:
        actions = validate_action_response({"actions": np.zeros((12, 16)).tolist()})
        self.assertEqual(actions.shape, (12, 16))
        with self.assertRaises(RemoteInferenceError):
            validate_action_response({"actions": np.zeros((12, 14)).tolist()})
        invalid = np.zeros((1, 16), dtype=np.float32)
        invalid[0, 0] = np.nan
        with self.assertRaises(RemoteInferenceError):
            validate_action_response({"actions": invalid.tolist()})

    def test_policy_metadata_supplies_robot_contract_without_local_checkpoint(self) -> None:
        response = {
            "camera_keys": [
                "observation.images.cam_high",
                "observation.images.cam_left_wrist",
                "observation.images.cam_right_wrist",
            ],
            "action_channels": list(range(14, 30)),
            "action_min": np.arange(16, dtype=np.float32).tolist(),
            "action_max": (np.arange(16, dtype=np.float32) + 1).tolist(),
        }
        camera_keys, channels, lower, upper = validate_policy_metadata(response)
        self.assertEqual(len(camera_keys), 3)
        self.assertEqual(channels, list(range(14, 30)))
        self.assertEqual(lower.shape, (16,))
        self.assertEqual(upper.shape, (16,))

        response["action_max"] = [0.0] * 15
        with self.assertRaises(RemoteInferenceError):
            validate_policy_metadata(response)


if __name__ == "__main__":
    unittest.main()
