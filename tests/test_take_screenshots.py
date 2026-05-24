import tempfile
import unittest
from pathlib import Path

from take_screenshots import write_camera_cfg


class TakeScreenshotsTests(unittest.TestCase):
    def test_write_camera_cfg_uses_camera_pickle_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "q3ut4" / "_testmap.cam.cfg"
            cameras = [
                {
                    "type": "anchor_cluster",
                    "position": [128.2, 256.9, 512.0],
                    "yaw": 90.4,
                    "score": 12.5,
                }
            ]

            write_camera_cfg(cfg_path, "testmap", cameras, wait=50, base_cfg="base_cam.cfg")

            text = cfg_path.read_text()
            self.assertIn("exec base_cam.cfg", text)
            self.assertIn("setviewpos 128 256 512 90", text)
            self.assertIn('screenshotJPEG "testmap_0_anchor_cluster"', text)
            self.assertIn("quit", text)


if __name__ == "__main__":
    unittest.main()
