import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from mapindexer import flood_fill_volume


class FloodFillVolumeTests(unittest.TestCase):
    def test_verbose_false_suppresses_flood_fill_details(self):
        bsp = SimpleNamespace(
            MODELS=[
                SimpleNamespace(
                    bounds=SimpleNamespace(
                        mins=[0.0, 0.0, 0.0],
                        maxs=[32.0, 32.0, 32.0],
                    )
                )
            ]
        )

        with (
            patch.object(flood_fill_volume, "make_blocked_fn", return_value=lambda _: False),
            patch.object(flood_fill_volume, "get_spawn_origins", return_value=[np.array([16.0, 16.0, 16.0], dtype=np.float32)]),
            patch.object(flood_fill_volume, "has_fly_clearance", return_value=True),
            patch.object(flood_fill_volume, "can_fly_between", return_value=True),
        ):
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                visited, aabb = flood_fill_volume.flood_fill_flyable_volume_from_spawns(bsp, verbose=False)

        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(visited, {(0, 0, 0)})
        np.testing.assert_array_equal(aabb[0], np.array([0.0, 0.0, 0.0], dtype=np.float32))
        np.testing.assert_array_equal(aabb[1], np.array([32.0, 32.0, 32.0], dtype=np.float32))

    def test_verbose_true_prints_flood_fill_details(self):
        bsp = SimpleNamespace(
            MODELS=[
                SimpleNamespace(
                    bounds=SimpleNamespace(
                        mins=[0.0, 0.0, 0.0],
                        maxs=[32.0, 32.0, 32.0],
                    )
                )
            ]
        )

        with (
            patch.object(flood_fill_volume, "make_blocked_fn", return_value=lambda _: False),
            patch.object(flood_fill_volume, "get_spawn_origins", return_value=[np.array([16.0, 16.0, 16.0], dtype=np.float32)]),
            patch.object(flood_fill_volume, "has_fly_clearance", return_value=True),
            patch.object(flood_fill_volume, "can_fly_between", return_value=True),
        ):
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                flood_fill_volume.flood_fill_flyable_volume_from_spawns(bsp)

        output = stdout.getvalue()
        self.assertIn("World bounds:", output)
        self.assertIn("Voxel grid dims:", output)
        self.assertIn("Seed spawns:", output)
        self.assertIn("Initial reachable cells:", output)
        self.assertIn("flood_fill_flyable_volume_from_spawns completed", output)


if __name__ == "__main__":
    unittest.main()
