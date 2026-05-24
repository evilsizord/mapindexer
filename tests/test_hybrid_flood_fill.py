import unittest
from types import SimpleNamespace

import numpy as np

from mapindexer.hybrid_flood_fill import build_leaf_adjacency, leaf_bounds_aabb


def leaf(mins, maxs, cluster=0):
    return SimpleNamespace(
        cluster=cluster,
        bounds=SimpleNamespace(mins=mins, maxs=maxs),
    )


class HybridFloodFillTests(unittest.TestCase):
    def test_build_leaf_adjacency_connects_face_touching_leafs(self):
        bsp = SimpleNamespace(
            LEAVES=[
                leaf([0, 0, 0], [10, 10, 10]),
                leaf([10, 0, 0], [20, 10, 10]),
                leaf([20, 20, 0], [30, 30, 10]),
            ]
        )

        adjacency = build_leaf_adjacency(bsp, {0, 1, 2}, cell_size=64.0)

        self.assertEqual(adjacency[0], {1})
        self.assertEqual(adjacency[1], {0})
        self.assertNotIn(2, adjacency)

    def test_leaf_bounds_aabb_wraps_reachable_leaf_bounds(self):
        bsp = SimpleNamespace(
            LEAVES=[
                leaf([0, 0, 0], [10, 10, 10]),
                leaf([-5, 2, 1], [3, 20, 9]),
            ]
        )

        mins, maxs = leaf_bounds_aabb(bsp, {0, 1})

        np.testing.assert_array_equal(mins, np.array([-5, 0, 0], dtype=np.float32))
        np.testing.assert_array_equal(maxs, np.array([10, 20, 10], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
