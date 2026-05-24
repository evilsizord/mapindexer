import unittest

import numpy as np

from mapindexer.camera_candidates import (
    Anchor,
    CameraCandidate,
    cluster_anchors,
    entity_weight,
    select_diverse_candidates,
)


class CameraCandidatesTests(unittest.TestCase):
    def test_entity_weight_prioritizes_gameplay_entities(self):
        self.assertGreater(entity_weight("team_CTF_redflag"), entity_weight("info_player_deathmatch"))
        self.assertGreater(entity_weight("weapon_lr300"), 0)
        self.assertEqual(entity_weight("misc_model"), 0)

    def test_cluster_anchors_groups_nearby_points(self):
        anchors = [
            Anchor("info_player_deathmatch", np.array([0, 0, 0], dtype=np.float32), 4.0),
            Anchor("weapon_lr300", np.array([100, 0, 0], dtype=np.float32), 2.5),
            Anchor("team_ctf_redflag", np.array([2000, 0, 0], dtype=np.float32), 10.0),
        ]

        clusters = cluster_anchors(anchors, radius=256.0)

        self.assertEqual(len(clusters), 2)
        self.assertEqual(len(clusters[0].anchors), 1)
        self.assertEqual(clusters[0].anchors[0].classname, "team_ctf_redflag")
        self.assertEqual(len(clusters[1].anchors), 2)

    def test_select_diverse_candidates_enforces_minimum_separation(self):
        candidates = [
            CameraCandidate("test", np.array([0, 0, 0], dtype=np.float32), np.zeros(3), 0, 0, 10, 1, 1),
            CameraCandidate("test", np.array([10, 0, 0], dtype=np.float32), np.zeros(3), 0, 0, 9, 1, 1),
            CameraCandidate("test", np.array([1000, 0, 0], dtype=np.float32), np.zeros(3), 0, 0, 8, 1, 1),
        ]

        selected = select_diverse_candidates(candidates, max_cameras=3, min_separation=128.0)

        self.assertEqual([c.score for c in selected], [10, 8])


if __name__ == "__main__":
    unittest.main()
