from copy import deepcopy
import unittest

import numpy as np

from adp.stress import StressEligibilityError, apply_stress


def scene_fixture():
    history = np.zeros((3, 50, 8), dtype=np.float32)
    history[:, :, 0] = np.arange(50) * 0.2
    history[1, :, 1] = 3
    history[:, :, 2] = 2
    history[:, :, 5:7] = 1
    history[:, :, 7] = np.linspace(0, 1, 50)
    future = np.repeat(history[:, -1:, :2], 60, axis=1)
    future[:, :, 0] += np.arange(1, 61) * 0.2
    return {"id": "fixture", "history": history, "history_mask": np.ones((3, 50), dtype=bool),
            "future": future, "future_mask": np.ones((3, 60), dtype=bool),
            "actor_ids": ["focal", "neighbor", "static"], "actor_types": np.array([0, 0, 5]),
            "map_lanes": [{"id": "current", "centerline": [[-10, 3], [100, 3]],
                           "left_neighbor_id": "adjacent", "right_neighbor_id": None, "is_drivable": True},
                          {"id": "adjacent", "centerline": [[-10, 6.5], [100, 6.5]],
                           "left_neighbor_id": None, "right_neighbor_id": "current", "is_drivable": True}]}


class StressTests(unittest.TestCase):
    def test_cut_in_stitches_only_unambiguous_contiguous_segments(self):
        scene = scene_fixture()
        scene['map_lanes'][0]['centerline'] = [[-10,3],[10,3]]
        scene['map_lanes'][1]['centerline'] = [[-10,6.5],[10,6.5]]
        scene['map_lanes'] += [
            {'id': 'continue-current', 'centerline': [[10,3],[100,3]], 'is_drivable': True},
            {'id': 'continue-adjacent', 'centerline': [[10,6.5],[100,6.5]], 'is_drivable': True},
        ]
        result = apply_stress(scene, {'kind': 'cut_in', 'severity': .5, 'actor_id': 'neighbor'})
        self.assertGreater(result['future'][1,-1,0],10)
        self.assertAlmostEqual(float(result['future'][1,-1,1]),6.5,places=4)
        scene['map_lanes'].append({'id': 'ambiguous', 'centerline': [[10,3],[100,4]], 'is_drivable': True})
        with self.assertRaises(StressEligibilityError):
            apply_stress(scene, {'kind': 'cut_in', 'severity': .5, 'actor_id': 'neighbor'})

    def test_observation_noise_is_seeded_independent_and_preserves_futures(self):
        scene = scene_fixture()
        before = deepcopy(scene)
        spec = {"kind": "position_noise", "severity": 0.35, "actor_id": "neighbor", "seed": 9}
        result = apply_stress(scene, spec)
        repeated = apply_stress(scene, spec)
        np.testing.assert_array_equal(result["history"], repeated["history"])
        np.testing.assert_array_equal(scene["history"], before["history"])
        np.testing.assert_array_equal(result["future"], before["future"])
        np.testing.assert_array_equal(result["history"][[0, 2]], before["history"][[0, 2]])
        self.assertFalse(np.array_equal(result["history"][1], before["history"][1]))

    def test_dropout_preserves_last_observation_and_clears_invalid_features(self):
        scene = scene_fixture()
        result = apply_stress(scene, {"kind": "dropout", "severity": 1, "seed": 1})
        self.assertGreater(result["stress"]["dropped_states"], 0)
        self.assertTrue(result["history_mask"][:, -1].all())
        np.testing.assert_array_equal(result["future"], scene["future"])
        self.assertTrue((result["history"][~result["history_mask"], :7] == 0).all())
        np.testing.assert_array_equal(result["history"][2], scene["history"][2])

    def test_braking_stops_monotonically_and_only_changes_selected_actor(self):
        scene = scene_fixture()
        result = apply_stress(scene, {"kind": "braking", "severity": 1, "actor_id": "neighbor"})
        np.testing.assert_array_equal(result["future"][[0, 2]], scene["future"][[0, 2]])
        np.testing.assert_array_equal(result["history"][:, :-11], scene["history"][:, :-11])
        x = np.r_[result["history"][1, -11:, 0], result["future"][1, :, 0]]
        self.assertTrue((np.diff(x) >= -1e-6).all())
        self.assertAlmostEqual(x[-1], scene["history"][1, -11, 0] + 2 ** 2 / 12, places=5)
        self.assertEqual(result["stress"]["label"], "counterfactual_nonreactive_replay")
        self.assertFalse(result["stress"]["other_agents_reactive"])

    def test_cut_in_requires_declared_supported_neighbor(self):
        scene = scene_fixture()
        result = apply_stress(scene, {"kind": "cut_in", "severity": 0.5, "actor_id": "neighbor"})
        np.testing.assert_array_equal(result["history"][1, -11, :2], scene["history"][1, -11, :2])
        self.assertAlmostEqual(result["future"][1, -1, 1], 6.5, places=5)
        self.assertEqual(result["stress"]["target_lane_id"], "adjacent")
        unsupported = deepcopy(scene)
        unsupported["map_lanes"][0]["left_neighbor_id"] = None
        with self.assertRaisesRegex(StressEligibilityError, "neighbor"):
            apply_stress(unsupported, {"kind": "cut_in", "severity": 0.5, "actor_id": "neighbor"})

    def test_stress_uses_history_only_and_zero_severity_is_noop(self):
        scene = scene_fixture()
        altered_future = deepcopy(scene)
        altered_future["future"][:] = 999
        spec = {"kind": "cut_in", "severity": 0.35, "actor_id": "neighbor"}
        a, b = apply_stress(scene, spec), apply_stress(altered_future, spec)
        np.testing.assert_array_equal(a["history"], b["history"])
        np.testing.assert_array_equal(a["future"][1], b["future"][1])
        for kind in ("position_noise", "dropout", "braking", "cut_in"):
            result = apply_stress(scene, {"kind": kind, "severity": 0})
            np.testing.assert_array_equal(result["history"], scene["history"])
            np.testing.assert_array_equal(result["future"], scene["future"])

    def test_rejects_invalid_severity_and_unobserved_intervention(self):
        for severity in (-0.1, 1.1, float("nan")):
            with self.assertRaisesRegex(ValueError, "severity"):
                apply_stress(scene_fixture(), {"kind": "braking", "severity": severity})
        scene = scene_fixture()
        scene["history_mask"][1, -2] = False
        with self.assertRaises(StressEligibilityError):
            apply_stress(scene, {"kind": "braking", "severity": 0.5, "actor_id": "neighbor"})


if __name__ == "__main__":
    unittest.main()
