import math
import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from adp.metrics import (average_precision, evaluate_predictions,
                             oriented_boxes_overlap, reliability, risk_labels,
                             summarize_records)


@unittest.skipIf(torch is None, "PyTorch is required for forecast metrics")
class MetricsTests(unittest.TestCase):
    def test_risk_ignores_padding_but_incomplete_dynamic_disables_eligibility(self):
        future = torch.zeros(1, 3, 60, 2)
        future[:, 1, :, 0] = 2.5
        mask = torch.ones(1, 3, 60, dtype=torch.bool)
        mask[:, 2] = False
        dynamic = torch.tensor([[True, True, False]])
        labels, eligible = risk_labels(future, mask, dynamic)
        self.assertEqual(labels.tolist(), [[1, 1, 0]])
        self.assertEqual(eligible.tolist(), [[True, True, False]])
        mask[:, 1, -1] = False
        labels, eligible = risk_labels(future, mask, dynamic)
        self.assertEqual(labels.tolist(), [[1, 1, 0]])
        self.assertFalse(eligible.any())

    def test_no_self_risk_or_risk_at_exact_threshold(self):
        future = torch.zeros(1, 2, 60, 2)
        future[:, 1, :, 0] = 3
        mask = torch.ones(1, 2, 60, dtype=torch.bool)
        labels, eligible = risk_labels(future, mask, torch.ones(1, 2, dtype=torch.bool))
        self.assertFalse(labels.any())
        self.assertTrue(eligible.all())
        _, eligible = risk_labels(future, mask, torch.tensor([[True, False]]))
        self.assertFalse(eligible.any())

    def test_joint_metric_never_combines_incompatible_actor_modes(self):
        future = torch.zeros(1, 2, 60, 2)
        future[:, 1, :, 1] = 10
        means = future[:, None].repeat(1, 2, 1, 1, 1)
        means[:, 0, 1, :, 1] += 10
        means[:, 1, 0, :, 1] -= 10
        batch = {"future": future, "future_mask": torch.ones(1, 2, 60, dtype=torch.bool),
                 "dynamic_mask": torch.ones(1, 2, dtype=torch.bool), "scenario_ids": ["one"]}
        outputs = {"means": means, "scales": torch.ones_like(means),
                   "logits": torch.zeros(1, 2), "risk_logits": torch.zeros(1, 2)}
        records = evaluate_predictions(outputs, batch)
        self.assertEqual(records[0]["minFDE"], [0, 0])
        self.assertEqual(records[0]["jointFDE"], 5)
        self.assertEqual(records[0]["jointADE"], 5)
        self.assertEqual(records[0]["brierFDE"], [0.25, 0.25])
        outputs["means"] = future[:, None].repeat(1, 2, 1, 1, 1)
        record = evaluate_predictions(outputs, batch)[0]
        self.assertAlmostEqual(record["NLL"], 0.5 * math.log(2 * math.pi), places=5)

    def test_calibrator_is_applied_without_changing_targets(self):
        future = torch.zeros(1, 2, 60, 2)
        batch = {"future": future, "future_mask": torch.ones(1, 2, 60, dtype=torch.bool),
                 "dynamic_mask": torch.ones(1, 2, dtype=torch.bool)}
        means = future[:, None]
        outputs = {"means": means, "scales": torch.ones_like(means),
                   "logits": torch.zeros(1, 1), "risk_logits": torch.full((1, 2), 2.0)}
        record = evaluate_predictions(outputs, batch, {"risk_temperature": 2, "risk_bias": -1})[0]
        self.assertEqual(record["risk_labels"], [1, 1])
        self.assertEqual(record["risk_calibrated_probabilities"], [0.5, 0.5])
        self.assertGreater(record["risk_raw_probabilities"][0], 0.8)
        outputs['means'] = means.repeat(1,2,1,1,1)
        outputs['means'][:,1] += 1
        outputs['scales'] = torch.ones_like(outputs['means'])
        outputs['logits'] = torch.tensor([[2.,0.]])
        record = evaluate_predictions(outputs, batch, {'mode_temperature': 2})[0]
        self.assertGreater(record['mode_confidence_raw'],record['mode_confidence'])
        summary = summarize_records([record], bootstrap=0)
        self.assertIsNotNone(summary['modeECE_raw'])

    def test_focal_scores_and_exclusion_coverage_are_separate(self):
        future = torch.zeros(1, 2, 60, 2)
        future[:, 1, :, 0] = 10
        mask = torch.ones(1, 2, 60, dtype=torch.bool)
        mask[:, 1, -1] = False
        means = future[:, None].clone()
        means[:, :, 0, :, 1] = 3
        outputs = {"means": means, "scales": torch.ones_like(means),
                   "logits": torch.zeros(1, 1), "risk_logits": torch.zeros(1, 2)}
        batch = {"future": future, "future_mask": mask, "dynamic_mask": torch.ones(1, 2, dtype=torch.bool),
                 "selections": [{"retained_dynamic": 2, "total_dynamic": 4,
                                 "retained_scored": 2, "total_scored": 3}]}
        record = evaluate_predictions(outputs, batch)[0]
        summary = summarize_records([record], bootstrap=0)
        self.assertEqual(summary["focal_minADE"], 3)
        self.assertEqual(summary["focal_minFDE"], 3)
        self.assertEqual(summary["focal_miss_rate"], 1)
        self.assertEqual(summary["coverage"]["scored_excluded_incomplete_future"], 1)
        self.assertEqual(summary["coverage"]["risk_eligible_dynamic"], 0)
        self.assertEqual(summary["coverage"]["scored_evaluation_fraction"], 0.5)
        self.assertEqual(summary["coverage"]["dynamic_retention_fraction"], 0.5)
        self.assertEqual(summary["definitions"]["footprint_collision_proxy"]["dimensions_length_width_m"]["bus"], [12, 2.5])

    def test_oriented_rectangles_distinguish_width_from_length(self):
        self.assertFalse(oriented_boxes_overlap(np.array([0, 0]), 0, (4.5, 1.8),
                                               np.array([0, 2.5]), 0, (4.5, 1.8)))
        self.assertTrue(oriented_boxes_overlap(np.array([0, 0]), 0, (4.5, 1.8),
                                              np.array([0, 2.5]), math.pi / 2, (4.5, 1.8)))

    def test_reliability_endpoint_and_average_precision_ties(self):
        diagram = reliability([0, 0.5, 1], [0, 1, 1], bins=2)
        self.assertEqual([item["count"] for item in diagram["bins"]], [1, 2])
        self.assertAlmostEqual(diagram["ece"], 1 / 6)
        self.assertEqual(average_precision([0.5, 0.5], [1, 0]), 0.5)
        self.assertEqual(average_precision([0.9, 0.1], [1, 0]), 1)
        self.assertIsNone(average_precision([0.5], [0]))

    def test_bootstrap_clusters_by_scenario_and_flags_sparse_events(self):
        records = [{"scenario_id": "many", "minFDE": [0.0] * 100},
                   {"scenario_id": "one", "minFDE": [1.0], "risk_labels": [1],
                    "risk_raw_probabilities": [0.5], "risk_calibrated_probabilities": [0.7]}]
        summary = summarize_records(records, bootstrap=200, seed=42)
        self.assertAlmostEqual(summary["minFDE"], 1 / 101)
        self.assertEqual(summary["confidence_intervals"]["minFDE"], [0, 1])
        self.assertEqual(summary["risk"]["status"], "insufficient_positive_events")
        self.assertEqual(summary, summarize_records(records, bootstrap=200, seed=42))

    def test_risk_evidence_requires_fitted_calibration_and_both_positive_counts(self):
        for calibration_positives, evaluation_positives, fitted, expected in (
            (2, 50, True, "insufficient_positive_events"),
            (50, 49, True, "insufficient_positive_events"),
            (50, 50, False, "not_fitted"),
            (50, 50, True, "sufficient"),
        ):
            with self.subTest(calibration_positives=calibration_positives,
                              evaluation_positives=evaluation_positives, fitted=fitted):
                record = {"risk_labels": [1] * evaluation_positives,
                          "risk_raw_probabilities": [0.5] * evaluation_positives,
                          "risk_calibrated_probabilities": [0.6] * evaluation_positives}
                calibration = {"risk": {"positives": calibration_positives,
                                        "count": calibration_positives + 20, "fitted": fitted}}
                risk = summarize_records([record], bootstrap=0, calibration=calibration)["risk"]
                self.assertEqual(risk["status"], expected)
                self.assertEqual(risk["positive_count"], evaluation_positives)
                self.assertEqual(risk["calibration"]["positive_count"], calibration_positives)
                self.assertEqual(risk["calibration"]["eligible_count"], calibration_positives + 20)
                self.assertEqual(risk["calibration"]["fitted"], fitted)

    def test_raw_risk_scores_are_not_called_calibrated_with_many_events(self):
        record = {"risk_labels": [1] * 50, "risk_raw_probabilities": [0.5] * 50,
                  "risk_calibrated_probabilities": [0.5] * 50}
        risk = summarize_records([record], bootstrap=0)["risk"]
        self.assertEqual(risk["status"], "uncalibrated")
        self.assertEqual(risk["evaluation_status"], "sufficient")
        self.assertEqual(risk["calibration"]["status"], "not_applied")
        unknown = summarize_records([record], bootstrap=0, calibration={"risk": {"fitted": True}})["risk"]
        self.assertEqual(unknown["status"], "insufficient_positive_events")


if __name__ == "__main__":
    unittest.main()
