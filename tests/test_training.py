"""Small CPU tests for resumability and held-out calibration."""

import json
from pathlib import Path
import random
import tempfile
import types
import unittest
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class CalibrationTests(unittest.TestCase):
    def test_temperature_reduces_held_out_nll(self):
        from adp.calibration import fit_temperature

        result = fit_temperature(torch.tensor([[8., 0.], [8., 0.], [8., 0.], [8., 0.]]), torch.tensor([0, 0, 0, 1]))
        self.assertTrue(result["fitted"])
        self.assertGreater(result["mode_temperature"], 1)
        self.assertLess(result["nll_after"], result["nll_before"])

    def test_empty_mode_calibration_stays_identity(self):
        from adp.calibration import fit_temperature

        result = fit_temperature(torch.empty(0, 6), torch.empty(0, dtype=torch.long))
        self.assertFalse(result["fitted"])
        self.assertEqual(result["mode_temperature"], 1)

    def test_platt_needs_both_classes(self):
        from adp.calibration import fit_platt

        result = fit_platt(torch.tensor([1., 2.]), torch.tensor([0., 0.]))
        self.assertFalse(result["fitted"])
        self.assertEqual(result["risk_temperature"], 1)
        self.assertEqual(result["risk_bias"], 0)

    def test_platt_reduces_held_out_nll(self):
        from adp.calibration import fit_platt

        result = fit_platt(torch.tensor([-5., -4., 4., 5.]), torch.tensor([0., 1., 0., 1.]))
        self.assertTrue(result["fitted"])
        self.assertLessEqual(result["nll_after"], result["nll_before"])
        self.assertGreater(result["risk_temperature"], 0)

    def test_calibration_preserves_trajectories(self):
        from adp.calibration import apply_calibration

        outputs = {"means": torch.zeros(1, 6, 1, 60, 2), "logits": torch.ones(1, 6), "risk_logits": torch.tensor([[2.]])}
        calibrated = apply_calibration(outputs, {"mode_temperature": 2., "risk_temperature": 2., "risk_bias": -1.})
        self.assertIs(calibrated["means"], outputs["means"])
        torch.testing.assert_close(calibrated["logits"].softmax(-1).sum(-1), torch.ones(1))
        self.assertEqual(float(calibrated["risk_logits"][0, 0]), 0)

    def test_mode_targets_ignore_missing_and_static_actors(self):
        from adp.calibration import mode_targets

        means = torch.zeros(1, 2, 3, 2, 2)
        means[:, 0, 0] = 10
        means[:, 1, 1] = 100
        batch = {
            "future": torch.zeros(1, 3, 2, 2), "future_mask": torch.tensor([[[True, True], [True, True], [False, False]]]),
            "dynamic_mask": torch.tensor([[True, False, True]]), "actor_mask": torch.ones(1, 3, dtype=torch.bool),
        }
        batch["future"][:, 2] = float("nan")
        targets, eligible = mode_targets({"means": means}, batch)
        self.assertEqual(targets.tolist(), [1])
        self.assertEqual(eligible.tolist(), [True])

    def test_mode_targets_use_scored_joint_fde(self):
        from adp.calibration import mode_targets

        means = torch.zeros(1, 2, 2, 3, 2)
        means[:, 0, 0, -1] = 4
        means[:, 1, 0, :-1] = 20
        means[:, 1, 1, -1] = 100
        batch = {
            "future": torch.zeros(1, 2, 3, 2), "future_mask": torch.ones(1, 2, 3, dtype=torch.bool),
            "dynamic_mask": torch.ones(1, 2, dtype=torch.bool), "actor_mask": torch.ones(1, 2, dtype=torch.bool),
            "scored_mask": torch.tensor([[True, False]]),
        }
        targets, eligible = mode_targets({"means": means}, batch)
        self.assertEqual(targets.tolist(), [1])
        self.assertEqual(eligible.tolist(), [True])


@unittest.skipIf(torch is None, "PyTorch is not installed")
class CheckpointTests(unittest.TestCase):
    def test_portable_checkpoint_restores_random_state(self):
        import numpy as np
        from adp.training import _atomic_checkpoint, _restore_rng, _rng_state, load_checkpoint

        random.seed(8)
        np.random.seed(8)
        torch.manual_seed(8)
        state = _rng_state()
        expected = (random.random(), float(np.random.random()), torch.rand(3))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            _atomic_checkpoint(path, {"format_version": 1, "rng_state": state})
            loaded = load_checkpoint(path)
            _restore_rng(loaded["rng_state"])
        self.assertEqual(random.random(), expected[0])
        self.assertEqual(float(np.random.random()), expected[1])
        torch.testing.assert_close(torch.rand(3), expected[2])

    def test_failed_save_keeps_previous_checkpoint(self):
        from adp.training import _atomic_checkpoint, load_checkpoint

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            _atomic_checkpoint(path, {"format_version": 1, "step": 1})
            with patch("adp.training.torch.save", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    _atomic_checkpoint(path, {"format_version": 1, "step": 2})
            self.assertEqual(load_checkpoint(path)["step"], 1)
            self.assertFalse(path.with_suffix(".pt.tmp").exists())

    def test_manifest_hash_changes_when_manifest_changes(self):
        from adp.training import manifest_hash

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text('{"seed":1}', encoding="utf-8")
            before = manifest_hash(directory)
            path.write_text('{"seed":2}', encoding="utf-8")
            self.assertNotEqual(before, manifest_hash(directory))

    def test_gpu_smoke_refuses_cpu(self):
        from adp.training import gpu_smoke_test

        with patch("adp.training.torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "requires a CUDA device"):
                gpu_smoke_test()

    def test_training_resume_and_manifest_guard(self):
        from adp.training import load_checkpoint, train
        from adp.data import preprocessing_hash, read_json

        class TinyModel(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                self.linear = torch.nn.Linear(1, 1)

            def forward(self, history):
                return {"value": self.linear(history)}

        fake_data = types.SimpleNamespace(
            CachedDataset=lambda cache_dir, split: [{"history": torch.ones(1)} for _ in range(3)],
            collate_scenes=lambda scenes: {"history": torch.stack([scene["history"] for scene in scenes])},
            preprocessing_hash=preprocessing_hash,
            read_json=read_json,
        )
        fake_model = types.SimpleNamespace(
            ForecastModel=TinyModel,
            model_inputs=lambda batch: batch,
            forecast_loss=lambda outputs, batch: {"loss": outputs["value"].square().mean()},
        )
        with tempfile.TemporaryDirectory() as directory, patch.dict("sys.modules", {"adp.data": fake_data, "adp.model": fake_model}):
            cache = Path(directory) / "cache"
            cache.mkdir()
            manifest = cache / "manifest.json"
            manifest.write_text(json.dumps({"splits": {"train": ["a"], "validation": ["b"]}}), encoding="utf-8")
            output = Path(directory) / "run"
            config = {"model": {}, "training": {"batch_size": 2, "accumulation_steps": 2}}
            first = train({**config, "data": {"max_actors": 32}}, cache, output, epochs=1, device="cpu")
            foreign_output = Path(directory)/"other-run"
            foreign_output.mkdir()
            for checkpoint_name in ("best.pt", "last.pt"):
                foreign_checkpoint = foreign_output/checkpoint_name
                foreign_checkpoint.write_bytes(b"unrelated checkpoint")
                with self.assertRaisesRegex(ValueError, "already contains checkpoints"):
                    train(config, cache, foreign_output, epochs=2, device="cpu", resume=first["checkpoint"])
                self.assertEqual(foreign_checkpoint.read_bytes(), b"unrelated checkpoint")
                self.assertEqual(list(foreign_output.iterdir()), [foreign_checkpoint])
                foreign_checkpoint.unlink()
            for changed in ({"seed": 43}, {"model": {"graph": False}}, {"training": {"batch_size": 1}}):
                with self.assertRaisesRegex(ValueError, "configuration differs"):
                    train({**config, **changed}, cache, output, epochs=2, device="cpu", resume=first["checkpoint"])
            second = train(config, cache, output, epochs=2, device="cpu", resume=first["checkpoint"])
            saved = load_checkpoint(second["checkpoint"])
            self.assertEqual(saved["completed_epochs"], 2)
            self.assertEqual(saved["global_step"], 2)
            self.assertEqual(len(saved["history"]), 2)
            self.assertTrue((output / "best.pt").exists())
            prep = cache/"preprocessing.json"
            prep.write_text('{}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, "preprocessing differs"):
                train(config, cache, output, epochs=3, device="cpu", resume=second["checkpoint"])
            prep.unlink()
            manifest.write_text('{"changed":true}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "dataset manifest differs"):
                train(config, cache, output, epochs=3, device="cpu", resume=second["checkpoint"])


if __name__ == "__main__":
    unittest.main()
