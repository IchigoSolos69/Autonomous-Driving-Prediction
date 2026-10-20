"""Small CPU tests for resumability and held-out calibration."""

import json
import itertools
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

    def test_checkpoint_retries_transient_destination_lock(self):
        import os
        from adp.training import _atomic_checkpoint, load_checkpoint
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            _atomic_checkpoint(path, {"format_version": 1, "step": 1})
            replace = os.replace
            attempts = []
            def locked_then_replace(source, destination):
                attempts.append(1)
                if len(attempts) < 3:
                    raise PermissionError("transient Windows lock")
                return replace(source, destination)
            with patch("adp.training.os.replace", side_effect=locked_then_replace), patch("adp.training.time.sleep") as sleep:
                _atomic_checkpoint(path, {"format_version": 1, "step": 2})
            self.assertEqual(len(attempts), 3)
            self.assertEqual(sleep.call_count, 2)
            self.assertEqual(load_checkpoint(path)["step"], 2)
            with patch("adp.training.os.replace", side_effect=PermissionError("persistent lock")), patch("adp.training.time.sleep"):
                with self.assertRaises(PermissionError):
                    _atomic_checkpoint(path, {"format_version": 1, "step": 3})
            self.assertEqual(load_checkpoint(path)["step"], 2)
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
        import adp.research
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
            protected = Path(directory)/"protected"
            protected_config = {**config, "training": {**config["training"], "minimum_free_gib": 8}}
            with patch("adp.research.ensure_space", return_value=16*1024**3) as space:
                protected_first = train(protected_config, cache, protected, epochs=1, device="cpu")
                self.assertEqual(space.call_count, 2)
                self.assertTrue(all(call.args[1] == 8 and call.args[2] >= 64*1024**2 for call in space.call_args_list))
            previous = {name: (protected/name).read_bytes() for name in ("last.pt", "best.pt")}
            with patch("adp.research.ensure_space", side_effect=RuntimeError("Storage paused")):
                with self.assertRaisesRegex(RuntimeError, "Storage paused"):
                    train(protected_config, cache, protected, epochs=2, device="cpu", resume=protected_first["checkpoint"])
            for name, value in previous.items():
                self.assertEqual((protected/name).read_bytes(), value)
            self.assertFalse(list(protected.glob("*.tmp")))
            prep = cache/"preprocessing.json"
            prep.write_text('{}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, "preprocessing differs"):
                train(config, cache, output, epochs=3, device="cpu", resume=second["checkpoint"])
            prep.unlink()
            manifest.write_text('{"changed":true}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "dataset manifest differs"):
                train(config, cache, output, epochs=3, device="cpu", resume=second["checkpoint"])

    def test_mid_epoch_resume_matches_uninterrupted_order_rng_and_updates(self):
        import numpy as np
        from adp.data import preprocessing_hash, read_json
        from adp.training import _atomic_checkpoint, load_checkpoint, train

        seen = []
        validation_budget = {"armed": False, "expired": False}

        class TinyModel(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                self.linear = torch.nn.Linear(1, 1)
                self.dropout = torch.nn.Dropout(.3)

            def forward(self, history):
                if self.training:
                    seen.extend(int(value) for value in history.flatten())
                elif validation_budget["armed"]:
                    validation_budget["expired"] = True
                noise = .01*(random.random()+np.random.random()) if self.training else 0
                return {"value": self.linear(self.dropout(history))+noise}

        fake_data = types.SimpleNamespace(
            CachedDataset=lambda cache_dir, split: [{"history": torch.tensor([float(i)])} for i in range(7 if split == "train" else 3)],
            collate_scenes=lambda scenes: {"history": torch.stack([scene["history"] for scene in scenes])},
            preprocessing_hash=preprocessing_hash, read_json=read_json,
        )

        def loss(outputs, batch):
            value = outputs["value"].square().mean()
            return {"loss": value, "nll": value, "risk_supervision_count": len(batch["history"])}

        fake_model = types.SimpleNamespace(ForecastModel=TinyModel, model_inputs=lambda batch: batch, forecast_loss=loss)
        with tempfile.TemporaryDirectory() as directory, patch.dict("sys.modules", {"adp.data": fake_data, "adp.model": fake_model}), \
                patch("adp.training.time.monotonic", side_effect=itertools.count().__next__):
            cache = Path(directory)/"cache"
            cache.mkdir()
            (cache/"manifest.json").write_text('{"splits":{"train":["a"],"validation":["b"]}}')
            config = {"seed": 6, "model": {}, "training": {"batch_size": 2, "accumulation_steps": 2,
                                                               "checkpoint_seconds": .000001}}
            whole = train(config, cache, Path(directory)/"whole", epochs=2, device="cpu")
            expected = load_checkpoint(whole["checkpoint"])
            expected_order = seen.copy()
            seen.clear()
            interrupted_dir = Path(directory)/"interrupted"

            def interrupt_after_commit(path, checkpoint):
                _atomic_checkpoint(path, checkpoint)
                if checkpoint["partial_epoch"]:
                    raise RuntimeError("simulated interruption")

            with patch("adp.training._atomic_checkpoint", side_effect=interrupt_after_commit):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    train(config, cache, interrupted_dir, epochs=2, device="cpu")
            partial = load_checkpoint(interrupted_dir/"last.pt")
            self.assertEqual(partial["training_progress"]["cursor"], 4)
            self.assertEqual(seen, expected_order[:4])
            budget_dir = Path(directory)/"budget"
            budget = train(config, cache, budget_dir, epochs=2, device="cpu", max_runtime_seconds=.000001)
            self.assertTrue(budget["budget_exhausted"])
            self.assertEqual(budget["completed_epochs"], 0)
            json.dumps(budget, allow_nan=False)
            self.assertEqual(load_checkpoint(budget["checkpoint"])["training_progress"]["cursor"], 0)
            budget_continued = train(config, cache, budget_dir, epochs=2, device="cpu", resume=budget["checkpoint"])
            budget_saved = load_checkpoint(budget_continued["checkpoint"])
            self.assertFalse(budget_continued["budget_exhausted"])
            self.assertEqual(budget_saved["history"], expected["history"])
            for name in expected["model_state_dict"]:
                torch.testing.assert_close(budget_saved["model_state_dict"][name], expected["model_state_dict"][name], rtol=0, atol=0)
            seen.clear()
            validation_budget["armed"] = True
            validation_dir = Path(directory)/"validation-budget"
            with patch("adp.training.time.monotonic", side_effect=lambda: 100 if validation_budget["expired"] else 0):
                stopped_validation = train(config, cache, validation_dir, epochs=2, device="cpu", max_runtime_seconds=10)
            validation_budget["armed"] = False
            self.assertTrue(stopped_validation["budget_exhausted"])
            self.assertEqual(stopped_validation["completed_epochs"], 0)
            self.assertEqual(load_checkpoint(stopped_validation["checkpoint"])["training_progress"]["cursor"], 7)
            self.assertEqual(seen, expected_order[:7])
            stopped_again = train(config, cache, validation_dir, epochs=2, device="cpu",
                                  resume=stopped_validation["checkpoint"], max_runtime_seconds=.000001)
            self.assertTrue(stopped_again["budget_exhausted"])
            self.assertEqual(seen, expected_order[:7])
            validation_continued = train(config, cache, validation_dir, epochs=2, device="cpu", resume=stopped_again["checkpoint"])
            validation_saved = load_checkpoint(validation_continued["checkpoint"])
            self.assertEqual(seen, expected_order)
            self.assertEqual(validation_saved["history"], expected["history"])
            for name in expected["model_state_dict"]:
                torch.testing.assert_close(validation_saved["model_state_dict"][name], expected["model_state_dict"][name], rtol=0, atol=0)
            seen[:] = expected_order[:4]
            resumed = train(config, cache, interrupted_dir, epochs=2, device="cpu", resume=interrupted_dir/"last.pt")
            actual = load_checkpoint(resumed["checkpoint"])
            self.assertEqual(seen, expected_order)
            self.assertEqual(actual["history"], expected["history"])
            self.assertEqual(actual["statistics"]["examples_seen"], 14)
            self.assertEqual(actual["statistics"]["objective_totals"], expected["statistics"]["objective_totals"])
            self.assertEqual(actual["risk_supervision_observations"], 14)
            self.assertEqual(actual["global_step"], 4)
            self.assertEqual(actual["scheduler_state_dict"], expected["scheduler_state_dict"])
            for key in ("model_state_dict", "optimizer_state_dict", "rng_state", "scaler_state_dict"):
                def compare(left, right):
                    if isinstance(left, torch.Tensor):
                        torch.testing.assert_close(left, right, rtol=0, atol=0)
                    elif isinstance(left, dict):
                        self.assertEqual(left.keys(), right.keys())
                        for name in left:
                            compare(left[name], right[name])
                    elif isinstance(left, (list, tuple)):
                        self.assertEqual(len(left), len(right))
                        for item, other in zip(left, right):
                            compare(item, other)
                    else:
                        self.assertEqual(left, right)
                compare(actual[key], expected[key])
            legacy = {**partial, "format_version": 1}
            legacy_path = Path(directory)/"legacy.pt"
            _atomic_checkpoint(legacy_path, legacy)
            with self.assertRaisesRegex(ValueError, "legacy partial-epoch"):
                train(config, cache, Path(directory)/"legacy-run", epochs=2, device="cpu", resume=legacy_path)
            _atomic_checkpoint(legacy_path, {**actual, "format_version": 1})
            boundary = train(config, cache, Path(directory)/"legacy-boundary", epochs=3, device="cpu", resume=legacy_path)
            self.assertEqual(boundary["completed_epochs"], 3)

    def test_joint_minfde_selection_breaks_ties_by_focal_then_earliest(self):
        from adp.data import preprocessing_hash, read_json
        from adp.training import _selection_key, _validation_displacement, load_checkpoint, train

        self.assertLess(_selection_key("joint_minFDE", 100, 3, 4, 2), _selection_key("joint_minFDE", 1, 3, 5, 1))
        self.assertLess(_selection_key("joint_minFDE", 100, 3, 4, 2), _selection_key("joint_minFDE", 1, 3, 4, 3))
        with self.assertRaisesRegex(ValueError, "complete scored"):
            _selection_key("joint_minFDE", 1, None, None, 1)
        means = torch.zeros(1, 2, 3, 2, 2)
        means[:, 0, 0, -1, 0] = 6
        means[:, 1, 1, -1, 0] = 4
        means[:, :, 2] = 100
        batch = {"future": torch.zeros(1, 3, 2, 2), "future_mask": torch.ones(1, 3, 2, dtype=torch.bool),
                 "dynamic_mask": torch.ones(1, 3, dtype=torch.bool), "actor_mask": torch.ones(1, 3, dtype=torch.bool),
                 "scored_mask": torch.tensor([[True, True, False]])}
        measured = _validation_displacement({"means": means}, batch)
        self.assertEqual(measured, {"joint_sum": 2., "joint_count": 1, "focal_sum": 0., "focal_count": 1})

        class TinyModel(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                self.linear = torch.nn.Linear(1, 1)

            def forward(self, history):
                return {"value": self.linear(history)}

        fake_data = types.SimpleNamespace(
            CachedDataset=lambda cache_dir, split: [{"history": torch.ones(1)}],
            collate_scenes=lambda scenes: {"history": torch.stack([scene["history"] for scene in scenes])},
            preprocessing_hash=preprocessing_hash, read_json=read_json,
        )
        fake_model = types.SimpleNamespace(ForecastModel=TinyModel, model_inputs=lambda batch: batch,
                                          forecast_loss=lambda outputs, batch: {"loss": outputs["value"].square().mean()})
        validation = [{"joint_sum": joint, "joint_count": 1, "focal_sum": focal, "focal_count": 1}
                      for joint, focal in ((3, 5), (3, 4), (3, 4), (3.1, 1))]
        with tempfile.TemporaryDirectory() as directory, patch.dict("sys.modules", {"adp.data": fake_data, "adp.model": fake_model}), \
                patch("adp.training._validation_displacement", side_effect=validation):
            cache = Path(directory)/"cache"
            cache.mkdir()
            (cache/"manifest.json").write_text('{"splits":{"train":["a"],"validation":["b"]}}')
            result = train({"model": {}, "training": {"checkpoint_selection": "joint_minFDE"}}, cache,
                           Path(directory)/"run", epochs=4, device="cpu")
            best = load_checkpoint(result["best_checkpoint"])
            self.assertEqual(best["completed_epochs"], 2)
            self.assertEqual(result["best_selection"], {"metric": "joint_minFDE", "key": [3., 4.], "epoch": 2})


if __name__ == "__main__":
    unittest.main()
