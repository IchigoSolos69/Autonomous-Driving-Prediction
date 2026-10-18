import json
from pathlib import Path
import stat
import zipfile

import pytest
import torch

from adp.bundles import export_model, input_schema, install_bundle, load_bundle, schemas_compatible
from adp.data import manifest_hash, write_json
from adp.model import ForecastModel
from adp.training import load_checkpoint


@pytest.fixture
def source(tmp_path):
    cache = tmp_path / "cache"
    write_json(cache / "manifest.json", {"splits": {"train": ["train-a"], "validation": ["select-a"], "calibration": ["cal-a"], "evaluation": ["test-a"]}})
    write_json(cache / "preprocessing.json", {"version": 1, "max_actors": 32, "max_polylines": 128, "map_radius": 100})
    config = {"width": 16, "heads": 4, "temporal_layers": 1, "graph_layers": 1}
    checkpoint = tmp_path / "train.pt"
    torch.save({"format_version": 1, "config": {"model": config, "training": {"batch_size": 4}},
                "model_state_dict": ForecastModel(config).state_dict(), "optimizer_state_dict": {"unused": True},
                "manifest_hash": manifest_hash(cache / "manifest.json"),
                "preprocessing_hash": manifest_hash(cache / "preprocessing.json"),
                "risk_supervision_observations": 3}, checkpoint)
    return checkpoint, cache


def test_export_roundtrip_is_model_only_and_normalizes_schema(source, tmp_path):
    checkpoint, cache = source
    output = tmp_path / "inference"
    metadata = export_model(checkpoint, cache, output)
    loaded = load_bundle(output)
    saved = load_checkpoint(loaded["paths"]["checkpoint"])
    assert "optimizer_state_dict" not in saved
    assert set(saved["config"]) == {"model"}
    assert metadata["provenance"]["training_scenario_ids"] == ["train-a"]
    assert metadata["evaluation"] is None
    assert metadata["evaluation_provenance_verified"] is False
    equivalent = input_schema(cache)
    equivalent["map_radius_m"] = 100.0
    assert schemas_compatible(saved["input_schema"], equivalent)
    equivalent["max_actors"] = 16
    assert not schemas_compatible(saved["input_schema"], equivalent)
    with pytest.raises(FileExistsError):
        export_model(checkpoint, cache, output)


def test_format2_training_checkpoint_exports_portable_format1_inference(source, tmp_path):
    checkpoint, cache = source
    saved = load_checkpoint(checkpoint)
    saved["format_version"] = 2
    saved["optimizer_state_dict"] = {"unused": True}
    torch.save(saved, checkpoint)

    exported = export_model(checkpoint, cache, tmp_path / "format2-bundle")
    loaded = load_bundle(tmp_path / "format2-bundle")
    inference = load_checkpoint(loaded["paths"]["checkpoint"])
    assert inference["format_version"] == 1
    assert "optimizer_state_dict" not in inference
    assert exported["provenance"]["source_checkpoint_sha256"] == manifest_hash(checkpoint)
    for name, tensor in saved["model_state_dict"].items():
        torch.testing.assert_close(inference["model_state_dict"][name], tensor)


def test_export_binds_calibration_to_inference_weights(source, tmp_path):
    checkpoint, cache = source
    calibration = tmp_path / "cal.json"
    write_json(calibration, {"format_version": 1, "checkpoint_sha256": manifest_hash(checkpoint),
                            "manifest_hash": manifest_hash(cache / "manifest.json"),
                            "risk": {"status": "insufficient_positive_events", "positives": 2}})
    evidence = tmp_path / "evidence.json"
    write_json(evidence, {"risk": {"positive_count": 14, "eligible_count": 95}})
    output = tmp_path / "bundle"
    exported = export_model(checkpoint, cache, output, calibration=calibration, evaluation=evidence)
    loaded = load_bundle(output)
    calibrated = json.loads(loaded["paths"]["calibration"].read_text())
    assert calibrated["checkpoint_sha256"] == exported["bundle_id"]
    assert calibrated["source_checkpoint_sha256"] == manifest_hash(checkpoint)
    assert calibrated["risk"]["positives"] == 2
    assert exported["evaluation"]["risk"]["positive_count"] == 14
    assert exported["evaluation_provenance_verified"] is False
    write_json(evidence, {"checkpoint_sha256": "wrong-model"})
    with pytest.raises(ValueError, match="Evaluation belongs"):
        export_model(checkpoint, cache, tmp_path / "wrong-evidence", evaluation=evidence)


def test_export_refuses_unrelated_provenance_and_cleans_staging(source, tmp_path):
    checkpoint, cache = source
    calibration = tmp_path / "wrong.json"
    write_json(calibration, {"checkpoint_sha256": "wrong"})
    with pytest.raises(ValueError, match="Calibration does not belong"):
        export_model(checkpoint, cache, tmp_path / "bad", calibration=calibration)
    assert not list(tmp_path.glob(".bundle-export-*"))
    write_json(cache / "manifest.json", {"splits": {"train": ["changed"]}})
    with pytest.raises(ValueError, match="provenance"):
        export_model(checkpoint, cache, tmp_path / "bad")


def test_hash_failure_does_not_replace_active_model(source, tmp_path):
    checkpoint, cache = source
    bundle = tmp_path / "bundle"
    export_model(checkpoint, cache, bundle)
    models = tmp_path / "models"
    first = install_bundle(bundle, models)
    active = (models / "active.json").read_bytes()
    with (bundle / "weights.pt").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        install_bundle(bundle, models)
    assert (models / "active.json").read_bytes() == active
    assert first["paths"]["checkpoint"].is_file()
    assert not list(models.glob(".bundle-import-*"))


def test_load_rejects_nonobject_metadata(source, tmp_path):
    checkpoint, cache = source
    bundle = tmp_path / "bundle"
    export_model(checkpoint, cache, bundle)
    (bundle / "bundle.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="object"):
        load_bundle(bundle)


@pytest.mark.parametrize("name", ["../weights.pt", "C:/weights.pt", "dir/weights.pt", "weights.pt:stream", "\\weights.pt"])
def test_zip_rejects_unsafe_paths(tmp_path, name):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as target:
        target.writestr(name, b"bad")
    with pytest.raises(ValueError, match="Unsafe"):
        install_bundle(archive, tmp_path / "models")
    assert not (tmp_path / "models" / "active.json").exists()


def test_zip_rejects_duplicates_and_expansion_budget(tmp_path, monkeypatch):
    import adp.bundles as bundles
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as target:
        target.writestr("weights.pt", b"1234")
        target.writestr("WEIGHTS.PT", b"1234")
    with pytest.raises(ValueError, match="Duplicate"):
        install_bundle(archive, tmp_path / "models")
    with zipfile.ZipFile(archive, "w") as target:
        target.writestr("weights.pt", b"1234")
    monkeypatch.setattr(bundles, "MAX_TOTAL_BYTES", 3)
    with pytest.raises(ValueError, match="Expanded"):
        install_bundle(archive, tmp_path / "models")


def test_zip_rejects_symlinks(tmp_path):
    archive = tmp_path / "link.zip"
    entry = zipfile.ZipInfo("weights.pt")
    entry.create_system = 3
    entry.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as target:
        target.writestr(entry, "../../outside")
    with pytest.raises(ValueError, match="Unsafe"):
        install_bundle(archive, tmp_path / "models")


def test_zip_install_and_keep_only_previous_registered_model(source, tmp_path):
    checkpoint, cache = source
    models = tmp_path / "models"
    ids = []
    for index in range(3):
        saved = load_checkpoint(checkpoint)
        saved["model_state_dict"]["world_queries"] += .1
        torch.save(saved, checkpoint)
        bundle = tmp_path / f"bundle-{index}"
        metadata = export_model(checkpoint, cache, bundle)
        archive = tmp_path / f"model-{index}.zip"
        with zipfile.ZipFile(archive, "w") as target:
            for path in bundle.iterdir():
                target.write(path, path.name)
        installed = install_bundle(archive, models)
        ids.append(metadata["bundle_id"])
        assert installed["paths"]["checkpoint"].is_file()
    assert json.loads((models / "active.json").read_text()) == {"format_version": 1, "current": ids[2], "previous": ids[1]}
    assert not (models / ids[0]).exists()
    assert (models / ids[1]).is_dir()
