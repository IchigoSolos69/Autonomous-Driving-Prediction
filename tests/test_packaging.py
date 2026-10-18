from pathlib import Path
import hashlib
import importlib.metadata
import runpy
import subprocess
import sys
import zipfile

import pytest

from adp.data import write_json


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def test_demo_is_bounded_reproducible_and_preserves_source(tmp_path):
    prepare = runpy.run_path(str(SCRIPTS / "prepare-demo.py"))["prepare_demo"]
    source, output = tmp_path / "source", tmp_path / "demo"
    manifest = {"splits": {"train": ["train-a"], "evaluation": ["eval-a", "eval-b"]}}
    write_json(source / "manifest.json", manifest)
    write_json(source / "preprocessing.json", {"version": 1})
    write_json(source / "catalog.json", {"scenarios": [{"id": sid} for sid in ("train-a", "eval-a", "eval-b")]})
    for split, ids in manifest["splits"].items():
        (source / split).mkdir()
        for scenario_id in ids:
            (source / split / f"{scenario_id}.npz").write_bytes(b"existing processed bytes")
    original = (source / "manifest.json").read_bytes()
    first = prepare(source, output, 1)
    assert first["scenario_count"] == 1
    assert list(output.glob("*/*.npz")) == [output / "evaluation/eval-a.npz"]
    assert "© 2021 Argo AI, LLC" in (output / "ATTRIBUTION.txt").read_text(encoding="utf-8")
    (output / "ATTRIBUTION.txt").write_text("outdated notice", encoding="utf-8")
    assert prepare(source, output, 1) == first
    assert "© 2021 Argo AI, LLC" in (output / "ATTRIBUTION.txt").read_text(encoding="utf-8")
    assert (source / "manifest.json").read_bytes() == original
    with pytest.raises(FileExistsError):
        prepare(source, output, 2)
    with pytest.raises(ValueError, match="1–100"):
        prepare(source, tmp_path / "bad", 101)


def test_runtime_qualification_checks_resume_and_rejects_wrong_build():
    qualify = runpy.run_path(str(SCRIPTS / "qualify-runtime.py"))["qualify"]
    result = qualify(device="cpu")
    assert result["actual_forward_backward"]
    assert result["optimizer_rng_continuation"] == "passed"
    with pytest.raises(RuntimeError, match="Expected torch"):
        qualify(expected_torch="does-not-exist")


def test_runtime_lock_verifies_transitive_versions(monkeypatch):
    verify = runpy.run_path(str(SCRIPTS / "verify-runtime-lock.py"))["verify"]
    requirements = SCRIPTS.parent / "requirements"
    lines = (requirements / "runtime-common.lock.txt").read_text().splitlines() + (requirements / "runtime-cpu.lock.txt").read_text().splitlines()
    versions = dict(line.split("==") for line in lines if line and not line.startswith(("#", "-")))
    monkeypatch.setattr(importlib.metadata, "version", versions.__getitem__)
    assert verify("cpu")["exact_runtime_lock"]
    versions["numpy"] = "wrong"
    with pytest.raises(RuntimeError, match="numpy"):
        verify("cpu")


@pytest.mark.skipif(sys.platform != "win32", reason="Windows packaging process guard")
def test_builder_refuses_running_output_before_modifying_it(tmp_path):
    wrapper = tmp_path / "guard.ps1"
    wrapper.write_text('''param([string]$BuildScript,[string]$TestBuild)
function Get-Process {
    [CmdletBinding()]param([string]$Name)
    return [pscustomobject]@{Path=(Join-Path $TestBuild "dist\\Interaction Lab\\Interaction Lab.exe")}
}
try { & $BuildScript -BuildRoot $TestBuild -Python does-not-exist }
catch { Write-Output $_.Exception.Message; exit 41 }
exit 0
''', encoding="utf-8")
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-File", str(wrapper),
                             "-BuildScript", str(SCRIPTS / "build-release.ps1"), "-TestBuild", str(tmp_path / "build")],
                            capture_output=True, text=True, errors="replace", timeout=30)
    assert result.returncode == 41 and "Close the generated" in result.stdout
    assert not (tmp_path / "build/environment").exists()


def _install_payload(archive, destination, expected=None):
    return subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-File", str(SCRIPTS / "install-payload.ps1"),
                           "-Archive", str(archive), "-ExpectedSha", expected or hashlib.sha256(archive.read_bytes()).hexdigest(),
                           "-Destination", str(destination)], capture_output=True, text=True, errors="replace", timeout=30)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows installer extraction")
def test_windows_payload_upgrade_preserves_data_and_rejects_corruption(tmp_path):
    archive = tmp_path / "payload's Ω.zip"
    destination = tmp_path / "application's Ω"
    for version in (b"version1", b"version2"):
        with zipfile.ZipFile(archive, "w") as target:
            target.writestr("Interaction Lab.exe", version)
            target.writestr("_internal/library.dll", version)
        result = _install_payload(archive, destination)
        assert result.returncode == 0, result.stdout + result.stderr
        assert (destination / "Interaction Lab.exe").read_bytes() == version
        (destination / "user-data.txt").write_text("preserve me")
    assert (destination / "user-data.txt").read_text() == "preserve me"
    assert not list(destination.glob(".interaction-*"))
    result = _install_payload(archive, destination, "0" * 64)
    assert result.returncode != 0 and "SHA-256" in result.stderr
    assert (destination / "Interaction Lab.exe").read_bytes() == b"version2"
    with zipfile.ZipFile(archive, "w") as target:
        target.writestr("_internal/../../outside.txt", "bad")
    result = _install_payload(archive, destination)
    assert result.returncode != 0 and "Unsafe" in result.stderr
    assert not (tmp_path / "outside.txt").exists()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows sharing semantics")
def test_windows_payload_rolls_back_when_existing_executable_is_locked(tmp_path):
    import ctypes
    from ctypes import wintypes
    archive, destination = tmp_path / "app.zip", tmp_path / "app"
    with zipfile.ZipFile(archive, "w") as target:
        target.writestr("Interaction Lab.exe", b"old")
        target.writestr("_internal/library.dll", b"old")
    assert _install_payload(archive, destination).returncode == 0
    with zipfile.ZipFile(archive, "w") as target:
        target.writestr("Interaction Lab.exe", b"new")
        target.writestr("_internal/library.dll", b"new")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = kernel.CreateFileW(str(destination / "Interaction Lab.exe"), 0x80000000, 1, None, 3, 0, None)
    assert handle != wintypes.HANDLE(-1).value
    try:
        result = _install_payload(archive, destination)
        assert result.returncode != 0
    finally:
        kernel.CloseHandle(handle)
    assert (destination / "Interaction Lab.exe").read_bytes() == b"old"
    assert (destination / "_internal/library.dll").read_bytes() == b"old"
