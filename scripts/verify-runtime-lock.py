"""Refuse reuse of a runtime that differs from its exact edition lock."""
import argparse
import importlib.metadata
import json
from pathlib import Path


def verify(edition):
    root = Path(__file__).resolve().parents[1] / "requirements"
    lines = (root / "runtime-common.lock.txt").read_text().splitlines()
    lines += (root / f"runtime-{edition}.lock.txt").read_text().splitlines()
    mismatches = []
    for line in lines:
        if not line or line.startswith(("#", "-")):
            continue
        name, expected = line.split("==")
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        if actual != expected:
            mismatches.append({"package": name, "expected": expected, "actual": actual})
    if mismatches:
        raise RuntimeError("Runtime does not match its edition lock: " + json.dumps(mismatches))
    return {"edition": edition, "exact_runtime_lock": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--edition", choices=["cpu", "nvidia"], required=True)
    print(json.dumps(verify(parser.parse_args().edition)))
