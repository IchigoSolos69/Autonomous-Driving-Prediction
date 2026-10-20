"""Standalone local monitor; run with the project's qualified Python runtime."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adp.compute_monitor import main


if __name__ == "__main__":
    raise SystemExit(main())
