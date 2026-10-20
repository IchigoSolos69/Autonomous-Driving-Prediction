"""Inspect or claim finite development work; this command never launches a GPU job."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adp.compute_queue import main

if __name__ == "__main__":
    main()
