"""Run one authenticated, verified private-server transfer cycle."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adp.remote_control import sync_private_server


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--no-resume", action="store_true", help="Transfer only; never start or resume training")
    args = parser.parse_args()
    print(json.dumps(sync_private_server(args.config, resume_training=not args.no_resume), indent=2, allow_nan=False))
