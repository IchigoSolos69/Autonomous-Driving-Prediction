"""Point read-only monitoring at an existing, user-synced saved queue folder."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adp.drive_evidence import DEFAULT_CONFIG, configure_folder


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=("colab", "kaggle"), default="colab")
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    print(json.dumps(configure_folder(args.directory, args.host, args.config), sort_keys=True))


if __name__ == "__main__":
    main()
