"""Stream a large frozen application into an adjacent, checksum-bound payload."""
import argparse
import json
from pathlib import Path
import shutil
import zipfile


parser = argparse.ArgumentParser()
parser.add_argument("--app-dir", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
root = args.app_dir.resolve()
paths = sorted(path for path in root.rglob("*") if path.is_file())
if not (root / "Interaction Lab.exe").is_file() or not (root / "_internal").is_dir():
    raise ValueError("Expected a complete one-directory frozen application")
if any(path.is_symlink() or root not in path.resolve().parents for path in paths):
    raise ValueError("Application payload links are not supported")
temporary = args.output.with_suffix(".zip.tmp")
try:
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for path in paths:
            if shutil.disk_usage(args.output.parent).free < 8 * 1024**3 + path.stat().st_size + 16 * 1024**2:
                raise OSError("Payload build stopped to preserve 8 GiB free space; use a larger build drive.")
            archive.write(path, path.relative_to(root).as_posix())
    temporary.replace(args.output)
finally:
    temporary.unlink(missing_ok=True)
print(json.dumps({"payload": str(args.output), "files": len(paths), "size_bytes": args.output.stat().st_size}))
