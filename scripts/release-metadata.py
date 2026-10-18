"""Record the actual distribution set and copy shipped Python license notices."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re


parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--edition", choices=["cpu", "nvidia"], required=True)
parser.add_argument("--verify-source", action="store_true")
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
project = Path(__file__).resolve().parents[1]
inputs = [path for name in ("src/adp", "frontend/src", "frontend/public", "frontend/dist", "scripts", "requirements")
          for path in (project / name).rglob("*")
          if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"]
inputs += [project / name for name in ("pyproject.toml", "frontend/package.json", "frontend/package-lock.json", "frontend/index.html", "frontend/vite.config.ts", "frontend/tsconfig.json")]
source_files = {path.relative_to(project).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(inputs)}
source_id = hashlib.sha256(json.dumps(source_files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
if args.verify_source:
    expected = json.loads((args.output / "source.json").read_text(encoding="utf-8"))
    if expected["sha256"] != source_id:
        raise RuntimeError("Source changed during freezing. Rebuild after the implementation stops changing.")
    print("Frozen build source snapshot verified:", source_id)
    raise SystemExit(0)
(args.output / "source.json").write_text(json.dumps({"sha256": source_id, "files": source_files}, indent=2), encoding="utf-8")
items = []
for distribution in sorted(importlib.metadata.distributions(), key=lambda item: item.metadata["Name"].lower()):
    name, version = distribution.metadata["Name"], distribution.version
    copied = []
    for relative in distribution.files or []:
        if re.search(r"(^|/)(licen[cs]e|copying|notice)([./_-]|$)", str(relative), re.I):
            source = Path(distribution.locate_file(relative))
            if source.is_file():
                safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
                target = args.output / "licenses" / safe_name / f"{len(copied):03d}-{source.name}"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
                copied.append(str(target.relative_to(args.output)).replace("\\", "/"))
    items.append({"name": name, "version": version, "license_expression": distribution.metadata.get("License-Expression"),
                  "license": distribution.metadata.get("License"), "license_files": copied})
(args.output / "dependencies.json").write_text(json.dumps({"edition": args.edition, "packages": items}, indent=2), encoding="utf-8")
paths = [path for path in args.output.rglob("*") if path.is_file() and path != args.output / "SHA256SUMS"]
(args.output / "SHA256SUMS").write_text("".join(hashlib.sha256(path.read_bytes()).hexdigest() + "  " + str(path.relative_to(args.output)).replace("\\", "/") + "\n" for path in sorted(paths)), encoding="utf-8")
