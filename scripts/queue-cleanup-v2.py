"""Keep the approved cleanup rules, allowing the existing larger source manifest."""
from __future__ import annotations

import hashlib
from pathlib import Path
import stat
from types import FunctionType, ModuleType

BASE_SHA256 = "9ef81922a40a35de0ac6bee47a85a3cfdf8367d8951e348aadd094c7f009168a"
MAX_SOURCE_MANIFEST = 8 * 1024**2
MAX_BASE_CODE = 2 * 1024**2


def load_base():
    """Execute only the exact reviewed sibling bytes, without a second file read."""
    path = Path(__file__).absolute().parent / "queue_cleanup.py"
    for item in (path, *path.parents):
        attributes = item.lstat()
        if (stat.S_ISLNK(attributes.st_mode)
                or getattr(attributes, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)):
            raise ValueError("Cleanup base paths must not contain links or junctions")
    before = path.stat()
    if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
            or not 0 < before.st_size <= MAX_BASE_CODE):
        raise ValueError("Cleanup base must be a bounded regular unshared file")
    with path.open("rb") as stream:
        data = stream.read(MAX_BASE_CODE + 1)
    after = path.stat()
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size,
                              value.st_mtime_ns, value.st_ctime_ns, value.st_nlink, value.st_mode)
    if identity(after) != identity(before) or hashlib.sha256(data).hexdigest() != BASE_SHA256:
        raise ValueError("Cleanup base SHA-256 differs from the pinned reviewed source")
    base = ModuleType("interaction_queue_cleanup_v2_base")
    base.__file__ = str(path)
    exec(compile(data, str(path), "exec"), vars(base))
    original_sha, original_read = base._sha, base._read
    # Reuse every parser check with a local bound; other metadata stays at 2 MiB.
    parse_manifest = FunctionType(base._json_bytes.__code__,
                                 {**vars(base), "MAX_JSON": MAX_SOURCE_MANIFEST})

    def sha(path, limit=base.MAX_FILE):
        path = base._safe(path)
        if path.name == "source-manifest.json" and limit == base.MAX_JSON:
            limit = MAX_SOURCE_MANIFEST
        return original_sha(path, limit)

    def read(path):
        path = base._safe(path)
        if path.name != "source-manifest.json":
            return original_read(path)
        base._regular(path, MAX_SOURCE_MANIFEST)
        with path.open("rb") as stream:
            return parse_manifest(stream.read(MAX_SOURCE_MANIFEST + 1))

    base._sha, base._read = sha, read
    return base


def main(argv=None):
    return load_base().main(argv)


if __name__ == "__main__":
    main()
