# Build only on Windows with one of the isolated, pinned runtime environments.
import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, copy_metadata

root = Path(SPECPATH).parent
resources = "adp/resources"
datas = [(str(root / "frontend/dist"), resources + "/frontend")]
datas += collect_data_files("webview") + collect_data_files("pythonnet") + collect_data_files("clr_loader")
for package in ("torch", "fastapi", "pywebview", "numpy", "pandas", "pyarrow", "scipy", "shapely", "uvicorn"):
    datas += copy_metadata(package)
for variable, destination in (("ADP_DEMO_CACHE", "demo/cache"), ("ADP_MODEL_BUNDLE", "demo/bundle")):
    source = os.environ.get(variable)
    if source:
        datas.append((source, resources + "/" + destination))
release_metadata = os.environ.get("ADP_RELEASE_METADATA")
if release_metadata:
    datas.append((release_metadata, resources + "/release"))

analysis = Analysis(
    [str(root / "scripts/frozen-entry.py")], pathex=[str(root / "src")],
    binaries=collect_dynamic_libs("webview"), datas=datas,
    hiddenimports=["webview.platforms.winforms", "webview.platforms.edgechromium", "clr", "pythonnet",
                   "clr_loader", "_cffi_backend", "uvicorn.logging", "uvicorn.loops.asyncio",
                   "uvicorn.protocols.http.h11_impl", "uvicorn.protocols.http.auto",
                   "uvicorn.protocols.websockets.auto", "uvicorn.lifespan.on", "uvicorn.lifespan.off"],
    excludes=["PyQt5", "PyQt6", "PySide2", "PySide6", "cefpython3", "IPython", "notebook", "pytest", "av2"],
    noarchive=False,
)
pyz = PYZ(analysis.pure)
exe = EXE(pyz, analysis.scripts, [], exclude_binaries=True, name="Interaction Lab",
          debug=False, strip=False, upx=False, console=False)
app = COLLECT(exe, analysis.binaries, analysis.datas, strip=False, upx=False, name="Interaction Lab")
