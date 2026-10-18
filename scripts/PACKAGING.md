# Windows private beta packaging

The current project-root `Interaction Lab.exe` is a developer checkout launcher. A release produced by `build-release.ps1` bundles Python, the application, its scientific dependencies and its built interface. End users do not install Python, Node or this repository.

Use Windows 11 x64, Python 3.12 with pip, Node 24, and the patched NSIS 3.13 compiler to build. Keep at least 8 GiB free in addition to the estimated build output. CUDA builds and their temporary files need several GiB; use a build drive with sufficient room. The two runtime lock files select exact PyTorch 2.13.0 CPU and CUDA 13.0 wheels. No driver or CUDA toolkit is installed automatically.

```powershell
# CPU edition in an isolated build environment.
.\scripts\build-release.ps1 -Edition cpu -BuildRoot D:\InteractionLab\release-cpu

# NVIDIA edition, reusing an already qualified runtime without modifying it.
.\scripts\build-release.ps1 -Edition nvidia -RuntimePython D:\InteractionLab\runtime-cu130\Scripts\python.exe -BuildRoot D:\InteractionLab\release-nvidia
```

`-Python` selects the builder's base Python with pip. `-RuntimePython` reuses a prequalified environment and installs packaging tools in the build directory instead. Build checks include dependency consistency, a real model forward/backward pass, optimizer/RNG continuation, a source snapshot/hash, the frozen application's inference/resources/HTTP lifecycle self-test, and file checksums. The build refuses a source tree that changes while being frozen.

Include a bounded real demonstration and an explicitly selected model:

```powershell
.\.venv\Scripts\python.exe -m adp export-model --checkpoint artifacts/experiment/best.pt --cache-dir data/cache --calibration artifacts/experiment/calibration.json --model-card MODEL_CARD.md --output-dir artifacts/reference-bundle
.\scripts\build-release.ps1 -Edition cpu -DemoCache data/cache -ModelBundle artifacts/reference-bundle
```

The builder copies at most 24 scenarios, capped at 64 MiB, without changing the source cache. It records the original manifest and the AV2 noncommercial share-alike attribution. Its selection is for demonstration, not a new benchmark. Omit these parameters to ship an empty import-ready library. Original anonymous evaluation files remain unverified evidence; packaging never changes the risk calibration status.

The CPU build normally produces one setup executable. The NVIDIA build exceeds NSIS's embedded-data limit and produces **two required files**: `Interaction-Lab-VERSION-nvidia-Setup.exe` and the adjacent `Interaction-Lab-VERSION-nvidia-payload.zip`. Distribute both from the same build with their `.sha256` files. Setup checks the embedded payload SHA-256 before reading or extracting it, stages its contents, and restores the prior application if promotion fails. Missing payloads, mismatched releases and insufficient disk space produce an error. The fixed Windows PowerShell script uses literal file parameters and does not disable execution policy.

Installers run per user, check/install Microsoft's signature-verified Evergreen WebView2 Runtime when required, and preserve datasets, models, settings and logs. First-time WebView2 setup may need internet; normal scenario inspection is local. Close the app before upgrading or uninstalling. Install on a drive with enough space; the NVIDIA runtime is several GiB. Manual versioned updates replace the application; there is no automatic updater or telemetry.

For an existing frozen app, build only its installer with `scripts/build-installer.ps1`. The app supports `--verify-installation --device cpu --output C:\path\check.json` for a headless installed-resource, trained-demo and service check. Use `--device cuda` on the installed NVIDIA edition to exercise its actual bundled GPU runtime. Run `scripts/qualify-runtime.py --device cuda --expected-torch 2.13.0+cu130 --checkpoint PATH --output PATH` to qualify the development runtime separately; a CPU test of the CUDA edition does not establish GPU support. NVIDIA payload failures write `%TEMP%\InteractionLab-install.log`.

The installers are unsigned private-beta artifacts. A clean Windows-machine install, native-window flow, offline relaunch, upgrade/uninstall and physical CPU/NVIDIA acceptance must be recorded separately. Local source tests and headless freezing do not substitute for those checks. No driving-safety, calibrated-risk, or stronger-model claim follows from producing an installer.

The manual GitHub release workflow uploads small qualification records by default. Large installer/payload uploads are opt-in and expire after one day; private repositories consume the owner's included Actions minutes and storage. Keep an account spending budget of zero when using only free resources. No workflow has been triggered or billing setting changed by these scripts. See [GitHub's included quotas](https://docs.github.com/en/billing/concepts/product-billing/github-actions).
