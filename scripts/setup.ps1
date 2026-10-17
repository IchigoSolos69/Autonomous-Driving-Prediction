param(
    [string]$Python = "",
    [switch]$CpuOnly,
    [switch]$WithAv2
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$ProjectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$EnvironmentPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not $Python) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $Python = & py -3.12 -c "import sys; print(sys.executable)"
        if ($LASTEXITCODE -ne 0) { $Python = "" }
    }
    $BundledPython = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
    if (-not $Python -and (Test-Path -LiteralPath $BundledPython)) { $Python = $BundledPython }
    if (-not $Python -and (Test-Path -LiteralPath $EnvironmentPython)) { $Python = $EnvironmentPython }
    if (-not $Python -and (Get-Command python -ErrorAction SilentlyContinue)) {
        $Python = (Get-Command python).Source
    }
}
if (-not $Python) { throw "Install Python 3.12, or pass -Python C:\path\to\python.exe." }
& $Python -c "import sys; assert sys.version_info[:2] == (3,12), 'This project requires Python 3.12'; print(sys.executable)"
if ($LASTEXITCODE -ne 0) { throw "Select a working Python 3.12 interpreter using -Python." }

if (-not (Test-Path -LiteralPath $EnvironmentPython)) {
    # Reuse an existing base PyTorch installation through this local environment.
    $ExistingTorch = & $Python -c "import importlib.util; print(int(importlib.util.find_spec('torch') is not None))"
    $VenvArguments = @("-m", "venv", "--without-pip")
    if ($ExistingTorch -eq "1") { $VenvArguments += "--system-site-packages" }
    $VenvArguments += (Join-Path $ProjectRoot ".venv")
    & $Python @VenvArguments
    if ($LASTEXITCODE -ne 0) { throw "Unable to create .venv. Select a standard Python 3.12 installation." }
}

$PipPython = $EnvironmentPython
& $EnvironmentPython -m pip --version 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    $PipPython = $Python
    & $PipPython -m pip --version 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "The selected Python has no pip. Install pip in that Python, then rerun setup." }
}
function Invoke-ProjectPip {
    param([string[]]$PipArguments)
    & $PipPython -m pip --python $EnvironmentPython @PipArguments
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed. See the pip error above." }
}

if ($CpuOnly) {
    & $EnvironmentPython -c "import torch; assert tuple(map(int, torch.__version__.split('+')[0].split('.')[:2])) >= (2,7)" 2>$null
} else {
    & $EnvironmentPython -c "import torch; assert torch.cuda.is_available(); x=torch.randn(32,32,device='cuda',requires_grad=True); (x@x.T).sum().backward(); torch.cuda.synchronize()" 2>$null
}
if ($LASTEXITCODE -ne 0) {
    $TorchIndex = "https://download.pytorch.org/whl/cu128"
    $TorchVersion = "torch==2.11.0+cu128"
    if ($CpuOnly) { $TorchIndex = "https://download.pytorch.org/whl/cpu"; $TorchVersion = "torch==2.11.0+cpu" }
    Write-Host "Installing $TorchVersion once into .venv. CUDA wheels require several GB of free space."
    Invoke-ProjectPip -PipArguments @("install", "--no-cache-dir", $TorchVersion, "--index-url", $TorchIndex)
} else {
    Write-Host "Reusing the existing compatible PyTorch installation."
}

$ProjectExtras = "$ProjectRoot`[dev,desktop]"
if ($WithAv2) { $ProjectExtras = "$ProjectRoot`[dev,av2,desktop]" }
Invoke-ProjectPip -PipArguments @("install", "--no-cache-dir", "-e", $ProjectExtras)
if (-not $CpuOnly) {
    & $EnvironmentPython -c "from adp.training import gpu_smoke_test; import json; print(json.dumps(gpu_smoke_test(),indent=2))"
    if ($LASTEXITCODE -ne 0) { throw "The GPU smoke test failed. Check the installed driver and CUDA PyTorch build." }
}
Write-Host "Ready. Use .venv\Scripts\python.exe -m adp --help"
