param(
    [ValidateSet("cpu", "nvidia")][string]$Edition = "cpu",
    [string]$Python = "python",
    [string]$RuntimePython = "",
    [string]$BuildRoot = "",
    [string]$DemoCache = "",
    [string]$ModelBundle = "",
    [string]$Version = "0.1.0-beta",
    [string]$Nsis = "",
    [switch]$SkipInstaller
)
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$ProjectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
if (-not $BuildRoot) { $BuildRoot = Join-Path $ProjectRoot "artifacts\release-build\$Edition" }
$BuildRoot = [IO.Path]::GetFullPath($BuildRoot)
New-Item -ItemType Directory -Force -Path $BuildRoot | Out-Null
$GeneratedExecutable = Join-Path $BuildRoot "dist\Interaction Lab\Interaction Lab.exe"
$RunningBuild = Get-Process -Name "Interaction Lab" -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq $GeneratedExecutable }
if ($RunningBuild) { throw "Close the generated Interaction Lab application before rebuilding. Its existing output has not been changed." }
$CompletionRecord = Join-Path $BuildRoot "build-complete.json"
if (Test-Path -LiteralPath $CompletionRecord) { Remove-Item -LiteralPath $CompletionRecord }
$BuildDrive = New-Object IO.DriveInfo ([IO.Path]::GetPathRoot($BuildRoot))
$BuildHeadroom = 4GB
if ($Edition -eq "nvidia") { $BuildHeadroom = 8GB }
if ($BuildDrive.AvailableFreeSpace -lt (8GB + $BuildHeadroom)) { throw "Not enough build-drive space to preserve 8 GiB free through installation, freezing and staging. Use a larger -BuildRoot drive." }
$EnvironmentRoot = Join-Path $BuildRoot "environment"
$EnvironmentPython = Join-Path $EnvironmentRoot "Scripts\python.exe"
& $Python -c "import sys; assert sys.version_info[:2] == (3,12) and sys.maxsize > 2**32, 'Use Windows x64 Python 3.12'"
if ($LASTEXITCODE -ne 0) { throw "Select Windows x64 Python 3.12 with -Python." }
if ($RuntimePython) {
    $EnvironmentPython = [IO.Path]::GetFullPath($RuntimePython)
    $Tools = Join-Path $BuildRoot "tools"
    & $Python -m pip install --no-cache-dir --upgrade --target $Tools -r (Join-Path $ProjectRoot "requirements\build.lock.txt")
    if ($LASTEXITCODE -ne 0) { throw "Could not install isolated release tools." }
    $env:PYTHONPATH = (Join-Path $ProjectRoot "src") + [IO.Path]::PathSeparator + $Tools
} else {
    if (-not (Test-Path -LiteralPath $EnvironmentPython)) {
        & $Python -m venv --without-pip $EnvironmentRoot
        if ($LASTEXITCODE -ne 0) { throw "Could not create the isolated release environment." }
    }
    & $Python -m pip --python $EnvironmentPython install --no-cache-dir -r (Join-Path $ProjectRoot "requirements\runtime-$Edition.lock.txt") -r (Join-Path $ProjectRoot "requirements\build.lock.txt")
    if ($LASTEXITCODE -ne 0) { throw "Exact release dependencies could not be installed." }
    & $Python -m pip --python $EnvironmentPython install --no-deps --no-build-isolation $ProjectRoot
    if ($LASTEXITCODE -ne 0) { throw "Could not install the application into the release environment." }
}
& $Python -m pip --python $EnvironmentPython check
if ($LASTEXITCODE -ne 0) { throw "Release dependency consistency check failed." }
& $EnvironmentPython (Join-Path $PSScriptRoot "verify-runtime-lock.py") --edition $Edition
if ($LASTEXITCODE -ne 0) { throw "The reused runtime differs from its exact release lock. Qualify the pinned edition runtime before building." }
Push-Location (Join-Path $ProjectRoot "frontend")
try {
    & npm.cmd ci
    if ($LASTEXITCODE -ne 0) { throw "Frontend dependency installation failed." }
    & npm.cmd run build
    if ($LASTEXITCODE -ne 0) { throw "Frontend build failed." }
} finally { Pop-Location }
$TorchVersion = "2.13.0+cpu"
if ($Edition -eq "nvidia") { $TorchVersion = "2.13.0+cu130" }
& $EnvironmentPython (Join-Path $PSScriptRoot "qualify-runtime.py") --expected-torch $TorchVersion --device cpu --output (Join-Path $BuildRoot "runtime-qualification.json")
if ($LASTEXITCODE -ne 0) { throw "The pinned runtime failed a real model forward/backward pass." }
$Metadata = Join-Path $BuildRoot "release-metadata"
& $EnvironmentPython (Join-Path $PSScriptRoot "release-metadata.py") --output $Metadata --edition $Edition
if ($LASTEXITCODE -ne 0) { throw "Dependency/license inventory failed." }
$env:ADP_RELEASE_METADATA = $Metadata
$env:ADP_DEMO_CACHE = ""
$env:ADP_MODEL_BUNDLE = ""
if ($DemoCache) {
    $SourceCache = [IO.Path]::GetFullPath($DemoCache)
    if (-not (Test-Path -LiteralPath (Join-Path $SourceCache "manifest.json"))) { throw "Demo cache manifest is missing." }
    $env:ADP_DEMO_CACHE = Join-Path $BuildRoot "demo-cache"
    & $EnvironmentPython (Join-Path $PSScriptRoot "prepare-demo.py") --cache-dir $SourceCache --output-dir $env:ADP_DEMO_CACHE --limit 24
    if ($LASTEXITCODE -ne 0) { throw "Bounded real demonstration preparation failed." }
}
if ($ModelBundle) {
    $env:ADP_MODEL_BUNDLE = [IO.Path]::GetFullPath($ModelBundle)
    & $EnvironmentPython -c "import os; from adp.bundles import load_bundle; load_bundle(os.environ['ADP_MODEL_BUNDLE'])"
    if ($LASTEXITCODE -ne 0) { throw "The bundled inference model did not validate." }
}
$Dist = Join-Path $BuildRoot "dist"
if ($BuildDrive.AvailableFreeSpace -lt (8GB + $BuildHeadroom)) { throw "Not enough remaining space to freeze the application while preserving 8 GiB free." }
& $EnvironmentPython -m PyInstaller --noconfirm --clean --distpath $Dist --workpath (Join-Path $BuildRoot "work") (Join-Path $PSScriptRoot "interaction-lab.spec")
if ($LASTEXITCODE -ne 0) { throw "Frozen application build failed." }
& $EnvironmentPython (Join-Path $PSScriptRoot "release-metadata.py") --output $Metadata --edition $Edition --verify-source
if ($LASTEXITCODE -ne 0) { throw "Source changed during the frozen build. Wait for edits to finish and rebuild." }
$Application = Join-Path $Dist "Interaction Lab"
$SelfTest = Join-Path $BuildRoot "frozen-self-test.json"
if (Test-Path -LiteralPath $SelfTest) { Remove-Item -LiteralPath $SelfTest }
$SelfTestProcess = Start-Process -FilePath (Join-Path $Application "Interaction Lab.exe") -ArgumentList @("--self-test", "--output", "`"$SelfTest`"") -WindowStyle Hidden -PassThru
if (-not $SelfTestProcess.WaitForExit(120000)) { $SelfTestProcess.Kill(); throw "Frozen application self-test timed out." }
if ($SelfTestProcess.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $SelfTest)) { throw "Frozen application self-test failed." }
if (-not (Get-Content -LiteralPath $SelfTest -Raw | ConvertFrom-Json).passed) { throw "Frozen application reported a failed self-test." }
if (-not $SkipInstaller) {
    & (Join-Path $PSScriptRoot "build-installer.ps1") -AppDirectory $Application -OutputDir $Dist -Edition $Edition -Version $Version -Python $EnvironmentPython -Nsis $Nsis
}
$Checksums = Get-ChildItem -LiteralPath $Application -Recurse -File | ForEach-Object {
    $Relative = $_.FullName.Substring($Dist.Length + 1).Replace("\", "/")
    (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant() + "  " + $Relative
}
$Checksums | Set-Content -LiteralPath (Join-Path $Dist "SHA256SUMS") -Encoding utf8
& $EnvironmentPython (Join-Path $PSScriptRoot "release-metadata.py") --output $Metadata --edition $Edition --verify-source
if ($LASTEXITCODE -ne 0) { throw "Source changed during installer creation. Rebuild before distributing these artifacts." }
@{ edition = $Edition; torch = $TorchVersion; source_sha256 = (Get-Content -LiteralPath (Join-Path $Metadata "source.json") -Raw | ConvertFrom-Json).sha256;
   frozen_self_test = (Get-Content -LiteralPath $SelfTest -Raw | ConvertFrom-Json); installer_built = -not $SkipInstaller;
   clean_windows_vm_checked = $false } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $CompletionRecord -Encoding utf8
Write-Host "Frozen application: $Application"
Write-Host "This build is unsigned. Clean Windows install and native GPU qualification remain separate acceptance checks."
