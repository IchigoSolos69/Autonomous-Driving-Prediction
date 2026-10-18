param(
    [Parameter(Mandatory=$true)][string]$AppDirectory,
    [Parameter(Mandatory=$true)][string]$OutputDir,
    [ValidateSet("cpu", "nvidia")][string]$Edition = "cpu",
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9.-]{0,63}$')][string]$Version = "0.1.0-beta",
    [string]$Python = "python",
    [string]$Nsis = ""
)
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$AppDirectory = [IO.Path]::GetFullPath($AppDirectory)
$OutputDir = [IO.Path]::GetFullPath($OutputDir)
if (-not (Test-Path -LiteralPath (Join-Path $AppDirectory "Interaction Lab.exe"))) { throw "Build the frozen application first." }
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
if (-not $Nsis) {
    $Candidate = Get-Command makensis.exe -ErrorAction SilentlyContinue
    if ($Candidate) { $Nsis = $Candidate.Source } else { $Nsis = "${env:ProgramFiles(x86)}\NSIS\makensis.exe" }
}
if (-not (Test-Path -LiteralPath $Nsis)) { throw "Install NSIS 3.13 or pass -Nsis C:\path\to\makensis.exe." }
$CompilerVersion = & $Nsis /VERSION
if ($CompilerVersion -notmatch 'v?3\.(1[3-9]|[2-9][0-9])') { throw "Use the patched NSIS 3.13 or newer compiler." }
$Bootstrapper = Join-Path $OutputDir "MicrosoftEdgeWebview2Setup.exe"
Invoke-WebRequest -Uri "https://go.microsoft.com/fwlink/p/?LinkId=2124703" -OutFile $Bootstrapper
$Signature = Get-AuthenticodeSignature -LiteralPath $Bootstrapper
if ($Signature.Status -ne "Valid" -or $Signature.SignerCertificate.Subject -notmatch "Microsoft Corporation") { throw "Microsoft WebView2 bootstrapper signature is invalid." }
$Installer = Join-Path $OutputDir "Interaction-Lab-$Version-$Edition-Setup.exe"
$CompilerArguments = @("/DAPP_VERSION=$Version", "/DAPP_EDITION=$Edition", "/DAPP_DIRECTORY=$AppDirectory", "/DOUTPUT_FILE=$Installer", "/DWEBVIEW_BOOTSTRAPPER=$Bootstrapper")
$PayloadRecord = $null
$ApplicationSize = (Get-ChildItem -LiteralPath $AppDirectory -Recurse -File | Measure-Object -Property Length -Sum).Sum
$OutputDrive = New-Object IO.DriveInfo ([IO.Path]::GetPathRoot($OutputDir))
if ($OutputDrive.AvailableFreeSpace -lt (8GB + 2 * $ApplicationSize)) { throw "Installer staging needs more disk space to preserve 8 GiB free. Use another output drive." }
if ($ApplicationSize -gt 1.5GB) {
    $Payload = Join-Path $OutputDir "Interaction-Lab-$Version-$Edition-payload.zip"
    & $Python (Join-Path $PSScriptRoot "archive-app.py") --app-dir $AppDirectory --output $Payload
    if ($LASTEXITCODE -ne 0) { throw "Application payload creation failed." }
    $PayloadHash = (Get-FileHash -LiteralPath $Payload -Algorithm SHA256).Hash.ToLowerInvariant()
    $CompilerArguments += @("/DAPP_ARCHIVE_NAME=$([IO.Path]::GetFileName($Payload))", "/DAPP_ARCHIVE_SHA256=$PayloadHash", "/DPAYLOAD_SCRIPT=$(Join-Path $PSScriptRoot 'install-payload.ps1')")
    "$PayloadHash  $([IO.Path]::GetFileName($Payload))" | Set-Content -LiteralPath "$Payload.sha256" -Encoding utf8
    $PayloadRecord = @{ name = [IO.Path]::GetFileName($Payload); sha256 = $PayloadHash; size_bytes = (Get-Item -LiteralPath $Payload).Length }
    Write-Host "Distribute the setup executable AND its adjacent payload ZIP together."
}
$CompilerTemp = Join-Path $OutputDir ".compiler-temp"
New-Item -ItemType Directory -Force -Path $CompilerTemp | Out-Null
$PreviousTemp, $PreviousTmp = $env:TEMP, $env:TMP
try {
    $env:TEMP, $env:TMP = $CompilerTemp, $CompilerTemp
    & $Nsis @CompilerArguments (Join-Path $PSScriptRoot "installer.nsi")
    if ($LASTEXITCODE -ne 0) { throw "Installer compilation failed." }
} finally { $env:TEMP, $env:TMP = $PreviousTemp, $PreviousTmp }
$InstallerHash = (Get-FileHash -LiteralPath $Installer -Algorithm SHA256).Hash.ToLowerInvariant()
"$InstallerHash  $([IO.Path]::GetFileName($Installer))" | Set-Content -LiteralPath "$Installer.sha256" -Encoding utf8
$SourceRecord = Join-Path $AppDirectory "_internal\adp\resources\release\source.json"
$SourceId = $null
if (Test-Path -LiteralPath $SourceRecord) { $SourceId = (Get-Content -LiteralPath $SourceRecord -Raw | ConvertFrom-Json).sha256 }
@{ edition = $Edition; version = $Version; compiler = $CompilerVersion; source_sha256 = $SourceId;
   installer = @{ name = [IO.Path]::GetFileName($Installer); sha256 = $InstallerHash; size_bytes = (Get-Item -LiteralPath $Installer).Length };
   payload = $PayloadRecord; application_uncompressed_bytes = $ApplicationSize;
   webview_bootstrapper_sha256 = (Get-FileHash -LiteralPath $Bootstrapper -Algorithm SHA256).Hash.ToLowerInvariant();
   signed = $false } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $OutputDir "installer-build.json") -Encoding utf8
Write-Host "Installer: $Installer"
