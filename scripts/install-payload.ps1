param(
    [Parameter(Mandatory=$true)][string]$Archive,
    [Parameter(Mandatory=$true)][ValidatePattern('^[a-fA-F0-9]{64}$')][string]$ExpectedSha,
    [Parameter(Mandatory=$true)][string]$Destination
)
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
Add-Type -AssemblyName System.IO.Compression.FileSystem
$Archive = [IO.Path]::GetFullPath($Archive)
$Destination = [IO.Path]::GetFullPath($Destination).TrimEnd([IO.Path]::DirectorySeparatorChar)
if ($Destination -eq [IO.Path]::GetPathRoot($Destination).TrimEnd([IO.Path]::DirectorySeparatorChar)) { throw "Choose an application folder, not a drive root." }
if (-not (Test-Path -LiteralPath $Archive -PathType Leaf)) { throw "Keep the NVIDIA payload ZIP beside its matching setup executable." }
$HashStream = [IO.File]::OpenRead($Archive)
$Hasher = [Security.Cryptography.SHA256]::Create()
try { $ActualSha = [BitConverter]::ToString($Hasher.ComputeHash($HashStream)).Replace("-", "") } finally { $HashStream.Dispose(); $Hasher.Dispose() }
if ($ActualSha -ne $ExpectedSha) { throw "The NVIDIA payload SHA-256 does not match this installer. Download both files from the same release." }
New-Item -ItemType Directory -Force -Path $Destination | Out-Null
$DestinationPrefix = $Destination + [IO.Path]::DirectorySeparatorChar
$Stage = Join-Path $Destination (".interaction-install-" + [guid]::NewGuid().ToString("N"))
$Backup = Join-Path $Destination (".interaction-previous-" + [guid]::NewGuid().ToString("N"))
$Moved = @()
$Promoted = @()
$ArchiveFile = $null
$Committed = $false
function Assert-WithinInstallation([string]$Candidate) {
    $Resolved = [IO.Path]::GetFullPath($Candidate)
    if (-not $Resolved.StartsWith($DestinationPrefix, [StringComparison]::OrdinalIgnoreCase)) { throw "Unsafe installation target." }
    return $Resolved
}
try {
    $ArchiveFile = [IO.Compression.ZipFile]::OpenRead($Archive)
    $Total = [long]0
    if ($ArchiveFile.Entries.Count -gt 20000) { throw "Payload contains too many entries." }
    foreach ($Entry in $ArchiveFile.Entries) {
        $Name = $Entry.FullName.Replace("/", "\")
        if ($Name -notmatch '^(Interaction Lab\.exe|_internal\\.+)$' -or $Name -match '(^|\\)\.\.(\\|$)|:' -or (($Entry.ExternalAttributes -shr 16) -band 0xF000) -eq 0xA000) { throw "Unsafe application payload path." }
        $Target = Assert-WithinInstallation (Join-Path $Stage $Name)
        $Total += $Entry.Length
        if ($Entry.Length -gt 2GB -or $Total -gt 16GB) { throw "Application payload exceeds the supported size." }
    }
    $Drive = New-Object IO.DriveInfo ([IO.Path]::GetPathRoot($Destination))
    if ($Drive.AvailableFreeSpace -lt ($Total + 512MB)) { throw ("Not enough free disk space. The new application needs " + [math]::Ceiling(($Total + 512MB) / 1GB) + " GiB available on " + $Drive.Name + ". Choose another installation drive or free space.") }
    New-Item -ItemType Directory -Path $Stage | Out-Null
    foreach ($Entry in $ArchiveFile.Entries) {
        $Target = Assert-WithinInstallation (Join-Path $Stage $Entry.FullName.Replace("/", "\"))
        if ($Entry.FullName.EndsWith("/")) { New-Item -ItemType Directory -Force -Path $Target | Out-Null; continue }
        New-Item -ItemType Directory -Force -Path ([IO.Path]::GetDirectoryName($Target)) | Out-Null
        [IO.Compression.ZipFileExtensions]::ExtractToFile($Entry, $Target, $false)
    }
    if (-not (Test-Path -LiteralPath (Join-Path $Stage "Interaction Lab.exe")) -or -not (Test-Path -LiteralPath (Join-Path $Stage "_internal"))) { throw "Incomplete application payload." }
    New-Item -ItemType Directory -Path $Backup | Out-Null
    foreach ($Name in @("_internal", "Interaction Lab.exe")) {
        $Active = Assert-WithinInstallation (Join-Path $Destination $Name)
        if (Test-Path -LiteralPath $Active) { Move-Item -LiteralPath $Active -Destination (Join-Path $Backup $Name); $Moved += $Name }
        Move-Item -LiteralPath (Join-Path $Stage $Name) -Destination $Active
        $Promoted += $Name
    }
    $Committed = $true
    Write-Output "Verified application payload installed. User datasets, models and settings were preserved."
} catch {
    foreach ($Name in $Promoted) {
        $Target = Assert-WithinInstallation (Join-Path $Destination $Name)
        if (Test-Path -LiteralPath $Target) { Remove-Item -LiteralPath $Target -Recurse -Force }
    }
    foreach ($Name in $Moved) {
        Move-Item -LiteralPath (Assert-WithinInstallation (Join-Path $Backup $Name)) -Destination (Assert-WithinInstallation (Join-Path $Destination $Name))
    }
    throw
} finally {
    if ($ArchiveFile) { $ArchiveFile.Dispose() }
    if (Test-Path -LiteralPath $Stage) { Remove-Item -LiteralPath (Assert-WithinInstallation $Stage) -Recurse -Force }
    if ($Committed -and (Test-Path -LiteralPath $Backup)) { Remove-Item -LiteralPath (Assert-WithinInstallation $Backup) -Recurse -Force -ErrorAction Continue }
}
