param(
    [string]$Pythonw = 'D:\InteractionLab\runtime-cu130\Scripts\pythonw.exe',
    [string]$StateDir = 'D:\InteractionLab\remote-compute\monitor',
    [switch]$RemoveStartup
)

$ErrorActionPreference = 'Stop'
$project = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$entry = Join-Path $project 'scripts\start-compute-monitor.py'
$startup = Join-Path ([Environment]::GetFolderPath('Startup')) 'Interaction Lab Compute Monitor.lnk'
$desktop = Join-Path ([Environment]::GetFolderPath('Desktop')) 'Interaction Lab Compute Monitor.lnk'
$shell = New-Object -ComObject WScript.Shell

if ($RemoveStartup) {
    if (Test-Path -LiteralPath $startup) {
        $link = $shell.CreateShortcut($startup)
        if ($link.Arguments -notlike ('*' + $entry + '*')) {
            throw 'The startup shortcut belongs to another program; leaving it unchanged.'
        }
        Remove-Item -LiteralPath $startup
    }
    Write-Output 'Automatic sign-in launch removed. The running monitor and its data are preserved.'
    return
}

$pythonPath = (Resolve-Path -LiteralPath $Pythonw).Path
if ([IO.Path]::GetFileName($pythonPath) -ne 'pythonw.exe' -or -not (Test-Path -LiteralPath $entry -PathType Leaf)) {
    throw 'Use the qualified pythonw.exe and restore scripts\start-compute-monitor.py before installing.'
}
foreach ($value in @($entry, $StateDir)) {
    if ($value.Contains('"') -or $value.Contains("`r") -or $value.Contains("`n")) {
        throw 'Shortcut paths cannot contain quotation marks or newlines.'
    }
}
$arguments = '"' + $entry + '" --state-dir "' + $StateDir.TrimEnd('\') + '"'
foreach ($path in @($startup, $desktop)) {
    if (Test-Path -LiteralPath $path) {
        $existing = $shell.CreateShortcut($path)
        if ($existing.Arguments -notlike ('*' + $entry + '*')) {
            throw ('The shortcut belongs to another program; leaving it unchanged: ' + $path)
        }
    }
}
foreach ($path in @($startup, $desktop)) {
    $shortcutArguments = if ($path -eq $desktop) { $arguments + ' --show-status' } else { $arguments }
    $link = $shell.CreateShortcut($path)
    $link.TargetPath = $pythonPath
    $link.Arguments = $shortcutArguments
    $link.WorkingDirectory = $project
    $link.Description = 'Independent Interaction Lab compute checks, transfers and guarded cleanup every five minutes'
    $link.Save()
    $saved = $shell.CreateShortcut($path)
    if ($saved.TargetPath -ne $pythonPath -or $saved.Arguments -ne $shortcutArguments) {
        throw ('Shortcut verification failed: ' + $path)
    }
    Write-Output ('Created: ' + $path)
}
Write-Output 'The desktop shortcut starts the monitor and shows its status. Sign-in startup stays quiet.'
