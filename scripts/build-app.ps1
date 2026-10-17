param([switch]$SkipFrontend)

$ErrorActionPreference = "Stop"
$ProjectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
if (-not $SkipFrontend) {
    Push-Location (Join-Path $ProjectRoot "frontend")
    try {
        if (-not (Test-Path -LiteralPath "node_modules")) {
            & npm.cmd ci
            if ($LASTEXITCODE -ne 0) { throw "Frontend dependencies could not be installed." }
        }
        & npm.cmd run build
        if ($LASTEXITCODE -ne 0) { throw "The application interface could not be built." }
    } finally { Pop-Location }
}
if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot "frontend\dist\index.html"))) {
    throw "Build the interface first: npm.cmd run build in frontend."
}
$Compiler = Join-Path $env:WINDIR "Microsoft.NET\Framework64\v4.0.30319\csc.exe"
if (-not (Test-Path -LiteralPath $Compiler)) { throw "The Windows .NET Framework C# compiler was not found." }
$Executable = Join-Path $ProjectRoot "Interaction Lab.exe"
& $Compiler /nologo /target:winexe /reference:System.Windows.Forms.dll "/out:$Executable" (Join-Path $PSScriptRoot "AppLauncher.cs")
if ($LASTEXITCODE -ne 0) { throw "The Windows launcher could not be compiled. Close Interaction Lab before rebuilding if the executable is in use." }
Write-Host "Ready: $Executable"
Write-Host "Double-click it to open the desktop application. Keep it in this project beside .venv, frontend, data and artifacts."
