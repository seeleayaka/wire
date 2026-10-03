[CmdletBinding()]
param(
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

$portablePython = Join-Path $root "runtime\windows\python\python.exe"
$portableMainSite = Join-Path $root "runtime\windows\main_site_packages"
$portableSam3Site = Join-Path $root "runtime\windows\sam3_site_packages"
$portableSam3Base = Join-Path $root "runtime\windows\sam3_base_site_packages"
$sam3Source = Join-Path $root "runtime\sam3\source"
if ((Test-Path -LiteralPath $portablePython) -and (Test-Path -LiteralPath $portableMainSite) -and (Test-Path -LiteralPath $portableSam3Site) -and (Test-Path -LiteralPath $portableSam3Base)) {
    $env:PYTHONHOME = Split-Path -Parent $portablePython
    $env:PYTHONPATH = "$portableMainSite;$root\models\dinov2;$root\prototype"
    & $portablePython -B -c "import cv2, numpy, torch; from PyQt5 import QtCore; print('bundled main runtime ready')"
    if ($LASTEXITCODE -ne 0) { throw "The bundled Windows main runtime could not be imported." }
    $env:PYTHONPATH = "$sam3Source;$portableSam3Site;$portableSam3Base"
    & $portablePython -B -c "import cv2, numpy, torch, sam3; print('bundled SAM3 runtime ready')"
    if ($LASTEXITCODE -ne 0) { throw "The bundled Windows SAM3 runtime could not be imported." }
    Write-Host "Bundled Windows runtime verified. No Python installation or package download is needed."
    return
}

$systemPython = (Get-Command python -ErrorAction Stop).Source
$version = & $systemPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($version -ne "3.11") {
    throw "Python 3.11 is required; found $version at $systemPython"
}

function Ensure-Venv([string]$venvPath, [string]$requirementsPath) {
    $venvPython = Join-Path $venvPath "Scripts\python.exe"
    if ($Recreate -and (Test-Path -LiteralPath $venvPath)) {
        Remove-Item -LiteralPath $venvPath -Recurse -Force
    }
    if (-not (Test-Path -LiteralPath $venvPython)) {
        if (Test-Path -LiteralPath $venvPath) {
            throw "Existing non-portable environment at $venvPath. Re-run with -Recreate to replace it."
        }
        & $systemPython -m venv $venvPath
    }
    & $venvPython -m pip install --upgrade pip
    & $venvPython -m pip install -r $requirementsPath
}

$mainVenv = Join-Path $root ".venv"
$mainRequirements = Join-Path $root "runtime\requirements-main.txt"
Ensure-Venv $mainVenv $mainRequirements

$sam3Venv = Join-Path $root "runtime\sam3\.venv"
$sam3Requirements = Join-Path $root "runtime\sam3\requirements.txt"
Ensure-Venv $sam3Venv $sam3Requirements
$sam3Python = Join-Path $sam3Venv "Scripts\python.exe"
& $sam3Python -m pip install --no-deps -e $sam3Source

& $mainVenv\Scripts\python.exe -B -c "import cv2, numpy, torch; from PyQt5 import QtCore; print('main runtime ready')"
$env:PYTHONPATH = $sam3Source
& $sam3Python -B -c "import cv2, numpy, torch, sam3; print('sam3 runtime ready')"
Write-Host "Runtime setup complete. Start with: $root\启动装配自动定位复核_DINO融合版.bat"
