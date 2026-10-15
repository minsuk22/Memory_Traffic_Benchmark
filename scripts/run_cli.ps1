# Pushes the CLI binary to the connected device and runs it. Extra arguments are passed
# to membench, e.g.:
#   scripts\run_cli.ps1 --size-mb 2048 --sweep --peak-gbps 76.8
# The log is also saved to results\<device>_<timestamp>.txt
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "env.ps1")
$ProjectDir = Split-Path -Parent $PSScriptRoot
$Bin = Join-Path $ProjectDir "build\android-cli\membench"
if (-not (Test-Path $Bin)) { throw "build first: scripts\build_cli.ps1" }

$devices = (adb devices) | Select-String "\tdevice$"
if (-not $devices) { throw "no device: enable USB debugging and check 'adb devices'" }

adb push $Bin /data/local/tmp/membench | Out-Null
adb shell chmod 755 /data/local/tmp/membench

$model = ((adb shell getprop ro.product.model) -join "").Trim() -replace "[^A-Za-z0-9_-]", "_"
$soc   = ((adb shell getprop ro.soc.model) -join "").Trim()
$out   = Join-Path $ProjectDir ("results\{0}_{1}.txt" -f $model, (Get-Date -Format "yyyyMMdd_HHmmss"))
New-Item -ItemType Directory -Force (Split-Path $out) | Out-Null

$argLine = ($args | ForEach-Object { "$_" }) -join " "
"device: $model  soc: $soc  args: $argLine" | Tee-Object -FilePath $out
adb shell "cd /data/local/tmp && ./membench $argLine; echo exit code `$?" | Tee-Object -FilePath $out -Append
Write-Host "saved: $out"
