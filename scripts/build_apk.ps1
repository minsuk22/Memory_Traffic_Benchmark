# Builds app\build\outputs\apk\release\app-release.apk (installs it with -Install).
param([switch]$Install)
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "env.ps1")
$ProjectDir = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectDir
try {
    & .\gradlew.bat assembleRelease --console=plain
    if ($LASTEXITCODE -ne 0) { throw "gradle build failed" }
    $apk = Join-Path $ProjectDir "app\build\outputs\apk\release\app-release.apk"
    Write-Host "Built: $apk"
    if ($Install) {
        adb install -r $apk
        Write-Host "Run: adb shell am start -n com.memtraffic.bench/.MainActivity --es args `"--size-mb 1024`""
        Write-Host "Log: adb logcat -s membench"
    }
} finally { Pop-Location }
