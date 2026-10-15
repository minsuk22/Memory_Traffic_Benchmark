# Installs a self-contained Android native toolchain (no admin rights needed) and prepares
# this project for building. Components that already exist are skipped.
#
#   powershell -ExecutionPolicy Bypass -File scripts\setup_android_env.ps1 [-Root D:\Android]
#
# Installs: Temurin JDK 17, Android cmdline-tools, platform-tools (adb), platform 34,
#           build-tools 34, NDK 26.3, CMake 3.22.1 (with ninja), Gradle 8.7 distribution.
param(
    [string]$Root = "D:\Android"
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest is very slow with progress bars
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$ProjectDir = Split-Path -Parent $PSScriptRoot
$Sdk        = Join-Path $Root "sdk"
$Downloads  = Join-Path $Root "downloads"
$NdkVersion = "26.3.11579264"
$Packages   = @("platform-tools", "platforms;android-34", "build-tools;34.0.0",
                "ndk;$NdkVersion", "cmake;3.22.1")

New-Item -ItemType Directory -Force $Root, $Sdk, $Downloads | Out-Null

function Get-File($Url, $Dest) {
    if (-not (Test-Path $Dest)) {
        Write-Host "  downloading $Url"
        Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile "$Dest.part"
        Move-Item "$Dest.part" $Dest
    }
}

# --- JDK 17 ---------------------------------------------------------------------------
$jdk = Get-ChildItem (Join-Path $Root "jdk") -Directory -ErrorAction SilentlyContinue |
       Where-Object { Test-Path (Join-Path $_.FullName "bin\java.exe") } | Select-Object -First 1
if (-not $jdk) {
    Write-Host "[1/5] JDK 17"
    $zip = Join-Path $Downloads "jdk17.zip"
    Get-File "https://api.adoptium.net/v3/binary/latest/17/ga/windows/x64/jdk/hotspot/normal/eclipse" $zip
    Expand-Archive $zip (Join-Path $Root "jdk") -Force
    $jdk = Get-ChildItem (Join-Path $Root "jdk") -Directory | Select-Object -First 1
} else { Write-Host "[1/5] JDK present: $($jdk.FullName)" }
$env:JAVA_HOME = $jdk.FullName

# --- cmdline-tools --------------------------------------------------------------------
$sdkmanager = Join-Path $Sdk "cmdline-tools\latest\bin\sdkmanager.bat"
if (-not (Test-Path $sdkmanager)) {
    Write-Host "[2/5] Android cmdline-tools"
    $zip = Join-Path $Downloads "cmdline-tools.zip"
    Get-File "https://dl.google.com/android/repository/commandlinetools-win-13114758_latest.zip" $zip
    $tmp = Join-Path $Sdk "cmdline-tools\_tmp"
    Expand-Archive $zip $tmp -Force
    Move-Item (Join-Path $tmp "cmdline-tools") (Join-Path $Sdk "cmdline-tools\latest")
    Remove-Item $tmp -Recurse -Force
} else { Write-Host "[2/5] cmdline-tools present" }

# --- SDK packages ---------------------------------------------------------------------
$missing = @()
foreach ($p in $Packages) {
    if (-not (Test-Path (Join-Path $Sdk ($p -replace ";", "\")))) { $missing += $p }
}
if ($missing.Count -gt 0) {
    Write-Host "[3/5] sdkmanager: $($missing -join ', ')"
    $yes = ("y`n" * 30)
    $yes | & $sdkmanager "--sdk_root=$Sdk" --licenses | Out-Null
    & $sdkmanager "--sdk_root=$Sdk" @missing
    if ($LASTEXITCODE -ne 0) { throw "sdkmanager failed" }
} else { Write-Host "[3/5] SDK packages present" }

# --- Gradle distribution (only used once to create the wrapper) -----------------------
$gradle = Join-Path $Root "gradle-dist\gradle-8.7\bin\gradle.bat"
if (-not (Test-Path $gradle)) {
    Write-Host "[4/5] Gradle 8.7"
    $zip = Join-Path $Downloads "gradle.zip"
    Get-File "https://services.gradle.org/distributions/gradle-8.7-bin.zip" $zip
    Expand-Archive $zip (Join-Path $Root "gradle-dist") -Force
} else { Write-Host "[4/5] Gradle present" }

# --- project files --------------------------------------------------------------------
Write-Host "[5/5] project setup"
$sdkEsc = ($Sdk -replace "\\", "\\")
Set-Content -Encoding ascii (Join-Path $ProjectDir "local.properties") "sdk.dir=$sdkEsc"
if (-not (Test-Path (Join-Path $ProjectDir "gradlew.bat"))) {
    Push-Location $ProjectDir
    & $gradle wrapper --gradle-version 8.7 --no-daemon -q
    Pop-Location
}

# Environment file used by the other scripts
@"
`$env:JAVA_HOME        = "$($jdk.FullName)"
`$env:ANDROID_HOME     = "$Sdk"
`$env:ANDROID_SDK_ROOT = "$Sdk"
`$env:ANDROID_NDK_HOME = "$Sdk\ndk\$NdkVersion"
`$env:Path = "`$env:JAVA_HOME\bin;$Sdk\platform-tools;$Sdk\cmake\3.22.1\bin;" + `$env:Path
"@ | Set-Content -Encoding ascii (Join-Path $PSScriptRoot "env.ps1")

Write-Host ""
Write-Host "Done. Toolchain in $Root"
Write-Host "  CLI build : powershell -ExecutionPolicy Bypass -File scripts\build_cli.ps1"
Write-Host "  APK build : powershell -ExecutionPolicy Bypass -File scripts\build_apk.ps1"
