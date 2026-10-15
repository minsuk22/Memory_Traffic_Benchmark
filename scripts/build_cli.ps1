# Builds the standalone arm64 executable build\android-cli\membench (NDK + CMake + Ninja).
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "env.ps1")
$ProjectDir = Split-Path -Parent $PSScriptRoot
$Build = Join-Path $ProjectDir "build\android-cli"
$Ninja = Join-Path $env:ANDROID_HOME "cmake\3.22.1\bin\ninja.exe"

cmake -S (Join-Path $ProjectDir "native") -B $Build -G Ninja `
    "-DCMAKE_MAKE_PROGRAM=$Ninja" `
    "-DCMAKE_TOOLCHAIN_FILE=$env:ANDROID_NDK_HOME\build\cmake\android.toolchain.cmake" `
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DANDROID_STL=c++_static `
    -DCMAKE_BUILD_TYPE=Release
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }
cmake --build $Build
if ($LASTEXITCODE -ne 0) { throw "build failed" }
Write-Host "Built: $Build\membench"
Write-Host "SPIR-V disassembly for inspection: $Build\generated\*.spvasm"
