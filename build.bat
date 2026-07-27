@echo off
REM ---------------------------------------------------------------------------
REM Build FRETBursts' Cython extensions on Windows and sync the uv environment.
REM
REM Why this script exists:
REM   setuptools/distutils cannot auto-detect Visual Studio 2026 (v18). It runs
REM   `cmd /u /c vcvarsall.bat && set` and decodes the output as UTF-16LE, but
REM   VS 2026's vcvarsall emits ANSI, so the parse yields zero environment
REM   variables and the build fails with:
REM       "Unable to find a compatible Visual Studio installation."
REM   The cl.exe compiler itself works fine. The fix is to initialize the MSVC
REM   environment ourselves with vcvarsall and set DISTUTILS_USE_SDK=1, which
REM   makes setuptools skip its (broken) detection and use the current env.
REM
REM Usage:  build.bat            (defaults to `uv sync`)
REM         build.bat <args...>  (runs `uv <args...>` instead, e.g. run pytest)
REM ---------------------------------------------------------------------------
setlocal

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo [build] vswhere.exe not found at "%VSWHERE%".
    echo [build] Install Visual Studio with the "Desktop development with C++" workload.
    exit /b 1
)

for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -prerelease -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSPATH=%%i"
if not defined VSPATH (
    echo [build] No Visual Studio install with the VC C++ toolset was found.
    exit /b 1
)

set "VCVARS=%VSPATH%\VC\Auxiliary\Build\vcvarsall.bat"
if not exist "%VCVARS%" (
    echo [build] vcvarsall.bat not found at "%VCVARS%".
    exit /b 1
)

echo [build] Using Visual Studio at: %VSPATH%
call "%VCVARS%" x64
if errorlevel 1 (
    echo [build] vcvarsall.bat failed.
    exit /b 1
)

REM Make setuptools skip its broken VS detection and use the env cl.exe above.
set "DISTUTILS_USE_SDK=1"
set "MSSdk=1"

cd /d "%~dp0"

if "%~1"=="" (
    echo [build] Running: uv sync
    uv sync
) else (
    echo [build] Running: uv %*
    uv %*
)

exit /b %ERRORLEVEL%
