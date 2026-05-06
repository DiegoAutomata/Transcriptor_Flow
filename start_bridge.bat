@echo off
title Transcriptor Flow Bridge
REM ============================================================
REM Transcriptor Flow - Bridge Win32
REM Hold Ctrl+Alt to dictate in WSL apps
REM ============================================================

set BRIDGE_DIR=\\wsl$\Ubuntu\home\diego\Transcriptor-Flow

if not exist "%BRIDGE_DIR%" (
    echo Project not found at: %BRIDGE_DIR%
    echo Verify your WSL distro is named "Ubuntu"
    pause
    exit /b 1
)

echo ============================================================
echo   Transcriptor Flow Bridge
echo   Hold Ctrl+Alt to dictate in WSL apps
echo ============================================================
echo.

powershell.exe -ExecutionPolicy Bypass -File "%BRIDGE_DIR%\start_bridge.ps1"

if errorlevel 1 (
    echo PowerShell failed, trying Python...

    set PYTHON=
    for %%p in (
        "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
        "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
        "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
        "%PROGRAMFILES%\Python312\python.exe"
        "%PROGRAMFILES%\Python311\python.exe"
        "C:\Python312\python.exe"
        "C:\Python311\python.exe"
    ) do (
        if not defined PYTHON (
            if exist "%%p" set PYTHON=%%p
        )
    )

    if not defined PYTHON (
        where python.exe >nul 2>&1 && set PYTHON=python.exe
    )

    if defined PYTHON (
        "%PYTHON%" "%BRIDGE_DIR%\src\bridge_win32.py"
    ) else (
        echo Python not found on Windows.
    )
)

pause
