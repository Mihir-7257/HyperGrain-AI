@echo off
title Push HyperGrain-AI to GitHub
echo =====================================================================
echo Pushing HyperGrain-AI to https://github.com/Mihir-7257/HyperGrain-AI.git
echo =====================================================================
echo.

set GIT_EXE=C:\Users\ASUS\AppData\Local\Programs\MinGit\cmd\git.exe
if not exist "%GIT_EXE%" set GIT_EXE=git

"%GIT_EXE%" push -u origin main

echo.
echo =====================================================================
if %ERRORLEVEL% EQU 0 (
    echo Successfully pushed HyperGrain-AI to GitHub!
    echo Visit: https://github.com/Mihir-7257/HyperGrain-AI
) else (
    echo Push encountered an authentication step.
    echo If prompted above, enter your GitHub Username and Personal Access Token (PAT).
)
echo =====================================================================
pause
