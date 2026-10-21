@echo off
title Grain Quality Hyperspectral Dashboard
echo =====================================================================
echo Starting Grain Quality Hyperspectral Web Dashboard...
echo Opening http://localhost:8501 in your default browser...
echo =====================================================================

start "" http://localhost:8501
.venv\Scripts\python.exe app.py
pause
