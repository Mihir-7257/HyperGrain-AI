@echo off
title Grain Quality Hyperspectral Pipeline
echo =====================================================================
echo Running Complete Grain Quality Training and Heatmap Pipeline...
echo =====================================================================

.venv\Scripts\python.exe main.py

echo.
echo =====================================================================
echo Pipeline finished! All plots and heatmaps are saved in the outputs/ folder.
echo =====================================================================
pause
