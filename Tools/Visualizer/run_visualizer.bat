@echo off
title EdgeSense VL53L8CH Real-Time Visualizer
cd /d "%~dp0"
echo =======================================================
echo   Starting EdgeSense 3D ToF & CNH Histogram Visualizer
echo =======================================================
python edgesense_visualizer.py
pause
