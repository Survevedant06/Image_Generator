@echo off
title Automated Batch Background Removal
echo =======================================================
echo Running Automated Batch Background Removal (BiRefNet)
echo =======================================================
python batch_remove_bg.py
echo.
echo Process complete. Press any key to exit.
pause >nul
