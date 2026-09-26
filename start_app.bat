@echo off
title E-Commerce AI Studio Dashboard
echo =========================================================================
echo Starting E-Commerce AI Studio Web App
echo =========================================================================
echo Dashboard URL: http://localhost:8000
echo API Docs     : http://localhost:8000/docs
echo =========================================================================
echo.

:: Open the default browser after 2 seconds
start "" http://localhost:8000

:: Start the FastAPI server
python server.py
pause
