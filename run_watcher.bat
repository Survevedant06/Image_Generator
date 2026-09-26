@echo off
title BiRefNet Automated Image Watcher
echo =========================================================================
echo BiRefNet Automated Background Removal Watcher
echo =========================================================================
echo Drop any new images into 'inputs/' or this folder.
echo The watcher will detect and process them automatically in the background.
echo Press Ctrl+C in this window at any time to stop.
echo =========================================================================
echo.
python batch_remove_bg.py --watch
pause
