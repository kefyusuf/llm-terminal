@echo off
pushd "%~dp0.."

echo Stopping all Python processes...
taskkill /F /IM python.exe 2>nul
timeout /t 2 /nobreak >nul

echo Deleting cache database...
del /Q data\cache.db 2>nul
if exist data\cache.db (
    echo Warning: Could not delete cache.db - it may still be locked
) else (
    echo Success: cache.db deleted
)

echo Deleting download database...
del /Q data\downloads.db 2>nul
if exist data\downloads.db (
    echo Warning: Could not delete downloads.db - it may still be locked
) else (
    echo Success: downloads.db deleted
)

echo.
echo Cache cleared! You can now start the app with: python app.py

popd
