@echo off
cd /d %~dp0

echo Copying pictures for build...
xcopy pictures source\pictures\ /s /i /y /q

echo Building Sphinx documentation...
sphinx-build -b html source build

echo Cleaning up copied pictures...
rd /s /q source\pictures

echo Build complete. The HTML pages are in the 'build' directory.
