@echo off
REM Angelo - install the OPTIONAL SAM 3 "Detect" feature (Windows).
REM
REM CLOSE ComfyUI before running this (installing into a running env can
REM fail on locked/loaded packages), then start it again when done.
REM
REM Angelo's core needs no extra dependencies; this is only for the SAM 3
REM text-segmentation Detect button. It tries to find your ComfyUI Python
REM automatically (recorded by the node, portable embedded, or a venv
REM beside ComfyUI). If it picks the wrong one, set PYTHON yourself, e.g.:
REM    set "PYTHON=C:\ComfyUI\python_embeded\python.exe"
REM    install_sam3_support.bat

setlocal
cd /d "%~dp0"

echo Angelo SAM 3 installer - make sure ComfyUI is CLOSED before continuing.

REM Pick a Python in priority order: PYTHON env var, then the interpreter
REM ComfyUI recorded on its last start (.comfy_python.txt - the reliable
REM one for any launcher), then portable embedded / venv beside ComfyUI,
REM then `python` on PATH as a last resort.
set "PY="
if defined PYTHON set "PY=%PYTHON%"
if not defined PY if exist ".comfy_python.txt" set /p PY=<".comfy_python.txt"
if not defined PY if exist "..\..\..\python_embeded\python.exe" set "PY=..\..\..\python_embeded\python.exe"
if not defined PY if exist "..\..\venv\Scripts\python.exe" set "PY=..\..\venv\Scripts\python.exe"
if not defined PY if exist "..\..\.venv\Scripts\python.exe" set "PY=..\..\.venv\Scripts\python.exe"
if not defined PY set "PY=python"
if not exist ".comfy_python.txt" echo NOTE: start ComfyUI once so Angelo can record its Python, for the most reliable install.

echo Using Python: %PY%
"%PY%" --version
if errorlevel 1 (
  echo Python not found. Set PYTHON to your ComfyUI python.exe and retry.
  goto :fail
)

REM Pin numpy to whatever this env already has, so nothing installed below
REM can downgrade it (a pinned numpy in the deps used to drag ComfyUI's
REM numpy 2.x down to 1.26 and break it - issue #31). An empty constraints
REM file is harmless if numpy somehow isn't present yet.
set "CONSTRAINTS=%TEMP%\angelo_sam3_constraints.txt"
type nul > "%CONSTRAINTS%"
"%PY%" -c "import numpy; v=numpy.__version__; open(r'%CONSTRAINTS%','a').write('numpy>='+v+'\n'); print('Pinning numpy>='+v+' so the SAM 3 deps cannot downgrade it.')" 2>nul

"%PY%" -c "import sam3" 1>nul 2>nul
if not errorlevel 1 (
  echo SAM 3 is already installed in this environment.
  echo Applying Angelo's SAM 3 compatibility fixes if needed...
  "%PY%" "sam3_postinstall.py"
  echo Restart ComfyUI and use the Detect button in Angelo.
  goto :done
)

echo Installing SAM 3 runtime dependencies...
"%PY%" -m pip install -r "sam3_requirements.txt" -c "%CONSTRAINTS%"
if errorlevel 1 (
  echo Dependency install failed.
  goto :fail
)

REM OpenCV (cv2) - only install if it isn't already importable, so we don't
REM clobber a full opencv-python another node already provides. Headless
REM avoids pulling GUI/Qt deps we don't need for the image path.
"%PY%" -c "import cv2" 1>nul 2>nul
if errorlevel 1 (
  echo Installing opencv-python-headless ^(cv2 not found^)...
  "%PY%" -m pip install "opencv-python-headless>=4.8.0" -c "%CONSTRAINTS%"
  if errorlevel 1 (
    echo OpenCV install failed.
    goto :fail
  )
) else (
  echo cv2 already available - leaving OpenCV as-is.
)

if not exist "sam3" (
  echo Cloning SAM 3 from GitHub...
  git clone https://github.com/facebookresearch/sam3.git sam3
  if errorlevel 1 (
    echo git clone failed - is git installed and on PATH?
    goto :fail
  )
) else (
  echo sam3 folder already present - skipping clone.
)

echo Installing SAM 3 ^(editable, no deps^)...
"%PY%" -m pip install -e "sam3" --no-deps
if errorlevel 1 (
  echo SAM 3 install failed.
  goto :fail
)

echo Applying Angelo's SAM 3 compatibility fixes...
"%PY%" "sam3_postinstall.py"

echo.
echo Done. Start ComfyUI again, then use the Detect button in Angelo.
echo The SAM 3 weights ^(sam3.pt^) download automatically on first Detect.

:done
echo.
pause
endlocal
exit /b 0

:fail
echo.
echo Install did NOT complete - see the messages above.
pause
endlocal
exit /b 1
