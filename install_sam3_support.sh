#!/usr/bin/env bash
# Angelo — install the OPTIONAL SAM 3 "Detect" feature (macOS / Linux).
#
# CLOSE ComfyUI before running this (installing into a running env can
# fail on locked/loaded packages), then start it again when done.
#
# Angelo's core needs no extra dependencies; this is only for the SAM 3
# text-segmentation Detect button. It tries to find your ComfyUI Python
# automatically (recorded by the node, or a venv beside ComfyUI). If it
# picks the wrong one, set PYTHON yourself, e.g.:
#     PYTHON=/path/to/ComfyUI/venv/bin/python bash install_sam3_support.sh
set -e

# Keep the window open on finish/error when launched interactively (e.g.
# double-clicked), so the user can read the result. No-op when piped/CI.
_pause() { if [ -t 0 ]; then printf '\nPress Enter to close...'; read -r _; fi; }
trap _pause EXIT

cd "$(dirname "$0")"
echo "Angelo SAM 3 installer — make sure ComfyUI is CLOSED before continuing."

# Pick a Python: PYTHON env var, then the interpreter ComfyUI recorded on
# its last start (.comfy_python.txt — reliable for any launcher), then a
# venv beside ComfyUI (../../), then `python` on PATH as a last resort.
PY="${PYTHON:-}"
if [ -z "$PY" ] && [ -f ".comfy_python.txt" ]; then PY="$(cat .comfy_python.txt)"; fi
if [ -z "$PY" ] && [ -x "../../venv/bin/python" ]; then PY="../../venv/bin/python"; fi
if [ -z "$PY" ] && [ -x "../../.venv/bin/python" ]; then PY="../../.venv/bin/python"; fi
if [ -z "$PY" ]; then PY="python"; fi
if [ ! -f ".comfy_python.txt" ]; then
  echo "NOTE: start ComfyUI once so Angelo can record its Python, for the most reliable install."
fi

echo "Angelo SAM 3 installer — using Python: $PY"
"$PY" --version || { echo "Python not found. Set PYTHON to your ComfyUI python and retry."; exit 1; }

# Pin numpy to whatever this env already has, so nothing installed below can
# downgrade it (a pinned numpy in the deps used to drag ComfyUI's numpy 2.x
# down to 1.26 and break it — issue #31). An empty constraints file is
# harmless if numpy somehow isn't present yet.
CONSTRAINTS="$(mktemp -t angelo_sam3_constraints.XXXXXX 2>/dev/null || echo "${TMPDIR:-/tmp}/angelo_sam3_constraints.txt")"
: > "$CONSTRAINTS"
NPVER="$("$PY" -c 'import numpy,sys;sys.stdout.write(numpy.__version__)' 2>/dev/null || true)"
if [ -n "$NPVER" ]; then
  echo "numpy>=$NPVER" >> "$CONSTRAINTS"
  echo "Pinning numpy>=$NPVER so the SAM 3 deps cannot downgrade it."
fi

if "$PY" -c "import sam3" >/dev/null 2>&1; then
  echo "SAM 3 is already installed in this environment."
  echo "Applying Angelo's SAM 3 compatibility fixes if needed..."
  "$PY" sam3_postinstall.py
  echo "Restart ComfyUI and use the Detect button in Angelo."
  exit 0
fi

echo "Installing SAM 3 runtime dependencies..."
"$PY" -m pip install -r sam3_requirements.txt -c "$CONSTRAINTS"

# OpenCV (cv2) — only install if it isn't already importable, so we don't
# clobber a full opencv-python another node already provides. Headless
# avoids pulling GUI/Qt deps we don't need for the image path.
if "$PY" -c "import cv2" >/dev/null 2>&1; then
  echo "cv2 already available — leaving OpenCV as-is."
else
  echo "Installing opencv-python-headless (cv2 not found)..."
  "$PY" -m pip install "opencv-python-headless>=4.8.0" -c "$CONSTRAINTS"
fi

if [ ! -d sam3 ]; then
  echo "Cloning SAM 3 from GitHub..."
  git clone https://github.com/facebookresearch/sam3.git sam3 || {
    echo "git clone failed — is git installed?"; exit 1; }
else
  echo "sam3/ already present — skipping clone."
fi

echo "Installing SAM 3 (editable, no deps)..."
"$PY" -m pip install -e sam3 --no-deps

echo "Applying Angelo's SAM 3 compatibility fixes..."
"$PY" sam3_postinstall.py

echo ""
echo "Done. Restart ComfyUI, then use the Detect button in Angelo."
echo "(The SAM 3 weights, sam3.pt, download automatically on first Detect.)"
