"""Angelo — SAM 3 auto-segmentation backend.

Our own lazy loader for Meta's SAM 3 (the pip-installed `sam3` package —
not the other node packs). SAM 3's headline capability is *concept /
text* prompts: give it a noun phrase ("the front wheel", "her face") and
it segments every matching instance. We run it on Angelo's current
preview image and return the detections as polygons + boxes so the JS
can overlay them, let the user click-to-confirm, and feed the chosen
shape into the refine pipeline (silhouette for Refine, bbox for Smart
Inpaint).

The model is loaded once on first use and cached. It loads to GPU on
demand and is offloaded back to CPU after each detection so it doesn't
sit on VRAM next to the diffusion model between clicks.

Routes are namespaced under /angelo/ to avoid clashing with other packs.
"""

from __future__ import annotations

import os
import threading
import traceback

_LOCK = threading.Lock()
_STATE = {
    "model": None,        # the SAM3 image model
    "processor": None,    # Sam3Processor
    "device": None,       # torch device the model loads onto
    "import_error": None, # cached import failure message, if any
}

# Checkpoint lives in <models>/sam3/sam3.pt. Auto-downloaded from the
# public HF mirror the installed comfyui-sam3 pack uses (no HF token
# needed — the official facebook/sam3 repo is gated).
_SAM3_HF_REPO = "1038lab/sam3"
_SAM3_CKPT_NAME = "sam3.pt"
# SHA256 of the current 1038lab/sam3 sam3.pt. Verified against the file
# the public mirror serves today (and as a checksum, also a guard against
# pickle tampering at rest, since SAM 3 loads via torch.load). Update if
# the upstream repo republishes — bump it, do not silently delete this
# check. Thanks to @FNGarvin (#19) for both the suggestion and value.
_SAM3_EXPECTED_SHA256 = "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"


def _torch_devices():
    import comfy.model_management as mm
    return mm.get_torch_device(), mm.unet_offload_device()


def _checkpoint_path():
    """Absolute path where sam3.pt should live (<models>/sam3/sam3.pt)."""
    import folder_paths
    return os.path.join(folder_paths.models_dir, "sam3", _SAM3_CKPT_NAME)


def _download_checkpoint(target_path):
    """Fetch sam3.pt into <models>/sam3/ from the public HF mirror, using
    the proven download+move pattern from the installed SAM 3 node packs.
    The downloaded blob is SHA256-verified before being moved into place;
    a mismatch deletes the bad copy and raises so the next attempt retries
    cleanly."""
    import hashlib
    import shutil
    from huggingface_hub import hf_hub_download

    target_dir = os.path.dirname(target_path)
    os.makedirs(target_dir, exist_ok=True)
    print(f"[Angelo/SAM3] sam3.pt not found — downloading from {_SAM3_HF_REPO} "
          f"(this is a one-time ~GB download)...")
    downloaded = hf_hub_download(
        repo_id=_SAM3_HF_REPO,
        filename=_SAM3_CKPT_NAME,
        local_dir=target_dir,
        local_dir_use_symlinks=False,
    )

    h = hashlib.sha256()
    with open(downloaded, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != _SAM3_EXPECTED_SHA256:
        try:
            os.remove(downloaded)
        except OSError:
            pass
        raise RuntimeError(
            f"[Angelo/SAM3] Downloaded sam3.pt failed SHA256 check.\n"
            f"  expected: {_SAM3_EXPECTED_SHA256}\n"
            f"  actual:   {actual}\n"
            "Refusing to load a tampered or unexpected checkpoint. The "
            "bad copy has been deleted; re-run to retry the download."
        )

    if os.path.normpath(downloaded) != os.path.normpath(target_path):
        shutil.move(downloaded, target_path)
    print(f"[Angelo/SAM3] Downloaded to {target_path}")
    return target_path


def _find_bpe(sam3_module):
    """Locate bpe_simple_vocab_16e6.txt.gz next to the imported sam3
    package. Returns the path or None (let build use its default)."""
    name = "bpe_simple_vocab_16e6.txt.gz"
    try:
        pkg_dir = os.path.dirname(os.path.abspath(sam3_module.__file__))
    except Exception:
        return None
    for c in (
        os.path.join(pkg_dir, "assets", name),         # rmbg layout
        os.path.join(pkg_dir, name),
        os.path.join(pkg_dir, "sam3", "assets", name),  # nested (tbg layout)
        os.path.join(pkg_dir, "sam3_lib", name),
    ):
        if os.path.exists(c):
            return c
    return None


def _ensure_model():
    """Lazily build + cache the SAM3 image model and processor. Returns
    (model, processor) or raises RuntimeError with a clear message."""
    if _STATE["import_error"]:
        raise RuntimeError(_STATE["import_error"])
    if _STATE["model"] is not None:
        return _STATE["model"], _STATE["processor"]

    with _LOCK:
        if _STATE["model"] is not None:
            return _STATE["model"], _STATE["processor"]
        try:
            # SAM 3 — whatever `import sam3` resolves to in this env. This
            # may be the pip package OR a copy vendored inside another
            # custom node (rmbg / tbg-sam3) that's on sys.path first.
            import sam3
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except Exception as e:  # pragma: no cover - environment dependent
            if "pkg_resources" in str(e):
                # SAM 3 IS installed, but setuptools 82+ removed the legacy
                # pkg_resources module it imports (issue #34) — don't tell
                # these users SAM 3 "isn't installed". Angelo patches this
                # automatically at startup (__init__.py self-heal), so
                # update + restart is normally enough; the installer is the
                # reliable fallback if the self-heal couldn't reach their
                # copy (permissions, load order).
                _STATE["import_error"] = (
                    "Your SAM 3 install needs a one-time fix: setuptools 82+ "
                    "removed the pkg_resources module SAM 3 relied on. "
                    "Update ComfyUI-Angelo and RESTART ComfyUI — Angelo "
                    "patches SAM 3 automatically at startup. If this notice "
                    "comes back after a restart, close ComfyUI and re-run "
                    "the installer in the ComfyUI-Angelo folder — "
                    "install_sam3_support.bat (Windows) or "
                    "install_sam3_support.sh (macOS/Linux). No setuptools "
                    f"downgrade needed. ({e})"
                )
            else:
                _STATE["import_error"] = (
                    "SAM 3 Detect isn't installed (it's optional). CLOSE ComfyUI, "
                    "then run the installer in the ComfyUI-Angelo folder — "
                    "install_sam3_support.bat (Windows) or install_sam3_support.sh "
                    "(macOS/Linux) — and start ComfyUI again. "
                    f"(sam3 package not importable: {e})"
                )
            raise RuntimeError(_STATE["import_error"])

        load_device, _ = _torch_devices()
        ckpt = _checkpoint_path()
        if not os.path.exists(ckpt):
            try:
                _download_checkpoint(ckpt)
            except Exception as e:
                # Not cached in import_error — a network blip should retry
                # on the next Detect, not block until restart.
                raise RuntimeError(
                    f"SAM 3 weights missing and auto-download failed: {e}\n"
                    f"Place sam3.pt manually at: {ckpt}"
                )
        # build_sam3_image_model's default bpe_path is computed relative to
        # the package and is unreliable when sam3 is a vendored copy inside
        # another node (it pointed at a non-existent assets/ dir). Locate
        # the BPE tokenizer co-located with the loaded sam3 and pass it.
        bpe_path = _find_bpe(sam3)
        print(f"[Angelo/SAM3] Building image model (checkpoint={ckpt}, "
              f"bpe={'auto' if not bpe_path else bpe_path})...")
        # Build under a forced float32 default dtype, then hard-cast to fp32.
        # Some setups leave torch's GLOBAL default dtype at bfloat16; on
        # torch >= 2.10 that bakes bf16 weights into SAM 3 during the build,
        # but SAM 3's image processor always feeds an fp32 image — so the
        # backbone crashes with "mat1 and mat2 must have the same dtype, got
        # BFloat16 and Float". fp32 is SAM 3's known-good precision (and a
        # no-op when the default is already fp32). Restore the previous
        # default afterwards so we don't disturb the rest of ComfyUI.
        # (Root-caused by reproducing on a torch 2.10.0+cu130 RC venv.)
        import torch
        _prev_default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        try:
            if bpe_path:
                model = build_sam3_image_model(checkpoint_path=ckpt, bpe_path=bpe_path)
            else:
                model = build_sam3_image_model(checkpoint_path=ckpt)
        finally:
            torch.set_default_dtype(_prev_default_dtype)
        model.float()
        processor = Sam3Processor(model)
        model.processor = processor
        model.eval()

        _STATE["model"] = model
        _STATE["processor"] = processor
        _STATE["device"] = load_device
        print("[Angelo/SAM3] Model ready.")
        return model, processor


def _mask_to_polygons(mask_np, min_area=64, epsilon_frac=0.004):
    """Binary mask (H, W) -> list of polygons, each a flat [x0,y0,x1,y1,...]
    list in image-pixel coords. Contours are simplified so the payload and
    the JS overlay stay light. Tiny specks are dropped."""
    import numpy as np
    polys = []
    m = (mask_np > 0.5).astype("uint8")
    if m.sum() == 0:
        return polys
    try:
        import cv2
    except Exception:
        # No OpenCV — fall back to the mask's bounding box as a 4-pt polygon.
        ys, xs = np.where(m)
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        return [[x1, y1, x2, y1, x2, y2, x1, y2]]

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon_frac * peri, True)
        pts = approx.reshape(-1, 2)
        if len(pts) < 3:
            continue
        polys.append([int(v) for xy in pts for v in xy])
    return polys


def detect_text(pil_image, text, confidence_threshold=0.3, max_detections=20):
    """Run SAM 3 text grounding on a PIL image. Returns a list of
    detections: {"polygons": [[x,y,...], ...], "bbox": [x1,y1,x2,y2],
    "score": float}, in image-pixel coords, sorted by score."""
    import torch
    import numpy as np

    model, processor = _ensure_model()
    load_device, offload_device = _torch_devices()

    img_w, img_h = pil_image.size
    results = []
    try:
        model.to(load_device)
        if hasattr(processor, "set_confidence_threshold"):
            processor.set_confidence_threshold(float(confidence_threshold))

        # SAM 3 runs in fp32 (forced at build in _ensure_model) and its
        # processor feeds an fp32 image, so weights and activations agree —
        # no dtype juggling needed here.
        state = processor.set_image(pil_image)
        if text and text.strip():
            state = processor.set_text_prompt(text.strip(), state)

        masks = state.get("masks", None)
        boxes = state.get("boxes", None)
        scores = state.get("scores", None)
        if masks is None or len(masks) == 0:
            return []

        # Sort by score (desc) and cap.
        if scores is not None and len(scores) > 0:
            order = torch.argsort(scores, descending=True)
            masks = masks[order]
            boxes = boxes[order] if boxes is not None else None
            scores = scores[order]
        n = len(masks)
        if max_detections > 0:
            n = min(n, max_detections)

        for i in range(n):
            mask_np = masks[i].detach().to("cpu").float().numpy()
            if mask_np.ndim == 3:
                mask_np = mask_np[0]
            polys = _mask_to_polygons(mask_np)
            if not polys:
                continue
            if boxes is not None:
                b = boxes[i].detach().to("cpu").float().tolist()
                bbox = [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
            else:
                ys, xs = np.where(mask_np > 0.5)
                bbox = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
            score = float(scores[i]) if scores is not None else 0.0
            results.append({"polygons": polys, "bbox": bbox, "score": score})
    finally:
        # Offload off the GPU so we don't squat VRAM between clicks.
        try:
            model.to(offload_device)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    return {"width": img_w, "height": img_h, "detections": results}


# --- HTTP route ------------------------------------------------------

try:
    from server import PromptServer
    _HAS_SERVER = True
except ImportError:
    _HAS_SERVER = False


def _load_preview_image(filename, subfolder, type_):
    """Resolve a ComfyUI /view-style image ref to a PIL image."""
    import folder_paths
    from PIL import Image
    type_ = type_ or "temp"
    if type_ == "output":
        base = folder_paths.get_output_directory()
    elif type_ == "input":
        base = folder_paths.get_input_directory()
    else:
        base = folder_paths.get_temp_directory()
    path = os.path.join(base, subfolder or "", filename or "")
    path = os.path.normpath(path)
    # Guard against path traversal outside the base dir.
    if not os.path.normpath(path).startswith(os.path.normpath(base)):
        raise ValueError("invalid image path")
    return Image.open(path).convert("RGB")


if _HAS_SERVER:
    from aiohttp import web

    routes = PromptServer.instance.routes

    @routes.post("/angelo/detect")
    async def _angelo_detect(request):
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)

        method = (data or {}).get("method", "sam3_text")
        text = (data or {}).get("text", "")
        threshold = float((data or {}).get("confidence_threshold", 0.3))
        max_det = int((data or {}).get("max_detections", 20))
        filename = (data or {}).get("filename", "")
        subfolder = (data or {}).get("subfolder", "")
        type_ = (data or {}).get("type", "temp")

        if not filename:
            return web.json_response({"error": "no preview image (generate one first)"}, status=400)

        try:
            pil = _load_preview_image(filename, subfolder, type_)
        except Exception as e:
            return web.json_response({"error": f"could not load preview: {e}"}, status=400)

        # Run the (blocking, GPU) detection off the event loop. On WSL /
        # containerised CUDA, the executor thread occasionally trips a
        # spurious torch.OutOfMemoryError on first allocation despite ample
        # free VRAM (CUDA's per-thread context bookkeeping doesn't see what
        # the main thread already reserved). Fall back to a synchronous
        # main-thread run when that specific failure mode shows up — it
        # blocks the event loop briefly but unblocks the user. Thanks to
        # @FNGarvin (#19) for the diagnosis.
        import asyncio
        import torch
        loop = asyncio.get_event_loop()
        try:
            if method == "sam3_text":
                try:
                    result = await loop.run_in_executor(
                        None, detect_text, pil, text, threshold, max_det
                    )
                except torch.cuda.OutOfMemoryError:
                    print("[Angelo/SAM3] Background-thread CUDA allocation failed — "
                          "retrying on the main thread (WSL/container quirk).")
                    result = detect_text(pil, text, threshold, max_det)
            else:
                return web.json_response({"error": f"unknown method '{method}'"}, status=400)
        except Exception as e:
            traceback.print_exc()
            return web.json_response({"error": str(e)}, status=500)

        return web.json_response({"ok": True, **result})
