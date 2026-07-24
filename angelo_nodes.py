"""AngeloRefine — single-node click-to-refine sampler.

How it works:
  - On first run (no clicks yet), the node decodes the incoming latent
    via VAE and shows the image in its own preview area. The latent is
    cached server-side keyed by node id.
  - The JS frontend attaches click handling to that preview. When the
    user clicks a region they want improved, the JS updates this node's
    click_x / click_y / click_seq widgets and re-queues the workflow.
  - On re-run, the node detects the new click_seq, builds a feathered
    circular mask centred on (click_x, click_y) with radius click_radius,
    re-noises the cached latent partially (`denoise`), and re-samples
    `steps` steps with the mask applied. ComfyUI's noise_mask handling
    re-stitches the original outside the mask at each step (the noise-
    injection inpaint pattern), so the unclicked region stays preserved.
  - The refined latent is cached for the next click, so successive clicks
    keep refining the SAME working latent rather than starting over.
  - Toggle the `reset` widget (or press the Reset button the JS adds) to
    throw away the cache and start fresh from the incoming latent.

The cache is in-process state only — restart of ComfyUI clears it.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import torch

import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview
import folder_paths
import node_helpers

# Reuse PreviewImage's save machinery for the preview output.
import nodes as comfy_nodes


# Per-node state cache:
#   unique_id -> {
#     "history":   list[tuple[Tensor, Tensor | None]], # stack of (latent, pixels) (oldest first, current = [-1])
#     "click_seq": int,            # last processed click_seq from JS
#     "undo_seq":  int,            # last processed undo_seq from JS
#     "redo_seq":  int,            # last processed redo_seq from JS
#     "redo_stack": list,          # entries Undo popped, awaiting Redo (cleared on new edit)
#     "source_latent": Tensor,     # session base latent, for the source_image output
#     "source_pixels": Tensor|None,# decoded source base (lazy, cached)
#     "fingerprint": str,          # hash of incoming latent; mismatch = upstream changed
#   }
_STATE: dict[str, dict] = {}

# ---- Custom-sampler helpers (#8) -----------------------------------------
# These three helpers + the _do_sample wrapper below let users wire a
# GUIDER + SAMPLER + SIGMAS bundle through Angelo — Overrides into Angelo
# in place of the toolbar's steps / cfg / sampler_name / scheduler. The
# helpers themselves are taken verbatim from @KursatAs's customSampler
# branch (https://github.com/KursatAs/ComfyUI-Angelo/tree/customSampler);
# the integration shape (optional via Overrides rather than required on
# the main node) is the Angelo adaptation. Full credit to him for the
# NAG-Extended fix encoded in _guider_sample.

def _guider_sample(
        temp_g,
        noise: torch.Tensor,
        latent: torch.Tensor,
        sampler,
        sigmas: torch.Tensor,
        *,
        denoise_mask: torch.Tensor | None = None,
        callback=None,
        disable_pbar: bool = False,
        seed: int | None = None,
) -> torch.Tensor:
    """Device-safe wrapper around guider.sample(). (From @KursatAs.)

    ComfyUI's built-in CFGGuider.sample() moves noise, latent, and
    denoise_mask to the model's load device before sampling. Some
    third-party extensions (e.g. ComfyUI-NAG-Extended) override
    inner_sample() without repeating that movement, so CPU tensors
    survive into the k-sampler's inpaint path where they collide with
    GPU tensors and raise a device-mismatch RuntimeError. Moving
    everything to load_device here is a safe no-op when the built-in
    already does it, and fixes the crash for extensions that don't."""
    device = temp_g.model_patcher.load_device
    noise = noise.to(device)
    latent = latent.to(device)
    if denoise_mask is not None:
        denoise_mask = denoise_mask.to(device)
    samples = temp_g.sample(
        noise, latent, sampler, sigmas,
        denoise_mask=denoise_mask,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=seed,
    )
    # Mirror comfy.sample.sample()'s exit move so the rest of Angelo's
    # pipeline (VAE decode, pixel composite, latent blend in Fine Upscale)
    # sees the same intermediate device + dtype it does on the default
    # path. Without this, the guider path returns the sampled latent on
    # the model's load_device (cuda) while cached_pixels / mask are on
    # intermediate_device (typically CPU), causing a device-mismatch at
    # the composite step in _refine_with_fine_upscaling.
    return samples.to(
        device=comfy.model_management.intermediate_device(),
        dtype=comfy.model_management.intermediate_dtype(),
    )


def _guider_with_conds(guider, positive, negative):
    """Copy a wired GUIDER and apply Angelo's per-call positive/negative
    conds to it. Handles both CFGGuider (takes both conds) and
    BasicGuider (positive only). (From @KursatAs.) Lets the user wire
    a generic guider once and Angelo keeps using its dynamic conds
    (Refine vs Area Prompt vs Smart Inpaint reference_latents etc.)
    for each sample call."""
    g = copy.copy(guider)
    try:
        g.set_conds(positive, negative)  # comfy.samplers.CFGGuider
    except TypeError:
        g.set_conds(positive)            # BasicGuider / BaseGuider
    return g


def _truncate_sigmas_for_denoise(sigmas: torch.Tensor, denoise: float) -> torch.Tensor:
    """Tail-slice the wired SIGMAS tensor by Angelo's per-call denoise,
    matching ComfyUI's SplitSigmasDenoise convention. (From @KursatAs.)
    A wired sigmas tensor comes pre-baked from a scheduler node at
    denoise=1.0; Angelo applies its own refine denoise here so the
    refine slider keeps meaning what it did before."""
    if denoise >= 1.0:
        return sigmas
    if denoise <= 0.0:
        return sigmas[-1:].new_zeros(2)
    n_total = len(sigmas) - 1
    n_refine = max(1, round(n_total * denoise))
    return sigmas[-(n_refine + 1):]


def _do_sample(
        *,
        guider,
        sampler,
        sigmas,
        model,
        noise: torch.Tensor,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        positive,
        negative,
        source_latent: torch.Tensor,
        denoise: float,
        callback,
        disable_pbar: bool,
        seed: int,
        noise_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Single sample dispatch for every Angelo sample call. If a custom
    guider + sampler + sigmas trio is wired through Overrides, take the
    custom path (via _guider_sample); otherwise fall through to the
    standard comfy.sample.sample(...) path that's existed since v1.0.

    All-or-nothing on the trio: partial wiring (e.g. sampler without
    sigmas) silently falls through to the default path. Users see the
    custom-sampler kwargs only by wiring the full bundle from a proper
    GUIDER + SAMPLER + SIGMAS chain in their workflow."""
    if guider is not None and sampler is not None and sigmas is not None:
        g = _guider_with_conds(guider, positive, negative)
        s = _truncate_sigmas_for_denoise(sigmas, denoise)
        return _guider_sample(
            g, noise, source_latent, sampler, s,
            denoise_mask=noise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=seed,
        )
    return comfy.sample.sample(
        model, noise, steps, cfg, sampler_name, scheduler,
        positive, negative, source_latent,
        denoise=denoise,
        noise_mask=noise_mask,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=seed,
    )


def _zero_out_conditioning(cond):
    """Zeroed-out copy of a conditioning list (same maths as ComfyUI's
    ConditioningZeroOut node). Used as the gen bundle's negative when
    gen_negative isn't wired — the main `negative` input can't stand in
    because it's encoded with the EDIT model's CLIP and conditioning is
    not portable across model families."""
    out = []
    for t, d in cond:
        d = d.copy()
        pooled = d.get("pooled_output")
        if pooled is not None:
            d["pooled_output"] = torch.zeros_like(pooled)
        out.append([torch.zeros_like(t), d])
    return out


# Max number of latents to keep in the undo stack per node. Each FLUX 2
# latent at 832x1776 is ~180 KB (bf16); 10 = ~1.8 MB per node. Cheap.
_HISTORY_CAP: int = 10

# Valid resize methods for the Fine Upscaling crop. All routed through
# comfy.utils.common_upscale which accepts both 4D image tensors and 4D
# latent tensors. lanczos is image-quality; bislerp is latent-aware.
# Default nearest-exact preserves exact sample values (good for
# latents); for the pixel-space path lanczos / bicubic typically look
# better. The user picks one for both ops in the Fine Upscale flow.
_FINE_UPSCALE_RESIZE_METHODS = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp", "lanczos"]

# Smart Guided Inpaint: maps a location dropdown LABEL (set by the JS
# location selector) to a natural-language prefix that's prepended to
# the Area Prompt text before CLIP encoding. The labels here are the
# single source of truth — the JS dropdown lists exactly these keys.
# Order is preserved (py3.7+ dicts) so the JS can build its dropdown
# from the same list via the widget's tooltip / a mirrored array.
_GUIDED_LOCATION_PREFIXES = {
    "(none)":               "",
    "Whole image":          "Across the whole image, ",
    "Top left":             "In the top left of the image, ",
    "Top middle":           "In the top middle of the image, ",
    "Top right":            "In the top right of the image, ",
    "Middle left":          "On the left middle of the image, ",
    "Center":               "In the center of the image, ",
    "Middle right":         "On the right middle of the image, ",
    "Bottom left":          "In the bottom left of the image, ",
    "Bottom middle":        "In the bottom middle of the image, ",
    "Bottom right":         "In the bottom right of the image, ",
    "Left edge":            "Along the left edge of the image, ",
    "Right edge":           "Along the right edge of the image, ",
    "Top edge":             "Along the top edge of the image, ",
    "Bottom edge":          "Along the bottom edge of the image, ",
    "Top half":             "In the top half of the image, ",
    "Bottom half":          "In the bottom half of the image, ",
    "Left half":            "In the left half of the image, ",
    "Right half":           "In the right half of the image, ",
    "Top of the image":     "At the top of the image, ",
    "Bottom of the image":  "At the bottom of the image, ",
}


def _latent_is_blank(latent: torch.Tensor) -> bool:
    """True when the latent is a constant fill — i.e. an EMPTY latent that
    was never sampled. Empty-latent nodes fill with a constant (0.0 for
    EmptyLatentImage and most others; the latent shift, e.g. 0.0609, for
    EmptySD3LatentImage), while a real image latent is never perfectly
    uniform. Used to tell "base was never generated" from "a real latent
    was wired in" on a fresh Edit-Mode queue."""
    v = latent.flatten()[0]
    return bool((latent == v).all())


def _latent_fingerprint(latent: torch.Tensor) -> str:
    """Quick non-cryptographic fingerprint of an incoming latent.

    Used to detect when the upstream KSampler has produced a fresh
    latent (e.g. user changed the prompt + re-queued), so we can
    automatically reset the cache instead of layering refinements
    onto a now-irrelevant base.
    """
    flat = latent.detach().to(torch.float32).flatten()
    n = min(flat.numel(), 1024)
    sample = flat[::max(1, flat.numel() // n)][:n]
    h = hashlib.sha1()
    h.update(str(tuple(latent.shape)).encode())
    h.update(sample.cpu().numpy().tobytes())
    return h.hexdigest()


def _parse_stroke_points(raw: str) -> list[tuple[float, float]]:
    """Parse the JS-set stroke_points widget. Empty / malformed → []."""
    raw = (raw or "").strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except Exception:
        return []
    out = []
    if not isinstance(data, list):
        return []
    for item in data:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                out.append((float(item[0]), float(item[1])))
            except (TypeError, ValueError):
                continue
    return out


def _stroke_mask_latent(
    latent_h: int,
    latent_w: int,
    stroke_points_pixel: list[tuple[float, float]],
    r_latent: float,
    scale_x: float,
    scale_y: float,
    device: torch.device,
) -> torch.Tensor:
    """Vectorised union of circles in latent space, one circle per
    point in stroke_points_pixel. Points come in image-pixel coords;
    we scale them to latent space here using the per-axis ratios.

    Memory: one [N_points, H, W] float tensor briefly. For N=200, a
    typical FLUX 2 latent (~52x111), that's about 4 MB peak — fine.
    """
    if not stroke_points_pixel:
        return torch.zeros((1, latent_h, latent_w), device=device, dtype=torch.float32)

    pts = torch.tensor(stroke_points_pixel, device=device, dtype=torch.float32)
    cx = pts[:, 0] * scale_x  # [N]
    cy = pts[:, 1] * scale_y  # [N]

    ys = torch.arange(latent_h, device=device, dtype=torch.float32).view(1, -1, 1)
    xs = torch.arange(latent_w, device=device, dtype=torch.float32).view(1, 1, -1)
    cxv = cx.view(-1, 1, 1)
    cyv = cy.view(-1, 1, 1)
    dist2 = (xs - cxv) ** 2 + (ys - cyv) ** 2

    circles = (dist2 <= r_latent * r_latent).to(torch.float32)  # [N, H, W]
    mask = circles.amax(dim=0)  # union via max
    return mask.unsqueeze(0)  # [1, H, W]


def _parse_rect_points(raw: str) -> tuple[float, float, float, float] | None:
    """Parse the JS-set rect_points widget — JSON list of one or more
    [x1, y1, x2, y2] entries in image-pixel coords. We use only the
    LAST rectangle (the most recent drag); earlier entries are kept
    in the widget history for the undo stack to consume but don't
    affect mask building. Returns None for empty / malformed input.
    """
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except Exception:
        return None
    if not isinstance(data, list) or not data:
        return None
    last = data[-1]
    if not isinstance(last, (list, tuple)) or len(last) < 4:
        return None
    try:
        return (float(last[0]), float(last[1]), float(last[2]), float(last[3]))
    except (TypeError, ValueError):
        return None


def _rect_mask_latent(
    latent_h: int,
    latent_w: int,
    rect_pixel: tuple[float, float, float, float],
    scale_x: float,
    scale_y: float,
    device: torch.device,
) -> torch.Tensor:
    """Build a [1, H, W] filled-rectangle mask in latent space.

    rect_pixel is (x1, y1, x2, y2) in image-pixel coords; corners
    aren't required to be ordered (the user may drag in any
    direction). Output is clamped to the latent bounds.
    """
    x1, y1, x2, y2 = rect_pixel
    x_lo_p, x_hi_p = min(x1, x2), max(x1, x2)
    y_lo_p, y_hi_p = min(y1, y2), max(y1, y2)

    xlat_lo = max(0, min(latent_w, int(round(x_lo_p * scale_x))))
    xlat_hi = max(0, min(latent_w, int(round(x_hi_p * scale_x))))
    ylat_lo = max(0, min(latent_h, int(round(y_lo_p * scale_y))))
    ylat_hi = max(0, min(latent_h, int(round(y_hi_p * scale_y))))

    mask = torch.zeros((1, latent_h, latent_w), device=device, dtype=torch.float32)
    if xlat_hi > xlat_lo and ylat_hi > ylat_lo:
        mask[0, ylat_lo:ylat_hi, xlat_lo:xlat_hi] = 1.0
    return mask


def _parse_seg_polygons(raw: str):
    """Parse the seg_polygon widget — a JSON list of polygons, each a flat
    [x0,y0,x1,y1,...] coord list in image-pixel space (a SAM 3 / YOLO
    detection's silhouette). Returns the list, or None if empty/invalid."""
    if not raw or not str(raw).strip():
        return None
    try:
        polys = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(polys, list) or not polys:
        return None
    return polys


def _polygons_mask_latent(
    latent_h: int,
    latent_w: int,
    polygons_pixel,
    scale_x: float,
    scale_y: float,
    device: torch.device,
) -> torch.Tensor:
    """Rasterise one or more silhouette polygons (image-pixel coords) into
    a filled [1, H, W] latent-space mask (their union)."""
    import numpy as np
    from PIL import Image, ImageDraw

    img = Image.new("L", (latent_w, latent_h), 0)
    draw = ImageDraw.Draw(img)
    for poly in (polygons_pixel or []):
        if not poly or len(poly) < 6:
            continue
        pts = [
            (float(poly[i]) * scale_x, float(poly[i + 1]) * scale_y)
            for i in range(0, len(poly) - 1, 2)
        ]
        draw.polygon(pts, fill=255)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...].to(device)


def _raster_mask_latent(latent_h, latent_w, png_b64, device):
    """Decode a base64-PNG touch-up mask (image-pixel resolution, white =
    masked) into a [1, H, W] latent-space mask. The Detect Shift/Alt brush
    produces this — a raster handles brushed holes / unions that a polygon
    silhouette can't. Resized straight to the latent grid (no scale args)."""
    import base64
    import io
    import numpy as np
    from PIL import Image

    raw = base64.b64decode(png_b64)
    img = Image.open(io.BytesIO(raw)).convert("L")
    img = img.resize((latent_w, latent_h), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...].to(device)


def _mask_bbox_latent(mask: torch.Tensor) -> tuple[int, int, int, int] | None:
    """Tight latent-space bbox of non-zero mask values. Returns
    (y_min, y_max, x_min, x_max) or None if the mask is empty.

    Threshold of 0.01 includes the feathered edge — bbox covers the
    full soft-edge region not just the binary interior.
    """
    m = mask[0] if mask.dim() == 3 else mask
    nz = m > 0.01
    if not nz.any():
        return None
    rows = nz.any(dim=1)
    cols = nz.any(dim=0)
    ridx = rows.nonzero(as_tuple=False).squeeze(-1)
    cidx = cols.nonzero(as_tuple=False).squeeze(-1)
    return (
        int(ridx[0].item()),
        int(ridx[-1].item()) + 1,
        int(cidx[0].item()),
        int(cidx[-1].item()) + 1,
    )


def _fine_upscale_factor(
    bbox_w_latent: int,
    bbox_h_latent: int,
    scale_x: float,
    scale_y: float,
    target_mp: float,
    max_linear: float,
) -> float:
    """Linear scale factor to apply to the cropped latent so that the
    crop is processed at ≥ target_mp (in image-pixel-equivalent terms),
    clamped to max_linear. Returns 1.0 when the crop already meets
    target — no upscale needed."""
    if scale_x <= 0 or scale_y <= 0:
        return 1.0
    bbox_w_pix = bbox_w_latent / scale_x
    bbox_h_pix = bbox_h_latent / scale_y
    current_mp = bbox_w_pix * bbox_h_pix / 1_000_000.0
    if current_mp <= 0 or current_mp >= target_mp:
        return 1.0
    needed = math.sqrt(target_mp / current_mp)
    return min(needed, max_linear)


def _resize_latent(t: torch.Tensor, target_h: int, target_w: int, method: str) -> torch.Tensor:
    """Resize the spatial dims of a latent or mask tensor using one of
    ComfyUI's standard latent-resize methods. Accepts [C,H,W], [B,C,H,W],
    or [1,H,W] (mask). Returns the same rank as input.

    `method` is one of _FINE_UPSCALE_RESIZE_METHODS. Routes through
    comfy.utils.common_upscale so bislerp + lanczos custom paths work."""
    method = method if method in _FINE_UPSCALE_RESIZE_METHODS else "nearest-exact"
    if t.dim() == 3:
        t4 = t.unsqueeze(0)
        out = comfy.utils.common_upscale(t4, target_w, target_h, method, "disabled")
        return out.squeeze(0)
    if t.dim() == 4:
        return comfy.utils.common_upscale(t, target_w, target_h, method, "disabled")
    raise ValueError(f"_resize_latent: unexpected ndim {t.dim()}")


# ----- VAE boundary -----
# Every latent->pixel and pixel->latent conversion in the node routes
# through these two functions. Centralising them means model-family
# quirks live in ONE place instead of at ~7 scattered call sites. The
# motivating case is Qwen Image Edit / Wan-derived VAEs, whose latents
# carry an extra temporal axis ([B, C, T, H, W] with T=1) that the rest
# of the node — and PIL's previewer — expect to be absent ([B, C, H, W]).
# Both helpers are thin pass-throughs today; this is the seam where that
# 5D normalisation will land without disturbing any caller.
# Above this output megapixel count, route VAE encode/decode through the
# TILED path. A single whole-canvas VAE pass at very large sizes (e.g. a
# 7744x2176 = 16.8MP canvas) can OOM or numerically overflow to NaN — which
# is silent (no exception), poisons everything downstream, and shows up as a
# black image. Tiling keeps each VAE pass small. Below the threshold we use
# the plain (exact, faster) path; ~1.4k tiles and normal canvases stay plain.
_VAE_TILE_MP = 4.0


def _vae_decode(vae, latent: torch.Tensor) -> torch.Tensor:
    """Decode a latent to pixels. Single decode chokepoint — see the
    VAE-boundary note above. Always returns a 4D image batch
    (B, H, W, C) float in [0, 1].

    Large canvases decode TILED (see _VAE_TILE_MP) so a single huge VAE pass
    can't OOM/overflow to NaN and blacken the output.

    Temporal/video VAEs (Qwen Image Edit, Wan) keep a frame axis: their
    latents are 5D ([B, C, T, H, W]) and `vae.decode` accordingly returns
    a 5D frame stack ([B, T, H, W, C] — ComfyUI moves channels last). The
    rest of the node, and ComfyUI's PreviewImage/PIL path, only understand
    4D image batches, so fold the frame axis into the batch dim. For image
    editing T is 1, so this is just dropping the singleton frame axis; if a
    future model ever produces T>1 the frames surface as extra batch items
    rather than crashing. The latent is passed through to `vae.decode`
    untouched — the video VAE wants its native 5D input — we only normalise
    the *pixels* it returns."""
    try:
        # Ask the VAE for its true spatial ratio — hardcoding 8 under-
        # estimated the output 4× on 16×-per-axis VAEs (FLUX 2, the node's
        # main target), so tiling only engaged ~4× past the danger point.
        try:
            ratio = int(vae.spacial_compression_decode())
        except Exception:
            ratio = int(getattr(vae, "downscale_ratio", 8) or 8)
        out_mp = (latent.shape[-2] * ratio) * (latent.shape[-1] * ratio) / 1e6
    except Exception:
        out_mp = 0.0
    image = vae.decode_tiled(latent) if out_mp > _VAE_TILE_MP else vae.decode(latent)
    if image.ndim == 5:
        b, t, h, w, c = image.shape
        image = image.reshape(b * t, h, w, c)
    return image


def _vae_encode(vae, pixels: torch.Tensor) -> torch.Tensor:
    """Encode pixels to latent samples. Single encode chokepoint —
    counterpart to _vae_decode. See the VAE-boundary note above.

    Deliberately returns the VAE's native latent shape WITHOUT collapsing
    it: a temporal/video VAE (Qwen, Wan) returns a 5D latent
    ([B, C, T, H, W]) and the sampler + model require that 5D shape to flow
    through unchanged (comfy.sample.sample is ndim-agnostic and prepare_noise
    matches the latent's shape exactly). Squeezing the frame axis here would
    break Qwen sampling — do not add a squeeze.

    Large canvases encode TILED (see _VAE_TILE_MP) so a single huge VAE pass
    can't OOM/overflow to NaN (which would silently poison the latent)."""
    try:
        in_mp = pixels.shape[1] * pixels.shape[2] / 1e6
    except Exception:
        in_mp = 0.0
    if in_mp > _VAE_TILE_MP:
        return vae.encode_tiled(pixels)
    return vae.encode(pixels)


def _grow_mask_latent(mask: torch.Tensor, frac: float) -> torch.Tensor:
    """Dilate a latent-space mask outward by `frac` of its bbox extent per side
    (a maxpool dilation). Used by the Remove brush so the erase fully covers the
    object plus its immediate halo. `mask`: [1, H, W] in [0, 1]. Returns the
    same shape."""
    import torch.nn.functional as F
    bbox = _mask_bbox_latent(mask)
    if bbox is None or frac <= 0:
        return mask
    y0, y1, x0, x1 = bbox
    mean_ext = 0.5 * ((y1 - y0) + (x1 - x0))
    grow = int(round(frac * mean_ext))
    if grow < 1:
        return mask
    k = 2 * grow + 1
    m = mask
    while m.dim() < 4:
        m = m.unsqueeze(0)                               # [1,H,W] -> [1,1,H,W]
    grown = F.max_pool2d(m, k, stride=1, padding=grow)
    return grown.view(mask.shape).clamp(0.0, 1.0)


def _refine_with_fine_upscaling(
    *,
    model,
    vae,
    current: torch.Tensor,               # [B, C, H_lat, W_lat] cached full-res latent
    current_pixels: torch.Tensor | None, # [B, H_pix, W_pix, C] cached full-res pixels to avoid redundant VAE decode
    mask: torch.Tensor,                  # [1, H_lat, W_lat] feathered mask, latent res
    scale_x: float,
    scale_y: float,
    target_mp: float,
    max_linear: float,
    resize_method: str,
    context_pad_pixel: int,
    inpainting_mode: str,
    reference_strength: float = 0.0,  # Refine + Ref value: anchor on the
                                      # (pre-refine) crop for this fraction
                                      # of the schedule (_apply_reference) —
                                      # WITHOUT Smart Inpaint's latent-zeroing
    remove_mode: bool = False,        # Remove brush via Xtra-Fine: reference the
                                      # crop with its masked region ZEROED (a real
                                      # hole) so the edit model rebuilds the
                                      # background; regenerate at full denoise
    seed: int = 0,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    positive,
    negative,
    denoise: float,
    callback,
    disable_pbar: bool,
    # #8 custom-sampler trio — None = use the default comfy.sample.sample
    # path; all three set = dispatch via _do_sample to the guider path.
    ov_guider=None,
    ov_sampler=None,
    ov_sigmas=None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Pixel-space crop + upscale + VAE encode + refine + VAE decode +
    downscale + composite + VAE encode. The latent-space crop+upscale
    approach smears bilinearly-interpolated latents into a low-freq
    starting state that the model can't recover detail from. Going
    through pixel space (where there's an image-upscale toolkit that's
    been tuned for natural images) and re-encoding gives the model a
    "natural" latent at the higher resolution to denoise from.

    Returns a tuple of (new_latent, new_pixels). Returns (current, current_pixels)
    if the mask bbox is empty / degenerate.
    """
    bbox = _mask_bbox_latent(mask)
    if bbox is None:
        return current, current_pixels
    y0_tight, y1_tight, x0_tight, x1_tight = bbox

    # Apply context padding: grow the bbox outward by context_pad_pixel
    # in every direction (clamped to the latent boundaries). This is
    # the area the model SEES during refine. The painted-shape mask
    # stays unchanged — areas inside the padded bbox but outside the
    # painted shape have mask=0 in the cropped tensor, so the noise-
    # injection inpaint preserves them as context (the model uses them
    # to inform what to draw inside the mask, but doesn't overwrite
    # them). All downstream code uses the PADDED bbox.
    H_lat = current.shape[-2]
    W_lat = current.shape[-1]
    pad_lat_y = max(0, round(context_pad_pixel * scale_y))
    pad_lat_x = max(0, round(context_pad_pixel * scale_x))
    y0 = max(0, y0_tight - pad_lat_y)
    y1 = min(H_lat, y1_tight + pad_lat_y)
    x0 = max(0, x0_tight - pad_lat_x)
    x1 = min(W_lat, x1_tight + pad_lat_x)

    bbox_h_lat = y1 - y0
    bbox_w_lat = x1 - x0
    if bbox_h_lat <= 0 or bbox_w_lat <= 0:
        return current, current_pixels

    scale = _fine_upscale_factor(bbox_w_lat, bbox_h_lat, scale_x, scale_y, target_mp, max_linear)
    if scale <= 1.0 and inpainting_mode != "Smart Inpaint" and not remove_mode:
        # Refine with no upscale needed — fall back to the standard latent-space
        # noise-injection inpaint. Avoids unnecessary VAE round-trips when the
        # painted region already meets the MP target.
        #
        # Smart Inpaint must NOT take this shortcut. It needs the crop +
        # reference_latents + masked-zero treatment below regardless of rect
        # size; skipping it made a LARGE rectangle (already at/above the MP
        # target, so scale<=1.0 — roughly >1024px on FLUX 2) degrade to a
        # whole-latent edit with NO crop reference, so the model worked on the
        # whole image instead of the selected rect.
        print(f"[Angelo fine-upscale] scale=1.0 — using latent-space path (no VAE round-trip)")
        if reference_strength > 0.0:
            # Refine + Reference on the no-upscale path: anchor on the whole
            # current image (there's no crop on this path).
            positive = _apply_reference(positive, current.clone(), reference_strength)
        noise = comfy.sample.prepare_noise(current, seed, None)
        new_latent = _do_sample(
            guider=ov_guider, sampler=ov_sampler, sigmas=ov_sigmas,
            model=model, noise=noise,
            steps=steps, cfg=cfg, sampler_name=sampler_name, scheduler=scheduler,
            positive=positive, negative=negative,
            source_latent=current,
            denoise=denoise,
            noise_mask=mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=seed,
        )
        # Return None for pixels because the latent was modified directly;
        # this forces a fresh VAE decode for the preview in the main run() method.
        return new_latent, None

    # Smart Inpaint with a large rectangle still crops + references the selected
    # region; it just doesn't upscale (and must never downscale) — clamp the
    # factor to identity so the crop is taken at native resolution. Remove takes
    # the same crop+reference path regardless of size, so clamp it too.
    if inpainting_mode == "Smart Inpaint" or remove_mode:
        scale = max(1.0, scale)

    # ----- VAE decode the full cached latent → cached pixels -----
    # Optimization: Reuse cached pixels if available to prevent VAE degradation 
    # (loss of high-frequency details) across multiple consecutive edits.
    if current_pixels is not None:
        cached_pixels = current_pixels
    else:
        cached_pixels = _vae_decode(vae, current)  # (B, H_pix, W_pix, C) float [0,1]
        
    H_pix = cached_pixels.shape[1]
    W_pix = cached_pixels.shape[2]
    # Pixel-per-latent ratio per axis (16 for FLUX 2, 8 for SDXL/SD1.5).
    # round(), not floor-divide: a non-integer true ratio (exotic VAEs)
    # floor-divided gives e.g. 15 for 15.8, drifting the pixel-space bbox
    # ~1px against the latent bbox and leaving a seam in the composite.
    # (#28, from @KursatAs.)
    px_per_lat_y = max(1, round(H_pix / current.shape[-2]))
    px_per_lat_x = max(1, round(W_pix / current.shape[-1]))

    # Pixel-space bbox derived from the latent-space bbox.
    y0_p = y0 * px_per_lat_y
    y1_p = y1 * px_per_lat_y
    x0_p = x0 * px_per_lat_x
    x1_p = x1 * px_per_lat_x
    bbox_h_p = y1_p - y0_p
    bbox_w_p = x1_p - x0_p

    # Upscaled target dims in pixel space. Snap to multiples of the
    # VAE downscale (16 for FLUX 2) so the subsequent VAE encode
    # produces a clean integer-dim latent.
    vae_snap = max(px_per_lat_y, px_per_lat_x)
    target_h_p = max(vae_snap, math.ceil(bbox_h_p * scale / vae_snap) * vae_snap)
    target_w_p = max(vae_snap, math.ceil(bbox_w_p * scale / vae_snap) * vae_snap)

    print(f"[Angelo fine-upscale] bbox_lat=(h={bbox_h_lat}, w={bbox_w_lat}) "
          f"bbox_px=(h={bbox_h_p}, w={bbox_w_p}) scale={scale:.2f} "
          f"target_px=(h={target_h_p}, w={target_w_p}) "
          f"resize={resize_method} max_linear={max_linear} "
          f"vae_ratio=(x={px_per_lat_x}, y={px_per_lat_y})")

    # ----- Crop pixel image + upscale in pixel space -----
    pixel_crop = cached_pixels[:, y0_p:y1_p, x0_p:x1_p, :]  # (B, h, w, C)
    # common_upscale expects (B, C, H, W) — permute, upscale, permute back.
    pixel_crop_chw = pixel_crop.movedim(-1, 1)
    pixel_crop_up_chw = comfy.utils.common_upscale(
        pixel_crop_chw, target_w_p, target_h_p, resize_method, "disabled",
    )
    pixel_crop_up = pixel_crop_up_chw.movedim(1, -1)  # back to (B, H, W, C)

    # ----- VAE encode the upscaled pixel crop → latent at high res -----
    latent_up = _vae_encode(vae, pixel_crop_up)
    target_h_lat = latent_up.shape[-2]
    target_w_lat = latent_up.shape[-1]

    # ----- Build mask at the upscaled latent resolution -----
    # Mask resizing always uses bilinear regardless of the user's choice.
    # The user's resize_method is for the IMAGE content upscale (where
    # lanczos / bicubic / etc. have real quality differences). The mask
    # is a 1-channel feathered alpha where we just want smooth values;
    # lanczos's grayscale-branch returns a transposed 3D tensor (PIL
    # quirk) and bislerp's spherical-vector math is semantically wrong
    # on a single channel.
    mask_crop = mask[..., y0:y1, x0:x1].contiguous()
    mask_crop_up = _resize_latent(mask_crop, target_h_lat, target_w_lat, "bilinear").clamp(0.0, 1.0)

    # ===== Smart Inpaint pre-processing on the upscaled patch =====
    # Klein 9B's edit branch only activates when reference_latents is
    # present on the conditioning. We then zero the masked area so the
    # sampler regenerates that region from full noise at sigma_max
    # (the denoise=1.0 lock makes this clean: every pixel in the
    # painted rect is brand-new content, with the surrounding context
    # band restored each step by the noise_mask compositing). The
    # reference uses the PRE-ZERO upscaled patch so Klein still sees
    # what was there before we blanked it.
    # POSITIVE ONLY — putting reference_latents on negative would tell
    # CFG>1 samplers to steer AWAY from the reference scene. Non-edit
    # models ignore the field, so this is harmless on any checkpoint.
    #
    # append=False (REPLACE, not append): the reference must be ONLY this
    # upscaled crop. When the Area Prompt is empty, refine_positive falls back
    # to the node's `positive` input, which in a Klein edit workflow already
    # carries reference_latents = the WHOLE source image (from an upstream
    # ReferenceLatent node). append=True stacked the crop onto that whole-image
    # reference, and the whole-image one dominated — so the patch reproduced
    # the entire original scene instead of editing the selected region.
    # Replacing guarantees Klein sees the crop and nothing else.
    # Mask used for the SAMPLING step (both the masked-zero and the noise_mask).
    # Defaults to the feathered mask; hardened to binary for Smart Inpaint on
    # 5D temporal latents — see the note in the Smart Inpaint block below.
    sample_mask = mask_crop_up
    if remove_mode:
        # Remove via Xtra-Fine: reference the crop with its masked region ZEROED
        # (a real hole), so the edit model rebuilds the background from the
        # surrounding crop rather than anchoring to the object. The source latent
        # is NOT zeroed — at denoise 1.0 the masked region is noised anyway, and
        # outside the mask the crop is kept. 5D uses a hard sampling mask for the
        # same reason as Smart Inpaint below.
        if latent_up.ndim == 5:
            sample_mask = (mask_crop_up >= 0.5).to(mask_crop_up.dtype)
        ref_hole = latent_up.clone() * (1.0 - sample_mask.unsqueeze(0))
        positive = node_helpers.conditioning_set_values(
            positive, {"reference_latents": [ref_hole]}, append=False,
        )
    elif inpainting_mode == "Smart Inpaint":
        reference_latent = latent_up.clone()
        positive = node_helpers.conditioning_set_values(
            positive, {"reference_latents": [reference_latent]}, append=False,
        )
        # Feather goes in the COMPOSITE, not the denoise mask — for Qwen/Wan.
        #
        # FLUX/Klein (4D, ~zero-mean latents): a soft mask works directly as
        # both the masked-zero and the noise_mask; their inpaint blend handles a
        # soft boundary cleanly, so keep the feathered mask.
        #
        # Qwen Image Edit / Wan (5D temporal latents): sample with a HARD
        # (binary) mask. These models have no clean soft-mask-during-denoise
        # behaviour — the community-standard Qwen-edit inpaint recipe is "blank
        # the region, regenerate, then composite back with a feathered blend",
        # NOT feathering the denoise mask. A soft noise_mask at denoise=1.0 on a
        # non-zero-mean latent space distorts exactly the feather band (the
        # symptom: artifacts only where the feathering happens). The smooth
        # visible edge is still produced downstream by the feathered PIXEL
        # composite + final latent blend (both use the full-res feathered
        # `mask`), so the feather is preserved without the sampling artifacts.
        if latent_up.ndim == 5:
            sample_mask = (mask_crop_up >= 0.5).to(mask_crop_up.dtype)
        latent_up = (1.0 - sample_mask.unsqueeze(0)) * latent_up
    elif reference_strength > 0.0:
        # Refine + Reference value: the upscaled (pre-refine) crop anchors
        # identity/content through the edit branch for the first
        # reference_strength fraction of the schedule, so a high denoise
        # can fully re-render texture without losing the subject — the
        # photo-restoration recipe. Unlike Smart Inpaint, the latent is NOT
        # zeroed: the existing content remains the starting state.
        positive = _apply_reference(positive, latent_up.clone(), reference_strength)

    # ----- Refine via noise-injection inpaint on the upscaled latent -----
    noise = comfy.sample.prepare_noise(latent_up, seed, None)
    refined_latent_up = _do_sample(
        guider=ov_guider, sampler=ov_sampler, sigmas=ov_sigmas,
        model=model, noise=noise,
        steps=steps, cfg=cfg, sampler_name=sampler_name, scheduler=scheduler,
        positive=positive, negative=negative,
        source_latent=latent_up,
        denoise=denoise,
        noise_mask=sample_mask,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=seed,
    )

    # ----- VAE decode refined latent → high-res pixel patch -----
    refined_pixel_up = _vae_decode(vae, refined_latent_up)  # (B, target_h_p, target_w_p, C)

    # ----- Downscale refined patch back to original bbox pixel size -----
    refined_pixel_up_chw = refined_pixel_up.movedim(-1, 1)
    refined_pixel_chw = comfy.utils.common_upscale(
        refined_pixel_up_chw, bbox_w_p, bbox_h_p, resize_method, "disabled",
    )
    refined_pixel = refined_pixel_chw.movedim(1, -1)  # (B, bbox_h_p, bbox_w_p, C)

    # ----- Composite refined patch into the cached pixel image -----
    # Build a pixel-space alpha by resizing the latent feathered mask to
    # full pixel resolution, cropping to the bbox. Always bilinear for
    # the same reasons as the mask upscale above — lanczos's grayscale
    # path is broken, bislerp doesn't apply to 1-channel.
    mask_4d = mask.unsqueeze(0)  # [1, 1, H_lat, W_lat]
    pixel_mask = comfy.utils.common_upscale(
        mask_4d, W_pix, H_pix, "bilinear", "disabled",
    ).clamp(0.0, 1.0)  # [1, 1, H_pix, W_pix]
    pixel_alpha_crop = pixel_mask[0, 0, y0_p:y1_p, x0_p:x1_p]  # [bbox_h_p, bbox_w_p]
    pixel_alpha_crop = pixel_alpha_crop.unsqueeze(0).unsqueeze(-1)  # [1, h, w, 1]

    new_pixels = cached_pixels.clone()
    pixel_orig_crop = cached_pixels[:, y0_p:y1_p, x0_p:x1_p, :]
    composited = refined_pixel * pixel_alpha_crop + pixel_orig_crop * (1.0 - pixel_alpha_crop)
    new_pixels[:, y0_p:y1_p, x0_p:x1_p, :] = composited

    # ----- VAE encode the composited full image → encoded latent -----
    encoded_latent = _vae_encode(vae, new_pixels)

    # ----- Blend in LATENT space using the feathered mask as alpha -----
    # The VAE encode is lossy, so naively returning encoded_latent would
    # mean the *unaltered* regions of the image drift slightly with every
    # Fine Upscale click. Avoidable: keep the original cached latent
    # outside the mask, take the encoded latent inside the mask. Mask is
    # already feathered, so the transition is smooth. Now unaltered
    # regions stay bit-exact across successive clicks; only the masked
    # area accumulates any VAE-roundtrip cost (and it gets a fresh
    # refine each click anyway, so any drift there is overwritten).
    alpha_lat = mask.unsqueeze(0)  # [1, 1, H_lat, W_lat]
    new_latent = encoded_latent * alpha_lat + current * (1.0 - alpha_lat)
    
    return new_latent, new_pixels


# Quick Photo Refine: the one-click restoration recipe — a MAGIC BUTTON
# with its own fixed settings, deliberately independent of every toolbar
# box. The whole canvas is re-rendered with this prompt, anchored to the
# current image via reference_latents — identity from the reference,
# texture from the re-render. Tested values; tuned here in ONE place.
# The winning prompt is an INSTRUCTION — edit models are instruction-
# trained, so speaking their vocabulary beat every descriptive/
# constraint phrasing tried ("high quality photo", keep-the-colours
# variants — see git history). At ref 1.0 the blend collapses to a
# single cond, so this recipe also costs nothing extra.
#
# Per-model phrasing: Qwen-Image-Edit responds better to a fuller,
# gentler instruction than Klein does. Selected at run time by latent
# dimensionality — Qwen/Wan-family latents are 5D [B,C,T,H,W], the
# same load-bearing check the inpaint hard-mask logic uses.
_QUICK_REFINE_PROMPT = "Keep the identity from image 1. make the image high quality."
_QUICK_REFINE_PROMPT_QWEN = ("lightly restore this old photo, remove dust and scratches, "
                             "improve sharpness and contrast, preserve original feel")
_QUICK_REFINE_DENOISE = 1.0
# Lite mode (toggle beside the prompt selector): identical recipe, gentler
# denoise — a lighter restore that re-renders less and stays closer to the
# input. Everything else (ref, target, prompt, tiling, re-roll) is unchanged.
_QUICK_REFINE_LITE_DENOISE = 0.8
_QUICK_REFINE_REF = 1.0
# ✨ v2 runs the whole image through the XTRA-FINE pipeline: whole-canvas
# mask, target 1.3MP (small images get internally supersampled to 1.3MP,
# refined there, composited back), ctx pad 128 (no-op at full coverage),
# feather 0.
_QUICK_REFINE_TARGET_MP = 1.3
_QUICK_REFINE_CTX_PAD = 128

# ✨ prompt presets (the selector beside the button). All instruction-
# register with the image-1 identity clause — the phrasing family testing
# proved out. "Use Area Prompt" hands the text box over instead (falls
# back to the default preset when the box is empty).
_QUICK_REFINE_PROMPTS = {
    "Identity + Quality": _QUICK_REFINE_PROMPT,
    "Restore Photo": "Keep the identity from image 1. restore the photo.",
    "Identity + Colours": ("Keep the identity and colours from image 1. "
                           "make the image high quality."),
}
_QUICK_REFINE_PROMPT_MODES = list(_QUICK_REFINE_PROMPTS.keys()) + ["Use Area Prompt"]

# 🩹 Remove brush — toggle beside Restore (Refine mode, edit models only).
# Erases the painted region to background in a SINGLE denoise-1.0 sampling pass
# over the grown, hard-edged mask (no pixel fill): regenerate the masked region
# with a background-fill instruction (_REMOVE_PROMPT) and a reference latent whose masked
# region is ZEROED — a real hole in the reference, so the edit model has nothing
# to anchor to there and rebuilds the background from the surrounding scene. Full
# denoise removes the object from the starting state; the zeroed reference stops
# it being reconstructed (the ghost). This is the direct analog of outpaint's
# protect brush, which excludes a region so it isn't repeated.
#
# feather 0 (hard mask, no soft-edge bleed) + grow the mask ~10% (cover the
# object's halo — a tight mask leaves a rim of the original behind). The Denoise
# box is unused — the pass is always full denoise.
#
# ALWAYS runs on the Xtra-Fine crop path (Remove forces the toggle on and won't
# run without it): crop the region at the toolbar ctx-pad / MP / method and do
# the removal on a high-res crop, with the zeroed reference built from that crop.
# This is what rebuilds fine background detail behind the removed object.
# FILL-style, not "remove"-style: the object is already gone from what the model
# sees (the reference has its masked region zeroed), so the prompt's only job is
# to fill the hole with plausible surrounding background — telling it to "remove
# the object" is redundant/confusing when it's looking at a hole.
_REMOVE_PROMPT = ("Seamlessly continue the surrounding background across the masked area. "
                  "A natural, coherent extension of the scene behind it, matching the "
                  "surrounding lighting, texture, colour and perspective. No objects, no new "
                  "content — background only.")
_REMOVE_PROMPT_QWEN = ("seamlessly continue the surrounding background across the masked area, "
                       "a natural coherent extension of the scene behind it, matching the "
                       "surrounding lighting, texture, colour and perspective; background only, "
                       "no objects and no new content")
_REMOVE_MASK_GROW_FRAC = 0.05  # ~10% larger area

# Tiled restore engine (2× Restore Upscale + big-canvas Quick Refine).
# Working tile size + overlap in PIXELS: tiles are sampled at ~1MP no
# matter how large the canvas is, so the latent fed to the model never
# outgrows the resolution it renders well at. Quick Photo Refine
# auto-routes through the tiled engine above the MP threshold for the
# same reason.
_TILE_PX = 1400
_TILE_OVERLAP_PX = 128


def _tile_min_overlap_px(canvas_long_edge: int) -> int:
    """Canvas-proportional MINIMUM tile overlap. Bigger canvases mean each
    tile sees a smaller slice of the scene, so neighbours need more shared
    context to agree. Linear in the long edge, calibrated on tested
    points: 3872px -> 172, 7744px -> 256; clamps to the 128px floor below
    ~1840px."""
    return max(_TILE_OVERLAP_PX, 88 + round(canvas_long_edge / 46.0))
_QUICK_REFINE_TILE_THRESHOLD_MP = 1.6




def _apply_reference(positive, ref_latent: torch.Tensor, strength: float):
    """Attach ref_latent as reference_latents at a TRUE fractional strength
    via dual-conditioning prediction blending.

    There is no native scalar weight on reference_latents — the reference
    becomes extra image tokens, take-it-or-leave-it. But ComfyUI's cond
    `strength` field is a RELATIVE weight: each cond's prediction is
    multiplied by its strength, accumulated, then normalised by the total.
    So two full-coverage conds — [with-reference, strength s] + [plain,
    strength 1-s] — yield exactly  s·pred_ref + (1-s)·pred_plain  at every
    step: a genuine prediction-space lerp between anchored and free.

      1.0      → one cond, fully anchored (no extra cost)
      0 < s <1 → the dual-cond blend above. COSTS A SECOND positive-side
                 model evaluation per step (like CFG's negative does) —
                 the price of a real interpolation.
      0.0      → untouched (no reference, no extra cost)

    (A timestep-range variant — strength as DURATION of anchoring — was
    tried first and rejected by testing; see git history.)
    """
    s = max(0.0, min(1.0, float(strength)))
    if s <= 0.0:
        return positive
    with_ref = node_helpers.conditioning_set_values(
        positive, {"reference_latents": [ref_latent]}, append=False,
    )
    if s >= 1.0:
        return with_ref
    head = node_helpers.conditioning_set_values(with_ref, {"strength": s})
    tail = node_helpers.conditioning_set_values(positive, {"strength": 1.0 - s})
    return head + tail


def _tile_positions(total: int, tile: int, overlap: int) -> list[int]:
    """Start offsets covering [0, total) with `tile`-sized windows,
    UNIFORMLY distributed so every seam gets the same overlap (>= the
    requested minimum). The old flush-to-end layout produced wildly
    inconsistent seams (128px here, 900px there) — uniform spacing means
    every boundary shares at least the minimum context, evenly."""
    if total <= tile:
        return [0]
    stride = max(1, tile - max(0, overlap))
    n = max(2, math.ceil((total - overlap) / stride))
    step = (total - tile) / (n - 1)
    return [round(i * step) for i in range(n)]


def _tiled_restore_pass(
    *,
    model,
    vae,
    current: torch.Tensor,
    current_pixels: torch.Tensor | None,
    scale: float,
    positive_base,
    negative,
    seed: int,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    callback,
    disable_pbar: bool,
    ov_guider=None,
    ov_sampler=None,
    ov_sigmas=None,
    tile_ref: float = _QUICK_REFINE_REF,
    tile_denoise: float = _QUICK_REFINE_DENOISE,
    dual_grid: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """The tiled restore engine: the Quick Photo Refine recipe run over
    overlapping ~1MP tiles, so the model never samples a latent bigger
    than its happy place — regardless of canvas size.

    scale=2.0 → 2× Restore Upscale: decode, lanczos-upscale the PIXELS
    (latent upscaling smears; pixel space is the Xtra-Fine lesson),
    re-encode, then restore tile by tile. scale=1.0 → in-place tiled
    restore (how Quick Photo Refine handles big canvases).

    Each tile is sampled at denoise 1.0 anchored to ITS OWN pre-refine
    latent via reference_latents. The anchor is what makes tiled
    full-denoise viable: tiles can't hallucinate new content (the
    classic tiled-upscaler failure), and adjacent tiles agree because
    they're anchored to the same underlying image — the reference does
    the job tile-ControlNets do in other pipelines.

    Tiles composite in pixel space under a separable ramp weight
    (linear fade over the overlap), normalised by the accumulated
    weight, so seams blend and un-overlapped edges keep weight 1.

    Returns (final_latent, final_pixels, n_tiles).
    """
    pixels = current_pixels if current_pixels is not None else _vae_decode(vae, current)
    B, H, W, C = pixels.shape

    if scale != 1.0:
        new_W = max(16, int(round(W * scale / 16.0)) * 16)
        new_H = max(16, int(round(H * scale / 16.0)) * 16)
        chw = pixels.movedim(-1, 1)
        chw = comfy.utils.common_upscale(chw, new_W, new_H, "lanczos", "disabled")
        big_pixels = chw.movedim(1, -1).contiguous()
    else:
        big_pixels = pixels
    new_H, new_W = big_pixels.shape[1], big_pixels.shape[2]

    big_latent = _vae_encode(vae, big_pixels)
    nlh, nlw = big_latent.shape[-2], big_latent.shape[-1]
    ppl_x = max(1, round(new_W / nlw))
    ppl_y = max(1, round(new_H / nlh))

    tile_lx = max(8, _TILE_PX // ppl_x)
    tile_ly = max(8, _TILE_PX // ppl_y)
    min_ov_px = _tile_min_overlap_px(max(new_W, new_H))
    ov_lx = max(1, min_ov_px // ppl_x)
    ov_ly = max(1, min_ov_px // ppl_y)
    xs = _tile_positions(nlw, tile_lx, ov_lx)
    ys = _tile_positions(nlh, tile_ly, ov_ly)
    n_tiles = len(xs) * len(ys)
    print(f"[Angelo tiled-restore] scale={scale} canvas={new_W}x{new_H} "
          f"grid={len(xs)}x{len(ys)} ({n_tiles} tiles of ~{_TILE_PX}px, "
          f"min overlap {min_ov_px}px, uniform spacing)")

    out = torch.zeros_like(big_pixels)
    wsum = torch.zeros((1, new_H, new_W, 1), dtype=big_pixels.dtype, device=big_pixels.device)

    # ONE global noise field, cropped per tile. This is the anti-ghosting
    # measure: neighbouring tiles see IDENTICAL init noise (and identical
    # reference content) in their shared overlap, so at denoise 1.0 their
    # trajectories converge on essentially the same fine detail there —
    # the blend averages two near-identical renderings instead of two
    # independently-imagined ones (decorrelated per-tile seeds produced
    # "double eyelash" ghosting at seams). Deterministic samplers (euler,
    # the default) get the full benefit; ancestral samplers re-noise per
    # step in tile-local coordinates, so some residual seam softness is
    # expected there.
    global_noise = comfy.sample.prepare_noise(big_latent, seed, None)

    def _ramp(length_px: int, ov_px: int, device):
        r = torch.ones(length_px, dtype=torch.float32, device=device)
        n = min(ov_px, length_px)
        fade = (torch.arange(n, dtype=torch.float32, device=device) + 1.0) / (n + 1.0)
        fade = fade * fade * (3.0 - 2.0 * fade)   # smoothstep — softer crossover
        r[:n] = torch.minimum(r[:n], fade)
        r[-n:] = torch.minimum(r[-n:], fade.flip(0))
        return r

    def _accumulate(xs_grid, ys_grid):
        for y0 in ys_grid:
            for x0 in xs_grid:
                tw = min(tile_lx, nlw - x0)
                th = min(tile_ly, nlh - y0)
                tile_lat = big_latent[..., y0:y0 + th, x0:x0 + tw].clone().contiguous()
                tile_pos = _apply_reference(positive_base, tile_lat.clone(), tile_ref)
                noise = global_noise[..., y0:y0 + th, x0:x0 + tw].clone().contiguous()
                refined = _do_sample(
                    guider=ov_guider, sampler=ov_sampler, sigmas=ov_sigmas,
                    model=model, noise=noise,
                    steps=steps, cfg=cfg, sampler_name=sampler_name, scheduler=scheduler,
                    positive=tile_pos, negative=negative,
                    source_latent=tile_lat,
                    denoise=tile_denoise,
                    noise_mask=None,
                    callback=callback,
                    disable_pbar=disable_pbar,
                    seed=int(seed),
                )
                tile_px = _vae_decode(vae, refined).to(big_pixels.device, big_pixels.dtype)

                py0, px0 = y0 * ppl_y, x0 * ppl_x
                pth, ptw = tile_px.shape[1], tile_px.shape[2]
                wy = _ramp(pth, min_ov_px, big_pixels.device)
                wx = _ramp(ptw, min_ov_px, big_pixels.device)
                w = (wy.view(1, -1, 1, 1) * wx.view(1, 1, -1, 1)).to(big_pixels.dtype)
                out[:, py0:py0 + pth, px0:px0 + ptw, :] += tile_px * w
                wsum[:, py0:py0 + pth, px0:px0 + ptw, :] += w

    _accumulate(xs, ys)

    # Seam-erase pass: a SECOND tile grid offset by ~half a tile, so its
    # tiles are CENTRED on the first grid's seams (a tile start at the
    # midpoint of two grid-A starts puts the new tile's centre on the seam
    # between them). Both grids refine the same source and accumulate into
    # the same weighted-blend buffers, so where grid A is weakest — its seam
    # lines, where its tiles taper to low ramp weight — the offset tiles sit
    # at full weight and dominate, and vice versa. Any residual seam from one
    # grid is overwritten by the other grid's continuous tile interior.
    # Costs a second set of tiles (~2× sampling). Skipped when a tiny canvas
    # produced a single tile on either axis (nothing to offset).
    offset_tiles = 0
    if dual_grid and len(xs) > 1 and len(ys) > 1:
        xs_b = [round((xs[i] + xs[i + 1]) / 2) for i in range(len(xs) - 1)]
        ys_b = [round((ys[i] + ys[i + 1]) / 2) for i in range(len(ys) - 1)]
        offset_tiles = len(xs_b) * len(ys_b)
        print(f"[Angelo tiled-restore] seam-erase pass: offset grid "
              f"{len(xs_b)}x{len(ys_b)} ({offset_tiles} tiles)")
        _accumulate(xs_b, ys_b)

    final_pixels = out / wsum.clamp(min=1e-6)
    final_latent = _vae_encode(vae, final_pixels)
    return final_latent, final_pixels, n_tiles + offset_tiles


# Direction-aware instruction prepended to the outpaint conditioning when a
# CLIP is wired (same pattern as Smart Guided's location prefixes). Edit
# models (Klein / Qwen) follow explicit continue-don't-repeat instructions
# well — this is the main lever against the classic outpainting failure of
# duplicating the scene's subject into the new strip.
_OUTPAINT_INSTRUCTIONS = {
    "left":  "Extend the image to the left, continuing the scene and background naturally. Do not repeat or add new subjects. ",
    "right": "Extend the image to the right, continuing the scene and background naturally. Do not repeat or add new subjects. ",
    "up":    "Extend the image upward, continuing the scene and background naturally. Do not repeat or add new subjects. ",
    "down":  "Extend the image downward, continuing the scene and background naturally. Do not repeat or add new subjects. ",
    "all":   "Extend the image outward on all sides, continuing the scene and background naturally. Do not repeat or add new subjects. ",
}


def _outpaint_prepare(
    *,
    vae,
    current: torch.Tensor,
    current_pixels: torch.Tensor | None,
    direction: str,
    amount_px: int,
    overlap_px: int,
    protect_json: str = "",
) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
    """Build the init latent + mask for one outpaint pass.

    Pads the decoded canvas in `direction` by `amount_px` (snapped to a
    /16 multiple so both 8x and 16x VAEs land on integer latent cells),
    fills the new strip with edge-replicated pixels (a colour-continuity
    hint that costs nothing — at denoise 1.0 the masked region starts from
    pure noise on flow models anyway), VAE-encodes the padded canvas, then
    PASTES the original latent back over the old region so the existing
    image stays bit-exact in latent terms (no VAE round-trip drift — the
    same trick as Xtra-Fine's final blend).

    The mask covers the new strip at 1.0 plus a feathered band reaching
    `overlap_px` into the old image, so the seam is redrawn rather than
    butted. The pure-new region is re-hardened to 1.0 after the blur.

    Every extension spans the FULL current edge, which is what makes
    corners a non-problem: extend right then down, and the bottom strip
    spans the new wider canvas — the corner is generated by whichever
    direction runs second, with both neighbours as context.

    `protect_json` is the protect-brush payload — a JSON list of
    [x, y, r] circles in OLD-image pixel coords. Protected pixels are
    subtracted from the overlap band (clamped so the pure-new strip can
    never be protected), keeping e.g. a car at the frame edge frozen
    while the rest of the band still blends generously.

    Also returns the REFERENCE latent for edit models: the edge-adjacent
    band (~512px deep) for directional extensions — showing the model
    the texture/lighting to continue WITHOUT re-showing it the scene's
    subject (whole-image references invite the edit branch to reproduce
    the subject into the strip — the Smart Inpaint lesson again). "all"
    keeps the whole image (everything borders the new space).

    Returns (latent_init, mask, snapped_amount_px, reference_latent).
    """
    pixels = current_pixels if current_pixels is not None else _vae_decode(vae, current)
    B, H, W, C = pixels.shape
    amt = max(16, int(round(amount_px / 16.0)) * 16)
    pl, pr, pt, pb = {
        "left":  (amt, 0, 0, 0),
        "right": (0, amt, 0, 0),
        "up":    (0, 0, amt, 0),
        "down":  (0, 0, 0, amt),
        "all":   (amt, amt, amt, amt),
    }.get(str(direction), (0, amt, 0, 0))

    chw = pixels.movedim(-1, 1)
    chw = torch.nn.functional.pad(chw, (pl, pr, pt, pb), mode="replicate")
    new_pixels = chw.movedim(1, -1).contiguous()
    new_H, new_W = new_pixels.shape[1], new_pixels.shape[2]

    latent_init = _vae_encode(vae, new_pixels)
    nlh, nlw = latent_init.shape[-2], latent_init.shape[-1]
    px_per_lat_x = max(1, round(new_W / nlw))
    px_per_lat_y = max(1, round(new_H / nlh))
    lat_l = pl // px_per_lat_x
    lat_t = pt // px_per_lat_y
    lh, lw = current.shape[-2], current.shape[-1]
    latent_init[..., lat_t:lat_t + lh, lat_l:lat_l + lw] = current.to(
        device=latent_init.device, dtype=latent_init.dtype,
    )

    ov_lx = max(0, round(int(overlap_px) / px_per_lat_x))
    ov_ly = max(0, round(int(overlap_px) / px_per_lat_y))
    mask = torch.ones((1, nlh, nlw), dtype=torch.float32, device=latent_init.device)
    y0 = lat_t + (ov_ly if pt > 0 else 0)
    y1 = lat_t + lh - (ov_ly if pb > 0 else 0)
    x0 = lat_l + (ov_lx if pl > 0 else 0)
    x1 = lat_l + lw - (ov_lx if pr > 0 else 0)
    if y1 > y0 and x1 > x0:
        mask[..., y0:y1, x0:x1] = 0.0
    # strip_only = 1.0 over ONLY the genuinely-new pixels. It's the floor
    # the blur and the protect brush are both clamped against: nothing may
    # reduce the strip below full regeneration.
    strip_only = torch.ones_like(mask)
    strip_only[..., lat_t:lat_t + lh, lat_l:lat_l + lw] = 0.0
    if int(overlap_px) > 0:
        sigma = max(0.5, (ov_lx + ov_ly) / 4.0)
        mask = torch.maximum(_gaussian_blur_2d(mask, sigma), strip_only).clamp(0.0, 1.0)

    # Protect brush: subtract painted circles from the overlap band so
    # e.g. a car at the frame edge stays bit-exact frozen while the rest
    # of the band still blends. Circles arrive in OLD-image pixel coords;
    # offset by the pad, scale to latent cells, union, soften 1 cell.
    circles = []
    try:
        for item in json.loads(protect_json or "[]"):
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                circles.append((float(item[0]), float(item[1]), float(item[2])))
    except (ValueError, TypeError):
        circles = []
    if circles:
        pts = torch.tensor(circles, dtype=torch.float32, device=mask.device)
        cx = (pts[:, 0] + pl) / px_per_lat_x
        cy = (pts[:, 1] + pt) / px_per_lat_y
        rl = pts[:, 2] / math.sqrt(px_per_lat_x * px_per_lat_y)
        ys = torch.arange(nlh, device=mask.device, dtype=torch.float32).view(1, -1, 1)
        xs = torch.arange(nlw, device=mask.device, dtype=torch.float32).view(1, 1, -1)
        dist2 = (xs - cx.view(-1, 1, 1)) ** 2 + (ys - cy.view(-1, 1, 1)) ** 2
        protect = (dist2 <= (rl.view(-1, 1, 1) ** 2)).to(torch.float32).amax(dim=0).unsqueeze(0)
        protect = _gaussian_blur_2d(protect, 1.0).clamp(0.0, 1.0)
        mask = torch.maximum(mask * (1.0 - protect), strip_only).clamp(0.0, 1.0)

    # Reference for edit models: the edge-adjacent band (~512px deep) for
    # a directional extension; the whole image for "all" (see docstring).
    ref_latent = current
    if direction in ("left", "right", "up", "down"):
        d_x = max(8, round(512.0 / px_per_lat_x))
        d_y = max(8, round(512.0 / px_per_lat_y))
        if direction == "right":
            ref_latent = current[..., :, max(0, lw - d_x):]
        elif direction == "left":
            ref_latent = current[..., :, :min(lw, d_x)]
        elif direction == "down":
            ref_latent = current[..., max(0, lh - d_y):, :]
        else:  # up
            ref_latent = current[..., :min(lh, d_y), :]
    ref_latent = ref_latent.clone().contiguous()

    return latent_init, mask, amt, ref_latent


def _circle_mask_latent_direct(
    latent_h: int,
    latent_w: int,
    cx_latent: float,
    cy_latent: float,
    r_latent: float,
    device: torch.device,
) -> torch.Tensor:
    """Build a binary [1, latent_h, latent_w] mask with a filled circle
    centred on (cx_latent, cy_latent) of radius `r_latent`, all in
    latent-space coordinates. The caller is responsible for converting
    pixel-space click coords to latent space using the correct per-axis
    scale (image_dim_pixel → latent_dim) so this function stays VAE-
    agnostic.
    """
    ys = torch.arange(latent_h, device=device, dtype=torch.float32).view(-1, 1)
    xs = torch.arange(latent_w, device=device, dtype=torch.float32).view(1, -1)
    dist2 = (xs - cx_latent) ** 2 + (ys - cy_latent) ** 2
    mask = (dist2 <= r_latent * r_latent).to(torch.float32)
    return mask.unsqueeze(0)  # [1, H, W]


def _gaussian_blur_2d(mask: torch.Tensor, sigma_latent: float) -> torch.Tensor:
    """Separable gaussian blur on a [B, H, W] or [H, W] mask tensor.

    sigma_latent is in latent-space units.
    """
    if sigma_latent <= 0:
        return mask
    # Kernel covers ~±3σ. Force odd size.
    ksize = int(2 * math.ceil(3 * sigma_latent) + 1)
    half = ksize // 2
    x = torch.arange(ksize, device=mask.device, dtype=torch.float32) - half
    k1d = torch.exp(-0.5 * (x / sigma_latent) ** 2)
    k1d = k1d / k1d.sum()

    # Reshape mask to [N, 1, H, W]
    orig_ndim = mask.dim()
    if orig_ndim == 2:
        m = mask.unsqueeze(0).unsqueeze(0)
    elif orig_ndim == 3:
        m = mask.unsqueeze(1)
    elif orig_ndim == 4:
        m = mask
    else:
        raise ValueError(f"gaussian_blur_2d: unexpected mask ndim {orig_ndim}")

    kh = k1d.view(1, 1, 1, ksize)
    kv = k1d.view(1, 1, ksize, 1)

    m = torch.nn.functional.pad(m, (half, half, 0, 0), mode="replicate")
    m = torch.nn.functional.conv2d(m, kh)
    m = torch.nn.functional.pad(m, (0, 0, half, half), mode="replicate")
    m = torch.nn.functional.conv2d(m, kv)

    if orig_ndim == 2:
        return m.squeeze(0).squeeze(0)
    if orig_ndim == 3:
        return m.squeeze(1)
    return m


def _encode_loaded_image(vae, ref_json: str, resize_mode: str, target_mp: float):
    """Load an image the user picked via the Load Image button and
    VAE-encode it into a base latent.

    `ref_json` is a JSON {name, subfolder, type} ref returned by
    ComfyUI's /upload/image (falls back to treating the string as a bare
    input-dir filename). resize_mode is "keep" (native res) or "mp"
    (scaled to ~target_mp megapixels). In both cases dimensions are
    rounded to a multiple of 16 so any supported VAE (8x or 16x) is
    happy. Returns (latent samples, original pixels) — the pixels so a
    freshly loaded image previews as ITSELF, not its VAE round-trip.
    """
    import numpy as np
    from PIL import Image, ImageOps

    # Resolve the image reference to a path.
    name, subfolder, type_ = ref_json, "", "input"
    try:
        ref = json.loads(ref_json)
        name = ref.get("name") or ref.get("filename") or ""
        subfolder = ref.get("subfolder", "") or ""
        type_ = ref.get("type", "input") or "input"
    except (ValueError, TypeError):
        pass

    if type_ == "output":
        base_dir = folder_paths.get_output_directory()
    elif type_ == "temp":
        base_dir = folder_paths.get_temp_directory()
    else:
        base_dir = folder_paths.get_input_directory()
    path = os.path.normpath(os.path.join(base_dir, subfolder, name))
    if not path.startswith(os.path.normpath(base_dir)):
        raise ValueError("Angelo: invalid loaded-image path")
    if not os.path.exists(path):
        raise ValueError(f"Angelo: loaded image not found: {name}")

    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size

    if resize_mode == "mp" and target_mp > 0:
        cur_mp = (w * h) / 1.0e6
        if cur_mp > 0:
            s = math.sqrt(target_mp / cur_mp)
            w = int(round(w * s))
            h = int(round(h * s))

    # Round down to a multiple of 16 (divisible by both 8x and 16x VAEs).
    w = max(16, (w // 16) * 16)
    h = max(16, (h // 16) * 16)
    if (w, h) != img.size:
        img = img.resize((w, h), Image.LANCZOS)

    arr = np.array(img).astype(np.float32) / 255.0      # (H, W, 3)
    pixels = torch.from_numpy(arr)[None, ...]            # (1, H, W, 3)
    samples = _vae_encode(vae, pixels[:, :, :, :3])
    # Hand back the ORIGINAL pixels (matched to the latent's device) so the
    # load previews the real file. The VAE round-trip is near-lossless on
    # small images — looks like a plain load — but visibly lossy on large
    # ones, which reads as "it edited my image on load".
    return samples, pixels[:, :, :, :3].to(samples.device)


def _pixels_to_preview(image: torch.Tensor):
    """Save an already-decoded (B, H, W, C) float image batch to the temp
    directory in the same format PreviewImage uses. Returns
    (image_tensor, list_of_image_refs). Split out of _decode_to_preview
    for paths that already hold pixels (the gen-bundle base generation
    previews the GEN VAE's decode, not its edit-VAE round-trip)."""
    previewer = comfy_nodes.PreviewImage()
    ui = previewer.save_images(image, filename_prefix="Angelo_preview")
    return image, ui["ui"]["images"]


def _decode_to_preview(vae, latent_samples: torch.Tensor):
    """Decode the latent and save to the temp directory in the same
    format PreviewImage uses, so we can return the same ui dict shape.

    Returns (image_tensor, list_of_image_refs). Each image ref is a
    {filename, subfolder, type} dict.
    """
    image = _vae_decode(vae, latent_samples)  # (B, H, W, C) float in [0, 1]
    return _pixels_to_preview(image)


class AngeloRefine:
    """Click-to-refine sampler. See module docstring."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),

                # ===== Sampler Mode controls (top of native widget area) =====
                # Visible in the node body. When `mode == "Sampler Mode"`, the
                # toolbar is greyed and canvas clicks do nothing — this widget
                # group is the active one, producing the base latent from the
                # incoming latent via comfy.sample.sample. When mode flips to
                # Refinement, sampler_seed_control is auto-forced to "fixed"
                # so subsequent Queue presses don't regenerate the base.
                "mode": (["Sampler Mode", "Edit Mode"], {"default": "Sampler Mode",
                                                                "tooltip": "Sampler Mode: AngeloRefine acts "
                                                                           "like a KSampler — generates the "
                                                                           "base latent from the incoming "
                                                                           "(usually empty) latent. Toolbar "
                                                                           "and canvas clicks are inert. "
                                                                           "Edit Mode: click / paint / drag "
                                                                           "to refine or inpaint the cached "
                                                                           "base. Switching to Edit Mode auto-"
                                                                           "locks sampler_seed_control to "
                                                                           "'fixed' so the base stays stable."}),
                "sampler_denoise": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 1.0, "step": 0.05,
                                              "tooltip": "[Sampler Mode] Denoise level for the base "
                                                         "generation. 1.0 = generate fully from noise "
                                                         "(KSampler default). Lower = img2img-style from "
                                                         "the incoming latent."}),
                "sampler_seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                                         "control_after_generate": False,
                                         "tooltip": "[Sampler Mode] Seed for the base generation. "
                                                    "Controlled by sampler_seed_control after each run."}),
                "sampler_seed_control": (["fixed", "increment", "decrement", "randomize"],
                                         {"default": "randomize",
                                          "tooltip": "[Sampler Mode] After-generation seed behaviour. "
                                                     "Defaults to randomize so each base generation gets "
                                                     "a fresh seed. Auto-forced to 'fixed' (locked to the "
                                                     "seed that produced the base) when you switch to Edit "
                                                     "Mode so refine clicks stay stable."}),

                # ===== Edit Mode controls (driven from toolbar) =====
                # Hidden from the native widget area; the toolbar above the
                # preview canvas drives these. seed_control follows the same
                # fixed/random/increment/decrement pattern as sampler_seed.
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                                 "control_after_generate": False,
                                 "tooltip": "[Edit Mode] Seed for the refine pass. Hidden — "
                                            "controlled by the Seed input on the toolbar."}),
                "seed_control": (["fixed", "increment", "decrement", "randomize"],
                                 {"default": "randomize",
                                  "tooltip": "[Edit Mode] After-click seed behaviour. Hidden — "
                                             "controlled by the Seed Ctrl dropdown on the toolbar. "
                                             "Defaults to randomize so each refine click produces a "
                                             "fresh variation rather than repeating the same result."}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 100,
                                  "tooltip": "Match the model's expected step count. "
                                             "FLUX 2 Klein distilled = 4."}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1,
                                  "tooltip": "FLUX 2 Klein distilled uses CFG=1 (no negative)."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                "denoise": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 1.0, "step": 0.05,
                                      "tooltip": "How much noise to add back for the refinement. "
                                                 "0.3 = subtle touch-up, 0.6 = real redo, "
                                                 "0.9+ = essentially regenerate that region."}),

                "click_radius": ("INT", {"default": 96, "min": 8, "max": 1024, "step": 4,
                                         "tooltip": "Pixel-space radius of the refinement region. "
                                                    "Updated automatically by the JS click widget."}),
                "feather_radius": ("INT", {"default": 24, "min": 0, "max": 256, "step": 4,
                                           "tooltip": "Pixel-space gaussian blur applied to the mask "
                                                      "before sampling. Smooths the seam between the "
                                                      "refined region and the preserved surroundings. "
                                                      "Roughly half of click_radius is a good default."}),

                # JS-driven widgets. click_x/y in pixel space; -1 = no click yet.
                # click_seq increments per click to force ComfyUI to detect a change
                # and re-execute this node (otherwise identical inputs would skip).
                # image_w/h are the actual decoded image dimensions in pixels —
                # set by the JS whenever a fresh preview is loaded into the canvas,
                # so we can compute the correct pixel→latent scale without
                # hardcoding the VAE downscale factor (FLUX 2 is 16, FLUX 1 / SDXL
                # are 8, exotic VAEs vary).
                "click_x": ("INT", {"default": -1, "min": -1, "max": 16384}),
                "click_y": ("INT", {"default": -1, "min": -1, "max": 16384}),
                "click_seq": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFF}),
                "image_w": ("INT", {"default": 0, "min": 0, "max": 16384}),
                "image_h": ("INT", {"default": 0, "min": 0, "max": 16384}),
                # Undo bookkeeping: undo_seq increments when the user clicks
                # the Undo button. Python pops the last refined latent off
                # the history stack on each new undo_seq value.
                "undo_seq": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFF}),

                "reset": ("BOOLEAN", {"default": False,
                                      "tooltip": "Tick + re-queue to discard the cached refined "
                                                 "latent and start over from the incoming latent."}),
                # DEPRECATED as a control — always-on now. Kept declared
                # (not removed) so its slot in widgets_values stays put;
                # deleting it would shift every later widget's position
                # and drift old saved workflows on load. Hidden in the UI;
                # run() ignores the value and always decodes the preview.
                "auto_decode": ("BOOLEAN", {"default": True,
                                            "tooltip": "(Deprecated — preview always decodes now.)"}),

                # STEP 2 of the Fine Upscaling re-introduction: JS toggle
                # bar drives these. Both widgets are hidden from the node
                # UI; the green "Fine Upscale" toggle and the small "MP"
                # numeric input on the bar above the preview canvas are
                # the user-facing surface. Python's run() still prints the
                # received values so we can verify the JS bar correctly
                # drives them. SAMPLING IS STILL UNCHANGED — neither value
                # is read by any code path below the print. Step 3 wires
                # the actual crop+upscale+refine.
                "fine_upscaling": ("BOOLEAN", {"default": False,
                                               "tooltip": "When ON, the painted region is cropped, "
                                                          "bilinear-upscaled in latent space to hit "
                                                          "min_megapixels, refined, downscaled, and "
                                                          "composited back. Gives the model more "
                                                          "effective resolution on small painted "
                                                          "regions. Upscale capped at 8× linear "
                                                          "internally. Set via the Fine Upscale "
                                                          "toggle above the preview."}),
                "min_megapixels": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 4.0, "step": 0.1,
                                             "tooltip": "Target megapixels for the refine pass when "
                                                        "Fine Upscaling is on. Only used in that mode. "
                                                        "Set via the MP input above the preview."}),
                "max_upscale": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 16.0, "step": 0.5,
                                          "tooltip": "Hard cap on the linear upscale factor in Fine "
                                                     "Upscaling. 8.0 was the original internal default. "
                                                     "Lower values reduce smoothing artifacts from "
                                                     "extreme upscales at the cost of less effective "
                                                     "resolution gain. Set via the Max input above "
                                                     "the preview."}),
                "resize_method": (_FINE_UPSCALE_RESIZE_METHODS, {"default": "lanczos",
                                                                 "tooltip": "Pixel-space upscale method for "
                                                                            "Fine Upscale. lanczos is the "
                                                                            "default — sharpest detail recovery "
                                                                            "for natural images. bilinear is "
                                                                            "smooth (good for skin/faces); "
                                                                            "bicubic middle ground; nearest-"
                                                                            "exact preserves exact sample values; "
                                                                            "area/bislerp niche."}),
                "inpainting_mode": (["Refine", "Smart Inpaint", "Smart Guided Inpaint", "Outpaint"], {"default": "Refine",
                                                                  "tooltip": "How the painted region is treated.\n\n"
                                                                             "Refine — the painted region is partially "
                                                                             "denoised from the existing content. Best "
                                                                             "for refining what's already there (faces, "
                                                                             "hands, textures). Paint/click as normal.\n\n"
                                                                             "Smart Inpaint — adds NEW content where you "
                                                                             "drag a rectangle. Click+hold one corner, "
                                                                             "release at the opposite corner. Locks "
                                                                             "denoise=1.0, Fine Upscale=ON, Ctx Pad=0 "
                                                                             "(the right defaults for adding new subjects "
                                                                             "with an edit model — Klein 9B etc.). "
                                                                             "Reference_latents are auto-injected so the "
                                                                             "model's edit branch activates.\n\n"
                                                                             "Smart Guided Inpaint — no painting or boxes. "
                                                                             "Pick a location from the dropdown above the "
                                                                             "Area Prompt; it's prepended to your prompt "
                                                                             "(e.g. 'In the top left of the image, ...') "
                                                                             "and the edit model places the content there "
                                                                             "across the whole image. Locks denoise=1.0, "
                                                                             "Fine Upscale=OFF, Area Prompt=ON; press "
                                                                             "'Generate Guided Edit' to run.\n\n"
                                                                             "Outpaint — extend the canvas. Pick a direction "
                                                                             "(arrows, or click near an edge of the preview), "
                                                                             "review the result, Accept to commit. Accepting "
                                                                             "installs the new canvas as a fresh session base "
                                                                             "— history resets."}),

                # Hidden — DOM location dropdown (Smart Guided Inpaint)
                # drives this. Holds a LABEL key from
                # _GUIDED_LOCATION_PREFIXES; run() maps it to the prompt
                # prefix at encode time.
                "guided_location": ("STRING", {"default": "(none)", "multiline": False}),

                "fine_context_pad": ("INT", {"default": 64, "min": 0, "max": 512, "step": 8,
                                              "tooltip": "[Fine Upscale] Pixel-space padding around the "
                                                         "painted-shape bbox before cropping. Gives the "
                                                         "model surrounding context (skin, hair, "
                                                         "background) so a tight face mask at high denoise "
                                                         "still produces a coherent face that matches its "
                                                         "surroundings. The painted shape is unchanged — "
                                                         "only the area the model SEES grows. Outside the "
                                                         "painted shape, the surrounding pixels are "
                                                         "preserved (not refined). Larger pad = more "
                                                         "context + less effective resolution on the "
                                                         "painted area (bump MP to compensate)."}),

                "persistent_mask": ("BOOLEAN", {"default": False,
                                                "tooltip": "When ON, the last mask used (click point or paint "
                                                           "stroke) is held in the node. Pressing the standard "
                                                           "ComfyUI Queue button re-runs that same mask + a fresh "
                                                           "seed on the LATEST result, so each press builds further "
                                                           "on that region — use it to gradually morph an area over "
                                                           "several presses while the rest of the image stays "
                                                           "unchanged. (To re-roll the same edit on the ORIGINAL "
                                                           "image instead of building on it, use the Re-roll "
                                                           "button.) Toggled via the Persistent Mask button above "
                                                           "the preview."}),

                "paint_mode": ("BOOLEAN", {"default": False,
                                           "tooltip": "When ON, hold + drag on the preview paints a "
                                                      "freeform brush stroke (each dragged point is the "
                                                      "centre of a circle of click_radius; the union is "
                                                      "the refine mask). Release to submit. Single clicks "
                                                      "in paint mode work as one-point strokes. When OFF, "
                                                      "clicks behave as before (single-circle refine)."}),

                # Hidden — JS-set JSON list of [x_pixel, y_pixel] points
                # captured during a paint drag. Empty when no paint stroke
                # is pending (e.g. the user just single-clicked instead).
                "stroke_points": ("STRING", {"default": "", "multiline": False}),

                # Hidden — JS-set JSON list of [x1, y1, x2, y2] rectangles
                # in image-pixel coords, captured during Smart Inpaint
                # rectangle drags. The backend uses only the most recent
                # entry as the active mask.
                "rect_points": ("STRING", {"default": "", "multiline": False}),

                "area_prompt": ("BOOLEAN", {"default": False,
                                            "tooltip": "When ON, refinements encode the Area Prompt text "
                                                       "(typed in the box below the canvas) with the "
                                                       "connected CLIP and use it instead of the main "
                                                       "positive/negative — useful for reshaping a region "
                                                       "with a different prompt (e.g. main prompt = wide "
                                                       "shot, area prompt = \"detailed face, photorealistic "
                                                       "eyes\"). The click-and-mask behaviour is unchanged; "
                                                       "only the prompt differs. If CLIP isn't connected or "
                                                       "the area text is empty, the main positive/negative "
                                                       "are used. Forced ON in Smart Inpaint."}),

                # Hidden — DOM text box below the canvas drives these.
                # The Area Prompt input has a Pos/Neg toggle that decides
                # which of these two the textarea is editing. Encoded
                # with the connected CLIP at run time when area_prompt is
                # on. Negative is usually unused (Klein / CFG=1) but kept
                # for non-distilled models.
                "area_text_positive": ("STRING", {"default": "", "multiline": True}),
                "area_text_negative": ("STRING", {"default": "", "multiline": True}),

                # Hidden — the Load Image button drives these. loaded_image
                # is a JSON ref {name,subfolder,type} from /upload/image;
                # loaded_image_seq bumps on each load so run() knows a new
                # image arrived; resize_mode/target_mp come from the popup.
                "loaded_image": ("STRING", {"default": "", "multiline": False}),
                "loaded_image_seq": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
                "loaded_resize_mode": (["keep", "mp"], {"default": "keep"}),
                "loaded_target_mp": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 8.0, "step": 0.1}),

                # Hidden — the Detect (SAM 3 / YOLO) flow drives this. A JSON
                # list of silhouette polygons (image-pixel coords) for the
                # confirmed detection; in Refine mode it becomes the mask.
                # Cleared by the JS after the confirm run.
                "seg_polygon": ("STRING", {"default": "", "multiline": False}),

                # Hidden — the Re-roll button drives this. Bumps on each
                # press; run() pops the most recent edit to expose its
                # pre-edit base, rebuilds the SAME mask from the widgets
                # above, and re-samples with a fresh seed — swapping the
                # new variation in place of the last attempt. Declared LAST
                # so older saved workflows (which lack it) don't shift their
                # positional widgets_values; it just defaults to 0.
                "reroll_seq": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFF}),

                # Hidden — the Detect Shift/Alt touch-up brush drives this. A
                # base64-PNG mask at image resolution (white = masked) that, in
                # Refine, takes priority over seg_polygon. Lets the user add to
                # or subtract from (incl. holes) a SAM mask before committing.
                # Declared LAST so older saved workflows don't shift their
                # positional widgets_values; defaults to "".
                "seg_mask_png": ("STRING", {"default": "", "multiline": False}),

                # Redo (#6): the Redo button / Ctrl-Y bumps this. Python pushes
                # back onto the history stack the entry that Undo most recently
                # popped off (held on a per-node redo stack). Declared LAST so
                # older saved workflows don't shift their positional
                # widgets_values; defaults to 0.
                "redo_seq": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFF}),

                # Restore brush (#12): when ON, clicks / paint strokes / detect
                # masks RESTORE the painted region back to the session's base
                # image via a feathered latent blend — no sampling at all, so
                # it's effectively instant. The Lightroom "erase part of an
                # edit" gesture: refine a region, then brush back the bits of
                # the original you wanted kept (the face inside the helmet).
                # Honoured in Refine mode only; the Smart modes ignore it.
                # Declared LAST so older saved workflows don't shift their
                # positional widgets_values; defaults to False.
                "restore_mode": ("BOOLEAN", {"default": False,
                                             "tooltip": "When ON, clicks and paint strokes restore "
                                                        "the painted region back to the session base "
                                                        "image (a feathered latent blend — no "
                                                        "sampling, instant). Refine mode only. "
                                                        "Toggled via the Restore button on the "
                                                        "toolbar."}),

                # Prompt slots (#12): JSON persistence for the Area Prompt
                # slot strip ({active, slots: [{pos, neg} × 6]}). Pure UI
                # state — Python never reads it; the JS mirrors the active
                # slot's text into area_text_positive/negative, which remain
                # the encode source of truth. Declared LAST (see above).
                "area_prompt_slots": ("STRING", {"default": "", "multiline": False}),

                # Vary ×4: the Vary button bumps vary_seq. run() re-runs the
                # most recent edit's mask from its PRE-edit base four times
                # with different seeds, stashes the candidates in state
                # WITHOUT touching history, and ships one preview ref per
                # candidate (Angelo_vary_previews) to the JS chooser overlay.
                # Clicking a candidate sets vary_pick + bumps vary_pick_seq;
                # run() then swaps the chosen candidate in for the last
                # attempt — a pure restore, no sampling, mirroring Re-roll's
                # replace semantics. Cancelling the chooser needs no Python
                # round-trip (the stash is invalidated by the next edit).
                # Declared LAST (see redo_seq note).
                "vary_seq": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFF}),
                "vary_pick": ("INT", {"default": -1, "min": -1, "max": 15}),
                "vary_pick_seq": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFF}),

                # Outpaint: the JS arrows / edge-click bump outpaint_seq with
                # a direction + amount + overlap. run() pads the canvas, fills
                # the new strip at denoise 1.0, and STASHES the result for
                # review (Angelo_outpaint_preview → the JS Accept overlay) —
                # history is untouched. Accepting bumps outpaint_accept_seq;
                # run() then installs the new canvas as a FRESH session base
                # (Load-Image semantics: history resets — a documented fact of
                # outpainting, which is why nothing commits without review).
                # Cancel needs no Python round-trip. Declared LAST.
                "outpaint_seq": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFF}),
                "outpaint_dir": ("STRING", {"default": "right", "multiline": False}),
                "outpaint_amount": ("INT", {"default": 256, "min": 16, "max": 2048, "step": 16,
                                            "tooltip": "Outpaint: how many pixels to extend the "
                                                       "canvas by (snapped to /16). Set via the "
                                                       "Amount input on the Outpaint row."}),
                "outpaint_overlap": ("INT", {"default": 64, "min": 0, "max": 512, "step": 8,
                                             "tooltip": "Outpaint: feathered band reaching this many "
                                                        "pixels INTO the existing image, redrawn so "
                                                        "the seam blends instead of butting. Set via "
                                                        "the Overlap input on the Outpaint row."}),
                "outpaint_accept_seq": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFF}),

                # Outpaint protect brush: JSON list of [x, y, r] circles
                # (old-image pixel coords) painted on the canvas in Outpaint
                # mode. Subtracted from the overlap band so marked content
                # (a car at the frame edge) stays frozen while the rest of
                # the band blends. Declared LAST.
                "outpaint_protect": ("STRING", {"default": "", "multiline": False}),

                # Where the outpaint instruction sits relative to the user's
                # Area Prompt text: "prepend" = instruction first (default),
                # "append" = the user's text first, instruction after. Some
                # models weight the head of the prompt more heavily, so the
                # order can change reliability — the JS shows a live preview
                # of the combined prompt so there's no guessing. The exact
                # composition here MUST stay in lockstep with the JS preview
                # (syncOutpaintPromptPreview). Declared LAST.
                "outpaint_instruction_pos": (["prepend", "append"], {"default": "prepend"}),

                # Reference toggle: ON/OFF for the anchored-refine feature;
                # reference_strength (declared further down) is the 0–1
                # blend it applies when ON. Manual refines only — Quick
                # Photo Refine runs its own fixed recipe and ignores both.
                "refine_reference": ("BOOLEAN", {"default": False,
                                                 "tooltip": "Anchor refines to the current image "
                                                            "(edit models). The strength box "
                                                            "beside the toolbar button sets the "
                                                            "blend. Toggled via the Reference "
                                                            "button."}),

                # Quick Photo Refine: the ✨ button bumps this. One-shot
                # restoration pass — whole canvas, with the button's OWN
                # fixed settings (_QUICK_REFINE_PROMPT / _DENOISE / _REF).
                # A magic button: no toolbar box affects it except the
                # seed. Pushes a normal history entry so Undo covers it.
                # Declared LAST.
                "quick_refine_seq": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFF}),

                # Reference strength (Refine + Quick Photo Refine): a TRUE
                # 0–1 blend between the reference-anchored prediction and
                # the plain one — see _apply_reference. 0 = no reference
                # (classic refine), 1 = fully anchored. Declared LAST.
                "reference_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                                 "tooltip": "How strongly the current image anchors "
                                                            "a refine — a true blend between the "
                                                            "anchored and free predictions. 0 = "
                                                            "off, 1 = fully anchored. Set via the "
                                                            "Ref box on the toolbar."}),

                # ⬆ 2× Pixel: the button bumps this. A PURE pixel-space
                # lanczos 2× + re-encode — no AI, deterministic — committed
                # directly as a fresh session base (dimension change =
                # Load-Image semantics, history resets). The AI step is the
                # user's next move (e.g. ✨, which auto-tiles when large).
                "upscale_seq": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFF}),

                # ✨ prompt selector — which preset (or the Area Prompt box)
                # drives Quick Photo Refine. Declared LAST.
                "quick_prompt_mode": (_QUICK_REFINE_PROMPT_MODES,
                                      {"default": "Identity + Quality"}),

                # ⬇ Shrink Image: the button bumps shrink_seq with the chosen
                # shrink_scale (0–1). A PURE pixel-space AREA-resample
                # downscale + re-encode — no AI, deterministic — committed
                # directly as a fresh session base (dimension change =
                # Load-Image semantics, history resets). Declared LAST.
                "shrink_seq": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFF}),
                "shrink_scale": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 0.95, "step": 0.05}),

                # ✨ Lite mode — the toggle beside the prompt selector. ON =
                # the regular Quick Photo Refine recipe at a gentler denoise
                # (_QUICK_REFINE_LITE_DENOISE); everything else identical.
                # Declared LAST.
                "quick_lite": ("BOOLEAN", {"default": False}),

                # 🩹 Remove brush — the toggle beside Restore. ON = clicks /
                # paint strokes / Detect masks ERASE the painted object and
                # fill the hole with background, using the edit branch (blank
                # masked region + reference, fixed removal instruction, denoise
                # 1.0). Edit models only; Refine mode only. Declared LAST so
                # older saved workflows don't shift their positional
                # widgets_values; defaults to False.
                "remove_mode": ("BOOLEAN", {"default": False,
                                            "tooltip": "When ON, clicks / paint / Detect masks "
                                                       "REMOVE the painted object: the hole is "
                                                       "filled with surrounding background, a "
                                                       "removal pass reconstructs it, then a "
                                                       "neutral refine cleans it up (edit models "
                                                       "only). Refine mode only; toggled via the "
                                                       "Remove button on the toolbar."}),

            },
            "optional": {
                # CLIP / text encoder for the Area Prompt. Optional —
                # without it the area text is ignored and the main
                # positive/negative are used.
                "clip": ("CLIP",),
                # Base latent. OPTIONAL now: if the Load Image button is
                # used, the loaded photo becomes the base and no latent
                # input is needed. Sampler Mode still needs one (it defines
                # the output dimensions for a fresh generation).
                "latent": ("LATENT",),

                # Optional sampler-quintet overrides for users who want a
                # single source of truth for these values across the
                # workflow (issue #25). Pipe-pattern: a single ANGELO_OVERRIDES
                # slot, fed by the companion `Angelo — Overrides` node which
                # bundles steps / cfg / sampler / scheduler into one dict.
                # If unwired, the toolbar widget values flow through as
                # before. Picked over per-param forceInput slots because
                # forceInput on optional inputs proved unreliable across
                # ComfyUI versions for delivering wired primitive values.
                "overrides": ("ANGELO_OVERRIDES", {"tooltip": "Optional. Wire from an "
                                                              "'Angelo — Overrides' node to "
                                                              "drive steps / cfg / sampler / "
                                                              "scheduler from your workflow "
                                                              "instead of the toolbar — and/or "
                                                              "to supply a dedicated GENERATION "
                                                              "model (gen bundle): Sampler Mode "
                                                              "generates the base with it, then "
                                                              "the main model handles all edits. "
                                                              "Also carries an external Area "
                                                              "Prompt text (wire a wildcard / "
                                                              "prompt-generator node into the "
                                                              "Overrides node's area_prompt_text)."}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    # loaded_filename (#30) is APPENDED — output slots serialize by index
    # in saved workflow links, so new outputs go last, same rule as widgets.
    RETURN_TYPES = ("IMAGE", "LATENT", "IMAGE", "STRING")
    RETURN_NAMES = ("image", "latent", "source_image", "loaded_filename")
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "sampling/Angelo"
    DESCRIPTION = (
        "Click-to-refine sampler. Click a region in the preview to do "
        "extra denoising on just that area while the rest of the latent "
        "is preserved. Repeated clicks layer refinements on top of each other."
    )

    def run(
        self,
        model,
        positive,
        negative,
        vae,
        mode,
        sampler_denoise,
        sampler_seed,
        sampler_seed_control,
        seed,
        seed_control,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        click_radius,
        feather_radius,
        click_x,
        click_y,
        click_seq,
        image_w,
        image_h,
        undo_seq,
        reset,
        auto_decode,
        fine_upscaling,
        min_megapixels,
        max_upscale,
        resize_method,
        inpainting_mode,
        fine_context_pad,
        persistent_mask,
        paint_mode,
        stroke_points,
        rect_points,
        area_prompt,
        guided_location="(none)",
        area_text_positive="",
        area_text_negative="",
        loaded_image="",
        loaded_image_seq=0,
        loaded_resize_mode="keep",
        loaded_target_mp=1.5,
        seg_polygon="",
        reroll_seq=0,
        seg_mask_png="",
        redo_seq=0,
        restore_mode=False,
        area_prompt_slots="",
        vary_seq=0,
        vary_pick=-1,
        vary_pick_seq=0,
        outpaint_seq=0,
        outpaint_dir="right",
        outpaint_amount=256,
        outpaint_overlap=64,
        outpaint_accept_seq=0,
        outpaint_protect="",
        outpaint_instruction_pos="prepend",
        refine_reference=False,
        quick_refine_seq=0,
        reference_strength=0.0,
        upscale_seq=0,
        shrink_seq=0,
        shrink_scale=0.5,
        quick_lite=False,
        quick_prompt_mode="Identity + Quality",
        remove_mode=False,
        latent=None,
        clip=None,
        overrides=None,
        unique_id=None,
    ):
        # Optional upstream overrides: if an ANGELO_OVERRIDES bundle is
        # wired in, its non-None entries beat the toolbar widget values.
        # Kept here at the top so every downstream code path sees the
        # effective value without each one having to know about the override.
        # Carries sampler-config (#25), display flags (#21
        # disable_live_preview), and the custom-sampler trio (#8
        # guider/sampler/sigmas — all-or-nothing, applied inside _do_sample).
        disable_live_preview = False
        ov_guider = None
        ov_sampler = None
        ov_sigmas = None
        ov_gen_model = None
        ov_gen_positive = None
        ov_gen_negative = None
        ov_gen_vae = None
        ov_gen_steps = 25
        ov_gen_cfg = 5.0
        ov_gen_sampler_name = "euler"
        ov_gen_scheduler = "normal"
        if isinstance(overrides, dict):
            if overrides.get("steps") is not None:
                steps = overrides["steps"]
            if overrides.get("cfg") is not None:
                cfg = overrides["cfg"]
            if overrides.get("sampler_name") is not None:
                sampler_name = overrides["sampler_name"]
            if overrides.get("scheduler") is not None:
                scheduler = overrides["scheduler"]
            disable_live_preview = bool(overrides.get("disable_live_preview"))
            ov_guider = overrides.get("guider")
            ov_sampler = overrides.get("sampler")
            ov_sigmas = overrides.get("sigmas")
            # Gen bundle: a second model stack used ONLY by Sampler Mode's
            # base generation (see the Sampler Mode branch). Edit passes
            # always run on the main edit model regardless of this bundle.
            ov_gen_model = overrides.get("gen_model")
            ov_gen_positive = overrides.get("gen_positive")
            ov_gen_negative = overrides.get("gen_negative")
            ov_gen_vae = overrides.get("gen_vae")
            # Explicit None checks, NOT `or` fallbacks: gen_cfg 0.0 is a
            # legal widget value and falsy — `or` would silently sample at
            # the default instead.
            if overrides.get("gen_steps") is not None:
                ov_gen_steps = int(overrides["gen_steps"])
            if overrides.get("gen_cfg") is not None:
                ov_gen_cfg = float(overrides["gen_cfg"])
            if overrides.get("gen_sampler_name") is not None:
                ov_gen_sampler_name = overrides["gen_sampler_name"]
            if overrides.get("gen_scheduler") is not None:
                ov_gen_scheduler = overrides["gen_scheduler"]
            # External Area Prompt (#30): a wired STRING (wildcard resolver /
            # prompt generator) that REPLACES the Area Prompt box's positive
            # text for this run when non-empty. Substituted here at the top
            # so every downstream reader of area_text_positive picks it up
            # (area conditioning, Smart Guided prefixing, Outpaint, the ✨
            # "Use Area Prompt" mode). The area_prompt TOGGLE still decides
            # whether area text is used at all — this only swaps the text.
            ov_area_text = overrides.get("area_prompt_text")
            if ov_area_text is not None and str(ov_area_text).strip():
                area_text_positive = str(ov_area_text)

        node_id = str(unique_id)
        state = _STATE.get(node_id)

        # ===== Base latent selection =====
        # While an image is LOADED (loaded_image non-empty), it owns the
        # base and the wired `latent` input is ignored — until the user
        # hits Unload (which clears loaded_image). This stops the wired
        # latent from re-winning after a load (which made undo snap back
        # to the latent image). Priority:
        #   1. a freshly loaded image (encode it, force reset)
        #   2. an already-loaded image still active → keep the cached base
        #   3. the wired `latent` input
        #   4. the cached base (no latent, no load)
        loaded_ref = str(loaded_image).strip()
        loaded_active = bool(loaded_ref)
        loaded_seq = int(loaded_image_seq)
        # loaded_filename output (#30): the bare filename of the image
        # currently loaded via Load Image / drag-drop / paste (useful for
        # tagging on save), "" when the base is a generated latent. Stays
        # populated across runs because loaded_image holds the ref until
        # Unload clears it.
        loaded_filename = ""
        if loaded_active:
            try:
                _lref = json.loads(loaded_ref)
                loaded_filename = str(_lref.get("name") or _lref.get("filename") or "")
            except (ValueError, TypeError):
                loaded_filename = loaded_ref
        new_loaded = loaded_active and (
            state is None or state.get("loaded_seq") != loaded_seq
        )
        forced_base = None
        forced_pixels = None
        if new_loaded:
            forced_base, forced_pixels = _encode_loaded_image(
                vae, loaded_ref, str(loaded_resize_mode), float(loaded_target_mp)
            )

        # base_from_wired_latent: only the wired-`latent` path carries a
        # meaningful fingerprint for upstream-change detection. The cache /
        # loaded-image paths take incoming from history[0], which MUTATES as
        # the undo stack is capped (the original base is evicted after
        # _HISTORY_CAP refines), so a fingerprint comparison there is bogus
        # and would spuriously reset mid-session. See the need_reset note.
        # Cached-base fallbacks prefer source_latent over history[0]:
        # once _HISTORY_CAP evicts the true base, history[0] is a mid-
        # session edit — a Reset would land on it AND overwrite the
        # session base (Restore anchor, compare image, source_image
        # output) with it, unrecoverably.
        def _cached_base(st):
            src = st.get("source_latent")
            if src is not None:
                return src
            h0 = st["history"][0]
            return h0[0] if isinstance(h0, tuple) else h0

        base_from_wired_latent = False
        if forced_base is not None:
            incoming = forced_base
        elif loaded_active and state is not None and state.get("history"):
            # Loaded image still active — keep its base, ignore `latent`.
            incoming = _cached_base(state)
        elif latent is not None:
            incoming = latent["samples"]
            base_from_wired_latent = True
        elif state is not None and state.get("history"):
            incoming = _cached_base(state)
        else:
            raise ValueError(
                "Angelo: no base image — connect a `latent` input or use the "
                "Load Image button."
            )

        # Normalise the base latent to the dimensionality the MODEL expects.
        # ComfyUI's stock KSampler runs the latent through
        # fix_empty_latent_channels before sampling; Angelo calls
        # comfy.sample.sample directly, so it must do this itself. The
        # load-bearing case is video/temporal VAEs (Qwen Image Edit, Wan):
        # their diffusion model wants a 5D latent [B, C, T, H, W], and a 4D
        # [B, C, H, W] base (e.g. from a plain EmptyLatentImage, or any path
        # that didn't go through a Qwen-aware latent node) makes process_img
        # fail with "not enough values to unpack (expected 5, got 4)". This
        # unsqueezes the temporal axis (and channel-pads an empty latent) for
        # those models; for ordinary 4D models (FLUX/SDXL/SD) it is a no-op,
        # and an already-5D latent is returned unchanged. Done once here so
        # every downstream path — Sampler Mode, the Edit-Mode history seed,
        # the fingerprint, and the refine round-trips — sees the right shape.
        # incoming_pre_fix: kept for the gen-bundle path, which must run
        # fix_empty_latent_channels against the GEN model instead (its
        # latent channel count / dimensionality can differ from the edit
        # model's). The fingerprint below stays EDIT-model-fixed so Edit
        # Mode's next-run comparison sees the same value.
        incoming_pre_fix = incoming
        incoming = comfy.sample.fix_empty_latent_channels(model, incoming)
        incoming_fp = _latent_fingerprint(incoming)

        # ===== Smart Inpaint locks =====
        # The whole point of Smart Inpaint as a mode is to opinionate
        # the params that matter for "add new content in a rectangle"
        # workflows. Override the widget values up front so every code
        # path downstream sees the locked values regardless of what
        # the user set the toolbar to.
        # area_prompt ON makes Smart Inpaint encode the Area Prompt text
        # (typed below the canvas) with the connected CLIP and use it — a
        # separate "what to insert" prompt, kept independent of the main scene
        # prompt. While area_prompt is on the refine uses the Area text ONLY
        # (empty Area text → empty conditioning), never the main positive — see
        # the area-conditioning block further down in run().
        if inpainting_mode == "Smart Inpaint":
            denoise = 1.0
            fine_context_pad = 0
            fine_upscaling = True
            area_prompt = True
            # feather_radius is left under user control — a soft edge can
            # help blend the inserted content into the surroundings.
        elif inpainting_mode == "Smart Guided Inpaint":
            # No painting / boxes — the whole image is edited and the
            # location dropdown supplies a prompt prefix that tells the
            # edit model where to place the content. No region to crop
            # (Fine Upscale OFF), no mask edge to feather, and no mask to
            # persist (Persistent Mask is meaningless here).
            denoise = 1.0
            fine_context_pad = 0
            fine_upscaling = False
            area_prompt = True
            feather_radius = 0
            persistent_mask = False
        elif inpainting_mode == "Outpaint":
            # Canvas extension, not a masked edit — there's no held mask to
            # re-run, so the Persistent Mask queue hook must not fire. The
            # other edit params aren't read on the outpaint path (denoise is
            # fixed at 1.0 inside it; the JS dims their controls).
            persistent_mask = False

        # ===== Auto base generation on a fresh Edit-Mode queue =====
        # Queueing in Edit Mode with NO session, NO loaded image, and a
        # BLANK wired latent (a workflow opened saved-in-Edit-Mode, or a
        # ComfyUI restart) means the base was never generated — "editing"
        # would just decode an empty grey canvas. Run the Sampler-Mode base
        # generation instead (both queue buttons arrive here identically)
        # and flag the JS to flip the Mode widget so the UI matches what
        # actually ran. A wired NON-blank latent is left alone — that's the
        # downstream-refiner pattern (edit an upstream sampler's output
        # directly). A mid-session browser refresh is also left alone: the
        # server-side state survives it, so state is not None.
        auto_sampler = False
        if (mode == "Edit Mode"
                and state is None
                and not loaded_active
                and base_from_wired_latent
                and _latent_is_blank(incoming)):
            print("[Angelo] Edit Mode queued with no session and an empty "
                  "wired latent — running the base generation instead "
                  "(Mode flips to Sampler Mode).")
            mode = "Sampler Mode"
            auto_sampler = True

        # ===== Sampler Mode branch =====
        # Acts like a KSampler: take the incoming latent (typically empty),
        # run a fresh denoise pass with sampler_seed + sampler_denoise, cache
        # the result as the new base. All toolbar / canvas / refine logic is
        # skipped — those are Edit Mode concerns.
        if mode == "Sampler Mode":
            disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
            new_pixels = None
            gen_wired = (ov_gen_model is not None or ov_gen_positive is not None
                         or ov_gen_vae is not None or ov_gen_negative is not None)
            if gen_wired:
                # ===== Gen-bundle path =====
                # Generate the base with a DEDICATED generation model, then
                # cross the VAE boundary into the edit model's latent space:
                # gen sample → gen-VAE decode → edit-VAE encode. Everything
                # after this (history, refines, outpaint) runs on the edit
                # model exactly as if the pixels had come from Load Image.
                # Raise on partial wiring rather than silently falling back
                # — a user who wired gen_model clearly intended this path.
                if ov_gen_model is None or ov_gen_positive is None or ov_gen_vae is None:
                    raise ValueError(
                        "Angelo: the Overrides gen bundle needs gen_model + "
                        "gen_positive + gen_vae all wired (gen_negative is "
                        "optional). Wire the missing input(s) or disconnect "
                        "the bundle."
                    )
                if base_from_wired_latent:
                    # The wired latent defines the output dimensions. It must
                    # be in the GEN model's latent space (typically an
                    # EmptyLatentImage sized for the GEN VAE's ratio) — fix
                    # channels for the GEN model, then sanity-check the shape
                    # so a wrong-space latent fails HERE with a clear message
                    # instead of deep inside the gen UNet with a conv error.
                    gen_in = comfy.sample.fix_empty_latent_channels(
                        ov_gen_model, incoming_pre_fix
                    )
                    try:
                        gen_lf = ov_gen_model.get_model_object("latent_format")
                        gen_lf_channels = int(gen_lf.latent_channels)
                        gen_lf_5d = getattr(gen_lf, "latent_dimensions", 2) == 3
                    except Exception:
                        gen_lf_channels = None
                        gen_lf_5d = None
                    if gen_lf_5d is False and gen_in.ndim == 5:
                        # 5D latent (Qwen/Wan-native empty latent for the EDIT
                        # model) into a 4D-latent gen model: drop a singleton
                        # frame axis; anything else is a real mismatch.
                        if gen_in.shape[2] == 1:
                            gen_in = gen_in.squeeze(2)
                        else:
                            raise ValueError(
                                "Angelo: the wired latent is a multi-frame video "
                                "latent — gen_model is an image model and can't "
                                "generate from it. Wire an empty latent sized "
                                "for the GEN model instead."
                            )
                    if gen_lf_channels is not None and gen_in.shape[1] != gen_lf_channels:
                        raise ValueError(
                            f"Angelo: the wired latent has {gen_in.shape[1]} "
                            f"channels but gen_model expects {gen_lf_channels} — "
                            "it isn't in the gen model's latent space. Wire an "
                            "empty latent made for the GEN model (a plain "
                            "EmptyLatentImage for SD-class models). Note the "
                            "latent also sets the output size via the GEN "
                            "VAE's ratio, not the edit VAE's."
                        )
                else:
                    # No wired latent (cached base / loaded image): derive the
                    # gen-space latent by encoding the base PIXELS with the
                    # gen VAE. At denoise 1.0 only the shape matters; below
                    # 1.0 this is a true img2img on the current base.
                    base_px = forced_pixels
                    if base_px is None and state is not None:
                        base_px = state.get("source_pixels")
                    if base_px is None:
                        base_px = _vae_decode(vae, incoming)
                    gen_in = _vae_encode(ov_gen_vae, base_px[:, :, :, :3])
                # Announce the generation resolution: an empty latent authored
                # for the EDIT VAE's ratio silently generates at the wrong
                # size with a different-ratio gen VAE (64x64 latent = 1024px
                # at 16x but 512px at 8x) — undetectable from the latent
                # alone, so at least make it visible in the console.
                try:
                    _gr = int(ov_gen_vae.spacial_compression_decode())
                except Exception:
                    _gr = int(getattr(ov_gen_vae, "downscale_ratio", 8) or 8)
                print(f"[Angelo] gen bundle: generating "
                      f"{gen_in.shape[-1] * _gr}x{gen_in.shape[-2] * _gr} "
                      f"with the gen model")
                # #21: same live-preview opt-out as the standard path.
                callback = None if disable_live_preview else latent_preview.prepare_callback(
                    ov_gen_model, ov_gen_steps
                )
                noise = comfy.sample.prepare_noise(gen_in, sampler_seed, None)
                gen_negative_eff = (ov_gen_negative if ov_gen_negative is not None
                                    else _zero_out_conditioning(ov_gen_positive))
                # Deliberately NOT passing the guider trio: a wired GUIDER
                # wraps the EDIT model's conds; it can't drive the gen model.
                gen_latent = _do_sample(
                    guider=None, sampler=None, sigmas=None,
                    model=ov_gen_model, noise=noise,
                    steps=ov_gen_steps, cfg=ov_gen_cfg,
                    sampler_name=ov_gen_sampler_name, scheduler=ov_gen_scheduler,
                    positive=ov_gen_positive, negative=gen_negative_eff,
                    source_latent=gen_in,
                    denoise=sampler_denoise,
                    callback=callback,
                    disable_pbar=disable_pbar,
                    seed=sampler_seed,
                )
                gen_pixels = _vae_decode(ov_gen_vae, gen_latent)[:, :, :, :3]
                # Crop to a multiple of 16 so any edit VAE (8x or 16x) is
                # happy — same rule as _encode_loaded_image. An 8x gen VAE
                # can emit dims the 16x edit VAE can't take.
                h16 = max(16, (gen_pixels.shape[1] // 16) * 16)
                w16 = max(16, (gen_pixels.shape[2] // 16) * 16)
                if (h16, w16) != (gen_pixels.shape[1], gen_pixels.shape[2]):
                    gen_pixels = gen_pixels[:, :h16, :w16, :]
                new_latent = _vae_encode(vae, gen_pixels)
                # Preview + history pixels are the GEN VAE's decode — what
                # the generation model actually produced — not the edit-VAE
                # round-trip of it.
                new_pixels = gen_pixels.to(new_latent.device)
            else:
                # #21: skip the live-preview callback when the user has
                # disabled it via Overrides — keeps ComfyUI's global preview
                # on for other samplers but stops it squashing Angelo's editor.
                callback = None if disable_live_preview else latent_preview.prepare_callback(model, steps)
                noise = comfy.sample.prepare_noise(incoming, sampler_seed, None)
                # #8: _do_sample dispatches to the custom guider path when
                # ov_guider/ov_sampler/ov_sigmas are all wired, otherwise
                # falls back to the standard comfy.sample.sample(...) call.
                new_latent = _do_sample(
                    guider=ov_guider, sampler=ov_sampler, sigmas=ov_sigmas,
                    model=model, noise=noise,
                    steps=steps, cfg=cfg, sampler_name=sampler_name, scheduler=scheduler,
                    positive=positive, negative=negative,
                    source_latent=incoming,
                    denoise=sampler_denoise,
                    callback=callback,
                    disable_pbar=disable_pbar,
                    seed=sampler_seed,
                )
            # Replace the cache with the freshly-sampled base. Drops the
            # undo history (it's irrelevant — we have a brand-new image).
            #
            # CRITICAL: store the fingerprint of the INCOMING latent, not
            # the freshly-generated new_latent. The fingerprint is the
            # "did upstream change?" signal used by Edit Mode's
            # auto-reset check. If we stored new_latent's fingerprint,
            # Edit Mode would always see (cached_fp != incoming_fp)
            # → think upstream changed → wipe cache → refine the empty
            # latent. Storing incoming_fp means Refinement only resets
            # when the user actually rewires upstream.
            #
            # sampler_seed_at_run = the seed Python used. Sent back to JS
            # via the ui message so JS can (a) apply after-gen control
            # next, and (b) restore this value if the user later switches
            # the control to "fixed".
            _STATE[node_id] = {
                "history": [(new_latent, new_pixels)],
                "click_seq": click_seq,
                "undo_seq": undo_seq,
                # Anchor every action-seq to its current widget value so a
                # stale bump can't refire its gate against the new base.
                "redo_seq": redo_seq,
                "reroll_seq": reroll_seq,
                "vary_seq": vary_seq,
                "vary_pick_seq": vary_pick_seq,
                "outpaint_seq": outpaint_seq,
                "outpaint_accept_seq": outpaint_accept_seq,
                "quick_refine_seq": quick_refine_seq,
                "upscale_seq": upscale_seq,
                "shrink_seq": shrink_seq,
                "fingerprint": incoming_fp,
                "sampler_seed_at_run": int(sampler_seed),
                "loaded_seq": loaded_seq,
                # Source image (#3/#9): the session base, captured once so the
                # source_image output survives _HISTORY_CAP eviction of history[0].
                "source_latent": new_latent,
                "source_pixels": None,
            }
            out_latent = {"samples": new_latent}
            ui_msg = {
                "Angelo_preview": [],
                "Angelo_mode": ["Sampler Mode"],
                "Angelo_sampler_seed_at_run": [int(sampler_seed)],
            }
            if auto_sampler:
                # Tell the JS this run auto-generated the base so it can
                # flip the Mode widget to Sampler Mode (see onExecuted).
                ui_msg["Angelo_auto_sampler"] = [True]
            # Preview always decodes now (auto_decode deprecated). Gen-bundle
            # path already holds the gen VAE's pixels — preview those rather
            # than the edit-VAE round-trip of them.
            if new_pixels is not None:
                image, image_refs = _pixels_to_preview(new_pixels)
            else:
                image, image_refs = _decode_to_preview(vae, new_latent)
            ui_msg["Angelo_preview"] = image_refs
            # Freshly-generated base IS the source image — cache + emit it.
            # The preview refs double as the source refs for the JS
            # hold-to-compare key (the base and the preview are the same
            # image until the first edit).
            _STATE[node_id]["source_pixels"] = image
            _STATE[node_id]["source_preview_refs"] = image_refs
            ui_msg["Angelo_source_preview"] = image_refs
            return {"ui": ui_msg, "result": (image, out_latent, image, loaded_filename)}

        # ===== Edit Mode branch (existing behaviour) =====

        # Decide whether to (re)seed the cache from the incoming latent.
        # Reset on explicit toggle, first run, or fingerprint change —
        # but NOT on fingerprint change while persistent_mask is on, because
        # the whole point of persistent_mask is to keep refining the cached
        # latent across upstream re-rolls (the user's pressing Queue
        # specifically because they want a variation of the held region,
        # not a fresh image).
        #
        # The fingerprint check ONLY applies when the base is the wired
        # `latent` input (where a change really does mean "upstream produced
        # a fresh latent"). When the base comes from the cache / loaded
        # image, `incoming` is history[0] — which mutates as the undo stack
        # is capped (_HISTORY_CAP). Comparing against it there would
        # spuriously reset to a mid-stage latent after enough refines (the
        # "suddenly reverts to an earlier stage while painting" bug). Those
        # bases only change via explicit Load / Unload / Reset, all handled
        # by `reset` / `new_loaded` above — so the fingerprint isn't needed.
        state = _STATE.get(node_id)
        fingerprint_changed = (
            base_from_wired_latent
            and state is not None
            and state.get("fingerprint") != incoming_fp
        )
        need_reset = (
            reset
            or state is None
            or new_loaded
            or (fingerprint_changed and not persistent_mask)
        )

        if need_reset:
            # Any reset means the base just changed (load, unload, upstream
            # rewire). Anchor click_seq/undo_seq to the CURRENT widget
            # values so a click that was meant for the OLD base can't trip
            # the new-click gate and replay a stale inpaint onto the new
            # base. The user's next genuine click bumps click_seq and fires
            # normally.
            _STATE[node_id] = {
                # On a fresh LOAD, pair the base latent with the real loaded
                # pixels (forced_pixels) so the preview is the file itself,
                # not its VAE round-trip. Other resets have no pixels → None,
                # and the preview decodes the latent as before.
                "history": [(incoming.clone(), forced_pixels)],
                "click_seq": click_seq,
                "undo_seq": undo_seq,
                # Anchor every action-seq (see the Sampler Mode dict note).
                "redo_seq": redo_seq,
                "reroll_seq": reroll_seq,
                "vary_seq": vary_seq,
                "vary_pick_seq": vary_pick_seq,
                "outpaint_seq": outpaint_seq,
                "outpaint_accept_seq": outpaint_accept_seq,
                "quick_refine_seq": quick_refine_seq,
                "upscale_seq": upscale_seq,
                "shrink_seq": shrink_seq,
                "fingerprint": incoming_fp,
                "loaded_seq": loaded_seq,
                # Source image (#3/#9): capture the base once, independent of
                # history[0] (which mutates under _HISTORY_CAP eviction). On a
                # load this is the real loaded image, so hold-to-compare and
                # the source_image output show the file, not a round-trip.
                "source_latent": incoming.clone(),
                "source_pixels": forced_pixels,
            }
            state = _STATE[node_id]

        # Undo: if undo_seq advanced and we have history to pop, pop it.
        # We always keep at least one latent (the base / earliest refine)
        # so the preview stays valid.
        new_undo = undo_seq > 0 and undo_seq != state.get("undo_seq", -1)
        if new_undo:
            if len(state["history"]) > 1:
                # Move the popped entry onto the redo stack so Redo (#6) can
                # restore it. Bounded like the history stack.
                popped = state["history"].pop()
                redo = state.setdefault("redo_stack", [])
                redo.append(popped)
                if len(redo) > _HISTORY_CAP:
                    state["redo_stack"] = redo[-_HISTORY_CAP:]
            state["undo_seq"] = undo_seq
            state["quick_last"] = False
            # history[-1] is no longer the edit the mask widgets describe —
            # block Re-roll / Vary until a fresh edit re-arms them, and drop
            # any stashed Vary candidates (picking one after an Undo would
            # overwrite the wrong history entry).
            state["last_edit_kind"] = None
            state["vary_candidates"] = None
            # Undo is a PURE restore — pop the cached latent and decode it.
            # It must NEVER re-sample, or it would produce a different image
            # than the one being restored. Absorb the current click_seq so
            # the new-click gate below stays False on this run. This is
            # load-bearing because the Persistent Mask queue hook bumps
            # click_seq on EVERY queue, including the Undo button's — without
            # this, an undo while Persistent Mask is on would look like a new
            # click and re-run the last mask with the (since-randomized) seed,
            # restoring the WRONG result. (Harmless no-op when the hook didn't
            # bump it.)
            state["click_seq"] = click_seq

        # ===== Redo (#6): restore an entry Undo moved to the redo stack =====
        # A PURE restore — never re-samples — and runs BEFORE `current` is read
        # below, so the restored entry becomes history[-1]. Absorbs click_seq
        # for the same reason Undo does (the Persistent Mask queue hook bumps
        # it on every queue, including the Redo button's). No-op if the redo
        # stack is empty. A genuine new edit clears the redo stack (below).
        new_redo = redo_seq > 0 and redo_seq != state.get("redo_seq", -1)
        if new_redo:
            redo = state.get("redo_stack") or []
            if redo:
                state["history"].append(redo.pop())
                if len(state["history"]) > _HISTORY_CAP:
                    state["history"] = state["history"][-_HISTORY_CAP:]
            state["redo_seq"] = redo_seq
            state["quick_last"] = False
            # Same invalidation as Undo — the restored entry may not be the
            # edit the mask widgets currently describe.
            state["last_edit_kind"] = None
            state["vary_candidates"] = None
            state["click_seq"] = click_seq

        # ===== Vary pick: commit a chosen variation (pure restore) =====
        # The JS chooser sets vary_pick + bumps vary_pick_seq after a Vary
        # run stashed candidates. Swap the chosen candidate in for the last
        # attempt — replace, not stack, exactly like Re-roll — and clear the
        # stash. Never samples. Absorbs click_seq for the same queue-hook
        # reason as Undo/Redo.
        new_vary_pick = vary_pick_seq > 0 and vary_pick_seq != state.get("vary_pick_seq", -1)
        if new_vary_pick:
            state["vary_pick_seq"] = vary_pick_seq
            state["quick_last"] = False
            cands = state.get("vary_candidates")
            if cands and 0 <= int(vary_pick) < len(cands):
                state["history"][-1] = cands[int(vary_pick)]
                # Picking a variation is a real edit decision — the redo
                # branch (which predates it) no longer applies.
                state["redo_stack"] = []
            state["vary_candidates"] = None
            state["click_seq"] = click_seq

        # ===== Outpaint accept: commit the reviewed canvas as a NEW session =====
        # The JS Accept button bumps outpaint_accept_seq after the review
        # overlay. The stashed canvas becomes a fresh session base with
        # Load-Image semantics — history resets, the Restore anchor and the
        # hold-to-compare base move to the new canvas. This wholesale state
        # replacement is what keeps every same-shape assumption in the rest
        # of the node true across a dimension change.
        new_op_accept = (
            outpaint_accept_seq > 0
            and outpaint_accept_seq != state.get("outpaint_accept_seq", -1)
        )
        if new_op_accept:
            pend = state.get("outpaint_pending")
            if pend is not None:
                op_lat, op_px = pend
                _STATE[node_id] = {
                    "history": [(op_lat, op_px)],
                    "click_seq": click_seq,
                    "undo_seq": undo_seq,
                    "redo_seq": redo_seq,
                    "reroll_seq": reroll_seq,
                    "vary_seq": vary_seq,
                    "vary_pick_seq": vary_pick_seq,
                    "outpaint_seq": outpaint_seq,
                    "outpaint_accept_seq": outpaint_accept_seq,
                "quick_refine_seq": quick_refine_seq,
                "upscale_seq": upscale_seq,
                "shrink_seq": shrink_seq,
                    # Preserve the wired-latent fingerprint + load marker so
                    # the next run doesn't read the unchanged upstream as a
                    # fresh latent and blow away the canvas we just committed.
                    "fingerprint": state.get("fingerprint"),
                    "loaded_seq": state.get("loaded_seq"),
                    "sampler_seed_at_run": state.get("sampler_seed_at_run", int(sampler_seed)),
                    "refine_seed_at_run": (state.get("refine_seed_at_run")
                                           if state.get("refine_seed_at_run") is not None
                                           else int(seed)),
                    "source_latent": op_lat,
                    "source_pixels": op_px,
                }
                state = _STATE[node_id]
            else:
                state["outpaint_accept_seq"] = outpaint_accept_seq
                state["click_seq"] = click_seq

        # ===== Re-roll: redo the most recent edit with a fresh seed =====
        # The Re-roll button bumps reroll_seq (and sets a new seed) without
        # touching the mask widgets. Re-run the SAME mask on the SAME pre-
        # edit base and swap the result in place of the last attempt — so
        # the user can cycle seeds on one edit without reset → re-mask →
        # rerun. History is NOT popped up front (a Cancel/OOM mid-sample
        # would permanently lose the attempt): like Vary, the edit block
        # samples from history[-2] and the commit REPLACES history[-1]
        # only on success. Gated on last_edit_kind == "mask" — after an
        # Undo/Redo or a ✨ pass the mask widgets no longer describe
        # history[-1], so re-running them would corrupt the session
        # (e.g. a click_x=-1 fresh session fires a bogus corner edit).
        new_reroll = reroll_seq > 0 and reroll_seq != state.get("reroll_seq", -1)
        reroll_now = (
            new_reroll
            and len(state["history"]) > 1
            and inpainting_mode != "Outpaint"
            and state.get("last_edit_kind") == "mask"
        )
        state["reroll_seq"] = reroll_seq

        # Re-roll after ✨: the last entry is a ✨ pass, so there are no mask
        # widgets to re-run — route the press into a FRESH ✨ pass instead
        # ("re-roll the last thing, whatever it was"). Picked up by the quick
        # block below; quick_last's replace semantics make it swap the
        # previous ✨ result, exactly like pressing ✨ again. The JS already
        # forced a new random seed for the press, so it's a real variation.
        reroll_quick = (
            new_reroll
            and inpainting_mode == "Refine"
            and state.get("last_edit_kind") == "quick"
        )

        # ===== Vary ×4 gate =====
        # Re-run the most recent edit's mask from its PRE-edit base, four
        # times, WITHOUT touching history — candidates are stashed and the
        # user commits one via the chooser (vary_pick above). Needs a prior
        # edit (history > 1) whose mask widgets are still valid (same
        # last_edit_kind gate as Re-roll). Excluded when the Restore brush
        # is active (restores are deterministic — four identical candidates
        # would be noise); the JS blocks that combination too.
        new_vary = vary_seq > 0 and vary_seq != state.get("vary_seq", -1)
        vary_now = (
            new_vary
            and len(state["history"]) > 1
            and inpainting_mode != "Outpaint"
            and not (bool(restore_mode) and inpainting_mode == "Refine")
            and state.get("last_edit_kind") == "mask"
        )
        state["vary_seq"] = vary_seq

        hist_last = state["history"][-1]
        if isinstance(hist_last, tuple):
            current, current_pixels = hist_last
        else:
            current = hist_last
            current_pixels = None

        # ===== Outpaint (generate): pad + fill, stash for review =====
        # Driven by outpaint_seq from the JS arrows / edge-click, gated on
        # Outpaint mode. The result is NOT committed — it's stashed with a
        # preview ref so the JS can show the Accept / Try again / Cancel
        # overlay. The node keeps outputting the current canvas underneath.
        new_outpaint = (
            inpainting_mode == "Outpaint"
            and outpaint_seq > 0
            and outpaint_seq != state.get("outpaint_seq", -1)
        )
        state["outpaint_seq"] = outpaint_seq
        if new_outpaint:
            latent_init, op_mask, op_amt, op_ref = _outpaint_prepare(
                vae=vae, current=current, current_pixels=current_pixels,
                direction=str(outpaint_dir), amount_px=int(outpaint_amount),
                overlap_px=int(outpaint_overlap),
                protect_json=str(outpaint_protect),
            )
            # Conditioning. With a CLIP wired, the outpaint encodes its OWN
            # prompt: a direction-aware continue-don't-repeat instruction +
            # the Area Prompt text (if armed) describing the new space. The
            # main scene prompt is deliberately NOT used — conditioning the
            # strip on "a red car on a road" is precisely what paints a
            # second car into it. Without a CLIP we can't encode, so the
            # main conditioning flows through (with its known duplication
            # risk — wiring CLIP is the documented recommendation).
            if clip is not None:
                instr = _OUTPAINT_INSTRUCTIONS.get(
                    str(outpaint_dir), _OUTPAINT_INSTRUCTIONS["right"])
                # Instruction order: "prepend" (default) puts the instruction
                # first; "append" puts the user's Area text first. MUST stay
                # in lockstep with the JS preview (syncOutpaintPromptPreview).
                user_txt = str(area_text_positive).strip() if area_prompt else ""
                if user_txt and str(outpaint_instruction_pos) == "append":
                    op_text = user_txt.rstrip(" .,") + ". " + instr.strip()
                else:
                    op_text = instr + user_txt
                tokens_p = clip.tokenize(op_text)
                op_positive = clip.encode_from_tokens_scheduled(tokens_p)
                if area_prompt and str(area_text_negative).strip():
                    tokens_n = clip.tokenize(str(area_text_negative))
                    op_negative = clip.encode_from_tokens_scheduled(tokens_n)
                else:
                    op_negative = negative
            else:
                op_positive = positive
                op_negative = negative
            # Edge-band reference for edit models (Klein / Qwen): texture +
            # lighting to continue, without re-showing the scene's subject
            # (a whole-image reference invites the edit branch to reproduce
            # it into the strip). REPLACE (append=False) per the Smart
            # Inpaint lesson — non-edit models ignore the field entirely.
            op_positive = node_helpers.conditioning_set_values(
                op_positive, {"reference_latents": [op_ref]}, append=False,
            )
            # 5D temporal latents (Qwen/Wan) sample with a HARD mask — same
            # reasoning as Smart Inpaint (no clean soft-mask-during-denoise
            # behaviour on non-zero-mean latent spaces).
            op_sample_mask = op_mask
            if latent_init.ndim == 5:
                op_sample_mask = (op_mask >= 0.5).to(op_mask.dtype)
            op_seed = int(seed)
            callback = None if disable_live_preview else latent_preview.prepare_callback(model, steps)
            disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
            print(f"[Angelo outpaint] dir={outpaint_dir} amount={op_amt}px "
                  f"overlap={int(outpaint_overlap)}px "
                  f"new_latent=({latent_init.shape[-2]}x{latent_init.shape[-1]})")
            noise = comfy.sample.prepare_noise(latent_init, op_seed, None)
            op_refined = _do_sample(
                guider=ov_guider, sampler=ov_sampler, sigmas=ov_sigmas,
                model=model, noise=noise,
                steps=steps, cfg=cfg, sampler_name=sampler_name, scheduler=scheduler,
                positive=op_positive, negative=op_negative,
                source_latent=latent_init,
                denoise=1.0,
                noise_mask=op_sample_mask,
                callback=callback,
                disable_pbar=disable_pbar,
                seed=op_seed,
            )
            op_img, op_refs = _decode_to_preview(vae, op_refined)
            state["outpaint_pending"] = (op_refined, op_img)
            state["outpaint_preview_refs"] = op_refs
            state["click_seq"] = click_seq
            state["refine_seed_at_run"] = op_seed

        # ===== Quick Photo Refine: the one-click restoration recipe =====
        # Whole canvas, the internal restoration prompt, and the Reference
        # anchor — identity reconstructs from the reference while the
        # texture re-renders. The toolbar DENOISE applies (1.0 = full
        # re-render, the strongest restoration; lower = gentler clean-up);
        # area prompt / toggles are ignored. The seed also applies, so
        # seed_control = randomize lets you mash the button for variations.
        new_quick = (
            inpainting_mode == "Refine"
            and quick_refine_seq > 0
            and quick_refine_seq != state.get("quick_refine_seq", -1)
        )
        state["quick_refine_seq"] = quick_refine_seq
        # Re-roll pressed while the last entry is a ✨ pass → run a fresh ✨
        # (see the reroll_quick note above).
        if reroll_quick:
            new_quick = True
        if new_quick:
            # Re-roll semantics: if the latest entry is itself a ✨ result,
            # this press re-draws from the same source the last press used,
            # REPLACING it, instead of compounding on its output (anchor-
            # chaining converged to a fixed point and made mashing for
            # variations useless). Manual edits in between clear the flag,
            # so they're never discarded. History is NOT popped up front —
            # a Cancel/OOM mid-sample must not lose the previous ✨ result;
            # the source is read from history[-2] and history[-1] is
            # replaced only on success (commit below).
            qr_replace = bool(state.get("quick_last")) and len(state["history"]) > 1
            if qr_replace:
                _hl = state["history"][-2]
                if isinstance(_hl, tuple):
                    current, current_pixels = _hl
                else:
                    current, current_pixels = _hl, None
            # Prompt selection: preset from the ✨ selector, or the Area
            # Prompt text. The default preset keeps the model-tuned Qwen
            # variant on 5D latents; explicit presets are used verbatim.
            qp_mode = str(quick_prompt_mode)
            if qp_mode == "Use Area Prompt" and str(area_text_positive).strip():
                qr_prompt = str(area_text_positive)
            elif qp_mode == "Use Area Prompt":
                print("[Angelo quick-refine] Area Prompt empty — using the default preset")
                qr_prompt = (_QUICK_REFINE_PROMPT_QWEN if current.dim() == 5
                             else _QUICK_REFINE_PROMPT)
            elif qp_mode == "Identity + Quality":
                qr_prompt = (_QUICK_REFINE_PROMPT_QWEN if current.dim() == 5
                             else _QUICK_REFINE_PROMPT)
            else:
                qr_prompt = _QUICK_REFINE_PROMPTS.get(qp_mode, _QUICK_REFINE_PROMPT)
            print(f"[Angelo quick-refine] prompt mode: {qp_mode}")
            if clip is not None:
                tokens_q = clip.tokenize(qr_prompt)
                qr_base = clip.encode_from_tokens_scheduled(tokens_q)
            else:
                # No CLIP → can't encode the restoration prompt; the main
                # positive flows through. Still works, just less targeted.
                qr_base = positive
            qr_seed = int(seed)
            # Lite mode (toggle) just swaps the denoise — everything else is
            # the regular Quick Photo Refine recipe.
            qr_denoise = _QUICK_REFINE_LITE_DENOISE if bool(quick_lite) else _QUICK_REFINE_DENOISE
            callback = None if disable_live_preview else latent_preview.prepare_callback(model, steps)
            disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
            # Big canvases (e.g. after a 2× Restore Upscale) auto-route
            # through the tiled engine — same recipe per ~1MP tile, each
            # anchored to its own content, seams feathered — so the model
            # never samples a latent beyond the size it renders well at.
            canvas_mp = (image_w * image_h) / 1e6 if (image_w > 0 and image_h > 0) else 0.0
            # Tiling is for RESTORATION (every tile's job is local). An Area
            # Prompt is a SEMANTIC edit — changing a person can't be decided
            # per-tile (each tile would change them differently: the broken
            # half-swapped result). Area-Prompt mode therefore always runs
            # the single global pass, like the manual brush does.
            if (canvas_mp > _QUICK_REFINE_TILE_THRESHOLD_MP
                    and qp_mode != "Use Area Prompt"):
                print(f"[Angelo quick-refine] {canvas_mp:.1f}MP canvas — tiled restore pass")
                qr_refined, qr_pixels, _n = _tiled_restore_pass(
                    model=model, vae=vae,
                    current=current, current_pixels=current_pixels,
                    scale=1.0,
                    positive_base=qr_base, negative=negative,
                    seed=qr_seed, steps=steps, cfg=cfg,
                    sampler_name=sampler_name, scheduler=scheduler,
                    callback=callback, disable_pbar=disable_pbar,
                    ov_guider=ov_guider, ov_sampler=ov_sampler, ov_sigmas=ov_sigmas,
                    tile_denoise=qr_denoise,
                    dual_grid=True,   # seam-erase offset grid (option 4)
                )
            else:
                # ✨ v2: the whole image through the XTRA-FINE pipeline —
                # whole-canvas mask, full reference anchor, full denoise,
                # 1.3MP working target (sub-1.3MP canvases get internally
                # supersampled to 1.3MP, refined there, composited back),
                # feather 0. Magic button: NO toolbar box affects it (only
                # the seed, for variation mashing).
                qr_px_in = current_pixels if current_pixels is not None else _vae_decode(vae, current)
                Hq, Wq = qr_px_in.shape[1], qr_px_in.shape[2]
                qr_mask = torch.ones((1, current.shape[-2], current.shape[-1]),
                                     device=current.device, dtype=torch.float32)
                print(f"[Angelo quick-refine] whole-image Xtra-Fine pass "
                      f"({Wq}x{Hq}, target {_QUICK_REFINE_TARGET_MP}MP), "
                      f"denoise={qr_denoise}, ref={_QUICK_REFINE_REF}, seed={qr_seed}"
                      f"{' (Lite)' if bool(quick_lite) else ''}")
                qr_refined, qr_pixels = _refine_with_fine_upscaling(
                    model=model, vae=vae, current=current, current_pixels=qr_px_in,
                    mask=qr_mask,
                    scale_x=current.shape[-1] / Wq, scale_y=current.shape[-2] / Hq,
                    target_mp=_QUICK_REFINE_TARGET_MP,
                    max_linear=8.0, resize_method="lanczos",
                    context_pad_pixel=_QUICK_REFINE_CTX_PAD,
                    inpainting_mode="Refine",
                    reference_strength=_QUICK_REFINE_REF,
                    seed=qr_seed, steps=steps, cfg=cfg,
                    sampler_name=sampler_name, scheduler=scheduler,
                    positive=qr_base, negative=negative,
                    denoise=qr_denoise,
                    callback=callback, disable_pbar=disable_pbar,
                    ov_guider=ov_guider, ov_sampler=ov_sampler, ov_sigmas=ov_sigmas,
                )
            if qr_replace:
                # Replace the previous ✨ result (cancel-safe: it stayed in
                # history while this pass sampled).
                state["history"][-1] = (qr_refined, qr_pixels)
            else:
                state["history"].append((qr_refined, qr_pixels))
                if len(state["history"]) > _HISTORY_CAP:
                    state["history"] = state["history"][-_HISTORY_CAP:]
            state["quick_last"] = True
            state["redo_stack"] = []
            state["vary_candidates"] = None
            state["outpaint_pending"] = None
            state["click_seq"] = click_seq
            state["refine_seed_at_run"] = qr_seed
            # ✨'s whole-canvas mask has nothing to do with the click/paint
            # widgets — Re-roll / Vary must not re-run them over this result
            # (press ✨ again to re-roll it instead).
            state["last_edit_kind"] = "quick"
            current = qr_refined
            current_pixels = qr_pixels

        # ===== ⬆ 2× Pixel: pure pixel-space upscale (no AI) =====
        # Lanczos 2× of the decoded canvas, re-encoded, committed DIRECTLY
        # as a fresh session base — dimension change = Load-Image semantics
        # (history resets). Deterministic, so no review step. The AI
        # enhancement is the user's next, separate move — e.g. ✨ Quick
        # Photo Refine, which auto-tiles on the now-large canvas.
        new_upscale = (
            inpainting_mode == "Refine"
            and upscale_seq > 0
            and upscale_seq != state.get("upscale_seq", -1)
        )
        state["upscale_seq"] = upscale_seq
        if new_upscale:
            up_px_in = current_pixels if current_pixels is not None else _vae_decode(vae, current)
            in_H, in_W = up_px_in.shape[1], up_px_in.shape[2]
            out_W = max(16, int(round(in_W * 2.0 / 16.0)) * 16)
            out_H = max(16, int(round(in_H * 2.0 / 16.0)) * 16)
            up_chw = up_px_in.movedim(-1, 1)
            up_chw = comfy.utils.common_upscale(up_chw, out_W, out_H, "lanczos", "disabled")
            up_pixels = up_chw.movedim(1, -1).contiguous()
            up_latent = _vae_encode(vae, up_pixels)
            print(f"[Angelo 2x-pixel] {in_W}x{in_H} -> {out_W}x{out_H} (lanczos, no AI)")
            _STATE[node_id] = {
                "history": [(up_latent, up_pixels)],
                "click_seq": click_seq,
                "undo_seq": undo_seq,
                "redo_seq": redo_seq,
                "reroll_seq": reroll_seq,
                "vary_seq": vary_seq,
                "vary_pick_seq": vary_pick_seq,
                "outpaint_seq": outpaint_seq,
                "outpaint_accept_seq": outpaint_accept_seq,
                "quick_refine_seq": quick_refine_seq,
                "upscale_seq": upscale_seq,
                "shrink_seq": shrink_seq,
                "fingerprint": state.get("fingerprint"),
                "loaded_seq": state.get("loaded_seq"),
                "sampler_seed_at_run": state.get("sampler_seed_at_run", int(sampler_seed)),
                "refine_seed_at_run": (state.get("refine_seed_at_run")
                                       if state.get("refine_seed_at_run") is not None
                                       else int(seed)),
                "source_latent": up_latent,
                "source_pixels": up_pixels,
            }
            state = _STATE[node_id]
            current, current_pixels = up_latent, up_pixels

        # ===== ⬇ Shrink Image: pure pixel-space downscale =====
        # AREA resampling (box averaging) is the right tool for shrinking —
        # it averages the pixels being discarded, so it anti-aliases; lanczos
        # (great for UPscaling) can ring/alias on heavy reduction. No AI;
        # committed straight as a fresh session base (dimension change =
        # Load-Image semantics, history resets). shrink_scale is the chosen
        # factor; out dims snap to a multiple of 16 (latent alignment).
        new_shrink = (
            inpainting_mode == "Refine"
            and shrink_seq > 0
            and shrink_seq != state.get("shrink_seq", -1)
        )
        state["shrink_seq"] = shrink_seq
        if new_shrink:
            sk_px_in = current_pixels if current_pixels is not None else _vae_decode(vae, current)
            in_H, in_W = sk_px_in.shape[1], sk_px_in.shape[2]
            sc = max(0.05, min(0.95, float(shrink_scale)))
            out_W = max(16, int(round(in_W * sc / 16.0)) * 16)
            out_H = max(16, int(round(in_H * sc / 16.0)) * 16)
            sk_chw = sk_px_in.movedim(-1, 1)
            sk_chw = comfy.utils.common_upscale(sk_chw, out_W, out_H, "area", "disabled")
            sk_pixels = sk_chw.movedim(1, -1).contiguous()
            sk_latent = _vae_encode(vae, sk_pixels)
            print(f"[Angelo shrink] {in_W}x{in_H} -> {out_W}x{out_H} "
                  f"(scale {sc:.2f}, area, no AI)")
            _STATE[node_id] = {
                "history": [(sk_latent, sk_pixels)],
                "click_seq": click_seq,
                "undo_seq": undo_seq,
                "redo_seq": redo_seq,
                "reroll_seq": reroll_seq,
                "vary_seq": vary_seq,
                "vary_pick_seq": vary_pick_seq,
                "outpaint_seq": outpaint_seq,
                "outpaint_accept_seq": outpaint_accept_seq,
                "quick_refine_seq": quick_refine_seq,
                "upscale_seq": upscale_seq,
                "shrink_seq": shrink_seq,
                "fingerprint": state.get("fingerprint"),
                "loaded_seq": state.get("loaded_seq"),
                "sampler_seed_at_run": state.get("sampler_seed_at_run", int(sampler_seed)),
                "refine_seed_at_run": (state.get("refine_seed_at_run")
                                       if state.get("refine_seed_at_run") is not None
                                       else int(seed)),
                "source_latent": sk_latent,
                "source_pixels": sk_pixels,
            }
            state = _STATE[node_id]
            current, current_pixels = sk_latent, sk_pixels

        # Has the user clicked since our last execution for this node?
        # Never treat a queue-hook click_seq bump as an edit in Outpaint
        # mode — the canvas there is a direction picker, not a mask tool.
        new_click = (
            click_x >= 0
            and click_y >= 0
            and click_seq != state["click_seq"]
            and inpainting_mode != "Outpaint"
        )

        # Source latent for every edit is the current cached latent, so all
        # paths build ON TOP of the previous result:
        #   - normal clicks / paints iterate on the latest image
        #   - Persistent Mask Queue presses re-run the held mask on the
        #     latest result with a fresh seed, so the region gradually
        #     morphs further with each press (change something into
        #     something else)
        # Re-roll and Vary are the exceptions: they re-run the last edit's
        # mask from its PRE-edit base, so their sampling source is
        # redirected to history[-2] below. History itself stays untouched
        # until the commit (cancel-safety — a Cancel/OOM mid-sample must
        # never lose the existing attempt).

        if new_click or reroll_now or vary_now:
            # Re-roll and Vary ×4 both re-run the same mask widgets from
            # the UN-popped pre-edit base: history stays intact (the last
            # attempt remains history[-1]) and only the sampling source is
            # redirected to history[-2]. Re-roll REPLACES history[-1] on
            # success (commit below); Vary stashes candidates and commits
            # nothing until the user picks one in the chooser.
            if vary_now or reroll_now:
                hist_prev = state["history"][-2]
                if isinstance(hist_prev, tuple):
                    current, current_pixels = hist_prev
                else:
                    current, current_pixels = hist_prev, None
            # Pixel → latent conversion. The JS sends us the actual image
            # dimensions (image_w, image_h) so we can derive the per-axis
            # scale dynamically rather than hardcoding a VAE downscale
            # factor — that breaks for FLUX 2 (16×) vs FLUX 1 / SDXL (8×).
            # Fall back to 8× only if image dims weren't provided (shouldn't
            # happen in normal use).
            latent_h = current.shape[-2]
            latent_w = current.shape[-1]
            if image_w > 0 and image_h > 0:
                scale_x = latent_w / image_w
                scale_y = latent_h / image_h
            else:
                # Fallback for headless tests / direct node use without the
                # JS widget populating image_w/h. 8× is the most common VAE
                # downscale (FLUX 1, SDXL, SD1.5) but breaks for FLUX 2 (16×).
                scale_x = scale_y = 1.0 / 8.0
                print("[Angelo] warning: image_w/h not set by JS; "
                      "falling back to 8x VAE assumption — may be wrong for FLUX 2")

            # Geometric mean of the two axis scales (#28, from @KursatAs):
            # using scale_x alone maps the on-screen circle to an ellipse in
            # latent space whenever the axes scale differently (non-/16
            # image dims), refining a taller/wider region than was painted.
            # The geometric mean keeps the latent circle area-true on both
            # portrait and landscape images.
            scale_geom = math.sqrt(scale_x * scale_y)
            r_latent = max(1.0, click_radius * scale_geom)

            # Remove brush: Refine-only, edit models only, mutually exclusive
            # with Restore. Computed HERE (before the mask is built) because it
            # forces feather 0 and grows the mask below.
            remove_now = (bool(remove_mode) and not bool(restore_mode)
                          and inpainting_mode == "Refine")

            # Feather is forced to 0 for Remove regardless of the box — a hard
            # mask stops the object bleeding back through a soft edge.
            sigma_latent = 0.0 if remove_now else (
                (feather_radius * scale_geom) if feather_radius > 0 else 0.0)

            # Build the mask. Sources of mask shape, in priority:
            #   1. Smart Guided Inpaint: full-image (no region — the whole
            #      image is edited, location comes from the prompt prefix).
            #   2. Smart Inpaint: a single rectangle from rect_points.
            #   3. Refine + paint_mode + stroke points: union of brush
            #      circles along the drag path.
            #   4. Refine single-click: one circle at (click_x, click_y).
            if inpainting_mode == "Smart Guided Inpaint":
                mask = torch.ones((1, latent_h, latent_w),
                                  device=current.device, dtype=torch.float32)
            elif inpainting_mode == "Smart Inpaint":
                rect = _parse_rect_points(rect_points)
                if rect is not None:
                    mask = _rect_mask_latent(
                        latent_h, latent_w, rect,
                        scale_x, scale_y, current.device,
                    )
                else:
                    # No rectangle drawn yet — nothing to do. Caller
                    # already gates on click_seq change so this is the
                    # "user switched into Smart Inpaint without dragging
                    # a rect" case; fall through with an empty mask and
                    # let downstream noise_mask handling preserve the
                    # cached latent.
                    mask = torch.zeros((1, latent_h, latent_w),
                                       device=current.device, dtype=torch.float32)
            else:
                # Refine mask sources, in priority:
                #   1. a brushed touch-up raster mask (Detect Shift/Alt brush)
                #   2. a confirmed segmentation silhouette (SAM 3 / YOLO)
                #   3. a paint stroke (union of brush circles)
                #   4. a single click circle
                raster_png = (seg_mask_png or "").strip()
                seg_polys = _parse_seg_polygons(seg_polygon)
                stroke_pts = _parse_stroke_points(stroke_points) if paint_mode else []
                if raster_png:
                    mask = _raster_mask_latent(
                        latent_h, latent_w, raster_png, current.device,
                    )
                elif seg_polys:
                    mask = _polygons_mask_latent(
                        latent_h, latent_w, seg_polys,
                        scale_x, scale_y, current.device,
                    )
                elif stroke_pts:
                    mask = _stroke_mask_latent(
                        latent_h, latent_w,
                        stroke_pts, r_latent,
                        scale_x, scale_y, current.device,
                    )
                elif click_x >= 0 and click_y >= 0:
                    cx_latent = click_x * scale_x
                    cy_latent = click_y * scale_y
                    mask = _circle_mask_latent_direct(
                        latent_h, latent_w,
                        cx_latent, cy_latent, r_latent,
                        current.device,
                    )
                else:
                    # No click recorded (click_x = -1, e.g. right after a
                    # Reset). Nothing to mask — an empty mask makes the
                    # edit a no-op instead of a bogus corner-circle edit.
                    mask = torch.zeros((1, latent_h, latent_w),
                                       device=current.device, dtype=torch.float32)
            if sigma_latent > 0:
                mask = _gaussian_blur_2d(mask, max(0.5, sigma_latent))
                mask = mask.clamp(0.0, 1.0)

            # Remove: grow the mask outward so it fully covers the object plus
            # its immediate halo — a tight mask leaves a rim of the original
            # object behind.
            if remove_now:
                mask = _grow_mask_latent(mask, _REMOVE_MASK_GROW_FRAC)

            # Restore brush (#12): Refine-only — the Smart modes regenerate
            # content, so "restore to base" has no meaning there and the JS
            # dims the toggle. Decided before the conditioning block so a
            # restore never pays for a CLIP encode it won't use.
            restore_now = bool(restore_mode) and inpainting_mode == "Refine"

            # Remove: a single denoise-1.0 pass regenerates the masked region as
            # background using a removal instruction + a ZEROED reference (a real
            # hole in the reference) — see the pass loop. It ALWAYS runs on the
            # Xtra-Fine crop path (force it on): the removal happens on a high-res
            # crop with a crop-built zeroed reference, respecting the toolbar
            # ctx-pad / MP / method. The JS mirrors this (forces the Xtra-Fine
            # toggle on while Remove is active). Denoise box unused; pass is 1.0.
            if remove_now:
                fine_upscaling = True

            # Area-prompt conditioning selection. When area_prompt is on AND a
            # CLIP is connected, the refine uses the AREA text ONLY and NEVER
            # the main prompt — even when the Area text is empty, in which case
            # we encode the empty string (→ an empty conditioning) rather than
            # falling back to `positive`. This is load-bearing for the edit
            # modes: the main positive can carry whole-image reference_latents
            # (a Klein edit workflow's ReferenceLatent), and letting it leak in
            # made an empty-Area-Prompt Smart Inpaint reproduce the whole scene.
            # Negative area text is optional — falls back to the main negative
            # when empty (fine for CFG=1 / distilled models that ignore it).
            #
            # Smart Guided Inpaint prepends a location prefix to the positive
            # text (e.g. "In the top left of the image, ") so the edit model
            # places the content at the chosen spot.
            #
            # Without a CLIP we can't encode anything, so we must use the
            # already-encoded main conditioning (an unavoidable degenerate case
            # — area prompts need a CLIP connected).
            area_pos_text = str(area_text_positive)
            if inpainting_mode == "Smart Guided Inpaint":
                prefix = _GUIDED_LOCATION_PREFIXES.get(str(guided_location), "")
                area_pos_text = prefix + area_pos_text
            if area_prompt and clip is not None and not restore_now:
                tokens_p = clip.tokenize(area_pos_text)
                refine_positive = clip.encode_from_tokens_scheduled(tokens_p)
                if str(area_text_negative).strip():
                    tokens_n = clip.tokenize(str(area_text_negative))
                    refine_negative = clip.encode_from_tokens_scheduled(tokens_n)
                else:
                    refine_negative = negative
            else:
                refine_positive = positive
                refine_negative = negative

            # NOTE: the Remove brush ignores this selection — it drives its
            # single pass with a fixed background-fill instruction (never the
            # main/Area prompt, which would draw content back into the hole). See
            # the Remove pre-processing block.

            # Sample with the mask. Use the seed widget value as-is —
            # NO click_seq offset. Per-Queue variation (when persistent_mask
            # is on) and per-click variation (when user wants different
            # attempts on the same spot) are now controlled by seed_control:
            #   fixed     → same seed each run, repeatable result
            #   randomize → seed changes after each run (via JS after-gen),
            #               so each Queue / click produces a different result
            #   increment/decrement → +1/-1 each run
            # An older version of this code did `(seed + click_seq) & mask`
            # to fake per-click variation in the absence of after-gen
            # control. That broke "fixed means fixed" — even with the seed
            # widget locked, click_seq's increment still moved the effective
            # sampling seed. Now the user has explicit control.
            this_seed = int(seed)
            # #21: skip the live-preview callback when the user has
            # disabled it via Overrides — see Sampler Mode branch above.
            callback = None if disable_live_preview else latent_preview.prepare_callback(model, steps)
            disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

            # Source latent = the current cached latent for every path (see
            # the note above). Re-roll already exposed the pre-edit base as
            # `current` via its pop, so it needs no special-casing here.
            refine_source = current

            # ===== Inpainting Mode pre-processing =====
            # Refine: no pre-processing — partial denoise from the
            #   existing latent does the work.
            # Smart Inpaint: forces fine_upscaling=True up top, so its
            #   pre-processing (latent zero + reference_latents) lives
            #   inside _refine_with_fine_upscaling.
            # Smart Guided Inpaint: whole-image edit through the non-
            #   fine-upscale path below — inject the scene as
            #   reference_latents so Klein's edit branch keeps the rest
            #   of the image faithful while applying the location-guided
            #   change. POSITIVE ONLY (negative reference would steer
            #   CFG>1 samplers away from the scene).
            if inpainting_mode == "Smart Guided Inpaint":
                reference_latent = refine_source.clone()
                refine_positive = node_helpers.conditioning_set_values(
                    refine_positive, {"reference_latents": [reference_latent]}, append=True,
                )

            # Reference strength (Refine only): anchor identity/content from
            # the current image for the first reference_strength fraction of
            # the schedule (timestep-range construction — _apply_reference),
            # so high-denoise refines keep the subject. Plain path anchors
            # on the WHOLE current image here; the Xtra-Fine path anchors on
            # its upscaled crop instead (reference_strength passed below).
            # Restore never samples, so it's excluded.
            ref_strength = 0.0
            if (inpainting_mode == "Refine" and not restore_now and not remove_now
                    and bool(refine_reference)):
                ref_strength = max(0.0, min(1.0, float(reference_strength)))
            if ref_strength > 0.0 and not fine_upscaling:
                refine_positive = _apply_reference(
                    refine_positive, refine_source.clone(), ref_strength)

            # ===== Remove: build the remove-pass conditioning =====
            # There is NO pixel fill — the erase is done by the pass's full
            # denoise plus a ZEROED reference (a real hole in the reference
            # latent), so the edit model rebuilds the background from the
            # surroundings rather than anchoring to the object. refine_source
            # stays `current` (the true latent); at denoise 1.0 the masked region
            # is noised out anyway and outside it the original is kept. The mask
            # was already grown ~10% and hardened (feather 0) higher up.
            remove_pass_positive = None
            if remove_now and clip is not None:
                # Just the background-fill instruction — the ZEROED reference is built
                # from the CROP inside _refine_with_fine_upscaling (remove_mode).
                rm_txt = (_REMOVE_PROMPT_QWEN if current.dim() == 5 else _REMOVE_PROMPT)
                remove_pass_positive = clip.encode_from_tokens_scheduled(clip.tokenize(rm_txt))

            # One pass normally; four for Vary ×4. The conditioning above is
            # shared across passes — only the noise seed differs per pass.
            n_passes = 4 if vary_now else 1
            pass_results = []
            for _k in range(n_passes):
                pass_seed = this_seed if _k == 0 else (
                    (this_seed + _k * 1000003) & 0xFFFFFFFFFFFFFFFF
                )
                if restore_now:
                    # ===== Restore brush: heal back to the base, no sampling =====
                    # current = mask * base + (1-mask) * current, in latent space.
                    # Outside the (feathered) mask the current latent is kept
                    # bit-exact; inside it the session base is brought back. The
                    # result is pushed as a normal history entry so Undo / Redo /
                    # Re-roll keep working unchanged. (vary_now excludes
                    # restore_now, so n_passes is always 1 here.)
                    src = state.get("source_latent")
                    if src is None:
                        h0 = state["history"][0]
                        src = h0[0] if isinstance(h0, tuple) else h0
                    if tuple(src.shape) != tuple(current.shape):
                        # Shouldn't happen (source_latent resets with every base
                        # change), but never crash an edit session over it.
                        print("[Angelo restore] base/current latent shape mismatch "
                              f"({tuple(src.shape)} vs {tuple(current.shape)}) — restore skipped")
                        refined = current
                    else:
                        src = src.to(device=current.device, dtype=current.dtype)
                        alpha = mask
                        while alpha.dim() < current.dim():
                            alpha = alpha.unsqueeze(0)   # [1,H,W] → [1,1,H,W] (→ [1,1,1,H,W] for 5D)
                        refined = src * alpha + current * (1.0 - alpha)
                    refined_pixels = None
                elif remove_now:
                    # ===== Remove: denoise-1.0 removal on the Xtra-Fine crop ====
                    # Crop the region (toolbar ctx-pad / MP / method), regenerate
                    # the masked area as background from the removal instruction +
                    # a crop-built ZEROED reference (remove_mode), so the edit
                    # model rebuilds from the surrounding crop rather than
                    # redrawing the object. Always the crop path (forced above).
                    if remove_pass_positive is None:
                        # No CLIP → can't build the removal conditioning; leave
                        # the image untouched rather than erroring.
                        refined, refined_pixels = refine_source, None
                    else:
                        refined, refined_pixels = _refine_with_fine_upscaling(
                            model=model, vae=vae, current=refine_source, current_pixels=current_pixels, mask=mask,
                            scale_x=scale_x, scale_y=scale_y,
                            target_mp=float(min_megapixels),
                            max_linear=float(max_upscale),
                            resize_method=str(resize_method),
                            context_pad_pixel=int(fine_context_pad),
                            inpainting_mode="Refine",
                            remove_mode=True,
                            seed=pass_seed, steps=steps, cfg=cfg,
                            sampler_name=sampler_name, scheduler=scheduler,
                            positive=remove_pass_positive, negative=refine_negative,
                            denoise=1.0, callback=callback, disable_pbar=disable_pbar,
                            ov_guider=ov_guider, ov_sampler=ov_sampler, ov_sigmas=ov_sigmas,
                        )
                elif fine_upscaling:
                    refined, refined_pixels = _refine_with_fine_upscaling(
                        model=model, vae=vae, current=refine_source, current_pixels=current_pixels, mask=mask,
                        scale_x=scale_x, scale_y=scale_y,
                        target_mp=float(min_megapixels),
                        max_linear=float(max_upscale),
                        resize_method=str(resize_method),
                        context_pad_pixel=int(fine_context_pad),
                        inpainting_mode=str(inpainting_mode),
                        reference_strength=ref_strength,
                        seed=pass_seed, steps=steps, cfg=cfg,
                        sampler_name=sampler_name, scheduler=scheduler,
                        positive=refine_positive, negative=refine_negative,
                        denoise=denoise, callback=callback, disable_pbar=disable_pbar,
                        # #8: pass the custom-sampler trio through so the
                        # internal sample call inside Fine Upscale uses the
                        # same dispatch logic.
                        ov_guider=ov_guider, ov_sampler=ov_sampler, ov_sigmas=ov_sigmas,
                    )
                else:
                    noise = comfy.sample.prepare_noise(refine_source, pass_seed, None)
                    # #8: dispatched via _do_sample, see Sampler Mode note.
                    refined = _do_sample(
                        guider=ov_guider, sampler=ov_sampler, sigmas=ov_sigmas,
                        model=model, noise=noise,
                        steps=steps, cfg=cfg, sampler_name=sampler_name, scheduler=scheduler,
                        positive=refine_positive, negative=refine_negative,
                        source_latent=refine_source,
                        denoise=denoise,
                        noise_mask=mask,
                        callback=callback,
                        disable_pbar=disable_pbar,
                        seed=pass_seed,
                    )
                    refined_pixels = None
                pass_results.append((refined, refined_pixels))

            if vary_now:
                # Stash candidates + one preview ref each for the JS chooser.
                # History is untouched — the node keeps outputting the
                # existing last attempt until the user picks (vary_pick).
                state["vary_candidates"] = pass_results
                vary_refs = []
                for (lat_k, px_k) in pass_results:
                    if px_k is not None:
                        previewer = comfy_nodes.PreviewImage()
                        ui_k = previewer.save_images(px_k, filename_prefix="Angelo_vary")
                        vary_refs.append(ui_k["ui"]["images"][0])
                    else:
                        _img_k, refs_k = _decode_to_preview(vae, lat_k)
                        vary_refs.append(refs_k[0])
                state["vary_preview_refs"] = vary_refs
                state["click_seq"] = click_seq
                state["refine_seed_at_run"] = int(seed)
                # Reload the real last attempt for the node's outputs —
                # `current` was redirected to history[-2] for sampling.
                hist_last = state["history"][-1]
                if isinstance(hist_last, tuple):
                    current, current_pixels = hist_last
                else:
                    current, current_pixels = hist_last, None
            else:
                refined, refined_pixels = pass_results[0]
                if refined is refine_source:
                    # The pass bailed and returned the source unchanged
                    # (empty mask bbox, or Remove without a CLIP). Don't
                    # push a duplicate history entry or wipe the redo /
                    # vary stashes over a no-op — that killed Redo and made
                    # the next Undo appear dead. Reload history[-1] for the
                    # outputs (a re-roll redirected `current` to the pre-
                    # edit base for sampling).
                    print("[Angelo] edit was a no-op (empty mask or missing "
                          "CLIP) — history unchanged")
                    state["click_seq"] = click_seq
                    hist_last = state["history"][-1]
                    if isinstance(hist_last, tuple):
                        current, current_pixels = hist_last
                    else:
                        current, current_pixels = hist_last, None
                else:
                    if reroll_now:
                        # Cancel-safe replace: the attempt being re-rolled
                        # stayed in history while sampling ran; only now,
                        # on success, is it swapped for the fresh take.
                        state["history"][-1] = (refined, refined_pixels)
                    else:
                        state["history"].append((refined, refined_pixels))
                        if len(state["history"]) > _HISTORY_CAP:
                            state["history"] = state["history"][-_HISTORY_CAP:]
                    # A genuine new edit (click or re-roll) invalidates the
                    # redo branch and any stale Vary / Outpaint stash.
                    state["redo_stack"] = []
                    state["vary_candidates"] = None
                    state["outpaint_pending"] = None
                    state["click_seq"] = click_seq
                    state["refine_seed_at_run"] = int(seed)
                    state["quick_last"] = False
                    # Provenance: history[-1] is a mask-widget edit, so
                    # Re-roll / Vary may re-run those widgets. Cleared by
                    # Undo/Redo and the ✨ pass, whose masks don't match.
                    state["last_edit_kind"] = "mask"
                    current = refined
                    current_pixels = refined_pixels

        out_latent = {"samples": current}
        ui_msg = {
            "Angelo_preview": [],
            "Angelo_mode": ["Edit Mode"],
            "Angelo_refine_seed_at_run": [
                int(state["refine_seed_at_run"])
                if state.get("refine_seed_at_run") is not None else int(seed)
            ],
        }
        
        if current_pixels is not None:
            image = current_pixels
            previewer = comfy_nodes.PreviewImage()
            ui = previewer.save_images(image, filename_prefix="Angelo_preview")
            ui_msg["Angelo_preview"] = ui["ui"]["images"]
        else:
            image, image_refs = _decode_to_preview(vae, current)
            ui_msg["Angelo_preview"] = image_refs

        # Source image (#3/#9): the session base, decoded once and cached so
        # repeated edits don't re-decode it (and so it survives history[0]
        # eviction under _HISTORY_CAP). source_latent is set at every base
        # (re)establishment; the history[0] fallback only covers pre-existing
        # in-memory state from before this feature.
        source_image = state.get("source_pixels")
        if source_image is None:
            src_latent = state.get("source_latent")
            if src_latent is None:
                h0 = state["history"][0]
                src_latent = h0[0] if isinstance(h0, tuple) else h0
            source_image = _vae_decode(vae, src_latent)
            state["source_pixels"] = source_image

        # Hold-to-compare: ship a viewable ref to the BASE image so the JS
        # can flash it under the '\' key. Saved to temp once per session
        # (per base) and the refs cached — repeat edits reuse them.
        src_refs = state.get("source_preview_refs")
        if src_refs is None:
            previewer = comfy_nodes.PreviewImage()
            ui_src = previewer.save_images(source_image, filename_prefix="Angelo_source")
            src_refs = ui_src["ui"]["images"]
            state["source_preview_refs"] = src_refs
        ui_msg["Angelo_source_preview"] = src_refs

        # Vary ×4: ship the candidate previews exactly once (popped, not
        # read) — only the run that generated them opens the JS chooser;
        # later runs must not re-open it.
        vary_refs_out = state.pop("vary_preview_refs", None)
        if vary_refs_out:
            ui_msg["Angelo_vary_previews"] = vary_refs_out

        # Outpaint: same pop-once contract for the review overlay.
        op_refs_out = state.pop("outpaint_preview_refs", None)
        if op_refs_out:
            ui_msg["Angelo_outpaint_preview"] = op_refs_out

        return {"ui": ui_msg, "result": (image, out_latent, source_image, loaded_filename)}


class AngeloOverrides:
    """Companion node: bundle non-default Angelo settings into a single
    ANGELO_OVERRIDES dict, wire it into Angelo's `overrides` slot, and
    Angelo applies any non-sentinel field for that run. Originally added
    for #25 (sampler-config overrides). Also carries display flags like
    `disable_live_preview` (#21) so a single companion node covers every
    "drive this from the workflow instead of the toolbar" use case.

    Sampler-config fields use sentinel defaults ("(toolbar)" for the
    combo dropdowns, -1 for INT / FLOAT) meaning "don't override this
    one — use Angelo's toolbar value". Display flags default to the
    same behaviour Angelo has without an Overrides node connected, so
    leaving them alone is a no-op.

    Gen bundle (generation model): wire gen_model + gen_positive +
    gen_vae (gen_negative optional) to make Sampler Mode generate the
    base with a DIFFERENT model than the edit model — e.g. generate
    with FLUX dev / SDXL for look, then click-to-edit with FLUX 2 Klein
    or Qwen-Image-Edit. Angelo samples with the gen stack using the
    gen_* settings below, decodes with gen_vae, and re-encodes the
    pixels with the main edit VAE so the whole edit session stays in
    the edit model's latent space. Edit Mode never uses the gen bundle.
    Seed and denoise for the base generation stay on Angelo's toolbar
    (sampler_seed / Smpl Denoise) so the seed controls keep working.

    External Area Prompt (#30): wire any STRING output into
    area_prompt_text and, when non-empty, it replaces the text typed in
    Angelo's Area Prompt box each run — lets wildcard / prompt-generator
    nodes drive region edits. The Area Prompt toggle still gates whether
    area text is used at all.

    WIDGET ORDER IS APPEND-ONLY: this node lives in saved workflows
    with positional widgets_values — new widgets go after gen_scheduler,
    never between existing ones."""

    _SENTINEL = "(toolbar)"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps": ("INT", {"default": -1, "min": -1, "max": 100, "step": 1,
                                  "tooltip": "-1 = use Angelo's toolbar value."}),
                "cfg": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 30.0, "step": 0.1,
                                  "tooltip": "-1 = use Angelo's toolbar value."}),
                "sampler_name": ([cls._SENTINEL] + list(comfy.samplers.KSampler.SAMPLERS),
                                 {"default": cls._SENTINEL,
                                  "tooltip": "(toolbar) = use Angelo's toolbar value."}),
                "scheduler": ([cls._SENTINEL] + list(comfy.samplers.KSampler.SCHEDULERS),
                              {"default": cls._SENTINEL,
                               "tooltip": "(toolbar) = use Angelo's toolbar value."}),
                "disable_live_preview": ("BOOLEAN", {"default": False,
                                                     "tooltip": "When ON, suppress ComfyUI's "
                                                                "live latent preview for this "
                                                                "Angelo node only (#21). "
                                                                "Useful if the global preview "
                                                                "is squashing Angelo's editor "
                                                                "area but you still want previews "
                                                                "on your other samplers."}),
            },
            "optional": {
                # Custom-sampler trio (#8). Wire ALL THREE from a proper
                # SamplerCustomAdvanced-style chain (a GUIDER node like
                # CFGGuider/BasicGuider + a KSamplerSelect SAMPLER + a
                # BasicScheduler/Karras/etc. SIGMAS at denoise=1.0) to
                # replace Angelo's toolbar sampler/scheduler entirely.
                # When all three are wired, Angelo uses the guider path
                # and the toolbar's steps/cfg/sampler_name/scheduler are
                # ignored (sigmas encode the schedule; guider encodes
                # cfg + the sampling math). Partial wiring (e.g. sampler
                # without sigmas) silently falls through to the default.
                # Angelo's per-call denoise still applies — the sigmas
                # are tail-sliced to honour it (same as
                # SplitSigmasDenoise). Conds (Refine vs Area Prompt vs
                # Smart Inpaint reference_latents) are re-set on the
                # guider per call so the wired guider is generic.
                "guider": ("GUIDER",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                # Gen bundle: a SECOND model stack used ONLY for Sampler
                # Mode base generation. gen_positive/gen_negative must be
                # encoded with the GEN model's own CLIP — conditioning is
                # not portable across model families. gen_negative is
                # optional (falls back to a zeroed-out copy of
                # gen_positive, same as ConditioningZeroOut).
                "gen_model": ("MODEL", {"tooltip": "Generation model for Sampler Mode. "
                                                   "When wired (with gen_positive + "
                                                   "gen_vae), Angelo generates the base "
                                                   "with THIS model, then hands the "
                                                   "pixels to the edit model for the "
                                                   "editing session. Size Angelo's wired "
                                                   "latent for THIS model's VAE ratio "
                                                   "(a plain EmptyLatentImage for "
                                                   "SD-class models)."}),
                "gen_positive": ("CONDITIONING", {"tooltip": "Positive conditioning for "
                                                             "gen_model — encode with the "
                                                             "GEN model's CLIP, not the "
                                                             "edit model's."}),
                "gen_negative": ("CONDITIONING", {"tooltip": "Optional negative for "
                                                             "gen_model. Unwired = zeroed "
                                                             "gen_positive (like "
                                                             "ConditioningZeroOut)."}),
                "gen_vae": ("VAE", {"tooltip": "VAE matching gen_model — used to decode "
                                               "the generated base before it is re-"
                                               "encoded with the edit VAE."}),
                # ===== Gen-bundle settings (append-only widget tail) =====
                # Only read when gen_model is wired above. Real defaults, no
                # sentinels — there is no toolbar value to fall back to
                # because these describe the SECOND model. OPTIONAL, not
                # required: pre-gen-bundle API-format exports don't carry
                # these keys, and ComfyUI's prompt validation rejects a
                # missing REQUIRED input before build()'s signature defaults
                # can apply. Optional widgets look identical in the UI but
                # let old API prompts keep validating.
                "gen_steps": ("INT", {"default": 25, "min": 1, "max": 100, "step": 1,
                                      "tooltip": "[Gen bundle] Steps for the base "
                                                 "generation with gen_model."}),
                "gen_cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 30.0, "step": 0.1,
                                      "tooltip": "[Gen bundle] CFG for the base "
                                                 "generation with gen_model."}),
                "gen_sampler_name": (list(comfy.samplers.KSampler.SAMPLERS),
                                     {"default": "euler",
                                      "tooltip": "[Gen bundle] Sampler for the base "
                                                 "generation with gen_model."}),
                "gen_scheduler": (list(comfy.samplers.KSampler.SCHEDULERS),
                                  {"default": "normal",
                                   "tooltip": "[Gen bundle] Scheduler for the base "
                                              "generation with gen_model."}),
                # External Area Prompt (#30): wire a prompt-generator /
                # wildcard-resolver node's STRING output here and, when it's
                # non-empty, it REPLACES the text typed in Angelo's Area
                # Prompt box each run. forceInput = a socket only, no widget
                # — widgets_values order is untouched, and it's the wired-
                # node use case anyway (typing text belongs in the box).
                "area_prompt_text": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Optional. Wire a STRING output (wildcard "
                               "resolver, prompt generator, ...) here and "
                               "when it's non-empty it replaces the text in "
                               "Angelo's Area Prompt box for that run. The "
                               "Area Prompt toggle still decides whether "
                               "area text is used at all (the Smart modes "
                               "force it ON as usual). Note the on-node box "
                               "keeps showing its own text — the wired text "
                               "wins at run time."}),
            },
        }

    RETURN_TYPES = ("ANGELO_OVERRIDES",)
    RETURN_NAMES = ("overrides",)
    FUNCTION = "build"
    CATEGORY = "sampling/Angelo"
    DESCRIPTION = (
        "Bundle Angelo settings into one wire that drives them from "
        "your workflow instead of the toolbar: steps / cfg / sampler / "
        "scheduler (per-field opt-in via sentinel defaults), plus the "
        "disable_live_preview flag for #21. Also carries an optional "
        "gen bundle (gen_model / gen_positive / gen_negative / gen_vae "
        "+ gen_* settings): generate the base in Sampler Mode with a "
        "dedicated generation model, then edit with the main edit model. "
        "And an optional area_prompt_text input: wire any STRING output "
        "(wildcard / prompt-generator nodes) to replace the Area Prompt "
        "box's text at run time."
    )

    def build(self, steps, cfg, sampler_name, scheduler, disable_live_preview,
              gen_steps=25, gen_cfg=5.0, gen_sampler_name="euler", gen_scheduler="normal",
              guider=None, sampler=None, sigmas=None,
              gen_model=None, gen_positive=None, gen_negative=None, gen_vae=None,
              area_prompt_text=None):
        bundle = {
            "steps": steps if isinstance(steps, int) and steps >= 1 else None,
            "cfg": float(cfg) if isinstance(cfg, (int, float)) and cfg >= 0 else None,
            "sampler_name": sampler_name if sampler_name != self._SENTINEL else None,
            "scheduler": scheduler if scheduler != self._SENTINEL else None,
            "disable_live_preview": bool(disable_live_preview),
            # Custom-sampler trio (#8) — None if not wired, all-or-nothing
            # at use time inside _do_sample.
            "guider": guider,
            "sampler": sampler,
            "sigmas": sigmas,
            # Gen bundle — None if not wired. Angelo validates the
            # model+positive+vae triple at use time (Sampler Mode only).
            "gen_model": gen_model,
            "gen_positive": gen_positive,
            "gen_negative": gen_negative,
            "gen_vae": gen_vae,
            "gen_steps": int(gen_steps),
            "gen_cfg": float(gen_cfg),
            "gen_sampler_name": gen_sampler_name,
            "gen_scheduler": gen_scheduler,
            # External Area Prompt (#30) — None/empty means "use the box".
            "area_prompt_text": area_prompt_text,
        }
        return (bundle,)


NODE_CLASS_MAPPINGS = {
    "AngeloRefine": AngeloRefine,
    "AngeloOverrides": AngeloOverrides,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AngeloRefine": "Angelo — click to refine",
    "AngeloOverrides": "Angelo — Overrides",
}
