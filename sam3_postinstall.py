"""Angelo — SAM 3 post-install fixups.

Run by install_sam3_support.{bat,sh} after SAM 3 is installed/cloned, and
also on the "already installed" path so existing installs get fixed too.
Idempotent and safe to run repeatedly; always exits 0 so it can never block
the installer.

Patch 1 — dtype crash in SAM 3.1 (issue #31)
--------------------------------------------
`sam3/perflib/fused.py::addmm_act` casts its inputs to bfloat16 and returns
a bfloat16 tensor *unconditionally*. Angelo builds SAM 3 in float32 (its
known-good precision — see angelo_segment.py::_ensure_model), so that bf16
output then collides with the fp32 weights one layer downstream:

    "mat1 and mat2 must have the same dtype, but got BFloat16 and Float"

The fix records the caller's input dtype and casts addmm_act's result back
to it — a no-op when SAM 3 genuinely runs in bf16. `vitdet.py` binds the
symbol at import time (`from sam3.perflib.fused import addmm_act`), so a
runtime monkey-patch wouldn't reach that reference; the file on disk has to
be edited. Reported in Angelo issue #31.

Patch 2 — pkg_resources gone on setuptools >= 82 (issue #34)
------------------------------------------------------------
`sam3/model_builder.py` has a top-level `import pkg_resources`, whose only
use is a fallback that locates the BPE tokenizer when no bpe_path is given.
setuptools 82+ no longer ships the legacy pkg_resources module, so on fresh
Python 3.12/3.13 envs the import — and with it all of SAM 3 — dies with
"No module named 'pkg_resources'". Angelo always passes bpe_path explicitly
(angelo_segment.py::_find_bpe), so the fallback is dead code for us, but
the top-level import still kills `from sam3.model_builder import ...`.
The fix guards the import (None when missing) and rewrites the fallback
call sites to a plain __file__-relative path — byte-for-byte the same
location resource_filename resolves for a filesystem install.
"""

from __future__ import annotations

import os
import sys

_MARKER = "angelo-dtype-fix"

# The exact upstream (SAM 3.1) function we expect to find. If upstream
# changes this, the patch no-ops loudly rather than corrupting the file.
_BROKEN = (
    "def addmm_act(activation, linear, mat1):\n"
    "    if torch.is_grad_enabled():\n"
    '        raise ValueError("Expected grad to be disabled.")\n'
    "    self = linear.bias.detach()\n"
    "    mat2 = linear.weight.detach()\n"
    "    self = self.to(torch.bfloat16)\n"
    "    mat1 = mat1.to(torch.bfloat16)\n"
    "    mat2 = mat2.to(torch.bfloat16)\n"
    "    mat1_flat = mat1.view(-1, mat1.shape[-1])\n"
    "    if activation in [torch.nn.functional.relu, torch.nn.ReLU]:\n"
    "        y = addmm_act_op(self, mat1_flat, mat2.t(), beta=1, alpha=1, use_gelu=False)\n"
    "        return y.view(mat1.shape[:-1] + (y.shape[-1],))\n"
    "    if activation in [torch.nn.functional.gelu, torch.nn.GELU]:\n"
    "        y = addmm_act_op(self, mat1_flat, mat2.t(), beta=1, alpha=1, use_gelu=True)\n"
    "        return y.view(mat1.shape[:-1] + (y.shape[-1],))\n"
    "    raise ValueError(f\"Unexpected activation {activation}\")"
)

_FIXED = (
    "def addmm_act(activation, linear, mat1):  # " + _MARKER + "\n"
    "    if torch.is_grad_enabled():\n"
    '        raise ValueError("Expected grad to be disabled.")\n'
    "    out_dtype = mat1.dtype  # " + _MARKER + ": preserve caller dtype (SAM 3 runs fp32)\n"
    "    self = linear.bias.detach()\n"
    "    mat2 = linear.weight.detach()\n"
    "    self = self.to(torch.bfloat16)\n"
    "    mat1 = mat1.to(torch.bfloat16)\n"
    "    mat2 = mat2.to(torch.bfloat16)\n"
    "    mat1_flat = mat1.view(-1, mat1.shape[-1])\n"
    "    if activation in [torch.nn.functional.relu, torch.nn.ReLU]:\n"
    "        y = addmm_act_op(self, mat1_flat, mat2.t(), beta=1, alpha=1, use_gelu=False)\n"
    "        return y.view(mat1.shape[:-1] + (y.shape[-1],)).to(out_dtype)\n"
    "    if activation in [torch.nn.functional.gelu, torch.nn.GELU]:\n"
    "        y = addmm_act_op(self, mat1_flat, mat2.t(), beta=1, alpha=1, use_gelu=True)\n"
    "        return y.view(mat1.shape[:-1] + (y.shape[-1],)).to(out_dtype)\n"
    "    raise ValueError(f\"Unexpected activation {activation}\")"
)

_PKGRES_MARKER = "angelo-pkgres-fix"

# Upstream model_builder.py, exact bytes. The import is a lone top-level
# line; the bpe fallback block appears (identically) in three builders.
_PKGRES_BROKEN_IMPORT = "\nimport pkg_resources\n"

_PKGRES_FIXED_IMPORT = (
    "\ntry:  # " + _PKGRES_MARKER + ": setuptools >= 82 no longer ships pkg_resources\n"
    "    import pkg_resources\n"
    "except ImportError:\n"
    "    pkg_resources = None\n"
)

_PKGRES_BROKEN_BPE = (
    "        bpe_path = pkg_resources.resource_filename(\n"
    '            "sam3", "assets/bpe_simple_vocab_16e6.txt.gz"\n'
    "        )"
)

# Same file the resource_filename call resolves to for a normal on-disk
# install (model_builder.py sits in the sam3 package dir). `os` is already
# imported at the top of model_builder.py.
_PKGRES_FIXED_BPE = (
    "        bpe_path = os.path.join(  # " + _PKGRES_MARKER + "\n"
    "            os.path.dirname(os.path.abspath(__file__)),\n"
    '            "assets", "bpe_simple_vocab_16e6.txt.gz",\n'
    "        )"
)


def _candidate_paths(*relpath):
    """Yield possible locations of sam3/<relpath>, best-guess first.

    The installer clones into <this dir>/sam3, so that path covers the
    normal flow exactly. A light importlib fallback catches the case where
    sam3 was pip-installed elsewhere (vendored copy, site-packages)."""
    here = os.path.dirname(os.path.abspath(__file__))
    yield os.path.join(here, "sam3", "sam3", *relpath)

    # Fallback: locate the installed package without importing it (importing
    # sam3 pulls in torch + the whole model stack, which we don't need here).
    try:
        import importlib.util
        spec = importlib.util.find_spec("sam3")
        if spec is not None:
            for loc in (spec.submodule_search_locations or []):
                yield os.path.join(loc, *relpath)
    except Exception:
        pass


def _patch_fused(path):
    """Apply the dtype fix to one fused.py. Returns a short status string."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
    except OSError as e:
        return f"unreadable ({e})"

    if _MARKER in src:
        return "already patched"
    if _BROKEN not in src:
        return ("skipped — addmm_act doesn't match the known-broken SAM 3.1 "
                "version (upstream may have fixed it; nothing to do)")

    patched = src.replace(_BROKEN, _FIXED, 1)
    try:
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(patched)
    except OSError as e:
        return f"FAILED to write ({e})"
    return "patched"


def _patch_model_builder(path):
    """Apply the pkg_resources fix to one model_builder.py. Returns a short
    status string."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
    except OSError as e:
        return f"unreadable ({e})"

    if _PKGRES_MARKER in src:
        return "already patched"
    if _PKGRES_BROKEN_IMPORT not in src:
        return ("skipped — no top-level `import pkg_resources` found "
                "(upstream may have fixed it; nothing to do)")

    patched = src.replace(_PKGRES_BROKEN_IMPORT, _PKGRES_FIXED_IMPORT, 1)
    # The bpe fallback appears identically in each builder; rewrite them all.
    # Even if upstream reshapes those blocks, guarding the import alone keeps
    # SAM 3 importable — Angelo always passes bpe_path, never the fallback.
    n_bpe = patched.count(_PKGRES_BROKEN_BPE)
    if n_bpe:
        patched = patched.replace(_PKGRES_BROKEN_BPE, _PKGRES_FIXED_BPE)
    try:
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(patched)
    except OSError as e:
        return f"FAILED to write ({e})"
    return f"patched (import guarded, {n_bpe} bpe fallback(s) rewritten)"


# (relative path inside the sam3 package, patch function, what-if-it-fails note)
_PATCHES = [
    (("model_builder.py",), _patch_model_builder,
     "If SAM 3 later fails to import with \"No module named "
     "'pkg_resources'\", see issue #34."),
    (("perflib", "fused.py"), _patch_fused,
     "If Detect later crashes with a 'BFloat16 and Float' dtype error, "
     "see issue #31."),
]


def main():
    for relpath, patch_fn, fail_note in _PATCHES:
        seen = set()
        patched_any = False
        found_any = False
        for path in _candidate_paths(*relpath):
            path = os.path.normpath(path)
            if path in seen or not os.path.isfile(path):
                continue
            seen.add(path)
            found_any = True
            status = patch_fn(path)
            print(f"[Angelo/SAM3 post-install] {path}: {status}")
            if status.startswith(("patched", "already patched")):
                patched_any = True

        rel = "/".join(relpath)
        if not found_any:
            print(f"[Angelo/SAM3 post-install] no sam3/{rel} found "
                  "(nothing to patch — fine if SAM 3 isn't installed yet).")
        elif not patched_any:
            print(f"[Angelo/SAM3 post-install] note: sam3/{rel} looked "
                  f"different from the version we patch. {fail_note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
