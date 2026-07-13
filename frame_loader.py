"""
frame_loader.py

Unified fractal keyframe loader for Kalles Fraktaler Movie Maker.

.kfb loading is delegated to your existing kfb.KFB class, untouched.
.rfm loading (RFF's "Dynamic Map" binary format) is new here.

Both paths produce an object exposing the same attributes render.py already
reads: w, h, iter, smooth, max_iter, colour_div, log_zoom.

NOTE ON THE .rfm FORMAT
------------------------
Reverse-engineered from RFFDynamicMapBinary::read()/exportFile() in
Merutilm's RFF-2.0 C++ source. On-disk layout, in order:

    uint16   width
    uint16   height
    float32  logZoom
    uint64   period
    uint64   maxIteration
    float64 x (width*height)   one double per pixel (smoothed iteration count)

ASSUMPTION - VERIFY: IOUtilities::readAndDecode/encodeAndWrite is assumed to be
a plain uncompressed little-endian read/write. load_frame() checks the payload
size against width*height*8 bytes and raises a clear error if that's wrong.

ASSUMPTION - VERIFY: the pixel data is assumed width-major, same as KFB's own
`.reshape((w,h)).T` convention. Flip RFM_COLUMN_MAJOR if a rendered .rfm frame
looks transposed/mirrored relative to a .kfb frame of the same location.

NOTE ON FILE NUMBERING
------------------------
RFF's exporter numbers dynamic-map exports in the opposite direction from
Kalles Fraktaler's .kfb exporter (id 0 = deepest keyframe, not shallowest).
discover_frames() sorts .kfb descending (matches existing behavior) and .rfm
ascending, so files[0] is always the shallow/start frame either way. Flip the
entry in _REVERSE_ORDER if a real RFF export folder turns out not to follow
this convention.
"""

import struct
from pathlib import Path

import numpy as np
import mpmath as mp

from kfb import KFB  # your existing, working .kfb parser -- left untouched

# Flip this if .rfm frames come out transposed/mirrored relative to .kfb frames.
RFM_COLUMN_MAJOR = False

# Sort direction per format so files[0] is always the shallow/start frame.
_REVERSE_ORDER = {
    ".kfb": True,   # existing behavior: sort ascending, then reverse
    ".rfm": True,  # RFF numbers deepest-first, so ascending is already correct
}

# uint16 w, uint16 h, float32 logZoom, uint64 period, uint64 maxIteration
_RFM_HEADER = struct.Struct("<HHfQQ")

SUPPORTED_EXTENSIONS = (".kfb", ".rfm")


class RFMFrame:
    """In-memory representation of one .rfm keyframe. Same attribute surface
    as a loaded kfb.KFB object, so render.py's build_render_cache() etc. don't
    need to know or care which format a given frame came from."""
    __slots__ = ("path", "w", "h", "iter", "smooth", "max_iter", "colour_div", "log_zoom")

    def __init__(self, path, w, h, iter_arr, smooth_arr, max_iter, colour_div, log_zoom):
        self.path = path
        self.w = w
        self.h = h
        self.iter = iter_arr
        self.smooth = smooth_arr
        self.max_iter = max_iter
        self.colour_div = colour_div
        self.log_zoom = log_zoom


def _log_zoom_from_filename(path):
    """Best-effort log10(zoom) parse from a filename's trailing _<zoom> segment."""
    name = Path(path).stem
    try:
        zoom = mp.mpf(name.split("_")[-1])
    except Exception:
        return None
    if zoom is None or not mp.isfinite(zoom) or zoom <= 0:
        return None
    return float(mp.log10(zoom))


def _load_kfb(path):
    k = KFB(str(path))
    k.read()
    return k


def _load_rfm(path):
    data = Path(path).read_bytes()
    header_size = _RFM_HEADER.size  # 24 bytes

    if len(data) < header_size:
        raise ValueError(f"File too small to contain an .rfm header: {path}")

    w, h, log_zoom_f32, period, max_iter = _RFM_HEADER.unpack_from(data, 0)

    expected_payload = w * h * 8  # float64 per pixel
    actual_payload = len(data) - header_size
    if actual_payload != expected_payload:
        raise ValueError(
            f"Payload size mismatch in {path}: expected {expected_payload} bytes "
            f"({w}x{h} float64) but found {actual_payload}. This means "
            f"IOUtilities::readAndDecode/encodeAndWrite does more than a plain "
            f"binary write (likely compression) -- this loader's format "
            f"assumption is wrong. Send over src/ui/IOUtilities.h from RFF-2.0 "
            f"and the decode step can be fixed."
        )

    raw = np.frombuffer(data, dtype="<f8", count=w * h, offset=header_size)

    if RFM_COLUMN_MAJOR:
        iterations = raw.reshape((w, h)).T.copy()
    else:
        iterations = raw.reshape((h, w)).copy()

    # RFF's dynamic map is one continuous double per pixel (already a smoothed
    # iteration count) -- there's no separate int/fractional split like KFB.
    smooth = iterations
    iter_arr = np.floor(iterations).astype(np.int64)

    log_zoom = float(log_zoom_f32)
    if not (log_zoom == log_zoom) or log_zoom == 0.0:  # NaN or unset
        fallback = _log_zoom_from_filename(path)
        log_zoom = fallback if fallback is not None else None

    return RFMFrame(
        path=path, w=w, h=h,
        iter_arr=iter_arr, smooth_arr=smooth,
        max_iter=int(max_iter), colour_div=int(period) if period else 1,
        log_zoom=log_zoom,
    )


_LOADERS = {
    ".kfb": _load_kfb,
    ".rfm": _load_rfm,
}


def discover_frames(folder, extension=None):
    """
    Find fractal keyframe files in `folder`.

    extension: None auto-detects whichever supported extension is present
               (raises if the folder mixes both -- keep each sequence in its
               own folder). Pass ".kfb" or ".rfm" explicitly to force one.

    Returns (files, ext), sorted per-format via _REVERSE_ORDER so files[0] is
    always the shallow/start frame regardless of which exporter wrote them.
    """
    folder = Path(folder).expanduser().resolve()
    if not folder.exists():
        raise ValueError(f"[discover_frames] Folder does not exist: {folder}")

    exts = [extension] if extension is not None else SUPPORTED_EXTENSIONS

    found = {}
    for ext in exts:
        files = sorted(folder.glob(f"*{ext}"))
        if files:
            found[ext] = files

    if not found:
        raise ValueError(
            f"[discover_frames] No {'/'.join(SUPPORTED_EXTENSIONS)} files "
            f"found in: {folder}"
        )
    if len(found) > 1:
        raise ValueError(
            f"[discover_frames] Folder contains more than one keyframe format "
            f"({', '.join(found)}) -- put each format in its own folder: {folder}"
        )

    (ext, files), = found.items()
    if _REVERSE_ORDER.get(ext, True):
        files.reverse()
    return files, ext


def load_frame(path):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"[load_frame] File not found: {path}")

    ext = path.suffix.lower()
    loader = _LOADERS.get(ext)
    if loader is None:
        raise ValueError(
            f"[load_frame] Unsupported extension '{ext}' for {path}. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    try:
        return loader(path)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"[load_frame] Failed to load: {path}\n{e}")