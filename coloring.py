from numba import cuda
import config
import numpy as np
import math
import xml.etree.ElementTree as ET
import cv2
import librosa
import scipy.signal as signal

inv_tau = 1 / (2 * math.pi)

_d_cols_cache: dict = {}

def _get_device_palette():
    key = config.PALETTE
    if key not in _d_cols_cache:
        cols = np.asarray(config.PALETTES[key], dtype=np.float32)
        if cols.shape[1] == 3:
            alpha = np.full((cols.shape[0], 1), 255.0, dtype=np.float32)
            cols = np.concatenate([cols, alpha], axis=1)
        _d_cols_cache[key] = cuda.to_device(cols)
    return _d_cols_cache[key]

_frame_buf_cache: dict = {}  # key: (h, w)

def _get_frame_bufs(h, w):
    key = (h, w)
    if key not in _frame_buf_cache:
        # Pinned host buffer — DMA-able, no page-lock overhead per frame
        pinned = cuda.pinned_array((h, w, 3), dtype=np.uint8)
        # Device output buffer — stays on GPU, reused every frame
        d_out = cuda.device_array((h, w, 3), dtype=np.uint8)
        _frame_buf_cache[key] = (pinned, d_out)
    return _frame_buf_cache[key]

def colorize(cache, flow, d_out):
    """
    cache: (d_lighting, d_base_phase, d_iters, max_iter)  — all device arrays
    d_out: preallocated device array (h, w, 3) uint8
    Returns d_out (still on device — caller copies to pinned buf)
    """
    d_lighting, d_base_phase, d_iters, max_iter = cache

    h, w = d_base_phase.shape
    d_cols = _get_device_palette()
    ncol = d_cols.shape[0]

    threads = (32, 8)
    blocks = (
        (w + threads[0] - 1) // threads[0],
        (h + threads[1] - 1) // threads[1],
    )

    colorize_core[blocks, threads](
        d_base_phase,
        d_lighting,
        d_iters,
        max_iter,
        d_cols,
        flow,
        config.BLACK_TOWARDS_END,
        config.BLACK_START,
        config.BLACK_STRENGTH,
        d_out,
        inv_tau,
        ncol,
        config.PERIOD
    )

    return d_out

@cuda.jit
def colorize_core(base_phase, lighting, iters, max_iter, cols, flow,
                  black_towards_end, black_start, black_strength,
                  out, inv_tau, ncol, period):

    x, y = cuda.grid(2)
    h, w = base_phase.shape
    if x >= w or y >= h:
        return

    if iters[y, x] >= max_iter:
        out[y, x, 0] = 0
        out[y, x, 1] = 0
        out[y, x, 2] = 0
        return

    bp = base_phase[y, x] /period
    lt = lighting[y, x]

    u = bp * inv_tau + flow
    u -= math.floor(u)

    idx = int(u * ncol)
    if idx >= ncol:
        idx = ncol - 1

    c = cols[idx]

    val = c[3] * 0.00392156862 * lt

    if black_towards_end and u > black_start:
        t = (u - black_start) / (1.0 - black_start)
        t = t * t * t
        darken = 1.0 - t * black_strength
        if darken < 0.0:
            darken = 0.0
        val *= darken

    out[y, x, 0] = int(c[0] * val)
    out[y, x, 1] = int(c[1] * val)
    out[y, x, 2] = int(c[2] * val)

def colorize_contour(cache, flow, d_out):
    d_lighting, d_base_phase, d_iters, max_iter = cache

    h, w = d_base_phase.shape
    d_cols = _get_device_palette()
    ncol = d_cols.shape[0]

    threads = (64, 8)
    blocks = (
        (w + threads[0] - 1) // threads[0],
        (h + threads[1] - 1) // threads[1],
    )

    colorize_contour_core[blocks, threads](
        d_base_phase,
        d_lighting,
        d_iters,
        max_iter,
        d_cols,
        flow,
        config.BLACK_TOWARDS_END,
        config.BLACK_START,
        config.BLACK_STRENGTH,
        d_out,
        inv_tau,
        ncol,
        config.PERIOD,
        config.CONTOURS,
        config.CONTOUR_LIFT,
        config.CONTOUR_WIDTH,
        config.FADE_WIDTH,
        config.SECTION_REPEAT,
        config.SECTION_DARKEN,
    )

    return d_out

@cuda.jit
def colorize_contour_core(base_phase, lighting, iters, max_iter, cols, flow,
                  black_towards_end, black_start, black_strength,
                  out, inv_tau, ncol, period,
                  contour_repeats, contour_lift, contour_width, fade_width,
                  section_repeat, section_darken):

    x, y = cuda.grid(2)
    h, w = base_phase.shape
    if x >= w or y >= h:
        return

    if iters[y, x] >= max_iter:
        out[y, x, 0] = 0
        out[y, x, 1] = 0
        out[y, x, 2] = 0
        return

    iter_val = iters[y, x] * iters[y, x]

    bp = base_phase[y, x] / period
    lt = lighting[y, x]

    u = bp * inv_tau + flow
    u -= math.floor(u)

    # ── which color slot are we in, and how far through it? ──────────────────
    scaled = u * ncol
    slot = int(scaled)
    if slot >= ncol:
        slot = ncol - 1
    t_slot = scaled - slot          # 0.0 → 1.0 within this color's region

    c = cols[slot]

    # ── big fade to black at the tail of the color slot ──────────────────────
    fade_start = 1.0 - fade_width
    darken = 1.0
    if t_slot > fade_start:
        fade_t = (t_slot - fade_start) / fade_width   # 0→1
        fade_t = fade_t * fade_t                       # ease-in
        darken = 1.0 - fade_t
        if darken < 0.0:
            darken = 0.0

    # ── section + contour ─────────────────────────────────────────────────────
    lift = 1.0
    if contour_repeats > 0 and section_repeat > 0:
        t_active = t_slot / fade_start              # 0→1 over non-fade region

        # which section are we in?
        t_section_raw = t_active * section_repeat
        section_idx = int(t_section_raw)
        if section_idx >= section_repeat:
            section_idx = section_repeat - 1
        t_section = t_section_raw - math.floor(t_section_raw)  # 0→1 within section

        # per-section darkening: section 0 = full brightness, last = darkest
        # section_darken controls how much darker each successive section gets
        # e.g. section_darken=0.25 means each section is 25% darker than previous
        section_scale = 1.0 - section_darken * section_idx

        # smooth fade at the end of each section (mini-fade)
        section_fade = 1.0
        section_fade_width = 0.15
        section_fade_start = 1.0 - section_fade_width
        if t_section > section_fade_start:
            sf_t = (t_section - section_fade_start) / section_fade_width
            sf_t = sf_t * sf_t
            section_fade = 1.0 - sf_t
            if section_fade < 0.0:
                section_fade = 0.0

        # tile contour bands within the section
        t_tiled = (t_section * contour_repeats) - math.floor(t_section * contour_repeats)

        if t_tiled < contour_width:
            band_frac = t_tiled / contour_width
            bump = math.sin(band_frac * math.pi)
            lift = 1.0 + (contour_lift - 1.0) * bump

        lift = lift * section_scale * section_fade

    # ── optional global black-towards-end ────────────────────────────────────
    global_darken = 1.0
    if black_towards_end and u > black_start:
        gt = (u - black_start) / (1.0 - black_start)
        gt = gt * gt * gt
        global_darken = 1.0 - gt * black_strength
        if global_darken < 0.0:
            global_darken = 0.0

    val = c[3] * 0.00392156862 * lt * lift * darken * global_darken

    out[y, x, 0] = min(255, int(c[0] * val))
    out[y, x, 1] = min(255, int(c[1] * val))
    out[y, x, 2] = min(255, int(c[2] * val))

_h_audio_raw = None
_audio_sr = None
_audio_frame = 0
_samples_per_frame = 0


def load_audio(path):
    global _h_audio_raw, _audio_sr, _samples_per_frame, _audio_frame

    audio, sr = librosa.load(path, mono=True, sr=None)

    # optional lowpass
    if getattr(config, "AUDIO_LOWPASS_ENABLE", False):
        nyq = 0.5 * sr
        b, a = signal.butter(
            4,
            config.AUDIO_LOWPASS_CUTOFF / nyq,
            btype='low'
        )
        audio = signal.filtfilt(b, a, audio)

    _h_audio_raw = audio
    _audio_sr = sr

    # exact frame → sample mapping
    _samples_per_frame = sr / config.FPS

    # reset frame timeline
    _audio_frame = 0


def colorize_audio(cache, flow, d_out):
    global _audio_frame, _h_audio_raw, _samples_per_frame

    if _h_audio_raw is None:
        load_audio(config.AUDIO_PATH)

    # derive audio window DIRECTLY from frame (no cursor, no drift possible)
    start = int(_audio_frame * _samples_per_frame)
    window = int(_samples_per_frame)

    chunk = _h_audio_raw[start:start + window]

    if len(chunk) == 0:
        energy = 0.0
    else:
        energy = (chunk * chunk).mean() ** 0.5
    for val in config.AUDIO_MODULATED_VALUES:
        adjusted_value=energy * config.AUDIO_PHASE_STRENGTH
        setattr(config,val,adjusted_value)

    _audio_frame += 1
    c=config.AUDIO_BASE_COLORING
    if c == "standard": return colorize(cache, flow, d_out)
    elif c == "contour": return colorize_contour(cache,flow,d_out)
    elif c == "image": return colorize_image(cache,flow,d_out)
    if _audio_frame % 60 == 0:
        t_video = _audio_frame / config.FPS
        t_audio = start / _audio_sr
        print(f"frame={_audio_frame}  video={t_video:.6f}s  audio={t_audio:.6f}s")

_d_image_cache = None

def _get_device_image():
    global _d_image_cache
    if _d_image_cache is None:
        img = cv2.imread(config.IMAGE_PATH)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        _d_image_cache = cuda.to_device(img)
    return _d_image_cache

def colorize_image(cache, flow, d_out):
    """
    cache: (d_lighting, d_base_phase, d_iters, max_iter)  — all device arrays
    d_out: preallocated device array (h, w, 3) uint8
    Returns d_out (still on device — caller copies to pinned buf)
    """
    d_lighting, d_base_phase, d_iters, max_iter = cache
 
    h, w = d_base_phase.shape
    d_img = _get_device_image()
 
    threads = (32, 8)
    blocks = (
        (w + threads[0] - 1) // threads[0],
        (h + threads[1] - 1) // threads[1],
    )
 
    colorize_image_core[blocks, threads](
        d_base_phase,
        d_lighting,
        d_iters,
        max_iter,
        d_img,
        flow,
        d_out,
        inv_tau,
        config.PERIOD
    )
 
    return d_out
 
@cuda.jit
def colorize_image_core(base_phase, lighting, iters, max_iter, img, flow,
                        out, inv_tau, period):
 
    x, y = cuda.grid(2)
    h, w = base_phase.shape
    if x >= w or y >= h:
        return
 
    if iters[y, x] >= max_iter:
        out[y, x, 0] = 0
        out[y, x, 1] = 0
        out[y, x, 2] = 0
        return
 
    img_h, img_w = img.shape[0], img.shape[1]
 
    # X axis: phase cycle position + flow
    bp = base_phase[y, x] / period
    u = bp * inv_tau + flow
    u -= math.floor(u)
    # triangle wave — mirrors 0→1→0 so wrap seam is always smooth
    u = 1.0 - math.fabs(u * 2.0 - 1.0)
 
    # Y axis: gradient angle normalized to 0-1
    if 0 < x < w - 1 and 0 < y < h - 1:
        gx = base_phase[y, x + 1] - base_phase[y, x - 1]
        gy = base_phase[y + 1, x] - base_phase[y - 1, x]
        angle = math.atan2(gy, gx)
    else:
        angle = 0.0
    v = (angle + math.pi) * (1.0 / (2.0 * math.pi))
    v -= math.floor(v)
    # triangle wave on Y as well
    v = 1.0 - math.fabs(v * 2.0 - 1.0)
 
    img_x = int(u * img_w) % img_w
    img_y = int(v * img_h) % img_h
 
    c = img[img_y, img_x]
    lt = lighting[y, x]
 
    out[y, x, 0] = min(int(c[0] * lt), 255)
    out[y, x, 1] = min(int(c[1] * lt), 255)
    out[y, x, 2] = min(int(c[2] * lt), 255)
