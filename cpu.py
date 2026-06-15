from pathlib import Path
import numpy as np
import subprocess
import math
import cv2
from kfb import KFB
import config
from numba import njit, prange
import time
import json

inv_tau = 1 / (2 * math.pi)


def discover(folder):
    folder = Path(folder).expanduser().resolve()
    if not folder.exists():
        raise ValueError(f"[discover] Folder does not exist: {folder}")
    files = sorted(folder.glob("*.kfb"))
    if len(files) == 0:
        raise ValueError(f"[discover] No .kfb files found in: {folder}")
    files.reverse()
    return files


def load_kfb(path):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"[load_kfb] File not found: {path}")
    try:
        k = KFB(str(path))
        k.read()
        return k
    except Exception as e:
        raise ValueError(f"[load_kfb] Failed to load: {path}\n{e}")


def lerp(a, b, t):
    return a + (b - a) * t


class StageClock:
    def __init__(self):
        self.timers = {}
        self.starts = {}

    def start(self, name):
        self.starts[name] = time.perf_counter()

    def end(self, name):
        t = time.perf_counter() - self.starts[name]
        self.timers[name] = self.timers.get(name, 0.0) + t

    def reset(self):
        self.timers.clear()

    def report(self, frames=1):
        print("\n--- PERF REPORT ---")
        for k, v in sorted(self.timers.items(), key=lambda x: -x[1]):
            print(f"{k:20s}: {v:.4f}s  ({v/frames:.6f}s/frame)")


# -------------------------------------------------------------------
# PALETTE CACHE
# -------------------------------------------------------------------
_cols_cache: dict = {}

def _get_palette():
    key = config.PALETTE
    if key not in _cols_cache:
        cols = np.asarray(config.PALETTES[key], dtype=np.float32)
        if cols.shape[1] == 3:
            alpha = np.full((cols.shape[0], 1), 255.0, dtype=np.float32)
            cols = np.concatenate([cols, alpha], axis=1)
        _cols_cache[key] = cols
    return _cols_cache[key]


# -------------------------------------------------------------------
# JIT KERNELS
# -------------------------------------------------------------------

@njit(parallel=True, cache=True)
def lighting_core(smooth, light_angle, lighting):
    h, w = smooth.shape
    for y in prange(h):
        for x in range(w):
            if 0 < x < w - 1 and 0 < y < h - 1:
                gx = smooth[y, x + 1] - smooth[y, x - 1]
                gy = smooth[y + 1, x] - smooth[y - 1, x]
                angle = math.atan2(gy, gx)
                diff = angle - light_angle
                shade = 0.5 + 0.5 * math.cos(diff)
                shade *= shade
                shade = 0.5 + (shade - 0.5) * 0.5
                lighting[y, x] = shade
            else:
                lighting[y, x] = 1.0


@njit(parallel=True, cache=True)
def colorize_core(base_phase, lighting, iters, max_iter, cols, flow,
                  black_towards_end, black_start, black_strength,
                  out, inv_tau, ncol):
    h, w = base_phase.shape
    for y in prange(h):
        for x in range(w):
            if iters[y, x] >= max_iter:
                out[y, x, 0] = 0
                out[y, x, 1] = 0
                out[y, x, 2] = 0
                continue

            bp = base_phase[y, x]
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


# -------------------------------------------------------------------
# BUILD RENDER CACHE
# -------------------------------------------------------------------

def build_render_cache(kfb, light_angle=0.7):
    smooth = kfb.smooth.astype(np.float32)
    h, w = smooth.shape
    lighting = np.empty((h, w), dtype=np.float32)
    lighting_core(smooth, light_angle, lighting)
    base_phase = smooth / float(max(kfb.colour_div, 1))
    return lighting, base_phase, kfb.iter, kfb.max_iter


# -------------------------------------------------------------------
# COLORIZE
# -------------------------------------------------------------------

def colorize(cache, flow, out):
    lighting, base_phase, iters, max_iter = cache
    cols = _get_palette()
    ncol = cols.shape[0]
    colorize_core(base_phase, lighting, iters, max_iter, cols, flow,
                  config.BLACK_TOWARDS_END, config.BLACK_START, config.BLACK_STRENGTH,
                  out, inv_tau, ncol)


# -------------------------------------------------------------------
# MISC
# -------------------------------------------------------------------

def zoom(img, scale):
    h, w = img.shape[:2]
    nw = max(1, int(w / scale))
    nh = max(1, int(h / scale))
    x0 = (w - nw) // 2
    y0 = (h - nh) // 2
    crop = img[y0:y0 + nh, x0:x0 + nw]
    return cv2.resize(crop, (w, h), interpolation=config.INTERPOLATION)


def create_encoder(path, fps, w, h, codec="libx265"):
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "-",
        "-an", "-vcodec", codec,
        "-pix_fmt", "yuv420p",
        "-cq", str(config.CQ), "-crf", str(config.CQ),
        path
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def ffmpeg_concat_segments(tmp_dir, out_path):
    tmp_dir = Path(tmp_dir)
    parts = sorted(tmp_dir.glob("part_*.mp4"))
    if not parts:
        raise ValueError(f"No segments found in {tmp_dir}")
    concat_file = tmp_dir / "concat.txt"
    with open(concat_file, "w") as f:
        for p in parts:
            f.write(f"file '{p.resolve().as_posix()}'\n")
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-c", "copy",
        str(out_path)
    ], check=True)


# -------------------------------------------------------------------
# MAIN RENDER
# -------------------------------------------------------------------

def render_sequence(folder, out="out.mp4", fps=60, segment_size=100):
    import json
    from pathlib import Path
    import subprocess

    STATE_FILE = Path(config.STATE_FILE)
    TMP_DIR = Path(config.TMP)
    TMP_DIR.mkdir(exist_ok=True)

    def save_state(seg_i, frame_id):
        STATE_FILE.write_text(json.dumps({
            "seg_i": seg_i,
            "frame_id": frame_id
        }))

    def load_state():
        if not STATE_FILE.exists():
            return 0, 0
        d = json.loads(STATE_FILE.read_text())
        return d.get("seg_i", 0), d.get("frame_id", 0)

    def segment_path(i):
        return TMP_DIR / f"part_{i:05d}.mp4"

    clock = StageClock()

    files = discover(folder)
    print(f"[render] Found {len(files)} KFB files")

    start_seg, frame_id = load_state()
    print(f"[render] Resuming at segment={start_seg}, frame={frame_id}")

    kfb_a = load_kfb(files[0])
    cache_a = build_render_cache(kfb_a)

    h, w = cache_a[1].shape
    pinned, d_out = _get_frame_bufs(h, w)

    flow = -frame_id * (config.FLOW_SPEED / 3)

    seg_count = len(files) - 1

    # ----------------------------
    # SEGMENT LOOP
    # ----------------------------
    for seg in range(start_seg, seg_count):

        if segment_path(seg).exists():
            print(f"[skip] segment {seg} exists")
            continue

        print(f"[render] segment {seg}/{seg_count}")

        kfb_b = load_kfb(files[seg + 1])
        cache_b = build_render_cache(kfb_b)

        z0 = kfb_a.log_zoom
        z1 = kfb_b.log_zoom

        writer = create_encoder(
            str(segment_path(seg)),
            fps,
            w,
            h,
            codec="hevc_nvenc"
        )

        start_frame = frame_id % segment_size if seg == start_seg else 0

        # reset frame_id once we leave first segment
        if seg != start_seg:
            frame_id = seg * segment_size

        segment_frames = segment_size

        for f in range(start_frame, segment_frames):

            # -----------------------
            # GPU COLORIZE
            # -----------------------
            clock.start("colorize")
            colorize(cache_a, flow, d_out)
            d_out.copy_to_host(pinned)
            clock.end("colorize")

            # -----------------------
            # ZOOM
            # -----------------------
            t = f / segment_frames
            z = lerp(z0, z1, t)
            scale = 10 ** (z - z0)

            clock.start("zoom")
            frame = zoom(pinned, scale)
            clock.end("zoom")

            # -----------------------
            # FLOW
            # -----------------------
            flow -= config.FLOW_SPEED

            # -----------------------
            # WRITE FRAME
            # -----------------------
            clock.start("ffmpeg_write")
            writer.stdin.write(frame.tobytes())
            clock.end("ffmpeg_write")

            frame_id += 1

            save_state(seg, frame_id)

        writer.stdin.close()
        writer.wait()

        kfb_a, cache_a = kfb_b, cache_b

        clock.report(frames=segment_frames)
        clock.reset()

    # ----------------------------
    # CONCAT FINAL OUTPUT (LOSSLESS)
    # ----------------------------

    def ffmpeg_concat_segments(tmp_dir, out_path):
        tmp_dir = Path(tmp_dir)

        parts = sorted(tmp_dir.glob("part_*.mp4"))
        if not parts:
            raise ValueError(f"No segments found in {tmp_dir}")

        concat_file = tmp_dir / "concat.txt"

        with open(concat_file, "w") as f:
            for p in parts:
                f.write(f"file '{p.resolve().as_posix()}'\n")

        subprocess.run([
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-c", "copy",
            str(out_path)
        ], check=True)

    ffmpeg_concat_segments(TMP_DIR, out)

    print(f"[render] DONE -> {out}")
