from pathlib import Path
import numpy as np
import subprocess
import math
import cv2
from frame_loader import discover_frames, load_frame
import config
from numba import cuda
from numba import njit, prange
import time
import coloring

inv_tau = 1 / (2 * math.pi)


def discover(folder):
    target_ext = ".rfm" if config.USE_RFF else ".kfb"
    files, ext = discover_frames(folder, extension=target_ext)
    return files, ext


def load_kfb(path):
    try:
        return load_frame(path)
    except ValueError:
        raise
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

@cuda.jit
def lighting_core(smooth, light_angle, lighting):
    x, y = cuda.grid(2)
    h, w = smooth.shape
    if x >= w or y >= h:
        return

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


# -------------------------------------------------------------------
# BUILD RENDER CACHE  (everything uploaded to device once per kfb)
# -------------------------------------------------------------------

def build_render_cache(kfb, light_angle=0.7):
    h, w = kfb.smooth.shape

    # lighting computed on GPU, stays on GPU
    d_smooth = cuda.to_device(kfb.smooth.astype(np.float32))
    d_lighting = cuda.device_array((h, w), dtype=np.float32)

    threads = (16, 16)
    blocks = (
        (w + threads[0] - 1) // threads[0],
        (h + threads[1] - 1) // threads[1],
    )
    lighting_core[blocks, threads](d_smooth, light_angle, d_lighting)

    base_phase = kfb.smooth.astype(np.float32) / float(max(kfb.colour_div, 1))
    d_base_phase = cuda.to_device(base_phase)

    iters = kfb.iter.astype(np.float32)
    if config.USE_RFF:
        iters *= iters          # undo RFF's square root export
    d_iters = cuda.to_device(iters)

    # d_smooth no longer needed after lighting is computed
    return d_lighting, d_base_phase, d_iters, kfb.max_iter


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


def create_encoder(path, w, h):
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(config.FPS),
        "-i", "-",
        "-an", "-vcodec", config.CODEC,
        "-pix_fmt", "yuv420p",
        "-cq", str(config.CQ), "-crf", str(config.CQ),
        path
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


# -------------------------------------------------------------------
# MAIN RENDER
# -------------------------------------------------------------------

def render_sequence(folder, out="out.mp4", segment_size=100):
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

    files, ext = discover(folder)
    print(f"[render] Found {len(files)} {ext} files")

    start_seg, frame_id = load_state()
    print(f"[render] Resuming at segment={start_seg}, frame={frame_id}")

    kfb_a = load_kfb(files[0])
    cache_a = build_render_cache(kfb_a)

    h, w = cache_a[1].shape
    pinned, d_out = coloring._get_frame_bufs(h, w)

    flow = -frame_id * config.FLOW_SPEED

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
            path=str(segment_path(seg)),
            w=w,
            h=h
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
            if config.COLORING == "standard":
                coloring.colorize(cache_a, flow, d_out)
            elif config.COLORING == "contour":
                coloring.colorize_contour(cache_a,flow,d_out)
            elif config.COLORING == "audio":
                coloring.colorize_audio(cache_a,flow,d_out)
            elif config.COLORING == "image":
                coloring.colorize_image(cache_a,flow,d_out)
            d_out.copy_to_host(pinned)
            clock.end("colorize")

            # -----------------------
            # ZOOMd
            # -----------------------
            try:
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
            except Exception: frame_id += 1;flow -= config.FLOW_SPEED
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

        concat_file = "concat.txt"

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
