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
import pygame

inv_tau = 1 / (2 * math.pi)

_preview_screen = None
_preview_w = None
_preview_h = None


def init_preview(w, h, max_dim=900):
    """Initialize a pygame window for live preview, scaled down if needed."""
    global _preview_screen, _preview_w, _preview_h

    scale = min(1.0, max_dim / max(w, h))
    pw, ph = max(1, int(w * scale)), max(1, int(h * scale))

    pygame.init()
    _preview_screen = pygame.display.set_mode((pw, ph))
    pygame.display.set_caption("Render Preview")
    _preview_w, _preview_h = pw, ph


def update_preview(frame_rgb):
    """Push an RGB24 (H,W,3) numpy frame to the pygame preview window."""
    global _preview_screen, _preview_w, _preview_h

    if _preview_screen is None:
        return True  # preview not initialized, just continue

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            return False

    # pygame surfarray expects (W, H, 3), our frame is (H, W, 3)
    surf = pygame.surfarray.make_surface(np.transpose(frame_rgb, (1, 0, 2)))

    if (_preview_w, _preview_h) != frame_rgb.shape[1::-1]:
        surf = pygame.transform.smoothscale(surf, (_preview_w, _preview_h))

    _preview_screen.blit(surf, (0, 0))
    pygame.display.flip()
    return True


def close_preview():
    global _preview_screen
    if _preview_screen is not None:
        pygame.quit()
        _preview_screen = None


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

def render_sequence(folder, out="out.mp4", segment_size=100, preview=True):
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

    def render_color(cache):
        if config.COLORING == "standard":
            coloring.colorize(cache, flow, d_out)
        elif config.COLORING == "contour":
            coloring.colorize_contour(cache, flow, d_out)
        elif config.COLORING == "audio":
            coloring.colorize_audio(cache, flow, d_out)
        elif config.COLORING == "image":
            coloring.colorize_image(cache, flow, d_out)
        elif config.COLORING == "void":
            coloring.colorize_void(cache, flow, d_out)

        d_out.copy_to_host(pinned)

    def resize_layer(img, scale):
        h0, w0 = img.shape[:2]

        new_w = max(1, int(w0 * scale))
        new_h = max(1, int(h0 * scale))

        return cv2.resize(
            img,
            (new_w, new_h),
            interpolation=config.INTERPOLATION
        )


    def paste_center(base, layer):
        h0, w0 = base.shape[:2]
        h1, w1 = layer.shape[:2]

        x = (w0 - w1) // 2
        y = (h0 - h1) // 2

        x0 = max(0, x)
        y0 = max(0, y)

        x1 = min(w0, x + w1)
        y1 = min(h0, y + h1)

        lx0 = x0 - x
        ly0 = y0 - y

        lx1 = lx0 + (x1 - x0)
        ly1 = ly0 + (y1 - y0)

        roi = base[y0:y1, x0:x1]
        overlay = layer[ly0:ly1, lx0:lx1]

        blend = config.SEAM_BLEND

        if blend <= 0:
            roi[:] = overlay
            return

        h, w = overlay.shape[:2]

        yy, xx = np.indices((h, w))

        edge = np.minimum.reduce([
            xx,
            yy,
            w - 1 - xx,
            h - 1 - yy
        ]).astype(np.float32)

        alpha = np.clip(
            edge / blend,
            0.0,
            1.0
        )

        alpha = alpha[..., None]

        roi[:] = (
            overlay.astype(np.float32) * alpha +
            roi.astype(np.float32) * (1.0 - alpha)
        ).astype(np.uint8)

    def load_layer(index):
        kfb = load_kfb(files[index])
        cache = build_render_cache(kfb)

        return {
            "cache": cache,
            "zoom": kfb.log_zoom,
            "index": index
        }

    clock = StageClock()

    files, ext = discover(folder)
    print(f"[render] Found {len(files)} {ext} files")

    start_seg, frame_id = load_state()
    print(f"[render] Resuming at segment={start_seg}, frame={frame_id}")

    layers = []

    warmup = min(config.KEYFRAMES, len(files))

    for i in range(warmup):
        layers.append(load_layer(i))

    h, w = layers[0]["cache"][1].shape

    pinned, d_out = coloring._get_frame_bufs(h, w)

    if preview:
        init_preview(w, h)

    flow = -frame_id * config.FLOW_SPEED

    seg_count = len(files) - 1

    aborted = False

    for seg in range(start_seg, seg_count):

        if aborted:
            break

        if segment_path(seg).exists():
            print(f"[skip] segment {seg} exists")
            continue

        print(f"[render] segment {seg}/{seg_count}")

        writer = create_encoder(
            path=str(segment_path(seg)),
            w=w,
            h=h
        )

        current = layers[0]

        z0 = current["zoom"]

        if len(layers) > 1:
            z1 = layers[1]["zoom"]
        else:
            z1 = z0

        start_frame = (
            frame_id % segment_size
            if seg == start_seg
            else 0
        )

        for f in range(start_frame, segment_size):

            try:
                t = f / segment_size
                z = lerp(z0, z1, t)

                # base keyframe
                render_color(
                    layers[0]["cache"]
                )

                frame = zoom(
                    pinned,
                    10 ** (z - layers[0]["zoom"])
                )

                # detail keyframes
                for layer in layers[1:]:

                    render_color(
                        layer["cache"]
                    )

                    scale = (
                        10 ** (z - layer["zoom"])
                    )

                    img = resize_layer(
                        pinned,
                        scale
                    )

                    paste_center(
                        frame,
                        img
                    )

                if preview:
                    keep_going = update_preview(frame)

                    if not keep_going:
                        print("[render] Preview closed")
                        aborted = True
                        break

                flow -= config.FLOW_SPEED

                writer.stdin.write(
                    frame.tobytes()
                )

                frame_id += 1

                save_state(
                    seg,
                    frame_id
                )

            except Exception as e:
                print("[render] frame error:", e)

                frame_id += 1
                flow -= config.FLOW_SPEED

        writer.stdin.close()
        writer.wait()

        # slide warmup window
        if layers:
            layers.pop(0)

        next_index = (
            layers[-1]["index"] + 1
            if layers
            else seg + warmup
        )

        if next_index < len(files):
            layers.append(
                load_layer(next_index)
            )

        clock.report(frames=segment_size)
        clock.reset()

    if preview:
        close_preview()

    if aborted:
        print(f"[render] ABORTED at segment {seg}")
        return

    parts = sorted(
        TMP_DIR.glob("part_*.mp4")
    )

    if not parts:
        raise ValueError(
            f"No segments found in {TMP_DIR}"
        )

    concat_file = "concat.txt"

    with open(concat_file, "w") as f:
        for p in parts:
            f.write(
                f"file '{p.resolve().as_posix()}'\n"
            )

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_file,
            "-c",
            "copy",
            str(out)
        ],
        check=True
    )

    print(f"[render] DONE -> {out}")
