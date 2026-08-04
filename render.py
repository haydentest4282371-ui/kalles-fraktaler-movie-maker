from pathlib import Path
import numpy as np
import subprocess
import math
import cv2
from frame_loader import discover_frames, load_frame
import config
from numba import cuda
import time
import coloring
from numba import prange, njit
import json
import shutil

print("import")

inv_tau = 1 / (2 * math.pi)


# ============================================================
# FILES
# ============================================================

def discover(folder):

    ext = ".rfm" if config.USE_RFF else ".kfb"

    files, actual = discover_frames(
        folder,
        extension=ext
    )

    return files, actual



def load_kfb(path):

    return load_frame(path)



def lerp(a,b,t):

    return a + (b-a)*t




# ============================================================
# TIMING
# ============================================================

class StageClock:

    def __init__(self):

        self.timers = {}
        self.starts = {}


    def start(self,name):

        self.starts[name] = time.perf_counter()


    def end(self,name):

        t = time.perf_counter() - self.starts[name]

        self.timers[name] = (
            self.timers.get(name,0)
            +
            t
        )


    def reset(self):

        self.timers.clear()


    def report(self,frames=1):

        print("\n--- PERF ---")

        for k,v in sorted(
            self.timers.items(),
            key=lambda x:-x[1]
        ):

            print(
                f"{k:20s}: {v:.4f}s ({v/frames:.6f}/frame)"
            )




# ============================================================
# GPU ROLLING ZOOM
# ============================================================

@cuda.jit
def zoom_field(
    src,
    dst,
    scale
):

    x,y = cuda.grid(2)

    h,w = dst.shape

    if x >= w or y >= h:
        return


    cx = w * 0.5
    cy = h * 0.5


    sx = (
        (x-cx)/scale
        +
        cx
    )

    sy = (
        (y-cy)/scale
        +
        cy
    )


    ix = int(sx)
    iy = int(sy)


    if (
        ix >= 0
        and iy >= 0
        and ix < w
        and iy < h
    ):

        dst[y,x] = src[iy,ix]

    else:

        dst[y,x] = 0




@cuda.jit
def insert_keyframe(
    src_phase,
    src_light,
    src_iters,

    dst_phase,
    dst_light,
    dst_iters,

    offset_x,
    offset_y,

    max_iter
):

    x,y = cuda.grid(2)


    h,w = src_phase.shape


    if x >= w or y >= h:
        return


    dx = x + offset_x
    dy = y + offset_y


    dh,dw = dst_phase.shape


    if (
        dx < 0
        or dy < 0
        or dx >= dw
        or dy >= dh
    ):
        return



    # only replace valid fractal data

    if src_iters[y,x] < max_iter:

        dst_phase[dy,dx] = src_phase[y,x]

        dst_light[dy,dx] = src_light[y,x]

        dst_iters[dy,dx] = src_iters[y,x]

@cuda.jit
def lighting_core(smooth, light_angle, lighting):

    x, y = cuda.grid(2)

    h, w = smooth.shape

    if x >= w or y >= h:
        return


    if 0 < x < w - 1 and 0 < y < h - 1:

        gx = smooth[y, x + 1] - smooth[y, x - 1]
        gy = smooth[y + 1, x] - smooth[y - 1, x]

        angle = math.atan2(
            gy,
            gx
        )

        diff = angle - light_angle

        shade = (
            0.5
            +
            0.5 * math.cos(diff)
        )

        shade *= shade

        shade = (0.5+(shade - 0.5) * 0.5)

        lighting[y, x] = shade

    else:

        lighting[y, x] = 1.0

# ============================================================
# GPU CACHE
# ============================================================

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

    base_phase = kfb.smooth.astype(np.float32)
    d_base_phase = cuda.to_device(base_phase)

    iters = kfb.iter.astype(np.float32)
    d_iters = cuda.to_device(iters)

    # d_smooth no longer needed after lighting is computed
    return d_lighting, d_base_phase, d_iters, kfb.max_iter




# ============================================================
# WORK BUFFER
# ============================================================

def allocate_work_buffers(
    h,
    w
):

    phase_a = cuda.device_array(
        (h,w),
        dtype=np.float32
    )

    phase_b = cuda.device_array(
        (h,w),
        dtype=np.float32
    )


    light_a = cuda.device_array(
        (h,w),
        dtype=np.float32
    )

    light_b = cuda.device_array(
        (h,w),
        dtype=np.float32
    )


    iter_a = cuda.device_array(
        (h,w),
        dtype=np.float32
    )

    iter_b = cuda.device_array(
        (h,w),
        dtype=np.float32
    )


    return {
        "phase": [phase_a, phase_b],
        "light": [light_a, light_b],
        "iters": [iter_a, iter_b],
        "index": 0
    }




def swap_work_buffers(work):

    work["index"] ^= 1



def current_buffers(work):

    i = work["index"]

    return (
        work["phase"][i],
        work["light"][i],
        work["iters"][i]
    )



def next_buffers(work):

    i = work["index"] ^ 1

    return (
        work["phase"][i],
        work["light"][i],
        work["iters"][i]
    )





# ============================================================
# ZOOM CURRENT IMAGE
# ============================================================

def gpu_zoom_work(
    work,
    scale
):

    src_phase, src_light, src_iters = current_buffers(work)

    dst_phase, dst_light, dst_iters = next_buffers(work)


    h,w = src_phase.shape


    threads = (32,8)

    blocks = (
        (w+threads[0]-1)//threads[0],
        (h+threads[1]-1)//threads[1]
    )


    zoom_field[blocks,threads](
        src_phase,
        dst_phase,
        scale
    )


    zoom_field[blocks,threads](
        src_light,
        dst_light,
        scale
    )


    zoom_field[blocks,threads](
        src_iters,
        dst_iters,
        scale
    )


    swap_work_buffers(work)




# ============================================================
# INSERT KEYFRAME
# ============================================================

def gpu_insert_keyframe(
    work,
    cache
):

    light, phase, iters, max_iter, _ = cache


    dst_phase, dst_light, dst_iters = current_buffers(work)


    h,w = phase.shape


    oh,ow = dst_phase.shape


    ox = (ow-w)//2
    oy = (oh-h)//2


    threads=(32,8)

    blocks=(
        (w+threads[0]-1)//threads[0],
        (h+threads[1]-1)//threads[1]
    )


    insert_keyframe[blocks,threads](
        phase,
        light,
        iters,

        dst_phase,
        dst_light,
        dst_iters,

        ox,
        oy,

        max_iter
    )





# ============================================================
# KEYFRAME LOADING
# ============================================================

def load_layer(index, files):

    kfb = load_kfb(
        files[index]
    )


    cache = build_render_cache(
        kfb
    )


    return {
        "cache": cache,
        "zoom": cache[4],
        "index": index
    }





# ============================================================
# ENCODER
# ============================================================

def create_encoder(
    path,
    w,
    h
):

    cmd = [

        "ffmpeg","-hide_banner","-loglevel","error",
        "-y",

        "-f",
        "rawvideo",

        "-pix_fmt",
        "rgb24",

        "-s",
        f"{w}x{h}",

        "-r",
        str(config.FPS),

        "-i",
        "-",

        "-an",

        "-vcodec",
        config.CODEC,

        "-pix_fmt",
        "yuv420p",

        "-cq",
        str(config.CQ),

        "-crf",
        str(config.CQ),

        path
    ]


    return subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE
    )
# ============================================================
# MAIN RENDER
# ============================================================

def zoom(img, scale):
    h, w = img.shape[:2]

    nw = max(1, int(w / scale))
    nh = max(1, int(h / scale))

    x0 = (w - nw) // 2
    y0 = (h - nh) // 2

    crop = img[
        y0:y0 + nh,
        x0:x0 + nw
    ]

    return cv2.resize(
        crop,
        (w, h),
        interpolation=config.INTERPOLATION
    )

@njit(parallel=True)
def composite_center(base, overlay, blend):
    bh, bw, _ = base.shape
    oh, ow, _ = overlay.shape

    xoff = (bw - ow) // 2
    yoff = (bh - oh) // 2

    for y in prange(oh):
        by = y + yoff

        if by < 0 or by >= bh:
            continue

        for x in range(ow):
            bx = x + xoff

            if bx < 0 or bx >= bw:
                continue

            if blend <= 0:
                base[by, bx, 0] = overlay[y, x, 0]
                base[by, bx, 1] = overlay[y, x, 1]
                base[by, bx, 2] = overlay[y, x, 2]

            else:
                edge = x
                if y < edge:
                    edge = y
                if ow - 1 - x < edge:
                    edge = ow - 1 - x
                if oh - 1 - y < edge:
                    edge = oh - 1 - y

                alpha = edge / blend

                if alpha > 1:
                    alpha = 1

                inv = 1 - alpha

                base[by, bx, 0] = (
                    overlay[y, x, 0] * alpha +
                    base[by, bx, 0] * inv
                )

                base[by, bx, 1] = (
                    overlay[y, x, 1] * alpha +
                    base[by, bx, 1] * inv
                )

                base[by, bx, 2] = (
                    overlay[y, x, 2] * alpha +
                    base[by, bx, 2] * inv
                )

def render_sequence(folder, out="out.mp4", segment_size=100):
    config.FLOW_SPEED = config.USER_FLOW_SPEED/config.FPS/config.PERIOD
    print("render start")

    STATE_FILE = Path(config.STATE_FILE)
    TMP_DIR = Path(config.TMP)
    TMP_DIR.mkdir(exist_ok=True)

    if STATE_FILE.exists():
        STATE_FILE.unlink()

    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR, ignore_errors=True)

    TMP_DIR.mkdir(parents=True, exist_ok=True)

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

    def render_color(layer):
        cache = layer["cache"]

        clock.start("colorize_kernel")

        if config.COLORING == "standard": coloring.colorize(cache, flow, d_out)
        elif config.COLORING == "contour": coloring.colorize_contour(cache, flow, d_out)
        elif config.COLORING == "audio": coloring.colorize_audio(cache, flow, d_out)
        elif config.COLORING == "image": coloring.colorize_image(cache, flow, d_out)
        elif config.COLORING == "linear": coloring.colorize_linear(cache,flow,d_out)
        elif config.COLORING == "distance": coloring.colorize_distance(cache, flow, d_out)
        elif config.COLORING == "de_angle": coloring.colorize_de_angle(cache,flow,d_out)
        cuda.synchronize()

        clock.end("colorize_kernel")

        clock.start("gpu_download")
        d_out.copy_to_host(pinned)
        clock.end("gpu_download")

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
        composite_center(
            base,
            layer,
            config.SEAM_BLEND
        )

    def load_layer(index):
        clock.start("keyframe_load")

        kfb = load_kfb(files[index])
        cache = build_render_cache(kfb)

        layer = {
            "cache": cache,
            "zoom": kfb.log_zoom,
            "index": index
        }

        clock.end("keyframe_load")

        return layer

    files, ext = discover(folder)

    print(f"[render] Found {len(files)} {ext} files")

    start_seg, frame_id = load_state()

    print(
        f"[render] Resuming at segment={start_seg}, frame={frame_id}"
    )

    layers = []

    warmup = min(config.KEYFRAMES, len(files))

    for i in range(warmup):
        layers.append(load_layer(i))

    # fixed export resolution from config
    w, h = config.DIMS

    # coloring buffers stay at keyframe resolution
    kh, kw = layers[0]["cache"][1].shape

    pinned, d_out = coloring._get_frame_bufs(kh, kw)

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
            str(segment_path(seg)),
            w,
            h
        )

        z0 = layers[0]["zoom"]
        z1 = layers[1]["zoom"] if len(layers) > 1 else z0

        start_frame = frame_id % segment_size if seg == start_seg else 0

        for f in range(start_frame, segment_size):

            try:
                t = f / segment_size
                z = lerp(z0, z1, t)

                clock.start("composite")

                render_color(layers[0])

                frame = zoom(
                    pinned,
                    10 ** (z - layers[0]["zoom"])
                )

                for layer in layers[1:]:

                    render_color(layer)

                    scale = 10 ** (z - layer["zoom"])

                    img = resize_layer(
                        pinned,
                        scale
                    )

                    paste_center(
                        frame,
                        img
                    )

                clock.end("composite")

                flow -= config.FLOW_SPEED

                if frame.shape[1] != w or frame.shape[0] != h:
                    frame = cv2.resize(
                        frame,
                        (w, h),
                        interpolation=config.INTERPOLATION
                    )

                clock.start("encode")
                
                writer.stdin.write(
                    frame.astype(np.uint8).tobytes()
                )

                clock.end("encode")

                frame_id += 1

                save_state(
                    seg,
                    frame_id
                )

                if frame_id % config.PERF_REPORT_INTERVAL == 0:
                    print(f"[perf] frame {frame_id}")
                    clock.report(frames=config.PERF_REPORT_INTERVAL)
                    clock.reset()

            except Exception as e:
                print("[render] frame error:", e)
                frame_id += 1
                flow -= config.FLOW_SPEED

        writer.stdin.close()
        writer.wait()

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

    if aborted:
        print(f"[render] ABORTED at segment {seg}")
        return

    parts = sorted(
        TMP_DIR.glob("part_*.mp4")
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
