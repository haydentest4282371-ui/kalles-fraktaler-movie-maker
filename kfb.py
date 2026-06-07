import struct
import numpy as np
import mpmath as mp


class KFB:
    def __init__(self, path):
        self.path = path
        self.w = self.h = 0
        self.max_iter = 0
        self.colour_div = 0
        self.colours = []
        self.iter = None
        self.smooth = None

        # NEW: safe zoom in log-space
        self.log_zoom = None

    def read(self):
        with open(self.path, "rb") as f:
            if f.read(3) != b"KFB":
                raise ValueError("Not a KFB file")

            self.w, self.h = struct.unpack("<ii", f.read(8))
            size = self.w * self.h

            raw_iter = np.frombuffer(f.read(size * 4), dtype=np.int32)
            self.iter = raw_iter.reshape((self.w, self.h)).T.copy()

            self.colour_div = struct.unpack("<i", f.read(4))[0]
            ncol = struct.unpack("<i", f.read(4))[0]

            self.colours = []
            for _ in range(ncol):
                r, g, b = struct.unpack("BBB", f.read(3))
                self.colours.append((r, g, b))

            self.max_iter = struct.unpack("<i", f.read(4))[0]

            raw_smooth = np.frombuffer(f.read(size * 4), dtype=np.float32)
            raw_smooth = raw_smooth.reshape((self.w, self.h)).T.copy()
            self.smooth = self.iter.astype(np.float64) + 1.0 - raw_smooth

            # ----------------------------------------------------
            # SAFE ZOOM PARSE (mpmath ONLY HERE)
            # ----------------------------------------------------
            name = self.path.split("/")[-1].replace(".kfb", "")

            try:
                zoom = mp.mpf(name.split("_")[-1])
            except:
                zoom = None

            if zoom is None or not mp.isfinite(zoom) or zoom <= 0:
                self.log_zoom = None
            else:
                self.log_zoom = float(mp.log10(zoom))