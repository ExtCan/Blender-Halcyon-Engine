"""Texture storage and sampling.

Point sampling is the default because that is what mid-90s software rasterisers
actually did; bilinear, trilinear and the N64's 3-point filter are all here too.
Images are stored (H,W,4) float32 with row 0 at the *bottom*, matching Blender's
pixel order and UV convention.
"""

import numpy as np

WRAP_MODES = ('REPEAT', 'EXTEND', 'CLIP', 'MIRROR')


class Texture:
    __slots__ = ('pixels', 'width', 'height', 'mips', 'name', 'colorspace', 'wrap',
                 'filter', 'quantized')

    def __init__(self, pixels, name='', colorspace='sRGB', wrap='REPEAT',
                 filt='NEAREST'):
        px = np.asarray(pixels, dtype=np.float32)
        if px.ndim == 2:
            px = px[:, :, None]
        if px.shape[2] == 1:
            px = np.repeat(px, 4, axis=2)
            px[:, :, 3] = 1.0
        elif px.shape[2] == 3:
            px = np.concatenate([px, np.ones(px.shape[:2] + (1,), np.float32)], axis=2)
        self.pixels = np.ascontiguousarray(px)
        self.height, self.width = self.pixels.shape[:2]
        self.mips = None
        self.name = name
        self.colorspace = colorspace
        self.wrap = wrap
        self.filter = filt
        self.quantized = False

    # ------------------------------------------------------------- authoring
    def to_linear(self):
        if self.colorspace == 'sRGB':
            from .mathx import srgb_to_linear
            # decode the CONTIGUOUS RGBA buffer and put alpha back,
            # instead of slicing a strided (H,W,3) view: the slice
            # forced a copy and strided arithmetic on multi-megapixel
            # images, and alpha restored afterwards is the same bits
            # as alpha never touched
            rgba = self.pixels
            out = srgb_to_linear(rgba)
            out[:, :, 3] = rgba[:, :, 3]
            self.pixels = np.ascontiguousarray(out)
            self.colorspace = 'Linear'
        return self

    def clamp_size(self, max_size):
        """Downsample to a hardware-style texture budget (box filter)."""
        if not max_size or (self.width <= max_size and self.height <= max_size):
            return self
        while self.width > max_size or self.height > max_size:
            h = max(1, self.height // 2)
            w = max(1, self.width // 2)
            self.pixels = _box_half(self.pixels)
            self.height, self.width = self.pixels.shape[:2]
            if self.height == h and self.width == w and (h == 1 and w == 1):
                break
        self.mips = None
        return self

    def quantize(self, colors):
        """Reduce the texture to N colours -- 90s texture memory was tiny."""
        if not colors or colors >= 256 * 256:
            return self
        from .palette import quantize_image
        rgb = self.pixels[:, :, :3]
        out, _pal = quantize_image(rgb, colors, method='MEDIAN_CUT', dither='NONE')
        self.pixels = np.concatenate([out, self.pixels[:, :, 3:]], axis=2)
        self.quantized = True
        self.mips = None
        return self

    def build_mips(self):
        if self.mips is not None:
            return self.mips
        mips = [self.pixels]
        cur = self.pixels
        while cur.shape[0] > 1 or cur.shape[1] > 1:
            cur = _box_half(cur)
            mips.append(cur)
        self.mips = mips
        return mips

    # -------------------------------------------------------------- sampling
    def sample(self, u, v, filt=None, wrap=None, lod=None, aniso=1,
               duv=None, dvv=None, bias=0.0):
        filt = filt or self.filter
        wrap = wrap or self.wrap
        u = np.asarray(u, np.float32)
        v = np.asarray(v, np.float32)
        if filt == 'TRILINEAR' and duv is not None and int(aniso) > 1:
            return self._sample_aniso(u, v, duv, dvv, float(bias), wrap,
                                      int(aniso))
        if filt == 'TRILINEAR' and lod is not None:
            return self._sample_trilinear(u, v, lod, wrap)
        if filt == 'NEAREST':
            return self._sample_nearest(self.pixels, u, v, wrap)
        if filt == 'N64_3POINT':
            return self._sample_3point(self.pixels, u, v, wrap)
        return self._sample_bilinear(self.pixels, u, v, wrap)

    # -------------------------------------------------------------- internals
    @staticmethod
    def _wrap_index(i, n, mode):
        if mode == 'REPEAT':
            return np.mod(i, n)
        if mode == 'MIRROR':
            period = 2 * n
            m = np.mod(i, period)
            return np.where(m < n, m, period - 1 - m)
        return np.clip(i, 0, n - 1)

    def _oob(self, u, v, wrap):
        if wrap != 'CLIP':
            return None
        return (u < 0.0) | (u > 1.0) | (v < 0.0) | (v > 1.0)

    def _sample_nearest(self, img, u, v, wrap):
        h, w = img.shape[:2]
        x = np.floor(u * w).astype(np.int64)
        y = np.floor(v * h).astype(np.int64)
        oob = self._oob(u, v, wrap)
        x = self._wrap_index(x, w, wrap)
        y = self._wrap_index(y, h, wrap)
        out = img[y, x]
        if oob is not None:
            out = out.copy()
            out[oob] = 0.0
        return out

    def _sample_bilinear(self, img, u, v, wrap):
        h, w = img.shape[:2]
        fx = u * w - 0.5
        fy = v * h - 0.5
        x0 = np.floor(fx).astype(np.int64)
        y0 = np.floor(fy).astype(np.int64)
        tx = (fx - x0).astype(np.float32)[:, None]
        ty = (fy - y0).astype(np.float32)[:, None]
        oob = self._oob(u, v, wrap)
        x0w = self._wrap_index(x0, w, wrap)
        x1w = self._wrap_index(x0 + 1, w, wrap)
        y0w = self._wrap_index(y0, h, wrap)
        y1w = self._wrap_index(y0 + 1, h, wrap)
        c00 = img[y0w, x0w]
        c10 = img[y0w, x1w]
        c01 = img[y1w, x0w]
        c11 = img[y1w, x1w]
        top = c00 + (c10 - c00) * tx
        bot = c01 + (c11 - c01) * tx
        out = top + (bot - top) * ty
        if oob is not None:
            out = out.copy()
            out[oob] = 0.0
        return out

    def _sample_3point(self, img, u, v, wrap):
        """Nintendo 64 3-point (triangular) filter: cheaper, visibly different."""
        h, w = img.shape[:2]
        fx = u * w - 0.5
        fy = v * h - 0.5
        x0 = np.floor(fx).astype(np.int64)
        y0 = np.floor(fy).astype(np.int64)
        tx = (fx - x0).astype(np.float32)
        ty = (fy - y0).astype(np.float32)
        x0w = self._wrap_index(x0, w, wrap)
        x1w = self._wrap_index(x0 + 1, w, wrap)
        y0w = self._wrap_index(y0, h, wrap)
        y1w = self._wrap_index(y0 + 1, h, wrap)
        c00 = img[y0w, x0w]
        c10 = img[y0w, x1w]
        c01 = img[y1w, x0w]
        c11 = img[y1w, x1w]
        upper = (tx + ty) > 1.0
        a = np.where(upper[:, None], c11, c00)
        s = np.where(upper[:, None], 1.0 - ty[:, None], tx[:, None])
        t = np.where(upper[:, None], 1.0 - tx[:, None], ty[:, None])
        b = np.where(upper[:, None], c01, c10)
        c = np.where(upper[:, None], c10, c01)
        out = a + (b - a) * s + (c - a) * t
        oob = self._oob(u, v, wrap)
        if oob is not None:
            out = out.copy()
            out[oob] = 0.0
        return out

    def _sample_aniso(self, u, v, duv, dvv, bias, wrap, max_aniso):
        """Hardware-style N-tap anisotropic filtering.

        The pixel's screen-x and screen-y footprints in texel units pick
        a MAJOR and a minor axis; the mip level follows the MINOR axis
        (so the texture keeps its detail along the stretch -- the whole
        point of anisotropy on a grazing floor), and `max_aniso`
        trilinear taps average along the major axis in UV space.
        Deterministic and vectorised: every pixel takes the same tap
        count, weights uniform -- the era's box approximation of EWA,
        honestly, rather than EWA itself.
        """
        tw, th = float(self.width), float(self.height)
        vx2 = (duv[:, 0] * tw) ** 2 + (dvv[:, 0] * th) ** 2
        vy2 = (duv[:, 1] * tw) ** 2 + (dvv[:, 1] * th) ** 2
        lx = np.sqrt(vx2)
        ly = np.sqrt(vy2)
        major_x = lx >= ly
        major = np.where(major_x, lx, ly)
        minor = np.maximum(np.where(major_x, ly, lx), 1e-6)
        ratio = np.clip(major / minor, 1.0, float(max(max_aniso, 1)))
        lod = (np.log2(np.maximum(major / ratio, 1e-6)) + bias) \
            .astype(np.float32)
        ax_u = np.where(major_x, duv[:, 0], duv[:, 1]).astype(np.float32)
        ax_v = np.where(major_x, dvv[:, 0], dvv[:, 1]).astype(np.float32)
        n_taps = max(int(max_aniso), 1)
        acc = np.zeros((u.shape[0], 4), np.float32)
        for k in range(n_taps):
            t = np.float32((k + 0.5) / n_taps - 0.5)
            acc += self._sample_trilinear(u + ax_u * t, v + ax_v * t,
                                          lod, wrap)
        return (acc / np.float32(n_taps)).astype(np.float32)

    def _sample_trilinear(self, u, v, lod, wrap):
        mips = self.build_mips()
        lod = np.clip(np.asarray(lod, np.float32), 0.0, len(mips) - 1.0)
        l0 = np.floor(lod).astype(np.int32)
        frac = (lod - l0)[:, None]
        out = np.zeros((u.shape[0], 4), np.float32)
        for lvl in np.unique(l0):
            m = l0 == lvl
            a = self._sample_bilinear(mips[int(lvl)], u[m], v[m], wrap)
            b = self._sample_bilinear(mips[min(int(lvl) + 1, len(mips) - 1)],
                                      u[m], v[m], wrap)
            out[m] = a + (b - a) * frac[m]
        return out


def _box_half(img):
    h, w = img.shape[:2]
    hh, ww = max(1, h // 2), max(1, w // 2)
    if h >= 2 and w >= 2:
        return (img[0:hh * 2:2, 0:ww * 2:2] + img[1:hh * 2:2, 0:ww * 2:2] +
                img[0:hh * 2:2, 1:ww * 2:2] + img[1:hh * 2:2, 1:ww * 2:2]) * 0.25
    if h >= 2:
        return (img[0:hh * 2:2] + img[1:hh * 2:2]) * 0.5
    return (img[:, 0:ww * 2:2] + img[:, 1:ww * 2:2]) * 0.5


def compute_lod(du, dv, width, height, bias=0.0):
    """Screen-space UV derivatives -> mip level."""
    dx = np.sqrt((du[:, 0] * width) ** 2 + (du[:, 1] * height) ** 2)
    dy = np.sqrt((dv[:, 0] * width) ** 2 + (dv[:, 1] * height) ** 2)
    rho = np.maximum(np.maximum(dx, dy), 1e-6)
    return np.log2(rho) + bias


def env_sphere_uv(d):
    """Mirror-ball / sphere-map lookup (the 90s way to fake reflections)."""
    m = 2.0 * np.sqrt(np.maximum(d[:, 0] ** 2 + d[:, 1] ** 2 + (d[:, 2] + 1.0) ** 2, 1e-8))
    return d[:, 0] / m + 0.5, d[:, 1] / m + 0.5


def env_equirect_uv(d):
    u = np.arctan2(d[:, 1], -d[:, 0]) / (2.0 * np.pi) + 0.5
    v = np.arctan2(d[:, 2], np.sqrt(np.maximum(d[:, 0] ** 2 + d[:, 1] ** 2, 1e-12))) / np.pi + 0.5
    return u.astype(np.float32), v.astype(np.float32)
