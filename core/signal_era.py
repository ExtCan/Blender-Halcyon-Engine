"""R251: the post-signal pack's CPU stages -- the machine's scan-out.

Every function here is bpy-free and device-free: `(rgb, st, ...)` in,
float32 `(H, W, 3)` out, row 0 the bottom of the picture (post.process).
The stages run on the frame the machine WROTE (after the quant block)
and reproduce what sat between that framebuffer and the glass:

    C005  vi_filter    the N64 Video Interface's dither filter and gamma
    C019  copy_filter  the GameCube / Wii EFB-to-XFB copy's 7-tap deflicker
    C033  crtc_blend   the PS2 GS CRTC's two-circuit PMODE mix
    C069  video_filter the 3dfx Voodoo "22-bit" scan-out filter (v1 / v2)
    C014  threedo_2x   the 3DO display generator's cornerweight doubling
    C047  gba_stretch  the GBA Mode 5 affine stretch (8.8 fixed point)

Every chain-stage function OPENS with `if not <name>_on(st): return rgb`,
so it is the identity on any settings that do not switch it on (the
activity test lives in the wiring AND here). The two output-scale
functions (`threedo_2x`, `gba_stretch`) run only from `post.upscale`
under their own `output_scale` items and change the shape by design.

The arithmetic is INTEGER end to end on the frame's own 8-bit (or
5/6-bit) values: bytes travel as int32, every product and shift is
order-free, and the picture returns through the one byte table `U8`
(`i / 255` in float32, computed once), so the GPU twins in
gpu/stages_signal.py fetch the same table and are bitwise this code.

The signed-hash rule (review 0-1): `film._hash_u32_raw` returns uint32
and under the NumPy Blender ships `uint32_array - 32768` STAYS uint32 and
wraps. Every hash field that feeds a signed operation is cast to int32
FIRST, through `hash_s16`; unsigned products stay unsigned.

Shared helpers for the pack's wave-2 modules (signal_tape, signal_codec):
`U8`, `to_u8`, `from_u8`, `to_bits`, `clamp255`, `_once`, `y601` /
`cb601` / `cr601` / `yuv601_decode`, `yiq_int` / `yiq_back`, `pal_yuv` /
`pal_back`, `hash_s16`, `SIN16`, `isqrt16`, `gauss16`, `fir16`,
`u8_image`, `sin16_image`, `cached_peek` (post-palette's, imported by name).
"""

import dataclasses
import math

import numpy as np

from .film import _hash_u32_raw
from .wear import _sheet_key
from .palette_era import cached_peek        # noqa: F401  (R251 design 2.6 #46)
from .post import DEPTH_BITS

# ------------------------------------------------------------ the byte table

#: the byte table: i / 255 in float32, computed once; every stage's
#: FINAL values come through it on both roads
U8 = (np.arange(256, dtype=np.float32) / np.float32(255.0)).astype(np.float32)

_ONCE = set()


def _once(msg):
    """Print `[Halcyon] {msg}` once per process (tests clear `_ONCE`)."""
    if msg not in _ONCE:
        _ONCE.add(msg)
        print(f'[Halcyon] {msg}')


def to_u8(rgb):
    """The frame's bytes: clip, scale by 255, round half to even (NumPy's
    round; the GPU's `roundEven` is the same rule). int32."""
    v = np.clip(np.asarray(rgb, np.float32), 0.0, 1.0) * np.float32(255.0)
    return np.round(v).astype(np.int32)


def from_u8(i):
    """Bytes back to float32 through the byte table."""
    return U8[np.clip(np.asarray(i, np.int32), 0, 255)]


def to_bits(v, bits):
    """The stored field of one channel at `bits` bits: the QUANT stage's
    own index (clip, one multiply, round half to even), exact on a frame
    that palette.snap_bits wrote."""
    lv = np.float32((1 << int(bits)) - 1)
    t = np.clip(np.asarray(v, np.float32), 0.0, 1.0) * lv
    return np.round(t).astype(np.int32)


def clamp255(x):
    """Clamp an integer array to 0..255 (int32)."""
    return np.clip(np.asarray(x, np.int32), 0, 255)


def u8_image():
    """`U8` as a (1, 256, 4) float32 image for a data texture (channel r)."""
    img = np.zeros((1, 256, 4), np.float32)
    img[0, :, 0] = U8
    img[0, :, 3] = 1.0
    return img


# ------------------------------------------- integer colour spaces (shared)

def _ash(x, n):
    """Arithmetic shift right on int32 (floor division by 2**n)."""
    return np.asarray(x, np.int32) >> np.int32(n)


def y601(R, G, B):
    """BT.601 8-bit luma (the Microsoft / Dolphin integers), 16..235."""
    R, G, B = (np.asarray(a, np.int32) for a in (R, G, B))
    return _ash(66 * R + 129 * G + 25 * B + 128, 8) + 16


def cb601(R, G, B):
    R, G, B = (np.asarray(a, np.int32) for a in (R, G, B))
    return _ash(-38 * R - 74 * G + 112 * B + 128, 8) + 128


def cr601(R, G, B):
    R, G, B = (np.asarray(a, np.int32) for a in (R, G, B))
    return _ash(112 * R - 94 * G - 18 * B + 128, 8) + 128


def yuv601_decode(Y, Cb, Cr):
    """BT.601 8-bit decode: (R, G, B) int32 clamped 0..255."""
    C = np.asarray(Y, np.int32) - 16
    D = np.asarray(Cb, np.int32) - 128
    E = np.asarray(Cr, np.int32) - 128
    R = clamp255(_ash(298 * C + 409 * E + 128, 8))
    G = clamp255(_ash(298 * C - 100 * D - 208 * E + 128, 8))
    B = clamp255(_ash(298 * C + 516 * D + 128, 8))
    return R, G, B


def yiq_int(R, G, B):
    """Integer YIQ (the composite stage's matrix scaled by 256): Y 0..255,
    I in -152..152, Q in -137..137 -- SIGNED, no offset."""
    R, G, B = (np.asarray(a, np.int32) for a in (R, G, B))
    Y = _ash(77 * R + 150 * G + 29 * B, 8)
    I = _ash(153 * R - 70 * G - 82 * B, 8)
    Q = _ash(54 * R - 134 * G + 80 * B, 8)
    return Y, I, Q


def yiq_back(Y, I, Q):
    Y, I, Q = (np.asarray(a, np.int32) for a in (Y, I, Q))
    R = clamp255(Y + _ash(245 * I + 158 * Q + 128, 8))
    G = clamp255(Y + _ash(-70 * I - 166 * Q + 128, 8))
    B = clamp255(Y + _ash(-283 * I + 436 * Q + 128, 8))
    return R, G, B


def pal_yuv(R, G, B):
    """Integer PAL YUV on the OFFSET-FREE luma (a grey encodes to
    (Y, 0, 0) exactly): U in -126..126, V in -224..224."""
    R, G, B = (np.asarray(a, np.int32) for a in (R, G, B))
    Y = _ash(77 * R + 150 * G + 29 * B, 8)
    U = _ash(126 * (B - Y) + 128, 8)
    V = _ash(224 * (R - Y) + 128, 8)
    return Y, U, V


def pal_back(Y, U, V):
    Y, U, V = (np.asarray(a, np.int32) for a in (Y, U, V))
    RY = _ash(292 * V + 128, 8)
    BY = _ash(520 * U + 128, 8)
    R = clamp255(Y + RY)
    B = clamp255(Y + BY)
    G = clamp255(Y - _ash(130 * RY + 50 * BY + 128, 8))
    return R, G, B


# ------------------------------------------------------ hashes and tables

def hash_s16(xx, yy, key):
    """A signed 16-bit hash field, -32768..32767 (int32): the low 16 bits
    of the 24-bit hash, cast to int32 BEFORE the subtraction (the
    signed-hash rule; the GLSL side's `int(h & 0xffffu) - 32768`)."""
    h = _hash_u32_raw(xx, yy, key)
    return (h & np.uint32(0xffff)).astype(np.int32) - np.int32(32768)


#: the sine as a 256-entry INTEGER table (float64 once on the CPU);
#: phases are 16-bit integer accumulators, `idx = (ph & 0xffff) >> 8`
SIN16 = np.round(32767.0 * np.sin(2.0 * np.pi * np.arange(256) / 256.0)
                 ).astype(np.int32)


def sin16_image():
    """`SIN16` as a (1, 256, 4) float32 image (channel r)."""
    img = np.zeros((1, 256, 4), np.float32)
    img[0, :, 0] = SIN16.astype(np.float32)
    img[0, :, 3] = 1.0
    return img


def isqrt16(n):
    """Exact integer square root of 0 <= n < 65536 (int32 arrays)."""
    n = np.asarray(n, np.int64)
    r = np.floor(np.sqrt(n.astype(np.float64))).astype(np.int64)
    r -= (r * r > n)
    r += ((r + 1) * (r + 1) <= n)
    return r.astype(np.int32)


def gauss16(sigma, R):
    """Gaussian FIR taps as integers summing to 65536 exactly: the
    residual goes to the centre tap. (2R+1,) int32."""
    k = np.arange(-int(R), int(R) + 1, dtype=np.float64)
    s = max(float(sigma), 1e-6)
    w = np.exp(-k * k / (2.0 * s * s))
    w = w / w.sum()
    w16 = np.round(w * 65536.0).astype(np.int64)
    w16[int(R)] += 65536 - int(w16.sum())
    return w16.astype(np.int32)


def fir16(v, w16, axis=1):
    """The integer FIR `(sum_k w16[k] * v[x+k] + 32768) >> 16` along
    `axis`, the index clamped to the edge (np.pad 'edge'). int32 in,
    int32 out; the sum is order-free."""
    v = np.asarray(v, np.int64)
    R = (len(w16) - 1) // 2
    pad = [(0, 0)] * v.ndim
    pad[axis] = (R, R)
    vp = np.pad(v, pad, mode='edge')
    acc = np.zeros(v.shape, np.int64)
    n = v.shape[axis]
    for i, w in enumerate(w16):
        sl = [slice(None)] * v.ndim
        sl[axis] = slice(i, i + n)
        acc += int(w) * vp[tuple(sl)]
    return ((acc + 32768) >> 16).astype(np.int32)


# =========================================================================
# C005  N64 Video Interface: DITHER_FILTER_ENABLE, GAMMA_ENABLE,
#       GAMMA_DITHER_ENABLE  (angrylion vi/restore.c, vi/gamma.c)
# =========================================================================

#: the N64's 5551 frame, and 5:6:5 as the extension the tooltip names
_VI_DEPTHS = ('15', '16')

#: the VI's gamma: gamma_table[i] = isqrt(i << 6) << 1
VI_GAMMA_LUT = np.array([2 * math.isqrt(64 * i) for i in range(256)],
                        np.int32)
#: the VI's gamma dither: six random bits under the root, index (v << 6) | d
VI_GAMMA_DITHER_LUT = np.array([2 * math.isqrt(i) for i in range(256 * 64)],
                               np.int32)


def vi_gamma_image():
    """`VI_GAMMA_LUT` as a (1, 256, 4) float32 image (channel r)."""
    img = np.zeros((1, 256, 4), np.float32)
    img[0, :, 0] = VI_GAMMA_LUT.astype(np.float32)
    img[0, :, 3] = 1.0
    return img


def vi_gdither_image():
    """`VI_GAMMA_DITHER_LUT` as a (128, 128, 4) float32 image laid out
    texel (i & 127, i >> 7) (channel r)."""
    img = np.zeros((128, 128, 4), np.float32)
    img[:, :, 0] = VI_GAMMA_DITHER_LUT.astype(np.float32).reshape(128, 128)
    img[:, :, 3] = 1.0
    return img


def vi_filter_why(st):
    """Why the VI stage is a named no-op on this frame, or None."""
    if str(st.color_depth) in _VI_DEPTHS:
        return None
    return ("VI filter: the Video Interface reads the N64's 15-bit "
            f"framebuffer; Colour Depth is '{st.color_depth}' (set 15, or 16)")


def vi_filter_on(st):
    dials = bool(getattr(st, 'vi_dither_filter', False)) or \
        str(getattr(st, 'vi_gamma', 'NONE')) != 'NONE'
    if dials and vi_filter_why(st):
        _once(vi_filter_why(st))            # named no-op, both roads
        return False
    return dials


#: the eight neighbours, a fixed order (an integer sum: order-free)
_VI_OFFSETS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1),
               (1, -1), (1, 0), (1, 1))


def vi_filter(rgb, st, frame=0, seed=0):
    """The N64 VI at scan-out: each stored field expanded to 8 bits and,
    with DITHER_FILTER_ENABLE, stepped by +-1 toward each of its eight
    neighbours (equal neighbours contribute 0 -- the tie rule); then
    GAMMA_ENABLE's `2 * isqrt(64 v)`, with GAMMA_DITHER_ENABLE's six
    hashed bits under the root, or the single-bit dither alone."""
    if not vi_filter_on(st):
        return rgb
    bits = DEPTH_BITS[str(st.color_depth)]
    h, w = rgb.shape[:2]
    out8 = np.zeros((h, w, 3), np.int32)
    dedither = bool(st.vi_dither_filter)
    for ch in range(3):
        k = to_bits(rgb[..., ch], bits[ch])
        c8 = k << np.int32(8 - bits[ch])
        if dedither:
            kp = np.pad(k, 1, mode='edge')     # a border neighbour equals the centre
            acc = np.zeros((h, w), np.int32)
            for dy, dx in _VI_OFFSETS:
                n = kp[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
                acc += np.sign(n - k).astype(np.int32)
            c8 = np.clip(c8 + acc, 0, 255)
        out8[..., ch] = c8
    mode = str(st.vi_gamma)
    if mode != 'NONE':
        xx, yy = np.meshgrid(np.arange(w, dtype=np.uint32),
                             np.arange(h, dtype=np.uint32))
        hsh = _hash_u32_raw(xx, yy, _sheet_key(frame, seed, 41))
        if mode == 'GAMMA':
            out8 = VI_GAMMA_LUT[out8]
        elif mode == 'GAMMA_DITHER':
            d = (hsh & np.uint32(63), (hsh >> np.uint32(6)) & np.uint32(63),
                 (hsh >> np.uint32(12)) & np.uint32(63))
            for ch in range(3):
                idx = (out8[..., ch] << np.int32(6)) | d[ch].astype(np.int32)
                out8[..., ch] = VI_GAMMA_DITHER_LUT[idx]
        elif mode == 'DITHER_ONLY':
            b = (hsh & np.uint32(1), (hsh >> np.uint32(1)) & np.uint32(1),
                 (hsh >> np.uint32(2)) & np.uint32(1))
            for ch in range(3):
                out8[..., ch] = np.minimum(255, out8[..., ch]
                                           + b[ch].astype(np.int32))
    return U8[out8]


# =========================================================================
# C019  GameCube / Wii EFB-to-XFB copy filter (libogc video.c vfilter,
#       Dolphin GetRAMCopyFilterCoefficients)
# =========================================================================

#: (U = c0+c1, M = c2+c3+c4, L = c5+c6) of the 7-tap sets, in 1/64: sum 64
COPY_TAPS = {'DEFLICKER': (16, 32, 16), 'DEFLICKER_AA': (12, 40, 12)}


def copy_filter_on(st):
    return str(getattr(st, 'copy_filter', 'NONE')) in COPY_TAPS


def copy_filter(rgb, st):
    """The copy's three-line vertical blur in 1/64 units, the 1/64 divide
    floored (a choice; the hardware's rounding is undocumented), sums
    over 64 clipped to white. Row 0 is the bottom: "the line above" on
    screen is row y+1; the first and last lines repeat themselves
    (GXSetCopyClamp)."""
    if not copy_filter_on(st):
        return rgb
    U, M, L = COPY_TAPS[str(st.copy_filter)]
    v = to_u8(rgb)
    h = v.shape[0]
    above = v[np.minimum(np.arange(h) + 1, h - 1)]
    below = v[np.maximum(np.arange(h) - 1, 0)]
    s = np.int32(U) * above
    t = np.int32(M) * v
    s = s + t
    t = np.int32(L) * below
    s = s + t
    out8 = np.minimum(s >> np.int32(6), 255)
    return U8[out8]


# =========================================================================
# C033  PlayStation 2 GS CRTC: PMODE's two-circuit blend
#       (ps2sdk libgs.h GS_PMODE, ps2tek)
# =========================================================================

#: the CRTC's last output, ONE entry: {'key', 'frame', 'out8'}
CRTC_STATE = {}

#: fields OUT of the fingerprint: _hold_fingerprint's own skip set, and
#: the device ROUTING fields (the same frame renders the same pixels on
#: either device, so a predecessor rendered on the other device IS RC2)
CRTC_SKIP = ('threads', 'process_count', 'use_processes',
             'render_device', 'gpu_post', 'gpu_shading', 'gpu_raster',
             'gpu_hold_context', 'layer_gpu_min_frac', 'gpu_scissor',
             'viewport_gpu')


def crtc_fingerprint(st):
    """Every PICTURE dial, the _hold_fingerprint shape."""
    parts = []
    for f in dataclasses.fields(st):
        if f.name in CRTC_SKIP:
            continue
        parts.append((f.name, repr(getattr(st, f.name))))
    return hash(tuple(parts))


def crtc_key(st, h, w):
    return (str(getattr(st, '_scene_name', '')), (int(h), int(w)),
            crtc_fingerprint(st))


def crtc_blend_on(st):
    return str(getattr(st, 'crtc_blend', 'NONE')) in ('PREVIOUS_FRAME',
                                                       'BG_COLOR')


def crtc_bg8(st, h, w):
    bg = to_u8(np.asarray(st.crtc_bg_color, np.float32)[:3]).reshape(1, 1, 3)
    return np.broadcast_to(bg, (h, w, 3)).astype(np.int32)


def crtc_rc2(st, frame, h, w):
    """RC2's picture and why: the previous CRTC output when this session
    rendered frame-1 at this size with these settings, else BGCOLOR
    (and a message that says so)."""
    bg8 = crtc_bg8(st, h, w)
    if str(st.crtc_blend) == 'BG_COLOR' or getattr(st, '_viewport', False):
        return bg8, None
    s = CRTC_STATE
    if s and s.get('key') == crtc_key(st, h, w) \
            and s.get('frame') == int(frame) - 1:
        return s['out8'], None
    return bg8, (f'CRTC blend: frame {int(frame)} has no rendered '
                 f'predecessor at {w}x{h} with these settings; RC2 reads '
                 'BGCOLOR')


def crtc_store(st, frame, out8):
    if getattr(st, '_viewport', False):
        return
    out8 = np.asarray(out8, np.int32)
    CRTC_STATE.clear()
    CRTC_STATE.update(key=crtc_key(st, *out8.shape[:2]), frame=int(frame),
                      out8=out8.copy())


def crtc_blend(rgb, st, frame=0):
    """`out = (RC1 * A + RC2 * (255 - A)) // 255` per byte: the GS's
    integer divide (floor). PREVIOUS_FRAME keeps its output as the next
    frame's RC2."""
    if not crtc_blend_on(st):
        return rgb
    h, w = rgb.shape[:2]
    A = min(max(int(st.crtc_alpha), 0), 255)
    cur8 = to_u8(rgb)
    prev8, why = crtc_rc2(st, frame, h, w)
    if why:
        _once(why)
    s = cur8 * np.int32(A)
    t = np.asarray(prev8, np.int32) * np.int32(255 - A)
    s = s + t
    out8 = s // 255
    if str(st.crtc_blend) == 'PREVIOUS_FRAME':
        crtc_store(st, frame, out8)
    return U8[out8]


# =========================================================================
# C069  3dfx Voodoo Graphics / Voodoo2 scan-out filter, "22-bit"
#       (86Box vid_voodoo_display.c voodoo_filterline_v1 / _v2)
# =========================================================================

def video_filter_why(st):
    if str(st.color_depth) == '16':
        return None
    return ("3dfx scan-out filter: the Voodoo's framebuffer is 16-bit "
            f"5:6:5; Colour Depth is '{st.color_depth}' (set 16)")


def video_filter_on(st):
    on = str(getattr(st, 'video_filter', 'NONE')) in ('VOODOO1', 'VOODOO2')
    if on and video_filter_why(st):
        _once(video_filter_why(st))         # named no-op, both roads
        return False
    return on


def expand565(rgb):
    """The RAMDAC's bit-replicating expansion of the 5:6:5 fields
    (not round(k * 255 / 31)). int32 (h, w, 3)."""
    k5 = to_bits(rgb[..., 0], 5)
    k6 = to_bits(rgb[..., 1], 6)
    k5b = to_bits(rgb[..., 2], 5)
    out = np.empty(rgb.shape[:2] + (3,), np.int32)
    out[..., 0] = (k5 << np.int32(3)) | (k5 >> np.int32(2))
    out[..., 1] = (k6 << np.int32(2)) | (k6 >> np.int32(4))
    out[..., 2] = (k5b << np.int32(3)) | (k5b >> np.int32(2))
    return out


def _voodoo_pass(p, direction, cap):
    """One 2-tap pass of the v1 line, reading the PREVIOUS pass's line:
    `c + ((clip(n - c, -cap, cap)) >> 1)` -- the table's `g + difference/2`
    stored to a byte, a FLOOR (arithmetic shift, both roads). The edge
    column with no neighbour keeps itself."""
    p = np.asarray(p, np.int32)
    q = p.copy()
    cap = np.int32(cap)
    if direction < 0:
        c = p[:, 1:]
        n = p[:, :-1]
        view = q[:, 1:]
    else:
        c = p[:, :-1]
        n = p[:, 1:]
        view = q[:, :-1]
    d = np.clip(n - c, -cap, cap)
    t = d >> np.int32(1)
    view[...] = c + t
    return q


def _voodoo2_step(g, h, cap, cap32):
    """86Box voodoo_generate_filter_v2's table thefilter[g][h] per pixel:
    only a BRIGHTER neighbour within the cap moves a pixel, by
    `min((g + 4h)//5 - (4g + h)//5, min(cap, 32))`."""
    g = np.asarray(g, np.int32)
    h = np.asarray(h, np.int32)
    d = h - g
    a = (4 * g + h) // 5
    b = (g + 4 * h) // 5
    ad = b - a
    step = np.minimum(ad, np.int32(cap32))
    return np.where((d > 0) & (d <= np.int32(cap)), g + step, g)


def _voodoo2_line(p, cap):
    """The v2 look-ahead line: per output pixel the cascade against the
    RAW source at -3, -2, -1, +1 (the order the emulator's loop resolves
    to); a clamped border neighbour equals the centre and moves nothing."""
    p = np.asarray(p, np.int32)
    w = p.shape[1]
    cap32 = min(int(cap), 32)
    q = p.copy()
    for k in (-3, -2, -1, 1):
        n = p[:, np.clip(np.arange(w) + k, 0, w - 1)]
        q = _voodoo2_step(q, n, cap, cap32)
    return q


def video_filter(rgb, st):
    """The 5:6:5 frame expanded as the RAMDAC does and filtered along the
    line: four cascaded 2-tap passes (VOODOO1: left, left, left, right)
    or the Voodoo2's one-sided look-ahead (VOODOO2)."""
    if not video_filter_on(st):
        return rgb
    p = expand565(rgb)
    cap = min(max(int(st.video_filter_threshold), 0), 255)
    if str(st.video_filter) == 'VOODOO2':
        return U8[np.clip(_voodoo2_line(p, cap), 0, 255)]
    p = _voodoo_pass(p, -1, cap)
    p = _voodoo_pass(p, -1, cap)
    p = _voodoo_pass(p, -1, cap)
    p = _voodoo_pass(p, +1, cap)
    return U8[np.clip(p, 0, 255)]


# =========================================================================
# C014  3DO display generator: 2x interpolated at a fixed cornerweight,
#       blue at 4 bits (3DO Portfolio, "Introduction to the 3DO Graphics
#       Systems")
# =========================================================================

_INDEXED_DEPTHS = ('8', '4', '1', 'HAM6', 'HAM8')


def threedo_2x_why(st):
    if DEPTH_BITS.get(str(st.color_depth)) == (5, 5, 5):
        return None
    return ("3DO 2x: the display generator reads a 15-bit frame; Colour "
            f"Depth is '{st.color_depth}': interpolating at the frame's own "
            'levels, blue LSB kept')


def threedo_2x(rgb, st):
    """(h, w, 3) -> (2h, 2w, 3): the sample of source pixel x sits at
    output column 2x, the column between is the floor-average with the
    right neighbour (the last column repeats itself); then the same over
    rows (row 0 = bottom). Blue's LSB is the cornerweight bit and is
    cleared on a 15-bit frame."""
    if threedo_2x_why(st):
        _once(threedo_2x_why(st))
    depth = str(st.color_depth)
    bits = (8, 8, 8) if depth in _INDEXED_DEPTHS else \
        DEPTH_BITS.get(depth, (8, 8, 8))
    h, w = rgb.shape[:2]
    k = np.empty((h, w, 3), np.int32)
    for ch in range(3):
        k[..., ch] = to_bits(rgb[..., ch], bits[ch])
    if bits[2] == 5:
        k[..., 2] &= ~np.int32(1)
    right = k[:, np.minimum(np.arange(w) + 1, w - 1)]
    kh = np.empty((h, 2 * w, 3), np.int32)
    kh[:, 0::2] = k
    kh[:, 1::2] = (k + right) >> np.int32(1)
    up = kh[np.minimum(np.arange(h) + 1, h - 1)]
    kv = np.empty((2 * h, 2 * w, 3), np.int32)
    kv[0::2] = kh
    kv[1::2] = (kh + up) >> np.int32(1)
    out = np.empty((2 * h, 2 * w, 3), np.float32)
    for ch in range(3):
        out[..., ch] = kv[..., ch].astype(np.float32) / \
            np.float32((1 << bits[ch]) - 1)
    return out


# =========================================================================
# C047  Game Boy Advance Mode 5: BG2's affine stretch in 8.8 fixed point
#       (GBATEK "BG Mode 5", Tonc "Affine backgrounds")
# =========================================================================

GBA_PA, GBA_PD = 0xAB, 0xCD          # 171/256, 205/256: the full-screen registers


def gba_source_size(tw, th):
    """The bitmap the LCD of (tw, th) reads through PA/PD: 240x160 -> 160x128."""
    return (((GBA_PA * (int(tw) - 1)) >> 8) + 1,
            ((GBA_PD * (int(th) - 1)) >> 8) + 1)


def gba_stretch(rgb, st, size=None):
    """(h, w, 3) -> (dh, dw, 3): output pixel X reads texel (PA*X) >> 8,
    truncated per pixel as the hardware's accumulator is; nearest, no
    filtering. Without `size` the output may carry one repeated trailing
    column or row (dw = ceil(256 w / PA)); `size=(tw, th)` crops to the LCD."""
    h, w = rgb.shape[:2]
    dw = (256 * w + GBA_PA - 1) // GBA_PA
    dh = (256 * h + GBA_PD - 1) // GBA_PD
    if size is not None:
        dw = min(dw, int(size[0]))
        dh = min(dh, int(size[1]))
    X = np.arange(dw)
    Y = np.arange(dh)
    px = (GBA_PA * X) >> 8
    py = (GBA_PD * Y) >> 8
    return np.ascontiguousarray(rgb[py][:, px])


gba_mode5 = gba_stretch
