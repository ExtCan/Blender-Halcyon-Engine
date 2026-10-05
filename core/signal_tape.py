"""R251: the post-signal pack's CPU stages -- the tape, the cable, the
receiver and the digital formats' chroma (SIG-2: C131, C129, C130, C128).

What sat between the machine's framebuffer and the glass once the
picture left the machine:

    C131  chroma_site   Y'CbCr at 8-bit legal levels, chroma sampled at
                        the format's rate and SITE (D1 4:2:2, DV 4:1:1,
                        MPEG-1 / MPEG-2 4:2:0, PAL DV, the GameCube XFB)
    C128  tape_path     the analogue tape: luma FM bandwidth, colour-under
                        chroma band-limited and delayed, the head-switch
                        tear, dropouts, every dub compounding all of it
    C129  svideo        the S-Video cable: the encoded chroma band-limited,
                        luma whole, no crawl
          rf_modulate   the RF modulator after the composite cable: luma
                        low-pass, the 920 kHz beat, snow, a ghost
    C130  pal_decode    the PAL receiver: the delay line, Hanover bars,
                        the eight-field crawl

Every function is bpy-free and device-free: `(rgb, st, ...)` in, float32
`(H, W, 3)` out, row 0 the bottom of the picture (post.process). Every
chain-stage function OPENS with `if not <name>_on(st): return rgb`, so it
is the identity on any settings that do not switch it on.

The arithmetic is INTEGER end to end on the frame's own 8-bit values
(core/signal_era's helpers: the BT.601 / YIQ / PAL YUV integers, `gauss16`
taps summing to 65536, `hash_s16`, `SIN16`): every sum is order-free,
every shift a floor, and the picture returns through the one byte table
`U8`, so the GPU twins in gpu/stages_tape.py are bitwise this code. The
tie rules, named: pair sums round half up (`+ 1 >> 1`), quads `+ 2 >> 2`,
FIRs `+ 32768 >> 16`, hash-to-noise floors (`>> 15`), the delay line's
two-line sum floors (`>> 1`), site neighbours and FIR taps clamp at the
frame edge.

Every scalar a stage needs (`*_params`) is computed ONCE here in float64
and rounded to an integer; both roads read the same dict.
"""

import math

import numpy as np

from .film import _hash_u32_raw
from .wear import _sheet_key
from .signal_era import (U8, to_u8, _once, y601, cb601, cr601, yuv601_decode,
                         yiq_int, yiq_back, pal_yuv, pal_back, hash_s16,
                         SIN16, isqrt16, gauss16)

#: the shader loop bound of every FIR here (taps -96..96, a 193-texel row)
FIR_BOUND = 96


def _stack8(r, g, b):
    """Three int32 planes -> the float32 picture through the byte table."""
    return U8[np.stack([np.clip(r, 0, 255), np.clip(g, 0, 255),
                        np.clip(b, 0, 255)], axis=-1)]


def isqrt_int(n):
    """Exact integer square root of a non-negative int array below 2**18
    (the GLSL twin is the nine-step binary search `hal_isqrt18`)."""
    n = np.asarray(n, np.int64)
    r = np.floor(np.sqrt(n.astype(np.float64))).astype(np.int64)
    r -= (r * r > n)
    r += ((r + 1) * (r + 1) <= n)
    return r.astype(np.int32)


def fir_h(plane, taps16, shift=0):
    """The integer FIR along a line, read at `x + shift`:
    `(sum_k taps16[k] * plane[clamp(x + shift + k)] + 32768) >> 16`, the
    index clamped to 0..w-1 AFTER the shift (the GLSL's clamp before
    `texelFetch`). int32 (h, w) in and out; the sum is order-free."""
    plane = np.asarray(plane, np.int64)
    w = plane.shape[1]
    R = (len(taps16) - 1) // 2
    x = np.arange(w)
    acc = np.zeros(plane.shape, np.int64)
    for k in range(-R, R + 1):
        t = int(taps16[k + R])
        if t == 0:
            continue
        acc += t * plane[:, np.clip(x + int(shift) + k, 0, w - 1)]
    return ((acc + 32768) >> 16).astype(np.int32)


def taps_image(taps_r, taps_g=None):
    """FIR taps as a (1, 193, 4) float32 image for a data texture: channel
    r = `taps_r` centred at index 96 (zero beyond its radius), channel g =
    `taps_g` likewise. Integers below 2**24 are exact as floats."""
    img = np.zeros((1, 2 * FIR_BOUND + 1, 4), np.float32)
    for ch, taps in ((0, taps_r), (1, taps_g)):
        if taps is None:
            continue
        R = (len(taps) - 1) // 2
        if R > FIR_BOUND:
            raise ValueError(f'FIR radius {R} exceeds the texture ({FIR_BOUND})')
        img[0, FIR_BOUND - R:FIR_BOUND + R + 1, ch] = \
            np.asarray(taps, np.float32)
    img[0, :, 3] = 1.0
    return img


# =========================================================================
# C131  Chroma subsampling and siting (Microsoft "Recommended 8-Bit YUV
#       Formats"; Dolphin TextureConversionShader.cpp for the XFB)
# =========================================================================

SITES = ('Y422', 'Y411', 'Y420_MPEG1', 'Y420_MPEG2', 'Y420_DVPAL', 'XFB_422')

#: the CHROMA_SITE stage's `fmt` uniform
SITE_FMT = {'Y422': 0, 'Y411': 1, 'Y420_MPEG1': 2, 'Y420_MPEG2': 3,
            'Y420_DVPAL': 4, 'XFB_422': 5}

_V420 = ('Y420_MPEG1', 'Y420_MPEG2', 'Y420_DVPAL')


def chroma_site_on(st):
    """The FILE formats: block A, before the display quantises."""
    fmt = str(getattr(st, 'chroma_format', 'NONE'))
    return fmt in SITES and fmt != 'XFB_422'


def xfb_on(st):
    """The machine's own copy (GameCube / Wii): block B, after copy_filter."""
    return str(getattr(st, 'chroma_format', 'NONE')) == 'XFB_422'


def chroma_sites(v8, fmt):
    """The stored chroma: (Cbs, Crs) int32 site arrays of the format.

    Y422 / XFB_422: (h, ceil(w/2)), sited on the even column. Y411:
    (h, ceil(w/4)). The 4:2:0 formats: (ceil(h/2), ceil(w/2)); row 0 is
    the bottom, a block's rows are 2j and 2j+1. Neighbours clamp at the
    frame edge; pair sums round half up, quads `+ 2 >> 2`. XFB_422
    averages the pair in RGB FIRST (floor), then runs the matrix."""
    h, w = v8.shape[:2]
    R, G, B = v8[..., 0], v8[..., 1], v8[..., 2]
    step = 4 if fmt == 'Y411' else 2
    nx = (w + step - 1) // step
    xs = [np.minimum(step * np.arange(nx) + o, w - 1) for o in range(step)]
    if fmt == 'XFB_422':
        m = (v8[:, xs[0]] + v8[:, xs[1]]) >> np.int32(1)
        return (np.clip(cb601(m[..., 0], m[..., 1], m[..., 2]), 16, 240),
                np.clip(cr601(m[..., 0], m[..., 1], m[..., 2]), 16, 240))
    Cb = np.clip(cb601(R, G, B), 16, 240)
    Cr = np.clip(cr601(R, G, B), 16, 240)
    if fmt == 'Y422':
        return ((Cb[:, xs[0]] + Cb[:, xs[1]] + 1) >> 1,
                (Cr[:, xs[0]] + Cr[:, xs[1]] + 1) >> 1)
    if fmt == 'Y411':
        return ((Cb[:, xs[0]] + Cb[:, xs[1]] + Cb[:, xs[2]] + Cb[:, xs[3]]
                 + 2) >> 2,
                (Cr[:, xs[0]] + Cr[:, xs[1]] + Cr[:, xs[2]] + Cr[:, xs[3]]
                 + 2) >> 2)
    ny = (h + 1) // 2
    y0 = 2 * np.arange(ny)
    y1 = np.minimum(y0 + 1, h - 1)
    if fmt == 'Y420_DVPAL':
        # Cb on the even line of the pair, Cr on the odd one
        return ((Cb[y0][:, xs[0]] + Cb[y0][:, xs[1]] + 1) >> 1,
                (Cr[y1][:, xs[0]] + Cr[y1][:, xs[1]] + 1) >> 1)

    def quad(C):
        return (C[y0][:, xs[0]] + C[y0][:, xs[1]]
                + C[y1][:, xs[0]] + C[y1][:, xs[1]] + 2) >> 2
    return quad(Cb), quad(Cr)


def chroma_upsample(Cbs, Crs, fmt, h, w, linear):
    """The decoder's full-resolution chroma from the sites: (Cb, Cr)
    int32 (h, w).

    HOLD: each pixel takes its site's sample. LINEAR, a fixed order --
    horizontal first, then vertical on the horizontal result, every sum
    half-up: co-sited formats (Y422, MPEG-2, PAL DV, XFB at 2i; Y411 at
    4i) interpolate between the site and the next one; MPEG-1's centred
    sites (2i + 0.5, 2j + 0.5) weigh 3:1 toward the pixel's own site;
    MPEG-2 is centred vertically; PAL DV's Cb sits on rows 2j and its Cr
    on rows 2j+1."""
    x = np.arange(w)
    y = np.arange(h)
    hs = 2 if fmt == 'Y411' else 1
    v420 = fmt in _V420
    i = x >> hs
    j = (y >> 1) if v420 else y
    if not linear:
        return Cbs[j][:, i], Crs[j][:, i]
    nx = Cbs.shape[1]

    def ir(a):
        return np.clip(a, 0, nx - 1)

    def horiz(Cs):
        if fmt == 'Y411':
            k = x & 3
            return (Cs[:, i] * (4 - k) + Cs[:, ir(i + 1)] * k + 2) >> 2
        if fmt == 'Y420_MPEG1':
            nb = np.where((x & 1) == 0, i - 1, i + 1)
            return (3 * Cs[:, i] + Cs[:, ir(nb)] + 2) >> 2
        return np.where((x & 1) == 0, Cs[:, i],
                        (Cs[:, i] + Cs[:, ir(i + 1)] + 1) >> 1)

    Hb, Hr = horiz(Cbs), horiz(Crs)
    if not v420:
        return Hb, Hr
    ny = Hb.shape[0]

    def jr(a):
        return np.clip(a, 0, ny - 1)

    odd = ((y & 1) == 1)[:, None]
    if fmt == 'Y420_DVPAL':
        cb = np.where(odd, (Hb[j] + Hb[jr(j + 1)] + 1) >> 1, Hb[j])
        cr = np.where(odd, Hr[j], (Hr[jr(j - 1)] + Hr[j] + 1) >> 1)
        return cb, cr
    nb = np.where((y & 1) == 0, j - 1, j + 1)
    return ((3 * Hb[j] + Hb[jr(nb)] + 2) >> 2,
            (3 * Hr[j] + Hr[jr(nb)] + 2) >> 2)


def chroma_site(rgb, st):
    """The frame through the format's Y'CbCr: luma per pixel at the legal
    16..235, chroma at the format's sites (16..240), rebuilt held or
    linear, decoded with the 298 / 409 / 100 / 208 / 516 integers. One
    function serves both wiring lines; the format decides the site."""
    if not (chroma_site_on(st) or xfb_on(st)):
        return rgb
    fmt = str(st.chroma_format)
    linear = str(getattr(st, 'chroma_upsample', 'HOLD')) == 'LINEAR'
    v8 = to_u8(rgb)
    h, w = v8.shape[:2]
    Y = np.clip(y601(v8[..., 0], v8[..., 1], v8[..., 2]), 16, 235)
    Cbs, Crs = chroma_sites(v8, fmt)
    Cb, Cr = chroma_upsample(Cbs, Crs, fmt, h, w, linear)
    return _stack8(*yuv601_decode(Y, Cb, Cr))


# =========================================================================
# C129  The cable: S-Video (NTSC / PAL) and the RF modulator
#       (Extron "NTSC Decoding Basics"; Analog Devices "Conditioning A/V
#       Signals for RF Modulation"; the 920 kHz sound-chroma beat)
# =========================================================================

#: (space, chroma band 1 MHz, chroma band 2 MHz, active line us)
CABLE_TABLE = {'SVIDEO': ('YIQ', 1.3, 0.5, 52.66),
               'SVIDEO_PAL': ('YUV', 1.3, 1.3, 52.0)}

NTSC_LINE_US = 52.66


def cable_chroma_on(st):
    on = str(getattr(st, 'signal', 'RGB')) in CABLE_TABLE
    if on and bool(getattr(st, 'composite', False)):
        _once('S-Video and Composite Video are both on; the cable is one or '
              'the other (the composite stage still runs on top)')
    return on


def _sigma_radius(sig):
    return max(int(math.ceil(3.0 * sig)), 1)


def cable_params(st, w):
    """The S-Video cable's two chroma low-passes at this width:
    sigma_px(f MHz) = W / (2 f * active line us)."""
    space, f1, f2, line_us = CABLE_TABLE[str(st.signal)]
    sig1 = float(w) / (2.0 * f1 * line_us)
    sig2 = float(w) / (2.0 * f2 * line_us)
    r1, r2 = _sigma_radius(sig1), _sigma_radius(sig2)
    return {'space': 0 if space == 'YIQ' else 1, 'r1': r1, 'r2': r2,
            'taps1': gauss16(sig1, r1), 'taps2': gauss16(sig2, r2)}


def svideo(rgb, st):
    """S-Video: luma whole; the encoded chroma (I 1.3 / Q 0.5 MHz as a
    wideband decoder passes the NTSC encoder's bands; U and V 1.3 MHz for
    PAL) through its Gaussian low-pass. No crawl: luma and chroma never
    shared a wire."""
    if not cable_chroma_on(st):
        return rgb
    P = cable_params(st, rgb.shape[1])
    v8 = to_u8(rgb)
    R, G, B = v8[..., 0], v8[..., 1], v8[..., 2]
    if P['space'] == 0:
        Y, C1, C2 = yiq_int(R, G, B)
    else:
        Y, C1, C2 = pal_yuv(R, G, B)
    C1 = fir_h(C1, P['taps1'])
    C2 = fir_h(C2, P['taps2'])
    back = yiq_back if P['space'] == 0 else pal_back
    return _stack8(*back(Y, C1, C2))


cable_chroma = svideo


def cable_rf_on(st):
    return str(getattr(st, 'signal', 'RGB')) == 'RF'


def rf_params(st, w, frame, seed):
    """Every scalar of the RF stage, once, as integers."""
    sig = float(w) / (2.0 * float(st.rf_bandwidth) * NTSC_LINE_US)
    R = _sigma_radius(sig)
    return {
        'r_fir': R, 'taps': gauss16(sig, R),
        # 920 kHz over the pixel rate W / 52.66 MHz, in 1/65536 cycle per pixel
        'a_ph': int(round(65536.0 * 0.92 * NTSC_LINE_US / float(w))),
        'b_ph': 30802,                     # 0.47 cycle per line
        'phase_f': (int(frame) * 42598) & 0xffff,   # 0.65 cycle per frame
        'amp_beat': int(round(float(st.rf_beat) * 256.0)),
        'amp_snow': int(round(255.0 * float(st.rf_snow))),
        'amp_snow_c': int(round(255.0 * 0.7 * float(st.rf_snow))),
        'ghost_g': int(round(float(st.rf_ghost) * 256.0)),
        'ghost_d': int(round(float(st.rf_ghost_delay) * float(w)
                             / NTSC_LINE_US)),
        'k_y': _sheet_key(frame, seed, 49),
        'k_i': _sheet_key(frame, seed, 50),
        'k_q': _sheet_key(frame, seed, 51),
    }


#: the CABLE_RF stage's int uniforms, in order
RF_INTS = ('r_fir', 'a_ph', 'b_ph', 'phase_f', 'amp_beat', 'amp_snow',
           'amp_snow_c', 'ghost_g', 'ghost_d', 'k_y', 'k_i', 'k_q')


def rf_modulate(rgb, st, frame=0, seed=0):
    """The RF modulator after the composite cable: the modulator's luma
    low-pass, the 920 kHz herringbone in proportion to chroma amplitude,
    thermal snow on luma and chroma, and a multipath ghost that reads
    the stage INPUT at x - d (a single-pass definition, named)."""
    if not cable_rf_on(st):
        return rgb
    h, w = rgb.shape[:2]
    P = rf_params(st, w, frame, seed)
    v8 = to_u8(rgb)
    Y, I, Q = yiq_int(v8[..., 0], v8[..., 1], v8[..., 2])
    Yf = fir_h(Y, P['taps'])
    C = isqrt16(I * I + Q * Q)                      # <= 216, exact
    xx, yy = np.meshgrid(np.arange(w, dtype=np.int64),
                         np.arange(h, dtype=np.int64))
    ph = (xx * P['a_ph'] + yy * P['b_ph'] + P['phase_f']) & 0xffff
    s16 = SIN16[(ph >> 8).astype(np.int32)]
    dbeat = (((C * s16) >> np.int32(8)) * np.int32(P['amp_beat'])) \
        >> np.int32(16)
    ux, uy = xx.astype(np.uint32), yy.astype(np.uint32)
    dsnow = (hash_s16(ux, uy, P['k_y']) * np.int32(P['amp_snow'])) \
        >> np.int32(15)
    dI = (hash_s16(ux, uy, P['k_i']) * np.int32(P['amp_snow_c'])) \
        >> np.int32(15)
    dQ = (hash_s16(ux, uy, P['k_q']) * np.int32(P['amp_snow_c'])) \
        >> np.int32(15)
    xg = np.clip(np.arange(w) - P['ghost_d'], 0, w - 1)
    g = np.int32(P['ghost_g'])
    Yg = (g * Y[:, xg]) >> np.int32(8)
    Ig = (g * I[:, xg]) >> np.int32(8)
    Qg = (g * Q[:, xg]) >> np.int32(8)
    Yo = np.clip(Yf + dbeat + dsnow + Yg, 0, 255)
    Io = I + dI + Ig
    Qo = Q + dQ + Qg
    return _stack8(*yiq_back(Yo, Io, Qo))


cable_rf = rf_modulate


# =========================================================================
# C130  The PAL receiver: delay line, Hanover bars, PAL crawl
#       (Wikipedia "PAL", "Hanover bars"; NESdev "PAL video")
# =========================================================================

PAL_MODES = {'NONE': 0, 'DELAY_LINE': 1, 'SIMPLE': 2}

PAL_LINE_US = 52.0
PAL_FSC_MHZ = 4.43361875


def pal_decode_on(st):
    return str(getattr(st, 'pal_decoder', 'NONE')) in ('DELAY_LINE', 'SIMPLE') \
        or float(getattr(st, 'pal_crawl', 0.0)) > 0.0


def pal_params(st, w, frame):
    """Every scalar of the PAL stage, once, as integers."""
    phi = math.radians(float(st.pal_phase_error))
    return {
        'mode': PAL_MODES.get(str(st.pal_decoder), 0),
        # the same-field neighbour when the frame is fielded
        's_rows': 2 if str(st.interlace) != 'NONE' else 1,
        'c16': int(round(65536.0 * math.cos(phi))),
        's16': int(round(65536.0 * math.sin(phi))),
        # subcarrier cycles per pixel over the 52 us line, in 1/65536
        'a_ph': int(round(65536.0 * PAL_FSC_MHZ * PAL_LINE_US / float(w))),
        'b_ph': 49152,                              # 0.75 cycle per line
        # 0.875 cycle per field: the 8-field sequence (57344 * 8 = 7 * 65536)
        'phase_f': (int(frame) * 57344) & 0xffff,
        'amp16': int(round(float(st.pal_crawl) * 0.06 * 65536.0)),
    }


#: the PAL_DECODE stage's int uniforms, in order
PAL_INTS = ('mode', 's_rows', 'c16', 's16', 'a_ph', 'b_ph', 'phase_f',
            'amp16')


def pal_decode(rgb, st, frame=0):
    """The PAL set's chroma decoder on the offset-free integer YUV: the
    delay line averages each line's chroma with the line above in the same
    field (the top line keeps its own); the simple decoder rotates (U, V)
    by +phi on even rows and -phi on odd rows (Hanover bars); then the
    4.43 MHz crawl on luma, in proportion to the DECODED chroma amplitude."""
    if not pal_decode_on(st):
        return rgb
    h, w = rgb.shape[:2]
    P = pal_params(st, w, frame)
    v8 = to_u8(rgb)
    Y, U, V = pal_yuv(v8[..., 0], v8[..., 1], v8[..., 2])
    if P['mode'] == 1:
        rows = np.arange(h)
        up = rows + P['s_rows']
        has = up <= h - 1
        upc = np.where(has, up, rows)               # no line above: itself
        U = (U + U[upc]) >> np.int32(1)
        V = (V + V[upc]) >> np.int32(1)
    elif P['mode'] == 2:
        sg = np.where((np.arange(h) & 1) == 0, P['s16'], -P['s16'])
        sg = sg.astype(np.int32)[:, None]
        c16 = np.int32(P['c16'])
        U2 = (c16 * U - sg * V + 32768) >> np.int32(16)
        V2 = (sg * U + c16 * V + 32768) >> np.int32(16)
        U, V = U2, V2
    if P['amp16'] > 0:
        C = isqrt_int(U * U + V * V)
        xx, yy = np.meshgrid(np.arange(w, dtype=np.int64),
                             np.arange(h, dtype=np.int64))
        ph = (xx * P['a_ph'] + yy * P['b_ph'] + P['phase_f']) & 0xffff
        s = SIN16[(ph >> 8).astype(np.int32)]
        dY = (((C * s) >> np.int32(8)) * np.int32(P['amp16'])) >> np.int32(23)
        Y = np.clip(Y + dY, 0, 255)
    return _stack8(*pal_back(Y, U, V))


# =========================================================================
# C128  The tape path with generation loss (Wikipedia "VHS", "Betacam";
#       US Patent 5,386,296 for the colour-under carrier)
# =========================================================================

#: (luma TVL, chroma TVL, Y/C delay us)
TAPE_TABLE = {'VHS': (240, 40, 0.6), 'SVHS': (400, 40, 0.6),
              'BETAMAX': (250, 40, 0.6), 'UMATIC': (260, 45, 0.5),
              'VIDEO8': (240, 40, 0.6), 'HI8': (400, 40, 0.6),
              'BETACAM': (300, 120, 0.0), 'BETACAM_SP': (340, 120, 0.0),
              'TYPE_C': (400, 120, 0.0)}


def tape_on(st):
    return str(getattr(st, 'tape', 'NONE')) in TAPE_TABLE


def tape_params(st, h, w, frame, seed):
    """Every scalar of the tape path ONCE, in float64, then an integer;
    both roads read this dict."""
    tvl_y, tvl_c, delay_us = TAPE_TABLE[str(st.tape)]
    N = min(max(int(st.tape_generations), 1), 8)
    sN = math.sqrt(N)
    # sigma_px(TVL) = 0.5 * W / (TVL * 4/3); N dubs = one Gaussian of sigma*sqrt(N)
    sig_y = 0.375 * w / tvl_y * sN
    sig_c = 0.375 * w / tvl_c * sN
    r_y, r_c = _sigma_radius(sig_y), _sigma_radius(sig_c)
    # (one-element arrays: NumPy warns on a uint32 SCALAR multiply that wraps)
    hf = int(_hash_u32_raw(np.array([int(frame) & 0xffffffff], np.uint32),
                           np.zeros(1, np.uint32),
                           _sheet_key(frame, seed, 45))[0]) / 16777216.0
    return {
        'N': N, 'r_y': r_y, 'r_c': r_c,
        'taps_y': gauss16(sig_y, r_y), 'taps_c': gauss16(sig_c, r_c),
        # the NTSC active line; the Y/C delay compounds per dub
        'delay': int(round(N * delay_us * w / NTSC_LINE_US)),
        'amp8': int(round(255.0 * float(st.tape_noise) * sN)),
        'head_rows': int(round(7.0 * h / 480.0))
        if bool(st.tape_head_switch) else 0,
        'head_j': int(round((hf - 0.5) * 4.0 * w / 320.0)),
        'drop_thr': min(max(int(float(st.tape_dropouts) / float(h)
                                * 16777216.0), 0), 16777216),
        'k_y': _sheet_key(frame, seed, 42),
        'k_cb': _sheet_key(frame, seed, 43),
        'k_cr': _sheet_key(frame, seed, 44),
        'k_row': _sheet_key(frame, seed, 46),
        'k_len': _sheet_key(frame, seed, 47),
        'k_x0': _sheet_key(frame, seed, 48),
        'frame_u': int(frame) & 0x7fffffff, 'wi': int(w),
    }


#: the TAPE_OUT stage's int uniforms, in order
TAPE_OUT_INTS = ('amp8', 'head_rows', 'head_j', 'drop_thr', 'k_y', 'k_cb',
                 'k_cr', 'k_row', 'k_len', 'k_x0', 'frame_u', 'wi')


def _tape_fir(v8, P):
    """The record / playback low-pass: (Yf, Cbf, Crf) int32. Luma at the
    format's TVL; the colour-under chroma at its own TVL, read at
    x - delay (it lands to the right of the luma)."""
    R, G, B = v8[..., 0], v8[..., 1], v8[..., 2]
    return (fir_h(y601(R, G, B), P['taps_y']),
            fir_h(cb601(R, G, B), P['taps_c'], shift=-P['delay']),
            fir_h(cr601(R, G, B), P['taps_c'], shift=-P['delay']))


def _tape_out(Yf, Cbf, Crf, P, h, w, frame=0):
    """The deck's output from the filtered planes, in the pinned order:
    the head-switch tear, the noise (luma, then chroma), the dropouts,
    the decode. Returns (R, G, B) int32 0..255; `_tape_ycc` is the
    Y'CbCr before the decode."""
    return yuv601_decode(*_tape_ycc(Yf, Cbf, Crf, P, h, w))


def _tape_ycc(Yf, Cbf, Crf, P, h, w):
    """(Y, Cb, Cr) int32 after tear, noise and dropouts: luma in 16..235,
    chroma in 16..240 (a deck's FM demodulator delivers nothing below
    black or above white); a dropout is legal white (235, 128, 128)."""
    Yf = np.asarray(Yf, np.int32).copy()
    xx, yy = np.meshgrid(np.arange(w, dtype=np.uint32),
                         np.arange(h, dtype=np.uint32))
    hr = int(P['head_rows'])
    band = np.zeros((h, 1), bool)
    if hr > 0:
        band[:hr] = True
        xs = np.clip(np.arange(w) + int(P['head_j']), 0, w - 1)
        Yf[:hr] = Yf[:hr][:, xs]                    # the tear (row 0 = bottom)
    amp = np.int32(P['amp8'])
    n = hash_s16(xx, yy, P['k_y']) * amp
    dY = np.where(band, n >> np.int32(13), n >> np.int32(15))   # x4 in the band
    Y = np.clip(Yf + dY, 16, 235)
    Cb = np.clip(np.asarray(Cbf, np.int32)
                 + ((hash_s16(xx, yy, P['k_cb']) * amp) >> np.int32(16)),
                 16, 240)
    Cr = np.clip(np.asarray(Crf, np.int32)
                 + ((hash_s16(xx, yy, P['k_cr']) * amp) >> np.int32(16)),
                 16, 240)
    rr = np.arange(h, dtype=np.uint32)
    fu = np.full(h, int(P['frame_u']), np.uint32)
    hit = _hash_u32_raw(rr, fu, P['k_row']).astype(np.int64) \
        < int(P['drop_thr'])
    hl = _hash_u32_raw(rr, fu, P['k_len'])
    length = 8 + (((hl & np.uint32(0xff)).astype(np.int64) * 41) >> 8)   # 8..48
    hx = _hash_u32_raw(rr, fu, P['k_x0'])
    x0 = ((hx & np.uint32(0xffff)).astype(np.int64) * int(P['wi'])) >> 16
    x = np.arange(w, dtype=np.int64)[None, :]
    mask = hit[:, None] & (x >= x0[:, None]) & (x < (x0 + length)[:, None])
    Y = np.where(mask, 235, Y).astype(np.int32)
    Cb = np.where(mask, 128, Cb).astype(np.int32)
    Cr = np.where(mask, 128, Cr).astype(np.int32)
    return Y, Cb, Cr


def tape_path(rgb, st, frame=0, seed=0):
    """The analogue tape between the machine and the cable: FIR luma, FIR
    and delay chroma, the head-switch tear, noise, dropouts, decode."""
    if not tape_on(st):
        return rgb
    h, w = rgb.shape[:2]
    P = tape_params(st, h, w, frame, seed)
    Yf, Cbf, Crf = _tape_fir(to_u8(rgb), P)
    return _stack8(*_tape_out(Yf, Cbf, Crf, P, h, w, frame))


tape = tape_path
