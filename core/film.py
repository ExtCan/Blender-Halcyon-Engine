"""The era looks (R230): the cel photographed and printed.

Everything a finished cel went through after the paint dried -- the
film stock's colour response, the rostrum camera's optics, the gate's
registration, the print's dust and grain, the projector's flicker, the
newsprint's dot screen -- and the animator's own economy, shooting on
twos and threes. Every stage here is a pure function of the frame, the
frame number and the seed, so a frame renders the same bits on either
device, in any band, on any worker: the frame number IS the clock.

Two halves:

- the FILM stages (`colour_process`, `grade`, `soften`, `weave`,
  `dust`, `grain`, `flicker`, `halftone`, and the print's wear in
  `wear.py`) run in the post chain -- the linear frame in, the linear
  frame out -- exactly where `post.process` calls them;
- the PAINT stages (`misregister_and_bleed`) run at the internal
  resolution BEFORE the ink lines are drawn, so the lines land crisp
  over paint that slipped and soaked, the way a hand-inked cel does.

Frame holds (`hold_key`) live in the engine: a held frame is not
rendered at all -- the key frame's cel is photographed again, and the
film stages still run per frame (fresh grain, fresh dust, the gate
weaving), exactly what the camera saw when the animator shot on twos.
"""

import numpy as np

from .ink import _hash_u32, value_noise


def _hash_u32_raw(ix, iy, k):
    """ink._hash_u32's integer, before the float scaling: 24 usable
    bits per pixel to slice. Takes uint32 grids as they are."""
    x = np.asarray(ix)
    y = np.asarray(iy)
    if x.dtype != np.uint32:
        x = (np.asarray(x, np.int64) & 0xffffffff).astype(np.uint32)
    if y.dtype != np.uint32:
        y = (np.asarray(y, np.int64) & 0xffffffff).astype(np.uint32)
    h = x * np.uint32(0x9E3779B1) ^ (y + np.uint32(0x85EBCA77)) \
        ^ np.uint32((int(k) * 0xC2B2AE3D) & 0xffffffff)
    h ^= h >> np.uint32(15)
    h *= np.uint32(0x2C1B3C6D)
    h ^= h >> np.uint32(12)
    h *= np.uint32(0x297A2D39)
    h ^= h >> np.uint32(15)
    return h & np.uint32(0xffffff)

#: the stock grades. Each is (name, label, description) for the UI and a
#: recipe for `grade`: a 3x3 matrix in linear light (the dyes' cross
#: talk), lift/gain per channel, a mid-preserving contrast power, and a
#: saturation factor -- applied in that order. The numbers are the
#: documented character of each process, not a measured LUT: the
#: three-strip's purified primaries and cyan-leaning shadows, the faded
#: Eastmancolor print's magenta cast and lifted blacks, the panchromatic
#: negative's red weighting.
FILM_GRADE_ITEMS = (
    ('NONE', "None", "No stock: the render's own colour"),
    ('TECHNICOLOR', "Technicolor (three-strip)",
     "The dye-transfer print: purified, saturated primaries, deep reds, "
     "shadows leaning cyan, a firm contrast curve -- the Golden Age "
     "feature look"),
    ('EASTMAN_70S', "Eastmancolor, faded (1970s print)",
     "A print whose cyan dye has faded: magenta-red cast, lifted blacks, "
     "softened contrast and saturation -- the television syndication "
     "print of the 70s"),
    ('TV_80S', "Video (1980s broadcast)",
     "The telecine chain: slightly warm, a touch less saturated, "
     "highlights held back -- the OVA on a good tape"),
    ('VHS', "VHS dub",
     "The worn tape: low saturation, milky blacks, dim highlights (pair "
     "with the Composite Video stages for the chroma smear)"),
    ('MONO', "Black and white (panchromatic)",
     "The 1930s negative: luminance weighted toward red, no colour at "
     "all"),
)

_GRADES = {
    'TECHNICOLOR': dict(
        matrix=((1.14, -0.09, -0.05), (-0.07, 1.06, 0.01),
                (-0.05, -0.09, 1.14)),
        lift=(0.0, 0.006, 0.014), gain=(1.03, 1.0, 0.97),
        contrast=1.12, saturation=1.22),
    'EASTMAN_70S': dict(
        matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        lift=(0.07, 0.035, 0.045), gain=(1.0, 0.93, 0.88),
        contrast=0.9, saturation=0.8),
    'TV_80S': dict(
        matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        lift=(0.02, 0.018, 0.015), gain=(0.97, 0.955, 0.93),
        contrast=0.97, saturation=0.9),
    'VHS': dict(
        matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        lift=(0.08, 0.075, 0.085), gain=(0.86, 0.85, 0.84),
        contrast=0.92, saturation=0.68),
    'MONO': dict(
        matrix=((0.30, 0.55, 0.15), (0.30, 0.55, 0.15), (0.30, 0.55, 0.15)),
        lift=(0.0, 0.0, 0.0), gain=(1.0, 1.0, 1.0),
        contrast=1.06, saturation=1.0),
}

#: the grey the contrast curve pivots on (linear 18%)
_MID = np.float32(0.18)

#: mid-tone luminance weights, linear light
_LUM = np.array([0.2126, 0.7152, 0.0722], np.float32)


def grade(rgb, st):
    """The stock's colour response, in linear light. Amount mixes the
    graded frame back over the original."""
    key = str(getattr(st, 'film_grade', 'NONE') or 'NONE').upper()
    amt = float(np.clip(getattr(st, 'film_grade_amount', 1.0), 0.0, 1.0))
    rec = _GRADES.get(key)
    if rec is None or amt <= 0.0:
        return rgb
    x = np.asarray(rgb, np.float32)
    m = np.asarray(rec['matrix'], np.float32)
    lift = np.asarray(rec['lift'], np.float32)
    gain = np.asarray(rec['gain'], np.float32)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    out = np.empty_like(x)
    for c in range(3):
        # the dyes' cross talk, then this channel's gain and lift --
        # written per channel: a 3-wide matmul over a frame is the slow
        # path of every BLAS
        out[..., c] = (r * m[c, 0] + g * m[c, 1] + b * m[c, 2]) \
            * gain[c] + lift[c]
    c = float(rec['contrast'])
    if abs(c - 1.0) > 1e-6:
        # mid-preserving power: (x/mid)^c * mid -- as exp(c log x), six
        # times cheaper than numpy's float32 pow, exact zero kept
        pos = np.maximum(out, 0.0)
        out = np.where(pos > 0.0,
                       np.exp(np.float32(c) * np.log(
                           np.maximum(pos, np.float32(1e-30)) / _MID)) * _MID,
                       np.float32(0.0)).astype(np.float32)
    s = float(rec['saturation'])
    if abs(s - 1.0) > 1e-6:
        lum = (out[..., 0] * _LUM[0] + out[..., 1] * _LUM[1]
               + out[..., 2] * _LUM[2])[..., None]
        out = lum + (out - lum) * np.float32(s)
    out = np.maximum(out, 0.0).astype(np.float32)
    if amt < 1.0:
        out = x + (out - x) * np.float32(amt)
    return out.astype(np.float32)


def _gauss_kernel(sigma):
    r = int(np.ceil(3.0 * sigma))
    xs = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-0.5 * (xs / np.float32(sigma)) ** 2).astype(np.float32)
    return k / k.sum()


def gaussian(img, sigma):
    """A separable Gaussian blur with a FLOAT sigma in pixels, edge-
    padded, as direct taps summed in a fixed order -- a cumsum box blur
    would be cheaper but its running sums depend on where the rows start,
    and a pooled band must blur bitwise what the whole frame blurs."""
    sigma = float(sigma)
    if sigma <= 1e-3:
        return img
    k = _gauss_kernel(sigma)
    r = (k.size - 1) // 2
    x = np.asarray(img, np.float32)
    pad = np.pad(x, ((0, 0), (r, r), (0, 0)), mode='edge')
    out = pad[:, 0:x.shape[1]] * k[0]
    for i in range(1, k.size):
        out += pad[:, i:i + x.shape[1]] * k[i]
    pad = np.pad(out, ((r, r), (0, 0), (0, 0)), mode='edge')
    out2 = pad[0:x.shape[0]] * k[0]
    for i in range(1, k.size):
        out2 += pad[i:i + x.shape[0]] * k[i]
    return out2.astype(np.float32)


def soften(rgb, st):
    """The rostrum camera and the optical printer: a Gaussian of Film
    Softness pixels (sigma), the whole frame."""
    s = float(max(getattr(st, 'film_softness', 0.0), 0.0))
    if s <= 1e-3:
        return rgb
    return gaussian(rgb, s)


def _frame_noise(frame, seed, k):
    """One hashed value in [0, 1) per (frame, seed, k)."""
    return float(_hash_u32(int(frame), int(seed) & 0xffff, k)[0])


def weave_offset(frame, seed, amplitude):
    """The gate's registration error for `frame`, in pixels: a slow
    wander (value noise over the frame count) plus a per-frame jitter,
    both sides, vertical a touch larger than horizontal as a claw-fed
    gate has it. Deterministic per (frame, seed)."""
    a = float(max(amplitude, 0.0))
    if a <= 0.0:
        return 0.0, 0.0
    f = float(int(frame))
    sx = float(value_noise(f * 0.23 + 0.37, 0.5,
                           91 + (int(seed) & 0xffff))[0])
    sy = float(value_noise(f * 0.19 + 7.11, 3.5,
                           92 + (int(seed) & 0xffff))[0])
    jx = _frame_noise(frame, seed, 93)
    jy = _frame_noise(frame, seed, 94)
    dx = a * (0.6 * (2.0 * sx - 1.0) + 0.4 * (2.0 * jx - 1.0)) * 0.8
    dy = a * (0.6 * (2.0 * sy - 1.0) + 0.4 * (2.0 * jy - 1.0))
    return float(dx), float(dy)


def shift(img, dx, dy):
    """Translate an (H, W, C) image by a float offset, bilinear, edge
    padded (the frame edge smears the way a weaving print's does). Four
    integer-shifted slices weighted by the fraction: no gathers, and the
    same four terms in the same order wherever the rows start."""
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return img
    x = np.asarray(img, np.float32)
    h, w = x.shape[:2]
    # the source coordinate of pixel (i, j) is (i - dy, j - dx)
    fx0 = int(np.floor(-dx))
    fy0 = int(np.floor(-dy))
    tx = np.float32(-dx - fx0)
    ty = np.float32(-dy - fy0)
    px = abs(fx0) + 1
    py = abs(fy0) + 1
    pad = np.pad(x, ((py, py), (px, px), (0, 0)), mode='edge')

    def sl(oy, ox):
        return pad[py + oy:py + oy + h, px + ox:px + ox + w]

    a = sl(fy0, fx0) * (1.0 - tx) + sl(fy0, fx0 + 1) * tx
    b = sl(fy0 + 1, fx0) * (1.0 - tx) + sl(fy0 + 1, fx0 + 1) * tx
    return (a * (1.0 - ty) + b * ty).astype(np.float32)


def weave(rgb, st, frame=0, seed=0):
    """Gate weave: the whole frame shifted by this frame's registration
    error, Film Weave pixels of amplitude."""
    a = float(max(getattr(st, 'film_weave', 0.0), 0.0))
    if a <= 0.0:
        return rgb
    dx, dy = weave_offset(frame, seed, a)
    return shift(rgb, dx, dy)


def dust(rgb, st, frame=0, seed=0):
    """Dirt on the print (R237, `wear.print_dirt`): opaque, nearly
    black, fresh every frame, Film Dust the density -- about forty
    specks on a 1080p frame at 1.0, scaling with the frame's area,
    less the shares Negative Dust and Cel Dust take for themselves."""
    from . import wear as WEAR
    return WEAR.print_dirt(rgb, st, frame, seed)


def grain(rgb, st, frame=0, seed=0):
    """The emulsion's grain on a frame that went through no colour
    process (R237, `wear.grain_plain`): the frame's channels read as a
    print's transmittance, density noise strongest where half the
    grains developed, sheets of Grain Size, in clumps, coloured by
    Grain Chroma. Every frame a fresh sheet; the same frame always the
    same sheet."""
    from . import wear as WEAR
    return WEAR.grain_plain(rgb, st, frame, seed)


def flicker(rgb, st, frame=0, seed=0):
    """Projector flicker: the whole frame's exposure wanders by up to
    Film Flicker * 12 percent, per frame, by the frame's hash."""
    f = float(np.clip(getattr(st, 'film_flicker', 0.0), 0.0, 1.0))
    if f <= 0.0:
        return rgb
    k = 1.0 + f * 0.12 * (2.0 * _frame_noise(frame, seed, 21) - 1.0)
    return (np.asarray(rgb, np.float32) * np.float32(k)).astype(np.float32)


#: the classic four-colour screen angles, degrees: cyan 15, magenta 75,
#: yellow 0, black 45 -- the rosette that hides the moire
_SCREEN_ANGLES = {'C': 15.0, 'M': 75.0, 'Y': 0.0, 'K': 45.0}


def halftone(rgb, st):
    """Ben-Day / newsprint dots on the display-referred frame: four
    ink screens (CMY plus a partial black) at the classic angles, dots
    sized by the square root of each ink's coverage at the CELL CENTRE,
    anti-aliased over one pixel, ink laid subtractively. Film Halftone
    mixes the print over the frame; Halftone Pitch is the screen's cell
    in pixels."""
    amt = float(np.clip(getattr(st, 'film_halftone', 0.0), 0.0, 1.0))
    if amt <= 0.0:
        return rgb
    pitch = float(max(getattr(st, 'film_halftone_pitch', 6.0), 2.0))
    x = np.clip(np.asarray(rgb, np.float32), 0.0, 1.0)
    h, w = x.shape[:2]
    from .post import _resample
    # the inks: CMY with a partial black pulled from the common shadow
    c = 1.0 - x[..., 0]
    m = 1.0 - x[..., 1]
    y = 1.0 - x[..., 2]
    # full under-colour removal: the grey component prints as black
    # ink alone, so a black pixel is solid K and a grey a pure K screen
    k = np.minimum(np.minimum(c, m), y)
    inks = {'C': np.clip(c - k, 0.0, 1.0), 'M': np.clip(m - k, 0.0, 1.0),
            'Y': np.clip(y - k, 0.0, 1.0), 'K': k}
    # broadcast vectors, not grids: the rotated coordinates are the
    # first full-frame arrays that need to exist
    xx = np.arange(w, dtype=np.float32)[None, :]
    yy = np.arange(h, dtype=np.float32)[:, None]
    cov = {}
    for name, ang in _SCREEN_ANGLES.items():
        th = np.radians(ang)
        ca, sa = np.float32(np.cos(th)), np.float32(np.sin(th))
        # into the screen's frame, to the nearest cell, and back to the
        # cell centre's frame position
        u = xx * ca + yy * sa
        v = -xx * sa + yy * ca
        iu = np.round(u / pitch).astype(np.int32)
        iv = np.round(v / pitch).astype(np.int32)
        cu = iu.astype(np.float32) * pitch
        cv = iv.astype(np.float32) * pitch
        cxp = cu * ca - cv * sa
        cyp = cu * sa + cv * ca
        d = np.sqrt((xx - cxp) ** 2 + (yy - cyp) ** 2)
        # the ink's coverage at each CELL centre, sampled once per cell
        # on the lattice, then gathered per pixel
        u0, u1 = int(iu.min()), int(iu.max())
        v0, v1 = int(iv.min()), int(iv.max())
        gu, gv = np.mgrid[u0:u1 + 1, v0:v1 + 1].astype(np.float32) * pitch
        gx = gu * ca - gv * sa
        gy = gu * sa + gv * ca
        ink = inks[name][..., None]
        lattice = _resample(ink, gx, gy)[..., 0]
        val = np.clip(lattice[iu - u0, iv - v0], 0.0, 1.0)
        # the exact-tone screen: below 50 percent a dark dot whose area
        # is the coverage (r = p sqrt(v/pi)); above it the classic
        # inversion -- a white dot of the remaining area on the
        # corner lattice, so the print reaches solid ink at 100
        # percent and checkerboards at 50
        r_dark = pitch * np.sqrt(val / np.float32(np.pi))
        cov_dark = np.clip(r_dark - d + np.float32(0.5), 0.0, 1.0)
        # the nearest corner lattice point, by the same rounding on the
        # half-shifted screen coordinates
        ju = np.round(u / pitch - np.float32(0.5)).astype(np.int32)
        jv = np.round(v / pitch - np.float32(0.5)).astype(np.int32)
        cu2 = (ju.astype(np.float32) + np.float32(0.5)) * pitch
        cv2 = (jv.astype(np.float32) + np.float32(0.5)) * pitch
        cx2 = cu2 * ca - cv2 * sa
        cy2 = cu2 * sa + cv2 * ca
        d2 = np.sqrt((xx - cx2) ** 2 + (yy - cy2) ** 2)
        gu2, gv2 = (np.mgrid[u0 - 1:u1 + 2, v0 - 1:v1 + 2].astype(np.float32)
                    + np.float32(0.5)) * pitch
        gx2 = gu2 * ca - gv2 * sa
        gy2 = gu2 * sa + gv2 * ca
        lattice2 = _resample(ink, gx2, gy2)[..., 0]
        val2 = np.clip(lattice2[ju - (u0 - 1), jv - (v0 - 1)], 0.0, 1.0)
        r_light = pitch * np.sqrt((np.float32(1.0) - val2)
                                  / np.float32(np.pi))
        cov_light = np.float32(1.0) - np.clip(r_light - d2 + np.float32(0.5),
                                              0.0, 1.0)
        cov[name] = np.where(val2 > 0.5, cov_light, cov_dark) \
            .astype(np.float32)
    out = np.empty_like(x)
    dark = 1.0 - cov['K']
    out[..., 0] = (1.0 - cov['C']) * dark
    out[..., 1] = (1.0 - cov['M']) * dark
    out[..., 2] = (1.0 - cov['Y']) * dark
    if amt < 1.0:
        out = x + (out - x) * np.float32(amt)
    return out.astype(np.float32)


# ------------------------------------------------------------ the process
#
# R236: the colour PROCESS, not a grade. A 1940s Technicolor cartoon was
# photographed as three black-and-white records (the successive-exposure
# camera shot each drawing three times through red, green and blue
# filters; the three-strip camera split the light instead -- the same
# three records, printed the same way) and PRINTED by dye transfer: from
# each record a gelatin relief matrix soaked in one dye -- cyan from the
# red record, magenta from the green, yellow from the blue -- and rolled
# in register onto one strip. Everything the field calls "the Technicolor
# look" lives in that chain, and this stage walks it in order:
#
#   1. the records: the filters' spectral cross-talk (M_CAM), halation in
#      the negative (light scattering in the base blooms the bright
#      areas of each record, in that record's colour);
#   2. the print: each record's exposure through the negative-and-matrix
#      curve to a DYE AMOUNT -- the Hurter-Driffield straight line in
#      log exposure, timed so a white cel prints clear, with the toe
#      where the dye runs out at D_max and the matrix's knee at clear;
#      all dye where there was no light -- then the dyes' UNWANTED
#      ABSORPTIONS (a cyan dye also eats some
#      green and blue; the magenta some red and blue; the yellow some
#      green): the transmittance is 10^-(D_max * sum_c a_c B_c A_cj),
#      with B the printer's balance that keeps a grey scale neutral, so
#      the impurities show only on colour -- reds deep, greens darker
#      and leaning cyan, blues toward purple. A silver KEY image printed
#      from the green record under the dyes (the later three-strip
#      prints) adds density in the shadows;
#   3. registration: each dye layer lands a hair off -- the honest place
#      for misregistration in this pipeline; the ink line itself gets a
#      coloured edge.
#
# The two-colour processes (Cinecolor, the 1930s two-strip) are the same
# chain with two records (red-orange and blue-green) and two dyes; a grey
# scale can only be balanced in the least-squares sense, which is why
# those prints' neutrals lean.

FILM_PROCESS_ITEMS = (
    ('NONE', "None", "No colour process: the render's own colour"),
    ('THREE_STRIP', "Three-strip / successive exposure",
     "Three black-and-white records through red, green and blue filters, "
     "printed by dye transfer in cyan, magenta and yellow with the dyes' "
     "own impurities and a silver key: the 1935-55 Technicolor cartoon "
     "and feature"),
    ('TWO_COLOUR', "Two-colour (Cinecolor, two-strip)",
     "Two records, red-orange and blue-green, printed in two dyes: no true "
     "green, violet or pure blue -- skies cyan, foliage olive, skin salmon "
     "-- the 1930s two-strip Technicolor and the 1940s Cinecolor short"),
)

#: the camera records: rows = records, columns = scene R, G, B (linear).
#: The filters are not clean -- a red record sees a little green, the
#: green record a little of both -- and every row sums to one, so a
#: white cel exposes every record alike
_M_CAM = {
    'THREE_STRIP': ((0.92, 0.08, 0.00), (0.06, 0.88, 0.06), (0.00, 0.10, 0.90)),
    'TWO_COLOUR': ((0.78, 0.22, 0.00), (0.00, 0.40, 0.60)),
}

#: the dyes: rows = the dye each record prints (complementary to the
#: record's colour), columns = the dye's absorption of display R, G, B
#: relative to its own colour (1). Off the diagonal are the UNWANTED
#: absorptions -- the imbibition dyes' documented impurity, scaled by
#: Dye Purity
_DYES = {
    'THREE_STRIP': ((1.00, 0.22, 0.10), (0.14, 1.00, 0.30), (0.02, 0.10, 1.00)),
    'TWO_COLOUR': ((1.00, 0.50, 0.15), (0.05, 0.60, 1.00)),
}

#: the ideal dyes each process would want (Dye Purity 1): three that
#: each absorb exactly their own third, or a blue-green and a red-orange
#: that split the green between them -- the two-colour's green
#: absorptions are the process, not an impurity
_DYES_IDEAL = {
    'THREE_STRIP': ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    'TWO_COLOUR': ((1.0, 0.5, 0.0), (0.0, 0.5, 1.0)),
}

#: which record carries the key image (the green record in three-strip;
#: the blue-green record in two-colour)
_KEY_RECORD = {'THREE_STRIP': 1, 'TWO_COLOUR': 1}

#: the matrix's toe at clear, as a density: the soft knee where an
#: over-white exposure runs the dye out (the matrix relief cannot go
#: below nothing) rather than the straight line going negative
_CLEAR_KNEE = np.float32(0.04)


def print_curve(e, gamma, dmax):
    """The timed print's characteristic. In density,
    D = gamma * log10((1 + 1/E0) / (1 + E/E0)) with E0 = 1 / (10^(D_max/gamma) - 1):
    the Hurter-Driffield straight line of slope Print Gamma in log
    exposure, pinned by the timer's light so a white cel (E = 1) prints
    clear and no light prints D_max -- the toe is where the dye runs
    out, and it is the same H&D toe whatever the gamma. Over-white
    exposures roll into the matrix's toe at clear (a soft knee of
    `_CLEAR_KNEE`) instead of negative dye. Returns (T, a): the
    transmittance 10^-D and the dye amount D / D_max in [0, 1] that
    prints it."""
    e = np.maximum(np.asarray(e, np.float32), np.float32(0.0))
    g = np.float32(gamma)
    dm = np.float32(dmax)
    e0 = np.float32(1.0 / (10.0 ** (float(dmax) / float(gamma)) - 1.0))
    d = dm - g * np.log10(np.float32(1.0) + e / e0)
    k = _CLEAR_KNEE
    d = (k * np.logaddexp(np.float32(0.0), d / k)).astype(np.float32)
    d = np.minimum(d, dm)
    t = np.exp(d * np.float32(-np.log(10.0))).astype(np.float32)
    a = (d / dm).astype(np.float32)
    return t, a


def _dye_matrix(process, purity):
    """The dyes' absorption rows, the documented dyes pulled toward the
    process's ideal ones by `purity`, and the printer's balance B (one
    factor per dye) that prints a grey scale neutral: solves A^T B = 1
    (least squares for two dyes). Returns (A (n_dyes, 3), B (n_dyes,))."""
    A = np.asarray(_DYES[process], np.float64)
    own = np.asarray(_DYES_IDEAL[process], np.float64)
    off = A - own
    A = own + off * (1.0 - float(np.clip(purity, 0.0, 1.0)))
    B, _res, _rk, _sv = np.linalg.lstsq(A.T, np.ones(3), rcond=None)
    return A.astype(np.float32), np.maximum(B, 0.0).astype(np.float32)


def process_on(st):
    return (str(getattr(st, 'film_process', 'NONE') or 'NONE').upper()
            in _M_CAM
            and float(getattr(st, 'film_process_amount', 1.0)) > 0.0)


def _downblur(f, sigma):
    """A wide Gaussian as a quarter-size blur: box-downsample by four,
    blur at sigma / 4, bilinear back. The halation is a veil; a
    quarter-size veil is the veil."""
    h, w = f.shape[:2]
    h4 = max(h // 4, 1)
    w4 = max(w // 4, 1)
    q = f[:h4 * 4, :w4 * 4].reshape(h4, 4, w4, 4, -1).mean(axis=(1, 3))
    q = gaussian(q, max(float(sigma) / 4.0, 0.5))
    # bilinear back up, edge padded: the frame is the same veil at its
    # edges as one pixel in
    ys = (np.arange(h, dtype=np.float32) + 0.5) / 4.0 - 0.5
    xs = (np.arange(w, dtype=np.float32) + 0.5) / 4.0 - 0.5
    y0 = np.clip(np.floor(ys).astype(np.int64), 0, h4 - 1)
    x0 = np.clip(np.floor(xs).astype(np.int64), 0, w4 - 1)
    y1 = np.clip(y0 + 1, 0, h4 - 1)
    x1 = np.clip(x0 + 1, 0, w4 - 1)
    ty = np.clip(ys - y0, 0.0, 1.0).astype(np.float32)[:, None, None]
    tx = np.clip(xs - x0, 0.0, 1.0).astype(np.float32)[None, :, None]
    top = q[y0][:, x0] * (1.0 - tx) + q[y0][:, x1] * tx
    bot = q[y1][:, x0] * (1.0 - tx) + q[y1][:, x1] * tx
    return (top * (1.0 - ty) + bot * ty).astype(np.float32)


def register_offsets(frame, seed, amplitude, n):
    """Each dye layer's registration error for `frame`, in pixels:
    n (dx, dy) pairs in [-amplitude, amplitude], independent per frame,
    per dye. The first layer is the reference (no shift) -- what is
    seen is the OTHER layers' offset from it."""
    a = float(max(amplitude, 0.0))
    out = [(0.0, 0.0)]
    for i in range(1, int(n)):
        dx = a * (2.0 * _frame_noise(frame, seed, 61 + i * 2) - 1.0)
        dy = a * (2.0 * _frame_noise(frame, seed, 62 + i * 2) - 1.0)
        out.append((float(dx), float(dy)))
    return out


def colour_process(rgb, st, frame=0, seed=0, fps=24.0):
    """The colour process on the linear frame: records, halation, the
    curve to dye, the negative's dust and grain on the dye, registration,
    the emulsion's scratches, the key, the dyes' transmittance, the
    base's scratches and the cue marks on the print. Amount mixes the
    printed frame over the original."""
    from . import wear as WEAR
    key = str(getattr(st, 'film_process', 'NONE') or 'NONE').upper()
    amt = float(np.clip(getattr(st, 'film_process_amount', 1.0), 0.0, 1.0))
    if key not in _M_CAM or amt <= 0.0:
        return rgb
    x = np.asarray(rgb, np.float32)
    M = np.asarray(_M_CAM[key], np.float32)
    n = M.shape[0]
    gamma = float(np.clip(getattr(st, 'film_gamma', 1.5), 0.5, 4.0))
    dmax = float(np.clip(getattr(st, 'film_density', 2.4), 0.5, 4.0))
    purity = float(np.clip(getattr(st, 'film_dye_purity', 0.0), 0.0, 1.0))
    key_amt = float(np.clip(getattr(st, 'film_key', 0.0), 0.0, 1.0))
    exposure = float(np.clip(getattr(st, 'film_exposure', 0.0), -4.0, 4.0))
    hal = float(np.clip(getattr(st, 'film_halation', 0.0), 0.0, 1.0))
    hal_r = float(max(getattr(st, 'film_halation_radius', 12.0), 1.0))
    reg = float(max(getattr(st, 'film_register', 0.0), 0.0))
    A, B = _dye_matrix(key, purity)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    gain = np.float32(2.0 ** exposure)
    sharp = float(np.clip(getattr(st, 'film_filters', 0.0), -0.3, 0.6))
    if sharp != 0.0:
        # the taking filters' sharpness: sharp-cutting filters separate
        # the records further than the eye separates the colours (the
        # analysis matrix (1 + 2s) I - s J, every row summing to one, so
        # a white cel still exposes every record alike); an overlap
        # (s < 0) muddies them. A record cannot see less than no light
        sm = np.float32(sharp)
        tot = r + g + b
        r, g, b = (np.maximum(r * (1.0 + 3.0 * sm) - sm * tot, 0.0).astype(np.float32),
                   np.maximum(g * (1.0 + 3.0 * sm) - sm * tot, 0.0).astype(np.float32),
                   np.maximum(b * (1.0 + 3.0 * sm) - sm * tot, 0.0).astype(np.float32))
    # 1. the records, exposed
    recs = []
    for c in range(n):
        e = (r * M[c, 0] + g * M[c, 1] + b * M[c, 2]) * gain
        recs.append(np.maximum(e, 0.0).astype(np.float32))
    if hal > 0.0:
        # halation: each record's own light scattered through the base
        veil = _downblur(np.stack(recs, -1), hal_r)
        recs = [(recs[c] + np.float32(hal) * veil[..., c]).astype(np.float32)
                for c in range(n)]
    # 2. the curve to a dye amount: all dye where there was no light,
    # none on a white cel at the timer's light, the H&D toe between
    dyes = [print_curve(recs[c], gamma, dmax)[1] for c in range(n)]
    # 2b. the negative's dust blocked the printer light through one
    # record (no dye there); the negative's grain printed through as
    # density noise on each record's dye (R237)
    dyes = WEAR.negative_dust_dyes(dyes, st, frame, seed)
    dyes = WEAR.grain_dyes(dyes, st, frame, seed, dmax)
    # 3. registration: every dye layer but the first lands a hair off
    if reg > 0.0:
        offs = register_offsets(frame, seed, reg, n)
        dyes = [shift(d[..., None], dx, dy)[..., 0] if (dx or dy) else d
                for d, (dx, dy) in zip(dyes, offs)]
    # 3b. the emulsion side's scratches take the dye away (R237)
    dyes = WEAR.scratch_dyes(dyes, st, frame, seed, key)
    # 4. the print's transmittance: the balanced dyes' densities per
    # channel, the key's silver on top
    dens = np.zeros(x.shape, np.float32)
    for c in range(n):
        w = dyes[c] * np.float32(B[c])
        for j in range(3):
            dens[..., j] += w * np.float32(A[c, j])
    dens *= np.float32(dmax)
    if key_amt > 0.0:
        k = dyes[_KEY_RECORD[key]]
        dens += (k * np.float32(key_amt * dmax * 0.35))[..., None]
    out = np.exp(dens * np.float32(-np.log(10.0))).astype(np.float32)
    # 4b. the base side's scratches scatter the lamp; the projectionist's
    # cue marks are scraped into the print (R237)
    out = WEAR.scratch_base(out, st, frame, seed)
    out = WEAR.cue_marks(out, st, frame, fps)
    # 5. the projector: the screen's white is the clear gate -- a white
    # cel on a normally timed print (the knee's hair of dye and all) --
    # so a print printed down IS darker on the screen, as it is
    t_white, _a = print_curve(np.array([1.0], np.float32), gamma, dmax)
    out = (out / np.float32(max(float(t_white[0]), 1e-3))).astype(np.float32)
    if amt < 1.0:
        out = x + (out - x) * np.float32(amt)
    return out.astype(np.float32)


def film_on(st):
    """Whether any linear-light film stage is active (the post chain's
    fast exit)."""
    from . import wear as WEAR
    return (process_on(st)
            or str(getattr(st, 'film_grade', 'NONE') or 'NONE').upper()
            != 'NONE'
            or float(getattr(st, 'film_softness', 0.0)) > 1e-3
            or float(getattr(st, 'film_weave', 0.0)) > 0.0
            or float(getattr(st, 'film_flicker', 0.0)) > 0.0
            or WEAR.wear_on(st))


def process_linear(rgb, st, frame=0, seed=0, key_frame=None, fps=24.0):
    """The film stages on the LINEAR frame, in the order the light met
    them (R237 put each where it happened): dust on the cel under the
    camera, the camera's optics, the colour process (with the
    negative's dust and grain, the emulsion's scratches and the cue
    marks inside it), the stock's grade, the same wear on a print that
    went through no process, dirt on the print, the gate's weave, the
    hairs in the projector gate, the lamp's flicker. `key_frame` is the
    hold's key frame (the cel the frame was photographed from)."""
    if not film_on(st):
        return rgb
    from . import wear as WEAR
    kf = int(frame) if key_frame is None else int(key_frame)
    out = WEAR.cel_dust(rgb, st, kf, seed)
    out = soften(out, st)
    on = process_on(st)
    if on:
        out = colour_process(out, st, frame, seed, fps)
    out = grade(out, st)
    if not on:
        out = WEAR.plain_print(out, st, frame, seed, fps)
    out = dust(out, st, frame, seed)
    out = weave(out, st, frame, seed)
    out = WEAR.hairs(out, st, frame, seed)
    out = flicker(out, st, frame, seed)
    return out


# ------------------------------------------------------------ the paint


def misregister_offset(frame, seed, amplitude):
    """The cel's placement error for `frame`, in pixels: each cel is
    laid on the pegs a hair off, independently per frame."""
    a = float(max(amplitude, 0.0))
    if a <= 0.0:
        return 0.0, 0.0
    dx = a * (2.0 * _frame_noise(frame, seed, 41) - 1.0)
    dy = a * (2.0 * _frame_noise(frame, seed, 42) - 1.0)
    return float(dx), float(dy)


def paint_reach(st, ss=1.0):
    """How many internal rows the paint stages read past a band."""
    mis = float(max(getattr(st, 'film_misregister', 0.0), 0.0))
    bleed = float(max(getattr(st, 'film_bleed', 0.0), 0.0))
    if mis <= 0.0 and bleed <= 1e-3:
        return 0
    return int(np.ceil((mis + 3.0 * bleed) * float(ss) + 1.0))


def misregister_and_bleed(img, st, frame=0, seed=0, ss=1.0):
    """The paint before the ink: Film Bleed softens the painted colour
    (the paint soaking into the cel, sigma in output pixels), and Film
    Misregister slides the whole paint layer by this frame's placement
    error (amplitude in output pixels) -- then the lines are drawn where
    the drawing put them, crisp, over paint that missed. Alpha stays."""
    mis = float(max(getattr(st, 'film_misregister', 0.0), 0.0))
    bleed = float(max(getattr(st, 'film_bleed', 0.0), 0.0))
    if mis <= 0.0 and bleed <= 1e-3:
        return img
    rgb = np.asarray(img[..., :3], np.float32)
    if bleed > 1e-3:
        rgb = gaussian(rgb, bleed * float(ss))
    if mis > 0.0:
        dx, dy = misregister_offset(frame, seed, mis)
        rgb = shift(rgb, dx * float(ss), dy * float(ss))
    out = np.array(img, np.float32, copy=True)
    out[..., :3] = rgb
    return out


# -------------------------------------------------------------- the hold


def hold_key(frame, start, hold):
    """The key frame `frame` is photographed from when shooting on
    `hold`s (2 = twos, 3 = threes): the last frame at or before it that
    sits on the hold grid counted from `start`."""
    hold = int(max(int(hold), 1))
    if hold <= 1:
        return int(frame)
    f = int(frame)
    s = int(start)
    return s + ((f - s) // hold) * hold
