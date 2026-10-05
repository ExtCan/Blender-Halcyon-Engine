"""R251 C001: the N64 Video Interface's coverage half -- the RDP's 3-bit
coverage blended at scan-out (the VI's "AA" filter) and the divot median.

The 'N64 blur' is not the texture filter: the RDP reduces a 4x4 subpixel
mask to 3 bits of coverage per pixel (core/raster.py `coverage16`), and
at scan-out the VI pulls every PARTIALLY covered pixel toward the
penultimate extremes of its six fully covered neighbours by
(7 - cvg) / 8, then the divot filter medians any horizontal triple that
holds a partial pixel -- softening every polygon edge, interior seams
included, and only there. Evidence: N64 Programming Manual ch. 15.2
('the sixteen subpixels must be dithered to eight'); angrylion-rdp-plus
vi/video.c `video_filter16` (the CENTRE first in the list, backr[0] = r
and numoffull = 1, then the six neighbours whose coverage is full;
`video_max_optimized`'s penultimate max / min of that list -- the centre
itself when the centre is the extreme; coeff = 7 - centercvg,
(((penmin + penmax - 2c) * coeff) + 4) >> 3),
vi/divot.c (median when (center.a & left.a & right.a) != 7).

Pure NumPy, int32 arithmetic on the display-referred frame's RGBA5551
expansion (the VI fetched a 16-bit framebuffer: 5 bits per channel
replicated to 8, `quant5`); the GLSL twins in gpu/stages_vi.py are the
same integer statements and are held bitwise in the simulator. Row 0 is
the bottom of the picture everywhere; the six-neighbour set is symmetric
so orientation never matters. Nothing here reads bpy.

The VI's 2x bilinear scale of the 320x240 buffer is NOT modelled
(Halcyon's Pixel Scale is nearest-neighbour) and the CLAMP coverage
accumulate is not either -- both disclosed in the CHANGELOG.
"""

import numpy as np

#: the six VI neighbours: up-left, up-right, left, right, down-left,
#: down-right (dx, dy) -- the sixteen-bit filter's own set
NEIGHBOURS = ((-1, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (1, 1))


def _nb(a, dx, dy, fill):
    """`a` shifted so that out[y, x] = a[y + dy, x + dx], `fill` off-frame."""
    out = np.full_like(a, fill)
    h, w = a.shape[0], a.shape[1]
    ys0, ys1 = max(0, -dy), min(h, h - dy)
    xs0, xs1 = max(0, -dx), min(w, w - dx)
    if ys1 > ys0 and xs1 > xs0:
        out[ys0:ys1, xs0:xs1] = a[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx]
    return out


def quant5(rgb):
    """The RGBA5551 framebuffer as the VI fetches it: 5 bits per channel,
    expanded to 8 by bit replication -> int32 (H, W, 3) in 0..255. The
    round is `floor(c * 31 + 0.5)` in two float32 ops, the GLSL twin's
    own (`v = v * 31.0; v = v + 0.5; floor(v)`)."""
    f32 = np.float32
    t = np.clip(np.asarray(rgb, np.float32), f32(0.0), f32(1.0))
    t = t * f32(31.0)
    t = t + f32(0.5)
    c5 = np.floor(np.clip(t, f32(0.0), f32(31.0))).astype(np.int32)
    return (c5 << 3) | (c5 >> 2)


def to8(rgb):
    """An 8-bit frame decoded back from `k / 255` texels (the divot's input
    after the AA pass): floor(c * 255 + 0.5) in two float32 ops, exact for
    every 8-bit k."""
    f32 = np.float32
    t = np.clip(np.asarray(rgb, np.float32), f32(0.0), f32(1.0))
    t = t * f32(255.0)
    t = t + f32(0.5)
    return np.floor(t).astype(np.int32)


def vi_aa(c8, cvg3):
    """angrylion's video_filter16 over an 8-bit frame: every pixel with
    coverage < 7 moves by (7 - cvg) / 8 toward the penultimate max / min
    of the list {the centre, its fully covered neighbours}.
    `video_max_optimized` keeps the centre (index 0) as the penultimate
    whenever the centre is the extreme, else the list's second largest /
    smallest -- which is `max(c, second largest NEIGHBOUR)` /
    `min(c, second smallest NEIGHBOUR)` per channel, the form written
    here. So penmin <= c <= penmax always: a pull toward the
    neighbours, never past them, and no neighbour leaves the pixel where
    it is. int32 in, int32 out, `& 0xff` (angrylion's mask, kept; it
    never bites since the result stays inside [penmin, penmax])."""
    c8 = np.asarray(c8, np.int32)
    cvg3 = np.asarray(cvg3, np.int32)
    vals_max = []
    vals_min = []
    for dx, dy in NEIGHBOURS:
        v = _nb(c8, dx, dy, 0)
        ok = _nb(cvg3, dx, dy, -1) == 7          # off-frame: not a candidate
        vals_max.append(np.where(ok[..., None], v, -1))
        vals_min.append(np.where(ok[..., None], v, 1000))
    smax = np.sort(np.stack(vals_max), axis=0)[::-1]
    smin = np.sort(np.stack(vals_min), axis=0)
    penmax = np.maximum(c8, smax[1])             # sentinel -1: fewer than two
    penmin = np.minimum(c8, smin[1])             # sentinel 1000 likewise
    coeff = (7 - cvg3)[..., None]
    d = ((penmin + penmax) - 2 * c8) * coeff
    d = d + 4
    d = d >> 3                                   # arithmetic shift = floor
    out = (c8 + d) & 0xff
    return np.where((cvg3 < 7)[..., None], out, c8).astype(np.int32)


def vi_divot(c8, cvg3):
    """angrylion's divot filter over the AA pass's output: any pixel whose
    horizontal triple holds a partial-coverage pixel becomes the
    per-channel median of the three. At the frame's x-edges the missing
    neighbour IS the centre (its coverage 7), so the median is the
    identity there -- the rule the GLSL twin follows with explicit
    guards, because the simulator clamps an out-of-range texelFetch
    where a driver returns zero."""
    c8 = np.asarray(c8, np.int32)
    cvg3 = np.asarray(cvg3, np.int32)
    l = c8.copy()
    l[:, 1:] = c8[:, :-1]
    r = c8.copy()
    r[:, :-1] = c8[:, 1:]
    la = np.full_like(cvg3, 7)
    la[:, 1:] = cvg3[:, :-1]
    ra = np.full_like(cvg3, 7)
    ra[:, :-1] = cvg3[:, 1:]
    trip = (la & cvg3 & ra) != 7
    med = np.maximum(np.minimum(l, c8), np.minimum(np.maximum(l, c8), r))
    return np.where(trip[..., None], med, c8).astype(np.int32)


def n64_vi(rgb, st, *_a, **_k):
    """The post chain's CPU function: the AA pass on the 5551 expansion of
    the display-referred frame, then the divot when `n64_divot`; the
    result as float32 `k / 255`. Reads the coverage plane `process`
    parked on the settings (`st._n64_cvg`)."""
    cvg3 = getattr(st, '_n64_cvg', None)
    if cvg3 is None:
        return np.asarray(rgb, np.float32)
    c8 = vi_aa(quant5(rgb), cvg3)
    if bool(getattr(st, 'n64_divot', True)):
        c8 = vi_divot(c8, cvg3)
    return (c8.astype(np.float32) / np.float32(255.0)).astype(np.float32)


def vi_on(st):
    """Whether the VI stage runs on this frame: the flag is on AND the
    render left a coverage plane of the frame's size."""
    return bool(getattr(st, 'n64_coverage_aa', False)) and \
        getattr(st, '_n64_cvg', None) is not None


def plane_for_frame(cvg, shape_hw, st, warn=None):
    """The coverage plane the chain reads for this frame, or None: the
    flag off -> None (nothing to say); the plane absent (a supersampled
    frame, a held frame without one) -> None, named once by `warn`; a
    shape that is not the frame's (resized upstream) -> None, named."""
    if not bool(getattr(st, 'n64_coverage_aa', False)):
        return None
    if cvg is None:
        if warn is not None:
            warn('N64 VI filter skipped: no coverage plane for this frame '
                 '(1 sample per pixel needed)')
        return None
    cvg = np.asarray(cvg)
    if tuple(cvg.shape[:2]) != tuple(shape_hw):
        if warn is not None:
            warn("N64 VI filter skipped: the coverage plane is not the "
                 "frame's size")
        return None
    return np.ascontiguousarray(cvg.astype(np.int32))


def coverage_plane(gbuf, opts, ss):
    """The render's hand-off to the post chain (`scene.last_cvg`): the
    G-buffer's coverage plane when the frame asked for it at 1 sample per
    pixel, else None."""
    if gbuf is None or getattr(gbuf, 'cvg', None) is None:
        return None
    if opts is None or not bool(getattr(opts, 'cvg', False)) or int(ss) != 1:
        return None
    return gbuf.cvg


def cvg_image(cvg3):
    """The coverage plane as an (H, W, 4) float32 texture (.x = cvg3), the
    frame's own row order, for the GPU stages."""
    cvg3 = np.asarray(cvg3)
    img = np.zeros((cvg3.shape[0], cvg3.shape[1], 4), np.float32)
    img[:, :, 0] = cvg3.astype(np.float32)
    return img
