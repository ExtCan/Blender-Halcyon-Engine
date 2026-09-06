"""The print's wear (R237): grain, dust, hairs, scratches, cue marks.

Everything that reached the screen and was never part of the drawing,
each where it physically happened, each a pure function of the frame
number, the hold's key frame, the seed and the dials -- the same bits on
either device, in any band, on any worker:

- CEL DUST sat on the cel or the platen glass under the rostrum camera:
  dark grey under the lights, and photographed again on every frame of
  a hold (the frames that share a cel share its specks), so it is placed
  by the KEY frame, not the frame;
- NEGATIVE DUST blocked the printer light through one record, so the
  print has no dye there: a clear speck -- and on a three-strip print a
  coloured one, the one record's dye missing;
- the GRAIN is the negative's silver printed through: density noise on
  each record's dye, the emulsion's grains a random sheet of a size, in
  clumps if the stock was coarse; each record its own sheet (coloured
  grain) or one sheet for all (one black-and-white negative). The noise
  is strongest where half the grains developed and vanishes at the
  clear gate and at D-max: the print's whites and blacks are clean;
- SCRATCHES are the transport's: vertical, the length of a shot, at one
  place with a slow wander. On the emulsion side they take dye away --
  a dye-transfer print's one gelatin layer loses all its dyes together
  (a bright neutral line), a duplitized two-colour print, dyed on both
  sides, loses one side's dye (a coloured line), a chromogenic print
  loses its yellow first and its magenta next (the blue scratch); on
  the base side they scatter the lamp (a dark line);
- CUE MARKS are the projectionist's: a scraped circle top right, four
  frames long, twice near the end of every reel -- the motor cue eight
  seconds out, the changeover cue one second out;
- PRINT DIRT lies on the print and weaves with it; HAIRS sit in the
  projector gate, anchored at the aperture's edge, dancing for a run of
  frames, and do not weave.

Sizes are given at 1080 lines and scale with the frame's height, the
way the ink's widths do.
"""

import numpy as np

from .ink import _hash_u32, value_noise
from .film import _hash_u32_raw, _gauss_kernel, gaussian

#: the density RMS of the grain at Grain 1 where half the grains
#: developed, for grains a pixel wide or wider
_GRAIN_K = np.float32(0.4)

#: how many hairs the gate can hold at once, and how many scratches the
#: transport can run at once
_HAIR_SLOTS = 4
_SCRATCH_SLOTS = 3

FILM_SCRATCH_SIDE_ITEMS = (
    ('EMULSION', "Emulsion side",
     "The scratch took the dye away: a bright line, neutral on a dye-"
     "transfer print, one dye's colour on a two-colour print, yellow "
     "first (the blue scratch) on a chromogenic one"),
    ('BASE', "Base side",
     "The scratch is in the film's base and scatters the projector's "
     "lamp: a soft dark line"),
)


def scale_of(h):
    """The frame's size factor: sizes are given at 1080 lines."""
    return float(h) / 1080.0


def _hv(clock, seed, k, j=0):
    """One hashed value in [0, 1) per (clock, seed, k, j)."""
    return float(_hash_u32(int(clock), (int(j) * 4099 + (int(seed) & 0xffff))
                           & 0x7fffffff, int(k))[0])


# ------------------------------------------------------------- the grain


def _sheet_key(frame, seed, k):
    return (int(frame) * 1000003 + (int(seed) & 0xffff) * 7919
            + int(k) * 104729) & 0x7fffffff


def _white_sheets(h, w, frame, seed, k, step=1):
    """Four unit-variance white sheets from two hashes per pixel: one
    triangular (two 12-bit uniforms) and three uniform (8 bits each).
    `step` > 1 makes them on a coarser grid (every step-th pixel)."""
    xx, yy = np.meshgrid(np.arange(w, dtype=np.uint32) * np.uint32(step),
                         np.arange(h, dtype=np.uint32) * np.uint32(step))
    ha = _hash_u32_raw(xx, yy, _sheet_key(frame, seed, k))
    hb = _hash_u32_raw(xx, yy, _sheet_key(frame, seed, k + 1))
    # the triangular sheet: two 12-bit uniforms summed, scaled to unit
    # variance in one pass (variance 1/6 before the scale)
    tri = (ha & np.uint32(0xfff)).astype(np.float32)
    tri += ((ha >> np.uint32(12)) & np.uint32(0xfff)).astype(np.float32)
    tri *= np.float32(np.sqrt(6.0) / 4096.0)
    tri -= np.float32(np.sqrt(6.0))
    sheets = [tri]
    for c in range(3):
        u = ((hb >> np.uint32(8 * c)) & np.uint32(0xff)).astype(np.float32)
        u *= np.float32(np.sqrt(3.0) * 2.0 / 256.0)
        u -= np.float32(np.sqrt(3.0))
        sheets.append(u)
    return sheets


def _upsample(q, h, w, step):
    """Bilinear from a grid of every step-th pixel back to (h, w),
    pixel centres aligned, edges clamped."""
    hq, wq = q.shape[:2]
    ys = (np.arange(h, dtype=np.float32) + 0.5) / float(step) - 0.5
    xs = (np.arange(w, dtype=np.float32) + 0.5) / float(step) - 0.5
    y0 = np.clip(np.floor(ys).astype(np.int64), 0, hq - 1)
    x0 = np.clip(np.floor(xs).astype(np.int64), 0, wq - 1)
    y1 = np.clip(y0 + 1, 0, hq - 1)
    x1 = np.clip(x0 + 1, 0, wq - 1)
    ty = np.clip(ys - y0, 0.0, 1.0).astype(np.float32)[:, None, None]
    tx = np.clip(xs - x0, 0.0, 1.0).astype(np.float32)[None, :, None]
    top = q[y0][:, x0] * (1.0 - tx) + q[y0][:, x1] * tx
    bot = q[y1][:, x0] * (1.0 - tx) + q[y1][:, x1] * tx
    return (top * (1.0 - ty) + bot * ty).astype(np.float32)


def _unit_sheets(h, w, frame, seed, k, size_px):
    """Four unit-variance sheets of grains `size_px` wide (FWHM): white
    sheets blurred to the size and renormalised analytically -- the
    same factor whatever the frame's size. Wide grains are made on a
    coarser grid (a sigma of at least a grid pixel is left in the
    blur, so the bilinear step back adds nothing the eye can find) and
    cost the same as fine ones."""
    if size_px <= 1.2:
        return _white_sheets(h, w, frame, seed, k)
    sig = float(size_px) / 2.355
    step = 1 if sig <= 2.0 else (2 if sig <= 4.0 else 4)
    hq = (h + step - 1) // step
    wq = (w + step - 1) // step
    sheets = _white_sheets(hq, wq, frame, seed, k, step)
    kern = _gauss_kernel(sig / step)
    norm = np.float32(1.0 / float((kern * kern).sum()))
    out = gaussian(np.stack(sheets, -1), sig / step) * norm
    if step > 1:
        out = _upsample(out, h, w, step)
    return [np.ascontiguousarray(out[..., i]) for i in range(out.shape[-1])]


def grain_sheets(h, w, frame, seed, size_px, clump, chroma, n):
    """`n` unit-variance grain sheets for `frame`: the common sheet mixed
    with each record's own by Chroma (sqrt(1 - c) and sqrt(c), so the
    variance holds), grains `size_px` wide, a `clump` share of them in
    clumps three grains wide."""
    fine = _unit_sheets(h, w, frame, seed, 3, size_px)
    if clump > 1e-3:
        coarse = _unit_sheets(h, w, frame, seed, 7, 3.0 * size_px)
        c = float(np.clip(clump, 0.0, 1.0))
        nrm = np.float32(1.0 / np.sqrt((1.0 - c) ** 2 + c * c))
        fine = [((f * np.float32(1.0 - c) + g * np.float32(c)) * nrm)
                .astype(np.float32) for f, g in zip(fine, coarse)]
    ch = float(np.clip(chroma, 0.0, 1.0))
    if ch <= 0.0:
        return [fine[0]] * int(n)
    if ch >= 1.0:
        return [fine[1 + (r % 3)] for r in range(int(n))]
    a = np.float32(np.sqrt(1.0 - ch))
    b = np.float32(np.sqrt(ch))
    out = []
    for r in range(int(n)):
        m = fine[1 + (r % 3)] * b
        m += fine[0] * a
        out.append(m.astype(np.float32, copy=False))
    return out


def grain_sigma(a, amount, size_px):
    """The grain's density RMS on a dye amount `a` in [0, 1]:
    Grain * K * min(size, 1) * sqrt(a (1 - a)) -- one grain per pixel
    or more averages down, and the noise vanishes where no grain or
    every grain developed."""
    amp = float(amount) * float(_GRAIN_K) * min(float(size_px), 1.0)
    return (np.sqrt(np.clip(a * (1.0 - a), 0.0, 1.0))
            * np.float32(amp)).astype(np.float32)


def grain_on(st):
    return float(getattr(st, 'film_grain', 0.0)) > 0.0


def grain_params(st, h):
    g = float(np.clip(getattr(st, 'film_grain', 0.0), 0.0, 1.0))
    size_px = float(max(getattr(st, 'film_grain_size', 1.0), 0.1)) * scale_of(h)
    clump = float(np.clip(getattr(st, 'film_grain_clump', 0.0), 0.0, 1.0))
    chroma = float(np.clip(getattr(st, 'film_grain_chroma', 0.5), 0.0, 1.0))
    return g, size_px, clump, chroma


def grain_dyes(dyes, st, frame, seed, dmax):
    """The grain on the print's dye amounts, one sheet per record."""
    if not grain_on(st):
        return dyes
    h, w = dyes[0].shape
    g, size_px, clump, chroma = grain_params(st, h)
    sheets = grain_sheets(h, w, frame, seed, size_px, clump, chroma, len(dyes))
    out = []
    for a, sh in zip(dyes, sheets):
        sig = grain_sigma(a, g, size_px) * np.float32(1.0 / float(dmax))
        out.append(np.clip(a + sig * sh, 0.0, 1.0).astype(np.float32))
    return out


def grain_plain(rgb, st, frame=0, seed=0):
    """The grain on a frame with no process: the frame's own channels
    read as a print's transmittance (a density scale of 2.4), the
    density RMS of `grain_sigma` as a relative noise of ln(10) times it,
    mean-preserving, each channel its own sheet by Chroma."""
    if not grain_on(st):
        return rgb
    x = np.asarray(rgb, np.float32)
    h, w = x.shape[:2]
    g, size_px, clump, chroma = grain_params(st, h)
    sheets = grain_sheets(h, w, frame, seed, size_px, clump, chroma, 3)
    out = np.empty_like(x)
    ln10 = np.float32(np.log(10.0))
    for c in range(3):
        t = np.maximum(x[..., c], np.float32(1e-4))
        a = np.log10(t)
        a *= np.float32(-1.0 / 2.4)
        np.clip(a, 0.0, 1.0, out=a)
        sig = grain_sigma(a, g, size_px)
        sig *= sheets[c]
        sig *= ln10
        sig += np.float32(1.0)
        np.multiply(x[..., c], sig, out=out[..., c])
    return np.maximum(out, 0.0).astype(np.float32)


# -------------------------------------------------------------- the dust


def _blob(h, w, cx, cy, r, salt, irregular=1.0):
    """The coverage patch of a speck at (cx, cy): a disc of radius r
    whose edge wanders with the angle (three and five lobes hashed from
    `salt`). Returns (y0, y1, x0, x1, cov) or None off the frame."""
    rr = r * (1.0 + 0.55 * irregular)
    x0 = int(max(np.floor(cx - rr - 1), 0))
    x1 = int(min(np.ceil(cx + rr + 1), w - 1))
    y0 = int(max(np.floor(cy - rr - 1), 0))
    y1 = int(min(np.ceil(cy + rr + 1), h - 1))
    if x1 < x0 or y1 < y0:
        return None
    yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1].astype(np.float32)
    dx = xx - np.float32(cx)
    dy = yy - np.float32(cy)
    d = np.sqrt(dx * dx + dy * dy)
    # the ragged edge only reads on a speck a few pixels across; a
    # one-pixel speck is a dot
    irregular = float(irregular) * float(np.clip((r - 1.2) / 3.0, 0.0, 1.0))
    if irregular > 0.0:
        ph1 = _hash_u32(int(salt), 1, 77)[0] * 2.0 * np.pi
        ph2 = _hash_u32(int(salt), 2, 78)[0] * 2.0 * np.pi
        ang = np.arctan2(dy, dx)
        rad = np.float32(r) * (1.0 + np.float32(0.35 * irregular)
                               * np.sin(3.0 * ang + np.float32(ph1))
                               + np.float32(0.2 * irregular)
                               * np.sin(5.0 * ang + np.float32(ph2)))
    else:
        rad = np.float32(r)
    cov = np.clip(rad - d + 0.5, 0.0, 1.0).astype(np.float32)
    return y0, y1 + 1, x0, x1 + 1, cov


def speck_count(dens, clock, seed, salt, area):
    """How many specks a print of this area carries at density `dens`:
    about forty on a 1080p frame at 1.0 plus the frame's own scatter,
    and never none above zero -- a print is never clean."""
    if dens <= 0.0:
        return 0
    base = int(round(dens * 40.0 * area))
    extra = int(round(_hv(clock, seed, salt) * dens * 24.0 * area))
    return max(base + extra, 1)


def specks(dens, clock, seed, salt, h, w, size_px):
    """The specks of one population for `clock`: (cx, cy, r, salt_k)."""
    area = float(h * w) / (1920.0 * 1080.0)
    n = speck_count(dens, clock, seed, salt, area)
    sd = int(seed) & 0xffff
    out = []
    for k in range(n):
        u = float(_hash_u32(int(clock), k * 4 + 0, salt + 1 + sd)[0])
        v = float(_hash_u32(int(clock), k * 4 + 1, salt + 2 + sd)[0])
        sz = float(_hash_u32(int(clock), k * 4 + 2, salt + 3 + sd)[0])
        r = size_px * scale_of(h) * (0.35 + 1.3 * sz)
        out.append((u * (w - 1), v * (h - 1), max(r, 0.4),
                    (int(clock) * 131 + k * 7 + salt) & 0x7fffffff))
    return out


def dust_params(st, h):
    dens = float(np.clip(getattr(st, 'film_dust', 0.0), 0.0, 1.0))
    size = float(max(getattr(st, 'film_dust_size', 1.5), 0.1))
    neg = float(np.clip(getattr(st, 'film_dust_negative', 0.25), 0.0, 1.0))
    cel = float(np.clip(getattr(st, 'film_dust_cel', 0.0), 0.0, 1.0))
    return dens, size, neg, cel


def _paint(out, patch, colour, mix):
    y0, y1, x0, x1, cov = patch
    c = (cov * np.float32(mix))[..., None]
    out[y0:y1, x0:x1] = out[y0:y1, x0:x1] * (1.0 - c) \
        + np.asarray(colour, np.float32) * c


def cel_dust(rgb, st, key_frame=0, seed=0):
    """Dust on the cel and the platen glass: dark grey under the rostrum
    lights, placed by the hold's KEY frame so every frame photographed
    from the same cel carries the same specks."""
    dens, size, neg, cel = dust_params(st, 0)
    if dens <= 0.0 or cel <= 0.0:
        return rgb
    out = np.array(rgb, np.float32, copy=True)
    h, w = out.shape[:2]
    for cx, cy, r, salt in specks(dens * cel, key_frame, seed, 40, h, w, size):
        p = _blob(h, w, cx, cy, r, salt, 1.0)
        if p is not None:
            _paint(out, p, (0.12, 0.11, 0.10), 0.85)
    return out


def print_dirt(rgb, st, frame=0, seed=0):
    """Dirt on the print: opaque, nearly black, fresh every frame."""
    dens, size, neg, cel = dust_params(st, 0)
    share = dens * (1.0 - cel) * (1.0 - neg)
    if share <= 0.0:
        return rgb
    out = np.array(rgb, np.float32, copy=True)
    h, w = out.shape[:2]
    for cx, cy, r, salt in specks(share, frame, seed, 30, h, w, size):
        p = _blob(h, w, cx, cy, r, salt, 1.0)
        if p is not None:
            _paint(out, p, (0.02, 0.018, 0.015), 0.9)
    return out


def negative_specks(st, frame, seed, h, w, n_records):
    """Dust on the negative at printing: (patch, record) per speck --
    the record it blocked, hashed, so a three-strip print's speck is
    that record's colour missing."""
    dens, size, neg, cel = dust_params(st, 0)
    share = dens * (1.0 - cel) * neg
    if share <= 0.0:
        return []
    out = []
    for cx, cy, r, salt in specks(share, frame, seed, 50, h, w, size):
        p = _blob(h, w, cx, cy, r * 0.85, salt, 0.7)
        if p is not None:
            rec = int(_hash_u32(salt, 3, 79)[0] * n_records) % max(n_records, 1)
            out.append((p, rec))
    return out


def negative_dust_dyes(dyes, st, frame, seed):
    """The negative's dust on the dye amounts: no dye where it blocked
    the light, on that record alone."""
    h, w = dyes[0].shape
    sp = negative_specks(st, frame, seed, h, w, len(dyes))
    if not sp:
        return dyes
    out = [np.array(d, np.float32, copy=True) for d in dyes]
    for (y0, y1, x0, x1, cov), rec in sp:
        out[rec][y0:y1, x0:x1] *= (1.0 - cov)
    return out


def negative_dust_plain(rgb, st, frame=0, seed=0):
    """The negative's dust on a frame with no process: one black-and-
    white negative, so the speck is clear -- the transmittance raised
    to (1 - coverage), invisible where the print is white already."""
    x = np.asarray(rgb, np.float32)
    h, w = x.shape[:2]
    sp = negative_specks(st, frame, seed, h, w, 1)
    if not sp:
        return rgb
    out = np.array(x, copy=True)
    for (y0, y1, x0, x1, cov), _rec in sp:
        t = np.maximum(out[y0:y1, x0:x1], np.float32(1e-4))
        out[y0:y1, x0:x1] = np.power(t, (1.0 - cov)[..., None]).astype(np.float32)
    return out


# ------------------------------------------------------- the slot model


def _alive(frame, seed, amount, hold, slots, salt):
    """The slot model shared by hairs and scratches: `slots` slots, each
    running epochs of one hold (staggered between slots); slot k is in
    use with probability clip(slots * amount - k, 0, 1) per epoch, so
    Hairs 0.25 keeps one of four slots always busy and 1.0 keeps all
    four; a busy slot holds its thing for 0.6-1.4 holds (clipped to the
    epoch), starting somewhere in the slack. Returns
    [(slot, epoch, pos, life)] alive at `frame`, pos the frames since
    the thing arrived."""
    hold = max(int(hold), 1)
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 0.0:
        return []
    E = hold
    out = []
    f = int(frame)
    for s in range(int(slots)):
        p = float(np.clip(int(slots) * amount - s, 0.0, 1.0))
        if p <= 0.0:
            continue
        fs = f + (s * E) // int(slots)          # slots staggered
        e = fs // E
        clock = (e * 8 + s) & 0x7fffffff
        if p < 1.0 and _hv(clock, seed, salt) >= p:
            continue
        life = int(round(hold * (0.6 + 0.8 * _hv(clock, seed, salt + 1))))
        life = max(1, min(life, E))
        start = int(np.floor(_hv(clock, seed, salt + 2) * (E - life + 1)))
        pos = fs - e * E - start
        if 0 <= pos < life:
            out.append((s, e, pos, life))
    return out


def _polyline_patch(h, w, pts, width):
    """The soft coverage of a polyline `pts` ((N, 2) x, y) of `width`
    pixels: the distance to the nearest segment, on a bounding patch.
    Returns (y0, y1, x0, x1, cov) or None."""
    pts = np.asarray(pts, np.float32)
    hw = 0.5 * float(width)
    x0 = int(max(np.floor(pts[:, 0].min() - hw - 1), 0))
    x1 = int(min(np.ceil(pts[:, 0].max() + hw + 1), w - 1))
    y0 = int(max(np.floor(pts[:, 1].min() - hw - 1), 0))
    y1 = int(min(np.ceil(pts[:, 1].max() + hw + 1), h - 1))
    if x1 < x0 or y1 < y0:
        return None
    yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1].astype(np.float32)
    P = np.stack([xx.ravel(), yy.ravel()], -1)           # (M, 2)
    A = pts[:-1][None, :, :]                              # (1, N-1, 2)
    B = pts[1:][None, :, :]
    AB = B - A
    L2 = np.maximum((AB * AB).sum(-1), 1e-6)
    t = np.clip(((P[:, None, :] - A) * AB).sum(-1) / L2, 0.0, 1.0)
    C = A + AB * t[..., None]
    d = np.sqrt(((P[:, None, :] - C) ** 2).sum(-1)).min(1)
    cov = np.clip(hw - d + 0.5, 0.0, 1.0).astype(np.float32).reshape(yy.shape)
    return y0, y1 + 1, x0, x1 + 1, cov


# ------------------------------------------------------------- the hairs


def hair_params(st, h):
    amount = float(np.clip(getattr(st, 'film_hairs', 0.0), 0.0, 1.0))
    length = float(max(getattr(st, 'film_hair_length', 90.0), 2.0)) * scale_of(h)
    width = float(max(getattr(st, 'film_hair_width', 1.5), 0.3)) * scale_of(h)
    hold = int(max(getattr(st, 'film_hair_hold', 12), 1))
    return amount, length, width, hold


def hair_points(h, w, s, e, pos, seed, length, salt=60):
    """The hair in slot s, epoch e, `pos` frames after it arrived: a
    cubic curve anchored at a hashed point on a hashed edge of the
    gate, reaching inward, its two bends and its lean hashed once per
    hair, swaying frame by frame -- a slow sway from lattice noise and
    a flutter from the frame's hash."""
    clock = (e * 8 + s) & 0x7fffffff
    edge = int(_hv(clock, seed, salt) * 4.0) % 4
    u = _hv(clock, seed, salt + 1)
    lean = (_hv(clock, seed, salt + 2) - 0.5) * 1.75          # radians, +-50 deg
    bend = (_hv(clock, seed, salt + 3) - 0.5) * 0.7
    bend2 = (_hv(clock, seed, salt + 6) - 0.5) * 0.7
    L = length * (0.6 + 0.8 * _hv(clock, seed, salt + 4))
    if edge == 0:
        ax, ay, nx, ny = u * (w - 1), 0.0, 0.0, 1.0
    elif edge == 1:
        ax, ay, nx, ny = u * (w - 1), float(h - 1), 0.0, -1.0
    elif edge == 2:
        ax, ay, nx, ny = 0.0, u * (h - 1), 1.0, 0.0
    else:
        ax, ay, nx, ny = float(w - 1), u * (h - 1), -1.0, 0.0
    sway = (float(value_noise(pos * 0.35 + 0.5, s * 3.1 + 0.25,
                              (111 + (int(seed) & 0xffff)) & 0x7fffffff)[0])
            - 0.5) * 0.5
    flutter = (_hv(pos, seed, salt + 5, j=s + 1) - 0.5) * 0.08
    ang = lean + sway + flutter
    ca, sa = np.cos(ang), np.sin(ang)
    dx, dy = nx * ca - ny * sa, nx * sa + ny * ca           # the lean
    px, py = -dy, dx                                        # sideways
    flex = bend + (float(value_noise(pos * 0.27 + 3.0, s * 2.3 + 0.75,
                                     (112 + (int(seed) & 0xffff)) & 0x7fffffff)[0])
                   - 0.5) * 0.25
    flex2 = bend2 - (float(value_noise(pos * 0.31 + 6.0, s * 1.9 + 0.5,
                                       (115 + (int(seed) & 0xffff)) & 0x7fffffff)[0])
                     - 0.5) * 0.25
    p0 = np.array([ax, ay], np.float32)
    fwd = np.array([dx, dy], np.float32)
    side = np.array([px, py], np.float32)
    p3 = p0 + fwd * np.float32(L)
    p1 = p0 + fwd * np.float32(L / 3.0) + side * np.float32(flex * L)
    p2 = p0 + fwd * np.float32(2.0 * L / 3.0) + side * np.float32(flex2 * L)
    n = max(int(L / 2.0), 4)
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
    u1 = 1.0 - t
    return (u1 ** 3 * p0 + 3 * u1 * u1 * t * p1 + 3 * u1 * t * t * p2
            + t ** 3 * p3).astype(np.float32)


def hairs(rgb, st, frame=0, seed=0):
    """Hairs in the projector gate: dark, soft-edged, each one dancing
    at the aperture's edge for its run of frames."""
    x = np.asarray(rgb, np.float32)
    h, w = x.shape[:2]
    amount, length, width, hold = hair_params(st, h)
    alive = _alive(frame, seed, amount, hold, _HAIR_SLOTS, 60)
    if not alive:
        return rgb
    out = np.array(x, copy=True)
    for s, e, pos, life in alive:
        pts = hair_points(h, w, s, e, pos, seed, length)
        p = _polyline_patch(h, w, pts, width)
        if p is not None:
            _paint(out, p, (0.03, 0.028, 0.025), 0.85)
    return out


# --------------------------------------------------------- the scratches


def scratch_params(st, h):
    amount = float(np.clip(getattr(st, 'film_scratches', 0.0), 0.0, 1.0))
    width = float(max(getattr(st, 'film_scratch_width', 1.2), 0.3)) * scale_of(h)
    hold = int(max(getattr(st, 'film_scratch_hold', 48), 1))
    side = str(getattr(st, 'film_scratch_side', 'EMULSION') or 'EMULSION').upper()
    return amount, width, hold, side


def scratch_columns(h, w, s, e, pos, seed, width, salt=80):
    """One scratch's coverage: (x0, x1, cov (H, x1 - x0), depth, side)
    -- a vertical line at a hashed place with a slow wander and a
    per-frame jitter, its width and depth hashed once per scratch, its
    strength wandering along its length."""
    clock = (e * 8 + s) & 0x7fffffff
    xc = _hv(clock, seed, salt) * (w - 1)
    depth = 0.25 + 0.75 * _hv(clock, seed, salt + 1)
    wd = width * (0.7 + 0.6 * _hv(clock, seed, salt + 2))
    side = int(_hv(clock, seed, salt + 3) * 2.0) % 2
    sc = scale_of(h)
    wander = (float(value_noise(pos * 0.13 + 0.5, s * 1.7 + 0.25,
                                (113 + (int(seed) & 0xffff)) & 0x7fffffff)[0])
              - 0.5) * 12.0 * sc
    jitter = (_hv(pos, seed, salt + 4, j=s + 1) - 0.5) * 0.7 * sc
    xc = xc + wander + jitter
    x0 = int(max(np.floor(xc - wd - 1), 0))
    x1 = int(min(np.ceil(xc + wd + 1), w - 1))
    if x1 < x0:
        return None
    xs = np.arange(x0, x1 + 1, dtype=np.float32)
    across = np.clip(0.5 * wd - np.abs(xs - np.float32(xc)) + 0.5, 0.0, 1.0)
    ys = np.arange(h, dtype=np.float32)
    along = 0.7 + 0.3 * value_noise(ys / max(40.0 * sc, 1.0) + 0.5, s * 5.3 + 0.25,
                                    (114 + (int(seed) & 0xffff)) & 0x7fffffff)
    cov = (along.astype(np.float32)[:, None] * across[None, :]).astype(np.float32)
    return x0, x1 + 1, cov, depth, side


def scratches_alive(st, frame, seed, h, w):
    amount, width, hold, side_mode = scratch_params(st, h)
    alive = _alive(frame, seed, amount, hold, _SCRATCH_SLOTS, 80)
    out = []
    for s, e, pos, life in alive:
        c = scratch_columns(h, w, s, e, pos, seed, width)
        if c is not None:
            out.append(c)
    return out, side_mode


def scratch_dyes(dyes, st, frame, seed, process):
    """Emulsion-side scratches on the dye amounts: a dye-transfer print
    loses all its dyes together, a duplitized two-colour print one
    side's dye."""
    h, w = dyes[0].shape
    cols, side_mode = scratches_alive(st, frame, seed, h, w)
    if not cols or side_mode != 'EMULSION':
        return dyes
    out = [np.array(d, np.float32, copy=True) for d in dyes]
    for x0, x1, cov, depth, side in cols:
        if process == 'TWO_COLOUR':
            f = np.float32(min(1.5 * depth, 1.0))
            r = side % len(out)
            out[r][:, x0:x1] *= (1.0 - f * cov)
        else:
            for d in out:
                d[:, x0:x1] *= (1.0 - np.float32(depth) * cov)
    return out


def scratch_base(rgb, st, frame, seed):
    """Base-side scratches on the transmittance: the lamp scattered,
    a soft dark line."""
    x = np.asarray(rgb, np.float32)
    h, w = x.shape[:2]
    cols, side_mode = scratches_alive(st, frame, seed, h, w)
    if not cols or side_mode != 'BASE':
        return rgb
    out = np.array(x, copy=True)
    for x0, x1, cov, depth, side in cols:
        out[:, x0:x1] *= (1.0 - np.float32(0.45 * depth) * cov)[..., None]
    return out


def scratch_plain(rgb, st, frame=0, seed=0):
    """Scratches on a frame with no process, read as a chromogenic
    print: the emulsion side loses yellow first, magenta next, cyan
    last -- the blue scratch -- as a power on the channel's
    transmittance; the base side scatters."""
    x = np.asarray(rgb, np.float32)
    h, w = x.shape[:2]
    cols, side_mode = scratches_alive(st, frame, seed, h, w)
    if not cols:
        return rgb
    out = np.array(x, copy=True)
    for x0, x1, cov, depth, side in cols:
        if side_mode == 'BASE':
            out[:, x0:x1] *= (1.0 - np.float32(0.45 * depth) * cov)[..., None]
            continue
        fy = float(np.clip(3.0 * depth, 0.0, 1.0))
        fm = float(np.clip(3.0 * depth - 1.0, 0.0, 1.0))
        fc = float(np.clip(3.0 * depth - 2.0, 0.0, 1.0))
        t = np.maximum(out[:, x0:x1], np.float32(1e-4))
        for ch, f in ((2, fy), (1, fm), (0, fc)):
            if f > 0.0:
                out[:, x0:x1, ch] = np.power(t[..., ch], 1.0 - np.float32(f) * cov)
    return out


# --------------------------------------------------------- the cue marks


def cue_patch(frame, st, h, w, fps=24.0):
    """The cue mark's coverage for `frame`, or None: a reel of Reel
    Length minutes at `fps`; the motor cue starts eight seconds before
    the reel's end and the changeover cue one second and four frames
    before it, each four frames long; a scraped disc top right, 3.5
    percent of the frame's width across, with the burr of its rim.
    Returns (y0, y1, x0, x1, cov_disc, cov_rim)."""
    reel = float(max(getattr(st, 'film_reel', 0.0), 0.0))
    if reel <= 0.0:
        return None
    fps = float(max(fps, 1.0))
    R = int(round(reel * 60.0 * fps))
    if R < 8:
        return None
    pos = (int(frame) - 1) % R
    starts = (R - int(round(8.0 * fps)), R - int(round(fps)) - 4)
    if not any(s0 >= 0 and s0 <= pos < s0 + 4 for s0 in starts):
        return None
    r = 0.0175 * w
    cx = w - 0.055 * w
    cy = h - 0.075 * h
    rw = max(1.5 * scale_of(h), 1.0)
    x0 = int(max(np.floor(cx - r - rw - 1), 0))
    x1 = int(min(np.ceil(cx + r + rw + 1), w - 1))
    y0 = int(max(np.floor(cy - r - rw - 1), 0))
    y1 = int(min(np.ceil(cy + r + rw + 1), h - 1))
    if x1 < x0 or y1 < y0:
        return None
    yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1].astype(np.float32)
    d = np.sqrt((xx - np.float32(cx)) ** 2 + (yy - np.float32(cy)) ** 2)
    disc = np.clip(np.float32(r) - d + 0.5, 0.0, 1.0).astype(np.float32)
    rim = np.clip(np.float32(0.5 * rw) - np.abs(d - np.float32(r)) + 0.5,
                  0.0, 1.0).astype(np.float32)
    return y0, y1 + 1, x0, x1 + 1, disc, rim


def cue_marks(rgb, st, frame=0, fps=24.0):
    """The cue mark on the transmittance: the disc scraped clear (the
    transmittance raised to 1 - coverage), the rim's burr darkening."""
    x = np.asarray(rgb, np.float32)
    h, w = x.shape[:2]
    p = cue_patch(frame, st, h, w, fps)
    if p is None:
        return rgb
    y0, y1, x0, x1, disc, rim = p
    out = np.array(x, copy=True)
    t = np.maximum(out[y0:y1, x0:x1], np.float32(1e-4))
    t = np.power(t, (1.0 - disc)[..., None])
    out[y0:y1, x0:x1] = (t * (1.0 - np.float32(0.75) * rim)[..., None]).astype(np.float32)
    return out


# ------------------------------------------------------------ the chains


def wear_on(st):
    """Whether any wear stage that runs OUTSIDE the colour process is
    on (the post chain's fast exit)."""
    return (float(getattr(st, 'film_dust', 0.0)) > 0.0
            or grain_on(st)
            or float(getattr(st, 'film_hairs', 0.0)) > 0.0
            or float(getattr(st, 'film_scratches', 0.0)) > 0.0
            or float(getattr(st, 'film_reel', 0.0)) > 0.0)


def plain_print(rgb, st, frame=0, seed=0, fps=24.0):
    """The print's own wear on a frame that went through no colour
    process: the negative's dust, the grain, the scratches, the cue
    marks, in that order, on the frame's own transmittance."""
    out = negative_dust_plain(rgb, st, frame, seed)
    out = grain_plain(out, st, frame, seed)
    out = scratch_plain(out, st, frame, seed)
    out = cue_marks(out, st, frame, fps)
    return out
