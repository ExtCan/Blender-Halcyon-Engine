"""Software scanline / z-buffer rasteriser.

This is a real rasteriser, not a wrapper around anything: clip-space transform,
near-plane Sutherland-Hodgman clipping, integer-snappable screen coordinates,
top-left fill rule, perspective-correct barycentrics, and an A-buffer fragment
capture path for sorted transparency (Carpenter 1984 -- period correct).

The G-buffer stores, per pixel, the *original* triangle id plus barycentric
weights over that triangle's three corners. Every vertex attribute (normal, uv,
colour, tangent...) is reconstructed later from those weights, so the rasteriser
never has to know what attributes exist.
"""

import collections
import dataclasses

import numpy as np

from . import mathx as M

EMPTY = -1


# ---------------------------------------------------- R251 raster options

#: the camera's depth mapping, for the W encodings: z_ndc = A + B / w with
#: w the eye depth (w = near -> -1, w = far -> +1). Every reader takes the
#: fields BY NAME; nothing unpacks the tuple.
WParams = collections.namedtuple('WParams', 'near far A B is_ortho')

#: the depth encodings (C007 / C026 / C075), in `hal_zenc` order
DEPTH_ENCODINGS = ('LINEAR', 'N64_FLOAT18', 'GC_14E2', 'GC_13E3', 'GC_12E4',
                   'W_FIXED', 'VOODOO_W16')
W_ENCODINGS = ('W_FIXED', 'VOODOO_W16')


@dataclasses.dataclass
class RasterOpts:
    """R251: every new rasteriser dial, carried as ONE object through the
    camera-space `rasterize` call sites and the kernel road. The defaults
    reproduce 1.89.0 bitwise on both roads; `key()` is the G-buffer cache
    key's entry (the arrays of a backdrop never enter a key, its `sig`
    does). `depth_bits` is NOT a field: the fill's own kwarg carries it.
    """
    pixel_shift: float = 0.0      # C084: 0.5 under INTEGER_D3D
    reject: bool = False          # C027: whole-triangle near/far/guard rejection
    size_limit: object = None     # C004: (wmax, hmax) internal px, or None
    enc: str = 'LINEAR'           # C007/C026/C075: the depth encoding
    wparams: object = None        # WParams for the W encodings
    jitter: object = None         # C127 (wave 2)
    cvg: bool = False             # C001 (wave 2)
    clear: object = None          # C038 (wave 2)

    def key(self):
        clear_sig = None
        if self.clear is not None:
            try:
                clear_sig = self.clear[2]
            except Exception:                                   # noqa: BLE001
                clear_sig = None
        wp = None
        if self.wparams is not None:
            wp = tuple(float(v) if not isinstance(v, bool) else v
                       for v in self.wparams)
        return (float(self.pixel_shift), bool(self.reject),
                None if self.size_limit is None else tuple(self.size_limit),
                str(self.enc), wp,
                None if self.jitter is None else str(self.jitter),
                bool(self.cvg), clear_sig)


def pixel_shift_of(st):
    """C084: the raster's pixel-centre shift for a settings object."""
    return 0.5 if str(getattr(st, 'pixel_center', 'HALF')) == 'INTEGER_D3D' \
        else 0.0


# ------------------------------------------ R251 C012 vertex quantisation

#: the normal grids of the two consoles: the PS1's 1.3.12 fixed point
#: (4096 units per 1.0), the N64's signed 8-bit (127 per 1.0)
VERTEX_NORMAL_GRID = {'PS1': 4096.0, 'N64': 127.0}
VERTEX_QUANTIZE_MODES = ('NONE', 'PS1', 'N64')


def quantize_mesh(mesh, mode, units):
    """C012: the mesh rounded to a console's vertex format (a NEW MeshData
    sharing every unchanged array by reference), or the mesh itself.

    PS1 (GTE 16-bit vertices, 1.3.12 normals, the GPU's 8-bit texel
    coordinates, 8-bit colours) and N64 (libultra Vtx_t: short ob[3],
    signed char n[3], 8-bit colours; UVs untouched -- the S10.5 grid
    needs the texture's own size, which the mesh does not carry). All
    in float32, `np.round` (half to even) for positions, normals and
    colours; PS1 UVs TRUNCATE (`np.floor(uv * 256) / 256`): the PS1
    exporters truncated float UVs to the integer texel coordinate -- the
    one truncation in a function that otherwise rounds, disclosed here
    and in the CHANGELOG. The lattice is a WORLD grid of `units` per
    unit (disclosed: the consoles quantised in model space; Halcyon's
    mesh is one world-space soup). Face normals are recomputed over the
    quantised corners -- the crunched polygon is what the machine lit.

    Idempotent and re-diallable: the copy carries `_quant_source` (the
    original) and `_quant_tag = (mode, units)`; a call starts from the
    source, returns the input unchanged when its tag matches, and
    returns the SOURCE for mode NONE, so a persistent scene can turn the
    dial both ways without quantising twice.
    """
    mode = str(mode or 'NONE')
    if mesh is None:
        return mesh
    source = getattr(mesh, '_quant_source', None)
    if source is None:
        source = mesh
    if mode == 'NONE' or mode not in VERTEX_NORMAL_GRID:
        return source
    units = float(units)
    tag = (mode, units)
    if getattr(mesh, '_quant_tag', None) == tag:
        return mesh
    f32 = np.float32
    u = f32(units)
    verts = np.asarray(source.verts, np.float32)
    vq = np.round(verts * u)
    vq = np.clip(vq, f32(-32768.0), f32(32767.0))
    vq = (vq / u).astype(np.float32)
    normals = None
    if getattr(source, 'normals', None) is not None:
        q = f32(VERTEX_NORMAL_GRID[mode])
        nq = np.round(np.asarray(source.normals, np.float32) * q)
        nq = (nq / q).astype(np.float32)
        normals = M.normalize(nq).astype(np.float32)
    uvs = getattr(source, 'uvs', None)
    uvs2 = getattr(source, 'uvs2', None)
    if mode == 'PS1':
        if uvs is not None:
            uvs = (np.floor(np.asarray(uvs, np.float32) * f32(256.0))
                   / f32(256.0)).astype(np.float32)
        if uvs2 is not None:
            uvs2 = (np.floor(np.asarray(uvs2, np.float32) * f32(256.0))
                    / f32(256.0)).astype(np.float32)
    colors = getattr(source, 'colors', None)
    if colors is not None:
        colors = (np.round(np.asarray(colors, np.float32) * f32(255.0))
                  / f32(255.0)).astype(np.float32)
    face_normals = getattr(source, 'face_normals', None)
    tris = getattr(source, 'tris', None)
    if face_normals is not None and tris is not None \
            and np.asarray(tris).size:
        t = np.asarray(tris, np.int64)
        v0, v1, v2 = vq[t[:, 0]], vq[t[:, 1]], vq[t[:, 2]]
        fn = np.cross(v1 - v0, v2 - v0).astype(np.float32)
        ln = np.sqrt((fn * fn).sum(axis=1, keepdims=True))
        # a polygon the lattice collapsed has no normal of its own: it
        # keeps the export's (the machine lit a zero-area polygon with
        # nothing; a zero vector would NaN the shading), so a crunch
        # never flips a face against the export's winding
        src_fn = np.asarray(face_normals, np.float32)
        face_normals = np.where(ln > f32(1e-12), fn / np.maximum(ln, f32(1e-30)),
                                src_fn).astype(np.float32)
    out = dataclasses.replace(source, verts=vq, normals=normals, uvs=uvs,
                              uvs2=uvs2, colors=colors,
                              face_normals=face_normals)
    out._quant_source = source
    out._quant_tag = tag
    return out


# ------------------------------------ R251 C127 jittered sample positions

def jitter_key(frame, seed):
    """C127: the 24-bit hash key of a (frame, seed) pair -- 24 bits so it
    rides a float uniform exactly."""
    return (int(frame) * 65537 + int(seed)) & 0xffffff


def jitter_at(ix, iy, key):
    """C127: the per-pixel sample offsets (jx, jy) in (-0.5, 0.5) for
    integer pixel indices `ix, iy` (any shape) -- REYES's one jittered
    sample per subpixel, from `film._hash_u32_raw` (the integer hash the
    grain already ships; the kernel's `hal_jit_hash` is its verbatim GLSL
    twin). Every step is exact in float32: (n + 0.5) * 2^-12 - 0.5 for
    a 12-bit n, so a sample never sits on a cell edge."""
    from . import film as _film
    f32 = np.float32
    # arrays, never scalars: NumPy warns on a wrapping scalar multiply
    # where the array road wraps silently (the hash's own arithmetic)
    h = _film._hash_u32_raw(np.atleast_1d(np.asarray(ix)),
                            np.atleast_1d(np.asarray(iy)), int(key))
    jx = (h & np.uint32(0xfff)).astype(np.float32)
    jx = jx + f32(0.5)
    jx = jx * f32(0.000244140625)
    jx = jx - f32(0.5)
    jy = ((h >> np.uint32(12)) & np.uint32(0xfff)).astype(np.float32)
    jy = jy + f32(0.5)
    jy = jy * f32(0.000244140625)
    jy = jy - f32(0.5)
    return jx, jy


def jitter_offsets(width, height, key):
    """C127: the (JX, JY) offset planes, (H, W) float32, of one frame."""
    yy, xx = np.mgrid[0:int(height), 0:int(width)]
    return jitter_at(xx.astype(np.uint32), yy.astype(np.uint32), key)


# ------------------------------------------ R251 C001 the RDP's coverage

def _sub_offsets(n_sub):
    """The n_sub x n_sub subsample offsets of one pixel, float32:
    float(i) / n_sub - 0.5 + 0.5 / n_sub (for 4: exactly -0.375, -0.125,
    0.125, 0.375 -- the kernel's `float(i) * 0.25 - 0.375`)."""
    f32 = np.float32
    n = int(n_sub)
    i = np.arange(n, dtype=np.float32)
    o = i / f32(n)
    o = o - f32(0.5)
    o = o + f32(0.5) / f32(n)
    return o.astype(np.float32)


def coverage_count(xa, ya, xb, yb, xc, yc, ar, X, Y, n_sub=4):
    """C001: how many of the n_sub x n_sub subsamples of the pixel centred
    at (X, Y) lie inside the triangle (xa, ya), (xb, yb), (xc, yc) of
    signed area `ar` -- 0..n_sub**2, int32, elementwise over X / Y (the
    corners scalars or arrays of the same shape).

    The fill's own three edge expressions at every subsample, both
    edges INCLUSIVE and NO wobble window: this is a count, not coverage,
    and the same expressions run on both roads (the kernel's resolve).
    The subsamples are relative to the PIXEL (the RDP's), so the C127
    jitter does not move them. The DS's 32-level edge coverage or any
    other machine's grid is the `n_sub` argument away.
    """
    f32 = np.float32
    X = np.asarray(X, np.float32)
    Y = np.asarray(Y, np.float32)
    xa, ya = np.asarray(xa, np.float32), np.asarray(ya, np.float32)
    xb, yb = np.asarray(xb, np.float32), np.asarray(yb, np.float32)
    xc, yc = np.asarray(xc, np.float32), np.asarray(yc, np.float32)
    pos = np.asarray(ar, np.float32) > f32(0.0)
    offs = _sub_offsets(n_sub)
    n = np.zeros(np.broadcast(X, Y).shape, np.int32)
    for oy in offs:
        Ys = Y + oy
        for ox in offs:
            Xs = X + ox
            s0 = (xc - xb) * (Ys - yb) - (yc - yb) * (Xs - xb)
            s1 = (xa - xc) * (Ys - yc) - (ya - yc) * (Xs - xc)
            s2 = (xb - xa) * (Ys - ya) - (yb - ya) * (Xs - xa)
            ins = np.where(pos,
                           (s0 >= f32(0.0)) & (s1 >= f32(0.0)) & (s2 >= f32(0.0)),
                           (s0 <= f32(0.0)) & (s1 <= f32(0.0)) & (s2 <= f32(0.0)))
            n += ins.astype(np.int32)
    return n


def coverage16(xa, ya, xb, yb, xc, yc, ar, X, Y):
    """C001: the RDP's 3-bit coverage of the winning triangle at the pixel
    centred at (X, Y): the 16-subsample count dithered to eight
    (`cvg = clip((n16 + 1) >> 1, 1, 8)`, N64 Programming Manual 15.2),
    stored as cvg - 1 in 0..7 (7 = full). uint8-range int32."""
    n16 = coverage_count(xa, ya, xb, yb, xc, yc, ar, X, Y, 4)
    cvg = np.clip((n16 + 1) >> 1, 1, 8)
    return (cvg - 1).astype(np.int32)


def sliver16(xa, ya, xb, yb, xc, yc, X, Y):
    """C001: the extended sliver mark -- whether ANY of the 16 subsamples
    has an edge function within the wobble window (EDGE_WOBBLE of the
    product magnitudes, the fill's own law) of zero, so a driver's FMA
    could move its count. The CPU replay reads it for the tests; the
    kernel computes the same 48 tests under `hal_refer`."""
    f32 = np.float32
    X = np.asarray(X, np.float32)
    Y = np.asarray(Y, np.float32)
    offs = _sub_offsets(4)
    out = np.zeros(np.broadcast(X, Y).shape, bool)
    for oy in offs:
        Ys = Y + oy
        for ox in offs:
            Xs = X + ox
            s0 = (xc - xb) * (Ys - yb) - (yc - yb) * (Xs - xb)
            s1 = (xa - xc) * (Ys - yc) - (ya - yc) * (Xs - xc)
            s2 = (xb - xa) * (Ys - ya) - (yb - ya) * (Xs - xa)
            m0 = np.abs((xc - xb) * (Ys - yb)) + np.abs((yc - yb) * (Xs - xb))
            m1 = np.abs((xa - xc) * (Ys - yc)) + np.abs((ya - yc) * (Xs - xc))
            m2 = np.abs((xb - xa) * (Ys - ya)) + np.abs((yb - ya) * (Xs - xa))
            out |= (np.abs(s0) < EDGE_WOBBLE * m0) | (np.abs(s1) < EDGE_WOBBLE * m1) \
                | (np.abs(s2) < EDGE_WOBBLE * m2)
    return out


# --------------------------------- R251 C038 the rear-plane depth bitmap

def backdrop_clear(depth_img, offset, rw, rh, opts, depth_bits, ss=1):
    """C038: the DS's CLEAR_DEPTH bitmap as the frame's own z-buffer
    clear -> (key, dec, sig).

    `depth_img` is an (h, w) eye-space distance image in scene units (a
    Z pass; row 0 = bottom), tiled 1:1 over the frame's OUTPUT pixels with
    the integer `offset` (the DS's CLRIMAGE_OFFSET, wrap-around
    addressing) -- under a supersample factor `ss` every ss x ss block of
    internal pixels reads one bitmap pixel (nearest, the Z pass is at
    output size); pixels that are not finite or not positive carry no
    depth (+inf).
    The distance goes through the CAMERA's own depth mapping (rule 2's
    `WParams`, read BY NAME: perspective z = A + B / w, orthographic
    z = A * w + B) and then through `encode_depth` at the frame's own
    encoding and bit count -- the SAME code the frame's fragments get, so
    the compare is key against key. `sig` is a content hash for the
    cache keys (the arrays never enter a key). Disclosed: the DS expanded
    its 15-bit bitmap on its own 24-bit scale; here the backdrop takes
    the frame's encoding, so `depth_precision` 15 reproduces the DS's
    granularity.
    """
    f32 = np.float32
    img = np.asarray(depth_img, np.float32)
    if img.ndim == 3:
        img = img[:, :, 0]
    h_img, w_img = int(img.shape[0]), int(img.shape[1])
    try:
        ox, oy = int(offset[0]), int(offset[1])
    except Exception:                                           # noqa: BLE001
        ox, oy = 0, 0
    yy, xx = np.mgrid[0:int(rh), 0:int(rw)]
    ss = max(int(ss), 1)
    if ss > 1:
        yy, xx = yy // ss, xx // ss
    wd = img[(yy + oy) % h_img, (xx + ox) % w_img].astype(np.float32)
    valid = np.isfinite(wd) & (wd > 0)
    wd_safe = np.where(valid, wd, f32(1.0))
    wp = opts.wparams if opts is not None else None
    A = f32(wp.A) if wp is not None else f32(0.0)
    B = f32(wp.B) if wp is not None else f32(0.0)
    is_ortho = bool(wp.is_ortho) if wp is not None else False
    if is_ortho:
        iw = np.ones_like(wd_safe)
        dec_ndc = A * wd_safe
        dec_ndc = dec_ndc + B
    else:
        iw = f32(1.0) / wd_safe
        dec_ndc = B * iw
        dec_ndc = A + dec_ndc
    dec_ndc = np.clip(dec_ndc, f32(-1.0), f32(1.0)).astype(np.float32)
    opts_e = opts if opts is not None else RasterOpts()
    key, dec = encode_depth(dec_ndc, iw, opts_e, int(depth_bits))
    key = np.where(valid, key, np.inf).astype(np.float32)
    dec = np.where(valid, dec, np.inf).astype(np.float32)
    sig = hash(key.tobytes())
    return key, dec, sig


def jitter_planes(opts, width, height):
    """The fill's (JX, JY) for a RasterOpts, or None when the samples
    sit at the subpixel centres (the default, bitwise 1.89.0)."""
    if opts is None or getattr(opts, 'jitter', None) is None:
        return None
    frame, seed = opts.jitter
    return jitter_offsets(width, height, jitter_key(frame, seed))

#: A-buffer depth tolerance. A modeled contact -- glass resting ON an
#: opaque surface, a box standing on a floor -- interpolates its
#: transparent fragments to the opaque depth plus or minus a few float32
#: ULPs, and the CPU and compute rasterisers round those ULPs
#: DIFFERENTLY (measured ~9e-7 apart on zndc). A bare `<` therefore let
#: the render DEVICE decide, pixel by pixel, whether the coplanar layer
#: exists: 1036 px of salt-and-pepper between otherwise-identical
#: frames, found by the self test the moment it diffed a transparent
#: frame across rasterisers. Collection keeps anything within ~30 ULPs
#: of the surface (real geometric separation is orders of magnitude
#: larger), and the compositor tests against the SAME limit, so the tie
#: lands the same way under any rounding. Opaque hidden-surface removal
#: keeps its exact `<`: this tolerance is only for deciding whether a
#: see-through fragment sits on or behind the surface.
ABUF_DEPTH_TOL_REL = np.float32(4e-6)
ABUF_DEPTH_TOL_ABS = np.float32(1e-7)


#: the coverage wobble window, in the units of the mechanism: float32
#: rounding moves an edge function by a few ulps of its PRODUCT
#: magnitudes, and the two triangles at a shared edge compute that edge
#: with different expressions -- without the window both can exclude the
#: same boundary pixel and the background shows through (the field's
#: "faint wireframe on all objects" at high resolutions). One constant,
#: four engines: loop fill, batched fill, the compute kernel, the replay.
#: float32 ON PURPOSE -- a Python float would upcast the whole test.
EDGE_WOBBLE = np.float32(2.5e-7)


def abuf_depth_limit(opaque_z):
    """The keep limit for A-buffer fragments against the opaque depth."""
    return opaque_z + np.abs(opaque_z) * ABUF_DEPTH_TOL_REL \
        + ABUF_DEPTH_TOL_ABS


def quantize_depth(zz, depth_bits):
    """Round interpolated depth to the z-buffer's grid, PER PIXEL.

    What an N-bit z-buffer of the period actually did: interpolate at
    full precision, round when the value meets the buffer. Every stored
    depth lies exactly on the 2^N-step grid, so two surfaces fight only
    where they are genuinely within a step of each other -- thin bands
    at the crossing, the authentic artifact. (Quantizing the VERTEX z
    before interpolation -- the old way -- tilted whole depth planes
    and cut solid wedges through close-fitting geometry like a face.)
    float32 throughout, and the compute kernel applies the same formula
    with roundEven, so both rasterisers round the same half-cases the
    same way.
    """
    if depth_bits >= 32:
        return zz
    steps = np.float32((1 << int(max(2, int(depth_bits)))) - 1)
    z32 = np.asarray(zz, np.float32)
    return (np.round((z32 * np.float32(0.5) + np.float32(0.5)) * steps)
            / steps * np.float32(2.0) - np.float32(1.0))


# ----------------------------------------------- R251 the depth encodings

#: the GameCube compressed-Z layouts: (mantissa bits, exponent bits).
#: Reconstructed from the GX manual's stated resolutions (14e2: 15 bits
#: near / 17 far; 13e3: 14 / 20; 12e4: 13 / full 24) and disclosed as a
#: reconstruction in the CHANGELOG.
GC_LAYOUTS = {'GC_14E2': (14, 2), 'GC_13E3': (13, 3), 'GC_12E4': (12, 4)}

_F32 = np.float32


def _zn_of(zz):
    """NDC z -> [0, 1], one op per statement, exactly as the kernel."""
    f32 = _F32
    zn = np.asarray(zz, np.float32) * f32(0.5)
    zn = zn + f32(0.5)
    return np.clip(zn, f32(0.0), f32(1.0))


def _n64_code(z18):
    """The RDP's 14-bit piecewise-floating code of an 18-bit z (int32)."""
    e = z18 >> 11
    shift = np.select([e < 0x40, e < 0x60, e < 0x70, e < 0x78, e < 0x7c,
                       e < 0x7e], [6, 5, 4, 3, 2, 1], 0)
    return (z18 >> shift) << shift


def _gc_code(z24, M, E):
    """The GameCube compressed-Z code of a 24-bit z (int32)."""
    cap = (1 << E) - 1
    e = np.zeros_like(z24)
    live = np.ones(z24.shape, bool)
    for i in range(cap):                 # leading ones from bit 23, capped
        one = ((z24 >> (23 - i)) & 1) == 1
        live &= one
        e += live.astype(np.int32)
    kept = e + (e < cap).astype(np.int32) + M
    drop = np.maximum(24 - kept, 0)
    return (z24 >> drop) << drop


def _voodoo_code(t):
    """MAME's compute_wfloat on the float 1/w normalised to the near
    plane (`t = invw * near`): the octave by exact doubling, the 12
    inverted mantissa bits, plus one; 0 at and before the near plane,
    0xffff past the 16th octave. Returns int32 codes."""
    f32 = _F32
    x = np.asarray(t, np.float32).copy()
    d = np.zeros(x.shape, np.int32)
    for _ in range(17):                  # exact: multiplying by 2 never rounds
        up = x < f32(1.0)
        x = np.where(up, x * f32(2.0), x)
        d += up.astype(np.int32)
    lo = t >= f32(1.0)
    big = t < f32(1.52587890625e-05)
    e = np.maximum(d - 1, 0)
    m = x - f32(1.0)
    m = m * f32(4096.0)
    m12 = np.floor(m).astype(np.int32)
    d16 = ((e << 12) | (4095 - m12)) + 1
    d16 = np.minimum(d16, 65535)
    return np.where(big, 65535, np.where(lo, 0, d16)).astype(np.int32)


def decode_key(key, opts, depth_bits=24):
    """The DECODED NDC value of a stored depth code, elementwise float32.

    The one decode both roads share: the CPU fill stores it beside the
    key and the kernel road's host decode (`gbuffer_into`) calls it on
    the readback, so the driver's division never enters `depth`/`zndc`.
    """
    f32 = _F32
    key = np.asarray(key, np.float32)
    enc = str(opts.enc)
    if enc == 'LINEAR':
        return key
    if enc == 'N64_FLOAT18':
        dec = key / f32(262143.0)
        dec = dec * f32(2.0)
        return dec - f32(1.0)
    if enc in GC_LAYOUTS:
        dec = key / f32(16777215.0)
        dec = dec * f32(2.0)
        return dec - f32(1.0)
    wp = opts.wparams
    if enc == 'W_FIXED':
        bits = min(int(depth_bits), 24)
        steps = f32((1 << bits) - 1)
        rng = f32(wp.far) - f32(wp.near)
        wq = key / steps
        wq = wq * rng
        wq = wq + f32(wp.near)
        iwq = f32(1.0) / wq
        dec = f32(wp.B) * iwq
        return f32(wp.A) + dec
    if enc == 'VOODOO_W16':
        d16 = np.asarray(np.rint(key), np.int32)
        dm = np.maximum(d16 - 1, 0)
        ed = dm >> 12
        md = 4095 - (dm & 4095)
        ud = f32(1.0) + md.astype(np.float32) / f32(4096.0)
        td = np.ldexp(ud, -(ed + 1)).astype(np.float32)
        iwq = td / f32(wp.near)
        dec = f32(wp.B) * iwq
        dec = f32(wp.A) + dec
        return np.where(d16 >= 65535, f32(1.0),
                        np.where(d16 <= 0, f32(-1.0), dec)).astype(np.float32)
    raise ValueError(f'unknown depth encoding {enc!r}')


def encode_depth(zz, invw, opts, depth_bits):
    """(key, dec) of interpolated depth under `opts.enc`, elementwise.

    `key` is the z-buffer's stored code as an exact integer in float32
    (the fill compares on it, ties to the lowest id); `dec` the decoded
    NDC value that `depth`/`zndc` receive. LINEAR is `quantize_depth`
    at the fill's own `depth_bits` (bitwise 1.89.0). float32 and int32
    throughout, one op per statement, as the kernel computes it:
      N64_FLOAT18  the RDP's 14-bit piecewise-floating code of 18-bit z
      GC_14E2/13E3/12E4  the GameCube's leading-ones compressed 24-bit z
      W_FIXED      eye depth w on a 2^bits grid between near and far
      VOODOO_W16   1/w as a 4-bit octave + 12 inverted mantissa bits
    """
    f32 = _F32
    enc = str(opts.enc)
    if enc == 'LINEAR':
        q = quantize_depth(zz, depth_bits) if depth_bits < 32 else zz
        return q, q
    zz = np.asarray(zz, np.float32)
    if enc == 'N64_FLOAT18':
        zn = _zn_of(zz)
        zf = zn * f32(262143.0)
        z18 = np.floor(zf).astype(np.int32)
        key = _n64_code(z18).astype(np.float32)
        return key, decode_key(key, opts, depth_bits)
    if enc in GC_LAYOUTS:
        M, E = GC_LAYOUTS[enc]
        zn = _zn_of(zz)
        zf = zn * f32(16777215.0)
        z24 = np.floor(zf).astype(np.int32)
        key = _gc_code(z24, M, E).astype(np.float32)
        return key, decode_key(key, opts, depth_bits)
    wp = opts.wparams
    invw = np.asarray(invw, np.float32)
    if enc == 'W_FIXED':
        bits = min(int(depth_bits), 24)
        steps = f32((1 << bits) - 1)
        near = f32(wp.near)
        rng = f32(wp.far) - near
        w = f32(1.0) / invw
        t = w - near
        t = t / rng
        t = np.clip(t, f32(0.0), f32(1.0))
        v = t * steps
        key = np.floor(v).astype(np.float32)
        return key, decode_key(key, opts, depth_bits)
    if enc == 'VOODOO_W16':
        t = invw * f32(wp.near)
        key = _voodoo_code(t).astype(np.float32)
        return key, decode_key(key, opts, depth_bits)
    raise ValueError(f'unknown depth encoding {enc!r}')


def encoding_index(enc):
    """`hal_zenc`: the encoding's number in DEPTH_ENCODINGS order."""
    return float(DEPTH_ENCODINGS.index(str(enc)))


def encoding_uniforms(opts, depth_bits):
    """The kernel's encoding uniforms (host float32, pre-rounded):
    hal_zenc, hal_wob (the referral window in the encoding's pre-floor
    units, 2.5e-6 * full scale -- the LINEAR window's own law),
    hal_wnear, hal_wrange, hal_wsteps."""
    enc = str(opts.enc) if opts is not None else 'LINEAR'
    out = {'hal_zenc': 0.0, 'hal_wob': 0.0, 'hal_wnear': 0.0,
           'hal_wrange': 1.0, 'hal_wsteps': 1.0}
    if enc == 'LINEAR':
        return out
    out['hal_zenc'] = encoding_index(enc)
    if enc == 'N64_FLOAT18':
        out['hal_wob'] = float(np.float32(2.5e-6 * 262143.0))
    elif enc in GC_LAYOUTS:
        out['hal_wob'] = float(np.float32(2.5e-6 * 16777215.0))
    else:
        wp = opts.wparams
        near = np.float32(wp.near)
        rng = np.float32(np.float32(wp.far) - near)
        out['hal_wnear'] = float(near)
        out['hal_wrange'] = float(rng)
        if enc == 'W_FIXED':
            bits = min(int(depth_bits), 24)
            steps = float((1 << bits) - 1)
            out['hal_wsteps'] = steps
            out['hal_wob'] = float(np.float32(2.5e-6 * steps))
        else:
            out['hal_wsteps'] = 1.0
            out['hal_wob'] = float(np.float32(2.5e-6 * 4096.0))
    return out


class GBuffer:
    __slots__ = ('width', 'height', 'tri', 'bary', 'bary_lin', 'depth', 'zndc',
                 'overdraw', 'front', 'gpu_alpha', 'gpu_ids_texture',
                 'gpu_sky', 'gpu_sky_why', 'gpu_frame', 'gpu_frame_rgba',
                 'sim_sky', 'zkey', 'cvg')

    def __init__(self, width, height):
        self.width = int(width)
        self.height = int(height)
        self.tri = np.full((height, width), EMPTY, dtype=np.int32)
        self.bary = np.zeros((height, width, 3), dtype=np.float32)
        self.depth = np.full((height, width), np.inf, dtype=np.float32)
        self.zndc = np.full((height, width), 1.0, dtype=np.float32)
        self.front = np.ones((height, width), dtype=bool)
        self.overdraw = np.zeros((height, width), dtype=np.int32)
        self.bary_lin = None
        # the GPU frame's decoded Screen Door alpha (shade_frame /
        # simulate set it under Transparency STIPPLE; render.py reads it)
        self.gpu_alpha = None
        # R249: the ids texture the GPU shading uploaded for this frame,
        # kept so the GPU ink pass reads the same one instead of packing
        # and uploading the G-buffer twice (gpu/shade.shade_frame sets it)
        self.gpu_ids_texture = None
        # R250: the sky pass's verdict for this frame (drawn in the
        # shading burst, or why not), the frame kept on the GPU after
        # shading (a gpu/frame.Resident handle) and its raw readback,
        # and the simulator's sky plane for the suite
        self.gpu_sky = False
        self.gpu_sky_why = ''
        self.gpu_frame = None
        self.gpu_frame_rgba = None
        self.sim_sky = None
        # R251 (C007/C026/C075): the z-buffer's stored CODE under a depth
        # encoding -- exact integers in float32, +inf where empty. The
        # fill compares on it; `depth` / `zndc` keep the DECODED NDC value
        # of the winning code, so every consumer of `depth` stays on NDC.
        # None under LINEAR (the default): every 1.89.0 line runs as it did
        self.zkey = None
        # R251 (C001): the RDP's 3-bit coverage of the winning polygon,
        # uint8 (H, W), 7 = fully covered (uncovered pixels too, so the
        # VI blends silhouettes toward the sky and never touches it).
        # None unless the frame asked for it (`opts.cvg`)
        self.cvg = None

    def alloc_zkey(self):
        """The depth-key plane (R251 depth encodings): +inf everywhere."""
        if self.zkey is None:
            self.zkey = np.full((self.height, self.width), np.inf,
                                dtype=np.float32)
        return self.zkey

    def alloc_cvg(self):
        """The coverage plane (R251 C001): 7 (full) everywhere."""
        if self.cvg is None:
            self.cvg = np.full((self.height, self.width), 7, dtype=np.uint8)
        return self.cvg

    def clear_depth(self, key, dec, enc='LINEAR'):
        """R251 C038: initialise the z-buffer from a per-pixel clear -- the
        DS's rear-plane depth bitmap. `key` / `dec` are `backdrop_clear`'s
        planes (the stored CODE and the decoded NDC value, +inf where the
        bitmap has no depth); `tri` stays EMPTY, so the sky draws those
        pixels and the covered mask is untouched. The fills' strict `<`
        against these buffers and their tie branch (`src_tri < EMPTY`,
        never true) mean a fragment must be strictly nearer than the
        backdrop to draw: THE NAMED TIE RULE -- the backdrop wins an equal
        key -- on both roads (the kernel starts `best_z` at the same key).
        Under a depth encoding the key lands in `zkey`; under LINEAR
        `depth` IS the key, as everywhere."""
        dec = np.asarray(dec, np.float32)
        self.depth[:] = dec
        self.zndc[:] = np.where(np.isfinite(dec), dec, np.float32(1.0))
        if str(enc) != 'LINEAR':
            self.alloc_zkey()[:] = np.asarray(key, np.float32)
        return self

    def alloc_linear(self):
        """Screen-linear barycentrics -- the affine texture warp of the era."""
        if self.bary_lin is None:
            self.bary_lin = np.zeros((self.height, self.width, 3), dtype=np.float32)
        return self.bary_lin

    def mask(self):
        return self.tri >= EMPTY + 1


class FragmentList:
    """Unbounded A-buffer: one entry per transparent fragment."""

    def __init__(self):
        self.px = []
        self.py = []
        self.tri = []
        self.depth = []
        self.bary = []
        self.front = []

    def add(self, px, py, tri, depth, bary, front, bary_lin=None):
        if px.size == 0:
            return
        self.px.append(px.astype(np.int32))
        self.py.append(py.astype(np.int32))
        self.tri.append(np.full(px.size, tri, dtype=np.int32) if np.isscalar(tri) else tri.astype(np.int32))
        self.depth.append(depth.astype(np.float32))
        self.bary.append(bary.astype(np.float32))
        self.front.append(front.astype(bool) if not np.isscalar(front)
                          else np.full(px.size, bool(front)))

    def finish(self):
        if not self.px:
            z = np.zeros(0, np.int32)
            return (z, z, z, np.zeros(0, np.float32), np.zeros((0, 3), np.float32),
                    np.zeros(0, bool))
        return (np.concatenate(self.px), np.concatenate(self.py),
                np.concatenate(self.tri), np.concatenate(self.depth),
                np.concatenate(self.bary), np.concatenate(self.front))

    def __len__(self):
        return int(sum(a.size for a in self.px))


# ---------------------------------------------------------------- transform


def _clip_to_screen(clip, width, height, snap=0.0, pixel_shift=0.0,
                    near_eps=1e-5):
    """Clip-space points -> (screen (V,2), invw (V,), zndc (V,)).

    R251 (C084): the projection's second half, shared by `project` and the
    wire roads' clipped endpoints (clip-space points that must not pass
    through the matrix twice). Snap first, then the pixel-centre shift:
    Direct3D snapped in its own integer-centred frame, so a snapped
    vertex at k lands at k + 0.5, exactly on this raster's pixel centre.
    """
    w = clip[:, 3]
    safe_w = np.where(np.abs(w) < near_eps, near_eps, w)
    invw = (1.0 / safe_w).astype(np.float32)
    ndc = clip[:, :3] * invw[:, None]
    sx = (ndc[:, 0] * 0.5 + 0.5) * width
    sy = (ndc[:, 1] * 0.5 + 0.5) * height
    screen = np.stack([sx, sy], axis=1).astype(np.float32)
    if snap > 0.0:
        screen = np.round(screen / snap) * snap
    if pixel_shift != 0.0:
        screen = screen + np.float32(pixel_shift)
    return screen, invw, ndc[:, 2].astype(np.float32)


def project(verts, mvp, width, height, snap=0.0, near_eps=1e-5,
            pixel_shift=0.0):
    """World verts -> clip space, plus screen coords for the non-clipped case.

    Returns (clip (V,4), screen (V,2), invw (V,), zndc (V,)).
    Vertices behind the eye keep valid clip coords; the clipper deals with them.
    """
    n = verts.shape[0]
    ph = np.empty((n, 4), dtype=np.float32)
    ph[:, :3] = verts
    ph[:, 3] = 1.0
    clip = ph @ np.asarray(mvp, dtype=np.float32).T
    screen, invw, zndc = _clip_to_screen(clip, width, height, snap=snap,
                                         pixel_shift=pixel_shift,
                                         near_eps=near_eps)
    return clip, screen, invw, zndc


#: R251 (C027): the whole-triangle rejection guard band, in half-screens.
#: 6.4 is the GS's geometry (2048 / 320) and the PS1's saturation
#: (1024 / 160) alike, so it is the REJECT item's fixed value, not a dial.
REJECT_GUARD = np.float32(6.4)


def reject_mask(clip, tris, near_eps=1e-5):
    """C027: which triangles the PS2 VU1 / PS1 rule drops WHOLE.

    A triangle with any vertex past the near plane (`z + w < near_eps`,
    the clipper's own test), the far plane (`z > w`) or the 6.4-half-screen
    guard band (the four plane functions written exactly as the guard
    clip writes them) is marked. Returns a (T,) bool; the survivors are
    all-inside, so the near clipper never runs on them. The ONE place the
    six compares live: tests read this mask, never a re-derivation.
    """
    tris = np.asarray(tris, dtype=np.int32)
    cp = clip[tris]                                    # (T,3,4)
    x = cp[:, :, 0]
    y = cp[:, :, 1]
    z = cp[:, :, 2]
    wP = cp[:, :, 3]
    f = z + wP
    G = REJECT_GUARD
    code = (f < near_eps) | (z > wP) | (G * wP - x < 0) | (G * wP + x < 0) \
        | (G * wP - y < 0) | (G * wP + y < 0)
    return code.any(axis=1)


def _clip_poly(cp, cb, coeff, thresh):
    """Sutherland-Hodgman against dot(clip, coeff) >= thresh, one polygon.

    cp: (K,4) clip positions, cb: (K,3) barycentric-over-original weights.
    The near plane is coeff (0,0,1,1); the guard-band planes are
    (±1,0,0,G) and (0,±1,0,G).

    Every edge cut is computed in a CANONICAL direction -- from the
    lexicographically smaller endpoint -- so the two triangles that share
    an edge produce the bit-identical intersection point. They used to
    walk the edge in opposite directions, and the ulp of disagreement in
    clip space was AMPLIFIED by the perspective divide near w -> 0 into
    multi-pixel cracks along every near-clipped shared edge: background
    showing through the seams was the field's "faint wireframe on all
    objects". A crack the projection can widen must not exist at all.
    """
    out_p = []
    out_b = []
    k = len(cp)
    for i in range(k):
        a_p, a_b = cp[i], cb[i]
        b_p, b_b = cp[(i + 1) % k], cb[(i + 1) % k]
        fa = float(a_p @ coeff)
        fb = float(b_p @ coeff)
        ina = fa >= thresh
        inb = fb >= thresh
        if ina:
            out_p.append(a_p)
            out_b.append(a_b)
        if ina != inb:
            denom = fa - fb
            if abs(denom) < 1e-12:
                continue
            if tuple(b_p) < tuple(a_p):
                # canonical: interpolate from the smaller endpoint, so
                # both triangles at this edge do the SAME arithmetic
                t = fb / (fb - fa)
                out_p.append(b_p + (a_p - b_p) * t)
                out_b.append(b_b + (a_b - b_b) * t)
            else:
                t = fa / denom
                out_p.append(a_p + (b_p - a_p) * t)
                out_b.append(a_b + (b_b - a_b) * t)
    return out_p, out_b


_NEAR_COEFF = np.array([0.0, 0.0, 1.0, 1.0], np.float32)

#: the guard band, in NDC units: triangles are clipped so no vertex lands
#: beyond GUARD screens from the viewport. Without it, a triangle that
#: barely survives the near clip projects to coordinates in the hundreds
#: of thousands of pixels, and the float32 edge functions lose so much
#: precision to cancellation that interior pixels misclassify -- holes
#: along every horizon-region edge, which the field photographed as "a
#: faint wireframe on all objects". Every period rasteriser guard-band
#: clipped for exactly this reason. The guard edges land 1.5+ screens
#: outside the viewport, so visible coverage only ever CORRECTS.
GUARD_BAND = np.float32(4.0)

_GUARD_COEFFS = (np.array([-1.0, 0.0, 0.0, GUARD_BAND], np.float32),
                 np.array([1.0, 0.0, 0.0, GUARD_BAND], np.float32),
                 np.array([0.0, -1.0, 0.0, GUARD_BAND], np.float32),
                 np.array([0.0, 1.0, 0.0, GUARD_BAND], np.float32))


def _clip_near(cp, cb, near_eps):
    """The near clip, as it always was: z >= -w plus the epsilon."""
    return _clip_poly(cp, cb, _NEAR_COEFF, near_eps)


IDENT_BARY = np.eye(3, dtype=np.float32)


def _subdivide_screen_tris(sx, sy, iw, z, bw, src, limit_px, max_passes=6):
    """Affine-correction subdivision: 4-way split until edges fit the cap.

    The PS1 warp gets less wrong when triangles are smaller -- period
    engines subdivided for exactly this reason, and `tex_affine_subdiv`
    is that dial: the maximum screen-space edge length, in pixels,
    before a triangle splits. Splitting AFTER projection is exact for
    every rasteriser input: screen position, 1/w, ndc z and the
    original-triangle weights `bw` are all screen-affine, so midpoints
    are plain averages and the perspective-correct barycentrics
    reconstructed from any sub-triangle are the original triangle's
    own. Only the screen-LINEAR interpolation -- the warp itself --
    changes, which is the entire point. Deterministic and shared: both
    rasterisers receive the same emitted list in the same order.
    """
    lim2 = np.float32(float(limit_px) * float(limit_px))
    half = np.float32(0.5)
    for _ in range(int(max_passes)):
        e01 = (sx[:, 0] - sx[:, 1]) ** 2 + (sy[:, 0] - sy[:, 1]) ** 2
        e12 = (sx[:, 1] - sx[:, 2]) ** 2 + (sy[:, 1] - sy[:, 2]) ** 2
        e20 = (sx[:, 2] - sx[:, 0]) ** 2 + (sy[:, 2] - sy[:, 0]) ** 2
        big = np.maximum(np.maximum(e01, e12), e20) > lim2
        if not big.any():
            break

        def _split(arr):
            # arr (E,3,...) -> midpoints m01,m12,m20 and the 4 children
            a, b, c = arr[big, 0], arr[big, 1], arr[big, 2]
            m01 = (a + b) * half
            m12 = (b + c) * half
            m20 = (c + a) * half
            kids = np.stack([
                np.stack([a, m01, m20], axis=1),
                np.stack([m01, b, m12], axis=1),
                np.stack([m20, m12, c], axis=1),
                np.stack([m01, m12, m20], axis=1),
            ], axis=1)                       # (Nbig, 4, 3, ...)
            return kids.reshape((-1,) + arr.shape[1:])

        stay = ~big
        parts = []
        for arr in (sx, sy, iw, z, bw):
            parts.append(np.concatenate([arr[stay], _split(arr)], axis=0))
        sx, sy, iw, z, bw = [p.astype(np.float32) for p in parts]
        src = np.concatenate([src[stay], np.repeat(src[big], 4)], axis=0)
    return sx, sy, iw, z, bw, src.astype(np.int32)


def build_screen_tris(clip, tris, width, height, snap=0.0, near_eps=1e-5,
                      depth_bits=24, subdiv_px=0, pixel_shift=0.0,
                      reject=False, size_limit=None):
    """Clip + project every triangle. Returns flat arrays ready for filling.

    Output arrays are per *emitted* triangle (clipping can create more than one):
      sx,sy  (E,3) screen coords
      iw     (E,3) 1/w
      z      (E,3) ndc z
      bw     (E,3,3) barycentric weight of each emitted corner over the original
      src    (E,)  index into `tris`

    `subdiv_px` > 0 turns on affine-correction subdivision (see
    _subdivide_screen_tris): triangles split until no screen edge
    exceeds that many pixels. Applied AFTER vertex snapping, so the
    PS1 combo subdivides the snapped geometry, as the era did.

    R251: `pixel_shift` (C084) moves the projected vertices AFTER the snap
    and BEFORE the subdivision; `reject` (C027) drops whole triangles by
    `reject_mask` before the near clipper; `size_limit` (C004, a
    (wmax, hmax) in internal pixels) drops every SOURCE polygon whose
    projected extent exceeds it, before the subdivision. The dead
    `clip_far` kwarg is retired: `reject` carries the far test as the
    machines did.
    """
    tris = np.asarray(tris, dtype=np.int32)
    src_ids = None
    if reject:
        drop = reject_mask(clip, tris, near_eps=near_eps)
        if drop.any():
            src_ids = np.nonzero(~drop)[0].astype(np.int32)
            tris = tris[~drop]
    cp = clip[tris]                                    # (T,3,4)
    f = cp[:, :, 2] + cp[:, :, 3]                      # near plane function
    inside = f >= near_eps
    n_in = inside.sum(axis=1)
    all_in = n_in == 3
    straddle = (n_in > 0) & ~all_in

    idx_in = np.nonzero(all_in)[0]
    parts_p = [cp[idx_in]]
    # a broadcast VIEW, not a copy: for an all-inside mesh this was a
    # (T,3,3) materialised identity per rasterisation -- 30 MB of writes
    # at 800k triangles, for values np.concatenate materialises anyway
    parts_b = [np.broadcast_to(IDENT_BARY, (idx_in.size, 3, 3))]
    parts_src = [idx_in.astype(np.int32)]

    for t in np.nonzero(straddle)[0]:
        poly_p, poly_b = _clip_near(list(cp[t]), list(IDENT_BARY), near_eps)
        if len(poly_p) < 3:
            continue
        for j in range(1, len(poly_p) - 1):
            parts_p.append(np.stack([poly_p[0], poly_p[j], poly_p[j + 1]])[None])
            parts_b.append(np.stack([poly_b[0], poly_b[j], poly_b[j + 1]])[None])
            parts_src.append(np.array([t], dtype=np.int32))

    if len(parts_p) == 1 and idx_in.size == 0:
        e = np.zeros((0, 3), np.float32)
        return (e, e.copy(), e.copy(), e.copy(),
                np.zeros((0, 3, 3), np.float32), np.zeros(0, np.int32))

    def _one_or_cat(parts, want):
        # the high-poly common case is NO straddlers: one part, already
        # the right dtype -- concatenate+astype was two full copies of
        # a 40 MB array for nothing. Values identical either way.
        if len(parts) == 1:
            a = parts[0]
            return a if a.dtype == want else a.astype(want)
        return np.concatenate(parts, axis=0).astype(want)

    P = _one_or_cat(parts_p, np.float32)                     # (E,3,4)
    B = _one_or_cat(parts_b, np.float32)                     # (E,3,3)
    S = parts_src[0] if len(parts_src) == 1 else np.concatenate(parts_src,
                                                                axis=0)

    # ---- the guard band: clip anything reaching past GUARD_BAND screens
    # (see the constant above for why; the common all-on-screen mesh
    # never enters this branch)
    wP = P[:, :, 3]
    # the four plane functions, homogeneous: f < 0 = outside that plane.
    # A triangle whose three vertices are outside the SAME plane is
    # outside it everywhere (f is linear), so its projection can never
    # reach the screen: CULLED, not clipped. The field's close-up put
    # thousands of beside-the-camera triangles into the python clip loop
    # (clip 128 -> 1392 ms); nearly all of them die here in one
    # vectorised test instead
    f_xp = GUARD_BAND * wP - P[:, :, 0]
    f_xn = GUARD_BAND * wP + P[:, :, 0]
    f_yp = GUARD_BAND * wP - P[:, :, 1]
    f_yn = GUARD_BAND * wP + P[:, :, 1]
    dead = ((f_xp < 0).all(axis=1) | (f_xn < 0).all(axis=1)
            | (f_yp < 0).all(axis=1) | (f_yn < 0).all(axis=1))
    touches = ((f_xp < 0) | (f_xn < 0) | (f_yp < 0)
               | (f_yn < 0)).any(axis=1)
    need = touches & ~dead
    if dead.any() and not need.any():
        keep = ~dead
        P, B, S = P[keep], B[keep], S[keep]
    if need.any():
        keep = ~need & ~dead
        g_p = [P[keep]]
        g_b = [B[keep]]
        g_s = [S[keep]]
        for i in np.nonzero(need)[0]:
            poly_p, poly_b = list(P[i]), list(B[i])
            for coeff in _GUARD_COEFFS:
                poly_p, poly_b = _clip_poly(poly_p, poly_b, coeff, 0.0)
                if len(poly_p) < 3:
                    break
            if len(poly_p) < 3:
                continue
            for j in range(1, len(poly_p) - 1):
                g_p.append(np.stack([poly_p[0], poly_p[j],
                                     poly_p[j + 1]])[None])
                g_b.append(np.stack([poly_b[0], poly_b[j],
                                     poly_b[j + 1]])[None])
                g_s.append(S[i:i + 1])
        P = np.concatenate(g_p, axis=0).astype(np.float32)
        B = np.concatenate(g_b, axis=0).astype(np.float32)
        S = np.concatenate(g_s, axis=0).astype(np.int32)

    w = P[:, :, 3]
    w = np.where(np.abs(w) < near_eps, near_eps, w)
    iw = 1.0 / w                    # float32 in, float32 out -- no copy
    ndc = P[:, :, :3] * iw[:, :, None]
    sx = (ndc[:, :, 0] * 0.5 + 0.5) * width
    sy = (ndc[:, :, 1] * 0.5 + 0.5) * height
    # sx/sy/iw are fresh contiguous arrays from the arithmetic; z alone
    # would be a strided VIEW into ndc, which every downstream gather
    # pays for -- one contiguous copy beats many strided reads
    z = np.ascontiguousarray(ndc[:, :, 2])
    if sx.dtype != np.float32:      # belt and braces for exotic inputs
        sx = sx.astype(np.float32)
        sy = sy.astype(np.float32)
        z = z.astype(np.float32)
        iw = iw.astype(np.float32)
    if snap > 0.0:
        sx = np.round(sx / snap) * snap
        sy = np.round(sy / snap) * snap
    if pixel_shift != 0.0:
        # C084: Direct3D 3-9 sampled at the integer corner. Shifting the
        # vertices by +0.5 is the same picture as sampling at (x, y)
        # instead of (x + 0.5, y + 0.5): the grid moves, the geometry
        # does not, and the interpolants move with the coverage. After
        # the snap (D3D snapped in its own integer-centred frame), before
        # the subdivision. float32 add, exact at pixel scale
        sx = sx + np.float32(pixel_shift)
        sy = sy + np.float32(pixel_shift)
    if size_limit is not None and sx.shape[0]:
        # C004: the PS1 GPU never drew a polygon past 1023 x 511 pixels.
        # Measured per SOURCE polygon (the guard clip may have cut one
        # into several rows: the union bbox over every row sharing an S),
        # BEFORE the subdivision so a big triangle cannot dodge it by
        # splitting. Order-free reductions
        wmax, hmax = (np.float32(size_limit[0]), np.float32(size_limit[1]))
        T = int(S.max()) + 1
        xmin = np.full(T, np.inf, np.float32)
        xmax = np.full(T, -np.inf, np.float32)
        ymin = np.full(T, np.inf, np.float32)
        ymax = np.full(T, -np.inf, np.float32)
        np.minimum.at(xmin, S, sx.min(1))
        np.maximum.at(xmax, S, sx.max(1))
        np.minimum.at(ymin, S, sy.min(1))
        np.maximum.at(ymax, S, sy.max(1))
        big_src = (xmax - xmin > wmax) | (ymax - ymin > hmax)
        if big_src.any():
            keep = ~big_src[S]
            sx, sy, iw, z, B, S = (a[keep] for a in (sx, sy, iw, z, B, S))
            sx = np.ascontiguousarray(sx)
            sy = np.ascontiguousarray(sy)
            iw = np.ascontiguousarray(iw)
            z = np.ascontiguousarray(z)
            B = np.ascontiguousarray(B)
            S = np.ascontiguousarray(S)
    if subdiv_px and int(subdiv_px) > 0:
        sx, sy, iw, z, B, S = _subdivide_screen_tris(
            sx, sy, iw, z, B, S, max(int(subdiv_px), 4))
    if src_ids is not None:
        # C027: the ids stay the ORIGINAL triangle indices
        S = src_ids[S]
    # depth_bits is accepted (and threaded to the fillers by rasterize)
    # but NOT applied here any more. Quantizing the VERTEX z and then
    # interpolating tilted every triangle's whole depth plane by up to
    # half a step -- two close surfaces' planes then CROSS, and a
    # low-bit z-buffer showed big solid wedges punching through faces
    # instead of the thin dithered bands real hardware showed. Period
    # hardware interpolated depth at full precision and rounded PER
    # PIXEL at the buffer; quantize_depth() in the fillers does exactly
    # that now, on both rasterisers.
    return sx, sy, iw, z, B, S


# ------------------------------------------------------------------- filling


def fill(gbuf, sx, sy, iw, z, bw, src, cull='NONE', frags=None, flat_depth=None,
         depth_write=True, depth_test=True, count_overdraw=False,
         tri_offset=0, z_offset=0.0, tri_map=None, depth_bits=32, frag_test=None,
         opts=None, jit=None):
    """Fill emitted triangles into a GBuffer (and/or a FragmentList).

    cull: 'NONE' | 'BACK' | 'FRONT'
    frags: FragmentList to append to instead of / as well as writing the gbuf.
    opts: R251 RasterOpts; under a non-LINEAR `opts.enc` the depth test
    runs on the encoding's CODE (`gbuf.zkey`, the named tie rule on it)
    and `depth`/`zndc` receive the decoded value. LINEAR keeps the
    1.89.0 statements in their 1.89.0 order.
    jit: R251 C127, the (JX, JY) sample offset planes of `jitter_planes`
    (None = the subpixel centres); the pixel INDEX stays the integer cell.
    """
    W, H = gbuf.width, gbuf.height
    n = sx.shape[0]
    if n == 0:
        return 0
    written = 0
    enc_on = opts is not None and str(opts.enc) != 'LINEAR'
    zkey_b = gbuf.alloc_zkey() if enc_on else None
    JX = JY = None
    if jit is not None:
        JX, JY = jit
    # C001: the coverage plane, written for the winners when the frame
    # asked for it (rasterize allocates it under opts.cvg)
    cvg_b = gbuf.cvg if (opts is not None and getattr(opts, 'cvg', False)
                         and depth_write) else None

    x0, x1, x2 = sx[:, 0], sx[:, 1], sx[:, 2]
    y0, y1, y2 = sy[:, 0], sy[:, 1], sy[:, 2]
    area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)

    live = np.abs(area) > 1e-9
    if cull == 'BACK':
        live &= area > 0.0          # CCW front faces, y-up screen. See below.
    elif cull == 'FRONT':
        live &= area < 0.0

    bxmin = np.maximum(np.floor(np.minimum(np.minimum(x0, x1), x2)).astype(np.int32), 0)
    bxmax = np.minimum(np.ceil(np.maximum(np.maximum(x0, x1), x2)).astype(np.int32), W - 1)
    bymin = np.maximum(np.floor(np.minimum(np.minimum(y0, y1), y2)).astype(np.int32), 0)
    bymax = np.minimum(np.ceil(np.maximum(np.maximum(y0, y1), y2)).astype(np.int32), H - 1)
    live &= (bxmax >= bxmin) & (bymax >= bymin)

    order = np.nonzero(live)[0]
    depth = gbuf.depth
    tri_b = gbuf.tri
    bary_b = gbuf.bary
    zndc_b = gbuf.zndc
    front_b = gbuf.front

    for t in order:
        xa, xb, xc = x0[t], x1[t], x2[t]
        ya, yb, yc = y0[t], y1[t], y2[t]
        ar = area[t]
        inv_area = 1.0 / ar
        xs = np.arange(bxmin[t], bxmax[t] + 1, dtype=np.float32) + 0.5
        ys = np.arange(bymin[t], bymax[t] + 1, dtype=np.float32) + 0.5
        X = xs[None, :]
        Y = ys[:, None]
        if JX is not None:
            # C127: the centre first, then the jitter -- the order the
            # kernel keeps (X = pix.x + jit.x)
            X = X + JX[bymin[t]:bymax[t] + 1, bxmin[t]:bxmax[t] + 1]
            Y = Y + JY[bymin[t]:bymax[t] + 1, bxmin[t]:bxmax[t] + 1]

        e0 = (xc - xb) * (Y - yb) - (yc - yb) * (X - xb)
        e1 = (xa - xc) * (Y - yc) - (ya - yc) * (X - xc)
        e2 = (xb - xa) * (Y - ya) - (yb - ya) * (X - xa)
        # WATERTIGHT coverage: the two triangles at a shared edge compute
        # that edge with DIFFERENT float expressions, so both can land a
        # few ulps below zero and both exclude the pixel -- background
        # bleeding through every dense-mesh edge was the field's "faint
        # wireframe on all objects". Coverage therefore widens by the
        # wobble window the referral already established (2.5e-7 of the
        # product magnitudes -- ulps of the arithmetic, R104's law);
        # overlapped edge pixels resolve by depth and the named tie rule
        w0 = np.abs((xc - xb) * (Y - yb)) + np.abs((yc - yb) * (X - xb))
        w1 = np.abs((xa - xc) * (Y - yc)) + np.abs((ya - yc) * (X - xc))
        w2 = np.abs((xb - xa) * (Y - ya)) + np.abs((yb - ya) * (X - xa))
        if ar > 0:
            inside = (e0 >= EDGE_WOBBLE * -w0) & (e1 >= EDGE_WOBBLE * -w1) \
                & (e2 >= EDGE_WOBBLE * -w2)
        else:
            inside = (e0 <= EDGE_WOBBLE * w0) & (e1 <= EDGE_WOBBLE * w1) \
                & (e2 <= EDGE_WOBBLE * w2)
        if not inside.any():
            continue

        yy, xx = np.nonzero(inside)
        l0 = e0[yy, xx] * inv_area
        l1 = e1[yy, xx] * inv_area
        l2 = e2[yy, xx] * inv_area

        src_tri = int(src[t]) + tri_offset
        if tri_map is not None:
            src_tri = int(tri_map[int(src[t])])
        if flat_depth is not None:
            # Painter's algorithm: the whole polygon carries one depth, so the
            # depth test decides between polygons rather than between fragments
            zz = np.full(l0.shape, float(flat_depth[src_tri]) + z_offset,
                         np.float32)
        else:
            zz = (l0 * z[t, 0] + l1 * z[t, 1] + l2 * z[t, 2]) + z_offset
        key = None
        if enc_on:
            # R251: the encoding reads 1/w (the W items) -- the exact
            # expression of the perspective divide below, computed
            # before the test; elementwise identical wherever it runs
            invw_e = l0 * iw[t, 0] + l1 * iw[t, 1] + l2 * iw[t, 2]
            invw_e = np.where(np.abs(invw_e) < 1e-20, 1e-20, invw_e)
            key, zz = encode_depth(zz, invw_e, opts, depth_bits)
        elif depth_bits < 32:
            zz = quantize_depth(zz, depth_bits)
        px = xx + bxmin[t]
        py = yy + bymin[t]

        if depth_test:
            if frags is not None:
                # A-buffer collection: the tolerant limit, so a modeled
                # contact survives whichever rasteriser wrote the depth
                keep = zz < abuf_depth_limit(depth[py, px])
            elif key is not None:
                # R251: the named tie rule on the CODE
                keep = (key < zkey_b[py, px]) | \
                    ((key == zkey_b[py, px]) & (src_tri < tri_b[py, px]))
            else:
                # THE NAMED TIE RULE: equal depth goes to the LOWEST
                # triangle id. Exact ties are common the moment depth is
                # quantised (coincident contacts land on shared steps),
                # and the old strict `<` left the winner to whichever
                # ORDER a code path happened to test in -- this loop
                # tested in submission order while fill_batched drew
                # big triangles first and size-bucketed the rest, and
                # the two disagreed on 3 pixels of the demo scene at 16
                # bits. A rule keyed on the triangle ID is order-free:
                # every path (this loop, the batched resolve, the
                # compute kernel, the referral replay) lands the same
                # winner whatever it tested first.
                keep = (zz < depth[py, px]) | \
                    ((zz == depth[py, px]) & (src_tri < tri_b[py, px]))
            if not keep.any():
                continue
            if not keep.all():
                px, py, l0, l1, l2, zz = px[keep], py[keep], l0[keep], l1[keep], l2[keep], zz[keep]
                if key is not None:
                    key = key[keep]

        iw0, iw1, iw2 = iw[t, 0], iw[t, 1], iw[t, 2]
        invw = l0 * iw0 + l1 * iw1 + l2 * iw2
        invw = np.where(np.abs(invw) < 1e-20, 1e-20, invw)
        p0 = l0 * iw0 / invw
        p1 = l1 * iw1 / invw
        p2 = l2 * iw2 / invw
        b = (p0[:, None] * bw[t, 0][None, :] +
             p1[:, None] * bw[t, 1][None, :] +
             p2[:, None] * bw[t, 2][None, :])

        src_tri = int(src[t]) + tri_offset
        if tri_map is not None:
            src_tri = int(tri_map[int(src[t])])
        is_front = ar < 0.0

        if frag_test is not None:
            # R212 punch-through: the alpha test lives INSIDE the
            # raster, exactly where the era's hardware ran it -- a
            # fragment that fails never touches the depth buffer, and
            # one behind an already-written solid never gets evaluated
            keep_a = frag_test(
                np.full(px.size, src_tri, np.int32), px, py, b,
                np.full(px.size, is_front, bool))
            if not keep_a.any():
                continue
            if not keep_a.all():
                px, py, zz, b = (px[keep_a], py[keep_a], zz[keep_a],
                                 b[keep_a])
                l0, l1, l2 = l0[keep_a], l1[keep_a], l2[keep_a]
                if key is not None:
                    key = key[keep_a]

        if count_overdraw:
            np.add.at(gbuf.overdraw, (py, px), 1)

        if frags is not None:
            frags.add(px, py, src_tri, zz, b, is_front)
        if depth_write:
            if key is not None:
                zkey_b[py, px] = key
            depth[py, px] = zz
            tri_b[py, px] = src_tri
            bary_b[py, px] = b
            if cvg_b is not None:
                # C001: the RDP's coverage of the WINNER at the pixel
                # centre (order-free: the winning triangle and the pixel)
                cvg_b[py, px] = coverage16(
                    xa, ya, xb, yb, xc, yc, ar,
                    px.astype(np.float32) + np.float32(0.5),
                    py.astype(np.float32) + np.float32(0.5))
            if gbuf.bary_lin is not None:
                lb = (l0[:, None] * bw[t, 0][None, :] +
                      l1[:, None] * bw[t, 1][None, :] +
                      l2[:, None] * bw[t, 2][None, :])
                gbuf.bary_lin[py, px] = lb
            zndc_b[py, px] = zz
            front_b[py, px] = is_front
        written += px.size
    return written


BATCH_MIN_TRIS = 24          # below this the loop wins; setup dominates


def rasterize(verts, tris, mvp, width, height, cull='NONE', snap=0.0,
              depth_bits=24, subset=None, gbuf=None, frags=None,
              depth_write=True, depth_test=True, count_overdraw=False,
              z_offset=0.0, near_eps=1e-5, batched=None, flat_depth=None,
              scissor=None, subdiv_px=0, frag_test=None, opts=None):
    """Convenience: project + clip + fill in one call.

    `batched` selects the loop-free rasteriser; None picks automatically. The
    reference per-triangle path is kept because it is the simpler code and the
    batched one is validated against it in the test suite.

    `scissor` is a (y0, y1) row range. Triangles that fall entirely outside it
    are dropped before filling. That is what makes splitting a frame across
    processes worth doing: without it every worker rasterises the whole mesh
    for its own slice, and sixty slices means sixty rasterisations.

    `opts` (R251) is a RasterOpts; None means the 1.89.0 behaviour (the
    SSS, UV-tile, shadow-bake and volume rasters never pass one).
    """
    if gbuf is None:
        gbuf = GBuffer(width, height)
    if opts is None:
        opts = RasterOpts()
    if str(opts.enc) != 'LINEAR':
        gbuf.alloc_zkey()
    # C127: the jittered sample positions, once per call (None = centres)
    jit = jitter_planes(opts, width, height)
    if getattr(opts, 'cvg', False):
        gbuf.alloc_cvg()            # C001: the RDP's coverage plane
    tri_map = None
    if subset is not None:
        tri_map = np.asarray(subset, dtype=np.int32)
        tris = tris[tri_map]
    clip, _, _, _ = project(verts, mvp, width, height, snap=0.0, near_eps=near_eps)

    if scissor is not None and tris.shape[0]:
        # Drop triangles outside the band *before* clipping, not after. Clipping
        # every triangle in every band is the cost that made splitting a frame
        # across processes lose to not splitting it. Only triangles wholly in
        # front of the near plane can be judged this cheaply; any that straddle
        # it are kept and sorted out by the clipper as usual.
        y0s, y1s = scissor
        w = clip[:, 3]
        tw = w[tris]
        infront = (tw > near_eps).all(axis=1)
        if infront.any():
            ndc_y = clip[:, 1] / np.where(np.abs(w) < near_eps, near_eps, w)
            sy_all = (ndc_y * 0.5 + 0.5) * height
            ty = sy_all[tris]
            lo = ty.min(axis=1)
            hi = ty.max(axis=1)
            outside = infront & ((hi < y0s - 1.0) | (lo > y1s + 1.0))
            if outside.any():
                keep_tris = ~outside
                tris = tris[keep_tris]
                if tri_map is not None:
                    tri_map = tri_map[keep_tris]
                elif subset is None:
                    tri_map = np.nonzero(keep_tris)[0].astype(np.int32)

    sx, sy, iw, z, bw, src = build_screen_tris(clip, tris, width, height, snap=snap,
                                               near_eps=near_eps, depth_bits=depth_bits,
                                               subdiv_px=subdiv_px,
                                               pixel_shift=opts.pixel_shift,
                                               reject=opts.reject,
                                               size_limit=opts.size_limit)
    if scissor is not None and sx.shape[0]:
        y0, y1 = scissor
        lo = sy.min(axis=1)
        hi = sy.max(axis=1)
        keep = (hi >= y0) & (lo < y1)
        if not keep.all():
            sx, sy, iw, z, bw, src = (a[keep] for a in (sx, sy, iw, z, bw, src))
    if batched is None:
        # overdraw counting needs the sequential semantics to stay exact
        batched = (sx.shape[0] >= BATCH_MIN_TRIS) and not count_overdraw
    if batched:
        fill_batched(gbuf, sx, sy, iw, z, bw, src, cull=cull, frags=frags,
                     flat_depth=flat_depth, depth_write=depth_write,
                     depth_test=depth_test, z_offset=z_offset, tri_map=tri_map,
                     depth_bits=depth_bits, frag_test=frag_test, opts=opts,
                     jit=jit)
    else:
        fill(gbuf, sx, sy, iw, z, bw, src, cull=cull, frags=frags,
             flat_depth=flat_depth, depth_write=depth_write,
             depth_test=depth_test, count_overdraw=count_overdraw,
             z_offset=z_offset, tri_map=tri_map, depth_bits=depth_bits, frag_test=frag_test,
             opts=opts, jit=jit)
    return gbuf


# ------------------------------------------------------- attribute fetching


def fetch(attr, tris, tri_idx, bary):
    """Interpolate a per-vertex attribute at shaded fragments.

    attr: (V,C) or (V,)   tris: (T,3)   tri_idx: (N,)   bary: (N,3)
    """
    idx = tris[tri_idx]                       # (N,3)
    a = attr[idx]                             # (N,3[,C])
    if a.ndim == 2:
        return (a * bary).sum(axis=1)
    return (a * bary[:, :, None]).sum(axis=1)


def fetch_face(attr, tri_idx):
    return attr[tri_idx]


def screen_derivatives(image, valid, tri_id):
    """Finite-difference derivatives that respect triangle boundaries.

    image: (H,W,C) attribute laid out in screen space.
    Returns (ddx, ddy) with the same shape.
    """
    ddx = np.zeros_like(image)
    ddy = np.zeros_like(image)
    same_x = np.zeros(tri_id.shape, dtype=bool)
    same_x[:, :-1] = (tri_id[:, :-1] == tri_id[:, 1:]) & valid[:, :-1] & valid[:, 1:]
    same_y = np.zeros(tri_id.shape, dtype=bool)
    same_y[:-1, :] = (tri_id[:-1, :] == tri_id[1:, :]) & valid[:-1, :] & valid[1:, :]

    fwd_x = np.zeros_like(image)
    fwd_x[:, :-1] = image[:, 1:] - image[:, :-1]
    fwd_y = np.zeros_like(image)
    fwd_y[:-1, :] = image[1:, :] - image[:-1, :]

    ddx[same_x] = fwd_x[same_x]
    ddy[same_y] = fwd_y[same_y]
    # backward difference where the forward neighbour was a different triangle
    bx = ~same_x & valid
    bx[:, 1:] &= (tri_id[:, 1:] == tri_id[:, :-1]) & valid[:, :-1]
    bx[:, 0] = False
    ddx[bx] = fwd_x[np.roll(bx, -1, axis=1)]
    by = ~same_y & valid
    by[1:, :] &= (tri_id[1:, :] == tri_id[:-1, :]) & valid[:-1, :]
    by[0, :] = False
    ddy[by] = fwd_y[np.roll(by, -1, axis=0)]
    return ddx, ddy


# ------------------------------------------------------- batched rasteriser

def _size_classes(span):
    """Round each triangle's bounding box up to a power of two."""
    span = np.maximum(span, 1)
    return (1 << np.ceil(np.log2(span.astype(np.float64))).astype(np.int64))


_GRID_CACHE = {}


def _grid(h, w):
    key = (h, w)
    g = _GRID_CACHE.get(key)
    if g is None:
        oy, ox = np.mgrid[0:h, 0:w]
        g = (oy[None, :, :].astype(np.int64), ox[None, :, :].astype(np.int64))
        if len(_GRID_CACHE) < 128:
            _GRID_CACHE[key] = g
    return g


LARGE_TRI_PX = 16384         # a 128x128 box; above this the loop amortises fine


def fill_batched(gbuf, sx, sy, iw, z, bw, src, cull='NONE', frags=None,
                 flat_depth=None, depth_write=True, depth_test=True, tri_offset=0,
                 z_offset=0.0, tri_map=None, max_batch_px=4_000_000,
                 depth_bits=32, frag_test=None, opts=None, jit=None):
    """Same result as fill(), without the per-triangle Python loop.

    Small triangles are bucketed by bounding-box size class -- separately in
    width and height, so a long thin triangle is not padded out to a square --
    and every candidate pixel in a bucket is tested in one vectorised sweep.
    Triangles with large boxes are handed to the sequential path instead, where
    the per-triangle overhead is amortised by the pixel count and the batched
    path's padding would be pure waste.

    The z-resolve sorts fragments per pixel and takes the nearest, which picks
    the same winner a sequential depth test would: a fragment only survives if
    it beats both the existing buffer and every other fragment on its pixel.

    Verified bit-identical to fill() by tests/test_render.py.
    """
    W, H = gbuf.width, gbuf.height
    n = sx.shape[0]
    if n == 0:
        return 0
    enc_on = opts is not None and str(opts.enc) != 'LINEAR'
    zkey_b = gbuf.alloc_zkey() if enc_on else None
    JX = JY = None
    if jit is not None:
        JX, JY = jit

    x0, x1, x2 = sx[:, 0], sx[:, 1], sx[:, 2]
    y0, y1, y2 = sy[:, 0], sy[:, 1], sy[:, 2]
    area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)

    live = np.abs(area) > 1e-9
    if cull == 'BACK':
        live &= area > 0.0
    elif cull == 'FRONT':
        live &= area < 0.0

    bxmin = np.maximum(np.floor(np.minimum(np.minimum(x0, x1), x2)), 0).astype(np.int64)
    bxmax = np.minimum(np.ceil(np.maximum(np.maximum(x0, x1), x2)), W - 1).astype(np.int64)
    bymin = np.maximum(np.floor(np.minimum(np.minimum(y0, y1), y2)), 0).astype(np.int64)
    bymax = np.minimum(np.ceil(np.maximum(np.maximum(y0, y1), y2)), H - 1).astype(np.int64)
    live &= (bxmax >= bxmin) & (bymax >= bymin)

    bw_px = bxmax - bxmin + 1
    bh_px = bymax - bymin + 1

    big = live & ((bw_px * bh_px) > LARGE_TRI_PX)
    written = 0
    if big.any():
        sel = np.nonzero(big)[0]
        written += fill(gbuf, sx[sel], sy[sel], iw[sel], z[sel], bw[sel], src[sel],
                        cull=cull, frags=frags, flat_depth=flat_depth,
                        depth_write=depth_write,
                        depth_test=depth_test, tri_offset=tri_offset,
                        z_offset=z_offset, tri_map=tri_map,
                        depth_bits=depth_bits, frag_test=frag_test,
                        opts=opts, jit=jit)

    idx_all = np.nonzero(live & ~big)[0]
    if idx_all.size == 0:
        return written
    if idx_all.size < BATCH_MIN_TRIS:
        # too few small triangles to pay for bucketing -- this is the common
        # case at high supersampling, where everything is large in pixels
        return written + fill(
            gbuf, sx[idx_all], sy[idx_all], iw[idx_all], z[idx_all],
            bw[idx_all], src[idx_all], cull=cull, frags=frags,
            depth_write=depth_write, depth_test=depth_test,
            tri_offset=tri_offset, z_offset=z_offset, tri_map=tri_map,
            depth_bits=depth_bits, frag_test=frag_test, opts=opts, jit=jit)

    wc = _size_classes(bw_px[idx_all])
    hc = _size_classes(bh_px[idx_all])
    key = wc * 65536 + hc

    px_all, py_all, zz_all, b_all, blin_all = [], [], [], [], []
    tri_all, front_all = [], []
    zk_all = []
    e_all = []                      # C001: the emitted triangle per fragment
    cvg_b = gbuf.cvg if (opts is not None and getattr(opts, 'cvg', False)
                         and depth_write) else None

    for k in np.unique(key):
        members = idx_all[key == k]
        SW = int(k // 65536)
        SH = int(k % 65536)
        per = max(int(max_batch_px // max(SW * SH, 1)), 1)
        for start in range(0, members.size, per):
            t = members[start:start + per]
            oy, ox = _grid(SH, SW)
            bx = bxmin[t][:, None, None]
            by = bymin[t][:, None, None]
            X = (bx + ox).astype(np.float32) + 0.5
            Y = (by + oy).astype(np.float32) + 0.5
            if JX is not None:
                # C127: the offsets gathered at the padded grid's cells
                # (indices clipped; out-of-box lanes are masked below)
                gy = np.minimum(by + oy, H - 1)
                gx = np.minimum(bx + ox, W - 1)
                X = X + JX[gy, gx]
                Y = Y + JY[gy, gx]

            xa, xb, xc = x0[t][:, None, None], x1[t][:, None, None], x2[t][:, None, None]
            ya, yb, yc = y0[t][:, None, None], y1[t][:, None, None], y2[t][:, None, None]
            e0 = (xc - xb) * (Y - yb) - (yc - yb) * (X - xb)
            e1 = (xa - xc) * (Y - yc) - (ya - yc) * (X - xc)
            e2 = (xb - xa) * (Y - ya) - (yb - ya) * (X - xa)
            # the watertight window, exactly as the loop fill and the
            # kernel compute it (see fill() for the story)
            w0 = np.abs((xc - xb) * (Y - yb)) + np.abs((yc - yb) * (X - xb))
            w1 = np.abs((xa - xc) * (Y - yc)) + np.abs((ya - yc) * (X - xc))
            w2 = np.abs((xb - xa) * (Y - ya)) + np.abs((yb - ya) * (X - xa))
            pos = (area[t] > 0)[:, None, None]
            inside = np.where(
                pos,
                (e0 >= EDGE_WOBBLE * -w0) & (e1 >= EDGE_WOBBLE * -w1)
                & (e2 >= EDGE_WOBBLE * -w2),
                (e0 <= EDGE_WOBBLE * w0) & (e1 <= EDGE_WOBBLE * w1)
                & (e2 <= EDGE_WOBBLE * w2))
            inside &= (ox < bw_px[t][:, None, None]) & (oy < bh_px[t][:, None, None])
            if not inside.any():
                continue

            ti, yy, xx = np.nonzero(inside)
            e_t = t[ti]                         # C001: emitted index per fragment
            inv_area = (1.0 / area[t])[ti]
            l0 = e0[ti, yy, xx] * inv_area
            l1 = e1[ti, yy, xx] * inv_area
            l2 = e2[ti, yy, xx] * inv_area
            px = (bxmin[t][ti] + xx).astype(np.int32)
            py = (bymin[t][ti] + yy).astype(np.int32)
            src_tri = src[t][ti].astype(np.int32) + tri_offset
            if tri_map is not None:
                src_tri = tri_map[src[t][ti]].astype(np.int32)
            if flat_depth is not None:
                zz = flat_depth[src_tri].astype(np.float32) + z_offset
            else:
                zz = (l0 * z[t, 0][ti] + l1 * z[t, 1][ti] +
                      l2 * z[t, 2][ti]) + z_offset
            zk = None
            if enc_on:
                # R251: the code, from the same 1/w the divide below
                # computes (elementwise identical)
                invw_e = l0 * iw[t, 0][ti] + l1 * iw[t, 1][ti] \
                    + l2 * iw[t, 2][ti]
                invw_e = np.where(np.abs(invw_e) < 1e-20, 1e-20, invw_e)
                zk, zz = encode_depth(zz, invw_e, opts, depth_bits)
            elif depth_bits < 32:
                zz = quantize_depth(zz, depth_bits)

            if depth_test:
                if frags is not None:
                    # A-buffer collection: the tolerant limit (see
                    # abuf_depth_limit) -- modeled contacts survive
                    # whichever rasteriser wrote the opaque depth
                    keep = zz < abuf_depth_limit(gbuf.depth[py, px])
                elif zk is not None:
                    keep = zk < zkey_b[py, px]
                else:
                    keep = zz < gbuf.depth[py, px]
                if not keep.any():
                    continue
                ti, l0, l1, l2, px, py, zz, src_tri = (
                    a[keep] for a in (ti, l0, l1, l2, px, py, zz, src_tri))
                e_t = e_t[keep]
                if zk is not None:
                    zk = zk[keep]

            iw0, iw1, iw2 = iw[t, 0][ti], iw[t, 1][ti], iw[t, 2][ti]
            invw = l0 * iw0 + l1 * iw1 + l2 * iw2
            invw = np.where(np.abs(invw) < 1e-20, 1e-20, invw)
            P = np.stack([l0 * iw0 / invw, l1 * iw1 / invw, l2 * iw2 / invw], axis=1)
            bwt = bw[t][ti]
            b = np.einsum('nk,nkc->nc', P, bwt)

            if frag_test is not None:
                # R212: alpha-tested rasterisation (see fill). The mask
                # runs after the z candidacy test, so fragments behind
                # solid ground never pay a chain evaluation
                fr = (area[t] < 0.0)[ti]
                keep_a = frag_test(src_tri, px, py, b, fr)
                if not keep_a.any():
                    continue
                if not keep_a.all():
                    ti, l0, l1, l2 = (ti[keep_a], l0[keep_a],
                                      l1[keep_a], l2[keep_a])
                    px, py, zz = px[keep_a], py[keep_a], zz[keep_a]
                    src_tri, b = src_tri[keep_a], b[keep_a]
                    e_t = e_t[keep_a]
                    if zk is not None:
                        zk = zk[keep_a]

            px_all.append(px)
            py_all.append(py)
            zz_all.append(zz.astype(np.float32))
            b_all.append(b.astype(np.float32))
            tri_all.append(src_tri)
            front_all.append((area[t] < 0.0)[ti])
            e_all.append(e_t.astype(np.int64))
            if zk is not None:
                zk_all.append(zk.astype(np.float32))
            if gbuf.bary_lin is not None:
                L = np.stack([l0, l1, l2], axis=1)
                blin_all.append(np.einsum('nk,nkc->nc', L, bwt).astype(np.float32))

    if not px_all:
        return written
    px = np.concatenate(px_all)
    py = np.concatenate(py_all)
    zz = np.concatenate(zz_all)
    b = np.concatenate(b_all)
    tri = np.concatenate(tri_all)
    front = np.concatenate(front_all)
    blin = np.concatenate(blin_all) if blin_all else None

    if frags is not None:
        frags.add(px, py, tri, zz, b, front)

    if depth_write:
        pix = py.astype(np.int64) * W + px
        # THE NAMED TIE RULE (see fill): equal depth -> lowest triangle
        # id. The id joins the sort key, so within this batch the
        # winner is order-free; against depths already in the gbuf
        # (the big-triangle pre-pass, an earlier rasterize call) the
        # equal-depth comparison consults the stored id the same way.
        # R251: under an encoding the CODE is the sort key and the
        # stored comparand; `depth` receives the decoded value
        zk = np.concatenate(zk_all) if zk_all else None
        cmp_v = zz if zk is None else zk
        order = np.lexsort((tri, cmp_v, pix))
        pix_s = pix[order]
        first = np.empty(pix_s.size, bool)
        first[0] = True
        np.not_equal(pix_s[1:], pix_s[:-1], out=first[1:])
        win = order[first]
        wx, wy = px[win], py[win]
        d0 = gbuf.depth[wy, wx] if zk is None else zkey_b[wy, wx]
        better = (cmp_v[win] < d0) | ((cmp_v[win] == d0)
                                      & (tri[win] < gbuf.tri[wy, wx]))
        win = win[better]
        wx, wy = wx[better], wy[better]
        if zk is not None:
            zkey_b[wy, wx] = zk[win]
        gbuf.depth[wy, wx] = zz[win]
        gbuf.zndc[wy, wx] = zz[win]
        gbuf.tri[wy, wx] = tri[win]
        gbuf.bary[wy, wx] = b[win]
        gbuf.front[wy, wx] = front[win]
        if blin is not None:
            gbuf.bary_lin[wy, wx] = blin[win]
        if cvg_b is not None:
            # C001: coverage16 once over the winners (the winning
            # triangle's own corners at the pixel centre)
            e_win = np.concatenate(e_all)[win]
            cvg_b[wy, wx] = coverage16(
                x0[e_win], y0[e_win], x1[e_win], y1[e_win], x2[e_win],
                y2[e_win], area[e_win],
                wx.astype(np.float32) + np.float32(0.5),
                wy.astype(np.float32) + np.float32(0.5))
    return written + int(px.size)


# ---- R251 AA resolve helpers ----
# RAST-B (1.90.0): C094 LightWave's Limit Dynamic Range, C122 Blender
# 2.41's gamma-2 OSA blend, C063 Elite's wireframe rule, C055 the vector
# monitor beam. Float32 / integer NumPy that both devices run: the two
# resolve helpers have GLSL twins in gpu/stages.py (`resolve_source`,
# drawn by gpu/frame.resolve as the RESOLVE_C / _G / _CG variants,
# bitwise in the simulator); the two wire roads are CPU-after-readback
# by name (the wire road's 'wireframe' release in render.py), so a GPU
# frame carries the same bits as a CPU one. Nothing above this line is
# this section's to edit.


def clamp_samples(tile):
    """C094: LightWave 5.6-7.5's Limit Dynamic Range -- every sample's
    colour clipped at 1.0 BEFORE the anti-aliasing filter (LightWave 7
    manual ch. 14), so an overbright edge blends between two displayable
    values instead of bleeding a halo through the reconstruction filter.
    A per-sample minimum on RGB, alpha untouched; returns a copy. The
    GLSL twin is one `min(t.rgb, vec3(1.0))` per tap (stages.RESOLVE_CLAMP).
    """
    out = np.array(tile, np.float32, copy=True)
    out[..., :3] = np.minimum(out[..., :3], np.float32(1.0))
    return out


# ------------------------------------------------ C122: the gamma-2 blend

#: Blender 2.41's RE_GAMMA_TABLE_SIZE (gammaCorrectionTables.c)
GAMMA2_SIZE = 400


class Gamma2Tables:
    """Blender 2.41's `makeGammaTables(2.0)` verbatim, in float32:
    `dom` the 401 knots i * 0.0025 (a multiply, never a division, so the
    GLSL twin recomputes it bitwise), `g` the squares (g[400] = 1.0),
    `gf` the per-cell slopes 400 * (g[i+1] - g[i]) (gf[400] = 0), `ig`
    the square roots (ig[400] = 1.0) and `igf` their slopes."""

    __slots__ = ('dom', 'g', 'gf', 'ig', 'igf')

    def __init__(self):
        n = GAMMA2_SIZE
        idx = np.arange(n + 1, dtype=np.float32)
        dom = (idx * np.float32(0.0025)).astype(np.float32)
        g = (dom.astype(np.float64) ** 2.0).astype(np.float32)
        g[n] = np.float32(1.0)
        ig = np.sqrt(dom.astype(np.float64)).astype(np.float32)
        ig[n] = np.float32(1.0)
        gf = np.zeros(n + 1, np.float32)
        igf = np.zeros(n + 1, np.float32)
        gf[:n] = (float(n) * (g[1:].astype(np.float64)
                              - g[:-1].astype(np.float64))).astype(np.float32)
        igf[:n] = (float(n) * (ig[1:].astype(np.float64)
                               - ig[:-1].astype(np.float64))).astype(np.float32)
        self.dom, self.g, self.gf, self.ig, self.igf = dom, g, gf, ig, igf


_GAMMA2 = {}


def gamma2_tables():
    """The cached Gamma2Tables (built once per process)."""
    T = _GAMMA2.get('T')
    if T is None:
        T = _GAMMA2['T'] = Gamma2Tables()
    return T


def _gamma2_lookup(c, rng, fac, T):
    """Blender's `gammaCorrect`: `range[i] + (c - domain[i]) * factor[i]`
    in the C's own three statements (i = floor(c * 400)); float32 in,
    float32 out, c already in [0, 1]."""
    c = np.asarray(c, np.float32)
    i = np.floor(c * np.float32(GAMMA2_SIZE)).astype(np.int32)
    i = np.clip(i, 0, GAMMA2_SIZE)
    d = (c - T.dom[i]).astype(np.float32)
    d = (d * fac[i]).astype(np.float32)
    return (rng[i] + d).astype(np.float32)


def gamma2_correct(c, T=None):
    """C122: c in [0, 1] squared through the 400-entry piecewise-linear
    table (the sample side of Blender 2.41's OSA gamma blend)."""
    T = T or gamma2_tables()
    return _gamma2_lookup(c, T.g, T.gf, T)


def gamma2_inverse(c, T=None):
    """C122: c in [0, 1] square-rooted through the inverse table (the
    finished-pixel side of the blend)."""
    T = T or gamma2_tables()
    return _gamma2_lookup(c, T.ig, T.igf, T)


def gamma2_texture():
    """The tables as one (1, 401, 4) float32 texture, (g, gf, ig, igf)
    per texel, for the RESOLVE_G / _CG variants' `gtab` sampler."""
    T = gamma2_tables()
    tex = np.zeros((1, GAMMA2_SIZE + 1, 4), np.float32)
    tex[0, :, 0] = T.g
    tex[0, :, 1] = T.gf
    tex[0, :, 2] = T.ig
    tex[0, :, 3] = T.igf
    return tex


def gamma2_blend(tile, kern):
    """C122: render._resolve's einsum with Blender 2.41's gamma-2 sample
    blend around it -- every subsample clamped to [0, 1] and squared
    through the table, the filter's fixed-order sum, the finished pixel
    clamped and square-rooted through the inverse table. Alpha is
    filtered plainly, never gamma'd. `tile` is (H, ss, W, ss, 4)."""
    T = gamma2_tables()
    t = np.array(tile, np.float32, copy=True)
    t[..., :3] = gamma2_correct(np.clip(t[..., :3], 0, 1), T)
    out = np.einsum('hiwjc,ij->hwc', t, kern).astype(np.float32)
    out[..., :3] = gamma2_inverse(np.clip(out[..., :3], 0, 1), T)
    return out


# --------------------------------------------- C063: Elite's wire rule

_EDGE_CACHE = {}


def mesh_edges(mesh):
    """(edges (E, 2) int32 low-high, faces (E, 2) int32): every unique
    vertex pair of the triangle soup with the first two (lowest-index)
    triangles that reference it, -1 for a boundary edge's missing second
    face. Vertices are WELDED by position first (an exported mesh splits
    its corners per face for normals and UVs -- a cube arrives as 24
    vertices -- and an edge shared by two faces must know both, as
    Elite's edge lists did), each welded vertex keeping its first index
    into `mesh.verts`. Order-free (a mesh's edge list is a fixed function
    of its triangles); cached on the tris array's identity with the
    source pinned in the value."""
    tris = np.asarray(mesh.tris, np.int32)
    key = (id(tris), tris.shape, int(tris[::17].sum()) if tris.size else 0)
    hit = _EDGE_CACHE.get(key)
    if hit is not None and hit[0] is tris:
        return hit[1], hit[2]
    T = tris.shape[0]
    if T == 0:
        return np.zeros((0, 2), np.int32), np.zeros((0, 2), np.int32)
    verts = np.asarray(mesh.verts, np.float32)
    _pos, first, inv = np.unique(verts, axis=0, return_index=True,
                                 return_inverse=True)
    weld = first.astype(np.int32)[np.asarray(inv).ravel()]
    wt = weld[tris]
    pairs = np.concatenate([wt[:, [0, 1]], wt[:, [1, 2]],
                            wt[:, [2, 0]]], axis=0)
    pairs = np.sort(pairs, axis=1)
    tri_of = np.tile(np.arange(T, dtype=np.int32), 3)
    live = pairs[:, 0] != pairs[:, 1]
    pairs, tri_of = pairs[live], tri_of[live]
    uniq, inv = np.unique(pairs, axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel().astype(np.int64)
    faces = np.full((uniq.shape[0], 2), -1, np.int32)
    order = np.lexsort((tri_of, inv))
    e_s, t_s = inv[order], tri_of[order]
    first = np.ones(e_s.size, bool)
    first[1:] = e_s[1:] != e_s[:-1]
    faces[e_s[first], 0] = t_s[first]
    second = np.zeros(e_s.size, bool)
    second[1:] = (e_s[1:] == e_s[:-1]) & first[:-1]
    faces[e_s[second], 1] = t_s[second]
    edges = uniq.astype(np.int32)
    if len(_EDGE_CACHE) >= 8:
        _EDGE_CACHE.pop(next(iter(_EDGE_CACHE)))
    _EDGE_CACHE[key] = (tris, edges, faces)
    return edges, faces


def _face_normals(mesh):
    fn = getattr(mesh, 'face_normals', None)
    if fn is not None:
        return np.asarray(fn, np.float32)
    V = np.asarray(mesh.verts, np.float32)
    T = np.asarray(mesh.tris, np.int32)
    n = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return (n / np.where(ln < 1e-12, 1.0, ln)).astype(np.float32)


def elite_edges(mesh, eye):
    """C063: Elite's visibility rule per edge -- an edge draws when at
    least one of its two faces faces the viewer, judged by the sign of
    the dot between the face's 8-bit normal (each axis rounded to 1/127,
    never renormalised: the sign is all Elite used) and the vector from
    the face's first vertex to the eye. No depth test anywhere. Returns
    the (E,) bool mask over mesh_edges(mesh)."""
    edges, faces = mesh_edges(mesh)
    if edges.shape[0] == 0:
        return np.zeros(0, bool)
    fn = _face_normals(mesh)
    nq = (np.round(fn * np.float32(127.0)) / np.float32(127.0)).astype(np.float32)
    verts = np.asarray(mesh.verts, np.float32)
    pf = verts[np.asarray(mesh.tris, np.int32)[:, 0]]
    eye = np.asarray(eye, np.float32)
    front = (nq * (eye[None, :] - pf)).sum(axis=1) > 0.0
    f0, f1 = faces[:, 0], faces[:, 1]
    vis = (f0 >= 0) & front[np.maximum(f0, 0)]
    vis |= (f1 >= 0) & front[np.maximum(f1, 0)]
    # a flat quad's triangulation diagonal is not an authored edge: Elite's
    # ships listed edges between FACES, so an edge whose two faces are
    # coplanar (a diagonal, or the seam of two coplanar quads) never draws
    both = (f0 >= 0) & (f1 >= 0)
    flat = (fn[np.maximum(f0, 0)] * fn[np.maximum(f1, 0)]).sum(axis=1)         >= np.float32(0.9999)
    vis &= ~(both & flat)
    return vis


def elite_visible_edges(mesh, vp, eye, **_unused):
    """The design's name for the edge list under the rule: (edges (K, 2)
    int32 of the visible edges, faces (K, 2)). `vp` is accepted for the
    signature's sake; visibility is a function of the eye alone."""
    edges, faces = mesh_edges(mesh)
    vis = elite_edges(mesh, eye)
    return edges[vis], faces[vis]


def _clip_to_screen_wire(clip, width, height, snap=0.0, near_eps=1e-5,
                    pixel_shift=0.0):
    """`project`'s screen formula over CLIP-space points (the near-clipped
    segment endpoints must not pass through mvp again): the same float32
    expressions, so a wire lands on exactly the raster's grid."""
    clip = np.asarray(clip, np.float32)
    w = clip[:, 3]
    safe_w = np.where(np.abs(w) < near_eps, near_eps, w)
    invw = (1.0 / safe_w).astype(np.float32)
    ndc = clip[:, :3] * invw[:, None]
    sx = (ndc[:, 0] * 0.5 + 0.5) * width
    sy = (ndc[:, 1] * 0.5 + 0.5) * height
    screen = np.stack([sx, sy], axis=1).astype(np.float32)
    if pixel_shift:
        screen = (screen + np.float32(pixel_shift)).astype(np.float32)
    if snap > 0.0:
        screen = np.round(screen / snap) * snap
    return np.asarray(screen, np.float32)


def clip_segments(c0, c1, near_eps=1e-5):
    """Near-plane clip of clip-space segments (f = z + w against
    near_eps): both ends behind -> dropped; one behind -> that endpoint
    moved onto the plane along the segment. Returns (a, b, keep)."""
    c0 = np.asarray(c0, np.float32)
    c1 = np.asarray(c1, np.float32)
    eps = np.float32(near_eps)
    f0 = (c0[:, 2] + c0[:, 3]).astype(np.float32)
    f1 = (c1[:, 2] + c1[:, 3]).astype(np.float32)
    keep = (f0 >= eps) | (f1 >= eps)
    a, b = c0.copy(), c1.copy()
    m0 = keep & (f0 < eps)
    if m0.any():
        t = ((eps - f0[m0]) / (f1[m0] - f0[m0])).astype(np.float32)
        a[m0] = c0[m0] + (c1[m0] - c0[m0]) * t[:, None]
    m1 = keep & (f1 < eps)
    if m1.any():
        t = ((eps - f1[m1]) / (f0[m1] - f1[m1])).astype(np.float32)
        b[m1] = c1[m1] + (c0[m1] - c1[m1]) * t[:, None]
    return a, b, keep


def _clip_rect(p0, p1, width, height, margin=2.0):
    """Liang-Barsky clip of screen-space segments to the frame plus a
    margin, so a segment that leaves the frame by a mile (an endpoint
    just in front of the near plane) costs a frame's worth of DDA steps,
    not a mile's. Returns (q0, q1, keep), float32."""
    p0 = np.asarray(p0, np.float32)
    p1 = np.asarray(p1, np.float32)
    d = p1 - p0
    t0 = np.zeros(p0.shape[0], np.float32)
    t1 = np.ones(p0.shape[0], np.float32)
    keep = np.ones(p0.shape[0], bool)
    lo = np.float32(-margin)
    hi = (np.float32(width + margin), np.float32(height + margin))
    for axis in (0, 1):
        for sign, bound in ((-1.0, lo), (1.0, hi[axis])):
            pk = np.float32(sign) * d[:, axis]
            qk = (bound - p0[:, axis]) * np.float32(sign)
            par = pk == 0.0
            keep &= ~(par & (qk < 0.0))
            safe = np.where(par, np.float32(1.0), pk)
            r = (qk / safe).astype(np.float32)
            enter = (~par) & (pk < 0.0)
            leave = (~par) & (pk > 0.0)
            t0 = np.where(enter, np.maximum(t0, r), t0).astype(np.float32)
            t1 = np.where(leave, np.minimum(t1, r), t1).astype(np.float32)
    keep &= t0 <= t1
    q0 = (p0 + d * t0[:, None]).astype(np.float32)
    q1 = (p0 + d * t1[:, None]).astype(np.float32)
    return q0, q1, keep


def draw_lines(mask, p0, p1):
    """Rasterise screen-space segments into a bool mask by a DDA: n =
    max(ceil(max(|dx|, |dy|)), 1) steps, t = arange(n + 1) / n (float32
    by construction), each sample landing in the pixel whose centre is
    nearest -- floor of the screen coordinate, the raster's own px + 0.5
    convention. An OR, so the segment order cannot change a bit."""
    H, W = mask.shape
    p0 = np.asarray(p0, np.float32).reshape(-1, 2)
    p1 = np.asarray(p1, np.float32).reshape(-1, 2)
    if p0.shape[0] == 0:
        return mask
    p0, p1, keep = _clip_rect(p0, p1, W, H)
    p0, p1 = p0[keep], p1[keep]
    if p0.shape[0] == 0:
        return mask
    dx = (p1[:, 0] - p0[:, 0]).astype(np.float32)
    dy = (p1[:, 1] - p0[:, 1]).astype(np.float32)
    n = np.maximum(np.ceil(np.maximum(np.abs(dx), np.abs(dy))),
                   np.float32(1.0)).astype(np.float32)
    counts = n.astype(np.int64) + 1
    seg = np.repeat(np.arange(n.size), counts)
    k = np.arange(counts.sum(), dtype=np.int64) - np.repeat(
        np.cumsum(counts) - counts, counts)
    t = (k.astype(np.float32) / n[seg]).astype(np.float32)
    x = (p0[seg, 0] + dx[seg] * t).astype(np.float32)
    y = (p0[seg, 1] + dy[seg] * t).astype(np.float32)
    ix = np.floor(x).astype(np.int64)
    iy = np.floor(y).astype(np.int64)
    inside = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
    mask[iy[inside], ix[inside]] = True
    return mask


def _dilate_square(mask, w):
    """Grow a mask by a w x w square of shifted ORs (order-free)."""
    w = int(w)
    if w <= 1:
        return mask
    H, W = mask.shape
    lo = -(w // 2)
    out = np.zeros_like(mask)
    for dy in range(lo, lo + w):
        for dx in range(lo, lo + w):
            ys = slice(max(dy, 0), H + min(dy, 0))
            yd = slice(max(-dy, 0), H + min(-dy, 0))
            xs = slice(max(dx, 0), W + min(dx, 0))
            xd = slice(max(-dx, 0), W + min(-dx, 0))
            out[yd, xd] |= mask[ys, xs]
    return out


def _object_verts(mesh):
    """object index -> the unique vertex indices its triangles use."""
    tris = np.asarray(mesh.tris, np.int32)
    obj = getattr(mesh, 'obj_index', None)
    if obj is None:
        return {0: np.unique(tris)}
    obj = np.asarray(obj, np.int32)
    out = {}
    for o in np.unique(obj):
        out[int(o)] = np.unique(tris[obj == o])
    return out


def draw_elite_wire(img, mesh, vp, eye, st, snap=0.0, near_eps=1e-5,
                    pixel_shift=0.0):
    """C063: the wire overlay under Elite's rule (BBC Micro 1984; Mark
    Moxon, 'Drawing ships'): every edge with at least one front face is
    drawn, no depth test at all -- the far edges of a concave hull show
    through, nothing hides behind anything -- and an object farther than
    `wire_dot_distance` collapses to one dot at its centroid. Lines are
    near-clipped in clip space, projected on the raster's grid, DDA'd
    and dilated to `wire_width`; the colour overwrites (Elite's 1-bit
    lines). Edits `img` in place and returns it. CPU on both devices
    through the wire road's readback by name."""
    H, W = img.shape[:2]
    edges, faces = mesh_edges(mesh)
    if edges.shape[0] == 0:
        return img
    verts = np.asarray(mesh.verts, np.float32)
    vis = elite_edges(mesh, eye)
    eye = np.asarray(eye, np.float32)
    clip, _screen, _invw, _z = project(verts, vp, W, H, snap=snap,
                                       near_eps=near_eps)
    mask = np.zeros((H, W), bool)
    dot_d = float(getattr(st, 'wire_dot_distance', 0.0) or 0.0)
    dots = 0
    if dot_d > 0.0:
        obj = getattr(mesh, 'obj_index', None)
        edge_obj = (np.asarray(obj, np.int32)[np.maximum(faces[:, 0], 0)]
                    if obj is not None else np.zeros(edges.shape[0], np.int32))
        for o, vidx in _object_verts(mesh).items():
            centre = verts[vidx].mean(axis=0).astype(np.float32)
            d = float(np.linalg.norm(eye - centre))
            if d <= dot_d:
                continue
            vis &= edge_obj != o
            ph = np.append(centre, np.float32(1.0)).astype(np.float32)
            c = (ph @ np.asarray(vp, np.float32).T).astype(np.float32)
            if float(c[2] + c[3]) < near_eps:
                continue
            s = _clip_to_screen_wire(c[None, :], W, H, snap, near_eps, pixel_shift)
            ix, iy = int(np.floor(s[0, 0])), int(np.floor(s[0, 1]))
            if 0 <= ix < W and 0 <= iy < H:
                mask[iy, ix] = True
                dots += 1
    sel = edges[vis]
    if sel.shape[0]:
        a, b, keep = clip_segments(clip[sel[:, 0]], clip[sel[:, 1]], near_eps)
        if keep.any():
            s0 = _clip_to_screen_wire(a[keep], W, H, snap, near_eps, pixel_shift)
            s1 = _clip_to_screen_wire(b[keep], W, H, snap, near_eps, pixel_shift)
            lines = np.zeros((H, W), bool)
            draw_lines(lines, s0, s1)
            wpx = max(int(np.floor(float(getattr(st, 'wire_width', 1.0)) + 0.5)), 1)
            mask |= _dilate_square(lines, wpx)
    col = np.asarray(getattr(st, 'wire_color', (0.0, 0.0, 0.0)), np.float32)
    n = int(mask.sum())
    if n:
        img[mask, :3] = col[None, :]
        img[mask, 3] = 1.0
    if not getattr(st, '_viewport', False) and n:
        print(f'[Halcyon] wireframe overlay: inked {n} pixels (ELITE: '
              f'{int(vis.sum())} of {edges.shape[0]} edges face the viewer, '
              f'no depth test, {dots} dot(s)) -- the Wireframe panel\'s '
              'Wireframe Overlay checkbox turns it off')
    return img


# ------------------------------------------ C055: the vector monitor beam

#: per vector generator: the intensity quantiser's top code (zmax), the
#: colour's top code (cmax: the DVG's overlay gel is 8-bit and unquantised,
#: the AVG's colour RAM and Star Wars' 3 direct bits are 1 bit per channel)
#: and the end dwell in 1/64 of the stroke's own energy -- the deflection
#: amplifier's settling, a property of the generator the era never
#: exposed, so a constant per item, not a dial (chosen, not measured: the
#: DVG's slower vector timer makes its vertex dots the brighter, the
#: Asteroids look)
BEAM_MACHINES = {
    'DVG': dict(zmax=15, cmax=255, dwell64=32),
    'AVG': dict(zmax=14, cmax=1, dwell64=16),
    'STARWARS': dict(zmax=255, cmax=1, dwell64=16),
}

#: the phosphor spot as an INTEGER table: entry i is
#: round(65536 * exp(-((i + 0.5) / 64) / 2)) over q = (d / sigma)^2 in
#: 1/64 steps to 3 sigma (576 entries), computed once when this table was
#: written and pinned as literals so two machines with different libm
#: `exp` last bits render the same beam; `exp` never runs at render time
#: (the CIRCLE256 idiom). Index 576 (beyond 3 sigma) contributes nothing.
BEAM_LUT = (
    65280, 64772, 64268, 63768, 63272, 62780, 62291, 61806, 61325, 60848, 60375, 59905,
    59439, 58976, 58517, 58062, 57610, 57162, 56717, 56275, 55837, 55403, 54972, 54544,
    54119, 53698, 53280, 52866, 52454, 52046, 51641, 51239, 50841, 50445, 50052, 49663,
    49276, 48893, 48512, 48135, 47760, 47389, 47020, 46654, 46291, 45931, 45573, 45218,
    44867, 44517, 44171, 43827, 43486, 43148, 42812, 42479, 42148, 41820, 41495, 41172,
    40851, 40534, 40218, 39905, 39595, 39286, 38981, 38677, 38376, 38078, 37781, 37487,
    37196, 36906, 36619, 36334, 36051, 35771, 35492, 35216, 34942, 34670, 34400, 34133,
    33867, 33604, 33342, 33083, 32825, 32570, 32316, 32065, 31815, 31568, 31322, 31078,
    30836, 30596, 30358, 30122, 29888, 29655, 29424, 29195, 28968, 28743, 28519, 28297,
    28077, 27858, 27642, 27426, 27213, 27001, 26791, 26583, 26376, 26170, 25967, 25765,
    25564, 25365, 25168, 24972, 24778, 24585, 24394, 24204, 24015, 23828, 23643, 23459,
    23276, 23095, 22916, 22737, 22560, 22385, 22211, 22038, 21866, 21696, 21527, 21360,
    21193, 21029, 20865, 20703, 20541, 20382, 20223, 20066, 19909, 19754, 19601, 19448,
    19297, 19147, 18998, 18850, 18703, 18558, 18413, 18270, 18128, 17987, 17847, 17708,
    17570, 17433, 17298, 17163, 17029, 16897, 16765, 16635, 16505, 16377, 16250, 16123,
    15998, 15873, 15750, 15627, 15505, 15385, 15265, 15146, 15028, 14911, 14795, 14680,
    14566, 14453, 14340, 14229, 14118, 14008, 13899, 13791, 13684, 13577, 13471, 13367,
    13263, 13159, 13057, 12955, 12854, 12754, 12655, 12557, 12459, 12362, 12266, 12170,
    12076, 11982, 11888, 11796, 11704, 11613, 11523, 11433, 11344, 11256, 11168, 11081,
    10995, 10909, 10825, 10740, 10657, 10574, 10492, 10410, 10329, 10248, 10169, 10090,
    10011, 9933, 9856, 9779, 9703, 9628, 9553, 9478, 9405, 9331, 9259, 9187,
    9115, 9044, 8974, 8904, 8835, 8766, 8698, 8630, 8563, 8496, 8430, 8365,
    8299, 8235, 8171, 8107, 8044, 7982, 7919, 7858, 7797, 7736, 7676, 7616,
    7557, 7498, 7440, 7382, 7324, 7267, 7211, 7155, 7099, 7044, 6989, 6934,
    6881, 6827, 6774, 6721, 6669, 6617, 6565, 6514, 6464, 6413, 6363, 6314,
    6265, 6216, 6168, 6120, 6072, 6025, 5978, 5931, 5885, 5839, 5794, 5749,
    5704, 5660, 5616, 5572, 5529, 5486, 5443, 5401, 5359, 5317, 5275, 5234,
    5194, 5153, 5113, 5073, 5034, 4995, 4956, 4917, 4879, 4841, 4803, 4766,
    4729, 4692, 4656, 4619, 4583, 4548, 4512, 4477, 4442, 4408, 4374, 4339,
    4306, 4272, 4239, 4206, 4173, 4141, 4109, 4077, 4045, 4013, 3982, 3951,
    3920, 3890, 3860, 3830, 3800, 3770, 3741, 3712, 3683, 3654, 3626, 3598,
    3570, 3542, 3514, 3487, 3460, 3433, 3406, 3380, 3353, 3327, 3301, 3276,
    3250, 3225, 3200, 3175, 3150, 3126, 3101, 3077, 3053, 3029, 3006, 2982,
    2959, 2936, 2913, 2891, 2868, 2846, 2824, 2802, 2780, 2758, 2737, 2716,
    2694, 2673, 2653, 2632, 2612, 2591, 2571, 2551, 2531, 2512, 2492, 2473,
    2453, 2434, 2415, 2396, 2378, 2359, 2341, 2323, 2305, 2287, 2269, 2251,
    2234, 2216, 2199, 2182, 2165, 2148, 2131, 2115, 2098, 2082, 2066, 2050,
    2034, 2018, 2002, 1987, 1971, 1956, 1941, 1926, 1911, 1896, 1881, 1866,
    1852, 1837, 1823, 1809, 1795, 1781, 1767, 1753, 1740, 1726, 1713, 1699,
    1686, 1673, 1660, 1647, 1634, 1622, 1609, 1596, 1584, 1572, 1559, 1547,
    1535, 1523, 1511, 1500, 1488, 1476, 1465, 1454, 1442, 1431, 1420, 1409,
    1398, 1387, 1376, 1365, 1355, 1344, 1334, 1323, 1313, 1303, 1293, 1283,
    1273, 1263, 1253, 1243, 1234, 1224, 1214, 1205, 1196, 1186, 1177, 1168,
    1159, 1150, 1141, 1132, 1123, 1114, 1106, 1097, 1089, 1080, 1072, 1063,
    1055, 1047, 1039, 1031, 1023, 1015, 1007, 999, 991, 984, 976, 968,
    961, 953, 946, 938, 931, 924, 917, 910, 903, 896, 889, 882,
    875, 868, 861, 854, 848, 841, 835, 828, 822, 815, 809, 803,
    796, 790, 784, 778, 772, 766, 760, 754, 748, 742, 737, 731,
)

_BEAM_LUT_ARR = np.array(BEAM_LUT + (0,), np.int64)


def dvg_program(I, color):
    """DVG (Asteroids / Battlezone): 4-bit intensity round(I * 15) and the
    overlay gel's colour as 8-bit integers floored at 1 (the tube was
    monochrome; the floor keeps a set intensity visible). (Z (E,), C (3,))"""
    Z = np.round(np.asarray(I, np.float32) * np.float32(15.0)).astype(np.int64)
    C = np.maximum(np.round(np.asarray(color, np.float32) * np.float32(255.0)),
                   1.0).astype(np.int64)
    return Z, C


def avg_program(I, color, zmax=14):
    """AVG (Tempest / Major Havoc): 3-bit doubled intensity -- 0 draws
    nothing, otherwise 2 * clamp(round(I * 7), 2, 7) in 4..14 -- or Star
    Wars' 8-bit STATZ round(I * 255) when zmax is 255; colour 1 bit per
    channel (wire_color >= 0.5). (Z (E,), C (3,))"""
    I = np.asarray(I, np.float32)
    if int(zmax) == 255:
        Z = np.round(I * np.float32(255.0)).astype(np.int64)
    else:
        q = np.clip(np.round(I * np.float32(7.0)), 2.0, 7.0).astype(np.int64) * 2
        Z = np.where(I == 0.0, 0, q).astype(np.int64)
    C = (np.asarray(color, np.float32) >= np.float32(0.5)).astype(np.int64)
    return Z, C


def beam_strokes(mesh, angle_deg=25.0):
    """The stroke list: the mesh's feature edges -- boundary edges and
    edges whose two face normals differ by more than `angle_deg` (the
    CREASE dial reused) -- with no visibility test (the games drew the
    list). Returns (K, 2) int32 vertex pairs."""
    edges, faces = mesh_edges(mesh)
    if edges.shape[0] == 0:
        return edges
    fn = _face_normals(mesh)
    f0, f1 = faces[:, 0], faces[:, 1]
    boundary = (f1 < 0) | (f0 < 0)
    cos_lim = np.float32(np.cos(np.radians(max(min(float(angle_deg), 179.0), 0.0))))
    dot = (fn[np.maximum(f0, 0)] * fn[np.maximum(f1, 0)]).sum(axis=1)
    return edges[boundary | (dot < cos_lim)]


def _beam_index(q2, sig2):
    """LUT index of a squared distance: min(floor(q2 / sigma^2 * 64), 576)."""
    q = (q2 / sig2).astype(np.float32)
    i = np.floor(q * np.float32(64.0))
    i = np.where(np.isfinite(i), i, 1e9)
    return np.minimum(i, 576.0).astype(np.int64)


def beam_trace(segs, Z, C, machine, sigma_px, w, h):
    """C055: the accumulator, integer and order-free. `segs` (K, 2, 2)
    float32 screen-space strokes, `Z` (K,) the quantised intensities,
    `C` (3,) the colour codes, `machine` a BEAM_MACHINES key, `sigma_px`
    the spot's sigma in pixels. Per stroke, over its bbox expanded by
    3 sigma: d^2 to the segment (t = clamp(dot(p - a, b - a) / |b - a|^2,
    0, 1), float32), da^2 / db^2 to the ends, and
    `acc += Z * C * (LUT[i_d] + ((dwell64 * (LUT[i_a] + LUT[i_b])) >> 6))`
    -- the whole product int64, summed in int64, so the stroke order
    cannot change a bit. Returns the int64 (h, w, 3) accumulator."""
    consts = BEAM_MACHINES[str(machine)]
    dwell = int(consts['dwell64'])
    acc = np.zeros((int(h), int(w), 3), np.int64)
    segs = np.asarray(segs, np.float32).reshape(-1, 2, 2)
    Z = np.asarray(Z, np.int64).reshape(-1)
    C = np.asarray(C, np.int64).reshape(3)
    sig = np.float32(sigma_px)
    sig2 = np.float32(sig * sig)
    if sig2 <= 0.0:
        return acc
    reach = int(np.ceil(3.0 * float(sig))) + 1
    lut = _BEAM_LUT_ARR
    for k in range(segs.shape[0]):
        if Z[k] <= 0:
            continue
        ax, ay = segs[k, 0]
        bx, by = segs[k, 1]
        x0 = max(int(np.floor(min(ax, bx))) - reach, 0)
        x1 = min(int(np.ceil(max(ax, bx))) + reach, int(w) - 1)
        y0 = max(int(np.floor(min(ay, by))) - reach, 0)
        y1 = min(int(np.ceil(max(ay, by))) + reach, int(h) - 1)
        if x1 < x0 or y1 < y0:
            continue
        X = (np.arange(x0, x1 + 1, dtype=np.float32) + np.float32(0.5))[None, :]
        Y = (np.arange(y0, y1 + 1, dtype=np.float32) + np.float32(0.5))[:, None]
        abx = np.float32(bx - ax)
        aby = np.float32(by - ay)
        pax = (X - ax).astype(np.float32)
        pay = (Y - ay).astype(np.float32)
        L2 = np.float32(abx * abx + aby * aby)
        if L2 > 0.0:
            t = np.clip(((pax * abx + pay * aby) / L2).astype(np.float32),
                        np.float32(0.0), np.float32(1.0)).astype(np.float32)
        else:
            t = np.zeros(np.broadcast(pax, pay).shape, np.float32)
        cx = (pax - abx * t).astype(np.float32)
        cy = (pay - aby * t).astype(np.float32)
        d2 = (cx * cx + cy * cy).astype(np.float32)
        da2 = (pax * pax + pay * pay).astype(np.float32)
        pbx = (X - bx).astype(np.float32)
        pby = (Y - by).astype(np.float32)
        db2 = (pbx * pbx + pby * pby).astype(np.float32)
        val = lut[_beam_index(d2, sig2)] + (
            (dwell * (lut[_beam_index(da2, sig2)]
                      + lut[_beam_index(db2, sig2)])) >> 6)
        zval = int(Z[k]) * val
        for ch in range(3):
            if C[ch]:
                acc[y0:y1 + 1, x0:x1 + 1, ch] += int(C[ch]) * zval
    return acc


def beam_image(acc, machine):
    """The accumulator as float32 light: clip(acc / (65536 * zmax * cmax), 0, 1)."""
    consts = BEAM_MACHINES[str(machine)]
    denom = 65536.0 * float(consts['zmax']) * float(consts['cmax'])
    return np.clip(acc.astype(np.float64) / denom, 0.0, 1.0).astype(np.float32)


def draw_beam_wire(img, mesh, view, vp, eye, st, rh=None, snap=0.0,
                   near_eps=1e-5, pixel_shift=0.0):
    """C055: the wire overlay as a vector monitor's beam (Atari DVG / AVG,
    Vectrex): every feature edge a phosphor stroke -- a Gaussian spot from
    the pinned integer table, intensity quantised by the machine (fogged
    by the era's fade to black when fog is on), colour 1 bit per channel
    on the AVG and Star Wars, dwell dots at the stroke ends, additive where
    strokes cross -- and no depth test. When the 1-bit colour rounds to
    black the road prints why once and returns the frame untouched (a
    named no-op, never a silent zero). Edits `img` in place. CPU on both
    devices through the wire road's readback by name."""
    H, W = img.shape[:2]
    rh = int(rh or H)
    machine = str(getattr(st, 'beam_machine', 'AVG') or 'AVG').upper()
    if machine not in BEAM_MACHINES:
        machine = 'AVG'
    color = np.asarray(getattr(st, 'wire_color', (0.0, 0.0, 0.0)), np.float32)
    if machine != 'DVG' and not bool((color >= 0.5).any()):
        print('[Halcyon] vector beam: the wire colour rounds to black under '
              '1 bit per channel; nothing drawn')
        return img
    edges = beam_strokes(mesh, float(getattr(st, 'wire_angle', 25.0)))
    if edges.shape[0] == 0:
        return img
    verts = np.asarray(mesh.verts, np.float32)
    I = np.ones(edges.shape[0], np.float32)
    if getattr(st, 'fog', False):
        mid = ((verts[edges[:, 0]] + verts[edges[:, 1]]) * np.float32(0.5)).astype(np.float32)
        vm = np.asarray(view, np.float32)
        zv = -(mid @ vm[:3, :3].T + vm[:3, 3][None, :])[:, 2]
        f0 = np.float32(getattr(st, 'fog_start', 5.0))
        f1 = np.float32(getattr(st, 'fog_end', 40.0))
        span = np.float32(f1 - f0) if f1 > f0 else np.float32(1e-6)
        I = np.clip(((f1 - zv.astype(np.float32)) / span).astype(np.float32),
                    np.float32(0.0), np.float32(1.0)).astype(np.float32)
    consts = BEAM_MACHINES[machine]
    if machine == 'DVG':
        Z, C = dvg_program(I, color)
    else:
        Z, C = avg_program(I, color, zmax=consts['zmax'])
    clip, _screen, _invw, _z = project(verts, vp, W, H, snap=snap,
                                       near_eps=near_eps)
    a, b, keep = clip_segments(clip[edges[:, 0]], clip[edges[:, 1]], near_eps)
    if not keep.any():
        return img
    s0 = _clip_to_screen_wire(a[keep], W, H, snap, near_eps, pixel_shift)
    s1 = _clip_to_screen_wire(b[keep], W, H, snap, near_eps, pixel_shift)
    segs = np.stack([s0, s1], axis=1).astype(np.float32)
    sigma_px = np.float32(float(getattr(st, 'beam_sigma', 0.7)) * rh / 480.0)
    acc = beam_trace(segs, Z[keep], C, machine, sigma_px, W, H)
    beam = beam_image(acc, machine)
    img[:, :, :3] = np.minimum(img[:, :, :3] + beam, np.float32(1.0))
    lit = (beam > 0.0).any(axis=2)
    img[:, :, 3] = np.maximum(img[:, :, 3], lit.astype(np.float32))
    if not getattr(st, '_viewport', False):
        print(f'[Halcyon] vector beam ({machine}): {int(keep.sum())} strokes, '
              f'{int(lit.sum())} pixels lit, spot sigma {float(sigma_px):.3f} px '
              "-- the Wireframe panel's Wireframe Overlay checkbox turns it off")
    return img
