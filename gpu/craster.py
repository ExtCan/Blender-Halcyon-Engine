"""The compute rasteriser: fill()'s exact rules, one thread per pixel.

Hardware rasterisation can never reproduce this renderer's G-buffer -- the
fill conventions differ at triangle edges -- so the port is the CPU's own
algorithm as a compute kernel. The design that makes exactness POSSIBLE is
per-pixel sequential resolve: the screen is cut into tiles, triangles are
binned per tile on the CPU (cheap, vectorised), and every pixel walks its
tile's bin IN SUBMISSION ORDER with the strict `<` depth test. First
triangle wins ties because it is literally tested first, exactly as fill()
writes first -- no atomics, no resolve pass, nothing order-dependent.

Everything numerical is float32 on both sides, so agreement is to the ulp;
the residual risk is a pixel whose edge function sits within one ulp of
zero, which is measure-zero in practice and the tests would show it.

The kernel's core is one shared GLSL text with two thin wrappers: a
fragment-style one the NumPy front-end can run headlessly (per tile, with
the bin range as uniforms so the loop bound is lane-uniform), and the
compute one the driver runs (bin ranges read per pixel from the tile
texture). What is verified headlessly is the mathematics; what only the
driver can prove is the dispatch, exactly the split the deferred pass
established.
"""

import numpy as np

TILE = 16

#: the shared core: everything between "which pixel" and "who won".
#: `hal_rc` carries two texels per emitted triangle corner:
#:   texel 2*(e*3+c)+0 = (sx, sy, z, iw)     texel 2*(e*3+c)+1 = (bw, src)
KERNEL_CORE = """
uniform sampler2D hal_rc;            // packed corners, two texels per corner
uniform float hal_rc_side;
uniform sampler2D hal_rbins;         // triangle indices, four per texel
uniform float hal_rbins_side;
uniform float hal_zsteps;            // z-buffer grid steps (0 = 32-bit off)

vec4 hal_rc_fetch(float index)
{
    float x = mod(index, hal_rc_side);
    float y = floor(index / hal_rc_side);
    return texture(hal_rc, (vec2(x, y) + vec2(0.5)) / hal_rc_side);
}

float hal_rbin_entry(float i)
{
    float texel = floor(i / 4.0);
    float x = mod(texel, hal_rbins_side);
    float y = floor(texel / hal_rbins_side);
    vec4 v = texture(hal_rbins,
                     (vec2(x, y) + vec2(0.5)) / hal_rbins_side);
    float c = i - texel * 4.0;
    if (c < 0.5) { return v.x; }
    if (c < 1.5) { return v.y; }
    if (c < 2.5) { return v.z; }
    return v.w;
}

// walk one pixel's bin in submission order; fill()'s exact rules.
// returns (winner emitted-tri index or -1, l0, l1, depth)
vec4 hal_raster_pixel(vec2 pix, float start, float count, float cull)
{
    float best_z = 1e30;
    float win = -1.0;
    float wl0 = 0.0;
    float wl1 = 0.0;
    float X = pix.x;
    float Y = pix.y;
    for (int i = 0; i < int(count); i++) {
        float e = hal_rbin_entry(start + float(i));
        vec4 ca = hal_rc_fetch(e * 6.0);
        vec4 cb = hal_rc_fetch(e * 6.0 + 2.0);
        vec4 cc = hal_rc_fetch(e * 6.0 + 4.0);
        float xa = ca.x; float ya = ca.y;
        float xb = cb.x; float yb = cb.y;
        float xc = cc.x; float yc = cc.y;
        float ar = (xb - xa) * (yc - ya) - (xc - xa) * (yb - ya);
        if (abs(ar) <= 1e-9) { continue; }
        if (cull > 0.5 && cull < 1.5 && ar <= 0.0) { continue; }
        if (cull > 1.5 && ar >= 0.0) { continue; }
        // the CPU tests only pixels inside the CLAMPED bounding box
        float bxmin = max(floor(min(min(xa, xb), xc)), 0.0);
        float bxmax = ceil(max(max(xa, xb), xc));
        float bymin = max(floor(min(min(ya, yb), yc)), 0.0);
        float bymax = ceil(max(max(ya, yb), yc));
        float ix = X - 0.5;
        float iy = Y - 0.5;
        if (ix < bxmin || ix > bxmax || iy < bymin || iy > bymax) {
            continue;
        }
        float e0 = (xc - xb) * (Y - yb) - (yc - yb) * (X - xb);
        float e1 = (xa - xc) * (Y - yc) - (ya - yc) * (X - xc);
        float e2 = (xb - xa) * (Y - ya) - (yb - ya) * (X - xa);
        bool inside = (ar > 0.0)
            ? (e0 >= 0.0 && e1 >= 0.0 && e2 >= 0.0)
            : (e0 <= 0.0 && e1 <= 0.0 && e2 <= 0.0);
        if (!inside) { continue; }
        float inv_area = 1.0 / ar;
        float l0 = e0 * inv_area;
        float l1 = e1 * inv_area;
        float l2 = e2 * inv_area;
        float zz = l0 * ca.z + l1 * cb.z + l2 * cc.z;
        // PER-PIXEL depth quantization, the same formula the CPU's
        // quantize_depth applies -- roundEven matches NumPy's
        // half-to-even, so both rasterisers round half-cases alike
        if (hal_zsteps > 0.5) {
            zz = roundEven((zz * 0.5 + 0.5) * hal_zsteps)
                 / hal_zsteps * 2.0 - 1.0;
        }
        if (zz < best_z) {
            best_z = zz;
            win = e;
            wl0 = l0;
            wl1 = l1;
        }
    }
    return vec4(win, wl0, wl1, best_z);
}

// the winner's outputs, exactly as fill() writes them: perspective-correct
// barycentrics over the ORIGINAL triangle, the source id, the front flag
// ids  = (b0, b1, 1-b0-b1, src_tri)  -- byte for byte what pack_ids packs
// aux  = (zndc, front, b2, 0) -- b2 is the CPU's OWN third barycentric, so
// the reconstructed G-buffer carries fill()'s exact value, not 1-b0-b1
void hal_raster_resolve(vec4 winner, vec2 pix,
                        out vec4 ids, out vec4 aux)
{
    if (winner.x < -0.5) {
        ids = vec4(0.0, 0.0, 0.0, -1.0);
        aux = vec4(1.0, 1.0, 0.0, 0.0);
        return;
    }
    float e = winner.x;
    vec4 ca = hal_rc_fetch(e * 6.0);
    vec4 cb = hal_rc_fetch(e * 6.0 + 2.0);
    vec4 cc = hal_rc_fetch(e * 6.0 + 4.0);
    vec4 ba = hal_rc_fetch(e * 6.0 + 1.0);
    vec4 bb = hal_rc_fetch(e * 6.0 + 3.0);
    vec4 bc = hal_rc_fetch(e * 6.0 + 5.0);
    float l0 = winner.y;
    float l1 = winner.z;
    float l2 = 1.0 - l0 - l1;
    // recompute l2 the CPU's way: e2*inv_area, not 1-l0-l1
    float xa = ca.x; float ya = ca.y;
    float xb = cb.x; float yb = cb.y;
    float xc = cc.x; float yc = cc.y;
    float ar = (xb - xa) * (yc - ya) - (xc - xa) * (yb - ya);
    float e2f = (xb - xa) * (pix.y - ya) - (yb - ya) * (pix.x - xa);
    l2 = e2f / ar;
    float invw = l0 * ca.w + l1 * cb.w + l2 * cc.w;
    if (abs(invw) < 1e-20) { invw = 1e-20; }
    float p0 = l0 * ca.w / invw;
    float p1 = l1 * cb.w / invw;
    float p2 = l2 * cc.w / invw;
    vec3 b = p0 * ba.xyz + p1 * bb.xyz + p2 * bc.xyz;
    float front = (ar < 0.0) ? 1.0 : 0.0;
    ids = vec4(b.x, b.y, 1.0 - b.x - b.y, ba.w);
    aux = vec4(winner.w, front, b.z, 0.0);
}
"""

#: fragment-style wrapper: one tile at a time, bin range as uniforms so the
#: loop bound is uniform across lanes -- what the NumPy front-end can run
FRAGMENT_SOURCE = KERNEL_CORE + """
uniform vec2 hal_pix;                // this lane's pixel centre
uniform float hal_bin_start;
uniform float hal_bin_count;
uniform float hal_cull;
out vec4 Ids;
out vec4 Aux;

void main()
{
    vec4 win = hal_raster_pixel(hal_pix, hal_bin_start, hal_bin_count,
                                hal_cull);
    vec4 ids;
    vec4 aux;
    hal_raster_resolve(win, hal_pix, ids, aux);
    Ids = ids;
    Aux = aux;
}
"""

#: exact-fetch data reads for the COMPUTE build only. The reflected-frame
#: field rounds proved that texture() with computed normalized coordinates
#: can misread row-boundary-adjacent texels of a data texture on a real
#: driver (95 deterministic wrong rays at one texture side, zero at
#: another). This rasteriser's measured zeros were earned at the sides its
#: scenes happened to produce; texelFetch with integer coordinates removes
#: the size lottery. The FRAGMENT source keeps texture(): it is what the
#: NumPy front-end verifies, and it is byte-identical to what it always
#: was.
_EXACT_RC = """
vec4 hal_rc_fetch(float index)
{
    int side = int(hal_rc_side);
    int i = int(index);
    return texelFetch(hal_rc, ivec2(i % side, i / side), 0);
}
"""

_EXACT_BIN = """
float hal_rbin_entry(float i)
{
    int side = int(hal_rbins_side);
    int t = int(i) / 4;
    vec4 v = texelFetch(hal_rbins, ivec2(t % side, t / side), 0);
    int c = int(i) - t * 4;
    if (c == 0) { return v.x; }
    if (c == 1) { return v.y; }
    if (c == 2) { return v.z; }
    return v.w;
}
"""


def _exact_core():
    """KERNEL_CORE with its two data fetchers swapped for texelFetch."""
    old_rc = KERNEL_CORE[KERNEL_CORE.index('vec4 hal_rc_fetch'):
                         KERNEL_CORE.index('float hal_rbin_entry')]
    old_bin = KERNEL_CORE[KERNEL_CORE.index('float hal_rbin_entry'):
                          KERNEL_CORE.index('// walk one pixel')]
    src = KERNEL_CORE.replace(old_rc, _EXACT_RC + '\n')
    src = src.replace(old_bin, _EXACT_BIN + '\n')
    assert 'texelFetch' in src and src != KERNEL_CORE
    return src


#: compute wrapper: bin ranges per pixel from the tile texture; the driver
#: runs this one. Images are written, samplers are read -- by texelFetch,
#: per the note above (the tiles fetch included)
COMPUTE_SOURCE = _exact_core() + """
uniform sampler2D hal_rtiles;        // per tile: (start, count, 0, 0)
uniform float hal_rtiles_w;
uniform float hal_rtiles_h;
uniform float hal_cull;
uniform float hal_rw;
uniform float hal_rh;

void main()
{
    ivec2 xy = ivec2(gl_GlobalInvocationID.xy);
    if (float(xy.x) >= hal_rw || float(xy.y) >= hal_rh) { return; }
    vec2 pix = vec2(float(xy.x) + 0.5, float(xy.y) + 0.5);
    vec4 trange = texelFetch(hal_rtiles,
                             ivec2(xy.x / TILE_I, xy.y / TILE_I), 0);
    vec4 win = hal_raster_pixel(pix, trange.x, trange.y, hal_cull);
    vec4 ids;
    vec4 aux;
    hal_raster_resolve(win, pix, ids, aux);
    imageStore(hal_out_ids, xy, ids);
    imageStore(hal_out_aux, xy, aux);
}
""".replace('TILE_I', str(TILE))


def _square(texels):
    side = int(np.ceil(np.sqrt(max(texels, 1))))
    return side


def pack_raster_inputs(sx, sy, iw, z, bw, src, tri_map, width, height):
    """Pack build_screen_tris' outputs for the kernel.

    Returns (corners, c_side, bins, b_side, tiles, tw, th). `src` is mapped
    through `tri_map` here, so the id the kernel writes is the final one.
    """
    e = sx.shape[0]
    src_final = np.asarray(src, np.int64)
    if tri_map is not None:
        src_final = np.asarray(tri_map, np.int64)[src_final]

    c_side = _square(e * 6)
    corners = np.zeros((c_side * c_side, 4), np.float32)
    for c in range(3):
        base = np.arange(e) * 6 + c * 2
        corners[base, 0] = sx[:, c]
        corners[base, 1] = sy[:, c]
        corners[base, 2] = z[:, c]
        corners[base, 3] = iw[:, c]
        corners[base + 1, :3] = bw[:, c]
        corners[base + 1, 3] = src_final.astype(np.float32)
    corners = corners.reshape(c_side, c_side, 4)

    # --- binning: conservative bbox coverage, submission order preserved
    tw = (width + TILE - 1) // TILE
    th = (height + TILE - 1) // TILE
    x0 = np.minimum(np.minimum(sx[:, 0], sx[:, 1]), sx[:, 2])
    x1 = np.maximum(np.maximum(sx[:, 0], sx[:, 1]), sx[:, 2])
    y0 = np.minimum(np.minimum(sy[:, 0], sy[:, 1]), sy[:, 2])
    y1 = np.maximum(np.maximum(sy[:, 0], sy[:, 1]), sy[:, 2])
    tx0 = np.clip(np.floor(x0).astype(np.int64) // TILE, 0, tw - 1)
    tx1 = np.clip(np.ceil(x1).astype(np.int64) // TILE, 0, tw - 1)
    ty0 = np.clip(np.floor(y0).astype(np.int64) // TILE, 0, th - 1)
    ty1 = np.clip(np.ceil(y1).astype(np.int64) // TILE, 0, th - 1)
    degenerate = ~(np.isfinite(x0) & np.isfinite(x1)
                   & np.isfinite(y0) & np.isfinite(y1))

    # vectorised (tile, triangle) pair expansion: the Python triple loop
    # here was most of the pack cost the self-test measured
    keep = np.nonzero(~degenerate)[0]
    if keep.size:
        kx0, kx1 = tx0[keep], tx1[keep]
        ky0, ky1 = ty0[keep], ty1[keep]
        nx = kx1 - kx0 + 1
        ny = ky1 - ky0 + 1
        per = nx * ny
        total = int(per.sum())
        rep = np.repeat(np.arange(keep.size), per)
        block_start = np.zeros(keep.size, np.int64)
        block_start[1:] = np.cumsum(per)[:-1]
        k = np.arange(total, dtype=np.int64) - block_start[rep]
        nx_rep = nx[rep]
        dx = k % nx_rep
        dy = k // nx_rep
        pt = (ky0[rep] + dy) * tw + (kx0[rep] + dx)
        pe = keep[rep]
        order = np.argsort(pt, kind='stable')     # keeps tri order per tile
        pt = pt[order]
        pe = pe[order]
    else:
        pt = np.zeros(0, np.int64)
        pe = np.zeros(0, np.int64)

    counts = np.bincount(pt, minlength=tw * th)
    starts = np.zeros(tw * th, np.int64)
    starts[1:] = np.cumsum(counts)[:-1]

    b_side = _square(int(np.ceil(pe.size / 4.0)))
    bins = np.zeros((b_side * b_side * 4,), np.float32)
    bins[:pe.size] = pe.astype(np.float32)
    bins = bins.reshape(b_side, b_side, 4)

    tiles = np.zeros((th, tw, 4), np.float32)
    tiles[:, :, 0] = starts.reshape(th, tw)
    tiles[:, :, 1] = counts.reshape(th, tw)
    return corners, c_side, bins, b_side, tiles, tw, th


def simulate_raster(sx, sy, iw, z, bw, src, tri_map, width, height,
                    cull='NONE', depth_bits=32):
    """Run the kernel through Halcyon's own front-end, tile by tile.

    Returns (tri (H,W) int32, bary (H,W,3), zndc (H,W), front (H,W) bool,
    b2 (H,W)) -- the G-buffer the driver's dispatch would produce, computed
    without a driver. bary[..., 2] carries the CPU's own third barycentric
    (from aux), not 1-b0-b1. The per-tile shape keeps the kernel's loop
    bound uniform across lanes, which is the one thing the front-end's SIMT
    model requires.
    """
    from ..core.texture import Texture
    from ..shaders.compiler import try_compile

    corners, c_side, bins, b_side, tiles, tw, th = pack_raster_inputs(
        sx, sy, iw, z, bw, src, tri_map, width, height)
    prog, err = try_compile(FRAGMENT_SOURCE, 'GLSL')
    if prog is None:
        raise RuntimeError(f'raster kernel does not compile: {err}')

    tri = np.full((height, width), -1, np.int32)
    bary = np.zeros((height, width, 3), np.float32)
    zndc = np.full((height, width), 1.0, np.float32)
    front = np.ones((height, width), bool)
    b2 = np.zeros((height, width), np.float32)
    cull_f = {'NONE': 0.0, 'BACK': 1.0, 'FRONT': 2.0}.get(cull, 0.0)
    zsteps = float((1 << int(max(2, depth_bits))) - 1) \
        if depth_bits < 32 else 0.0

    tex = {'hal_rc': Texture(corners, colorspace='Non-Color',
                             filt='NEAREST', wrap='EXTEND'),
           'hal_rbins': Texture(bins, colorspace='Non-Color',
                                filt='NEAREST', wrap='EXTEND')}
    for tyy in range(th):
        for txx in range(tw):
            start = float(tiles[tyy, txx, 0])
            count = float(tiles[tyy, txx, 1])
            if count < 0.5:
                continue
            xs = np.arange(txx * TILE, min((txx + 1) * TILE, width))
            ys = np.arange(tyy * TILE, min((tyy + 1) * TILE, height))
            X, Y = np.meshgrid(xs, ys)
            n = X.size
            pix = np.stack([X.ravel() + 0.5, Y.ravel() + 0.5],
                           1).astype(np.float32)
            uni = dict(tex)
            uni['hal_rc_side'] = np.full(n, float(c_side), np.float32)
            uni['hal_rbins_side'] = np.full(n, float(b_side), np.float32)
            uni['hal_pix'] = pix
            uni['hal_bin_start'] = np.full(n, start, np.float32)
            uni['hal_bin_count'] = np.full(n, count, np.float32)
            uni['hal_cull'] = np.full(n, cull_f, np.float32)
            uni['hal_zsteps'] = np.full(n, zsteps, np.float32)
            outs = prog.run(uni, {}, n)[0]
            ids = outs['Ids']
            aux = outs['Aux']
            yy = Y.ravel()
            xx = X.ravel()
            tri[yy, xx] = np.round(ids[:, 3]).astype(np.int32)
            bary[yy, xx, 0] = ids[:, 0]
            bary[yy, xx, 1] = ids[:, 1]
            bary[yy, xx, 2] = aux[:, 2]      # the CPU's own b2
            zndc[yy, xx] = aux[:, 0]
            front[yy, xx] = aux[:, 1] > 0.5
            b2[yy, xx] = aux[:, 2]
    return tri, bary, zndc, front, b2


def raster_inputs_for(scene_mesh, vp, width, height, near_eps=1e-5,
                      depth_bits=24, snap=0.0, subset=None):
    """build_screen_tris on a mesh, exactly as rasterize() calls it.

    With `subset`, the triangle list is cut down and the ORIGINAL indices
    ride along as the tri_map -- rasterize()'s own convention -- so the ids
    the kernel writes are final. Returns (sx, sy, iw, z, bw, src, tri_map).
    """
    from ..core import raster as CR
    tris = np.asarray(scene_mesh.tris, np.int32)
    tri_map = None
    if subset is not None:
        tri_map = np.asarray(subset, np.int32)
        tris = tris[tri_map]
    clip, _s, _iw, _z = CR.project(scene_mesh.verts, vp, width, height,
                                   snap=0.0, near_eps=near_eps)
    sx, sy, iw, z, bw, src = CR.build_screen_tris(
        clip, tris, width, height, snap=snap, near_eps=near_eps,
        depth_bits=depth_bits)
    return sx, sy, iw, z, bw, src, tri_map


def raster_on_device(mesh, vp, width, height, cull='NONE', snap=0.0,
                     depth_bits=24, subset=None):
    """Pack, upload, dispatch, read back: the raster on the real driver.

    Returns ({'ids', 'aux', 'timings'}, None) or (None, why). `timings`
    splits the milliseconds -- clip+project, pack+bin, upload, dispatch and
    read -- because "56 ms" is a number and a split is a diagnosis.
    """
    import time as _time

    from . import device

    t0 = _time.perf_counter()
    sx, sy, iw, z, bw, src, tri_map = raster_inputs_for(
        mesh, vp, width, height, depth_bits=depth_bits, snap=snap,
        subset=subset)
    t_clip = _time.perf_counter() - t0
    t0 = _time.perf_counter()
    corners, c_side, bins, b_side, tiles, tw, th = pack_raster_inputs(
        sx, sy, iw, z, bw, src, tri_map, width, height)
    t_pack = _time.perf_counter() - t0
    shader, err = device.compile_compute(
        'HAL_RASTER', COMPUTE_SOURCE,
        samplers=('hal_rc', 'hal_rbins', 'hal_rtiles'),
        floats=('hal_rc_side', 'hal_rbins_side', 'hal_rtiles_w',
                'hal_rtiles_h', 'hal_cull', 'hal_rw', 'hal_rh',
                'hal_zsteps'),
        images=('hal_out_ids', 'hal_out_aux'))
    if shader is None:
        return None, err
    try:
        t0 = _time.perf_counter()
        t_rc = device.upload(corners)
        t_bins = device.upload(bins)
        t_tiles = device.upload(tiles)
        t_upload = _time.perf_counter() - t0
        cull_f = {'NONE': 0.0, 'BACK': 1.0, 'FRONT': 2.0}.get(cull, 0.0)
        t0 = _time.perf_counter()
        out = device.dispatch_compute(
            shader, width, height,
            uniforms={'hal_rc_side': float(c_side),
                      'hal_rbins_side': float(b_side),
                      'hal_rtiles_w': float(tw), 'hal_rtiles_h': float(th),
                      'hal_cull': cull_f,
                      'hal_rw': float(width), 'hal_rh': float(height),
                      'hal_zsteps': float((1 << int(max(2, depth_bits))) - 1)
                      if depth_bits < 32 else 0.0},
            samplers={'hal_rc': t_rc, 'hal_rbins': t_bins,
                      'hal_rtiles': t_tiles},
            images=('hal_out_ids', 'hal_out_aux'))
        t_run = _time.perf_counter() - t0
    except Exception as exc:                                    # noqa: BLE001
        return None, f'raster dispatch failed: {type(exc).__name__}: {exc}'
    return {'ids': out['hal_out_ids'], 'aux': out['hal_out_aux'],
            'timings': {'clip_ms': t_clip * 1000.0,
                        'pack_ms': t_pack * 1000.0,
                        'upload_ms': t_upload * 1000.0,
                        'dispatch_read_ms': t_run * 1000.0}}, None


def gbuffer_into(gbuf, ids, aux):
    """Fill a GBuffer from the kernel's two images, fill()'s conventions.

    Covered pixels carry the winner; empty ones keep the CPU's own init
    values -- depth +inf (the transparent pass depth-tests against it),
    zndc 1.0, front True. b2 comes from aux, where the kernel put the
    CPU's OWN third barycentric rather than 1-b0-b1.
    """
    tri = np.rint(ids[:, :, 3]).astype(np.int32)
    cov = tri >= 0
    gbuf.tri[:] = tri
    gbuf.bary[:, :, 0] = ids[:, :, 0]
    gbuf.bary[:, :, 1] = ids[:, :, 1]
    gbuf.bary[:, :, 2] = aux[:, :, 2]
    gbuf.depth[:] = np.where(cov, aux[:, :, 0], np.inf)
    gbuf.zndc[:] = np.where(cov, aux[:, :, 0], 1.0)
    gbuf.front[:] = aux[:, :, 1] > 0.5
    return gbuf


def raster_into_gbuffer(mesh, vp, width, height, gbuf, cull='NONE',
                        snap=0.0, depth_bits=24, subset=None):
    """The render() hook: rasterise on the driver, reconstruct the GBuffer.

    Returns (True, None) on success or (False, why); the caller falls back
    to the CPU rasteriser and prints the reason. Qualification (bands,
    Painter's, overdraw, affine texture mode) is the caller's job -- this
    function is mechanism.
    """
    out, why = raster_on_device(mesh, vp, width, height, cull=cull,
                                snap=snap, depth_bits=depth_bits,
                                subset=subset)
    if out is None:
        return False, why
    gbuffer_into(gbuf, out['ids'], out['aux'])
    return True, None
