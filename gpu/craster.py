"""The compute rasteriser: fill()'s exact rules, one thread per pixel.

Hardware rasterisation can never reproduce this renderer's G-buffer -- the
fill conventions differ at triangle edges -- so the port is the CPU's own
algorithm as a compute kernel. The design that makes exactness POSSIBLE is
per-pixel sequential resolve: the screen is cut into tiles, triangles are
binned per tile on the CPU (cheap, vectorised), and every pixel walks its
tile's bin with the strict `<` depth test and THE NAMED TIE RULE -- equal
depth goes to the lowest triangle id. The rule is order-free by
construction, which is what lets this kernel, the loop fill AND the
batched fill land the same winner: "first tested wins" silently depended
on each path's internal order (the batched path draws big triangles
first), and at quantised depths -- where exact ties are common -- the two
CPU paths themselves disagreed on 3 pixels of the demo scene until the
rule was named.

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

uniform float hal_refer;             // 1 = mark fragile decisions (aux.w)

// walk one pixel's bin in submission order; fill()'s exact rules.
// returns (winner emitted-tri index or -1, l0, l1, depth); `fragile`
// comes back 1.0 when this pixel's DECISION sat inside a cross-device
// noise window -- a candidate depth within an ulp-wobble of a
// quantisation boundary, two candidates within a couple of quantised
// steps of each other (coincident surfaces on shared steps), a pixel
// centre within a sliver of a triangle edge, or a triangle at the
// degenerate-area gate. The caller replays marked pixels with the
// CPU's own arithmetic: the raster tie referral, exactly the ray
// referral's shape (name the noise-window decisions, route them to
// the reference).
vec4 hal_raster_pixel(vec2 pix, float start, float count, float cull,
                      out float fragile)
{
    float best_z = 1e30;
    float min_v = 1e30;
    float second_v = 1e30;
    float win = -1.0;
    float win_src = 1e30;
    float wl0 = 0.0;
    float wl1 = 0.0;
    float frag = 0.0;
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
        if (abs(ar) <= 1e-9) {
            // a 1-ulp ar on the other device could clear this gate
            if (hal_refer > 0.5 && abs(ar) > 5e-10) { frag = 1.0; }
            continue;
        }
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
        // product magnitudes: the units float wobble is measured in.
        // Coverage below widens by a few ulps of these, so a shared
        // edge cannot be excluded by BOTH triangles (the watertight
        // rule; holes along dense-mesh edges were the field's faint
        // wireframe). The referral reuses them for its sliver window
        float w0 = abs((xc - xb) * (Y - yb)) + abs((yc - yb) * (X - xb));
        float w1 = abs((xa - xc) * (Y - yc)) + abs((ya - yc) * (X - xc));
        float w2 = abs((xb - xa) * (Y - ya)) + abs((yb - ya) * (X - xa));
        if (hal_refer > 0.5) {
            // sliver window: coverage flips when a driver's fma
            // contraction moves an edge function across zero. The
            // wobble is at most a few ulps of the PRODUCT magnitudes,
            // so the window is sized to exactly that -- not to the
            // area, not to a guess. An e of exactly 0.0 is carved out
            // ONLY when the exactness is PROVABLE: all four factors of
            // that edge exact half-integers below 2^21 (integer vertex
            // snapping plus half-integer pixel centres), where the
            // products and their difference are exactly representable
            // and fma cannot move them. The first carve-out trusted
            // e == 0.0 unconditionally, and the field flipped one
            // snapped pixel whose zero came from INEXACT clipped
            // corners rounding to it -- exactly the hole this test
            // closes.
            float d0a = xc - xb; float d0b = Y - yb;
            float d0c = yc - yb; float d0d = X - xb;
            float d1a = xa - xc; float d1b = Y - yc;
            float d1c = ya - yc; float d1d = X - xc;
            float d2a = xb - xa; float d2b = Y - ya;
            float d2c = yb - ya; float d2d = X - xa;
            float m0 = w0;
            float m1 = w1;
            float m2 = w2;
            float x0 = (fract(abs(d0a) * 2.0) == 0.0
                        && fract(abs(d0b) * 2.0) == 0.0
                        && fract(abs(d0c) * 2.0) == 0.0
                        && fract(abs(d0d) * 2.0) == 0.0
                        && m0 < 2097152.0) ? 1.0 : 0.0;
            float x1 = (fract(abs(d1a) * 2.0) == 0.0
                        && fract(abs(d1b) * 2.0) == 0.0
                        && fract(abs(d1c) * 2.0) == 0.0
                        && fract(abs(d1d) * 2.0) == 0.0
                        && m1 < 2097152.0) ? 1.0 : 0.0;
            float x2 = (fract(abs(d2a) * 2.0) == 0.0
                        && fract(abs(d2b) * 2.0) == 0.0
                        && fract(abs(d2c) * 2.0) == 0.0
                        && fract(abs(d2d) * 2.0) == 0.0
                        && m2 < 2097152.0) ? 1.0 : 0.0;
            if ((x0 < 0.5 && abs(e0) < 2.5e-7 * m0)
                    || (x1 < 0.5 && abs(e1) < 2.5e-7 * m1)
                    || (x2 < 0.5 && abs(e2) < 2.5e-7 * m2)) {
                frag = 1.0;
            }
        }
        bool inside = (ar > 0.0)
            ? (e0 >= -2.5e-7 * w0 && e1 >= -2.5e-7 * w1
               && e2 >= -2.5e-7 * w2)
            : (e0 <= 2.5e-7 * w0 && e1 <= 2.5e-7 * w1
               && e2 <= 2.5e-7 * w2);
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
            float v = (zz * 0.5 + 0.5) * hal_zsteps;
            // the RAW pre-rounding value, tracked for the fragility
            // gate below: a cross-device winner flip requires the two
            // best raw values to sit within the arithmetic wobble of
            // each other -- boundary proximity alone flips nothing
            if (hal_refer > 0.5) {
                if (v < min_v) { second_v = min_v; min_v = v; }
                else if (v < second_v) { second_v = v; }
            }
            zz = roundEven(v) / hal_zsteps * 2.0 - 1.0;
        }
        if (zz < best_z) {
            best_z = zz;
            win = e;
            win_src = hal_rc_fetch(e * 6.0 + 1.0).w;
            wl0 = l0;
            wl1 = l1;
        } else if (zz == best_z && win > -0.5) {
            // THE NAMED TIE RULE, shared with both CPU fill paths:
            // equal depth goes to the LOWEST triangle id -- order-free,
            // so an exact quantised tie is NOT fragile by itself
            float s = hal_rc_fetch(e * 6.0 + 1.0).w;
            if (s < win_src) {
                win = e;
                win_src = s;
                wl0 = l0;
                wl1 = l1;
            }
        }
    }
    if (hal_refer > 0.5 && hal_zsteps > 0.5 && win > -0.5
            && (second_v - min_v)
               <= (0.25 + hal_zsteps * 2.5e-6)) {
        // the z fragility window, in RAW value units: the depth wobble
        // between devices is ulp-scale in zz (a few e-7), which is
        // hal_zsteps * ~1e-7 in v units -- MANY steps at 24 bits, a
        // fraction of one at 16. The first window was sized in STEPS
        // and the field flipped a snapped 24-bit pixel straight through
        // it: at fine quantisation the wobble dwarfs the step, and only
        // a raw-gap window scaled to the arithmetic (plus a quarter
        // step for the coarse-bits rounding case) names every fragile
        // competition at every depth precision.
        frag = 1.0;
    }
    fragile = frag;
    return vec4(win, wl0, wl1, best_z);
}

// the winner's outputs, exactly as fill() writes them: perspective-correct
// barycentrics over the ORIGINAL triangle, the source id, the front flag
// ids  = (b0, b1, 1-b0-b1, src_tri)  -- byte for byte what pack_ids packs
// aux  = (zndc, front, b2, fragile) -- b2 is the CPU's OWN third
// barycentric; fragile is the referral mark hal_raster_pixel raised
// lin  = (lb0, lb1, lb2, 0) -- the SCREEN-LINEAR barycentrics over the
// original triangle (l . bw, no perspective division): the affine
// texture warp's own interpolants, fill()'s bary_lin
void hal_raster_resolve(vec4 winner, vec2 pix, float fragile,
                        out vec4 ids, out vec4 aux, out vec4 lin)
{
    if (winner.x < -0.5) {
        ids = vec4(0.0, 0.0, 0.0, -1.0);
        aux = vec4(1.0, 1.0, 0.0, fragile);
        lin = vec4(0.0);
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
    vec3 lb = l0 * ba.xyz + l1 * bb.xyz + l2 * bc.xyz;
    float front = (ar < 0.0) ? 1.0 : 0.0;
    ids = vec4(b.x, b.y, 1.0 - b.x - b.y, ba.w);
    aux = vec4(winner.w, front, b.z, fragile);
    lin = vec4(lb, 0.0);
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
out vec4 Lin;

void main()
{
    float fragile = 0.0;
    vec4 win = hal_raster_pixel(hal_pix, hal_bin_start, hal_bin_count,
                                hal_cull, fragile);
    vec4 ids;
    vec4 aux;
    vec4 lin;
    hal_raster_resolve(win, hal_pix, fragile, ids, aux, lin);
    Ids = ids;
    Aux = aux;
    Lin = lin;
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
_COMPUTE_SHELL = """
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
    float fragile = 0.0;
    vec4 win = hal_raster_pixel(pix, trange.x, trange.y, hal_cull,
                                fragile);
    vec4 ids;
    vec4 aux;
    vec4 lin;
    hal_raster_resolve(win, pix, fragile, ids, aux, lin);
    imageStore(hal_out_ids, xy, ids);
    imageStore(hal_out_aux, xy, aux);
    LIN_STORE
}
""".replace('TILE_I', str(TILE))

COMPUTE_SOURCE = _exact_core() + _COMPUTE_SHELL.replace('LIN_STORE', '')
#: the affine variant: a third image carries the screen-linear
#: barycentrics. A separate compiled shader because image bindings are
#: part of the compile signature; perspective frames never pay for it.
COMPUTE_SOURCE_LIN = _exact_core() + _COMPUTE_SHELL.replace(
    'LIN_STORE', 'imageStore(hal_out_lin, xy, lin);')


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
                    cull='NONE', depth_bits=32, refer=False):
    """Run the kernel through Halcyon's own front-end, tile by tile.

    Returns (tri (H,W) int32, bary (H,W,3), zndc (H,W), front (H,W) bool,
    b2 (H,W), lin (H,W,3), mark (H,W) bool) -- the G-buffer the driver's
    dispatch would produce, computed without a driver. bary[..., 2]
    carries the CPU's own third barycentric (from aux), not 1-b0-b1;
    `lin` carries the screen-linear barycentrics (the affine warp's
    interpolants); `mark` the referral flags when `refer`. The per-tile
    shape keeps the kernel's loop bound uniform across lanes, which is
    the one thing the front-end's SIMT model requires.
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
    lin = np.zeros((height, width, 3), np.float32)
    mark = np.zeros((height, width), bool)
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
            uni['hal_refer'] = np.full(n, 1.0 if refer else 0.0,
                                       np.float32)
            outs = prog.run(uni, {}, n)[0]
            ids = outs['Ids']
            aux = outs['Aux']
            lin_o = outs['Lin']
            yy = Y.ravel()
            xx = X.ravel()
            tri[yy, xx] = np.round(ids[:, 3]).astype(np.int32)
            bary[yy, xx, 0] = ids[:, 0]
            bary[yy, xx, 1] = ids[:, 1]
            bary[yy, xx, 2] = aux[:, 2]      # the CPU's own b2
            zndc[yy, xx] = aux[:, 0]
            front[yy, xx] = aux[:, 1] > 0.5
            b2[yy, xx] = aux[:, 2]
            lin[yy, xx, :] = lin_o[:, :3]
            mark[yy, xx] = aux[:, 3] > 0.5
    return tri, bary, zndc, front, b2, lin, mark


def replay_pixels(pxs, pys, sx, sy, iw, z, bw, src_final,
                  tiles, bins_flat, tw, cull, depth_bits):
    """The raster tie referral's CPU half: re-decide marked pixels.

    Vectorised over (marked pixel x bin entry) with the CPU fill's own
    float32 expressions -- e_i, l_i = e_i * (1/ar), zz = l.z,
    quantize_depth, and the named tie rule (equal depth -> lowest
    triangle id) -- so the answer at a fragile pixel is the
    reference's own, not the driver's last bit, at NumPy speed rather
    than a Python loop's. Returns (tri, bary, lin, zndc, front) over
    the given pixels; tri -1 where nothing covers.
    """
    from ..core.raster import quantize_depth
    f32 = np.float32
    n = int(pxs.size)
    out_tri = np.full(n, -1, np.int32)
    out_b = np.zeros((n, 3), np.float32)
    out_lb = np.zeros((n, 3), np.float32)
    out_z = np.full(n, 1.0, np.float32)
    out_front = np.ones(n, bool)
    if n == 0:
        return out_tri, out_b, out_lb, out_z, out_front

    starts = tiles[:, :, 0].ravel().astype(np.int64)
    counts = tiles[:, :, 1].ravel().astype(np.int64)
    t_idx = (pys.astype(np.int64) // TILE) * tw + \
        (pxs.astype(np.int64) // TILE)
    s0 = starts[t_idx]
    cnt = counts[t_idx]
    maxc = int(cnt.max()) if cnt.size else 0
    if maxc == 0:
        return out_tri, out_b, out_lb, out_z, out_front
    lane = np.arange(maxc, dtype=np.int64)[None, :]
    valid = lane < cnt[:, None]
    entry = np.where(valid, s0[:, None] + lane, 0)
    T = bins_flat[entry].astype(np.int64)               # (n, maxc)

    X = (pxs.astype(np.float32) + f32(0.5))[:, None]
    Y = (pys.astype(np.float32) + f32(0.5))[:, None]
    xa, xb, xc = sx[T, 0], sx[T, 1], sx[T, 2]
    ya, yb, yc = sy[T, 0], sy[T, 1], sy[T, 2]
    ar = (xb - xa) * (yc - ya) - (xc - xa) * (yb - ya)
    ok = valid & (np.abs(ar) > 1e-9)
    if cull == 'BACK':
        ok &= ar > 0.0
    elif cull == 'FRONT':
        ok &= ar < 0.0
    bxmin = np.maximum(np.floor(np.minimum(np.minimum(xa, xb), xc)), 0.0)
    bxmax = np.ceil(np.maximum(np.maximum(xa, xb), xc))
    bymin = np.maximum(np.floor(np.minimum(np.minimum(ya, yb), yc)), 0.0)
    bymax = np.ceil(np.maximum(np.maximum(ya, yb), yc))
    ix = X - f32(0.5)
    iy = Y - f32(0.5)
    ok &= (ix >= bxmin) & (ix <= bxmax) & (iy >= bymin) & (iy <= bymax)
    e0 = (xc - xb) * (Y - yb) - (yc - yb) * (X - xb)
    e1 = (xa - xc) * (Y - yc) - (ya - yc) * (X - xc)
    e2 = (xb - xa) * (Y - ya) - (yb - ya) * (X - xa)
    # the watertight window, bit-for-bit the kernel's own (raster.fill
    # tells the story)
    from ..core.raster import EDGE_WOBBLE as _EW
    w0 = np.abs((xc - xb) * (Y - yb)) + np.abs((yc - yb) * (X - xb))
    w1 = np.abs((xa - xc) * (Y - yc)) + np.abs((ya - yc) * (X - xc))
    w2 = np.abs((xb - xa) * (Y - ya)) + np.abs((yb - ya) * (X - xa))
    pos = ar > 0.0
    inside = np.where(pos,
                      (e0 >= _EW * -w0) & (e1 >= _EW * -w1)
                      & (e2 >= _EW * -w2),
                      (e0 <= _EW * w0) & (e1 <= _EW * w1)
                      & (e2 <= _EW * w2))
    ok &= inside
    inv_area = f32(1.0) / np.where(ar == 0.0, f32(1.0), ar)
    l0 = e0 * inv_area
    l1 = e1 * inv_area
    l2 = e2 * inv_area
    zz = l0 * z[T, 0] + l1 * z[T, 1] + l2 * z[T, 2]
    if depth_bits < 32:
        zz = quantize_depth(zz, depth_bits)
    src_t = src_final[T]

    # the winner: minimum quantised depth, ties to the lowest triangle
    # id, then to the earliest bin entry (submission order) -- exactly
    # the batched resolve's stable lexsort
    zbig = np.where(ok, zz, np.float32(np.inf))
    zmin = zbig.min(axis=1)
    covered = np.isfinite(zmin)
    at_min = ok & (zbig == zmin[:, None])
    sbig = np.where(at_min, src_t, np.int64(2**62))
    smin = sbig.min(axis=1)
    pick_mask = at_min & (src_t == smin[:, None])
    pick = np.argmax(pick_mask, axis=1)                 # first True
    rows = np.arange(n)
    tW = T[rows, pick]
    l0w = l0[rows, pick]
    l1w = l1[rows, pick]
    l2w = l2[rows, pick]
    arw = ar[rows, pick]
    iw0, iw1, iw2 = iw[tW, 0], iw[tW, 1], iw[tW, 2]
    invw = l0w * iw0 + l1w * iw1 + l2w * iw2
    invw = np.where(np.abs(invw) < 1e-20, f32(1e-20), invw)
    p0 = l0w * iw0 / invw
    p1 = l1w * iw1 / invw
    p2 = l2w * iw2 / invw
    bwt = bw[tW]                                        # (n, 3, 3)
    P = np.stack([p0, p1, p2], axis=1).astype(np.float32)
    L = np.stack([l0w, l1w, l2w], axis=1).astype(np.float32)
    b_all = np.einsum('nk,nkc->nc', P, bwt).astype(np.float32)
    lb_all = np.einsum('nk,nkc->nc', L, bwt).astype(np.float32)

    out_tri[covered] = src_t[rows, pick][covered].astype(np.int32)
    out_b[covered] = b_all[covered]
    out_lb[covered] = lb_all[covered]
    out_z[covered] = zmin[covered].astype(np.float32)
    out_front[covered] = (arw < 0.0)[covered]
    return out_tri, out_b, out_lb, out_z, out_front


def raster_inputs_for(scene_mesh, vp, width, height, near_eps=1e-5,
                      depth_bits=24, snap=0.0, subset=None, subdiv_px=0):
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
        depth_bits=depth_bits, subdiv_px=subdiv_px)
    return sx, sy, iw, z, bw, src, tri_map


def raster_on_device(mesh, vp, width, height, cull='NONE', snap=0.0,
                     depth_bits=24, subset=None, subdiv_px=0,
                     near_eps=1e-5, want_lin=False, refer=False):
    """Pack, upload, dispatch, read back: the raster on the real driver.

    Returns ({'ids', 'aux', 'lin'?, 'inputs', 'timings'}, None) or
    (None, why). `want_lin` runs the affine variant (a third image with
    the screen-linear barycentrics); `refer` turns the fragility marks
    on (aux.w). `inputs` carries the packed arrays so the caller can
    run the tie-referral replay without re-packing. `timings` splits
    the milliseconds -- clip+project, pack+bin, upload, dispatch and
    read -- because "56 ms" is a number and a split is a diagnosis.
    """
    import time as _time

    from . import device

    t0 = _time.perf_counter()
    sx, sy, iw, z, bw, src, tri_map = raster_inputs_for(
        mesh, vp, width, height, near_eps=near_eps,
        depth_bits=depth_bits, snap=snap,
        subset=subset, subdiv_px=subdiv_px)
    t_clip = _time.perf_counter() - t0
    t0 = _time.perf_counter()
    corners, c_side, bins, b_side, tiles, tw, th = pack_raster_inputs(
        sx, sy, iw, z, bw, src, tri_map, width, height)
    t_pack = _time.perf_counter() - t0
    images = ('hal_out_ids', 'hal_out_aux') + \
        (('hal_out_lin',) if want_lin else ())
    shader, err = device.compile_compute(
        'HAL_RASTER_LIN' if want_lin else 'HAL_RASTER',
        COMPUTE_SOURCE_LIN if want_lin else COMPUTE_SOURCE,
        samplers=('hal_rc', 'hal_rbins', 'hal_rtiles'),
        floats=('hal_rc_side', 'hal_rbins_side', 'hal_rtiles_w',
                'hal_rtiles_h', 'hal_cull', 'hal_rw', 'hal_rh',
                'hal_zsteps', 'hal_refer'),
        images=images)
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
                      if depth_bits < 32 else 0.0,
                      'hal_refer': 1.0 if refer else 0.0},
            samplers={'hal_rc': t_rc, 'hal_rbins': t_bins,
                      'hal_rtiles': t_tiles},
            images=images)
        t_run = _time.perf_counter() - t0
        dd = dict(getattr(device, 'LAST_DISPATCH', None) or {})
    except Exception as exc:                                    # noqa: BLE001
        return None, f'raster dispatch failed: {type(exc).__name__}: {exc}'
    # src mapped exactly as the packer maps it, for the replay
    src_final = np.asarray(src, np.int64)
    if tri_map is not None:
        src_final = np.asarray(tri_map, np.int64)[src_final]
    return {'ids': out['hal_out_ids'], 'aux': out['hal_out_aux'],
            'lin': out.get('hal_out_lin') if want_lin else None,
            'inputs': (sx, sy, iw, z, bw, src_final,
                       tiles, bins.reshape(-1), tw),
            'timings': {'clip_ms': t_clip * 1000.0,
                        'pack_ms': t_pack * 1000.0,
                        'upload_ms': t_upload * 1000.0,
                        'dispatch_ms': dd.get('dispatch_ms',
                                              t_run * 1000.0),
                        'read_ms': dd.get('read_ms', 0.0),
                        'dispatch_read_ms': t_run * 1000.0}}, None


def gbuffer_into(gbuf, ids, aux, lin=None):
    """Fill a GBuffer from the kernel's images, fill()'s conventions.

    Covered pixels carry the winner; empty ones keep the CPU's own init
    values -- depth +inf (the transparent pass depth-tests against it),
    zndc 1.0, front True. b2 comes from aux, where the kernel put the
    CPU's OWN third barycentric rather than 1-b0-b1. `lin` (the affine
    variant's third image) fills bary_lin when the gbuf carries one.
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
    if lin is not None and gbuf.bary_lin is not None:
        gbuf.bary_lin[:, :, :] = lin[:, :, :3]
    return gbuf


#: referral bail-out: when fragile pixels exceed this fraction of the
#: frame, the replay stops being a footnote and the frame falls back
#: whole, with the count in the printed reason. The replay is fully
#: vectorised, so the budget is generous; a frame past it is
#: pathological (everything coincident with everything).
REFER_BAIL_FRAC = 0.10


def raster_into_gbuffer(mesh, vp, width, height, gbuf, cull='NONE',
                        snap=0.0, depth_bits=24, subset=None,
                        subdiv_px=0, near_eps=1e-5):
    """The render() hook: rasterise on the driver, reconstruct the GBuffer.

    Returns (True, None) on success or (False, why); the caller falls back
    to the CPU rasteriser and prints the reason. Qualification (bands,
    Painter's, overdraw) is the caller's job -- this function is
    mechanism. Affine frames run the lin variant (bary_lin filled when
    the gbuf carries one); quantised-depth and snapped frames run with
    the fragility marks on, and marked pixels are REPLAYED with the CPU
    fill's own arithmetic -- the raster tie referral. LAST_REFERRED
    carries the replay count for the tests and the self-test.
    """
    want_lin = gbuf.bary_lin is not None
    refer = int(depth_bits) < 24 or float(snap) > 0.0
    out, why = raster_on_device(mesh, vp, width, height, cull=cull,
                                snap=snap, depth_bits=depth_bits,
                                subset=subset, subdiv_px=subdiv_px,
                                near_eps=near_eps,
                                want_lin=want_lin, refer=refer)
    if out is None:
        return False, why
    LAST_REFERRED['count'] = 0
    if refer:
        mark = out['aux'][:, :, 3] > 0.5
        n_mark = int(mark.sum())
        if n_mark:
            covered = max(int((np.rint(out['ids'][:, :, 3]) >= 0).sum()),
                          1)
            if n_mark > REFER_BAIL_FRAC * covered:
                return False, (f'{n_mark} fragile pixels of {covered} '
                               f'covered under quantised depth -- past '
                               f'the referral budget, the CPU '
                               f'rasterises this frame')
            pys, pxs = np.nonzero(mark)
            sx, sy, iw, z, bw, src_final, tiles, bins_flat, tw = \
                out['inputs']
            r_tri, r_b, r_lb, r_z, r_front = replay_pixels(
                pxs, pys, sx, sy, iw, z, bw, src_final,
                tiles, bins_flat, tw, cull, depth_bits)
            ids, aux = out['ids'], out['aux']
            ids[pys, pxs, 0] = r_b[:, 0]
            ids[pys, pxs, 1] = r_b[:, 1]
            ids[pys, pxs, 3] = r_tri.astype(np.float32)
            aux[pys, pxs, 0] = np.where(r_tri >= 0, r_z, 1.0)
            aux[pys, pxs, 1] = r_front.astype(np.float32)
            aux[pys, pxs, 2] = r_b[:, 2]
            if out.get('lin') is not None:
                out['lin'][pys, pxs, :3] = r_lb
            LAST_REFERRED['count'] = n_mark
            # the field instrument: one line names how many pixels the
            # referral handed back to the CPU's own arithmetic
            print(f'[Halcyon GPU] raster tie referral: {n_mark} of '
                  f'{covered} covered pixels replayed with the CPU '
                  f"fill's arithmetic")
    import time as _time
    t0 = _time.perf_counter()
    gbuffer_into(gbuf, out['ids'], out['aux'], out.get('lin'))
    tm = dict(out.get('timings') or {})
    tm['decode_ms'] = (_time.perf_counter() - t0) * 1000.0
    if tm.get('read_ms'):
        # the device gave the finer split; the aggregate would double-print
        tm.pop('dispatch_read_ms', None)
    LAST_RASTER.clear()
    LAST_RASTER.update(tm)
    return True, None


#: how many pixels the last driver raster referred to the CPU replay
LAST_REFERRED = {'count': 0}

#: the last driver raster's stage split in milliseconds (clip_ms, pack_ms,
#: upload_ms, dispatch_read_ms, decode_ms) -- "218 ms" is a number, a split
#: is a diagnosis. render.py prints it under the frame breakdown so a
#: high-resolution F12 names which half of the road got slow
LAST_RASTER = {}
