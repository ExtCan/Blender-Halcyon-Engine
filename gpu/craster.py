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
#: R251 C001: the coverage block of hal_raster_resolve is a placeholder
#: (CVG_CODE) that `kernel_core(cvg)` fills -- the 16 subsample tests are
#: assembled into the source ONLY under the cvg flag, so a default frame
#: pays nothing for a plane nobody reads; KERNEL_CORE is the cvg-off text
_KERNEL_TEMPLATE = """
uniform sampler2D hal_rc;            // packed corners, two texels per corner
uniform float hal_rc_side;
uniform sampler2D hal_rbins;         // triangle indices, four per texel
uniform float hal_rbins_side;
uniform float hal_zsteps;            // z-buffer grid steps (0 = 32-bit off)
// R251: the flat (Painter's) depth read and the depth-encoding family.
// Every function here sits BEFORE hal_rc_fetch on purpose: the compute
// build replaces the two data-fetch spans by string index and keeps
// everything above them
uniform float hal_flat;              // 1 = corner z holds the polygon's one depth (C004)
uniform float hal_zenc;              // 0 LINEAR, 1 N64_FLOAT18, 2/3/4 GC_14E2/13E3/12E4, 5 W_FIXED, 6 VOODOO_W16
uniform float hal_wob;               // the encoding referral window, pre-floor units
uniform float hal_wnear;             // W encodings: the camera's near plane (eye depth)
uniform float hal_wrange;            // W_FIXED: far - near
uniform float hal_wsteps;            // W_FIXED: 2^bits - 1 (its OWN steps; hal_zsteps is 0 under every encoding)
uniform float hal_jit;               // C127: 1 = jittered sample positions (REYES)
uniform float hal_jit_key;           // C127: the 24-bit (frame, seed) hash key

// C127: film._hash_u32_raw's GLSL twin -- a VERBATIM copy of the GRAIN
// stage's hal_grain_hash (gpu/stages.py), so two copies of one hash
// cannot drift; the uint arithmetic wraps at 2^32 on both roads
uint hal_jit_hash(uint x, uint y, int k)
{
    uint h = x * 0x9E3779B1u ^ (y + 0x85EBCA77u) ^ (uint(k) * 0xC2B2AE3Du);
    h ^= h >> 15u;
    h *= 0x2C1B3C6Du;
    h ^= h >> 12u;
    h *= 0x297A2D39u;
    h ^= h >> 15u;
    return h & 0xffffffu;
}

// C127: the per-pixel sample offset, (n + 0.5) * 2^-12 - 0.5 from two
// 12-bit fields of the hash -- exact in float32, never on a cell edge
vec2 hal_jitter(vec2 pix)
{
    if (hal_jit < 0.5) { return vec2(0.0); }
    uint ix = uint(pix.x - 0.5);  uint iy = uint(pix.y - 0.5);
    uint h = hal_jit_hash(ix, iy, int(hal_jit_key));
    float jx = float(h & 0xfffu);  jx = jx + 0.5;  jx = jx * 0.000244140625;  jx = jx - 0.5;
    float jy = float((h >> 12u) & 0xfffu);  jy = jy + 0.5;  jy = jy * 0.000244140625;  jy = jy - 0.5;
    return vec2(jx, jy);
}

// C007: the RDP's 14-bit piecewise-floating code of an 18-bit screen z
// (angrylion rdp/zbuffer.c z_build_com_table): 64-unit steps over the
// near half, then 32, 16, 8, 4, 2, 1 as z approaches the far end
float hal_zkey_n64(float zz, out float v, out float stepc)
{
    float zn = zz * 0.5;  zn = zn + 0.5;  zn = clamp(zn, 0.0, 1.0);
    float zf = zn * 262143.0;  v = zf;
    int z18 = int(floor(zf));
    int e = z18 >> 11;
    int shift = 0;
    if (e < 64) { shift = 6; } else if (e < 96) { shift = 5; } else if (e < 112) { shift = 4; }
    else if (e < 120) { shift = 3; } else if (e < 124) { shift = 2; } else if (e < 126) { shift = 1; }
    int code = (z18 >> shift) << shift;
    stepc = float(1 << shift);
    return float(code);
}

// C026: the GameCube's compressed 24-bit z -- leading ones from bit 23
// (capped by the exponent field) select how many mantissa bits survive
float hal_zkey_gc(float zz, float mbits, float ebits, out float v, out float stepc)
{
    float zn = zz * 0.5;  zn = zn + 0.5;  zn = clamp(zn, 0.0, 1.0);
    float zf = zn * 16777215.0;  v = zf;
    int z24 = int(floor(zf));
    int cap = (1 << int(ebits)) - 1;
    int e = 0;
    for (int i = 0; i < 15; i++) {
        if (i >= cap) { break; }
        if (((z24 >> (23 - i)) & 1) == 1) { e = e + 1; } else { break; }
    }
    int kept = e + ((e < cap) ? 1 : 0) + int(mbits);
    int drop = max(24 - kept, 0);
    int code = (z24 >> drop) << drop;
    stepc = float(1 << drop);
    return float(code);
}

// C026: the fixed-point W-buffer (Xbox NV2A, D3DZB_USEW): eye depth w
// on a 2^bits grid between near and far
float hal_zkey_w(float invw, out float v, out float stepc)
{
    float w = 1.0 / invw;
    float t = w - hal_wnear;  t = t / hal_wrange;  t = clamp(t, 0.0, 1.0);
    v = t * hal_wsteps;  stepc = 1.0;
    return floor(v);
}

// C075: 3dfx's 16-bit floating W (MAME voodoo_render.cpp compute_wfloat)
// on the float 1/w normalised to the near plane: the octave by exact
// doubling, 12 inverted mantissa bits, plus one
float hal_zkey_voodoo(float invw, out float v, out float stepc)
{
    float t = invw * hal_wnear;
    float x = t;
    int d = 0;
    for (int i = 0; i < 17; i++) { if (x < 1.0) { x = x * 2.0; d = d + 1; } }
    int e = max(d - 1, 0);
    float m = x - 1.0;  m = m * 4096.0;  v = m;  stepc = 1.0;
    int m12 = int(floor(m));
    int d16 = ((e << 12) | (4095 - m12)) + 1;
    d16 = min(d16, 65535);
    if (t < 1.52587890625e-05) { d16 = 65535; }
    if (t >= 1.0) { d16 = 0; }
    return float(d16);
}

// the dispatch on hal_zenc (1..6); v = the pre-floor value, stepc = the
// code's step at that value, both for the two-candidate referral
float hal_zkey(float zz, float invw, out float v, out float stepc)
{
    if (hal_zenc < 1.5) { return hal_zkey_n64(zz, v, stepc); }
    if (hal_zenc < 2.5) { return hal_zkey_gc(zz, 14.0, 2.0, v, stepc); }
    if (hal_zenc < 3.5) { return hal_zkey_gc(zz, 13.0, 3.0, v, stepc); }
    if (hal_zenc < 4.5) { return hal_zkey_gc(zz, 12.0, 4.0, v, stepc); }
    if (hal_zenc < 5.5) { return hal_zkey_w(invw, v, stepc); }
    return hal_zkey_voodoo(invw, v, stepc);
}

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
vec4 hal_raster_pixel(vec2 pix, vec2 jit, float start, float count,
                      float cull, float clear_key, out float fragile)
{
    // C038: the z-buffer starts at the pixel's CLEAR (the DS rear-plane
    // depth bitmap's key; 1e30 without one): a fragment must be strictly
    // nearer to draw, and the tie branch below never replaces an
    // unbeaten backdrop (win stays -1) -- the CPU fill's own rule
    float best_z = clear_key;
    float min_v = 1e30;
    float second_v = 1e30;
    // R251: the two smallest encoding KEYS with their boundary distance
    // and step (the two-candidate referral of the depth encodings). The
    // STEPS start at 0.0, not 1e30: a lone candidate leaves k2 at 1e30
    // and `k2 - k1` rounds to 1e30, which a 1e30 step would let through
    // (the whole frame marked -- measured, 7566 of 11844 covered px)
    float k1 = 1e30; float d1 = 1e30; float s1 = 0.0;
    float k2 = 1e30; float d2 = 1e30; float s2 = 0.0;
    if (hal_refer > 0.5 && clear_key < 1e29) {
        // C038: the clear is a candidate of the referral windows -- a
        // fragment within the wobble of the backdrop's own value may
        // flip across it on the other device
        if (hal_zenc > 0.5) { k1 = clear_key; d1 = 0.0; s1 = 0.0; }
        else if (hal_zsteps > 0.5) { min_v = (clear_key * 0.5 + 0.5) * hal_zsteps; }
    }
    float win = -1.0;
    float win_src = 1e30;
    float wl0 = 0.0;
    float wl1 = 0.0;
    float frag = 0.0;
    // C127: the sample sits at the (jittered) position; the bbox test
    // below keeps the integer cell (ix = pix.x - 0.5)
    float X = pix.x + jit.x;
    float Y = pix.y + jit.y;
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
        float ix = pix.x - 0.5;
        float iy = pix.y - 0.5;
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
        float zz;
        if (hal_flat > 0.5) { zz = ca.z; } else { zz = l0 * ca.z + l1 * cb.z + l2 * cc.z; }
        // PER-PIXEL depth quantization, the same formula the CPU's
        // quantize_depth applies -- roundEven matches NumPy's
        // half-to-even, so both rasterisers round half-cases alike.
        // R251: under a depth ENCODING the code replaces zz and the
        // LINEAR block never runs (the host passes hal_zsteps 0.0)
        if (hal_zenc > 0.5) {
            float invw_c = l0 * ca.w + l1 * cb.w + l2 * cc.w;
            if (abs(invw_c) < 1e-20) { invw_c = 1e-20; }
            float stepc = 1.0;
            float v = 0.0;
            float key = hal_zkey(zz, invw_c, v, stepc);
            if (hal_refer > 0.5) {
                // distance of the pre-floor value to the nearest
                // boundary of its OWN step; the two smallest keys kept
                float m = v - stepc * floor(v / stepc);
                float dist = min(m, stepc - m);
                if (key < k1) { k2 = k1; d2 = d1; s2 = s1; k1 = key; d1 = dist; s1 = stepc; }
                else if (key < k2) { k2 = key; d2 = dist; s2 = stepc; }
            }
            zz = key;
        } else if (hal_zsteps > 0.5) {
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
    if (hal_refer > 0.5 && hal_zenc > 0.5 && win > -0.5
            && k2 - k1 <= max(s1, s2) && min(d1, d2) <= hal_wob) {
        // R251: the flip condition of a truncating code -- the winner
        // can change only when the two best keys lie within one step
        // of each other (equal keys included) AND one pre-floor value
        // sits within the arithmetic wobble of a step boundary. A lone
        // candidate (k2 = 1e30) never marks
        frag = 1.0;
    }
    if (hal_refer > 0.5 && hal_zsteps > 0.5 && hal_flat < 0.5 && win > -0.5
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
void hal_raster_resolve(vec4 winner, vec2 pix, vec2 jit, float fragile,
                        float clear_key, out vec4 ids, out vec4 aux,
                        out vec4 lin, out vec4 cvgv)
{
    // C001: uncovered pixels are FULLY covered (7) on both roads, so the
    // VI blends silhouettes toward the sky and never touches the sky
    cvgv = vec4(7.0, 0.0, 0.0, 0.0);
    if (winner.x < -0.5) {
        ids = vec4(0.0, 0.0, 0.0, -1.0);
        // C038: an unbeaten backdrop reports its own key (the host
        // decodes it beside the winners'); 1.0 without one, as before
        aux = vec4((clear_key < 1e29) ? clear_key : 1.0, 1.0, 0.0, fragile);
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
    // C127: the sample position (the centre plus the jitter) for the
    // barycentrics; the coverage block below reads the centre `pix`
    float sX = pix.x + jit.x;
    float sY = pix.y + jit.y;
    float e2f = (xb - xa) * (sY - ya) - (yb - ya) * (sX - xa);
    l2 = e2f / ar;
    float invw = l0 * ca.w + l1 * cb.w + l2 * cc.w;
    if (abs(invw) < 1e-20) { invw = 1e-20; }
    float p0 = l0 * ca.w / invw;
    float p1 = l1 * cb.w / invw;
    float p2 = l2 * cc.w / invw;
    vec3 b = p0 * ba.xyz + p1 * bb.xyz + p2 * bc.xyz;
    vec3 lb = l0 * ba.xyz + l1 * bb.xyz + l2 * bc.xyz;
    float front = (ar < 0.0) ? 1.0 : 0.0;
    float frag2 = fragile;
    CVG_CODE
    ids = vec4(b.x, b.y, 1.0 - b.x - b.y, ba.w);
    aux = vec4(winner.w, front, b.z, frag2);
    lin = vec4(lb, 0.0);
}
"""

#: C001: the RDP's coverage of the winner at the PIXEL CENTRE (not the
#: jittered sample: the RDP's subsamples are relative to the pixel) --
#: the fill's own three edge expressions at 16 subsamples, both edges
#: inclusive, no wobble window (a count, not coverage), then
#: clip((n16 + 1) / 2, 1, 8) - 1. Under hal_refer the same loop ORs the
#: extended sliver mark (any subsample's edge function within the
#: wobble window of the product magnitudes) into the fragility flag, so
#: a marked pixel's replay recomputes coverage16 on the CPU
_CVG_CODE = """
    float cx = pix.x - 0.0;  // the pixel centre (the wrappers pass it)
    float cy = pix.y - 0.0;
    float n16 = 0.0;
    float sliver = 0.0;
    for (int i = 0; i < 4; i++) {
        float oy = float(i) * 0.25 - 0.375;
        for (int j = 0; j < 4; j++) {
            float ox = float(j) * 0.25 - 0.375;
            float Xs = cx + ox;
            float Ys = cy + oy;
            float s0 = (xc - xb) * (Ys - yb) - (yc - yb) * (Xs - xb);
            float s1 = (xa - xc) * (Ys - yc) - (ya - yc) * (Xs - xc);
            float s2 = (xb - xa) * (Ys - ya) - (yb - ya) * (Xs - xa);
            bool ins = (ar > 0.0) ? (s0 >= 0.0 && s1 >= 0.0 && s2 >= 0.0)
                                  : (s0 <= 0.0 && s1 <= 0.0 && s2 <= 0.0);
            if (ins) { n16 = n16 + 1.0; }
            if (hal_refer > 0.5) {
                float m0 = abs((xc - xb) * (Ys - yb)) + abs((yc - yb) * (Xs - xb));
                float m1 = abs((xa - xc) * (Ys - yc)) + abs((ya - yc) * (Xs - xc));
                float m2 = abs((xb - xa) * (Ys - ya)) + abs((yb - ya) * (Xs - xa));
                if (abs(s0) < 2.5e-7 * m0 || abs(s1) < 2.5e-7 * m1
                        || abs(s2) < 2.5e-7 * m2) { sliver = 1.0; }
            }
        }
    }
    if (sliver > 0.5) { frag2 = 1.0; }
    float cvg = clamp(floor((n16 + 1.0) * 0.5), 1.0, 8.0);
    cvgv = vec4(cvg - 1.0, 0.0, 0.0, 0.0);
"""
_CVG_OFF = ""

_CORE_CACHE = {}


def kernel_core(cvg=False):
    """The shared core with the coverage block in (`cvg`) or out."""
    key = bool(cvg)
    hit = _CORE_CACHE.get(key)
    if hit is None:
        hit = _CORE_CACHE[key] = _KERNEL_TEMPLATE.replace(
            'CVG_CODE', _CVG_CODE if cvg else _CVG_OFF)
    return hit


KERNEL_CORE = kernel_core(False)

#: fragment-style wrapper: one tile at a time, bin range as uniforms so the
#: loop bound is uniform across lanes -- what the NumPy front-end can run
_FRAGMENT_SHELL = """
uniform vec2 hal_pix;                // this lane's pixel centre
uniform float hal_bin_start;
uniform float hal_bin_count;
uniform float hal_cull;
uniform float hal_clear_key;         // C038: this lane's z-buffer clear (1e30 = none)
out vec4 Ids;
out vec4 Aux;
out vec4 Lin;
CVG_OUT
void main()
{
    float fragile = 0.0;
    vec2 jit = hal_jitter(hal_pix);
    vec4 win = hal_raster_pixel(hal_pix, jit, hal_bin_start, hal_bin_count,
                                hal_cull, hal_clear_key, fragile);
    vec4 ids;
    vec4 aux;
    vec4 lin;
    vec4 cvgv;
    hal_raster_resolve(win, hal_pix, jit, fragile, hal_clear_key, ids, aux,
                       lin, cvgv);
    Ids = ids;
    Aux = aux;
    Lin = lin;
    CVG_STORE
}
"""


def fragment_source(cvg=False):
    """The simulator's wrapper, with the coverage output under `cvg`."""
    src = _FRAGMENT_SHELL.replace('CVG_OUT\n', 'out vec4 Cvg;\n' if cvg else '')
    src = src.replace('CVG_STORE', 'Cvg = cvgv;' if cvg else '')
    return kernel_core(cvg) + src


FRAGMENT_SOURCE = fragment_source(False)

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


def _exact_core(cvg=False):
    """kernel_core(cvg) with its two data fetchers swapped for texelFetch."""
    core = kernel_core(cvg)
    old_rc = core[core.index('vec4 hal_rc_fetch'):
                  core.index('float hal_rbin_entry')]
    old_bin = core[core.index('float hal_rbin_entry'):
                   core.index('// walk one pixel')]
    src = core.replace(old_rc, _EXACT_RC + '\n')
    src = src.replace(old_bin, _EXACT_BIN + '\n')
    assert 'texelFetch' in src and src != core
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
CLEAR_DECL
void main()
{
    ivec2 xy = ivec2(gl_GlobalInvocationID.xy);
    if (float(xy.x) >= hal_rw || float(xy.y) >= hal_rh) { return; }
    vec2 pix = vec2(float(xy.x) + 0.5, float(xy.y) + 0.5);
    vec4 trange = texelFetch(hal_rtiles,
                             ivec2(xy.x / TILE_I, xy.y / TILE_I), 0);
    float fragile = 0.0;
    vec2 jit = hal_jitter(pix);
    CLEAR_READ
    vec4 win = hal_raster_pixel(pix, jit, trange.x, trange.y, hal_cull,
                                ck, fragile);
    vec4 ids;
    vec4 aux;
    vec4 lin;
    vec4 cvgv;
    hal_raster_resolve(win, pix, jit, fragile, ck, ids, aux, lin, cvgv);
    imageStore(hal_out_ids, xy, ids);
    imageStore(hal_out_aux, xy, aux);
    LIN_STORE
    CVG_STORE
}
""".replace('TILE_I', str(TILE))

#: C038: the rear-plane depth bitmap's key, one texel per pixel (.x)
_CLEAR_DECL = 'uniform sampler2D hal_rclear;       // C038: the z-buffer clear (key per pixel)\n'
_CLEAR_READ = 'float ck = texelFetch(hal_rclear, xy, 0).x;'
_NO_CLEAR_READ = 'float ck = 1e30;'

_COMPUTE_CACHE = {}


def compute_variant_name(lin=False, clear=False, cvg=False):
    """The compiled shader's name: image and sampler bindings are part
    of the compile signature (gpu/device.py), so every flag set is its
    own name."""
    return (f'HAL_RASTER_L{1 if lin else 0}C{1 if clear else 0}'
            f'V{1 if cvg else 0}')


def compute_source(lin=False, clear=False, cvg=False):
    """The compute rasteriser's source for one flag set: `lin` adds the
    third image (the screen-linear barycentrics), `clear` (C038) the
    `hal_rclear` sampler read per pixel, `cvg` (C001) the coverage
    block and its fourth image `hal_out_cvg`. Cached by the flag triple."""
    key = (bool(lin), bool(clear), bool(cvg))
    hit = _COMPUTE_CACHE.get(key)
    if hit is None:
        src = _COMPUTE_SHELL
        src = src.replace('CLEAR_DECL\n', _CLEAR_DECL if clear else '')
        src = src.replace('CLEAR_READ', _CLEAR_READ if clear
                          else _NO_CLEAR_READ)
        src = src.replace('LIN_STORE',
                          'imageStore(hal_out_lin, xy, lin);' if lin else '')
        src = src.replace('CVG_STORE',
                          'imageStore(hal_out_cvg, xy, cvgv);' if cvg else '')
        hit = _COMPUTE_CACHE[key] = _exact_core(cvg) + src
    return hit


COMPUTE_SOURCE = compute_source(False, False, False)
#: the affine variant: a third image carries the screen-linear
#: barycentrics. A separate compiled shader because image bindings are
#: part of the compile signature; perspective frames never pay for it.
COMPUTE_SOURCE_LIN = compute_source(True, False, False)


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


def flat_corner_z(z, src, tri_map, flat_depth):
    """C004: every corner texel carries the polygon's ONE depth.

    `src` is mapped through `tri_map` exactly as the packer maps it, so
    the flat value is looked up by the final id -- the CPU fill's own
    `flat_depth[src_tri]`, the same float32 bits.
    """
    src_final = np.asarray(src, np.int64)
    if tri_map is not None:
        src_final = np.asarray(tri_map, np.int64)[src_final]
    fd = np.asarray(flat_depth, np.float32)[src_final]
    return np.repeat(fd[:, None], 3, axis=1).astype(np.float32)


def kernel_uniforms(opts, depth_bits, flat):
    """The R251 kernel uniforms for one dispatch, host-computed."""
    from ..core import raster as CR
    enc = str(opts.enc) if opts is not None else 'LINEAR'
    u = CR.encoding_uniforms(opts, depth_bits)
    # THE LINEAR QUANTISE BLOCK IS BYPASSED under every encoding: a key
    # must never be re-quantised (review 4.B1)
    u['hal_zsteps'] = float((1 << int(max(2, depth_bits))) - 1) \
        if (depth_bits < 32 and enc == 'LINEAR') else 0.0
    u['hal_flat'] = 1.0 if flat else 0.0
    # C127: the jittered sample positions (a 24-bit key rides the float
    # push constant exactly)
    jit = getattr(opts, 'jitter', None) if opts is not None else None
    u['hal_jit'] = 1.0 if jit is not None else 0.0
    u['hal_jit_key'] = float(CR.jitter_key(jit[0], jit[1])) \
        if jit is not None else 0.0
    return u


def simulate_raster(sx, sy, iw, z, bw, src, tri_map, width, height,
                    cull='NONE', depth_bits=32, refer=False, opts=None,
                    flat_depth=None, want_cvg=False):
    """Run the kernel through Halcyon's own front-end, tile by tile.

    Returns (tri (H,W) int32, bary (H,W,3), zndc (H,W), front (H,W) bool,
    b2 (H,W), lin (H,W,3), mark (H,W) bool) -- the G-buffer the driver's
    dispatch would produce, computed without a driver. bary[..., 2]
    carries the CPU's own third barycentric (from aux), not 1-b0-b1;
    `lin` carries the screen-linear barycentrics (the affine warp's
    interpolants); `mark` the referral flags when `refer`. The per-tile
    shape keeps the kernel's loop bound uniform across lanes, which is
    the one thing the front-end's SIMT model requires.

    R251: `opts` (a RasterOpts) selects the depth encoding -- under one,
    the `zndc` plane returned holds the KEYS (aux.x), which the host
    decodes with `raster.decode_key`, exactly as `gbuffer_into` does;
    `flat_depth` (Painter's) replaces every corner z by the polygon's
    one depth (`hal_flat`); `want_cvg` (C001) returns the RDP's 3-bit
    coverage plane (uint8 (H, W), 7 where uncovered) as an EIGHTH value
    -- the seven 1.89.0 values and their unpackings stay as they are.
    """
    from ..core.texture import Texture
    from ..shaders.compiler import try_compile

    if flat_depth is not None:
        z = flat_corner_z(z, src, tri_map, flat_depth)
    ku = kernel_uniforms(opts, depth_bits, flat_depth is not None)
    corners, c_side, bins, b_side, tiles, tw, th = pack_raster_inputs(
        sx, sy, iw, z, bw, src, tri_map, width, height)
    prog, err = try_compile(fragment_source(bool(want_cvg)), 'GLSL')
    if prog is None:
        raise RuntimeError(f'raster kernel does not compile: {err}')
    cvg_out = np.full((height, width), 7, np.uint8) if want_cvg else None

    tri = np.full((height, width), -1, np.int32)
    bary = np.zeros((height, width, 3), np.float32)
    zndc = np.full((height, width), 1.0, np.float32)
    front = np.ones((height, width), bool)
    b2 = np.zeros((height, width), np.float32)
    lin = np.zeros((height, width, 3), np.float32)
    mark = np.zeros((height, width), bool)
    cull_f = {'NONE': 0.0, 'BACK': 1.0, 'FRONT': 2.0}.get(cull, 0.0)
    zsteps = ku['hal_zsteps']
    # C038: the z-buffer clear per pixel (the backdrop's key plane)
    clear_key = None
    if opts is not None and getattr(opts, 'clear', None) is not None:
        clear_key = np.asarray(opts.clear[0], np.float32)

    tex = {'hal_rc': Texture(corners, colorspace='Non-Color',
                             filt='NEAREST', wrap='EXTEND'),
           'hal_rbins': Texture(bins, colorspace='Non-Color',
                                filt='NEAREST', wrap='EXTEND')}
    for tyy in range(th):
        for txx in range(tw):
            start = float(tiles[tyy, txx, 0])
            count = float(tiles[tyy, txx, 1])
            if count < 0.5 and clear_key is None:
                # an empty tile keeps the init values -- unless a clear
                # (C038) gives its pixels a key to report, as the compute
                # shell does for every pixel
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
            for nm in ('hal_flat', 'hal_zenc', 'hal_wob', 'hal_wnear',
                       'hal_wrange', 'hal_wsteps', 'hal_jit', 'hal_jit_key'):
                uni[nm] = np.full(n, ku[nm], np.float32)
            if clear_key is not None:
                uni['hal_clear_key'] = clear_key[Y.ravel(), X.ravel()].astype(np.float32)
            else:
                uni['hal_clear_key'] = np.full(n, 1e30, np.float32)
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
            if cvg_out is not None:
                cvg_out[yy, xx] = np.rint(outs['Cvg'][:, 0]).astype(np.uint8)
    if want_cvg:
        return tri, bary, zndc, front, b2, lin, mark, cvg_out
    return tri, bary, zndc, front, b2, lin, mark


def replay_pixels(pxs, pys, sx, sy, iw, z, bw, src_final,
                  tiles, bins_flat, tw, cull, depth_bits, opts=None,
                  flat=False, want_cvg=False):
    """The raster tie referral's CPU half: re-decide marked pixels.

    Vectorised over (marked pixel x bin entry) with the CPU fill's own
    float32 expressions -- e_i, l_i = e_i * (1/ar), zz = l.z,
    quantize_depth, and the named tie rule (equal depth -> lowest
    triangle id) -- so the answer at a fragile pixel is the
    reference's own, not the driver's last bit, at NumPy speed rather
    than a Python loop's. Returns (tri, bary, lin, zndc, front) over
    the given pixels; tri -1 where nothing covers.

    R251: under `opts.enc` the candidates are compared on the CODE
    (`raster.encode_depth`) and `zndc` returns the KEY, so aux.x holds
    keys on both roads and the host decodes once; `flat` reads the
    packed corner z as the polygon's one depth (Painter's).
    """
    from ..core.raster import coverage16, encode_depth, quantize_depth
    f32 = np.float32
    enc_on = opts is not None and str(opts.enc) != 'LINEAR'
    n = int(pxs.size)
    out_tri = np.full(n, -1, np.int32)
    out_b = np.zeros((n, 3), np.float32)
    out_lb = np.zeros((n, 3), np.float32)
    out_z = np.full(n, 1.0, np.float32)
    out_front = np.ones(n, bool)
    out_cvg = np.full(n, 7, np.int32)

    def _ret():
        if want_cvg:
            return out_tri, out_b, out_lb, out_z, out_front, out_cvg
        return out_tri, out_b, out_lb, out_z, out_front
    if n == 0:
        return _ret()

    starts = tiles[:, :, 0].ravel().astype(np.int64)
    counts = tiles[:, :, 1].ravel().astype(np.int64)
    t_idx = (pys.astype(np.int64) // TILE) * tw + \
        (pxs.astype(np.int64) // TILE)
    s0 = starts[t_idx]
    cnt = counts[t_idx]
    maxc = int(cnt.max()) if cnt.size else 0
    if maxc == 0:
        return _ret()
    lane = np.arange(maxc, dtype=np.int64)[None, :]
    valid = lane < cnt[:, None]
    entry = np.where(valid, s0[:, None] + lane, 0)
    T = bins_flat[entry].astype(np.int64)               # (n, maxc)

    X = (pxs.astype(np.float32) + f32(0.5))[:, None]
    Y = (pys.astype(np.float32) + f32(0.5))[:, None]
    ix = X - f32(0.5)
    iy = Y - f32(0.5)
    jit_p = getattr(opts, 'jitter', None) if opts is not None else None
    if jit_p is not None:
        # C127: the same offsets the fill added, at these pixels only
        from ..core.raster import jitter_at, jitter_key
        jx, jy = jitter_at(pxs.astype(np.uint32), pys.astype(np.uint32),
                           jitter_key(jit_p[0], jit_p[1]))
        X = X + jx[:, None]
        Y = Y + jy[:, None]
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
    if flat:
        zz = z[T, 0]
    else:
        zz = l0 * z[T, 0] + l1 * z[T, 1] + l2 * z[T, 2]
    if enc_on:
        invw_c = l0 * iw[T, 0] + l1 * iw[T, 1] + l2 * iw[T, 2]
        invw_c = np.where(np.abs(invw_c) < 1e-20, f32(1e-20), invw_c)
        zz, _dec = encode_depth(zz, invw_c, opts, depth_bits)
    elif depth_bits < 32:
        zz = quantize_depth(zz, depth_bits)
    src_t = src_final[T]

    # the winner: minimum quantised depth, ties to the lowest triangle
    # id, then to the earliest bin entry (submission order) -- exactly
    # the batched resolve's stable lexsort
    zbig = np.where(ok, zz, np.float32(np.inf))
    zmin = zbig.min(axis=1)
    covered = np.isfinite(zmin)
    clear_p = getattr(opts, 'clear', None) if opts is not None else None
    if clear_p is not None:
        # C038: a fragment must be strictly nearer than the backdrop's
        # key (the backdrop wins an equal key); an unbeaten backdrop
        # reports its own key, as the kernel does
        ck = np.asarray(clear_p[0], np.float32)[pys, pxs]
        covered &= zmin < ck
        out_z[:] = np.where(np.isfinite(ck), ck, np.float32(1.0))
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
    if want_cvg and covered.any():
        # C001: the winner's coverage at the pixel CENTRE (never the
        # jittered sample), the CPU fill's own coverage16
        cx = pxs.astype(np.float32) + f32(0.5)
        cy = pys.astype(np.float32) + f32(0.5)
        cv = coverage16(sx[tW, 0], sy[tW, 0], sx[tW, 1], sy[tW, 1],
                        sx[tW, 2], sy[tW, 2], arw, cx, cy)
        out_cvg[covered] = cv[covered]
    return _ret()


def raster_inputs_for(scene_mesh, vp, width, height, near_eps=1e-5,
                      depth_bits=24, snap=0.0, subset=None, subdiv_px=0,
                      opts=None):
    """build_screen_tris on a mesh, exactly as rasterize() calls it.

    With `subset`, the triangle list is cut down and the ORIGINAL indices
    ride along as the tri_map -- rasterize()'s own convention -- so the ids
    the kernel writes are final. Returns (sx, sy, iw, z, bw, src, tri_map).
    R251: `opts` (a RasterOpts) carries the pre-fill dials -- the pixel
    shift (C084), whole-triangle rejection (C027) and the size limit
    (C004) -- shared with the CPU road before either fill runs.
    """
    from ..core import raster as CR
    if opts is None:
        opts = CR.RasterOpts()
    tris = np.asarray(scene_mesh.tris, np.int32)
    tri_map = None
    if subset is not None:
        tri_map = np.asarray(subset, np.int32)
        tris = tris[tri_map]
    clip, _s, _iw, _z = CR.project(scene_mesh.verts, vp, width, height,
                                   snap=0.0, near_eps=near_eps)
    sx, sy, iw, z, bw, src = CR.build_screen_tris(
        clip, tris, width, height, snap=snap, near_eps=near_eps,
        depth_bits=depth_bits, subdiv_px=subdiv_px,
        pixel_shift=opts.pixel_shift, reject=opts.reject,
        size_limit=opts.size_limit)
    return sx, sy, iw, z, bw, src, tri_map


def raster_on_device(mesh, vp, width, height, cull='NONE', snap=0.0,
                     depth_bits=24, subset=None, subdiv_px=0,
                     near_eps=1e-5, want_lin=False, refer=False,
                     opts=None, flat_depth=None, want_cvg=False):
    """Pack, upload, dispatch, read back: the raster on the real driver.

    Returns ({'ids', 'aux', 'lin'?, 'inputs', 'timings'}, None) or
    (None, why). `want_lin` runs the affine variant (a third image with
    the screen-linear barycentrics); `refer` turns the fragility marks
    on (aux.w). `inputs` carries the packed arrays so the caller can
    run the tie-referral replay without re-packing. `timings` splits
    the milliseconds -- clip+project, pack+bin, upload, dispatch and
    read -- because "56 ms" is a number and a split is a diagnosis.

    R251: `opts` selects the depth encoding (its uniforms ride as push
    constants, 15 floats in all: 60 of the ~128 bytes) and the pre-fill
    dials; `flat_depth` (Painter's, C004) replaces every corner z by the
    polygon's one depth, so the kernel compares whole polygons.
    """
    import time as _time

    from . import device

    t0 = _time.perf_counter()
    sx, sy, iw, z, bw, src, tri_map = raster_inputs_for(
        mesh, vp, width, height, near_eps=near_eps,
        depth_bits=depth_bits, snap=snap,
        subset=subset, subdiv_px=subdiv_px, opts=opts)
    if flat_depth is not None:
        z = flat_corner_z(z, src, tri_map, flat_depth)
    ku = kernel_uniforms(opts, depth_bits, flat_depth is not None)
    t_clip = _time.perf_counter() - t0
    t0 = _time.perf_counter()
    corners, c_side, bins, b_side, tiles, tw, th = pack_raster_inputs(
        sx, sy, iw, z, bw, src, tri_map, width, height)
    t_pack = _time.perf_counter() - t0
    # binding order (device binds by index): hal_out_cvg (C001) is
    # appended LAST in images and hal_rclear (C038) LAST in samplers,
    # each under its flag, so the existing indices never move
    images = ('hal_out_ids', 'hal_out_aux') + \
        (('hal_out_lin',) if want_lin else ()) + \
        (('hal_out_cvg',) if want_cvg else ())
    clear = getattr(opts, 'clear', None) if opts is not None else None
    want_clear = clear is not None
    samplers = ('hal_rc', 'hal_rbins', 'hal_rtiles') + \
        (('hal_rclear',) if want_clear else ())
    shader, err = device.compile_compute(
        compute_variant_name(want_lin, want_clear, want_cvg),
        compute_source(want_lin, want_clear, want_cvg),
        samplers=samplers,
        floats=('hal_rc_side', 'hal_rbins_side', 'hal_rtiles_w',
                'hal_rtiles_h', 'hal_cull', 'hal_rw', 'hal_rh',
                'hal_zsteps', 'hal_refer',
                # R251: the flat read and the encoding family (6 more
                # float push constants), the jitter (2 more): 17 in all,
                # 68 of the ~128 bytes
                'hal_flat', 'hal_zenc', 'hal_wob', 'hal_wnear',
                'hal_wrange', 'hal_wsteps', 'hal_jit', 'hal_jit_key'),
        images=images)
    if shader is None:
        return None, err
    try:
        t0 = _time.perf_counter()
        t_rc = device.upload(corners)
        t_bins = device.upload(bins)
        t_tiles = device.upload(tiles)
        sampler_binds = {'hal_rc': t_rc, 'hal_rbins': t_bins,
                         'hal_rtiles': t_tiles}
        if want_clear:
            # C038: one texel per pixel, .x = the key; an unchanged
            # backdrop uploads once (keyed on its content hash)
            key_plane = np.asarray(clear[0], np.float32)

            def _clear_image(_k=key_plane):
                img = np.zeros((_k.shape[0], _k.shape[1], 4), np.float32)
                img[:, :, 0] = _k
                return img
            sampler_binds['hal_rclear'] = device.upload_cached(
                ('rclear', clear[2], int(width), int(height)), _clear_image)
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
                      # R251: 0.0 under every depth encoding (the key is
                      # never re-quantised); W_FIXED rides hal_wsteps
                      'hal_zsteps': float(ku['hal_zsteps']),
                      'hal_refer': 1.0 if refer else 0.0,
                      'hal_flat': float(ku['hal_flat']),
                      'hal_zenc': float(ku['hal_zenc']),
                      'hal_wob': float(ku['hal_wob']),
                      'hal_wnear': float(ku['hal_wnear']),
                      'hal_wrange': float(ku['hal_wrange']),
                      'hal_wsteps': float(ku['hal_wsteps']),
                      'hal_jit': float(ku['hal_jit']),
                      'hal_jit_key': float(ku['hal_jit_key'])},
            samplers=sampler_binds,
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
            'cvg': out.get('hal_out_cvg') if want_cvg else None,
            'inputs': (sx, sy, iw, z, bw, src_final,
                       tiles, bins.reshape(-1), tw),
            'timings': {'clip_ms': t_clip * 1000.0,
                        'pack_ms': t_pack * 1000.0,
                        'upload_ms': t_upload * 1000.0,
                        'dispatch_ms': dd.get('dispatch_ms',
                                              t_run * 1000.0),
                        'read_ms': dd.get('read_ms', 0.0),
                        'dispatch_read_ms': t_run * 1000.0}}, None


def gbuffer_into(gbuf, ids, aux, lin=None, opts=None, depth_bits=24,
                 cvg=None):
    """Fill a GBuffer from the kernel's images, fill()'s conventions.

    Covered pixels carry the winner; empty ones keep the CPU's own init
    values -- depth +inf (the transparent pass depth-tests against it),
    zndc 1.0, front True. b2 comes from aux, where the kernel put the
    CPU's OWN third barycentric rather than 1-b0-b1. `lin` (the affine
    variant's third image) fills bary_lin when the gbuf carries one.

    R251: under `opts.enc` aux.x holds the z-buffer's CODE; it lands in
    `gbuf.zkey` and `depth` / `zndc` receive `raster.decode_key` of it --
    the host decode, so the driver's division never enters the stored
    depth and both roads hold the CPU's own bits.
    """
    # (R168 note: masked-copyto and channel-pair variants of this body
    # were benched at field size and LOST to the plain form below --
    # the decode is memory-bandwidth-bound and numpy's where is already
    # at the floor. Measured, kept.)
    tri = np.rint(ids[:, :, 3]).astype(np.int32)
    cov = tri >= 0
    gbuf.tri[:] = tri
    gbuf.bary[:, :, 0] = ids[:, :, 0]
    gbuf.bary[:, :, 1] = ids[:, :, 1]
    gbuf.bary[:, :, 2] = aux[:, :, 2]
    # C038: the uncovered pixels carry the backdrop's own key / decoded
    # value (the host has both planes), +inf / 1.0 without a backdrop
    clear = getattr(opts, 'clear', None) if opts is not None else None
    if clear is not None:
        key_c = np.asarray(clear[0], np.float32)
        dec_c = np.asarray(clear[1], np.float32)
        zndc_c = np.where(np.isfinite(dec_c), dec_c, np.float32(1.0))
    else:
        key_c = dec_c = np.inf
        zndc_c = 1.0
    if opts is not None and str(opts.enc) != 'LINEAR':
        from ..core.raster import decode_key
        key = np.where(cov, aux[:, :, 0], key_c).astype(np.float32)
        zk = gbuf.alloc_zkey()
        zk[:] = key
        dec = decode_key(np.where(cov, aux[:, :, 0], 0.0), opts,
                         depth_bits)
        gbuf.depth[:] = np.where(cov, dec, dec_c)
        gbuf.zndc[:] = np.where(cov, dec, zndc_c)
    else:
        gbuf.depth[:] = np.where(cov, aux[:, :, 0], dec_c)
        gbuf.zndc[:] = np.where(cov, aux[:, :, 0], zndc_c)
    gbuf.front[:] = aux[:, :, 1] > 0.5
    if lin is not None and gbuf.bary_lin is not None:
        gbuf.bary_lin[:, :, :] = lin[:, :, :3]
    if cvg is not None and gbuf.cvg is not None:
        # C001: the winners' coverage; uncovered pixels are full (7)
        gbuf.cvg[:] = np.where(cov, np.rint(cvg[:, :, 0]), 7).astype(np.uint8)
    return gbuf


#: referral bail-out: when fragile pixels exceed this fraction of the
#: frame, the replay stops being a footnote and the frame falls back
#: whole, with the count in the printed reason. The replay is fully
#: vectorised, so the budget is generous; a frame past it is
#: pathological (everything coincident with everything).
REFER_BAIL_FRAC = 0.10


def raster_into_gbuffer(mesh, vp, width, height, gbuf, cull='NONE',
                        snap=0.0, depth_bits=24, subset=None,
                        subdiv_px=0, near_eps=1e-5, opts=None,
                        flat_depth=None):
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
    enc_on = opts is not None and str(opts.enc) != 'LINEAR'
    # C001: the coverage plane, when the frame asked for it (the
    # extended sliver mark rides the referral: a marked pixel's
    # coverage is the CPU's replay)
    want_cvg = gbuf.cvg is not None and opts is not None \
        and bool(getattr(opts, 'cvg', False))
    # R251: a depth ENCODING runs with the two-candidate referral on
    # (C007); a flat (Painter's) depth adds no cross-device decision,
    # so `refer` is unchanged by it
    # C038: a backdrop is one more compare a driver's wobble can flip
    refer = int(depth_bits) < 24 or float(snap) > 0.0 or enc_on \
        or (opts is not None and getattr(opts, 'clear', None) is not None) \
        or want_cvg
    out, why = raster_on_device(mesh, vp, width, height, cull=cull,
                                snap=snap, depth_bits=depth_bits,
                                subset=subset, subdiv_px=subdiv_px,
                                near_eps=near_eps,
                                want_lin=want_lin, refer=refer,
                                opts=opts, flat_depth=flat_depth,
                                want_cvg=want_cvg)
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
            rep = replay_pixels(
                pxs, pys, sx, sy, iw, z, bw, src_final,
                tiles, bins_flat, tw, cull, depth_bits, opts=opts,
                flat=flat_depth is not None, want_cvg=want_cvg)
            r_tri, r_b, r_lb, r_z, r_front = rep[:5]
            if want_cvg and out.get('cvg') is not None:
                out['cvg'][pys, pxs, 0] = rep[5].astype(np.float32)
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
    gbuffer_into(gbuf, out['ids'], out['aux'], out.get('lin'), opts=opts,
                 depth_bits=depth_bits, cvg=out.get('cvg'))
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
