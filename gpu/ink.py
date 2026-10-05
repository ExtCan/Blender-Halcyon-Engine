"""The ink pass on the GPU -- R249 (1.88.0).

Every GPU frame used to read its shaded picture back and then spend the
better part of a second on the CPU drawing its lines: at 1280x720 the
CEL_ANIME_MODERN preset measured 514 ms in the outline bucket, 237 of
them in the (5,7,11) chamfer transform and 193 in the crease test that
extracts the seeds, against a few milliseconds of shading. This module
is that pass as full-screen fragment passes, on the same G-buffer the
shading already uploaded, drawing the same line:

- SEEDS: the 4-neighbourhood compares of render.apply_outline -- object
  ids, the depth break, the crease, the material break -- read from the
  ids texture, the per-triangle data and a texture of the RAW face
  normals (the CPU's own float32 values, so the crease test compares
  the same bits), plus the depth plane.
- The mask roads (plain and classed) as ONE pass: a Manhattan-diamond
  search of radius width-1 around every pixel -- exactly the 4-connected
  dilation the CPU runs -- with the classed road's owner rule (the
  nearer neighbour owns a silhouette pixel's class) and its paint
  order (the last class in sorted order wins).
- The style road: the chamfer distance transform as an ITERATED LOCAL
  RELAXATION over the full (5,7,11) mask -- sixteen fetches a pass,
  ping-pong targets, one pass per pixel of reach. Every pass propagates
  one move of every shortest path, so after K passes every pixel within
  K pixels of a seed holds its exact chamfer distance and nearest-seed
  feature, and the band is never wider than that. The feature's tie
  rule is order-free -- (distance, seed row, seed column) -- and
  core/ink.chamfer keeps the same rule, so the two roads own the same
  line to the same seed.
- STYLE: the per-band-pixel maths of core/ink.apply, transcribed:
  taper, shadow side, weight noise, end taper (the thinning passes and
  a second distance chain), roughness, drift, gaps, the streak and
  charcoal textures, the profile, the colour modes and Iro-Trace. The
  noises are the CPU's own integer hashes, in uint arithmetic the GLSL
  front-end evaluates bit for bit.
- COMPOSITE: boil and pencil (manual bilinear taps, the CPU's
  bilinear_at), grain, opacity, over the frame.

What stays on the CPU, by name: the stroke road (Smooth, Pressure,
Overshoot), the isophote weight, the SURFACE anchor, the form / shadow /
tone line sources, the Guilty Gear vertex-colour line control, and any
band render. A frame using one of those inks on the CPU exactly as
before and the reason is printed once.

The simulator twin (`simulate`) runs the same sources through
halcyon.shaders, so the suite holds the GPU line against the CPU line
without a driver: the mask roads bitwise, the style road to float
tolerance (the CPU keeps a few float64 intermediates).
"""

import re

import numpy as np

INF = 268435456.0            # 2**28, the CPU chamfer's INF, exact in float32

# --------------------------------------------------------------- GLSL

PRELUDE_SRC = """
uniform sampler2D hal_gb_ids;      // rgb = barycentric, a = triangle id
uniform sampler2D hal_gb_tris;     // per triangle: (material, object, -, -)
uniform sampler2D hal_ink_depth;   // r = depth (1e12 off-surface), g = marked-edge distance
uniform sampler2D hal_ink_lut;     // per material class: see build_lut
in vec2 vUV;
out vec4 Color;

PARAM_ACCESSORS

ivec2 hal_ink_px()
{
    return ivec2(clamp(vUV * hal_ink_size(), vec2(0.0),
                       hal_ink_size() - vec2(1.0)));
}

bool hal_ink_in(ivec2 p)
{
    return p.x >= 0 && p.y >= 0 && p.x < int(hal_ink_size().x)
        && p.y < int(hal_ink_size().y);
}

float hal_ink_tri(ivec2 p)
{
    return texelFetch(hal_gb_ids, p, 0).a;
}

vec4 hal_ink_tridata(float tri)
{
    int side = int(hal_tri_side());
    int t = clamp(int(tri), 0, side * side - 1);
    return texelFetch(hal_gb_tris, ivec2(t % side, t / side), 0);
}

// the pixel's material class: its material index, the sky class off-surface
int hal_ink_pmat(ivec2 p)
{
    float tri = hal_ink_tri(p);
    if (tri < 0.0) return hal_ink_nm();
    return clamp(int(hal_ink_tridata(tri).x), 0, hal_ink_nm());
}

vec4 hal_ink_lut0(int m) { return texelFetch(hal_ink_lut, ivec2(m, 0), 0); }
vec4 hal_ink_lut1(int m) { return texelFetch(hal_ink_lut, ivec2(m, 1), 0); }
vec4 hal_ink_lut2(int m) { return texelFetch(hal_ink_lut, ivec2(m, 2), 0); }

// core/ink._hash_u32, in the uint arithmetic every driver wraps the same way
uint hal_ink_hash(int ix, int iy, int k)
{
    uint x = uint(ix);
    uint y = uint(iy);
    uint h = x * 0x9E3779B1u ^ (y + 0x85EBCA77u) ^ (uint(k) * 0xC2B2AE3Du);
    h ^= h >> 15u;
    h *= 0x2C1B3C6Du;
    h ^= h >> 12u;
    h *= 0x297A2D39u;
    h ^= h >> 15u;
    return h;
}

float hal_ink_hashf(int ix, int iy, int k)
{
    return float(hal_ink_hash(ix, iy, k) & 0xffffffu) * (1.0 / 16777216.0);
}

// core/ink.value_noise
float hal_ink_vnoise(float x, float y, int k)
{
    float x0 = floor(x);
    float y0 = floor(y);
    float tx = x - x0;
    float ty = y - y0;
    tx = tx * tx * (3.0 - 2.0 * tx);
    ty = ty * ty * (3.0 - 2.0 * ty);
    int ix = int(x0);
    int iy = int(y0);
    float a = hal_ink_hashf(ix, iy, k);
    float b = hal_ink_hashf(ix + 1, iy, k);
    float c = hal_ink_hashf(ix, iy + 1, k);
    float d = hal_ink_hashf(ix + 1, iy + 1, k);
    float top = a + (b - a) * tx;
    return top + ((c + (d - c) * tx) - top) * ty;
}

// core/ink.fbm: three octaves, amplitudes halving, normalised
float hal_ink_fbm(float x, float y, int k)
{
    float total = 0.0;
    total += 1.0 * (2.0 * hal_ink_vnoise(x, y, k) - 1.0);
    total += 0.5 * (2.0 * hal_ink_vnoise(x * 2.0 + 13.1, y * 2.0 + 7.7,
                                         k + 17) - 1.0);
    total += 0.25 * (2.0 * hal_ink_vnoise(x * 4.0 + 26.2, y * 4.0 + 15.4,
                                          k + 34) - 1.0);
    return total / 1.75;
}
"""

PARAMS = (
    ('hal_ink_size', 'v2'),
    ('hal_ink_nm', 'i'),
    ('hal_tri_side', 'f'),
    ('hal_attr_side', 'f'),
    ('hal_slot_count', 'f'),
    ('hal_ink_f_obj', 'i'),
    ('hal_ink_f_mat', 'i'),
    ('hal_ink_f_depth', 'i'),
    ('hal_ink_f_nrm', 'i'),
    ('hal_ink_thr', 'f'),
    ('hal_ink_coslim', 'f'),
    ('hal_ink_radius', 'i'),
    ('hal_ink_oversky', 'i'),
    ('hal_ink_has_md', 'i'),
    ('hal_ink_opacity', 'f'),
    ('hal_ink_hmax', 'f'),
    ('hal_ink_margin', 'f'),
    ('hal_ink_inner', 'f'),
    ('hal_ink_has_int', 'i'),
    ('hal_ink_rs', 'f'),
    ('hal_ink_taper', 'f'),
    ('hal_ink_shadow', 'f'),
    ('hal_ink_wnoise', 'f'),
    ('hal_ink_wscale', 'f'),
    ('hal_ink_end_taper', 'f'),
    ('hal_ink_end_len', 'f'),
    ('hal_ink_rough', 'f'),
    ('hal_ink_rough_scale', 'f'),
    ('hal_ink_drift', 'f'),
    ('hal_ink_gaps', 'f'),
    ('hal_ink_tex_amt', 'f'),
    ('hal_ink_wmax', 'f'),
    ('hal_ink_darken', 'f'),
    ('hal_ink_color2', 'v3'),
    ('hal_ink_dmin', 'f'),
    ('hal_ink_dmax', 'f'),
    ('hal_ink_m22', 'f'),
    ('hal_ink_m23', 'f'),
    ('hal_ink_m32', 'f'),
    ('hal_ink_persp', 'i'),
    ('hal_ink_style', 'i'),
    ('hal_ink_mode', 'i'),
    ('hal_ink_axis', 'i'),
    ('hal_ink_texture', 'i'),
    ('hal_ink_has_end', 'i'),
    ('hal_ink_has_iro', 'i'),
    ('hal_ink_need_depth', 'i'),
    ('hal_ink_need_ndl', 'i'),
    ('hal_ink_lkind', 'i'),
    ('hal_ink_lvec', 'v3'),
    ('hal_ink_boil', 'f'),
    ('hal_ink_boil_scale', 'f'),
    ('hal_ink_spread', 'f'),
    ('hal_ink_strokes', 'i'),
    ('hal_ink_grain', 'f'),
    ('hal_ink_warp', 'i'),
    ('hal_ink_pencil', 'i'),
)
PARAM_INDEX = {name: i for i, (name, _kind) in enumerate(PARAMS)}


def _accessors():
    """One GLSL accessor per dial, reading its texel of the params
    texture. The dials ride a texture instead of push constants: the
    style pass has some sixty of them, past the 128-byte push-constant
    floor a Vulkan driver guarantees, and a texel read of a uniform
    value costs nothing measurable."""
    out = ['uniform sampler2D hal_ink_params;   // every dial of the frame, '
           'one texel each (PARAMS)', '',
           'vec4 hal_ink_p(int i) { return texelFetch(hal_ink_params, '
           'ivec2(i, 0), 0); }']
    for i, (name, kind) in enumerate(PARAMS):
        if kind == 'f':
            out.append(f'float {name}() {{ return hal_ink_p({i}).x; }}')
        elif kind == 'i':
            out.append(f'int {name}() {{ return int(hal_ink_p({i}).x); }}')
        elif kind == 'v2':
            out.append(f'vec2 {name}() {{ return hal_ink_p({i}).xy; }}')
        else:
            out.append(f'vec3 {name}() {{ return hal_ink_p({i}).xyz; }}')
    return '\n'.join(out)


PRELUDE = PRELUDE_SRC.replace('PARAM_ACCESSORS', _accessors())


def pack_params(values):
    """The dials as a (1, N, 4) float32 texture, one texel per entry of
    PARAMS (ints as exact floats; the two hash ints that can exceed
    2**24 -- the phase and the seed salt -- stay push constants)."""
    out = np.zeros((1, len(PARAMS), 4), np.float32)
    for i, (name, kind) in enumerate(PARAMS):
        v = values.get(name)
        if v is None:
            continue
        if kind in ('v2', 'v3'):
            arr = np.asarray(v, np.float32).ravel()
            out[0, i, :arr.size] = arr
        else:
            out[0, i, 0] = float(v)
    return out


SEEDS = PRELUDE + """
uniform sampler2D hal_ink_fn;      // the RAW face normals per triangle

vec3 hal_ink_fnorm(float tri)
{
    int side = int(hal_tri_side());
    int t = clamp(int(tri), 0, side * side - 1);
    return texelFetch(hal_ink_fn, ivec2(t % side, t / side), 0).xyz;
}

void main()
{
    ivec2 p = hal_ink_px();
    float tri = hal_ink_tri(p);
    bool cov = tri >= 0.0;
    float dp = texelFetch(hal_ink_depth, p, 0).r;
    vec4 td = cov ? hal_ink_tridata(tri) : vec4(-1.0);
    int op = cov ? int(td.y) : -1;
    int mp = cov ? int(td.x) : -1;
    vec3 np_ = cov ? hal_ink_fnorm(tri) : vec3(0.0);
    bool nzp = (abs(np_.x) + abs(np_.y) + abs(np_.z)) > 0.0;
    int pm = cov ? clamp(mp, 0, hal_ink_nm()) : hal_ink_nm();
    bool e_sil = false;
    bool e_int = false;
    bool one_sided = false;
    float best = dp;
    int own = pm;
    // the CPU's neighbour order: left, right, the row before, the row after
    for (int i = 0; i < 4; i++) {
        ivec2 q = p + ((i == 0) ? ivec2(-1, 0) : (i == 1) ? ivec2(1, 0)
                       : (i == 2) ? ivec2(0, -1) : ivec2(0, 1));
        if (!hal_ink_in(q)) continue;
        float tq = hal_ink_tri(q);
        bool covq = tq >= 0.0;
        float dq = texelFetch(hal_ink_depth, q, 0).r;
        vec4 tdq = covq ? hal_ink_tridata(tq) : vec4(-1.0);
        if (hal_ink_f_obj() == 1) {
            int oq = covq ? int(tdq.y) : -1;
            if (op != oq) e_sil = true;
        }
        if (hal_ink_f_mat() == 1) {
            int mq = covq ? int(tdq.x) : -1;
            if (mp != mq) e_int = true;
        }
        if (hal_ink_f_depth() == 1) {
            float nr = min(abs(dp), abs(dq));
            if (dp < 1e11 && dp < dq && (dq - dp) > hal_ink_thr() * max(nr, 1e-4))
                e_sil = true;
        }
        if (hal_ink_f_nrm() == 1) {
            vec3 nq = covq ? hal_ink_fnorm(tq) : vec3(0.0);
            float d = (np_.x * nq.x + np_.y * nq.y) + np_.z * nq.z;
            bool nzq = (abs(nq.x) + abs(nq.y) + abs(nq.z)) > 0.0;
            if (d < hal_ink_coslim() && nzp && nzq) e_int = true;
        }
        if (dp <= dq || !covq) one_sided = true;
        if (dq < best) {
            best = dq;
            own = covq ? clamp(int(tdq.x), 0, hal_ink_nm()) : hal_ink_nm();
        }
    }
    int owner = e_sil ? own : pm;
    Color = vec4(e_sil ? 1.0 : 0.0, e_int ? 1.0 : 0.0, float(owner),
                 one_sided ? 1.0 : 0.0);
}
"""

MASK = PRELUDE + """
uniform sampler2D hal_ink_seeds;
uniform sampler2D hal_ink_frame;

void main()
{
    ivec2 p = hal_ink_px();
    vec4 frame = texelFetch(hal_ink_frame, p, 0);
    float tri = hal_ink_tri(p);
    bool cov = tri >= 0.0;
    int pm = hal_ink_pmat(p);
    vec4 sp = texelFetch(hal_ink_seeds, p, 0);
    bool painted = false;
    int best = -1;
    for (int dy = -hal_ink_radius(); dy <= hal_ink_radius(); dy++) {
        for (int dx = -hal_ink_radius(); dx <= hal_ink_radius(); dx++) {
            int man = abs(dx) + abs(dy);
            if (man > hal_ink_radius()) continue;
            ivec2 q = p + ivec2(dx, dy);
            if (!hal_ink_in(q)) continue;
            vec4 s = texelFetch(hal_ink_seeds, q, 0);
            if (s.r < 0.5 && s.g < 0.5) continue;
            if (hal_ink_oversky() == 0 && hal_ink_tri(q) < 0.0) continue;
            int oq = int(s.b);
            vec4 l0 = hal_ink_lut0(oq);
            if (l0.x < 0.5) continue;
            if (man > int(l0.z) - 1) continue;
            int rank = int(hal_ink_lut1(oq).w);
            if (rank > best) best = rank;
            painted = true;
        }
    }
    if (hal_ink_has_md() == 1) {
        // marked interior ink: width-shaped by distance already, matched
        // to the dilation convention (w -> 2w-1 pixels)
        int po = (sp.r > 0.5) ? int(sp.b) : pm;
        vec4 l0 = hal_ink_lut0(po);
        float md = texelFetch(hal_ink_depth, p, 0).g;
        if (l0.x > 0.5 && md < float(int(l0.z)) - 0.5) {
            int rank = int(hal_ink_lut1(po).w);
            if (rank > best) best = rank;
            painted = true;
        }
    }
    if (hal_ink_lut0(pm).y > 0.5) painted = false;     // the pixel's own material inks Never
    if (hal_ink_oversky() == 0 && !cov) painted = false;
    if (!painted) {
        Color = frame;
        return;
    }
    vec3 col = hal_ink_lut2(best).rgb;
    vec3 rgb = frame.rgb * (1.0 - hal_ink_opacity()) + col * hal_ink_opacity();
    float a = (hal_ink_oversky() == 1) ? max(frame.a, hal_ink_opacity()) : frame.a;
    Color = vec4(rgb, a);
}
"""

INIT = PRELUDE + """
uniform sampler2D hal_ink_seeds;

void main()
{
    ivec2 p = hal_ink_px();
    float tri = hal_ink_tri(p);
    bool cov = tri >= 0.0;
    int pm = hal_ink_pmat(p);
    bool on = hal_ink_lut0(pm).x > 0.5;
    vec4 s = texelFetch(hal_ink_seeds, p, 0);
    // the silhouette seed is ONE-SIDED: the nearer pixel of every
    // boundary owns its line; interior seeds are two-sided
    bool sil = s.r > 0.5 && cov && on && s.a > 0.5;
    bool inter = s.g > 0.5 && cov && on;
    Color = vec4(sil ? 0.0 : 268435456.0, sil ? float(p.x) : -1.0,
                 sil ? float(p.y) : -1.0, inter ? 0.0 : 268435456.0);
}
"""

# one pass of the (5,7,11) relaxation: r = a distance with its nearest
# seed in (g, b) = (column, row), a = a second plain distance
RELAX = PRELUDE + """
uniform sampler2D hal_ink_field;

vec4 hal_ink_cand(ivec2 q, float cost)
{
    if (!hal_ink_in(q)) return vec4(268435456.0, -1.0, -1.0, 268435456.0);
    vec4 v = texelFetch(hal_ink_field, q, 0);
    return vec4(v.r + cost, v.g, v.b, v.a + cost);
}

void main()
{
    ivec2 p = hal_ink_px();
    vec4 c = texelFetch(hal_ink_field, p, 0);
    float d0 = c.r;
    float ix = c.g;
    float iy = c.b;
    float d1 = c.a;
    for (int i = 0; i < 16; i++) {
        ivec2 o;
        float cost;
        if (i == 0)       { o = ivec2(-1, 0); cost = 5.0; }
        else if (i == 1)  { o = ivec2(1, 0); cost = 5.0; }
        else if (i == 2)  { o = ivec2(0, -1); cost = 5.0; }
        else if (i == 3)  { o = ivec2(0, 1); cost = 5.0; }
        else if (i == 4)  { o = ivec2(-1, -1); cost = 7.0; }
        else if (i == 5)  { o = ivec2(1, -1); cost = 7.0; }
        else if (i == 6)  { o = ivec2(-1, 1); cost = 7.0; }
        else if (i == 7)  { o = ivec2(1, 1); cost = 7.0; }
        else if (i == 8)  { o = ivec2(-2, -1); cost = 11.0; }
        else if (i == 9)  { o = ivec2(2, -1); cost = 11.0; }
        else if (i == 10) { o = ivec2(-2, 1); cost = 11.0; }
        else if (i == 11) { o = ivec2(2, 1); cost = 11.0; }
        else if (i == 12) { o = ivec2(-1, -2); cost = 11.0; }
        else if (i == 13) { o = ivec2(1, -2); cost = 11.0; }
        else if (i == 14) { o = ivec2(-1, 2); cost = 11.0; }
        else              { o = ivec2(1, 2); cost = 11.0; }
        vec4 n = hal_ink_cand(p + o, cost);
        // the order-free tie rule: (distance, row, column)
        if (n.r < d0 || (n.r == d0 && (n.b < iy || (n.b == iy && n.g < ix)))) {
            d0 = n.r;
            ix = n.g;
            iy = n.b;
        }
        d1 = min(d1, n.a);
    }
    Color = vec4(d0, ix, iy, d1);
}
"""

# core/ink._thin_4to8, one orientation per pass
THIN = PRELUDE + """
uniform sampler2D hal_ink_field;
uniform int hal_ink_from_field;    // 1: the seed is field.r == 0; 0: field.r > 0.5
uniform int hal_ink_side;          // 0 left-and-down, 1 right-and-down

bool hal_ink_set(ivec2 q)
{
    if (!hal_ink_in(q)) return false;
    float r = texelFetch(hal_ink_field, q, 0).r;
    return (hal_ink_from_field == 1) ? (r == 0.0) : (r > 0.5);
}

void main()
{
    ivec2 p = hal_ink_px();
    bool s = hal_ink_set(p);
    bool horiz = hal_ink_set(p + ((hal_ink_side == 0) ? ivec2(-1, 0) : ivec2(1, 0)));
    bool down = hal_ink_set(p + ivec2(0, 1));
    bool keep = s && !(horiz && down);
    Color = vec4(keep ? 1.0 : 0.0, 0.0, 0.0, 0.0);
}
"""

# core/ink.stroke_ends on the thinned contour, as a distance seed
ENDS = PRELUDE + """
uniform sampler2D hal_ink_field;

bool hal_ink_set(ivec2 q)
{
    if (!hal_ink_in(q)) return false;
    return texelFetch(hal_ink_field, q, 0).r > 0.5;
}

void main()
{
    ivec2 p = hal_ink_px();
    bool thin = hal_ink_set(p);
    int n8 = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;
            if (hal_ink_set(p + ivec2(dx, dy))) n8 += 1;
        }
    }
    bool end = thin && (n8 <= 1 || n8 >= 3);
    Color = vec4(end ? 0.0 : 268435456.0, -1.0, -1.0, 268435456.0);
}
"""

# the per-pixel quantities and the band test, shared by STYLE and COMPOSITE
STYLE_COMMON = PRELUDE + """
uniform sampler2D hal_ink_field;   // r = sil distance units, (g, b) = feature, a = interior distance
uniform sampler2D hal_ink_frame;
uniform int hal_ink_phase;
uniform int hal_ink_salt;

float hal_ink_dsil(ivec2 q)
{
    return texelFetch(hal_ink_field, q, 0).r * 0.2;
}

float hal_ink_dint(ivec2 q)
{
    return texelFetch(hal_ink_field, q, 0).a * 0.2;
}

float hal_ink_dmd(ivec2 q)
{
    return texelFetch(hal_ink_depth, q, 0).g;
}

bool hal_ink_band(ivec2 p, vec4 f, bool cov, int pm)
{
    bool has = f.g >= 0.0;
    bool band = has && (f.r * 0.2 < hal_ink_hmax() + hal_ink_margin());
    if (hal_ink_has_int() == 1)
        band = band || (f.a * 0.2 < hal_ink_hmax() * hal_ink_inner() + hal_ink_margin());
    if (hal_ink_has_md() == 1)
        band = band || (texelFetch(hal_ink_depth, p, 0).g
                        < hal_ink_hmax() * hal_ink_inner() + hal_ink_margin());
    if (hal_ink_lut0(pm).y > 0.5) band = false;
    if (hal_ink_oversky() == 0 && !cov) band = false;
    return band;
}
"""

STYLE = STYLE_COMMON + """
uniform sampler2D hal_gb_attrs;
uniform sampler2D hal_ink_dend;

vec4 hal_ink_attr(float tri, int corner, int slot)
{
    int side = int(hal_attr_side());
    int index = clamp((int(tri) * 3 + corner) * int(hal_slot_count()) + slot,
                      0, side * side - 1);
    return texelFetch(hal_gb_attrs, ivec2(index % side, index / side), 0);
}

vec3 hal_ink_interp(float tri, vec3 b, int slot)
{
    vec3 a0 = hal_ink_attr(tri, 0, slot).xyz;
    vec3 a1 = hal_ink_attr(tri, 1, slot).xyz;
    vec3 a2 = hal_ink_attr(tri, 2, slot).xyz;
    return a0 * b.x + a1 * b.y + a2 * b.z;
}

// core/ink._linear_depth of the pixel's raw depth (0 off-surface)
float hal_ink_tdep(ivec2 s)
{
    float tri = hal_ink_tri(s);
    float z = (tri >= 0.0) ? texelFetch(hal_ink_depth, s, 0).r : 0.0;
    float lin;
    if (hal_ink_persp() == 1) {
        float den = hal_ink_m32() * z - hal_ink_m22();
        if (abs(den) < 1e-12) den = 1e-12;
        lin = abs(hal_ink_m23() / den);
    } else {
        float m22 = (abs(hal_ink_m22()) > 1e-12) ? hal_ink_m22() : 1e-12;
        lin = abs((z - hal_ink_m23()) / m22);
    }
    return clamp((lin - hal_ink_dmin()) / max(hal_ink_dmax() - hal_ink_dmin(), 1e-6),
                 0.0, 1.0);
}

// core/ink._key_ndl: n.l of the key lamp, clipped, 1 off-surface
float hal_ink_ndl(ivec2 s)
{
    float tri = hal_ink_tri(s);
    if (tri < 0.0 || hal_ink_lkind() == 0) return 1.0;
    vec4 ids = texelFetch(hal_gb_ids, s, 0);
    vec3 N = hal_ink_interp(tri, ids.rgb, 1);
    float ln = sqrt((N.x * N.x + N.y * N.y) + N.z * N.z);
    N = N / max(ln, 1e-12);
    vec3 L;
    if (hal_ink_lkind() == 1) {
        L = hal_ink_lvec();
    } else {
        vec3 P = hal_ink_interp(tri, ids.rgb, 0);
        L = hal_ink_lvec() - P;
        float ll = sqrt((L.x * L.x + L.y * L.y) + L.z * L.z);
        L = L / max(ll, 1e-9);
    }
    float ndl = (N.x * L.x + N.y * L.y) + N.z * L.z;
    return clamp(ndl, 0.0, 1.0);
}

// core/ink.apply's half_at: the half-width owned by a source pixel
float hal_ink_half(ivec2 s)
{
    float h = hal_ink_lut0(hal_ink_pmat(s)).z;
    if (hal_ink_taper() > 0.0)
        h = h * (1.0 + hal_ink_taper() * (1.0 - 2.0 * hal_ink_tdep(s)));
    if (hal_ink_shadow() > 0.0)
        h = h * (1.0 + hal_ink_shadow() * (1.0 - hal_ink_ndl(s)));
    return h;
}

float hal_ink_hand(ivec2 p, float scale, float ox, float oy, int k)
{
    return hal_ink_vnoise(float(p.x) / scale + ox, float(p.y) / scale + oy, k);
}

float hal_ink_profile(float c, float h)
{
    if (hal_ink_texture() == 2) {
        float tw = max(1.5, h * 1.1);
        float t = clamp((c + tw * 0.5) / tw, 0.0, 1.0);
        return t * t * (3.0 - 2.0 * t);
    }
    if (hal_ink_style() == 1) {
        float tw = max(1.0, h * 0.8);
        float t = clamp((c + tw * 0.5) / tw, 0.0, 1.0);
        return t * t * (3.0 - 2.0 * t);
    }
    return clamp(c, 0.0, 1.0);
}

void main()
{
    ivec2 p = hal_ink_px();
    float tri = hal_ink_tri(p);
    bool cov = tri >= 0.0;
    int pm = hal_ink_pmat(p);
    vec4 f = texelFetch(hal_ink_field, p, 0);
    if (!hal_ink_band(p, f, cov, pm)) {
        Color = vec4(0.0);
        return;
    }
    int phase = hal_ink_phase;
    int salt = hal_ink_salt;
    // near a silhouette: within the silhouette line's own reach (the
    // transform is exact there); farther pixels are their own source
    bool has_b = f.g >= 0.0 && (f.r * 0.2 < hal_ink_hmax() + hal_ink_margin());
    ivec2 feat = has_b ? ivec2(int(f.g), int(f.b)) : p;
    // inside: the pixel belongs to the owner's object (the line's inner
    // half); outside: the far side -- sky or the object behind
    float ftri = hal_ink_tri(feat);
    int op = cov ? int(hal_ink_tridata(tri).y) : -1;
    int of = (ftri >= 0.0) ? int(hal_ink_tridata(ftri).y) : -1;
    bool inside = cov && (op == of);
    float half_sil = hal_ink_half(feat);
    float d_b = f.r * 0.2;
    if (hal_ink_drift() > 0.0) {
        float dn = hal_ink_drift() * (2.0 * hal_ink_hand(p, hal_ink_wscale(), 5.3, 9.1,
                                                       phase * 3 + 23 + salt) - 1.0);
        d_b = inside ? d_b + dn : d_b - dn;
    }
    float db_sil = inside ? d_b + 0.5 : d_b - 0.5;
    float cov_sil = has_b ? half_sil + 0.5 - db_sil : -1e9;
    float half_self = hal_ink_half(p);
    float half_int = half_self;
    float wf = 1.0;
    bool has_wf = false;
    if (hal_ink_wnoise() > 0.0) {
        float nz = hal_ink_hand(p, hal_ink_wscale(), 0.0, 0.0, phase * 3 + 11 + salt);
        wf = 1.0 + hal_ink_wnoise() * (2.0 * nz - 1.0);
        has_wf = true;
    }
    if (hal_ink_has_end() == 1) {
        float te = clamp(texelFetch(hal_ink_dend, p, 0).r * 0.2 / hal_ink_end_len(),
                         0.0, 1.0);
        te = te * te * (3.0 - 2.0 * te);
        float ef = 1.0 - hal_ink_end_taper() * 0.85 * (1.0 - te);
        wf = has_wf ? wf * ef : ef;
        has_wf = true;
    }
    if (hal_ink_rough() > 0.0) {
        float pch = hal_ink_hand(p, hal_ink_rough_scale() * 7.0, 3.7, 1.9,
                                   phase * 3 + 31 + salt);
        pch = clamp((pch - 0.25) / 0.5, 0.0, 1.0);
        float rf = hal_ink_fbm(float(p.x) / hal_ink_rough_scale(),
                               float(p.y) / hal_ink_rough_scale(),
                               phase * 3 + 41 + salt);
        float rr = 1.0 + hal_ink_rough() * pch * rf;
        wf = has_wf ? wf * rr : rr;
        has_wf = true;
    }
    if (has_wf) {
        half_sil = half_sil * wf;
        half_self = half_self * wf;
        half_int = half_int * wf;
        cov_sil = has_b ? half_sil + 0.5 - db_sil : -1e9;
    }
    float C = hal_ink_profile(cov_sil, half_sil);
    if (hal_ink_has_int() == 1) {
        float cov_int = half_int * hal_ink_inner() - f.a * 0.2;
        C = max(C, hal_ink_profile(cov_int, half_int * hal_ink_inner()));
    }
    if (hal_ink_has_md() == 1) {
        float cov_md = half_self * hal_ink_inner() + 0.5
            - texelFetch(hal_ink_depth, p, 0).g;
        C = max(C, hal_ink_profile(cov_md, half_self * hal_ink_inner()));
    }
    if (hal_ink_gaps() > 0.0) {
        float gn = hal_ink_hand(p, 4.0 * hal_ink_rs(), 11.3, 4.7, phase * 3 + 53 + salt);
        float thin = 1.0 - clamp(half_sil / max(hal_ink_wmax(), 1e-6), 0.0, 1.0);
        float g0 = 1.0 - hal_ink_gaps() * (0.45 + 0.25 * thin);
        float cut = clamp((gn - g0) / 0.06, 0.0, 1.0);
        C = C * (1.0 - cut * cut * (3.0 - 2.0 * cut));
    }
    if (hal_ink_texture() == 1 && hal_ink_tex_amt() > 0.0) {
        // the brush's hairs: streaks along the stroke (core/ink.tangent_at)
        // of the line that put the pixel in the band -- the silhouette
        // within its reach, else the interior line, else the marked edge
        int W = int(hal_ink_size().x);
        int H = int(hal_ink_size().y);
        ivec2 xl = ivec2(max(p.x - 1, 0), p.y);
        ivec2 xr = ivec2(min(p.x + 1, W - 1), p.y);
        ivec2 yd = ivec2(p.x, max(p.y - 1, 0));
        ivec2 yu = ivec2(p.x, min(p.y + 1, H - 1));
        float gx;
        float gy;
        float fx;
        float fy;
        float v_d;
        bool on_int = !has_b && hal_ink_has_int() == 1
            && (f.a * 0.2 < hal_ink_hmax() * hal_ink_inner() + hal_ink_margin());
        if (has_b) {
            gx = hal_ink_dsil(xr) - hal_ink_dsil(xl);
            gy = hal_ink_dsil(yu) - hal_ink_dsil(yd);
            fx = float(feat.x);
            fy = float(feat.y);
            v_d = inside ? d_b : -d_b;
        } else if (on_int) {
            gx = hal_ink_dint(xr) - hal_ink_dint(xl);
            gy = hal_ink_dint(yu) - hal_ink_dint(yd);
            fx = float(p.x);
            fy = float(p.y);
            v_d = f.a * 0.2;
        } else {
            gx = hal_ink_dmd(xr) - hal_ink_dmd(xl);
            gy = hal_ink_dmd(yu) - hal_ink_dmd(yd);
            fx = float(p.x);
            fy = float(p.y);
            v_d = hal_ink_dmd(p);
        }
        float ln = sqrt(gx * gx + gy * gy);
        bool ok = ln > 1e-6;
        float tx = ok ? -gy / ln : 1.0;
        float ty = ok ? gx / ln : 0.0;
        bool flip = (tx < 0.0) || ((tx == 0.0) && (ty < 0.0));
        if (flip) { tx = -tx; ty = -ty; }
        float across = max(1.0, half_sil * 0.4);
        float u = (fx * tx + fy * ty) / (26.0 * hal_ink_rs());
        float v = v_d / across;
        float sn = hal_ink_vnoise(u + 2.1, v + 6.3, phase * 3 + 61 + salt);
        float streak = clamp((sn - 0.55) / 0.3, 0.0, 1.0);
        streak = streak * streak * (3.0 - 2.0 * streak);
        C = C * (1.0 - hal_ink_tex_amt() * 0.55 * streak);
    } else if (hal_ink_texture() == 2 && hal_ink_tex_amt() > 0.0) {
        float tooth = hal_ink_hashf(p.x, p.y, phase * 5 + 71 + salt);
        float blotch = hal_ink_vnoise(float(p.x) / (2.5 * hal_ink_rs()),
                                      float(p.y) / (2.5 * hal_ink_rs()),
                                      phase * 3 + 73 + salt);
        C = C * (1.0 - hal_ink_tex_amt() * (0.25 * tooth + 0.35 * blotch));
    }
    // the colour at the line's source pixel
    ivec2 src = (has_b && !inside) ? feat : p;
    int sm = hal_ink_pmat(src);
    vec3 K = hal_ink_lut1(sm).rgb;
    if (hal_ink_mode() == 1) {
        K = texelFetch(hal_ink_frame, src, 0).rgb * (1.0 - hal_ink_darken());
    } else if (hal_ink_mode() == 2) {
        float t;
        if (hal_ink_axis() == 1) t = float(p.y) / float(max(H_MINUS(), 1));
        else if (hal_ink_axis() == 2) t = 1.0 - hal_ink_ndl(src);
        else t = hal_ink_tdep(src);
        K = K + (hal_ink_color2() - K) * t;
    }
    if (hal_ink_has_iro() == 1) {
        float dk = hal_ink_lut0(sm).w;
        if (dk >= 0.0)
            K = texelFetch(hal_ink_frame, src, 0).rgb * max(1.0 - dk, 0.0);
    }
    Color = vec4(K, C);
}
""".replace('H_MINUS()', 'int(hal_ink_size().y) - 1')

COMPOSITE = STYLE_COMMON + """
uniform sampler2D hal_ink_layer;   // rgb = K, a = C, zero off the band

// core/ink.bilinear_at on the layer, clamped to the frame
vec4 hal_ink_bilin(float x, float y)
{
    float W = hal_ink_size().x;
    float H = hal_ink_size().y;
    x = clamp(x, 0.0, W - 1.0);
    y = clamp(y, 0.0, H - 1.0);
    float fx0 = floor(x);
    float fy0 = floor(y);
    int x0 = int(fx0);
    int y0 = int(fy0);
    int x1 = min(x0 + 1, int(W) - 1);
    int y1 = min(y0 + 1, int(H) - 1);
    float tx = x - fx0;
    float ty = y - fy0;
    vec4 a = texelFetch(hal_ink_layer, ivec2(x0, y0), 0);
    vec4 b = texelFetch(hal_ink_layer, ivec2(x1, y0), 0);
    vec4 c = texelFetch(hal_ink_layer, ivec2(x0, y1), 0);
    vec4 d = texelFetch(hal_ink_layer, ivec2(x1, y1), 0);
    vec4 top = a + (b - a) * tx;
    vec4 bot = c + (d - c) * tx;
    return top + (bot - top) * ty;
}

void main()
{
    ivec2 p = hal_ink_px();
    vec4 frame = texelFetch(hal_ink_frame, p, 0);
    float tri = hal_ink_tri(p);
    bool cov = tri >= 0.0;
    int pm = hal_ink_pmat(p);
    vec4 f = texelFetch(hal_ink_field, p, 0);
    if (!hal_ink_band(p, f, cov, pm)) {
        Color = frame;
        return;
    }
    int phase = hal_ink_phase;
    int salt = hal_ink_salt;
    vec4 layer = texelFetch(hal_ink_layer, p, 0);
    float C = layer.a;
    vec3 K = layer.rgb;
    if (hal_ink_warp() == 1) {
        float ux = 0.0;
        float uy = 0.0;
        if (hal_ink_boil() > 0.0) {
            float bs = hal_ink_boil_scale();
            ux = hal_ink_boil() * (2.0 * hal_ink_vnoise(float(p.x) / bs, float(p.y) / bs,
                                                      phase * 2 + 101 + salt) - 1.0);
            uy = hal_ink_boil() * (2.0 * hal_ink_vnoise(float(p.x) / bs + 31.7,
                                                      float(p.y) / bs + 17.3,
                                                      phase * 2 + 102 + salt) - 1.0);
        }
        if (hal_ink_pencil() == 1) {
            float acc = 0.0;
            vec3 Ksum = vec3(0.0);
            float wsum = 0.0;
            for (int k = 0; k < hal_ink_strokes(); k++) {
                float ox = hal_ink_spread() * (2.0 * hal_ink_hashf(k, phase, 7 + salt) - 1.0);
                float oy = hal_ink_spread() * (2.0 * hal_ink_hashf(k + 17, phase, 9 + salt) - 1.0);
                vec4 s = hal_ink_bilin(float(p.x) + ox + ux, float(p.y) + oy + uy);
                float ck = s.a * 0.72;
                acc = 1.0 - (1.0 - acc) * (1.0 - ck);
                Ksum += s.rgb * ck;
                wsum += ck;
            }
            C = acc;
            if (wsum > 1e-6) K = Ksum / max(wsum, 1e-6);
        } else {
            vec4 s = hal_ink_bilin(float(p.x) + ux, float(p.y) + uy);
            C = s.a;
            K = s.rgb;
        }
    }
    if (hal_ink_grain() > 0.0)
        C = C * (1.0 - hal_ink_grain() * hal_ink_hashf(p.x, p.y, phase * 5 + 3 + salt));
    C = clamp(C, 0.0, 1.0);
    float a = C * hal_ink_opacity();
    vec3 rgb = frame.rgb * (1.0 - a) + K * a;
    float fa = (hal_ink_oversky() == 1) ? max(frame.a, a) : frame.a;
    Color = vec4(rgb, fa);
}
"""

SOURCES = {'INK_SEEDS': SEEDS, 'INK_MASK': MASK, 'INK_INIT': INIT,
           'INK_RELAX': RELAX, 'INK_THIN': THIN, 'INK_ENDS': ENDS,
           'INK_STYLE': STYLE, 'INK_COMPOSITE': COMPOSITE}

_DECL = re.compile(r'^\s*uniform\s+(sampler2D|float|int|vec2|vec3)\s+(\w+)\s*;',
                   re.M)


def interface(src):
    """The CreateInfo spec of a source: its declared samplers and push
    constants, by kind, in declaration order."""
    spec = {'samplers': [], 'floats': [], 'ints': [], 'vec2': [], 'vec3': []}
    seen = set()
    for kind, name in _DECL.findall(src):
        if name in seen:
            continue
        seen.add(name)
        key = {'sampler2D': 'samplers', 'float': 'floats', 'int': 'ints',
               'vec2': 'vec2', 'vec3': 'vec3'}[kind]
        spec[key].append(name)
    return spec


# ------------------------------------------------------------- packing


def pack_tri_fn(mesh):
    """The RAW per-triangle face normals, one texel per triangle in the
    tri-data layout (pack_tri_data's side). The crease test compares
    mesh.face_normals' own float32 values on the CPU; hal_triaux carries
    them re-normalised, which is not the same bits."""
    tris = np.asarray(mesh.tris, np.int32)
    n = tris.shape[0]
    side = int(np.ceil(np.sqrt(max(n, 1))))
    out = np.zeros((side * side, 4), np.float32)
    fn = getattr(mesh, 'face_normals', None)
    if fn is not None:
        out[:n, :3] = np.asarray(fn, np.float32)
    return out.reshape(side, side, 4), side


def pack_depth(gbuf, cov, md=None):
    """r = the depth plane the CPU compares (1e12 off-surface), g = the
    marked-edge distance (1e9 without one)."""
    h, w = cov.shape
    out = np.zeros((h, w, 4), np.float32)
    out[:, :, 0] = np.where(cov, gbuf.depth, 1e12).astype(np.float32)
    out[:, :, 1] = np.float32(1e9) if md is None else np.asarray(md, np.float32)
    return out


def build_lut(tables, road):
    """The per-class table as a (3, n_m + 1, 4) texture.

    Row 0: (ink on, ink Never, width, iro darken or -1). Width is the
    integer mask width on the mask road and the style road's half-width
    (max(w - 0.5, 0.5) * rs). Row 1: (ink colour, class rank). Row 2:
    (colour of the class with that rank) -- the classed road paints its
    classes in sorted (width, colour) order and the last one wins, so
    the winner among several is the highest rank.
    """
    on = np.asarray(tables['on'], bool)
    off = np.asarray(tables['off'], bool)
    width = np.asarray(tables['width'], np.float32)
    col = np.asarray(tables['color'], np.float32)
    n = on.size
    lut = np.zeros((3, n, 4), np.float32)
    lut[0, :, 0] = on
    lut[0, :, 1] = off
    if road == 'STYLE':
        lut[0, :, 2] = tables['half']
    else:
        lut[0, :, 2] = width
    iro = tables.get('iro')
    lut[0, :, 3] = -1.0 if iro is None else np.asarray(iro, np.float32)
    lut[1, :, :3] = col
    # the classes: (width, colour) of every ON material, sorted -- the
    # CPU's `sorted(classes.items())`
    keys = sorted({(int(width[i]), tuple(float(c) for c in col[i]))
                   for i in range(n) if on[i]})
    rank = {k: r for r, k in enumerate(keys)}
    for i in range(n):
        if on[i]:
            lut[1, i, 3] = rank[(int(width[i]),
                                 tuple(float(c) for c in col[i]))]
    for (w_c, col_c), r in rank.items():
        lut[2, r, :3] = col_c
    return lut


# ---------------------------------------------------------------- plan


def refusal(scene, st):
    """Why this frame's ink stays on the CPU, or None."""
    mats = getattr(scene, 'materials', None) or []
    if float(getattr(st, 'ink_isophote', 0.0)) > 0.0:
        return 'Isophote Weight walks the surface on the CPU'
    if float(getattr(st, 'ink_smooth', 0.0)) > 0.0 or \
            float(getattr(st, 'ink_pressure', 0.0)) > 0.0 or \
            float(getattr(st, 'ink_overshoot', 0.0)) > 0.0:
        return 'the stroke road (Smooth, Pressure, Overshoot) redraws the ' \
               'seeds as chains on the CPU'
    if str(getattr(st, 'ink_anchor', 'SCREEN')).upper() == 'SURFACE':
        return 'the Surface anchor reads world positions under the line on ' \
               'the CPU'
    for name, label in (('outline_form', 'Form'), ('outline_shadow', 'Shadow'),
                        ('outline_tone', 'Tone')):
        if getattr(st, name, False):
            return f'{label} lines are a CPU line source'
    if any(str(getattr(m, 'ink_vc', 'OFF') or 'OFF').upper() != 'OFF'
           for m in mats):
        return 'the vertex-colour line control reads the mesh colours on ' \
               'the CPU'
    return None


def plan(scene, gbuf, st, tables, md, vp, proj, eye):
    """Everything the passes need, computed once: the road, the LUT,
    the uniforms of each pass and the relaxation count. Returns
    (plan, None) or (None, why)."""
    from ..core import ink as INK
    why = refusal(scene, st)
    if why:
        return None, why
    mesh = scene.mesh
    H, W = gbuf.tri.shape
    n_m = len(tables['on']) - 1
    opacity = float(tables['opacity'])
    over_sky = bool(tables['over_sky'])
    style_road = INK.styled_on(st, scene)
    seeds_uni = {
        'hal_ink_f_obj': int(bool(getattr(st, 'outline_objects', True))
                             and mesh.obj_index is not None),
        'hal_ink_f_mat': int(bool(getattr(st, 'outline_materials', False))
                             and mesh.mat_index is not None),
        'hal_ink_f_depth': int(bool(getattr(st, 'outline_depth', True))),
        'hal_ink_f_nrm': int(bool(getattr(st, 'outline_normals', True))
                             and mesh.face_normals is not None),
        'hal_ink_thr': float(max(float(getattr(st, 'outline_depth_threshold',
                                                0.02)), 1e-5)),
        'hal_ink_coslim': float(np.float32(np.cos(np.radians(
            float(getattr(st, 'outline_normal_angle', 60.0)))))),
    }
    params = {'hal_ink_size': (float(W), float(H)), 'hal_ink_nm': int(n_m),
              'hal_slot_count': 4.0}
    params.update(seeds_uni)
    out = {'road': 'STYLE' if style_road else 'MASK', 'params': params,
           'opacity': opacity, 'over_sky': over_sky,
           'has_md': md is not None, 'W': W, 'H': H, 'push': {}}
    if not style_road:
        on = np.asarray(tables['on'], bool)
        widths = np.asarray(tables['width'], np.int32)
        w_max = int(widths[on].max()) if on.any() else 1
        out['lut'] = build_lut(tables, 'MASK')
        params.update({'hal_ink_radius': int(w_max - 1),
                       'hal_ink_oversky': int(over_sky),
                       'hal_ink_has_md': int(md is not None),
                       'hal_ink_opacity': opacity})
        return out, None

    # ---- the style road's dials, exactly core/ink.apply's reading
    style = str(getattr(st, 'ink_style', 'CLEAN')).upper()
    rs = 1.0
    ref = int(getattr(st, 'ink_reference_height', 0) or 0)
    if ref > 0:
        rs = float(H) / float(ref)
    taper = float(np.clip(getattr(st, 'ink_taper', 0.0), 0.0, 1.0))
    mode = str(getattr(st, 'ink_color_mode', 'FIXED')).upper()
    axis = str(getattr(st, 'ink_gradient', 'DEPTH')).upper()
    shadow_side = float(np.clip(getattr(st, 'ink_shadow_side', 0.0), 0.0, 1.0))
    wnoise = float(np.clip(getattr(st, 'ink_weight_noise', 0.0), 0.0, 1.0))
    inner = float(max(getattr(st, 'ink_interior_scale', 1.0), 0.0))
    time = float(getattr(scene, 'time', 0.0) or 0.0)
    boil = float(max(getattr(st, 'ink_boil', 0.0), 0.0))
    phase = INK.boil_phase(time, getattr(st, 'ink_boil_fps', 12)) \
        if boil > 0.0 else 0
    seed_salt = int(getattr(st, 'seed', 0) or 0) * 7919
    spread = float(max(getattr(st, 'ink_pencil_spread', 1.5), 0.0)) \
        if style == 'PENCIL' else 0.0
    end_taper = float(np.clip(getattr(st, 'ink_end_taper', 0.0), 0.0, 1.0))
    end_len = float(max(getattr(st, 'ink_end_length', 12.0), 1.0)) * rs
    rough = float(np.clip(getattr(st, 'ink_roughness', 0.0), 0.0, 1.0))
    rough_scale = max(float(getattr(st, 'ink_roughness_scale', 6.0)), 1.0) * rs
    drift = float(max(getattr(st, 'ink_drift', 0.0), 0.0)) * rs
    gaps = float(np.clip(getattr(st, 'ink_gaps', 0.0), 0.0, 1.0))
    texture = str(getattr(st, 'ink_texture', 'SOLID')).upper()
    tex_amt = float(np.clip(getattr(st, 'ink_texture_amount', 0.6), 0.0, 1.0))
    pressure = float(np.clip(getattr(st, 'ink_pressure', 0.0), 0.0, 1.0))
    w_lut = np.maximum(np.asarray(tables['width'], np.float32)
                       - np.float32(0.5), np.float32(0.5)) * np.float32(rs)
    h_max = float(w_lut.max()) * (1.0 + taper) * (1.0 + shadow_side) \
        * (1.0 + wnoise) * (1.0 + rough) * (1.0 + 1.5 * pressure) + drift
    if texture == 'CHARCOAL':
        h_max = h_max * 1.1 + 1.0
    margin = boil + spread + 1.5
    need_depth = taper > 0.0 or (mode == 'GRADIENT' and axis == 'DEPTH')
    need_ndl = shadow_side > 0.0 or (mode == 'GRADIENT' and axis == 'LIGHT')
    tables = dict(tables)
    tables['half'] = w_lut
    out['lut'] = build_lut(tables, 'STYLE')
    # the relaxation reach: every consumer's threshold, in pixels (one
    # pass propagates at least one pixel of every shortest path)
    reach_sil = h_max + margin
    reach_int = h_max * inner + margin
    reach = max(reach_sil, reach_int)
    if texture == 'STREAKS':
        reach += 1.0        # the tangent reads the neighbours' distances
    K = int(np.ceil(reach)) + 1
    K_end = int(np.ceil(end_len)) + 1 if end_taper > 0.0 else 0
    # the taper's depth range and the projection's depth row
    dmin, dmax = 0.0, 1.0
    persp, m22, m23, m32 = 1, 0.0, 0.0, 0.0
    if need_depth:
        cov = gbuf.tri >= 0
        zz = np.where(cov, gbuf.depth, 0.0).astype(np.float32)
        depth = INK._linear_depth(zz, proj)
        dmin, dmax = INK._scene_depth_range(mesh, vp, proj, depth, cov)
        if proj is not None:
            pm = np.asarray(proj, np.float64)
            m22, m23, m32 = float(pm[2, 2]), float(pm[2, 3]), float(pm[3, 2])
            persp = int(abs(m32) > 1e-9)
        else:
            persp, m22, m23, m32 = 0, 1.0, 0.0, 0.0
    lkind, lvec = 0, (0.0, 0.0, 1.0)
    if need_ndl:
        from ..core import lines as LN
        light = LN.key_light(scene)
        if light is not None:
            kind = str(getattr(light, 'type', 'POINT')).upper()
            if kind in ('SUN', 'HEMI'):
                d = np.asarray(getattr(light, 'direction', (0.0, 0.0, -1.0)),
                               np.float32)
                ln = float(np.linalg.norm(d))
                L = (-d / ln) if ln > 1e-9 else np.array([0.0, 0.0, 1.0],
                                                          np.float32)
                lkind = 1
                lvec = tuple(float(v) for v in np.asarray(L, np.float32))
            else:
                lkind = 2
                lvec = tuple(float(v) for v in np.asarray(
                    getattr(light, 'position', (0.0, 0.0, 0.0)), np.float32))
    params.update({
        'hal_ink_hmax': float(h_max), 'hal_ink_margin': float(margin),
        'hal_ink_inner': float(inner), 'hal_ink_has_int': 1,
        'hal_ink_has_md': int(md is not None), 'hal_ink_oversky': int(over_sky),
        'hal_ink_rs': float(rs), 'hal_ink_taper': taper,
        'hal_ink_shadow': shadow_side, 'hal_ink_wnoise': wnoise,
        'hal_ink_wscale': float(max(float(getattr(st, 'ink_weight_scale', 24.0)),
                                    1.0)),
        'hal_ink_end_taper': end_taper, 'hal_ink_end_len': float(end_len),
        'hal_ink_rough': rough, 'hal_ink_rough_scale': float(rough_scale),
        'hal_ink_drift': float(drift), 'hal_ink_gaps': gaps,
        'hal_ink_tex_amt': tex_amt, 'hal_ink_wmax': float(w_lut.max()),
        'hal_ink_darken': float(np.clip(getattr(st, 'ink_fill_darken', 0.45),
                                        0.0, 1.0)),
        'hal_ink_color2': tuple(float(v) for v in np.asarray(
            getattr(st, 'ink_color2', (0.35, 0.1, 0.45)), np.float32)),
        'hal_ink_dmin': float(dmin), 'hal_ink_dmax': float(dmax),
        'hal_ink_m22': m22, 'hal_ink_m23': m23, 'hal_ink_m32': m32,
        'hal_ink_persp': persp,
        'hal_ink_style': {'CLEAN': 0, 'BRUSH': 1, 'PENCIL': 2}.get(style, 0),
        'hal_ink_mode': {'FIXED': 0, 'FILL': 1, 'GRADIENT': 2}.get(mode, 0),
        'hal_ink_axis': {'DEPTH': 0, 'VERTICAL': 1, 'LIGHT': 2}.get(axis, 0),
        'hal_ink_texture': {'SOLID': 0, 'STREAKS': 1,
                            'CHARCOAL': 2}.get(texture, 0),
        'hal_ink_has_end': int(end_taper > 0.0),
        'hal_ink_has_iro': int(tables.get('iro') is not None),
        'hal_ink_need_depth': int(need_depth), 'hal_ink_need_ndl': int(need_ndl),
        'hal_ink_lkind': int(lkind), 'hal_ink_lvec': lvec,
        'hal_ink_boil': float(boil),
        'hal_ink_boil_scale': float(max(float(getattr(st, 'ink_boil_scale',
                                                      18.0)), 1.0)),
        'hal_ink_spread': float(spread),
        'hal_ink_strokes': int(np.clip(getattr(st, 'ink_pencil_strokes', 3),
                                       1, 6)),
        'hal_ink_grain': float(np.clip(getattr(st, 'ink_grain', 0.0), 0.0,
                                       1.0)),
        'hal_ink_opacity': opacity,
        'hal_ink_warp': int(boil > 0.0 or style == 'PENCIL'),
        'hal_ink_pencil': int(style == 'PENCIL')})
    # the two hash ints ride as push constants: a seed salt can exceed
    # the 2**24 a float texel carries exactly
    out['push'] = {'hal_ink_phase': int(phase), 'hal_ink_salt': int(seed_salt)}
    out.update(K=K, K_end=K_end, need_ndl=need_ndl)
    return out, None


# -------------------------------------------------------------- runners


def _steps(p):
    """The pass list of a plan, backend-agnostic: (shader name, push
    constants, samplers {name: key}, output key). Keys name uploads
    ('ids', 'tris', 'fn', 'depth', 'lut', 'frame', 'attrs', 'params')
    or earlier outputs. The two thinning passes borrow the layer and
    output targets, dead until the style pass: seven targets in all."""
    gb = {'hal_gb_ids': 'ids', 'hal_gb_tris': 'tris', 'hal_ink_depth': 'depth',
          'hal_ink_lut': 'lut', 'hal_ink_params': 'params'}
    steps = [('INK_SEEDS', {}, {**gb, 'hal_ink_fn': 'fn'}, 'seeds')]
    if p['road'] == 'MASK':
        steps.append(('INK_MASK', {},
                      {**gb, 'hal_ink_seeds': 'seeds', 'hal_ink_frame': 'frame'},
                      'out'))
        return steps
    push = dict(p['push'])
    steps.append(('INK_INIT', {}, {**gb, 'hal_ink_seeds': 'seeds'}, 'field0'))
    cur = 'field0'
    for _i in range(p['K']):
        nxt = 'field1' if cur == 'field0' else 'field0'
        steps.append(('INK_RELAX', {}, {**gb, 'hal_ink_field': cur}, nxt))
        cur = nxt
    field = cur
    dend = None
    if p['K_end'] > 0:
        steps.append(('INK_THIN', {'hal_ink_from_field': 1, 'hal_ink_side': 0},
                      {**gb, 'hal_ink_field': 'field0'}, 'layer'))
        steps.append(('INK_THIN', {'hal_ink_from_field': 0, 'hal_ink_side': 1},
                      {**gb, 'hal_ink_field': 'layer'}, 'out'))
        steps.append(('INK_ENDS', {}, {**gb, 'hal_ink_field': 'out'}, 'dend0'))
        cur = 'dend0'
        for _i in range(p['K_end']):
            nxt = 'dend1' if cur == 'dend0' else 'dend0'
            steps.append(('INK_RELAX', {}, {**gb, 'hal_ink_field': cur}, nxt))
            cur = nxt
        dend = cur
    style_samp = {**gb, 'hal_ink_field': field, 'hal_ink_frame': 'frame',
                  'hal_gb_attrs': 'attrs',
                  'hal_ink_dend': dend if dend is not None else field}
    steps.append(('INK_STYLE', dict(push), style_samp, 'layer'))
    steps.append(('INK_COMPOSITE', dict(push),
                  {**gb, 'hal_ink_field': field, 'hal_ink_frame': 'frame',
                   'hal_ink_layer': 'layer'}, 'out'))
    return steps


def _tables_key(tables):
    return (tuple(np.asarray(tables['on'], bool).tolist()),
            tuple(np.asarray(tables['width'], np.float32).tolist()))


def simulate(scene, gbuf, img, st, tables, md=None, vp=None, proj=None,
             eye=None, debug=None):
    """The passes through Halcyon's own GLSL front-end: (image, None) or
    (None, why). The suite's twin of the driver road. `debug`, a dict,
    receives every pass's output texture by key and the plan."""
    from ..core.texture import Texture
    from ..shaders.compiler import try_compile
    from . import gbuffer as GB
    p, why = plan(scene, gbuf, st, tables, md, vp, proj, eye)
    if p is None:
        return None, why
    mesh = scene.mesh
    H, W = gbuf.tri.shape
    n = H * W
    cov = gbuf.tri >= 0
    ids = GB.pack_ids(gbuf)
    tris, tside = GB.pack_tri_data(mesh)
    fn, _fside = pack_tri_fn(mesh)
    attrs, aside = GB.pack_attributes(mesh, respect_smooth=True)

    def tex(arr):
        return Texture(arr, colorspace='Non-Color', filt='NEAREST',
                       wrap='EXTEND')

    frame = np.asarray(img, np.float32)
    if frame.shape[2] == 3:
        frame = np.concatenate([frame, np.ones((H, W, 1), np.float32)], 2)
    params = dict(p['params'])
    params['hal_tri_side'] = float(tside)
    params['hal_attr_side'] = float(aside)
    store = {'ids': tex(ids), 'tris': tex(tris), 'fn': tex(fn),
             'depth': tex(pack_depth(gbuf, cov, md)), 'lut': tex(p['lut']),
             'frame': tex(frame), 'attrs': tex(attrs),
             'params': tex(pack_params(params))}
    yy, xx = np.mgrid[0:H, 0:W]
    uv = np.stack([(xx.ravel() + 0.5) / W, (yy.ravel() + 0.5) / H],
                  1).astype(np.float32)
    progs = {}
    for name, uni, samp, out_key in _steps(p):
        prog = progs.get(name)
        if prog is None:
            src = SOURCES[name].replace('in vec2 vUV;', 'uniform vec2 vUV;')
            prog, err = try_compile(src, 'GLSL')
            if prog is None:
                return None, f'{name} does not compile: {err}'
            progs[name] = prog
        u = {'vUV': uv}
        for k, v in uni.items():
            u[k] = np.full(n, int(v), np.int32)
        for sname, key in samp.items():
            u[sname] = store[key]
        got = prog.run(u, {}, n)[0]['Color']
        store[out_key] = tex(np.asarray(got, np.float32).reshape(H, W, 4))
    out = store['out'].pixels
    if debug is not None:
        debug.update(store)
        debug['plan'] = p
    if img.shape[2] == 3:
        return out[:, :, :3].copy(), None
    return out.copy(), None


LAST_TIMINGS = {}
_WARNED = set()


def _warn(msg):
    if msg not in _WARNED:
        _WARNED.add(msg)
        print(f'[Halcyon GPU] ink on the CPU: {msg}')


def apply(scene, gbuf, img, st, tables, md=None, vp=None, proj=None,
          eye=None, frame=None, keep=None):
    """The driver road: (image, None) or (None, why). Every failure is a
    reason and the caller inks on the CPU exactly as before.

    R250: `frame` is the shading's resident frame (a gpu/frame.Resident
    whose target equals `img` byte for byte) -- the pass then samples it
    in place instead of uploading `img`. With `keep` (a dict) the 'out'
    target is NOT freed on success: it is handed over as keep['out'] and
    the caller owns it (the frame that stays on the GPU); every other
    target is freed as before, and on any failure all of them are."""
    import time as _time
    from . import device
    from . import gbuffer as GB
    from .shade import _mesh_key
    ok, why = device.probe()
    if not ok:
        return None, why
    p, why = plan(scene, gbuf, st, tables, md, vp, proj, eye)
    if p is None:
        return None, why
    t0 = _time.perf_counter()
    mesh = scene.mesh
    H, W = gbuf.tri.shape
    cov = gbuf.tri >= 0
    mkey = _mesh_key(mesh)
    side_holder = {}

    def build_tris():
        arr, sd = GB.pack_tri_data(mesh)
        side_holder['tside'] = sd
        return arr

    def build_attrs():
        arr, sd = GB.pack_attributes(mesh, respect_smooth=True)
        side_holder['aside'] = sd
        return arr

    frame_arr = np.asarray(img, np.float32)
    if frame_arr.shape[2] == 3:
        frame_arr = np.concatenate([frame_arr,
                                    np.ones((H, W, 1), np.float32)], 2)
    shaders = {}
    _frame_tex = None
    try:
        store = {}
        # the ids texture the shading uploaded this frame, when it is
        # still on the G-buffer; packed again otherwise
        got_ids = getattr(gbuf, 'gpu_ids_texture', None)
        store['ids'] = got_ids if got_ids is not None \
            else device.upload(GB.pack_ids(gbuf))
        store['tris'] = device.upload_cached(('gb_tris',) + mkey, build_tris)
        store['fn'] = device.upload_cached(('ink_fn',) + mkey,
                                           lambda: pack_tri_fn(mesh)[0])
        store['depth'] = device.upload(pack_depth(gbuf, cov, md))
        store['lut'] = device.upload(p['lut'])
        _frame_tex = None
        if frame is not None and getattr(frame, 'live', False) and \
                int(frame.width) == W and int(frame.height) == H:
            _frame_tex = frame.texture()
        if _frame_tex is not None:
            store['frame'] = _frame_tex
        else:
            store['frame'] = device.upload(frame_arr)
        if p['road'] == 'STYLE' and p.get('need_ndl'):
            store['attrs'] = device.upload_cached(('gb_attrs',) + mkey,
                                                  build_attrs)
        else:
            store['attrs'] = store['lut']       # bound, never read
        tside = side_holder.get('tside', int(store['tris'].width))
        aside = side_holder.get('aside', int(store['attrs'].width))
        params = dict(p['params'])
        params['hal_tri_side'] = float(tside)
        params['hal_attr_side'] = float(aside)
        store['params'] = device.upload(pack_params(params))
        steps = _steps(p)
        for name, _u, _s, _o in steps:
            if name in shaders:
                continue
            sh, err = device.compile_dynamic(name, SOURCES[name],
                                             interface(SOURCES[name]))
            if sh is None:
                return None, f'{name} refused by the driver: {err}'
            shaders[name] = sh
    except Exception as exc:                                    # noqa: BLE001
        return None, f'uploading the ink textures failed: {exc}'
    t_up = _time.perf_counter() - t0
    targets = {}
    draws = []
    try:
        for name, uni, samp, out_key in steps:
            tgt = targets.get(out_key)
            if tgt is None:
                tgt = device.Target(W, H)
                targets[out_key] = tgt
            binds = {}
            for sname, key in samp.items():
                src = store.get(key)
                if src is None:
                    src = device.target_texture(targets[key])
                    store[key] = src
                binds[sname] = src
            draws.append((shaders[name], dict(uni), binds, tgt, 'NONE',
                          False, None))
        out = device.draw_many(draws, read=targets['out'])
        if keep is not None:
            # ownership of the output target passes to the caller: it is
            # the frame now, and it stays on the GPU
            keep['out'] = targets.pop('out')
    except Exception as exc:                                    # noqa: BLE001
        return None, f'the ink passes failed: {type(exc).__name__}: {exc}'
    finally:
        for t in targets.values():
            t.free()
    LAST_TIMINGS.clear()
    LAST_TIMINGS.update(upload_ms=t_up * 1000.0,
                        total_ms=(_time.perf_counter() - t0) * 1000.0,
                        passes=len(draws), road=p['road'],
                        K=int(p.get('K', 0)) + int(p.get('K_end', 0)),
                        frame_reused=_frame_tex is not None)
    if img.shape[2] == 3:
        return np.ascontiguousarray(out[:, :, :3]), None
    return out, None
