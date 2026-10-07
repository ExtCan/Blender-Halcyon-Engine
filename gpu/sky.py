"""The sky / background pass on the GPU (R250, 1.89.0).

The frame's uncovered pixels used to be the CPU's: `render._background_image`
built a camera ray per pixel, normalised it, and evaluated the world along
it -- 109 ms of a 720p GPU frame for a FLAT colour (the rays and three
normalisations), 179 ms at the field's 2x supersample, and the shaded
readback was then copied over it pixel by pixel. This pass draws the same
picture as the LAST draw of the deferred shading's own burst (blend NONE):
it writes `vec4(sky, 1.0)` where the G-buffer holds no triangle and
`discard`s where it does, so the material passes' pixels stay exactly what
they drew (the design's first cut drew the sky FIRST under the materials'
blend; the review showed a material's blend over a non-zero destination
is not the CPU's composite, and a sky drawn last touches nothing covered).
The readback IS the frame; its alpha plane is composed on the CPU (1.0 or
the Screen Door code at covered pixels, the film's alpha elsewhere).

What it draws, transcribed op for op from `core/sky.py` and
`render.world_color` in float32: SOLID; a NODES world without a graph (the
flat colour, the `sky_blend` gradient, an `env_image` sampled bilinearly);
a NODES world whose graph is a plain Background node (constant Color and
Strength -- Blender's default world); GRADIENT; BANDS; HDRI (nearest or
bilinear, edge-extended, tinted; a missing texture falls to the solid
colour exactly as the CPU does). Refused by name (printed once; the CPU
draws the sky as before): BRYCE, PAINTED, STARFIELD, PHYSICAL, any other
world graph, the ground plane. Under a transparent film the pass still
draws (MODE_NONE) and writes exactly `vec4(0.0)` at every uncovered pixel:
a cleared target is not a promise (the pool hands back used targets), the
written zero is the 2.79 contract.

Twins: every mode here is BITWISE the CPU sky in the simulator on the
suite's frames (the rays included: the CPU's (N,4)x(4,4) matmul sums in
the shader's order, and `_rotate_z`'s float64 rounding lands on the same
float32). The driver is measured by the field test: its `pow`, `atan`
and division are its own, and a ray an ulp off can flip a pixel that sits
on a cliff (a band edge, a nearest texel edge) whole -- the field test
counts those pixels by name. The supersampled road reproduces
`fast_background` exactly: the low-resolution pixel centre's ray for every
pixel of a block, edge blocks folded into the last row and column.

Every dial rides a params texture (texelFetch is exact; ints to 2**24);
the mode is the one push-constant int. Nothing here is a setting: the
pass engages under the deferred shading's own gate.

R251 (sky-camera): two pixel-addressed backdrops and one more
direction mode. CYLINDER (Doom's angle-mapped sky, C056) reads the
CPU's own per-column / per-row integer tables (`hal_sky_tab`, built
by `core/sky.cylinder_tables` from the same projection, J and yaw on
both roads, uploaded per frame); LW_GRADIENT (LightWave's four-colour
backdrop, C100) is a whole-number squeeze as repeated multiplication;
the Mode 7 floor (C048) is a second stage of the same draw after the
sky colour: per-row 8.8 registers (`hal_sky_m7rows`, per frame) over
the palettised 1024x1024 map (`hal_sky_m7map`, cached by content),
integer multiply / add / shift / mask only. All three are bitwise
the CPU by construction: tables and map are CPU bytes.

R253 (cubemap-world): CUBEMAP, the 1990s skybox, is one more direction
mode (MODE_CUBE = 9) on the HDRI pattern: `core/sky.cube_atlas` builds
the six faces as ONE (6S x S) atlas in GL face order, already turned
into the canonical GL orientation on the CPU, and it rides the existing
`hal_sky_env` sampler (no new sampler, no new push constant; one PARAMS
texel `hal_sky_cube` = (S, filter, 0)). `hal_sky_sample_cube` is the
op-for-op transcription of `sky.cube_face_uv` + `sky.cube_sample`: the
OpenGL 4.6 table 8.19 face rule on the Z-up direction swapped to GL
axes, then exactly `hal_sky_sample_env`'s nearest / bilinear arithmetic
with every tap clamped to the face's own texels (the seam law: a tap
never leaves its face). Bitwise in the simulator; on the driver a ray
an ulp off can flip a nearest-texel cliff whole (HDRI's stance). The
plan refuses by name when 6 * S would exceed the driver's texture
height; a missing or malformed image folds to the flat colour x
strength exactly as HDRI's plan does.
"""

import re

import numpy as np

# the mode families, exactly the CPU's decision order in world_color /
# sky.evaluate
MODE_NONE = 0          # nothing to draw (transparent film)
MODE_FLAT = 1          # a constant colour (SOLID*strength, world.color, Background)
MODE_BLEND = 2         # NODES: world.sky_blend, hor -> zen on d.z
MODE_ENV = 3           # NODES: world.env_image, bilinear, extended
MODE_GRADIENT = 4      # sky.gradient (then strength)
MODE_BANDS = 5         # sky.bands (then strength)
MODE_HDRI = 6          # sky.hdri (tint, then strength)
MODE_CYL = 7           # R251 C056: sky.cylinder_pixels (the tables, then strength)
MODE_LWGRAD = 8        # R251 C100: sky.lw_gradient (then strength)
MODE_CUBE = 9          # R253: sky.cubemap (the atlas, tint, then strength)

M7_OVER = {'WRAP': 0, 'TRANSPARENT': 1, 'TILE0': 2}   # World.mode7_over (M7SEL)

#: R253: the tallest atlas the plan will hand the driver (6 * S rows). The
#: GL_MAX_TEXTURE_SIZE floor of every card the field runs is 16384; a
#: taller atlas is refused BY NAME at plan time (the CPU draws the sky),
#: never left to a failed upload
CUBE_ATLAS_MAX_HEIGHT = 16384

BLEND_CODES = {'LINEAR': 0, 'SMOOTH': 1, 'SHARP': 2, 'EASE': 3}

#: the dial table: one texel each, (name, kind) with kind in f|i|v2|v3|v4
PARAMS = [
    ('hal_sky_inv0', 'v4'),      # rows of inv(view-projection), float32
    ('hal_sky_inv1', 'v4'),
    ('hal_sky_inv2', 'v4'),
    ('hal_sky_inv3', 'v4'),
    ('hal_sky_eye', 'v3'),
    ('hal_sky_size', 'v2'),      # the target's (w, h)
    ('hal_sky_low', 'v3'),       # fast_background: (lw, lh, ss); ss 1 = off
    ('hal_sky_alpha', 'f'),      # the uncovered pixels' alpha (1.0)
    ('hal_sky_color', 'v3'),     # FLAT: the constant
    ('hal_sky_hor', 'v3'),
    ('hal_sky_zen', 'v3'),
    ('hal_sky_gnd', 'v3'),
    ('hal_sky_height', 'f'),     # horizon_height
    ('hal_sky_above_den', 'f'),  # max(1 - height, 1e-3), rounded as the CPU rounds it
    ('hal_sky_below_den', 'f'),  # max(1 + height, 1e-3)
    ('hal_sky_falloff', 'f'),    # max(gradient_falloff, 0.01)
    ('hal_sky_blend', 'i'),      # BLEND_CODES
    ('hal_sky_ground', 'i'),     # show_ground
    ('hal_sky_steps', 'i'),      # band_count (>= 1)
    ('hal_sky_soft', 'f'),       # band_softness in 0..1
    ('hal_sky_soft_lo', 'f'),    # 1 - soft
    ('hal_sky_soft_den', 'f'),   # max(soft, 1e-4)
    ('hal_sky_strength', 'f'),
    ('hal_sky_rot', 'v3'),       # cos, sin, on (|rotation| > 1e-6)
    ('hal_sky_env_size', 'v2'),  # the env image's (w, h)
    ('hal_sky_env_filt', 'i'),   # 0 NEAREST, 1 BILINEAR
    ('hal_sky_env_map', 'i'),    # 0 EQUIRECT (and SCREEN), 1 MIRRORBALL
    ('hal_sky_tint', 'v3'),      # env_tint (HDRI)
    # ---- R251 sky-camera
    ('hal_sky_pitch', 'v3'),     # (Wo, Ho, ss): the OUTPUT pixel pitch, always filled
    ('hal_sky_tab_wo', 'i'),     # CYLINDER: Wo (the row table starts at texel Wo)
    ('hal_sky_lw_zen', 'v3'),    # LW_GRADIENT: the four colours, float32
    ('hal_sky_lw_sky', 'v3'),
    ('hal_sky_lw_gnd', 'v3'),
    ('hal_sky_lw_nad', 'v3'),
    ('hal_sky_lw_sqs', 'i'),     # sky squeeze 1..20 (a lane-uniform loop bound)
    ('hal_sky_lw_sqg', 'i'),     # ground squeeze 1..20
    ('hal_sky_m7', 'i'),         # Mode 7 floor: 0 / 1
    ('hal_sky_m7over', 'i'),     # M7_OVER
    # ---- R251 LIGHT-A2 (F009): POV ground fog on a miss ray
    ('hal_sky_gfog', 'v3'),      # (alt*(pi/2 - atan(y1)) float32, density, on)
    ('hal_sky_gfcol', 'v3'),     # the fog colour x fog ambient
    # ---- R253 cube map (appended at the END: PARAM_INDEX is positional)
    ('hal_sky_cube', 'v3'),      # (S, filt 0 NEAREST / 1 BILINEAR, 0); read under mode 9 only
]
PARAM_INDEX = {name: i for i, (name, _k) in enumerate(PARAMS)}


def _accessors():
    out = ['uniform sampler2D hal_sky_params;   // every dial, one texel '
           'each (PARAMS)', '',
           'vec4 hal_sky_p(int i) { return texelFetch(hal_sky_params, '
           'ivec2(i, 0), 0); }']
    for i, (name, kind) in enumerate(PARAMS):
        if kind == 'f':
            out.append(f'float {name}() {{ return hal_sky_p({i}).x; }}')
        elif kind == 'i':
            out.append(f'int {name}() {{ return int(hal_sky_p({i}).x); }}')
        elif kind == 'v2':
            out.append(f'vec2 {name}() {{ return hal_sky_p({i}).xy; }}')
        elif kind == 'v3':
            out.append(f'vec3 {name}() {{ return hal_sky_p({i}).xyz; }}')
        else:
            out.append(f'vec4 {name}() {{ return hal_sky_p({i}); }}')
    return '\n'.join(out)


def pack_params(values):
    """The dials as a (1, N, 4) float32 texture, one texel per PARAMS
    entry (ints as exact floats; nothing here can exceed 2**24)."""
    out = np.zeros((1, len(PARAMS), 4), np.float32)
    for i, (name, kind) in enumerate(PARAMS):
        v = values.get(name)
        if v is None:
            continue
        if kind in ('v2', 'v3', 'v4'):
            arr = np.asarray(v, np.float32).ravel()
            out[0, i, :arr.size] = arr
        else:
            out[0, i, 0] = float(v)
    return out


SOURCE_SRC = """
uniform sampler2D hal_gb_ids;      // the shading's ids texture: a < 0 is open sky
uniform sampler2D hal_sky_env;     // the env / HDRI image (bound to the params when absent)
uniform sampler2D hal_sky_tab;     // R251 CYLINDER: (1, Wo+Ho) texel i<Wo col[i], Wo+j row[j] (bound to the params when unused)
uniform sampler2D hal_sky_m7rows;  // R251 Mode 7: (1, 2*Ho) texel 2*yo (A, C, X0, Y0), 2*yo+1 (hit, 0, 0, 0)
uniform sampler2D hal_sky_m7map;   // R251 Mode 7: the palettised 1024x1024 map
PARAM_ACCESSORS
uniform int hal_sky_mode;
in vec2 vUV;
out vec4 Color;

// mathx.normalize: a / max(sqrt((x*x + y*y) + z*z), 1e-8)
vec3 hal_sky_norm(vec3 a)
{
    float ln = sqrt((a.x * a.x + a.y * a.y) + a.z * a.z);
    return a / max(ln, 1e-8);
}

// render._background_image's camera ray of a pixel: the NDC centre of the
// pixel (or, under fast_background, of its low-resolution block), through
// the inverse view-projection, divided by w, less the eye, normalised
vec3 hal_sky_ray(ivec2 px)
{
    vec2 size = hal_sky_size();
    vec3 low = hal_sky_low();
    int ss = int(low.z);
    float fx = float(px.x);
    float fy = float(px.y);
    float fw = size.x;
    float fh = size.y;
    if (ss > 1) {
        int lw = int(low.x);
        int lh = int(low.y);
        // the block a pixel belongs to; the edge bands beyond lw*ss and
        // lh*ss read the last column / row (np.pad mode='edge')
        int lx = min(int(fx / float(ss)), lw - 1);
        int ly = min(int(fy / float(ss)), lh - 1);
        fx = float(lx);
        fy = float(ly);
        fw = float(lw);
        fh = float(lh);
    }
    float nx = (fx + 0.5) / fw * 2.0 - 1.0;
    float ny = (fy + 0.5) / fh * 2.0 - 1.0;
    vec4 i0 = hal_sky_inv0();
    vec4 i1 = hal_sky_inv1();
    vec4 i2 = hal_sky_inv2();
    vec4 i3 = hal_sky_inv3();
    float wx = ((nx * i0.x + ny * i0.y) + i0.z) + i0.w;
    float wy = ((nx * i1.x + ny * i1.y) + i1.z) + i1.w;
    float wz = ((nx * i2.x + ny * i2.y) + i2.z) + i2.w;
    float ww = ((nx * i3.x + ny * i3.y) + i3.z) + i3.w;
    float wd = (abs(ww) < 1e-9) ? 1e-9 : ww;
    vec3 world = vec3(wx / wd, wy / wd, wz / wd);
    return hal_sky_norm(world - hal_sky_eye());
}

// sky._blend
float hal_sky_blendf(float t, int mode)
{
    t = clamp(t, 0.0, 1.0);
    if (mode == 1) { return t * t * (3.0 - 2.0 * t); }
    if (mode == 2) { return t * t; }
    if (mode == 3) { return sqrt(t); }
    return t;
}

// np.power(x, 1.0) is x; the driver's pow need not be
float hal_sky_powf(float x, float e)
{
    return (e == 1.0) ? x : pow(x, e);
}

// sky.bands' quantise: steps colours INCLUDING both ends
float hal_sky_quant(float t)
{
    int steps = hal_sky_steps();
    if (steps == 1) { return 0.0; }
    float fs = float(steps);
    float s = min(floor(t * fs), fs - 1.0);
    float soft = hal_sky_soft();
    if (soft > 1e-4) {
        float frac = t * fs - floor(t * fs);
        float e = clamp((frac - hal_sky_soft_lo()) / hal_sky_soft_den(), 0.0, 1.0);
        s = min(s + e * e * (3.0 - 2.0 * e), fs - 1.0);
    }
    return clamp(s / (fs - 1.0), 0.0, 1.0);
}

// sky.gradient / sky.bands on the evaluated direction
vec3 hal_sky_gradient(vec3 d, int quantised)
{
    float up = clamp(d.z, -1.0, 1.0);
    vec3 hor = hal_sky_hor();
    vec3 zen = hal_sky_zen();
    vec3 gnd = hal_sky_gnd();
    float height = hal_sky_height();
    float falloff = hal_sky_falloff();
    int mode = hal_sky_blend();
    float above = clamp((up - height) / hal_sky_above_den(), 0.0, 1.0);
    float t = hal_sky_blendf(hal_sky_powf(above, falloff), mode);
    if (quantised == 1) { t = hal_sky_quant(t); }
    vec3 sky = hor + (zen - hor) * t;
    if (hal_sky_ground() == 1) {
        float below = clamp((height - up) / hal_sky_below_den(), 0.0, 1.0);
        float b = hal_sky_blendf(hal_sky_powf(below, falloff), mode);
        if (quantised == 1) { b = hal_sky_quant(b); }
        if (up < height) { sky = hor + (gnd - hor) * b; }
    }
    return sky;
}

vec4 hal_sky_fetch(int x, int y)
{
    return texelFetch(hal_sky_env, ivec2(x, y), 0);
}

// texture.env_equirect_uv / env_sphere_uv, then Texture.sample with
// wrap EXTEND (clip to the edge texel) and NEAREST or BILINEAR
vec3 hal_sky_sample_env(vec3 d, int filt)
{
    vec2 es = hal_sky_env_size();
    float w = es.x;
    float h = es.y;
    float u;
    float v;
    if (hal_sky_env_map() == 1) {
        float m = 2.0 * sqrt(max((d.x * d.x + d.y * d.y) + (d.z + 1.0) * (d.z + 1.0), 1e-8));
        u = d.x / m + 0.5;
        v = d.y / m + 0.5;
    } else {
        u = atan(d.y, -d.x) / 6.283185307179586 + 0.5;
        v = atan(d.z, sqrt(max(d.x * d.x + d.y * d.y, 1e-12))) / 3.141592653589793 + 0.5;
    }
    if (filt == 0) {
        int x = int(clamp(floor(u * w), 0.0, w - 1.0));
        int y = int(clamp(floor(v * h), 0.0, h - 1.0));
        return hal_sky_fetch(x, y).rgb;
    }
    float fx = u * w - 0.5;
    float fy = v * h - 0.5;
    float x0 = floor(fx);
    float y0 = floor(fy);
    float tx = fx - x0;
    float ty = fy - y0;
    int ix0 = int(clamp(x0, 0.0, w - 1.0));
    int ix1 = int(clamp(x0 + 1.0, 0.0, w - 1.0));
    int iy0 = int(clamp(y0, 0.0, h - 1.0));
    int iy1 = int(clamp(y0 + 1.0, 0.0, h - 1.0));
    vec3 c00 = hal_sky_fetch(ix0, iy0).rgb;
    vec3 c10 = hal_sky_fetch(ix1, iy0).rgb;
    vec3 c01 = hal_sky_fetch(ix0, iy1).rgb;
    vec3 c11 = hal_sky_fetch(ix1, iy1).rgb;
    vec3 top = c00 + (c10 - c00) * tx;
    vec3 bot = c01 + (c11 - c01) * tx;
    return top + (bot - top) * ty;
}

// R253: sky.cube_face_uv + sky.cube_sample -- the OpenGL 4.6 table 8.19
// face rule on the Z-up direction swapped to GL axes (gx, gy, gz) =
// (d.x, d.z, -d.y), the same tie order (X over Y over Z), then exactly
// hal_sky_sample_env's nearest / bilinear arithmetic with every tap
// clamped to the face's own S texels at atlas rows [face*S, face*S+S):
// a tap never leaves its face (the seam law on both roads)
vec3 hal_sky_sample_cube(vec3 d, int filt)
{
    float S = hal_sky_cube().x;
    float gx = d.x;
    float gy = d.z;
    float gz = -d.y;
    float ax = abs(gx);
    float ay = abs(gy);
    float az = abs(gz);
    int face;
    float ma;
    float sc;
    float tc;
    if (ax >= ay && ax >= az) {
        ma = ax;
        if (gx >= 0.0) { face = 0; sc = -gz; tc = -gy; } else { face = 1; sc = gz; tc = -gy; }
    } else if (ay >= az) {
        ma = ay;
        if (gy >= 0.0) { face = 2; sc = gx; tc = gz; } else { face = 3; sc = gx; tc = -gz; }
    } else {
        ma = az;
        if (gz >= 0.0) { face = 4; sc = gx; tc = -gy; } else { face = 5; sc = -gx; tc = -gy; }
    }
    float u = (sc / ma) * 0.5 + 0.5;
    float v = 0.5 - (tc / ma) * 0.5;      // GL's t runs down; the atlas rows run up
    int y0f = face * int(S);
    if (filt == 0) {
        int x = int(clamp(floor(u * S), 0.0, S - 1.0));
        int y = int(clamp(floor(v * S), 0.0, S - 1.0));
        return hal_sky_fetch(x, y0f + y).rgb;
    }
    float fx = u * S - 0.5;
    float fy = v * S - 0.5;
    float x0 = floor(fx);
    float y0 = floor(fy);
    float tx = fx - x0;
    float ty = fy - y0;
    int ix0 = int(clamp(x0, 0.0, S - 1.0));
    int ix1 = int(clamp(x0 + 1.0, 0.0, S - 1.0));
    int iy0 = int(clamp(y0, 0.0, S - 1.0));
    int iy1 = int(clamp(y0 + 1.0, 0.0, S - 1.0));
    vec3 c00 = hal_sky_fetch(ix0, y0f + iy0).rgb;
    vec3 c10 = hal_sky_fetch(ix1, y0f + iy0).rgb;
    vec3 c01 = hal_sky_fetch(ix0, y0f + iy1).rgb;
    vec3 c11 = hal_sky_fetch(ix1, y0f + iy1).rgb;
    vec3 top = c00 + (c10 - c00) * tx;
    vec3 bot = c01 + (c11 - c01) * tx;
    return top + (bot - top) * ty;
}

void main()
{
    vec2 size = hal_sky_size();
    ivec2 px = ivec2(clamp(vUV * size, vec2(0.0), size - vec2(1.0)));
    vec4 ids = texelFetch(hal_gb_ids, px, 0);
    if (ids.a >= 0.0) {
        // a covered pixel belongs to its material pass, drawn before this
        // one: the texel is left exactly as that pass wrote it
        discard;
    }
    if (hal_sky_mode == 0) {
        // the transparent film: premultiplied (0,0,0,0), written -- never
        // trusted to whatever a material pass left at a pixel it did not own
        Color = vec4(0.0);
        return;
    }
    vec3 d0 = hal_sky_ray(px);
    vec3 col = vec3(0.0);
    if (hal_sky_mode == 1) {
        col = hal_sky_color();
    } else if (hal_sky_mode == 2) {
        // world_color's sky_blend branch: its own normalise of the ray
        vec3 d = hal_sky_norm(d0);
        float t = clamp(d.z * 0.5 + 0.5, 0.0, 1.0);
        col = hal_sky_hor() + (hal_sky_zen() - hal_sky_hor()) * t;
    } else if (hal_sky_mode == 3) {
        vec3 d = hal_sky_norm(d0);
        col = hal_sky_sample_env(d, 1);
    } else {
        // sky.evaluate: normalise again, spin by -rotation, the mode,
        // then strength
        vec3 d = hal_sky_norm(d0);
        vec3 r = hal_sky_rot();
        if (r.z > 0.5) {
            d = vec3(d.x * r.x - d.y * r.y, d.x * r.y + d.y * r.x, d.z);
        }
        if (hal_sky_mode == 4) {
            col = hal_sky_gradient(d, 0);
        } else if (hal_sky_mode == 5) {
            col = hal_sky_gradient(d, 1);
        } else if (hal_sky_mode == 6) {
            col = hal_sky_sample_env(d, hal_sky_env_filt()) * hal_sky_tint();
        } else if (hal_sky_mode == 7) {
            // sky.cylinder_pixels: the CPU's own integer tables by output
            // pixel (the rotation is inside the table; d is unused)
            vec3 pt = hal_sky_pitch();
            int sst = int(pt.z);
            int xo = min(int(float(px.x) / float(sst)), int(pt.x) - 1);
            int yo = min(int(float(px.y) / float(sst)), int(pt.y) - 1);
            int wo = hal_sky_tab_wo();
            int cc = int(texelFetch(hal_sky_tab, ivec2(xo, 0), 0).x);
            int rr = int(texelFetch(hal_sky_tab, ivec2(wo + yo, 0), 0).x);
            col = hal_sky_fetch(cc, rr).rgb;
        } else if (hal_sky_mode == 8) {
            // sky.lw_gradient: a hard horizon, whole-number squeezes as
            // repeated multiplication, the fixed a + (b - a) * t order
            float u = clamp(d.z, -1.0, 1.0);
            if (u >= 0.0) {
                float p = 1.0 - u;
                float q = p;
                int n = hal_sky_lw_sqs();
                for (int i = 1; i < n; i++) { q = q * p; }
                float t = 1.0 - q;
                col = hal_sky_lw_sky() + (hal_sky_lw_zen() - hal_sky_lw_sky()) * t;
            } else {
                float p = 1.0 + u;
                float q = p;
                int n = hal_sky_lw_sqg();
                for (int i = 1; i < n; i++) { q = q * p; }
                float t = 1.0 - q;
                col = hal_sky_lw_gnd() + (hal_sky_lw_nad() - hal_sky_lw_gnd()) * t;
            }
        } else if (hal_sky_mode == 9) {
            // R253: sky.cubemap -- the atlas texel, then the tint as hdri()
            col = hal_sky_sample_cube(d, int(hal_sky_cube().y)) * hal_sky_tint();
        }
        col = col * hal_sky_strength();
    }
    if (hal_sky_m7() == 1) {
        // sky.mode7_overlay: the SNES PPU's per-scanline registers, X =
        // X0 + A * x in 8.8, the texel through the M7SEL over-map rule
        vec3 pt = hal_sky_pitch();
        int sst = int(pt.z);
        int xo = min(int(float(px.x) / float(sst)), int(pt.x) - 1);
        int yo = min(int(float(px.y) / float(sst)), int(pt.y) - 1);
        vec4 r0 = texelFetch(hal_sky_m7rows, ivec2(2 * yo, 0), 0);
        vec4 r1 = texelFetch(hal_sky_m7rows, ivec2(2 * yo + 1, 0), 0);
        if (r1.x > 0.5) {
            int A = int(r0.x);
            int C = int(r0.y);
            int X0 = int(r0.z);
            int Y0 = int(r0.w);
            int X = X0 + A * xo;
            int Y = Y0 + C * xo;
            int u = X >> 8;
            int v = Y >> 8;
            int over = hal_sky_m7over();
            bool inside = (u >= 0) && (u < 1024) && (v >= 0) && (v < 1024);
            if (over == 0) { u = u & 1023; v = v & 1023; inside = true; }
            if (over == 2 && !inside) { u = u & 7; v = v & 7; inside = true; }
            if (inside) { col = texelFetch(hal_sky_m7map, ivec2(u, v), 0).rgb; }
        }
    }
    // R251 sky tail
    col = hal_ground_fog_sky(col, d0);
    Color = vec4(col, hal_sky_alpha());
}
"""

SOURCE = SOURCE_SRC.replace('PARAM_ACCESSORS', _accessors())
SOURCES = {'SKY': SOURCE}

_DECL = re.compile(r'^\s*uniform\s+(sampler2D|float|int|vec2|vec3)\s+(\w+)\s*;',
                   re.M)


def interface(src=SOURCE):
    """The CreateInfo spec of the source: its declared samplers and push
    constants, by kind, in declaration order (the ink's rule)."""
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


# ------------------------------------------------------------------ the plan


def _f32(x):
    return float(np.float32(x))


def _plain_background(graph):
    """(rgb, strength) when the world graph is a lone Background node with
    constant Color and Strength -- Blender's default world -- else None.
    Anything else is the node road, which refuses by name."""
    if not isinstance(graph, dict):
        return None
    nodes = graph.get('nodes') or {}
    out_id = graph.get('output')
    out = nodes.get(out_id) if out_id else None
    if not out:
        return None
    surf = None
    for s in out.get('inputs', []):
        if s.get('name') == 'Surface' or s.get('identifier') == 'Surface':
            surf = s
            break
    if surf is None:
        return None
    link = surf.get('link')
    if not link:
        return None
    bg = nodes.get(link[0])
    if not bg or bg.get('bl_idname') != 'ShaderNodeBackground' or \
            int(link[1]) != 0:
        return None
    col = strength = None
    for s in bg.get('inputs', []):
        nm = s.get('name') or s.get('identifier')
        if nm == 'Color':
            if s.get('link'):
                return None
            col = s.get('default')
        elif nm == 'Strength':
            if s.get('link'):
                return None
            strength = s.get('default')
    try:
        rgb = np.asarray(col, np.float32).ravel()[:3]
        if rgb.size != 3:
            return None
        st = np.float32(float(np.asarray(strength, np.float32).ravel()[0]))
    except Exception:                                           # noqa: BLE001
        return None
    return rgb.astype(np.float32), st


def _env_texture(world, textures):
    """The texture the CPU would sample for the world's env image:
    core/sky's ONE lookup (R251), never a copy of it."""
    from ..core import sky as SKY
    return SKY.env_texture(world, textures)


def refusal(scene, st):
    """Why this frame's sky stays on the CPU, or None."""
    world = getattr(scene, 'world', None)
    if world is None:
        return None
    mode = str(getattr(world, 'mode', 'NODES') or 'NODES').upper()
    if mode in ('BRYCE', 'PAINTED', 'STARFIELD', 'PHYSICAL'):
        names = {'BRYCE': 'the Bryce Sky Lab', 'PAINTED': 'the painted sky',
                 'STARFIELD': 'the starfield', 'PHYSICAL': 'the physical sky'}
        return f'{names[mode]} is evaluated on the CPU'
    if mode not in ('NODES', 'SOLID', 'GRADIENT', 'BANDS', 'HDRI',
                    'CYLINDER', 'LW_GRADIENT', 'CUBEMAP'):   # R253: CUBEMAP
        return f"sky mode '{mode}' has no GPU pass"
    if getattr(world, 'ground_plane', False) and \
            str(getattr(world, 'ground_mode', 'SOLID')) != 'MODE7':
        # R251 (C048): the Mode 7 floor is integer per-row registers
        # over a palettised map -- drawn here; every analytic floor is
        # still lit and traced on the CPU
        return 'the infinite ground plane is lit and traced on the CPU'
    if mode == 'NODES' and getattr(world, 'graph', None):
        if _plain_background(world.graph) is None:
            return 'the world node graph is evaluated on the CPU'
    return None


def plan(scene, st, width, height, vp, eye, textures=None, ss=1):
    """The frame's sky plan: (plan, None) or (None, why).

    `width` x `height` is the target (the internal, supersampled frame);
    `ss` the supersample factor. The plan reads every dial the CPU road
    reads, in the CPU's decision order, so the two roads choose the same
    picture."""
    if getattr(st, 'film_transparent', False):
        return {'mode': MODE_NONE, 'params': {}, 'env': None,
                'env_key': None}, None
    why = refusal(scene, st)
    if why:
        return None, why
    world = getattr(scene, 'world', None)
    vals = {}
    # exactly render._background_image's inverse: the same LAPACK call on
    # the same float32 matrix
    inv = np.linalg.inv(vp).astype(np.float32)
    for i in range(4):
        vals[f'hal_sky_inv{i}'] = tuple(float(v) for v in inv[i])
    vals['hal_sky_eye'] = tuple(float(v) for v in np.asarray(eye, np.float32))
    vals['hal_sky_size'] = (float(width), float(height))
    ss = int(ss)
    fast = ss > 1 and bool(getattr(st, 'fast_background', True))
    if fast:
        lw, lh = max(width // ss, 1), max(height // ss, 1)
        vals['hal_sky_low'] = (float(lw), float(lh), float(ss))
    else:
        vals['hal_sky_low'] = (float(width), float(height), 1.0)
    # R251: the OUTPUT pixel pitch, whether fast_background is on or
    # off -- the pixel-addressed backdrops read it
    Wo, Ho = max(width // ss, 1), max(height // ss, 1)
    vals['hal_sky_pitch'] = (float(Wo), float(Ho), float(ss))
    vals['hal_sky_alpha'] = 1.0
    # R251 LIGHT-A2 (F009): the ground fog's frame constants
    vals.update(ground_fog_params(st, eye))
    env = None
    tab = None
    m7rows = None
    m7map = None
    m7_key = None
    mode = MODE_FLAT
    if world is None:
        vals['hal_sky_color'] = (0.0, 0.0, 0.0)
    else:
        wmode = str(getattr(world, 'mode', 'NODES') or 'NODES').upper()
        strength = _f32(getattr(world, 'strength', 1.0))
        if wmode == 'NODES':
            pb = _plain_background(world.graph) if getattr(world, 'graph', None) \
                else None
            tex = _env_texture(world, textures)
            if pb is not None:
                rgb, s = pb
                # nodeeval: acc += col * s * w with w = 1.0, float32
                c = (rgb * s) * np.float32(1.0)
                vals['hal_sky_color'] = tuple(float(v) for v in c)
            elif tex is not None:
                mode = MODE_ENV
                env = tex
                vals['hal_sky_env_map'] = 1 if str(getattr(
                    world, 'env_mapping', 'EQUIRECT')) == 'MIRRORBALL' else 0
                vals['hal_sky_env_filt'] = 1
            elif getattr(world, 'sky_blend', False):
                mode = MODE_BLEND
                vals['hal_sky_hor'] = tuple(float(np.float32(v))
                                            for v in world.horizon)
                vals['hal_sky_zen'] = tuple(float(np.float32(v))
                                            for v in world.zenith)
            else:
                vals['hal_sky_color'] = tuple(float(np.float32(v))
                                              for v in world.color)
        else:
            rot = float(getattr(world, 'rotation', 0.0))
            if abs(rot) > 1e-6:
                # R253 (pre-existing, found by the cube map's twin): the
                # texel must carry sky._rotate_z's OWN (c, s) for the angle
                # -rotation -- the shader's `x*c - y*s, x*s + y*c` is that
                # function's text, and with (cos(rot), sin(rot)) it spun the
                # sky the other way: every GPU HDRI pixel under a non-zero
                # Rotation had differed from the CPU's since 1.89.0 (no
                # direction mode had been twinned with a rotation)
                vals['hal_sky_rot'] = (_f32(np.cos(-rot)), _f32(np.sin(-rot)), 1.0)
            else:
                vals['hal_sky_rot'] = (1.0, 0.0, 0.0)
            vals['hal_sky_strength'] = strength
            if wmode == 'SOLID':
                # solid() then `col * float(strength)`: constant, folded
                c = np.asarray(world.color, np.float32) * np.float32(strength)
                vals['hal_sky_color'] = tuple(float(v) for v in c)
                vals['hal_sky_strength'] = 1.0
            elif wmode in ('GRADIENT', 'BANDS'):
                mode = MODE_GRADIENT if wmode == 'GRADIENT' else MODE_BANDS
                height_ = float(world.horizon_height)
                vals['hal_sky_hor'] = tuple(float(np.float32(v))
                                            for v in world.horizon)
                vals['hal_sky_zen'] = tuple(float(np.float32(v))
                                            for v in world.zenith)
                vals['hal_sky_gnd'] = tuple(float(np.float32(v)) for v in
                                            getattr(world, 'ground_color',
                                                    (0.0, 0.0, 0.0)))
                vals['hal_sky_height'] = _f32(height_)
                vals['hal_sky_above_den'] = _f32(max(1.0 - height_, 1e-3))
                vals['hal_sky_below_den'] = _f32(max(1.0 + height_, 1e-3))
                vals['hal_sky_falloff'] = _f32(
                    max(float(world.gradient_falloff), 0.01))
                vals['hal_sky_blend'] = BLEND_CODES.get(
                    str(getattr(world, 'blend_mode', 'LINEAR')), 0)
                vals['hal_sky_ground'] = 1 if getattr(world, 'show_ground',
                                                      False) else 0
                if wmode == 'BANDS':
                    steps = max(int(getattr(world, 'band_count', 8)), 1)
                    soft = float(np.clip(getattr(world, 'band_softness', 0.0),
                                         0.0, 1.0))
                    vals['hal_sky_steps'] = steps
                    vals['hal_sky_soft'] = _f32(soft)
                    vals['hal_sky_soft_lo'] = _f32(1.0 - soft)
                    vals['hal_sky_soft_den'] = _f32(max(soft, 1e-4))
            elif wmode == 'HDRI':
                tex = _env_texture(world, textures)
                if tex is None:
                    # sky.hdri: no texture -> solid(world), then strength
                    c = np.asarray(world.color, np.float32) * np.float32(strength)
                    vals['hal_sky_color'] = tuple(float(v) for v in c)
                    vals['hal_sky_strength'] = 1.0
                else:
                    mode = MODE_HDRI
                    env = tex
                    vals['hal_sky_env_map'] = 1 if str(getattr(
                        world, 'env_mapping', 'EQUIRECT')) == 'MIRRORBALL' \
                        else 0
                    vals['hal_sky_env_filt'] = 0 if str(getattr(
                        world, 'env_filter', 'BILINEAR')) == 'NEAREST' else 1
                    vals['hal_sky_tint'] = tuple(float(np.float32(v)) for v in
                                                 getattr(world, 'env_tint',
                                                         (1.0, 1.0, 1.0)))
            elif wmode == 'CYLINDER':
                # R251 C056: the CPU hook's own calls on the same st,
                # camera and OUTPUT size, so the tables are the same bytes
                from ..core import sky as SKY
                tex = _env_texture(world, textures)
                if tex is None:
                    # no image: the solid colour then strength, hdri's rule
                    c = np.asarray(world.color, np.float32) * np.float32(strength)
                    vals['hal_sky_color'] = tuple(float(v) for v in c)
                    vals['hal_sky_strength'] = 1.0
                else:
                    mode = MODE_CYL
                    env = tex
                    proj_c, jx, jy, yaw = SKY.cylinder_inputs(scene, st, Wo, Ho)
                    tab = SKY.cylinder_tables(world, yaw, proj_c, Wo, Ho,
                                              tex.width, tex.height, jx, jy)
                    vals['hal_sky_tab_wo'] = int(Wo)
            elif wmode == 'LW_GRADIENT':
                # R251 C100: four float32 colours, whole-number squeezes
                mode = MODE_LWGRAD
                for pname, fname in (('hal_sky_lw_zen', 'lw_zenith'),
                                     ('hal_sky_lw_sky', 'lw_sky'),
                                     ('hal_sky_lw_gnd', 'lw_ground'),
                                     ('hal_sky_lw_nad', 'lw_nadir')):
                    vals[pname] = tuple(float(np.float32(v)) for v in
                                        getattr(world, fname))
                vals['hal_sky_lw_sqs'] = int(np.clip(int(getattr(
                    world, 'lw_sky_squeeze', 2)), 1, 20))
                vals['hal_sky_lw_sqg'] = int(np.clip(int(getattr(
                    world, 'lw_ground_squeeze', 2)), 1, 20))
            elif wmode == 'CUBEMAP':
                # R253: the six faces as ONE atlas on the env sampler (the
                # CPU's own bytes: sky.cube_atlas builds and caches it for
                # both roads); no image, or a malformed one, folds to the
                # solid colour then strength -- hdri's rule, bitwise
                from ..core import sky as SKY
                atlas = SKY.cube_atlas(world, textures)
                if atlas is None:
                    c = np.asarray(world.color, np.float32) * np.float32(strength)
                    vals['hal_sky_color'] = tuple(float(v) for v in c)
                    vals['hal_sky_strength'] = 1.0
                else:
                    S = int(atlas.width)
                    if 6 * S > CUBE_ATLAS_MAX_HEIGHT:
                        # a refusal by name: render._warn prints it once and
                        # the CPU draws the sky as before
                        return None, (f'the cube map faces are {S} px; the '
                                      'six-face atlas would exceed the '
                                      "driver's texture height")
                    mode = MODE_CUBE
                    env = atlas
                    filt = 0.0 if str(getattr(world, 'cube_filter',
                                              'NEAREST')) == 'NEAREST' else 1.0
                    vals['hal_sky_cube'] = (float(S), filt, 0.0)
                    vals['hal_sky_tint'] = tuple(float(np.float32(v)) for v in
                                                 getattr(world, 'env_tint',
                                                         (1.0, 1.0, 1.0)))
    # R251 C048: the Mode 7 floor, a second stage of the same draw over
    # any sky mode drawn here; without a map the CPU draws nothing either
    vals['hal_sky_m7'] = 0
    if world is not None and mode != MODE_NONE and \
            getattr(world, 'ground_plane', False) and \
            str(getattr(world, 'ground_mode', 'SOLID')) == 'MODE7':
        from ..core import sky as SKY
        m7_key = SKY.mode7_key(world)
        if m7_key is not None:
            m7map = SKY.mode7_map(world)
            m7rows = SKY.mode7_rows(world, inv, eye, Wo, Ho)
            vals['hal_sky_m7'] = 1
            vals['hal_sky_m7over'] = M7_OVER.get(
                str(getattr(world, 'mode7_over', 'WRAP')), 0)
            m7_key = ('sky_m7map',) + tuple(m7_key)
    env_key = None
    if env is not None:
        px = np.asarray(env.pixels, np.float32)
        vals['hal_sky_env_size'] = (float(px.shape[1]), float(px.shape[0]))
        import zlib
        env_key = ('sky_env', getattr(env, 'name', ''), int(px.shape[1]),
                   int(px.shape[0]), int(zlib.adler32(px[::7].tobytes())),
                   int(zlib.adler32(px[3::11].tobytes())))
    return {'mode': mode, 'params': vals, 'env': env, 'env_key': env_key,
            'tab': tab, 'm7rows': m7rows, 'm7map': m7map,
            'm7_key': m7_key}, None


def env_pixels(p):
    """The env image as the (H, W, 4) float32 texture the CPU samples."""
    env = p.get('env')
    if env is None:
        return None
    px = np.asarray(env.pixels, np.float32)
    if px.ndim == 3 and px.shape[2] == 3:
        px = np.concatenate([px, np.ones(px.shape[:2] + (1,), np.float32)], 2)
    return np.ascontiguousarray(px)


def pack_tab(tab):
    """R251 C056: the cylinder tables as a (1, Wo+Ho, 4) float32 texture,
    texel i < Wo = col[i], texel Wo+j = row[j] (ints < 2**24: exact)."""
    col, row = tab
    out = np.zeros((1, int(col.size) + int(row.size), 4), np.float32)
    out[0, :col.size, 0] = np.asarray(col, np.float32)
    out[0, col.size:, 0] = np.asarray(row, np.float32)
    return out


def pack_m7rows(rows):
    """R251 C048: the per-row registers as a (1, 2*Ho, 4) float32 texture,
    texel 2*yo = (A, C, X0, Y0), texel 2*yo+1 = (hit, 0, 0, 0)."""
    hit, A, C, X0, Y0 = rows
    Ho = int(hit.size)
    out = np.zeros((1, 2 * Ho, 4), np.float32)
    out[0, 0::2, 0] = np.asarray(A, np.float32)
    out[0, 0::2, 1] = np.asarray(C, np.float32)
    out[0, 0::2, 2] = np.asarray(X0, np.float32)
    out[0, 0::2, 3] = np.asarray(Y0, np.float32)
    out[0, 1::2, 0] = np.asarray(hit, np.float32)
    return out


def m7map_pixels(p):
    """R251 C048: the palettised map as the (1024, 1024, 4) float32 texture."""
    rgb = p.get('m7map')
    if rgb is None:
        return None
    return np.ascontiguousarray(np.concatenate(
        [np.asarray(rgb, np.float32),
         np.ones(rgb.shape[:2] + (1,), np.float32)], 2))


def push(p):
    """The draw's push constants."""
    return {'hal_sky_mode': int(p['mode'])}


# ------------------------------------------------------------- the twin


def simulate(scene, gbuf, st, vp, eye, textures=None, ss=1, debug=None):
    """The suite's twin: the pass through the GLSL front-end over the
    whole frame. Returns ((H, W, 4) float32, None) or (None, why). The
    result is what the sky draw leaves in the target: the sky with alpha
    at uncovered pixels, zeros at covered ones (and zeros everywhere for
    MODE_NONE)."""
    from ..core.texture import Texture
    from ..shaders.compiler import try_compile
    from . import gbuffer as GB
    H, W = gbuf.tri.shape
    p, why = plan(scene, st, W, H, vp, eye, textures, ss=ss)
    if p is None:
        return None, why
    src = SOURCE.replace('in vec2 vUV;', 'uniform vec2 vUV;')
    prog, err = try_compile(src, 'GLSL')
    if prog is None:
        return None, f'SKY does not compile: {err}'
    n = H * W
    yy, xx = np.mgrid[0:H, 0:W]
    uv = np.stack([(xx.ravel() + 0.5) / W, (yy.ravel() + 0.5) / H],
                  1).astype(np.float32)
    params = pack_params(p['params'])
    envpx = env_pixels(p)
    tex_ids = Texture(GB.pack_ids(gbuf), colorspace='Non-Color',
                      filt='NEAREST', wrap='EXTEND')
    tex_par = Texture(params, colorspace='Non-Color', filt='NEAREST',
                      wrap='EXTEND')
    tex_env = Texture(envpx, colorspace='Non-Color', filt='NEAREST',
                      wrap='EXTEND') if envpx is not None else tex_par
    # R251: the cylinder tables, the Mode 7 rows and map -- the params
    # texel when unused, as the env image is
    tex_tab = Texture(pack_tab(p['tab']), colorspace='Non-Color',
                      filt='NEAREST', wrap='EXTEND') \
        if p.get('tab') is not None else tex_par
    m7px = m7map_pixels(p)
    if p.get('m7rows') is not None and m7px is not None:
        tex_m7r = Texture(pack_m7rows(p['m7rows']), colorspace='Non-Color',
                          filt='NEAREST', wrap='EXTEND')
        tex_m7m = Texture(m7px, colorspace='Non-Color', filt='NEAREST',
                          wrap='EXTEND')
    else:
        tex_m7r = tex_m7m = tex_par
    u = {'vUV': uv, 'hal_gb_ids': tex_ids, 'hal_sky_params': tex_par,
         'hal_sky_env': tex_env, 'hal_sky_tab': tex_tab,
         'hal_sky_m7rows': tex_m7r, 'hal_sky_m7map': tex_m7m,
         'hal_sky_mode': np.full(n, int(p['mode']), np.int32)}
    outs, discarded = prog.run(u, {}, n)
    out = np.asarray(outs['Color'], np.float32).reshape(H, W, 4)
    # the covered pixels discard: the target keeps the material passes'
    # texels there; the standalone twin reports them as zero
    out[np.asarray(discarded, bool).reshape(H, W)] = 0.0
    if debug is not None:
        debug['plan'] = p
        debug['params'] = params
    return out, None


# ------------------------------------------------------------ the driver


LAST_TIMINGS = {}
_WARNED = set()


def _warn(msg):
    if msg not in _WARNED:
        _WARNED.add(msg)
        print(f'[Halcyon GPU] sky on the CPU: {msg}')


def prepare(scene, gbuf, st, vp, eye, textures, ss, tex_ids):
    """Everything the shading burst needs to draw the sky first: the
    compiled shader, its uniforms and samplers, or (None, why).

    Returns (draw, plan) where draw = (shader, uniforms, samplers) ready
    for the LAST `draw_many` tuple of the burst, blend NONE, no clear
    (covered pixels discard, so the material passes' texels stand; the
    uncovered ones are replaced whatever a pass left there), or (None,
    why) -- the caller then leaves the sky to the CPU. Under a
    transparent film the draw writes exact zeros (MODE_NONE)."""
    import time as _time
    from . import device
    t0 = _time.perf_counter()
    H, W = gbuf.tri.shape
    p, why = plan(scene, st, W, H, vp, eye, textures, ss=ss)
    if p is None:
        return None, why
    shader, err = device.compile_dynamic('HAL_SKY', SOURCE, interface(SOURCE))
    if shader is None:
        return None, f'the driver rejected the sky pass: {err}'
    try:
        tex_par = device.upload(pack_params(p['params']))
        if p['env'] is not None:
            tex_env = device.upload_cached(p['env_key'], lambda: env_pixels(p))
        else:
            tex_env = tex_par            # bound, never read
        # R251: the cylinder tables and the Mode 7 rows are camera-
        # dependent (per frame, never cached); the map is cached by
        # content as the HDRI env is
        tex_tab = device.upload(pack_tab(p['tab'])) \
            if p.get('tab') is not None else tex_par
        if p.get('m7rows') is not None and p.get('m7map') is not None:
            tex_m7r = device.upload(pack_m7rows(p['m7rows']))
            tex_m7m = device.upload_cached(p['m7_key'],
                                           lambda: m7map_pixels(p))
        else:
            tex_m7r = tex_m7m = tex_par
    except Exception as exc:                                    # noqa: BLE001
        return None, f'uploading the sky textures failed: {exc}'
    LAST_TIMINGS.clear()
    LAST_TIMINGS.update(mode=int(p['mode']),
                        prepare_ms=(_time.perf_counter() - t0) * 1000.0)
    return (shader, push(p), {'hal_gb_ids': tex_ids, 'hal_sky_env': tex_env,
                              'hal_sky_params': tex_par,
                              'hal_sky_tab': tex_tab,
                              'hal_sky_m7rows': tex_m7r,
                              'hal_sky_m7map': tex_m7m}), p


# R251 sky tail (LIGHT-A2 appends after this line)


# ---- R251 LIGHT-A2 (F009): POV-Ray's ground fog on a ray that hits nothing
#
# core/fog.ground_fog_sky is the CPU (the tail of render._background_image);
# this is the twin on the sky pass's own ray d0 (hal_sky_ray: the CPU's
# normalised direction, bitwise). The frame constants ride two PARAMS
# texels (hal_sky_gfog = (alt * (pi/2 - atan(y1)) as ONE float32 product,
# density, on), hal_sky_gfcol = the fog colour x fog ambient); per pixel
# only the ray's elevation d0.z is read. T = exp(-(num / dz) * density)
# for a rising ray, 0 (the fog colour) for a level or falling one. Every
# operand is computed and the ternaries select (A32: the divide by zero
# of a level ray never reaches the colour). Bitwise in the simulator; on
# the driver `/` and `exp` are its own roundings -- the sky twin's bar
# under fog_mode GROUND is d <= 2e-5 (A33), every other mode stays 0.0
# (the function returns `col` untouched when the texel says off).
GROUND_FOG_GLSL = """
// R251 LIGHT-A2 (F009): core/fog.ground_fog_sky on the sky ray
vec3 hal_ground_fog_sky(vec3 col, vec3 d0)
{
    vec3 gf = hal_sky_gfog();
    float gdz = d0.z;
    float gq = gf.x / gdz;
    float ge = gq * gf.y;
    ge = -ge;
    float gT = exp(ge);
    gT = (gdz > 0.0) ? gT : 0.0;
    vec3 gfc = hal_sky_gfcol();
    vec3 go = col * gT;
    float gg = 1.0 - gT;
    go = go + gfc * gg;
    return (gf.z > 0.5) ? go : col;
}
"""


def ground_fog_params(st, eye):
    """The two PARAMS texels of the ground fog (plan() calls this): zeros
    -- the pass leaves the sky untouched -- unless fog is on in mode
    GROUND; then core/fog.ground_fog_sky_consts's own float32 numbers."""
    if not bool(getattr(st, 'fog', False)) or \
            str(getattr(st, 'fog_mode', 'LINEAR') or 'LINEAR') != 'GROUND':
        return {'hal_sky_gfog': (0.0, 0.0, 0.0),
                'hal_sky_gfcol': (0.0, 0.0, 0.0)}
    from ..core import fog as _FOG
    num32, den32, col32 = _FOG.ground_fog_sky_consts(st, eye)
    return {'hal_sky_gfog': (float(num32), float(den32), 1.0),
            'hal_sky_gfcol': tuple(float(v) for v in col32)}


# the function joins the fragment source before main(); the call is the
# ONE line after `// R251 sky tail` (LIGHT_A2_central.py)
SOURCE_SRC = SOURCE_SRC.replace('void main()', GROUND_FOG_GLSL + 'void main()', 1)
SOURCE = SOURCE_SRC.replace('PARAM_ACCESSORS', _accessors())
SOURCES = {'SKY': SOURCE}
