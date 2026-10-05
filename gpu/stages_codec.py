"""R251: the post-signal pack's codec and optical-printer GPU stages (the
twins of core/signal_codec).

Registered on gpu/stages.py's three registries by the integrator's
`update` line BEFORE `ENABLED` is derived; the pack pin checks
`set(STAGES_CODEC) <= set(stages.ENABLED)`.

    MATTE_BLUR, MATTE_ADD      C134: `2 * passes + 1` draws orchestrated by
                               chain_codec.matte_glow, recorded MATTE_GLOW
    MPEG_ENC, MPEG_DCT_ROW,    C132: five draws at the padded size
    MPEG_DCT_COL_Q,            orchestrated by chain_codec.mpeg1, recorded
    MPEG_IDCT_ROW,             MPEG1
    MPEG_IDCT_COL_OUT
    SMACKER                    C133: one draw (chain_codec.smacker)

The nine component stages of the two orchestrators are in
stages_signal.SIGNAL_MULTI (the self test measures them through their
orchestrators, never alone).

The lattice rule: integers and DCT coefficients travel in RGBA32F
targets as floats only in PRIVATE intermediates (the MPEG passes, the
matte's blur targets); every stage's FINAL draw writes what the CPU
returns -- U8-lattice floats through `u8lut` (MPEG), the palette's own
float32 colours fetched from `pal` (SMACKER), or the linear frame's
floats (the matte). Every body: `uniform sampler2D source;` first,
`in vec2 vUV; out vec4 Color;`, the QUANT fetch idiom, alpha passed
through, `texelFetch` only, neighbours clamped, loops with literal
bounds, no `int / int`, no `%`, `roundEven` where the CPU uses
`np.round`, `trunc` where it uses `np.trunc`, one operation per
statement wherever a float rounding matters.

The sampler budget, measured: MATTE_ADD binds 7 (source, gel, b1..b5),
the widest stage of the pack; MPEG_DCT_COL_Q, MPEG_IDCT_COL_OUT and
SMACKER bind 3. `device.max_fragment_samplers()` never answers under 16
(the GL / Vulkan guaranteed floor), so 7 can never be refused and the
chain carries no sampler refusal (a branch no driver can reach would be
a pointless option).
"""

from .stages_signal import _IDIV, _U8

# ------------------------------------------------------- C134: the matte

# one tap set along one axis: i in -R..R in order, one multiply and one
# add per tap (the CPU's signal_codec._blur_axis), the index clamped to
# the frame. The literal loop bound is 160; the chain refuses by name a
# radius beyond it
MATTE_BLUR = """
uniform sampler2D source;
uniform sampler2D taps;
uniform vec2 resolution;
uniform int radius;
uniform int dir_x;
uniform int dir_y;
in vec2 vUV;
out vec4 Color;
void main()
{
    ivec2 px = ivec2(clamp(vUV * resolution, vec2(0.0), resolution - vec2(1.0)));
    vec4 texel = texelFetch(source, px, 0);
    ivec2 hi = ivec2(resolution) - ivec2(1);
    vec3 acc = vec3(0.0);
    for (int i = -160; i <= 160; i++) {
        if (i >= -radius && i <= radius) {
            float wt = texelFetch(taps, ivec2(i + 160, 0), 0).r;
            ivec2 q = clamp(px + ivec2(i * dir_x, i * dir_y), ivec2(0), hi);
            vec3 v = texelFetch(source, q, 0).rgb;
            vec3 t = v * wt;
            acc = acc + t;
        }
    }
    Color = vec4(acc, texel.a);
}
"""

# the negative's sum: the frame, the crisp exposure, then passes 1..N in
# order, each `o = o + t`; no upper clamp (the display stage's clip is
# the film's shoulder)
MATTE_ADD = """
uniform sampler2D source;
uniform sampler2D gel;
uniform sampler2D b1;
uniform sampler2D b2;
uniform sampler2D b3;
uniform sampler2D b4;
uniform sampler2D b5;
uniform vec2 resolution;
uniform int passes;
uniform float e;
uniform float w1;
uniform float w2;
uniform float w3;
uniform float w4;
uniform float w5;
in vec2 vUV;
out vec4 Color;
void main()
{
    ivec2 px = ivec2(clamp(vUV * resolution, vec2(0.0), resolution - vec2(1.0)));
    vec4 texel = texelFetch(source, px, 0);
    vec3 g = texelFetch(gel, px, 0).rgb;
    vec3 t = g * e;
    vec3 o = texel.rgb + t;
    if (passes >= 1) { vec3 v = texelFetch(b1, px, 0).rgb; t = v * w1; o = o + t; }
    if (passes >= 2) { vec3 v = texelFetch(b2, px, 0).rgb; t = v * w2; o = o + t; }
    if (passes >= 3) { vec3 v = texelFetch(b3, px, 0).rgb; t = v * w3; o = o + t; }
    if (passes >= 4) { vec3 v = texelFetch(b4, px, 0).rgb; t = v * w4; o = o + t; }
    if (passes >= 5) { vec3 v = texelFetch(b5, px, 0).rgb; t = v * w5; o = o + t; }
    o = max(o, vec3(0.0));
    Color = vec4(o, texel.a);
}
"""

# ------------------------------------------------------ C132: MPEG-1

_YCC = """
int hal_y601(ivec3 c)
{
    int s = 66 * c.r + 129 * c.g + 25 * c.b + 128;
    return (s >> 8) + 16;
}
int hal_cb601(ivec3 c)
{
    int s = 112 * c.b - 38 * c.r - 74 * c.g + 128;
    return (s >> 8) + 128;
}
int hal_cr601(ivec3 c)
{
    int s = 112 * c.r - 94 * c.g - 18 * c.b + 128;
    return (s >> 8) + 128;
}
"""

# the frame (w x h) encoded into the padded target (Wp x Hp): every fetch
# is clamped to the FRAME's edge (`src_size`), never to `resolution - 1`
# -- that is the padded draw's edge, beyond the frame texture, where a
# driver's texelFetch is undefined (the simulator's clamps and would hide
# it). r = Y - 128; g, b = the 2x2 cell's Cb - 128, Cr - 128
MPEG_ENC = """
uniform sampler2D source;
uniform vec2 resolution;
uniform vec2 src_size;
in vec2 vUV;
out vec4 Color;
""" + _U8 + _YCC + """
ivec3 hal_bytes(ivec2 q, ivec2 hi)
{
    ivec2 c = clamp(q, ivec2(0), hi);
    vec4 t = texelFetch(source, c, 0);
    return ivec3(hal_u8(t.r), hal_u8(t.g), hal_u8(t.b));
}
void main()
{
    ivec2 px = ivec2(clamp(vUV * resolution, vec2(0.0), resolution - vec2(1.0)));
    ivec2 hi = ivec2(src_size) - ivec2(1);
    ivec2 pc = clamp(px, ivec2(0), hi);
    vec4 texel = texelFetch(source, pc, 0);
    ivec3 c0 = hal_bytes(px, hi);
    int y = hal_y601(c0) - 128;
    ivec2 cell = ivec2((px.x >> 1) << 1, (px.y >> 1) << 1);
    ivec3 qa = hal_bytes(cell, hi);
    ivec3 qb = hal_bytes(cell + ivec2(1, 0), hi);
    ivec3 qc = hal_bytes(cell + ivec2(0, 1), hi);
    ivec3 qd = hal_bytes(cell + ivec2(1, 1), hi);
    int cb = hal_cb601(qa) + hal_cb601(qb) + hal_cb601(qc) + hal_cb601(qd) + 2;
    cb = (cb >> 2) - 128;
    int cr = hal_cr601(qa) + hal_cr601(qb) + hal_cr601(qc) + hal_cr601(qd) + 2;
    cr = (cr >> 2) - 128;
    Color = vec4(float(y), float(cb), float(cr), texel.a);
}
"""

# row DCT: luma over the pixel's own 8 columns, chroma over the cell's
# (the chroma sample of cell i sits at texel 2 i)
MPEG_DCT_ROW = """
uniform sampler2D source;
uniform sampler2D dct;
uniform vec2 resolution;
in vec2 vUV;
out vec4 Color;
void main()
{
    ivec2 px = ivec2(clamp(vUV * resolution, vec2(0.0), resolution - vec2(1.0)));
    vec4 texel = texelFetch(source, px, 0);
    int u = px.x & 7;
    int col0 = (px.x >> 3) << 3;
    int i = px.x >> 1;
    int uc = i & 7;
    int c0 = (i >> 3) << 3;
    float ay = 0.0;
    float ab = 0.0;
    float ar = 0.0;
    for (int k = 0; k < 8; k++) {
        float d = texelFetch(dct, ivec2(k, u), 0).r;
        float v = texelFetch(source, ivec2(col0 + k, px.y), 0).r;
        float t = d * v;
        ay = ay + t;
        float dc = texelFetch(dct, ivec2(k, uc), 0).r;
        vec4 s = texelFetch(source, ivec2(2 * (c0 + k), px.y), 0);
        t = dc * s.g;
        ab = ab + t;
        t = dc * s.b;
        ar = ar + t;
    }
    Color = vec4(ay, ab, ar, texel.a);
}
"""

_QUANT = """
float hal_quant(float X, int u, int v)
{
    float dc = X * 0.125;
    dc = roundEven(dc);
    dc = dc * 8.0;
    vec4 w = texelFetch(qtab, ivec2(u, v), 0);
    float t = X * 16.0;
    t = t * w.g;
    float s = 0.0;
    if (X > 0.0) { s = 0.5; }
    if (X < 0.0) { s = -0.5; }
    t = t + s;
    int qf = int(trunc(t));
    int m = abs(qf);
    m = m * qs;
    m = m * int(w.r);
    m = m >> 3;
    if (m != 0 && (m & 1) == 0) { m = m - 1; }
    if (qf < 0) { m = -m; }
    m = clamp(m, -2048, 2047);
    float r = float(m);
    if (u == 0 && v == 0) { r = dc; }
    return r;
}
"""

# column DCT, then the quantiser and its reconstruction (the DC in steps
# of 8, the AC levels truncated, oddified toward zero)
MPEG_DCT_COL_Q = """
uniform sampler2D source;
uniform sampler2D dct;
uniform sampler2D qtab;
uniform vec2 resolution;
uniform int qs;
in vec2 vUV;
out vec4 Color;
""" + _QUANT + """
void main()
{
    ivec2 px = ivec2(clamp(vUV * resolution, vec2(0.0), resolution - vec2(1.0)));
    vec4 texel = texelFetch(source, px, 0);
    int v = px.y & 7;
    int row0 = (px.y >> 3) << 3;
    int j = px.y >> 1;
    int vc = j & 7;
    int r0 = (j >> 3) << 3;
    float ay = 0.0;
    float ab = 0.0;
    float ar = 0.0;
    for (int k = 0; k < 8; k++) {
        float d = texelFetch(dct, ivec2(k, v), 0).r;
        float sv = texelFetch(source, ivec2(px.x, row0 + k), 0).r;
        float t = d * sv;
        ay = ay + t;
        float dc = texelFetch(dct, ivec2(k, vc), 0).r;
        vec4 s = texelFetch(source, ivec2(px.x, 2 * (r0 + k)), 0);
        t = dc * s.g;
        ab = ab + t;
        t = dc * s.b;
        ar = ar + t;
    }
    int u = px.x & 7;
    int uc = (px.x >> 1) & 7;
    float qy = hal_quant(ay, u, v);
    float qb = hal_quant(ab, uc, vc);
    float qr = hal_quant(ar, uc, vc);
    Color = vec4(qy, qb, qr, texel.a);
}
"""

# row IDCT: D[k][u] is texel (u, k)
MPEG_IDCT_ROW = """
uniform sampler2D source;
uniform sampler2D dct;
uniform vec2 resolution;
in vec2 vUV;
out vec4 Color;
void main()
{
    ivec2 px = ivec2(clamp(vUV * resolution, vec2(0.0), resolution - vec2(1.0)));
    vec4 texel = texelFetch(source, px, 0);
    int u = px.x & 7;
    int col0 = (px.x >> 3) << 3;
    int i = px.x >> 1;
    int uc = i & 7;
    int c0 = (i >> 3) << 3;
    float ay = 0.0;
    float ab = 0.0;
    float ar = 0.0;
    for (int k = 0; k < 8; k++) {
        float d = texelFetch(dct, ivec2(u, k), 0).r;
        float v = texelFetch(source, ivec2(col0 + k, px.y), 0).r;
        float t = v * d;
        ay = ay + t;
        float dc = texelFetch(dct, ivec2(uc, k), 0).r;
        vec4 s = texelFetch(source, ivec2(2 * (c0 + k), px.y), 0);
        t = s.g * dc;
        ab = ab + t;
        t = s.b * dc;
        ar = ar + t;
    }
    Color = vec4(ay, ab, ar, texel.a);
}
"""

# column IDCT, the decoder's rounding (half to even) and the BT.601
# decode, drawn at the FRAME's size: the padded source is fetched at px,
# never clamped
MPEG_IDCT_COL_OUT = """
uniform sampler2D source;
uniform sampler2D dct;
uniform sampler2D u8lut;
uniform vec2 resolution;
in vec2 vUV;
out vec4 Color;
int hal_byte(float acc)
{
    float o = roundEven(acc);
    o = o + 128.0;
    o = clamp(o, 0.0, 255.0);
    return int(o);
}
void main()
{
    ivec2 px = ivec2(clamp(vUV * resolution, vec2(0.0), resolution - vec2(1.0)));
    vec4 texel = texelFetch(source, px, 0);
    int v = px.y & 7;
    int row0 = (px.y >> 3) << 3;
    int j = px.y >> 1;
    int vc = j & 7;
    int r0 = (j >> 3) << 3;
    float ay = 0.0;
    float ab = 0.0;
    float ar = 0.0;
    for (int k = 0; k < 8; k++) {
        float d = texelFetch(dct, ivec2(v, k), 0).r;
        float sv = texelFetch(source, ivec2(px.x, row0 + k), 0).r;
        float t = sv * d;
        ay = ay + t;
        float dc = texelFetch(dct, ivec2(vc, k), 0).r;
        vec4 s = texelFetch(source, ivec2(px.x, 2 * (r0 + k)), 0);
        t = s.g * dc;
        ab = ab + t;
        t = s.b * dc;
        ar = ar + t;
    }
    int c = hal_byte(ay) - 16;
    int d8 = hal_byte(ab) - 128;
    int e8 = hal_byte(ar) - 128;
    int r = (298 * c + 409 * e8 + 128) >> 8;
    int g = (298 * c - 100 * d8 - 208 * e8 + 128) >> 8;
    int b = (298 * c + 516 * d8 + 128) >> 8;
    r = clamp(r, 0, 255);
    g = clamp(g, 0, 255);
    b = clamp(b, 0, 255);
    float fr = texelFetch(u8lut, ivec2(r, 0), 0).r;
    float fg = texelFetch(u8lut, ivec2(g, 0), 0).r;
    float fb = texelFetch(u8lut, ivec2(b, 0), 0).r;
    Color = vec4(fr, fg, fb, texel.a);
}
"""

# ------------------------------------------------------ C133: Smacker

# one 4x4 block per pixel, in the CPU's own integers: the sixteen bytes,
# their lumas, the fill / mono / full decision, and this pixel's entry.
# The palette's float32 colour is FETCHED (never recomputed). The
# sixteen fetches are written out with LITERAL array indices (generated
# here): the front-end's simulator stores into an array only at a
# constant index (it reads at any), and an unrolled store is plain GLSL
_SMK_FETCH = ''.join("""
    c = hal_blk(b0 + ivec2(%d, %d), hi);
    p[%d] = c;
    l = (77 * c.r + 150 * c.g + 29 * c.b) >> 8;
    lum[%d] = l;
    sum = sum + c;
    lmin = min(lmin, l);
    lmax = max(lmax, l);""" % (i & 3, i >> 2, i, i) for i in range(16))

SMACKER = """
uniform sampler2D source;
uniform sampler2D pal;
uniform sampler2D icm;
uniform vec2 resolution;
uniform float inv255;
uniform int t_fill;
uniform int t_full;
in vec2 vUV;
out vec4 Color;
""" + _U8 + _IDIV + """
int hal_near(ivec3 c)
{
    int qr = hal_idiv(c.r * 64, 255, inv255);
    qr = min(qr, 63);
    int qg = hal_idiv(c.g * 64, 255, inv255);
    qg = min(qg, 63);
    int qb = hal_idiv(c.b * 64, 255, inv255);
    qb = min(qb, 63);
    int i = (qr << 12) | (qg << 6) | qb;
    return int(texelFetch(icm, ivec2(i & 511, i >> 9), 0).r);
}
ivec3 hal_pal8(int i)
{
    vec3 c = texelFetch(pal, ivec2(i, 0), 0).rgb;
    return ivec3(hal_u8(c.r), hal_u8(c.g), hal_u8(c.b));
}
ivec3 hal_blk(ivec2 q, ivec2 hi)
{
    ivec2 qc = clamp(q, ivec2(0), hi);
    vec4 t = texelFetch(source, qc, 0);
    return ivec3(hal_u8(t.r), hal_u8(t.g), hal_u8(t.b));
}
int hal_d2(ivec3 a, ivec3 b)
{
    ivec3 d = a - b;
    return d.r * d.r + d.g * d.g + d.b * d.b;
}
void main()
{
    ivec2 px = ivec2(clamp(vUV * resolution, vec2(0.0), resolution - vec2(1.0)));
    vec4 texel = texelFetch(source, px, 0);
    ivec2 b0 = ivec2((px.x >> 2) << 2, (px.y >> 2) << 2);
    ivec2 hi = ivec2(resolution) - ivec2(1);
    ivec3 p[16];
    int lum[16];
    ivec3 sum = ivec3(0);
    int lmin = 255;
    int lmax = 0;
    ivec3 c;
    int l;""" + _SMK_FETCH + """
    ivec3 mine = ivec3(hal_u8(texel.r), hal_u8(texel.g), hal_u8(texel.b));
    ivec3 mean = (sum + ivec3(8)) >> 4;
    int idx = hal_near(mean);
    if (lmax - lmin >= t_fill) {
        ivec3 s0 = ivec3(0);
        ivec3 s1 = ivec3(0);
        for (int i = 0; i < 16; i++) {
            int r = 0;
            for (int j = 0; j < 16; j++) {
                if (lum[j] < lum[i]) { r += 1; }
                else if (lum[j] == lum[i] && j < i) { r += 1; }
            }
            if (r < 8) { s0 = s0 + p[i]; } else { s1 = s1 + p[i]; }
        }
        ivec3 m0 = (s0 + ivec3(4)) >> 3;
        ivec3 m1 = (s1 + ivec3(4)) >> 3;
        int i0 = hal_near(m0);
        int i1 = hal_near(m1);
        ivec3 p0 = hal_pal8(i0);
        ivec3 p1 = hal_pal8(i1);
        int E = 0;
        for (int i = 0; i < 16; i++) {
            int d0 = hal_d2(p[i], p0);
            int d1 = hal_d2(p[i], p1);
            E = E + min(d0, d1);
        }
        if (E > t_full) { idx = hal_near(mine); }
        else {
            int e0 = hal_d2(mine, p0);
            int e1 = hal_d2(mine, p1);
            idx = (e0 <= e1) ? i0 : i1;
        }
    }
    Color = vec4(texelFetch(pal, ivec2(idx, 0), 0).rgb, texel.a);
}
"""

STAGES_CODEC = {
    'MATTE_BLUR': MATTE_BLUR,
    'MATTE_ADD': MATTE_ADD,
    'MPEG_ENC': MPEG_ENC,
    'MPEG_DCT_ROW': MPEG_DCT_ROW,
    'MPEG_DCT_COL_Q': MPEG_DCT_COL_Q,
    'MPEG_IDCT_ROW': MPEG_IDCT_ROW,
    'MPEG_IDCT_COL_OUT': MPEG_IDCT_COL_OUT,
    'SMACKER': SMACKER,
}

INTERFACE_CODEC = {
    'MATTE_BLUR': {'samplers': ['source', 'taps'],
                   'ints': ['radius', 'dir_x', 'dir_y'],
                   'vec2': ['resolution']},
    'MATTE_ADD': {'samplers': ['source', 'gel', 'b1', 'b2', 'b3', 'b4', 'b5'],
                  'floats': ['e', 'w1', 'w2', 'w3', 'w4', 'w5'],
                  'ints': ['passes'], 'vec2': ['resolution']},
    'MPEG_ENC': {'samplers': ['source'], 'vec2': ['resolution', 'src_size']},
    'MPEG_DCT_ROW': {'samplers': ['source', 'dct'], 'vec2': ['resolution']},
    'MPEG_DCT_COL_Q': {'samplers': ['source', 'dct', 'qtab'], 'ints': ['qs'],
                       'vec2': ['resolution']},
    'MPEG_IDCT_ROW': {'samplers': ['source', 'dct'], 'vec2': ['resolution']},
    'MPEG_IDCT_COL_OUT': {'samplers': ['source', 'dct', 'u8lut'],
                          'vec2': ['resolution']},
    'SMACKER': {'samplers': ['source', 'pal', 'icm'], 'floats': ['inv255'],
                'ints': ['t_fill', 't_full'], 'vec2': ['resolution']},
}

# MATTE_*: bitwise the CPU in the simulator (float32 taps in a fixed
# order, one op per statement); a driver's FMA contraction of
# `t = v * wt; acc = acc + t` is an ulp per tap (the GRAIN precedent) --
# the bar is 5e-4, read in the self test.
# MPEG_*: bitwise the CPU in the simulator (fixed-order float32 sums,
# the quantiser's reciprocal fetched from the CPU's table); under a
# driver an FMA contraction can move a coefficient by an ulp and a
# truncation across a boundary by one quantiser step, which the IDCT
# spreads as at most one 8-bit level: the bar is 1/255, read in the
# self test (the NTSC precedent).
# SMACKER: integer reductions in a fixed order, table fetches only.
VALIDATION_CODEC = {
    'MATTE_BLUR': ('CLOSE', 0.0005),
    'MATTE_ADD': ('CLOSE', 0.0005),
    'MPEG_ENC': ('CLOSE', 0.004),
    'MPEG_DCT_ROW': ('CLOSE', 0.004),
    'MPEG_DCT_COL_Q': ('CLOSE', 0.004),
    'MPEG_IDCT_ROW': ('CLOSE', 0.004),
    'MPEG_IDCT_COL_OUT': ('CLOSE', 0.004),
    'SMACKER': ('EXACT', 0.00001),
}
