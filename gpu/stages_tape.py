"""R251: the post-signal pack's tape / cable / receiver / chroma stages
(SIG-2: the GPU twins of core/signal_tape).

Registered on gpu/stages.py's three registries by the integrator's
`update` line BEFORE `ENABLED` is derived; the pack pin checks
`set(STAGES_TAPE) <= set(stages.ENABLED)`.

The lattice rule of gpu/stages_signal holds here too: integers travel in
RGBA32F targets as `float(i)` only in the ONE private intermediate
(TAPE_FIR's three filtered planes); every stage's FINAL draw writes
U8-lattice floats through `u8lut`. Every body is integer arithmetic on
the frame's own bytes -- `hal_u8` (clip, * 255, roundEven) in, table
fetches out -- with `texelFetch` only, neighbours and FIR taps clamped to
the frame before the fetch, loops with literal bounds (the FIRs run
-96..96 over a 193-texel taps row, zero beyond the radius), no `int / int`,
no `%`, no float arithmetic a driver's FMA could move. `>>` on a negative
int is an arithmetic shift (a floor), the same as NumPy's.

    CHROMA_SITE   C131  one draw; every site value is recomputed per
                        pixel from the source in the CPU's fixed order
    CABLE_CHROMA  C129  one draw, before the composite stage
    CABLE_RF      C129  one draw, after it
    PAL_DECODE    C130  one draw
    TAPE_FIR      C128  the three filtered planes as floats (private)
    TAPE_OUT      C128  tear, noise, dropouts, decode; drawn by
                        chain_tape.tape after TAPE_FIR, recorded as TAPE

`TAPE_FIR` and `TAPE_OUT` are named in stages_signal.SIGNAL_MULTI: only
the orchestrator draws them, and the self test measures them through it.
"""

from .stages_signal import _HASH, _U8

# the frame's bytes, clamped fetch
_RGB8 = _U8 + """
ivec3 hal_rgb8(ivec2 q)
{
    q = clamp(q, ivec2(0), ivec2(resolution) - ivec2(1));
    vec4 t = texelFetch(source, q, 0);
    return ivec3(hal_u8(t.r), hal_u8(t.g), hal_u8(t.b));
}
"""

# BT.601 8-bit integers (the Microsoft / Dolphin constants), one op per statement
_ENC601 = """
int hal_y601(ivec3 c)
{
    int s = 66 * c.r;
    int t = 129 * c.g;
    s = s + t;
    t = 25 * c.b;
    s = s + t;
    s = s + 128;
    s = s >> 8;
    return s + 16;
}
int hal_cb601(ivec3 c)
{
    int s = -38 * c.r;
    int t = 74 * c.g;
    s = s - t;
    t = 112 * c.b;
    s = s + t;
    s = s + 128;
    s = s >> 8;
    return s + 128;
}
int hal_cr601(ivec3 c)
{
    int s = 112 * c.r;
    int t = 94 * c.g;
    s = s - t;
    t = 18 * c.b;
    s = s - t;
    s = s + 128;
    s = s >> 8;
    return s + 128;
}
"""

_DEC601 = """
ivec3 hal_decode601(int y, int cb, int cr)
{
    int c = y - 16;
    int d = cb - 128;
    int e = cr - 128;
    int base = 298 * c;
    base = base + 128;
    int r = 409 * e;
    r = base + r;
    r = r >> 8;
    int g = 100 * d;
    g = base - g;
    int t = 208 * e;
    g = g - t;
    g = g >> 8;
    int b = 516 * d;
    b = base + b;
    b = b >> 8;
    return ivec3(clamp(r, 0, 255), clamp(g, 0, 255), clamp(b, 0, 255));
}
"""

# the picture out through the byte table
_OUT8 = """
vec3 hal_out8(ivec3 o)
{
    float r = texelFetch(u8lut, ivec2(o.r, 0), 0).r;
    float g = texelFetch(u8lut, ivec2(o.g, 0), 0).r;
    float b = texelFetch(u8lut, ivec2(o.b, 0), 0).r;
    return vec3(r, g, b);
}
"""

# C131: chroma subsampling and siting. fmt 0 Y422, 1 Y411, 2 MPEG-1,
# 3 MPEG-2, 4 PAL DV, 5 XFB. A site (i, j) is recomputed from the source
# wherever it is needed (a redundant gather: at most 6 sites x 4 pixels)
CHROMA_SITE = """
uniform sampler2D source;
uniform sampler2D u8lut;
uniform vec2 resolution;
uniform int fmt;
uniform int interp;
in vec2 vUV;
out vec4 Color;
""" + _RGB8 + _ENC601 + _DEC601 + _OUT8 + """
int hal_cb(ivec3 c) { return clamp(hal_cb601(c), 16, 240); }
int hal_cr(ivec3 c) { return clamp(hal_cr601(c), 16, 240); }
ivec2 hal_site(int si, int sj)
{
    int wd = int(resolution.x);
    int ht = int(resolution.y);
    int nx = (fmt == 1) ? ((wd + 3) >> 2) : ((wd + 1) >> 1);
    int ny = (fmt >= 2 && fmt <= 4) ? ((ht + 1) >> 1) : ht;
    int i = clamp(si, 0, nx - 1);
    int j = clamp(sj, 0, ny - 1);
    if (fmt == 5) {
        ivec3 a = hal_rgb8(ivec2(2 * i, j));
        ivec3 b = hal_rgb8(ivec2(2 * i + 1, j));
        ivec3 m = ivec3((a.r + b.r) >> 1, (a.g + b.g) >> 1, (a.b + b.b) >> 1);
        return ivec2(hal_cb(m), hal_cr(m));
    }
    if (fmt == 0) {
        ivec3 a = hal_rgb8(ivec2(2 * i, j));
        ivec3 b = hal_rgb8(ivec2(2 * i + 1, j));
        int cb = hal_cb(a) + hal_cb(b);
        cb = cb + 1;
        int cr = hal_cr(a) + hal_cr(b);
        cr = cr + 1;
        return ivec2(cb >> 1, cr >> 1);
    }
    if (fmt == 1) {
        ivec3 a = hal_rgb8(ivec2(4 * i, j));
        ivec3 b = hal_rgb8(ivec2(4 * i + 1, j));
        ivec3 c = hal_rgb8(ivec2(4 * i + 2, j));
        ivec3 d = hal_rgb8(ivec2(4 * i + 3, j));
        int cb = hal_cb(a) + hal_cb(b);
        cb = cb + hal_cb(c);
        cb = cb + hal_cb(d);
        cb = cb + 2;
        int cr = hal_cr(a) + hal_cr(b);
        cr = cr + hal_cr(c);
        cr = cr + hal_cr(d);
        cr = cr + 2;
        return ivec2(cb >> 2, cr >> 2);
    }
    ivec3 a = hal_rgb8(ivec2(2 * i, 2 * j));
    ivec3 b = hal_rgb8(ivec2(2 * i + 1, 2 * j));
    ivec3 c = hal_rgb8(ivec2(2 * i, 2 * j + 1));
    ivec3 d = hal_rgb8(ivec2(2 * i + 1, 2 * j + 1));
    if (fmt == 4) {
        int cb = hal_cb(a) + hal_cb(b);
        cb = cb + 1;
        int cr = hal_cr(c) + hal_cr(d);
        cr = cr + 1;
        return ivec2(cb >> 1, cr >> 1);
    }
    int cb = hal_cb(a) + hal_cb(b);
    cb = cb + hal_cb(c);
    cb = cb + hal_cb(d);
    cb = cb + 2;
    int cr = hal_cr(a) + hal_cr(b);
    cr = cr + hal_cr(c);
    cr = cr + hal_cr(d);
    cr = cr + 2;
    return ivec2(cb >> 2, cr >> 2);
}
ivec2 hal_hline(int x, int j)
{
    if (fmt == 1) {
        int i = x >> 2;
        int k = x & 3;
        ivec2 s0 = hal_site(i, j);
        ivec2 s1 = hal_site(i + 1, j);
        int cb = s0.x * (4 - k);
        int t = s1.x * k;
        cb = cb + t;
        cb = cb + 2;
        int cr = s0.y * (4 - k);
        t = s1.y * k;
        cr = cr + t;
        cr = cr + 2;
        return ivec2(cb >> 2, cr >> 2);
    }
    int i = x >> 1;
    int odd = x & 1;
    ivec2 s0 = hal_site(i, j);
    if (fmt == 2) {
        int nb = (odd == 0) ? (i - 1) : (i + 1);
        ivec2 s1 = hal_site(nb, j);
        int cb = 3 * s0.x;
        cb = cb + s1.x;
        cb = cb + 2;
        int cr = 3 * s0.y;
        cr = cr + s1.y;
        cr = cr + 2;
        return ivec2(cb >> 2, cr >> 2);
    }
    if (odd == 0) { return s0; }
    ivec2 s1 = hal_site(i + 1, j);
    int cb = s0.x + s1.x;
    cb = cb + 1;
    int cr = s0.y + s1.y;
    cr = cr + 1;
    return ivec2(cb >> 1, cr >> 1);
}
void main()
{
    ivec2 px = ivec2(clamp(vUV * resolution, vec2(0.0), resolution - vec2(1.0)));
    vec4 texel = texelFetch(source, px, 0);
    ivec3 c0 = hal_rgb8(px);
    int y = clamp(hal_y601(c0), 16, 235);
    int v420 = (fmt >= 2 && fmt <= 4) ? 1 : 0;
    int j = (v420 == 1) ? (px.y >> 1) : px.y;
    int cb = 128;
    int cr = 128;
    if (interp == 0) {
        int i = (fmt == 1) ? (px.x >> 2) : (px.x >> 1);
        ivec2 s = hal_site(i, j);
        cb = s.x;
        cr = s.y;
    } else {
        ivec2 h0 = hal_hline(px.x, j);
        cb = h0.x;
        cr = h0.y;
        if (v420 == 1) {
            int odd = px.y & 1;
            if (fmt == 4) {
                if (odd == 1) {
                    ivec2 h1 = hal_hline(px.x, j + 1);
                    cb = h0.x + h1.x;
                    cb = cb + 1;
                    cb = cb >> 1;
                } else {
                    ivec2 h1 = hal_hline(px.x, j - 1);
                    cr = h1.y + h0.y;
                    cr = cr + 1;
                    cr = cr >> 1;
                }
            } else {
                int nb = (odd == 0) ? (j - 1) : (j + 1);
                ivec2 h1 = hal_hline(px.x, nb);
                cb = 3 * h0.x;
                cb = cb + h1.x;
                cb = cb + 2;
                cb = cb >> 2;
                cr = 3 * h0.y;
                cr = cr + h1.y;
                cr = cr + 2;
                cr = cr >> 2;
            }
        }
    }
    ivec3 o = hal_decode601(y, cb, cr);
    Color = vec4(hal_out8(o), texel.a);
}
"""

# integer YIQ (the composite stage's matrix scaled by 256) and PAL YUV on
# the offset-free luma; signed, floor shifts
_YIQ = """
int hal_yq(ivec3 c)
{
    int s = 77 * c.r;
    int t = 150 * c.g;
    s = s + t;
    t = 29 * c.b;
    s = s + t;
    return s >> 8;
}
int hal_i(ivec3 c)
{
    int s = 153 * c.r;
    int t = 70 * c.g;
    s = s - t;
    t = 82 * c.b;
    s = s - t;
    return s >> 8;
}
int hal_q(ivec3 c)
{
    int s = 54 * c.r;
    int t = 134 * c.g;
    s = s - t;
    t = 80 * c.b;
    s = s + t;
    return s >> 8;
}
ivec3 hal_yiq_decode(int y, int i, int q)
{
    int r = 245 * i;
    int t = 158 * q;
    r = r + t;
    r = r + 128;
    r = r >> 8;
    r = y + r;
    int g = -70 * i;
    t = 166 * q;
    g = g - t;
    g = g + 128;
    g = g >> 8;
    g = y + g;
    int b = -283 * i;
    t = 436 * q;
    b = b + t;
    b = b + 128;
    b = b >> 8;
    b = y + b;
    return ivec3(clamp(r, 0, 255), clamp(g, 0, 255), clamp(b, 0, 255));
}
"""

_YUV = """
int hal_u(ivec3 c, int y)
{
    int d = c.b - y;
    d = 126 * d;
    d = d + 128;
    return d >> 8;
}
int hal_v(ivec3 c, int y)
{
    int d = c.r - y;
    d = 224 * d;
    d = d + 128;
    return d >> 8;
}
ivec3 hal_yuv_decode(int y, int u, int v)
{
    int ry = 292 * v;
    ry = ry + 128;
    ry = ry >> 8;
    int by = 520 * u;
    by = by + 128;
    by = by >> 8;
    int g = 130 * ry;
    int t = 50 * by;
    g = g + t;
    g = g + 128;
    g = g >> 8;
    g = y - g;
    int r = y + ry;
    int b = y + by;
    return ivec3(clamp(r, 0, 255), clamp(g, 0, 255), clamp(b, 0, 255));
}
"""

# C129: the S-Video cable -- the two chroma planes through their own
# Gaussian FIRs (taps.r, taps.g), luma whole. space 0 YIQ, 1 PAL YUV
CABLE_CHROMA = """
uniform sampler2D source;
uniform sampler2D taps;
uniform sampler2D u8lut;
uniform vec2 resolution;
uniform int space;
uniform int r1;
uniform int r2;
in vec2 vUV;
out vec4 Color;
""" + _RGB8 + _YIQ + _YUV + _OUT8 + """
void main()
{
    ivec2 px = ivec2(clamp(vUV * resolution, vec2(0.0), resolution - vec2(1.0)));
    vec4 texel = texelFetch(source, px, 0);
    int wmax = int(resolution.x) - 1;
    int rmax = max(r1, r2);
    int a1 = 0;
    int a2 = 0;
    for (int i = -96; i <= 96; i++) {
        int k = (i < 0) ? -i : i;
        if (k <= rmax) {
            vec4 tw = texelFetch(taps, ivec2(i + 96, 0), 0);
            int w1 = int(tw.r);
            int w2 = int(tw.g);
            ivec3 c = hal_rgb8(ivec2(clamp(px.x + i, 0, wmax), px.y));
            int c1 = 0;
            int c2 = 0;
            if (space == 0) {
                c1 = hal_i(c);
                c2 = hal_q(c);
            } else {
                int yn = hal_yq(c);
                c1 = hal_u(c, yn);
                c2 = hal_v(c, yn);
            }
            a1 += w1 * c1;
            a2 += w2 * c2;
        }
    }
    a1 = a1 + 32768;
    a1 = a1 >> 16;
    a2 = a2 + 32768;
    a2 = a2 >> 16;
    ivec3 c0 = hal_rgb8(px);
    int y = hal_yq(c0);
    ivec3 o = ivec3(0);
    if (space == 0) { o = hal_yiq_decode(y, a1, a2); } else { o = hal_yuv_decode(y, a1, a2); }
    Color = vec4(hal_out8(o), texel.a);
}
"""

_ISQRT16 = """
int hal_isqrt16(int n)
{
    int r = 0;
    for (int b = 7; b >= 0; b--) {
        int t = r + (1 << b);
        int tt = t * t;
        if (tt <= n) { r = t; }
    }
    return r;
}
"""

_ISQRT18 = """
int hal_isqrt18(int n)
{
    int r = 0;
    for (int b = 8; b >= 0; b--) {
        int t = r + (1 << b);
        int tt = t * t;
        if (tt <= n) { r = t; }
    }
    return r;
}
"""

# C129: the RF modulator -- the luma FIR, the 920 kHz beat in proportion
# to chroma amplitude, snow on Y / I / Q, the ghost reading the stage
# INPUT at x - ghost_d
CABLE_RF = """
uniform sampler2D source;
uniform sampler2D taps;
uniform sampler2D sin16;
uniform sampler2D u8lut;
uniform vec2 resolution;
uniform int r_fir;
uniform int a_ph;
uniform int b_ph;
uniform int phase_f;
uniform int amp_beat;
uniform int amp_snow;
uniform int amp_snow_c;
uniform int ghost_g;
uniform int ghost_d;
uniform int k_y;
uniform int k_i;
uniform int k_q;
in vec2 vUV;
out vec4 Color;
""" + _HASH + _RGB8 + _YIQ + _ISQRT16 + _OUT8 + """
void main()
{
    ivec2 px = ivec2(clamp(vUV * resolution, vec2(0.0), resolution - vec2(1.0)));
    vec4 texel = texelFetch(source, px, 0);
    int wmax = int(resolution.x) - 1;
    int yf = 0;
    for (int n = -96; n <= 96; n++) {
        int k = (n < 0) ? -n : n;
        if (k <= r_fir) {
            int wy = int(texelFetch(taps, ivec2(n + 96, 0), 0).r);
            ivec3 c = hal_rgb8(ivec2(clamp(px.x + n, 0, wmax), px.y));
            yf += wy * hal_yq(c);
        }
    }
    yf = yf + 32768;
    yf = yf >> 16;
    ivec3 c0 = hal_rgb8(px);
    int i = hal_i(c0);
    int q = hal_q(c0);
    int cc = i * i;
    int t = q * q;
    cc = cc + t;
    int amp = hal_isqrt16(cc);
    int ph = px.x * a_ph;
    t = px.y * b_ph;
    ph = ph + t;
    ph = ph + phase_f;
    ph = ph & 0xffff;
    int s16 = int(texelFetch(sin16, ivec2(ph >> 8, 0), 0).r);
    int beat = amp * s16;
    beat = beat >> 8;
    beat = beat * amp_beat;
    beat = beat >> 16;
    uint hy = hal_sig_hash(uint(px.x), uint(px.y), k_y);
    int nz = int(hy & 0xffffu) - 32768;
    int snow = nz * amp_snow;
    snow = snow >> 15;
    uint hi = hal_sig_hash(uint(px.x), uint(px.y), k_i);
    nz = int(hi & 0xffffu) - 32768;
    int di = nz * amp_snow_c;
    di = di >> 15;
    uint hq = hal_sig_hash(uint(px.x), uint(px.y), k_q);
    nz = int(hq & 0xffffu) - 32768;
    int dq = nz * amp_snow_c;
    dq = dq >> 15;
    ivec3 cg = hal_rgb8(ivec2(clamp(px.x - ghost_d, 0, wmax), px.y));
    int yg = ghost_g * hal_yq(cg);
    yg = yg >> 8;
    int ig = ghost_g * hal_i(cg);
    ig = ig >> 8;
    int qg = ghost_g * hal_q(cg);
    qg = qg >> 8;
    int yo = yf + beat;
    yo = yo + snow;
    yo = yo + yg;
    yo = clamp(yo, 0, 255);
    int io = i + di;
    io = io + ig;
    int qo = q + dq;
    qo = qo + qg;
    ivec3 o = hal_yiq_decode(yo, io, qo);
    Color = vec4(hal_out8(o), texel.a);
}
"""

# C130: the PAL receiver. mode 0 none, 1 delay line (the line `s_rows`
# above, the top lines keep their own chroma), 2 simple (the +-phi
# rotation in 16.16); then the crawl on the decoded chroma's amplitude
PAL_DECODE = """
uniform sampler2D source;
uniform sampler2D sin16;
uniform sampler2D u8lut;
uniform vec2 resolution;
uniform int mode;
uniform int s_rows;
uniform int c16;
uniform int s16;
uniform int a_ph;
uniform int b_ph;
uniform int phase_f;
uniform int amp16;
in vec2 vUV;
out vec4 Color;
""" + _RGB8 + _YIQ + _YUV + _ISQRT18 + _OUT8 + """
void main()
{
    ivec2 px = ivec2(clamp(vUV * resolution, vec2(0.0), resolution - vec2(1.0)));
    vec4 texel = texelFetch(source, px, 0);
    ivec3 c = hal_rgb8(px);
    int y = hal_yq(c);
    int u = hal_u(c, y);
    int v = hal_v(c, y);
    if (mode == 1) {
        int up = px.y + s_rows;
        if (up <= int(resolution.y) - 1) {
            ivec3 cu = hal_rgb8(ivec2(px.x, up));
            int yu = hal_yq(cu);
            int uu = hal_u(cu, yu);
            int vu = hal_v(cu, yu);
            u = u + uu;
            u = u >> 1;
            v = v + vu;
            v = v >> 1;
        }
    } else if (mode == 2) {
        int sg = ((px.y & 1) == 0) ? s16 : -s16;
        int a = c16 * u;
        int b = sg * v;
        a = a - b;
        a = a + 32768;
        int u2 = a >> 16;
        a = sg * u;
        b = c16 * v;
        a = a + b;
        a = a + 32768;
        int v2 = a >> 16;
        u = u2;
        v = v2;
    }
    if (amp16 > 0) {
        int cc = u * u;
        int t = v * v;
        cc = cc + t;
        int amp = hal_isqrt18(cc);
        int ph = px.x * a_ph;
        t = px.y * b_ph;
        ph = ph + t;
        ph = ph + phase_f;
        ph = ph & 0xffff;
        int s = int(texelFetch(sin16, ivec2(ph >> 8, 0), 0).r);
        int d = amp * s;
        d = d >> 8;
        d = d * amp16;
        d = d >> 23;
        y = clamp(y + d, 0, 255);
    }
    ivec3 o = hal_yuv_decode(y, u, v);
    Color = vec4(hal_out8(o), texel.a);
}
"""

# C128, draw 1: the record / playback low-pass. The target holds the
# three filtered INTEGER planes as floats (private: only TAPE_OUT reads it)
TAPE_FIR = """
uniform sampler2D source;
uniform sampler2D taps;
uniform vec2 resolution;
uniform int r_y;
uniform int r_c;
uniform int delay;
in vec2 vUV;
out vec4 Color;
""" + _RGB8 + _ENC601 + """
void main()
{
    ivec2 px = ivec2(clamp(vUV * resolution, vec2(0.0), resolution - vec2(1.0)));
    vec4 texel = texelFetch(source, px, 0);
    int wmax = int(resolution.x) - 1;
    int rmax = max(r_y, r_c);
    int ay = 0;
    int ab = 0;
    int ar = 0;
    for (int i = -96; i <= 96; i++) {
        int k = (i < 0) ? -i : i;
        if (k <= rmax) {
            vec4 tw = texelFetch(taps, ivec2(i + 96, 0), 0);
            int wy = int(tw.r);
            int wc = int(tw.g);
            ivec3 c = hal_rgb8(ivec2(clamp(px.x + i, 0, wmax), px.y));
            ay += wy * hal_y601(c);
            ivec3 cd = hal_rgb8(ivec2(clamp(px.x - delay + i, 0, wmax), px.y));
            ab += wc * hal_cb601(cd);
            ar += wc * hal_cr601(cd);
        }
    }
    ay = ay + 32768;
    ay = ay >> 16;
    ab = ab + 32768;
    ab = ab >> 16;
    ar = ar + 32768;
    ar = ar >> 16;
    Color = vec4(float(ay), float(ab), float(ar), texel.a);
}
"""

# C128, draw 2 (source = the FIR target): the head-switch tear, the
# noise (luma, then chroma), the dropouts, the decode
TAPE_OUT = """
uniform sampler2D source;
uniform sampler2D u8lut;
uniform vec2 resolution;
uniform int amp8;
uniform int head_rows;
uniform int head_j;
uniform int drop_thr;
uniform int k_y;
uniform int k_cb;
uniform int k_cr;
uniform int k_row;
uniform int k_len;
uniform int k_x0;
uniform int frame_u;
uniform int wi;
in vec2 vUV;
out vec4 Color;
""" + _HASH + _DEC601 + _OUT8 + """
void main()
{
    ivec2 px = ivec2(clamp(vUV * resolution, vec2(0.0), resolution - vec2(1.0)));
    vec4 f = texelFetch(source, px, 0);
    int y = int(f.r);
    int cb = int(f.g);
    int cr = int(f.b);
    int band = (px.y < head_rows) ? 1 : 0;
    if (band == 1) {
        int sx = clamp(px.x + head_j, 0, int(resolution.x) - 1);
        y = int(texelFetch(source, ivec2(sx, px.y), 0).r);
    }
    uint hy = hal_sig_hash(uint(px.x), uint(px.y), k_y);
    int nz = int(hy & 0xffffu) - 32768;
    int dy = nz * amp8;
    dy = (band == 1) ? (dy >> 13) : (dy >> 15);
    y = clamp(y + dy, 16, 235);
    uint hb = hal_sig_hash(uint(px.x), uint(px.y), k_cb);
    nz = int(hb & 0xffffu) - 32768;
    int dc = nz * amp8;
    dc = dc >> 16;
    cb = clamp(cb + dc, 16, 240);
    uint hr = hal_sig_hash(uint(px.x), uint(px.y), k_cr);
    nz = int(hr & 0xffffu) - 32768;
    dc = nz * amp8;
    dc = dc >> 16;
    cr = clamp(cr + dc, 16, 240);
    uint rowh = hal_sig_hash(uint(px.y), uint(frame_u), k_row);
    if (int(rowh) < drop_thr) {
        uint hl = hal_sig_hash(uint(px.y), uint(frame_u), k_len);
        int len = int(hl & 0xffu) * 41;
        len = len >> 8;
        len = len + 8;
        uint hx = hal_sig_hash(uint(px.y), uint(frame_u), k_x0);
        int x0 = int(hx & 0xffffu) * wi;
        x0 = x0 >> 16;
        if (px.x >= x0 && px.x < x0 + len) {
            y = 235;
            cb = 128;
            cr = 128;
        }
    }
    ivec3 o = hal_decode601(y, cb, cr);
    Color = vec4(hal_out8(o), f.a);
}
"""

STAGES_TAPE = {
    'CHROMA_SITE': CHROMA_SITE,
    'CABLE_CHROMA': CABLE_CHROMA,
    'CABLE_RF': CABLE_RF,
    'PAL_DECODE': PAL_DECODE,
    'TAPE_FIR': TAPE_FIR,
    'TAPE_OUT': TAPE_OUT,
}

INTERFACE_TAPE = {
    'CHROMA_SITE': {'samplers': ['source', 'u8lut'],
                    'ints': ['fmt', 'interp'], 'vec2': ['resolution']},
    'CABLE_CHROMA': {'samplers': ['source', 'taps', 'u8lut'],
                     'ints': ['space', 'r1', 'r2'], 'vec2': ['resolution']},
    # 12 ints + a vec2 = 56 bytes of push constants
    'CABLE_RF': {'samplers': ['source', 'taps', 'sin16', 'u8lut'],
                 'ints': ['r_fir', 'a_ph', 'b_ph', 'phase_f', 'amp_beat',
                          'amp_snow', 'amp_snow_c', 'ghost_g', 'ghost_d',
                          'k_y', 'k_i', 'k_q'],
                 'vec2': ['resolution']},
    'PAL_DECODE': {'samplers': ['source', 'sin16', 'u8lut'],
                   'ints': ['mode', 's_rows', 'c16', 's16', 'a_ph', 'b_ph',
                            'phase_f', 'amp16'],
                   'vec2': ['resolution']},
    'TAPE_FIR': {'samplers': ['source', 'taps'],
                 'ints': ['r_y', 'r_c', 'delay'], 'vec2': ['resolution']},
    'TAPE_OUT': {'samplers': ['source', 'u8lut'],
                 'ints': ['amp8', 'head_rows', 'head_j', 'drop_thr', 'k_y',
                          'k_cb', 'k_cr', 'k_row', 'k_len', 'k_x0',
                          'frame_u', 'wi'],
                 'vec2': ['resolution']},
}

# integer arithmetic on the frame's bytes, integer FIRs summing to 65536,
# integer hashes, table fetches: bitwise the CPU in the simulator, and
# nothing a driver's FMA or division can move
VALIDATION_TAPE = {
    'CHROMA_SITE': ('EXACT', 0.00001),
    'CABLE_CHROMA': ('EXACT', 0.00001),
    'CABLE_RF': ('EXACT', 0.00001),
    'PAL_DECODE': ('EXACT', 0.00001),
    'TAPE_FIR': ('EXACT', 0.00001),
    'TAPE_OUT': ('EXACT', 0.00001),
}
