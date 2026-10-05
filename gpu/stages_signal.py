"""R251: the post-signal pack's GPU stages (the twins of core/signal_era).

Registered on gpu/stages.py's three registries by the integrator's
`update` lines BEFORE `ENABLED` is derived; the pack pin checks
`set(STAGES_SIGNAL) <= set(stages.ENABLED)`.

The lattice rule: integers travel in RGBA32F targets as `float(i)` only
in PRIVATE intermediates; `Frame.finish` returns the LAST target as the
picture, so every stage's FINAL draw writes U8-lattice floats through
`u8lut` (the CPU's own `i / 255` table, fetched, never divided). Every
body: `uniform sampler2D source;` first, `in vec2 vUV; out vec4 Color;`,
the QUANT fetch idiom, alpha passed through as `texel.a`, `texelFetch`
only, neighbours clamped to the frame, loops with literal bounds, no
`int / int`, no `%`, `roundEven` where the CPU uses `np.round`, one
operation per statement wherever a float rounding matters. The shared
helpers (`hal_sig_hash` == film._hash_u32_raw, `hal_idiv` exact whatever
the driver's division does) are pasted into every body that needs them
(CreateInfo compiles each stage alone).

`SIGNAL_MULTI` names the component stages only an orchestrator draws
(VOODOO_LINE is drawn four times by chain_signal.video_filter); the self
test skips them in its per-stage loop and measures them through their
orchestrator.
"""

# the pack's shared GLSL helpers (pasted; CreateInfo compiles each stage alone)
_HASH = """
uint hal_sig_hash(uint x, uint y, int k)
{
    uint h = x * 0x9E3779B1u ^ (y + 0x85EBCA77u) ^ (uint(k) * 0xC2B2AE3Du);
    h ^= h >> 15u;
    h *= 0x2C1B3C6Du;
    h ^= h >> 12u;
    h *= 0x297A2D39u;
    h ^= h >> 15u;
    return h & 0xffffffu;
}
"""

_IDIV = """
int hal_idiv(int s, int d, float inv)
{
    int q = int(floor(float(s) * inv));
    int r = s - q * d;
    if (r < 0) { q -= 1; }
    if (r >= d) { q += 1; }
    return q;
}
"""

_FIELD = """
int hal_field(float v, float lv)
{
    float t = clamp(v, 0.0, 1.0);
    t = t * lv;
    t = roundEven(t);
    return int(t);
}
"""

_U8 = """
int hal_u8(float v)
{
    float t = clamp(v, 0.0, 1.0);
    t = t * 255.0;
    t = roundEven(t);
    return int(t);
}
"""

# C005: the N64 VI -- dedither (+-1 toward each of eight neighbours on
# the 8-bit expansion), then the gamma / gamma-dither / dither-only LUTs
VI = """
uniform sampler2D source;
uniform sampler2D u8lut;
uniform sampler2D glut;
uniform vec2 resolution;
uniform vec3 levels;
uniform int shift_r;
uniform int shift_g;
uniform int shift_b;
uniform int dedither;
uniform int gamma_mode;
uniform int key;
in vec2 vUV;
out vec4 Color;
""" + _HASH + _FIELD + """
int hal_vi_channel(int ch, int k, int shift, ivec2 px, float lv)
{
    int c8 = k << shift;
    if (dedither == 1) {
        int acc = 0;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0) { continue; }
                ivec2 q = clamp(px + ivec2(dx, dy), ivec2(0), ivec2(resolution) - ivec2(1));
                vec4 tn = texelFetch(source, q, 0);
                float vn = (ch == 0) ? tn.r : ((ch == 1) ? tn.g : tn.b);
                int n = hal_field(vn, lv);
                if (n > k) { acc += 1; } else if (n < k) { acc -= 1; }
            }
        }
        c8 = clamp(c8 + acc, 0, 255);
    }
    return c8;
}
void main()
{
    ivec2 px = ivec2(clamp(vUV * resolution, vec2(0.0), resolution - vec2(1.0)));
    vec4 texel = texelFetch(source, px, 0);
    int kr = hal_field(texel.r, levels.x);
    int kg = hal_field(texel.g, levels.y);
    int kb = hal_field(texel.b, levels.z);
    int r8 = hal_vi_channel(0, kr, shift_r, px, levels.x);
    int g8 = hal_vi_channel(1, kg, shift_g, px, levels.y);
    int b8 = hal_vi_channel(2, kb, shift_b, px, levels.z);
    if (gamma_mode != 0) {
        uint h = hal_sig_hash(uint(px.x), uint(px.y), key);
        if (gamma_mode == 1) {
            r8 = int(texelFetch(glut, ivec2(r8, 0), 0).r);
            g8 = int(texelFetch(glut, ivec2(g8, 0), 0).r);
            b8 = int(texelFetch(glut, ivec2(b8, 0), 0).r);
        } else if (gamma_mode == 2) {
            int ir = (r8 << 6) | int(h & 63u);
            int ig = (g8 << 6) | int((h >> 6u) & 63u);
            int ib = (b8 << 6) | int((h >> 12u) & 63u);
            r8 = int(texelFetch(glut, ivec2(ir & 127, ir >> 7), 0).r);
            g8 = int(texelFetch(glut, ivec2(ig & 127, ig >> 7), 0).r);
            b8 = int(texelFetch(glut, ivec2(ib & 127, ib >> 7), 0).r);
        } else {
            r8 = min(255, r8 + int(h & 1u));
            g8 = min(255, g8 + int((h >> 1u) & 1u));
            b8 = min(255, b8 + int((h >> 2u) & 1u));
        }
    }
    float r = texelFetch(u8lut, ivec2(r8, 0), 0).r;
    float g = texelFetch(u8lut, ivec2(g8, 0), 0).r;
    float b = texelFetch(u8lut, ivec2(b8, 0), 0).r;
    Color = vec4(r, g, b, texel.a);
}
"""

# C019: the GameCube / Wii copy filter -- three lines in 1/64 units,
# floored; row 0 is the bottom, so "the line above" is y + 1
COPY_FILTER = """
uniform sampler2D source;
uniform sampler2D u8lut;
uniform vec2 resolution;
uniform int tap_u;
uniform int tap_m;
uniform int tap_l;
in vec2 vUV;
out vec4 Color;
""" + _U8 + """
int hal_copy(int a, int c, int b)
{
    int s = tap_u * a;
    int t = tap_m * c;
    s = s + t;
    t = tap_l * b;
    s = s + t;
    s = s >> 6;
    return min(s, 255);
}
void main()
{
    ivec2 px = ivec2(clamp(vUV * resolution, vec2(0.0), resolution - vec2(1.0)));
    ivec2 hi = ivec2(resolution) - ivec2(1);
    vec4 texel = texelFetch(source, px, 0);
    vec4 ta = texelFetch(source, clamp(px + ivec2(0, 1), ivec2(0), hi), 0);
    vec4 tb = texelFetch(source, clamp(px - ivec2(0, 1), ivec2(0), hi), 0);
    int r8 = hal_copy(hal_u8(ta.r), hal_u8(texel.r), hal_u8(tb.r));
    int g8 = hal_copy(hal_u8(ta.g), hal_u8(texel.g), hal_u8(tb.g));
    int b8 = hal_copy(hal_u8(ta.b), hal_u8(texel.b), hal_u8(tb.b));
    float r = texelFetch(u8lut, ivec2(r8, 0), 0).r;
    float g = texelFetch(u8lut, ivec2(g8, 0), 0).r;
    float b = texelFetch(u8lut, ivec2(b8, 0), 0).r;
    Color = vec4(r, g, b, texel.a);
}
"""

# C033: the PS2 CRTC PMODE mix -- (cur * A + other * (255 - A)) / 255 in
# integers, the divide exact through hal_idiv; `other` is the previous
# output (mode 0, the `prev` texture) or BGCOLOR (mode 1)
CRTC_BLEND = """
uniform sampler2D source;
uniform sampler2D u8lut;
uniform sampler2D prev;
uniform vec2 resolution;
uniform vec3 bg;
uniform int alpha;
uniform int mode;
uniform float inv255;
in vec2 vUV;
out vec4 Color;
""" + _U8 + _IDIV + """
int hal_mix(int c, int p)
{
    int s = c * alpha;
    int t = p * (255 - alpha);
    s = s + t;
    return hal_idiv(s, 255, inv255);
}
void main()
{
    ivec2 px = ivec2(clamp(vUV * resolution, vec2(0.0), resolution - vec2(1.0)));
    vec4 texel = texelFetch(source, px, 0);
    vec3 other = bg;
    if (mode == 0) { other = texelFetch(prev, px, 0).rgb; }
    int r8 = hal_mix(hal_u8(texel.r), hal_u8(other.r));
    int g8 = hal_mix(hal_u8(texel.g), hal_u8(other.g));
    int b8 = hal_mix(hal_u8(texel.b), hal_u8(other.b));
    float r = texelFetch(u8lut, ivec2(r8, 0), 0).r;
    float g = texelFetch(u8lut, ivec2(g8, 0), 0).r;
    float b = texelFetch(u8lut, ivec2(b8, 0), 0).r;
    Color = vec4(r, g, b, texel.a);
}
"""

# C069 (VOODOO1): one 2-tap pass of the Voodoo Graphics line; drawn four
# times (direction -1, -1, -1, +1) by chain_signal.video_filter. `expand`
# is 1 on the first draw (the source is the 5:6:5 float frame) and 0
# after (the source holds U8 floats)
VOODOO_LINE = """
uniform sampler2D source;
uniform sampler2D u8lut;
uniform vec2 resolution;
uniform int direction;
uniform int cap;
uniform int expand;
in vec2 vUV;
out vec4 Color;
""" + _FIELD + """
int hal_in(float v, int ch)
{
    if (expand == 0) { return hal_field(v, 255.0); }
    if (ch == 1) { int k = hal_field(v, 63.0); return (k << 2) | (k >> 4); }
    int k = hal_field(v, 31.0);
    return (k << 3) | (k >> 2);
}
int hal_half(int c, int n)
{
    int d = n - c;
    d = clamp(d, -cap, cap);
    d = d >> 1;
    return c + d;
}
void main()
{
    ivec2 px = ivec2(clamp(vUV * resolution, vec2(0.0), resolution - vec2(1.0)));
    vec4 texel = texelFetch(source, px, 0);
    int r = hal_in(texel.r, 0);
    int g = hal_in(texel.g, 1);
    int b = hal_in(texel.b, 2);
    int nx = px.x + direction;
    if (nx >= 0 && nx < int(resolution.x)) {
        vec4 tn = texelFetch(source, ivec2(nx, px.y), 0);
        r = hal_half(r, hal_in(tn.r, 0));
        g = hal_half(g, hal_in(tn.g, 1));
        b = hal_half(b, hal_in(tn.b, 2));
    }
    r = clamp(r, 0, 255);
    g = clamp(g, 0, 255);
    b = clamp(b, 0, 255);
    float fr = texelFetch(u8lut, ivec2(r, 0), 0).r;
    float fg = texelFetch(u8lut, ivec2(g, 0), 0).r;
    float fb = texelFetch(u8lut, ivec2(b, 0), 0).r;
    Color = vec4(fr, fg, fb, texel.a);
}
"""

# C069 (VOODOO2): the one-sided look-ahead line, one draw, five fetches:
# the cascade against the raw source at -3, -2, -1, +1
VOODOO_LOOK = """
uniform sampler2D source;
uniform sampler2D u8lut;
uniform vec2 resolution;
uniform int cap;
uniform int cap32;
uniform float inv5;
in vec2 vUV;
out vec4 Color;
""" + _FIELD + _IDIV + """
int hal_in565(float v, int ch)
{
    if (ch == 1) { int k = hal_field(v, 63.0); return (k << 2) | (k >> 4); }
    int k = hal_field(v, 31.0);
    return (k << 3) | (k >> 2);
}
int hal_step(int g, int h)
{
    int d = h - g;
    if (d <= 0 || d > cap) { return g; }
    int a = 4 * g;
    a = a + h;
    a = hal_idiv(a, 5, inv5);
    int b = 4 * h;
    b = b + g;
    b = hal_idiv(b, 5, inv5);
    int ad = b - a;
    ad = min(ad, cap32);
    return g + ad;
}
void main()
{
    ivec2 px = ivec2(clamp(vUV * resolution, vec2(0.0), resolution - vec2(1.0)));
    vec4 texel = texelFetch(source, px, 0);
    int wmax = int(resolution.x) - 1;
    int r = hal_in565(texel.r, 0);
    int g = hal_in565(texel.g, 1);
    int b = hal_in565(texel.b, 2);
    for (int i = 0; i < 4; i++) {
        int k = (i == 0) ? -3 : ((i == 1) ? -2 : ((i == 2) ? -1 : 1));
        vec4 tn = texelFetch(source, ivec2(clamp(px.x + k, 0, wmax), px.y), 0);
        r = hal_step(r, hal_in565(tn.r, 0));
        g = hal_step(g, hal_in565(tn.g, 1));
        b = hal_step(b, hal_in565(tn.b, 2));
    }
    r = clamp(r, 0, 255);
    g = clamp(g, 0, 255);
    b = clamp(b, 0, 255);
    float fr = texelFetch(u8lut, ivec2(r, 0), 0).r;
    float fg = texelFetch(u8lut, ivec2(g, 0), 0).r;
    float fb = texelFetch(u8lut, ivec2(b, 0), 0).r;
    Color = vec4(fr, fg, fb, texel.a);
}
"""

STAGES_SIGNAL = {
    'VI': VI,
    'COPY_FILTER': COPY_FILTER,
    'CRTC_BLEND': CRTC_BLEND,
    'VOODOO_LINE': VOODOO_LINE,
    'VOODOO_LOOK': VOODOO_LOOK,
}

INTERFACE_SIGNAL = {
    'VI': {'samplers': ['source', 'u8lut', 'glut'],
           'ints': ['shift_r', 'shift_g', 'shift_b', 'dedither',
                    'gamma_mode', 'key'],
           'vec2': ['resolution'], 'vec3': ['levels']},
    'COPY_FILTER': {'samplers': ['source', 'u8lut'],
                    'ints': ['tap_u', 'tap_m', 'tap_l'],
                    'vec2': ['resolution']},
    'CRTC_BLEND': {'samplers': ['source', 'u8lut', 'prev'],
                   'floats': ['inv255'], 'ints': ['alpha', 'mode'],
                   'vec2': ['resolution'], 'vec3': ['bg']},
    'VOODOO_LINE': {'samplers': ['source', 'u8lut'],
                    'ints': ['direction', 'cap', 'expand'],
                    'vec2': ['resolution']},
    'VOODOO_LOOK': {'samplers': ['source', 'u8lut'],
                    'floats': ['inv5'], 'ints': ['cap', 'cap32'],
                    'vec2': ['resolution']},
}

# every stage here is integer arithmetic, table fetches and roundEven on
# the QUANT's own products: bitwise the CPU in the simulator, and
# nothing a driver's FMA or division can move (hal_idiv is exact
# whatever the division rounds to)
VALIDATION_SIGNAL = {
    'VI': ('EXACT', 0.00001),
    'COPY_FILTER': ('EXACT', 0.00001),
    'CRTC_BLEND': ('EXACT', 0.00001),
    'VOODOO_LINE': ('EXACT', 0.00001),
    'VOODOO_LOOK': ('EXACT', 0.00001),
}

#: the component stages only an orchestrator draws (the self test skips
#: them in its per-stage loop, and measures them through their
#: orchestrators): SIG-1's VOODOO_LINE and, named here once for the
#: whole pack, the wave-2 modules' (a name absent from STAGES skips
#: nothing)
SIGNAL_MULTI = ('VOODOO_LINE', 'TAPE_FIR', 'TAPE_OUT', 'MPEG_ENC',
                'MPEG_DCT_ROW', 'MPEG_DCT_COL_Q', 'MPEG_IDCT_ROW',
                'MPEG_IDCT_COL_OUT', 'MATTE_BLUR', 'MATTE_ADD')
