"""R251: the era colour stages of the post chain (post-palette pack).

GLSL bodies in the front-end's accepted subset, each the bitwise twin of a
CPU function in core/palette_era.py, core/dither.py or core/palette.py:
one op per statement wherever the CPU's rounding matters, `texelFetch`
with in-range integer indices only, `roundEven` for np.round, no `int /
int`, no `texture()`, no `mix()`, no division except by a power of two
-- every other quotient is a CPU-built table texel. One `uniform` per
line (tests/fakedevice checks the first declaration of a line only).

`STAGES`, `INTERFACE`, `VALIDATION` are merged into gpu/stages.py by three
`update` lines above its `ENABLED` derivation.

  PALETTE  the inverse-colormap palette snap (P0): the CPU's own 64^3
           index table as a 512 x 512 texture, the palette as a row, an
           optional ordered dither (DI.ordered_palette's perturbation);
           C061's register snap rides it through the palette texture
  ORDERED  the ordered dither to a bit depth (P1): DI.ordered_bits --
           the tile texture, one multiply, one add, roundEven, the QUANT
           level table
  EHB      Extra Half-Brite (C054): the 4:4:4 lattice round and a 64-way
           integer argmin table over the 32 registers and their halves
  CRY16    the Atari Jaguar's CRY pixel (C011): integer end to end from
           three constant tables
  YJK      the MSX2+ V9958's YJK pixel (C059): four texel fetches per
           aligned group, integer sums, one half-even round of the mean
  SUPERBLACK  Super Black (C092): an IEEE max against the CPU's own
           T / 255 where the packed coverage plane is set
  LEGALISE Video Color Check (C093): the composite envelope test and
           3ds Max's three corrections (the pack's one CLOSE stage:
           one sqrt, one division)
  CELLS_FIT / CELLS_SNAP  attribute cells (C050), two passes: the
           least-squares set per cell, then the nearest of its colours
           per pixel -- integers after the 8-bit round

Hardware grade of the ordered roads (PALETTE with a dither, ORDERED, EHB
with a dither): the perturbation is a multiply followed by an add that a
driver may contract to an FMA; the simulator is bitwise, the driver's
number is Run Self Test's (expected 0 of 921,600 at 720p). The NONE roads
(one multiply, a truncation or roundEven, integer fetches) are bitwise on
every device.
"""

# P0: the palette snap. InverseColormap.lookup is q = clip(int32(c * 64),
# 0, 63) per channel (one multiply, truncation toward zero) and the packed
# code (r << 12) | (g << 6) | b indexes the CPU's own LUT; the dither is
# DI.ordered_palette's pert = img + ((tm - 0.5) * strength * spacing),
# left to right in float32.
PALETTE = """
uniform sampler2D source;
uniform sampler2D icm;
uniform sampler2D pal;
uniform sampler2D tile;
uniform vec2 resolution;
uniform float strength;
uniform float spacing;
uniform int tile_mask;
uniform int dither_on;
in vec2 vUV;
out vec4 Color;
void main()
{
    ivec2 px = ivec2(clamp(vUV * resolution, vec2(0.0), resolution - vec2(1.0)));
    vec4 texel = texelFetch(source, px, 0);
    vec3 c = texel.rgb;
    if (dither_on == 1) {
        ivec2 tp = ivec2(px.x & tile_mask, px.y & tile_mask);
        float t = texelFetch(tile, tp, 0).r;
        t = t - 0.5;
        t = t * strength;
        t = t * spacing;
        c = c + vec3(t);
    }
    c = clamp(c, vec3(0.0), vec3(1.0));
    vec3 q = c * 64.0;
    ivec3 qi = ivec3(int(q.r), int(q.g), int(q.b));
    qi = clamp(qi, ivec3(0), ivec3(63));
    int code = qi.r * 4096 + qi.g * 64 + qi.b;
    int cy = code >> 9;
    int cx = code & 511;
    int idx = int(texelFetch(icm, ivec2(cx, cy), 0).r);
    vec4 p = texelFetch(pal, ivec2(idx, 0), 0);
    Color = vec4(p.rgb, texel.a);
}
"""

# P1: the ordered dither to a bit depth. DI.ordered_bits per channel:
# v = clip(c) * levels; v = v + (tm - 0.5) * strength; clip(round(v), 0,
# levels) / levels -- the quotient fetched from the CPU's own k / levels
# table (chain.quant_lut).
ORDERED = """
uniform sampler2D source;
uniform sampler2D tile;
uniform sampler2D lut;
uniform vec2 resolution;
uniform vec3 levels;
uniform float strength;
uniform int tile_mask;
in vec2 vUV;
out vec4 Color;
void main()
{
    ivec2 px = ivec2(clamp(vUV * resolution, vec2(0.0), resolution - vec2(1.0)));
    vec4 texel = texelFetch(source, px, 0);
    ivec2 tp = ivec2(px.x & tile_mask, px.y & tile_mask);
    float t = texelFetch(tile, tp, 0).r;
    t = t - 0.5;
    t = t * strength;
    vec3 v = clamp(texel.rgb, vec3(0.0), vec3(1.0));
    v = v * levels;
    v = v + vec3(t);
    v = roundEven(v);
    v = clamp(v, vec3(0.0), levels);
    float r = texelFetch(lut, ivec2(int(v.r), 0), 0).r;
    float g = texelFetch(lut, ivec2(int(v.g), 0), 0).g;
    float b = texelFetch(lut, ivec2(int(v.b), 0), 0).b;
    Color = vec4(r, g, b, texel.a);
}
"""

# C054: Extra Half-Brite. palette_era.ehb_reduce: the ordered
# perturbation as PALETTE's, then q = round(clip(c) * 15) half to even,
# code = (r << 8) | (g << 4) | b into the 64-way argmin table (a 64 x 64
# index texture: code >> 6, code & 63), the register from the 64-row
# palette texture.
EHB = """
uniform sampler2D source;
uniform sampler2D ehb_lut;
uniform sampler2D pal;
uniform sampler2D tile;
uniform vec2 resolution;
uniform float strength;
uniform float spacing;
uniform int tile_mask;
uniform int dither_on;
in vec2 vUV;
out vec4 Color;
void main()
{
    ivec2 px = ivec2(clamp(vUV * resolution, vec2(0.0), resolution - vec2(1.0)));
    vec4 texel = texelFetch(source, px, 0);
    vec3 c = texel.rgb;
    if (dither_on == 1) {
        ivec2 tp = ivec2(px.x & tile_mask, px.y & tile_mask);
        float t = texelFetch(tile, tp, 0).r;
        t = t - 0.5;
        t = t * strength;
        t = t * spacing;
        c = c + vec3(t);
    }
    c = clamp(c, vec3(0.0), vec3(1.0));
    vec3 q = c * 15.0;
    q = roundEven(q);
    ivec3 qi = ivec3(int(q.r), int(q.g), int(q.b));
    int code = qi.r * 256 + qi.g * 16 + qi.b;
    int cy = code >> 6;
    int cx = code & 63;
    int idx = int(texelFetch(ehb_lut, ivec2(cx, cy), 0).r);
    vec4 p = texelFetch(pal, ivec2(idx, 0), 0);
    Color = vec4(p.rgb, texel.a);
}
"""

# C011: the Jaguar's CRY pixel. palette_era.cry16: R8 = round(clip(c) *
# 255) half to even; I = max(r, g, b); the FIRST maximal channel picks the
# face (r, then g, then b); ca, cb = CRY_DIV[max(I, 1), other channels];
# cr = CRY_ARGMIN[face, ca, cb]; t = CRY_TABLE[cr]; out = CRY_MUL[I, t];
# the frame's float from the k / 255 table. Integer end to end.
CRY16 = """
uniform sampler2D source;
uniform sampler2D cry_tabs;
uniform sampler2D cry_argmin;
uniform sampler2D cry_table;
uniform sampler2D lut255;
uniform vec2 resolution;
in vec2 vUV;
out vec4 Color;
void main()
{
    ivec2 px = ivec2(clamp(vUV * resolution, vec2(0.0), resolution - vec2(1.0)));
    vec4 texel = texelFetch(source, px, 0);
    vec3 q = clamp(texel.rgb, vec3(0.0), vec3(1.0));
    q = q * 255.0;
    q = roundEven(q);
    int r8 = int(q.r);
    int g8 = int(q.g);
    int b8 = int(q.b);
    int I = max(r8, max(g8, b8));
    int Is = max(I, 1);
    int f = 2;
    int a = r8;
    int b = g8;
    if (g8 == I) {
        f = 1;
        a = r8;
        b = b8;
    }
    if (r8 == I) {
        f = 0;
        a = g8;
        b = b8;
    }
    int ca = int(texelFetch(cry_tabs, ivec2(a, Is), 0).r);
    int cb = int(texelFetch(cry_tabs, ivec2(b, Is), 0).r);
    int cr = int(texelFetch(cry_argmin, ivec2(f * 256 + ca, cb), 0).r);
    vec4 t = texelFetch(cry_table, ivec2(cr, 0), 0);
    int tr = int(t.r);
    int tg = int(t.g);
    int tb = int(t.b);
    int orr = int(texelFetch(cry_tabs, ivec2(256 + tr, I), 0).r);
    int og = int(texelFetch(cry_tabs, ivec2(256 + tg, I), 0).r);
    int ob = int(texelFetch(cry_tabs, ivec2(256 + tb, I), 0).r);
    float R = texelFetch(lut255, ivec2(orr, 0), 0).r;
    float G = texelFetch(lut255, ivec2(og, 0), 0).g;
    float B = texelFetch(lut255, ivec2(ob, 0), 0).b;
    Color = vec4(R, G, B, texel.a);
}
"""

# C059: the V9958's YJK pixel. palette_era.yjk: every pixel refetches its
# own aligned group of four (the right edge repeats its last texel),
# integer sums, J and K = roundEven(sum * 0.25) clamped to -32..31, the
# decode with floor divisions by powers of two (floor(n * 0.25)), the
# 5-bit value widened as (v << 3) | (v >> 2) and fetched from k / 255.
YJK = """
uniform sampler2D source;
uniform sampler2D lut255;
uniform vec2 resolution;
in vec2 vUV;
out vec4 Color;
void main()
{
    ivec2 px = ivec2(clamp(vUV * resolution, vec2(0.0), resolution - vec2(1.0)));
    vec4 texel = texelFetch(source, px, 0);
    int x0 = px.x - (px.x & 3);
    int xmax = int(resolution.x) - 1;
    int sj = 0;
    int sk = 0;
    int ys = 0;
    for (int i = 0; i < 4; i++) {
        int xi = min(x0 + i, xmax);
        vec4 t = texelFetch(source, ivec2(xi, px.y), 0);
        vec3 q = clamp(t.rgb, vec3(0.0), vec3(1.0));
        q = q * 255.0;
        q = roundEven(q);
        int r = int(floor(q.r * 0.125));
        int g = int(floor(q.g * 0.125));
        int b = int(floor(q.b * 0.125));
        int n = -2 * r - g + 2;
        int fl = int(floor(float(n) * 0.25));
        int u = b - fl;
        int yy = int(floor(float(u + 1) * 0.5));
        sj = sj + (r - yy);
        sk = sk + (g - yy);
        if (xi == px.x) {
            ys = yy;
        }
    }
    float jf = roundEven(float(sj) * 0.25);
    float kf = roundEven(float(sk) * 0.25);
    int J = clamp(int(jf), -32, 31);
    int K = clamp(int(kf), -32, 31);
    int R = clamp(ys + J, 0, 31);
    int G = clamp(ys + K, 0, 31);
    int nb = 5 * ys - 2 * J - K + 2;
    int B = clamp(int(floor(float(nb) * 0.25)), 0, 31);
    int r8 = R * 8 + (R >> 2);
    int g8 = G * 8 + (G >> 2);
    int b8 = B * 8 + (B >> 2);
    float fr = texelFetch(lut255, ivec2(r8, 0), 0).r;
    float fg = texelFetch(lut255, ivec2(g8, 0), 0).g;
    float fb = texelFetch(lut255, ivec2(b8, 0), 0).b;
    Color = vec4(fr, fg, fb, texel.a);
}
"""

# ---- wave 2 (PAL-2) ----

# C092: Super Black. palette_era.super_black is np.maximum(rgb, thr) where
# the coverage plane is set; the plane rides a packed texture (64 pixels a
# texel, 16 bits a channel as exact integer-valued floats), the floor is
# the CPU's own float32 quotient T / 255 as a uniform.
SUPERBLACK = """
uniform sampler2D source;
uniform sampler2D coverage;
uniform vec2 resolution;
uniform float threshold;
in vec2 vUV;
out vec4 Color;
void main()
{
    ivec2 px = ivec2(clamp(vUV * resolution, vec2(0.0), resolution - vec2(1.0)));
    vec4 texel = texelFetch(source, px, 0);
    int cx = px.x >> 6;
    int ch = (px.x >> 4) & 3;
    int bit = px.x & 15;
    vec4 t = texelFetch(coverage, ivec2(cx, px.y), 0);
    float wf = t.r;
    if (ch == 1) { wf = t.g; }
    if (ch == 2) { wf = t.b; }
    if (ch == 3) { wf = t.a; }
    int w = int(wf);
    int on = (w >> bit) & 1;
    vec3 c = texel.rgb;
    if (on == 1) { c = max(c, vec3(threshold)); }
    Color = vec4(c, texel.a);
}
"""

# C093: Video Color Check. palette_era.video_color_check statement for
# statement: the catalogue's 3-decimal YIQ, the squared-amplitude envelope
# test (root-free), then the mode's correction. FLAG_BLACK takes no root
# and no division; the scale modes take one sqrt and one division each
# (the CLOSE grade: 1 ulp + 2.5 ulp on a driver, bitwise in the simulator).
LEGALISE = """
uniform sampler2D source;
uniform vec2 resolution;
uniform int mode;
uniform float hi;
uniform float lo;
in vec2 vUV;
out vec4 Color;
void main()
{
    ivec2 px = ivec2(clamp(vUV * resolution, vec2(0.0), resolution - vec2(1.0)));
    vec4 texel = texelFetch(source, px, 0);
    vec3 c = texel.rgb;
    float t = 0.0;
    float y = c.r * 0.299;
    t = c.g * 0.587;
    y = y + t;
    t = c.b * 0.114;
    y = y + t;
    float i = c.r * 0.596;
    t = c.g * 0.274;
    i = i - t;
    t = c.b * 0.322;
    i = i - t;
    float q = c.r * 0.211;
    t = c.g * 0.523;
    q = q - t;
    t = c.b * 0.312;
    q = q + t;
    float ii = i * i;
    float qq = q * q;
    float c2 = ii + qq;
    float a = hi - y;
    float b = y - lo;
    float aa = a * a;
    float bb = b * b;
    bool peak = (a < 0.0) || (c2 > aa);
    bool trough = (c2 > bb);
    if (mode == 1) {
        if (peak || trough) { c = vec3(0.0); }
    }
    if (mode == 2) {
        float yy = y;
        float cc = sqrt(c2);
        if (peak) {
            float d = yy + cc;
            float s = hi / d;
            c = c * s;
            yy = yy * s;
            cc = cc * s;
        }
        float b2 = yy - lo;
        if (cc > b2 && cc > 1e-6) {
            float k = b2 / cc;
            k = min(k, 1.0);
            k = max(k, 0.0);
            vec3 dlt = c - vec3(yy);
            vec3 e = dlt * k;
            c = vec3(yy) + e;
        }
    }
    if (mode == 3) {
        float cs = sqrt(c2);
        if ((peak || trough) && cs > 1e-6) {
            float k1 = a / cs;
            float k2 = b / cs;
            float ks = min(k1, k2);
            ks = min(ks, 1.0);
            ks = max(ks, 0.0);
            vec3 ds = c - vec3(y);
            vec3 es = ds * ks;
            c = vec3(y) + es;
        }
    }
    Color = vec4(clamp(c, vec3(0.0), vec3(1.0)), texel.a);
}
"""

# C050: attribute cells, two passes. CELLS_FIT draws one texel per
# CELL: the lowest set index k minimising the cell's summed squared error
# (per pixel the nearest of the set's colours), integers end to end after
# the CPU's 8-bit round -- palette_era.attribute_cells' cost / argmin.
# A partial cell at the top or right edge is the pixels it has. Strict <
# counting up = np.argmin's lowest index. Measured through the CELLS
# pair; never drawn on its own.
CELLS_FIT = """
uniform sampler2D source;
uniform sampler2D sets;
uniform sampler2D tile;
uniform vec2 resolution;
uniform vec2 frame_size;
uniform int cell_w;
uniform int cell_h;
uniform int n_sets;
uniform int n_colors;
uniform int tile_mask;
uniform int dither_on;
uniform float strength;
uniform float spacing;
in vec2 vUV;
out vec4 Color;
void main()
{
    ivec2 cp = ivec2(clamp(vUV * resolution, vec2(0.0), resolution - vec2(1.0)));
    int x0 = cp.x * cell_w;
    int y0 = cp.y * cell_h;
    int W = int(frame_size.x);
    int H = int(frame_size.y);
    int best = 0;
    int best_cost = 2147483647;
    for (int k = 0; k < 560; k++) {
        if (k >= n_sets) { break; }
        int cost = 0;
        for (int j = 0; j < 8; j++) {
            if (j >= cell_h) { break; }
            for (int i = 0; i < 8; i++) {
                if (i >= cell_w) { break; }
                int x = x0 + i;
                int y = y0 + j;
                if (x < W && y < H) {
                    vec4 t = texelFetch(source, ivec2(x, y), 0);
                    vec3 c = t.rgb;
                    if (dither_on == 1) {
                        float d = texelFetch(tile, ivec2(x & tile_mask, y & tile_mask), 0).r;
                        d = d - 0.5;
                        d = d * strength;
                        d = d * spacing;
                        c = c + vec3(d);
                    }
                    c = clamp(c, vec3(0.0), vec3(1.0));
                    c = c * 255.0;
                    c = roundEven(c);
                    int r = int(c.r);
                    int g = int(c.g);
                    int b = int(c.b);
                    int dmin = 2147483647;
                    for (int s = 0; s < 4; s++) {
                        if (s >= n_colors) { break; }
                        vec4 e = texelFetch(sets, ivec2(k, s), 0);
                        int dr = r - int(e.r);
                        int dg = g - int(e.g);
                        int db = b - int(e.b);
                        int dd = dr * dr + dg * dg + db * db;
                        if (dd < dmin) { dmin = dd; }
                    }
                    cost = cost + dmin;
                }
            }
        }
        if (cost < best_cost) { best_cost = cost; best = k; }
    }
    Color = vec4(float(best), 0.0, 0.0, 1.0);
}
"""

# CELLS_SNAP: every pixel takes the nearest colour of ITS cell's chosen
# set (the same perturbed 8-bit value, the same integer distance, lowest s
# on ties), returned through the CPU's own k / 255 table.
CELLS_SNAP = """
uniform sampler2D source;
uniform sampler2D cells;
uniform sampler2D sets;
uniform sampler2D tile;
uniform sampler2D lut255;
uniform vec2 resolution;
uniform int cell_shift_x;
uniform int cell_shift_y;
uniform int n_colors;
uniform int tile_mask;
uniform int dither_on;
uniform float strength;
uniform float spacing;
in vec2 vUV;
out vec4 Color;
void main()
{
    ivec2 px = ivec2(clamp(vUV * resolution, vec2(0.0), resolution - vec2(1.0)));
    vec4 texel = texelFetch(source, px, 0);
    vec3 c = texel.rgb;
    if (dither_on == 1) {
        float d = texelFetch(tile, ivec2(px.x & tile_mask, px.y & tile_mask), 0).r;
        d = d - 0.5;
        d = d * strength;
        d = d * spacing;
        c = c + vec3(d);
    }
    c = clamp(c, vec3(0.0), vec3(1.0));
    c = c * 255.0;
    c = roundEven(c);
    int r = int(c.r);
    int g = int(c.g);
    int b = int(c.b);
    int k = int(texelFetch(cells, ivec2(px.x >> cell_shift_x, px.y >> cell_shift_y), 0).r);
    int best_s = 0;
    int dmin = 2147483647;
    for (int s = 0; s < 4; s++) {
        if (s >= n_colors) { break; }
        vec4 e = texelFetch(sets, ivec2(k, s), 0);
        int dr = r - int(e.r);
        int dg = g - int(e.g);
        int db = b - int(e.b);
        int dd = dr * dr + dg * dg + db * db;
        if (dd < dmin) { dmin = dd; best_s = s; }
    }
    vec4 ef = texelFetch(sets, ivec2(k, best_s), 0);
    float R = texelFetch(lut255, ivec2(int(ef.r), 0), 0).r;
    float G = texelFetch(lut255, ivec2(int(ef.g), 0), 0).g;
    float B = texelFetch(lut255, ivec2(int(ef.b), 0), 0).b;
    Color = vec4(R, G, B, texel.a);
}
"""

# WAVE2-STAGES-END

STAGES = {
    'PALETTE': PALETTE,
    'ORDERED': ORDERED,
    'EHB': EHB,
    'CRY16': CRY16,
    'YJK': YJK,
    'SUPERBLACK': SUPERBLACK,
    'LEGALISE': LEGALISE,
    'CELLS_FIT': CELLS_FIT,
    'CELLS_SNAP': CELLS_SNAP,
}

INTERFACE = {
    'PALETTE': {'samplers': ['source', 'icm', 'pal', 'tile'],
                'floats': ['strength', 'spacing'],
                'ints': ['tile_mask', 'dither_on'],
                'vec2': ['resolution']},
    'ORDERED': {'samplers': ['source', 'tile', 'lut'],
                'floats': ['strength'], 'ints': ['tile_mask'],
                'vec2': ['resolution'], 'vec3': ['levels']},
    'EHB': {'samplers': ['source', 'ehb_lut', 'pal', 'tile'],
            'floats': ['strength', 'spacing'],
            'ints': ['tile_mask', 'dither_on'],
            'vec2': ['resolution']},
    'CRY16': {'samplers': ['source', 'cry_tabs', 'cry_argmin', 'cry_table',
                           'lut255'],
              'vec2': ['resolution']},
    'YJK': {'samplers': ['source', 'lut255'], 'vec2': ['resolution']},
    'SUPERBLACK': {'samplers': ['source', 'coverage'],
                   'floats': ['threshold'], 'vec2': ['resolution']},
    'LEGALISE': {'samplers': ['source'], 'floats': ['hi', 'lo'],
                 'ints': ['mode'], 'vec2': ['resolution']},
    'CELLS_FIT': {'samplers': ['source', 'sets', 'tile'],
                  'floats': ['strength', 'spacing'],
                  'ints': ['cell_w', 'cell_h', 'n_sets', 'n_colors',
                           'tile_mask', 'dither_on'],
                  'vec2': ['resolution', 'frame_size']},
    'CELLS_SNAP': {'samplers': ['source', 'cells', 'sets', 'tile',
                                'lut255'],
                   'floats': ['strength', 'spacing'],
                   'ints': ['cell_shift_x', 'cell_shift_y', 'n_colors',
                            'tile_mask', 'dither_on'],
                   'vec2': ['resolution']},
}

# R251: every stage is bitwise its CPU function in the simulator (integer
# index arithmetic on CPU-built tables; the only float ops are the CPU's
# own, one per statement). The ordered roads' driver number is Run Self
# Test's (a multiply-add a driver may contract, expected 0 of 921,600).
VALIDATION = {
    'PALETTE': ('EXACT', 0.00001),
    'ORDERED': ('EXACT', 0.00001),
    'EHB': ('EXACT', 0.00001),
    'CRY16': ('EXACT', 0.00001),
    'YJK': ('EXACT', 0.00001),
    # wave 2: a max against a pre-rounded uniform and integer bit tests
    'SUPERBLACK': ('EXACT', 0.00001),
    # one sqrt (1 ulp) and one division (2.5 ulp) in the scale modes, and
    # a multiply-add a driver may contract: within 1e-5 on hardware,
    # bitwise in the simulator; FLAG_BLACK is root-free and can differ
    # only by a whole pixel sitting on the envelope (measured, 0 expected)
    'LEGALISE': ('CLOSE', 0.00001),
    # the CELLS pair: integer sums and compares on the CPU's 8-bit
    # round; CELLS_FIT is measured through the pair, never drawn on its
    # own (the NTSC_BLUR shape)
    'CELLS_FIT': ('EXACT', 0.00001),
    'CELLS_SNAP': ('EXACT', 0.00001),
}
