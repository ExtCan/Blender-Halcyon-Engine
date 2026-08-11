"""The period pattern library, as GLSL: patterns.py moved, not reinvented.

Halcyon's own procedural textures (Marble, Wood, Granite, Dents, Crackle and
the rest) ride an INTEGER hash -- multiply-accumulate, xorshift, mask -- and
that is what makes them portable at all: masking a uint32 product to 31 bits
equals masking an exact int64 product to 31 bits, because 2^31 divides 2^32.
The front-end learned real uint semantics for exactly this file, so every
function below is verified against its patterns.py original through the same
compiler that runs the deferred pass headlessly.

Blender's Noise/Voronoi/White Noise/Musgrave textures are a different story:
their hash is fract(sin(x)*43758.5453) evaluated in float64 on the CPU, and a
driver's float32 sin decorrelates completely after that amplification. Those
refuse by name rather than render a different picture -- the reasons live in
emit.REFUSED.

Every function here mirrors its patterns.py original line for line, in the
same operation order, so a change on one side is a diff away from being seen
on the other.
"""

#: the primitives, shared by every pattern below. Split from the patterns so
#: a shader carries only what its material actually uses.
PRIM_GLSL = """
// --- patterns.py primitives, exactly ------------------------------------
// hash3: (ix*K1 + iy*K2 + iz*K3) & 0x7fffffff, xorshift, 16-bit mask.
// uint32 wrap reproduces the CPU's int64-then-mask arithmetic bit for bit.
float hal_pt_hash3(int ix, int iy, int iz)
{
    uint h = (uint(ix) * 374761393u + uint(iy) * 668265263u
              + uint(iz) * 1274126177u) & 0x7fffffffu;
    h = (h ^ (h >> 13u)) * 1274126177u;
    return float((h ^ (h >> 16u)) & 0xffffu) / 65535.0;
}

float hal_pt_hash3f(vec3 p, float salt)
{
    vec3 c = floor(p);
    return hal_pt_hash3(int(c.x), int(c.y), int(c.z + salt));
}

// trilinear value noise in 0..1, patterns.value_noise
float hal_pt_vnoise(vec3 p)
{
    vec3 i = floor(p);
    vec3 f = p - i;
    f = f * f * (3.0 - 2.0 * f);
    int ix = int(i.x);
    int iy = int(i.y);
    int iz = int(i.z);
    float c000 = hal_pt_hash3(ix, iy, iz);
    float c100 = hal_pt_hash3(ix + 1, iy, iz);
    float c010 = hal_pt_hash3(ix, iy + 1, iz);
    float c110 = hal_pt_hash3(ix + 1, iy + 1, iz);
    float c001 = hal_pt_hash3(ix, iy, iz + 1);
    float c101 = hal_pt_hash3(ix + 1, iy, iz + 1);
    float c011 = hal_pt_hash3(ix, iy + 1, iz + 1);
    float c111 = hal_pt_hash3(ix + 1, iy + 1, iz + 1);
    float x00 = c000 + (c100 - c000) * f.x;
    float x10 = c010 + (c110 - c010) * f.x;
    float x01 = c001 + (c101 - c001) * f.x;
    float x11 = c011 + (c111 - c011) * f.x;
    float y0 = x00 + (x10 - x00) * f.y;
    float y1 = x01 + (x11 - x01) * f.y;
    return y0 + (y1 - y0) * f.z;
}

// patterns._hash_mix reached through 1/2/4 lattice terms: the same
// accumulate-mask-xorshift family, one extra constant for the 4th axis
float hal_pt_mix16(uint h)
{
    h = h & 0x7fffffffu;
    h = (h ^ (h >> 13u)) * 1274126177u;
    return float((h ^ (h >> 16u)) & 0xffffu) / 65535.0;
}

// patterns.value_noise1: linear value noise on a 1D lattice
float hal_pt_vnoise1(float x)
{
    float fl = floor(x);
    float f = x - fl;
    f = f * f * (3.0 - 2.0 * f);
    uint b = uint(int(fl)) * 374761393u;
    float c0 = hal_pt_mix16(b);
    float c1 = hal_pt_mix16(b + 374761393u);
    return c0 + (c1 - c0) * f;
}

// patterns.value_noise2: bilinear value noise on a 2D lattice
float hal_pt_vnoise2(vec2 p)
{
    vec2 fl = floor(p);
    vec2 f = p - fl;
    f = f * f * (3.0 - 2.0 * f);
    uint b = uint(int(fl.x)) * 374761393u + uint(int(fl.y)) * 668265263u;
    float c00 = hal_pt_mix16(b);
    float c10 = hal_pt_mix16(b + 374761393u);
    float c01 = hal_pt_mix16(b + 668265263u);
    float c11 = hal_pt_mix16(b + 374761393u + 668265263u);
    float x0 = c00 + (c10 - c00) * f.x;
    float x1 = c01 + (c11 - c01) * f.x;
    return x0 + (x1 - x0) * f.y;
}

// patterns.value_noise4: quadrilinear value noise on a 4D lattice --
// the seamless-loop axis pair lives in (z, w)
float hal_pt_vn4_plane(uint b, vec2 f)
{
    float c00 = hal_pt_mix16(b);
    float c10 = hal_pt_mix16(b + 374761393u);
    float c01 = hal_pt_mix16(b + 668265263u);
    float c11 = hal_pt_mix16(b + 374761393u + 668265263u);
    float x0 = c00 + (c10 - c00) * f.x;
    float x1 = c01 + (c11 - c01) * f.x;
    return x0 + (x1 - x0) * f.y;
}

float hal_pt_vnoise4(vec4 p)
{
    vec4 fl = floor(p);
    vec4 f = p - fl;
    f = f * f * (3.0 - 2.0 * f);
    uint b = uint(int(fl.x)) * 374761393u + uint(int(fl.y)) * 668265263u
             + uint(int(fl.z)) * 1274126177u
             + uint(int(fl.w)) * 1911520717u;
    float z0w0 = hal_pt_vn4_plane(b, f.xy);
    float z1w0 = hal_pt_vn4_plane(b + 1274126177u, f.xy);
    float z0w1 = hal_pt_vn4_plane(b + 1911520717u, f.xy);
    float z1w1 = hal_pt_vn4_plane(b + 1274126177u + 1911520717u, f.xy);
    float w0 = z0w0 + (z1w0 - z0w0) * f.z;
    float w1 = z0w1 + (z1w1 - z0w1) * f.z;
    return w0 + (w1 - w0) * f.w;
}

// patterns.fbm, normalised octave sum
float hal_pt_fbm(vec3 p, int octaves, float lacunarity, float gain)
{
    float total = 0.0;
    float amp = 1.0;
    float norm = 0.0;
    float freq = 1.0;
    for (int i = 0; i < octaves; i++) {
        total = total + hal_pt_vnoise(p * freq) * amp;
        norm = norm + amp;
        amp = amp * gain;
        freq = freq * lacunarity;
    }
    return total / max(norm, 1e-6);
}

// patterns.turbulence: sum of |signed noise|, the cusps are the point
float hal_pt_turb(vec3 p, int octaves, float lacunarity, float gain)
{
    float total = 0.0;
    float amp = 1.0;
    float norm = 0.0;
    float freq = 1.0;
    for (int i = 0; i < octaves; i++) {
        total = total + abs(hal_pt_vnoise(p * freq) * 2.0 - 1.0) * amp;
        norm = norm + amp;
        amp = amp * gain;
        freq = freq * lacunarity;
    }
    return total / max(norm, 1e-6);
}

// patterns.worley: F1 and F2 over the 27 neighbouring cells
vec2 hal_pt_worley(vec3 p, float jitter)
{
    vec3 cell = floor(p);
    float f1 = 1e9;
    float f2 = 1e9;
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dz = -1; dz <= 1; dz++) {
                vec3 c = cell + vec3(float(dx), float(dy), float(dz));
                float jx = hal_pt_hash3f(c, 0.0);
                float jy = hal_pt_hash3f(c, 31.0);
                float jz = hal_pt_hash3f(c, 71.0);
                vec3 pt = c + vec3(jx, jy, jz) * jitter
                          + (0.5 * (1.0 - jitter));
                float d = length(p - pt);
                if (d < f1) { f2 = f1; f1 = d; }
                else { f2 = min(f2, d); }
            }
        }
    }
    return vec2(f1, f2);
}

// patterns.worley with the winning cell's id -- the update order mirrors
// the CPU exactly: `closer` is decided against the OLD F1, the id follows
// that same decision, then F1 takes the min. Left separate from the vec2
// version so dents/crackle keep their proven source byte for byte.
vec3 hal_pt_worley3(vec3 p, float jitter)
{
    vec3 cell = floor(p);
    float f1 = 1e9;
    float f2 = 1e9;
    float cid = 0.0;
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dz = -1; dz <= 1; dz++) {
                vec3 c = cell + vec3(float(dx), float(dy), float(dz));
                float jx = hal_pt_hash3f(c, 0.0);
                float jy = hal_pt_hash3f(c, 31.0);
                float jz = hal_pt_hash3f(c, 71.0);
                vec3 pt = c + vec3(jx, jy, jz) * jitter
                          + (0.5 * (1.0 - jitter));
                float d = length(p - pt);
                bool closer = d < f1;
                if (closer) { f2 = f1; }
                else { f2 = min(f2, d); }
                if (closer) { cid = hal_pt_hash3f(c, 137.0); }
                f1 = min(f1, d);
            }
        }
    }
    return vec3(f1, f2, cid);
}

// patterns.ridged: fold, square, sum at decaying amplitude
float hal_pt_ridged(vec3 p, int octaves, float lacunarity, float gain)
{
    float total = 0.0;
    float amp = 1.0;
    float norm = 0.0;
    float freq = 1.0;
    for (int i = 0; i < octaves; i++) {
        float s = 1.0 - abs(hal_pt_vnoise(p * freq) * 2.0 - 1.0);
        total = total + (s * s) * amp;
        norm = norm + amp;
        amp = amp * gain;
        freq = freq * lacunarity;
    }
    return total / max(norm, 1e-6);
}
"""

#: pattern name -> which primitives its GLSL needs, so the assembler can
#: include only what is used (worley's 27-cell loop is not free to compile)
PATTERN_GLSL = {
    # patterns.marble: sine banding displaced by turbulence
    'marble': """
float hal_pat_marble(vec3 p, float turb, int octaves, float veins,
                     float sharpness, int axis)
{
    float t = hal_pt_turb(p, octaves, 2.0, 0.5) * turb;
    float coord = (axis == 0) ? p.x : ((axis == 1) ? p.y : p.z);
    float v = sin((coord + t * 4.0) * 3.14159265358979 * max(veins, 1e-3));
    v = v * 0.5 + 0.5;
    return pow(clamp(v, 0.0, 1.0), max(sharpness, 0.01));
}
""",
    # patterns.wood: concentric rings + fine cross grain
    'wood': """
float hal_pat_wood(vec3 p, float rings, float turb, int octaves,
                   float grain, int axis)
{
    float pa = (axis == 0) ? p.y : p.x;
    float pb = (axis == 2) ? p.y : p.z;
    float pax = (axis == 0) ? p.x : ((axis == 1) ? p.y : p.z);
    float r = sqrt(pa * pa + pb * pb);
    r = r + hal_pt_turb(p, octaves, 2.0, 0.5) * turb;
    float v = sin(r * max(rings, 1e-3) * 6.28318530717959) * 0.5 + 0.5;
    if (grain > 0.0) {
        float fine = hal_pt_vnoise(vec3(pa * 48.0, pb * 48.0, pax * 3.0));
        v = clamp(v + (fine - 0.5) * grain, 0.0, 1.0);
    }
    return v;
}
""",
    # patterns.granite: stacked noise, contrast-stretched
    'granite': """
float hal_pat_granite(vec3 p, int octaves, float contrast, float speckle)
{
    float v = hal_pt_fbm(p, octaves, 2.4, 0.62);
    if (speckle > 0.0) {
        v = v + (hal_pt_vnoise(p * 9.0) - 0.5) * speckle;
    }
    return clamp((v - 0.5) * max(contrast, 0.01) + 0.5, 0.0, 1.0);
}
""",
    # patterns.dents: sparse worley pits
    'dents': """
float hal_pat_dents(vec3 p, float size, int octaves, float depth)
{
    vec2 w = hal_pt_worley(p / max(size, 1e-3), 1.0);
    float v = clamp(1.0 - w.x, 0.0, 1.0);
    v = pow(v, max(1.0 / max(depth, 0.01), 0.01));
    if (octaves > 1) {
        v = v * 0.7 + hal_pt_turb(p, octaves, 2.0, 0.5) * 0.3;
    }
    return clamp(v, 0.0, 1.0);
}
""",
    # patterns.crackle: the boundary network between cells
    'crackle': """
float hal_pat_crackle(vec3 p, float jitter, float width, float smoothw)
{
    vec2 w = hal_pt_worley(p, jitter);
    float edge = w.y - w.x;
    float ww = max(width, 1e-4);
    float v = 1.0 - clamp((edge - ww) / max(smoothw, 1e-4), 0.0, 1.0);
    return clamp(v, 0.0, 1.0);
}
""",
    # patterns.plasma: interfering sine fields, the demoscene one
    'plasma': """
float hal_pat_plasma(vec3 p, float ptime, float complexity)
{
    float x = p.x;
    float y = p.y;
    float c = max(complexity, 0.1);
    float v = sin(x * c + ptime);
    v = v + sin((y * c + ptime) * 0.7);
    v = v + sin((x + y) * c * 0.5 + ptime * 1.3);
    v = v + sin(sqrt(x * x + y * y) * c * 1.1 + ptime * 0.8);
    return (v * 0.25) * 0.5 + 0.5;
}
""",
    # patterns.starfield: points on a grid with hashed brightness
    'starfield': """
float hal_pat_starfield(vec3 p, float density, float size, float twinkle,
                        float stime)
{
    vec3 cell = floor(p);
    float h = hal_pt_hash3f(p, 0.0);
    float thresh = 1.0 - clamp(density, 0.0, 1.0) * 0.25;
    vec3 fr = p - cell;
    vec3 centre = vec3(hal_pt_hash3f(p, 11.0), hal_pt_hash3f(p, 23.0),
                       hal_pt_hash3f(p, 47.0));
    float d = length(fr - centre);
    float radius = max(size, 1e-3) * 0.5;
    float disc = clamp(1.0 - d / radius, 0.0, 1.0);
    float mag = (h > thresh) ? (h - thresh) / max(1.0 - thresh, 1e-4) : 0.0;
    if (twinkle > 0.0) {
        float phase = hal_pt_hash3f(p, 91.0) * 6.28318530717959;
        mag = mag * (1.0 - twinkle * 0.5
                     * (1.0 + sin(stime * 3.0 + phase)) * 0.5);
    }
    return clamp(disc * mag, 0.0, 1.0);
}
""",
    # patterns.weave: over-under fabric; returns (value, is_warp)
    'weave': """
vec2 hal_pat_weave(vec3 p, float thickness, float gap, float warp)
{
    float x = p.x;
    float y = p.y;
    if (warp > 0.0) {
        x = x + (hal_pt_vnoise(p * 3.0) * 2.0 - 1.0) * warp;
        y = y + (hal_pt_vnoise(p * 3.0 + 17.0) * 2.0 - 1.0) * warp;
    }
    float fx = x - floor(x);
    float fy = y - floor(y);
    bool over = mod(floor(x) + floor(y), 2.0) == 0.0;
    float t = clamp(thickness, 0.01, 0.99);
    bool band_x = abs(fx - 0.5) < t * 0.5;
    bool band_y = abs(fy - 0.5) < t * 0.5;
    float shade_x = cos((fx - 0.5) / max(t, 1e-3) * 3.14159265358979)
                    * 0.5 + 0.5;
    float shade_y = cos((fy - 0.5) / max(t, 1e-3) * 3.14159265358979)
                    * 0.5 + 0.5;
    bool on_warp = over ? band_x : band_y;
    float val = on_warp ? shade_x : shade_y;
    if (!(band_x || band_y)) { val = 0.0; }
    if (gap > 0.0) {
        float edge = min(abs(fx - 0.5), abs(fy - 0.5));
        val = val * clamp(edge / max(gap, 1e-4), 0.0, 1.0);
    }
    return vec2(clamp(val, 0.0, 1.0), on_warp ? 1.0 : 0.0);
}
""",
    # patterns.tiles: (value, tile id, inside)
    'tiles': """
vec3 hal_pat_tiles(vec3 p, float rows, float columns, float grout,
                   float offset, float bevel)
{
    float y = p.y * max(rows, 1e-3);
    float row = floor(y);
    float x = p.x * max(columns, 1e-3) + row * offset;
    float fx = x - floor(x);
    float fy = y - row;
    float g = max(grout, 0.0) * 0.5;
    bool inside = (fx > g) && (fx < 1.0 - g) && (fy > g) && (fy < 1.0 - g);
    float edge = min(min(fx - g, 1.0 - g - fx), min(fy - g, 1.0 - g - fy));
    float shade = clamp(edge / max(bevel, 1e-4), 0.0, 1.0);
    float val = inside ? (0.35 + 0.65 * shade) : 0.0;
    float tid = hal_pt_hash3(int(floor(x)), int(row), 0);
    return vec3(val, tid, inside ? 1.0 : 0.0);
}
""",
    # patterns.spiral: Archimedean banding around an axis
    'spiral': """
float hal_pat_spiral(vec3 p, float turns, float sharpness, int axis,
                     float twist)
{
    float pa = (axis == 0) ? p.y : p.x;
    float pb = (axis == 2) ? p.y : p.z;
    float pax = (axis == 0) ? p.x : ((axis == 1) ? p.y : p.z);
    float ang = atan(pb, pa);
    float r = sqrt(pa * pa + pb * pb);
    float v = sin(ang * max(turns, 1e-3) + r * 6.28318530717959
                  + pax * twist);
    v = v * 0.5 + 0.5;
    return pow(clamp(v, 0.0, 1.0), max(sharpness, 0.01));
}
""",
    # patterns.bozo: value noise, optionally turbulence-displaced
    'bozo': """
float hal_pat_bozo(vec3 p, float turb, int octaves, float lacunarity)
{
    vec3 q = p;
    if (turb > 1e-6) {
        vec3 d = vec3(hal_pt_vnoise(p + 11.3) * 2.0 - 1.0,
                      hal_pt_vnoise(p + 47.1) * 2.0 - 1.0,
                      hal_pt_vnoise(p + 83.7) * 2.0 - 1.0);
        float f = 1.0;
        for (int i = 1; i < octaves; i++) {
            f = f * lacunarity;
            d = d + vec3(hal_pt_vnoise((p + 11.3) * f) * 2.0 - 1.0,
                         hal_pt_vnoise((p + 47.1) * f) * 2.0 - 1.0,
                         hal_pt_vnoise((p + 83.7) * f) * 2.0 - 1.0) / f;
        }
        q = p + d * turb;
    }
    return clamp(hal_pt_vnoise(q), 0.0, 1.0);
}
""",
    # patterns.agate: POV's sine band thrown about by turbulence, pow 0.77
    'agate': """
float hal_pat_agate(vec3 p, float turb, int octaves, float bands,
                    float sharpness, int axis)
{
    float pax = (axis == 0) ? p.x : ((axis == 1) ? p.y : p.z);
    float t = hal_pt_turb(p, octaves, 2.0, 0.5) * 2.0 - 1.0;
    float v = 0.5 * (sin(1.3 * t * max(turb, 0.0)
                         + max(bands, 1e-3) * pax) + 1.0);
    return pow(clamp(v, 0.0, 1.0), max(sharpness, 0.01));
}
""",
    # patterns.leopard: three interfering sines squared
    'leopard': """
float hal_pat_leopard(vec3 p, float spot)
{
    float s = (sin(p.x) + sin(p.y) + sin(p.z)) / 3.0;
    return pow(clamp(s * s, 0.0, 1.0), max(spot, 0.01));
}
""",
    # patterns.onion: concentric spherical shells
    'onion': """
float hal_pat_onion(vec3 p, float thickness, float sharpness)
{
    float r = length(p) / max(thickness, 1e-4);
    float v = r - floor(r);
    return pow(clamp(v, 0.0, 1.0), max(sharpness, 0.01));
}
""",
    # patterns.bumps: smooth noise as a height field
    'bumps': """
float hal_pat_bumps(vec3 p, float roundness, int octaves, float lacunarity,
                    float gain)
{
    float v = hal_pt_fbm(p, octaves, lacunarity, gain);
    v = clamp(v, 0.0, 1.0);
    v = v * v * (3.0 - 2.0 * v);
    return pow(v, max(roundness, 0.01));
}
""",
    # patterns.wrinkles: folded noise at halving amplitude, lifted 1.4x
    'wrinkles': """
float hal_pat_wrinkles(vec3 p, int octaves, float lacunarity, float crease)
{
    float v = clamp(1.4 * hal_pt_turb(p, octaves, lacunarity, 0.5),
                    0.0, 1.0);
    return pow(v, max(crease, 0.01));
}
""",
    # patterns.noise_fractal: the raw field, three profiles by kind
    'noise': """
float hal_pat_noise(vec3 p, int kind, int octaves, float lacunarity,
                    float gain)
{
    if (kind == 1) { return hal_pt_turb(p, octaves, lacunarity, gain); }
    if (kind == 2) { return hal_pt_ridged(p, octaves, lacunarity, gain); }
    return hal_pt_fbm(p, octaves, lacunarity, gain);
}
""",
    # patterns.noise_fractal over the 1/2/4-D lattices: the same three
    # profiles rebuilt from the dimension's own value noise, exactly the
    # CPU's dispatcher branch
    'noise_nd': """
float hal_pat_noise_nd(vec4 p, int dims, int kind, int octaves,
                       float lacunarity, float gain)
{
    float total = 0.0;
    float amp = 1.0;
    float norm = 0.0;
    float freq = 1.0;
    for (int i = 0; i < octaves; i++) {
        float v;
        if (dims == 1) { v = hal_pt_vnoise1(p.x * freq); }
        else if (dims == 2) { v = hal_pt_vnoise2(p.xy * freq); }
        else { v = hal_pt_vnoise4(p * freq); }
        if (kind == 1) { v = abs(v * 2.0 - 1.0); }
        else if (kind == 2) {
            float s = 1.0 - abs(v * 2.0 - 1.0);
            v = s * s;
        }
        total = total + v * amp;
        norm = norm + amp;
        amp = amp * gain;
        freq = freq * lacunarity;
    }
    return total / max(norm, 1e-6);
}
""",
    # patterns.water: layered directional 4D noise; the per-layer drift
    # directions arrive baked as literals from the CPU's own WATER_DIRS
    # table, and the loop moves all of time onto a (z, w) circle
    'water': """
float hal_pat_water_layer(vec2 xy, float ca, float sa, float freq,
                          float rate, float salt, float t, float speed,
                          float chop, float loopz, float loopw,
                          int looping)
{
    float xx = (xy.x * ca - xy.y * sa) * freq;
    float yy = (xy.x * sa + xy.y * ca) * freq;
    vec4 p4;
    if (looping == 1) {
        p4 = vec4(xx, yy, loopz * rate + salt, loopw * rate + salt);
    } else {
        float zt = t * speed * rate + salt;
        p4 = vec4(xx + zt * 0.35, yy, zt, salt);
    }
    float v = hal_pt_vnoise4(p4);
    float fold = 1.0 - abs(v * 2.0 - 1.0);
    return v + (fold - v) * clamp(chop, 0.0, 1.0);
}
""",
    # the shaped gradient: centre, rotation, eight shapes, repeat, easing
    'gradientshape': """
float hal_pat_gradient(vec3 q, int shape, int rep, int ease)
{
    float x = q.x;
    float y = q.y;
    float z = q.z;
    float f;
    if (shape == 1) { f = 1.0 - abs(x); }
    else if (shape == 2) { f = 1.0 - sqrt(x * x + y * y + z * z); }
    else if (shape == 3) {
        float r = max(1.0 - sqrt(x * x + y * y + z * z), 0.0);
        f = r * r;
    }
    else if (shape == 4) { f = 1.0 - max(abs(x), abs(y)); }
    else if (shape == 5) { f = 1.0 - (abs(x) + abs(y)); }
    else if (shape == 6) { f = atan(y, x) / 6.28318530717959 + 0.5; }
    else if (shape == 7) {
        f = atan(y, x) / 6.28318530717959 + 0.5 + sqrt(x * x + y * y);
        f = f - floor(f);
    }
    else { f = x + 0.5; }
    if (rep == 1) { f = f - floor(f); }
    else if (rep == 2) {
        float h = (f * 0.5 - floor(f * 0.5)) * 2.0;
        f = 1.0 - abs(h - 1.0);
    }
    f = clamp(f, 0.0, 1.0);
    if (ease == 1) { f = f * f * (3.0 - 2.0 * f); }
    else if (ease == 2) { f = f * f; }
    return f;
}
""",
    # patterns.cells: Worley by feature; returns (fac, cell id)
    'cells': """
vec2 hal_pat_cells(vec3 p, float jitter, int feature)
{
    vec3 w = hal_pt_worley3(p, jitter);
    float v = w.x;
    if (feature == 1) { v = w.y; }
    if (feature == 2) { v = w.y - w.x; }
    if (feature == 3) { v = w.z; }
    return vec2(clamp(v, 0.0, 1.0), w.z);
}
""",
    # patterns.tv_static: per-cell hash, frame-salted
    'static': """
float hal_pat_static(vec3 p, float frame)
{
    vec3 c = floor(p);
    return hal_pt_hash3(int(c.x), int(c.y), int(c.z) + int(frame) * 7919);
}
""",
    # patterns.brick: running bond; (bevel ramp, brick id, inside)
    'brick': """
vec3 hal_pat_brick(vec3 p, float width, float height, float mortar,
                   float offset, float bevel)
{
    float w = max(width, 1e-3);
    float h = max(height, 1e-3);
    float row = floor(p.z / h);
    float shift = row * offset * w;
    float u = (p.x + shift) / w;
    float col = floor(u);
    float fu = u - col;
    float fv = p.z / h - row;
    float m = clamp(mortar, 0.0, 0.49);
    bool inside = (fu > m) && (fu < 1.0 - m) && (fv > m) && (fv < 1.0 - m);
    float b = max(bevel, 1e-4);
    float du = min(fu - m, (1.0 - m) - fu) / b;
    float dv = min(fv - m, (1.0 - m) - fv) / b;
    float ramp = clamp(min(du, dv), 0.0, 1.0);
    float fac = inside ? ramp : 0.0;
    float bid = hal_pt_hash3(int(col), int(row), 0);
    return vec3(fac, bid, inside ? 1.0 : 0.0);
}
""",
}

#: which patterns need the worley primitive (the others skip its loops)
NEEDS_WORLEY = frozenset({'dents', 'crackle'})


#: the colour-space ramp's conversion helpers: OKLab (Ottosson 2020),
#: OKLCh's short-way hue walk, HSV -- mirroring nodeeval's NumPy originals
#: constant for constant
OKRAMP_GLSL = """
vec3 hal_rgb_to_oklab(vec3 c)
{
    float l = 0.4122214708 * c.r + 0.5363325363 * c.g + 0.0514459929 * c.b;
    float m = 0.2119034982 * c.r + 0.6806995451 * c.g + 0.1073969566 * c.b;
    float s = 0.0883024619 * c.r + 0.2817188376 * c.g + 0.6299787005 * c.b;
    l = pow(max(l, 0.0), 1.0 / 3.0);
    m = pow(max(m, 0.0), 1.0 / 3.0);
    s = pow(max(s, 0.0), 1.0 / 3.0);
    return vec3(0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
                1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
                0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s);
}

vec3 hal_oklab_to_rgb(vec3 lab)
{
    float l = lab.x + 0.3963377774 * lab.y + 0.2158037573 * lab.z;
    float m = lab.x - 0.1055613458 * lab.y - 0.0638541728 * lab.z;
    float s = lab.x - 0.0894841775 * lab.y - 1.2914855480 * lab.z;
    l = l * l * l;
    m = m * m * m;
    s = s * s * s;
    return vec3(+4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
                -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
                -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s);
}

vec3 hal_rgb_to_hsv(vec3 c)
{
    float mx = max(c.r, max(c.g, c.b));
    float mn = min(c.r, min(c.g, c.b));
    float d = mx - mn;
    float h = 0.0;
    if (d > 1e-12) {
        if (mx == c.r) { h = (mx - c.b) / d - (mx - c.g) / d; }
        else if (mx == c.g) { h = 2.0 + (mx - c.r) / d - (mx - c.b) / d; }
        else { h = 4.0 + (mx - c.g) / d - (mx - c.r) / d; }
    }
    h = fract(h / 6.0);
    float s = (mx > 1e-12) ? d / mx : 0.0;
    return vec3(h, s, mx);
}

vec3 hal_hsv_to_rgb(vec3 hsv)
{
    float h = fract(hsv.x) * 6.0;
    float i = floor(h);
    float f = h - i;
    float p = hsv.z * (1.0 - hsv.y);
    float q = hsv.z * (1.0 - hsv.y * f);
    float t = hsv.z * (1.0 - hsv.y * (1.0 - f));
    int ii = int(i);
    if (ii == 0) { return vec3(hsv.z, t, p); }
    if (ii == 1) { return vec3(q, hsv.z, p); }
    if (ii == 2) { return vec3(p, hsv.z, t); }
    if (ii == 3) { return vec3(p, q, hsv.z); }
    if (ii == 4) { return vec3(t, p, hsv.z); }
    return vec3(hsv.z, p, q);
}

vec3 hal_ramp_blend(vec3 a, vec3 b, float t, int space)
{
    if (space == 1) {
        vec3 la = hal_rgb_to_oklab(a);
        vec3 lb = hal_rgb_to_oklab(b);
        return hal_oklab_to_rgb(la + (lb - la) * t);
    }
    if (space == 2) {
        vec3 la = hal_rgb_to_oklab(a);
        vec3 lb = hal_rgb_to_oklab(b);
        float ca = sqrt(la.y * la.y + la.z * la.z);
        float cb = sqrt(lb.y * lb.y + lb.z * lb.z);
        float ha = atan(la.z, la.y);
        float hb = atan(lb.z, lb.y);
        float dh = hb - ha;
        dh = dh - floor(dh / 6.28318530717959 + 0.5) * 6.28318530717959;
        float L = la.x + (lb.x - la.x) * t;
        float C = ca + (cb - ca) * t;
        float H = ha + dh * t;
        return hal_oklab_to_rgb(vec3(L, C * cos(H), C * sin(H)));
    }
    if (space == 3) {
        vec3 ha = hal_rgb_to_hsv(a);
        vec3 hb = hal_rgb_to_hsv(b);
        float dh = hb.x - ha.x;
        dh = dh - floor(dh + 0.5);
        return hal_hsv_to_rgb(vec3(fract(ha.x + dh * t),
                                   ha.y + (hb.y - ha.y) * t,
                                   ha.z + (hb.z - ha.z) * t));
    }
    return a + (b - a) * t;
}
"""
