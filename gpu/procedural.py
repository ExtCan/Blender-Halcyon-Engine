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


# ------------------------------------------------------- Blender Internal
# The BI texture engine's GPU twin. The GLSL is generated at import time
# from core/bitex_tables.py, so the CPU and GPU read the SAME tables from
# the same module -- they cannot drift apart. Algorithms match
# core/bitex.py line for line; see that module for provenance.


def _bitex_glsl():
    # The tables travel as a DATA TEXTURE (hal_bitex_tab, bound once per
    # program), not as inline const arrays: 2048 constants compiled into
    # every material shader is exactly the kind of thing that grinds or
    # crashes real drivers, and texelFetch of engine-made data textures
    # is the proven R115 pattern. Layout: red channel of a 2048x1 image,
    # hash at [0..511], hashvectf at [512..1279], hashpntf at [1280..2047]
    # -- packed by core/bitex_tables.table_pixels(), the same module the
    # CPU reads, so the two devices cannot drift.
    return """
uniform sampler2D hal_bitex_tab;

// forward declarations: bi_bricontrgb calls the okramp chunk's HSV
// pair, and the okramp chunk lands AFTER this one in the assembled
// source. A real GLSL compiler requires declaration before use -- the
// field driver rejected every material carrying this chunk ('the
// driver rejected <material>: HAL_MAT_1: CreateInfo failed') while the
// name-resolving front-end and simulator compiled it happily. The
// prototypes make this chunk correct under ANY chunk order
vec3 hal_rgb_to_hsv(vec3 c);
vec3 hal_hsv_to_rgb(vec3 hsv);

// clamp: texelFetch outside the texture is UNDEFINED on Vulkan -- on a
// real driver that is a device fault, and a lost device kills Blender
// with no crash log at all. Every table read funnels through here; an
// index bug upstream now costs a wrong texel, never the GPU
float bi_tab(int i) { return texelFetch(hal_bitex_tab,
                                        ivec2(clamp(i, 0, 2047), 0), 0).r; }
int bi_hashi(int i) { return int(bi_tab(i) + 0.5); }
vec3 bi_hvecv(int h)
{
    int b = 512 + 3 * h;
    return vec3(bi_tab(b), bi_tab(b + 1), bi_tab(b + 2));
}
vec3 bi_hpntv(int h)
{
    int b = 1280 + 3 * h;
    return vec3(bi_tab(b), bi_tab(b + 1), bi_tab(b + 2));
}

float bi_onoise(vec3 p)
{
    vec3 fp = floor(p);
    vec3 o = p - fp;
    vec3 j = o - 1.0;
    ivec3 ip = ivec3(fp);
    vec3 cn_o = 1.0 - 3.0 * o * o + 2.0 * o * o * o;
    vec3 cn_j = 1.0 - 3.0 * j * j - 2.0 * j * j * j;
    int b00 = bi_hashi(bi_hashi(ip.x & 255) + (ip.y & 255));
    int b10 = bi_hashi(bi_hashi((ip.x + 1) & 255) + (ip.y & 255));
    int b01 = bi_hashi(bi_hashi(ip.x & 255) + ((ip.y + 1) & 255));
    int b11 = bi_hashi(bi_hashi((ip.x + 1) & 255) + ((ip.y + 1) & 255));
    int b20 = ip.z & 255, b21 = (ip.z + 1) & 255;
    float n = 0.5;
    vec3 h;
    h = bi_hvecv(bi_hashi(b20 + b00));
    n += (cn_o.x * cn_o.y * cn_o.z) * (h.x * o.x + h.y * o.y + h.z * o.z);
    h = bi_hvecv(bi_hashi(b21 + b00));
    n += (cn_o.x * cn_o.y * cn_j.z) * (h.x * o.x + h.y * o.y + h.z * j.z);
    h = bi_hvecv(bi_hashi(b20 + b01));
    n += (cn_o.x * cn_j.y * cn_o.z) * (h.x * o.x + h.y * j.y + h.z * o.z);
    h = bi_hvecv(bi_hashi(b21 + b01));
    n += (cn_o.x * cn_j.y * cn_j.z) * (h.x * o.x + h.y * j.y + h.z * j.z);
    h = bi_hvecv(bi_hashi(b20 + b10));
    n += (cn_j.x * cn_o.y * cn_o.z) * (h.x * j.x + h.y * o.y + h.z * o.z);
    h = bi_hvecv(bi_hashi(b21 + b10));
    n += (cn_j.x * cn_o.y * cn_j.z) * (h.x * j.x + h.y * o.y + h.z * j.z);
    h = bi_hvecv(bi_hashi(b20 + b11));
    n += (cn_j.x * cn_j.y * cn_o.z) * (h.x * j.x + h.y * j.y + h.z * o.z);
    h = bi_hvecv(bi_hashi(b21 + b11));
    n += (cn_j.x * cn_j.y * cn_j.z) * (h.x * j.x + h.y * j.y + h.z * j.z);
    return clamp(n, 0.0, 1.0);
}

float bi_operlin(vec3 p)
{
    // orgPerlinNoise(): Perlin's ORIGINAL 1985 noise, Blender's exact
    // form -- +10000 shift, hashvectf gradients, s-curve fades, the
    // 1.5 scale. The CPU twin is bitex.org_perlin_noise
    vec3 t = p + 10000.0;
    ivec3 b0 = ivec3(t) & 255;
    ivec3 b1 = (b0 + 1) & 255;
    vec3 r0 = t - floor(t);
    vec3 r1 = r0 - 1.0;
    int i = bi_hashi(b0.x);
    int j = bi_hashi(b1.x);
    int b00 = bi_hashi(i + b0.y);
    int b10 = bi_hashi(j + b0.y);
    int b01 = bi_hashi(i + b1.y);
    int b11 = bi_hashi(j + b1.y);
    float sx = r0.x * r0.x * (3.0 - 2.0 * r0.x);
    float sy = r0.y * r0.y * (3.0 - 2.0 * r0.y);
    float sz = r0.z * r0.z * (3.0 - 2.0 * r0.z);
    vec3 h;
    float u, v, a, b, c, d;
    h = bi_hvecv(bi_hashi(b00 + b0.z));
    u = r0.x * h.x + r0.y * h.y + r0.z * h.z;
    h = bi_hvecv(bi_hashi(b10 + b0.z));
    v = r1.x * h.x + r0.y * h.y + r0.z * h.z;
    a = u + sx * (v - u);
    h = bi_hvecv(bi_hashi(b01 + b0.z));
    u = r0.x * h.x + r1.y * h.y + r0.z * h.z;
    h = bi_hvecv(bi_hashi(b11 + b0.z));
    v = r1.x * h.x + r1.y * h.y + r0.z * h.z;
    b = u + sx * (v - u);
    c = a + sy * (b - a);
    h = bi_hvecv(bi_hashi(b00 + b1.z));
    u = r0.x * h.x + r0.y * h.y + r1.z * h.z;
    h = bi_hvecv(bi_hashi(b10 + b1.z));
    v = r1.x * h.x + r0.y * h.y + r1.z * h.z;
    a = u + sx * (v - u);
    h = bi_hvecv(bi_hashi(b01 + b1.z));
    u = r0.x * h.x + r1.y * h.y + r1.z * h.z;
    h = bi_hvecv(bi_hashi(b11 + b1.z));
    v = r1.x * h.x + r1.y * h.y + r1.z * h.z;
    b = u + sx * (v - u);
    d = a + sy * (b - a);
    return 1.5 * (c + sz * (d - c));
}

float bi_fade(float t) { return t * t * t * (t * (t * 6.0 - 15.0) + 10.0); }

float bi_grad(int h, float x, float y, float z)
{
    h = h & 15;
    float u = h < 8 ? x : y;
    float v = h < 4 ? y : ((h == 12 || h == 14) ? x : z);
    return (((h & 1) == 0) ? u : -u) + (((h & 2) == 0) ? v : -v);
}

float bi_nperlin(vec3 p)
{
    vec3 fp = floor(p);
    ivec3 I = ivec3(fp) & 255;
    vec3 r = p - fp;
    vec3 f = vec3(bi_fade(r.x), bi_fade(r.y), bi_fade(r.z));
    int A = bi_hashi(I.x) + I.y;
    int AA = bi_hashi(A) + I.z, AB = bi_hashi(A + 1) + I.z;
    int B = bi_hashi(I.x + 1) + I.y;
    int BA = bi_hashi(B) + I.z, BB = bi_hashi(B + 1) + I.z;
    return mix(mix(mix(bi_grad(bi_hashi(AA), r.x, r.y, r.z),
                       bi_grad(bi_hashi(BA), r.x - 1.0, r.y, r.z), f.x),
                   mix(bi_grad(bi_hashi(AB), r.x, r.y - 1.0, r.z),
                       bi_grad(bi_hashi(BB), r.x - 1.0, r.y - 1.0, r.z), f.x), f.y),
               mix(mix(bi_grad(bi_hashi(AA + 1), r.x, r.y, r.z - 1.0),
                       bi_grad(bi_hashi(BA + 1), r.x - 1.0, r.y, r.z - 1.0), f.x),
                   mix(bi_grad(bi_hashi(AB + 1), r.x, r.y - 1.0, r.z - 1.0),
                       bi_grad(bi_hashi(BB + 1), r.x - 1.0, r.y - 1.0, r.z - 1.0), f.x), f.y), f.z);
}

float bi_cell_u(vec3 p)
{
    p = (p + 0.000001) * 1.00001;
    ivec3 ip = ivec3(floor(p));
    uint n = uint(ip.x + ip.y * 1301 + ip.z * 314159);
    n = n ^ (n << 13u);
    uint v = n * (n * n * 15731u + 789221u) + 1376312589u;
    return float(v) / 4294967296.0;
}

vec3 bi_cell_v3(vec3 p)
{
    return vec3(bi_cell_u(p), bi_cell_u(p.yxz), bi_cell_u(p.yzx));
}

float bi_vdist(vec3 d, float e, int dtype)
{
    vec3 a = abs(d);
    if (dtype == 1) { return dot(d, d); }
    if (dtype == 2) { return a.x + a.y + a.z; }
    if (dtype == 3) { return max(a.x, max(a.y, a.z)); }
    if (dtype == 4) { float s = sqrt(a.x) + sqrt(a.y) + sqrt(a.z); return s * s; }
    if (dtype == 5) { vec3 q = d * d; return sqrt(sqrt(dot(q, q))); }
    if (dtype == 6) { e = max(e, 1e-6); return pow(pow(a.x, e) + pow(a.y, e) + pow(a.z, e), 1.0 / e); }
    return sqrt(dot(d, d));
}

void bi_voronoi(vec3 p, float me, int dtype, out vec4 da, out vec3 pa[4])
{
    ivec3 base = ivec3(floor(p));
    da = vec4(1e10);
    pa[0] = vec3(0.0); pa[1] = vec3(0.0); pa[2] = vec3(0.0); pa[3] = vec3(0.0);
    for (int xx = -1; xx <= 1; xx++)
    for (int yy = -1; yy <= 1; yy++)
    for (int zz = -1; zz <= 1; zz++) {
        ivec3 c = base + ivec3(xx, yy, zz);
        int hi = bi_hashi((bi_hashi((bi_hashi(c.z & 255) + c.y) & 255) + c.x) & 255);
        vec3 pt = bi_hpntv(hi) + vec3(c);
        float d = bi_vdist(p - pt, me, dtype);
        if (d < da.x) {
            da = vec4(d, da.xyz);
            pa[3] = pa[2]; pa[2] = pa[1]; pa[1] = pa[0]; pa[0] = pt;
        } else if (d < da.y) {
            da.yzw = vec3(d, da.yz);
            pa[3] = pa[2]; pa[2] = pa[1]; pa[1] = pt;
        } else if (d < da.z) {
            da.zw = vec2(d, da.z);
            pa[3] = pa[2]; pa[2] = pt;
        } else if (d < da.w) {
            da.w = d; pa[3] = pt;
        }
    }
}

float bi_basis_u(int nbas, vec3 p)
{
    if (nbas == 0) { return bi_onoise(p); }
    if (nbas == 1) { return 0.5 + 0.5 * bi_operlin(p); }
    if (nbas == 2) { return 0.5 + 0.5 * bi_nperlin(p); }
    if (nbas == 14) { return bi_cell_u(p); }
    vec4 da; vec3 pa[4];
    bi_voronoi(p, 2.5, 0, da, pa);
    if (nbas == 3) { return da.x; }
    if (nbas == 4) { return da.y; }
    if (nbas == 5) { return da.z; }
    if (nbas == 6) { return da.w; }
    if (nbas == 7) { return da.y - da.x; }
    if (nbas == 8) { return min(10.0 * (da.y - da.x), 1.0); }
    return bi_onoise(p);
}

float bi_basis_s(int nbas, vec3 p)
{
    if (nbas == 1) { return bi_operlin(p); }
    if (nbas == 2) { return bi_nperlin(p); }
    return 2.0 * bi_basis_u(nbas, p) - 1.0;
}

float bi_gnoise(float nsize, vec3 p, int hard, int nbas)
{
    if (nsize != 0.0) { p /= nsize; }
    float t = bi_basis_u(nbas, p);
    return (hard != 0) ? abs(2.0 * t - 1.0) : t;
}

float bi_gturb(float nsize, vec3 p, int oct, int hard, int nbas)
{
    if (nsize != 0.0) { p /= nsize; }
    float total = 0.0, amp = 1.0;
    for (int i = 0; i <= oct; i++) {
        float t = bi_basis_u(nbas, p);
        if (hard != 0) { t = abs(2.0 * t - 1.0); }
        total += t * amp;
        amp *= 0.5;
        p *= 2.0;
    }
    total *= float(1 << oct) / float((1 << (oct + 1)) - 1);
    return total;
}

float bi_mg_fbm(vec3 p, float H, float lac, float oct, int nbas)
{
    float pwHL = pow(lac, -H), pwr = 1.0, value = 0.0;
    int io = int(oct);
    for (int i = 0; i < io; i++) {
        value += bi_basis_s(nbas, p) * pwr;
        pwr *= pwHL;
        p *= lac;
    }
    float rmd = oct - floor(oct);
    if (rmd != 0.0) { value += rmd * bi_basis_s(nbas, p) * pwr; }
    return value;
}

float bi_mg_multifractal(vec3 p, float H, float lac, float oct, int nbas)
{
    float pwHL = pow(lac, -H), pwr = 1.0, value = 1.0;
    int io = int(oct);
    for (int i = 0; i < io; i++) {
        value *= bi_basis_s(nbas, p) * pwr + 1.0;
        pwr *= pwHL;
        p *= lac;
    }
    float rmd = oct - floor(oct);
    if (rmd != 0.0) { value *= rmd * bi_basis_s(nbas, p) * pwr + 1.0; }
    return value;
}

float bi_mg_hetero(vec3 p, float H, float lac, float oct, float ofs, int nbas)
{
    float pwHL = pow(lac, -H), pwr = pwHL;
    float value = ofs + bi_basis_s(nbas, p);
    p *= lac;
    int io = int(oct);
    for (int i = 1; i < io; i++) {
        value += (bi_basis_s(nbas, p) + ofs) * pwr * value;
        pwr *= pwHL;
        p *= lac;
    }
    float rmd = oct - floor(oct);
    if (rmd != 0.0) { value += rmd * (bi_basis_s(nbas, p) + ofs) * pwr * value; }
    return value;
}

float bi_mg_hybrid(vec3 p, float H, float lac, float oct, float ofs, float gain, int nbas)
{
    float pwHL = pow(lac, -H), pwr = pwHL;
    float result = bi_basis_s(nbas, p) + ofs;
    float weight = gain * result;
    p *= lac;
    int io = int(oct);
    for (int i = 1; i < io; i++) {
        weight = min(weight, 1.0);
        float signal = (bi_basis_s(nbas, p) + ofs) * pwr;
        pwr *= pwHL;
        result += weight * signal;
        weight *= gain * signal;
        p *= lac;
    }
    float rmd = oct - floor(oct);
    if (rmd != 0.0) { result += rmd * (bi_basis_s(nbas, p) + ofs) * pwr; }
    return result;
}

float bi_mg_ridged(vec3 p, float H, float lac, float oct, float ofs, float gain, int nbas)
{
    float pwHL = pow(lac, -H), pwr = pwHL;
    float signal = ofs - abs(bi_basis_s(nbas, p));
    signal *= signal;
    float result = signal;
    int io = int(oct);
    for (int i = 1; i < io; i++) {
        p *= lac;
        float weight = clamp(signal * gain, 0.0, 1.0);
        signal = ofs - abs(bi_basis_s(nbas, p));
        signal *= signal * weight;
        result += signal * pwr;
        pwr *= pwHL;
    }
    return result;
}

float bi_vlnoise(vec3 p, float dist, int b1, int b2)
{
    vec3 r = vec3(bi_basis_s(b1, p + 13.5) * dist,
                  bi_basis_s(b1, p) * dist,
                  bi_basis_s(b1, p - 13.5) * dist);
    return bi_basis_s(b2, p + r);
}

float bi_wave(int wf, float a)
{
    if (wf == 1) {
        float b = 6.2831853;
        a = mod(a, b);
        if (a < 0.0) { a += b; }
        return a / b;
    }
    if (wf == 2) {
        float b = 6.2831853;
        return 1.0 - 2.0 * abs(floor(a * (1.0 / b) + 0.5) - a * (1.0 / b));
    }
    return 0.5 + 0.5 * sin(a);
}

float bi_tex_clouds(vec3 p, float nsize, int depth, int hard, int nbas)
{
    return bi_gturb(nsize, p, depth, hard, nbas);
}

vec3 bi_tex_clouds_col(vec3 p, float nsize, int depth, int hard, int nbas)
{
    return vec3(bi_gturb(nsize, p, depth, hard, nbas),
                bi_gturb(nsize, p.yxz, depth, hard, nbas),
                bi_gturb(nsize, p.yzx, depth, hard, nbas));
}

float bi_tex_wood(vec3 p, int stype, int wf, float nsize, float turb, int hard, int nbas)
{
    if (stype == 0) { return bi_wave(wf, (p.x + p.y + p.z) * 10.0); }
    if (stype == 1) { return bi_wave(wf, length(p) * 20.0); }
    float wi = turb * bi_gnoise(nsize, p, hard, nbas);
    if (stype == 2) { return bi_wave(wf, (p.x + p.y + p.z) * 10.0 + wi); }
    return bi_wave(wf, length(p) * 20.0 + wi);
}

float bi_tex_marble(vec3 p, int stype, int wf, float nsize, float turb, int depth, int hard, int nbas)
{
    float n = 5.0 * (p.x + p.y + p.z);
    float mi = n + turb * bi_gturb(nsize, p, depth, hard, nbas);
    mi = bi_wave(wf, mi);
    if (stype == 1) { mi = sqrt(mi); }
    else if (stype == 2) { mi = sqrt(sqrt(mi)); }
    return mi;
}

vec4 bi_tex_magic(vec3 p, int depth, float turbul)
{
    float turb = turbul / 5.0;
    float x = sin((p.x + p.y + p.z) * 5.0);
    float y = cos((-p.x + p.y - p.z) * 5.0);
    float z = -cos((-p.x - p.y + p.z) * 5.0);
    if (depth > 0) {
        x *= turb; y *= turb; z *= turb;
        y = -cos(x - y + z) * turb;
        if (depth > 1) { x = cos(x - y - z) * turb;
        if (depth > 2) { z = sin(-x - y - z) * turb;
        if (depth > 3) { x = -cos(-x + y - z) * turb;
        if (depth > 4) { y = -sin(-x + y + z) * turb;
        if (depth > 5) { y = -cos(-x + y + z) * turb;
        if (depth > 6) { x = cos(x + y + z) * turb;
        if (depth > 7) { z = sin(x + y - z) * turb;
        if (depth > 8) { x = -cos(-x - y + z) * turb;
        if (depth > 9) { y = -sin(x - y + z) * turb; } } } } } } } } }
    }
    if (turb != 0.0) {
        turb *= 2.0;
        x /= turb; y /= turb; z /= turb;
    }
    vec3 rgb = vec3(0.5 - x, 0.5 - y, 0.5 - z);
    return vec4(rgb, (rgb.r + rgb.g + rgb.b) / 3.0);
}

float bi_tex_blend(vec3 p, int stype, int flip)
{
    float x = (flip != 0) ? p.y : p.x;
    float y = (flip != 0) ? p.x : p.y;
    if (stype == 0) { return (1.0 + x) / 2.0; }
    if (stype == 1) { float t = (1.0 + x) / 2.0; return t < 0.0 ? 0.0 : t * t; }
    if (stype == 2) {
        float t = clamp((1.0 + x) / 2.0, 0.0, 1.0);
        return 3.0 * t * t - 2.0 * t * t * t;
    }
    if (stype == 3) { return (2.0 + x + y) / 4.0; }
    if (stype == 6) { return atan(y, x) / 6.2831853 + 0.5; }
    float t = max(1.0 - sqrt(x * x + y * y + p.z * p.z), 0.0);
    return (stype == 5) ? t * t : t;
}

float bi_tex_stucci(vec3 p, int stype, float nsize, float turb, int hard, int nbas)
{
    float b2 = bi_gnoise(nsize, p, hard, nbas);
    float ofs = turb / 200.0;
    if (stype != 0) { ofs *= b2 * b2; }
    float tin = bi_gnoise(nsize, vec3(p.x, p.y, p.z + ofs), hard, nbas);
    if (stype == 2) { tin = 1.0 - tin; }
    return max(tin, 0.0);
}

float bi_tex_noise(vec3 p, int depth, float frame)
{
    ivec2 ip = ivec2(floor(p.xy * 10000.0));
    uint n = uint(ip.x + ip.y * 1301 + (int(frame) + 7) * 314159);
    n = n ^ (n << 13u);
    uint ran = n * (n * n * 15731u + 789221u) + 1376312589u;
    float div = 3.0;
    uint shift = 29u;
    float val = float((ran >> shift) & 3u);
    for (int i = 0; i < depth; i++) {
        shift -= 2u;
        val *= float((ran >> shift) & 3u);
        div *= 3.0;
    }
    return val / div;
}

float bi_tex_musgrave(vec3 p, int stype, float H, float lac, float oct,
                      float ofs, float gain, float outscale, int nbas)
{
    if (stype == 0) { return outscale * bi_mg_multifractal(p, H, lac, oct, nbas); }
    if (stype == 1) { return outscale * bi_mg_ridged(p, H, lac, oct, ofs, gain, nbas); }
    if (stype == 2) { return outscale * bi_mg_hybrid(p, H, lac, oct, ofs, gain, nbas); }
    if (stype == 4) { return outscale * bi_mg_hetero(p, H, lac, oct, ofs, nbas); }
    return outscale * bi_mg_fbm(p, H, lac, oct, nbas);
}

vec4 bi_tex_voronoi(vec3 p, float w1, float w2, float w3, float w4,
                    float mexp, int distm, float outscale, int coltype)
{
    float aw1 = abs(w1), aw2 = abs(w2), aw3 = abs(w3), aw4 = abs(w4);
    float sc = aw1 + aw2 + aw3 + aw4;
    if (sc != 0.0) { sc = outscale / sc; }
    vec4 da; vec3 pa[4];
    bi_voronoi(p, mexp, distm, da, pa);
    float tin = sc * abs(dot(vec4(w1, w2, w3, w4), da));
    if (coltype == 0) { return vec4(tin, tin, tin, tin); }
    vec3 col = aw1 * bi_cell_v3(pa[0]) + aw2 * bi_cell_v3(pa[1])
             + aw3 * bi_cell_v3(pa[2]) + aw4 * bi_cell_v3(pa[3]);
    if (coltype >= 2) {
        float t1 = min((da.y - da.x) * 10.0, 1.0);
        t1 *= (coltype == 3) ? tin : sc;
        col *= t1;
    } else {
        col *= sc;
    }
    return vec4(col, tin);
}

float bi_tex_distnoise(vec3 p, float dist, int b1, int b2)
{
    return bi_vlnoise(p, dist, b1, b2);
}

float bi_bricont(float tin, float bright, float contrast, int noclamp)
{
    tin = (tin - 0.5) * contrast + bright - 0.5;
    if (noclamp == 0) { tin = clamp(tin, 0.0, 1.0); }
    return tin;
}

vec3 bi_bricontrgb(vec3 rgb, float bright, float contrast, float sat,
                   vec3 fac, int noclamp)
{
    rgb = fac * ((rgb - 0.5) * contrast + bright - 0.5);
    if (noclamp == 0) { rgb = max(rgb, vec3(0.0)); }
    if (sat != 1.0) {
        vec3 hsv = hal_rgb_to_hsv(rgb);
        hsv.y *= sat;
        rgb = hal_hsv_to_rgb(hsv);
        if (sat > 1.0 && noclamp == 0) { rgb = max(rgb, vec3(0.0)); }
    }
    return rgb;
}

vec3 bi_classic_texvec(vec3 v, vec3 ofs, vec3 size, int classic)
{
    if (classic != 0) { v = v * 2.0 - 1.0; }
    return size * (v + ofs);
}
"""


PATTERN_GLSL['bitex'] = _bitex_glsl()
