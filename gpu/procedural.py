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

// patterns.value_noise2s: value_noise2 on the z = salt plane (R232, the
// 2D media): the salt rides the third lattice constant
float hal_pt_vnoise2s(vec2 p, int salt)
{
    vec2 fl = floor(p);
    vec2 f = p - fl;
    f = f * f * (3.0 - 2.0 * f);
    uint b = uint(int(fl.x)) * 374761393u + uint(int(fl.y)) * 668265263u
             + uint(salt) * 1274126177u;
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
    # patterns.leopard: jittered rosette cells -- dark broken rings with a
    # warm interior and the odd solid spot; returns (ring, interior)
    'leopard': """
vec2 hal_pat_leopard(vec3 p, float spot, float jitter, float breakup)
{
    float size = clamp(spot, 0.05, 1.6);
    float jit = clamp(jitter, 0.0, 1.0);
    float brk = clamp(breakup, 0.0, 1.0);
    float wx = hal_pt_vnoise(p * 2.3) - 0.5;
    float wy = hal_pt_vnoise(p * 2.3 + vec3(37.0)) - 0.5;
    float x = p.x + wx * 0.30;
    float y = p.y + wy * 0.30;
    float cx0 = floor(x);
    float cy0 = floor(y);
    float ring = 0.0;
    float inner = 0.0;
    float skip = 0.10 + 0.45 * brk;
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            float cxf = cx0 + float(dx);
            float cyf = cy0 + float(dy);
            int cx = int(cxf);
            int cy = int(cyf);
            float jx = hal_pt_hash3(cx, cy, 0);
            float jy = hal_pt_hash3(cx, cy, 31);
            float sz = hal_pt_hash3(cx, cy, 71);
            float rot = hal_pt_hash3(cx, cy, 137);
            float so = hal_pt_hash3(cx, cy, 197);
            float ox = cxf + 0.5 + (jx - 0.5) * (0.8 * jit);
            float oy = cyf + 0.5 + (jy - 0.5) * (0.8 * jit);
            float r0 = min(size * (0.30 + 0.13 * sz), 0.58);
            float ddx = x - ox;
            float ddy = y - oy;
            float d = sqrt(ddx * ddx + ddy * ddy);
            float theta = atan(ddy, ddx + 1e-12);
            float is_solid = 1.0 - step(0.22, so);
            float blobs = 0.0;
            for (int k = 0; k < 5; k++) {
                float hk = hal_pt_hash3(cx, cy, 211 + 13 * k);
                float h2 = hal_pt_hash3(cx, cy, 311 + 13 * k);
                float keep = step(skip, hk);
                float ang = 6.2831853 * (float(k) / 5.0)
                            + rot * 6.2831853 + (hk - 0.5) * 0.5;
                float dr = (d - r0) / (r0 * 0.30);
                float sarc = (mod(theta - ang + 3.14159265, 6.2831853)
                              - 3.14159265) / (0.55 + 0.35 * h2);
                float e = sqrt(dr * dr + sarc * sarc);
                float blob = (1.0 - smoothstep(0.72, 1.0, e)) * keep;
                blobs = max(blobs, blob);
            }
            float disc = 1.0 - smoothstep(r0 * 0.34, r0 * 0.42, d);
            float cell_ring = blobs + (disc - blobs) * is_solid;
            float cell_int = (1.0 - smoothstep(r0 * 0.68, r0 * 0.84, d))
                             * (1.0 - is_solid);
            ring = max(ring, cell_ring);
            inner = max(inner, cell_int);
        }
    }
    return vec2(ring, inner);
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
    # patterns.fur_tufts: round tapering cross-sections; (height, tuft id)
    'furtufts': """
vec2 hal_pat_fur_tufts(vec3 p, float coverage, float taper, float variation)
{
    float cx0 = floor(p.x);
    float cy0 = floor(p.y);
    float best = 0.0;
    float rnd = 0.0;
    float var = clamp(variation, 0.0, 1.0);
    float tap = max(taper, 1e-3);
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            float cxf = cx0 + float(dx);
            float cyf = cy0 + float(dy);
            int cx = int(cxf);
            int cy = int(cyf);
            float jx = hal_pt_hash3(cx, cy, 0);
            float jy = hal_pt_hash3(cx, cy, 31);
            float hv = hal_pt_hash3(cx, cy, 71);
            float id = hal_pt_hash3(cx, cy, 137);
            float ddx = (p.x - (cxf + 0.5 + (jx - 0.5) * 0.75)) / 0.62;
            float ddy = (p.y - (cyf + 0.5 + (jy - 0.5) * 0.75)) / 0.62;
            float q = max(1.0 - (ddx * ddx + ddy * ddy), 0.0);
            float prof = (tap == 1.0) ? q : pow(q, tap);
            float h = (id < coverage)
                      ? (1.0 - var + var * hv) * prof : 0.0;
            if (h > best) { best = h; rnd = id; }
        }
    }
    return vec2(best, rnd);
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


# --------------------------------------------------------- the 2D media (R232)
#
# core/media.py moved, not reinvented: hatching, scribble, stipple, charcoal,
# brush strokes, wash and paper, each the same operation order as its NumPy
# original. The per-node salt and the hash channel share the lattice's z
# axis in blocks of 64 (media.SALT_STRIDE); the baked rotations arrive as
# vec2 (cos, sin) literals from the emitter, exactly the float32 pairs the
# CPU multiplied with.

MEDIA_GLSL = {
    'md_prims': """
// media._n2 / media._h: value noise and a cell hash on the (salt, channel)
// plane
float hal_md_n2(vec2 p, int salt, int k)
{
    return hal_pt_vnoise2s(p, salt * 64 + k);
}

float hal_md_h(int ix, int iy, int salt, int k)
{
    return hal_pt_hash3(ix, iy, salt * 64 + k);
}

// media.view_coords: the eye-to-point direction unfolded octahedrally,
// L1-normalised (abs, add, divide: no sqrt), lower hemisphere folded
// into the corners -- the View space of the media
vec2 hal_md_view(vec3 d)
{
    float ax = abs(d.x);
    float ay = abs(d.y);
    float az = abs(d.z);
    float s = max(ax + ay + az, 1e-12);
    float u = d.x / s;
    float v = d.y / s;
    float au = abs(u);
    float av = abs(v);
    float fu = (1.0 - av) * sign(u);
    float fv = (1.0 - au) * sign(v);
    bool lower = d.z < 0.0;
    return vec2(lower ? fu : u, lower ? fv : v);
}

// media.tangent_2d (R235): the stroke direction of a FORM (mode 0, N x V,
// silhouette-parallel) or SLOPE (mode 1, V on the tangent plane) hatch,
// unit length or zero, in the matcap frame on the camera's paper (screen)
// or as world x, y
vec2 hal_md_tangent(vec3 n, vec3 v, int mode, int screen)
{
    vec3 t3 = (mode == 0) ? cross(n, v) : (v - n * dot(n, v));
    float tl = sqrt(dot(t3, t3));
    bool ok = tl > 1e-6;
    t3 = ok ? t3 / tl : vec3(0.0, 0.0, 0.0);
    if (screen == 1) {
        vec3 r0 = cross(vec3(0.0, 0.0, 1.0), v);
        float rl = dot(r0, r0);
        bool deg = rl < 1e-8;
        vec3 right = deg ? vec3(1.0, 0.0, 0.0) : r0 / sqrt(rl);
        vec3 upv = cross(v, right);
        return vec2(dot(t3, right), dot(t3, upv));
    }
    return vec2(t3.x, t3.y);
}

// media.direction_bins: the angle folded onto 180 degrees in units of 15,
// [0, 12); a zero direction reads bin 0
float hal_md_bins(float tx, float ty)
{
    float th = atan(ty, tx);
    float b = th * 3.8197186;
    b = b - floor(b / 12.0) * 12.0;
    b = min(max(b, 0.0), 11.999);
    return ((tx * tx + ty * ty) < 1e-8) ? 0.0 : b;
}

// media.salt_for: seed folded with the boil key, 16 bits. frame // boil
// is taken as floor((frame + 0.5) / boil): the half keeps the quotient at
// least 0.5/boil clear of every whole number, so a driver's 2.5-ulp float
// division (the GLSL spec's allowance) cannot land a key frame on the
// wrong side, and no integer division is asked of any front-end
int hal_md_salt(int seed, float frame, int boil)
{
    int key = (boil > 0)
              ? int(floor((float(max(int(frame), 0)) + 0.5) / float(boil)))
              : 0;
    return (seed * 7919 + key * 104729) & 0xffff;
}
""",
    # media.hatching: L layers of lanes, per-lane hashes, tonal fill
    'md_hatching': """
float hal_md_hatching(vec2 p, float d, int L, vec2 r0, vec2 r1, vec2 r2,
                      vec2 r3, float width, float length, float wobble,
                      float breaks, int salt)
{
    float remain = 1.0;
    float wid = clamp(width, 0.0, 1.0);
    float ln = max(length, 1e-3);
    float wob_amt = wobble * 0.6;
    float brk = breaks * 0.6;
    float fill = clamp((d - 0.8) / 0.2, 0.0, 1.0);
    for (int k = 0; k < L; k++) {
        vec2 r = (k == 0) ? r0 : ((k == 1) ? r1 : ((k == 2) ? r2 : r3));
        float c = r.x;
        float s = r.y;
        float u = p.x * c + p.y * s;
        float v = -p.x * s + p.y * c;
        float lane = floor(v);
        int li = int(lane);
        float h0 = hal_md_h(li, 0, salt, k * 4 + 0);
        float h1 = hal_md_h(li, 0, salt, k * 4 + 1);
        float h2 = hal_md_h(li, 0, salt, k * 4 + 2);
        float off = h0 * 512.0;
        float wob = (hal_md_n2(vec2(u * 0.35 + off, 0.5), salt, k) - 0.5)
                    * wob_amt;
        float a = (v - lane) - 0.5 + wob;
        float wk = clamp((d - float(k) / float(L)) * float(L), 0.0, 1.0);
        float hw = 0.5 * (wid * (0.35 + 0.65 * wk) * (0.8 + 0.4 * h1)
                          + (1.0 - wid) * fill);
        float t = u / ln + h2;
        float tl = t - floor(t);
        float ends = clamp(min(tl, 1.0 - tl) / 0.12, 0.0, 1.0);
        ends = ends + (1.0 - ends) * fill;
        float press = 0.75 + 0.25 * hal_md_n2(vec2(u * 1.7 + off, 2.5),
                                              salt, k);
        float gap = smoothstep(brk - 0.06, brk + 0.06,
                               hal_md_n2(vec2(u * 1.1 + off, 4.5), salt, k));
        gap = gap + (1.0 - gap) * fill;
        float line = clamp((hw * ends - abs(a)) / 0.08 + 0.5, 0.0, 1.0)
                     * press * gap * smoothstep(0.0, 0.15, wk);
        remain = remain * (1.0 - line);
    }
    return 1.0 - remain;
}
""",
    # media.scribble: warped lanes, graphite on the tooth
    'md_scribble': """
float hal_md_scribble(vec2 p, float d, int L, vec2 r0, vec2 r1, vec2 r2,
                      vec2 r3, float width, float curl, float pressure,
                      float grain, int salt, float blend)
{
    float bl = clamp(blend, 0.0, 1.0);
    float amp = curl * 2.5;
    float xw = p.x + ((hal_md_n2(vec2(p.x * 0.4, p.y * 0.4), salt, 40) - 0.5)
                      + (hal_md_n2(vec2(p.x * 0.9, p.y * 0.9), salt, 43)
                         - 0.5) * 0.4) * amp;
    float yw = p.y + ((hal_md_n2(vec2(p.x * 0.4, p.y * 0.4), salt, 41) - 0.5)
                      + (hal_md_n2(vec2(p.x * 0.9, p.y * 0.9), salt, 47)
                         - 0.5) * 0.4) * amp;
    float tooth = hal_md_n2(vec2(p.x * 18.0, p.y * 18.0), salt, 42);
    float fill = clamp((d - 0.8) / 0.2, 0.0, 1.0);
    float g = 0.12 + 0.5 * (1.0 - clamp(grain, 0.0, 1.0));
    float pd_base = pressure;
    float wid = width;
    float remain = 1.0;
    for (int k = 0; k < L; k++) {
        vec2 r = (k == 0) ? r0 : ((k == 1) ? r1 : ((k == 2) ? r2 : r3));
        float c = r.x;
        float s = r.y;
        float u = xw * c + yw * s;
        float v = -xw * s + yw * c;
        float wob = (hal_md_n2(vec2(u * 0.6, v * 0.6 + 7.3 * float(k)),
                               salt, 44 + k) - 0.5) * 0.5;
        float vv = v + wob;
        float a = (vv - floor(vv)) - 0.5;
        float wk = clamp((d - float(k) / float(L)) * float(L), 0.0, 1.0);
        float hw = 0.5 * (wid * (0.4 + 0.6 * wk) + (1.0 - wid) * fill);
        float press = (0.75 + 0.25 * hal_md_n2(vec2(u * 4.0, v * 0.9),
                                               salt, 48 + k)) * pd_base;
        float line = clamp((hw - abs(a)) / 0.1 + 0.5, 0.0, 1.0);
        float pd = press * (0.6 + 0.6 * d);
        float dep = line * clamp((tooth - (1.0 - pd)) / g + 0.5 + fill,
                                 0.0, 1.0);
        if (bl > 0.0) {
            float hw2 = max(hw * 2.5, 1e-4);
            float halo = clamp((hw2 - abs(a)) / hw2, 0.0, 1.0) * pd * 0.5;
            float smear = max(line * pd * 0.5, halo);
            dep = dep * (1.0 - bl) + smear * bl;
        }
        dep = dep * smoothstep(0.0, 0.15, wk);
        remain = remain * (1.0 - dep);
    }
    return 1.0 - remain;
}
""",
    # media.stipple: two jittered lattices of dots, present by darkness
    'md_stipple': """
float hal_md_stipple(vec2 p, float d, float size, float jitter, float fine,
                     int salt, int placement)
{
    float fill = clamp((d - 0.8) / 0.2, 0.0, 1.0);
    float best = 0.0;
    float jit = jitter * 0.9;
    float fn = clamp(fine, 0.0, 1.0);
    if (placement == 1) {
        // COUNT: three nested lattices, one dot size, ranks in fixed
        // ranges (media._COUNT_LEVELS)
        for (int level = 0; level < 3; level++) {
            float sc = (level == 0) ? 0.25 : ((level == 1) ? 0.5 : 1.0);
            float lo = (level == 0) ? 0.0 : ((level == 1) ? 0.04761905
                                                          : 0.23809524);
            float hi = (level == 0) ? 0.04761905 : ((level == 1) ? 0.23809524
                                                                 : 1.0);
            float px = p.x * sc;
            float py = p.y * sc;
            int cx0 = int(floor(px));
            int cy0 = int(floor(py));
            float rad = size * 0.5 * (1.0 + 1.5 * fill) * sc;
            for (int dx = -1; dx <= 1; dx++) {
                for (int dy = -1; dy <= 1; dy++) {
                    int cx = cx0 + dx;
                    int cy = cy0 + dy;
                    float jx = hal_md_h(cx, cy, salt, 30 + level * 4);
                    float jy = hal_md_h(cx, cy, salt, 31 + level * 4);
                    float pres = hal_md_h(cx, cy, salt, 32 + level * 4);
                    float rs = hal_md_h(cx, cy, salt, 33 + level * 4);
                    float rank = lo + pres * (hi - lo);
                    float ddx = px - (float(cx) + 0.5 + (jx - 0.5) * jit);
                    float ddy = py - (float(cy) + 0.5 + (jy - 0.5) * jit);
                    float dist = sqrt(ddx * ddx + ddy * ddy);
                    float r = rad * (0.85 + 0.3 * rs);
                    float dot_ = clamp((r - dist) / 0.06 + 0.5, 0.0, 1.0);
                    dot_ = (rank < d) ? dot_ : 0.0;
                    best = max(best, dot_);
                }
            }
        }
        return best;
    }
    for (int level = 0; level < 2; level++) {
        if (level == 1 && fn <= 0.0) { break; }
        float sc = (level == 0) ? 1.0 : 2.0;
        float lv = (level == 0) ? 1.0 : 0.7;
        float px = p.x * sc;
        float py = p.y * sc;
        int cx0 = int(floor(px));
        int cy0 = int(floor(py));
        float dens = (level == 0) ? d : clamp((d - 0.5) * 2.0, 0.0, 1.0) * fn;
        float rad = size * 0.5 * (0.6 + 0.7 * d + 1.5 * fill) * lv;
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                int cx = cx0 + dx;
                int cy = cy0 + dy;
                float jx = hal_md_h(cx, cy, salt, 20 + level * 4);
                float jy = hal_md_h(cx, cy, salt, 21 + level * 4);
                float pres = hal_md_h(cx, cy, salt, 22 + level * 4);
                float rs = hal_md_h(cx, cy, salt, 23 + level * 4);
                float ddx = px - (float(cx) + 0.5 + (jx - 0.5) * jit);
                float ddy = py - (float(cy) + 0.5 + (jy - 0.5) * jit);
                float dist = sqrt(ddx * ddx + ddy * ddy);
                float r = rad * (0.75 + 0.5 * rs);
                float dot_ = clamp((r - dist) / 0.06 + 0.5, 0.0, 1.0);
                dot_ = (pres < dens) ? dot_ : 0.0;
                best = max(best, dot_);
            }
        }
    }
    return best;
}
""",
    # media.charcoal: the stick on the tooth, streaked, smudged, banded
    'md_charcoal': """
float hal_md_charcoal(vec2 p, float d, vec2 rot, float grain, float streak,
                      float smudge, int salt, float blend)
{
    float bl = clamp(blend, 0.0, 1.0);
    float c = rot.x;
    float s = rot.y;
    float u = p.x * c + p.y * s;
    float v = -p.x * s + p.y * c;
    float ex = 1.0 / (1.0 + streak * 4.0);
    float t1 = hal_md_n2(vec2(u * 6.0 * ex, v * 6.0), salt, 60);
    float t2 = hal_md_n2(vec2(u * 14.0 * ex, v * 14.0), salt, 61);
    float t3 = hal_md_n2(vec2(u * 30.0 * ex, v * 30.0), salt, 63);
    float t4 = hal_md_h(int(floor(u * 30.0 * ex)), int(floor(v * 30.0)),
                        salt, 65);
    float tooth = clamp((t1 * 0.3 + t2 * 0.3 + t3 * 0.2 + t4 * 0.2 - 0.5)
                        * 2.5 + 0.5, 0.0, 1.0);
    float sm = ((hal_md_n2(vec2(p.x * 0.6, p.y * 0.6), salt, 62) - 0.5) * smudge
                + (hal_md_n2(vec2(u * 0.25, v * 1.6), salt, 64) - 0.5) * 0.45)
               * clamp(4.0 * d * (1.0 - d), 0.0, 1.0);
    float g = 0.04 + 0.25 * (1.0 - clamp(grain, 0.0, 1.0));
    if (bl > 0.0) {
        tooth = tooth + (0.5 - tooth) * (bl * 0.6);
        g = g + bl * 0.4;
    }
    float P = d * (1.0 + 2.0 * g) - g + sm;
    float cov = smoothstep(1.0 - P - g, 1.0 - P + g, tooth);
    return clamp(cov, 0.0, 1.0);
}
""",
    # media.paint_strokes: dabs over a 5x5 search, two-deep painter's order;
    # returns (value, alpha, id)
    'md_paint': """
vec3 hal_md_paint(vec2 p, vec2 rot, float length, float width, float slope,
                  float bristles, float variation, int salt)
{
    float c0 = rot.x;
    float s0 = rot.y;
    float hl0 = max(length, 1e-3) * 0.5;
    float hw0 = max(width, 1e-3) * 0.5;
    int cx0 = int(floor(p.x));
    int cy0 = int(floor(p.y));
    float o1 = -1.0;
    float v1 = 0.0;
    float a1 = 0.0;
    float i1 = 0.0;
    float o2 = -1.0;
    float v2 = 0.0;
    float a2 = 0.0;
    float br = bristles;
    float var_ = variation;
    for (int dx = -2; dx <= 2; dx++) {
        for (int dy = -2; dy <= 2; dy++) {
            int cx = cx0 + dx;
            int cy = cy0 + dy;
            float jx = hal_md_h(cx, cy, salt, 0);
            float jy = hal_md_h(cx, cy, salt, 1);
            float hj = hal_md_h(cx, cy, salt, 2);
            float order = hal_md_h(cx, cy, salt, 3);
            float val = hal_md_h(cx, cy, salt, 4);
            float bid = hal_md_h(cx, cy, salt, 5);
            float ex = p.x - (float(cx) + 0.5 + (jx - 0.5) * 0.9);
            float ey = p.y - (float(cy) + 0.5 + (jy - 0.5) * 0.9);
            float t = (hj - 0.5) * (2.0 * slope);
            float inv = 1.0 / sqrt(1.0 + t * t);
            float dxr = (c0 - s0 * t) * inv;
            float dyr = (s0 + c0 * t) * inv;
            float al = ex * dxr + ey * dyr;
            float ac = -ex * dyr + ey * dxr;
            float hl = hl0 * (0.7 + 0.6 * bid);
            float hw = hw0 * (0.8 + 0.4 * hj)
                       * (1.0 - 0.3 * clamp(al / hl, 0.0, 1.0));
            float qa = al / hl;
            float qc = ac / hw;
            float q = (qa * qa) * (qa * qa) + (qc * qc) * (qc * qc);
            float alpha = clamp((1.0 - q) / 0.12, 0.0, 1.0);
            bool cover = alpha > 0.0;
            float b1 = hal_md_n2(vec2(ac * 9.0 + bid * 64.0,
                                      al * 0.5 + val * 64.0), salt, 6);
            float b2 = hal_md_n2(vec2(ac * 21.0 + val * 64.0,
                                      al * 1.1 + bid * 64.0), salt, 7);
            float bri = b1 * 0.6 + b2 * 0.4;
            float value = (1.0 + var_ * (val - 0.5)) * (1.0 + br * (bri - 0.5));
            bool win = cover && (order > o1);
            bool second = cover && !win && (order > o2);
            o2 = win ? o1 : (second ? order : o2);
            v2 = win ? v1 : (second ? value : v2);
            a2 = win ? a1 : (second ? alpha : a2);
            o1 = win ? order : o1;
            v1 = win ? value : v1;
            a1 = win ? alpha : a1;
            i1 = win ? bid : i1;
        }
    }
    float under_v = v2 * a2;
    float under_a = a2;
    float value = under_v + (v1 - under_v) * a1;
    float alpha = under_a + (1.0 - under_a) * a1;
    return vec3(value, alpha, i1);
}
""",
    # media.wash: n flat washes, pooled edges, granulation
    'md_wash': """
float hal_md_wash(vec2 p, float d, int n, float pooling, float granulation,
                  float bleed, int salt)
{
    float dn = d + (hal_md_n2(vec2(p.x * 0.9, p.y * 0.9), salt, 70) - 0.5)
                   * (0.5 * bleed);
    float cov = 0.0;
    float pool = pooling;
    float share = 1.0 / float(n);
    for (int k = 1; k <= n; k++) {
        float e = dn - float(k) / float(n + 1);
        float ak = smoothstep(0.0, 0.03, e);
        float r = clamp(1.0 - e / 0.06, 0.0, 1.0);
        float rim = pool * (r * r) * ak;
        cov = cov + share * ak * (1.0 + rim);
    }
    float g1 = hal_md_n2(vec2(p.x * 9.0, p.y * 9.0), salt, 71);
    float g2 = hal_md_n2(vec2(p.x * 21.0, p.y * 21.0), salt, 72);
    float gr = 1.0 + granulation * ((g1 * 0.6 + g2 * 0.4) - 0.5);
    return clamp(cov * gr, 0.0, 1.0);
}
""",
    # media.paper: tooth, fibres, mottle
    'md_paper': """
float hal_md_paper(vec2 p, float tooth, float fibres, float mottle, int salt)
{
    float t1 = hal_md_n2(vec2(p.x * 9.0, p.y * 9.0), salt, 80);
    float t2 = hal_md_n2(vec2(p.x * 23.0, p.y * 23.0), salt, 81);
    float tv = t1 * 0.6 + t2 * 0.4;
    float f1 = hal_md_n2(vec2(p.x * 1.3, p.y * 16.0), salt, 82);
    float f2 = hal_md_n2(vec2(p.x * 16.0, p.y * 1.3), salt, 83);
    float fb = (f1 + f2) * 0.5;
    float m1 = hal_md_n2(vec2(p.x * 0.8, p.y * 0.8), salt, 84);
    float m2 = hal_md_n2(vec2(p.x * 1.6, p.y * 1.6), salt, 85);
    float mv = m1 * 0.65 + m2 * 0.35;
    float h = 0.5 + tooth * (tv - 0.5) + fibres * (fb - 0.5)
              + mottle * (mv - 0.5);
    return clamp(h, 0.0, 1.0);
}
""",
}

PATTERN_GLSL.update(MEDIA_GLSL)


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


# ===================================================== R242/R243: 3ds Max
# maxmaps.py line for line, on Max's own algorithms (R243): the tables
# the CPU materialised are computed here inline from the same hash
# (hal_pt_hash3), so a lattice corner has one value on both devices; the
# arithmetic keeps maxmaps' operation order so the two sides split every
# band at the same bit. Every control is a batch constant, so the loops
# take literal counts.
MAX_GLSL = {
    # ------------------------------------------------------------ the core
    # maxmaps' tables and noise: the permutation, gradient and random
    # tables as the SAME hash the CPU materialised, Perlin's 512-lattice
    # noise3 in Max's op order, noise3DS / NOISE, and the small helpers.
    'mx_prims': """
int hal_mx_perm(int i)
{
    return int(hal_pt_hash3(i, 7, 13) * 511.99);
}

vec3 hal_mx_g3(int i)
{
    vec3 g = vec3(hal_pt_hash3(i, 31, 71), hal_pt_hash3(i, 32, 71),
                  hal_pt_hash3(i, 33, 71)) * 2.0 - 1.0;
    return g / sqrt(g.x * g.x + g.y * g.y + g.z * g.z);
}

vec4 hal_mx_g4(int i)
{
    vec4 g = vec4(hal_pt_hash3(i, 31, 73), hal_pt_hash3(i, 32, 73),
                  hal_pt_hash3(i, 33, 73), hal_pt_hash3(i, 34, 73)) * 2.0 - 1.0;
    return g / sqrt(g.x * g.x + g.y * g.y + g.z * g.z + g.w * g.w);
}

float hal_mx_rand01(int i) { return hal_pt_hash3(i, 3, 5); }
float hal_mx_rand02(int i) { return hal_pt_hash3(i, 11, 17); }

int hal_mx_b0(float v) { return int(v + 10000.0) & 511; }
float hal_mx_r0(float v) { float t = v + 10000.0; return t - float(int(t)); }
float hal_mx_scurve(float t) { return t * t * (3.0 - 2.0 * t); }
float hal_mx_lerp(float t, float a, float b) { return a + t * (b - a); }

float hal_mx_noise3(vec3 p)
{
    int bx0 = hal_mx_b0(p.x); int bx1 = (bx0 + 1) & 511;
    float rx0 = hal_mx_r0(p.x); float rx1 = rx0 - 1.0;
    int by0 = hal_mx_b0(p.y); int by1 = (by0 + 1) & 511;
    float ry0 = hal_mx_r0(p.y); float ry1 = ry0 - 1.0;
    int bz0 = hal_mx_b0(p.z); int bz1 = (bz0 + 1) & 511;
    float rz0 = hal_mx_r0(p.z); float rz1 = rz0 - 1.0;
    int i = hal_mx_perm(bx0);
    int j = hal_mx_perm(bx1);
    int b00 = hal_mx_perm(i + by0);
    int b10 = hal_mx_perm(j + by0);
    int b01 = hal_mx_perm(i + by1);
    int b11 = hal_mx_perm(j + by1);
    float sx = hal_mx_scurve(rx0);
    float sy = hal_mx_scurve(ry0);
    float sz = hal_mx_scurve(rz0);
    vec3 g;
    g = hal_mx_g3(b00 + bz0); float u = rx0 * g.x + ry0 * g.y + rz0 * g.z;
    g = hal_mx_g3(b10 + bz0); float v = rx1 * g.x + ry0 * g.y + rz0 * g.z;
    float a = hal_mx_lerp(sx, u, v);
    g = hal_mx_g3(b01 + bz0); u = rx0 * g.x + ry1 * g.y + rz0 * g.z;
    g = hal_mx_g3(b11 + bz0); v = rx1 * g.x + ry1 * g.y + rz0 * g.z;
    float b = hal_mx_lerp(sx, u, v);
    float c = hal_mx_lerp(sy, a, b);
    g = hal_mx_g3(b00 + bz1); u = rx0 * g.x + ry0 * g.y + rz1 * g.z;
    g = hal_mx_g3(b10 + bz1); v = rx1 * g.x + ry0 * g.y + rz1 * g.z;
    a = hal_mx_lerp(sx, u, v);
    g = hal_mx_g3(b01 + bz1); u = rx0 * g.x + ry1 * g.y + rz1 * g.z;
    g = hal_mx_g3(b11 + bz1); v = rx1 * g.x + ry1 * g.y + rz1 * g.z;
    b = hal_mx_lerp(sx, u, v);
    float d = hal_mx_lerp(sy, a, b);
    return hal_mx_lerp(sz, c, d);
}

float hal_mx_noise3ds(vec3 p)
{
    return clamp(1.65 * hal_mx_noise3(p), -1.0, 1.0);
}

float hal_mx_NOISE(vec3 p)
{
    return (1.0 + hal_mx_noise3ds(p)) * 0.5;
}

float hal_mx_threshold(float x, float a, float b)
{
    if (a == b) return clamp(x, 0.0, 1.0);
    return clamp((x - a) / (b - a), 0.0, 1.0);
}

float hal_mx_smoothstep(float a, float b, float x)
{
    b = max(b, a + 1e-6);
    float t = clamp((x - a) / (b - a), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

float hal_mx_mixcurve(float lo, float hi, float x)
{
    if (hi <= lo) return (x >= hi) ? 1.0 : 0.0;
    float t = clamp((x - lo) / (hi - lo), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}
""",
    # maxmaps.noise4 + mx_noise: the 4D lattice noise with Phase as time;
    # Regular / Fractal / Turbulence with fractional Levels
    'mx_noise': """
float hal_mx_noise4(vec3 p, float w)
{
    int bx0 = hal_mx_b0(p.x); int bx1 = (bx0 + 1) & 511;
    float rx0 = hal_mx_r0(p.x); float rx1 = rx0 - 1.0;
    int by0 = hal_mx_b0(p.y); int by1 = (by0 + 1) & 511;
    float ry0 = hal_mx_r0(p.y); float ry1 = ry0 - 1.0;
    int bz0 = hal_mx_b0(p.z); int bz1 = (bz0 + 1) & 511;
    float rz0 = hal_mx_r0(p.z); float rz1 = rz0 - 1.0;
    int bw0 = hal_mx_b0(w); int bw1 = (bw0 + 1) & 511;
    float rw0 = hal_mx_r0(w); float rw1 = rw0 - 1.0;
    int i = hal_mx_perm(bx0);
    int j = hal_mx_perm(bx1);
    int b00 = hal_mx_perm(i + by0);
    int b10 = hal_mx_perm(j + by0);
    int b01 = hal_mx_perm(i + by1);
    int b11 = hal_mx_perm(j + by1);
    int c00 = hal_mx_perm(b00 + bz0);
    int c10 = hal_mx_perm(b10 + bz0);
    int c01 = hal_mx_perm(b01 + bz0);
    int c11 = hal_mx_perm(b11 + bz0);
    int d00 = hal_mx_perm(b00 + bz1);
    int d10 = hal_mx_perm(b10 + bz1);
    int d01 = hal_mx_perm(b01 + bz1);
    int d11 = hal_mx_perm(b11 + bz1);
    float sx = hal_mx_scurve(rx0);
    float sy = hal_mx_scurve(ry0);
    float sz = hal_mx_scurve(rz0);
    float sw = hal_mx_scurve(rw0);
    vec4 g;
    float u; float v; float a; float b; float c; float d;
    float e; float f;
    // the w0 octant
    g = hal_mx_g4(c00 + bw0); u = rx0 * g.x + ry0 * g.y + rz0 * g.z + rw0 * g.w;
    g = hal_mx_g4(c10 + bw0); v = rx1 * g.x + ry0 * g.y + rz0 * g.z + rw0 * g.w;
    a = hal_mx_lerp(sx, u, v);
    g = hal_mx_g4(c01 + bw0); u = rx0 * g.x + ry1 * g.y + rz0 * g.z + rw0 * g.w;
    g = hal_mx_g4(c11 + bw0); v = rx1 * g.x + ry1 * g.y + rz0 * g.z + rw0 * g.w;
    b = hal_mx_lerp(sx, u, v);
    c = hal_mx_lerp(sy, a, b);
    g = hal_mx_g4(d00 + bw0); u = rx0 * g.x + ry0 * g.y + rz1 * g.z + rw0 * g.w;
    g = hal_mx_g4(d10 + bw0); v = rx1 * g.x + ry0 * g.y + rz1 * g.z + rw0 * g.w;
    a = hal_mx_lerp(sx, u, v);
    g = hal_mx_g4(d01 + bw0); u = rx0 * g.x + ry1 * g.y + rz1 * g.z + rw0 * g.w;
    g = hal_mx_g4(d11 + bw0); v = rx1 * g.x + ry1 * g.y + rz1 * g.z + rw0 * g.w;
    b = hal_mx_lerp(sx, u, v);
    d = hal_mx_lerp(sy, a, b);
    e = hal_mx_lerp(sz, c, d);
    // the w1 octant
    g = hal_mx_g4(c00 + bw1); u = rx0 * g.x + ry0 * g.y + rz0 * g.z + rw1 * g.w;
    g = hal_mx_g4(c10 + bw1); v = rx1 * g.x + ry0 * g.y + rz0 * g.z + rw1 * g.w;
    a = hal_mx_lerp(sx, u, v);
    g = hal_mx_g4(c01 + bw1); u = rx0 * g.x + ry1 * g.y + rz0 * g.z + rw1 * g.w;
    g = hal_mx_g4(c11 + bw1); v = rx1 * g.x + ry1 * g.y + rz0 * g.z + rw1 * g.w;
    b = hal_mx_lerp(sx, u, v);
    c = hal_mx_lerp(sy, a, b);
    g = hal_mx_g4(d00 + bw1); u = rx0 * g.x + ry0 * g.y + rz1 * g.z + rw1 * g.w;
    g = hal_mx_g4(d10 + bw1); v = rx1 * g.x + ry0 * g.y + rz1 * g.z + rw1 * g.w;
    a = hal_mx_lerp(sx, u, v);
    g = hal_mx_g4(d01 + bw1); u = rx0 * g.x + ry1 * g.y + rz1 * g.z + rw1 * g.w;
    g = hal_mx_g4(d11 + bw1); v = rx1 * g.x + ry1 * g.y + rz1 * g.z + rw1 * g.w;
    b = hal_mx_lerp(sx, u, v);
    d = hal_mx_lerp(sy, a, b);
    f = hal_mx_lerp(sz, c, d);
    return hal_mx_lerp(sw, e, f);
}

float hal_mx_noise(vec3 p, int kind, float levels, float low, float high,
                   float phase)
{
    float lev = clamp(levels, 1.0, 10.0);
    float res;
    if (kind == 0) {
        res = (1.0 + hal_mx_noise4(p, phase)) * 0.5;
    } else {
        int count = int(ceil(lev));
        float rest = lev - floor(lev);
        float total = 0.0;
        float f = 1.0;
        for (int i = 0; i < 10; i++) {
            if (i >= count) break;
            float factor = (i == count - 1 && rest > 0.0) ? rest : 1.0;
            float nz = hal_mx_noise4(p * f, phase);
            if (kind == 2) nz = abs(nz);
            total = total + factor * nz / f;
            f = f * 2.0;
        }
        res = (kind == 1) ? 0.5 * (total + 1.0) : total;
    }
    if (low < high) res = hal_mx_threshold(res, low, high);
    return clamp(res, 0.0, 1.0);
}
""",
    # maxmaps.cell_function / fractal_cell_function / mx_cellular /
    # cellular_colors: Max's 27-cell Poisson Worley, squared distances,
    # the fractal sum by lacunarity, the colour rule with Variation
    'mx_cellular': """
int hal_mx_cell_count(float u)
{
    if (u < 0.049787) return 0;
    if (u < 0.199148) return 1;
    if (u < 0.423190) return 2;
    if (u < 0.647232) return 3;
    if (u < 0.815263) return 4;
    if (u < 0.916082) return 5;
    if (u < 0.966492) return 6;
    if (u < 0.988096) return 7;
    return 8;
}

// (f1, f2, the nearest point's id mod 10000)
vec3 hal_mx_cell_function(vec3 p, int want2)
{
    vec3 fp = floor(p);
    int ipx = int(fp.x); int ipy = int(fp.y); int ipz = int(fp.z);
    vec3 fip0 = fp - p;
    float f1 = 1e30;
    float f2 = 1e30;
    int fid = 0;
    for (int dx = -1; dx <= 1; dx++) {
        int px = hal_mx_perm((ipx + dx) & 511);
        for (int dy = -1; dy <= 1; dy++) {
            int py = hal_mx_perm((ipy + dy) & 511) * 512;
            for (int dz = -1; dz <= 1; dz++) {
                int pz = hal_mx_perm((ipz + dz) & 511) * 262144;
                int cid = px + py + pz;
                int y = cid % 10000;
                int ct = hal_mx_cell_count(hal_mx_rand01(y));
                float fipx = fip0.x + float(dx);
                float fipy = fip0.y + float(dy);
                float fipz = fip0.z + float(dz);
                int cntr = 1;
                for (int k = 0; k < 9; k++) {
                    if (k >= ct) break;
                    float qz = hal_mx_rand02(y + cntr) + fipz;
                    float qy = hal_mx_rand02(y + cntr + 1) + fipy;
                    float qx = hal_mx_rand02(y + cntr + 2) + fipx;
                    cntr = cntr + 3;
                    float d = qx * qx + qy * qy + qz * qz;
                    if (want2 == 1) f2 = min(f2, max(f1, d));
                    if (d < f1) { fid = cid + k; }
                    f1 = min(f1, d);
                }
            }
        }
    }
    return vec3(f1, f2, float(fid % 10000));
}

vec3 hal_mx_fractal_cell(vec3 p, float iterations, float lacunarity, int want2)
{
    int it = int(min(iterations, 25.0));
    float rem = iterations - float(it);
    vec3 r = hal_mx_cell_function(p, want2);
    float u = lacunarity;
    for (int i = 1; i < 25; i++) {
        if (i >= it) break;
        vec3 d = hal_mx_cell_function(p * u, want2);
        r.x = r.x + d.x / u;
        r.y = r.y + d.y / u;
        r.z = d.z;
        u = u * lacunarity;
    }
    if (rem > 0.0) {
        vec3 d = hal_mx_cell_function(p * u, want2);
        r.x = r.x + rem * d.x / u;
        r.y = r.y + rem * d.y / u;
        r.z = d.z;
    }
    return r;
}

// (u, the cell id mod 10000)
vec2 hal_mx_cellular(vec3 p, int chips, float spread, int fractal,
                     float iterations, float roughness)
{
    vec3 q = p + 1000.0;
    float lac = 2.0 - roughness;
    vec3 r;
    if (fractal == 1) r = hal_mx_fractal_cell(q, iterations, lac, chips);
    else r = hal_mx_cell_function(q, chips);
    float u;
    if (chips == 1) u = 1.0 - (r.y - r.x) / max(spread * 0.5, 1e-6);
    else u = r.x / max(spread, 1e-6);
    return vec2(u, r.z);
}

vec4 hal_mx_cellular_colors(vec2 uf, vec4 cell, vec4 div1, vec4 div2,
                            float low, float mid, float high, float variation)
{
    float u = uf.x;
    float var = variation / 50.0;
    if (var > 0.0) {
        float vr = hal_mx_rand01(int(uf.y)) * var + (1.0 - var * 0.5);
        cell.rgb = clamp(cell.rgb * vr, 0.0, 1.0);
    }
    float ml = max(mid - low, 1e-6);
    float hm = max(high - mid, 1e-6);
    vec4 outc;
    if (u < low) outc = cell;
    else if (u > high) outc = div2;
    else if (u < mid) { float t = (u - low) / ml; outc = div1 * t + (1.0 - t) * cell; }
    else { float t = (u - mid) / hm; outc = div2 * t + (1.0 - t) * div1; }
    outc.a = 1.0;
    return outc;
}
""",
    # maxmaps.mx_smoke: |noise3| octaves drifted by their velocities
    'mx_smoke': """
vec3 hal_mx_smoke_vel(int i)
{
    return vec3(hal_pt_hash3(i, 41, 43) * 2.0 - 1.0,
                hal_pt_hash3(i, 47, 53) * 2.0 - 1.0,
                hal_pt_hash3(i, 59, 61) * 2.0 - 1.0);
}

float hal_mx_smoke(vec3 p, int iterations, float phase, float exponent)
{
    vec3 x = p;
    float mag = 0.0;
    float s = 1.0;
    float ft = 1.0;
    for (int i = 0; i < 20; i++) {
        if (i >= iterations) break;
        float k = ft * phase;
        vec3 r = x + hal_mx_smoke_vel(i) * k;
        mag = mag + abs(hal_mx_noise3(r)) / s;
        x = x * 2.0;
        s = s * 2.0;
        ft = ft * 2.4;
    }
    float d = min(mag, 1.0);
    return pow(d, max(exponent, 1e-4));
}
""",
    # maxmaps.mx_speckle: the point x10, six octaves of NOISE, capped
    'mx_speckle': """
float hal_mx_speckle(vec3 p)
{
    vec3 q = p * 10.0;
    float total = 0.0;
    float s = 1.0;
    for (int i = 0; i < 6; i++) {
        total = total + hal_mx_NOISE(q) / s;
        s = s * 2.0;
        q = q * 2.0;
    }
    return min(total, 1.0);
}
""",
    # maxmaps.mx_splat: one minus the product of smoothsteps of NOISE
    'mx_splat': """
float hal_mx_splat(vec3 p, int iterations, float threshold)
{
    vec3 q = p;
    float fact = 1.0;
    for (int i = 0; i < 5; i++) {
        if (i >= iterations) break;
        float t = min(hal_mx_NOISE(q), 1.0);
        fact = fact * hal_mx_smoothstep(threshold - 0.02, threshold + 0.02, t);
        q = q * 2.0;
    }
    return 1.0 - fact;
}
""",
    # maxmaps.mx_stucco: the knee curve past Threshold over Thickness
    'mx_stucco': """
float hal_mx_stucco(vec3 p, float thickness, float threshold)
{
    float f = 0.5 * (hal_mx_noise3(p) + 1.0);
    f = (f - threshold) / max(thickness, 1e-6);
    if (f <= 0.0) return 0.0;
    if (f >= 1.0) return 1.0;
    if (f < 0.2) return 3.125 * f * f;
    if (f < 1.0 - 0.2) return 0.625 * (2.0 * f - 0.2);
    return 1.0 - 3.125 * (1.0 - f) * (1.0 - f);
}
""",
    # maxmaps.mx_swirl: the twisted UV, octaves of noise3 at gain Contrast
    'mx_swirl': """
float hal_mx_swirl(vec3 p, float cx, float cy, float twist, float intensity,
                   float amount, int detail, float contrast, float seed)
{
    float u = p.x + cx;
    float v = p.y + cy;
    float rsq = u * u + v * v;
    float ang = twist * 6.2831853 * rsq;
    float sn = sin(ang);
    float cs = cos(ang);
    float ppx = v * cs - u * sn;
    float ppy = v * sn + u * cs;
    float ppz = seed;
    float a = 0.0;
    float l = 1.0;
    float o = 1.0;
    for (int i = 0; i < 7; i++) {
        if (i >= detail) break;
        a = a + o * hal_mx_noise3(vec3(ppx * l, ppy * l, ppz * l));
        l = l * 2.0;
        o = o * contrast;
    }
    return amount * intensity * a;
}
""",
    # maxmaps.dent_noise / wood_noise: the 21-cube linear table noise,
    # periodic every 20 (Dent, Planet, Wood)
    'mx_dentnoise': """
float hal_mx_dent_table(int ix, int iy, int iz)
{
    return hal_pt_hash3((ix % 20) + 101, (iy % 20) + 203, (iz % 20) + 307);
}

float hal_mx_dent_noise(vec3 p)
{
    vec3 m = mod(p, 20.0);
    int ix = int(m.x); int iy = int(m.y); int iz = int(m.z);
    float fx = mod(m.x, 1.0); float fy = mod(m.y, 1.0); float fz = mod(m.z, 1.0);
    float n = hal_mx_dent_table(ix, iy, iz);
    float n00 = n + fx * (hal_mx_dent_table(ix + 1, iy, iz) - n);
    n = hal_mx_dent_table(ix, iy, iz + 1);
    float n01 = n + fx * (hal_mx_dent_table(ix + 1, iy, iz + 1) - n);
    n = hal_mx_dent_table(ix, iy + 1, iz);
    float n10 = n + fx * (hal_mx_dent_table(ix + 1, iy + 1, iz) - n);
    n = hal_mx_dent_table(ix, iy + 1, iz + 1);
    float n11 = n + fx * (hal_mx_dent_table(ix + 1, iy + 1, iz + 1) - n);
    float n0 = n00 + fy * (n10 - n00);
    float n1 = n01 + fy * (n11 - n01);
    return n0 + fz * (n1 - n0);
}

float hal_mx_wood_noise(float x)
{
    float m = mod(x, 20.0);
    int ix = int(m);
    float fx = mod(m, 1.0);
    float n0 = hal_mx_dent_table(ix, 0, 0);
    float n1 = hal_mx_dent_table(ix + 1, 0, 0);
    return n0 + fx * (n1 - n0);
}
""",
    # maxmaps.mx_planet / planet_colors
    'mx_planet': """
float hal_mx_planet(vec3 p, float island)
{
    return hal_mx_dent_noise(p) + hal_mx_dent_noise(p * island) / 5.0;
}

vec4 hal_mx_planet_colors(float d, vec4 c0, vec4 c1, vec4 c2, vec4 c3,
                          vec4 c4, vec4 c5, vec4 c6, vec4 c7,
                          float ocean_pct, int blend)
{
    float land = clamp(ocean_pct, 0.0, 100.0) / 100.0;
    land = min(max(land, 1e-4), 1.0 - 1e-4);
    if (d < land) {
        float dw = d / land * 3.0;
        int iw = min(int(dw), 2);
        float fw = dw - float(iw);
        float omf = 1.0 - fw;
        if (iw == 0) return omf * c0 + fw * c1;
        if (iw == 1) return omf * c1 + fw * c2;
        if (blend == 1) return omf * c2 + fw * c3;
        return c2;
    }
    float dl = (d - land) / (1.0 - land) * 5.0;
    int il = min(int(max(dl, 0.0)), 5);
    float fl = dl - float(il);
    float omf = 1.0 - fl;
    if (il == 0) return omf * c3 + fl * c4;
    if (il == 1) return omf * c4 + fl * c5;
    if (il == 2) return omf * c5 + fl * c6;
    if (il == 3) return omf * c6 + fl * c7;
    return c7;
}
""",
    # maxmaps.mx_waves: the C runtime's rand() walk seeds the wave sets
    'mx_waves': """
float hal_mx_waves(vec3 p, int sets, float radius, float len_min, float len_max,
                   float amplitude, float phase, int dist3d, int seed)
{
    int count = max(1, min(sets, 50));
    uint h = uint(seed);
    float n = 0.0;
    float lmax = max(len_max, 1e-6);
    for (int i = 0; i < 50; i++) {
        if (i >= count) break;
        h = h * 214013u + 2531011u;
        float cx = float(int((h >> 16u) & 0x7fffu)) / 16384.0 - 1.0;
        float cy = 0.0;
        if (dist3d == 1) {
            h = h * 214013u + 2531011u;
            cy = float(int((h >> 16u) & 0x7fffu)) / 16384.0 - 1.0;
        }
        h = h * 214013u + 2531011u;
        float cz = float(int((h >> 16u) & 0x7fffu)) / 16384.0 - 1.0;
        float ln = sqrt(cx * cx + cy * cy + cz * cz);
        float dd = radius / max(ln, 1e-6);
        h = h * 214013u + 2531011u;
        float period = float(int((h >> 16u) & 0x7fffu)) / 32768.0 * (len_max - len_min) + len_min;
        period = max(period, 1e-6);
        float rate = sqrt(lmax / period);
        float vx = (p.x - cx * dd) / period;
        float vy = (p.y - cy * dd) / period;
        float vz = (p.z - cz * dd) / period;
        float d = sqrt(vx * vx + vy * vy + vz * vz);
        float t = 0.5 * (1.0 + sin((d - phase * rate) * 6.2831853));
        n = n + t * period / lmax;
    }
    float v = n * amplitude / float(count);
    return min(v, 1.0);
}
""",
    # maxmaps.mx_checker: Max's integrated soften, or the parity test
    'mx_checker': """
float hal_mx_sintegral(float x)
{
    float fl = floor(x);
    return fl * 0.5 + max(0.0, (x - fl) - 0.5);
}

float hal_mx_checker(vec3 p, float soften)
{
    float u = p.x;
    float v = p.y;
    if (soften <= 0.0) {
        float fu = u - floor(u);
        float fv = v - floor(v);
        bool a = fu > 0.5;
        bool b = fv > 0.5;
        return ((a || b) && !(a && b)) ? 0.0 : 1.0;
    }
    float du = soften;
    float hdu = du * 0.5;
    float s = (hal_mx_sintegral(u + hdu) - hal_mx_sintegral(u - hdu)) / du;
    float t = (hal_mx_sintegral(v + hdu) - hal_mx_sintegral(v - hdu)) / du;
    return s * t + (1.0 - s) * (1.0 - t);
}
""",
    # maxmaps.mx_tiles -> vec3(mixer, brick random, fade factor)
    'mx_tiles': """
float hal_mx_boxstep(float a, float b, float x)
{
    return clamp((x - a) / (b - a), 0.0, 1.0);
}

vec3 hal_mx_tiles(vec3 p, int pattern, float hcount, float vcount,
                  float hgap, float vgap, float line_shift, float random_shift,
                  float holes, float fade, float color_var, int seed)
{
    float mtx = hgap * 0.01;
    float mty = vgap * 0.01;
    float bw = 1.0 / (hcount + mtx);
    float bh = 1.0 / (vcount + mty);
    float mwf = mtx / bw;
    float mhf = mty / bh;
    float ss = p.x / bw;
    float tt = p.y / bh;
    float tbrick = floor(tt);
    bool odd = mod(tbrick, 2.0) > 0.5;
    ss = ss + random_shift * (hal_pt_hash3(int(tbrick), seed, 29) - 0.5);
    float w = 1.0;
    if (pattern == 1) { if (odd) ss = ss + line_shift; }
    else if (pattern == 2) { if (odd) { w = 0.5; ss = ss + line_shift * 0.5; } }
    else if (pattern == 3) { if (odd) ss = ss + line_shift * 1.5; }
    float sbrick;
    float fs;
    if (pattern == 3) {
        float t = mod(ss, 1.5);
        bool head = t >= 1.0;
        sbrick = floor(ss / 1.5) * 2.0 + (head ? 1.0 : 0.0);
        fs = head ? (t - 1.0) / 0.5 : t;
        w = head ? 0.5 : 1.0;
    } else {
        sbrick = floor(ss / w);
        fs = (ss - sbrick * w) / w;
    }
    float ft = tt - tbrick;
    float mw = mwf / w;
    float wx = hal_mx_boxstep(mw, mw + 1e-4, fs) - hal_mx_boxstep(1.0 - mw, 1.0 - mw + 1e-4, fs);
    float wy = hal_mx_boxstep(mhf, mhf + 1e-4, ft) - hal_mx_boxstep(1.0 - mhf, 1.0 - mhf + 1e-4, ft);
    float mixer = wx * wy;
    int sb = int(sbrick);
    int tb = int(tbrick);
    float rnd = hal_pt_hash3(sb, tb, seed);
    if (hal_pt_hash3(sb, tb, seed + 97) < holes / 100.0) mixer = 0.0;
    float n1 = rnd - 0.5;
    float fade_v = (1.0 + color_var * n1) * (1.0 + fade * n1);
    return vec3(mixer, rnd, max(fade_v, 0.0));
}
""",
    # maxmaps.mx_marble: the point x500, the seventeen-band veins
    'mx_marble': """
float hal_mx_marble(vec3 p, float width)
{
    vec3 q = p * 500.0;
    float x = q.x; float y = q.y; float z = q.z;
    vec3 r = vec3(x / 100.0, y / 200.0, z / 200.0);
    float d = (x + 10000.0) * width + 7.0 * hal_mx_NOISE(r);
    float idx = mod(d, 17.0);
    float outv;
    if (idx < 4.0) {
        vec3 r2 = vec3(x / 70.0, y / 50.0, z / 50.0);
        outv = 0.7 + 0.2 * hal_mx_NOISE(r2);
    } else {
        vec3 r3 = vec3(x / 100.0, y / 100.0, x / 100.0);
        float n3 = hal_mx_NOISE(r3);
        if (idx < 9.0 || idx >= 12.0) {
            float dd = abs(d - floor(d / 17.0) * 17.0 - 10.5) * 0.1538462;
            outv = 0.4 + 0.3 * dd + 0.2 * n3;
        } else {
            outv = 0.2 * (1.0 + n3);
        }
    }
    return clamp(outv, 0.0, 1.0);
}
""",
    # maxmaps.mx_perlin_marble / perlin_marble_colors: the turbulence
    # vein coordinate and Perlin's thirteen-knot Catmull-Rom
    'mx_perlin_marble': """
float hal_mx_perlin_marble(vec3 p, int levels)
{
    float turb = 0.0;
    float freq = 1.0;
    for (int i = 0; i < 8; i++) {
        if (i >= levels) break;
        turb = turb + abs(hal_mx_NOISE(p * freq)) / freq;
        freq = freq * 2.0;
    }
    return clamp(sin(p.x + (4.0 * turb - 3.0)), 0.0, 1.0);
}

vec3 hal_mx_pm_knot(int i, vec3 c0, vec3 c1, vec3 lc0, vec3 dc1)
{
    if (i <= 1) return lc0;
    if (i <= 4) return c0;
    if (i <= 6) return lc0;
    if (i <= 8) return c1;
    if (i <= 10) return dc1;
    if (i == 11) return lc0;
    return dc1;
}

vec4 hal_mx_perlin_marble_colors(float csp, vec4 col0, vec4 col1,
                                 float sat1, float sat2)
{
    vec3 c0 = col0.rgb;
    vec3 c1 = col1.rgb;
    vec3 lc0 = clamp(c0 * (2.0 * sat1 - 1.0), 0.0, 1.0);
    vec3 dc1 = clamp(c1 * (2.0 * sat2 - 1.0), 0.0, 1.0);
    float x = clamp(csp, 0.0, 1.0) * 10.0;
    int span = min(int(x), 9);
    x = x - float(span);
    vec3 k0 = hal_mx_pm_knot(span, c0, c1, lc0, dc1);
    vec3 k1 = hal_mx_pm_knot(span + 1, c0, c1, lc0, dc1);
    vec3 k2 = hal_mx_pm_knot(span + 2, c0, c1, lc0, dc1);
    vec3 k3 = hal_mx_pm_knot(span + 3, c0, c1, lc0, dc1);
    vec3 c3 = -0.5 * k0 + 1.5 * k1 + -1.5 * k2 + 0.5 * k3;
    vec3 c2 = 1.0 * k0 + -2.5 * k1 + 2.0 * k2 + -0.5 * k3;
    vec3 cc1 = -0.5 * k0 + 0.0 * k1 + 0.5 * k2 + 0.0 * k3;
    vec3 cc0 = 0.0 * k0 + 1.0 * k1 + 0.0 * k2 + 0.0 * k3;
    vec3 rgb = ((c3 * x + c2) * x + cc1) * x + cc0;
    return vec4(rgb, 1.0);
}
""",
    # maxmaps.mx_wood: the ring bands on the jittered radius
    'mx_wood': """
float hal_mx_wood(vec3 p, float radial, float axial)
{
    float px = p.x + hal_mx_wood_noise(p.x) * radial;
    float py = p.y + hal_mx_wood_noise(p.y) * radial;
    float pz = p.z + hal_mx_wood_noise(p.z) * radial;
    float r = sqrt(py * py + pz * pz);
    px = px / 4.0;
    r = r + (hal_mx_wood_noise(r) + axial * hal_mx_wood_noise(px));
    r = mod(r, 1.0);
    return hal_mx_smoothstep(0.0, 0.8, r) - hal_mx_smoothstep(0.83, 1.0, r);
}
""",
    # maxmaps.mx_dent: the point x50, |0.5 - dent noise| octaves, cubed
    'mx_dent': """
float hal_mx_dent(vec3 p, float strength, int iterations)
{
    vec3 q = p * 50.0;
    float s = 1.0;
    float mag = 0.0;
    for (int i = 0; i < 10; i++) {
        if (i >= iterations) break;
        mag = mag + abs(0.5 - hal_mx_dent_noise(q)) / s;
        s = s * 2.0;
        q = q * 2.0;
    }
    return clamp(mag * mag * mag * strength, 0.0, 1.0);
}
""",
    # maxmaps.mx_gradient_ramp: the ramp coordinate for each gradient type
    'mx_gradient_ramp': """
float hal_mx_gradient_ramp(float u, float v, int kind, float ndv, float mapped)
{
    float a;
    if (kind == 0) {
        a = (u > 0.0) ? (v * v) / max(u, 1e-6) * u : v;
    } else if (kind == 1) {
        float lu = abs(u - 0.5) * 2.0;
        float lv = abs(v - 0.5) * 2.0;
        a = min(max(lu, lv), 1.0);
    } else if (kind == 2) {
        a = sqrt(2.0 * (v - u) * (v - u));
    } else if (kind == 4) {
        a = 1.0 - clamp(abs(ndv), 0.0, 1.0);
    } else if (kind == 5) {
        a = (v > 0.0) ? u / max(v, 1e-6) : v;
        if (a > 1.0) a = (u > 0.0) ? v / max(u, 1e-6) : 0.0;
        a = min(a, 1.0);
    } else if (kind == 6) {
        a = min(sqrt((u - 0.5) * (u - 0.5) + (v - 0.5) * (v - 0.5)) * 2.0, 1.0);
    } else if (kind == 7) {
        float ln = sqrt(u * u + v * v);
        float c = clamp((ln > 0.0) ? u / max(ln, 1e-6) : 1.0, -1.0, 1.0);
        a = min(acos(c) * 57.29578 / 90.0, 1.0);
    } else if (kind == 8) {
        float lu = u - 0.5;
        float lv = v - 0.5;
        float ln = sqrt(lu * lu + lv * lv);
        float c = clamp((ln > 0.0) ? lu / max(ln, 1e-6) : 1.0, -1.0, 1.0);
        float x = acos(c) * 57.29578;
        if (lv > 0.0) x = 360.0 - x;
        a = min(x / 360.0, 1.0);
    } else if (kind == 9) {
        float lu = abs(u - 0.5) * 2.0;
        float lv = abs(v - 0.5) * 2.0;
        a = 1.0 - min(min(lu, lv), 1.0);
    } else if (kind == 10) {
        a = mapped;
    } else {
        a = v;
    }
    return clamp(a, 0.0, 1.0);
}
""",
    # maxmaps.max_fresnel: the unpolarised dielectric equation on |N.V|
    # ------------------------------------------ R243: the remaining maps
    # tex_gradient's noise, sramp and shape; Color Correction's HSL and
    # its two lightness roads; the Composite map's twenty-five blend
    # modes and its "over" -- each the CPU function above, op for op.
    'mx_gradient': """
float hal_mx_sramp(float x, float a, float b, float d)
{
    if (d <= 0.0) return clamp(x, a, b);
    float p0 = a - d;
    float p1 = a + d;
    float p2 = b - d;
    float p3 = b + d;
    if (x <= p0) return a;
    if (x >= p3) return b;
    if (x >= p1 && x <= p2) return x;
    if (x > p0 && x < p1) {
        float q = (x - p0) / (2.0 * d);
        return a + q * q * d;
    }
    float q = (p3 - x) / (2.0 * d);
    return b - q * q * d;
}

float hal_mx_gradient_noise(vec3 p, int kind, float levels, float low,
                            float high, float smth)
{
    float res = 0.0;
    if (kind == 0 || (kind == 1 && levels == 1.0)) {
        res = hal_mx_noise3(p);
    } else {
        float f = 1.0;
        float lv = levels;
        for (int i = 0; i < 12; i++) {
            if (lv < 1.0) break;
            float n = hal_mx_noise3(p * f);
            if (kind == 2) n = abs(n);
            res = res + n / f;
            f = f * 2.0;
            lv -= 1.0;
        }
        if (lv > 0.0) {
            float n = hal_mx_noise3(p * f);
            if (kind == 2) n = abs(n);
            res = res + lv * n / f;
        }
    }
    if (low < high) {
        float sd = (high - low) * 0.5 * smth;
        res = 2.0 * hal_mx_sramp((res + 1.0) / 2.0, low, high, sd) - 1.0;
    }
    return res;
}

float hal_mx_gradient(float u, float v, int kind, float amount, float noise)
{
    float a;
    if (kind == 1) {
        float lu = u - 0.5;
        float lv = v - 0.5;
        a = min(sqrt(lu * lu + lv * lv) * 2.0, 1.0);
    } else {
        a = v;
    }
    if (amount != 0.0) a = clamp(a + amount * noise, 0.0, 1.0);
    return a;
}

vec4 hal_mx_gradient_colors(float a, float pos, vec4 c1, vec4 c2, vec4 c3)
{
    float lo = (pos > 0.0) ? pos : 1.0;
    float hi = (pos < 1.0) ? 1.0 - pos : 1.0;
    float t1 = a / lo;
    float t2 = (a - pos) / hi;
    vec4 below = c3 * (1.0 - t1) + c2 * t1;
    vec4 above = c2 * (1.0 - t2) + c1 * t2;
    return (a < pos) ? below : ((a > pos) ? above : c2);
}
""",
    'mx_hsl': """
float hal_mx_rotate(float a, float b, float c)
{
    float delta = c - b;
    if (a < b) {
        float f = float(int((c - a) / delta));
        a = a + delta * f;
    }
    if (a > c) {
        float f = float(int((a - b) / delta));
        a = a - delta * f;
    }
    return a;
}

float hal_mx_hue(vec3 c)
{
    float mn = min(c.r, min(c.g, c.b));
    float mx = max(c.r, max(c.g, c.b));
    float delta = mx - mn;
    if (delta < 0.00001) return 0.0;
    float h;
    if (c.r == mx) h = (c.g - c.b) / delta;
    else if (c.g == mx) h = 2.0 + (c.b - c.r) / delta;
    else h = 4.0 + (c.r - c.g) / delta;
    return hal_mx_rotate(h * 60.0, 0.0, 360.0);
}

float hal_mx_lum(vec3 c)
{
    float mn = min(c.r, min(c.g, c.b));
    float mx = max(c.r, max(c.g, c.b));
    return clamp((mx + mn) / 2.0, 0.0, 1.0);
}

float hal_mx_sat(vec3 c)
{
    float mn = min(c.r, min(c.g, c.b));
    float mx = max(c.r, max(c.g, c.b));
    float lum = hal_mx_lum(c);
    if (mn == mx || lum == 0.0) return 0.0;
    if (lum <= 0.5) return clamp((mx - mn) / (2.0 * lum), 0.0, 1.0);
    float d2 = 2.0 - 2.0 * lum;
    if (d2 == 0.0) d2 = 1.0;
    return clamp((mx - mn) / d2, 0.0, 1.0);
}

float hal_mx_hue2rgb(float v1, float v2, float h)
{
    h = hal_mx_rotate(h, 0.0, 1.0);
    if (6.0 * h < 1.0) return v1 + (v2 - v1) * 6.0 * h;
    if (2.0 * h < 1.0) return v2;
    if (3.0 * h < 2.0) return v1 + (v2 - v1) * (0.6666667 - h) * 6.0;
    return v1;
}

vec3 hal_mx_hsl2rgb(float h, float s, float lum)
{
    float q = (lum < 0.5) ? lum * (1.0 + s) : lum + s - lum * s;
    float p = 2.0 * lum - q;
    float hk = h / 360.0;
    return clamp(vec3(hal_mx_hue2rgb(p, q, hk + 0.33333334),
                      hal_mx_hue2rgb(p, q, hk),
                      hal_mx_hue2rgb(p, q, hk - 0.33333334)), 0.0, 1.0);
}
""",
    'mx_colorcorr': """
float hal_mx_rewire(int k, vec4 c)
{
    if (k == 0) return c.r;
    if (k == 1) return c.g;
    if (k == 2) return c.b;
    if (k == 3) return c.a;
    if (k == 4) return 1.0 - c.r;
    if (k == 5) return 1.0 - c.g;
    if (k == 6) return 1.0 - c.b;
    if (k == 7) return 1.0 - c.a;
    if (k == 8) return (c.r + c.g + c.b) / 3.0;
    if (k == 9) return 1.0;
    return 0.0;
}

vec4 hal_mx_color_correction(vec4 c, int rr, int rg, int rb, int ra,
                             float hue_shift, float saturation, vec3 tint,
                             float strength, int advanced, float brightness,
                             float contrast, float gain, float gamma,
                             float pivot, float lift)
{
    vec4 t = c;
    c = vec4(hal_mx_rewire(rr, t), hal_mx_rewire(rg, t), hal_mx_rewire(rb, t),
             hal_mx_rewire(ra, t));
    float h = hal_mx_hue(c.rgb);
    float s = hal_mx_sat(c.rgb);
    float lum = hal_mx_lum(c.rgb);
    h = hal_mx_rotate(h + hue_shift, 0.0, 360.0);
    s = clamp(s + saturation / 100.0, 0.0, 1.0);
    float ht = hal_mx_hue(tint);
    h = h + (ht - h) * (strength / 100.0);
    vec3 rgb = hal_mx_hsl2rgb(h, s, lum);
    if (advanced == 0) {
        float b = brightness / 100.0;
        float k = contrast / 100.0;
        rgb = (rgb - 0.5) * (1.0 + k) + 0.5 + b;
    } else {
        float gm = max(gamma, 1e-6);
        float pv = (pivot == 0.0) ? 1e-6 : pivot;
        vec3 base = max(((rgb * gain) / 100.0) / pv, vec3(0.0));
        float e = 1.0 / gm;
        rgb = pv * vec3(pow(base.x, e), pow(base.y, e), pow(base.z, e)) + lift;
    }
    return vec4(rgb, c.a);
}
""",
    'mx_composite': """
vec4 hal_mx_hsl_blend(vec3 hsrc, vec3 ssrc, vec3 lsrc, vec4 fg)
{
    return vec4(hal_mx_hsl2rgb(hal_mx_hue(hsrc), hal_mx_sat(ssrc), hal_mx_lum(lsrc)), fg.a);
}

vec4 hal_mx_blend(int m, vec4 fg, vec4 bg)
{
    if (m == 0) return fg;
    if (m == 1) return (fg + bg) / 2.0;
    if (m == 2) return fg + bg;
    if (m == 3) return bg - fg;
    if (m == 4) return min(fg, bg);
    if (m == 5) return fg * bg;
    if (m == 6) {
        vec4 r = vec4(0.0);
        if (fg.r != 0.0) r.r = max(1.0 - (1.0 - bg.r) / fg.r, 0.0);
        if (fg.g != 0.0) r.g = max(1.0 - (1.0 - bg.g) / fg.g, 0.0);
        if (fg.b != 0.0) r.b = max(1.0 - (1.0 - bg.b) / fg.b, 0.0);
        if (fg.a != 0.0) r.a = max(1.0 - (1.0 - bg.a) / fg.a, 0.0);
        return r;
    }
    if (m == 7) return max(fg + bg - 1.0, vec4(0.0));
    if (m == 8) return max(fg, bg);
    if (m == 9) return fg + bg - fg * bg;
    if (m == 10) {
        vec4 r = vec4(1.0);
        if (fg.r != 1.0) r.r = min(bg.r / (1.0 - fg.r), 1.0);
        if (fg.g != 1.0) r.g = min(bg.g / (1.0 - fg.g), 1.0);
        if (fg.b != 1.0) r.b = min(bg.b / (1.0 - fg.b), 1.0);
        if (fg.a != 1.0) r.a = min(bg.a / (1.0 - fg.a), 1.0);
        return r;
    }
    if (m == 11) return min(fg + bg, vec4(1.0));
    if (m == 12) return min(2.0 * fg * bg, vec4(1.0));
    if (m == 13) return min(fg * bg + bg, vec4(1.0));
    if (m == 14) {
        vec4 r = 2.0 * fg * bg;
        if (bg.r > 0.5) r.r = 1.0 - 2.0 * (1.0 - fg.r) * (1.0 - bg.r);
        if (bg.g > 0.5) r.g = 1.0 - 2.0 * (1.0 - fg.g) * (1.0 - bg.g);
        if (bg.b > 0.5) r.b = 1.0 - 2.0 * (1.0 - fg.b) * (1.0 - bg.b);
        if (bg.a > 0.5) r.a = 1.0 - 2.0 * (1.0 - fg.a) * (1.0 - bg.a);
        return clamp(r, 0.0, 1.0);
    }
    if (m == 15) {
        vec4 r = bg * (bg + 2.0 * fg * (1.0 - bg));
        if (fg.r > 0.5) r.r = bg.r + (2.0 * fg.r - 1.0) * sqrt(max(bg.r, 0.0)) - bg.r;
        if (fg.g > 0.5) r.g = bg.g + (2.0 * fg.g - 1.0) * sqrt(max(bg.g, 0.0)) - bg.g;
        if (fg.b > 0.5) r.b = bg.b + (2.0 * fg.b - 1.0) * sqrt(max(bg.b, 0.0)) - bg.b;
        if (fg.a > 0.5) r.a = bg.a + (2.0 * fg.a - 1.0) * sqrt(max(bg.a, 0.0)) - bg.a;
        return clamp(r, 0.0, 1.0);
    }
    if (m == 16) {
        vec4 r = 2.0 * fg * bg;
        if (fg.r > 0.5) r.r = 1.0 - 2.0 * (1.0 - fg.r) * (1.0 - bg.r);
        if (fg.g > 0.5) r.g = 1.0 - 2.0 * (1.0 - fg.g) * (1.0 - bg.g);
        if (fg.b > 0.5) r.b = 1.0 - 2.0 * (1.0 - fg.b) * (1.0 - bg.b);
        if (fg.a > 0.5) r.a = 1.0 - 2.0 * (1.0 - fg.a) * (1.0 - bg.a);
        return clamp(r, 0.0, 1.0);
    }
    if (m == 17) {
        vec4 r = bg;
        if ((fg.r > 0.5 && fg.r > bg.r) || (fg.r < 0.5 && fg.r < bg.r)) r.r = fg.r;
        if ((fg.g > 0.5 && fg.g > bg.g) || (fg.g < 0.5 && fg.g < bg.g)) r.g = fg.g;
        if ((fg.b > 0.5 && fg.b > bg.b) || (fg.b < 0.5 && fg.b < bg.b)) r.b = fg.b;
        if ((fg.a > 0.5 && fg.a > bg.a) || (fg.a < 0.5 && fg.a < bg.a)) r.a = fg.a;
        return r;
    }
    if (m == 18) {
        vec4 r = vec4(1.0);
        if (fg.r + bg.r <= 1.0) r.r = 0.0;
        if (fg.g + bg.g <= 1.0) r.g = 0.0;
        if (fg.b + bg.b <= 1.0) r.b = 0.0;
        if (fg.a + bg.a <= 1.0) r.a = 0.0;
        return r;
    }
    if (m == 19) return abs(fg - bg);
    if (m == 20) return fg + bg - 2.0 * fg * bg;
    if (m == 21) return hal_mx_hsl_blend(fg.rgb, bg.rgb, bg.rgb, fg);
    if (m == 22) return hal_mx_hsl_blend(bg.rgb, fg.rgb, bg.rgb, fg);
    if (m == 23) return hal_mx_hsl_blend(fg.rgb, fg.rgb, bg.rgb, fg);
    if (m == 24) return hal_mx_hsl_blend(bg.rgb, bg.rgb, fg.rgb, fg);
    return fg;
}

vec4 hal_mx_comp_layer(vec4 res, vec4 fg, float opacity, float mask, int m)
{
    float a = fg.a;
    if (a != 1.0 && a != 0.0) fg.rgb = fg.rgb / a;
    float fa = a * mask;
    fa = fa * (opacity / 100.0);
    fg.a = fa;
    float ra = res.a;
    if (ra == 0.0) return fg;
    vec4 bl = hal_mx_blend(m, fg, res);
    float alpha = fa + (1.0 - fa) * ra;
    float safe = (alpha != 0.0) ? alpha : 1.0;
    vec3 rgb = (bl.rgb * (fa * ra) + fg.rgb * (fa * (1.0 - ra))
                + res.rgb * ((1.0 - fa) * ra)) / safe;
    return vec4(rgb, alpha);
}

vec4 hal_mx_comp_finish(vec4 res)
{
    if (res.a != 1.0) res.rgb = res.rgb * res.a;
    return res;
}
""",
    'mx_fresnel': """
float hal_mx_fresnel(float ndv, float ior)
{
    float c = clamp(abs(ndv), 0.0, 1.0);
    float g2 = ior * ior + c * c - 1.0;
    if (g2 < 0.0) return 1.0;
    float g = sqrt(max(g2, 0.0));
    float gc = max(g + c, 1e-6);
    float t = (c * gc - 1.0) / (c * gc + 1.0);
    float f = ((g - c) * (g - c)) / (2.0 * gc * gc) * (1.0 + t * t);
    return clamp(f, 0.0, 1.0);
}
""",
}

for _k, _v in MAX_GLSL.items():
    PATTERN_GLSL[_k] = _v
del _k, _v
