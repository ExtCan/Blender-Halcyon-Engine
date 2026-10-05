"""R251 C001: the N64 Video Interface's coverage half as two resident post
stages -- the GLSL twins of core/n64vi.py's `vi_aa` and `vi_divot`.

Integer arithmetic on the frame's own 8-bit values (the 5551 expansion
`(c5 << 3) | (c5 >> 2)` for the AA pass, `floor(c * 255 + 0.5)` for the
divot, which reads the AA pass's `k / 255` texels back as the integers
they encode); every fetch is a `texelFetch` with integer coordinates,
the six-neighbour walk is guarded with `continue` at the frame edge and
the divot's missing neighbour IS the centre (the simulator CLAMPS an
out-of-range fetch where a driver returns zero, so a frame-edge rule
must be explicit). The two-largest / two-smallest update is the standard
order-free multiset selection, so ties land identically on both roads.
Held bitwise (0.0) against the CPU in the simulator; the `/ 255.0` output
is the one non-integer op and encodes an integer exactly on both roads.

Registered into gpu/stages.py's STAGES / INTERFACE / VALIDATION by the
integrator's three `update` lines; `set(STAGES_VI) <= set(stages.ENABLED)`.
"""

N64VI_AA = """
uniform sampler2D source;
uniform sampler2D cvg;
uniform vec2 resolution;
in vec2 vUV;
out vec4 Color;

// the RGBA5551 framebuffer as the VI fetches it: 5 bits, expanded by
// bit replication (core/n64vi.quant5: two float32 ops, then floor)
ivec3 hal_c8(ivec2 p)
{
    vec4 t = texelFetch(source, p, 0);
    vec3 v = clamp(t.rgb, 0.0, 1.0);
    v = v * 31.0;
    v = v + 0.5;
    ivec3 c5 = ivec3(floor(v));
    return (c5 << 3) | (c5 >> 2);
}

int hal_cvg(ivec2 p)
{
    return int(texelFetch(cvg, p, 0).x + 0.5);
}

void main()
{
    ivec2 px = ivec2(clamp(vUV * resolution, vec2(0.0), resolution - vec2(1.0)));
    ivec3 c = hal_c8(px);
    int cv = hal_cvg(px);
    vec4 t = texelFetch(source, px, 0);
    if (cv >= 7) {
        Color = vec4(vec3(c) / 255.0, t.a);
        return;
    }
    ivec3 max1 = ivec3(-1);
    ivec3 max2 = ivec3(-1);
    ivec3 min1 = ivec3(1000);
    ivec3 min2 = ivec3(1000);
    for (int k = 0; k < 6; k++) {
        ivec2 d = (k == 0) ? ivec2(-1, -1) : (k == 1) ? ivec2(1, -1)
                : (k == 2) ? ivec2(-1, 0) : (k == 3) ? ivec2(1, 0)
                : (k == 4) ? ivec2(-1, 1) : ivec2(1, 1);
        ivec2 q = px + d;
        if (q.x < 0 || q.y < 0 || float(q.x) >= resolution.x
                || float(q.y) >= resolution.y) { continue; }
        if (hal_cvg(q) != 7) { continue; }
        ivec3 v = hal_c8(q);
        // per channel, the running two largest and two smallest
        // (order-free: a multiset selection)
        ivec3 hi = max(v, max1);
        ivec3 lo = min(v, max1);
        max2 = max(max2, lo);
        max1 = hi;
        ivec3 lo2 = min(v, min1);
        ivec3 hi2 = max(v, min1);
        min2 = min(min2, hi2);
        min1 = lo2;
    }
    // angrylion's list holds the centre first: the penultimate of
    // {c, full neighbours} is max(c, second largest neighbour)
    ivec3 penmax = max(c, max2);
    ivec3 penmin = min(c, min2);
    int coeff = 7 - cv;
    ivec3 dd = ((penmin + penmax) - 2 * c) * coeff;
    dd = dd + 4;
    dd = dd >> 3;
    ivec3 o = (c + dd) & 255;
    Color = vec4(vec3(o) / 255.0, t.a);
}
"""

N64VI_DIVOT = """
uniform sampler2D source;
uniform sampler2D cvg;
uniform vec2 resolution;
in vec2 vUV;
out vec4 Color;

// the AA pass's texels are the integers o / 255.0: decoded back to 8
// bits, never re-quantised to 5 (core/n64vi.to8)
ivec3 hal_c8(ivec2 p)
{
    vec4 t = texelFetch(source, p, 0);
    vec3 v = clamp(t.rgb, 0.0, 1.0);
    v = v * 255.0;
    v = v + 0.5;
    return ivec3(floor(v));
}

int hal_cvg(ivec2 p)
{
    return int(texelFetch(cvg, p, 0).x + 0.5);
}

void main()
{
    ivec2 px = ivec2(clamp(vUV * resolution, vec2(0.0), resolution - vec2(1.0)));
    ivec3 c = hal_c8(px);
    vec4 t = texelFetch(source, px, 0);
    // the frame's x-edges: the missing neighbour IS the centre (coverage
    // 7), so the median is the identity there -- the CPU's own rule
    bool hasl = px.x > 0;
    bool hasr = float(px.x) < resolution.x - 1.0;
    ivec3 l = hasl ? hal_c8(px - ivec2(1, 0)) : c;
    ivec3 r = hasr ? hal_c8(px + ivec2(1, 0)) : c;
    int la = hasl ? hal_cvg(px - ivec2(1, 0)) : 7;
    int ra = hasr ? hal_cvg(px + ivec2(1, 0)) : 7;
    bool trip = ((la & hal_cvg(px)) & ra) != 7;
    ivec3 med = max(min(l, c), min(max(l, c), r));
    ivec3 o = trip ? med : c;
    Color = vec4(vec3(o) / 255.0, t.a);
}
"""

STAGES_VI = {
    'N64VI_AA': N64VI_AA,
    'N64VI_DIVOT': N64VI_DIVOT,
}

INTERFACE_VI = {
    'N64VI_AA': {'samplers': ['source', 'cvg'], 'vec2': ['resolution']},
    'N64VI_DIVOT': {'samplers': ['source', 'cvg'], 'vec2': ['resolution']},
}

# integer arithmetic and table-free: bitwise the CPU in the simulator;
# the 5-bit quantise is floor(t) after t = c * 31.0; t = t + 0.5 (the
# CPU's own two ops), and the output division encodes an integer
VALIDATION_VI = {
    'N64VI_AA': ('EXACT', 0.00001),
    'N64VI_DIVOT': ('EXACT', 0.00001),
}
