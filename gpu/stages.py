"""The post chain as GLSL.

Each stage is a full-screen fragment shader taking `source` and writing a
colour. They are deliberately written in the subset that both Blender's
GPUShader and Halcyon's own GLSL front-end accept, because that is what makes
them testable: the same source is compiled by `halcyon.shaders` and run through
NumPy, and the result compared against the CPU implementation in `core/post.py`.

That check is the only reason to trust this code. It was written on a machine
with no GPU, so nothing here has been executed by a driver -- but the *logic*
has been executed, and shown to agree with the CPU path it replaces.

Only the parallel stages are here. Error diffusion is sequential by nature and
stays on the CPU, where the wavefront schedule already handles it.
"""

DISPLAY = """
uniform sampler2D source;
uniform float exposure;
uniform float brightness;
uniform float contrast;
uniform float saturation;
uniform float gamma;
uniform int cm_mode;
in vec2 vUV;
out vec4 Color;
void main()
{
    vec4 texel = texture(source, vUV);
    vec3 c = texel.rgb * exposure;
    // the view-transform curve, exactly core/post.display_transform:
    // 1 FILMIC, 2 REINHARD, 3 SRGB (the piecewise OETF -- 2.79's
    // 'Default' view). 0 is the period-correct no-op. Before this
    // uniform the GPU stage silently SKIPPED the curve whenever
    // color_management was set: a live CPU/GPU divergence.
    if (cm_mode == 1) { c = c / (c + vec3(0.6)); }
    if (cm_mode == 2) { c = c / (vec3(1.0) + c); }
    if (cm_mode == 3) {
        c = clamp(c, vec3(0.0), vec3(1.0));
        vec3 hi = 1.055 * pow(max(c, vec3(0.0)), vec3(1.0 / 2.4))
                  - vec3(0.055);
        vec3 lo = c * 12.92;
        c = mix(hi, lo, step(c, vec3(0.0031308)));
    }
    c = c + vec3(brightness);
    c = (c - vec3(0.5)) * (1.0 + contrast) + vec3(0.5);
    c = max(c, vec3(0.0));
    float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
    c = vec3(l) + (c - vec3(l)) * saturation;
    c = pow(max(c, vec3(0.0)), vec3(1.0 / gamma));
    Color = vec4(clamp(c, 0.0, 1.0), texel.a);
}
"""

# Barrel or pincushion, with the channels displaced by different amounts.
LENS = """
uniform sampler2D source;
uniform float distortion;
uniform float aberration;
uniform float edges;
uniform vec2 resolution;
in vec2 vUV;
out vec4 Color;

vec2 warp(vec2 uv, float k)
{
    // work in pixel indices, as the CPU path does: normalising over width - 1
    // rather than over the 0..1 texture range, or the two disagree by half a
    // texel and the difference grows with the distortion
    vec2 span = max(resolution - vec2(1.0), vec2(1.0));
    vec2 p = uv * resolution - vec2(0.5);
    vec2 n = (p / span) * 2.0 - vec2(1.0);
    float r2 = dot(n, n);
    vec2 s = n * (1.0 + k * r2);
    vec2 sp = (s * 0.5 + vec2(0.5)) * span;
    // clamp, do not let the sampler wrap: distortion pushes coordinates past
    // the edge of the frame, and a repeating sampler brings the far side of
    // the picture back round into the corners
    return clamp((sp + vec2(0.5)) / resolution, vec2(0.0), vec2(1.0));
}

void main()
{
    float ca = aberration * 0.02;
    vec2 ur = warp(vUV, distortion + ca);
    vec2 ug = warp(vUV, distortion);
    vec2 ub = warp(vUV, distortion - ca);
    vec3 c = vec3(texture(source, ur).r,
                  texture(source, ug).g,
                  texture(source, ub).b);
    float inside = 1.0;
    if (edges > 0.5) {
        vec2 raw = vUV * 2.0 - vec2(1.0);
        float rr = dot(raw, raw);
        vec2 s = raw * (1.0 + distortion * rr);
        if (abs(s.x) > 1.0 || abs(s.y) > 1.0) { inside = 0.0; }
    }
    Color = vec4(c * inside, texture(source, ug).a);
}
"""

# Scanlines, phosphor mask and vignette. Curvature is a coordinate warp and is
# folded into LENS, which runs first.
CRT = """
uniform sampler2D source;
uniform float scanlines;
uniform float mask_strength;
uniform int mask_kind;
uniform float vignette;
uniform vec2 resolution;
in vec2 vUV;
out vec4 Color;
void main()
{
    vec4 texel = texture(source, vUV);
    vec3 c = texel.rgb;
    vec2 px = vUV * resolution;

    if (mask_strength > 0.0 && mask_kind > 0) {
        float col = floor(px.x);
        float row = floor(px.y);
        float shift = 0.0;
        if (mask_kind == 2) { shift = floor(mod(floor(row * 0.5), 2.0)); }
        if (mask_kind == 3) { shift = mod(row, 2.0) * 2.0; }
        float idx = mod(col + shift, 3.0);
        vec3 m = vec3(1.0 - mask_strength);
        if (idx < 0.5) { m.r = 1.0; }
        else if (idx < 1.5) { m.g = 1.0; }
        else { m.b = 1.0; }
        c = c * m;
    }

    if (scanlines > 0.0) {
        float s = 1.0 - scanlines * (0.5 + 0.5 * cos(floor(px.y) * 3.14159265));
        c = c * clamp(s, 0.0, 1.0);
    }

    if (vignette > 0.0) {
        vec2 n = vUV * 2.0 - vec2(1.0);
        c = c * clamp(1.0 - dot(n, n) * vignette * 0.5, 0.0, 1.0);
    }
    Color = vec4(c, texel.a);
}
"""

# Composite chroma bleed: separable, so the horizontal blur is a fixed tap set.
# The composite cable, structured as the CPU structures it: the chroma is
# blurred by a box blur RUN THREE TIMES (a triple box is the CPU's fast
# Gaussian), I and Q at DIFFERENT radii -- Q got about half the bandwidth I
# did -- and Y sharpened against a radius-2 blur of itself for ringing. One
# shader cannot reproduce three passes exactly, because the CPU re-pads the
# edges before every pass; so it is three draws of NTSC_BLUR and one of NTSC,
# orchestrated by chain.ntsc(), and each draw's edge clamp matches np.pad
# exactly. The old single-pass version blurred both chroma channels with one
# 13-tap triangle, which is why it never validated.

NTSC_BLUR = """
uniform sampler2D source;
uniform float ri;
uniform float rq;
uniform float ry;
uniform float to_yiq;
uniform vec2 resolution;
in vec2 vUV;
out vec4 Color;

vec3 rgb2yiq(vec3 c)
{
    return vec3(dot(c, vec3(0.299, 0.587, 0.114)),
                dot(c, vec3(0.5959, -0.2746, -0.3213)),
                dot(c, vec3(0.2115, -0.5227, 0.3112)));
}

vec3 fetch(vec2 uv)
{
    vec3 c = texture(source, uv).rgb;
    return (to_yiq > 0.5) ? rgb2yiq(c) : c;
}

void main()
{
    float step_u = 1.0 / resolution.x;
    float lo = 0.5 * step_u;
    float hi = 1.0 - 0.5 * step_u;
    float rmax = max(max(ri, rq), ry);
    vec3 acc = vec3(0.0);
    vec3 total = vec3(0.0);
    for (int i = -96; i <= 96; i++) {
        float k = abs(float(i));
        if (k <= rmax) {
            vec2 p = vec2(clamp(vUV.x + float(i) * step_u, lo, hi), vUV.y);
            vec3 v = fetch(p);
            vec3 w = vec3(k <= ry ? 1.0 : 0.0,
                          k <= ri ? 1.0 : 0.0,
                          k <= rq ? 1.0 : 0.0);
            acc = acc + v * w;
            total = total + w;
        }
    }
    Color = vec4(acc / max(total, vec3(0.0001)), 1.0);
}
"""

NTSC = """
uniform sampler2D source;
uniform sampler2D blurred;
uniform float ringing;
in vec2 vUV;
out vec4 Color;

vec3 rgb2yiq(vec3 c)
{
    return vec3(dot(c, vec3(0.299, 0.587, 0.114)),
                dot(c, vec3(0.5959, -0.2746, -0.3213)),
                dot(c, vec3(0.2115, -0.5227, 0.3112)));
}

vec3 yiq2rgb(vec3 c)
{
    return vec3(c.x + 0.956 * c.y + 0.619 * c.z,
                c.x - 0.272 * c.y - 0.647 * c.z,
                c.x - 1.106 * c.y + 1.703 * c.z);
}

void main()
{
    vec3 centre = rgb2yiq(texture(source, vUV).rgb);
    vec3 soft = texture(blurred, vUV).rgb;      // (Y blurred, I blurred, Q blurred)
    vec3 outc = vec3(centre.x + (centre.x - soft.x) * ringing * 2.0,
                     soft.y, soft.z);
    Color = vec4(clamp(yiq2rgb(outc), 0.0, 1.0), texture(source, vUV).a);
}
"""

# R250: the film's grain on the GPU -- wear.grain_plain for grains at or
# under 1.2 px (the white sheets: two hashes per pixel sliced into one
# triangular and three uniform unit-variance sheets, no blur), the
# frame's transmittance read as a print's density, the density noise
# strongest where half the grains developed, each channel its own sheet
# by Chroma; the projector's flicker factor last, as film.process_linear
# orders them. Larger grains and clumps blur the sheets on the CPU and
# refuse by name. The hash is the ink's uint hash, masked to 24 bits as
# film._hash_u32_raw masks it; log10 is the one library call (CLOSE).
GRAIN = """
uniform sampler2D source;
uniform vec2 resolution;
uniform int key_a;
uniform int key_b;
uniform int chroma_mode;
uniform float mix_a;
uniform float mix_b;
uniform float amp;
uniform float tri_scale;
uniform float tri_off;
uniform float u_scale;
uniform float u_off;
uniform float a_coef;
uniform float ln10;
uniform float flicker;
in vec2 vUV;
out vec4 Color;

uint hal_grain_hash(uint x, uint y, int k)
{
    uint h = x * 0x9E3779B1u ^ (y + 0x85EBCA77u) ^ (uint(k) * 0xC2B2AE3Du);
    h ^= h >> 15u;
    h *= 0x2C1B3C6Du;
    h ^= h >> 12u;
    h *= 0x297A2D39u;
    h ^= h >> 15u;
    return h & 0xffffffu;
}

// wear.grain_plain, one channel: the transmittance's density, the noise
// on it, mean-preserving
float hal_grain_ch(float x, float sh)
{
    float t = max(x, 1e-4);
    float a = log2(t) * 0.30102999566398120;
    a = a * a_coef;
    a = clamp(a, 0.0, 1.0);
    float sig = sqrt(clamp(a * (1.0 - a), 0.0, 1.0)) * amp;
    sig = sig * sh;
    sig = sig * ln10;
    sig = sig + 1.0;
    return x * sig;
}

void main()
{
    ivec2 px = ivec2(clamp(vUV * resolution, vec2(0.0), resolution - vec2(1.0)));
    vec4 texel = texelFetch(source, px, 0);
    uint ha = hal_grain_hash(uint(px.x), uint(px.y), key_a);
    uint hb = hal_grain_hash(uint(px.x), uint(px.y), key_b);
    // the triangular sheet: two 12-bit uniforms summed, scaled to unit variance
    float tri = float(ha & 0xfffu);
    tri += float((ha >> 12u) & 0xfffu);
    tri *= tri_scale;
    tri -= tri_off;
    // three uniform sheets, 8 bits each
    float u0 = float(hb & 0xffu);
    u0 *= u_scale;
    u0 -= u_off;
    float u1 = float((hb >> 8u) & 0xffu);
    u1 *= u_scale;
    u1 -= u_off;
    float u2 = float((hb >> 16u) & 0xffu);
    u2 *= u_scale;
    u2 -= u_off;
    vec3 sheet = vec3(tri);
    if (chroma_mode == 1) {
        sheet = vec3(u0, u1, u2);
    } else if (chroma_mode == 2) {
        sheet = vec3(u0 * mix_b + tri * mix_a, u1 * mix_b + tri * mix_a,
                     u2 * mix_b + tri * mix_a);
    }
    vec3 outc = vec3(hal_grain_ch(texel.r, sheet.x),
                     hal_grain_ch(texel.g, sheet.y),
                     hal_grain_ch(texel.b, sheet.z));
    outc = max(outc, vec3(0.0));
    Color = vec4(outc * flicker, texel.a);
}
"""

# R250: the framebuffer's bit depth without a dither -- palette.snap_bits:
# clip, scale to the channel's levels, round half to even (NumPy's
# round), back. roundEven is the CPU's tie rule exactly.
QUANT = """
uniform sampler2D source;
uniform sampler2D lut;
uniform vec2 resolution;
uniform vec3 levels;
in vec2 vUV;
out vec4 Color;
void main()
{
    ivec2 px = ivec2(clamp(vUV * resolution, vec2(0.0), resolution - vec2(1.0)));
    vec4 texel = texelFetch(source, px, 0);
    // the level index is one multiply and one round-half-even (both
    // exact everywhere); the level's VALUE is fetched from the CPU's own
    // k / levels table, so the driver's division never enters
    vec3 idx = roundEven(clamp(texel.rgb, 0.0, 1.0) * levels);
    float r = texelFetch(lut, ivec2(int(idx.r), 0), 0).r;
    float g = texelFetch(lut, ivec2(int(idx.g), 0), 0).g;
    float b = texelFetch(lut, ivec2(int(idx.b), 0), 0).b;
    Color = vec4(r, g, b, texel.a);
}
"""

# R250: the supersample resolve -- render._resolve's filter over each
# output pixel's ss x ss block, the taps summed row by row in the order
# the CPU's einsum sums them (row i outer, column j inner; measured
# bitwise for every filter and factor). The kernel rides a texture.
RESOLVE = """
uniform sampler2D source;
uniform sampler2D kernel;
uniform vec2 resolution;
uniform int ss;
in vec2 vUV;
out vec4 Color;
void main()
{
    ivec2 px = ivec2(clamp(vUV * resolution, vec2(0.0), resolution - vec2(1.0)));
    vec4 acc = vec4(0.0);
    for (int i = 0; i < 8; i++) {
        if (i >= ss) { break; }
        for (int j = 0; j < 8; j++) {
            if (j >= ss) { break; }
            vec4 t = texelFetch(source, ivec2(px.x * ss + j, px.y * ss + i), 0);
            float k = texelFetch(kernel, ivec2(j, i), 0).x;
            acc = acc + t * k;
        }
    }
    Color = acc;
}
"""


# R251 (RAST-B, 1.90.0): the two resolve variants -- C094 LightWave's
# Limit Dynamic Range (a per-tap `min` before the same fixed-order sum)
# and C122 Blender 2.41's gamma-2 OSA blend (each tap squared through the
# CPU's own 400-entry table, the sum square-rooted through the inverse
# table). NOT an edit to the pinned RESOLVE stage: `resolve_source` with
# both flags off returns RESOLVE byte-identical, and the variants are
# compiled through `device.compile_dynamic` from gpu/frame.resolve with
# `resolve_spec`'s interface, so STAGES / INTERFACE / VALIDATION and the
# self test's stage table never see them. `vec3(float(ii.r), ...) * 0.0025`
# is the same float32 multiply as raster.gamma2_tables' `dom` (the
# per-component float() matters: the simulator's vec3(ivec3) keeps the
# integers and a later float op promotes to float64 -- a front-end blind
# spot this pack's twin test found), the fetches return the
# CPU's table bits and the accumulation order is RESOLVE's, so both
# variants are EXACT in the simulator (tests/test_r251_raster_wire.py).
RESOLVE_CLAMP = """
            if (clamp_samples > 0.5) { t.rgb = min(t.rgb, vec3(1.0)); }
"""

RESOLVE_GAMMA2 = """
            if (gamma_blend > 0.5) {
                vec3 c = clamp(t.rgb, 0.0, 1.0);
                ivec3 ii = ivec3(floor(c * 400.0));
                vec3 dom = vec3(float(ii.r), float(ii.g), float(ii.b)) * 0.0025;
                vec4 tr = texelFetch(gtab, ivec2(ii.r, 0), 0);
                vec4 tg = texelFetch(gtab, ivec2(ii.g, 0), 0);
                vec4 tb = texelFetch(gtab, ivec2(ii.b, 0), 0);
                vec3 d = c - dom;
                d = d * vec3(tr.y, tg.y, tb.y);
                t.rgb = vec3(tr.x, tg.x, tb.x) + d;
            }
"""

RESOLVE_GAMMA2_INVERSE = """
    if (gamma_blend > 0.5) {
        vec3 a = clamp(acc.rgb, 0.0, 1.0);
        ivec3 ii = ivec3(floor(a * 400.0));
        vec3 dom = vec3(float(ii.r), float(ii.g), float(ii.b)) * 0.0025;
        vec4 tr = texelFetch(gtab, ivec2(ii.r, 0), 0);
        vec4 tg = texelFetch(gtab, ivec2(ii.g, 0), 0);
        vec4 tb = texelFetch(gtab, ivec2(ii.b, 0), 0);
        vec3 d = a - dom;
        d = d * vec3(tr.w, tg.w, tb.w);
        acc.rgb = vec3(tr.z, tg.z, tb.z) + d;
    }
"""

_RESOLVE_DECL = 'uniform int ss;\n'
_RESOLVE_TAP = ('            vec4 t = texelFetch(source, ivec2(px.x * ss + j, '
                'px.y * ss + i), 0);\n')
_RESOLVE_OUT = '    Color = acc;\n'


def resolve_flags(st):
    """(clamp, gamma): the two resolve dials as read from the settings."""
    return (bool(getattr(st, 'aa_clamp_samples', False)),
            bool(getattr(st, 'aa_gamma_blend', False)))


def resolve_variant_name(clamp=False, gamma=False):
    """'RESOLVE' when both are off, else RESOLVE_C / RESOLVE_G / RESOLVE_CG."""
    if not clamp and not gamma:
        return 'RESOLVE'
    return 'RESOLVE_' + ('C' if clamp else '') + ('G' if gamma else '')


def resolve_source(clamp=False, gamma=False):
    """The RESOLVE stage's source with the C094 clamp and/or the C122
    gamma blend spliced in: with both flags off, RESOLVE byte-identical."""
    src = RESOLVE
    if not clamp and not gamma:
        return src
    decl = ''
    if clamp:
        decl += 'uniform float clamp_samples;\n'
    if gamma:
        decl += 'uniform float gamma_blend;\nuniform sampler2D gtab;\n'
    assert src.count(_RESOLVE_DECL) == 1
    src = src.replace(_RESOLVE_DECL, _RESOLVE_DECL + decl, 1)
    ins = ''
    if clamp:
        ins += RESOLVE_CLAMP
    if gamma:
        ins += RESOLVE_GAMMA2
    assert src.count(_RESOLVE_TAP) == 1
    src = src.replace(_RESOLVE_TAP, _RESOLVE_TAP + ins, 1)
    if gamma:
        assert src.count(_RESOLVE_OUT) == 1
        src = src.replace(_RESOLVE_OUT, RESOLVE_GAMMA2_INVERSE + _RESOLVE_OUT, 1)
    return src


def resolve_spec(clamp=False, gamma=False):
    """INTERFACE['RESOLVE'] plus the variant's uniforms: `clamp_samples`
    under clamp, `gamma_blend` and the `gtab` sampler under gamma -- a
    declared sampler must be bound on every draw, so the table exists
    only in the _G / _CG variants."""
    spec = {'samplers': [], 'floats': [], 'ints': [], 'vec2': [], 'vec3': []}
    spec.update({k: list(v) for k, v in INTERFACE['RESOLVE'].items()})
    if clamp:
        spec['floats'].append('clamp_samples')
    if gamma:
        spec['floats'].append('gamma_blend')
        spec['samplers'].append('gtab')
    return spec


# Vulkan has no legacy GPUShader(vertex, fragment) constructor: shaders are
# built from a GPUShaderCreateInfo, which carries the interface itself and
# wants the GLSL *without* its declarations. One spec per stage, so the
# declarations and the CreateInfo cannot disagree.
INTERFACE = {
    'DISPLAY': {'samplers': ['source'],
                'floats': ['exposure', 'brightness', 'contrast',
                           'saturation', 'gamma'],
                'ints': ['cm_mode']},
    'LENS': {'samplers': ['source'],
             'floats': ['distortion', 'aberration', 'edges'],
             'vec2': ['resolution']},
    'CRT': {'samplers': ['source'],
            'floats': ['scanlines', 'mask_strength', 'vignette'],
            'ints': ['mask_kind'], 'vec2': ['resolution']},
    'NTSC_BLUR': {'samplers': ['source'],
                  'floats': ['ri', 'rq', 'ry', 'to_yiq'],
                  'vec2': ['resolution']},
    'NTSC': {'samplers': ['source', 'blurred'],
             'floats': ['ringing']},
    'GRAIN': {'samplers': ['source'],
              'floats': ['mix_a', 'mix_b', 'amp', 'tri_scale', 'tri_off',
                         'u_scale', 'u_off', 'a_coef', 'ln10', 'flicker'],
              'ints': ['key_a', 'key_b', 'chroma_mode'],
              'vec2': ['resolution']},
    'QUANT': {'samplers': ['source', 'lut'], 'vec2': ['resolution'],
              'vec3': ['levels']},
    'RESOLVE': {'samplers': ['source', 'kernel'], 'ints': ['ss'],
                'vec2': ['resolution']},
}


def body(name):
    """The stage source with its declarations removed, for a CreateInfo build.

    Delegates to the one stripper, because the two used to disagree: this one
    also required the semicolon to end the line, and a declaration with a
    trailing comment survived into a source whose CreateInfo already declared
    it -- a redeclaration the driver refuses and our own front-end does not.
    """
    from .device import strip_declarations
    return strip_declarations(STAGES[name])


STAGES = {
    'DISPLAY': DISPLAY,
    'LENS': LENS,
    'CRT': CRT,
    'NTSC_BLUR': NTSC_BLUR,
    'NTSC': NTSC,
    'GRAIN': GRAIN,
    'QUANT': QUANT,
    'RESOLVE': RESOLVE,
}

# How far each stage has been shown to agree with the CPU function it replaces,
# by compiling it with Halcyon's own GLSL front-end and running it through
# NumPy. Nothing here has been executed by a real driver.
#
#   EXACT     bit-identical to the CPU path on the test image
#   CLOSE     agrees within the stated tolerance; the difference is a known
#             formulation detail, not an error
#   UNPROVEN  a real disagreement remains -- not enabled
# Measured on an RTX 5060 Ti under Vulkan, not only against the NumPy backend.
# Where the two disagree the hardware number wins, because that is the one that
# reaches the screen.
VALIDATION = {
    'DISPLAY': ('EXACT', 0.0001),    # 0.00001 measured on hardware at 32F
    'CRT': ('CLOSE', 0.03),          # 0.0113 measured
    'LENS': ('CLOSE', 0.01),         # 0.00426 measured after the half-texel fix
    'NTSC': ('CLOSE', 0.001),        # 0.00037 measured on an RTX 5060 Ti
                                     # under Vulkan, run as its real shape:
                                     # three blur draws and a combine, I and
                                     # Q at their own radii. Two rounds of
                                     # self test bought this line
    'NTSC_BLUR': ('CLOSE', 0.001),   # measured as part of the NTSC pipeline;
                                     # never drawn on its own
    # R250: QUANT and RESOLVE are bitwise the CPU in the simulator
    # (roundEven on the CPU's own level LUT; the resolve's summation
    # order). GRAIN is one float32 ulp off in the simulator (1.2e-7):
    # `log2(t) * 0.30102999566398120` is not `np.log10(t)` at 5133 of
    # 9216 float32 inputs, the one library call the stage makes. The
    # tolerance is the driver's (FMA contraction, its log2 / division);
    # the field test measured 0.00000 on an RTX 5060 Ti under Vulkan
    'GRAIN': ('CLOSE', 0.0005),      # log10 through log2: the one library call
    'QUANT': ('EXACT', 0.00001),
    'RESOLVE': ('EXACT', 0.00001),
}

# R251: the post-palette pack's era colour stages (PALETTE, ORDERED, EHB,
# CRY16, YJK), merged before ENABLED is derived
from . import stages_palette as _SP  # noqa: E402
STAGES.update(_SP.STAGES)
INTERFACE.update(_SP.INTERFACE)
VALIDATION.update(_SP.VALIDATION)

# R251: the post-signal pack's stages (gpu/stages_signal.py)
from .stages_signal import STAGES_SIGNAL, INTERFACE_SIGNAL, VALIDATION_SIGNAL  # noqa: E402
STAGES.update(STAGES_SIGNAL); INTERFACE.update(INTERFACE_SIGNAL); VALIDATION.update(VALIDATION_SIGNAL)
# R251 SIG-2: the tape / cable / PAL / chroma-siting stages (gpu/stages_tape.py)
from .stages_tape import STAGES_TAPE, INTERFACE_TAPE, VALIDATION_TAPE  # noqa: E402
STAGES.update(STAGES_TAPE); INTERFACE.update(INTERFACE_TAPE); VALIDATION.update(VALIDATION_TAPE)

# R251 SIG-3: the codec / optical-printer stages (gpu/stages_codec.py)
from .stages_codec import STAGES_CODEC, INTERFACE_CODEC, VALIDATION_CODEC  # noqa: E402
STAGES.update(STAGES_CODEC); INTERFACE.update(INTERFACE_CODEC); VALIDATION.update(VALIDATION_CODEC)

# R251 C001 (raster pack): the N64 VI's coverage blend and divot (gpu/stages_vi.py)
from .stages_vi import STAGES_VI, INTERFACE_VI, VALIDATION_VI  # noqa: E402
STAGES.update(STAGES_VI); INTERFACE.update(INTERFACE_VI); VALIDATION.update(VALIDATION_VI)

#: stages the engine is allowed to run. Widening this needs evidence, not hope.
ENABLED = tuple(k for k, (grade, _tol) in VALIDATION.items()
                if grade in ('EXACT', 'CLOSE'))

MASK_KINDS = {'NONE': 0, 'APERTURE': 1, 'SLOT': 2, 'SHADOW': 3}

#: color_management -> the DISPLAY stage's cm_mode uniform, matching
#: core/post.display_transform branch for branch
CM_MODES = {'NONE': 0, 'FILMIC': 1, 'REINHARD': 2, 'SRGB': 3}
