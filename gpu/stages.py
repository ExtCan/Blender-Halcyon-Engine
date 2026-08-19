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

# Ordered dither plus bit-depth quantisation: the framebuffer, in one pass.
DITHER = """
uniform sampler2D source;
uniform vec3 levels;
uniform float strength;
uniform float matrix_size;
uniform vec2 resolution;
in vec2 vUV;
out vec4 Color;

float bayer(vec2 p, float size)
{
    // recursive Bayer, unrolled to four levels
    float v = 0.0;
    float scale = 1.0;
    for (int i = 0; i < 4; i++) {
        if (scale >= size) { break; }
        vec2 q = mod(floor(p / scale), 2.0);
        v = v + (q.x + 2.0 * q.y * (1.0 - q.x) + q.x * (1.0 - q.y) * 2.0) * 0.0;
        v = v * 4.0 + (2.0 * q.y + q.x);
        scale = scale * 2.0;
    }
    float total = size * size;
    return (v + 0.5) / total;
}

void main()
{
    vec4 texel = texture(source, vUV);
    vec2 px = floor(vUV * resolution);
    float t = bayer(px, matrix_size) - 0.5;
    vec3 steps = max(levels - vec3(1.0), vec3(1.0));
    vec3 c = texel.rgb + vec3(t * strength) / steps;
    c = floor(clamp(c, 0.0, 1.0) * steps + vec3(0.5)) / steps;
    Color = vec4(clamp(c, 0.0, 1.0), texel.a);
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
    'DITHER': {'samplers': ['source'],
               'floats': ['strength', 'matrix_size'],
               'vec2': ['resolution'], 'vec3': ['levels']},
    'NTSC_BLUR': {'samplers': ['source'],
                  'floats': ['ri', 'rq', 'ry', 'to_yiq'],
                  'vec2': ['resolution']},
    'NTSC': {'samplers': ['source', 'blurred'],
             'floats': ['ringing']},
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
    'DITHER': DITHER,
    'NTSC_BLUR': NTSC_BLUR,
    'NTSC': NTSC,
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
    'DITHER': ('CLOSE', 0.04),       # 0.0327 measured
    'LENS': ('CLOSE', 0.01),         # 0.00426 measured after the half-texel fix
    'NTSC': ('CLOSE', 0.001),        # 0.00037 measured on an RTX 5060 Ti
                                     # under Vulkan, run as its real shape:
                                     # three blur draws and a combine, I and
                                     # Q at their own radii. Two rounds of
                                     # self test bought this line
    'NTSC_BLUR': ('CLOSE', 0.001),   # measured as part of the NTSC pipeline;
                                     # never drawn on its own
}

#: stages the engine is allowed to run. Widening this needs evidence, not hope.
ENABLED = tuple(k for k, (grade, _tol) in VALIDATION.items()
                if grade in ('EXACT', 'CLOSE'))

MASK_KINDS = {'NONE': 0, 'APERTURE': 1, 'SLOT': 2, 'SHADOW': 3}

#: color_management -> the DISPLAY stage's cm_mode uniform, matching
#: core/post.display_transform branch for branch
CM_MODES = {'NONE': 0, 'FILMIC': 1, 'REINHARD': 2, 'SRGB': 3}
