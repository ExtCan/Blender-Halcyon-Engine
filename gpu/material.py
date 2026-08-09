"""Turning one material into one complete fragment shader.

This is where the pieces meet: the node emitter produces the surface
parameters, `glsl_shading` provides the reflectance model, and a light loop
here sums them. What comes out is a single GLSL source that a driver can
compile and that shades a fragment exactly as `core/render.py` does.

The whole thing is checkable without a GPU. Running the assembled shader
through Halcyon's own NumPy backend and comparing against the CPU shading path
tests the emitter, the models and the light loop *together*, which catches the
seams between them that testing each alone does not.

A material that uses a node with no emitter produces no shader at all. That is
the correct outcome and the caller renders it on the CPU.
"""

from . import glsl_shading as GS
from .emit import Emitter, Unsupported

MAX_LIGHTS = 8

#: light types, matching core.scene.Light.type
LIGHT_KIND = {'SUN': 0, 'POINT': 1, 'SPOT': 2, 'AREA': 3}

def lighting(count):
    """The light loop, unrolled to `count` lights.

    Unrolled rather than indexed into a uniform array. A driver prefers it --
    no dynamic indexing, and the light count is a compile-time constant so the
    whole thing folds -- and it removes the one construct that a uniform array
    of vectors introduces, where a vec3 read out of `uniform vec3 x[N]` is
    ambiguous with N lanes of a scalar.
    """
    count = max(int(count), 0)
    decls = ['uniform vec3 hal_ambient;', 'uniform int hal_model;']
    for i in range(count):
        decls += [
            f'uniform int   hal_lkind{i};',
            f'uniform vec3  hal_lpos{i};',
            f'uniform vec3  hal_ldir{i};',
            f'uniform vec3  hal_lcol{i};',
            f'uniform float hal_lenergy{i};',
            f'uniform float hal_lradius{i};',
        ]
    body = ['', '// One light, matching light_surface() on the CPU. The 1/pi is',
            '// Lambertian normalisation for Blender\'s watt-based units, and',
            '// without it every surface comes out white.',
            'vec3 hal_one_light(int kind, vec3 lpos, vec3 ldir, vec3 lcol,',
            '                   float energy, float radius,',
            '                   HalcyonSurface s, vec3 P, vec3 N, vec3 V)',
            '{',
            '    vec3 L;',
            '    float atten = 1.0;',
            '    if (kind == 0) {',
            '        L = normalize(-ldir);',
            '    } else {',
            '        vec3 d = lpos - P;',
            '        float dist = length(d);',
            '        L = d / max(dist, 1e-6);',
            '        atten = 1.0 / max(dist * dist, 1e-6);',
            '        if (kind == 2) {',
            '            float cd = dot(normalize(ldir), -L);',
            '            float edge = cos(max(radius, 1e-3));',
            '            atten *= clamp((cd - edge) / max(1.0 - edge, 1e-4),',
            '                           0.0, 1.0);',
            '        }',
            '    }',
            '    vec4 ds = hal_evaluate(hal_model, s, N, L, V);',
            '    vec3 radiance = lcol * energy * atten * (1.0 / 3.14159265);',
            '    vec3 diff = s.diffuse * s.diffuse_level * ds.x;',
            '    // the specular colour is already in ds.yzw: hal_evaluate',
            '    // folds it, as the CPU does, so Metal can tint by diffuse',
            '    vec3 spec = s.specular_level * ds.yzw;',
            '    return (diff + spec) * radiance;',
            '}',
            '',
            'vec3 hal_shade(HalcyonSurface s, vec3 P, vec3 N, vec3 V, vec3 emission)',
            '{',
            '    vec3 total = s.diffuse * hal_ambient;']
    for i in range(count):
        body.append(f'    total += hal_one_light(hal_lkind{i}, hal_lpos{i}, '
                    f'hal_ldir{i}, hal_lcol{i},')
        body.append(f'                           hal_lenergy{i}, '
                    f'hal_lradius{i}, s, P, N, V);')
    body += ['    return total + emission;', '}']
    return '\n'.join(decls) + '\n' + '\n'.join(body) + '\n'


VARYINGS = """
uniform vec3 hal_P;
uniform vec3 hal_N;
uniform vec3 hal_V;
uniform vec3 hal_T;
uniform vec3 hal_generated;
uniform vec2 hal_uv;
"""


def surface_setup(indent='    '):
    """GLSL that fills a HalcyonSurface from the material uniforms."""
    fields = ('diffuse_level', 'specular_level', 'glossiness', 'roughness',
              'metallic', 'anisotropy', 'aniso_rot', 'soften', 'ior',
              'translucency', 'toon_size', 'toon_smooth', 'opacity')
    lines = [f'{indent}HalcyonSurface s;']
    for f in fields:
        lines.append(f'{indent}s.{f} = hal_{f};')
    lines.append(f'{indent}s.toon_steps = hal_toon_steps;')
    lines.append(f'{indent}s.tangent = hal_T;')
    lines.append(f'{indent}s.bitangent = normalize(cross(hal_N, hal_T));')
    return '\n'.join(lines)


SURFACE_UNIFORMS = """
uniform float hal_diffuse_level;
uniform float hal_specular_level;
uniform float hal_glossiness;
uniform float hal_roughness;
uniform float hal_metallic;
uniform float hal_anisotropy;
uniform float hal_aniso_rot;
uniform float hal_soften;
uniform float hal_ior;
uniform float hal_translucency;
uniform float hal_toon_size;
uniform float hal_toon_smooth;
uniform float hal_toon_steps;
uniform float hal_opacity;
uniform vec3  hal_specular_tint;
"""


def find_surface_link(graph):
    """The node feeding the output's Surface socket, if any."""
    out = (graph or {}).get('output')
    node = (graph or {}).get('nodes', {}).get(out)
    if node is None:
        return None
    for sock in node.get('inputs', ()):
        if sock.get('name') == 'Surface' and sock.get('link'):
            return sock['link']
    return None


def assemble(graph, model_index=0, light_count=0):
    """Build a complete fragment shader for one material.

    Returns (source, samplers) or (None, reason). The reason names the node
    types with no emitter, so the caller can say why a material stayed on the
    CPU rather than merely that it did.
    """
    link = find_surface_link(graph)
    em = Emitter(graph)
    base_colour = 'vec3(0.8)'
    body = ''
    if link is not None:
        try:
            var, vt = em.output(link[0], link[1])
            base_colour = em.cast(var, vt, 'vec3')
            body = em.body()
        except Unsupported as exc:
            missing = ', '.join(sorted(em.unsupported)) or str(exc)
            return None, f'no GLSL emitter for {missing}'

    src = (GS.GLSL + GS.DISPATCH + VARYINGS + SURFACE_UNIFORMS
           + lighting(light_count) + '\n'.join(em.inline) + """
out vec4 Color;
void main()
{
""" + body + '\n' + surface_setup() + """
    s.diffuse = """ + base_colour + """;
    s.specular = hal_specular_tint;
    vec3 lit = hal_shade(s, hal_P, normalize(hal_N), normalize(hal_V),
                         vec3(0.0));
    Color = vec4(lit, hal_opacity);
}
""")
    return src, em.samplers


def can_assemble(graph, light_count=0):
    src, _info = assemble(graph, light_count=light_count)
    return src is not None


# ------------------------------------------------------------ frame shading
#
# The shader above proves the emitters and the models agree with the CPU at a
# point. The one below shades a *frame*: it reads the packed G-buffer, keeps
# to its own material's pixels, and reproduces `light_surface` -- the real
# light loop, not a replica of it. Everything constant for the frame is baked
# into the source as literals, which sidesteps Vulkan's push-constant budget
# entirely: the only uniforms left are the three G-buffer samplers and their
# sizes, and the shader is recompiled only when the scene's constants change,
# which the compile cache already handles by hashing the source.


def _f(x):
    """A float literal a strict GLSL front-end cannot mistake for an int.

    `%.9g` renders 6.0 as `6`, which is an *integer* literal. Implicit
    int-to-float conversion makes that legal everywhere that matters, but a
    baked shader full of them is one strict-profile driver away from a
    mystery, so every literal carries its point.
    """
    s = f'{float(x):.9g}'
    if 'e' not in s and 'E' not in s and '.' not in s and 'inf' not in s \
            and 'nan' not in s:
        s += '.0'
    return s


def _v3(t):
    t = tuple(float(v) for v in t)[:3]
    return 'vec3({}, {}, {})'.format(*(_f(v) for v in t))


def _attenuation(light, falloff_default, eps=1e-6):
    """GLSL for one light's distance falloff, matching lights.attenuate."""
    mode = getattr(light, 'decay', 'DEFAULT')
    if mode == 'DEFAULT':
        mode = falloff_default
    start = float(getattr(light, 'decay_start', 0.0))
    if mode == 'NONE':
        return '1.0'
    d = f'max(dist - {_f(start)}, {_f(eps)})' if start > 0 else \
        f'max(dist, {_f(eps)})'
    if mode == 'INVERSE':
        return f'(1.0 / {d})'
    if mode == 'CUSTOM':
        end = max(float(getattr(light, 'decay_end', 40.0)), start + eps)
        return (f'clamp(1.0 - (dist - {_f(start)}) / {_f(end - start)}, '
                f'0.0, 1.0)')
    return f'(1.0 / ({d} * {d}))'


def _v4(t):
    t = tuple(float(v) for v in t)[:4]
    return 'vec4({}, {}, {}, {})'.format(*(_f(v) for v in t))


#: the deterministic-sampling primitives: the pattern library's integer
#: hash under a sampling name (a material may inline PRIM_GLSL too, and a
#: driver rejects a redefinition), plus the shared 256-entry unit-circle
#: table as a texture fetch. The table exists because a driver's own
#: sin/cos round differently than NumPy's, and an occlusion ray is a
#: cliff: reading the SAME float32 data on every device keeps the jitter
#: bit-identical, which is what keeps the averaged visibility identical.
SAMPLING_GLSL = """
uniform sampler2D hal_circle;
float hal_smp_hash3(int ix, int iy, int iz)
{
    uint h = (uint(ix) * 374761393u + uint(iy) * 668265263u
              + uint(iz) * 1274126177u) & 0x7fffffffu;
    h = (h ^ (h >> 13u)) * 1274126177u;
    return float((h ^ (h >> 16u)) & 0xffffu) / 65535.0;
}
vec2 hal_smp_circle(float u)
{
    int ai = int(u * 65535.0 + 0.5) & 255;
    return texelFetch(hal_circle, ivec2(ai, 0), 0).rg;
}
"""


def _ao_function(ao, consts):
    """`ambient_occlusion` as GLSL: the same rays, the same average.

    Cosine-weighted hemisphere directions from the hash and the circle
    table -- the identical draws the CPU makes for this pixel, because
    both sides compute (pixel, sample, seed) -> jitter with the same
    integer arithmetic and the same table data. Each direction asks the
    shared BVH traversal, and the occlusion scales the ambient term
    exactly as `light_surface` does it.
    """
    w, h = consts['resolution']
    seed = int(consts.get('seed', 0))
    samples = max(int(ao['samples']), 1)
    L = ['float hal_ao(vec3 P, vec3 N)',
         '{',
         '    vec3 up = (abs(N.z) < 0.999) ? vec3(0.0, 0.0, 1.0) '
         ': vec3(1.0, 0.0, 0.0);',
         '    vec3 t = normalize(cross(up, N));',
         '    vec3 b = cross(N, t);',
         f'    vec3 org = P + N * {_f(float(ao["bias"]))};',
         f'    int sx = int(vUV.x * {_f(float(w))});',
         f'    int sy = int(vUV.y * {_f(float(h))});',
         '    float occ = 0.0;']
    for k in range(samples):
        z = 2 * k + 8389 + 7919 * seed
        L += ['    {',
              f'    float u1 = hal_smp_hash3(sx, sy, {z});',
              f'    vec2 cs = hal_smp_circle(hal_smp_hash3(sx, sy, '
              f'{z + 1}));',
              '    float r = sqrt(u1);',
              '    vec3 d = normalize(t * (r * cs.x) + b * (r * cs.y) '
              '+ N * sqrt(max(1.0 - u1, 0.0)));',
              f'    occ += hal_bvh_occluded(org, d, '
              f'{_f(float(ao["distance"]))});',
              '    }']
    L += [f'    return clamp(1.0 - (occ / {_f(float(samples))}) * '
          f'{_f(float(ao["intensity"]))}, 0.0, 1.0);',
          '}']
    return '\n'.join(L) + '\n'


def _rad_function(rad, consts):
    """`radiosity_gather` as GLSL: the same rays, the same bleed.

    Directions come from the identical hash draws (the gather's own
    salt); each asks the closest-hit kernel. A miss inside the gather
    distance returns the scene's ambient colour, a hit returns that
    surface's flat diffuse from a baked table -- looked up through
    hal_tri_data, whose texel already carries the material index the
    G-buffer uses -- scaled by the linear falloff and the intensity,
    exactly as the CPU gathers it.

    The one seam is a NAMED one: the kernel reports id -2.0 when two
    surfaces tie inside float noise (the glass-mirror lesson), and a
    fragment shader cannot re-route single samples to the CPU the way
    the sweeps do. A tie sample here reads the TABLE MEAN instead of
    the winner's albedo. Gather rays stop at first hit, so the exposed
    coplanar overlaps that made sweep ties common are occluded from
    them; what remains is edge-grazes, single samples of N, bounded by
    |mean - winner| / samples.
    """
    w, h = consts['resolution']
    seed = int(consts.get('seed', 0))
    samples = max(int(rad['samples']), 1)
    dist = max(float(rad['distance']), 1e-4)
    albedo = rad['albedo']
    mean = [sum(c[i] for c in albedo) / max(len(albedo), 1)
            for i in range(3)] if albedo else [0.8, 0.8, 0.8]
    L = ['vec3 hal_rad_alb(float m)', '{']
    for i, col in enumerate(albedo[:-1] if len(albedo) > 1 else albedo):
        L.append(f'    if (m < {_f(i + 0.5)}) return {_v3(col)};')
    L.append(f'    return {_v3(albedo[-1] if albedo else (0.8, 0.8, 0.8))};')
    L.append('}')
    L += ['vec3 hal_rad_at(vec3 P, vec3 N, int sx, int sy)',
          '{',
          '    vec3 up = (abs(N.z) < 0.999) ? vec3(0.0, 0.0, 1.0) '
          ': vec3(1.0, 0.0, 0.0);',
          '    vec3 t = normalize(cross(up, N));',
          '    vec3 b = cross(N, t);',
          f'    vec3 org = P + N * {_f(float(rad["bias"]))};',
          '    vec3 gather = vec3(0.0, 0.0, 0.0);']
    for k in range(samples):
        z = 2 * k + int(rad['salt']) + 7919 * seed
        L += ['    {',
              f'    float u1 = hal_smp_hash3(sx, sy, {z});',
              f'    vec2 cs = hal_smp_circle(hal_smp_hash3(sx, sy, '
              f'{z + 1}));',
              '    float r = sqrt(u1);',
              '    vec3 d = normalize(t * (r * cs.x) + b * (r * cs.y) '
              '+ N * sqrt(max(1.0 - u1, 0.0)));',
              f'    vec4 hh = hal_bvh_intersect(org, d, {_f(dist)});',
              f'    if (hh.x < -1.5) {{',
              f'        float fall = clamp(1.0 - hh.y / {_f(dist)}, '
              f'0.0, 1.0);',
              f'        gather += {_v3(mean)} * (fall '
              f'* {_f(float(rad["intensity"]))});',
              '    } else if (hh.x < -0.5) {',
              f'        gather += {_v3(tuple(rad["ambient"]))};',
              '    } else {',
              f'        float fall = clamp(1.0 - hh.y / {_f(dist)}, '
              f'0.0, 1.0);',
              '        float hm = hal_tri_data(hh.x).x;',
              f'        gather += hal_rad_alb(hm) * (fall '
              f'* {_f(float(rad["intensity"]))});',
              '    }',
              '    }']
    L += [f'    return gather / {_f(float(samples))};', '}']
    L += ['vec3 hal_rad(vec3 P, vec3 N)',
          '{',
          f'    int sx = int(vUV.x * {_f(float(w))});',
          f'    int sy = int(vUV.y * {_f(float(h))});',
          '    return hal_rad_at(P, N, sx, sy);',
          '}']
    return '\n'.join(L) + '\n'


def _rad_lookup_function(rad, consts):
    """`radiosity_lookup` as GLSL: the field's bilinear, texel for texel.

    Reads the grid the radfield pre-pass drew (or, on the CPU, the grid
    `radiosity_field` computed -- same sources, same rays, same numbers)
    and blends four points with validity-weighted bilinear weights in
    the exact float order the CPU uses. All four corners invalid falls
    back to the flat ambient colour, exactly as the CPU does.
    """
    w, h = consts['resolution']
    n = float(max(int(rad.get('spacing', 1)), 1))
    gw, gh = rad['grid']
    L = ['vec3 hal_rad_lookup()',
         '{',
         f'    float fx = float(int(vUV.x * {_f(float(w))})) / {_f(n)};',
         f'    float fy = float(int(vUV.y * {_f(float(h))})) / {_f(n)};',
         f'    float gx0 = min(floor(fx), {_f(gw - 1.0)});',
         f'    float gy0 = min(floor(fy), {_f(gh - 1.0)});',
         f'    float gx1 = min(gx0 + 1.0, {_f(gw - 1.0)});',
         f'    float gy1 = min(gy0 + 1.0, {_f(gh - 1.0)});',
         '    float tx = clamp(fx - gx0, 0.0, 1.0);',
         '    float ty = clamp(fy - gy0, 0.0, 1.0);',
         '    vec4 c00 = texelFetch(hal_radfield, '
         'ivec2(int(gx0), int(gy0)), 0);',
         '    vec4 c10 = texelFetch(hal_radfield, '
         'ivec2(int(gx1), int(gy0)), 0);',
         '    vec4 c01 = texelFetch(hal_radfield, '
         'ivec2(int(gx0), int(gy1)), 0);',
         '    vec4 c11 = texelFetch(hal_radfield, '
         'ivec2(int(gx1), int(gy1)), 0);',
         '    float w00 = (1.0 - tx) * (1.0 - ty) * c00.a;',
         '    float w10 = tx * (1.0 - ty) * c10.a;',
         '    float w01 = (1.0 - tx) * ty * c01.a;',
         '    float w11 = tx * ty * c11.a;',
         '    float total = w00 + w10 + w01 + w11;',
         '    vec3 rgb = c00.rgb * w00 + c10.rgb * w10 '
         '+ c01.rgb * w01 + c11.rgb * w11;',
         f'    return (total > 1e-6) ? rgb / max(total, 1e-6) '
         f': {_v3(tuple(rad["ambient"]))};',
         '}']
    return 'uniform sampler2D hal_radfield;\n' + '\n'.join(L) + '\n'


def radiosity_field_pass(rad, consts, sides):
    """The grid pre-pass: one fragment per grid point, gathering at the
    first covered pixel of its block, row-major -- the CPU's own source
    rule, so both devices cast identical rays. Writes (irradiance,
    valid). Drawn once per frame at grid resolution, before every
    material pass, and bound to them as `hal_radfield`.
    """
    from . import gbuffer as GB
    from .rtrace import INTERSECT_GLSL, TRAVERSE_GLSL
    w, h = consts['resolution']
    n = int(max(int(rad.get('spacing', 1)), 1))
    gw, gh = rad['grid']
    trav = TRAVERSE_GLSL + INTERSECT_GLSL
    for cname in ('hal_bvh_side', 'hal_btris_side'):
        trav = trav.replace(f'uniform float {cname};', '')
        trav = trav.replace(cname, _f(float(sides[cname])))
    rad_fns = _rad_function(rad, consts)
    src = '\n'.join([
        GB.GLSL,
        'in vec2 vUV;',
        'out vec4 Color;',
        'uniform vec3 hal_eye;',
        trav,
        SAMPLING_GLSL,
        rad_fns,
        'void main()',
        '{',
        f'    int gx = int(vUV.x * {_f(float(gw))});',
        f'    int gy = int(vUV.y * {_f(float(gh))});',
        # float flags, no bare bool declarations, no struct assignment:
        # the front-end runs this too, and it carries neither
        '    float found = 0.0;',
        '    float fspx = 0.0;',
        '    float fspy = 0.0;',
        # the CPU walks the block ROW-MAJOR (dy outer, dx inner) and
        # keeps the FIRST covered pixel; same walk, same winner
        f'    for (int dy = 0; dy < {n}; dy++) {{',
        f'        for (int dx = 0; dx < {n}; dx++) {{',
        '            if (found < 0.5) {',
        f'                int px = gx * {n} + dx;',
        f'                int py = gy * {n} + dy;',
        f'                if (px < {int(w)} && py < {int(h)}) {{',
        '                    vec2 uv = vec2((float(px) + 0.5) '
        f'/ {_f(float(w))}, (float(py) + 0.5) / {_f(float(h))});',
        '                    HalcyonFragment c = hal_read_gbuffer(uv);',
        '                    if (c.covered) {',
        '                        found = 1.0;',
        '                        fspx = float(px);',
        '                        fspy = float(py);',
        '                    }',
        '                }',
        '            }',
        '        }',
        '    }',
        '    if (found < 0.5) {',
        '        Color = vec4(0.0, 0.0, 0.0, 0.0);',
        '        return;',
        '    }',
        f'    vec2 suv = vec2((fspx + 0.5) / {_f(float(w))}, '
        f'(fspy + 0.5) / {_f(float(h))});',
        '    HalcyonFragment f = hal_read_gbuffer(suv);',
        '    vec3 N = normalize(f.N);',
        '    Color = vec4(hal_rad_at(f.P, N, int(fspx), int(fspy)), 1.0);',
        '}'])
    binds = {'radfield': True, 'size': (int(gw), int(gh)),
             'samplers': ['hal_gb_ids', 'hal_gb_attrs', 'hal_gb_tris',
                          'hal_bvh', 'hal_btris', 'hal_circle'],
             'frame_uniforms': [], 'textures': {}, 'prepasses': ()}
    return src, binds


def _cookie_function(i, spec):
    """GLSL for one light's projected texture, mirroring cookie_factor.

    The lookup is the CPU's own bilinear texel arithmetic written out --
    floor, fract, per-texel wrap, two lerps -- reading the uploaded image
    at texel centres, so both devices filter with the same float math
    instead of trusting a driver's sampler. SPOT clamps (EXTEND: the cone
    edge lands on the image edge), SUN wraps (REPEAT: the tiled cloud
    shadow of the era).
    """
    w = float(spec['w'])
    h = float(spec['h'])
    if spec['kind'] == 'SUN':
        wx = f'mod(x0, {_f(w)})'
        wx1 = f'mod(x0 + 1.0, {_f(w)})'
        wy = f'mod(y0, {_f(h)})'
        wy1 = f'mod(y0 + 1.0, {_f(h)})'
    else:
        wx = f'clamp(x0, 0.0, {_f(w - 1.0)})'
        wx1 = f'clamp(x0 + 1.0, 0.0, {_f(w - 1.0)})'
        wy = f'clamp(y0, 0.0, {_f(h - 1.0)})'
        wy1 = f'clamp(y0 + 1.0, 0.0, {_f(h - 1.0)})'
    L = [f'uniform sampler2D hal_cookie{i};',
         f'vec3 hal_cookie_rgb{i}(vec2 uv)',
         '{',
         f'    float fx = uv.x * {_f(w)} - 0.5;',
         f'    float fy = uv.y * {_f(h)} - 0.5;',
         '    float x0 = floor(fx);',
         '    float y0 = floor(fy);',
         '    float tx = fx - x0;',
         '    float ty = fy - y0;',
         f'    float x0w = {wx};',
         f'    float x1w = {wx1};',
         f'    float y0w = {wy};',
         f'    float y1w = {wy1};',
         f'    vec3 c00 = texture(hal_cookie{i}, vec2((x0w + 0.5) / {_f(w)}, '
         f'(y0w + 0.5) / {_f(h)})).rgb;',
         f'    vec3 c10 = texture(hal_cookie{i}, vec2((x1w + 0.5) / {_f(w)}, '
         f'(y0w + 0.5) / {_f(h)})).rgb;',
         f'    vec3 c01 = texture(hal_cookie{i}, vec2((x0w + 0.5) / {_f(w)}, '
         f'(y1w + 0.5) / {_f(h)})).rgb;',
         f'    vec3 c11 = texture(hal_cookie{i}, vec2((x1w + 0.5) / {_f(w)}, '
         f'(y1w + 0.5) / {_f(h)})).rgb;',
         '    vec3 top = c00 + (c10 - c00) * tx;',
         '    vec3 bot = c01 + (c11 - c01) * tx;',
         '    return top + (bot - top) * ty;',
         '}']
    return '\n'.join(L) + '\n'


def _shadow_function(i, meta, consts):
    """GLSL for one light's shadow term, mirroring ShadowMap.lookup exactly.

    Everything is baked: the light-space matrix as four row vectors (no mat4,
    which Halcyon's own front-end does not carry), the linearise constants,
    the PCF offsets unrolled tap by tap with the softness already multiplied
    in, and for a point light the six cube faces as an atlas with the face
    chosen by the major axis, exactly as CubeShadow does. The function
    returns the same 1 - (1 - lit) * density the CPU returns.

    A ray meta emits `visibility`'s RAY branch instead: offset the origin
    along the shading normal AND the light direction by the ray bias, clip
    the ray just short of the light, and ask the BVH -- the same
    `hal_bvh_occluded` the occlusion kernel proved against `bvh.occluded()`
    ray for ray. No density term: the CPU's RAY branch applies none.
    """
    import numpy as np
    from ..core.lights import _pcf_offsets

    if meta.get('ray'):
        bias = _f(float(meta['bias']))
        samples = int(meta.get('samples', 1))
        if samples <= 1:
            return '\n'.join([
                f'float hal_shadow_vis{i}(vec3 P, vec3 N, vec3 L, '
                'float dist)',
                '{',
                f'    vec3 org = P + N * {bias} + L * {bias};',
                '    float maxt = (dist > 1e8) ? 1e9 : dist * (1.0 - 1e-3);',
                '    return 1.0 - hal_bvh_occluded(org, L, maxt);',
                '}']) + '\n'
        # SOFT: visibility()'s deterministic branch, sample for sample --
        # the same hash draws, the same table angles, the same jittered
        # rays this pixel's CPU shade would build, averaged the same way
        w, h = consts['resolution']
        seed = int(consts.get('seed', 0))
        radius = _f(float(meta['radius']))
        L = [f'float hal_shadow_vis{i}(vec3 P, vec3 N, vec3 L, float dist)',
             '{',
             f'    vec3 org = P + N * {bias} + L * {bias};',
             '    float maxt = (dist > 1e8) ? 1e9 : dist * (1.0 - 1e-3);',
             '    vec3 up = (abs(L.z) < 0.999) ? vec3(0.0, 0.0, 1.0) '
             ': vec3(1.0, 0.0, 0.0);',
             '    vec3 t = normalize(cross(up, L));',
             '    vec3 b = cross(L, t);',
             f'    int sx = int(vUV.x * {_f(float(w))});',
             f'    int sy = int(vUV.y * {_f(float(h))});',
             '    float acc = 0.0;']
        for k in range(samples):
            z = 2 * k + 131 * int(i) + 7919 * seed
            L += ['    {',
                  f'    float u1 = hal_smp_hash3(sx, sy, {z});',
                  f'    vec2 cs = hal_smp_circle(hal_smp_hash3(sx, sy, '
                  f'{z + 1}));',
                  f'    float r = sqrt(u1) * {radius};',
                  '    vec3 jit = t * (r * cs.x) + b * (r * cs.y);',
                  '    vec3 Lj = normalize(L * dist + jit);',
                  '    acc += 1.0 - hal_bvh_occluded(org, Lj, maxt);',
                  '    }']
        L += [f'    return acc / {_f(float(samples))};', '}']
        return '\n'.join(L) + '\n'

    faces = meta['faces']                    # list of per-face dicts
    size = meta['size']
    near, far, persp = meta['near'], meta['far'], meta['persp']
    grid_w = meta['grid'][0]
    atlas_w, atlas_h = size * meta['grid'][0], size * meta['grid'][1]
    taps = max(int(consts.get('shadow_samples', 4)), 1)
    soft = float(meta['softness'])
    bias = float(meta['bias'])
    density = float(meta['density'])
    origin = meta['origin']

    L = [f'uniform sampler2D hal_shadow{i};',
         f'float hal_shadow_vis{i}(vec3 P, vec3 N, vec3 L)',
         '{',
         f'    float ndl = clamp(dot(N, L), 0.0, 1.0);',
         f'    float slope = {_f(bias)} * (1.0 + 2.0 * (1.0 - ndl));',
         f'    float pdist = length(P - {_v3(origin)});']
    # texel_size, then the normal offset the CPU applies before the lookup
    if persp:
        L.append(f'    float texel = 2.0 * {_f(meta["extent"])} * '
                 f'max(pdist, {_f(near)}) / {_f(size)};')
    else:
        L.append(f'    float texel = {_f(2.0 * meta["extent"] / size)};')
    L += [f'    float off_amt = texel * (1.5 + 2.5 * '
          f'sqrt(max(1.0 - ndl * ndl, 0.0))) * {_f(max(1.0, soft))};',
          '    vec4 ph = vec4(P + N * off_amt, 1.0);']

    if len(faces) > 1:
        # cube: the face is the major axis of the vector from the light
        L += ['    vec3 dvec = ph.xyz - ' + _v3(origin) + ';',
              '    vec3 avec = abs(dvec);',
              '    int face = 0;',
              '    if (avec.x >= avec.y && avec.x >= avec.z) '
              '{ face = dvec.x >= 0.0 ? 0 : 1; }',
              '    else if (avec.y >= avec.z) '
              '{ face = dvec.y >= 0.0 ? 2 : 3; }',
              '    else { face = dvec.z >= 0.0 ? 4 : 5; }',
              '    float clipx = 0.0; float clipy = 0.0;',
              '    float clipz = 0.0; float clipw = 1.0;',
              '    float cellx = 0.0; float celly = 0.0;']
        for fi, fc in enumerate(faces):
            vp = np.asarray(fc['vp'], np.float32)
            L += [f'    if (face == {fi}) {{',
                  f'        clipx = dot(ph, {_v4(vp[0])});',
                  f'        clipy = dot(ph, {_v4(vp[1])});',
                  f'        clipz = dot(ph, {_v4(vp[2])});',
                  f'        clipw = dot(ph, {_v4(vp[3])});',
                  f'        cellx = {_f((fi % grid_w) * size)};',
                  f'        celly = {_f((fi // grid_w) * size)};',
                  '    }']
    else:
        vp = np.asarray(faces[0]['vp'], np.float32)
        L += [f'    float clipx = dot(ph, {_v4(vp[0])});',
              f'    float clipy = dot(ph, {_v4(vp[1])});',
              f'    float clipz = dot(ph, {_v4(vp[2])});',
              f'    float clipw = dot(ph, {_v4(vp[3])});',
              '    float cellx = 0.0; float celly = 0.0;']

    eps = 1e-6
    L += [f'    float w = abs(clipw) < {_f(eps)} ? {_f(eps)} : clipw;',
          '    float nx = clipx / w;',
          '    float ny = clipy / w;',
          '    float nz = clipz / w;',
          '    bool inside = abs(nx) <= 1.0 && abs(ny) <= 1.0 && nz <= 1.0']
    if persp:
        L[-1] += ' && clipw > 0.0;'
    else:
        L[-1] += ';'
    # linearise the fragment's own depth exactly as _linearise does
    L.append('    float zc = clamp(nz, -1.0, 1.0);')
    if persp:
        L.append(f'    float den = ({_f(far + near)}) - zc * {_f(far - near)};')
        L.append(f'    den = abs(den) < {_f(eps)} ? {_f(eps)} : den;')
        L.append(f'    float sdist = {_f(2.0 * near * far)} / den;')
    else:
        L.append(f'    float sdist = (zc * 0.5 + 0.5) * {_f(far - near)}'
                 f' + {_f(near)};')
    L += [f'    float u = (nx * 0.5 + 0.5) * {_f(size)};',
          f'    float v = (ny * 0.5 + 0.5) * {_f(size)};',
          '    float lit = 0.0;']
    offs = _pcf_offsets(taps) * max(soft, 0.0)
    for ox, oy in offs:
        L += [f'    {{',
              f'    float xi = clamp(floor(u + {_f(ox)}), 0.0, '
              f'{_f(size - 1)});',
              f'    float yi = clamp(floor(v + {_f(oy)}), 0.0, '
              f'{_f(size - 1)});',
              f'    vec2 suv = vec2((cellx + xi + 0.5) / {_f(atlas_w)}, '
              f'(celly + yi + 0.5) / {_f(atlas_h)});',
              f'    float occ = texture(hal_shadow{i}, suv).r;',
              '    lit += (sdist - slope <= occ) ? 1.0 : 0.0;',
              '    }']
    L += [f'    lit /= {_f(len(offs))};',
          '    if (!inside) { lit = 1.0; }',
          f'    return 1.0 - (1.0 - lit) * {_f(density)};',
          '}']
    return '\n'.join(L) + '\n'


def _sky_blend(x, mode):
    """`sky._blend`, as an expression: the input is already clamped."""
    if mode == 'SMOOTH':
        return f'({x} * {x} * (3.0 - 2.0 * {x}))'
    if mode == 'SHARP':
        return f'({x} * {x})'
    if mode == 'EASE':
        return f'sqrt({x})'
    return x


def _sky_env_lines(env_spec):
    """The GRADIENT and BANDS skies along hal_R, exactly `sky.gradient` /
    `sky.bands` with every world constant baked.

    Only the ray's z enters the formulas, so the world rotation (which
    spins x and y) is correctly absent. Strength multiplies at the end,
    as `evaluate` does. Bands quantise in the blend parameter with the
    same steps/soft smoothing arithmetic, ground colour takes the same
    below-horizon branch.
    """
    sp = env_spec[1]
    bands = env_spec[0] == 'SKY_BANDS'
    hor = _v3(sp['horizon'])
    zen = _v3(sp['zenith'])
    gnd = _v3(sp['ground'])
    height = float(sp['height'])
    inv_above = 1.0 / max(1.0 - height, 1e-3)
    inv_below = 1.0 / max(1.0 + height, 1e-3)
    falloff = float(sp['falloff'])
    mode = sp['blend']
    L = ['    float hal_sky_up = clamp(hal_R.z, -1.0, 1.0);']

    def t_expr(raw, tag):
        # clip -> pow -> blend -> (bands: quantise), returning the var name
        L.append(f'    float hal_st{tag} = '
                 f'{_sky_blend(f"pow({raw}, {_f(falloff)})", mode)};')
        name = f'hal_st{tag}'
        if not bands:
            return name
        steps = int(sp['steps'])
        if steps == 1:
            L.append(f'    float hal_sq{tag} = 0.0;')
            return f'hal_sq{tag}'
        soft = float(sp['soft'])
        L.append(f'    float hal_ss{tag} = min(floor({name} * {_f(float(steps))}), '
                 f'{_f(float(steps - 1))});')
        if soft > 1e-4:
            L.extend([
                f'    float hal_sf{tag} = {name} * {_f(float(steps))} - '
                f'floor({name} * {_f(float(steps))});',
                f'    float hal_se{tag} = clamp((hal_sf{tag} - '
                f'{_f(1.0 - soft)}) / {_f(max(soft, 1e-4))}, 0.0, 1.0);',
                f'    hal_ss{tag} = min(hal_ss{tag} + hal_se{tag} * '
                f'hal_se{tag} * (3.0 - 2.0 * hal_se{tag}), '
                f'{_f(float(steps - 1))});'])
        L.append(f'    float hal_sq{tag} = clamp(hal_ss{tag} / '
                 f'{_f(float(steps - 1))}, 0.0, 1.0);')
        return f'hal_sq{tag}'

    ta = t_expr(f'clamp((hal_sky_up - {_f(height)}) * {_f(inv_above)}, '
                '0.0, 1.0)', 'a')
    L.append(f'    vec3 hal_env = {hor} + ({zen} - {hor}) * {ta};')
    if sp['show_ground']:
        tb = t_expr(f'clamp(({_f(height)} - hal_sky_up) * {_f(inv_below)}, '
                    '0.0, 1.0)', 'b')
        L.append(f'    if (hal_sky_up < {_f(height)}) '
                 f'{{ hal_env = {hor} + ({gnd} - {hor}) * {tb}; }}')
    strength = float(sp['strength'])
    if abs(strength - 1.0) > 1e-6:
        L.append(f'    hal_env = hal_env * {_f(strength)};')
    return L


#: filters the manual sampler reproduces in every pass variant. TRILINEAR
#: and anisotropy additionally need the per-pixel footprint field and the
#: mip atlas, which the FRAME and vertex-rate passes carry (hal_uvgrad);
#: secondary passes mirror the CPU's ray hits, which have no footprint
#: and sample the top level. N64 3-point needs no footprint at all.
SUPPORTED_TEX_FILTERS = ('NEAREST', 'BILINEAR', 'TRILINEAR', 'N64_3POINT')


def mip_atlas(tex):
    """(atlas (H2,W,4) float32, [(y0, w, h) per level]) from tex's OWN mips.

    The CPU's build_mips output packed in a vertical stack -- the driver
    samples the very texels the CPU filters, so the two trilinears can
    only disagree by lerp rounding. Level 0 sits at y0 = 0.
    """
    import numpy as np
    mips = tex.build_mips()
    W = int(tex.width)
    H2 = int(sum(m.shape[0] for m in mips))
    atlas = np.zeros((H2, W, 4), np.float32)
    levels = []
    y = 0
    for m in mips:
        h, w = m.shape[:2]
        atlas[y:y + h, :w] = m
        levels.append((float(y), float(w), float(h)))
        y += h
    return atlas, levels


def _wrap_dyn(var, dim, wrap):
    """A wrap expression against a RUNTIME dimension (mip levels vary)."""
    if wrap == 'REPEAT':
        return f'({var} - floor({var} / {dim}) * {dim})'
    if wrap == 'MIRROR':
        return (f'(mod({var}, 2.0 * {dim}) < {dim} ? '
                f'mod({var}, 2.0 * {dim}) : '
                f'2.0 * {dim} - 1.0 - mod({var}, 2.0 * {dim}))')
    return f'clamp({var}, 0.0, {dim} - 1.0)'          # EXTEND and CLIP


def _mip_sampler(name, tex, wrap, levels, aniso, bias):
    """GLSL for one TRILINEAR (optionally anisotropic) footprint sampler.

    Mirrors Texture._sample_trilinear and _sample_aniso line for line:
    compute_lod from the hal_uvgrad field (the CPU's OWN derivatives,
    uploaded), a per-level manual bilinear inside the atlas stack, the
    a + (b - a) * frac level blend, and -- under anisotropy -- the mip
    level of the MINOR footprint axis with uniform trilinear taps along
    the major. The level table is a select ladder: no arrays, nothing
    the front-end cannot run.
    """
    w0, h0 = float(tex.width), float(tex.height)
    n = len(levels)
    L = [f'uniform sampler2D {name};',
         f'vec4 hal_afetch_{name}(float x, float y)',
         '{',
         f'    return texture({name}, vec2((x + 0.5) / {_f(w0)}, '
         f'(y + 0.5) / {_f(float(sum(lv[2] for lv in levels)))}));',
         '}',
         f'vec3 hal_mipof_{name}(float l)',
         '{']
    for k, (y0, lw, lh) in enumerate(levels[:-1]):
        L.append(f'    if (l < {_f(k + 0.5)}) '
                 f'{{ return vec3({_f(y0)}, {_f(lw)}, {_f(lh)}); }}')
    y0, lw, lh = levels[-1]
    L += [f'    return vec3({_f(y0)}, {_f(lw)}, {_f(lh)});',
          '}',
          f'vec4 hal_lvl_{name}(vec2 uv, float y0, float lw, float lh)',
          '{',
          '    float fx = uv.x * lw - 0.5;',
          '    float fy = uv.y * lh - 0.5;',
          '    float x0 = floor(fx);',
          '    float y0i = floor(fy);',
          '    float tx = fx - x0;',
          '    float ty = fy - y0i;',
          '    float x1 = x0 + 1.0;',
          '    float y1 = y0i + 1.0;',
          f'    float wx0 = {_wrap_dyn("x0", "lw", wrap)};',
          f'    float wx1 = {_wrap_dyn("x1", "lw", wrap)};',
          f'    float wy0 = {_wrap_dyn("y0i", "lh", wrap)};',
          f'    float wy1 = {_wrap_dyn("y1", "lh", wrap)};',
          f'    vec4 c00 = hal_afetch_{name}(wx0, y0 + wy0);',
          f'    vec4 c10 = hal_afetch_{name}(wx1, y0 + wy0);',
          f'    vec4 c01 = hal_afetch_{name}(wx0, y0 + wy1);',
          f'    vec4 c11 = hal_afetch_{name}(wx1, y0 + wy1);',
          '    vec4 top = c00 + (c10 - c00) * tx;',
          '    vec4 bot = c01 + (c11 - c01) * tx;',
          '    return top + (bot - top) * ty;',
          '}',
          f'vec4 hal_trilerp_{name}(vec2 uv, float lod)',
          '{',
          f'    float l = clamp(lod, 0.0, {_f(n - 1.0)});',
          '    float l0 = floor(l);',
          '    float f = l - l0;',
          f'    vec3 A = hal_mipof_{name}(l0);',
          f'    vec3 B = hal_mipof_{name}(min(l0 + 1.0, {_f(n - 1.0)}));',
          f'    vec4 a = hal_lvl_{name}(uv, A.x, A.y, A.z);',
          f'    vec4 b = hal_lvl_{name}(uv, B.x, B.y, B.z);',
          '    return a + (b - a) * f;',
          '}',
          f'vec4 hal_sample_{name}(vec2 uv)',
          '{',
          '    vec4 g = texture(hal_uvgrad, vUV);']
    if int(aniso) > 1:
        A = float(int(aniso))
        L += [
            # exactly _sample_aniso: screen-x/-y footprints in texels
            f'    float lx = sqrt((g.x * {_f(w0)}) * (g.x * {_f(w0)}) + '
            f'(g.z * {_f(h0)}) * (g.z * {_f(h0)}));',
            f'    float ly = sqrt((g.y * {_f(w0)}) * (g.y * {_f(w0)}) + '
            f'(g.w * {_f(h0)}) * (g.w * {_f(h0)}));',
            '    float mx = (lx >= ly) ? 1.0 : 0.0;',
            '    float major = (mx > 0.5) ? lx : ly;',
            '    float minor = max((mx > 0.5) ? ly : lx, 1e-6);',
            f'    float ratio = clamp(major / minor, 1.0, {_f(A)});',
            f'    float lod = log2(max(major / ratio, 1e-6)) + {_f(bias)};',
            '    float axu = (mx > 0.5) ? g.x : g.y;',
            '    float axv = (mx > 0.5) ? g.z : g.w;',
            '    vec4 acc = vec4(0.0);',
            f'    for (int k = 0; k < {int(aniso)}; k++) {{',
            f'        float t = (float(k) + 0.5) / {_f(A)} - 0.5;',
            f'        acc = acc + hal_trilerp_{name}(uv + vec2(axu, axv) '
            f'* t, lod);',
            '    }',
            f'    vec4 c = acc / {_f(A)};']
    else:
        L += [
            # exactly compute_lod: per-axis texel footprints, max, log2
            f'    float dx = sqrt((g.x * {_f(w0)}) * (g.x * {_f(w0)}) + '
            f'(g.y * {_f(h0)}) * (g.y * {_f(h0)}));',
            f'    float dy = sqrt((g.z * {_f(w0)}) * (g.z * {_f(w0)}) + '
            f'(g.w * {_f(h0)}) * (g.w * {_f(h0)}));',
            '    float rho = max(max(dx, dy), 1e-6);',
            f'    vec4 c = hal_trilerp_{name}(uv, log2(rho) '
            f'+ {_f(bias)});']
    if wrap == 'CLIP':
        L.append('    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || '
                 'uv.y > 1.0) { c = vec4(0.0); }')
    L += ['    return c;', '}']
    return '\n'.join(L) + '\n'


def _block(pieces):
    """Join GLSL pieces as LINES: newline-separated, newline-TERMINATED.

    The assemblers build their source with ''.join(parts), which glues
    adjacent blocks character to character. A block whose last line lacks
    its newline splices onto the next block's first line -- and when both
    halves are declarations, the spliced line matches neither the
    stripper nor the lint (both read whole lines), survives into the
    CreateInfo build, and the driver rejects the shader as a
    redefinition. The field found exactly that: the Bump emitter's bare
    `uniform sampler2D hal_bump0;` glued onto `uniform sampler2D
    hal_shadow0;` whenever a bump material had no image textures, and
    every bump frame silently shaded on the CPU. Every multi-line block
    goes through here now, so no emitter has to remember its own
    trailing newline.
    """
    txt = '\n'.join(pieces)
    return txt + '\n' if txt else ''


def _wrap_expr(var, n, mode):
    """Index wrapping, exactly as Texture._wrap_index does it."""
    if mode == 'REPEAT':
        return f'mod({var}, {_f(n)})'
    if mode == 'MIRROR':
        return (f'(mod({var}, {_f(2 * n)}) < {_f(n)} '
                f'? mod({var}, {_f(2 * n)}) '
                f': {_f(2 * n - 1)} - mod({var}, {_f(2 * n)}))')
    return f'clamp({var}, 0.0, {_f(n - 1)})'     # EXTEND, and CLIP's indices


def _texture_sampler(name, tex, filt, wrap):
    """A manual sampler matching Texture.sample, driver-independent.

    `texture()` with driver filtering would put the sampling arithmetic in
    the driver's hands; fetching texel centres and doing the filter in the
    shader keeps it in ours, which is what makes the GPU pixel the CPU pixel.
    """
    w, h = float(tex.width), float(tex.height)
    L = [f'uniform sampler2D {name};',
         f'vec4 hal_fetch_{name}(float x, float y)',
         '{',
         f'    return texture({name}, vec2((x + 0.5) / {_f(w)}, '
         f'(y + 0.5) / {_f(h)}));',
         '}',
         f'vec4 hal_sample_{name}(vec2 uv)',
         '{']
    if filt == 'NEAREST':
        L += [f'    float x = floor(uv.x * {_f(w)});',
              f'    float y = floor(uv.y * {_f(h)});',
              f'    x = {_wrap_expr("x", w, wrap)};',
              f'    y = {_wrap_expr("y", h, wrap)};',
              f'    vec4 c = hal_fetch_{name}(x, y);']
    elif filt == 'N64_3POINT':
        # exactly Texture._sample_3point: the triangular filter picks the
        # dominant corner and blends along the two edges -- three taps,
        # the N64's own arithmetic, no footprint needed
        L += [f'    float fx = uv.x * {_f(w)} - 0.5;',
              f'    float fy = uv.y * {_f(h)} - 0.5;',
              '    float x0 = floor(fx);',
              '    float y0 = floor(fy);',
              '    float tx = fx - x0;',
              '    float ty = fy - y0;',
              '    float x1 = x0 + 1.0;',
              '    float y1 = y0 + 1.0;',
              f'    float wx0 = {_wrap_expr("x0", w, wrap)};',
              f'    float wx1 = {_wrap_expr("x1", w, wrap)};',
              f'    float wy0 = {_wrap_expr("y0", h, wrap)};',
              f'    float wy1 = {_wrap_expr("y1", h, wrap)};',
              f'    vec4 c00 = hal_fetch_{name}(wx0, wy0);',
              f'    vec4 c10 = hal_fetch_{name}(wx1, wy0);',
              f'    vec4 c01 = hal_fetch_{name}(wx0, wy1);',
              f'    vec4 c11 = hal_fetch_{name}(wx1, wy1);',
              '    float up = ((tx + ty) > 1.0) ? 1.0 : 0.0;',
              '    vec4 a = (up > 0.5) ? c11 : c00;',
              '    float s = (up > 0.5) ? (1.0 - ty) : tx;',
              '    float t = (up > 0.5) ? (1.0 - tx) : ty;',
              '    vec4 b = (up > 0.5) ? c01 : c10;',
              '    vec4 cc = (up > 0.5) ? c10 : c01;',
              '    vec4 c = a + (b - a) * s + (cc - a) * t;']
    else:
        L += [f'    float fx = uv.x * {_f(w)} - 0.5;',
              f'    float fy = uv.y * {_f(h)} - 0.5;',
              '    float x0 = floor(fx);',
              '    float y0 = floor(fy);',
              '    float tx = fx - x0;',
              '    float ty = fy - y0;',
              '    float x1 = x0 + 1.0;',
              '    float y1 = y0 + 1.0;',
              f'    float wx0 = {_wrap_expr("x0", w, wrap)};',
              f'    float wx1 = {_wrap_expr("x1", w, wrap)};',
              f'    float wy0 = {_wrap_expr("y0", h, wrap)};',
              f'    float wy1 = {_wrap_expr("y1", h, wrap)};',
              f'    vec4 c00 = hal_fetch_{name}(wx0, wy0);',
              f'    vec4 c10 = hal_fetch_{name}(wx1, wy0);',
              f'    vec4 c01 = hal_fetch_{name}(wx0, wy1);',
              f'    vec4 c11 = hal_fetch_{name}(wx1, wy1);',
              '    vec4 c = mix(mix(c00, c10, tx), mix(c01, c11, tx), ty);']
    if wrap == 'CLIP':
        L.append('    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || '
                 'uv.y > 1.0) { c = vec4(0.0); }')
    L += ['    return c;', '}']
    return '\n'.join(L) + '\n'


def resolve_tex_filter(interp, settings_filter):
    """The filter the CPU will actually use, from n_tex_image's own logic:
    the node asks, the period settings override."""
    filt = {'Closest': 'NEAREST', 'Linear': 'BILINEAR', 'Cubic': 'BILINEAR',
            'Smart': 'BILINEAR'}.get(interp, 'BILINEAR')
    if settings_filter:
        filt = settings_filter
    return filt


def _one_light_source(i, light, consts, shadowed=None, bake=None):
    """The unrolled loop body for one light, mirroring light_surface.

    AREA lights take the position branch: the CPU's sample() treats an area
    light without an explicit surface sample exactly as a point at its
    centre, and its softness lives in the shadow term, not the direct math.

    `shadowed` is the light's shadow meta (or None): a ray meta's visibility
    function takes the distance to the light -- in scope for a positional
    light, 1e9 for a SUN, exactly the `dist` sample() feeds `visibility`.
    """
    import numpy as np
    bake = bake or {}
    kind = getattr(light, 'type', 'POINT')
    col = _v3(getattr(light, 'color', (1, 1, 1)))
    energy = float(getattr(light, 'energy', 1.0))
    lines = ['    {']
    ck = (consts.get('cookies') or {}).get(i)
    if kind == 'SUN':
        d = np.asarray(light.direction, np.float32)
        d = d / max(float(np.linalg.norm(d)), 1e-9)
        lines += [f'    vec3 L = {_v3(-d)};',
                  f'    vec3 rad = {col} * {_f(energy)};']
        if ck is not None:
            # exactly cookie_factor's SUN branch: world position projected
            # on the light's own axes, one tile per cookie_scale units
            lines += [
                f'    vec2 ckuv = vec2(dot(P, {_v3(ck["side"])}) / '
                f'{_f(ck["scale"])}, dot(P, {_v3(ck["up"])}) / '
                f'{_f(ck["scale"])});',
                f'    rad = rad * (vec3(1.0) + (hal_cookie_rgb{i}(ckuv) '
                f'- vec3(1.0)) * {_f(ck["strength"])});']
    else:
        lines += [f'    vec3 delta = {_v3(light.position)} - P;',
                  '    float dist = max(length(delta), 1e-6);',
                  '    vec3 L = delta / dist;',
                  f'    float att = '
                  f'{_attenuation(light, consts["falloff_default"])};',
                  f'    vec3 rad = {col} * ({_f(energy / (4.0 * np.pi))}'
                  f' * att);']
        if kind == 'SPOT':
            sd = np.asarray(light.direction, np.float32)
            sd = sd / max(float(np.linalg.norm(sd)), 1e-9)
            half = float(light.spot_size) * 0.5
            blend = max(float(light.spot_blend), 1e-4)
            outer = np.cos(half)
            inner = np.cos(half * (1.0 - blend))
            lines += [
                f'    float cosang = -dot(L, {_v3(sd)});',
                f'    float spot_t = clamp((cosang - {_f(outer)}) / '
                f'{_f(max(inner - outer, 1e-5))}, 0.0, 1.0);',
                '    rad = rad * (spot_t * spot_t);']
            if ck is not None:
                # exactly cookie_factor's SPOT branch: light->surface
                # direction in the light's own frame, the full cone
                # spanning the image
                lines += [
                    '    vec3 ckd = -L;',
                    f'    float ckz = dot(ckd, {_v3(ck["fwd"])});',
                    '    if (ckz > 1e-6) {',
                    f'    float cks = max(ckz, 1e-6) * '
                    f'{_f(2.0 * ck["tanh"])};',
                    f'    vec2 ckuv = vec2(dot(ckd, {_v3(ck["side"])}) '
                    f'/ cks + 0.5, dot(ckd, {_v3(ck["up"])}) / cks '
                    f'+ 0.5);',
                    f'    rad = rad * (vec3(1.0) + (hal_cookie_rgb{i}'
                    f'(ckuv) - vec3(1.0)) * {_f(ck["strength"])});',
                    '    }']
    if getattr(light, 'negative', False):
        lines.append('    rad = -rad;')
    lines.append('    vec4 ds = hal_evaluate(hal_model_i, s, N, L, V);')
    lines.append('    vec3 contrib = vec3(0.0);')
    if getattr(light, 'affect_diffuse', True) and \
            not getattr(light, 'specular_only', False):
        lines.append('    contrib += (ds.x * s.diffuse * s.diffuse_level)'
                     ' * rad;')
    if getattr(light, 'affect_specular', True) and \
            not getattr(light, 'diffuse_only', False):
        spec = 'ds.yzw'
        if not consts.get('specular_in_gamma', True):
            spec = 'pow(max(ds.yzw, vec3(0.0)), vec3(2.2))'
        lines.append(f'    contrib += {spec} * s.specular_level * rad;')
        if float(bake.get('sheen', 0.0)) > 1e-4:
            # the velvet lobe, exactly as light_surface: scattered back at
            # grazing angles, needing a light, vanishing face-on
            sr = min(max(float(bake.get('sheen_roughness', 0.3)), 0.0), 1.0)
            sheen_exp = 1.0 + (1.0 - sr) * 15.0
            lines.append(
                f'    contrib += {_v3(bake.get("sheen_color", (1, 1, 1)))}'
                f' * (pow(hal_edge_vn, {_f(sheen_exp)})'
                f' * max(dot(N, L), 0.0) * {_f(bake["sheen"])}) * rad;')
    lines.append('    contrib *= 0.318309886;')          # 1/pi, as the CPU
    if shadowed:
        # the CPU order: 1/pi, then visibility, then the clamp
        if isinstance(shadowed, dict) and shadowed.get('ray'):
            dist_arg = '1e9' if kind == 'SUN' else 'dist'
            lines.append(f'    contrib *= '
                         f'hal_shadow_vis{i}(P, N, L, {dist_arg});')
        else:
            lines.append(f'    contrib *= hal_shadow_vis{i}(P, N, L);')
    link = (consts.get('light_links') or {}).get(i)
    if link:
        # light linking, exactly light_surface's mask and order: 1/pi,
        # visibility, THEN the mask, then the clamp. The ladder tests
        # td.y (an exact integer float) against the light's linked
        # object list -- np.isin, unrolled, no texture and no cliff.
        # ONLY lights light just their list; EXCLUDE lights light
        # everything else. Hits carry the same object id through
        # hal_tri_data, exactly as ctx.object_index_raw does at hits.
        tests = ' + '.join(f'((abs(td.y - {_f(float(o))}) < 0.5) '
                           '? 1.0 : 0.0)'
                           for o in link['objects'])
        lines.append(f'    float hal_lk{i} = min({tests}, 1.0);')
        if str(link.get('mode', 'EXCLUDE')).upper() == 'ONLY':
            lines.append(f'    contrib *= hal_lk{i};')
        else:
            lines.append(f'    contrib *= (1.0 - hal_lk{i});')
    clamp = float(consts.get('light_clamp', 0.0))
    if clamp > 0.0:
        lines.append(f'    contrib = min(contrib, vec3({_f(clamp)}));')
    lines.append('    total += contrib;')
    lines.append('    }')
    return lines


#: surface fields baked from the probe, in HalcyonSurface field order
BAKE_FIELDS = ('diffuse_level', 'specular_level', 'glossiness', 'roughness',
               'metallic', 'anisotropy', 'aniso_rot', 'soften', 'ior',
               'translucency', 'toon_size', 'toon_smooth', 'toon_steps',
               'opacity')

#: master-node sockets that may vary per pixel: when LINKED, the chain is
#: emitted and assigned to the surface field; unlinked, the probed constant
#: bakes as before. socket name -> (surface field, glsl type). This is what
#: lets a texture drive Roughness without pushing the material off the GPU.
PER_PIXEL_SOCKETS = {
    'Diffuse Level': ('diffuse_level', 'float'),
    'Specular Level': ('specular_level', 'float'),
    'Specular Color': ('specular', 'vec3'),
    'Glossiness': ('glossiness', 'float'),
    'Roughness': ('roughness', 'float'),
    'Metalness': ('metallic', 'float'),
    'Soften': ('soften', 'float'),
    'IOR': ('ior', 'float'),
    'Translucency': ('translucency', 'float'),
    'Anisotropy': ('anisotropy', 'float'),
    'Anisotropic Rotation': ('aniso_rot', 'float'),
    'Toon Size': ('toon_size', 'float'),
    'Toon Smooth': ('toon_smooth', 'float'),
    'Self-Illumination': ('emission', 'vec3'),
    # the matcap COLOUR is a chain by design -- the documented workflow
    # is an Image Texture through Matcap Coordinates -- and it refused
    # from the day the override was ported, because only this table
    # grants per-pixel rights. The field found it the hard way: one
    # 'Eyes' material put a whole 640x640 frame on the CPU, and the
    # radiosity gather with it. The BLEND stays baked (varying blend
    # still refuses by name).
    'Matcap': ('matcap', 'vec3'),
}


def master_node(graph):
    """The HALCYON_ShaderNode feeding the output, if that is what feeds it."""
    link = find_surface_link(graph)
    if link is None:
        return None
    node = (graph or {}).get('nodes', {}).get(link[0])
    if node is not None and node.get('bl_idname') == 'HALCYON_ShaderNode':
        return node
    return None


def _replace_all(text, pairs):
    for old, new in pairs:
        text = text.replace(old, new)
    return text


def _socket(node, name):
    """One named input socket of a node, or None."""
    for sock in (node or {}).get('inputs', ()):
        if sock.get('name') == name:
            return sock
    return None


def master_normal_linked(graph):
    """Whether the master node's Normal socket drives the shading normal.

    The probe reads this to know that the bent normal it just watched
    `closure_to_surface` produce is one the frame shader will bend
    identically -- the assembler emits exactly that chain. A normal bent by
    anything else (a BSDF lobe's Normal input on a non-master graph) still
    moves the material to the CPU, because nothing below emits it.
    """
    sock = _socket(master_node(graph), 'Normal')
    return bool(sock and sock.get('link'))


def per_pixel_fields(graph):
    """Surface fields the frame shader will compute per pixel for `graph`.

    The probe uses this to know which constancy checks to skip -- a linked
    Roughness varying across the frame is the point, not a disqualifier --
    and the assembler uses it to know which chains to emit. One rule, read
    from the graph by both sides, so they cannot disagree.
    """
    node = master_node(graph)
    if node is None:
        return {}
    out = {}
    for sock in node.get('inputs', ()):
        name = sock.get('name')
        if name in PER_PIXEL_SOCKETS and sock.get('link'):
            field, gtype = PER_PIXEL_SOCKETS[name]
            out[field] = (name, gtype)
    return out


def _assemble_height_pass(graph, mat_id, bump_node, consts, textures,
                          programs):
    """One Bump node's HEIGHT chain as its own full-screen pass.

    Renders (height, 0, 0, keep) over the same ids texture the main pass
    reads, so the main pass can take the CPU's exact one-sided neighbour
    differences by texelFetch. The support machinery mirrors
    `assemble_frame`'s own: manual texture samplers with the filter
    arithmetic in the shader, generated coordinates from the baked
    per-object bounds, the same replacement pass over inlined code.
    Returns (source, binds) or (None, why).
    """
    from . import gbuffer as GB

    em = Emitter(graph or {})
    em.frame_mode = True
    em.resolution = consts.get('resolution')
    em.camera = consts.get('camera')
    em.uv_names = tuple(consts.get('uv_names') or ())
    em.programs = programs if programs is not None else {}
    try:
        expr = em.input(bump_node, 'Height', 'float')
        body = em.body()
    except Unsupported as exc:
        # a height chain the emitter cannot carry -- Blender's sin-fract
        # Noise family above all -- does NOT refuse the material: the
        # pre-pass is only an image, and the CPU can produce it with the
        # renderer's own evaluator, float64 sin and all, EXACTLY. The
        # frame pays one height evaluation over the material's pixels.
        missing = ', '.join(sorted(em.unsupported)) or str(exc)
        return '__CPU__', {'cpu': True, 'node': bump_node.get('id'),
                           'why': f'{missing} evaluates on the CPU into '
                                  'the height pre-pass',
                           'samplers': [], 'textures': {},
                           'frame_uniforms': [], 'uses_screen': False}
    if em.bump_passes:
        return None, 'a Bump node inside another Bump\'s height chain is ' \
                     'not in the deferred pass yet'

    tex_fns = []
    tex_binds = {}
    src = body + ('\n' if body else '')
    replacements = []
    for meta in em.samplers:
        sname = meta['uniform']
        key = meta.get('image')
        tex = (textures or {}).get(key)
        if meta.get('code'):
            if tex is None:
                return None, f"the coded shader's image '{key}' is not " \
                             'among the prepared textures'
            filt = str(consts.get('tex_filter', 'NEAREST'))
            wrap = 'REPEAT'
        else:
            if tex is None:
                return None, f"image '{key}' is not among the prepared " \
                             'textures'
            filt = resolve_tex_filter(meta.get('interpolation', 'Linear'),
                                      consts.get('tex_filter'))
            wrap = {'REPEAT': 'REPEAT', 'EXTEND': 'EXTEND', 'CLIP': 'CLIP',
                    'MIRROR': 'MIRROR'}.get(meta.get('extension', 'REPEAT'),
                                            'REPEAT')
        if filt not in SUPPORTED_TEX_FILTERS:
            return None, (f'the {filt} texture filter is not in the '
                          f'deferred pass yet')
        if filt == 'TRILINEAR':
            # the height pre-pass has no footprint field; the CPU height
            # image is exact by construction, so exotic-filtered heights
            # take that path instead of a wrong mip
            return '__CPU__', {'cpu': True, 'node': bump_node.get('id'),
                               'why': 'a TRILINEAR-filtered height chain '
                                      'evaluates on the CPU into the '
                                      'height pre-pass',
                               'samplers': [], 'textures': {},
                               'frame_uniforms': [], 'uses_screen': False}
        tex_fns.append(_texture_sampler(sname, tex, filt, wrap))
        tex_binds[sname] = key
        replacements.append((f'texture({sname},', f'hal_sample_{sname}('))
    for old, new in replacements:
        src = src.replace(old, new)
    inline_parts = list(em.inline)
    if replacements:
        inline_parts = [t if 'texture(' not in t else
                        _replace_all(t, replacements) for t in inline_parts]
    if 'hal_T' in src:
        return None, 'tangent texture coordinates are not in the ' \
                     'G-buffer yet (UV and generated coordinates are)'
    gen_fns = ''
    gen_line = ''
    if 'hal_generated' in src:
        bounds = consts.get('obj_bounds')
        if bounds is None:
            return None, 'generated coordinates need the per-object ' \
                         'bounds the caller did not supply'
        lo, span = bounds

        def _sel(name, rows):
            lines = [f'vec3 {name}(float obj)', '{']
            for i in range(len(rows) - 1):
                lines.append(f'    if (obj < {_f(i + 0.5)}) '
                             f'return {_v3(rows[i])};')
            lines.append(f'    return {_v3(rows[len(rows) - 1])};')
            lines.append('}')
            return '\n'.join(lines)

        gen_fns = _sel('hal_gen_lo', list(lo)) + '\n' \
            + _sel('hal_gen_span', list(span)) + '\n'
        gen_line = ('    vec3 hal_generated = (P - hal_gen_lo(td.y)) '
                    '/ hal_gen_span(td.y);\n')

    frame_unis = sorted(em.frame_uniforms)
    extra_unis = ''.join(f'uniform float {u};\n' for u in frame_unis)
    # affine texture mode reaches the height pre-pass too: the CPU's
    # bump fields interpolate uv by the screen-linear barycentrics
    # (bump_field_source carries gbuf.bary_lin), so the pre-pass reads
    # the same warp -- uv only, exactly as the main pass
    affine_uv = ''
    if consts.get('affine'):
        affine_uv = (
            '    vec4 hal_idslin = texture(hal_gb_idslin, vUV);\n'
            '    f.uv = hal_interp(f.tri, hal_idslin.rgb, 2).xy;\n'
            '    f.uv2 = hal_interp4(f.tri, hal_idslin.rgb, 2).zw;\n')
    parts = [GB.GLSL, gen_fns, _block(inline_parts), _block(tex_fns),
             f"""
in vec2 vUV;
out vec4 Color;
uniform vec3 hal_eye;
{'uniform sampler2D hal_gb_idslin;' if affine_uv else ''}
{extra_unis}

void main()
{{
    HalcyonFragment f = hal_read_gbuffer(vUV);
    vec4 td = hal_tri_data(max(f.tri, 0.0));
    float keep = (f.covered && abs(td.x - {_f(mat_id)}) < 0.5) ? 1.0 : 0.0;
    // EARLY OUT on the ownership mask. Every material pass draws the
    // full screen; this shader used to shade EVERY pixel and multiply
    // by keep at the end -- harmless while shading was ALU, catastrophic
    // once it carried BVH loops: an M-material frame ran the radiosity
    // gather, the AO rays and every ray-shadow tap M times per pixel.
    // The field measured it: 5.0s of a 5.8s frame. A keep=0 pixel wrote
    // exactly (0,0,0,0) before and writes exactly (0,0,0,0) now, so the
    // picture cannot move by a bit; nothing downstream reads implicit
    // derivatives (mips ride the explicit footprint field), so the
    // divergent return is safe by construction.
    if (keep < 0.5) {{
        Color = vec4(0.0, 0.0, 0.0, 0.0);
        return;
    }}
{affine_uv}    vec3 P = f.P;
    vec3 V = normalize(hal_eye - P);
    vec3 N0 = normalize(f.N);
    vec3 hal_P = P;
    vec3 hal_N = N0;
    vec3 hal_V = V;
    vec2 hal_uv = f.uv;
    vec2 hal_uv2 = f.uv2;
"""]
    if gen_line:
        parts.append(gen_line)
    if 'hal_vcol' in src:
        parts.append('    vec4 hal_vcol = hal_interp4(f.tri, f.bary, 3);\n')
    parts.append(src)
    parts.append(f'    Color = vec4({expr}, 0.0, 0.0, 1.0) * keep;\n}}')
    return ''.join(parts), {'samplers': sorted(tex_binds)
                            + (['hal_gb_idslin'] if affine_uv else []),
                            'textures': tex_binds,
                            'frame_uniforms': frame_unis,
                            'uses_screen': em.used_screen}


def assemble_frame(graph, mat_id, model_index, bake, lights, consts,
                   shadows=None, textures=None, programs=None,
                   secondary=False, layer=False, vertex_rate=None):
    """One material's full-screen deferred pass, as complete GLSL.

    `vertex_rate` ('VERTEX' or 'FACE') assembles the Gouraud/flat
    variant: NO lighting is emitted at all. The CPU lights the corners
    of every triangle over a white surface -- shadows, rays, env, the
    model's own formula, everything shade_batch runs -- and the pass
    fetches the three corner colours from the `hal_vlight` texture,
    interpolates them by the G-buffer's own barycentrics, and
    multiplies by the per-pixel albedo chain. MODULATE, the
    fixed-function combiner: the reason a Gouraud-shaded period model
    has soft banded light over a sharp texture, now with the driver
    doing the per-pixel half and the CPU the per-vertex half.

    `secondary=True` assembles the pass for REFLECTION HIT points instead
    of camera fragments. Almost nothing changes -- the CPU shades hits
    with the same camera-eye V (`ctx.I` is always `P - eye`) -- except the
    backface override: `trace()` builds its context with `front=None`, so
    `surf.backfacing` never sets and the override is inert on hits. The
    secondary pass emits no backface block for the same reason.

    `layer=True` assembles the TRANSPARENT-LAYER variant: the same lit
    surface, but the pass writes the material's REAL alpha (opacity,
    threshold, edge-opacity blend -- shade_batch's own chain) instead of
    the coverage flag, premultiplied by keep so disjoint materials merge
    additively into one layer image.

    `bake` carries the surface constants the probe harvested (BAKE_FIELDS
    plus 'specular', 'ambient', 'emission', and 'diffuse' when there is no
    graph to compute it). `consts` carries the frame constants: eye,
    ambient_color, two_sided, specular_in_gamma, clamp_specular, light_clamp,
    falloff_default, shadow_samples. `shadows` is a list parallel to
    `lights`: None for an unshadowed light, or the meta dict its shadow atlas
    was packed with.

    Returns (source, binds) or (None, why). `binds` carries 'samplers' --
    every sampler name the source declares beyond the G-buffer's own, in
    binding order -- and 'textures', mapping the image sampler names to the
    prepared-texture keys the caller must upload.
    """
    from . import gbuffer as GB

    em = Emitter(graph or {})
    em.frame_mode = True
    em.resolution = consts.get('resolution')
    em.secondary = secondary
    em.camera = consts.get('camera')
    em.uv_names = tuple(consts.get('uv_names') or ())
    # {} is authoritative "nothing compiled" (code nodes read as zeros, the
    # CPU's own answer); the frame path always knows, so it never passes None
    em.programs = programs if programs is not None else {}
    base = None
    body = ''
    perpix = {}
    perpix_exprs = {}
    normal_expr = None
    bump_expr = None
    link = find_surface_link(graph)
    if link is not None:
        try:
            var, vt = em.output(link[0], link[1])
            base = em.cast(var, vt, 'vec3')
            # linked surface-parameter sockets on the master node: their
            # chains are emitted here, through the same emitter -- shared
            # subexpressions and all -- and assigned per pixel below
            perpix = per_pixel_fields(graph)
            mnode = master_node(graph)
            for field, (sockname, gtype) in perpix.items():
                want = 'float' if gtype == 'float' else 'vec3'
                perpix_exprs[field] = em.input(mnode, sockname, want)
            # the master node's Normal chain, through the same emitter so a
            # texture feeding both the colour and a Normal Map is sampled
            # once, exactly as the evaluator's node cache does it
            nsock = _socket(mnode, 'Normal')
            if nsock is not None and nsock.get('link'):
                normal_expr = em.input(mnode, 'Normal', 'vec3')
                if _socket(mnode, 'Bump Strength') is not None:
                    bump_expr = em.input(mnode, 'Bump Strength', 'float')
                # a node missing the socket bends at full strength on the
                # CPU (_opt defaults to 1.0); emitting nothing does the same
            body = em.body()
        except Unsupported as exc:
            missing = ', '.join(sorted(em.unsupported)) or str(exc)
            return None, f'no GLSL emitter for {missing}'
    if base is None:
        base = _v3(bake.get('diffuse', (0.8, 0.8, 0.8)))

    # image textures: the CPU samples the *prepared* pixels -- resized,
    # quantised, colourspace-converted -- so those exact pixels travel, and
    # the filter arithmetic is reproduced in the shader rather than left to
    # the driver's sampler state
    for node in (graph or {}).get('nodes', {}).values():
        if node.get('bl_idname') == 'ShaderNodeTexImage' and \
                node.get('props', {}).get('projection', 'FLAT') != 'FLAT':
            return None, (f"{node['props']['projection']} projection is not "
                          f"in the deferred pass yet (FLAT is)")
    tex_fns = []
    tex_binds = {}
    tex_binds_mip = {}
    needs_uvgrad = []
    src = body + ('\n' if body else '')
    replacements = []
    for meta in em.samplers:
        sname = meta['uniform']
        key = meta.get('image')
        tex = (textures or {}).get(key)
        if meta.get('code'):
            # coded-shader images sample with the scene's filter and REPEAT
            # wrap, exactly the SCtx the evaluator hands the program. A
            # missing image breaks the node on the CPU too, and the probe
            # refuses those frames before this code ever runs
            if tex is None:
                return None, f"the coded shader's image '{key}' is not " \
                             'among the prepared textures'
            filt = str(consts.get('tex_filter', 'NEAREST'))
            wrap = 'REPEAT'
        else:
            if tex is None:
                return None, f"image '{key}' is not among the prepared " \
                             'textures'
            filt = resolve_tex_filter(meta.get('interpolation', 'Linear'),
                                      consts.get('tex_filter'))
            wrap = {'REPEAT': 'REPEAT', 'EXTEND': 'EXTEND', 'CLIP': 'CLIP',
                    'MIRROR': 'MIRROR'}.get(meta.get('extension', 'REPEAT'),
                                            'REPEAT')
        if filt not in SUPPORTED_TEX_FILTERS:
            return None, (f'the {filt} texture filter is not in the '
                          f'deferred pass yet')
        if filt == 'TRILINEAR':
            # the footprint rules, exactly the CPU's: a raw flat UV
            # lookup on a SCREEN point filters with the mip footprint; a
            # linked Vector chain, a coded-shader image, or a ray hit
            # (secondary pass -- no pixel footprint) samples the top
            # level, which is what lod=None does on the CPU
            fp = bool(meta.get('footprint')) and not meta.get('code') \
                and not secondary
            if fp and layer:
                return None, ('the TRILINEAR footprint is not in the '
                              'layer passes yet (the opaque frame has '
                              'it); this glass shades on the CPU')
            if fp:
                if not needs_uvgrad:
                    tex_fns.append('uniform sampler2D hal_uvgrad;\n')
                _atlas_px, levels = mip_atlas(tex)
                tex_fns.append(_mip_sampler(
                    sname, tex, wrap, levels,
                    int(consts.get('tex_aniso', 1) or 1),
                    float(consts.get('tex_mip_bias', 0.0) or 0.0)))
                tex_binds_mip[sname] = key
                needs_uvgrad.append(sname)
            else:
                tex_fns.append(_texture_sampler(sname, tex, 'BILINEAR',
                                                wrap))
                tex_binds[sname] = key
        else:
            tex_fns.append(_texture_sampler(sname, tex, filt, wrap))
            tex_binds[sname] = key
        # the emitter sampled with texture(); the frame pass samples with
        # the arithmetic above, so the same pixel comes back on any driver
        replacements.append((f'texture({sname},', f'hal_sample_{sname}('))
    for old, new in replacements:
        src = src.replace(old, new)
    # coded shaders call texture() inside their own inlined functions, so
    # the same rewrite runs over the inline blocks too
    inline_parts = list(em.inline)
    if replacements:
        inline_parts = [t if 'texture(' not in t else
                        _replace_all(t, replacements) for t in inline_parts]
    if 'hal_T' in src:
        return None, 'tangent texture coordinates are not in the G-buffer ' \
                     'yet (UV and generated coordinates are)'
    # generated coordinates: Blender normalises them over each object's own
    # bounding box, and those bounds are per-scene constants -- so they bake
    # as a pair of lookup functions keyed by the object index the tri_data
    # texture already carries. Exactly ctx.generated = (P - lo[obj])/span
    gen_fns = ''
    gen_line = ''
    if 'hal_generated' in src:
        bounds = consts.get('obj_bounds')
        if bounds is None:
            return None, 'generated coordinates need the per-object bounds ' \
                         'the caller did not supply'
        lo, span = bounds

        def _sel(name, rows):
            lines = [f'vec3 {name}(float obj)', '{']
            for i in range(len(rows) - 1):
                lines.append(f'    if (obj < {_f(i + 0.5)}) '
                             f'return {_v3(rows[i])};')
            lines.append(f'    return {_v3(rows[len(rows) - 1])};')
            lines.append('}')
            return '\n'.join(lines)

        gen_fns = _sel('hal_gen_lo', list(lo)) + '\n' \
            + _sel('hal_gen_span', list(span)) + '\n'
        gen_line = ('    vec3 hal_generated = (P - hal_gen_lo(td.y)) '
                    '/ hal_gen_span(td.y);\n')

    # Bump nodes recorded height pre-passes during the walk: each height
    # chain becomes its own full-screen pass whose target the main pass
    # reads by texelFetch. Assembled here so a chain the pre-pass cannot
    # carry refuses the whole material, by name, before anything draws.
    prepasses = []
    for k, bnode in enumerate(em.bump_passes):
        psrc, pinfo = _assemble_height_pass(graph, mat_id, bnode, consts,
                                            textures, programs)
        if psrc is None:
            return None, pinfo
        prepasses.append((f'hal_bump{k}', psrc, pinfo))

    # the environment-reflection term: sphere-map the world along R, as
    # shade_batch adds it after everything else. The world spec was decided
    # by the plan (solid, blend, or an environment texture); its sampler
    # joins the material's own so the same prepared pixels travel
    env_lines = []
    env_spec = consts.get('env')
    if env_spec and float(bake.get('reflect', 0.0)) > 1e-4:
        refl = float(bake.get('reflect', 0.0))    # raw, as surf.reflect is
        rcol = _v3(bake.get('reflect_color', (1, 1, 1)))
        env_lines.append('    vec3 hal_R = reflect(-V, Nsurf);')
        if env_spec[0] in ('SKY_GRAD', 'SKY_BANDS'):
            env_lines += _sky_env_lines(env_spec)
        elif env_spec[0] in ('SOLID', 'SKY_SOLID'):
            env_lines.append(f'    vec3 hal_env = {_v3(env_spec[1])};')
        elif env_spec[0] == 'BLEND':
            env_lines += [
                '    float hal_et = clamp(hal_R.z * 0.5 + 0.5, 0.0, 1.0);',
                f'    vec3 hal_env = {_v3(env_spec[1])} + '
                f'({_v3(env_spec[2])} - {_v3(env_spec[1])}) * hal_et;']
        else:
            key = env_spec[1]
            tex = (textures or {}).get(key)
            if tex is None:
                return None, 'the environment image is not among the ' \
                             'prepared textures'
            tex_fns.append(_texture_sampler('hal_env_tex', tex, 'BILINEAR',
                                            'EXTEND'))
            tex_binds['hal_env_tex'] = key
            if env_spec[0] == 'MIRRORBALL':
                env_lines += [
                    '    float hal_em = 2.0 * sqrt(max(hal_R.x * hal_R.x + '
                    'hal_R.y * hal_R.y + (hal_R.z + 1.0) * (hal_R.z + 1.0)'
                    ', 1e-8));',
                    '    vec2 hal_euv = vec2(hal_R.x / hal_em + 0.5, '
                    'hal_R.y / hal_em + 0.5);']
            else:
                env_lines += [
                    '    vec2 hal_euv = vec2('
                    'atan(hal_R.y, -hal_R.x) / 6.28318530717959 + 0.5, '
                    'atan(hal_R.z, sqrt(max(hal_R.x * hal_R.x + '
                    'hal_R.y * hal_R.y, 1e-12))) / 3.14159265358979 + 0.5);']
            env_lines.append('    vec3 hal_env = '
                             'hal_sample_hal_env_tex(hal_euv).rgb;')
        env_lines.append(f'    total += hal_env * ({_f(refl)} * s.specular'
                         f' * {rcol});')

    vlight_fns = ''
    vlight_spec = None
    if vertex_rate:
        tcount = int(consts.get('tri_count', 0))
        if tcount <= 0:
            return None, ('vertex-rate lighting needs the triangle count '
                          'the caller did not supply')
        import math
        vside = int(math.ceil(math.sqrt(float(max(tcount * 3, 1)))))
        # the same fetch-by-arithmetic every packed texture here uses:
        # texel centres, side baked as a literal
        vlight_fns = (
            'uniform sampler2D hal_vlight;\n'
            'vec3 hal_vlight_fetch(float i)\n'
            '{\n'
            f'    float x = mod(i, {_f(float(vside))});\n'
            f'    float y = floor(i / {_f(float(vside))});\n'
            '    return texture(hal_vlight, (vec2(x, y) + vec2(0.5)) / '
            f'{_f(float(vside))}).rgb;\n'
            '}\n')
        vlight_spec = {'rate': str(vertex_rate), 'side': int(vside),
                       'mat': int(mat_id)}
        env_lines = []                     # the corners carry the env term

    two_sided = bool(consts.get('two_sided', True))
    shadows = shadows or [None] * len(lights)
    shadow_fns = []
    samplers = []
    # CONSTANT / WIREFRAME: light_surface returns before the light loop,
    # the ambient term and every surface cheat -- full-bright albedo x
    # diffuse level, plus emission. The pass carries no lighting support
    # at all (the wires themselves are carved by apply_wireframe on the
    # readback -- the CPU's own separable stage, fog-doctrine style).
    # env and rays stay: the CPU applies those AFTER light_surface.
    shadeless = bool(bake.get('__shadeless'))
    if vertex_rate or shadeless:
        # the pass lights nothing, so it carries none of the lighting
        # support: no shadow taps, no BVH traversal, no AO -- for a
        # vertex-rate pass all of it already lives in the CPU-lit
        # corner values; for a shadeless one it never existed
        shadows = [None] * len(lights)
    ray_any = any(s is not None and s.get('ray') for s in shadows)
    soft_any = any(s is not None and s.get('ray')
                   and int(s.get('samples', 1)) > 1 for s in shadows)
    ao_spec = None if (vertex_rate or shadeless) else consts.get('ao')
    rad_spec = None if (vertex_rate or shadeless) \
        else consts.get('radiosity')
    # interpolated mode: SCREEN passes read the grid field (a texel
    # fetch), so they carry no gather and no traversal of their own;
    # secondary passes gather fully -- a traced hit has no place in a
    # screen-space cache, exactly as the CPU shades its hits
    rad_field = bool(rad_spec) and not secondary and \
        int(rad_spec.get('spacing', 1)) > 1
    rad_gather = bool(rad_spec) and not rad_field
    if ray_any or ao_spec or rad_gather:
        # the shared traversal, once, ahead of every hal_shadow_vis (and
        # hal_ao / hal_rad) that calls it. The texture sides bake as
        # literals: the plan signature fingerprints the mesh, so a
        # changed BVH re-plans and re-bakes.
        sides = consts.get('bvh_sides') or {}
        if 'hal_bvh_side' not in sides or 'hal_btris_side' not in sides:
            return None, 'ray shadows need the BVH textures the caller ' \
                         'did not pack'
        from .rtrace import INTERSECT_GLSL, TRAVERSE_GLSL
        trav = TRAVERSE_GLSL
        if rad_gather:
            # the gather needs the CLOSEST hit (id + t), not just
            # any-hit occlusion; the intersect kernel rides the same
            # texel fetchers the traversal just declared
            trav = trav + INTERSECT_GLSL
        for cname in ('hal_bvh_side', 'hal_btris_side'):
            trav = trav.replace(f'uniform float {cname};', '')
            trav = trav.replace(cname, _f(float(sides[cname])))
        shadow_fns.append(trav)
        samplers += ['hal_bvh', 'hal_btris']
    if soft_any or ao_spec or rad_gather:
        # the deterministic-sampling primitives: the pattern hash under a
        # sampling name (a material may inline the pattern library too),
        # and the shared unit-circle table
        shadow_fns.append(SAMPLING_GLSL)
        samplers.append('hal_circle')
    if ao_spec and not rad_spec:
        # radiosity supersedes plain AO exactly as light_surface does:
        # the gather is occlusion-aware by construction
        shadow_fns.append(_ao_function(ao_spec, consts))
    if rad_gather:
        shadow_fns.append(_rad_function(rad_spec, consts))
    if rad_field:
        shadow_fns.append(_rad_lookup_function(rad_spec, consts))
        samplers.append('hal_radfield')
    for i, smeta in enumerate(shadows):
        if smeta is not None:
            shadow_fns.append(_shadow_function(i, smeta, consts))
            if not smeta.get('ray'):
                samplers.append(f'hal_shadow{i}')
    if not vertex_rate:
        # projected light textures: one lookup function per cookie light.
        # A vertex-rate pass skips them for the same reason it skips the
        # whole loop -- the cookie is already IN the CPU-lit corner values
        for i, ckspec in sorted((consts.get('cookies') or {}).items()):
            shadow_fns.append(_cookie_function(i, ckspec))
            samplers.append(f'hal_cookie{i}')
    # the per-triangle auxiliary texture: STORED face normals + the
    # per-tri random, the CPU's own values baked (gbuffer.pack_tri_aux).
    # Wanted by Normal Source FACE (every pass) and by graphs reading
    # the Geometry node's True Normal / Random Per Island (the emitter
    # flags those). The fetch is the same packed-square arithmetic
    # every per-tri texture here uses; the side bakes as a literal.
    # affine texture mode: uv re-interpolates by the rasteriser's own
    # SCREEN-LINEAR barycentrics (hal_gb_idslin) -- uv ONLY, exactly
    # attributes(): P, N and colour stay perspective-correct. Ray hits
    # have no screen-linear bary on either device (the CPU's trace
    # passes bary_lin=None), so secondary passes keep true barycentrics.
    affine_uv = ''
    if consts.get('affine') and not secondary:
        samplers.append('hal_gb_idslin')
        affine_uv = (
            '    vec4 hal_idslin = texture(hal_gb_idslin, vUV);\n'
            '    f.uv = hal_interp(f.tri, hal_idslin.rgb, 2).xy;\n'
            '    f.uv2 = hal_interp4(f.tri, hal_idslin.rgb, 2).zw;\n')
    normal_face = bool(consts.get('normal_face'))
    needs_triaux = normal_face or bool(getattr(em, 'needs_triaux', False))
    triaux_fns = ''
    if needs_triaux:
        tcount = int(consts.get('tri_count', 0))
        if tcount <= 0:
            return None, ('the per-tri auxiliary texture needs the '
                          'triangle count the caller did not supply')
        import math
        aside = int(math.ceil(math.sqrt(float(max(tcount, 1)))))
        triaux_fns = (
            'uniform sampler2D hal_triaux;\n'
            'vec4 hal_triaux_fetch(float i)\n'
            '{\n'
            f'    float x = mod(i, {_f(float(aside))});\n'
            f'    float y = floor(i / {_f(float(aside))});\n'
            '    return texture(hal_triaux, (vec2(x, y) + vec2(0.5)) / '
            f'{_f(float(aside))});\n'
            '}\n')
        samplers.append('hal_triaux')
    if consts.get('stipple') and not secondary:
        # the Screen Door threshold map (see the composite below)
        samplers.append('hal_stipple')
    # the per-corner screen positions for the Wireframe node's Pixel
    # Size: (sx, sy, w) per triangle corner, the CPU's own sgrad-cache
    # projection baked -- same packed-square fetch as every data
    # texture here
    wirescreen_fns = ''
    if bool(getattr(em, 'needs_wirescreen', False)):
        tcount = int(consts.get('tri_count', 0))
        if tcount <= 0:
            return None, ('the wireframe screen texture needs the '
                          'triangle count the caller did not supply')
        import math as _math
        wside = int(_math.ceil(_math.sqrt(float(max(tcount * 3, 1)))))
        wirescreen_fns = (
            'uniform sampler2D hal_vscreen;\n'
            'vec4 hal_vscreen_fetch(float i)\n'
            '{\n'
            f'    float x = mod(i, {_f(float(wside))});\n'
            f'    float y = floor(i / {_f(float(wside))});\n'
            '    return texture(hal_vscreen, (vec2(x, y) + vec2(0.5)) / '
            f'{_f(float(wside))});\n'
            '}\n')
        samplers.append('hal_vscreen')
    frame_unis = sorted(em.frame_uniforms)
    extra_unis = ''.join(f'uniform float {u};\n' for u in frame_unis)
    # the interface declarations come FIRST: the sampling helpers and the
    # soft-shadow/AO functions read vUV inside function bodies, and GLSL
    # requires the declaration to precede the use in file order
    decls = ('in vec2 vUV;\n'
             'out vec4 Color;\n'
             '// the per-frame scalars that are NOT baked: baking the eye '
             'meant a moving\n'
             '// camera changed the source every frame, and every frame '
             'paid the driver\'s\n'
             '// shader compile. A still pays nothing for these; an orbit '
             'stops paying 20ms\n'
             '// -- and a coded shader reading the clock animates without '
             'recompiling\n'
             'uniform vec3 hal_eye;\n'
             + ('uniform sampler2D hal_gb_idslin;\n' if affine_uv else '')
             + ('uniform sampler2D hal_stipple;\n'
                if (consts.get('stipple') and not secondary) else '')
             + extra_unis)
    parts = [GS.GLSL, GS.DISPATCH, GB.GLSL, decls, gen_fns,
             _block(inline_parts), _block(tex_fns), _block(shadow_fns),
             vlight_fns, triaux_fns, wirescreen_fns, f"""
void main()
{{
    HalcyonFragment f = hal_read_gbuffer(vUV);
    vec4 td = hal_tri_data(max(f.tri, 0.0));
    float keep = (f.covered && abs(td.x - {_f(mat_id)}) < 0.5) ? 1.0 : 0.0;
    // EARLY OUT on the ownership mask. Every material pass draws the
    // full screen; this shader used to shade EVERY pixel and multiply
    // by keep at the end -- harmless while shading was ALU, catastrophic
    // once it carried BVH loops: an M-material frame ran the radiosity
    // gather, the AO rays and every ray-shadow tap M times per pixel.
    // The field measured it: 5.0s of a 5.8s frame. A keep=0 pixel wrote
    // exactly (0,0,0,0) before and writes exactly (0,0,0,0) now, so the
    // picture cannot move by a bit; nothing downstream reads implicit
    // derivatives (mips ride the explicit footprint field), so the
    // divergent return is safe by construction.
    if (keep < 0.5) {{
        Color = vec4(0.0, 0.0, 0.0, 0.0);
        return;
    }}
{affine_uv}    vec3 P = f.P;
    vec3 V = normalize(hal_eye - P);
    vec3 N0 = normalize(f.N);
    vec3 hal_P = P;
    vec3 hal_N = N0;
    vec3 hal_V = V;
    vec2 hal_uv = f.uv;
    vec2 hal_uv2 = f.uv2;
"""]
    if gen_line:
        parts.append(gen_line)
    if 'hal_vcol' in src:
        # three extra fetches, paid only by materials that read the paint
        parts.append('    vec4 hal_vcol = hal_interp4(f.tri, f.bary, 3);\n')
    parts.append(src)
    # The order the CPU actually runs: the graph evaluates against the
    # interpolated normal -- ctx.N, UNFLIPPED -- then closure_to_surface
    # lerps the chain's normal toward it by Bump Strength, and only
    # light_surface flips for two-sided lighting, testing the BENT normal
    # against V. Flipping first (as this shader once did) fed the chains a
    # normal the CPU never showed them, wrong on every back face.
    if normal_expr is not None:
        k = bump_expr if bump_expr is not None else '1.0'
        parts.append(f'    vec3 Nsurf = normalize(N0 + '
                     f'(normalize({normal_expr}) - N0) * ({k}));\n')
    else:
        parts.append('    vec3 Nsurf = N0;\n')
    if normal_face:
        # Normal Source FACE, the CPU's exact order: the graph just ran
        # against the INTERPOLATED normal (hal_N above), and only now
        # does the stored face normal replace the shading normal --
        # ctx.N = ctx.Ng, as render.py does it. The texel carries the
        # CPU's own normalized values; the two-sided flip and the
        # tangent frame below pick the replacement up exactly as
        # light_surface and shade_batch do.
        parts.append('    Nsurf = hal_triaux_fetch(max(f.tri, 0.0))'
                     '.xyz;\n')
    if two_sided:
        parts.append('    float side = (dot(Nsurf, V) < 0.0) ? -1.0 : 1.0;\n'
                     '    vec3 N = Nsurf * side;\n')
    else:
        parts.append('    vec3 N = Nsurf;\n')
    lines = ['    HalcyonSurface s;',
             f'    int hal_model_i = {int(model_index)};']
    for name in BAKE_FIELDS:
        if name in perpix_exprs:
            lines.append(f'    s.{name} = {perpix_exprs[name]};')
        else:
            lines.append(f'    s.{name} = {_f(bake.get(name, 0.0))};')
    lines += [
        # the same frame mathx.orthonormal_basis builds on the CPU -- and
        # from the same normal: shade_batch builds it from ctx.N, the bent
        # normal BEFORE the two-sided flip. Building it from the flipped N
        # negated the tangent on back faces, which anisotropy would notice.
        '    vec3 up = (abs(Nsurf.z) < 0.999) ? vec3(0.0, 0.0, 1.0)'
        ' : vec3(1.0, 0.0, 0.0);',
        '    s.tangent = normalize(cross(up, Nsurf));',
        '    s.bitangent = cross(Nsurf, s.tangent);',
    ]
    # Anisotropic Rotation turns the frame, exactly _aniso_frame's rotation
    # -- the term the GLSL silently dropped until the feature matrix put
    # every model on the driver and ANISOTROPIC came back 0.0627 off (405
    # px): the CPU's highlight sat 72 degrees from the GPU's. A baked
    # rotation lands as cos/sin LITERALS (same bits, no driver trig); a
    # per-pixel chain rotates with the driver's own cos/sin, which is
    # smooth (no decision cliff) and lands inside the deferred bar.
    rot_baked = float(bake.get('aniso_rot', 0.0) or 0.0)
    if 'aniso_rot' in perpix_exprs:
        lines += [
            '    {',
            '    float hal_ar = s.aniso_rot * 6.2831853071795862;',
            '    float hal_ca = cos(hal_ar);',
            '    float hal_sa = sin(hal_ar);',
            '    vec3 hal_t2 = s.tangent * hal_ca + s.bitangent * hal_sa;',
            '    s.bitangent = normalize(s.tangent * (-hal_sa) '
            '+ s.bitangent * hal_ca);',
            '    s.tangent = normalize(hal_t2);',
            '    }',
        ]
    elif abs(rot_baked) > 1e-6:
        import numpy as _np
        ca = float(_np.cos(rot_baked * 2.0 * _np.pi))
        sa = float(_np.sin(rot_baked * 2.0 * _np.pi))
        lines += [
            '    {',
            f'    vec3 hal_t2 = s.tangent * {_f(ca)} + s.bitangent '
            f'* {_f(sa)};',
            f'    s.bitangent = normalize(s.tangent * {_f(-sa)} '
            f'+ s.bitangent * {_f(ca)});',
            '    s.tangent = normalize(hal_t2);',
            '    }',
        ]
    if str(bake.get('__slot', '')) == 'specular':
        # the specular slot routing: a lone raw GLOSSY lobe's colour
        # chain is the SPECULAR colour (closure_to_surface puts it in
        # surf.specular), and the diffuse is the CPU's untouched flat
        # constant -- the exact struct the CPU shades from
        lines += [
            f'    s.diffuse = {_v3(bake.get("diffuse", (0.8, 0.8, 0.8)))};',
            f'    s.specular = {base};',
        ]
    else:
        lines += [
            f'    s.diffuse = {base};',
            (f'    s.specular = {perpix_exprs["specular"]};'
             if 'specular' in perpix_exprs else
             f'    s.specular = {_v3(bake.get("specular", (1, 1, 1)))};'),
        ]
    if vertex_rate:
        # Gouraud/flat: fetch the three CPU-lit corners of THIS pixel's
        # triangle, interpolate by the G-buffer's own perspective
        # barycentrics (the CPU interpolates the same ones), multiply by
        # the per-pixel albedo. Everything else the pixel path emits
        # below -- ambient, lights, clamp, emission, the silhouette
        # cheats, matcap, backface, env -- is already IN the corner
        # values, computed by the renderer's own CPU code
        lines += [
            '    float hal_vt = max(f.tri, 0.0) * 3.0;',
            '    vec3 hal_vl = hal_vlight_fetch(hal_vt) * f.bary.x',
            '        + hal_vlight_fetch(hal_vt + 1.0) * f.bary.y',
            '        + hal_vlight_fetch(hal_vt + 2.0) * f.bary.z;',
            '    vec3 total = s.diffuse * hal_vl;',
        ]
    elif shadeless:
        # light_surface's early return, verbatim: diffuse x level (+
        # emission, added by the shared block below). No ambient term,
        # no lights, no clamp, no silhouette cheats -- the CPU never
        # reaches apply_surface_effects for these models
        lines.append('    vec3 total = s.diffuse * s.diffuse_level;')
    elif rad_field:
        # interpolated radiosity: the pass blends the grid field the
        # pre-pass gathered -- a texel fetch where the full mode walks
        # the BVH
        lines.append(f'    vec3 total = s.diffuse * (hal_rad_lookup()'
                     f' * {_f(bake.get("ambient", 1.0))});')
    elif rad_spec:
        # the Radiosity checkbox: gathered ambient replaces the flat
        # term, exactly light_surface's branch -- and plain AO with it
        lines.append(f'    vec3 total = s.diffuse * (hal_rad(P, N)'
                     f' * {_f(bake.get("ambient", 1.0))});')
    else:
        lines.append(
            f'    vec3 total = s.diffuse * ({_v3(consts["ambient_color"])}'
            f' * {_f(bake.get("ambient", 1.0))});')
    if ao_spec and not rad_spec:
        # exactly light_surface's order: ambient occlusion scales the
        # ambient term (with the post-flip N) before any light adds
        lines.append('    total *= hal_ao(P, N);')
    if not vertex_rate and not shadeless \
            and float(bake.get('sheen', 0.0)) > 1e-4:
        lines.append('    float hal_edge_vn = clamp(1.0 - abs(dot(N, V)), '
                     '0.0, 1.0);')
    for i, light in enumerate(() if (vertex_rate or shadeless)
                               else lights):
        lines += _one_light_source(i, light, consts,
                                   shadowed=shadows[i], bake=bake)
    if not vertex_rate and not shadeless \
            and consts.get('clamp_specular', True):
        lines.append('    total = min(total, vec3(64.0));')
    if not vertex_rate:
        if 'emission' in perpix_exprs:
            lines.append(
                f'    total = total + {perpix_exprs["emission"]};')
        else:
            lines.append(f'    total = total + '
                         f'{_v3(bake.get("emission", (0, 0, 0)))};')
    # the era's silhouette cheats, exactly as apply_surface_effects: applied
    # after emission, outside the reflectance model, the same on every model
    if not vertex_rate and not shadeless and \
            (float(bake.get('fresnel', 0.0)) > 1e-4 or
             float(bake.get('rim', 0.0)) > 1e-4):
        lines.append('    float hal_facing = clamp(abs(dot(N, V)), 0.0, 1.0);')
        lines.append('    float hal_sil = 1.0 - hal_facing;')
    if not vertex_rate and not shadeless \
            and float(bake.get('fresnel', 0.0)) > 1e-4:
        fp = max(float(bake.get('fresnel_power', 3.0)), 0.01)
        lines.append(
            f'    total += {_v3(bake.get("fresnel_color", (1, 1, 1)))}'
            f' * (pow(hal_sil, {_f(fp)}) * {_f(bake["fresnel"])}'
            f' * {_f(bake.get("specular_level", 0.5))});')
    if not vertex_rate and not shadeless \
            and float(bake.get('rim', 0.0)) > 1e-4:
        rp = max(float(bake.get('rim_power', 3.0)), 0.01)
        lines.append(
            f'    total += {_v3(bake.get("rim_color", (1, 1, 1)))}'
            f' * (pow(hal_sil, {_f(rp)}) * {_f(bake["rim"])});')
    # matcap: the whole lit result lerps toward one colour, exactly as
    # apply_surface_effects -- after fresnel and rim, before the backface
    mk = min(max(float(bake.get('matcap_blend', 0.0)), 0.0), 1.0)
    if vertex_rate or shadeless:
        mk = 0.0            # in the corners already / never applied
    if mk > 1e-4:
        mc = perpix_exprs.get('matcap',
                              _v3(bake.get('matcap', (0, 0, 0))))
        lines.append(f'    total = total * {_f(1.0 - mk)} + '
                     f'({mc}) * {_f(mk)};')
    # the backface override: the rasteriser decides front by projected
    # winding, and for a perspective camera that is exactly the plane-side
    # test against the eye -- computed from the corner positions the
    # G-buffer already carries. front is (plane < 0); backfacing the rest
    kb = min(max(float(bake.get('backface_mix', 0.0)), 0.0), 1.0)
    if secondary:
        kb = 0.0                   # trace() shades hits with front=None
    if vertex_rate or shadeless:
        kb = 0.0           # in the corners already / never applied
    if kb > 1e-4:
        lines += [
            '    vec3 hal_bf_p0 = hal_fetch_attr(f.tri, 0, 0).xyz;',
            '    vec3 hal_bf_pl = cross('
            'hal_fetch_attr(f.tri, 1, 0).xyz - hal_bf_p0, '
            'hal_fetch_attr(f.tri, 2, 0).xyz - hal_bf_p0);',
            '    float hal_backfacing = '
            '(dot(hal_bf_pl, hal_eye - hal_bf_p0) < 0.0) ? 0.0 : 1.0;',
            f'    float hal_bk = {_f(kb)} * hal_backfacing;',
            '    total = total * (1.0 - hal_bk) + '
            f'{_v3(bake.get("backface_color", (0, 0, 0)))} * hal_bk;',
        ]
    lines += env_lines
    if layer:
        # a TRANSPARENT LAYER writes its real alpha -- the same chain
        # shade_batch runs for SORTED/ABUFFER fragments: opacity clamped,
        # the hard threshold, then the edge-opacity silhouette blend.
        # rgb and alpha premultiply by KEEP only, so per-pixel-disjoint
        # materials merge into one target as a plain sum (the driver
        # composites them under ALPHA_PREMULT, which for disjoint
        # writes IS that sum)
        aexpr = perpix_exprs.get('opacity',
                                 _f(float(bake.get('opacity', 1.0))))
        lines.append(f'    float hal_alpha = clamp({aexpr}, 0.0, 1.0);')
        thr = float(consts.get('alpha_threshold', 0.0))
        if thr > 0.0:
            lines.append(f'    hal_alpha = (hal_alpha >= {_f(thr)}) '
                         '? hal_alpha : 0.0;')
        eo = float(bake.get('edge_opacity', 1.0))
        if abs(eo - 1.0) > 1e-4:
            fp = max(float(bake.get('fresnel_power', 3.0)), 0.01)
            lines += [
                # the CPU tests ctx.N -- the BENT normal, before the
                # two-sided flip (abs() makes the flip moot) -- so a
                # Normal-Map material's silhouette matches: Nsurf, not N0
                '    float hal_eo_f = clamp(abs(dot(Nsurf, V)), 0.0, 1.0);',
                f'    float hal_eo_t = pow(1.0 - hal_eo_f, {_f(fp)});',
                '    hal_alpha = clamp(hal_alpha * (1.0 - hal_eo_t) + '
                f'{_f(eo)} * hal_eo_t, 0.0, 1.0);',
            ]
        lines.append('    Color = vec4(total * keep, hal_alpha * keep);')
    elif consts.get('stipple') and not secondary:
        # Screen Door: shade_batch's own chain -- clamp, the hard
        # cutoff, then keep-or-drop against the CPU's threshold map
        # (hal_stipple carries threshold_map(pattern, 64, 64) verbatim;
        # the CPU indexes it py % 64, px % 64). The alpha channel
        # doubles as the target's ownership flag, so the stipple bit
        # rides ENCODED above the coverage floor: 0.6 = covered but
        # dropped, 0.9 = covered and kept -- the readback decodes
        # coverage at > 0.5 and the stipple bit at > 0.75, and rgb
        # stays the full shaded colour either way, exactly the CPU's
        # (rgb shaded, alpha 0/1) split. Ray HITS never stipple: the
        # CPU gates on ctx.px, which a hit does not have.
        aexpr = perpix_exprs.get('opacity',
                                 _f(float(bake.get('opacity', 1.0))))
        lines.append(f'    float hal_alpha = clamp({aexpr}, 0.0, 1.0);')
        thr = float(consts.get('alpha_threshold', 0.0))
        if thr > 0.0:
            lines.append(f'    hal_alpha = (hal_alpha >= {_f(thr)}) '
                         '? hal_alpha : 0.0;')
        rw, rh = consts.get('resolution', (1.0, 1.0))
        lines += [
            f'    float hal_spx = mod(floor(vUV.x * {_f(float(rw))}), '
            '64.0);',
            f'    float hal_spy = mod(floor(vUV.y * {_f(float(rh))}), '
            '64.0);',
            '    float hal_sthr = texture(hal_stipple, '
            '(vec2(hal_spx, hal_spy) + vec2(0.5)) / 64.0).r;',
            '    hal_alpha = (hal_alpha > hal_sthr) ? 1.0 : 0.0;',
            '    Color = vec4(total, 0.6 + 0.3 * hal_alpha) * keep;',
        ]
    else:
        lines.append('    Color = vec4(total, 1.0) * keep;')
    lines.append('}')
    parts.append('\n'.join(lines))
    info = {'samplers': samplers
            + (['hal_vlight'] if vlight_spec is not None else [])
            + sorted(tex_binds) + sorted(tex_binds_mip)
            + (['hal_uvgrad'] if needs_uvgrad else [])
            + [p[0] for p in prepasses],
            'textures': tex_binds,
            'textures_mip': tex_binds_mip,
            'needs_uvgrad': bool(needs_uvgrad),
            'needs_wirescreen': bool(getattr(em, 'needs_wirescreen',
                                             False)),
            'frame_uniforms': frame_unis,
            'prepasses': prepasses,
            'uses_screen': em.used_screen
            or any(p[2].get('uses_screen') for p in prepasses)}
    if vlight_spec is not None:
        info['vlight'] = vlight_spec
    return ''.join(parts), info
