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
            '    vec3 spec = s.specular * s.specular_level * ds.yzw;',
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
