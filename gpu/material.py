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
from . import combine as GCB         # R251 material pack (MAT-A)
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
            '    if (kind == 0 || kind == 3) {',
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
            '    vec4 ds = (kind == 3)',
            '        ? vec4(0.5 * dot(N, L) + 0.5, s.specular',
            '               * pow(max(0.5 * dot(N, normalize(L + V))',
            '                         + 0.5, 0.0), s.glossiness))',
            '        : hal_evaluate(hal_model, s, N, L, V);',
            '    vec3 radiance = lcol * energy * atten * (1.0 / 3.14159265);',
            # R243: a Max shader whose diffuse carries its own colour
            # leaves it in hal_dif_rgb (the hemi override never runs
            # hal_evaluate, so it reads the diffuse socket)
            '    vec3 dcol = (kind == 3) ? s.diffuse : hal_dif_rgb;',
            '    vec3 diff = dcol * s.diffuse_level * ds.x;',
            '    // the specular colour is already in ds.yzw: hal_evaluate',
            '    // folds it, as the CPU does, so Metal can tint by diffuse',
            # R243: the level-free models (Strauss, Multi-Layer) scale by
            # nothing here, exactly the CPU loop
            f'    float lvl = ({_level_free_test("hal_model")}) ? 1.0 '
            ': s.specular_level;',
            '    vec3 spec = lvl * ds.yzw;',
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
              'translucency', 'toon_size', 'toon_smooth', 'toon_size2',
              'toon_smooth2', 'bi_fresnel', 'bi_fresnel_fac', 'bi_slope',
              'bi_transp_fresnel', 'bi_transp_blend', 'bi_spectra',
              'bi_cubic', 'bi_tangent', 'shadow_receive', 'cast_only',
              'shadows_only', 'opacity')
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
uniform float hal_toon_size2;
uniform float hal_toon_smooth2;
uniform float hal_bi_fresnel;
uniform float hal_bi_fresnel_fac;
uniform float hal_bi_slope;
uniform float hal_bi_transp_fresnel;
uniform float hal_bi_transp_blend;
uniform float hal_bi_spectra;
uniform float hal_bi_cubic;
uniform float hal_bi_tangent;
uniform float hal_shadow_receive;
uniform float hal_cast_only;
uniform float hal_shadows_only;
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


def _level_free_test(var):
    """R243: a GLSL test for the models the light loop must not scale by
    Specular Level (shading.LEVEL_FREE_MODELS), by index."""
    from ..core.shading import LEVEL_FREE_MODELS, MODEL_ITEMS
    idx = [k for k, m in enumerate(MODEL_ITEMS) if m[0] in LEVEL_FREE_MODELS]
    return ' || '.join(f'{var} == {k}' for k in idx) or 'false'


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


def declaration_order_violations(source):
    """Function names CALLED before any declaration of them: [(name,
    call line, declaration line)].

    Real GLSL compilers require every function declared before use;
    Halcyon's own front-end and simulator resolve calls by NAME at run
    time and shrug at order. That one divergence cost the field five
    rounds: the bitex chunk called the okramp chunk's HSV pair, the
    okramp chunk landed later in the assembly, and the driver rejected
    every material carrying a Blender Internal texture node while
    every headless check passed. This scan is the wall against the
    whole class -- run it over ASSEMBLED pass sources in tests.
    """
    import re
    types_ = r'(?:void|float|int|uint|bool|[iub]?vec[234]|mat[234])'
    decl_re = re.compile(rf'^\s*{types_}\s+(\w+)\s*\(', re.M)
    lines = source.splitlines()
    declared_at = {}
    decl_lines = set()
    for m in decl_re.finditer(source):
        ln = source.count('\n', 0, m.start()) + 1
        declared_at.setdefault(m.group(1), ln)
        decl_lines.add((m.group(1), ln))
    call_re = re.compile(r'\b(\w+)\s*\(')
    out = []
    flagged = set()
    for i, line in enumerate(lines, 1):
        code = line.split('//', 1)[0]
        for m in call_re.finditer(code):
            name = m.group(1)
            d = declared_at.get(name)
            if d is None or name in flagged:
                continue
            if i < d and (name, i) not in decl_lines:
                out.append((name, i, d))
                flagged.add(name)
    return out


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


def _mv(consts, x):
    """A per-MATERIAL value literal, marked for the texel lifter.

    R174: material constants baked as literals made every material its
    own shader -- 25 compiles, and every slider drag a recompile. Under
    consts['__mark_values'] the value is wrapped in a hal_MV() marker;
    the lifter in shade.py pulls every marker into the hal_mats texture
    and materials with the same STRUCTURE share one compiled shader.
    Values that GATE code generation (a sheen crossing zero, a shadeless
    flag) are read python-side and stay structure, exactly as before.
    Non-finite values stay literal."""
    s = _f(x)
    try:
        f = float(x)
    except (TypeError, ValueError):
        return s
    if consts.get('__mark_values') and f == f and abs(f) != float('inf'):
        return f'hal_MV({s})'
    return s


def _mv3(consts, t):
    return 'vec3({}, {}, {})'.format(*(_mv(consts, v) for v in t))


_MV_RE = None


def _lift_marked_values(src):
    """Replace every hal_MV(<literal>) with hal_mv(<index>); return values.

    The marker body is always a plain float literal (written by _f or the
    Emitter), so a paren-free regex is exact. Index order is source
    order, which is deterministic for a given structure -- the atlas row
    and the shader agree by construction."""
    global _MV_RE
    if _MV_RE is None:
        import re
        _MV_RE = re.compile(r'hal_MV\(([^()]*)\)')
    vals = []

    def _sub(m):
        vals.append(float(m.group(1)))
        return f'hal_mv({len(vals) - 1})'

    return _MV_RE.sub(_sub, src), vals


def _object_frame(src, consts):
    """R243: the object's own frame on the GPU. `hal_object` reads the
    world position through the per-object inverse matrix, baked as
    three row-lookup functions by object index -- exactly the CPU's
    n_tex_coord einsum (inv[:3, :3] . P + inv[:3, 3]). Returns the
    functions and the line, or ('', '') when no chain reads it, or
    (None, why) when the caller supplied no matrices."""
    if 'hal_object' not in src:
        return '', ''
    mats = consts.get('obj_inv')
    if mats is None:
        return None, 'object coordinates need the per-object matrices ' \
                     'the caller did not supply'
    rows = [[], [], []]
    for m in mats:
        for r in range(3):
            rows[r].append(tuple(float(x) for x in m[r][:4]))

    def _sel(name, vals):
        lines = [f'vec4 {name}(float obj)', '{']
        for i in range(len(vals) - 1):
            lines.append(f'    if (obj < {_f(i + 0.5)}) return vec4('
                         + ', '.join(_f(x) for x in vals[i]) + ');')
        lines.append('    return vec4(' + ', '.join(_f(x) for x in vals[-1]) + ');')
        lines.append('}')
        return '\n'.join(lines)
    fns = '\n'.join(_sel(f'hal_obj_r{r}', rows[r]) for r in range(3)) + '\n'
    line = ('    vec3 hal_object = vec3(dot(hal_obj_r0(td.y), vec4(P, 1.0)), '
            'dot(hal_obj_r1(td.y), vec4(P, 1.0)), '
            'dot(hal_obj_r2(td.y), vec4(P, 1.0)));\n')
    return fns, line


def _v3(t):
    t = tuple(float(v) for v in t)[:3]
    return 'vec3({}, {}, {})'.format(*(_f(v) for v in t))


#: texels per light in the hal_lights texture (R169). Layout:
#:   t0 = (position.xyz,            energy_pre [signed; /4pi for
#:                                   point-family, raw for SUN/HEMI])
#:   t1 = (L or spot dir .xyz,      spotsi = cos(spot_size/2))
#:   t2 = (colour.rgb,              spotbl = (1 - spotsi) * blend)
#:   t3 = (decay t3x, t3y, ld1, ld2) -- per decay mode: CUSTOM stores
#:        (start, end - start); the BI modes and SPHERE store
#:        (start, D = max(end, eps)); NONE/INVERSE/default (start, 0)
#: The values a user tweaks while lighting live HERE, not in shader
#: source: a lamp edit re-uploads one tiny texture instead of
#: recompiling every material pass (the field's 45-second viewport
#: refine and 30-second cold F12 were exactly that recompile storm).
#: STRUCTURE -- light count, types, decay modes, shadow modes, which
#: sliders are nonzero -- still bakes, still re-plans when it changes.
#: R251 LIGHT-B1: 8 texels per light -- 4 = (spot_exponent, cr, a0, a1)
#: and 5.x = a2 (F012), 5.yz = the GX (k1, k2) (F013), 6 = (cx, cy, w,
#: h) and 7 = (start, aext, 0, 0) the Model 3 screen spotlight (F015)
LIGHT_TEXEL_STRIDE = 8


def pack_light_texels(lights, job=None):
    """The per-light value texture, (1, n*STRIDE, 4) float32.

    Every value is np.float32 of the SAME python expression the old
    literal bake evaluated, so the arithmetic downstream sees the same
    numbers to within the literal's own %.9g decimal rounding -- and
    the CPU's exact float32 values, which is what parity is against.

    R251 (STRIDE 8): texel 4 = (spot_exponent, cr, a0, a1) and 5.x = a2,
    the cone law's values (F012); 5.yz = the GX distance coefficients
    (k1, k2) (F013); 6 = (cx, cy, w, h) and 7 = (start, aext, 0, 0), the
    Model 3 screen spotlight's ellipse in INTERNAL pixels (F015) --
    camera-dependent, so `job` (its settings, camera, width, height) is
    passed by every caller and the plan-cache HIT repack runs per frame.
    """
    import numpy as np
    from ..core import lights as LI
    n = max(len(lights), 1)
    st_job = getattr(job, 'settings', None)
    vp_job = None
    if job is not None and any(getattr(l, 'screen_spot', False)
                               for l in lights):
        from ..core import render as _R
        vp_job = _R.camera_matrices(job.scene.camera, job.width,
                                    job.height)[2]
    out = np.zeros((1, n * LIGHT_TEXEL_STRIDE, 4), np.float32)
    for i, light in enumerate(lights):
        b = i * LIGHT_TEXEL_STRIDE
        kind = getattr(light, 'type', 'POINT')
        sign = -1.0 if getattr(light, 'negative', False) else 1.0
        energy = float(getattr(light, 'energy', 1.0))
        if kind in ('SUN', 'HEMI'):
            d = np.asarray(light.direction, np.float32)
            d = d / max(float(np.linalg.norm(d)), 1e-9)
            out[0, b + 1, :3] = -d
            out[0, b + 0, 3] = np.float32(sign * energy)
        else:
            out[0, b + 0, :3] = np.asarray(
                getattr(light, 'position', (0, 0, 0)), np.float32)
            out[0, b + 0, 3] = np.float32(sign * energy / (4.0 * np.pi))
            if kind == 'SPOT':
                sd = np.asarray(light.direction, np.float32)
                sd = sd / max(float(np.linalg.norm(sd)), 1e-9)
                out[0, b + 1, :3] = sd
                spotsi = float(np.cos(float(light.spot_size) * 0.5))
                out[0, b + 1, 3] = np.float32(spotsi)
                out[0, b + 2, 3] = np.float32(
                    (1.0 - spotsi) * float(light.spot_blend))
            elif kind == 'AREA':
                # the form factor's value road: facing direction and
                # the dist^2/(sx*sy) normalisation (area_lamp_vectors'
                # areasize at one sample) -- Distance drags re-upload
                # this texel instead of recompiling
                ad = np.asarray(light.direction, np.float32)
                ad = ad / max(float(np.linalg.norm(ad)), 1e-9)
                out[0, b + 1, :3] = ad
                from ..core.lights import area_emit_area
                a_d = max(float(getattr(light, 'decay_end', 25.0)), 1e-6)
                # R225: divided by the face's EMITTING area (pi/4 of
                # the rectangle for the round shapes), as the CPU
                out[0, b + 1, 3] = np.float32(
                    a_d * a_d / area_emit_area(light))
        out[0, b + 2, :3] = np.asarray(
            getattr(light, 'color', (1, 1, 1)), np.float32)
        eps = 1e-6
        mode = getattr(light, 'decay', 'DEFAULT')
        start = float(getattr(light, 'decay_start', 0.0))
        out[0, b + 3, 0] = np.float32(start)
        if mode == 'CUSTOM':
            end = max(float(getattr(light, 'decay_end', 40.0)),
                      start + eps)
            out[0, b + 3, 1] = np.float32(end - start)
        else:
            out[0, b + 3, 1] = np.float32(
                max(float(getattr(light, 'decay_end', 25.0)), eps))
        out[0, b + 3, 2] = np.float32(
            float(getattr(light, 'decay_ld1', 0.0) or 0.0))
        out[0, b + 3, 3] = np.float32(
            float(getattr(light, 'decay_ld2', 0.0) or 0.0))
        if kind == 'SPOT':
            # R251 F012: the cone law's values -- the SAME expressions
            # spot_law_factor evaluates (spot_law_coeffs is the one
            # place the coefficients are computed)
            cr, a0, a1, a2 = LI.spot_law_coeffs(light)
            out[0, b + 4, 0] = np.float32(
                float(getattr(light, 'spot_exponent', 0.0) or 0.0))
            out[0, b + 4, 1] = cr
            out[0, b + 4, 2] = a0
            out[0, b + 4, 3] = a1
            out[0, b + 5, 0] = a2
            if getattr(light, 'screen_spot', False) and vp_job is not None:
                # R251 F015: the screen ellipse, per frame
                cx, cy, w_e, h_e, s0, aext = LI.screen_spot_params(
                    light, vp_job, job.width, job.height,
                    getattr(job.scene, 'camera', None))
                out[0, b + 6, 0] = cx
                out[0, b + 6, 1] = cy
                out[0, b + 6, 2] = w_e
                out[0, b + 6, 3] = h_e
                out[0, b + 7, 0] = s0
                out[0, b + 7, 1] = aext
        dmode = str(mode)
        if dmode == 'DEFAULT' and st_job is not None:
            dmode = str(getattr(st_job, 'light_falloff_default', 'DEFAULT'))
        if dmode.startswith('GX_') and kind not in ('SUN', 'HEMI'):
            # R251 F013: libogc's (k1, k2) for the lamp's ref_brite
            k1, k2 = LI.gx_dist_coeffs(light, dmode)
            out[0, b + 5, 1] = k1
            out[0, b + 5, 2] = k2
    return out


def spot_law_glsl(law, exp_on, r):
    """R251 F012: the GLSL of one cone law, defining `float spot_f` from
    `cosang` -- statement for statement lights.spot_law_factor. `r` maps
    'si' (the cutoff cosine), 'exp', 'cr', 'a0', 'a1', 'a2' to their
    expressions (hal_lights texel reads, or literals). Whether the
    exponent is nonzero is structure (`exp_on`); its value is a texel.
    The test module compiles exactly these lines standalone."""
    if law == 'GL11':
        if exp_on:
            return [f'    float spot_f = (cosang < {r["si"]}) ? 0.0 : '
                    f'pow(max(cosang, 0.0), {r["exp"]});']
        return [f'    float spot_f = (cosang < {r["si"]}) ? 0.0 : 1.0;']
    if law == 'POV':
        lines = [(f'    float pa = pow(max(cosang, 0.0), {r["exp"]});'
                  if exp_on else '    float pa = 1.0;'),
                 f'    float pd = {r["cr"]} - {r["si"]};',
                 f'    float pt0 = cosang - {r["si"]};',
                 # the division by a tiny pd lives in the DISCARDED
                 # operand of the select below
                 '    pt0 = pt0 / pd;',
                 '    pt0 = clamp(pt0, 0.0, 1.0);',
                 f'    float pt1 = (cosang < {r["cr"]}) ? 0.0 : 1.0;',
                 '    float pt = (pd < 1e-6) ? pt1 : pt0;',
                 '    float pt2 = pt * pt;',
                 '    float ps = 3.0 - 2.0 * pt;',
                 '    ps = ps * pt2;',
                 '    float pb = pa * ps;',
                 f'    pa = (cosang < {r["cr"]}) ? pb : pa;',
                 '    float spot_f = (cosang <= 0.0) ? 0.0 : pa;']
        return lines
    # the six GX angular functions: a quadratic in the cosine, saturated
    return [f'    float ga = {r["a1"]} * cosang;',
            f'    ga = {r["a0"]} + ga;',
            '    float gc = cosang * cosang;',
            f'    gc = {r["a2"]} * gc;',
            '    ga = ga + gc;',
            '    float spot_f = clamp(ga, 0.0, 1.0);']


def screen_spot_glsl(r):
    """R251 F015: the GLSL of the Model 3 screen-spot lobe, defining
    `ss_en`, `ss_el`, `ss_lobe` from `ss_px`, `ss_py`, `ss_depth` --
    statement for statement lights.screen_spot_lobe. `r` maps 'cx',
    'cy', 'w', 'h', 'start', 'aext' to expressions."""
    return ['    float ss_ex = ss_px + 0.5;',
            f'    ss_ex = ss_ex - {r["cx"]};',
            f'    ss_ex = ss_ex / {r["w"]};',
            '    float ss_ey = ss_py + 0.5;',
            f'    ss_ey = ss_ey - {r["cy"]};',
            f'    ss_ey = ss_ey / {r["h"]};',
            '    ss_ex = ss_ex * ss_ex;',
            '    ss_ey = ss_ey * ss_ey;',
            '    float ss_el = 1.0 - ss_ex;',
            '    ss_el = ss_el - ss_ey;',
            '    ss_el = max(ss_el, 0.0);',
            f'    float ss_en = (ss_depth >= {r["start"]}) ? 1.0 : 0.0;',
            '    float ss_z = -ss_depth;',
            f'    float ss_dd = {r["start"]} + {r["aext"]};',
            '    ss_dd = ss_dd + ss_z;',
            '    ss_dd = min(ss_dd, 0.0);',
            f'    float ss_qa = 1.0 + {r["aext"]};',
            '    float ss_q = ss_dd / ss_qa;',
            '    ss_q = ss_q - 1.0;',
            '    ss_q = ss_q * ss_q;',
            '    float ss_rng = ss_en / ss_q;',
            '    float ss_lobe = ss_rng * ss_el;']


def decay_law_glsl(mode, r):
    """R251 F013: the GLSL of one decay law, defining `float att` from
    `dist` -- statement for statement lights.decay_law. `r` maps 'D',
    'ld1', 'ld2', 'k1', 'k2' to expressions (texel reads or literals)."""
    lines = ['    float dm = max(dist, 0.0);']
    if mode in ('POV_FADE_LINEAR', 'POV_FADE_SQUARE'):
        lines.append(f'    float at_q = dm / {r["D"]};')
        if mode == 'POV_FADE_SQUARE':
            lines.append('    at_q = at_q * at_q;')
        lines += ['    float at_s = 1.0 + at_q;',
                  '    float att = 2.0 / at_s;']
        return lines
    ka, kb = (r['ld1'], r['ld2']) if mode == 'GL_3TERM' else (r['k1'],
                                                              r['k2'])
    lines += [f'    float at_t = {ka} * dm;',
              '    float at_u = dm * dm;',
              f'    at_u = {kb} * at_u;',
              '    float at_s = 1.0 + at_t;',
              '    at_s = at_s + at_u;',
              '    float att = 1.0 / at_s;']
    return lines


#: R251 F011: the fixed camera-axis viewer rides hal_fogtab texel 227
#: (the lighting pack's section-0 texel map: (vs.x, vs.y, vs.z, 0), the
#: normalised +Z row of the CPU's own view matrix, packed per frame by
#: pack_fog_texels). A constant vector read from a texture: bitwise.
AXIS_VIEWER_TEXEL = 227


def viewer_expr(consts, bake=None):
    """The GLSL viewer the reflectance models take: `V` (the true eye
    vector, PIXEL) or the frame's camera axis (AXIS, or a model that
    forces the axis -- F016/F018 mark `bake['__axis_viewer']`)."""
    if str((consts or {}).get('specular_viewer', 'PIXEL')) == 'AXIS' \
            or bool((bake or {}).get('__axis_viewer')) \
            or int((bake or {}).get('__model_i', -1)) in (32, 35):
        # (32 GX_LIGHT / 35 DS_FIXED force the axis: LIGHT-B2 F016/F018)
        return (f'texelFetch(hal_fogtab, ivec2({AXIS_VIEWER_TEXEL}, 0), 0)'
                '.xyz')
    return 'V'


def _lref(i, texel, comp):
    """The GLSL expression reading one packed light value."""
    return f'hal_ltex({i * LIGHT_TEXEL_STRIDE + texel}).{comp}'


def _attenuation(light, falloff_default, eps=1e-6, refs=None):
    """GLSL for one light's distance falloff, matching lights.attenuate.

    With `refs` (the light-texel expressions: {'D', 'start'}), the
    VALUES come from the hal_lights texture and only the branch
    structure bakes -- dragging the lamp Distance slider re-uploads a
    texel instead of recompiling the pass."""
    expr = _attenuation_curve(light, falloff_default, eps, refs=refs)
    if getattr(light, 'bi_sphere', False):
        # LA_SPHERE, verbatim (R155): *= (D-d)/D with a hard zero past
        # the lamp Distance -- applied OUTSIDE the falloff switch, so
        # even a Constant lamp clips (exactly lights.attenuate)
        D = refs['D'] if refs else _f(max(
            float(getattr(light, 'decay_end', 25.0)), eps))
        return (f'((max(dist, 0.0) < {D}) ? ({expr} '
                f'* (({D} - max(dist, 0.0)) / {D})) : 0.0)')
    return expr


def _attenuation_curve(light, falloff_default, eps=1e-6, refs=None):
    mode = getattr(light, 'decay', 'DEFAULT')
    if mode == 'DEFAULT':
        mode = falloff_default
    start = float(getattr(light, 'decay_start', 0.0))
    s_ref = refs['start'] if refs else _f(start)
    if mode == 'NONE':
        return '1.0'
    d = f'max(dist - {s_ref}, {_f(eps)})' if start > 0 else \
        f'max(dist, {_f(eps)})'
    if mode == 'INVERSE':
        return f'(1.0 / {d})'
    if mode == 'CUSTOM':
        if refs:
            # refs['D'] carries (end - start), packed as one float32
            return (f'clamp(1.0 - (dist - {s_ref}) / {refs["D"]}, '
                    f'0.0, 1.0)')
        end = max(float(getattr(light, 'decay_end', 40.0)), start + eps)
        return (f'clamp(1.0 - (dist - {_f(start)}) / {_f(end - start)}, '
                f'0.0, 1.0)')
    D = refs['D'] if refs else _f(max(
        float(getattr(light, 'decay_end', 25.0)), eps))
    if mode == 'BI_LINEAR':
        # Blender Internal's D/(D+d), D from the lamp Distance
        return f'({D} / ({D} + max(dist, 0.0)))'
    if mode == 'BI_SQUARE':
        # verbatim lamp_get_visibility (R155): D / (D + d*d) -- the
        # r12045 'hack' the C itself annotates, NOT a true inverse
        # square D^2/(D^2+d^2). The first cut assumed the tidy form;
        # the fetched source says otherwise.
        return (f'({D} / ({D} '
                '+ max(dist, 0.0) * max(dist, 0.0)))')
    if mode == 'BI_SLIDERS':
        # verbatim: D/(D+ld1*d) * D^2/(D^2+ld2*d^2), each factor only
        # when its slider is positive (the 2.4x Quad lamp's att1/att2)
        ld1 = float(getattr(light, 'decay_ld1', 0.0) or 0.0)
        ld2 = float(getattr(light, 'decay_ld2', 0.0) or 0.0)
        ld1_r = refs['ld1'] if refs else _f(ld1)
        ld2_r = refs['ld2'] if refs else _f(ld2)
        parts = []
        if ld1 > 0.0:
            parts.append(f'({D} / ({D} + {ld1_r} * dist))')
        if ld2 > 0.0:
            parts.append(f'(({D} * {D}) / (({D} * {D}) '
                         f'+ {ld2_r} * dist * dist))')
        return ' * '.join(parts) if parts else '1.0'
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


def _cel_lookup_function(consts):
    """R238: the cel field at this fragment's own pixel -- the screen
    shadow in .r, the depth rim in .g -- exactly celfield.lookup for
    the opaque G-buffer surface (which this pass is by construction)."""
    w, h = consts['resolution']
    L = ['vec4 hal_cel_at()',
         '{',
         f'    int sx = int(vUV.x * {_f(float(w))});',
         f'    int sy = int(vUV.y * {_f(float(h))});',
         '    return texelFetch(hal_celfield, ivec2(sx, sy), 0);',
         '}']
    return 'uniform sampler2D hal_celfield;\n' + '\n'.join(L) + '\n'


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

    The lookup is the CPU Texture sampler's own texel arithmetic written
    out -- floor, fract, per-texel wrap, the lerps or the B-spline
    weights -- reading the uploaded image at texel centres, so both
    devices filter with the same float math instead of trusting a
    driver's sampler. R219: the wrap comes from the light's resolved
    extension (REPEAT tiles, EXTEND clamps, CLIP returns zero outside
    the slide -- the projector's gate) and the filter from its
    interpolation choice (CLOSEST / BILINEAR / CUBIC), exactly the
    combinations _sample_nearest / _sample_bilinear / _sample_bicubic
    run on the prepared pixels.
    """
    w = float(spec['w'])
    h = float(spec['h'])
    ext = spec.get('extend', 'EXTEND' if spec['kind'] != 'SUN'
                  else 'REPEAT')
    filt = spec.get('filter', 'BILINEAR')

    def wrap(expr, n):
        if ext == 'REPEAT':
            return f'mod({expr}, {_f(n)})'
        return f'clamp({expr}, 0.0, {_f(n - 1.0)})'

    def fetch(xe, ye):
        return (f'texelFetch(hal_cookie{i}, ivec2(int({xe}), '
                f'int({ye})), 0).rgb')

    L = [f'uniform sampler2D hal_cookie{i};',
         f'vec3 hal_cookie_rgb{i}(vec2 uv)',
         '{']
    if ext == 'CLIP':
        # the CPU zeroes every out-of-square lookup after sampling;
        # returning early is the same value for less work
        L.append('    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || '
                 'uv.y > 1.0) return vec3(0.0);')
    if filt == 'NEAREST':
        # _sample_nearest: floor(u*w), no half-texel shift
        L += [f'    float x0 = floor(uv.x * {_f(w)});',
              f'    float y0 = floor(uv.y * {_f(h)});',
              f'    float x0w = {wrap("x0", w)};',
              f'    float y0w = {wrap("y0", h)};',
              f'    return {fetch("x0w", "y0w")};',
              '}']
        return '\n'.join(L) + '\n'
    L += [f'    float fx = uv.x * {_f(w)} - 0.5;',
          f'    float fy = uv.y * {_f(h)} - 0.5;',
          '    float x0 = floor(fx);',
          '    float y0 = floor(fy);',
          '    float tx = fx - x0;',
          '    float ty = fy - y0;']
    if filt == 'CUBIC':
        # _sample_bicubic: uniform B-spline weights, sixteen fetches,
        # rows summed in x then weighted in y -- the same order, the
        # same 1/6 factors
        L += ['    float tx2 = tx * tx;',
              '    float tx3 = tx2 * tx;',
              '    float ty2 = ty * ty;',
              '    float ty3 = ty2 * ty;',
              '    float wx0 = (1.0 - 3.0 * tx + 3.0 * tx2 - tx3) '
              '* 0.16666667;',
              '    float wx1 = (4.0 - 6.0 * tx2 + 3.0 * tx3) '
              '* 0.16666667;',
              '    float wx2 = (1.0 + 3.0 * tx + 3.0 * tx2 - 3.0 * tx3) '
              '* 0.16666667;',
              '    float wx3 = tx3 * 0.16666667;',
              '    float wy0 = (1.0 - 3.0 * ty + 3.0 * ty2 - ty3) '
              '* 0.16666667;',
              '    float wy1 = (4.0 - 6.0 * ty2 + 3.0 * ty3) '
              '* 0.16666667;',
              '    float wy2 = (1.0 + 3.0 * ty + 3.0 * ty2 - 3.0 * ty3) '
              '* 0.16666667;',
              '    float wy3 = ty3 * 0.16666667;']
        for k in range(4):
            L.append(f'    float xw{k} = '
                     f'{wrap(f"x0 + {_f(k - 1.0)}", w)};')
        L.append('    vec3 acc = vec3(0.0);')
        for j in range(4):
            L.append(f'    float yw{j} = '
                     f'{wrap(f"y0 + {_f(j - 1.0)}", h)};')
            row = ' + '.join(
                f'{fetch(f"xw{k}", f"yw{j}")} * wx{k}' for k in range(4))
            L.append(f'    acc = acc + ({row}) * wy{j};')
        L += ['    return acc;', '}']
        return '\n'.join(L) + '\n'
    # BILINEAR -- the pre-R219 body, wrap generalized
    L += [f'    float x0w = {wrap("x0", w)};',
          f'    float x1w = {wrap("x0 + 1.0", w)};',
          f'    float y0w = {wrap("y0", h)};',
          f'    float y1w = {wrap("y0 + 1.0", h)};',
          f'    vec3 c00 = {fetch("x0w", "y0w")};',
          f'    vec3 c10 = {fetch("x1w", "y0w")};',
          f'    vec3 c01 = {fetch("x0w", "y1w")};',
          f'    vec3 c11 = {fetch("x1w", "y1w")};',
          '    vec3 top = c00 + (c10 - c00) * tx;',
          '    vec3 bot = c01 + (c11 - c01) * tx;',
          '    return top + (bot - top) * ty;',
          '}']
    return '\n'.join(L) + '\n'


def lint_declaration_order(src):
    """R193: every hal_* call must follow a declaration of its name.

    GLSL requires declaration before use in FILE ORDER, and the real
    driver enforces it -- the simulator, which resolves names after
    parsing, never did. The field found the gap: the area form
    factor's texel reads called hal_ltex from the shadow-function
    block, whose definition lands later in the assembly; every
    material pass of every area-lamp scene failed CreateInfo, and the
    refusal storm ended in a native crash. This walk is the suite's
    stand-in for the driver's rule: it returns a list of
    '<name> called at <pos> before its declaration at <pos>' strings,
    empty when the source is honest. Only hal_-prefixed functions are
    judged -- built-ins carry no declaration in the source.
    """
    import re
    # comments out first, preserving offsets line-for-line is not
    # needed -- positions only order the report
    clean = re.sub(r'//[^\n]*', '', src)
    clean = re.sub(r'/\*.*?\*/', '', clean, flags=re.S)
    decl_re = re.compile(
        r'\b(?:void|float|int|bool|uint|vec[234]|ivec[234]|uvec[234]'
        r'|mat[234])\s+(hal_\w+)\s*\(')
    first_decl = {}
    decl_spans = []
    for m in decl_re.finditer(clean):
        name = m.group(1)
        decl_spans.append((m.start(1), m.end(1)))
        if name not in first_decl:
            first_decl[name] = m.start(1)
    spans = set()
    for a, b in decl_spans:
        spans.add(a)
    out = []
    for m in re.finditer(r'\b(hal_\w+)\s*\(', clean):
        name = m.group(1)
        pos = m.start(1)
        if pos in spans:
            continue                      # this IS a declaration head
        d = first_decl.get(name)
        if d is not None and pos < d:
            out.append(f'{name} called at {pos} before its '
                       f'declaration at {d}')
    return out


def _area_function(i, light, consts):
    """vec2 hal_area_inp{i}(P, N): BI's area lamp energy, both sides.

    The GLSL twin of lights.area_inp -- ONE Stokes contour over the
    rectangle's corners, front and flipped-normal energies read off
    its sign (the sum is linear in the normal). The edge angle is
    2*asin(||a-b||/2), mathematically acos(dot(a,b)) but stable in
    float32 where acos of a near-1 dot sheds digits -- the C could
    afford plain acos because area_lamp_energy ran in doubles.

    Corners derive in-shader from the texel POSITION plus literal
    half-extent offsets, so dragging the lamp re-uploads a texel; the
    facing direction and the dist^2/(sx*sy) normalisation ride texel 1
    (pack_light_texels), so Distance drags stay recompile-free too.
    Size, axes and Gamma are structure and recompile, like the shadow
    twin's own geometry bakes.
    """
    import numpy as np
    from ..core import lights as LI
    use_tx = bool(consts.get('light_texels', True))
    ax = np.asarray(getattr(light, 'area_x', None)
                    if getattr(light, 'area_x', None) is not None
                    else (1.0, 0.0, 0.0), np.float64)
    ay = np.asarray(getattr(light, 'area_y', None)
                    if getattr(light, 'area_y', None) is not None
                    else (0.0, 1.0, 0.0), np.float64)
    asz = getattr(light, 'area_size', (1.0, 1.0))
    sx = max(float(asz[0]), 1e-6)
    sy = max(float(asz[1]) if len(asz) > 1 else float(asz[0]), 1e-6)
    hx = (ax * (sx * 0.5)).astype(np.float32)
    hy = (ay * (sy * 0.5)).astype(np.float32)
    disc = LI.area_is_disc(light)
    if use_tx:
        pos = _lref(i, 0, 'xyz')
        vdir = _lref(i, 1, 'xyz')
        asize = _lref(i, 1, 'w')
    else:
        d = np.asarray(light.direction, np.float32)
        d = d / max(float(np.linalg.norm(d)), 1e-9)
        dist = max(float(getattr(light, 'decay_end', 25.0)), 1e-6)
        pos = _v3(light.position)
        vdir = _v3(d)
        asize = _f(dist * dist / LI.area_emit_area(light))
    k = float(getattr(light, 'area_gamma', 1.0) or 1.0)
    L = [f'vec2 hal_area_inp{i}(vec3 P, vec3 N)',
         '{',
         f'    vec3 hp = {pos};']
    if not disc:
        # area_lamp_vectors' corner order: -x-y, -x+y, +x+y, +x-y
        L += [f'    vec3 hax = {_v3(hx)};',
              f'    vec3 hay = {_v3(hy)};',
              '    vec3 av0 = normalize(P - (hp - hax - hay));',
              '    vec3 av1 = normalize(P - (hp - hax + hay));',
              '    vec3 av2 = normalize(P - (hp + hax + hay));',
              '    vec3 av3 = normalize(P - (hp + hax - hay));']
        nc = 4
    else:
        # R225: the round shapes' equal-area polygon (lights.
        # area_corners), each corner a literal OFFSET from the texel
        # position so the lamp still drags recompile-free
        pos64 = np.asarray(light.position, np.float64)
        offs = (LI.area_corners(light) - pos64[None, :]).astype(np.float32)
        nc = offs.shape[0]
        for c in range(nc):
            L.append(f'    vec3 av{c} = normalize(P - (hp + {_v3(offs[c])}));')
    L += [
         # BI shades with normals flipped along the view ray; the
         # contour sees the normal bare, so -N is the C's own frame
         # (core/lights.area_inp has the derivation and the compiled
         # proof)
         '    vec3 avn = -N;',
         '    float fac = 0.0;',
         '    vec3 acr; float acl;']
    for a in range(nc):
        b = (a + 1) % nc
        L += [f'    acr = cross(av{a}, av{b});',
              '    acl = max(length(acr), 1e-30);',
              f'    fac += (2.0 * asin(clamp(0.5 * length(av{a} - av{b}),'
              ' 0.0, 1.0))) * (dot(avn, acr) / acl);']
    L += [f'    float afront = (dot(P - hp, {vdir}) >= 0.0) '
          '? 1.0 : 0.0;',
          f'    float aas = {asize};',
          '    float af = max(fac, 0.0) * aas;',
          '    float ab = max(-fac, 0.0) * aas;']
    if abs(k - 1.0) > 1e-9:
        # pow(0, k) is undefined territory on real drivers
        L += [f'    af = (af > 0.0) ? pow(af, {_f(k)}) : 0.0;',
              f'    ab = (ab > 0.0) ? pow(ab, {_f(k)}) : 0.0;']
    L += ['    return vec2(af, ab) * afront;',
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
        # rays this pixel's CPU shade would build, averaged the same way.
        # Three shapes, matching _soft_ray on the CPU: AREA aims at real
        # points on its rectangle (or ellipse), SUN tilts inside its
        # angular disc, POINT/SPOT jitter the target in a world sphere
        w, h = consts['resolution']
        seed = int(consts.get('seed', 0))
        kind = str(meta.get('kind', 'POINT')).upper()
        area = meta.get('area')
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
            if kind == 'AREA' and area is not None:
                L += ['    {',
                      f'    float u1 = hal_smp_hash3(sx, sy, {z});',
                      f'    float u2 = hal_smp_hash3(sx, sy, {z + 1});']
                if area['disk']:
                    L += ['    float rr = sqrt(u1);',
                          '    vec2 cs = hal_smp_circle(u2);',
                          '    float uu = rr * cs.x;',
                          '    float vv = rr * cs.y;']
                else:
                    L += ['    float uu = u1 * 2.0 - 1.0;',
                          '    float vv = u2 * 2.0 - 1.0;']
                L += [f'    vec3 ps = {_v3(area["pos"])} + '
                      f'{_v3(area["ax"])} * ({_f(area["hx"])} * uu) + '
                      f'{_v3(area["ay"])} * ({_f(area["hy"])} * vv);',
                      '    vec3 dl = ps - P;',
                      '    float ds = max(length(dl), 1e-6);',
                      '    vec3 Lj = dl / ds;',
                      '    float mt = ds * (1.0 - 1e-3);',
                      '    acc += 1.0 - hal_bvh_occluded(org, Lj, mt);',
                      '    }']
            elif kind == 'SUN':
                tanh = _f(float(np.tan(float(meta['radius']) * 0.5)))
                L += ['    {',
                      f'    float u1 = hal_smp_hash3(sx, sy, {z});',
                      f'    vec2 cs = hal_smp_circle(hal_smp_hash3(sx, sy, '
                      f'{z + 1}));',
                      f'    float r = sqrt(u1) * {tanh};',
                      '    vec3 jit = t * (r * cs.x) + b * (r * cs.y);',
                      '    vec3 Lj = normalize(L + jit);',
                      '    acc += 1.0 - hal_bvh_occluded(org, Lj, maxt);',
                      '    }']
            else:
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

    voff = int(meta.get('voff', 0))
    L = ['uniform sampler2D hal_shadowpack;',
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
    if meta.get('midpoint'):
        # R251 C117: the MIDPOINT map's compare has no normal offset (and
        # `slope` above folds to 0.0 * (...) = 0.0 from the zero bias):
        # `P + N * 0.0 == P` bitwise, the CPU's own zeroed inputs
        L.append('    float off_amt = 0.0;')
    else:
        L.append(f'    float off_amt = texel * (1.5 + 2.5 * '
                 f'sqrt(max(1.0 - ndl * ndl, 0.0))) * {_f(max(1.0, soft))};')
    L.append('    vec4 ph = vec4(P + N * off_amt, 1.0);')

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
                  f'        celly = {_f((fi // grid_w) * size + voff)};',
                  '    }']
    else:
        vp = np.asarray(faces[0]['vp'], np.float32)
        L += [f'    float clipx = dot(ph, {_v4(vp[0])});',
              f'    float clipy = dot(ph, {_v4(vp[1])});',
              f'    float clipz = dot(ph, {_v4(vp[2])});',
              f'    float clipw = dot(ph, {_v4(vp[3])});',
              f'    float cellx = 0.0; float celly = {_f(voff)};']

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
              '    float occ = texelFetch(hal_shadowpack, '
              'ivec2(int(cellx + xi), int(celly + yi)), 0).r;',
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
SUPPORTED_TEX_FILTERS = ('NEAREST', 'BILINEAR', 'TRILINEAR', 'N64_3POINT',
                         'POV_NORMDIST', 'SUMMED_AREA')

# R251 texture pack: one definition of the pyramid / footprint filter sets
# (core/texture.py) and ONE `sample_opts` for both devices, so the three
# rules of `sample_opts` cannot drift between the CPU and the GPU
from ..core import texture as _TX                                   # noqa: E402
import numpy as np                                                  # noqa: E402
PYRAMID_FILTERS = _TX.PYRAMID_FILTERS
FOOTPRINT_TEX_FILTERS = _TX.FOOTPRINT_TEX_FILTERS
#: what a footprint filter samples where no footprint exists (secondary
#: passes, coded images, linked Vector chains): the CPU's lod=None road
NOFOOTPRINT_FILTER = {'TRILINEAR': 'BILINEAR', 'SUMMED_AREA': 'NEAREST'}
_OPT_KEYS = ('tex_filter', 'tex_mipmap', 'tex_aniso', 'tex_frac_bits',
             'tex_clamp_mode', 'tex_colorkey', 'tex_colorkey_range',
             'tex_mip_select', 'tex_lod_sharpen', 'tex_lod_source',
             'tex_lod_k', 'tex_lod_l')


def _tex_opts(consts):
    """The sample-time dials of R251 from the plan's consts: the SAME
    `sample_opts` the CPU reads, on the same twelve keys."""
    import types
    ns = types.SimpleNamespace(**{k: consts[k] for k in _OPT_KEYS if k in consts})
    return _TX.sample_opts(ns)


def _frac_lines(frac_bits):
    """C080: the Voodoo's coarse texel fraction, exactly _sample_bilinear's
    four float32 operations (floor of an exact power-of-two product)."""
    n = float(2 ** int(frac_bits))
    inv = float(1.0 / 2 ** int(frac_bits))
    return [f'    float txq = floor(tx * {_f(n)});',
            f'    tx = txq * {_f(inv)};',
            f'    float tyq = floor(ty * {_f(n)});',
            f'    ty = tyq * {_f(inv)};']


def _border_wrap(wrap, opts):
    """C077: `Texture.sample`'s own mapping -- an Extend wrap under
    GL_CLAMP is the internal BORDER wrap (the GL 1.1 coordinate clamp with
    transparent-black border taps)."""
    if opts is not None and wrap == 'EXTEND' and opts.get('clamp_mode') == 'GL_CLAMP':
        return 'BORDER'
    return wrap


def _colorkey_lines(tex, opts):
    """C083: the chroma key's tail, after the filter and after CLIP: a
    sample below the threshold on every channel is discarded (alpha 0), a
    survivor keeps its own alpha. The literal is `colorkey_threshold` on
    the texture's OWN colourspace, `_f` round-trips the float32."""
    if opts is None or not opts.get('colorkey'):
        return []
    thr = _TX.colorkey_threshold(int(opts.get('colorkey_range', 0)),
                                 getattr(tex, 'colorspace', 'sRGB') == 'Linear')
    t = _f(thr)
    return [f'    if (c.r < {t} && c.g < {t} && c.b < {t}) {{ c.a = 0.0; }}']


def _clip_lines(wrap):
    if wrap == 'CLIP':
        return ['    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || '
                'uv.y > 1.0) { c = vec4(0.0); }']
    return []


def _border_select(wrap, v00, v10, v01, v11, wdim, hdim, x0='x0', y0='y0',
                   x1='x1', y1='y1'):
    """C077: the four validity selects of the BORDER taps against the
    dims `wdim`/`hdim` (literals or runtime names): the low side can only
    fail for x0/y0 and the high side for x1/y1 once s, t are clamped --
    the CPU's `_border_taps` mask, tap by tap."""
    if wrap != 'BORDER':
        return []
    return [f'    {v00} = ({x0} >= 0.0 && {y0} >= 0.0) ? {v00} : vec4(0.0);',
            f'    {v10} = ({x1} <= {wdim} - 1.0 && {y0} >= 0.0) ? {v10} : vec4(0.0);',
            f'    {v01} = ({x0} >= 0.0 && {y1} <= {hdim} - 1.0) ? {v01} : vec4(0.0);',
            f'    {v11} = ({x1} <= {wdim} - 1.0 && {y1} <= {hdim} - 1.0) ? {v11} : vec4(0.0);']


def _uv_clamp_lines(wrap, sx='s', sy='t'):
    """C077: GL_CLAMP clamps the coordinate; `sx`, `sy` replace uv.x, uv.y
    (the 3-point branch names its own s / t, so it passes cs / ct)."""
    if wrap == 'BORDER':
        return [f'    float {sx} = clamp(uv.x, 0.0, 1.0);',
                f'    float {sy} = clamp(uv.y, 0.0, 1.0);']
    return []


def _uvx(wrap, sx='s'):
    return sx if wrap == 'BORDER' else 'uv.x'


def _uvy(wrap, sy='t'):
    return sy if wrap == 'BORDER' else 'uv.y'


def _lod_field_key(opts, tex):
    """Which CPU-decided LOD field a footprint sampler reads: None (the
    driver's own log2(rho) + bias, as today), (0, 0) under the GS's Q rule
    (one size-independent field) or (w, h) under a per-texture-size level
    road. Wave 1 emits none of the field roads; the emission site refuses
    them by name until TEX-2 lands the samplers."""
    if opts is None:
        return None
    if opts.get('lod_source') == 'GS_Q':
        return (0, 0)
    if opts.get('lod_source') == 'TRIANGLE' or opts.get('mip_select') != 'FILTER' \
            or opts.get('lod_sharpen'):
        return (int(tex.width), int(tex.height))
    return None


def _lod_uniform(key):
    return 'hal_lodq' if key == (0, 0) else f'hal_lod_{key[0]}x{key[1]}'


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


def _atlas_head(name, tex, levels):
    """The atlas fetch and the level table shared by every pyramid
    sampler: `hal_afetch` reads a texel centre of the vertical mip stack,
    `hal_mipof(l)` is the select ladder (y0, w, h) of level l -- no
    arrays, nothing the front-end cannot run."""
    w0 = float(tex.width)
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
          '}']
    return L


def _lvl_function(name, filt, wrap, frac_bits):
    """`hal_lvl_NAME(uv, y0, lw, lh)`: the FILTER's own tap set against the
    runtime level dims (mip levels vary) -- exactly Texture._sample_nearest
    / _sample_bilinear(frac_bits) / _sample_3point on one level, with the
    C077 BORDER coordinate clamp and validity selects and the C080 fraction
    bits. BILINEAR is TRILINEAR's level sampler (1.89.0's text under
    neutral opts)."""
    L = [f'vec4 hal_lvl_{name}(vec2 uv, float y0, float lw, float lh)',
         '{']
    if filt == 'NEAREST':
        # _sample_nearest: the coordinate is NOT clamped under BORDER
        # (GL 1.1 special-cases s = 1 for NEAREST: no seam), the index is
        L += ['    float x = floor(uv.x * lw);',
              '    float y = floor(uv.y * lh);',
              f'    x = {_wrap_dyn("x", "lw", wrap)};',
              f'    y = {_wrap_dyn("y", "lh", wrap)};',
              f'    return hal_afetch_{name}(x, y0 + y);',
              '}']
        return L
    sx, sy = ('cs', 'ct') if filt == 'N64_3POINT' else ('s', 't')
    L += _uv_clamp_lines(wrap, sx, sy)
    L += [f'    float fx = {_uvx(wrap, sx)} * lw - 0.5;',
          f'    float fy = {_uvy(wrap, sy)} * lh - 0.5;',
          '    float x0 = floor(fx);',
          '    float y0i = floor(fy);',
          '    float tx = fx - x0;',
          '    float ty = fy - y0i;']
    if filt != 'N64_3POINT' and int(frac_bits):
        L += _frac_lines(frac_bits)                      # C080, per level
    L += ['    float x1 = x0 + 1.0;',
          '    float y1 = y0i + 1.0;',
          f'    float wx0 = {_wrap_dyn("x0", "lw", wrap)};',
          f'    float wx1 = {_wrap_dyn("x1", "lw", wrap)};',
          f'    float wy0 = {_wrap_dyn("y0i", "lh", wrap)};',
          f'    float wy1 = {_wrap_dyn("y1", "lh", wrap)};',
          f'    vec4 c00 = hal_afetch_{name}(wx0, y0 + wy0);',
          f'    vec4 c10 = hal_afetch_{name}(wx1, y0 + wy0);',
          f'    vec4 c01 = hal_afetch_{name}(wx0, y0 + wy1);',
          f'    vec4 c11 = hal_afetch_{name}(wx1, y0 + wy1);']
    L += _border_select(wrap, 'c00', 'c10', 'c01', 'c11', 'lw', 'lh', y0='y0i')
    if filt == 'N64_3POINT':
        L += ['    float up = ((tx + ty) > 1.0) ? 1.0 : 0.0;',
              '    vec4 a = (up > 0.5) ? c11 : c00;',
              '    float s = (up > 0.5) ? (1.0 - ty) : tx;',
              '    float t = (up > 0.5) ? (1.0 - tx) : ty;',
              '    vec4 b = (up > 0.5) ? c01 : c10;',
              '    vec4 cc = (up > 0.5) ? c10 : c01;',
              '    return a + (b - a) * s + (cc - a) * t;',
              '}']
    else:
        L += ['    vec4 top = c00 + (c10 - c00) * tx;',
              '    vec4 bot = c01 + (c11 - c01) * tx;',
              '    return top + (bot - top) * ty;',
              '}']
    return L


def _trilerp_function(name, n):
    return [f'vec4 hal_trilerp_{name}(vec2 uv, float lod)',
            '{',
            f'    float l = clamp(lod, 0.0, {_f(n - 1.0)});',
            '    float l0 = floor(l);',
            '    float f = l - l0;',
            f'    vec3 A = hal_mipof_{name}(l0);',
            f'    vec3 B = hal_mipof_{name}(min(l0 + 1.0, {_f(n - 1.0)}));',
            f'    vec4 a = hal_lvl_{name}(uv, A.x, A.y, A.z);',
            f'    vec4 b = hal_lvl_{name}(uv, B.x, B.y, B.z);',
            '    return a + (b - a) * f;',
            '}']


_UVGRAD_FETCH = [
    '    ivec2 hal_gsz = textureSize(hal_uvgrad, 0);',
    '    vec4 g = texelFetch(hal_uvgrad, '
    'ivec2(clamp(vUV * vec2(hal_gsz), vec2(0.0), '
    'vec2(hal_gsz) - vec2(1.0))), 0);']


def _mip_sampler(name, tex, wrap, levels, aniso, bias, frac_bits=0, opts=None):
    """GLSL for one TRILINEAR (optionally anisotropic) footprint sampler on
    the DERIVATIVE road (1.89.0's: the level is the driver's log2(rho) +
    bias).

    Mirrors Texture._sample_trilinear and _sample_aniso line for line:
    compute_lod from the hal_uvgrad field (the CPU's OWN derivatives,
    uploaded), a per-level manual bilinear inside the atlas stack, the
    a + (b - a) * frac level blend, and -- under anisotropy -- the mip
    level of the MINOR footprint axis with uniform trilinear taps along
    the major. The level table is a select ladder: no arrays, nothing
    the front-end cannot run. R251: `wrap` may be BORDER (C077), `opts`
    carries the chroma key (C083); the CPU-decided level roads are
    `_level_sampler`'s.
    """
    w0, h0 = float(tex.width), float(tex.height)
    n = len(levels)
    L = _atlas_head(name, tex, levels)
    L += _lvl_function(name, 'BILINEAR', wrap, frac_bits)
    L += _trilerp_function(name, n)
    L += [f'vec4 hal_sample_{name}(vec2 uv)',
          '{']
    L += _UVGRAD_FETCH
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
    L += _clip_lines(wrap)
    L += _colorkey_lines(tex, opts)
    L += ['    return c;', '}']
    return '\n'.join(L) + '\n'


_DITHER_LINES = [
    # the same pixel the uvgrad fetch reads (gbpx's expression; uploads are
    # bottom-row-first, the CPU's row 0 is the bottom, so gp.y == py) and
    # the 4x4 Bayer matrix as bit arithmetic: 0 8 2 10 / 12 4 14 6 /
    # 3 11 1 9 / 15 7 13 5 == DI.BAYER4 * 16 indexed [py & 3, px & 3]
    '    int bx = gp.x & 3;',
    '    int by = gp.y & 3;',
    '    int xy = bx ^ by;',
    '    int d0 = (xy & 1) << 3;',
    '    int d1 = (by & 1) << 2;',
    '    int d2 = ((xy >> 1) & 1) << 1;',
    '    int d3 = (by >> 1) & 1;',
    '    float d16 = float(d0 + d1 + d2 + d3);']


def _level_sampler(name, tex, filt, wrap, opts, bias, lod_key):
    """R251 (C072 / C022 / C079 / C008): a pyramid filter on a CPU-decided
    level road. The LOD is NEVER computed on the driver here -- it is read
    per pixel from the uploaded field (`hal_lod_WxH`: compute_lod or the
    per-triangle table; `hal_lodq`: the GS rule), so no sqrt / log2 ULP
    can become a whole level. `hal_pick` selects or blends levels with
    the filter's own `hal_lvl` taps (BLEND = the lod_frac lerp,
    NEAREST_LEVEL = GL 1.1's ceil(l + 1/2) - 1 written with floor,
    DITHER_VOODOO = MAME's 8.8 fixed lod + (bayer << 4) >> 8); FILTER
    (TRILINEAR under the GS rule, or a Sharpen alone) is the trilerp for
    TRILINEAR and level 0 for the others. Sharpen (C008) extrapolates
    level 0 away from level 1 under magnification with roundEven and the
    9-bit clamp, one shared float32 constant. Bitwise the CPU."""
    _atlas_px, levels = mip_atlas(tex)
    n = len(levels)
    w0, h0 = float(tex.width), float(tex.height)
    sel = str(opts.get('mip_select', 'FILTER'))
    sharpen = bool(opts.get('lod_sharpen')) and n >= 2
    lod_uniform = _lod_uniform(lod_key)
    L = _atlas_head(name, tex, levels)
    L += _lvl_function(name, filt, wrap, int(opts.get('frac_bits', 0)))
    L += [f'vec4 hal_pick_{name}(vec2 uv, float lod, float d16)',
          '{',
          f'    float l = clamp(lod, 0.0, {_f(n - 1.0)});']
    if sel == 'BLEND' or (sel == 'FILTER' and filt == 'TRILINEAR'):
        L += ['    float l0 = floor(l);',
              '    float f = l - l0;',
              f'    vec3 A = hal_mipof_{name}(l0);',
              f'    vec3 B = hal_mipof_{name}(min(l0 + 1.0, {_f(n - 1.0)}));',
              f'    vec4 a = hal_lvl_{name}(uv, A.x, A.y, A.z);',
              f'    vec4 b = hal_lvl_{name}(uv, B.x, B.y, B.z);',
              '    return a + (b - a) * f;',
              '}']
    elif sel == 'NEAREST_LEVEL':
        L += ['    float lq = l + 0.5;',
              '    float ln = 0.0 - lq;',
              '    float lf = floor(ln);',
              '    float lc2 = 0.0 - lf;',
              '    float lc = lc2 - 1.0;',
              f'    lc = clamp(lc, 0.0, {_f(n - 1.0)});',
              f'    vec3 A = hal_mipof_{name}(lc);',
              f'    return hal_lvl_{name}(uv, A.x, A.y, A.z);',
              '}']
    elif sel == 'DITHER_VOODOO':
        L += ['    float lod8 = floor(l * 256.0);',
              '    float dd = d16 * 16.0;',
              '    float ls = lod8 + dd;',
              '    float lq = ls * 0.00390625;',
              '    float lc = floor(lq);',
              f'    lc = clamp(lc, 0.0, {_f(n - 1.0)});',
              f'    vec3 A = hal_mipof_{name}(lc);',
              f'    return hal_lvl_{name}(uv, A.x, A.y, A.z);',
              '}']
    else:
        # FILTER for NEAREST / BILINEAR / N64_3POINT: level 0, as the CPU's
        # plain filter (a Sharpen alone reaches this road)
        L += [f'    vec3 A = hal_mipof_{name}(0.0);',
              f'    return hal_lvl_{name}(uv, A.x, A.y, A.z);',
              '}']
    if sharpen:
        L += [f'float hal_clamp9_{name}(float v8)',
              '{',
              '    int i = int(v8);',
              '    int w9 = i & 511;',
              '    int b8 = w9 & 256;',
              '    int b7 = w9 & 128;',
              '    float o = float(w9);',
              '    if (b8 != 0) { o = (b7 != 0) ? 0.0 : 255.0; }',
              f'    return o * {_f(np.float32(1.0 / 255.0))};',
              '}']
    L += [f'vec4 hal_sample_{name}(vec2 uv)',
          '{']
    L += _UVGRAD_FETCH
    L += ['    ivec2 gp = ivec2(clamp(vUV * vec2(hal_gsz), vec2(0.0), '
          'vec2(hal_gsz) - vec2(1.0)));']
    if sel == 'DITHER_VOODOO':
        L += _DITHER_LINES
    else:
        L += ['    float d16 = 0.0;']
    L += [f'    float lodf = texelFetch({lod_uniform}, gp, 0).r;',
          f'    vec4 c = hal_pick_{name}(uv, lodf, d16);']
    if sharpen:
        L += [f'    float ax = abs(g.x) * {_f(w0)};',
              f'    float ay = abs(g.y) * {_f(w0)};',
              f'    float bx = abs(g.z) * {_f(h0)};',
              f'    float by = abs(g.w) * {_f(h0)};',
              '    float mx = max(ax, ay);',
              '    float my = max(bx, by);',
              '    float m = max(mx, my);',
              '    if (m < 1.0) {',
              f'        vec3 L0 = hal_mipof_{name}(0.0);',
              f'        vec3 L1 = hal_mipof_{name}(1.0);',
              f'        vec4 t0 = hal_lvl_{name}(uv, L0.x, L0.y, L0.z);',
              f'        vec4 t1 = hal_lvl_{name}(uv, L1.x, L1.y, L1.z);',
              '        float f = m - 1.0;',
              '        vec4 d = t1 - t0;',
              '        vec4 df = d * f;',
              '        vec4 v = t0 + df;',
              '        vec4 v255 = v * 255.0;',
              '        vec4 v8 = roundEven(v255);',
              f'        c = vec4(hal_clamp9_{name}(v8.r), hal_clamp9_{name}(v8.g), '
              f'hal_clamp9_{name}(v8.b), hal_clamp9_{name}(v8.a));',
              '    }']
    L += _clip_lines(wrap)
    L += _colorkey_lines(tex, opts)
    L += ['    return c;', '}']
    return '\n'.join(L) + '\n'


def sat_atlas(tex):
    """C088: the summed-area table as one (H, 2W, 4) float32 image --
    the hi plane in the left half, the lo plane in the right (integer
    DATA: read with texelFetch). Built from the texture's OWN table."""
    hi, lo = tex.build_sat()
    h, w = hi.shape[:2]
    atlas = np.zeros((h, 2 * w, 4), np.float32)
    atlas[:, :w] = hi
    atlas[:, w:] = lo
    return atlas


def _sat_sampler(name, tex, wrap, opts, bias):
    """C088: Crow's summed-area box over the footprint rectangle, exactly
    Texture._sample_sat -- half-extents from the uploaded derivatives
    (at least half a texel, at most 127), the continuous coordinate
    wrapped (REPEAT; the continuous MIRROR with no "- 1"), ceil - 1
    written with floor, the box clamped at the edge, four texelFetch
    pairs, a uint wrap-around difference, two RECIP256 reads and one
    multiply by 1/65535. `hal_recip256` is declared ONCE per pass by the
    emission site (the hal_uvgrad idiom, A12.1). No division, no
    transcendental: bitwise."""
    w, h = float(tex.width), float(tex.height)
    W, H = _f(w), _f(h)
    scale = _f(np.float32(2.0 ** float(bias)))
    L = [f'uniform sampler2D {name};',
         f'vec4 hal_satfetch_{name}(float x, float y)',
         '{',
         f'    return texelFetch({name}, ivec2(int(x), int(y)), 0);',
         '}',
         f'float hal_box_{name}(float Ah, float Al, float Bh, float Bl, '
         'float Ch, float Cl, float Dh, float Dl, float rx, float ry)',
         '{',
         '    uint A = uint(Ah) * 65536u + uint(Al);',
         '    uint B = uint(Bh) * 65536u + uint(Bl);',
         '    uint C = uint(Ch) * 65536u + uint(Cl);',
         '    uint D = uint(Dh) * 65536u + uint(Dl);',
         '    uint AB = A - B;',
         '    uint ABC = AB - C;',
         '    uint box = ABC + D;',
         '    uint qh = box >> 16u;',
         '    uint ql = box & 65535u;',
         '    float fh = float(qh) * 65536.0;',
         '    float f = fh + float(ql);',
         '    float m1 = f * rx;',
         '    float m2 = m1 * ry;',
         f'    return m2 * {_f(np.float32(1.0 / 65535.0))};',
         '}',
         f'vec4 hal_sample_{name}(vec2 uv)',
         '{']
    L += _UVGRAD_FETCH
    L += ['    float ax = abs(g.x);',
          '    float ay = abs(g.y);',
          '    float sx = ax + ay;',
          '    float hx = sx * 0.5;',
          f'    float hw = hx * {W};',
          f'    float hs = hw * {scale};',
          '    float hu = max(hs, 0.5);',
          '    hu = min(hu, 127.0);',
          '    float bx = abs(g.z);',
          '    float by = abs(g.w);',
          '    float sy = bx + by;',
          '    float hy = sy * 0.5;',
          f'    float hh = hy * {H};',
          f'    float ht = hh * {scale};',
          '    float hv = max(ht, 0.5);',
          '    hv = min(hv, 127.0);',
          f'    float uc = uv.x * {W};',
          f'    float vc = uv.y * {H};']
    if wrap == 'REPEAT':
        L += [f'    uc = uc - floor(uc / {W}) * {W};',
              f'    vc = vc - floor(vc / {H}) * {H};']
    elif wrap == 'MIRROR':
        W2, H2 = _f(2.0 * w), _f(2.0 * h)
        L += [f'    float mu = uc - floor(uc / {W2}) * {W2};',
              f'    uc = (mu < {W}) ? mu : {W2} - mu;',
              f'    float mv = vc - floor(vc / {H2}) * {H2};',
              f'    vc = (mv < {H}) ? mv : {H2} - mv;']
    L += ['    float x0 = floor(uc - hu);',
          '    float nx1 = 0.0 - (uc + hu);',
          '    float fx1 = floor(nx1);',
          '    float x1 = (0.0 - fx1) - 1.0;',
          '    float y0 = floor(vc - hv);',
          '    float ny1 = 0.0 - (vc + hv);',
          '    float fy1 = floor(ny1);',
          '    float y1 = (0.0 - fy1) - 1.0;',
          f'    x0 = clamp(x0, 0.0, {W} - 1.0);',
          f'    x1 = clamp(x1, 0.0, {W} - 1.0);',
          f'    y0 = clamp(y0, 0.0, {H} - 1.0);',
          f'    y1 = clamp(y1, 0.0, {H} - 1.0);',
          '    float xm = x0 - 1.0;',
          '    float ym = y0 - 1.0;',
          f'    vec4 Ah = hal_satfetch_{name}(x1, y1);',
          f'    vec4 Al = hal_satfetch_{name}(x1 + {W}, y1);',
          f'    vec4 Bh = (xm < 0.0) ? vec4(0.0) : hal_satfetch_{name}(xm, y1);',
          f'    vec4 Bl = (xm < 0.0) ? vec4(0.0) : hal_satfetch_{name}(xm + {W}, y1);',
          f'    vec4 Ch = (ym < 0.0) ? vec4(0.0) : hal_satfetch_{name}(x1, ym);',
          f'    vec4 Cl = (ym < 0.0) ? vec4(0.0) : hal_satfetch_{name}(x1 + {W}, ym);',
          f'    vec4 Dh = (xm < 0.0 || ym < 0.0) ? vec4(0.0) : hal_satfetch_{name}(xm, ym);',
          f'    vec4 Dl = (xm < 0.0 || ym < 0.0) ? vec4(0.0) : hal_satfetch_{name}(xm + {W}, ym);',
          '    float nx = x1 - x0;',
          '    float ny = y1 - y0;',
          '    float rx = texelFetch(hal_recip256, ivec2(int(nx), 0), 0).r;',
          '    float ry = texelFetch(hal_recip256, ivec2(int(ny), 0), 0).r;',
          f'    vec4 c = vec4(hal_box_{name}(Ah.r, Al.r, Bh.r, Bl.r, Ch.r, Cl.r, Dh.r, Dl.r, rx, ry),',
          f'                  hal_box_{name}(Ah.g, Al.g, Bh.g, Bl.g, Ch.g, Cl.g, Dh.g, Dl.g, rx, ry),',
          f'                  hal_box_{name}(Ah.b, Al.b, Bh.b, Bl.b, Ch.b, Cl.b, Dh.b, Dl.b, rx, ry),',
          f'                  hal_box_{name}(Ah.a, Al.a, Bh.a, Bl.a, Ch.a, Cl.a, Dh.a, Dl.a, rx, ry));']
    L += _clip_lines(wrap)
    L += _colorkey_lines(tex, opts)
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


def _footprint_sampler(name, tex, filt, wrap, opts, bias, lod_key):
    """The footprint road's sampler (the seven roads of R251):
    `_sat_sampler` for SUMMED_AREA (C088); `_mip_sampler` for TRILINEAR
    on the derivative road with no level select (1.89.0's driver log2,
    untouched); `_level_sampler` for every CPU-decided level road -- a Mip
    Level Select (C072), the GS rule (C022), the per-polygon level (C079)
    or Sharpen (C008) -- reading the uploaded `hal_lod_WxH` / `hal_lodq`
    field. GL_CLAMP maps Extend to BORDER here as `Texture.sample` does
    (C077); every sampler carries the chroma-key tail (C083)."""
    wrap = _border_wrap(wrap, opts)
    if filt == 'SUMMED_AREA':
        return _sat_sampler(name, tex, wrap, opts, float(bias))
    if filt == 'TRILINEAR' and lod_key is None:
        _atlas_px, levels = mip_atlas(tex)
        return _mip_sampler(name, tex, wrap, levels, int(opts['aniso']),
                            float(bias), int(opts['frac_bits']), opts=opts)
    if filt not in PYRAMID_FILTERS or lod_key is None:
        raise ValueError(f'no GLSL footprint sampler for {filt} on the '
                         f'{lod_key} level road')
    return _level_sampler(name, tex, filt, wrap, opts, float(bias), lod_key)


def _texture_sampler(name, tex, filt, wrap, opts=None):
    """A manual sampler matching Texture.sample, driver-independent.

    `texture()` with driver filtering would put the sampling arithmetic in
    the driver's hands; fetching texel centres and doing the filter in the
    shader keeps it in ours, which is what makes the GPU pixel the CPU pixel.

    R251: `opts` (from `_tex_opts`) carries the sample-time dials;
    `opts=None` is 1.89.0's text byte for byte. The filter ladder is
    explicit -- an unlisted filter raises instead of falling through to
    bilinear.
    """
    w, h = float(tex.width), float(tex.height)
    frac_bits = int(opts['frac_bits']) if opts is not None else 0
    wrap = _border_wrap(wrap, opts)                       # C077: EXTEND + GL_CLAMP
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
        # the N64's own arithmetic, no footprint needed (C077: under
        # BORDER the coordinate is clamped and the taps select the border)
        L += _uv_clamp_lines(wrap, 'cs', 'ct')
        L += [f'    float fx = {_uvx(wrap, "cs")} * {_f(w)} - 0.5;',
              f'    float fy = {_uvy(wrap, "ct")} * {_f(h)} - 0.5;',
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
              f'    vec4 c11 = hal_fetch_{name}(wx1, wy1);']
        L += _border_select(wrap, 'c00', 'c10', 'c01', 'c11', _f(w), _f(h))
        L += ['    float up = ((tx + ty) > 1.0) ? 1.0 : 0.0;',
              '    vec4 a = (up > 0.5) ? c11 : c00;',
              '    float s = (up > 0.5) ? (1.0 - ty) : tx;',
              '    float t = (up > 0.5) ? (1.0 - tx) : ty;',
              '    vec4 b = (up > 0.5) ? c01 : c10;',
              '    vec4 cc = (up > 0.5) ? c10 : c01;',
              '    vec4 c = a + (b - a) * s + (cc - a) * t;']
    elif filt == 'POV_NORMDIST':
        # C111: exactly Texture._sample_normdist -- POV-Ray's interpolate
        # 4, inverse-square weights to the four texel centres, normalised;
        # one operation per statement (the driver's fma contraction and
        # its `/` ULPs are the stated bar; the simulator is bitwise)
        L += [f'    float xp = uv.x * {_f(w)} + 0.5;',
              f'    float yp = uv.y * {_f(h)} + 0.5;',
              '    float ix = floor(xp);',
              '    float iy = floor(yp);',
              '    float p = xp - ix;',
              '    float q = yp - iy;',
              '    float ixm = ix - 1.0;',
              '    float iym = iy - 1.0;',
              f'    float wx0 = {_wrap_expr("ix", w, wrap)};',
              f'    float wx1 = {_wrap_expr("ixm", w, wrap)};',
              f'    float wy0 = {_wrap_expr("iy", h, wrap)};',
              f'    float wy1 = {_wrap_expr("iym", h, wrap)};',
              f'    vec4 c0 = hal_fetch_{name}(wx0, wy0);',
              f'    vec4 c1 = hal_fetch_{name}(wx1, wy0);',
              f'    vec4 c2 = hal_fetch_{name}(wx0, wy1);',
              f'    vec4 c3 = hal_fetch_{name}(wx1, wy1);',
              '    float pm = 1.0 - p;',
              '    float qm = 1.0 - q;',
              '    float pm2 = pm * pm;',
              '    float qm2 = qm * qm;',
              '    float p2 = p * p;',
              '    float q2 = q * q;',
              '    float d0 = pm2 + qm2;',
              '    float d1 = p2 + qm2;',
              '    float d2 = pm2 + q2;',
              '    float d3 = p2 + q2;',
              '    float w0 = 1.0 / max(d0, 1e-12);',
              '    float w1 = 1.0 / max(d1, 1e-12);',
              '    float w2 = 1.0 / max(d2, 1e-12);',
              '    float w3 = 1.0 / max(d3, 1e-12);',
              '    float s01 = w0 + w1;',
              '    float s012 = s01 + w2;',
              '    float s = s012 + w3;',
              '    vec4 a0 = c0 * w0;',
              '    vec4 a1 = c1 * w1;',
              '    vec4 a2 = c2 * w2;',
              '    vec4 a3 = c3 * w3;',
              '    vec4 t01 = a0 + a1;',
              '    vec4 t012 = t01 + a2;',
              '    vec4 t = t012 + a3;',
              '    vec4 c = t / s;']
    elif filt == 'BILINEAR':
        L += _uv_clamp_lines(wrap)                        # C077
        L += [f'    float fx = {_uvx(wrap)} * {_f(w)} - 0.5;',
              f'    float fy = {_uvy(wrap)} * {_f(h)} - 0.5;',
              '    float x0 = floor(fx);',
              '    float y0 = floor(fy);',
              '    float tx = fx - x0;',
              '    float ty = fy - y0;']
        if frac_bits:
            L += _frac_lines(frac_bits)                  # C080
        L += ['    float x1 = x0 + 1.0;',
              '    float y1 = y0 + 1.0;',
              f'    float wx0 = {_wrap_expr("x0", w, wrap)};',
              f'    float wx1 = {_wrap_expr("x1", w, wrap)};',
              f'    float wy0 = {_wrap_expr("y0", h, wrap)};',
              f'    float wy1 = {_wrap_expr("y1", h, wrap)};',
              f'    vec4 c00 = hal_fetch_{name}(wx0, wy0);',
              f'    vec4 c10 = hal_fetch_{name}(wx1, wy0);',
              f'    vec4 c01 = hal_fetch_{name}(wx0, wy1);',
              f'    vec4 c11 = hal_fetch_{name}(wx1, wy1);']
        L += _border_select(wrap, 'c00', 'c10', 'c01', 'c11', _f(w), _f(h))
        if frac_bits or wrap == 'BORDER':
            # the CPU's own lerp order: bitwise, unlike mix() (C080, C077)
            L += ['    vec4 top = c00 + (c10 - c00) * tx;',
                  '    vec4 bot = c01 + (c11 - c01) * tx;',
                  '    vec4 c = top + (bot - top) * ty;']
        else:
            L += ['    vec4 c = mix(mix(c00, c10, tx), mix(c01, c11, tx), ty);']
    else:
        raise ValueError(f'no GLSL sampler for {filt}')
    if wrap == 'CLIP':
        L.append('    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || '
                 'uv.y > 1.0) { c = vec4(0.0); }')
    L += _colorkey_lines(tex, opts)                        # C083
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


def _cel_composition(lines, bake, consts, affect_diffuse, affect_specular,
                     link_c=None, link_i=None):
    """R218/R228/R238: the cel models' composition for ONE light block
    whose `ds` (the wrapped cosine), `hal_sv` (visibility), `rad`, `L`
    and `V` are in scope -- the cartoon's strongest-verdict
    accumulation or the anime's banded tint, the stepped highlight,
    the airbrush. Shared by the scene's lamps and the fixed key.
    `link_c` is the light's link entry (objects, mode) for the
    cartoon's mask; `link_i` the anime's (index for its mask name).
    Returns True when the block was closed (the cartoon path)."""
    anime = bool(bake.get('__anime'))
    if bake.get('__cartoon'):
        # R228: the paint keeps the STRONGEST lamp's verdict, exactly
        # light_surface's CARTOON branch -- lit term = clamp(wrap) x
        # vis (x the light-link mask), max over lamps; the lamp's
        # radiance only through the Lamp Influence energy; the
        # highlight gate on the wrapped half-vector, inside this lamp's
        # own lit step. The generic tail is bypassed: the block closes
        # here.
        if link_c:
            tests = ' + '.join(f'((abs(td.y - {_f(float(o))}) < 0.5) '
                               '? 1.0 : 0.0)'
                               for o in link_c['objects'])
            lines.append(f'    float hal_clk = min({tests}, 1.0);')
            if str(link_c.get('mode', 'EXCLUDE')).upper() == 'ONLY':
                lines.append('    float hal_cm = hal_clk;')
            else:
                lines.append('    float hal_cm = 1.0 - hal_clk;')
        else:
            lines.append('    float hal_cm = 1.0;')
        lines.append('    float hal_cxr = clamp(ds.x, 0.0, 1.0) '
                     '* hal_sv * hal_cm;')
        if affect_diffuse:
            lines.append('    float hal_cx = hal_cxr;')
        else:
            lines.append('    float hal_cx = 0.0;')
        lines += [
            '    hal_cart_lit = max(hal_cart_lit, hal_cx);',
            '    hal_cart_rad += rad * hal_cx * 0.318309886;',
        ]
        if affect_specular:
            lines += [
                '    vec3 hal_ch = normalize(L + V);',
                '    float hal_csn = clamp(dot(N, hal_ch) * 0.5 + 0.5, '
                '0.0, 1.0);',
                '    float hal_chs = clamp(s.cartoon_hl_size, 0.0, 1.0);',
                '    float hal_ced = 1.0 - 0.25 * hal_chs * hal_chs;',
                '    float hal_cts = clamp((hal_csn - (hal_ced '
                '- s.cartoon_hl_soft)) / max(2.0 * s.cartoon_hl_soft, '
                '1e-6), 0.0, 1.0);',
                '    float hal_cg = hal_cts * hal_cts '
                '* (3.0 - 2.0 * hal_cts);',
                '    hal_cg = (s.cartoon_hl_size > 1e-6) ? hal_cg : 0.0;',
                '    float hal_ctl = clamp((hal_cxr - (s.cartoon_th '
                '- s.cartoon_soft)) / max(2.0 * s.cartoon_soft, 1e-6), '
                '0.0, 1.0);',
                '    hal_ctl = hal_ctl * hal_ctl * (3.0 - 2.0 * hal_ctl);',
                '    hal_cart_hl = max(hal_cart_hl, hal_cg * hal_ctl);',
            ]
        lines.append('    }')
        return True
    if anime:
        # R218: the cel composition, exactly render._anime_lamp -- the
        # shadow term inside the band input, tint multiplying the base,
        # the stepped highlight on the wrapped half-vector, 1/pi and
        # the per-material gain folded here. The generic assembly and
        # its 1/pi-times-vis tail are bypassed below.
        lines += [
            '    vec3 hal_arad = rad * s.anime_gain;',
            '    float hal_ax = clamp(ds.x + s.anime_bias, 0.0, 1.0)'
            ' * hal_sv;',
        ]
        fspec = bake.get('__anime_face')
        if fspec:
            # R239: the SDF face shadow -- the map's field against the
            # light's horizontal angle about the face's frame REPLACES
            # the lambert wrap, exactly render._anime_lamp's face road:
            # same projection, same acos, same endpoint-mapped bilinear
            # on the same uploaded texels, the side mirrored across the
            # face's centre line
            ffw = _v3(fspec['fwd'])
            fup = _v3(fspec['up'])
            frt = _v3(fspec['right'])
            fw = float(fspec['w'])
            fh = float(fspec['h'])
            fv0 = float(fspec['v0'])
            lines += [
                f'    vec3 hal_flh = L - {fup} * dot(L, {fup});',
                '    float hal_fln = max(length(hal_flh), 1e-9);',
                '    hal_flh = hal_flh / hal_fln;',
                f'    float hal_fct = clamp(dot(hal_flh, {ffw}), '
                '-1.0, 1.0);',
                '    float hal_ft = acos(hal_fct) * 0.318309873;',
                f'    float hal_fsde = dot(hal_flh, {frt});',
                '    float hal_fu = (hal_fsde >= 0.0) ? hal_uv.x '
                ': (1.0 - hal_uv.x);',
                f'    float hal_ffx = clamp(hal_fu, 0.0, 1.0) '
                f'* {_f(fw - 1.0)};',
                f'    float hal_ffy = clamp(hal_uv.y, 0.0, 1.0) '
                f'* {_f(fh - 1.0)};',
                '    float hal_fx0 = floor(hal_ffx);',
                '    float hal_fy0 = floor(hal_ffy);',
                '    float hal_ftx = hal_ffx - hal_fx0;',
                '    float hal_fty = hal_ffy - hal_fy0;',
                f'    float hal_fx1 = min(hal_fx0 + 1.0, {_f(fw - 1.0)});',
                f'    float hal_fy1 = min(hal_fy0 + 1.0, {_f(fh - 1.0)});',
                f'    float hal_fc00 = texelFetch(hal_facesdf, '
                f'ivec2(int(hal_fx0), int(hal_fy0 + {_f(fv0)})), 0).r;',
                f'    float hal_fc10 = texelFetch(hal_facesdf, '
                f'ivec2(int(hal_fx1), int(hal_fy0 + {_f(fv0)})), 0).r;',
                f'    float hal_fc01 = texelFetch(hal_facesdf, '
                f'ivec2(int(hal_fx0), int(hal_fy1 + {_f(fv0)})), 0).r;',
                f'    float hal_fc11 = texelFetch(hal_facesdf, '
                f'ivec2(int(hal_fx1), int(hal_fy1 + {_f(fv0)})), 0).r;',
                '    float hal_ftop = hal_fc00 + (hal_fc10 - hal_fc00)'
                ' * hal_ftx;',
                '    float hal_fbot = hal_fc01 + (hal_fc11 - hal_fc01)'
                ' * hal_ftx;',
                '    float hal_fval = hal_ftop + (hal_fbot - hal_ftop)'
                ' * hal_fty;',
                '    hal_ax = clamp(0.5 + (hal_fval - hal_ft), '
                '0.0, 1.0) * hal_sv;',
            ]
        rspec = bake.get('__anime_ramp')
        if rspec:
            # R221: the tint IS the baked ramp, sampled by the light
            # term with _anime_ramp_sample's own endpoint-mapped
            # bilinear arithmetic on the SAME uploaded texels
            rw = float(rspec['w'])
            rh = float(rspec['h'])
            rv0 = float(rspec['v0'])
            lines += [
                f'    float hal_rfx = clamp(hal_ax, 0.0, 1.0) '
                f'* {_f(rw - 1.0)};',
                f'    float hal_rfy = clamp(s.anime_ramp_row, 0.0, '
                f'1.0) * {_f(rh - 1.0)};',
                '    float hal_rx0 = floor(hal_rfx);',
                '    float hal_ry0 = floor(hal_rfy);',
                '    float hal_rtx = hal_rfx - hal_rx0;',
                '    float hal_rty = hal_rfy - hal_ry0;',
                f'    float hal_rx1 = min(hal_rx0 + 1.0, {_f(rw - 1.0)});',
                f'    float hal_ry1 = min(hal_ry0 + 1.0, {_f(rh - 1.0)});',
                f'    vec3 hal_rc00 = texelFetch(hal_animeramp, '
                f'ivec2(int(hal_rx0), int(hal_ry0 + {_f(rv0)})), 0).rgb;',
                f'    vec3 hal_rc10 = texelFetch(hal_animeramp, '
                f'ivec2(int(hal_rx1), int(hal_ry0 + {_f(rv0)})), 0).rgb;',
                f'    vec3 hal_rc01 = texelFetch(hal_animeramp, '
                f'ivec2(int(hal_rx0), int(hal_ry1 + {_f(rv0)})), 0).rgb;',
                f'    vec3 hal_rc11 = texelFetch(hal_animeramp, '
                f'ivec2(int(hal_rx1), int(hal_ry1 + {_f(rv0)})), 0).rgb;',
                '    vec3 hal_rtop = hal_rc00 + (hal_rc10 - hal_rc00)'
                ' * hal_rtx;',
                '    vec3 hal_rbot = hal_rc01 + (hal_rc11 - hal_rc01)'
                ' * hal_rtx;',
                '    vec3 hal_tint = hal_rtop + (hal_rbot - hal_rtop)'
                ' * hal_rty;',
            ]
        else:
            lines += [
            '    float hal_t1 = clamp((hal_ax - (s.anime_th1 '
            '- s.anime_soft1)) / max(2.0 * s.anime_soft1, 1e-6), '
            '0.0, 1.0);',
            '    float hal_b1 = hal_t1 * hal_t1 '
            '* (3.0 - 2.0 * hal_t1);',
            '    float hal_t2 = clamp((hal_ax - (s.anime_th2 '
            '- s.anime_soft2)) / max(2.0 * s.anime_soft2, 1e-6), '
            '0.0, 1.0);',
            '    float hal_b2 = hal_t2 * hal_t2 '
            '* (3.0 - 2.0 * hal_t2);',
            '    float hal_t3 = clamp(s.anime_tones - 2.0, 0.0, 1.0);',
            '    vec3 hal_sh3 = s.anime_shadow2 + (s.anime_shadow1 '
            '- s.anime_shadow2) * hal_b2;',
            '    vec3 hal_shade = s.anime_shadow1 + (hal_sh3 '
            '- s.anime_shadow1) * hal_t3;',
            '    vec3 hal_tint = hal_shade + (vec3(1.0) - hal_shade)'
            ' * hal_b1;',
            ]
        if 'anime_air' in perpix_names(bake) or \
                float(bake.get('anime_air', 0.0) or 0.0) > 1e-6:
            # R229: render._anime_airbrush, the same operations -- the
            # lit-side multiply toward the airbrush colour just above
            # the first threshold, the shadow-side blend toward it
            # just below, by side code (0 lit, 1 shadow, 2 both)
            lines += _airbrush_lines('hal_tint', 'hal_ax', 's.anime_th1')
        if affect_diffuse:
            lines.append('    hal_dcon = (s.diffuse * hal_tint '
                         '* s.diffuse_level) * hal_arad '
                         '* 0.318309886;')
        if affect_specular:
            lines += [
                '    vec3 hal_ah = normalize(L + V);',
                '    float hal_asn = clamp(dot(N, hal_ah) * 0.5 '
                '+ 0.5, 0.0, 1.0);',
                '    float hal_aed = 1.0 - clamp(s.anime_spec_size, '
                '0.0, 1.0);',
                '    float hal_ts = clamp((hal_asn - (hal_aed '
                '- s.anime_sharp)) / max(2.0 * s.anime_sharp, 1e-6), '
                '0.0, 1.0);',
                '    float hal_ag = hal_ts * hal_ts '
                '* (3.0 - 2.0 * hal_ts);',
                '    hal_scon = ((hal_ag * s.anime_mask * hal_sv) '
                '* s.specular * s.specular_level) * hal_arad '
                '* 0.318309886;',
            ]
    return False


def _airbrush_lines(tint_var, x_var, th_expr):
    """R229/R238: render._anime_airbrush's operations on `tint_var`
    against the band input `x_var` at the edge `th_expr` -- the anime
    master's first threshold, or the cartoon's."""
    return [
        '    float hal_aam = clamp(s.anime_air, 0.0, 1.0);',
        '    float hal_awd = max(s.anime_air_width, 1e-4);',
        f'    float hal_agl = (1.0 - hal_sstep({th_expr}, '
        f'{th_expr} + hal_awd, {x_var})) '
        f'* hal_sstep({th_expr} - 1e-4, {th_expr}, {x_var});',
        f'    float hal_ags = hal_sstep({th_expr} - hal_awd, '
        f'{th_expr}, {x_var}) * (1.0 - hal_sstep({th_expr}, '
        f'{th_expr} + 1e-4, {x_var}));',
        '    float hal_asd = floor(s.anime_air_side + 0.5);',
        '    float hal_akl = hal_aam * hal_agl '
        '* ((hal_asd != 1.0) ? 1.0 : 0.0);',
        '    float hal_aks = hal_aam * hal_ags '
        '* ((hal_asd != 0.0) ? 1.0 : 0.0);',
        f'    {tint_var} = {tint_var} * (vec3(1.0) '
        f'+ (s.anime_air_color - vec3(1.0)) * hal_akl);',
        f'    {tint_var} = {tint_var} + (s.anime_air_color - {tint_var}) '
        '* hal_aks;',
    ]


def _cel_key_source(consts, bake):
    """R238: the fixed key's own block, after the lamps -- exactly
    light_surface's cel_mode > 0 tail: the key's world direction (the
    camera's axes times the material's own-frame vector, in that
    order, or the world vector itself), a radiance of pi (the lit tone
    is the colour as painted), the strongest caster's visibility
    (hal_vkey, -1 when no caster reached the object: lit) times the
    screen shadow, then the cel composition a lamp gets."""
    lines = ['    {']
    mode = int(round(float(bake.get('cel_light', 0.0))))
    if mode == 1:
        lines.append('    vec3 L = normalize(hal_cam_right * s.cel_dir.x '
                     '+ hal_cam_up * s.cel_dir.y '
                     '+ hal_cam_back * s.cel_dir.z);')
    else:
        lines.append('    vec3 L = normalize(s.cel_dir);')
    lines += ['    vec3 rad = vec3(3.14159265);',
              '    float hal_sv = (hal_vkey >= 0.0) ? hal_vkey : 1.0;']
    if bake.get('__cel_field') and float(bake.get('cel_ss', 0.0) or 0.0) > 1e-6:
        lines.append('    hal_sv = hal_sv * (1.0 - hal_cel.r '
                     '* clamp(s.cel_ss, 0.0, 1.0));')
    lines += ['    vec3 hal_vs = ' + viewer_expr(consts, bake) + ';',
              '    vec4 ds = hal_evaluate(hal_model_i, s, N, L, hal_vs);',
              '    vec3 hal_dcon = vec3(0.0);',
              '    vec3 hal_scon = vec3(0.0);']
    closed = _cel_composition(lines, bake, consts, True, True)
    if closed:
        return lines
    lines.append('    vec3 contrib = hal_dcon + hal_scon;')
    clamp = float(consts.get('light_clamp', 0.0))
    if clamp > 0.0:
        lines.append(f'    contrib = min(contrib, vec3({_f(clamp)}));')
    lines.append('    total += contrib;')
    lines.append('    }')
    return lines


def _one_light_source(i, light, consts, shadowed=None, bake=None, bi=None):
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
    # R169: the light's VALUES come from the hal_lights texture (see
    # LIGHT_TEXEL_STRIDE), so a lamp edit re-uploads a texel instead of
    # changing this source and recompiling every pass. Only STRUCTURE
    # bakes below: the light's type, decay mode, which sliders are
    # nonzero, shadow mode, cookies, linking.
    use_tx = bool(consts.get('light_texels', True))
    col = _lref(i, 2, 'rgb') if use_tx else \
        _v3(getattr(light, 'color', (1, 1, 1)))
    energy = float(getattr(light, 'energy', 1.0))
    lines = ['    {',
             # R251 F011: the viewer the reflectance models take -- the
             # true eye vector, or the frame's camera axis (one texel)
             f'    vec3 hal_vs = {viewer_expr(consts, bake)};']
    if kind == 'SPOT' and getattr(light, 'screen_spot', False):
        # R251 F015: the Model 3 screen spotlight, exactly light_surface's
        # early branch. The lamp lives on the SCREEN: no block at all on
        # a secondary (hit) or layer pass (the CPU's px is None there,
        # A10) and none under a fixed cel key (it casts no shadow, A53).
        # Values ride hal_lights texels 6-7 (per frame); depth is the
        # F000 expression on the view row (hal_fogtab texel 3, so the
        # frame's consts['fogtab'] rule must cover a screen-spot lamp).
        if bake.get('__cel_key') or not consts.get('__screen_pass', True):
            return []
        w_px, h_px = consts['resolution']
        r = {'cx': _lref(i, 6, 'x'), 'cy': _lref(i, 6, 'y'),
             'w': _lref(i, 6, 'z'), 'h': _lref(i, 6, 'w'),
             'start': _lref(i, 7, 'x'), 'aext': _lref(i, 7, 'y')}
        lines += [
            f'    int ss_ix = int(vUV.x * {_f(float(w_px))});',
            f'    int ss_iy = int(vUV.y * {_f(float(h_px))});',
            '    float ss_px = float(ss_ix);',
            '    float ss_py = float(ss_iy);',
            '    vec3 ss_dp = P - hal_eye;',
            '    vec3 ss_r = texelFetch(hal_fogtab, ivec2(3, 0), 0).xyz;',
            '    float ss_dz = ss_dp.x * ss_r.x;',
            '    ss_dz += ss_dp.y * ss_r.y;',
            '    ss_dz += ss_dp.z * ss_r.z;',
            '    float ss_depth = abs(ss_dz);']
        lines += screen_spot_glsl(r)
        lines += [
            f'    vec3 ss_col = {col} * {_lref(i, 0, "w")};',
            '    vec3 hal_sst = (s.diffuse * s.diffuse_level) * ss_col;',
            '    hal_sst = hal_sst * ss_lobe;',
            '    hal_sst = hal_sst * 0.318309886;',
            '    vec3 ss_fog = ss_col * (ss_en * ss_el);']
        link_ss = (consts.get('light_links') or {}).get(i)
        if link_ss:
            tests = ' + '.join(f'((abs(td.y - {_f(float(o))}) < 0.5) '
                               '? 1.0 : 0.0)'
                               for o in link_ss['objects'])
            lines.append(f'    float hal_lk{i} = min({tests}, 1.0);')
            mask = f'hal_lk{i}' \
                if str(link_ss.get('mode', 'EXCLUDE')).upper() == 'ONLY' \
                else f'(1.0 - hal_lk{i})'
            lines += [f'    hal_sst = hal_sst * {mask};',
                      f'    ss_fog = ss_fog * {mask};']
        lines += ['    total += hal_sst;',
                  '    hal_spotfog += ss_fog;',
                  '    }']
        return lines
    ck = (consts.get('cookies') or {}).get(i)
    if kind in ('SUN', 'HEMI'):
        d = np.asarray(light.direction, np.float32)
        d = d / max(float(np.linalg.norm(d)), 1e-9)
        if use_tx:
            lines += [f'    vec3 L = {_lref(i, 1, "xyz")};',
                      f'    vec3 rad = {col} * {_lref(i, 0, "w")};']
        else:
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
        mode_bi = str(getattr(light, 'decay', '')).startswith('BI_') or \
            getattr(light, 'bi_sphere', False)
        refs = None
        if use_tx:
            refs = {'D': _lref(i, 3, 'y'), 'start': _lref(i, 3, 'x'),
                    'ld1': _lref(i, 3, 'z'), 'ld2': _lref(i, 3, 'w')}
        pos = _lref(i, 0, 'xyz') if use_tx else _v3(light.position)
        e_pre = _lref(i, 0, 'w') if use_tx else \
            _f(energy / (4.0 * np.pi))
        lines += [f'    vec3 delta = {pos} - P;',
                  '    float dist = max(length(delta), 1e-6);',
                  '    vec3 L = delta / dist;']
        dmode = str(getattr(light, 'decay', 'DEFAULT') or 'DEFAULT')
        if dmode == 'DEFAULT':
            dmode = str(consts['falloff_default'])
        if kind == 'AREA':
            # lamp_get_visibility skips its falloff switch for LA_AREA:
            # visifac stays 1.0 and distance speaks only through the
            # form factor -- exactly the CPU sample()'s area branch
            lines.append('    float att = 1.0;')
        elif dmode in ('GL_3TERM', 'POV_FADE_LINEAR', 'POV_FADE_SQUARE',
                       'GX_GENTLE', 'GX_MEDIUM', 'GX_STEEP'):
            # R251 F013: the GL / POV / GX decay laws, one op per
            # statement, exactly lights.decay_law (the mode is already
            # _light_sig structure; D, the sliders and the GX
            # coefficients are texels)
            from ..core import lights as _LI
            if use_tx:
                drefs = {'D': _lref(i, 3, 'y'), 'ld1': _lref(i, 3, 'z'),
                         'ld2': _lref(i, 3, 'w'), 'k1': _lref(i, 5, 'y'),
                         'k2': _lref(i, 5, 'z')}
            else:
                _k1, _k2 = _LI.gx_dist_coeffs(light, dmode)
                drefs = {'D': _f(max(float(getattr(light, 'decay_end',
                                                   25.0)), 1e-6)),
                         'ld1': _f(float(getattr(light, 'decay_ld1', 0.0)
                                         or 0.0)),
                         'ld2': _f(float(getattr(light, 'decay_ld2', 0.0)
                                         or 0.0)),
                         'k1': _f(_k1), 'k2': _f(_k2)}
            lines += decay_law_glsl(dmode, drefs)
            if getattr(light, 'bi_sphere', False):
                # LA_SPHERE outside the falloff switch, exactly
                # _attenuation's wrap
                _D = drefs['D']
                lines.append(f'    att = (max(dist, 0.0) < {_D}) ? (att '
                             f'* (({_D} - max(dist, 0.0)) / {_D})) : 0.0;')
        else:
            lines.append(
                f'    float att = '
                f'{_attenuation(light, consts["falloff_default"], refs=refs)};')
        if mode_bi and kind not in ('SPOT', 'AREA'):
            # lamp_get_visibility's tail: visifac <= 0.001 snaps to 0
            lines.append('    att = (att <= 0.001) ? 0.0 : att;')
        lines += [f'    vec3 rad = {col} * ({e_pre} * att);']
        if kind == 'SPOT':
            # BI's spot, verbatim (R155): hard cutoff at spotsi, a
            # smoothstep across spotbl = (1-spotsi)*blend in cosine
            # units, then the whole cone multiplied by the raw cosine
            sd = np.asarray(light.direction, np.float32)
            sd = sd / max(float(np.linalg.norm(sd)), 1e-9)
            spotsi = float(np.cos(float(light.spot_size) * 0.5))
            spotbl = (1.0 - spotsi) * float(light.spot_blend)
            sd_r = _lref(i, 1, 'xyz') if use_tx else _v3(sd)
            si_r = _lref(i, 1, 'w') if use_tx else _f(spotsi)
            bl_r = _lref(i, 2, 'w') if use_tx else _f(spotbl)
            law = str(getattr(light, 'spot_law', 'BLENDER') or 'BLENDER')
            lines.append(f'    float cosang = -dot(L, {sd_r});')
            if law in ('GL11', 'POV', 'GX_FLAT', 'GX_COS', 'GX_COS2',
                       'GX_SHARP', 'GX_RING1', 'GX_RING2'):
                # R251 F012: the OpenGL 1.1 / POV-Ray / GX cone laws,
                # exactly lights.spot_law_factor; the law and whether
                # the exponent is nonzero are _light_sig structure, the
                # exponent, POV's radius and the GX coefficients texels
                exp_v = float(getattr(light, 'spot_exponent', 0.0) or 0.0)
                if use_tx:
                    srefs = {'si': si_r, 'exp': _lref(i, 4, 'x'),
                             'cr': _lref(i, 4, 'y'), 'a0': _lref(i, 4, 'z'),
                             'a1': _lref(i, 4, 'w'), 'a2': _lref(i, 5, 'x')}
                else:
                    from ..core import lights as _LI
                    _cr, _a0, _a1, _a2 = _LI.spot_law_coeffs(light)
                    srefs = {'si': si_r, 'exp': _f(np.float32(exp_v)),
                             'cr': _f(_cr), 'a0': _f(_a0), 'a1': _f(_a1),
                             'a2': _f(_a2)}
                lines += spot_law_glsl(law, exp_v != 0.0, srefs)
            else:
                lines += [
                    f'    float spot_t = cosang - {si_r};',
                    '    float spot_f = 0.0;',
                    '    if (spot_t > 0.0) {']
                if spotbl != 0.0:
                    # whether a blend EXISTS is structure; its width is
                    # a texel (crossing zero re-plans, dragging it does
                    # not)
                    lines += [
                        f'        float spot_i = clamp(spot_t / '
                        f'{bl_r}, 0.0, 1.0);',
                        f'        float spot_s = (spot_t < {bl_r}) ? '
                        '(3.0 * spot_i * spot_i - 2.0 * spot_i * spot_i '
                        '* spot_i) : 1.0;',
                        '        spot_f = spot_s * cosang;']
                else:
                    lines += ['        spot_f = cosang;']
                lines += ['    }']
            if mode_bi:
                # the combined visifac (att * spot) snaps at 0.001,
                # exactly the CPU's sample()
                lines.append('    spot_f = ((att * spot_f) <= 0.001) '
                             '? 0.0 : spot_f;')
            lines.append('    rad = rad * spot_f;')
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
        if ck is not None and kind == 'POINT':
            # exactly cookie_factor's POINT branch: light->surface
            # direction in the lamp's own frame, unrolled lat-long --
            # atan2 around the forward axis, asin from the equator
            lines += [
                '    vec3 ckd = -L;',
                f'    float ckx = dot(ckd, {_v3(ck["side"])});',
                f'    float cky = dot(ckd, {_v3(ck["up"])});',
                f'    float ckz = dot(ckd, {_v3(ck["fwd"])});',
                '    vec2 ckuv = vec2(atan(ckx, ckz) * '
                f'{_f(float(np.float32(1.0 / (2.0 * np.pi))))} + 0.5, '
                'asin(clamp(cky, -1.0, 1.0)) * '
                f'{_f(float(np.float32(1.0 / np.pi)))} + 0.5);',
                f'    rad = rad * (vec3(1.0) + (hal_cookie_rgb{i}(ckuv) '
                f'- vec3(1.0)) * {_f(ck["strength"])});']
        if ck is not None and kind == 'AREA':
            # exactly cookie_factor's AREA branch: parallel throw off
            # the face, offset in the lamp plane over the face size;
            # behind the face the projection is undefined (factor 1)
            lines += [
                '    vec3 ckrel = -delta;',
                f'    float ckz = dot(ckrel, {_v3(ck["fwd"])});',
                '    if (ckz > 1e-6) {',
                f'    vec2 ckuv = vec2(dot(ckrel, {_v3(ck["side"])}) / '
                f'{_f(ck["sx"])} + 0.5, dot(ckrel, {_v3(ck["up"])}) / '
                f'{_f(ck["sy"])} + 0.5);',
                f'    rad = rad * (vec3(1.0) + (hal_cookie_rgb{i}(ckuv) '
                f'- vec3(1.0)) * {_f(ck["strength"])});',
                '    }']
    if getattr(light, 'negative', False) and not use_tx:
        # texel mode folds the sign into the packed energy (float32
        # negation is exact), so toggling Negative never recompiles
        lines.append('    rad = -rad;')
    bi = bi or {}
    rd = bi.get('ramp_dif')
    rs_ = bi.get('ramp_spec')
    # R251 (LIGHT-B1 F011 / LIGHT-B2 F016-F018): the view vector the
    # models take -- V, or ONE camera axis for the whole frame (hal_fogtab
    # texel 227: OpenGL 1.1's infinite viewer, the Sega boards' R.z, the
    # DS's line of sight) under Specular Viewer AXIS or for the console
    # light units (models 32 / 35), exactly light_surface's Vs. The Hemi
    # override keeps V, as the CPU's does.
    _model_i = int(bake.get('__model_i', -1))
    # (integrator) hal_vs is declared ONCE at the block's head -- LIGHT-B1's
    # viewer_expr, which also selects the axis for the console models
    _vs = 'hal_vs'
    if kind == 'HEMI':
        # BI's Hemi replaces the shaders on BOTH lobes, exactly
        # light_surface's override: the 0.5+0.5*N.L wrap and a wrapped
        # half-vector pow through the surface's own hardness and tint
        lines += [
            # R243: the override never runs hal_evaluate, so the coloured
            # diffuse slot reads the diffuse socket
            '    hal_dif_rgb = s.diffuse;',
            '    float hal_hd = 0.5 * dot(N, L) + 0.5;',
            '    float hal_ht = 0.5 * dot(N, normalize(L + V)) + 0.5;',
            # verbatim (R155): t = spec(t, shi->har) -- the integer-bit
            # spec() table, same as the CPU override
            '    vec4 ds = vec4(hal_hd, s.specular '
            '* hal_bi_spec_pow(hal_ht, s.glossiness));']
    elif kind == 'AREA':
        # BI's area lamp: the Stokes energy stands in for the diffuse
        # cosine and multiplies the finished specular (shade_one_light:
        # inp = area_lamp_energy_multisample, then specfac *= inp).
        # .y is the flipped-normal twin, consumed by BI translucency
        lines += [
            f'    vec2 hal_ainp = hal_area_inp{i}(P, N);',
            f'    vec4 ds = hal_evaluate2(hal_model_i, s, N, L, {_vs}, '
            'hal_ainp.x, hal_ainp.y, 1.0);']
        # R251: the console and POV finish terms sit BEFORE the area
        # correction, exactly light_surface's order (spec * area_nd last)
        lines += _console_finish_lines(kind, bake, consts, _model_i, i)
        lines.append('    ds.yzw *= hal_ainp.x;')
    else:
        lines.append(
            f'    vec4 ds = hal_evaluate(hal_model_i, s, N, L, {_vs});')
        lines += _console_finish_lines(kind, bake, consts, _model_i, i)
    # the shadow term -- evaluated AFTER the shaders, exactly the CPU's
    # R167 order (SH.evaluate before visibility), and R177 adds the
    # CPU's own skip to the twin: the traversal runs ONLY where this
    # lamp actually contributes (some shader channel nonzero AND the
    # attenuated radiance nonzero). A gated-out pixel keeps hal_sv at
    # 1.0 and contributes zero through every term that reads it -- the
    # ramp ENERGY factor, the shadow-colour tint, the receive fold --
    # so the picture cannot move; only the wasted BVH walks and map
    # taps stop. Shadows Only materials never reach this pass (they
    # refuse the GPU by name), so the gate needs no exception.
    if bake.get('__cel_key'):
        # R238: under a fixed key the scene's lamps only CAST their
        # shadows -- exactly light_surface's cel_mode > 0 branch: no
        # block at all for a lamp with no shadow term; a caster's
        # visibility (the CPU's own call, no contribution gate: the
        # mask is the link) joins the strongest-caster accumulator, -1
        # where the lamp does not reach this object
        if not shadowed:
            return []
        if isinstance(shadowed, dict) and shadowed.get('ray'):
            dist_arg = '1e9' if kind == 'SUN' else 'dist'
            _sv_call = f'hal_shadow_vis{i}(P, N, L, {dist_arg})'
        else:
            _sv_call = f'hal_shadow_vis{i}(P, N, L)'
        lines.append(f'    float hal_svk = {_sv_call};')
        link = (consts.get('light_links') or {}).get(i)
        if link:
            tests = ' + '.join(f'((abs(td.y - {_f(float(o))}) < 0.5) '
                               '? 1.0 : 0.0)'
                               for o in link['objects'])
            lines.append(f'    float hal_lk{i} = min({tests}, 1.0);')
            mask = f'hal_lk{i}' if str(link.get('mode', 'EXCLUDE')).upper() \
                == 'ONLY' else f'(1.0 - hal_lk{i})'
            lines.append(f'    hal_svk = ({mask} > 0.5) ? hal_svk : -1.0;')
        lines.append('    hal_vkey = max(hal_vkey, hal_svk);')
        lines.append('    }')
        return lines
    if shadowed:
        if isinstance(shadowed, dict) and shadowed.get('ray'):
            dist_arg = '1e9' if kind == 'SUN' else 'dist'
            _sv_call = f'hal_shadow_vis{i}(P, N, L, {dist_arg})'
        else:
            _sv_call = f'hal_shadow_vis{i}(P, N, L)'
        lines += [
            '    float hal_sv = 1.0;',
            '    if ((abs(ds.x) + abs(ds.y) + abs(ds.z) + abs(ds.w)) '
            '!= 0.0',
            '            && (abs(rad.x) + abs(rad.y) + abs(rad.z)) '
            '!= 0.0) {',
            f'        hal_sv = {_sv_call};',
            # Shadow > Receive off: shadows never darken this material
            # -- folds to a no-op at the baked default of 1.0 (and a
            # skipped pixel's 1.0 folds to 1.0 identically)
            '        hal_sv = 1.0 - s.shadow_receive '
            '+ s.shadow_receive * hal_sv;',
            '    }']
    else:
        lines.append('    float hal_sv = 1.0;')
    if bake.get('__cel_field') and int(consts.get('cel_key_index', -1)) == i \
            and float(bake.get('cel_ss', 0.0) or 0.0) > 1e-6:
        # R238: the screen shadow rides the scene's key lamp, after the
        # receive fold, exactly light_surface's order
        lines.append('    hal_sv = hal_sv * (1.0 - hal_cel.r '
                     '* clamp(s.cel_ss, 0.0, 1.0));')
    if getattr(light, 'only_shadow', False):
        # R251 F014: Blender Internal's LA_ONLYSHADOW, exactly
        # light_surface's hook -- the plain diffuse the lamp would have
        # given, times (1 - vis)*(1 - shadow colour), subtracted; no
        # specular, no ramp, no clamp; nothing when the lobe flags give
        # no diffuse. Same association as the CPU:
        # (((dif*col*level)*rad)*inv_pi)*dark. The block closes here:
        # an only-shadow lamp never reaches the contribution tail.
        if getattr(light, 'affect_diffuse', True) and \
                not getattr(light, 'specular_only', False):
            _oshc = tuple(getattr(light, 'shadow_color', (0.0, 0.0, 0.0))
                          or (0.0, 0.0, 0.0))
            lines += [
                '    vec3 hal_osh = (ds.x * hal_dif_rgb * s.diffuse_level)'
                ' * rad;',
                '    hal_osh = hal_osh * 0.318309886;',
                '    vec3 hal_osd = vec3(1.0 - hal_sv);']
            if max(_oshc) > 0.0:
                lines.append(f'    hal_osd = hal_osd * (vec3(1.0) - '
                             f'{_v3(_oshc)});')
            lines.append('    hal_osh = hal_osh * hal_osd;')
            link_os = (consts.get('light_links') or {}).get(i)
            if link_os:
                tests = ' + '.join(f'((abs(td.y - {_f(float(o))}) < 0.5) '
                                   '? 1.0 : 0.0)'
                                   for o in link_os['objects'])
                lines.append(f'    float hal_lk{i} = min({tests}, 1.0);')
                mask = f'hal_lk{i}' \
                    if str(link_os.get('mode', 'EXCLUDE')).upper() == 'ONLY' \
                    else f'(1.0 - hal_lk{i})'
                lines.append(f'    hal_osh = hal_osh * {mask};')
            if (bi or {}).get('result_mode'):
                lines.append('    hal_dacc -= hal_osh;')
            else:
                lines.append('    total = total - hal_osh;')
        lines.append('    }')
        return lines
    lines.append('    vec3 hal_dcon = vec3(0.0);')
    lines.append('    vec3 hal_scon = vec3(0.0);')
    anime = bool(bake.get('__anime'))
    if bake.get('__cartoon') or anime:
        link_c = (consts.get('light_links') or {}).get(i)
        closed = _cel_composition(
            lines, bake, consts,
            getattr(light, 'affect_diffuse', True)
            and not getattr(light, 'specular_only', False),
            getattr(light, 'affect_specular', True)
            and not getattr(light, 'diffuse_only', False),
            link_c=link_c if bake.get('__cartoon') else None)
        if closed:
            return lines
    if not anime and getattr(light, 'affect_diffuse', True) and \
            not getattr(light, 'specular_only', False):
        if rd is not None and rd.get('input') != 'RESULT':
            # the diffuse ramp, per light, exactly add_to_diffuse: the
            # band recolours the DIFFUSE COLOUR, the shader scalar stays
            fac = {'ENERGY': '(ds.x * hal_sv * '
                             'dot(rad, vec3(0.3, 0.58, 0.11)))',
                   'NORMAL': '(dot(N, V))'}.get(
                       rd.get('input', 'SHADER'), 'ds.x')
            bidx = _ramp_blend_index(rd.get('blend', 'MIX'))
            lines += [
                f'    vec4 hal_bd = hal_biband_dif({fac});',
                f'    vec3 hal_dcol = hal_ramp_blend({bidx}, s.diffuse, '
                f'hal_bd.a * {_mv(consts, float(rd.get("factor", 1.0)))}, '
                'hal_bd.rgb);',
                '    hal_dcon += (ds.x * hal_dcol * s.diffuse_level)'
                ' * rad;']
        else:
            # R243: hal_dif_rgb is s.diffuse for every model but the Max
            # shaders whose diffuse carries its own colour
            lines.append('    hal_dcon += (ds.x * hal_dif_rgb * '
                         's.diffuse_level) * rad;')
    if not anime and getattr(light, 'affect_specular', True) and \
            not getattr(light, 'diffuse_only', False):
        spec = 'ds.yzw'
        if rs_ is not None and rs_.get('input') != 'RESULT':
            # recover the shader scalar from the coloured return (spec =
            # scalar * specular colour; argmax channel, ties to red,
            # exactly the CPU's np.argmax) -- a black specular colour
            # has no recoverable scalar and stays dark
            lines += [
                '    float hal_smx = max(s.specular.r, '
                'max(s.specular.g, s.specular.b));',
                '    float hal_sr = (s.specular.r >= s.specular.g && '
                's.specular.r >= s.specular.b) ? ds.y : '
                '((s.specular.g >= s.specular.b) ? ds.z : ds.w);',
                '    float hal_sfac = (hal_smx > 1e-9) ? '
                'hal_sr / max(hal_smx, 1e-9) : 0.0;']
            fac = {'ENERGY': '(hal_sfac * hal_sv * '
                             'dot(rad, vec3(0.3, 0.58, 0.11)))',
                   'NORMAL': '(dot(N, V))'}.get(
                       rs_.get('input', 'SHADER'), 'hal_sfac')
            bidx = _ramp_blend_index(rs_.get('blend', 'MIX'))
            lines += [
                f'    vec4 hal_bs = hal_biband_spec({fac});',
                f'    vec3 hal_scol = hal_ramp_blend({bidx}, s.specular, '
                f'hal_bs.a * {_mv(consts, float(rs_.get("factor", 1.0)))}, '
                'hal_bs.rgb);',
                '    vec3 hal_sp = hal_sfac * hal_scol;']
            spec = 'hal_sp'
        if not consts.get('specular_in_gamma', True):
            spec = f'pow(max({spec}, vec3(0.0)), vec3(2.2))'
        if bake.get('__level_free'):
            # R243: Strauss has no Specular Level in Max, Multi-Layer
            # applies its two levels inside -- the loop scales by nothing
            lines.append(f'    hal_scon += {spec} * rad;')
        else:
            lines.append(f'    hal_scon += {spec} * s.specular_level * rad;')
        if float(bake.get('sheen', 0.0)) > 1e-4:
            # the velvet lobe, exactly as light_surface: scattered back at
            # grazing angles, needing a light, vanishing face-on
            sr = min(max(float(bake.get('sheen_roughness', 0.3)), 0.0), 1.0)
            sheen_exp = 1.0 + (1.0 - sr) * 15.0
            lines.append(
                f'    hal_scon += '
                f'{_mv3(consts, bake.get("sheen_color", (1, 1, 1)))}'
                f' * (pow(hal_edge_vn, {_mv(consts, sheen_exp)})'
                f' * max(dot(N, L), 0.0) * {_mv(consts, bake["sheen"])})'
                ' * rad;')
    # R164: shade_one_light's phongcorr -- the sbias branch: diffuse-
    # only, raw N.L threshold, receive-folded exactly like the CPU's
    # shadow_receive gate. R167 adds the RAYBIAS branch's twin: on
    # smooth pixels of a RAY-shadowed lamp the curve runs against the
    # object's Auto Smooth threshold (per-tri hal_sres), with the
    # CPU _pcurve's exact guards (threshold capped below 1, the
    # degenerate denominator falling back to 1); flat pixels keep the
    # sbias curve (or 1), exactly light_surface's done-mask order.
    sb = float((bi or {}).get('sbias', 0.0) or 0.0)
    _smode = getattr(light, 'shadow', 'NONE') \
        if str(consts.get('shadow_default', 'PER_LIGHT')) == 'PER_LIGHT' \
        else str(consts.get('shadow_default'))
    rb = bool((bi or {}).get('raybias')) and shadowed is not None \
        and bool((shadowed or {}).get('ray')) and _smode == 'RAY' \
        and kind not in ('HEMI', 'AREA')
    if rb:
        lines += [
            '    float hal_thr = min(hal_pc_thr, 1.0 - 1e-6);',
            '    float hal_pden = dot(N, L) * (1.0 - hal_thr);',
            '    float hal_pcr = (dot(N, L) > hal_thr)'
            ' ? ((dot(N, L) - hal_thr)'
            ' / ((abs(hal_pden) > 1e-20) ? hal_pden : 1.0)) : 0.0;']
        if sb != 0.0:
            lines += [
                f'    float hal_pcs = (dot(N, L) > {_mv(consts, sb)})'
                f' ? ((dot(N, L) - {_mv(consts, sb)})'
                f' / (dot(N, L) * {_mv(consts, 1.0 - sb)})) : 0.0;',
                '    float hal_pc = mix(hal_pcs, hal_pcr, hal_pc_sm);']
        else:
            lines += [
                '    float hal_pc = mix(1.0, hal_pcr, hal_pc_sm);']
        lines += [
            '    hal_pc = 1.0 - s.shadow_receive'
            ' + s.shadow_receive * hal_pc;',
            '    hal_dcon *= hal_pc;']
    elif sb != 0.0 and shadowed is not None and kind not in ('HEMI',
                                                             'AREA'):
        lines += [
            f'    float hal_pc = (dot(N, L) > {_mv(consts, sb)})'
            f' ? ((dot(N, L) - {_mv(consts, sb)})'
            f' / (dot(N, L) * {_mv(consts, 1.0 - sb)})) : 0.0;',
            '    hal_pc = 1.0 - s.shadow_receive'
            ' + s.shadow_receive * hal_pc;',
            '    hal_dcon *= hal_pc;']
    # R164: the lamp's SHADOW COLOUR tints the shadowed diffuse
    # (lashdw*(i_noshad - i) added back); spec keeps the plain factor
    _shc = tuple(getattr(light, 'shadow_color', (0.0, 0.0, 0.0))
                 or (0.0, 0.0, 0.0))
    if not anime:
        if max(_shc) > 0.0:
            lines.append(f'    hal_dcon *= 0.318309886 * (vec3(hal_sv)'
                         f' + {_v3(_shc)} * (1.0 - hal_sv));')
        else:
            lines.append('    hal_dcon *= 0.318309886 * hal_sv;')
        lines.append('    hal_scon *= 0.318309886 * hal_sv;')
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
        mask = f'hal_lk{i}' if str(link.get('mode', 'EXCLUDE')).upper() \
            == 'ONLY' else f'(1.0 - hal_lk{i})'
        lines.append(f'    hal_dcon *= {mask};')
        lines.append(f'    hal_scon *= {mask};')
    if bi.get('need_spec_acc'):
        # the unclamped specular accumulation, for spectra and the
        # RESULT spec ramp -- exactly the CPU's spec_acc
        lines.append('    hal_spec_acc += hal_scon;')
    if bi.get('result_mode'):
        # RESULT ramps read the whole accumulated colour: parts stay
        # apart, the clamp waits for the end (BI had none per light)
        lines.append('    hal_dacc += hal_dcon;')
        lines.append('    hal_sacc += hal_scon;')
    else:
        lines.append('    vec3 contrib = hal_dcon + hal_scon;')
        clamp = float(consts.get('light_clamp', 0.0))
        if clamp > 0.0:
            lines.append(f'    contrib = min(contrib, vec3({_f(clamp)}));')
        lines.append('    total += contrib;')
    lines.append('    }')
    return lines


def fog_material_dials(bake):
    """R251 LIGHT-A2 (F006): True when this material's bake carries a
    non-default fog dial (Fog Burn-Through, Fog Bias, or Fog Bank 1) --
    the per-material STRUCTURE decision core/fog.material_dials makes
    per point on the CPU. A material with a dial calls the five-argument
    hal_fog; every other material keeps the two-argument call."""
    if not bake:
        return False
    burn = float(bake.get('fog_burn', 0.0) or 0.0)
    bias = float(bake.get('fog_bias', 0.0) or 0.0)
    bank = float(bake.get('fog_bank', 0.0) or 0.0)
    import numpy as np
    return burn != 0.0 or bias != 0.0 or float(np.rint(bank)) >= 1.0


def FOG_BACKDROP_SAMPLER(consts):
    """R251 LIGHT-A2 (F008): whether the pass declares and binds the
    hal_backdrop texture -- exactly when it carries the hal_fog
    definition under a BACKDROP fog target (assemble_frame emits the
    definition on every pass of a fogged plan)."""
    return bool(consts.get('fogtab')) and bool(consts.get('fog')) \
        and str((consts.get('fog') or {}).get('source', 'FIXED')) == 'BACKDROP'


def FOG_CALL(consts, bake):
    """R251 LIGHT-A2 (F006): the hal_fog CALL the tail emits. A material
    with a fog dial hands its three values (hal_mats texels through
    _mv: a slider drag re-uploads a row, never recompiles); every other
    material emits the wave-1 call verbatim."""
    args = ['total', 'P']
    if fog_material_dials(bake):
        args += [_mv(consts, float(bake.get('fog_burn', 0.0) or 0.0)),
                 _mv(consts, float(bake.get('fog_bias', 0.0) or 0.0)),
                 _mv(consts, float(bake.get('fog_bank', 0.0) or 0.0))]
    if (consts.get('fog') or {}).get('spot'):
        # F015's fog half: the screen spotlights' lobe accumulator the
        # lamp loop declares on a lit primary pass (LIGHT-B1's
        # hal_spotfog); a shadeless pass has no loop and hands zeros
        args.append('vec3(0.0)' if bake.get('__shadeless') else 'hal_spotfog')
    if len(args) == 2:
        return '    total = hal_fog(total, P);'
    return '    total = hal_fog(' + ', '.join(args) + ');'


def FOG_GLSL(consts, bake=None, em=None):
    """R251: `hal_fog(rgb, P)` -- the GLSL twin of core/fog.py's `factor`
    + `blend`, one op per statement in the CPU's float32 order.

    STRUCTURE (which bracketed lines are emitted) comes from
    `consts['fog']` (core/fog.structure: mode, vertex, bands, height,
    table, dither, depth, ortho; wave 2: range_adjust, face, source,
    bank1, turb) and is in the plan signature; VALUES ride the
    `hal_fogtab` data texture (core/fog.pack_fog_texels, texelFetch
    only; the sampler is declared by assemble_frame under
    `consts['fogtab']`, never here), the backdrop target rides
    `hal_backdrop` (F008) and a material's fog dials arrive as the
    call's three extra arguments (F006, `FOG_CALL`; the definition is
    then `vec3 hal_fog(vec3 rgb, vec3 P, float fburn, float fbias,
    float fbank)`, otherwise the wave-1 `vec3 hal_fog(vec3 rgb, vec3 P)`).
    `em` is the pass's Emitter: F010's turbulence needs the pattern
    library (PRIM_GLSL) inlined exactly once per pass -- the once-guard
    is `'__pt_prims' in em.once` (gpu/emit._need_prims's key, A36).
    Accepted subset only: no `int/int`, no `%` (`& 3`), `roundEven` for
    np.round, `precise` where a driver could contract a sum of products.
    Bitwise the CPU in the simulator; on the driver `exp`, `atan`,
    `sqrt` and `/` are its own roundings (the EXP fog's and the LINEAR
    fog's existing class) and the blend is 1 ULP unless `precise` is
    honoured."""
    fog = consts.get('fog') or {}
    mode = str(fog.get('mode', 'LINEAR'))
    table = str(fog.get('table', 'NONE'))
    import numpy as np
    w, h = consts['resolution']
    f32 = np.float32
    dials = fog_material_dials(bake)
    bank1 = bool(fog.get('bank1')) and dials
    radj = bool(fog.get('range_adjust')) and not bool(fog.get('ortho'))
    backdrop = str(fog.get('source', 'FIXED')) == 'BACKDROP'
    turb = bool(fog.get('turb'))
    need_px = table == 'VOODOO64' or radj or backdrop
    # F006: the curve reads the point's bank (texel 5 / 6) or texel 0
    B = 'fpb' if bank1 else 'fp0'
    G = 'fgb' if bank1 else 'fpg'
    prefix = ''
    if turb:
        # F010: the pattern library once per pass (A36)
        from .procedural import PRIM_GLSL
        if em is None:
            prefix = PRIM_GLSL + '\n'
        elif '__pt_prims' not in em.once:
            em.once.add('__pt_prims')
            prefix = PRIM_GLSL + '\n'
    spot = bool(fog.get('spot'))
    params = 'vec3 rgb, vec3 P'
    if dials:
        params += ', float fburn, float fbias, float fbank'
    if spot:
        params += ', vec3 fspot'
    sig = 'vec3 hal_fog(' + params + ')'
    # (the wave-1 form, for the MARKER and the reader: vec3 hal_fog(vec3 rgb, vec3 P))
    L = [sig,
         '{',
         '    vec4 fp0 = texelFetch(hal_fogtab, ivec2(0, 0), 0);'
         '   // start, end, span, density',
         '    vec4 fp1 = texelFetch(hal_fogtab, ivec2(1, 0), 0);'
         '   // col.rgb * ambient, bands',
         '    vec4 fp2 = texelFetch(hal_fogtab, ivec2(2, 0), 0);'
         '   // top, falloff, near, k',
         '    vec4 fp3 = texelFetch(hal_fogtab, ivec2(3, 0), 0);'
         '   // job.view[2, :3], 0',
         '    vec4 fp4 = texelFetch(hal_fogtab, ivec2(4, 0), 0);'
         '   // ambient, cx, k, spot',
         '    vec3 dP = P - hal_eye;',
         '    precise float dz = dP.x * fp3.x;',
         '    dz = dz + dP.y * fp3.y;',
         '    dz = dz + dP.z * fp3.z;',
         '    precise float d = abs(dz);']
    if need_px:
        # the screen pixel (the F002 dither, the F004 column, the F008
        # backdrop texel): int(vUV * size), the fragment's own pixel
        L += [f'    int sx = int(vUV.x * {_f(float(w))});',
              f'    int sy = int(vUV.y * {_f(float(h))});']
    if fog.get('face'):
        # [fog_face, F005]: the polygon's mean corner depth replaces the
        # pixel's (the G-buffer's own corners, slot 0 = world position;
        # the fixed corner order 0, 1, 2; the mean as ONE multiply)
        L += ['    HalcyonFragment ff = hal_read_gbuffer(vUV);']
        for c in range(3):
            L += [f'    vec3 c{c} = hal_fetch_attr(ff.tri, {c}, 0).xyz;',
                  f'    vec3 e{c} = c{c} - hal_eye;',
                  f'    precise float z{c} = e{c}.x * fp3.x;',
                  f'    z{c} = z{c} + e{c}.y * fp3.y;',
                  f'    z{c} = z{c} + e{c}.z * fp3.z;',
                  f'    z{c} = abs(z{c});']
        L += ['    float ds = z0 + z1;',
              '    ds = ds + z2;',
              f'    d = ds * {_f(f32(1.0 / 3.0))};']
    if radj:
        # [range_adjust, F004]: GX_InitFogAdjTable's secant of the
        # column, interpolated between the ten knots (texels 8..17;
        # K[-1] = 1 at the centre), scales the planar depth
        L += ['    float dxp = float(sx) + 0.5;',
              '    dxp = dxp - fp4.y;',
              '    dxp = abs(dxp);',
              '    float tt = dxp * fp4.z;',
              '    float jj = floor(tt);',
              '    jj = min(jj, 9.0);',
              '    int ji = int(jj);',
              '    float kb = texelFetch(hal_fogtab, ivec2(8 + ji, 0), 0).x;',
              '    float ka = (ji == 0) ? 1.0 : texelFetch(hal_fogtab, '
              'ivec2(7 + ji, 0), 0).x;',
              '    float fr = tt - jj;',
              '    float kd = kb - ka;',
              '    kd = kd * fr;',
              '    precise float k = ka + kd;',
              '    d = d * k;']
    if fog.get('vertex'):
        # [fog_vertex, structure; skipped by GTE_1Z / tables / fog_face]
        L += ['    float d8 = d * 8.0;',
              '    d8 = roundEven(d8);',
              '    d = d8 / 8.0;']
    L.append('    d = max(d, 0.0);')
    if str(fog.get('depth', 'W')) == 'Z':
        # [fog_depth Z, F007]: D3D's z' = f/(f-n) * (1 - n/z), or the
        # linear ortho z-buffer
        if fog.get('ortho'):
            L += ['    float tz = d - fp2.z;',
                  '    d = tz * fp2.w;']
        else:
            L += [f'    float ddz = max(d, {_f(f32(1e-6))});',
                  '    float qz = fp2.z / ddz;',
                  '    float tz = 1.0 - qz;',
                  '    d = fp2.w * tz;']

    def turb_lines(var):
        # [turbulence, F010]: POV's one read at the segment's middle,
        # faded by exp(-distance * density), scales the distance `var`
        return ['    vec4 fpt = texelFetch(hal_fogtab, ivec2(226, 0), 0);',
                '    vec3 pm = hal_eye + P;',
                '    pm = pm * 0.5;',
                '    pm = pm * fpt.x;',
                '    float tu = hal_pt_turb(pm, 6, 2.0, 0.5);',
                f'    float te = -{var};',
                '    te = te * fp0.w;',
                '    float tk = exp(te);',
                '    float tuq = tu * fpt.y;',
                '    tuq = min(tuq, 1.0);',
                '    float tkk = tk * tuq;',
                '    float tsc = 1.0 - tkk;',
                f'    {var} = {var} * tsc;']
    if turb and mode != 'GROUND':
        L += turb_lines('d')
    L.append('    precise float f = 1.0;')
    if bank1:
        # [bank, F006]: the material's bank index (half-to-even, 0..1)
        # selects texel 5's Start / End / span when bank 1 is live
        L += ['    float bkf = roundEven(fbank);',
              '    bkf = clamp(bkf, 0.0, 1.0);',
              '    int bk = int(bkf);',
              '    vec4 fp5 = texelFetch(hal_fogtab, ivec2(5, 0), 0);',
              '    float ub = (bk == 1) ? fp5.w : 0.0;',
              '    vec4 fpb = (ub > 0.5) ? fp5 : fp0;']
    if mode == 'GROUND':
        # [GROUND, F009]: POV's ComputeGroundFogDepth on the eye -> P
        # segment; every candidate computed, the select picks (A32)
        L += ['    vec4 fpo = texelFetch(hal_fogtab, ivec2(225, 0), 0);',
              '    float gy1 = fpo.z;',
              '    float gy2 = P.z - fpo.x;',
              '    gy2 = gy2 * fpo.y;',
              '    float gsx = dP.x * dP.x;',
              '    gsx = gsx + dP.y * dP.y;',
              '    gsx = gsx + dP.z * dP.z;',
              '    float gdd = sqrt(gsx);']
        if turb:
            L += turb_lines('gdd')
        L += ['    float ga1 = atan(gy1);',
              '    float ga2 = atan(gy2);',
              '    float gmB = (ga2 - gy1) / (gy2 - gy1);',
              '    float gmC = (ga1 - gy2) / (gy1 - gy2);',
              '    float gmD = (ga1 - ga2) / (gy1 - gy2);',
              '    float gq = gy1 * gy1;',
              '    gq = gq + 1.0;',
              '    float gmE = 1.0 / gq;',
              '    float gm = ((gy1 <= 0.0) && (gy2 <= 0.0)) ? 1.0 : '
              '((gy1 <= 0.0) ? gmB : ((gy2 <= 0.0) ? gmC : '
              f'((abs(gy1 - gy2) > {_f(f32(1e-6))}) ? gmD : gmE)));',
              '    float ge = gdd * gm;',
              '    ge = ge * fp0.w;',
              '    ge = -ge;',
              '    f = exp(ge);']
    elif table == 'VOODOO64':
        # [VOODOO64, F002]: Glide's 64-entry w-table, MAME's shift chain
        L += ['    float wv = max(d, 1.0);',
              '    wv = min(wv, 65535.0);',
              '    int e = 0;']
        for k in range(1, 16):
            L.append(f'    e = e + ((wv >= {_f(f32(2.0 ** k))}) ? 1 : 0);')
        L.append('    float p = 1.0;')
        for k in range(1, 16):
            L.append(f'    p = (e >= {k}) ? p * 2.0 : p;')
        L += ['    float r = p / wv;',
              '    float m = 1.0 - r;',
              '    m = m * 8.0;',
              '    float fi = floor(m);',
              '    int i = 4 * e + int(fi);',
              '    float frac = m - fi;',
              '    float fr8 = frac * 256.0;',
              '    fr8 = floor(fr8);',
              '    int frac8 = int(fr8);',
              '    vec4 te = texelFetch(hal_fogtab, ivec2(32 + i, 0), 0);',
              '    int ti = int(te.x);',
              '    int dl = int(te.y);',
              '    int dv = (dl * frac8) >> 6;']
        if fog.get('dither'):
            # [dither]: Voodoo2 fogMode bit 6, the 4x4 matrix at the pixel
            L += ['    int dm = int(texelFetch(hal_fogtab, ivec2(240 + '
                  '((sy & 3) * 4 + (sx & 3)), 0), 0).x);',
                  '    dv = dv + dm;']
        L += ['    dv = dv >> 4;',
              '    int fbv = ti + dv + 1;',
              '    float fop = float(fbv) * 0.00390625;',
              '    f = 1.0 - fop;']
    elif table == 'PVR128':
        # [PVR128, F003]: the CLX2's log table on density / w
        L += ['    vec4 fpg = texelFetch(hal_fogtab, ivec2(224, 0), 0);',
              f'    float dd = max(d, {_f(f32(1e-6))});',
              '    float invw = 1.0 / dd;',
              '    float x = fpg.z * invw;',
              f'    x = clamp(x, 1.0, {_f(f32(255.999985))});',
              '    int e = 0;']
        for k in range(1, 8):
            L.append(f'    e = e + ((x >= {_f(f32(2.0 ** k))}) ? 1 : 0);')
        L.append('    float p = 1.0;')
        for k in range(1, 8):
            L.append(f'    p = (e >= {k}) ? p * 2.0 : p;')
        L += ['    float m = x / p;',
              '    float mm = m - 1.0;',
              '    float m4f = mm * 16.0;',
              '    m4f = floor(m4f);',
              '    int m4 = int(m4f);',
              '    int idx = 16 * e + m4;',
              '    float f8f = mm * 4096.0;',
              '    f8f = floor(f8f);',
              '    int f8 = int(f8f) - m4 * 256;',
              '    vec4 te = texelFetch(hal_fogtab, ivec2(96 + idx, 0), 0);',
              '    int hi = int(te.x);',
              '    int lo = int(te.y);',
              '    int a = (lo * f8 + hi * (255 - f8)) >> 8;',
              '    f = float(255 - a) / 255.0;']
    elif table == 'DS32':
        # [DS32, F022]: melonDS's CalculateFogDensity on eye depth
        L += ['    vec4 fpg = texelFetch(hal_fogtab, ivec2(224, 0), 0);',
              '    float dt = d - fp0.x;',
              '    dt = dt / fpg.w;',
              '    dt = clamp(dt, 0.0, 32.0);',
              '    float dfi = floor(dt);',
              '    float dfr = dt - dfi;',
              '    float d17 = dfr * 131072.0;',
              '    d17 = floor(d17);',
              '    int di = int(dfi);',
              '    int dq = int(d17);',
              '    int t0 = int(texelFetch(hal_fogtab, ivec2(256 + di, 0), '
              '0).x);',
              '    int t1 = int(texelFetch(hal_fogtab, ivec2(257 + di, 0), '
              '0).x);',
              '    int dsum = t0 * (131072 - dq);',
              '    dsum = dsum + t1 * dq;',
              '    int dD = dsum >> 17;',
              '    dD = (dD >= 127) ? 128 : dD;',
              '    float fop = float(dD) * 0.0078125;',
              '    f = 1.0 - fop;']
    elif mode == 'GTE_1Z':
        # [GTE_1Z, F001]: affine in 1/z, floored to 4.12 (IR0); bank 1
        # reads its own (A, B) from texel 6 (F006)
        L += ['    vec4 fpg = texelFetch(hal_fogtab, ivec2(224, 0), 0);']
        if bank1:
            L += ['    vec4 fp6 = texelFetch(hal_fogtab, ivec2(6, 0), 0);',
                  '    vec2 fgb = (ub > 0.5) ? fp6.xy : fpg.xy;']
        L += [f'    float dd = max(d, {_f(f32(1e-6))});',
              f'    float q = {G}.x / dd;',
              f'    float t = q + {G}.y;',
              '    t = clamp(t, 0.0, 1.0);',
              '    float ir = t * 4096.0;',
              '    ir = floor(ir);',
              '    t = ir * 0.000244140625;',
              '    f = 1.0 - t;']
    elif mode == 'EXP':
        L += ['    float e = -fp0.w;',
              '    e = e * d;',
              '    f = exp(e);']
    elif mode == 'EXP2':
        L += ['    float t = fp0.w * d;',
              '    float t2 = t * t;',
              '    float e = -t2;',
              '    f = exp(e);']
    elif mode == 'TABLE16':
        L += [f'    float t = d - {B}.x;',
              f'    t = t / {B}.z;',
              '    t = clamp(t, 0.0, 1.0);',
              '    float ft = t * 16.0;',
              '    ft = floor(ft);',
              '    ft = ft / 16.0;',
              '    f = 1.0 - ft;']
    else:
        # LINEAR (and any mode without a branch)
        L += [f'    f = {B}.y - d;',
              f'    f = f / {B}.z;']
    L.append('    f = clamp(f, 0.0, 1.0);')
    if dials:
        # [material fog, F006]: System 22's cz delta on the opacity, then
        # Model 3's light modifier (the share that burns through)
        L += ['    float op = 1.0 - f;',
              '    op = op + fbias;',
              '    op = clamp(op, 0.0, 1.0);',
              '    float bt = 1.0 - fburn;',
              '    op = op * bt;',
              '    f = 1.0 - op;']
    if fog.get('bands'):
        # [bands >= 2, structure]: round-to-band
        L += ['    float fb = f * fp1.w;',
              '    fb = fb + 0.5;',
              '    fb = floor(fb);',
              '    f = fb / fp1.w;']
    if fog.get('height'):
        # [fog_height, structure]: the fog AMOUNT scales by the height
        # falloff; h == 1 passes f through UNTOUCHED (an inert control
        # must be inert), exactly apply_fog's np.where
        L += ['    float above = P.z - fp2.x;',
              '    above = max(above, 0.0);',
              '    float hx = -above;',
              '    hx = hx * fp2.y;',
              '    float hh = exp(hx);',
              '    float omf = 1.0 - f;',
              '    omf = omf * hh;',
              '    float f2 = 1.0 - omf;',
              '    f = (hh >= 1.0) ? f : f2;']
    if backdrop:
        # [backdrop, F008]: the CPU's own backdrop at this pixel is the
        # target (LightWave's Use Backdrop Color); not scaled by ambient
        L += ['    vec3 col = texelFetch(hal_backdrop, ivec2(sx, sy), 0).rgb;']
    else:
        L += ['    vec3 col = fp1.xyz;']
    if spot:
        # [spot fog, F015]: Spotlight Fog x the screen spotlights' lobe
        # (Supermodel's spotFogColor x fogAttenuation) joins the target
        L += ['    vec3 sp = fp4.w * fspot;',
              '    col = col + sp;']
    L += ['    vec3 o = rgb * f;',
          '    float g = 1.0 - f;',
          '    o = o + col * g;',
          '    return o;',
          '}']
    return prefix + '\n'.join(L) + '\n'


def _console_finish_lines(kind, bake, consts, model_i, light_i=0):
    """R251 (LIGHT-B2): the lines that follow the evaluate call in ONE
    light's block, exactly light_surface's order after SH.evaluate --
    the GX Sun-only gate (F016), the DS table highlight (F018), then the
    POV-Ray finish dials on the scalar diffuse and the coloured
    highlight (F019 brilliance, F020 crand, F021 metallic). Emitted for
    the lamps that run the evaluate call (the Hemi override replaces
    both lobes afterwards on both roads, so it takes none of these).
    `light_i` is the lamp's index in the frame's light list (crand's
    salt, light_surface's own `li`). Every statement is one operation in
    the CPU's order."""
    lines = []
    if kind == 'HEMI':
        return lines
    if model_i == 32 and kind not in ('SUN', 'HEMI'):
        # GX_AF_SPEC lit from directional lights only (GX_InitSpecularDir):
        # a point or spot lamp adds no highlight on the GameCube
        lines.append('    ds.yzw = vec3(0.0);')
    if model_i == 35:
        # GBATEK: HalfVector = (LightVector + LineOfSight) / 2, NOT
        # normalised; ShininessLevel = max(0, -H.N)^2; then the 128-entry
        # 8-bit shininess table (this material's row of hal_dstab, one
        # `_f`-baked row per material), truncation as the hardware's
        # index, the entry as a 0.8 fixed value (1/256). hal_vs is the
        # fixed line of sight (texel 227). core/shading.ds_spec, line
        # for line.
        row = int(bake.get('__mat_id', 0))
        lines += [
            '    vec3 hal_dsh = L + hal_vs;',
            '    hal_dsh = hal_dsh * 0.5;',
            '    float hal_dsv = dot(N, hal_dsh);',
            '    hal_dsv = max(hal_dsv, 0.0);',
            '    hal_dsv = hal_dsv * hal_dsv;',
            '    float hal_dsi = hal_dsv * 128.0;',
            '    int hal_dsx = int(hal_dsi);',
            '    hal_dsx = min(hal_dsx, 127);',
            f'    float hal_dst = texelFetch(hal_dstab, ivec2(hal_dsx, '
            f'{int(row)}), 0).r;',
            '    ds.yzw = s.specular * (hal_dst * 0.00390625);']
    if 24 <= model_i <= 31:
        # the 3ds Max shaders keep their own laws (their diffuse carries
        # its colour): the POV finish is inert there on both roads
        return lines
    if float(bake.get('brilliance', 1.0)) != 1.0:
        # F019: POV's `if (Brilliance != 1.0) intensity = pow(fabs(cos),
        # Brilliance)` on the model's diffuse scalar (SH.apply_brilliance);
        # a per-material texel, the same np.power in the simulator
        lines.append('    ds.x = (s.brilliance == 1.0) ? ds.x : '
                     'pow(max(ds.x, 0.0), s.brilliance);')
    if float(bake.get('crand', 0.0)) > 0.0:
        # F020: POV's `intensity -= rand * Crand` per lamp, from the
        # integer hash of the pixel and a salt (977 + 131 lamp + 7919
        # seed, + 1013 frame under Crand Flickers per Frame), the salt
        # wrapped to 31 bits exactly as light_surface wraps it -- the
        # hash reads only the low 31 bits, so both roads hash ONE number
        # whatever the seed. The pixel identity is the soft-shadow
        # sampler's own (int(vUV * resolution)).
        w, h = consts['resolution']
        z0 = (977 + 131 * int(light_i) + 7919 * int(consts.get('seed', 0))) \
            & 0x7fffffff
        lines += [f'    int hal_csx = int(vUV.x * {_f(float(w))});',
                  f'    int hal_csy = int(vUV.y * {_f(float(h))});',
                  f'    int hal_cz = {int(z0)};']
        if consts.get('crand_per_frame'):
            lines += ['    hal_cz = hal_cz + int(hal_frame) * 1013;',
                      '    hal_cz = hal_cz & 0x7fffffff;']
        lines += ['    float hal_ch = hal_smp_hash3(hal_csx, hal_csy, '
                  'hal_cz);',
                  '    hal_ch = s.crand * hal_ch;',
                  '    ds.x = ds.x - hal_ch;',
                  '    ds.x = max(ds.x, 0.0);']
    if float(bake.get('pov_metallic', 0.0)) > 0.0:
        # F021: POV's ComputeMetallic -- x = acos(N.L)/(pi/2), F =
        # 0.014567225/(x - 1.12)^2 - 0.011612903 clamped, colour *=
        # 1 + M (1 - F)(pigment - 1) on the highlight. Every literal is
        # the float32 of the CPU's own constant through _f (never
        # hand-copied float64 digits); SH.pov_metallic_fresnel line for
        # line.
        import numpy as _np
        lines += [
            '    float hal_hx = clamp(dot(N, L), 0.0, 1.0);',
            '    hal_hx = acos(hal_hx);',
            f'    hal_hx = hal_hx * {_f(_np.float32(2.0 / _np.pi))};',
            f'    float hal_hxm = hal_hx - {_f(_np.float32(1.12))};',
            '    hal_hxm = hal_hxm * hal_hxm;',
            f'    float hal_hF = {_f(_np.float32(0.014567225))} / hal_hxm;',
            f'    hal_hF = hal_hF - {_f(_np.float32(0.011612903))};',
            '    hal_hF = clamp(hal_hF, 0.0, 1.0);',
            '    float hal_hm = 1.0 - hal_hF;',
            '    hal_hm = hal_hm * s.pov_metallic;',
            '    vec3 hal_ht = s.diffuse - vec3(1.0);',
            '    hal_ht = hal_ht * hal_hm;',
            '    hal_ht = hal_ht + vec3(1.0);',
            '    ds.yzw = ds.yzw * hal_ht;']
    return lines


def perpix_names(bake):
    """R229: the per-pixel field names the assembler recorded on the
    bake (so the lamp lines can gate their structure on 'this field
    varies' as well as 'this constant is nonzero')."""
    return frozenset((bake or {}).get('__perpix') or ())


def _ramp_blend_index(name):
    """A BI ramp blend mode name -> its MA_RAMP_* DNA index."""
    from ..core.shading import BI_RAMP_BLEND_ORDER
    try:
        return BI_RAMP_BLEND_ORDER.index(str(name))
    except ValueError:
        return 0


def _bi_graph_props(graph):
    """The BI material node's props from a graph dict, or None."""
    if not graph:
        return None
    for node in graph.get('nodes', {}).values():
        if node.get('bl_idname') == 'HALCYON_BIMaterialNode':
            return node.get('props', {})
    return None


def _bi_ramp_spec(props, prefix):
    """One ramp's spec dict from the BI node's props, or None.

    The same parse n_bi_material runs for the CPU closure, so both
    devices read the identical stops, input, blend, factor and ipo."""
    if not props or not props.get(f'use_{prefix}'):
        return None
    stops = props.get(f'{prefix}_stops')
    if not stops:
        return None
    return {'stops': [tuple(float(x) for x in s) for s in stops],
            'input': props.get(f'{prefix}_input', 'SHADER'),
            'blend': props.get(f'{prefix}_blend', 'MIX'),
            'factor': float(props.get(f'{prefix}_factor', 1.0)),
            'ipotype': int(props.get(f'{prefix}_ipo', 0))}


def _colorband_glsl(name, stops, ipo):
    """A colorband as one GLSL function, `vec4 name(float)`.

    The body is the bitex emitter's unroll verbatim (the proven twin of
    core.bitex.colorband_eval), wrapped as a callable so the light loop
    can read the band per light."""
    n = len(stops)
    if n == 0:
        return f'vec4 {name}(float tin) {{ return vec4(0.0); }}'
    if n == 1:
        s = stops[0]
        return (f'vec4 {name}(float tin) {{ return vec4({_f(s[1])}, '
                f'{_f(s[2])}, {_f(s[3])}, {_f(s[4])}); }}')
    arr = ', '.join(f'vec4({_f(s[1])}, {_f(s[2])}, {_f(s[3])}, {_f(s[4])})'
                    for s in stops)
    pos = ', '.join(_f(s[0]) for s in stops)
    ipo = int(ipo)
    return f"""vec4 {name}(float tin)
{{
    vec4 cbc[{n}] = vec4[{n}]({arr});
    float cbp[{n}] = float[{n}]({pos});
    vec4 cb = vec4(0.0);
    int a = 0;
    while (a < {n} && cbp[a] <= tin) {{ a++; }}
    int ir = (a < {n}) ? a : {n} - 1;
    int il = (a > 0) ? a - 1 : 0;
    float rpos = (a >= {n}) ? 1.0 : cbp[ir];
    float lpos = (a <= 0) ? 0.0 : cbp[il];
    vec4 rcol = cbc[ir];
    vec4 lcol = cbc[il];
    float span = lpos - rpos;
    float fac = (abs(span) > 1e-9) ? (tin - rpos) / span
                                   : ((a >= {n}) ? 1.0 : 0.0);
    int ipo = {ipo};
    if (ipo == 4) {{ cb = lcol; }}
    else if (ipo == 2 || ipo == 3) {{
        int i0 = (a >= {n} - 1) ? ir : min(ir + 1, {n} - 1);
        int i3 = (a < 2) ? il : max(il - 1, 0);
        vec4 c0 = cbc[i0]; vec4 c1 = rcol;
        vec4 c2 = lcol; vec4 c3 = cbc[i3];
        float t = clamp(fac, 0.0, 1.0);
        float t2 = t * t; float t3 = t2 * t;
        vec4 res;
        if (ipo == 3) {{
            res = (0.5 * t3 - 0.5 * t2) * c3
                + (-1.5 * t3 + 2.0 * t2 + 0.5 * t) * c2
                + (1.5 * t3 - 2.5 * t2 + 1.0) * c1
                + (-0.5 * t3 + t2 - 0.5 * t) * c0;
        }} else {{
            res = (0.16666666 * t3) * c3
                + (-0.5 * t3 + 0.5 * t2 + 0.5 * t + 0.16666666) * c2
                + (0.5 * t3 - t2 + 0.66666666) * c1
                + (-0.16666666 * t3 + 0.5 * t2 - 0.5 * t + 0.16666666)
                  * c0;
        }}
        cb = clamp(res, 0.0, 1.0);
    }} else {{
        float f2 = clamp(fac, 0.0, 1.0);
        if (ipo == 1) {{ f2 = 3.0 * f2 * f2 - 2.0 * f2 * f2 * f2; }}
        cb = (1.0 - f2) * rcol + f2 * lcol;
    }}
    return cb;
}}"""


#: surface fields baked from the probe, in HalcyonSurface field order
BAKE_FIELDS = ('diffuse_level', 'specular_level', 'glossiness', 'roughness',
               'metallic', 'anisotropy', 'aniso_rot', 'soften', 'ior',
               'translucency', 'toon_size', 'toon_smooth', 'toon_steps',
               'toon_size2', 'toon_smooth2', 'bi_fresnel', 'bi_fresnel_fac',
               'bi_slope', 'bi_transp_fresnel', 'bi_transp_blend',
               'bi_spectra', 'bi_cubic', 'bi_tangent', 'shadow_receive',
               'cast_only', 'shadows_only', 'opacity',
               # R251 lighting: the period finish dials (F006, F019-F021)
               'fog_burn', 'fog_bias', 'fog_bank', 'brilliance', 'crand',
               'pov_metallic',
               # the anime/cel drivers (R218)
               'anime_th1', 'anime_soft1', 'anime_th2', 'anime_soft2',
               'anime_bias', 'anime_tones', 'anime_spec_size',
               'anime_sharp', 'anime_mask', 'anime_gain',
               # R221: the ramp row pick (the LUT itself rides an atlas)
               'anime_ramp_row',
               # the cartoon/paint drivers (R228)
               'cartoon_amount', 'cartoon_th', 'cartoon_soft',
               'cartoon_smooth', 'cartoon_hl_size', 'cartoon_hl_soft',
               'cartoon_mode', 'cartoon_lamp',
               # the 80s anime additions (R229)
               'anime_shine', 'anime_shine_h', 'anime_shine_w',
               'anime_shine_wave', 'anime_shine_waves', 'anime_shine_soft',
               'anime_shine_second',
               # R241: the hair pass
               'anime_shine_shape', 'anime_shine_angle',
               'anime_shine_follow',
               'anime_air', 'anime_air_width',
               'anime_air_side',
               # R238: the cel's light (the key's vector rides
               # EXTRA_COLORS as cel_dir)
               'cel_light', 'cel_ss', 'cel_ss_len', 'cel_rim_mode',
               'cel_rim_width', 'cel_rim_side', 'cel_shape',
               # R243: the Max Multi-Layer's second highlight
               'specular_level2', 'glossiness2', 'anisotropy2', 'aniso_rot2')

#: master-node sockets that may vary per pixel: when LINKED, the chain is
#: emitted and assigned to the surface field; unlinked, the probed constant
#: bakes as before. socket name -> (surface field, glsl type). This is what
#: lets a texture drive Roughness without pushing the material off the GPU.
PER_PIXEL_SOCKETS = {
    # R213: a linked Opacity chain travels to the GPU -- the layer and
    # stipple passes consume it as their alpha (the plumbing always
    # could; the grant was missing), so genuine fractional transparency
    # stops refusing the driver with 'opacity varies across the frame'.
    # Punch-through materials never need it (their visibility resolves
    # in the z-pass), but Strauss reads 1-opacity per pixel either way.
    'Opacity': ('opacity', 'float'),
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
    # the anime master's own per-pixel rights (R218): each maps one
    # socket onto one surface field, so the generic machinery carries
    # a texture-driven tone or bias without pushing the frame off the
    # GPU. (Game Texture and Detail Texture decode to SEVERAL fields
    # and ride the bespoke branch in per_pixel_fields instead.)
    'Shadow Bias': ('anime_bias', 'float'),
    'Shadow 1 Color': ('anime_shadow1', 'vec3'),
    'Shadow 2 Color': ('anime_shadow2', 'vec3'),
    'Specular Size': ('anime_spec_size', 'float'),
    'Light Response': ('anime_gain', 'float'),
    'Ramp Row': ('anime_ramp_row', 'float'),
    # the cartoon master's per-pixel rights (R228): the sockets the CPU
    # reads RAW or clamps inside the same arithmetic the lamp lines
    # repeat (the two softness sockets are floored at the node and stay
    # baked -- a varying one refuses by name)
    'Shadow Color': ('cartoon_shadow', 'vec3'),
    'Shadow Amount': ('cartoon_amount', 'float'),
    'Shadow Threshold': ('cartoon_th', 'float'),
    'Shadow Smoothing': ('cartoon_smooth', 'float'),
    'Highlight Color': ('cartoon_hl_color', 'vec3'),
    'Highlight Size': ('cartoon_hl_size', 'float'),
    'Lamp Influence': ('cartoon_lamp', 'float'),
    # the 80s anime additions (R229): the amounts and colours are read
    # raw (or re-clamped in the same arithmetic); the band geometry and
    # the widths stay baked
    'Hair Shine': ('anime_shine', 'float'),
    'Hair Shine Color': ('anime_shine_color', 'vec3'),
    'Hair Shine Second Color': ('anime_shine_color2', 'vec3'),
    'Airbrush': ('anime_air', 'float'),
    'Airbrush Color': ('anime_air_color', 'vec3'),
}


def master_node(graph):
    """The master node feeding the output, if that is what feeds it.

    Two nodes qualify: the Halcyon Shader and the BI Material node --
    the second speaks the same surface vocabulary through its socket
    IDENTIFIERS, so every probe and bake below works on both."""
    link = find_surface_link(graph)
    if link is None:
        return None
    node = (graph or {}).get('nodes', {}).get(link[0])
    if node is not None and node.get('bl_idname') in (
            'HALCYON_ShaderNode', 'HALCYON_AnimeShaderNode',
            'HALCYON_CartoonNode', 'HALCYON_BIMaterialNode',
            # R243: Max's Standard and Raytrace materials, the BI idiom
            'HALCYON_MaxStandardNode', 'HALCYON_MaxRaytraceNode'):
        return node
    return None


def _replace_all(text, pairs):
    for old, new in pairs:
        text = text.replace(old, new)
    return text


def _socket(node, name):
    """One named input socket of a node, or None.

    Matches the display name OR the identifier: the BI Material node
    shows Blender Internal's own labels (Hardness, Refr, Alpha) while
    carrying master-shader identifiers underneath, and every seam that
    reads sockets goes through here or the evaluator, which already
    matches both."""
    for sock in (node or {}).get('inputs', ()):
        if sock.get('name') == name or sock.get('identifier') == name:
            return sock
    return None


def master_faceted(graph):
    """R242: True when the master node wears Max's Faceted flag -- the
    assembler then substitutes the stored face normal for this
    material, the CPU's closure_to_surface twin."""
    mn = master_node(graph) if graph else None
    return bool(mn is not None and (mn.get('props') or {}).get('faceted'))


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
    emit_sock = None
    color_linked = False
    for sock in node.get('inputs', ()):
        name = sock.get('name')
        ident = sock.get('identifier')
        if (ident or name) == 'Emit' or name == 'Emit':
            emit_sock = sock
        if (ident == 'Diffuse Color' or name == 'Color'
                or name == 'Diffuse Color') and sock.get('link'):
            color_linked = True
        # the BI Material node shows Blender Internal's own labels
        # (Hardness, Spec, Ref) over master-shader IDENTIFIERS -- the
        # grant must match either, like _socket and em.input do. Matching
        # the display name alone silently revoked per-pixel rights from
        # every renamed socket: the field's mask material put its whole
        # frame on the CPU over a granted, emittable Glossiness chain
        # ('glossiness varies across the frame'), five minutes a frame.
        key = name if name in PER_PIXEL_SOCKETS else \
            (ident if ident in PER_PIXEL_SOCKETS else None)
        if ident and str(ident).startswith('Max '):
            # R243: the Max material nodes' percentage and degree
            # sockets carry 'Max ...' identifiers because the evaluator
            # converts their units (percent / 100, degrees / 360); the
            # raw chain is not the master's field, so it earns no grant
            # -- a linked one makes the probe refuse by name instead of
            # the GPU shading a Specular Level of 60 as 6000 percent
            key = None
        if key is not None and sock.get('link'):
            field, gtype = PER_PIXEL_SOCKETS[key]
            # hand consumers the socket's real display name: em.input
            # and _socket match name-or-identifier, so either works,
            # but the name is what the graph actually shows
            out[field] = (name, gtype)
    # BI's Emit is a FLOAT that glows the DIFFUSE CHAIN's colour:
    # emission = base.rgb * emit (+ the Vertex Color Light add). It has
    # no vec3 master socket, so the generic table can never grant it --
    # and an EMIT texture slot (or a textured colour under a nonzero
    # Emit) pushed the whole frame onto the CPU by the constancy rule
    # ('emission varies across the frame': the field's Red material,
    # every viewport re-render at region resolution). The assembler
    # synthesizes the product from the SAME emitted chains.
    if node.get('bl_idname') == 'HALCYON_AnimeShaderNode':
        # R218: the game maps decode to SEVERAL fields, per compat
        # mode -- each becomes a per-pixel expression the assembler
        # synthesizes from the SAME emitted chains (textures sample
        # once). The probe reads the same table, so constancy checks
        # skip exactly the fields the shader will compute per pixel.
        p = node.get('props', {})
        mode = str(p.get('compat', 'GENERIC'))
        game_linked = det_linked = base_alpha = False
        for sock in node.get('inputs', ()):
            nm = sock.get('name')
            if nm == 'Game Texture' and sock.get('link'):
                game_linked = True
            if nm == 'Detail Texture' and sock.get('link'):
                det_linked = True
            if nm == 'Diffuse Color' and sock.get('link'):
                base_alpha = True
        if game_linked and mode in ('ARCSYS', 'DBFZ', 'KAKAROT'):
            # (R246: SPARKING left this family -- its Game Texture is
            # the Mask1 detail sheet, a colour multiply the emitter
            # folds into the base; no field decodes from it)
            out['anime_mask'] = ('Game Texture', 'anime_arcsys_mask')
            out['anime_bias'] = ('Game Texture', 'anime_arcsys_bias')
            out['anime_spec_size'] = ('Game Texture',
                                      'anime_arcsys_size')
        elif game_linked and mode == 'GENSHIN':
            out['anime_mask'] = ('Game Texture', 'anime_hoyo_mask')
            out['anime_bias'] = ('Game Texture', 'anime_hoyo_bias')
            out['anime_spec_size'] = ('Game Texture', 'anime_hoyo_size')
            # R221: with a ramp linked and no explicit row, the game
            # texture's alpha (the material id) picks the ramp row --
            # exactly n_anime_shader's wiring
            ramp_linked = row_linked = False
            for sock in node.get('inputs', ()):
                if sock.get('name') == 'Shadow Ramp' and \
                        sock.get('link'):
                    ramp_linked = True
                if sock.get('name') == 'Ramp Row' and sock.get('link'):
                    row_linked = True
            if ramp_linked and not row_linked:
                out['anime_ramp_row'] = ('Game Texture',
                                         'anime_hoyo_row')
        elif game_linked and mode == 'ZZZ':
            out['anime_mask'] = ('Game Texture', 'anime_zzz_mask')
            out['anime_bias'] = ('Game Texture', 'anime_zzz_bias')
            out['anime_spec_size'] = ('Game Texture', 'anime_zzz_size')
        if det_linked and mode in ('ARCSYS', 'DBFZ', 'KAKAROT'):
            out['anime_shadow1'] = ('Detail Texture', 'anime_sss')
        if bool(p.get('emission_alpha')) and base_alpha:
            out['emission'] = ('Diffuse Color', 'anime_emit')
        if bool(p.get('use_vertex_ao')):
            # the vertex-red push needs no chain, but the bias field
            # must go per-pixel so the term lands. R239: EVERY mode --
            # the GDC convention is not the ArcSys lineage's alone
            if 'anime_bias' not in out:
                out['anime_bias'] = ('Shadow Bias', 'anime_vao_only')
    if node.get('bl_idname') == 'HALCYON_BIMaterialNode' \
            and 'emission' not in out and emit_sock is not None:
        p = node.get('props', {})
        emit_linked = bool(emit_sock.get('link'))
        emit_default = float(emit_sock.get('default') or 0.0)
        base_varies = color_linked or bool(p.get('vcol_paint'))
        if emit_linked or (abs(emit_default) > 1e-9 and base_varies) \
                or bool(p.get('vcol_light')):
            out['emission'] = ('Emit', 'bi_emit')
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
    em.color_name = str(consts.get('color_name') or '')
    em.has_vcol = bool(consts.get('has_vcol'))
    em.seed = int(consts.get('seed', 0) or 0)          # R251 C099 (MAT-B)
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
    if getattr(em, 'used_shading_lib', False):
        # the chain called into the shading library (a ramp-delegating
        # BI colour blend), which the height pre-pass does not carry --
        # the CPU produces the pre-pass image exactly instead
        return '__CPU__', {'cpu': True, 'node': bump_node.get('id'),
                           'why': 'a shading-library call in the height '
                                  'chain evaluates on the CPU into the '
                                  'height pre-pass',
                           'samplers': [], 'textures': {},
                           'frame_uniforms': [], 'uses_screen': False}

    tex_fns = []
    tex_binds = {}
    src = body + ('\n' if body else '')
    replacements = []
    for meta in em.samplers:
        sname = meta['uniform']
        key = meta.get('image')
        tex = (textures or {}).get(key)
        if meta.get('raw'):
            # an engine data texture (the BI noise tables): the shader
            # declares the sampler itself and reads with texelFetch, so
            # only the binding is needed here
            if tex is None:
                return None, f"engine table texture '{key}' is not " \
                             'among the prepared textures'
            tex_binds[sname] = key
            continue
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
        opts = _tex_opts(consts)                                   # R251
        if filt not in SUPPORTED_TEX_FILTERS:
            return None, (f'the {filt} texture filter is not in the '
                          f'deferred pass yet')
        if filt in FOOTPRINT_TEX_FILTERS or (
                filt in PYRAMID_FILTERS
                and (opts['lod_sharpen'] or opts['mip_select'] != 'FILTER')):
            # the height pre-pass has no footprint field, and the CPU's
            # height chain runs through job.context (R:5289) WITH the
            # footprint: every footprint filter and level road takes the
            # proven CPU height-image pre-pass, as TRILINEAR did
            return '__CPU__', {'cpu': True, 'node': bump_node.get('id'),
                               'why': f'a {filt} footprint in a height chain '
                                      'evaluates on the CPU into the height '
                                      'pre-pass',
                               'samplers': [], 'textures': {},
                               'frame_uniforms': [], 'uses_screen': False}
        tex_fns.append(_texture_sampler(sname, tex, filt, wrap, opts))
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
    obj_fns, obj_line = _object_frame(src, consts)
    if obj_fns is None:
        return None, obj_line
    gen_fns += obj_fns
    gen_line += obj_line

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
uniform vec3 hal_cam_right;
uniform vec3 hal_cam_up;
uniform vec3 hal_cam_back;
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


def _layer_blend_lines(lines, consts, mode, var, f_expr, col_expr):
    """One silhouette layer in GLSL, by its blend code.

    The exact twin of core.render._blend_layer: 0 Add (the pre-1.38
    behaviour, kept in its historic single-line form), 1 Mix, 2
    Multiply, 3 Screen -- factor clamped for the bounded modes, free
    for Add, the layer product clamped for Screen, all as the CPU does.
    """
    if mode == 1:
        lines.append(f'    float {var} = clamp({f_expr}, 0.0, 1.0);')
        lines.append(f'    total = total * (1.0 - {var}) + '
                     f'{col_expr} * {var};')
    elif mode == 2:
        lines.append(f'    float {var} = clamp({f_expr}, 0.0, 1.0);')
        lines.append(f'    total = total * (vec3(1.0) - {var} * '
                     f'(vec3(1.0) - {col_expr}));')
    elif mode == 3:
        lines.append(f'    float {var} = {f_expr};')
        lines.append(f'    total = vec3(1.0) - (vec3(1.0) - total) * '
                     f'(vec3(1.0) - clamp({col_expr} * {var}, 0.0, 1.0));')
    else:
        lines.append(f'    total += {col_expr} * {f_expr};')


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
    from ..core.shading import MODEL_ITEMS as _MI

    # R218: the anime/cel model swaps the lamp assembly for the banded
    # composition; the flag rides bake so _one_light_source sees it
    _anime_idx = next((k for k, m in enumerate(_MI)
                       if m[0] == 'ANIME'), -1)
    bake = dict(bake or {})
    bake['__anime'] = int(model_index) == _anime_idx
    # R228: the cartoon/paint model -- the lamp lines accumulate the
    # strongest lamp's verdict, the assembly tail composes the paint
    _cartoon_idx = next((k for k, m in enumerate(_MI)
                         if m[0] == 'CARTOON'), -1)
    bake['__cartoon'] = int(model_index) == _cartoon_idx
    # R251 (LIGHT-B2): the model index and the material id, for the
    # per-light block's console branches (GX gate, DS table row)
    bake['__model_i'] = int(model_index)
    bake['__mat_id'] = int(mat_id)
    # R243: the Max shaders the light loop must not scale by Specular
    # Level (shading.LEVEL_FREE_MODELS), by index
    from ..core.shading import LEVEL_FREE_MODELS as _LF
    bake['__level_free'] = any(k == int(model_index) for k, m in enumerate(_MI)
                               if m[0] in _LF)
    # R221: this material's packed Shadow Ramp (atlas offsets),
    # when the scene collection found one
    bake['__anime_ramp'] = (consts.get('anime_ramps')
                            or {}).get(mat_id)
    # R239: this material's packed SDF face map (atlas offsets + the
    # face's frame, resolved per frame from the first object wearing
    # the material -- a moved head is a moved mesh, which re-plans
    # anyway)
    bake['__anime_face'] = (consts.get('anime_faces')
                            or {}).get(mat_id)
    # R238: the cel's light -- a fixed key swaps the lamp loop for the
    # shadow-only lamps plus the key's own block; the cel field (screen
    # shadow, depth rim) is read by the opaque frame pass alone (a
    # layer over the surface and a hit off it read nothing, exactly
    # the CPU's depth guard)
    _cel_model = bool(bake['__anime']) or bool(bake['__cartoon'])
    bake['__cel_key'] = _cel_model and \
        int(round(float(bake.get('cel_light', 0.0) or 0.0))) > 0
    bake['__cel_field'] = _cel_model and not secondary and not layer \
        and bool(consts.get('cel_field')) and (
            float(bake.get('cel_ss', 0.0) or 0.0) > 1e-6
            or (float(bake.get('cel_rim_mode', 0.0) or 0.0) > 0.5
                and float(bake.get('rim', 0.0) or 0.0) > 1e-4))

    em = Emitter(graph or {})
    em.frame_mode = True
    em.mark_values = bool(consts.get('__mark_values'))
    em.resolution = consts.get('resolution')
    em.secondary = secondary
    em.camera = consts.get('camera')
    em.uv_names = tuple(consts.get('uv_names') or ())
    em.color_name = str(consts.get('color_name') or '')
    em.has_vcol = bool(consts.get('has_vcol'))
    em.seed = int(consts.get('seed', 0) or 0)          # R251 C099 (MAT-B)
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
            bake['__perpix'] = frozenset(perpix)
            for field, (sockname, gtype) in perpix.items():
                if gtype.startswith('anime_'):
                    p = mnode.get('props', {})
                    mode = str(p.get('compat', 'GENERIC'))
                    kk = {'KAKAROT': ' + 0.06'}.get(mode, '')
                    vao = ''
                    if bool(p.get('use_vertex_ao')) and em.has_vcol:
                        vao = ' + (hal_vcol.r - 1.0) * 2.0'
                    sb = em.input(mnode, 'Shadow Bias', 'float')
                    ss = em.input(mnode, 'Specular Size', 'float')
                    if gtype == 'anime_vao_only':
                        expr = f'({sb}{vao})'
                    elif gtype == 'anime_emit':
                        e_str = em.input(mnode, 'Emission Strength',
                                         'float')
                        base4 = em.input(mnode, 'Diffuse Color', 'vec4')
                        expr = (f'(({base4}).rgb * clamp(({base4}).a '
                                f'- 0.03, 0.0, 1.0) * {e_str} + '
                                + em.input(mnode, 'Self-Illumination',
                                           'vec3')
                                + f' * {e_str})')
                    elif gtype == 'anime_sss':
                        det = em.input(mnode, 'Detail Texture', 'vec4')
                        s1 = em.input(mnode, 'Shadow 1 Color', 'vec3')
                        expr = f'({s1} * ({det}).rgb)'
                    else:
                        game = em.input(mnode, 'Game Texture', 'vec4')
                        fam, part = gtype.split('_')[1], \
                            gtype.split('_')[2]
                        if fam == 'arcsys':
                            table = {
                                'mask': f'({game}).r',
                                'bias': f'({sb} + (({game}).g * 2.0 '
                                        f'- 1.0){kk}{vao})',
                                'size': f'({ss} * clamp(({game}).b '
                                        f'* 2.0, 0.0, 2.0))'}
                        elif fam == 'hoyo':
                            table = {
                                'mask': f'(({game}).r + ((({game}).r '
                                        f'> 0.9) ? 0.6 : 0.0))',
                                'bias': f'({sb} + (({game}).g - 0.5) '
                                        f'* 2.0{vao})',
                                'size': f'({ss} * clamp(1.03 '
                                        f'- ({game}).b, 0.0, 1.0) '
                                        f'* 2.0)',
                                # R221: the material id picks the row
                                'row': f'clamp(({game}).a, 0.0, 1.0)'}
                        else:
                            det4 = em.input(mnode, 'Detail Texture',
                                            'vec4')
                            table = {
                                'mask': f'clamp(({game}).g '
                                        f'+ ({det4}).b, 0.0, 1.5)',
                                'bias': f'({sb} + (({game}).r - 0.5) '
                                        f'* 2.0{vao})',
                                'size': f'({ss} * clamp(({game}).b '
                                        f'* 2.0, 0.0, 2.0))'}
                        expr = table[part]
                    perpix_exprs[field] = expr
                    continue
                if gtype == 'bi_emit':
                    # BI emission = the diffuse chain's colour times the
                    # Emit float (n_bi_material verbatim), sharing the
                    # already-emitted base chain -- textures sample once.
                    # Vertex Color Light adds vcol.rgb * vcol.a when the
                    # mesh actually carries a layer, exactly the CPU's
                    # has_vcol gate.
                    e_expr = em.input(mnode, sockname, 'float')
                    expr = f'({base} * {e_expr})'
                    if (mnode.get('props', {}).get('vcol_light')
                            and em.has_vcol):
                        expr = (f'({expr} + hal_vcol.rgb '
                                f'* hal_vcol.w)')
                    perpix_exprs[field] = expr
                    continue
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
    if bake.get('__sss'):
        # the scatter tree rides a raw data texture, exactly the BI
        # noise tables' mechanism: the SSS library declares the
        # sampler, the plan binds the prepared texture by key
        em.samplers.append({'uniform': 'hal_sss_tree',
                            'image': bake['__sss']['key'],
                            'raw': True})
    tex_fns = []
    tex_binds = {}
    tex_binds_mip = {}
    tex_binds_sat = {}            # R251 C088 (TEX-2): summed-area atlases
    needs_lod = {}                # R251: sampler -> CPU-decided LOD field key (TEX-2)
    needs_uvgrad = []
    src = body + ('\n' if body else '')
    replacements = []
    for meta in em.samplers:
        sname = meta['uniform']
        key = meta.get('image')
        tex = (textures or {}).get(key)
        if meta.get('raw'):
            # an engine data texture (the BI noise tables): the shader
            # declares the sampler itself and reads with texelFetch, so
            # only the binding is needed here
            if tex is None:
                return None, f"engine table texture '{key}' is not " \
                             'among the prepared textures'
            tex_binds[sname] = key
            continue
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
        opts = _tex_opts(consts)                           # R251 texture pack (coded shaders too: B0.4)
        if filt not in SUPPORTED_TEX_FILTERS:
            return None, (f'the {filt} texture filter is not in the '
                          f'deferred pass yet')
        # the footprint rules, exactly the CPU's: a raw flat UV lookup on
        # a SCREEN point filters with the mip footprint; a linked Vector
        # chain, a coded-shader image, or a ray hit (secondary pass -- no
        # pixel footprint) samples the top level, which is what lod=None
        # does on the CPU. The predicate is sample_opts' (0.1 A / E)
        wants_fp = filt in FOOTPRINT_TEX_FILTERS or (
            filt in PYRAMID_FILTERS
            and (opts['lod_sharpen'] or opts['mip_select'] != 'FILTER'))
        fp = wants_fp and bool(meta.get('footprint')) and not meta.get('code') \
            and not secondary
        if fp and layer:
            return None, (f'the {filt} footprint is not in the layer '
                          'passes yet (the opaque frame has it); this '
                          'glass shades on the CPU')
        if fp:
            lod_key = _lod_field_key(opts, tex)     # R251 TEX-2: the level roads emit below
            if not needs_uvgrad:
                tex_fns.append('uniform sampler2D hal_uvgrad;\n')
            if filt == 'SUMMED_AREA' and not tex_binds_sat:
                tex_fns.append('uniform sampler2D hal_recip256;\n')      # declared ONCE per pass, as hal_uvgrad is (A12.1)
            if lod_key is not None and lod_key not in needs_lod.values():
                tex_fns.append(f'uniform sampler2D {_lod_uniform(lod_key)};\n')
            tex_fns.append(_footprint_sampler(
                sname, tex, filt, wrap, opts,
                float(consts.get('tex_mip_bias', 0.0) or 0.0), lod_key))
            (tex_binds_sat if filt == 'SUMMED_AREA' else tex_binds_mip)[sname] = key
            needs_uvgrad.append(sname)
            if lod_key is not None:
                needs_lod[sname] = lod_key
        else:
            tex_fns.append(_texture_sampler(
                sname, tex, NOFOOTPRINT_FILTER.get(filt, filt), wrap, opts))
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
    # R228: the cartoon's shape smoothing reads the same per-object
    # bounds table, so it forces the lookup functions in even when no
    # chain reads Generated coordinates
    _cartoon_gen = (bool(bake.get('__cartoon'))
                    or bool(bake.get('__anime'))) and (
        'cartoon_smooth' in perpix_exprs
        or float(bake.get('cartoon_smooth', 0.0) or 0.0) > 1e-6)
    # R229: the hair shine band reads Generated (its height) and the
    # bounds centre (its azimuth); R241: the cartoon wears it too
    _shine_gen = (bool(bake.get('__anime'))
                  or bool(bake.get('__cartoon'))) and (
        'anime_shine' in perpix_exprs
        or float(bake.get('anime_shine', 0.0) or 0.0) > 1e-6)
    if 'hal_generated' in src or _cartoon_gen or _shine_gen:
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
        if 'hal_generated' in src or _shine_gen:
            gen_line = ('    vec3 hal_generated = (P - hal_gen_lo(td.y)) '
                        '/ hal_gen_span(td.y);\n')
    obj_fns, obj_line = _object_frame(src, consts)
    if obj_fns is None:
        return None, obj_line
    gen_fns += obj_fns
    gen_line += obj_line

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
        rcol = _mv3(consts, bake.get('reflect_color', (1, 1, 1)))
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
        env_lines.append(f'    total += hal_env * '
                         f'({_mv(consts, refl)} * s.specular'
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
            f'    int vi = int(i);\n'
            f'    return texelFetch(hal_vlight, ivec2(vi % {vside}, '
            f'vi / {vside}), 0).rgb;\n'
            '}\n')
        # R251 (MAT-A): hal_vlight_fetch4 + the period combine's function
        # ('' for every pre-1.90 model: the text above is unchanged)
        vlight_fns += GCB.combine_fns(bake.get('__model'), consts, vside)
        vlight_spec = {'rate': str(vertex_rate), 'side': int(vside),
                       'mat': int(mat_id)}
        env_lines = []                     # the corners carry the env term

    two_sided = bool(consts.get('two_sided', True))
    shadows = shadows or [None] * len(lights)
    shadow_fns = []
    samplers = []
    # ---- the BI panel round: ramps, the light group, spectra -- all
    # read from the material's own node props, generated per material
    bi_props = _bi_graph_props(graph)
    bi_meta = {}
    if bi_props and not vertex_rate:
        rd = _bi_ramp_spec(bi_props, 'ramp_dif')
        rs_ = _bi_ramp_spec(bi_props, 'ramp_spec')
        need_spec_acc = (float(bake.get('bi_spectra', 0.0)) > 0.0
                         or (rs_ is not None
                             and rs_.get('input') == 'RESULT'))
        result_mode = ((rd is not None and rd.get('input') == 'RESULT')
                       or (rs_ is not None
                           and rs_.get('input') == 'RESULT'))
        bi_meta = {'ramp_dif': rd, 'ramp_spec': rs_,
                   'need_spec_acc': need_spec_acc,
                   'result_mode': result_mode,
                   # R164: the sbias terminator fix rides per light
                   'sbias': float(bi_props.get('sbias', 0.0) or 0.0),
                   # R167: the RAYBIAS branch's twin (probe-granted)
                   'raybias': bool(bake.get('__raybias'))}
        if rd is not None:
            shadow_fns.append(_colorband_glsl(
                'hal_biband_dif', rd['stops'], rd.get('ipotype', 0)))
        if rs_ is not None:
            shadow_fns.append(_colorband_glsl(
                'hal_biband_spec', rs_['stops'], rs_.get('ipotype', 0)))
    # CONSTANT / WIREFRAME: light_surface returns before the light loop,
    # the ambient term and every surface cheat -- full-bright albedo x
    # diffuse level, plus emission. The pass carries no lighting support
    # at all (the wires themselves are carved by apply_wireframe on the
    # readback -- the CPU's own separable stage, fog-doctrine style).
    # env and rays stay: the CPU applies those AFTER light_surface.
    shadeless = bool(bake.get('__shadeless'))
    if bake.get('__sss') and not vertex_rate and not shadeless:
        # SSS replaces the accumulated diffuse AFTER all lights
        # (shade_lamp_loop's block), so the pass rides the same
        # separated dif/spec bookkeeping the RESULT ramps use, and the
        # gather library joins the source
        if not bi_meta:
            bi_meta = {'ramp_dif': None, 'ramp_spec': None,
                       'need_spec_acc': False, 'result_mode': True}
        else:
            bi_meta['result_mode'] = True
        bi_meta['sss'] = bake['__sss']
        shadow_fns.append(GS.SSS_GLSL)
    # R164: BI's world Exposure corrects the accumulated DIFFUSE and
    # SPEC separately before ambient/emit join -- every material rides
    # the separated bookkeeping while it is active, non-BI included
    _wexp = consts.get('world_exposure')
    if _wexp and not vertex_rate and not shadeless:
        if not bi_meta:
            bi_meta = {'ramp_dif': None, 'ramp_spec': None,
                       'need_spec_acc': False, 'result_mode': True}
        else:
            bi_meta['result_mode'] = True
        bi_meta['exposure'] = _wexp
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
    # R251 (LIGHT-B2 F020, B13): a crand material reads hal_smp_hash3 --
    # the primitives (and the hal_circle sampler they declare) are
    # appended exactly once, here, whatever else the frame needs
    crand_on = float(bake.get('crand', 0.0)) > 0.0
    if soft_any or ao_spec or rad_gather or crand_on:
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
    if bake.get('__cel_field') and not vertex_rate:
        shadow_fns.append(_cel_lookup_function(consts))
        samplers.append('hal_celfield')
    mapped_shadow = False
    for i, smeta in enumerate(shadows):
        if smeta is not None:
            shadow_fns.append(_shadow_function(i, smeta, consts))
            if not smeta.get('ray'):
                mapped_shadow = True
    if mapped_shadow:
        # every mapped light reads the ONE combined atlas: eight
        # per-light samplers put real materials at the driver's
        # 16-sampler fragment cliff (the field's rejected compile)
        samplers.append('hal_shadowpack')
    if not vertex_rate:
        # projected light textures: one lookup function per cookie light.
        # A vertex-rate pass skips them for the same reason it skips the
        # whole loop -- the cookie is already IN the CPU-lit corner values
        for i, ckspec in sorted((consts.get('cookies') or {}).items()):
            shadow_fns.append(_cookie_function(i, ckspec))
            samplers.append(f'hal_cookie{i}')
        # R221: the packed anime Shadow Ramp atlas -- one sampler shared
        # by every ramped material's lamp loop (declared ahead of the
        # light functions, the R193 declaration-order law)
        if consts.get('anime_ramps'):
            shadow_fns.append('uniform sampler2D hal_animeramp;\n')
            samplers.append('hal_animeramp')
        # R239: the packed SDF face maps -- one sampler shared by every
        # face material's lamp loop
        if consts.get('anime_faces'):
            shadow_fns.append('uniform sampler2D hal_facesdf;\n')
            samplers.append('hal_facesdf')
        # BI's area lamp form factor: one Stokes-contour function per
        # AREA light, consumed by _one_light_source's evaluate2 road.
        # Gated exactly like the light loop (a shadeless pass emits no
        # caller, and an uncalled body still has to LINK its hal_ltex
        # reference on strict drivers). The prototype comes first:
        # these functions read the light-value texture, whose
        # definition lands AFTER shadow_fns in the assembly -- GLSL
        # wants declaration before use in file order, and the FIELD's
        # driver enforced what the simulator never did (R193: every
        # material pass rejected, 95% black, then the crash)
        if not shadeless:
            _area_lights = [
                (i, _al) for i, _al in enumerate(lights)
                if getattr(_al, 'type', 'POINT') == 'AREA']
            if _area_lights and bool(consts.get('light_texels', True)):
                shadow_fns.append('vec4 hal_ltex(int t);\n')
            for i, _al in _area_lights:
                shadow_fns.append(_area_function(i, _al, consts))
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
            '    ivec2 hal_lsz = textureSize(hal_gb_idslin, 0);\n'
            '    vec4 hal_idslin = texelFetch(hal_gb_idslin,\n'
            '        ivec2(clamp(vUV * vec2(hal_lsz), vec2(0.0),\n'
            '                    vec2(hal_lsz) - vec2(1.0))), 0);\n'
            '    f.uv = hal_interp(f.tri, hal_idslin.rgb, 2).xy;\n'
            '    f.uv2 = hal_interp4(f.tri, hal_idslin.rgb, 2).zw;\n')
    normal_face = bool(consts.get('normal_face'))
    # R242: Max's Faceted on the master node -- this material alone
    # shades by its stored face normal (closure_to_surface's twin)
    _mn = master_node(graph) if graph else None
    if _mn is not None and bool((_mn.get('props') or {}).get('faceted')):
        normal_face = True
    # R167: the RAYBIAS terminator twin reads the stored face normal
    # (hal_triaux) and the per-tri Auto Smooth threshold (hal_sres) --
    # frame passes only: a ray hit has no G-buffer triangle id. A
    # SECONDARY pass emits a STUB instead (the correction omitted, the
    # pass marked): the planner builds hit passes for every material
    # whenever ray tracing is on, but runs them only when something
    # reflective or refractive exists -- and only THEN does the stub
    # refuse the plan by name. A mirror-free frame (the field frame)
    # rides the GPU whole.
    raybias_on = bool(bake.get('__raybias')) and bool(bi_meta.get('raybias'))
    raybias_stub = False
    if raybias_on and secondary:
        raybias_on = False
        raybias_stub = True
        bi_meta['raybias'] = False
    needs_triaux = normal_face or bool(getattr(em, 'needs_triaux', False)) \
        or raybias_on
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
            f'    int ti = int(i);\n'
            f'    return texelFetch(hal_triaux, ivec2(ti % {aside}, '
            f'ti / {aside}), 0);\n'
            '}\n')
        samplers.append('hal_triaux')
    if raybias_on:
        tcount = int(consts.get('tri_count', 0))
        if tcount <= 0:
            return None, ('the per-tri smoothresh texture needs the '
                          'triangle count the caller did not supply')
        import math as _m2
        sside = int(_m2.ceil(_m2.sqrt(float(max(tcount, 1)))))
        triaux_fns += (
            'uniform sampler2D hal_sres;\n'
            'vec4 hal_sres_fetch(float i)\n'
            '{\n'
            f'    int si = int(i);\n'
            f'    return texelFetch(hal_sres, ivec2(si % {sside}, '
            f'si / {sside}), 0);\n'
            '}\n')
        samplers.append('hal_sres')
    # R169: the per-light VALUE texture -- one row of texels the light
    # loop fetches instead of baked literals, so lamp edits re-upload
    # instead of recompiling (see LIGHT_TEXEL_STRIDE)
    if lights and not (vertex_rate or shadeless) and \
            bool(consts.get('light_texels', True)):
        triaux_fns += (
            'uniform sampler2D hal_lights;\n'
            'vec4 hal_ltex(int t)\n'
            '{\n'
            '    return texelFetch(hal_lights, ivec2(t, 0), 0);\n'
            '}\n')
        samplers.append('hal_lights')
    if int(model_index) == 35 and lights and not (vertex_rate or shadeless):
        # R251 (LIGHT-B2 F018): the DS shininess table rides a data
        # texture declared ONLY in a DS pass (the shared GLSL dispatch
        # carries no sampler; a PHONG pass in the same frame declares
        # it not). The table is per material: a per-pixel Glossiness
        # chain has no row and refuses by name.
        if 'glossiness' in perpix_exprs:
            return None, ("the DS shininess table is one row per "
                          "material; a per-pixel Glossiness chain "
                          "shades on the CPU")
        triaux_fns += 'uniform sampler2D hal_dstab;\n'
        samplers.append('hal_dstab')
    if consts.get('stipple') and not secondary:
        # the Screen Door threshold map (see the composite below)
        samplers.append('hal_stipple')
    if consts.get('fogtab'):
        # R251: hal_fogtab rides every pass while the rule holds (+1
        # sampler against the driver's limit, named in capability.py)
        samplers.append('hal_fogtab')
    # R251 C119 (MAT-B): the REYES snap -- the camera G-buffer's passes
    # only (hit / layer passes refused by name in the planner); a Bump
    # pre-pass samples its heights at the pixel, the CPU at the cell
    # corner, so a Bump material refuses by name
    reyes_fns = ''
    reyes_splice = ''
    if consts.get('reyes') and not secondary and not layer:
        if prepasses:
            from ..core import reyes as _REYES
            return None, _REYES.REFUSE_BUMP
        from ..core import reyes as _REYES
        reyes_fns = _REYES.snap_glsl(int(consts['reyes']['side']))
        reyes_splice = '    f = hal_reyes_snap(f);\n'
        samplers.append('hal_reyes')
    if FOG_BACKDROP_SAMPLER(consts):
        # R251 LIGHT-A2 (F008): the backdrop target (+1 sampler more)
        samplers.append('hal_backdrop')
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
            f'    int wi = int(i);\n'
            f'    return texelFetch(hal_vscreen, ivec2(wi % {wside}, '
            f'wi / {wside}), 0);\n'
            '}\n')
        samplers.append('hal_vscreen')
    if consts.get('crand_per_frame') and \
            float(bake.get('crand', 0.0)) > 0.0:
        # R251 (LIGHT-B2 F020): crand folds the frame number into its
        # salt; the pass declares the per-frame scalar (already fed by
        # the driver and the simulator)
        em.frame_uniforms.add('hal_frame')
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
             # R238: the camera's axes, for a key fixed to the camera
             'uniform vec3 hal_cam_right;\n'
             'uniform vec3 hal_cam_up;\n'
             'uniform vec3 hal_cam_back;\n'
             + ('uniform sampler2D hal_gb_idslin;\n' if affine_uv else '')
             + ('uniform sampler2D hal_stipple;\n'
                if (consts.get('stipple') and not secondary) else '')
             # R251: the fog VALUE texture, on every pass of a plan
             # that needs it (lighting.md section 0's one rule)
             + ('uniform sampler2D hal_fogtab;\n'
                if consts.get('fogtab') else '')
             # R251 LIGHT-A2 (F008): the backdrop fog target, on every
             # pass that carries the hal_fog definition
             + ('uniform sampler2D hal_backdrop;\n'
                if FOG_BACKDROP_SAMPLER(consts) else '')
             + extra_unis)
    parts = [GS.GLSL, GS.DISPATCH, GB.GLSL, decls, gen_fns,
             _block(inline_parts), _block(tex_fns), _block(shadow_fns),
             # R251: hal_fog (FOG_GLSL), reading hal_fogtab and hal_eye
             (FOG_GLSL(consts, bake, em) if (consts.get('fogtab')
                                             and consts.get('fog')) else ''),
             vlight_fns, triaux_fns, wirescreen_fns, reyes_fns, f"""
void main()
{{
    HalcyonFragment f = hal_read_gbuffer(vUV);
{reyes_splice}    vec4 td = hal_tri_data(max(f.tri, 0.0));
    float keep = (f.covered && abs(td.x - {_mv(consts, mat_id)}) < 0.5) ? 1.0 : 0.0;
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
    if 'hal_vcol' in src or any('hal_vcol' in str(v)
                                for v in perpix_exprs.values()):
        # three extra fetches, paid only by materials that read the
        # paint -- including a Vertex Color Light emission term whose
        # only vcol read lives in a per-pixel expression, not in src
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
            # R174: the whole surface struct rides the material texels
            lines.append(f'    s.{name} = '
                         f'{_mv(consts, bake.get(name, 0.0))};')
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
            f'    vec3 hal_t2 = s.tangent * {_mv(consts, ca)} '
            f'+ s.bitangent * {_mv(consts, sa)};',
            f'    s.bitangent = normalize(s.tangent * {_mv(consts, -sa)} '
            f'+ s.bitangent * {_mv(consts, ca)});',
            '    s.tangent = normalize(hal_t2);',
            '    }',
        ]
    if str(bake.get('__slot', '')) == 'specular':
        # the specular slot routing: a lone raw GLOSSY lobe's colour
        # chain is the SPECULAR colour (closure_to_surface puts it in
        # surf.specular), and the diffuse is the CPU's untouched flat
        # constant -- the exact struct the CPU shades from
        lines += [
            f'    s.diffuse = '
            f'{_mv3(consts, bake.get("diffuse", (0.8, 0.8, 0.8)))};',
            f'    s.specular = {base};',
        ]
    else:
        lines += [
            f'    s.diffuse = {base};',
            (f'    s.specular = {perpix_exprs["specular"]};'
             if 'specular' in perpix_exprs else
             f'    s.specular = '
             f'{_mv3(consts, bake.get("specular", (1, 1, 1)))};'),
        ]
    lines += [
        (f'    s.anime_shadow1 = {perpix_exprs["anime_shadow1"]};'
         if 'anime_shadow1' in perpix_exprs else
         f'    s.anime_shadow1 = '
         f'{_mv3(consts, bake.get("anime_shadow1", (0.62, 0.44, 0.48)))};'),
        (f'    s.anime_shadow2 = {perpix_exprs["anime_shadow2"]};'
         if 'anime_shadow2' in perpix_exprs else
         f'    s.anime_shadow2 = '
         f'{_mv3(consts, bake.get("anime_shadow2", (0.38, 0.26, 0.38)))};'),
        # R228: the cartoon paint tones
        (f'    s.cartoon_shadow = {perpix_exprs["cartoon_shadow"]};'
         if 'cartoon_shadow' in perpix_exprs else
         f'    s.cartoon_shadow = '
         f'{_mv3(consts, bake.get("cartoon_shadow", (0.55, 0.45, 0.62)))};'),
        (f'    s.cartoon_hl_color = {perpix_exprs["cartoon_hl_color"]};'
         if 'cartoon_hl_color' in perpix_exprs else
         f'    s.cartoon_hl_color = '
         f'{_mv3(consts, bake.get("cartoon_hl_color", (1.0, 1.0, 1.0)))};'),
        # R229: the 80s anime tones
        (f'    s.anime_shine_color = {perpix_exprs["anime_shine_color"]};'
         if 'anime_shine_color' in perpix_exprs else
         f'    s.anime_shine_color = '
         f'{_mv3(consts, bake.get("anime_shine_color", (1.0, 1.0, 1.0)))};'),
        # R241: the second band's own tint
        (f'    s.anime_shine_color2 = {perpix_exprs["anime_shine_color2"]};'
         if 'anime_shine_color2' in perpix_exprs else
         f'    s.anime_shine_color2 = '
         f'{_mv3(consts, bake.get("anime_shine_color2", (1.0, 1.0, 1.0)))};'),
        (f'    s.anime_air_color = {perpix_exprs["anime_air_color"]};'
         if 'anime_air_color' in perpix_exprs else
         f'    s.anime_air_color = '
         f'{_mv3(consts, bake.get("anime_air_color", (0.82, 0.62, 0.62)))};'),
        # R243: the Max Multi-Layer's second highlight colour and the
        # Max Translucent's colour (constants: the probe refuses a chain)
        f'    s.specular2 = '
        f'{_mv3(consts, bake.get("specular2", (0.9, 0.9, 0.9)))};',
        f'    s.translucent_color = '
        f'{_mv3(consts, bake.get("translucent_color", (0.0, 0.0, 0.0)))};',
        # R238: the key's own-frame vector (per material, never per pixel)
        f'    s.cel_dir = '
        f'{_mv3(consts, bake.get("cel_dir", (0.0, 0.0, 1.0)))};',
    ]
    if (bool(bake.get('__cartoon')) or bool(bake.get('__anime'))) \
            and not vertex_rate and (
            'cartoon_smooth' in perpix_exprs
            or float(bake.get('cartoon_smooth', 0.0) or 0.0) > 1e-6):
        # R228: render.cartoon_smooth_normal -- the shading normal mixed
        # toward the normal of a sphere around the object's bounding-box
        # centre, from the SAME per-object bounds Generated coordinates
        # read (hal_gen_lo/span, keyed by td.y), BEFORE ambient, the
        # lamps, the shadow queries and the silhouette cheats -- exactly
        # where light_surface bends it
        # R238: the shape -- 0 the sphere, 1 an upright cylinder (the
        # radial with no z), 2 the camera (the normal bent toward V)
        shp = int(round(float(bake.get('cel_shape', 0.0) or 0.0)))
        if shp == 1:
            tgt = ['    vec3 hal_crel = P - hal_ccen;',
                   '    vec3 hal_csph = normalize(vec3(hal_crel.x, '
                   'hal_crel.y, 0.0));']
        elif shp == 2:
            tgt = ['    vec3 hal_csph = V;']
        else:
            tgt = ['    vec3 hal_csph = normalize(P - hal_ccen);']
        lines += [
            '    {',
            '    vec3 hal_ccen = hal_gen_lo(td.y) '
            '+ hal_gen_span(td.y) * 0.5;',
        ] + tgt + [
            '    N = normalize(N + (hal_csph - N) '
            '* clamp(s.cartoon_smooth, 0.0, 1.0));',
            '    }',
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
        ] + GCB.recombine_lines(bake.get('__model'), consts, bake)
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
                     f' * {_mv(consts, bake.get("ambient", 1.0))});')
    elif rad_spec:
        # the Radiosity checkbox: gathered ambient replaces the flat
        # term, exactly light_surface's branch -- and plain AO with it
        lines.append(f'    vec3 total = s.diffuse * (hal_rad(P, N)'
                     f' * {_mv(consts, bake.get("ambient", 1.0))});')
    elif int(model_index) >= 100:
        # verbatim shade_lamp_loop (R155/R156): BI's flat ambient rule
        # covers the WORLD pool only; the engine's Global Ambient stays
        # diffuse-tinted, exactly light_surface's BI branch
        eng = consts.get('ambient_engine', consts['ambient_color'])
        wrld = consts.get('ambient_world', (0.0, 0.0, 0.0))
        lines.append(
            f'    vec3 total = s.diffuse * ({_v3(eng)}'
            f' * {_mv(consts, bake.get("ambient", 1.0))})'
            f' + {_v3(wrld)} * {_mv(consts, bake.get("ambient", 1.0))};')
    else:
        lines.append(
            f'    vec3 total = s.diffuse * ({_v3(consts["ambient_color"])}'
            f' * {_mv(consts, bake.get("ambient", 1.0))});')
    if ao_spec and not rad_spec:
        # exactly light_surface's order: ambient occlusion scales the
        # ambient term (with the post-flip N) before any light adds
        lines.append('    total *= hal_ao(P, N);')
    if not vertex_rate and not shadeless \
            and float(bake.get('sheen', 0.0)) > 1e-4:
        lines.append('    float hal_edge_vn = clamp(1.0 - abs(dot(N, V)), '
                     '0.0, 1.0);')
    if bi_meta.get('need_spec_acc'):
        lines.append('    vec3 hal_spec_acc = vec3(0.0);')
    if bi_meta.get('result_mode'):
        lines.append('    vec3 hal_dacc = vec3(0.0);')
        lines.append('    vec3 hal_sacc = vec3(0.0);')
    cartoon_on = bool(bake.get('__cartoon')) and not (vertex_rate
                                                      or shadeless)
    if cartoon_on:
        # R228: the paint accumulators light_surface's CARTOON branch
        # keeps -- the strongest lit verdict, the strongest highlight
        # gate, the Lamp Influence energy
        lines += [
            '    float hal_cart_lit = 0.0;',
            '    float hal_cart_hl = 0.0;',
            '    vec3 hal_cart_rad = vec3(0.0);',
        ]
    cel_key_on = bool(bake.get('__cel_key')) and not (vertex_rate
                                                      or shadeless)
    cel_field_on = bool(bake.get('__cel_field')) and not (vertex_rate
                                                          or shadeless)
    if cel_field_on:
        # R238: the frame's cel field at this pixel (.r screen shadow,
        # .g depth rim), once before the lamps
        lines.append('    vec4 hal_cel = hal_cel_at();')
    if cel_key_on:
        # R238: the strongest caster's visibility, -1 until a caster
        # reaches this object -- light_surface's vis_acc
        lines.append('    float hal_vkey = -1.0;')
    # R167: the RAYBIAS terminator twin's per-pixel inputs, once before
    # the light loop -- light_surface's pc_smooth (the interpolated
    # normal differs from the STORED face normal: the same texel FACE
    # normal mode reads, the same bits the CPU's ctx.Ng carries) and
    # pc_thresh (the object's Auto Smooth threshold, per triangle)
    if raybias_on and not (vertex_rate or shadeless):
        lines += [
            '    float hal_pc_thr = hal_sres_fetch(max(f.tri, 0.0)).x;',
            '    float hal_pc_sm = (abs(dot(hal_triaux_fetch('
            'max(f.tri, 0.0)).xyz, N)) < (1.0 - 1e-6)) ? 1.0 : 0.0;']
    # the material's Light Group (and other materials' EXCLUSIVE
    # groups): a skipped lamp simply emits no block, per material --
    # exactly light_surface's continue
    bi_group = frozenset(bi_props.get('light_group_lights') or ()) \
        if bi_props else frozenset()
    excl_names = frozenset(consts.get('exclusive_lights') or ())
    # R251 F015 (LIGHT-B1): a screen-spot lamp lives on the SCREEN --
    # its block is emitted on primary passes only (hits and layers have
    # no pixel on the CPU either); the fog lobe accumulator LIGHT-A1's
    # hal_fog reads (SPOT32 * hal_spotfog)
    consts_l = dict(consts, __screen_pass=not (secondary or layer))
    if not (secondary or layer or vertex_rate or shadeless) and \
            any(getattr(l, 'screen_spot', False) for l in lights):
        lines.append('    vec3 hal_spotfog = vec3(0.0);')
    for i, light in enumerate(() if (vertex_rate or shadeless)
                               else lights):
        lname = getattr(light, 'name', None)
        if bi_group:
            if lname not in bi_group:
                continue
        elif excl_names and lname in excl_names:
            continue
        lines += _one_light_source(i, light, consts_l,
                                   shadowed=shadows[i], bake=bake,
                                   bi=bi_meta)
    if cel_key_on:
        # R238: the fixed key's own block, after the casters
        lines += _cel_key_source(consts, bake)
    if bi_meta.get('result_mode'):
        # the RESULT ramps: band and blend the whole accumulation, then
        # one clamp -- exactly light_surface's track_result tail
        rd = bi_meta.get('ramp_dif')
        if rd is not None and rd.get('input') == 'RESULT':
            bidx = _ramp_blend_index(rd.get('blend', 'MIX'))
            lines += [
                '    float hal_drfac = dot(hal_dacc, '
                'vec3(0.3, 0.58, 0.11));',
                '    vec4 hal_bdr = hal_biband_dif(hal_drfac);',
                f'    hal_dacc = hal_ramp_blend({bidx}, hal_dacc, '
                f'hal_bdr.a * {_mv(consts, float(rd.get("factor", 1.0)))}, '
                'hal_bdr.rgb);']
        rs_ = bi_meta.get('ramp_spec')
        if rs_ is not None and rs_.get('input') == 'RESULT':
            bidx = _ramp_blend_index(rs_.get('blend', 'MIX'))
            lines += [
                '    float hal_srfac = dot(hal_sacc, '
                'vec3(0.3, 0.58, 0.11));',
                '    vec4 hal_bsr = hal_biband_spec(hal_srfac);',
                f'    hal_sacc = hal_ramp_blend({bidx}, hal_sacc, '
                f'hal_bsr.a * {_mv(consts, float(rs_.get("factor", 1.0)))}, '
                'hal_bsr.rgb);']
        if bi_meta.get('need_spec_acc'):
            # spectra reads the POST-ramp accumulation, as the CPU's
            # extras['spec_acc'] does
            lines.append('    hal_spec_acc = hal_sacc;')
        if bi_meta.get('sss'):
            # shade_lamp_loop's SSS block: the gathered scatter
            # REPLACES the accumulated diffuse, shaped by the material
            # colour per sss_texfac (invalpha = 1/col[3], the pow
            # midpoint) -- exactly light_surface's CPU tail
            texfac = float(bi_meta['sss'].get('texfac', 0.0))
            lines.append('    vec3 hal_sssrad = hal_sss_sample(P);')
            lines.append('    float hal_sssia = '
                         '(s.opacity > 1.1920929e-07) '
                         '? 1.0 / s.opacity : 1.0;')
            if texfac == 0.0:
                col_expr = 's.diffuse * hal_sssia'
            elif texfac == 1.0:
                col_expr = 'vec3(hal_sssia)'
            else:
                col_expr = (f'pow(max(s.diffuse * hal_sssia, '
                            f'vec3(0.0)), '
                            f'vec3({_mv(consts, 1.0 - texfac)}))')
            lines.append(f'    hal_dacc = hal_sssrad * ({col_expr});')
        if bi_meta.get('exposure'):
            # wrld_exposure_correct, the 2003 letters: diffuse total
            # and spec separately, AFTER the SSS replacement, BEFORE
            # ambient and emit join -- baked linfac/logfac constants
            import numpy as _np
            _we, _wr = bi_meta['exposure']
            _lin = float(1.0 + (2.0 * float(_we) + 0.5) ** -10.0)
            _log = float(_np.log((_lin - 1.0) / _lin)
                         / (float(_wr) if abs(float(_wr)) > 1e-6
                            else 1e-6))
            lines += [
                f'    hal_dacc = {_f(_lin)} * (vec3(1.0)'
                f' - exp(hal_dacc * {_f(_log)}));',
                f'    hal_sacc = {_f(_lin)} * (vec3(1.0)'
                f' - exp(hal_sacc * {_f(_log)}));']
            if bi_meta.get('need_spec_acc'):
                # spectra reads the POST-exposure spec, as the CPU's
                # extras['spec_acc'] does (assigned after the tail)
                lines.append('    hal_spec_acc = hal_sacc;')
        lines.append('    vec3 hal_lpart = hal_dacc + hal_sacc;')
        clamp_l = float(consts.get('light_clamp', 0.0))
        if clamp_l > 0.0:
            lines.append(f'    hal_lpart = min(hal_lpart, '
                         f'vec3({_f(clamp_l)}));')
        lines.append('    total += hal_lpart;')
    if cartoon_on:
        # R228: render._cartoon_compose, the same operations in the
        # same order -- the shadow tone by mode (0 transparent cel, 1
        # painted, 2 none), the lit paint under Lamp Influence, the lit
        # step between them, the highlight painted OVER, then the
        # diffuse level. The paint REPLACES the accumulation: no
        # ambient, no lamp energy unless asked.
        lines += [
            '    float hal_ct0 = clamp((hal_cart_lit - (s.cartoon_th '
            '- s.cartoon_soft)) / max(2.0 * s.cartoon_soft, 1e-6), '
            '0.0, 1.0);',
            '    float hal_ct = hal_ct0 * hal_ct0 * (3.0 - 2.0 * hal_ct0);',
            '    float hal_camt = clamp(s.cartoon_amount, 0.0, 1.0);',
            '    vec3 hal_ctrans = s.diffuse * (vec3(1.0) '
            '+ (s.cartoon_shadow - vec3(1.0)) * hal_camt);',
            '    vec3 hal_cpaint = s.diffuse + (s.cartoon_shadow '
            '- s.diffuse) * hal_camt;',
            '    vec3 hal_cshade = (s.cartoon_mode > 1.5) ? s.diffuse : '
            '((s.cartoon_mode > 0.5) ? hal_cpaint : hal_ctrans);',
            '    float hal_cli = clamp(s.cartoon_lamp, 0.0, 1.0);',
            '    vec3 hal_clit = s.diffuse * (1.0 - hal_cli) '
            '+ s.diffuse * hal_cart_rad * hal_cli;',
            '    vec3 hal_cbase = hal_cshade + (hal_clit - hal_cshade) '
            '* hal_ct;',
        ]
        if 'anime_air' in perpix_exprs or \
                float(bake.get('anime_air', 0.0) or 0.0) > 1e-6:
            # R238: the airbrush against the paint's shadow edge --
            # render._cartoon_compose's call, on the cartoon threshold
            lines += _airbrush_lines('hal_cbase', 'hal_cart_lit',
                                     's.cartoon_th')
        lines += [
            '    float hal_chk = clamp(hal_cart_hl, 0.0, 1.0);',
            '    hal_cbase = hal_cbase + (s.cartoon_hl_color - hal_cbase) '
            '* hal_chk;',
            '    total = hal_cbase * s.diffuse_level;',
        ]
    if (bool(bake.get('__anime')) or bool(bake.get('__cartoon'))) \
            and not (vertex_rate or shadeless) and (
            'anime_shine' in perpix_exprs
            or float(bake.get('anime_shine', 0.0) or 0.0) > 1e-6):
        # R229: render._anime_hair_shine -- the band at a fraction of
        # the object's height (hal_generated.z), its edge waving with
        # the azimuth about the bounds centre, on the camera-facing
        # surface, fading only on the underside; a thinner second band
        # below; painted OVER the banded result. R241 (the hair pass):
        # the CARTOON master wears it too; the wave's shape and phase,
        # the follow toward the key's height and the second band's own
        # tint -- the same operations render.py runs, the shape and
        # the key picked STATICALLY per material (props and lamp
        # types, never per pixel)
        shape_i = int(round(float(bake.get('anime_shine_shape', 0.0)
                                  or 0.0)))
        if shape_i == 1:
            wav_l = ['    float hal_hq = hal_harg * 0.159154937 - 0.25;',
                     '    float hal_hwv = 4.0 * abs(hal_hq '
                     '- floor(hal_hq) - 0.5) - 1.0;']
        elif shape_i == 2:
            wav_l = ['    float hal_hwv = 1.0 - 2.0 '
                     '* abs(sin(hal_harg * 0.5));']
        elif shape_i == 3:
            wav_l = ['    float hal_hwv = (sin(hal_harg) >= 0.0) '
                     '? 1.0 : -1.0;']
        else:
            wav_l = ['    float hal_hwv = sin(hal_harg);']
        kz_expr = None
        if float(bake.get('anime_shine_follow', 0.0) or 0.0) > 1e-6:
            _clm = int(round(float(bake.get('cel_light', 0.0) or 0.0)))
            if _clm == 1:
                kz_expr = ('(hal_cam_right.z * s.cel_dir.x '
                           '+ hal_cam_up.z * s.cel_dir.y '
                           '+ hal_cam_back.z * s.cel_dir.z)')
            elif _clm == 2:
                kz_expr = 'normalize(s.cel_dir).z'
            else:
                _ki = int(consts.get('cel_key_index', -1))
                if 0 <= _ki < len(lights) and str(getattr(
                        lights[_ki], 'type', '')).upper() in ('SUN',
                                                              'HEMI'):
                    kz_expr = _lref(_ki, 1, 'z')
        hh_line = '    float hal_hh = s.anime_shine_h' + (
            f' + s.anime_shine_follow * (0.35 * {kz_expr});'
            if kz_expr else ';')
        lines += [
            '    {',
            '    vec3 hal_hcen = hal_gen_lo(td.y) '
            '+ hal_gen_span(td.y) * 0.5;',
            '    float hal_haz = atan(P.y - hal_hcen.y, P.x - hal_hcen.x);',
            '    float hal_ht = hal_generated.z;',
            '    float hal_ham = clamp(s.anime_shine, 0.0, 1.0);',
            '    float hal_hw = max(s.anime_shine_w, 1e-4) * 0.5;',
            '    float hal_hs = max(s.anime_shine_soft, 1e-4);',
            '    float hal_harg = s.anime_shine_waves * hal_haz '
            '+ s.anime_shine_angle * 0.0174532924;',
        ] + wav_l + [
            hh_line,
            '    float hal_h0 = hal_hh + s.anime_shine_wave * hal_hwv;',
            '    float hal_hb1 = hal_sstep(hal_h0 - hal_hw - hal_hs, '
            'hal_h0 - hal_hw + hal_hs, hal_ht) '
            '* (1.0 - hal_sstep(hal_h0 + hal_hw - hal_hs, '
            'hal_h0 + hal_hw + hal_hs, hal_ht));',
            '    float hal_h2 = hal_h0 - s.anime_shine_second;',
            '    float hal_hw2 = hal_hw * 0.5;',
            '    float hal_hb2 = hal_sstep(hal_h2 - hal_hw2 - hal_hs, '
            'hal_h2 - hal_hw2 + hal_hs, hal_ht) '
            '* (1.0 - hal_sstep(hal_h2 + hal_hw2 - hal_hs, '
            'hal_h2 + hal_hw2 + hal_hs, hal_ht)) * 0.85;',
            '    float hal_hb2m = (s.anime_shine_second > 1e-6) '
            '? hal_hb2 : 0.0;',
            '    float hal_hb = max(hal_hb1, hal_hb2m);',
            '    float hal_hf = hal_sstep(0.0, 0.35, dot(N, V));',
            '    float hal_hu = hal_sstep(-0.3, 0.1, N.z);',
            '    float hal_hk = hal_ham * hal_hb * hal_hf * hal_hu;',
            '    vec3 hal_hcol = (hal_hb2m > hal_hb1) '
            '? s.anime_shine_color * s.anime_shine_color2 '
            ': s.anime_shine_color;',
            '    total = total * (1.0 - hal_hk) '
            '+ hal_hcol * hal_hk;',
            '    }',
        ]
    if not vertex_rate and not shadeless and int(model_index) in (32, 35):
        # R251 (LIGHT-B2): the vertex unit's saturated integer output --
        # the GX's 8-bit lit colour (F016) or the DS's 5-bit one (F018)
        # -- before the clamp and the emission, exactly light_surface's
        # SH.quantize_lit: a DIVISION by the level count so full white
        # is exactly 1.0, the half-up tie, one operation per statement
        # (the driver may contract `* lv + 0.5`: one level at a tie)
        _lv = '255.0' if int(model_index) == 32 else '31.0'
        lines += ['    total = clamp(total, 0.0, 1.0);',
                  f'    total = total * {_lv};',
                  '    total = total + 0.5;',
                  '    total = floor(total);',
                  f'    total = total / {_lv};']
    if not vertex_rate and not shadeless \
            and consts.get('clamp_specular', True):
        lines.append('    total = min(total, vec3(64.0));')
    if not vertex_rate:
        if 'emission' in perpix_exprs:
            lines.append(
                f'    total = total + {perpix_exprs["emission"]};')
        else:
            lines.append(f'    total = total + '
                         f'{_mv3(consts, bake.get("emission", (0, 0, 0)))};')
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
        _layer_blend_lines(
            lines, consts, int(round(float(bake.get('fresnel_blend', 0.0)))),
            'hal_frf',
            f'(pow(hal_sil, {_mv(consts, fp)})'
            f' * {_mv(consts, bake["fresnel"])}'
            f' * {_mv(consts, bake.get("specular_level", 0.5))})',
            _mv3(consts, bake.get('fresnel_color', (1, 1, 1))))
    if not vertex_rate and not shadeless \
            and float(bake.get('rim', 0.0)) > 1e-4:
        rp = max(float(bake.get('rim_power', 3.0)), 0.01)
        if float(bake.get('cel_rim_mode', 0.0) or 0.0) > 0.5 and (
                bool(bake.get('__anime')) or bool(bake.get('__cartoon'))):
            # R238: the depth rim -- the cel field's mask times Rim
            # Amount, in place of the Fresnel power (zero where the
            # pass reads no field: a layer, a hit -- the CPU's own)
            rim_expr = (f'(hal_cel.g * {_mv(consts, bake["rim"])})'
                        if cel_field_on else '0.0')
        else:
            rim_expr = (f'(pow(hal_sil, {_mv(consts, rp)})'
                        f' * {_mv(consts, bake["rim"])})')
        _layer_blend_lines(
            lines, consts, int(round(float(bake.get('rim_blend', 0.0)))),
            'hal_rmf', rim_expr,
            _mv3(consts, bake.get('rim_color', (1, 1, 1))))
    # matcap: the whole lit result lands by its blend menu, exactly as
    # apply_surface_effects -- after fresnel and rim, before the backface.
    # Mode 0 (Mix) is the pre-1.38 behaviour verbatim
    mk = min(max(float(bake.get('matcap_blend', 0.0)), 0.0), 1.0)
    if vertex_rate or shadeless:
        mk = 0.0            # in the corners already / never applied
    if mk > 1e-4:
        mc = perpix_exprs.get('matcap',
                              _mv3(consts, bake.get('matcap', (0, 0, 0))))
        mmode = int(round(float(bake.get('matcap_mode', 0.0))))
        if mmode == 1:
            lines.append(f'    total = total + ({mc}) * {_mv(consts, mk)};')
        elif mmode == 2:
            lines.append(f'    total = total * (vec3(1.0) - '
                         f'{_mv(consts, mk)} * (vec3(1.0) - ({mc})));')
        elif mmode == 3:
            lines.append(f'    total = vec3(1.0) - (vec3(1.0) - total) * '
                         f'(vec3(1.0) - clamp(({mc}) * {_mv(consts, mk)}, '
                         f'0.0, 1.0));')
        else:
            lines.append(f'    total = total * {_mv(consts, 1.0 - mk)} + '
                         f'({mc}) * {_mv(consts, mk)};')
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
            f'    float hal_bk = {_mv(consts, kb)} * hal_backfacing;',
            '    total = total * (1.0 - hal_bk) + '
            f'{_mv3(consts, bake.get("backface_color", (0, 0, 0)))}'
            ' * hal_bk;',
        ]
    # R164: MA_OBCOLOR modulates the WHOLE combined at the very end of
    # shade_lamp_loop -- an unrolled ladder over the objects whose
    # colour is not white, selected by the G-buffer's own object id
    # (td.y), exactly the light-linking ladder's mechanism
    _obc = bake.get('__obcolor')
    if _obc and not vertex_rate and not shadeless:
        sel = ' + '.join(
            f'((abs(td.y - {_f(float(oi))}) < 0.5)'
            f' ? {_v3(c)} : vec3(0.0))'
            for oi, c in _obc)
        anyhit = ' + '.join(
            f'((abs(td.y - {_f(float(oi))}) < 0.5) ? 1.0 : 0.0)'
            for oi, c in _obc)
        lines += [
            f'    float hal_obhit = min({anyhit}, 1.0);',
            f'    vec3 hal_obc = ({sel})'
            ' + vec3(1.0 - hal_obhit);',
            '    total = total * hal_obc;']
    lines += env_lines
    if consts.get('fog') and not vertex_rate and not secondary \
            and not layer and float(bake.get('use_mist', 1.0)) >= 0.5 \
            and not consts.get('fog_cpu'):
        # R251: fog INSIDE the pass (FOG_GLSL's hal_fog) -- the CPU's
        # own order: light_surface -> apply_fog -> alpha. Vertex-rate
        # passes carry fog in their CPU-lit corners; hits and layers
        # fog on the CPU where their composites are; fog_cpu materials
        # keep the readback road (named by the planner)
        lines.append(FOG_CALL(consts, bake))
    if layer:
        # a TRANSPARENT LAYER writes its real alpha -- the same chain
        # shade_batch runs for SORTED/ABUFFER fragments: opacity clamped,
        # the hard threshold, then the edge-opacity silhouette blend.
        # rgb and alpha premultiply by KEEP only, so per-pixel-disjoint
        # materials merge into one target as a plain sum (the driver
        # composites them under ALPHA_PREMULT, which for disjoint
        # writes IS that sum)
        aexpr = perpix_exprs.get('opacity',
                                 _mv(consts,
                                     float(bake.get('opacity', 1.0))))
        lines.append(f'    float hal_alpha = clamp({aexpr}, 0.0, 1.0);')
        if abs(float(bake.get('bi_transp_fresnel', 0.0))) > 0.0:
            # Transparency > Fresnel REPLACES the alpha slider, exactly
            # shade_batch's chain (and 2.79's)
            lines.append(
                '    hal_alpha = clamp(hal_bi_fresnel_fac('
                '-dot(Nsurf, V), s.bi_transp_blend, '
                's.bi_transp_fresnel), 0.0, 1.0);')
        if float(bake.get('bi_spectra', 0.0)) > 0.0 and \
                bi_meta.get('need_spec_acc'):
            # Transparency > Specular: highlights turn opaque
            lines += [
                '    float hal_spt = clamp(max(hal_spec_acc.r, '
                'max(hal_spec_acc.g, hal_spec_acc.b)) * s.bi_spectra, '
                '0.0, 1.0);',
                '    hal_alpha = (1.0 - hal_spt) * hal_alpha + hal_spt;']
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
                f'    float hal_eo_t = pow(1.0 - hal_eo_f, '
                f'{_mv(consts, fp)});',
                '    hal_alpha = clamp(hal_alpha * (1.0 - hal_eo_t) + '
                f'{_mv(consts, eo)} * hal_eo_t, 0.0, 1.0);',
            ]
        ac = float(bake.get('alpha_clip', -1.0))
        if ac >= 0.0:
            # R211 punch-through law, exactly the CPU's: a CLIP material
            # is fully there or fully absent, after the whole chain.
            # R251 C031: a Clip+Blend layer keeps the sub-threshold alpha
            # (its blend half); the probe bakes the mode as structure
            cb_else = 'hal_alpha' if bake.get('__clip_blend') else '0.0'
            lines.append(f'    hal_alpha = (hal_alpha >= '
                         f'{_f(max(ac, 1e-6))}) ? 1.0 : {cb_else};')
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
                                 _mv(consts,
                                     float(bake.get('opacity', 1.0))))
        lines.append(f'    float hal_alpha = clamp({aexpr}, 0.0, 1.0);')
        thr = float(consts.get('alpha_threshold', 0.0))
        if thr > 0.0:
            lines.append(f'    hal_alpha = (hal_alpha >= {_f(thr)}) '
                         '? hal_alpha : 0.0;')
        ac = float(bake.get('alpha_clip', -1.0))
        if ac >= 0.0:
            lines.append(f'    hal_alpha = (hal_alpha >= '
                         f'{_f(max(ac, 1e-6))}) ? 1.0 : 0.0;')
        rw, rh = consts.get('resolution', (1.0, 1.0))
        if str((consts.get('stipple') or {}).get('pattern')) == 'N64_NOISE':
            # R251 C015 (N64 RDP dither_alpha_en): hal_stipple is the
            # CPU's FULL-FRAME 8-bit random map, four pixel columns per
            # RGBA32F texel (pixel x in texel x >> 2, channel x & 3 --
            # selected by three compares, never an integer divide: the
            # simulator's int/int is a float). round(alpha * 255) is
            # roundEven <-> np.round; 255.0 and the texel values 0..255
            # are exact, so `>` is the CPU's a8 > r8 bit for bit. One op
            # per statement where the rounding matters.
            lines += [
                f'    float hal_spx = floor(vUV.x * {_f(float(rw))});',
                f'    float hal_spy = floor(vUV.y * {_f(float(rh))});',
                '    float hal_sq = floor(hal_spx * 0.25);',
                '    float hal_sc = hal_spx - hal_sq * 4.0;',
                '    vec4 hal_st4 = texelFetch(hal_stipple, '
                'ivec2(int(hal_sq), int(hal_spy)), 0);',
                '    float hal_sthr = hal_st4.r;',
                '    hal_sthr = (hal_sc > 0.5) ? hal_st4.g : hal_sthr;',
                '    hal_sthr = (hal_sc > 1.5) ? hal_st4.b : hal_sthr;',
                '    hal_sthr = (hal_sc > 2.5) ? hal_st4.a : hal_sthr;',
                '    float hal_a8 = hal_alpha * 255.0;',
                '    hal_a8 = roundEven(hal_a8);',
                '    hal_alpha = (hal_a8 > hal_sthr) ? 1.0 : 0.0;',
                '    Color = vec4(total, 0.6 + 0.3 * hal_alpha) * keep;',
            ]
        else:
            lines += [
                f'    float hal_spx = mod(floor(vUV.x * {_f(float(rw))}), '
                '64.0);',
                f'    float hal_spy = mod(floor(vUV.y * {_f(float(rh))}), '
                '64.0);',
                '    float hal_sthr = texelFetch(hal_stipple, '
                'ivec2(int(hal_spx), int(hal_spy)), 0).r;',
                '    hal_alpha = (hal_alpha > hal_sthr) ? 1.0 : 0.0;',
                '    Color = vec4(total, 0.6 + 0.3 * hal_alpha) * keep;',
            ]
    else:
        lines.append('    Color = vec4(total, 1.0) * keep;')
    lines.append('}')
    parts.append('\n'.join(lines))
    all_samplers = (samplers
                    + (['hal_vlight'] if vlight_spec is not None else [])
                    + sorted(tex_binds) + sorted(tex_binds_mip)
                    + sorted(tex_binds_sat)                                   # R251: SAT atlases (TEX-2)
                    + (['hal_recip256'] if tex_binds_sat else [])             # R251: the reciprocal atlas (bound as hal_circle is)
                    + (['hal_uvgrad'] if needs_uvgrad else [])
                    + sorted({_lod_uniform(k) for k in needs_lod.values()})   # R251: the CPU-decided LOD fields (TEX-2)
                    + [p[0] for p in prepasses])
    _slim = int(consts.get('max_samplers', 16))
    if len(all_samplers) + 3 > _slim:
        # the three G-buffer samplers ride every pass. The limit is the
        # DRIVER'S OWN answer (device.max_fragment_samplers -- 32 on
        # desktop hardware), not the spec's worst case: a field
        # material needing 17 spent 16 minutes on the CPU under the
        # assumed 16, on a card that provides twice that. Refusing
        # HERE, by name and by count, still beats the driver rejecting
        # the compile -- but only past the real cliff.
        return None, (f'needs {len(all_samplers) + 3} texture samplers; '
                      f'this driver provides {_slim}')
    info = {'samplers': all_samplers,
            # R251 (LIGHT-B2 F018): (material index, Glossiness) -- the
            # hal_dstab row this pass reads, packed by the plan
            'dstab': ((int(mat_id), float(bake.get('glossiness', 25.0)))
                      if 'hal_dstab' in samplers else None),
            'textures': tex_binds,
            'textures_mip': tex_binds_mip,
            'textures_sat': tex_binds_sat,                                     # R251 (TEX-2 fills it)
            'textures_lod': {_lod_uniform(k): k for k in needs_lod.values()},  # R251 (TEX-2 fills it)
            'needs_uvgrad': bool(needs_uvgrad),
            'needs_wirescreen': bool(getattr(em, 'needs_wirescreen',
                                             False)),
            'frame_uniforms': frame_unis,
            'prepasses': prepasses,
            'uses_screen': em.used_screen
            or any(p[2].get('uses_screen') for p in prepasses)}
    if consts.get('fog_cpu'):
        # R251: the planner's reason this material fogs on the CPU
        info['fog_cpu'] = str(consts['fog_cpu'])
        # R251 LIGHT-A2 (F006): its fog dials for the refusal road
        info['fog_mat'] = (float(bake.get('fog_burn', 0.0) or 0.0),
                           float(bake.get('fog_bias', 0.0) or 0.0),
                           float(bake.get('fog_bank', 0.0) or 0.0))
    if vlight_spec is not None:
        info['vlight'] = vlight_spec
    if raybias_stub:
        # this hit pass is INCOMPLETE (Ray Bias omitted): valid to
        # build, fatal to run -- the planner refuses by name if any
        # reflective or refractive material would execute it
        info['raybias_secondary_stub'] = True
    src = ''.join(parts)
    if consts.get('__mark_values'):
        # R174: pull every marked material VALUE out of the source and
        # into the hal_mats texture row this pass will be handed
        # (hal_mrow). Two materials with the same STRUCTURE now produce
        # byte-identical sources -- one driver compile serves them all,
        # and a slider drag re-uploads a texel row instead of
        # recompiling. The values are the same float32 bits either way.
        src, mvals = _lift_marked_values(src)
        if 'hal_MV(' in src:
            # cannot happen by construction; refusing beats handing the
            # driver an undefined symbol
            return None, ('internal: a material-value marker survived '
                          'the lift')
        if mvals:
            if len(all_samplers) + 4 > _slim:
                return None, (f'needs {len(all_samplers) + 4} texture '
                              f'samplers; this driver provides {_slim}')
            decl = ('uniform sampler2D hal_mats;\n'
                    'uniform float hal_mrow;\n'
                    'float hal_mv(int t)\n'
                    '{\n'
                    '    return texelFetch(hal_mats, '
                    'ivec2(t, int(hal_mrow)), 0).r;\n'
                    '}\n')
            src = src.replace('void main()', decl + 'void main()', 1)
            info['samplers'] = list(info['samplers']) + ['hal_mats']
            info['frame_uniforms'] = list(frame_unis) + ['hal_mrow']
            info['mat_values'] = mvals
    return src, info
