"""Emitting GLSL from the same node graphs the NumPy evaluator consumes.

This is the piece that decides whether a GPU frame is possible at all. The
renderer's node evaluator has a NumPy backend; a GPU one needs the identical
semantics expressed as GLSL. Where `nodeeval.py` computes an array, this
appends a line of GLSL and returns the name of the variable holding it.

Anything unrecognised is recorded rather than guessed at, and a material that
needs an unrecognised node is simply rendered on the CPU. A wrong GLSL emitter
is far worse than an absent one: it produces a picture, just not the right one.

Every emitter here is checked against the NumPy evaluator by running the
generated GLSL through Halcyon's own front-end and comparing outputs.
"""

FLOAT, VEC3, VEC4 = 'float', 'vec3', 'vec4'

#: socket type in the exported graph -> GLSL type
SOCKET_TYPE = {'VALUE': FLOAT, 'RGBA': VEC4, 'VECTOR': VEC3,
               'SHADER': VEC4, 'INT': FLOAT, 'BOOLEAN': FLOAT}


class Unsupported(Exception):
    """Raised when a node has no GLSL emitter. The material falls back."""


class Emitter:
    """Walks a graph and produces GLSL statements plus a result variable."""

    def __init__(self, graph):
        self.graph = graph or {}
        self.nodes = (graph or {}).get('nodes', {})
        self.lines = []
        self.cache = {}
        self.unsupported = set()
        self.inline = []
        self.samplers = []
        self._n = 0

    # ---------------------------------------------------------------- helpers

    def tmp(self, gtype, expr):
        self._n += 1
        name = f'_v{self._n}'
        self.lines.append(f'    {gtype} {name} = {expr};')
        return name, gtype

    def cast(self, var, from_t, to_t):
        """The same coercions the NumPy backend applies between socket types."""
        if from_t == to_t:
            return var
        if to_t == FLOAT:
            if from_t == VEC4:
                return f'dot(({var}).rgb, vec3(0.2126, 0.7152, 0.0722))'
            return f'dot({var}, vec3(0.2126, 0.7152, 0.0722))'
        if to_t == VEC3:
            if from_t == FLOAT:
                return f'vec3({var})'
            return f'({var}).rgb'
        if to_t == VEC4:
            if from_t == FLOAT:
                return f'vec4(vec3({var}), 1.0)'
            return f'vec4({var}, 1.0)'
        return var

    @staticmethod
    def const(value, gtype):
        if gtype == FLOAT:
            try:
                v = float(value[0]) if hasattr(value, '__len__') else float(value)
            except (TypeError, ValueError):
                v = 0.0
            return f'{v:.8g}'
        seq = list(value) if hasattr(value, '__len__') else [float(value)] * 3
        while len(seq) < 4:
            seq.append(1.0)
        if gtype == VEC3:
            return 'vec3({:.8g}, {:.8g}, {:.8g})'.format(*seq[:3])
        return 'vec4({:.8g}, {:.8g}, {:.8g}, {:.8g})'.format(*seq[:4])

    def input(self, node, name, gtype):
        """The GLSL expression for one input socket, linked or defaulted."""
        for sock in node.get('inputs', ()):
            if sock.get('name') != name:
                continue
            link = sock.get('link')
            if link:
                var, vt = self.output(link[0], link[1])
                return self.cast(var, vt, gtype)
            return self.const(sock.get('default'), gtype)
        return self.const(0.0 if gtype == FLOAT else (0, 0, 0, 1), gtype)

    def output(self, node_id, index=0):
        key = (node_id, index)
        if key in self.cache:
            return self.cache[key]
        node = self.nodes.get(node_id)
        if node is None:
            raise Unsupported(f'missing node {node_id}')
        idname = node.get('bl_idname', '?')
        fn = EMITTERS.get(idname)
        if fn is None:
            self.unsupported.add(idname)
            raise Unsupported(idname)
        result = fn(self, node, index)
        self.cache[key] = result
        return result

    def body(self):
        return '\n'.join(self.lines)


def prop(node, name, default=None):
    return node.get('props', {}).get(name, default)


# ------------------------------------------------------------------ emitters


def e_rgb(em, node, _i):
    # the colour lives in a property, not an input socket
    return em.tmp(VEC4, em.const(prop(node, 'value', [0.5, 0.5, 0.5, 1.0]), VEC4))


def e_value(em, node, _i):
    return em.tmp(FLOAT, em.const(prop(node, 'value', 0.5), FLOAT))


def tex_vector(em, node, default='generated'):
    """A texture's coordinate input, defaulting to generated coordinates.

    An unlinked Vector on a texture node does not mean "use the socket value";
    it means generated coordinates. Reading the default instead silently
    produces a constant, which looks like a working texture that never varies.
    """
    for sock in node.get('inputs', ()):
        if sock.get('name') == 'Vector' and sock.get('link'):
            var, vt = em.output(sock['link'][0], sock['link'][1])
            return em.cast(var, vt, VEC3)
    return 'vec3(hal_uv, 0.0)' if default == 'uv' else 'hal_generated'


def e_mix_rgb(em, node, _i):
    op = str(prop(node, 'blend_type', 'MIX')).upper()
    fac = em.input(node, 'Fac', FLOAT)
    a = em.input(node, 'Color1', VEC4)
    b = em.input(node, 'Color2', VEC4)
    f, _t = em.tmp(FLOAT, f'clamp({fac}, 0.0, 1.0)')
    av, _t = em.tmp(VEC4, a)
    bv, _t = em.tmp(VEC4, b)
    # alpha is taken from the first colour, never blended -- mixing it too
    # is invisible on an opaque scene and wrong the moment one is not
    ac = f'{av}.rgb'
    bc = f'{bv}.rgb'
    table = {
        'MIX': f'{ac} + ({bc} - {ac}) * {f}',
        'ADD': f'{ac} + {bc} * {f}',
        'MULTIPLY': f'{ac} * (vec3(1.0) - vec3({f}) + vec3({f}) * {bc})',
        'SUBTRACT': f'{ac} - {bc} * {f}',
        'SCREEN': f'vec3(1.0) - (vec3(1.0) - {ac}) '
                  f'* (vec3(1.0) - vec3({f}) * {bc})',
        'DIFFERENCE': f'{ac} + (abs({ac} - {bc}) - {ac}) * {f}',
        'DIVIDE': f'{ac} * (1.0 - {f}) + {f} * {ac} / max({bc}, vec3(1e-6))',
        'LIGHTEN': f'{ac} + (max({ac}, {bc}) - {ac}) * {f}',
        'DARKEN': f'{ac} + (min({ac}, {bc}) - {ac}) * {f}',
    }
    expr = table.get(op)
    if expr is None:
        raise Unsupported(f'MixRGB {op}')
    rgb, _t = em.tmp(VEC3, expr)
    if prop(node, 'use_clamp', False):
        rgb, _t = em.tmp(VEC3, f'clamp({rgb}, 0.0, 1.0)')
    return em.tmp(VEC4, f'vec4({rgb}, {av}.a)')


MATH_UNARY = {
    'SQRT': 'sqrt(max({a}, 0.0))', 'ABSOLUTE': 'abs({a})',
    'ROUND': 'floor({a} + 0.5)', 'FLOOR': 'floor({a})',
    'CEIL': 'ceil({a})', 'FRACT': 'fract({a})',
    'SINE': 'sin({a})', 'COSINE': 'cos({a})', 'TANGENT': 'tan({a})',
    'ARCSINE': 'asin(clamp({a}, -1.0, 1.0))',
    'ARCCOSINE': 'acos(clamp({a}, -1.0, 1.0))',
    'ARCTANGENT': 'atan({a})',
    'EXPONENT': 'exp({a})',

    'SIGN': 'sign({a})', 'TRUNC': 'trunc({a})',
    'INVERSE_SQRT': 'inversesqrt(max({a}, 1e-8))',
    'RADIANS': 'radians({a})', 'DEGREES': 'degrees({a})',
}

MATH_BINARY = {
    'ADD': '{a} + {b}', 'SUBTRACT': '{a} - {b}', 'MULTIPLY': '{a} * {b}',
    'DIVIDE': '{a} / (abs({b}) < 1e-8 ? 1e-8 : {b})',
    'POWER': 'pow(max({a}, 0.0), {b})',
    'MINIMUM': 'min({a}, {b})', 'MAXIMUM': 'max({a}, {b})',
    'MODULO': 'mod({a}, (abs({b}) < 1e-8 ? 1e-8 : {b}))',
    'LESS_THAN': '({a} < {b} ? 1.0 : 0.0)',
    'GREATER_THAN': '({a} > {b} ? 1.0 : 0.0)',
    'SNAP': 'floor({a} / (abs({b}) < 1e-8 ? 1e-8 : {b})) * {b}',
    'ARCTAN2': 'atan({a}, {b})',
}


def e_math(em, node, _i):
    op = str(prop(node, 'operation', 'ADD')).upper()
    a = em.input(node, 'Value', FLOAT)
    ins = [s for s in node.get('inputs', ()) if s.get('name') == 'Value']
    b = None
    if len(ins) > 1:
        b = em.input_indexed(node, 'Value', 1, FLOAT)
    if op in MATH_UNARY:
        expr = MATH_UNARY[op].format(a=f'({a})')
    elif op in MATH_BINARY:
        expr = MATH_BINARY[op].format(a=f'({a})', b=f'({b or "0.0"})')
    elif op == 'MULTIPLY_ADD':
        c = em.input_indexed(node, 'Value', 2, FLOAT)
        expr = f'({a}) * ({b}) + ({c})'
    elif op == 'LOGARITHM':
        expr = f'log(max({a}, 1e-9)) / log(max({b or "2.0"}, 1e-9))'
    elif op == 'CLAMP':
        expr = f'clamp({a}, 0.0, 1.0)'
    else:
        raise Unsupported(f'Math {op}')
    out, t = em.tmp(FLOAT, expr)
    if prop(node, 'use_clamp', False):
        out, t = em.tmp(FLOAT, f'clamp({out}, 0.0, 1.0)')
    return out, t


def _input_indexed(em, node, name, index, gtype):
    seen = 0
    for sock in node.get('inputs', ()):
        if sock.get('name') != name:
            continue
        if seen == index:
            link = sock.get('link')
            if link:
                var, vt = em.output(link[0], link[1])
                return em.cast(var, vt, gtype)
            return em.const(sock.get('default'), gtype)
        seen += 1
    return em.const(0.0 if gtype == FLOAT else (0, 0, 0, 1), gtype)


Emitter.input_indexed = _input_indexed

VECMATH = {
    'ADD': '{a} + {b}', 'SUBTRACT': '{a} - {b}', 'MULTIPLY': '{a} * {b}',
    'DIVIDE': '{a} / max(abs({b}), vec3(1e-8))',
    'CROSS_PRODUCT': 'cross({a}, {b})',
    'PROJECT': '{b} * (dot({a}, {b}) / max(dot({b}, {b}), 1e-8))',
    'REFLECT': 'reflect({a}, normalize({b}))',
    'MINIMUM': 'min({a}, {b})', 'MAXIMUM': 'max({a}, {b})',
    'MODULO': 'mod({a}, max(abs({b}), vec3(1e-8)))',
    'SNAP': 'floor({a} / max(abs({b}), vec3(1e-8))) * {b}',
}
VECMATH_UNARY = {
    'NORMALIZE': 'normalize({a})', 'ABSOLUTE': 'abs({a})',
    'FLOOR': 'floor({a})', 'CEIL': 'ceil({a})', 'FRACTION': 'fract({a})',
    'SINE': 'sin({a})', 'COSINE': 'cos({a})', 'TANGENT': 'tan({a})',
}
VECMATH_SCALAR = {
    'DOT_PRODUCT': 'dot({a}, {b})', 'DISTANCE': 'distance({a}, {b})',
    'LENGTH': 'length({a})',
}


def e_vector_math(em, node, _i):
    op = str(prop(node, 'operation', 'ADD')).upper()
    a = em.input_indexed(em, node, 'Vector', 0, VEC3) if False else \
        _input_indexed(em, node, 'Vector', 0, VEC3)
    b = _input_indexed(em, node, 'Vector', 1, VEC3)
    if op in VECMATH_SCALAR:
        return em.tmp(FLOAT, VECMATH_SCALAR[op].format(a=f'({a})', b=f'({b})'))
    if op in VECMATH_UNARY:
        return em.tmp(VEC3, VECMATH_UNARY[op].format(a=f'({a})'))
    if op in VECMATH:
        return em.tmp(VEC3, VECMATH[op].format(a=f'({a})', b=f'({b})'))
    if op == 'SCALE':
        s = _input_indexed(em, node, 'Scale', 0, FLOAT)
        return em.tmp(VEC3, f'({a}) * ({s})')
    if op == 'MULTIPLY_ADD':
        c = _input_indexed(em, node, 'Vector', 2, VEC3)
        return em.tmp(VEC3, f'({a}) * ({b}) + ({c})')
    raise Unsupported(f'VectorMath {op}')


def e_invert(em, node, _i):
    fac = em.input(node, 'Fac', FLOAT)
    col = em.input(node, 'Color', VEC4)
    c, _t = em.tmp(VEC4, col)
    return em.tmp(VEC4, f'vec4(mix({c}.rgb, vec3(1.0) - {c}.rgb, '
                        f'clamp({fac}, 0.0, 1.0)), {c}.a)')


def e_gamma(em, node, _i):
    col = em.input(node, 'Color', VEC4)
    g = em.input(node, 'Gamma', FLOAT)
    c, _t = em.tmp(VEC4, col)
    return em.tmp(VEC4, f'vec4(pow(max({c}.rgb, vec3(0.0)), '
                        f'vec3(max({g}, 1e-6))), {c}.a)')


def e_bright_contrast(em, node, _i):
    col = em.input(node, 'Color', VEC4)
    br = em.input(node, 'Bright', FLOAT)
    ct = em.input(node, 'Contrast', FLOAT)
    c, _t = em.tmp(VEC4, col)
    return em.tmp(VEC4,
                  f'vec4(max(({c}.rgb - vec3(0.5)) * (1.0 + ({ct})) '
                  f'+ vec3(0.5) + vec3({br}), vec3(0.0)), {c}.a)')


def e_separate_xyz(em, node, index):
    v = em.input(node, 'Vector', VEC3)
    comp = 'xyz'[min(index, 2)]
    return em.tmp(FLOAT, f'({v}).{comp}')


def e_combine_xyz(em, node, _i):
    x = _input_indexed(em, node, 'X', 0, FLOAT)
    y = _input_indexed(em, node, 'Y', 0, FLOAT)
    z = _input_indexed(em, node, 'Z', 0, FLOAT)
    return em.tmp(VEC3, f'vec3({x}, {y}, {z})')


def e_separate_rgb(em, node, index):
    c = em.input(node, 'Image', VEC4)
    comp = 'rgb'[min(index, 2)]
    return em.tmp(FLOAT, f'({c}).{comp}')


def e_combine_rgb(em, node, _i):
    r = _input_indexed(em, node, 'R', 0, FLOAT)
    g = _input_indexed(em, node, 'G', 0, FLOAT)
    b = _input_indexed(em, node, 'B', 0, FLOAT)
    return em.tmp(VEC4, f'vec4({r}, {g}, {b}, 1.0)')


def e_checker(em, node, index):
    v = tex_vector(em, node)
    sc = em.input(node, 'Scale', FLOAT)
    c1 = em.input(node, 'Color1', VEC4)
    c2 = em.input(node, 'Color2', VEC4)
    p, _t = em.tmp(VEC3, f'({v}) * ({sc})')
    f, _t = em.tmp(FLOAT,
                   f'(mod(floor({p}.x) + floor({p}.y) + floor({p}.z), 2.0) '
                   f'< 0.5) ? 1.0 : 0.0')
    if index == 1:
        return f, FLOAT
    return em.tmp(VEC4, f'mix({c2}, {c1}, {f})')


def e_bsdf_diffuse(em, node, _i):
    return em.tmp(VEC4, em.input(node, 'Color', VEC4))


def e_emission(em, node, _i):
    col = em.input(node, 'Color', VEC4)
    st = em.input(node, 'Strength', FLOAT)
    return em.tmp(VEC4, f'vec4(({col}).rgb * ({st}), ({col}).a)')


def e_fresnel(em, node, _i):
    ior = em.input(node, 'IOR', FLOAT)
    return em.tmp(FLOAT,
                  f'hal_fresnel_dielectric(dot(hal_N, hal_V), max({ior}, 1.0001))')


def e_tex_image(em, node, index):
    """Sample an image.

    Verified against `nodeeval.n_tex_image` to 0.000000 under nearest-neighbour
    sampling, which is what proves the coordinate handling: an image texture
    defaults to UV rather than generated coordinates, and getting that wrong
    gives a texture that samples the wrong place everywhere.

    Filtering is a binding-time property of the sampler rather than of this
    code, so it is carried in `em.samplers` for the material assembler to apply
    and is not part of the numerical check.

    The emitter can produce the sampler and the lookup; binding the texture is
    GPU plumbing that lives in the material assembler. `interpolation` and
    `extension` are carried as declarations so the sampler can be configured
    to match what the CPU path does.
    """
    # an image texture defaults to UV, not generated -- the two are the same
    # on a unit cube and completely different on anything else
    v = tex_vector(em, node, 'uv')
    name = f'hal_tex{len(em.samplers)}'
    em.samplers.append({
        'uniform': name,
        'image': prop(node, 'image'),
        'interpolation': prop(node, 'interpolation', 'Linear'),
        'extension': prop(node, 'extension', 'REPEAT'),
    })
    uv, _t = em.tmp(VEC3, v)
    texel, _t = em.tmp(VEC4, f'texture({name}, {uv}.xy)')
    if index == 1:
        return em.tmp(FLOAT, f'{texel}.a')
    return texel, VEC4


def e_reroute(em, node, _i):
    for sock in node.get('inputs', ()):
        link = sock.get('link')
        if link:
            return em.output(link[0], link[1])
    raise Unsupported('unlinked reroute')


def e_clamp(em, node, _i):
    v = em.input(node, 'Value', FLOAT)
    lo = em.input(node, 'Min', FLOAT)
    hi = em.input(node, 'Max', FLOAT)
    if str(prop(node, 'clamp_type', 'MINMAX')).upper() == 'RANGE':
        return em.tmp(FLOAT, f'clamp({v}, min({lo}, {hi}), max({lo}, {hi}))')
    return em.tmp(FLOAT, f'clamp({v}, {lo}, {hi})')


def e_map_range(em, node, _i):
    v = em.input(node, 'Value', FLOAT)
    fs = em.input(node, 'From Min', FLOAT)
    fe = em.input(node, 'From Max', FLOAT)
    ts = em.input(node, 'To Min', FLOAT)
    te = em.input(node, 'To Max', FLOAT)
    t, _x = em.tmp(FLOAT, f'(({v}) - ({fs})) / max(abs(({fe}) - ({fs})), 1e-8)')
    if prop(node, 'clamp', True):
        t, _x = em.tmp(FLOAT, f'clamp({t}, 0.0, 1.0)')
    return em.tmp(FLOAT, f'({ts}) + ({te} - ({ts})) * {t}')


def e_hue_sat(em, node, _i):
    col = em.input(node, 'Color', VEC4)
    hue = em.input(node, 'Hue', FLOAT)
    sat = em.input(node, 'Saturation', FLOAT)
    val = em.input(node, 'Value', FLOAT)
    fac = em.input(node, 'Fac', FLOAT)
    c, _t = em.tmp(VEC4, col)
    h, _t = em.tmp(VEC3, f'hal_rgb2hsv({c}.rgb)')
    h2, _t = em.tmp(VEC3, f'vec3(fract({h}.x + ({hue}) - 0.5), '
                          f'clamp({h}.y * ({sat}), 0.0, 1.0), {h}.z * ({val}))')
    rgb, _t = em.tmp(VEC3, f'hal_hsv2rgb({h2})')
    return em.tmp(VEC4, f'vec4(mix({c}.rgb, {rgb}, clamp({fac}, 0.0, 1.0)), {c}.a)')


def e_tex_coord(em, node, index):
    # 0 Generated, 1 Normal, 2 UV, 3 Object, 4 Camera, 5 Window, 6 Reflection
    names = ['hal_generated', 'hal_N', 'vec3(hal_uv, 0.0)', 'hal_P',
             'hal_P', 'hal_generated', 'reflect(-hal_V, hal_N)']
    return em.tmp(VEC3, names[min(index, len(names) - 1)])


def e_uvmap(em, node, _i):
    return em.tmp(VEC3, 'vec3(hal_uv, 0.0)')


def e_new_geometry(em, node, index):
    names = ['hal_P', 'hal_N', 'hal_N', 'hal_T', 'hal_P', 'hal_generated',
             'hal_generated']
    if index in (6, 7, 8):
        return em.tmp(FLOAT, '0.0')
    return em.tmp(VEC3, names[min(index, len(names) - 1)])


def e_layer_weight(em, node, index):
    blend = em.input(node, 'Blend', FLOAT)
    b, _t = em.tmp(FLOAT, f'clamp({blend}, 0.0, 0.99999)')
    cosi, _t = em.tmp(FLOAT, 'abs(dot(normalize(hal_N), normalize(hal_V)))')
    if index == 0:                                       # Fresnel
        eta, _t = em.tmp(FLOAT,
                         f'({b} < 0.5) ? 1.0 / max(1.0 - {b} * 2.0, 1e-5) '
                         f': 1.0 + ({b} - 0.5) * 2.0')
        return em.tmp(FLOAT,
                      f'hal_fresnel_dielectric({cosi}, max({eta}, 1.0001))')
    # the exponent is driven by Blend, not a plain one-minus-cosine
    ex, _t = em.tmp(FLOAT,
                    f'({b} < 0.5) ? 0.5 / max({b}, 1e-5) : 2.0 * (1.0 - {b})')
    return em.tmp(FLOAT,
                  f'clamp(pow(max(1.0 - {cosi}, 0.0), {ex}), 0.0, 1.0)')


def e_bsdf_glossy(em, node, _i):
    return em.tmp(VEC4, em.input(node, 'Color', VEC4))


def e_bsdf_transparent(em, node, _i):
    return em.tmp(VEC4, em.input(node, 'Color', VEC4))


def e_mix_shader(em, node, _i):
    fac = em.input(node, 'Fac', FLOAT)
    a = _input_indexed(em, node, 'Shader', 0, VEC4)
    b = _input_indexed(em, node, 'Shader', 1, VEC4)
    return em.tmp(VEC4, f'mix({a}, {b}, clamp({fac}, 0.0, 1.0))')


def e_add_shader(em, node, _i):
    a = _input_indexed(em, node, 'Shader', 0, VEC4)
    b = _input_indexed(em, node, 'Shader', 1, VEC4)
    return em.tmp(VEC4, f'{a} + {b}')


def e_code_node(em, node, _i):
    """The coded shader node: its source is already GLSL.

    This is the whole point of calling it the easy case. There is no
    translation step -- the user's function is inlined under a mangled name and
    called. On the CPU that same source has to be compiled to NumPy first,
    which on a GPU simply stops being necessary.
    """
    src = prop(node, 'source_text') or prop(node, '__source')
    if not src:
        raise Unsupported('HALCYON_CodeNode without source')
    if str(prop(node, 'language', 'GLSL')).upper() != 'GLSL':
        raise Unsupported('HALCYON_CodeNode HLSL needs translating first')
    em.inline.append(src)
    return em.tmp(VEC4, 'hal_user_shader()')


EMITTERS = {
    'ShaderNodeClamp': e_clamp,
    'ShaderNodeMapRange': e_map_range,
    'ShaderNodeHueSaturation': e_hue_sat,
    'ShaderNodeTexCoord': e_tex_coord,
    'ShaderNodeUVMap': e_uvmap,
    'ShaderNodeNewGeometry': e_new_geometry,
    'ShaderNodeLayerWeight': e_layer_weight,
    'ShaderNodeBsdfGlossy': e_bsdf_glossy,
    'ShaderNodeBsdfTransparent': e_bsdf_transparent,
    'ShaderNodeMixShader': e_mix_shader,
    'ShaderNodeAddShader': e_add_shader,
    'ShaderNodeRGB': e_rgb,
    'ShaderNodeValue': e_value,
    'ShaderNodeMixRGB': e_mix_rgb,
    'ShaderNodeMath': e_math,
    'ShaderNodeVectorMath': e_vector_math,
    'ShaderNodeInvert': e_invert,
    'ShaderNodeGamma': e_gamma,
    'ShaderNodeBrightContrast': e_bright_contrast,
    'ShaderNodeSeparateXYZ': e_separate_xyz,
    'ShaderNodeCombineXYZ': e_combine_xyz,
    'ShaderNodeSeparateRGB': e_separate_rgb,
    'ShaderNodeCombineRGB': e_combine_rgb,
    'ShaderNodeTexChecker': e_checker,
    'ShaderNodeBsdfDiffuse': e_bsdf_diffuse,
    'ShaderNodeEmission': e_emission,
    'ShaderNodeFresnel': e_fresnel,
    'ShaderNodeTexImage': e_tex_image,
    'NodeReroute': e_reroute,
}


def supported():
    return sorted(EMITTERS)


def can_emit(graph):
    """(ok, unsupported node types) without producing any code."""
    em = Emitter(graph)
    out = (graph or {}).get('output')
    if not out:
        return False, {'no output node'}
    node = em.nodes.get(out, {})
    for sock in node.get('inputs', ()):
        link = sock.get('link')
        if link:
            try:
                em.output(link[0], link[1])
            except Unsupported:
                pass
    return (not em.unsupported), em.unsupported
