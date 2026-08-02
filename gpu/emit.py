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
        self.once = set()          # nodes whose setup lines are already down
        self.frame_uniforms = set()  # per-frame scalars a code node asked for
        self.programs = None       # material's compiled programs, or None
        self.frame_mode = False    # True only under assemble_frame
        self.secondary = False     # True for reflection-hit passes
        self.used_screen = False   # a code node read vScreenUV/iResolution
        self.bump_passes = []      # Bump nodes needing a height pre-pass
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
            note = REFUSED.get(idname)
            self.unsupported.add(idname if note is None
                                 else f'{idname} ({note})')
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


def e_mapping(em, node, _i):
    """ShaderNodeMapping: the CPU's exact transform order, trig baked.

    `n_mapping` multiplies by Scale, rotates X-then-Y-then-Z in a
    hand-rolled sequence, then adds Location (POINT) or normalizes
    (NORMAL); TEXTURE subtracts Location, rotates by the NEGATED angles
    through the SAME sequence (nodeeval's own quirk, reproduced rather
    than corrected), and divides by the Scale floored at 1e-8. When the
    Rotation socket is unlinked -- practically always -- its cos/sin
    are computed HERE with NumPy's float32 trig and baked as literals,
    so the driver does no trigonometry a CPU frame did not do (a
    driver's sin rounds differently, and a texture lookup at a texel
    boundary is a cliff). A per-pixel-driven Rotation falls back to
    in-shader trig; the field's mapped textures are constant mappings.
    """
    import numpy as np
    mode = str(prop(node, 'vector_type', 'POINT')).upper()

    def unlinked(namev, default):
        for sock in node.get('inputs', ()):
            if sock.get('name') == namev:
                if sock.get('link'):
                    return None
                d = sock.get('default')
                seq = list(d) if hasattr(d, '__len__') else [d, d, d]
                return np.asarray([float(x) for x in (seq + [0, 0, 0])[:3]],
                                  np.float32)
        return np.asarray(default, np.float32)

    rot_c = unlinked('Rotation', (0.0, 0.0, 0.0))
    scl_c = unlinked('Scale', (1.0, 1.0, 1.0))
    v, _t = em.tmp(VEC3, em.input(node, 'Vector', VEC3))

    if mode == 'TEXTURE':
        v, _t = em.tmp(VEC3, f'{v} - {em.input(node, "Location", VEC3)}')

    def rotate(vv, trig):
        cxe, sxe, cye, sye, cze, sze = trig
        y1, _ = em.tmp(FLOAT, f'{vv}.y * {cxe} - {vv}.z * {sxe}')
        z1, _ = em.tmp(FLOAT, f'{vv}.y * {sxe} + {vv}.z * {cxe}')
        x2, _ = em.tmp(FLOAT, f'{vv}.x * {cye} + {z1} * {sye}')
        z2, _ = em.tmp(FLOAT, f'-{vv}.x * {sye} + {z1} * {cye}')
        x3, _ = em.tmp(FLOAT, f'{x2} * {cze} - {y1} * {sze}')
        y3, _ = em.tmp(FLOAT, f'{x2} * {sze} + {y1} * {cze}')
        out, _ = em.tmp(VEC3, f'vec3({x3}, {y3}, {z2})')
        return out

    def trig_for(sign):
        if rot_c is not None:
            r = rot_c * np.float32(sign)
            if not np.any(np.abs(r) > 0.0):
                return None                     # identity: cx=1, sx=0 exact
            return tuple(f'{np.float32(f(c)):.8g}'
                         for c in r for f in (np.cos, np.sin))
        rr, _ = em.tmp(VEC3, em.input(node, 'Rotation', VEC3))
        if sign < 0:
            rr, _ = em.tmp(VEC3, f'-{rr}')
        parts = []
        for axis in ('x', 'y', 'z'):
            ce, _ = em.tmp(FLOAT, f'cos({rr}.{axis})')
            se, _ = em.tmp(FLOAT, f'sin({rr}.{axis})')
            parts += [ce, se]
        return tuple(parts)

    if mode == 'TEXTURE':
        trig = trig_for(-1.0)
        if trig is not None:
            v = rotate(v, trig)
        if scl_c is not None:
            floored = np.where(np.abs(scl_c) < 1e-8,
                               np.float32(1e-8), scl_c)
            v, _t = em.tmp(VEC3, f'{v} / {em.const(tuple(floored), VEC3)}')
        else:
            s, _ = em.tmp(VEC3, em.input(node, 'Scale', VEC3))
            sf, _ = em.tmp(
                VEC3,
                f'vec3(abs({s}.x) < 1e-8 ? 1e-8 : {s}.x, '
                f'abs({s}.y) < 1e-8 ? 1e-8 : {s}.y, '
                f'abs({s}.z) < 1e-8 ? 1e-8 : {s}.z)')
            v, _t = em.tmp(VEC3, f'{v} / {sf}')
        return v, VEC3

    v, _t = em.tmp(VEC3, f'{v} * {em.input(node, "Scale", VEC3)}')
    trig = trig_for(1.0)
    if trig is not None:
        v = rotate(v, trig)
    if mode == 'POINT':
        v, _t = em.tmp(VEC3, f'{v} + {em.input(node, "Location", VEC3)}')
    elif mode == 'NORMAL':
        v, _t = em.tmp(VEC3, f'normalize({v})')
    return v, VEC3


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


# --- the coded shader node ------------------------------------------------
#
# Its source is already GLSL -- the whole point of calling it the easy case.
# On the CPU that source is compiled to a NumPy program; on the GPU the
# translation step simply stops being necessary. What does NOT stop being
# necessary is the contract around the source: uniforms become sockets,
# declared `in` names bind to the renderer's varyings, `out` names become
# output sockets. All of that is reproduced below, and every piece of the
# contract the deferred pass cannot honour refuses by name.

#: CPU varying binding -> (frame-shader expression, its GLSL arity). The
#: expressions reference main() locals; the assignments are emitted inside
#: main(), so that is exactly where they are in scope. hal_time / hal_frame
#: are per-frame uniforms for the same reason hal_eye is: baking them meant
#: an animation recompiled every frame.
CODE_VARYINGS = {
    'position': ('hal_P', 3),
    'normal': ('hal_N', 3),
    'uv': ('hal_uv', 2),
    'color': ('hal_vcol', 4),
    'view': ('hal_V', 3),
    'incident': ('(-hal_V)', 3),
    'camera': ('hal_eye', 3),
    'time': ('hal_time', 1),
    'frame': ('hal_frame', 1),
    'tangent': (None, 3),               # built from hal_N at the call site
    'object': ('hal_generated', 3),     # per-object bounds bake per scene
    'screenuv': (None, 2),              # gl_FragCoord.xy / width, baked
    'resolution': (None, 3),            # (w, h, 1), baked per plan
}

#: varying bindings the G-buffer cannot honestly provide, and why
CODE_REFUSED_VARYINGS = {
    'fragcoord': 'its z is the view-space depth, which the fullscreen pass '
                 'does not carry (vScreenUV carries the xy)',
    'depth': 'view-space depth is not in the G-buffer',
    'backfacing': 'the rasteriser decides it from winding, which the '
                  'G-buffer does not carry',
    'geonormal': 'the face normal is not in the G-buffer',
    'uv2': 'the G-buffer carries one UV layer',
    'bitangent': 'the CPU leaves it unbound (zeros); declare nothing and it '
                 'matches, declare it and it would not',
    'random': 'the per-fragment random stream lives on the CPU',
}

#: HLSL-flavoured names Halcyon's own forgiving front-end accepts in GLSL
#: mode but a real driver will not. Refused by name here, because "compiles
#: in the preview, dies on the driver" is the worst possible seam.
CODE_HLSL_NAMES = ('saturate', 'lerp', 'frac', 'mul', 'rsqrt', 'ddx', 'ddy',
                   'tex2D', 'atan2', 'fmod', 'float2', 'float3', 'float4',
                   'half2', 'half3', 'half4', 'lit')

_CODE_PORTS = {}


def _strip_comments(text):
    import re
    text = re.sub(r'/\*.*?\*/', ' ', text, flags=re.S)
    return re.sub(r'//[^\n]*', ' ', text)


def _port_code_node(src, tag):
    """Transform one coded shader into frame-safe GLSL.

    Returns (inline_text, uniforms, ins, outs) where `uniforms` is
    [(glsl_type_str, name, mangled, default_list)], `ins` is
    [(name, mangled, binding, glsl_type_str, arity)] and `outs` maps output
    name -> (mangled, glsl_type_str). Raises Unsupported with the exact
    reason otherwise. Cached on (source, tag): the parse and the renames run
    once per plan, not once per socket read.
    """
    import re

    key = (src, tag)
    hit = _CODE_PORTS.get(key)
    if hit is not None:
        return hit

    from ..shaders.codegen import VARYINGS as VTAB
    from ..shaders.compiler import default_value
    from ..shaders.lexer import ShaderError
    from ..shaders.parser import parse

    bare = _strip_comments(src)
    if re.search(r'\bdiscard\b', bare):
        raise Unsupported('the coded shader discards fragments, which the '
                          'probe cannot rule out from sixteen samples')
    if re.search(r'\bgl_FragCoord\b', bare):
        raise Unsupported('gl_FragCoord.z is the view-space depth on the '
                          'CPU, which the fullscreen pass does not carry -- '
                          'declare `in vec2 vScreenUV;` for the xy')
    hlslish = sorted({m for m in CODE_HLSL_NAMES
                      if re.search(r'\b%s\s*\(' % m, bare)
                      or re.search(r'\b%s\b' % m, bare) and m[0] in 'fhi'})
    if hlslish:
        raise Unsupported('HLSL-flavoured GLSL (%s): Halcyon\'s preview '
                          'accepts it, a driver will not'
                          % ', '.join(hlslish))
    try:
        decls, structs = parse(src, hlsl=False,
                               defines={'HALCYON': ([], '1'),
                                        'GLSL': ([], '1')})
    except (ShaderError, RecursionError) as exc:
        raise Unsupported(f'the coded shader does not parse: {exc}')

    uniforms, ins, outs, names, samplers = [], [], {}, [], []
    has_main = False
    for d in decls:
        kind = d[0]
        if kind == 'global':
            quals, gtype, name = d[1], d[2], d[3]
            names.append(name)
            init = d[5] if len(d) > 5 else None    # d[4] is the array flag
            if 'uniform' in quals:
                if getattr(gtype, 'base', '') == 'sampler':
                    # image inputs travel: the socket's prepared pixels ride
                    # the same manual-sampler machinery as every texture
                    samplers.append((name, f'_cn{tag}_{name}'))
                    continue
                if getattr(gtype, 'is_matrix', False):
                    raise Unsupported('matrix uniforms are not in the '
                                      'deferred pass yet')
                if str(gtype) not in ('float', 'vec3', 'vec4'):
                    raise Unsupported(f'{gtype} uniforms are not in the '
                                      'deferred pass yet (float, vec3 and '
                                      'vec4 are)')
                uniforms.append((str(gtype), name,
                                 f'_cn{tag}_{name}',
                                 default_value(gtype, init)))
            elif 'in' in quals:
                bound = VTAB.get(name)
                binding = bound[0] if bound is not None else None
                if binding in CODE_REFUSED_VARYINGS:
                    raise Unsupported(
                        f'the coded shader reads {name}: '
                        f'{CODE_REFUSED_VARYINGS[binding]}')
                if binding is not None and binding not in CODE_VARYINGS:
                    raise Unsupported(f'the coded shader reads {name}, '
                                      'which the deferred pass does not '
                                      'carry')
                n_want = getattr(gtype, 'n', 1)
                if binding is not None and n_want == 1 and \
                        CODE_VARYINGS[binding][1] > 1:
                    raise Unsupported(f'{name} declared as a scalar; the '
                                      'CPU leaves the extra lanes in play '
                                      'and the pass cannot reproduce that')
                ins.append((name, f'_cn{tag}_{name}', binding,
                            str(gtype), n_want))
            elif 'out' in quals:
                outs[name] = (f'_cn{tag}_{name}', str(gtype))
            # plain globals stay in the user text, renamed below
        elif kind == 'func':
            names.append(d[2])
            if d[2] == 'main':
                has_main = True
                if str(d[1]) != 'void':
                    raise Unsupported('the coded shader returns a value '
                                      'from main; declare an out variable '
                                      'for the deferred pass')
            if d[2] == 'mainImage':
                raise Unsupported('mainImage entry points need fragcoord, '
                                  'which the fullscreen pass does not carry')
        elif kind == 'struct':
            names.append(d[1])
    if not has_main:
        raise Unsupported('the coded shader has no main()')
    if not outs:
        raise Unsupported('the coded shader declares no out variable')

    # strip the declaration statements (line-anchored so parameter `in`s and
    # locals survive), then rename every top-level identifier
    text = re.sub(r'(?m)^[ \t]*(?:layout\s*\([^)]*\)\s*)?'
                  r'(?:uniform|in|out)\b[^;]*;[ \t]*\n?', '', src)
    rename = {n: f'_cn{tag}_{n}' for n in names}
    for name, mangled, _b, _t, _n in ins:
        rename[name] = mangled
    for name, (mangled, _t) in outs.items():
        rename[name] = mangled
    for name, mangled in samplers:
        rename[name] = mangled
    for name in sorted(rename, key=len, reverse=True):
        text = re.sub(r'(?<!\.)\b%s\b' % re.escape(name), rename[name], text)

    decl_lines = ['#define HALCYON 1', '#define GLSL 1']
    for gtype, _name, mangled, _default in uniforms:
        decl_lines.append(f'{gtype} {mangled};')
    for _name, mangled, _b, gtype, _n in ins:
        decl_lines.append(f'{gtype} {mangled};')
    for _name, (mangled, gtype) in outs.items():
        decl_lines.append(f'{gtype} {mangled};')
    inline = '\n'.join(decl_lines) + '\n' + text
    result = (inline, uniforms, ins, outs, samplers)
    if len(_CODE_PORTS) > 64:
        _CODE_PORTS.clear()
    _CODE_PORTS[key] = result
    return result


def _code_socket_expr(em, node, uname, gtype, default):
    """The GLSL expression for one coded-shader uniform's value.

    Exactly `n_halcyon_code`: a socket whose uniform/identifier/name matches
    supplies the value (linked chain or socket default); no socket at all
    falls back to the default declared in the source.
    """
    want = {'float': FLOAT, 'vec3': VEC3, 'vec4': VEC4}[gtype]
    for sock in node.get('inputs', ()):
        gl = sock.get('uniform') or sock.get('identifier') or sock.get('name')
        if gl != uname:
            continue
        skind = SOCKET_TYPE.get(sock.get('type', 'VALUE'))
        if skind != want:
            raise Unsupported(f"socket '{sock.get('name')}' is "
                              f"{sock.get('type')} but the uniform is "
                              f'{gtype}; the CPU splats mismatches in ways '
                              'the pass does not reproduce')
        return em.input(node, sock.get('name'), want)
    return em.const(default if default is not None else 0.0, want)


def e_code_node(em, node, index):
    """The coded shader node, running natively in the deferred pass.

    The user's GLSL is inlined under per-node mangled names -- functions,
    structs, globals, everything, so two coded shaders sharing a uniform
    name cannot collide. Uniforms and varyings become plain globals the
    frame's main() assigns before calling the shader's own (renamed) main:
    that reproduces the CPU contract exactly, where varyings and socket
    values are visible from any helper function, not just the entry point.
    """
    if not em.frame_mode:
        raise Unsupported('the coded shader node emits only into the '
                          'deferred frame pass')
    if str(prop(node, 'language', 'GLSL')).upper() != 'GLSL':
        raise Unsupported('HALCYON_CodeNode HLSL needs translating first')
    if prop(node, 'as_surface', False):
        raise Unsupported('a coded shader used as a surface becomes an '
                          'emission closure, which the frame pass does not '
                          'reproduce yet')
    outs_meta = node.get('outputs') or []
    if not outs_meta:
        raise Unsupported('HALCYON_CodeNode without outputs')
    o = outs_meta[index] if index < len(outs_meta) else outs_meta[0]
    o_type = SOCKET_TYPE.get(o.get('type', 'RGBA'), VEC4)

    # the CPU's truth for a node whose program never compiled: every output
    # is zeros -- alpha included, coerce(None) has no opinion about alpha
    key = node.get('id')
    if em.programs is not None and key not in em.programs:
        zero = {FLOAT: '0.0', VEC3: 'vec3(0.0)',
                VEC4: 'vec4(0.0, 0.0, 0.0, 0.0)'}[o_type]
        return em.tmp(o_type, zero)

    src = prop(node, 'source_text') or prop(node, '__source')
    if not src:
        raise Unsupported('HALCYON_CodeNode without source')
    import re
    tag = re.sub(r'\W', '_', str(key))
    inline, uniforms, ins, outs, samplers = _port_code_node(src, tag)

    if key not in em.once:
        em.once.add(key)
        em.inline.append(inline)
        for uname, mangled in samplers:
            # the socket names the image; its prepared pixels ride the
            # manual-sampler machinery. A missing image samples zeros on
            # the CPU, and the assembler emits a zeros sampler to match
            img = None
            for s in node.get('inputs', ()):
                gl = s.get('uniform') or s.get('identifier') or s.get('name')
                if gl == uname and s.get('is_image'):
                    img = s.get('image')
                    break
            em.samplers.append({'uniform': mangled, 'image': img,
                                'code': True})
        for name, mangled, binding, gtype, n_want in ins:
            if binding is None:
                # an in-name the CPU cannot bind reads zeros there too
                em.lines.append(f'    {mangled} = {gtype}(0.0);')
                continue
            if binding in ('screenuv', 'resolution'):
                em.used_screen = True
                res = getattr(em, 'resolution', None)
                if res is None:
                    raise Unsupported('screen coordinates need the frame '
                                      'resolution, which only the deferred '
                                      'pass supplies')
                w, h = float(res[0]), float(res[1])
                if binding == 'screenuv':
                    # (px+0.5, py+0.5)/width, derived from vUV rather than
                    # gl_FragCoord: vUV's orientation is proven by every
                    # agreement test, the builtin's y origin is not
                    expr, arity = (f'(vec2(vUV.x * {_c(w)}, '
                                   f'vUV.y * {_c(h)}) / {_c(w)})', 2)
                else:
                    expr, arity = (f'vec3({_c(w)}, {_c(h)}, 1.0)', 3)
                if n_want == arity:
                    em.lines.append(f'    {mangled} = {expr};')
                else:
                    em.lines.append(f'    {mangled} = {gtype}(({expr}).x);')
                continue
            if binding == 'tangent':
                nn, _t = em.tmp(VEC3, 'normalize(hal_N)')
                up, _t = em.tmp(VEC3, f'(abs({nn}.z) < 0.999) '
                                      f'? vec3(0.0, 0.0, 1.0) '
                                      f': vec3(1.0, 0.0, 0.0)')
                expr, arity = f'normalize(cross({up}, {nn}))', 3
            else:
                expr, arity = CODE_VARYINGS[binding]
                if expr in ('hal_time', 'hal_frame'):
                    em.frame_uniforms.add(expr)
            if n_want == arity:
                em.lines.append(f'    {mangled} = {expr};')
            elif arity == 1:
                em.lines.append(f'    {mangled} = {gtype}({expr});')
            else:
                # the CPU's adapt() splats lane zero on arity mismatch
                em.lines.append(f'    {mangled} = {gtype}(({expr}).x);')
        for gtype, uname, mangled, default in uniforms:
            em.lines.append(f'    {mangled} = '
                            f'{_code_socket_expr(em, node, uname, gtype, default)};')
        em.lines.append(f'    _cn{tag}_main();')

    want = o.get('key') or o.get('name')
    got = outs.get(want)
    if got is None:
        # an output socket with no matching out variable reads zeros on the
        # CPU (the program never wrote that key)
        zero = {FLOAT: '0.0', VEC3: 'vec3(0.0)',
                VEC4: 'vec4(0.0, 0.0, 0.0, 0.0)'}[o_type]
        return em.tmp(o_type, zero)
    mangled, gtype = got
    src_t = {'float': FLOAT, 'vec3': VEC3, 'vec4': VEC4,
             'vec2': VEC3, 'int': FLOAT, 'bool': FLOAT}.get(gtype)
    if src_t is None or gtype == 'vec2':
        raise Unsupported(f'a {gtype} out variable is not in the deferred '
                          'pass yet (float, vec3 and vec4 are)')
    if gtype in ('int', 'bool'):
        return em.tmp(o_type, em.cast(f'float({mangled})', FLOAT, o_type))
    return em.tmp(o_type, em.cast(mangled, src_t, o_type))


def e_halcyon_shader(em, node, _i):
    """The master shader node, as the deferred pass needs it: its colour.

    Every other socket on this node is a surface parameter, and the frame
    probe harvests those through `closure_to_surface` itself -- baked when
    constant, refused by name when varying. What the emitter owes the frame
    is the one thing that may vary per pixel: the Diffuse Color chain. This
    is the node every converted material is built around, so until this
    existed, no converted scene ever qualified for the GPU.
    """
    vmix_on = False
    vcol_linked = False
    for sock in node.get('inputs', ()):
        name = sock.get('name')
        if name == 'Vertex Color Mix':
            try:
                default = float(sock.get('default') or 0.0)
            except (TypeError, ValueError):
                default = 0.0
            vmix_on = bool(sock.get('link')) or default > 1e-6
        if name == 'Vertex Color':
            vcol_linked = bool(sock.get('link'))
        # a linked Normal no longer refuses: the frame assembler emits that
        # chain itself and bends the geometric normal exactly as
        # closure_to_surface does, Bump Strength lerp included
    base, _t = em.tmp(VEC4, em.input(node, 'Diffuse Color', VEC4))
    if vmix_on:
        # vertex colour blends OVER the diffuse, exactly as the evaluator
        # does it; an unlinked socket reads the mesh's own painted colour,
        # which the G-buffer carries in the slot the tangent never used
        vmix, _t = em.tmp(FLOAT,
                          f'clamp({em.input(node, "Vertex Color Mix", FLOAT)}'
                          f', 0.0, 1.0)')
        vcol = em.input(node, 'Vertex Color', VEC4) if vcol_linked \
            else 'hal_vcol'
        base, _t = em.tmp(VEC4, f'{base} + ({vcol} - {base}) * {vmix}')
    return base, VEC4


def e_vertex_color(em, node, _i):
    """The mesh's painted colour, straight from the G-buffer.

    The G-buffer carries the active colour layer; a node naming some other
    layer refuses rather than quietly reading the wrong paint.
    """
    if prop(node, 'layer_name', ''):
        raise Unsupported('named colour layers are not in the G-buffer; '
                          'the active layer is')
    return em.tmp(VEC4, 'hal_vcol')


# --- the period pattern textures ------------------------------------------
#
# Halcyon's own procedurals (Marble, Wood, Granite, Dents, Crackle) ride the
# integer hash in core/patterns.py, which uint32 GLSL reproduces bit for bit
# -- the library lives in gpu/procedural.py, verified function by function
# against its NumPy original. The evaluator collapses every scalar pattern
# parameter to its batch mean, so a LINKED scalar socket refuses by name: a
# varying chain would render differently on the two paths. Colours and the
# Vector/Scale inputs stay per-pixel on both sides.


def _need_prims(em):
    from .procedural import PRIM_GLSL
    if '__pt_prims' not in em.once:
        em.once.add('__pt_prims')
        em.inline.append(PRIM_GLSL)


def _need_pattern(em, name):
    from .procedural import PATTERN_GLSL
    _need_prims(em)
    key = ('__pat', name)
    if key not in em.once:
        em.once.add(key)
        em.inline.append(PATTERN_GLSL[name])


def _pat_scalar(em, node, name, fallback=0.0):
    sock = None
    for s in node.get('inputs', ()):
        if s.get('name') == name:
            sock = s
            break
    if sock is not None and sock.get('link'):
        raise Unsupported(f"the evaluator collapses the pattern's '{name}' "
                          'to its batch mean; a varying chain would render '
                          'differently here, so the material shades on the '
                          'CPU')
    return em.const((sock or {}).get('default', fallback), FLOAT)


def _pat_vec(em, node):
    v = tex_vector(em, node, 'generated')
    scale = em.input(node, 'Scale', FLOAT)
    p, _t = em.tmp(VEC3, f'{v} * {scale}')
    return p


def _pat_output(em, node, index, f):
    """Colour/Fac pair exactly as `_pat_out`: clip, then ramp the colours."""
    outs = node.get('outputs') or []
    o = outs[index] if index < len(outs) else {}
    fac, _t = em.tmp(FLOAT, f'clamp({f}, 0.0, 1.0)')
    if o.get('name') == 'Fac':
        return fac, FLOAT
    a, _t = em.tmp(VEC4, em.input(node, 'Color 1', VEC4))
    b, _t = em.tmp(VEC4, em.input(node, 'Color 2', VEC4))
    return em.tmp(VEC4, f'{a} + ({b} - {a}) * {fac}')


_AXIS = {'X': 0, 'Y': 1, 'Z': 2}


def e_pat_marble(em, node, index):
    _need_pattern(em, 'marble')
    p = _pat_vec(em, node)
    return _pat_output(em, node, index, (
        f'hal_pat_marble({p}, {_pat_scalar(em, node, "Turbulence", 1.0)}, '
        f'{int(prop(node, "octaves", 5))}, '
        f'{_pat_scalar(em, node, "Veins", 1.0)}, '
        f'{_pat_scalar(em, node, "Sharpness", 1.0)}, '
        f'{_AXIS.get(str(prop(node, "axis", "X")), 0)})'))


def e_pat_wood(em, node, index):
    _need_pattern(em, 'wood')
    p = _pat_vec(em, node)
    return _pat_output(em, node, index, (
        f'hal_pat_wood({p}, {_pat_scalar(em, node, "Rings", 8.0)}, '
        f'{_pat_scalar(em, node, "Turbulence", 0.35)}, '
        f'{int(prop(node, "octaves", 4))}, '
        f'{_pat_scalar(em, node, "Grain", 0.4)}, '
        f'{_AXIS.get(str(prop(node, "axis", "Z")), 2)})'))


def e_pat_granite(em, node, index):
    _need_pattern(em, 'granite')
    p = _pat_vec(em, node)
    return _pat_output(em, node, index, (
        f'hal_pat_granite({p}, {int(prop(node, "octaves", 6))}, '
        f'{_pat_scalar(em, node, "Contrast", 1.6)}, '
        f'{_pat_scalar(em, node, "Speckle", 0.35)})'))


def e_pat_dents(em, node, index):
    _need_pattern(em, 'dents')
    p = _pat_vec(em, node)
    return _pat_output(em, node, index, (
        f'hal_pat_dents({p}, {_pat_scalar(em, node, "Size", 1.0)}, '
        f'{int(prop(node, "octaves", 3))}, '
        f'{_pat_scalar(em, node, "Depth", 1.0)})'))


def e_pat_crackle(em, node, index):
    _need_pattern(em, 'crackle')
    p = _pat_vec(em, node)
    return _pat_output(em, node, index, (
        f'hal_pat_crackle({p}, {_pat_scalar(em, node, "Randomness", 1.0)}, '
        f'{_pat_scalar(em, node, "Width", 0.06)}, '
        f'{_pat_scalar(em, node, "Smooth", 0.02)})'))


def _pat_time(em, node, animated=None):
    """The pattern's clock: hal_time when animated, exactly ctx.time."""
    if animated is None:
        animated = bool(prop(node, 'animate', True))
    if not animated:
        return '0.0'
    em.frame_uniforms.add('hal_time')
    return 'hal_time'


def e_pat_plasma(em, node, index):
    _need_pattern(em, 'plasma')
    p = _pat_vec(em, node)
    t = _pat_time(em, node)
    speed = _pat_scalar(em, node, 'Speed', 1.0)
    comp = _pat_scalar(em, node, 'Complexity', 3.0)
    f, _t = em.tmp(FLOAT, f'hal_pat_plasma({p}, {t} * {speed}, {comp})')
    if prop(node, 'cycle_palette', True):
        outs = node.get('outputs') or []
        o = outs[index] if index < len(outs) else {}
        fac, _t = em.tmp(FLOAT, f'clamp({f}, 0.0, 1.0)')
        if o.get('name') == 'Fac':
            return fac, FLOAT
        # the palette-cycled rainbow that made plasma demos move; the phase
        # rides the RAW clock, not the speed-scaled one, as the evaluator
        ph, _t = em.tmp(FLOAT, f'{f} * 6.2832 + {t} * 0.6')
        return em.tmp(VEC4, f'vec4(sin({ph}) * 0.5 + 0.5, '
                            f'sin({ph} + 2.094) * 0.5 + 0.5, '
                            f'sin({ph} + 4.188) * 0.5 + 0.5, 1.0)')
    return _pat_output(em, node, index, f)


def e_pat_ripples(em, node, index):
    import numpy as np
    p = _pat_vec(em, node)           # pure sin/exp; needs no primitives
    t = _pat_time(em, node)
    speed = _pat_scalar(em, node, 'Speed', 1.0)
    freq = _pat_scalar(em, node, 'Frequency', 8.0)
    decay = _pat_scalar(em, node, 'Decay', 0.6)
    # the source positions come from a seeded generator the evaluator runs
    # every frame; being seeded, they are per-scene constants -- so they
    # bake, and the GLSL never needs the generator at all
    rng = np.random.default_rng(int(prop(node, 'seed', 0)) + 1234)
    n = max(int(prop(node, 'sources', 3)), 1)
    tt, _t = em.tmp(FLOAT, f'{t} * {speed} * 3.0')
    total, _t = em.tmp(FLOAT, '0.0')
    for _ in range(n):
        c = (rng.random(3).astype(np.float32) - 0.5) * 2.0
        d, _t = em.tmp(FLOAT, f'length({p} - {Emitter.const(c, VEC3)})')
        total, _t = em.tmp(FLOAT,
                           f'{total} + sin({d} * max({freq}, 1e-3) - {tt})'
                           f' * exp(-{d} * max({decay}, 0.0))')
    f, _t = em.tmp(FLOAT, f'{total} / {float(n)!r} * 0.5 + 0.5')
    return _pat_output(em, node, index, f)


def e_pat_starfield(em, node, index):
    _need_pattern(em, 'starfield')
    p = _pat_vec(em, node)
    em.frame_uniforms.add('hal_time')    # the evaluator always reads it
    f, _t = em.tmp(FLOAT, (
        f'hal_pat_starfield({p}, {_pat_scalar(em, node, "Density", 0.5)}, '
        f'{_pat_scalar(em, node, "Size", 0.35)}, '
        f'{_pat_scalar(em, node, "Twinkle", 0.0)}, hal_time)'))
    fac, _t = em.tmp(FLOAT, f'clamp({f}, 0.0, 1.0)')
    outs = node.get('outputs') or []
    o = outs[index] if index < len(outs) else {}
    if o.get('name') == 'Fac':
        return fac, FLOAT
    sky, _t = em.tmp(VEC4, em.input(node, 'Sky Color', VEC4))
    star, _t = em.tmp(VEC4, em.input(node, 'Star Color', VEC4))
    return em.tmp(VEC4, f'{sky} + ({star} - {sky}) * {fac}')


def e_pat_weave(em, node, index):
    _need_pattern(em, 'weave')
    p = _pat_vec(em, node)
    res, _t = em.tmp('vec2', (
        f'hal_pat_weave({p}, {_pat_scalar(em, node, "Thickness", 0.35)}, '
        f'{_pat_scalar(em, node, "Gap", 0.08)}, '
        f'{_pat_scalar(em, node, "Distortion", 0.0)})'))
    outs = node.get('outputs') or []
    o = outs[index] if index < len(outs) else {}
    if o.get('name') == 'Thread':
        return em.tmp(FLOAT, f'{res}.y')
    if o.get('name') == 'Fac':
        return em.tmp(FLOAT, f'{res}.x')
    warp_c, _t = em.tmp(VEC4, em.input(node, 'Warp Color', VEC4))
    weft_c, _t = em.tmp(VEC4, em.input(node, 'Weft Color', VEC4))
    base, _t = em.tmp(VEC4, f'(({res}.y > 0.5) ? {warp_c} : {weft_c})'
                            f' * {res}.x')
    return em.tmp(VEC4, f'vec4({base}.rgb, 1.0)')


def e_pat_scratches(em, node, index):
    import numpy as np
    _need_prims(em)                  # the jag reads value noise
    p = _pat_vec(em, node)
    width = float((_socket_default(node, 'Width', 0.02)))
    length = float((_socket_default(node, 'Length', 1.0)))
    aniso = float((_socket_default(node, 'Anisotropy', 1.0)))
    for name in ('Width', 'Length', 'Anisotropy'):
        _pat_scalar(em, node, name)      # linked scalars refuse, as always
    rng = np.random.default_rng(int(prop(node, 'seed', 0)) + 77)
    n = max(int(prop(node, 'count', 6)), 1)
    total, _t = em.tmp(FLOAT, '0.0')
    w2 = max(width * width, 1e-8)
    for _ in range(n):
        # the seeded angles and offsets are constants; the evaluator's own
        # cos/sin bake as literals, so both sides read identical numbers
        ang = rng.random() * np.pi * (1.0 - aniso)
        dx, dy = float(np.cos(ang)), float(np.sin(ang))
        offset = (rng.random() - 0.5) * 4.0
        proj, _t = em.tmp(FLOAT, f'{p}.x * {_c(-dy)} + {p}.y * {_c(dx)}'
                                 f' + {_c(offset)}')
        along, _t = em.tmp(FLOAT, f'{p}.x * {_c(dx)} + {p}.y * {_c(dy)}')
        band, _t = em.tmp(FLOAT, f'exp(-({proj} * {proj}) / {_c(w2)})')
        mask, _t = em.tmp(FLOAT, f'clamp(1.0 - abs({along}) / '
                                 f'{_c(max(length, 1e-3))}, 0.0, 1.0)')
        jag, _t = em.tmp(FLOAT, f'0.6 + 0.4 * hal_pt_vnoise('
                                f'vec3({along} * 24.0, 0.0, 0.0))')
        total, _t = em.tmp(FLOAT, f'max({total}, {band} * {mask} * {jag})')
    return _pat_output(em, node, index, total)


def _socket_default(node, name, fallback):
    for s in node.get('inputs', ()):
        if s.get('name') == name:
            try:
                return float(s.get('default') if s.get('default') is not None
                             else fallback)
            except (TypeError, ValueError):
                return fallback
    return fallback


def _c(v):
    return f'{float(v):.8g}' if any(ch in f'{float(v):.8g}' for ch in '.e') \
        else f'{float(v):.8g}.0'


def _pat_cells(em, node, index, res, cell_color, field_color, id_name):
    """Tiles and brick share their colour composite exactly."""
    outs = node.get('outputs') or []
    o = outs[index] if index < len(outs) else {}
    if o.get('name') == id_name:
        return em.tmp(FLOAT, f'{res}.y')
    if o.get('name') == 'Fac':
        return em.tmp(FLOAT, f'{res}.x')
    cc, _t = em.tmp(VEC4, em.input(node, cell_color, VEC4))
    fc, _t = em.tmp(VEC4, em.input(node, field_color, VEC4))
    vary = em.input(node, 'Variation', FLOAT)       # per-pixel, unmeaned
    shade, _t = em.tmp(FLOAT, f'1.0 + ({res}.y - 0.5) * {vary}')
    col, _t = em.tmp(VEC4, f'({res}.z > 0.5) '
                           f'? {cc} * ({res}.x * {shade}) : {fc}')
    return em.tmp(VEC4, f'vec4({col}.rgb, 1.0)')


def e_pat_tiles(em, node, index):
    _need_pattern(em, 'tiles')
    p = _pat_vec(em, node)
    res, _t = em.tmp(VEC3, (
        f'hal_pat_tiles({p}, {_pat_scalar(em, node, "Rows", 4.0)}, '
        f'{_pat_scalar(em, node, "Columns", 4.0)}, '
        f'{_pat_scalar(em, node, "Grout", 0.06)}, '
        f'{_pat_scalar(em, node, "Offset", 0.0)}, '
        f'{_pat_scalar(em, node, "Bevel", 0.15)})'))
    return _pat_cells(em, node, index, res, 'Tile Color', 'Grout Color',
                      'Tile ID')


def e_pat_brick(em, node, index):
    _need_pattern(em, 'brick')
    p = _pat_vec(em, node)
    res, _t = em.tmp(VEC3, (
        f'hal_pat_brick({p}, {_pat_scalar(em, node, "Width", 0.25)}, '
        f'{_pat_scalar(em, node, "Height", 0.125)}, '
        f'{_pat_scalar(em, node, "Mortar", 0.05)}, '
        f'{_pat_scalar(em, node, "Offset", 0.5)}, '
        f'{_pat_scalar(em, node, "Bevel", 0.12)})'))
    return _pat_cells(em, node, index, res, 'Brick Color', 'Mortar Color',
                      'Brick ID')


def e_pat_spiral(em, node, index):
    _need_pattern(em, 'spiral')
    p = _pat_vec(em, node)
    return _pat_output(em, node, index, (
        f'hal_pat_spiral({p}, {_pat_scalar(em, node, "Turns", 4.0)}, '
        f'{_pat_scalar(em, node, "Sharpness", 1.0)}, '
        f'{_AXIS.get(str(prop(node, "axis", "Z")), 2)}, '
        f'{_pat_scalar(em, node, "Twist", 0.0)})'))


def e_pat_bozo(em, node, index):
    _need_pattern(em, 'bozo')
    p = _pat_vec(em, node)
    return _pat_output(em, node, index, (
        f'hal_pat_bozo({p}, {_pat_scalar(em, node, "Turbulence", 0.0)}, '
        f'{int(prop(node, "octaves", 4))}, '
        f'{_pat_scalar(em, node, "Lacunarity", 2.0)})'))


def e_pat_agate(em, node, index):
    _need_pattern(em, 'agate')
    p = _pat_vec(em, node)
    return _pat_output(em, node, index, (
        f'hal_pat_agate({p}, {_pat_scalar(em, node, "Turbulence", 1.0)}, '
        f'{int(prop(node, "octaves", 6))}, '
        f'{_pat_scalar(em, node, "Bands", 1.1)}, '
        f'{_pat_scalar(em, node, "Sharpness", 0.77)}, '
        f'{_AXIS.get(str(prop(node, "axis", "Z")), 2)})'))


def e_pat_leopard(em, node, index):
    _need_pattern(em, 'leopard')
    p = _pat_vec(em, node)
    return _pat_output(em, node, index, (
        f'hal_pat_leopard({p}, {_pat_scalar(em, node, "Spot", 1.0)})'))


def e_pat_onion(em, node, index):
    _need_pattern(em, 'onion')
    p = _pat_vec(em, node)
    return _pat_output(em, node, index, (
        f'hal_pat_onion({p}, {_pat_scalar(em, node, "Thickness", 1.0)}, '
        f'{_pat_scalar(em, node, "Sharpness", 1.0)})'))


def e_pat_bumps(em, node, index):
    _need_pattern(em, 'bumps')
    p = _pat_vec(em, node)
    return _pat_output(em, node, index, (
        f'hal_pat_bumps({p}, {_pat_scalar(em, node, "Roundness", 1.0)}, '
        f'{int(prop(node, "octaves", 1))}, '
        f'{_pat_scalar(em, node, "Lacunarity", 2.0)}, '
        f'{_pat_scalar(em, node, "Gain", 0.5)})'))


def e_pat_wrinkles(em, node, index):
    _need_pattern(em, 'wrinkles')
    p = _pat_vec(em, node)
    return _pat_output(em, node, index, (
        f'hal_pat_wrinkles({p}, {int(prop(node, "octaves", 8))}, '
        f'{_pat_scalar(em, node, "Lacunarity", 2.0)}, '
        f'{_pat_scalar(em, node, "Crease", 1.0)})'))


def e_tex_gradient(em, node, index):
    v = tex_vector(em, node)
    p, _t = em.tmp(VEC3, v)
    t = str(prop(node, 'gradient_type', 'LINEAR'))
    if t == 'QUADRATIC':
        r, _t = em.tmp(FLOAT, f'max({p}.x, 0.0)')
        expr = f'{r} * {r}'
    elif t == 'EASING':
        r, _t = em.tmp(FLOAT, f'clamp({p}.x, 0.0, 1.0)')
        expr = f'{r} * {r} * (3.0 - 2.0 * {r})'
    elif t == 'DIAGONAL':
        expr = f'({p}.x + {p}.y) * 0.5'
    elif t == 'RADIAL':
        expr = f'atan({p}.y, {p}.x) / 6.28318530717959 + 0.5'
    elif t in ('QUADRATIC_SPHERE', 'SPHERICAL'):
        r, _t = em.tmp(FLOAT, f'max(1.0 - length({p}), 0.0)')
        expr = f'{r} * {r}' if t == 'QUADRATIC_SPHERE' else r
    else:
        expr = f'{p}.x'
    fac, _t = em.tmp(FLOAT, f'clamp({expr}, 0.0, 1.0)')
    outs = node.get('outputs') or []
    o = outs[index] if index < len(outs) else {}
    if o.get('name') == 'Fac':
        return fac, FLOAT
    return em.tmp(VEC4, f'vec4({fac}, {fac}, {fac}, 1.0)')


def e_tex_magic(em, node, index):
    v = tex_vector(em, node)
    scale = em.input(node, 'Scale', FLOAT)
    dist = em.input(node, 'Distortion', FLOAT)
    depth = int(prop(node, 'turbulence_depth', 2))
    p, _t = em.tmp(VEC3, f'{v} * {scale}')
    x, _t = em.tmp(FLOAT, f'sin(({p}.x + {p}.y + {p}.z) * 5.0)')
    y, _t = em.tmp(FLOAT, f'cos((-{p}.x + {p}.y - {p}.z) * 5.0)')
    z, _t = em.tmp(FLOAT, f'-cos((-{p}.x - {p}.y + {p}.z) * 5.0)')
    d, _t = em.tmp(FLOAT, dist)
    for _ in range(max(depth, 1)):
        x2, _t = em.tmp(FLOAT, f'sin({y} * {d}) * 0.5 + 0.5')
        y2, _t = em.tmp(FLOAT, f'cos({z} * {d}) * 0.5 + 0.5')
        z2, _t = em.tmp(FLOAT, f'-cos({x} * {d}) * 0.5 + 0.5')
        x, _t = em.tmp(FLOAT, f'{x2} * 2.0 - 1.0')
        y, _t = em.tmp(FLOAT, f'{y2} * 2.0 - 1.0')
        z, _t = em.tmp(FLOAT, f'{z2} * 2.0 - 1.0')
        d, _t = em.tmp(FLOAT, f'{d} * 0.85 + 0.15')
    col, _t = em.tmp(VEC4, f'vec4(0.5 - {x} * 0.5, 0.5 - {y} * 0.5, '
                           f'0.5 - {z} * 0.5, 1.0)')
    outs = node.get('outputs') or []
    o = outs[index] if index < len(outs) else {}
    if o.get('name') == 'Fac':
        return em.tmp(FLOAT, em.cast(col, VEC4, FLOAT))
    return col, VEC4


def e_tex_wave(em, node, index):
    for s in node.get('inputs', ()):
        if s.get('name') == 'Distortion':
            try:
                dv = float(s.get('default') or 0.0)
            except (TypeError, ValueError):
                dv = 0.0
            if s.get('link') or abs(dv) > 1e-9:
                raise Unsupported('Wave distortion runs on Blender-style '
                                  "Perlin, whose sin-fract hash a driver's "
                                  'float32 cannot reproduce; undistorted '
                                  'waves travel')
    v = tex_vector(em, node)
    scale = em.input(node, 'Scale', FLOAT)
    p, _t = em.tmp(VEC3, f'{v} * {scale}')
    wt = str(prop(node, 'wave_type', 'BANDS'))
    direction = str(prop(node, 'bands_direction', 'X'))
    if wt == 'RINGS':
        base = f'length({p})'
    elif direction in ('X', 'Y', 'Z'):
        base = f'{p}.{direction.lower()}'
    else:
        base = f'{p}.x + {p}.y + {p}.z'
    nn, _t = em.tmp(FLOAT, f'({base}) * 20.0')
    profile = str(prop(node, 'wave_profile', 'SIN'))
    if profile == 'SAW':
        expr = f'mod({nn} / 6.28318530717959, 1.0)'
    elif profile == 'TRI':
        expr = f'abs(mod({nn} / 3.14159265358979, 2.0) - 1.0)'
    else:
        expr = f'0.5 + 0.5 * sin({nn} - 1.5707963267949)'
    fac, _t = em.tmp(FLOAT, f'clamp({expr}, 0.0, 1.0)')
    outs = node.get('outputs') or []
    o = outs[index] if index < len(outs) else {}
    if o.get('name') == 'Fac':
        return fac, FLOAT
    return em.tmp(VEC4, f'vec4({fac}, {fac}, {fac}, 1.0)')


def e_matcap_uv(em, node, index):
    """Sphere-map coordinates from the view-space normal.

    The 1990s environment-reflection trick, exactly as
    `n_halcyon_matcap_uv`: a frame built from the view direction, the
    normal projected into it, one image carrying an entire material. The
    degenerate guard tests the raw cross product rather than normalising a
    near-zero vector first -- same degenerate set, and no NaN on the way.
    """
    n, _t = em.tmp(VEC3, 'normalize(hal_N)')
    outs = node.get('outputs') or []
    o = outs[index] if index < len(outs) else {}
    if o.get('name') == 'Facing':
        return em.tmp(FLOAT, f'clamp(dot({n}, hal_V), 0.0, 1.0)')
    r0, _t = em.tmp(VEC3, 'cross(vec3(0.0, 0.0, 1.0), hal_V)')
    right, _t = em.tmp(VEC3, f'(dot({r0}, {r0}) < 1e-8) '
                             f'? vec3(1.0, 0.0, 0.0) : normalize({r0})')
    upv, _t = em.tmp(VEC3, f'cross(hal_V, {right})')
    scale, _t = em.tmp(FLOAT, em.input(node, 'Scale', FLOAT))
    return em.tmp(VEC3, f'vec3(dot({n}, {right}) * 0.5 * {scale} + 0.5, '
                        f'dot({n}, {upv}) * 0.5 * {scale} + 0.5, 0.0)')


def e_normal_map(em, node, _i):
    """Blender's Normal Map node, exactly as `nodeeval.n_normal_map`.

    The tangent frame is the renderer's own: the CPU never carries a UV
    tangent (`ctx.T` is never set anywhere), so the evaluator builds one
    from the geometric normal with `orthonormal_basis` -- deterministic,
    and therefore exactly reproducible from the G-buffer. OBJECT and WORLD
    space read the colour as a world-space normal (the evaluator draws no
    distinction), anything else takes the tangent construction, and both
    end on the same strength lerp toward the geometric normal.
    """
    col = em.input(node, 'Color', VEC4)
    strength = em.input(node, 'Strength', FLOAT)
    cv, _t = em.tmp(VEC4, col)
    tn, _t = em.tmp(VEC3, f'{cv}.rgb * 2.0 - 1.0')
    n, _t = em.tmp(VEC3, 'normalize(hal_N)')
    if str(prop(node, 'space', 'TANGENT')) in ('OBJECT', 'WORLD'):
        out, _t = em.tmp(VEC3, f'normalize({tn})')
    else:
        # mathx.orthonormal_basis, inlined: the same up-vector selection
        # the frame shader uses for s.tangent
        up, _t = em.tmp(VEC3, f'(abs({n}.z) < 0.999) '
                              f'? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0)')
        t, _t = em.tmp(VEC3, f'normalize(cross({up}, {n}))')
        b, _t = em.tmp(VEC3, f'cross({n}, {t})')
        out, _t = em.tmp(VEC3, f'normalize({t} * {tn}.x + {b} * {tn}.y '
                               f'+ {n} * {tn}.z)')
    return em.tmp(VEC3, f'normalize({n} + ({out} - {n}) * {strength})')


def e_bump(em, node, _i):
    """Blender's Bump node, exactly as `nodeeval.n_bump`.

    The CPU scatters the HEIGHT chain into the frame grid and takes
    one-sided differences toward the +x and +y neighbour pixels, gated on
    the neighbour being shaded by the same batch. The deferred pass
    reproduces that with a HEIGHT PRE-PASS: the chain renders to its own
    target over the same ids texture (alpha = the material's keep), and
    this emitter fetches the three texels by INTEGER coordinate --
    texelFetch, exact by specification, with explicit edge guards because
    out-of-bounds texelFetch is undefined on a driver -- then bends the
    normal with the CPU's own formula.

    Secondary (reflection-hit) passes emit the pass-through instead:
    `trace()` shades hits with ctx.px None, and `n_bump` returns its
    Normal input untouched there. Ray CONSTRUCTION for a bump material
    still bends: `_ray_context` runs this very evaluator code on the CPU
    for the ray pixels.
    """
    sock = next((s for s in node.get('inputs', ())
                 if s.get('name') == 'Normal'), None)
    nrm = em.input(node, 'Normal', VEC3) if (sock and sock.get('link')) \
        else 'hal_N'
    if em.secondary:
        # ctx.px is None on hit shading: the node is a wire
        return em.tmp(VEC3, nrm)
    if not em.frame_mode or getattr(em, 'resolution', None) is None:
        raise Unsupported("the Bump node's height pre-pass exists only in "
                          'the deferred frame')
    strength = em.input(node, 'Strength', FLOAT)
    dist = em.input(node, 'Distance', FLOAT)
    k = len(em.bump_passes)
    em.bump_passes.append(node)
    u = f'hal_bump{k}'
    em.inline.append(f'uniform sampler2D {u};')
    w, h = float(em.resolution[0]), float(em.resolution[1])
    wi, hi = int(w), int(h)
    pix, _t = em.tmp('ivec2', f'ivec2(int(vUV.x * {_c(w)}), '
                              f'int(vUV.y * {_c(h)}))')
    h0, _t = em.tmp(VEC4, f'texelFetch({u}, {pix}, 0)')
    hr, _t = em.tmp(VEC4, f'({pix}.x + 1 < {wi}) ? '
                          f'texelFetch({u}, {pix} + ivec2(1, 0), 0) '
                          f': vec4(0.0)')
    hu, _t = em.tmp(VEC4, f'({pix}.y + 1 < {hi}) ? '
                          f'texelFetch({u}, {pix} + ivec2(0, 1), 0) '
                          f': vec4(0.0)')
    dhdx, _t = em.tmp(FLOAT, f'({hr}.a > 0.5) ? ({hr}.r - {h0}.r) : 0.0')
    dhdy, _t = em.tmp(FLOAT, f'({hu}.a > 0.5) ? ({hu}.r - {h0}.r) : 0.0')
    n0, _t = em.tmp(VEC3, f'normalize({nrm})')
    up, _t = em.tmp(VEC3, f'(abs({n0}.z) < 0.999) '
                          f'? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0)')
    t, _t = em.tmp(VEC3, f'normalize(cross({up}, {n0}))')
    b, _t = em.tmp(VEC3, f'cross({n0}, {t})')
    inv = '-1.0' if prop(node, 'invert', False) else '1.0'
    return em.tmp(VEC3, f'normalize({n0} - ({inv} * {strength} * {dist}) * '
                        f'({t} * {dhdx} + {b} * {dhdy}) * 20.0)')


#: node types that will not get an emitter as things stand, and why. Not the
#: same thing as "nobody wrote one yet": these are refused for a reason worth
#: naming, so the fallback message says it instead of just the node's name.
REFUSED = {
    # the Blender-noise family rides fract(sin(x)*43758.5453), evaluated in
    # float64 on the CPU. A driver's float32 sin decorrelates completely
    # after that amplification, so the GPU would render a DIFFERENT pattern
    # -- the worst outcome there is. Halcyon's own pattern textures ride an
    # integer hash and travel exactly; use those.
    'ShaderNodeTexNoise': 'its Perlin rides a sin-fract hash a driver\'s '
                          'float32 sin decorrelates; Halcyon\'s pattern '
                          'textures use an integer hash and do travel',
    'ShaderNodeTexWhiteNoise': 'the same sin-fract hash as the Noise '
                               'texture',
    'ShaderNodeTexVoronoi': 'its cell jitter is the same sin-fract hash',
    'ShaderNodeTexMusgrave': 'its fractal is the same sin-fract Perlin',
    'ShaderNodeTexBrick': 'its per-brick tint is the same sin-fract hash',
}

EMITTERS = {
    'HALCYON_ShaderNode': e_halcyon_shader,
    'HALCYON_CodeNode': e_code_node,
    'HALCYON_MatcapUVNode': e_matcap_uv,
    'ShaderNodeVertexColor': e_vertex_color,
    'ShaderNodeNormalMap': e_normal_map,
    'ShaderNodeBump': e_bump,
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
    'ShaderNodeMapping': e_mapping,
    'ShaderNodeVectorMath': e_vector_math,
    'ShaderNodeInvert': e_invert,
    'ShaderNodeGamma': e_gamma,
    'ShaderNodeBrightContrast': e_bright_contrast,
    'ShaderNodeSeparateXYZ': e_separate_xyz,
    'ShaderNodeCombineXYZ': e_combine_xyz,
    'ShaderNodeSeparateRGB': e_separate_rgb,
    'ShaderNodeCombineRGB': e_combine_rgb,
    'ShaderNodeTexChecker': e_checker,
    'ShaderNodeTexGradient': e_tex_gradient,
    'ShaderNodeTexMagic': e_tex_magic,
    'ShaderNodeTexWave': e_tex_wave,
    'HALCYON_MarbleNode': e_pat_marble,
    'HALCYON_WoodNode': e_pat_wood,
    'HALCYON_GraniteNode': e_pat_granite,
    'HALCYON_DentsNode': e_pat_dents,
    'HALCYON_CrackleNode': e_pat_crackle,
    'HALCYON_PlasmaNode': e_pat_plasma,
    'HALCYON_RipplesNode': e_pat_ripples,
    'HALCYON_StarfieldNode': e_pat_starfield,
    'HALCYON_WeaveNode': e_pat_weave,
    'HALCYON_ScratchesNode': e_pat_scratches,
    'HALCYON_TilesNode': e_pat_tiles,
    'HALCYON_BrickNode': e_pat_brick,
    'HALCYON_SpiralNode': e_pat_spiral,
    'HALCYON_BozoNode': e_pat_bozo,
    'HALCYON_AgateNode': e_pat_agate,
    'HALCYON_LeopardNode': e_pat_leopard,
    'HALCYON_OnionNode': e_pat_onion,
    'HALCYON_BumpsNode': e_pat_bumps,
    'HALCYON_WrinklesNode': e_pat_wrinkles,
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
