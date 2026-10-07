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

FLOAT, VEC2, VEC3, VEC4 = 'float', 'vec2', 'vec3', 'vec4'

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
        self.camera = None         # camera TYPE ('PERSP'/'ORTHO'), stamped
        #                            by the assemblers; None refuses the
        #                            perspective-only answers rather than
        #                            guessing them right

        self.uv_names = ()         # the mesh's named UV maps, layer order
        self.color_name = ''       # R246: the colour layer the G-buffer
        #                            carries, by name
        self.has_vcol = False      # the mesh HAS a colour layer: BI
        #                            stripped vertex-colour modes when
        #                            none existed (convertblender.c),
        #                            and the ones-filled default plane
        #                            must never read as painted white
        self.used_screen = False   # a code node read vScreenUV/iResolution
        self.bump_passes = []      # Bump nodes needing a height pre-pass
        self.mark_values = False   # R174: wrap graph constants in hal_MV()
        #                            markers so the material-texel lifter
        #                            can pull them out of the source; set
        #                            ONLY by assemble_frame, so height
        #                            passes, worlds and every other
        #                            Emitter user emit plain literals
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

    def const(self, value, gtype):
        m = self._m
        if gtype == FLOAT:
            try:
                v = float(value[0]) if hasattr(value, '__len__') else float(value)
            except (TypeError, ValueError):
                v = 0.0
            return m(v)
        seq = list(value) if hasattr(value, '__len__') else [float(value)] * 3
        while len(seq) < 4:
            seq.append(1.0)
        if gtype == VEC3:
            return 'vec3({}, {}, {})'.format(*map(m, seq[:3]))
        return 'vec4({}, {}, {}, {})'.format(*map(m, seq[:4]))

    def _m(self, v):
        """One float, marked as a liftable VALUE when the flag is up.

        Graph constants -- socket defaults, RGB/Value nodes, computed
        node centres -- are exactly what differs between two materials
        with the same STRUCTURE. Marked, the material-texel lifter pulls
        them into the hal_mats texture and identical skeletons share one
        compiled shader. Non-finite floats stay literal (structure)."""
        try:
            f = float(v)
        except (TypeError, ValueError):
            f = 0.0
        s = f'{f:.8g}'
        if self.mark_values and f == f and abs(f) != float('inf'):
            return f'hal_MV({s})'
        return s

    def input(self, node, name, gtype):
        """The GLSL expression for one input socket, linked or defaulted.

        Matches the display name or the IDENTIFIER, exactly as the
        evaluator's ev.input does -- the Mix node's three data types
        share display names ('A' three times over) and only the
        identifier ('A_Color') is unambiguous."""
        for sock in node.get('inputs', ()):
            if sock.get('name') != name and sock.get('identifier') != name:
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


def _mix_expr_table(ac, bc, f):
    """The MixRGB blend modes as GLSL, mirroring _mix_blend exactly.

    Shared by the legacy MixRGB node and the modern Mix node's RGBA
    mode -- one table, one parity surface."""
    return {
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
        # per-component select, exactly np.where(ac < 0.5, lo, hi):
        # step(0.5, ac) is 1 at ac >= 0.5, so mix(lo, hi, sel) lands the
        # same side of the boundary the CPU does, 0.5 included
        'OVERLAY': f'{ac} + (mix(2.0 * {ac} * {bc}, '
                   f'vec3(1.0) - 2.0 * (vec3(1.0) - {ac}) '
                   f'* (vec3(1.0) - {bc}), '
                   f'step(vec3(0.5), {ac})) - {ac}) * {f}',
    }


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
    expr = _mix_expr_table(f'{av}.rgb', f'{bv}.rgb', f).get(op)
    if expr is None:
        raise Unsupported(f'MixRGB {op}')
    rgb, _t = em.tmp(VEC3, expr)
    if prop(node, 'use_clamp', False):
        rgb, _t = em.tmp(VEC3, f'clamp({rgb}, 0.0, 1.0)')
    return em.tmp(VEC4, f'vec4({rgb}, {av}.a)')


def e_mix(em, node, _i):
    """The modern Mix node: FLOAT and VECTOR lerp, RGBA blends.

    Socket resolution goes by IDENTIFIER ('A_Color') through the same
    helper the evaluator uses -- the node's three data types share
    display names, and plain 'A' silently reads the FLOAT socket.
    """
    from ..core.nodeeval import mix_socket
    dtype = str(prop(node, 'data_type', 'RGBA')).upper()
    op = str(prop(node, 'blend_type', 'MIX')).upper()
    fac = em.input(node, mix_socket(node, 'Factor', 'FLOAT'), FLOAT)
    if prop(node, 'clamp_factor', True):
        f, _t = em.tmp(FLOAT, f'clamp({fac}, 0.0, 1.0)')
    else:
        f, _t = em.tmp(FLOAT, fac)
    if dtype == 'FLOAT':
        a = em.input(node, mix_socket(node, 'A', dtype), FLOAT)
        b = em.input(node, mix_socket(node, 'B', dtype), FLOAT)
        av, _t = em.tmp(FLOAT, a)
        bv, _t = em.tmp(FLOAT, b)
        return em.tmp(FLOAT, f'({av} + ({bv} - {av}) * {f})')
    if dtype == 'VECTOR':
        a = em.input(node, mix_socket(node, 'A', dtype), VEC3)
        b = em.input(node, mix_socket(node, 'B', dtype), VEC3)
        av, _t = em.tmp(VEC3, a)
        bv, _t = em.tmp(VEC3, b)
        return em.tmp(VEC3, f'({av} + ({bv} - {av}) * {f})')
    a = em.input(node, mix_socket(node, 'A', dtype), VEC4)
    b = em.input(node, mix_socket(node, 'B', dtype), VEC4)
    av, _t = em.tmp(VEC4, a)
    bv, _t = em.tmp(VEC4, b)
    expr = _mix_expr_table(f'{av}.rgb', f'{bv}.rgb', f).get(op)
    if expr is None:
        raise Unsupported(f'Mix {op}')
    rgb, _t = em.tmp(VEC3, expr)
    if prop(node, 'clamp_result', False):
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


def e_separate_color(em, node, index):
    """R246: Blender 4/5's Separate Color -- the only separate node
    those versions offer -- exactly n_separate_color: the RGB channels,
    or hal_rgb2hsv's HSV for the HSV/HSL modes (the CPU reads both as
    HSV), the alpha as a fourth output where a tree asks for it."""
    c, _t = em.tmp(VEC4, em.input(node, 'Color', VEC4))
    if index >= 3:
        return em.tmp(FLOAT, f'{c}.a')
    mode = str(prop(node, 'mode', 'RGB'))
    if mode in ('HSV', 'HSL'):
        h, _t = em.tmp(VEC3, f'hal_rgb2hsv({c}.rgb)')
        return em.tmp(FLOAT, f'{h}.{"xyz"[index]}')
    return em.tmp(FLOAT, f'{c}.{"rgb"[index]}')


def e_combine_color(em, node, _i):
    """R246: the Combine Color twin of n_combine_color."""
    r = em.input(node, 'Red', FLOAT)
    g = em.input(node, 'Green', FLOAT)
    b = em.input(node, 'Blue', FLOAT)
    mode = str(prop(node, 'mode', 'RGB'))
    if mode in ('HSV', 'HSL'):
        rgb, _t = em.tmp(VEC3, f'hal_hsv2rgb(vec3({r}, {g}, {b}))')
        return em.tmp(VEC4, f'vec4({rgb}, 1.0)')
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
    # no image assigned: the CPU (n_tex_image) returns opaque black --
    # emit exactly that, and register NO sampler. Refusing here sent the
    # WHOLE frame plan to the CPU over a node that renders a constant.
    if not prop(node, 'image'):
        if index == 1:
            return em.tmp(FLOAT, '1.0')
        return em.tmp(VEC4, 'vec4(0.0, 0.0, 0.0, 1.0)')
    # an image texture defaults to UV, not generated -- the two are the same
    # on a unit cube and completely different on anything else
    v = tex_vector(em, node, 'uv')
    name = f'hal_tex{len(em.samplers)}'
    vec_linked = any(s.get('name') == 'Vector' and s.get('link')
                     for s in node.get('inputs', ()))
    em.samplers.append({
        'uniform': name,
        'image': prop(node, 'image'),
        'interpolation': prop(node, 'interpolation', 'Linear'),
        'extension': prop(node, 'extension', 'REPEAT'),
        # the mip footprint applies exactly where the CPU applies it: a
        # RAW flat-projection UV lookup. A linked Vector chain (or a
        # sphere/tube/box projection) resamples through a transform the
        # chain rule was never applied to, and keeps the top level
        'footprint': (not vec_linked
                      and str(prop(node, 'projection', 'FLAT')) == 'FLAT'),
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
    # Generated / Normal / UV / Object / Camera / Window / Reflection --
    # each the CPU's own n_tex_coord answer:
    #   Object: the object's own frame -- R243: hal_object, the world
    #           position through the per-object inverse matrix the
    #           material bakes as a lookup by object index (exactly
    #           n_tex_coord's einsum); before R243 the GPU answered
    #           world P here, a silent split on any moved object.
    #   Camera: c.P - camera_pos, NOT hal_P (the old emitter's silent
    #           wrong answer).
    #   Window: (px+0.5)/size -- the fullscreen pass's own vUV is that
    #           very number. A HIT context has no screen pixel (px is
    #           None -> zeros on the CPU), so secondary passes emit
    #           zeros rather than the hit's screen position.
    # Dispatch by the serialized output NAME first: the CPU evaluator
    # resolves links by name (n_tex_coord returns a name-keyed dict), so
    # the GPU must read the SAME source or the two devices shade
    # different pictures. Position is only the fallback for a nameless
    # graph.
    outs = node.get('outputs') or ()
    name = ''
    if index < len(outs):
        name = str(outs[index].get('name') or '').strip().lower()
    if not name:
        name = ['generated', 'normal', 'uv', 'object', 'camera', 'window',
                'reflection'][min(index, 6)]
    if name == 'camera':
        return em.tmp(VEC3, '(hal_P - hal_eye)')
    if name == 'window':
        if em.secondary:
            return em.tmp(VEC3, 'vec3(0.0)')
        return em.tmp(VEC3, 'vec3(vUV, 0.0)')
    table = {'generated': 'hal_generated', 'normal': 'hal_N',
             'uv': 'vec3(hal_uv, 0.0)', 'object': 'hal_object',
             'reflection': 'reflect(-hal_V, hal_N)'}
    return em.tmp(VEC3, table.get(name, 'hal_generated'))


def e_uvmap(em, node, _i):
    # the CPU resolves the LAYER NAME: ctx.attributes['uv:'+name],
    # falling back to the active layer when the name is empty or
    # unknown. Two UV sets travel (the period dual-texture budget);
    # the second rides the attribute texture's spare half.
    name = str(prop(node, 'uv_map', '') or '')
    if not name or not em.uv_names:
        return em.tmp(VEC3, 'vec3(hal_uv, 0.0)')
    if name == em.uv_names[0]:
        return em.tmp(VEC3, 'vec3(hal_uv, 0.0)')
    if len(em.uv_names) > 1 and name == em.uv_names[1]:
        return em.tmp(VEC3, 'vec3(hal_uv2, 0.0)')
    # an unknown name falls back to the active layer on the CPU too
    return em.tmp(VEC3, 'vec3(hal_uv, 0.0)')


def e_new_geometry(em, node, index):
    """Blender's Geometry node, output by output against `n_geometry`.

    This emitter's first life mapped outputs BY INDEX from a hand-written
    list, and the list was wrong: Tangent emitted the NORMAL, Incoming
    emitted the POSITION, Parametric emitted GENERATED coordinates, and
    Backfacing emitted a constant 0.0 whatever the winding -- four silent
    CPU/GPU divergences that survived because no test ever read those
    outputs (the R80 lesson: matrix blindness for untested sockets).
    Outputs resolve BY NAME now, each one either the CPU's exact
    expression or a refusal that says why:

    - Tangent: ctx.T is never filled, so the CPU always builds
      `orthonormal_basis(ctx.N)[0]` -- emitted verbatim.
    - True Normal: the face normal is STORED per triangle on the CPU
      (Blender's own, through the normal matrix); recomputing from
      corners flips on mirrored objects, so it is never recomputed --
      the hal_triaux texel carries the CPU's own normalized values,
      fetched by triangle id (frame passes only).
    - Incoming: the CPU returns -normalize(I) = normalize(eye - P),
      which IS hal_V.
    - Parametric: the CPU returns (uv, 0).
    - Backfacing: the rasteriser decides by projected winding; for a
      PERSPECTIVE camera that is exactly the plane-side test against the
      eye (the measured convention the backface override pinned).
      Orthographic refuses by name; secondary passes emit 0.0, because
      `trace()` shades hits with front=None and the CPU keeps zeros.
    - Random Per Island: the CPU's per-tri random is the sin-fract hash
      a driver would decorrelate -- so the CPU's own values bake into
      the hal_triaux texel's alpha and are fetched, never recomputed
      (frame passes only).
    """
    outs = node.get('outputs') or []
    o = outs[index] if index < len(outs) else {}
    name = o.get('name')
    if name == 'Position':
        return em.tmp(VEC3, 'hal_P')
    if name == 'Normal':
        return em.tmp(VEC3, 'hal_N')
    if name == 'Tangent':
        n0, _t = em.tmp(VEC3, 'normalize(hal_N)')
        up, _t = em.tmp(VEC3, f'(abs({n0}.z) < 0.999) '
                              f'? vec3(0.0, 0.0, 1.0) '
                              f': vec3(1.0, 0.0, 0.0)')
        return em.tmp(VEC3, f'normalize(cross({up}, {n0}))')
    if name == 'Incoming':
        return em.tmp(VEC3, 'hal_V')
    if name == 'Parametric':
        return em.tmp(VEC3, 'vec3(hal_uv, 0.0)')
    if name == 'Backfacing':
        if em.secondary:
            # trace() builds its context with front=None; ctx.backfacing
            # stays zeros on every hit, and this matches it
            return em.tmp(FLOAT, '0.0')
        if not em.frame_mode:
            raise Unsupported('backfacing exists only in the rasterised '
                              'frame')
        if str(em.camera or '').upper() != 'PERSP':
            raise Unsupported('backfacing under an orthographic camera '
                              'is not in the deferred pass -- the '
                              'plane-side test is the perspective '
                              'answer; the material shades on the CPU')
        p0, _t = em.tmp(VEC3, 'hal_fetch_attr(f.tri, 0, 0).xyz')
        pl, _t = em.tmp(VEC3, f'cross(hal_fetch_attr(f.tri, 1, 0).xyz '
                              f'- {p0}, hal_fetch_attr(f.tri, 2, 0).xyz '
                              f'- {p0})')
        return em.tmp(FLOAT, f'(dot({pl}, hal_eye - {p0}) < 0.0) '
                             f'? 0.0 : 1.0')
    if name == 'Pointiness':
        return em.tmp(FLOAT, '0.0')      # zeros on the CPU too
    if name == 'Random Per Island':
        # the CPU's own per-tri sin-fract values ride the hal_triaux
        # texel's alpha (gbuffer.pack_tri_aux bakes _hash1 of the tri
        # index) -- fetched, never recomputed, so no driver hash can
        # decorrelate them. Frame passes only: the fetch needs f.tri.
        if not em.frame_mode:
            raise Unsupported('the per-triangle random needs the '
                              'G-buffer triangle id of the deferred '
                              'pass')
        em.needs_triaux = True
        return em.tmp(FLOAT, 'hal_triaux_fetch(max(f.tri, 0.0)).w')
    if name == 'True Normal':
        # the STORED face normal rides the hal_triaux texel's rgb --
        # the CPU's own normalize(mesh.face_normals), same bits, the
        # exact values ctx.Ng carries. Frame passes only: f.tri again.
        if not em.frame_mode:
            raise Unsupported('the stored face normal needs the '
                              'G-buffer triangle id of the deferred '
                              'pass')
        em.needs_triaux = True
        return em.tmp(VEC3, 'hal_triaux_fetch(max(f.tri, 0.0)).xyz')
    raise Unsupported(f'Geometry output {name!r} is not in the deferred '
                      'pass')


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


def e_wireframe(em, node, _i):
    """The Wireframe node: exact edge distance as a 0/1 factor.

    ShadeJob.wire_fields, expression for expression: the world-space
    point-to-edge distance on the fragment's own triangle (corners from
    the attribute texture, P re-interpolated the CPU's way), and -- for
    Pixel Size -- the world-units-per-pixel scale from the per-corner
    screen positions the hal_vscreen texture carries (the CPU's own
    projected sx, sy, w, baked). Works in frame AND secondary passes:
    a hit has a triangle identity too, exactly as ctx.wire_fields does.
    """
    if not em.frame_mode:
        raise Unsupported('the wireframe ink needs a triangle identity '
                          'only the deferred passes carry')
    size = em.input(node, 'Size', FLOAT)
    a, _t = em.tmp(VEC3, 'hal_fetch_attr(max(f.tri, 0.0), 0, 0).xyz')
    b, _t = em.tmp(VEC3, 'hal_fetch_attr(max(f.tri, 0.0), 1, 0).xyz')
    c, _t = em.tmp(VEC3, 'hal_fetch_attr(max(f.tri, 0.0), 2, 0).xyz')
    p, _t = em.tmp(VEC3, f'{a} * f.bary.x + {b} * f.bary.y '
                         f'+ {c} * f.bary.z')

    def _edge(A, B):
        E, _e = em.tmp(VEC3, f'{B} - {A}')
        L2, _e = em.tmp(FLOAT, f'max(dot({E}, {E}), 1e-12)')
        X, _e = em.tmp(VEC3, f'cross({p} - {A}, {E})')
        d, _e = em.tmp(FLOAT, f'sqrt(max(dot({X}, {X}), 0.0) / {L2})')
        return d

    d0 = _edge(a, b)
    d1 = _edge(b, c)
    d2 = _edge(c, a)
    dmin, _t = em.tmp(FLOAT, f'min(min({d0}, {d1}), {d2})')
    half, _t = em.tmp(FLOAT, f'max({size}, 0.0) * 0.5')
    if prop(node, 'use_pixel_size', False):
        em.needs_wirescreen = True
        s0, _t = em.tmp(VEC4, 'hal_vscreen_fetch(max(f.tri, 0.0) * 3.0)')
        s1, _t = em.tmp(VEC4,
                        'hal_vscreen_fetch(max(f.tri, 0.0) * 3.0 + 1.0)')
        s2, _t = em.tmp(VEC4,
                        'hal_vscreen_fetch(max(f.tri, 0.0) * 3.0 + 2.0)')
        e1, _t = em.tmp(VEC2, f'{s1}.xy - {s0}.xy')
        e2, _t = em.tmp(VEC2, f'{s2}.xy - {s0}.xy')
        Dd, _t = em.tmp(FLOAT, f'{e1}.x * {e2}.y - {e1}.y * {e2}.x')
        Dc, _t = em.tmp(FLOAT, f'(abs({Dd}) < 1e-9) ? 1e-9 : {Dd}')
        g1, _t = em.tmp(VEC2, f'vec2({e2}.y, -{e2}.x) / {Dc}')
        g2, _t = em.tmp(VEC2, f'vec2(-{e1}.y, {e1}.x) / {Dc}')
        g0, _t = em.tmp(VEC2, f'-{g1} - {g2}')
        Wp, _t = em.tmp(FLOAT, f'f.bary.x * {s0}.z + f.bary.y * {s1}.z '
                               f'+ f.bary.z * {s2}.z')
        f0, _t = em.tmp(VEC2, f'({g0} / {s0}.z) * {Wp}')
        f1, _t = em.tmp(VEC2, f'({g1} / {s1}.z) * {Wp}')
        f2, _t = em.tmp(VEC2, f'({g2} / {s2}.z) * {Wp}')
        dpx, _t = em.tmp(VEC3, f'({a} - {p}) * {f0}.x + ({b} - {p}) '
                               f'* {f1}.x + ({c} - {p}) * {f2}.x')
        dpy, _t = em.tmp(VEC3, f'({a} - {p}) * {f0}.y + ({b} - {p}) '
                               f'* {f1}.y + ({c} - {p}) * {f2}.y')
        wpp, _t = em.tmp(FLOAT, f'0.5 * (sqrt(dot({dpx}, {dpx})) '
                                f'+ sqrt(dot({dpy}, {dpy})))')
        half, _t = em.tmp(FLOAT, f'{half} * {wpp}')
    return em.tmp(FLOAT, f'({dmin} <= {half}) ? 1.0 : 0.0')


def e_bsdf_metallic(em, node, _i):
    # the frame probe harvests the surface parameters through the closure;
    # what the emitter owes is the per-pixel colour chain, same as glossy
    return em.tmp(VEC4, em.input(node, 'Base Color', VEC4))


def e_bsdf_specular(em, node, _i):
    return em.tmp(VEC4, em.input(node, 'Base Color', VEC4))


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


def _console_lit(x):
    """R252: a float literal a strict GLSL front-end reads as a float
    (%.9g, the float32 round-trip, with a '.0' where it has none)."""
    t = f'{float(x):.9g}'
    if 'e' not in t and 'E' not in t and '.' not in t and 'inf' not in t \
            and 'nan' not in t:
        t += '.0'
    return t


def e_console_shader(em, node, _i):
    """R252: the Console Emulation Shader, as the deferred pass needs it:
    its colour -- the Diffuse Color chain, the vertex colour blended
    over it (as the master does it) or REPLACING it where the type makes
    the vertex colour the material (GX_SRC_VTX, Model 3 fixed shading,
    the N64 with lighting off, D3D's colour vertex), and RenderWare's
    dual-texture pass over the base: core/nodeeval.console_albedo's
    arithmetic, written out. Every other socket is a surface parameter
    the probe harvests (baked when constant, granted per pixel by the
    tables, refused by name otherwise)."""
    from ..core.console import resolve
    res = resolve(node.get('props', {}))
    vmix_on = res['vmix'] is not None
    vcol_linked = False
    dual_linked = False
    vmix_default = 0.0
    for sock in node.get('inputs', ()):
        name = sock.get('name')
        if name == 'Vertex Color Mix':
            try:
                vmix_default = float(sock.get('default') or 0.0)
            except (TypeError, ValueError):
                vmix_default = 0.0
            if bool(sock.get('link')) or vmix_default > 1e-6:
                vmix_on = True
        if name == 'Vertex Color':
            vcol_linked = bool(sock.get('link'))
        if name == 'Dual Texture':
            dual_linked = bool(sock.get('link'))
    if res['untextured']:
        # G_CC_SHADE: the socket's flat colour, the chain dropped
        # (console_albedo's own read)
        sock = next((sk for sk in node.get('inputs', ())
                     if sk.get('name') == 'Diffuse Color'), None)
        dv = list((sock or {}).get('default') or (0.8, 0.8, 0.8, 1.0))
        dv = (dv + [1.0, 1.0, 1.0, 1.0])[:4]
        base, _t = em.tmp(VEC4, 'vec4(' + ', '.join(_console_lit(v)
                                                     for v in dv) + ')')
    else:
        base, _t = em.tmp(VEC4, em.input(node, 'Diffuse Color', VEC4))
    if vmix_on:
        if res['vmix'] is not None:
            vmix = _console_lit(res['vmix'])
        else:
            vmix, _t = em.tmp(FLOAT,
                              f'clamp({em.input(node, "Vertex Color Mix", FLOAT)}'
                              f', 0.0, 1.0)')
        # the G-buffer's colour layer, or the socket (linked, or the
        # mesh unpainted -- console_albedo's own fallback)
        vcol = em.input(node, 'Vertex Color', VEC4) \
            if (vcol_linked or not em.has_vcol) else 'hal_vcol'
        base, _t = em.tmp(VEC4, f'{base} + ({vcol} - {base}) * {vmix}')
    cmb = res['combine']
    if cmb.get('rw_matfx') == 'DUAL' and dual_linked:
        dual, _t = em.tmp(VEC4, em.input(node, 'Dual Texture', VEC4))
        mode = str(cmb.get('rw_dual', 'MODULATE'))
        if mode == 'ADD':
            rgb = f'min({base}.rgb + {dual}.rgb, vec3(1.0))'
        elif mode == 'ALPHA':
            rgb = (f'{base}.rgb * (1.0 - clamp({dual}.a, 0.0, 1.0)) '
                   f'+ {dual}.rgb * clamp({dual}.a, 0.0, 1.0)')
        else:
            rgb = f'{base}.rgb * {dual}.rgb'
        base, _t = em.tmp(VEC4, f'vec4({rgb}, {base}.a)')
    return base, VEC4


def e_anime_shader(em, node, _i):
    """The anime master (R218), as the deferred pass needs it: its
    colour -- base times the Line Art chain, times the ILM's drawn
    line channel in the ArcSys-family modes. Every other socket is a
    surface parameter the probe harvests (baked when constant, granted
    per pixel by the tables, refused by name otherwise)."""
    base, _t = em.tmp(VEC4, em.input(node, 'Diffuse Color', VEC4))
    line, _t = em.tmp(VEC4, em.input(node, 'Line Art', VEC4))
    mode = str(prop(node, 'compat', 'GENERIC'))
    out, _t = em.tmp(VEC4, f'vec4({base}.rgb * {line}.rgb, {base}.a)')
    if mode in ('ARCSYS', 'DBFZ', 'KAKAROT'):
        game, _t = em.tmp(VEC4, em.input(node, 'Game Texture', VEC4))
        out, _t = em.tmp(VEC4, f'vec4({out}.rgb * {game}.a, {out}.a)')
    elif mode == 'SPARKING':
        # R246: Sparking! ZERO's Mask1 -- the greyscale detail sheet
        # multiplies the flat colour, exactly n_anime_shader
        game, _t = em.tmp(VEC4, em.input(node, 'Game Texture', VEC4))
        out, _t = em.tmp(VEC4, f'vec4({out}.rgb * {game}.rgb, {out}.a)')
    return out, VEC4


def e_cartoon_shader(em, node, _i):
    """The cartoon/paint master (R228), as the deferred pass needs it:
    the Paint Color chain. Every other socket is a surface parameter
    the probe harvests (baked when constant, granted per pixel by the
    tables, refused by name otherwise); the paint composition itself
    lives in the lamp lines and the assembly tail."""
    return em.tmp(VEC4, em.input(node, 'Paint Color', VEC4))


def e_bi_material(em, node, _i):
    """The BI material node, as the deferred pass needs it: its colour.

    Same contract as the Halcyon master: every other socket is a
    surface parameter the frame probe harvests through
    closure_to_surface (baked when constant, refused by name when
    varying); the emitter owes the frame the Diffuse Color chain.
    Options > Vertex Color Paint replaces an UNLINKED colour with the
    mesh vertex colours, exactly n_bi_material's CPU read (a linked
    chain is the artist's own business and wins)."""
    p = node.get('props', {})
    linked = False
    for sock in node.get('inputs', ()):
        nm = sock.get('identifier') or sock.get('name')
        if nm == 'Diffuse Color' or sock.get('name') == 'Color':
            linked = bool(sock.get('link'))
            break
    if p.get('vcol_paint') and not linked and em.has_vcol:
        # verbatim shade_color (R164): the paint ALPHA-LERPS over the
        # base colour, never replaces it outright
        base = em.input(node, 'Diffuse Color', VEC4)
        return em.tmp(VEC4, f'vec4(({base}).rgb * (1.0 - hal_vcol.a) '
                            '+ hal_vcol.rgb * hal_vcol.a, '
                            f'({base}).a)')
    return em.tmp(VEC4, em.input(node, 'Diffuse Color', VEC4))


def e_vertex_color(em, node, _i):
    """The mesh's painted colour, straight from the G-buffer.

    The G-buffer carries the active colour layer; a node naming THAT
    layer reads it (R246: the mesh carries the layer's name, so an
    imported character's 'COL0' resolves on both devices); a node naming
    some other layer refuses rather than quietly reading the wrong paint.
    """
    name = prop(node, 'layer_name', '')
    if name and name != getattr(em, 'color_name', ''):
        raise Unsupported(f"colour layer '{name}' is not in the G-buffer; "
                          f"the active layer '{em.color_name}' is")
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


_LUM709 = 'vec3(0.2126, 0.7152, 0.0722)'


def _e_bi_slot_tin(em, node, tin, want_rgb=False):
    """The slot prelude, exactly `_bi_slot_prelude`: RGBToIntensity
    collapses the Color chain to Rec.709 luminance, Negative inverts.
    All flags are node constants, so only the taken path is emitted.
    Returns (tin_expr, rgb_expr_or_None, alpha_expr_or_None, is_rgb)."""
    is_rgb = bool(prop(node, 'tex_rgb', False))
    rgb = alpha = None
    if is_rgb:
        col = em.tmp(VEC4, em.input(node, 'Color', VEC4))[0]
        rgb = em.tmp(VEC3, f'{col}.rgb')[0]
        alpha = em.tmp(FLOAT, em.input(node, 'Alpha', FLOAT))[0]
        # imagewrap's alpha law, verbatim (R158), same as the CPU
        if bool(prop(node, 'calc_alpha', False)):
            alpha = em.tmp(FLOAT, f'max({rgb}.r, max({rgb}.g, '
                           f'{rgb}.b))')[0]
        if bool(prop(node, 'neg_alpha', False)):
            alpha = em.tmp(FLOAT, f'1.0 - {alpha}')[0]
        if bool(prop(node, 'rgbtoint', False)):
            tin = em.tmp(FLOAT, f'dot({rgb}, {_LUM709})')[0]
            is_rgb = False
            rgb = alpha = None
    if bool(prop(node, 'negative', False)):
        if is_rgb:
            rgb = em.tmp(VEC3, f'vec3(1.0) - {rgb}')[0]
        tin = em.tmp(FLOAT, f'1.0 - {tin}')[0]
    return tin, rgb, alpha, is_rgb


def e_bi_influence(em, node, _i):
    """BI's value-channel influence, exactly `n_bi_influence`.

    The blend mode is a node constant; the scalars arrive as
    expressions, so a linked Factor keeps its per-pixel sign flip.
    The slot prelude runs first: an RGB-yielding texture drives the
    channel by luminance (or alpha under AlphaMix), never raw tin."""
    base = em.tmp(FLOAT, em.input(node, 'Base', FLOAT))[0]
    tin = em.tmp(FLOAT, em.input(node, 'Intensity', FLOAT))[0]
    tin, rgb, alpha, is_rgb = _e_bi_slot_tin(em, node, tin)
    if is_rgb:
        tin = alpha if bool(prop(node, 'alphamix', False)) \
            else em.tmp(FLOAT, f'dot({rgb}, {_LUM709})')[0]
    fac = em.tmp(FLOAT, em.input(node, 'Factor', FLOAT))[0]
    dvar = em.tmp(FLOAT, em.input(node, 'DVar', FLOAT))[0]
    mode = prop(node, 'blend', 'MIX')
    facg = em.tmp(FLOAT, f'abs({fac})')[0]
    fact0 = em.tmp(FLOAT, f'{tin} * {facg}')[0]
    fact = em.tmp(FLOAT,
                  f'({fac} < 0.0) ? (1.0 - {fact0}) : {fact0}')[0]
    facm = em.tmp(FLOAT,
                  f'({fac} < 0.0) ? {fact0} : (1.0 - {fact0})')[0]
    if mode == 'MIX':
        expr = f'{fact} * {dvar} + {facm} * {base}'
    elif mode == 'MUL':
        expr = f'(1.0 - {facg} + {fact} * {dvar}) * {base}'
    elif mode == 'SCREEN':
        expr = (f'1.0 - (1.0 - {facg} + {fact} * (1.0 - {dvar}))'
                f' * (1.0 - {base})')
    elif mode == 'SUB':
        expr = f'-{fact} * {dvar} + {base}'
    elif mode == 'ADD':
        expr = f'{fact} * {dvar} + {base}'
    elif mode == 'DIV':
        expr = (f'({dvar} != 0.0) ? ({facm} * {base} + {fact} * {base}'
                f' / {dvar}) : 0.0')
    elif mode == 'DIFF':
        expr = f'{facm} * {base} + {fact} * abs({dvar} - {base})'
    elif mode == 'DARK':
        # verbatim 2.79 (R155): min(out,tex)*fact + out*facm
        expr = f'min({base}, {dvar}) * {fact} + {base} * {facm}'
    elif mode == 'LIGHT':
        expr = f'max({fact} * {dvar}, {base})'
    elif mode == 'OVERLAY':
        expr = (f'({base} < 0.5) '
                f'? ({base} * (1.0 - {facg} + 2.0 * {fact} * {dvar})) '
                f': (1.0 - (1.0 - {facg} + 2.0 * {fact}'
                f' * (1.0 - {dvar})) * (1.0 - {base}))')
    elif mode == 'SOFT':
        # verbatim 2.79 (R155): the last (base*scf) term is UNSCALED
        scf = em.tmp(FLOAT,
                     f'1.0 - (1.0 - {dvar}) * (1.0 - {base})')[0]
        expr = (f'{facm} * {base} + {fact} * ((1.0 - {base})'
                f' * {dvar} * {base}) + ({base} * {scf})')
    elif mode == 'LINEAR':
        expr = (f'({dvar} > 0.5) '
                f'? ({base} + {fact} * (2.0 * ({dvar} - 0.5))) '
                f': ({base} + {fact} * (2.0 * {dvar} - 1.0))')
    else:
        expr = '0.0'
    return em.tmp(FLOAT, expr)


def e_bi_rgb_blend(em, node, _i):
    """BI's colour-channel influence, exactly `n_bi_rgb_blend`.

    texture_rgb_blend per channel: tcol is the Color chain for an
    RGB-yielding texture (its alpha the per-pixel factor) and the Slot
    Color otherwise (the intensity the factor). The blend mode is a
    node constant, so only its expression is emitted; the four
    ramp_blend-delegating modes call the shading library's own
    hal_ramp_blend, which every shading pass already carries."""
    from ..core.shading import BI_RAMP_BLEND_ORDER
    base = em.tmp(VEC3, f'({em.input(node, "Base", VEC4)}).rgb')[0]
    tin = em.tmp(FLOAT, em.input(node, 'Intensity', FLOAT))[0]
    tin, rgb, alpha, is_rgb = _e_bi_slot_tin(em, node, tin)
    if is_rgb:
        tcol = rgb
        if bool(prop(node, 'map_alpha', False)):
            if bool(prop(node, 'alphamix', False)):
                tin = alpha
        else:
            tin = alpha
    else:
        tcol = em.tmp(VEC3,
                      f'({em.input(node, "Slot Color", VEC4)}).rgb')[0]
    facg = em.tmp(FLOAT, em.input(node, 'Factor', FLOAT))[0]
    fact = em.tmp(FLOAT, f'{tin} * {facg}')[0]
    facm = em.tmp(FLOAT, f'1.0 - {fact}')[0]
    mode = prop(node, 'blend', 'MIX')
    if mode == 'MIX':
        expr = f'{fact} * {tcol} + {facm} * {base}'
    elif mode == 'MUL':
        # verbatim 2.79 (R155): the rgb fn's MUL/SCREEN/OVERLAY use
        # facm = 1 - fact, NOT 1 - facg (that is the VALUE twin's shape)
        expr = f'(vec3({facm}) + {fact} * {tcol}) * {base}'
    elif mode == 'SCREEN':
        expr = (f'vec3(1.0) - (vec3({facm}) + {fact} '
                f'* (vec3(1.0) - {tcol})) * (vec3(1.0) - {base})')
    elif mode == 'SUB':
        expr = f'-{fact} * {tcol} + {base}'
    elif mode == 'ADD':
        expr = f'{fact} * {tcol} + {base}'
    elif mode == 'DIV':
        div = em.tmp(VEC3, f'{facm} * {base} + {fact} * {base} '
                     f'/ mix(vec3(1.0), {tcol}, '
                     f'vec3(notEqual({tcol}, vec3(0.0))))')[0]
        expr = (f'mix({base}, {div}, '
                f'vec3(notEqual({tcol}, vec3(0.0))))')
    elif mode == 'DIFF':
        expr = f'{facm} * {base} + {fact} * abs({tcol} - {base})'
    elif mode == 'DARK':
        # verbatim: min(out,tex)*fact + out*facm -- mix toward min
        expr = f'min({base}, {tcol}) * {fact} + {base} * {facm}'
    elif mode == 'LIGHT':
        expr = f'max({fact} * {tcol}, {base})'
    elif mode == 'OVERLAY':
        low = em.tmp(VEC3, f'{base} * (vec3({facm}) '
                     f'+ 2.0 * {fact} * {tcol})')[0]
        high = em.tmp(VEC3, f'vec3(1.0) - (vec3({facm}) '
                      f'+ 2.0 * {fact} * (vec3(1.0) - {tcol})) '
                      f'* (vec3(1.0) - {base})')[0]
        expr = f'mix({low}, {high}, step(vec3(0.5), {base}))'
    elif mode in ('HUE', 'SAT', 'VAL', 'COLOR', 'SOFT', 'LINEAR'):
        em.used_shading_lib = True
        idx = BI_RAMP_BLEND_ORDER.index(mode)
        expr = f'hal_ramp_blend({idx}, {base}, {fact}, {tcol})'
    else:
        expr = base
    out = em.tmp(VEC3, expr)[0]
    return em.tmp(VEC4, f'vec4({out}, 1.0)')


def e_bi_texture(em, node, index):
    """The Blender Internal texture node on the GPU: every setting is a
    literal at emit time, so each node compiles to a direct call into
    the bitex GLSL library (whose tables are generated from the same
    module the CPU reads). The colorband is unrolled inline from the
    node's own stops."""
    from ..core.bitex import (TEX_BLEND, TEX_CLOUDS, TEX_DISTNOISE,
                              TEX_MAGIC, TEX_MARBLE, TEX_MUSGRAVE,
                              TEX_NOISE, TEX_STUCCI, TEX_VORONOI,
                              TEX_WOOD, node_params)
    _need_pattern(em, 'bitex')
    _need_okramp(em)
    key = ('__bitex_tab',)
    if key not in em.once:
        em.once.add(key)
        # the tables ride a data texture, bound like any image but raw:
        # no filter wrapper, texelFetch straight into the red channel
        em.samplers.append({'uniform': 'hal_bitex_tab',
                            'image': '__bitex_tables__', 'raw': True})

    props = {k: prop(node, k) for k in (
        'tex_type', 'noise_basis', 'noise_basis2', 'wave', 'hard_noise',
        'noise_size', 'noise_depth', 'turbulence', 'clouds_color',
        'wood_type', 'marble_type', 'blend_type', 'blend_flip',
        'stucci_type', 'musgrave_type', 'mg_h', 'mg_lacunarity',
        'mg_octaves', 'mg_offset', 'mg_gain', 'ns_outscale', 'vn_w1',
        'vn_w2', 'vn_w3', 'vn_w4', 'vn_mexp', 'vn_distm', 'vn_coltype',
        'dist_amount', 'bright', 'contrast', 'saturation', 'rgb_factors',
        'use_clamp', 'classic_space', 'tex_offset', 'tex_size',
        'use_colorband', 'coba_ipotype', 'coba')}
    if props.get('use_clamp') is None:
        props['use_clamp'] = True
    if props.get('classic_space') is None:
        props['classic_space'] = True
    p = node_params(props)

    def F(v):
        return f'{float(v):.8f}'

    ofs = props.get('tex_offset') or (0.0, 0.0, 0.0)
    size = props.get('tex_size') or (1.0, 1.0, 1.0)
    v = tex_vector(em, node, 'generated')
    tv, _t = em.tmp(VEC3, (
        f'bi_classic_texvec({v}, '
        f'vec3({F(ofs[0])}, {F(ofs[1])}, {F(ofs[2])}), '
        f'vec3({F(size[0])}, {F(size[1])}, {F(size[2])}), '
        f'{1 if props.get("classic_space", True) else 0})'))

    tt = p['type']
    ns, dep = F(p['noisesize']), int(p['noisedepth'])
    hard = int(p['noisetype'])
    nb, nb2 = int(p['noisebasis']), int(p['noisebasis2'])
    rgb_expr = None
    if tt in (TEX_MUSGRAVE, TEX_VORONOI, TEX_DISTNOISE):
        # multitex() pre-scales only these three by 1/noisesize
        inv = 1.0 / (p['noisesize'] if p['noisesize'] != 0 else 1e-6)
        tv, _t = em.tmp(VEC3, f'{tv} * {F(inv)}')
    if tt == TEX_CLOUDS:
        tin, _t = em.tmp(FLOAT,
                         f'bi_tex_clouds({tv}, {ns}, {dep}, {hard}, {nb})')
        if p['stype'] == 1:
            rgb_expr, _t = em.tmp(VEC3, (
                f'bi_tex_clouds_col({tv}, {ns}, {dep}, {hard}, {nb})'))
    elif tt == TEX_WOOD:
        tin, _t = em.tmp(FLOAT, (
            f'bi_tex_wood({tv}, {p["stype"]}, {nb2}, {ns}, '
            f'{F(p["turbul"])}, {hard}, {nb})'))
    elif tt == TEX_MARBLE:
        tin, _t = em.tmp(FLOAT, (
            f'bi_tex_marble({tv}, {p["stype"]}, {nb2}, {ns}, '
            f'{F(p["turbul"])}, {dep}, {hard}, {nb})'))
    elif tt == TEX_MAGIC:
        m4, _t = em.tmp(VEC4, f'bi_tex_magic({tv}, {dep}, '
                              f'{F(p["turbul"])})')
        tin, _t = em.tmp(FLOAT, f'{m4}.a')
        rgb_expr, _t = em.tmp(VEC3, f'{m4}.rgb')
    elif tt == TEX_BLEND:
        flip = 1 if (p['flag'] & 2) else 0
        tin, _t = em.tmp(FLOAT, f'bi_tex_blend({tv}, {p["stype"]}, '
                                f'{flip})')
    elif tt == TEX_STUCCI:
        tin, _t = em.tmp(FLOAT, (
            f'bi_tex_stucci({tv}, {p["stype"]}, {ns}, '
            f'{F(p["turbul"])}, {hard}, {nb})'))
    elif tt == TEX_NOISE:
        em.frame_uniforms.add('hal_frame')
        tin, _t = em.tmp(FLOAT, f'bi_tex_noise({tv}, {dep}, hal_frame)')
    elif tt == TEX_MUSGRAVE:
        tin, _t = em.tmp(FLOAT, (
            f'bi_tex_musgrave({tv}, {p["stype"]}, {F(p["mg_H"])}, '
            f'{F(p["mg_lacunarity"])}, {F(p["mg_octaves"])}, '
            f'{F(p["mg_offset"])}, {F(p["mg_gain"])}, '
            f'{F(p["ns_outscale"])}, {nb})'))
    elif tt == TEX_VORONOI:
        v4, _t = em.tmp(VEC4, (
            f'bi_tex_voronoi({tv}, {F(p["vn_w1"])}, {F(p["vn_w2"])}, '
            f'{F(p["vn_w3"])}, {F(p["vn_w4"])}, {F(p["vn_mexp"])}, '
            f'{p["vn_distm"]}, {F(p["ns_outscale"])}, '
            f'{p["vn_coltype"]})'))
        tin, _t = em.tmp(FLOAT, f'{v4}.a')
        if p['vn_coltype']:
            rgb_expr, _t = em.tmp(VEC3, f'{v4}.rgb')
    elif tt == TEX_DISTNOISE:
        tin, _t = em.tmp(FLOAT, (
            f'bi_tex_distnoise({tv}, {F(p["dist_amount"])}, {nb}, '
            f'{nb2})'))
    else:
        tin, _t = em.tmp(FLOAT, '0.0')

    # ---- colorband, unrolled from the node's stops
    alpha = '1.0'
    coba = p.get('coba')
    if (p['flag'] & 1) and coba:
        ipo = int(p.get('coba_ipotype', 0))
        cb, _t = em.tmp(VEC4, 'vec4(0.0)')
        n = len(coba)
        if n == 1:
            s = coba[0]
            em.lines.append(f'{cb} = vec4({F(s[1])}, {F(s[2])}, '
                            f'{F(s[3])}, {F(s[4])});')
        else:
            arr = ', '.join(
                f'vec4({F(s[1])}, {F(s[2])}, {F(s[3])}, {F(s[4])})'
                for s in coba)
            pos = ', '.join(F(s[0]) for s in coba)
            # array declarations use the `vec4 name[N]` form, never
            # `vec4[N] name`: both are GLSL, but the second is the form
            # shader translators (and Halcyon's own front-end, the
            # headless proof) are least likely to accept
            em._n += 1
            cbc = f'_v{em._n}'
            em.lines.append(f'    vec4 {cbc}[{n}] = vec4[{n}]({arr});')
            em._n += 1
            cbp = f'_v{em._n}'
            em.lines.append(f'    float {cbp}[{n}] = float[{n}]({pos});')
            body = (
                '{\n'
                '    int a = 0;\n'
                f'    while (a < {n} && {cbp}[a] <= {tin}) {{ a++; }}\n'
                f'    int ir = (a < {n}) ? a : {n} - 1;\n'
                '    int il = (a > 0) ? a - 1 : 0;\n'
                f'    float rpos = (a >= {n}) ? 1.0 : {cbp}[ir];\n'
                f'    float lpos = (a <= 0) ? 0.0 : {cbp}[il];\n'
                f'    vec4 rcol = {cbc}[ir];\n'
                f'    vec4 lcol = {cbc}[il];\n'
                '    float span = lpos - rpos;\n'
                f'    float fac = (abs(span) > 1e-9) ? ({tin} - rpos) / span\n'
                f'                                   : ((a >= {n}) ? 1.0 : 0.0);\n'
                f'    int ipo = {ipo};\n'
                f'    if (ipo == 4) {{ {cb} = lcol; }}\n'
                '    else if (ipo == 2 || ipo == 3) {\n'
                f'        int i0 = (a >= {n} - 1) ? ir : min(ir + 1, {n} - 1);\n'
                '        int i3 = (a < 2) ? il : max(il - 1, 0);\n'
                f'        vec4 c0 = {cbc}[i0]; vec4 c1 = rcol;\n'
                f'        vec4 c2 = lcol; vec4 c3 = {cbc}[i3];\n'
                '        float t = clamp(fac, 0.0, 1.0);\n'
                '        float t2 = t * t; float t3 = t2 * t;\n'
                '        vec4 res;\n'
                '        if (ipo == 3) {\n'
                '            res = (0.5 * t3 - 0.5 * t2) * c3\n'
                '                + (-1.5 * t3 + 2.0 * t2 + 0.5 * t) * c2\n'
                '                + (1.5 * t3 - 2.5 * t2 + 1.0) * c1\n'
                '                + (-0.5 * t3 + t2 - 0.5 * t) * c0;\n'
                '        } else {\n'
                '            res = (0.16666666 * t3) * c3\n'
                '                + (-0.5 * t3 + 0.5 * t2 + 0.5 * t + 0.16666666) * c2\n'
                '                + (0.5 * t3 - t2 + 0.66666666) * c1\n'
                '                + (-0.16666666 * t3 + 0.5 * t2 - 0.5 * t + 0.16666666) * c0;\n'
                '        }\n'
                f'        {cb} = clamp(res, 0.0, 1.0);\n'
                '    } else {\n'
                '        float f2 = clamp(fac, 0.0, 1.0);\n'
                '        if (ipo == 1) { f2 = 3.0 * f2 * f2 - 2.0 * f2 * f2 * f2; }\n'
                f'        {cb} = (1.0 - f2) * rcol + f2 * lcol;\n'
                '    }\n'
                '}')
            em.lines.append(body)
        rgb_expr, _t = em.tmp(VEC3, f'{cb}.rgb')
        alpha, _t = em.tmp(FLOAT, f'{cb}.a')

    # ---- BRICONT(RGB)
    noclamp = 1 if (p['flag'] & 1024) else 0
    rf = props.get('rgb_factors') or (1.0, 1.0, 1.0)
    if rgb_expr is not None:
        rgb_expr, _t = em.tmp(VEC3, (
            f'bi_bricontrgb({rgb_expr}, {F(p["bright"])}, '
            f'{F(p["contrast"])}, {F(p["saturation"])}, '
            f'vec3({F(rf[0])}, {F(rf[1])}, {F(rf[2])}), {noclamp})'))
        if not noclamp:
            tin, _t = em.tmp(FLOAT, f'clamp({tin}, 0.0, 1.0)')
    else:
        tin, _t = em.tmp(FLOAT, (
            f'bi_bricont({tin}, {F(p["bright"])}, {F(p["contrast"])}, '
            f'{noclamp})'))

    outs = node.get('outputs') or []
    o = outs[index] if index < len(outs) else {}
    name = o.get('name')
    if name == 'Fac':
        return tin, FLOAT
    if name == 'Alpha':
        return alpha, FLOAT
    if rgb_expr is not None:
        return em.tmp(VEC4, f'vec4({rgb_expr}, {alpha})')
    return em.tmp(VEC4, f'vec4({tin}, {tin}, {tin}, 1.0)')


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
        d, _t = em.tmp(FLOAT, f'length({p} - {em.const(c, VEC3)})')
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
    # R216 rosettes: the pattern hands back (ring, interior) and the
    # colour output lays ground -> Interior -> ring, exactly as the
    # evaluator does; Fac is the ring mask
    _need_pattern(em, 'leopard')
    p = _pat_vec(em, node)
    ri, _t = em.tmp('vec2', (
        f'hal_pat_leopard({p}, {_pat_scalar(em, node, "Spot", 1.0)}, '
        f'{_pat_scalar(em, node, "Jitter", 0.85)}, '
        f'{_pat_scalar(em, node, "Break", 0.55)})'))
    outs = node.get('outputs') or []
    o = outs[index] if index < len(outs) else {}
    if o.get('name') == 'Fac':
        return em.tmp(FLOAT, f'{ri}.x')
    a, _t = em.tmp(VEC4, em.input(node, 'Color 1', VEC4))
    b, _t = em.tmp(VEC4, em.input(node, 'Color 2', VEC4))
    ci, _t = em.tmp(VEC4, em.input(node, 'Interior', VEC4))
    col, _t = em.tmp(VEC4, f'{a} + ({ci} - {a}) * {ri}.y')
    return em.tmp(VEC4, f'{col} + ({b} - {col}) * {ri}.x')


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


_NOISE_KIND = {'SMOOTH': 0, 'TURBULENT': 1, 'RIDGED': 2}
_CELL_FEATURE = {'F1': 0, 'F2': 1, 'BORDER': 2, 'CELL': 3}


def e_pat_noise(em, node, index):
    p = _pat_vec(em, node)
    dims = str(prop(node, 'dims', '3D'))
    kind = _NOISE_KIND.get(str(prop(node, "kind", "SMOOTH")), 0)
    octv = int(prop(node, "octaves", 5))
    lac = _pat_scalar(em, node, "Lacunarity", 2.0)
    gain = _pat_scalar(em, node, "Gain", 0.5)
    if dims == '3D':
        _need_pattern(em, 'noise')
        return _pat_output(em, node, index, (
            f'hal_pat_noise({p}, {kind}, {octv}, {lac}, {gain})'))
    _need_pattern(em, 'noise_nd')
    if dims == '4D':
        w = em.input(node, 'W', FLOAT)
        scale = em.input(node, 'Scale', FLOAT)
        p4, _t = em.tmp(VEC4, f'vec4({p}, {w} * {scale})')
    else:
        p4, _t = em.tmp(VEC4, f'vec4({p}, 0.0)')
    d = {'1D': 1, '2D': 2, '4D': 4}.get(dims, 4)
    return _pat_output(em, node, index, (
        f'hal_pat_noise_nd({p4}, {d}, {kind}, {octv}, {lac}, {gain})'))


_CAUSTIC_GLSL = """
float hal_caustic(vec2 uv, float t, float per)
{
    vec2 ci = floor(uv);
    vec2 cf = uv - ci;
    float f1 = 1e9; float f2 = 1e9;
    for (int oy = -1; oy <= 1; oy++)
    for (int ox = -1; ox <= 1; ox++) {
        float cx = mod(ci.x + float(ox), per);
        cx = (cx < 0.0) ? cx + per : cx;
        float cy = mod(ci.y + float(oy), per);
        cy = (cy < 0.0) ? cy + per : cy;
        uint h = uint(int(cx + 0.5)) + uint(int(cy + 0.5)) * uint(int(per + 0.5));
        float a1 = hal_wang01(h ^ 2654435769u);
        float a2 = hal_wang01(h ^ 2246822507u);
        float ph = hal_wang01(h ^ 3266489909u) * 6.2831853;
        vec2 p = vec2(0.5 + (0.22 + 0.2 * a1) * cos(ph + t),
                      0.5 + (0.22 + 0.2 * a2) * sin(ph + t));
        vec2 dv = vec2(float(ox), float(oy)) + p - cf;
        float d = dot(dv, dv);
        float nf1 = min(f1, d);
        f2 = min(f2, max(f1, d));
        f1 = nf1;
    }
    float edge = sqrt(max(f2, 0.0)) - sqrt(max(f1, 0.0));
    float web = clamp(1.0 - edge * 3.2, 0.0, 1.0);
    return web * web * web;
}
"""


def e_pat_caustics(em, node, index):
    """The pool-light web on the GPU: the CPU's caustic_web, hash for
    hash -- the same uint32 Wang mix, the same wrapped lattice."""
    if '__wang' not in em.once:
        em.once.add('__wang')
        em.inline.append(_WANG_GLSL)
    if '__caustic' not in em.once:
        em.once.add('__caustic')
        em.inline.append(_CAUSTIC_GLSL)
    p = _pat_vec(em, node)
    t = _pat_time(em, node)
    speed = _pat_scalar(em, node, 'Speed', 1.0)
    f, _t2 = em.tmp(FLOAT,
                    f'hal_caustic(({p}).xy, ({t}) * ({speed}), 8.0)')
    return _pat_output(em, node, index, f)


def e_pat_water(em, node, index):
    from ..core.patterns import WATER_DIRS
    _need_pattern(em, 'water')
    p = _pat_vec(em, node)
    t = _pat_time(em, node)
    speed = _pat_scalar(em, node, 'Speed', 1.0)
    chop = _pat_scalar(em, node, 'Choppiness', 0.5)
    layers = max(1, min(int(prop(node, 'layers', 5)), 12))
    looping = bool(prop(node, 'loop', False))
    lf = max(int(prop(node, 'loop_frames', 48)), 1)
    fps = int(prop(node, 'fps', 24))
    if looping:
        ph, _t2 = em.tmp(FLOAT, f'6.28318530717959 * fract({t} * '
                                f'{float(fps)} / {float(lf)})')
        lz, _t2 = em.tmp(FLOAT, f'cos({ph}) * 1.5')
        lw, _t2 = em.tmp(FLOAT, f'sin({ph}) * 1.5')
    else:
        lz, lw = '0.0', '0.0'
    total, _t2 = em.tmp(FLOAT, '0.0')
    amp, freq, norm = 1.0, 1.0, 0.0
    for i in range(layers):
        ca = float(WATER_DIRS[i, 0])
        sa = float(WATER_DIRS[i, 1])
        rate = 0.6 + 0.13 * i
        salt = 17.0 * i + 3.0
        em.lines.append(
            f'    {total} = {total} + hal_pat_water_layer({p}.xy, '
            f'{ca!r}, {sa!r}, {freq!r}, {rate!r}, {salt!r}, '
            f'{t}, {speed}, {chop}, {lz}, {lw}, '
            f'{1 if looping else 0}) * {amp!r};')
        norm += amp
        amp *= 0.55
        freq *= 1.9
    return _pat_output(em, node, index,
                       f'{total} / {max(norm, 1e-6)!r}')


def e_pat_gradient_shaped(em, node, index):
    _need_pattern(em, 'gradientshape')
    v = tex_vector(em, node, 'generated')
    centre = em.input(node, 'Center', VEC3)
    scale = em.input(node, 'Scale', FLOAT)
    rot = em.input(node, 'Rotation', FLOAT)
    q0, _t = em.tmp(VEC3, f'({v} - {centre}) * {scale}')
    ca, _t = em.tmp(FLOAT, f'cos(-{rot})')
    sa, _t = em.tmp(FLOAT, f'sin(-{rot})')
    q, _t = em.tmp(VEC3, f'vec3({q0}.x * {ca} - {q0}.y * {sa}, '
                         f'{q0}.x * {sa} + {q0}.y * {ca}, {q0}.z)')
    shapes = {'LINEAR': 0, 'REFLECTED': 1, 'SPHERICAL': 2, 'QUADRATIC': 3,
              'SQUARE': 4, 'DIAMOND': 5, 'CONICAL': 6, 'SPIRAL': 7}
    reps = {'NONE': 0, 'REPEAT': 1, 'PINGPONG': 2}
    eases = {'NONE': 0, 'SMOOTH': 1, 'SHARP': 2}
    return _pat_output(em, node, index, (
        f'hal_pat_gradient({q}, '
        f'{shapes.get(str(prop(node, "shape", "LINEAR")), 0)}, '
        f'{reps.get(str(prop(node, "repeat", "NONE")), 0)}, '
        f'{eases.get(str(prop(node, "easing", "NONE")), 0)})'))


def e_pat_cells_tex(em, node, index):
    _need_pattern(em, 'cells')
    p = _pat_vec(em, node)
    res, _t = em.tmp('vec2', (
        f'hal_pat_cells({p}, {_pat_scalar(em, node, "Randomness", 1.0)}, '
        f'{_CELL_FEATURE.get(str(prop(node, "feature", "F1")), 0)})'))
    outs = node.get('outputs') or []
    o = outs[index] if index < len(outs) else {}
    if o.get('name') == 'Cell ID':
        return em.tmp(FLOAT, f'{res}.y')
    return _pat_output(em, node, index, f'{res}.x')


def e_pat_static(em, node, index):
    _need_pattern(em, 'static')
    p = _pat_vec(em, node)
    if prop(node, 'animate', True):
        em.frame_uniforms.add('hal_frame')
        frame = 'hal_frame'
    else:
        frame = '0.0'
    return _pat_output(em, node, index,
                       f'hal_pat_static({p}, {frame})')


def e_pat_fur_tufts(em, node, index):
    """R209: the shell-fur field -- same lattice hashes as the CPU, so the
    tuft discs land identically and taper identically on both devices."""
    _need_pattern(em, 'furtufts')
    p = _pat_vec(em, node)
    res, _t = em.tmp('vec2', (
        f'hal_pat_fur_tufts({p}, '
        f'{_pat_scalar(em, node, "Coverage", 1.0)}, '
        f'{_pat_scalar(em, node, "Taper", 1.0)}, '
        f'{_pat_scalar(em, node, "Variation", 0.35)})'))
    outs = node.get('outputs') or []
    o = outs[index] if index < len(outs) else {}
    if o.get('name') == 'Random':
        return em.tmp(FLOAT, f'{res}.y')
    if o.get('name') == 'Height':
        return em.tmp(FLOAT, f'{res}.x')
    return _pat_output(em, node, index, f'{res}.x')


# --- the 2D media (R232) ---------------------------------------------------
#
# core/media.py's twins, through the pattern conventions plus three new
# seams. TONE is per-pixel (a Shader to RGB luminance cannot travel, but a
# Facing, a Layer Weight or a dot with a light direction can), so it emits
# through em.input like a colour. SCREEN space reads the frame pass's own
# vUV back to the integer pixel and divides by the frame HEIGHT -- the
# CPU's (px + .5) / H in the same float32 ops -- and answers zeros off the
# pixel grid, exactly as the Window coordinate and Screen Info do. The
# BOIL rides hal_frame: an animation never recompiles a drawing, the salt
# is integer arithmetic on the uniform. Angles that reach a floor (the
# lanes, the brush direction, the charcoal streak axis) are baked to
# float32 cos/sin literals from the SAME float32 socket value the CPU
# read, so no driver's cos decides a lane; the scribble's fan and the
# brush's turn slope bake the same way.


def _lit32(v):
    """A float32 literal that round-trips: the repr of the float32's exact
    value parses back to that float32 on any front-end."""
    import numpy as np
    return repr(float(np.float32(v)))


def _md_default(node, name, fallback):
    """The socket default AS THE CPU READS IT: rounded to float32 first, so
    a baked cos/sin starts from the same number on both devices."""
    import numpy as np
    return float(np.float32(_socket_default(node, name, fallback)))


def _md_xy(em, node):
    space = str(prop(node, 'space', 'VECTOR'))
    if space == 'VIEW':
        # R233: the camera's sphere from the SAME two float32 vectors the
        # CPU subtracts (hal_P is the CPU's own P; hal_eye its eye)
        _need_pattern(em, 'md_prims')
        scale = em.input(node, 'Scale', FLOAT)
        p, _t = em.tmp('vec2', f'hal_md_view(hal_P - hal_eye) * {scale}')
        return p
    if space == 'SCREEN':
        on_grid = (em.frame_mode and not em.secondary
                   and getattr(em, 'resolution', None) is not None)
        if on_grid:
            w, h = float(em.resolution[0]), float(em.resolution[1])
            base = (f'vec2((floor(vUV.x * {_c(w)}) + 0.5) / {_c(h)}, '
                    f'(floor(vUV.y * {_c(h)}) + 0.5) / {_c(h)})')
        else:
            base = 'vec2(0.0, 0.0)'
        scale = em.input(node, 'Scale', FLOAT)
        p, _t = em.tmp('vec2', f'{base} * {scale}')
        return p
    p3 = _pat_vec(em, node)
    p, _t = em.tmp('vec2', f'{p3}.xy')
    return p


def _md_salt(em, node):
    boil = int(prop(node, 'boil', 0))
    if boil > 0:
        em.frame_uniforms.add('hal_frame')
        frame = 'hal_frame'
    else:
        frame = '0.0'
    seed = em.const(float(int(prop(node, 'seed', 0))), FLOAT)
    s, _t = em.tmp('int', f'hal_md_salt(int({seed}), {frame}, {boil})')
    return s


def _md_has_socket(node, name):
    return any(s.get('name') == name or s.get('identifier') == name
               for s in node.get('inputs', ()))


def _md_tone(em, node):
    tone = em.input(node, 'Tone', FLOAT)
    if _md_has_socket(node, 'Indication'):
        # R235: tone' = 1 - (1 - tone) * indication, exactly nodeeval
        ind = em.input(node, 'Indication', FLOAT)
        tone, _t = em.tmp(FLOAT, f'1.0 - (1.0 - ({tone})) * ({ind})')
    d, _t = em.tmp(FLOAT, f'clamp(1.0 - ({tone}), 0.0, 1.0)')
    return d


def _md_directed(em, node, fan, call):
    """R235: a lane medium's Direction. ANGLE emits the one call `call(rl)`
    with the baked layer rotations; FORM / SLOPE emit the tangent from
    hal_N and hal_V (the matcap frame on the camera's paper), the bin
    coordinate, and two calls with the bin's literal rotation tables
    cross-faded as f0 * (1 - w) + f1 * w -- media.blend_bins line for
    line."""
    from ..core import media as MD
    mode = str(prop(node, 'direction', 'ANGLE')).upper()
    if mode not in ('FORM', 'SLOPE'):
        _L, rl = _md_rots(em, node, fan)
        f, _t = em.tmp(FLOAT, call(rl))
        return f
    L = max(min(int(prop(node, 'layers', 2)), 4), 1)
    _pat_scalar(em, node, 'Angle')          # linked: refuse by name
    ang = _md_default(node, 'Angle', 45.0)
    table = MD.bin_rotations(ang, L, fan)
    screen = '1' if str(prop(node, 'space', 'VECTOR')) == 'SCREEN' else '0'
    n, _t = em.tmp(VEC3, 'normalize(hal_N)')
    t2, _t = em.tmp('vec2', f'hal_md_tangent({n}, hal_V, '
                            f'{0 if mode == "FORM" else 1}, {screen})')
    b, _t = em.tmp(FLOAT, f'hal_md_bins({t2}.x, {t2}.y)')
    bf, _t = em.tmp(FLOAT, f'floor({b})')
    i0, _t = em.tmp('int', f'min(int({bf}), {MD.DIRECTION_BINS - 1})')
    i1, _t = em.tmp('int', f'({i0} == {MD.DIRECTION_BINS - 1}) ? 0 : {i0} + 1')
    w, _t = em.tmp(FLOAT, f'smoothstep(0.35, 0.65, {b} - {bf})')

    def rots_for(idx):
        lits = []
        for k in range(4):
            if k < L:
                chain = f'vec2({_lit32(table[MD.DIRECTION_BINS - 1][k][0])}, ' \
                        f'{_lit32(table[MD.DIRECTION_BINS - 1][k][1])})'
                for i in range(MD.DIRECTION_BINS - 2, -1, -1):
                    c, sn = table[i][k]
                    chain = f'({idx} == {i}) ? vec2({_lit32(c)}, ' \
                            f'{_lit32(sn)}) : ({chain})'
                r, _t = em.tmp('vec2', chain)
                lits.append(r)
            else:
                lits.append('vec2(1.0, 0.0)')
        return ', '.join(lits)

    f0, _t = em.tmp(FLOAT, call(rots_for(i0)))
    f1, _t = em.tmp(FLOAT, call(rots_for(i1)))
    f, _t = em.tmp(FLOAT, f'{f0} * (1.0 - {w}) + {f1} * {w}')
    return f


def _md_rots(em, node, fan=180.0):
    from ..core import media as MD
    L = max(min(int(prop(node, 'layers', 2)), 4), 1)
    _pat_scalar(em, node, 'Angle')          # linked: refuse by name
    ang = _md_default(node, 'Angle', 45.0)
    lits = [f'vec2({_lit32(c)}, {_lit32(s)})'
            for c, s in (MD.rotation(ang, k, L, fan) for k in range(L))]
    while len(lits) < 4:
        lits.append('vec2(1.0, 0.0)')
    return L, ', '.join(lits)


def _md_rot1(em, node, fallback):
    from ..core import media as MD
    _pat_scalar(em, node, 'Angle')
    c, s = MD.rotation(_md_default(node, 'Angle', fallback))
    return f'vec2({_lit32(c)}, {_lit32(s)})'


def e_md_hatching(em, node, index):
    _need_pattern(em, 'md_prims')
    _need_pattern(em, 'md_hatching')
    p = _md_xy(em, node)
    d = _md_tone(em, node)
    L = max(min(int(prop(node, 'layers', 2)), 4), 1)
    width = _pat_scalar(em, node, "Width", 0.35)
    length = _pat_scalar(em, node, "Length", 6.0)
    wobble = _pat_scalar(em, node, "Wobble", 0.5)
    breaks = _pat_scalar(em, node, "Breaks", 0.2)
    salt = _md_salt(em, node)
    f = _md_directed(em, node, 180.0, lambda rl: (
        f'hal_md_hatching({p}, {d}, {L}, {rl}, {width}, {length}, '
        f'{wobble}, {breaks}, {salt})'))
    return _pat_output(em, node, index, f)


def e_md_scribble(em, node, index):
    from ..core import media as MD
    _need_pattern(em, 'md_prims')
    _need_pattern(em, 'md_scribble')
    p = _md_xy(em, node)
    d = _md_tone(em, node)
    L = max(min(int(prop(node, 'layers', 2)), 4), 1)
    width = _pat_scalar(em, node, "Width", 0.3)
    curl = _pat_scalar(em, node, "Curl", 0.5)
    pressure = _pat_scalar(em, node, "Pressure", 0.7)
    grain = _pat_scalar(em, node, "Grain", 0.6)
    salt = _md_salt(em, node)
    blend = _pat_scalar(em, node, "Blend", 0.0)
    f = _md_directed(em, node, MD.SCRIBBLE_FAN, lambda rl: (
        f'hal_md_scribble({p}, {d}, {L}, {rl}, {width}, {curl}, '
        f'{pressure}, {grain}, {salt}, {blend})'))
    return _pat_output(em, node, index, f)


def e_md_stipple(em, node, index):
    _need_pattern(em, 'md_prims')
    _need_pattern(em, 'md_stipple')
    p = _md_xy(em, node)
    d = _md_tone(em, node)
    placement = 1 if str(prop(node, 'placement', 'SIZE')).upper() == 'COUNT' \
        else 0
    f, _t = em.tmp(FLOAT, (
        f'hal_md_stipple({p}, {d}, '
        f'{_pat_scalar(em, node, "Size", 0.6)}, '
        f'{_pat_scalar(em, node, "Jitter", 0.8)}, '
        f'{_pat_scalar(em, node, "Fine", 1.0)}, {_md_salt(em, node)}, '
        f'{placement})'))
    return _pat_output(em, node, index, f)


def e_md_charcoal(em, node, index):
    _need_pattern(em, 'md_prims')
    _need_pattern(em, 'md_charcoal')
    p = _md_xy(em, node)
    d = _md_tone(em, node)
    rot = _md_rot1(em, node, 20.0)
    f, _t = em.tmp(FLOAT, (
        f'hal_md_charcoal({p}, {d}, {rot}, '
        f'{_pat_scalar(em, node, "Grain", 0.6)}, '
        f'{_pat_scalar(em, node, "Streak", 0.5)}, '
        f'{_pat_scalar(em, node, "Smudge", 0.4)}, {_md_salt(em, node)}, '
        f'{_pat_scalar(em, node, "Blend", 0.0)})'))
    return _pat_output(em, node, index, f)


def e_md_paint(em, node, index):
    from ..core import media as MD
    _need_pattern(em, 'md_prims')
    _need_pattern(em, 'md_paint')
    p = _md_xy(em, node)
    rot = _md_rot1(em, node, 25.0)
    _pat_scalar(em, node, 'Spread')
    slope = _lit32(MD.paint_slope(_md_default(node, 'Spread', 0.5)))
    res, _t = em.tmp(VEC3, (
        f'hal_md_paint({p}, {rot}, '
        f'{_pat_scalar(em, node, "Length", 2.6)}, '
        f'{_pat_scalar(em, node, "Width", 0.85)}, {slope}, '
        f'{_pat_scalar(em, node, "Bristles", 0.6)}, '
        f'{_pat_scalar(em, node, "Variation", 0.3)}, {_md_salt(em, node)})'))
    outs = node.get('outputs') or []
    o = outs[index] if index < len(outs) else {}
    if o.get('name') == 'Stroke ID':
        return em.tmp(FLOAT, f'{res}.z')
    if o.get('name') == 'Fac':
        return em.tmp(FLOAT, f'{res}.x')
    canvas, _t = em.tmp(VEC4, em.input(node, 'Color 1', VEC4))
    paint, _t = em.tmp(VEC4, em.input(node, 'Color 2', VEC4))
    col, _t = em.tmp(VEC4, f'{canvas} * (1.0 - {res}.y) + {paint} * {res}.x')
    return em.tmp(VEC4, f'vec4({col}.rgb, 1.0)')


def e_md_wash(em, node, index):
    _need_pattern(em, 'md_prims')
    _need_pattern(em, 'md_wash')
    p = _md_xy(em, node)
    d = _md_tone(em, node)
    n = max(min(int(prop(node, 'levels', 3)), 6), 1)
    f, _t = em.tmp(FLOAT, (
        f'hal_md_wash({p}, {d}, {n}, '
        f'{_pat_scalar(em, node, "Pooling", 0.6)}, '
        f'{_pat_scalar(em, node, "Granulation", 0.4)}, '
        f'{_pat_scalar(em, node, "Bleed", 0.4)}, {_md_salt(em, node)})'))
    return _pat_output(em, node, index, f)


def e_md_paper(em, node, index):
    _need_pattern(em, 'md_prims')
    _need_pattern(em, 'md_paper')
    p = _md_xy(em, node)
    f, _t = em.tmp(FLOAT, (
        f'hal_md_paper({p}, '
        f'{_pat_scalar(em, node, "Tooth", 0.6)}, '
        f'{_pat_scalar(em, node, "Fibres", 0.3)}, '
        f'{_pat_scalar(em, node, "Mottle", 0.4)}, {_md_salt(em, node)})'))
    return _pat_output(em, node, index, f)


# --- the retro utilities --------------------------------------------------
#
# Pure arithmetic start to finish, so they travel whole: the same floor,
# fract and roundEven the CPU runs (roundEven IS np.round -- the depth
# quantiser proved that pairing on real drivers). The two that read the
# camera's pixel grid (Ordered Dither, Screen Info) emit the frame pass's
# vUV form of it and mirror the CPU's px-is-None fallback in secondary
# passes, exactly as nodeeval does on ray hits.


def e_halcyon_posterize(em, node, _i):
    col, _t = em.tmp(VEC4, em.input(node, 'Color', VEC4))
    lv, _t = em.tmp(FLOAT, f'max({em.input(node, "Levels", FLOAT)}, 1.0)')
    q, _t = em.tmp(VEC3, f'floor(clamp({col}.rgb, 0.0, 1.0) * {lv}) '
                         f'/ max({lv} - 1.0, 1.0)')
    return em.tmp(VEC4, f'vec4(clamp({q}, 0.0, 1.0), {col}.a)')


#: the Bayer matrix, as arithmetic: entry (x, y) of the 2^bits square is
#: the base-4 number whose digit for bit i (LSB outermost) is
#: 2*(x_i XOR y_i) + y_i -- exactly the recursive [[4m, 4m+2], [4m+3,
#: 4m+1]] construction dither.bayer() runs, so float(v)/cells equals the
#: CPU's threshold map bit for bit (every value is a small integer over a
#: power of two). A test holds the equality against dither.threshold_map.
_BAYER_GLSL = """
float hal_bayer(int px, int py, int bits, float cells)
{
    uint x = uint(px);
    uint y = uint(py);
    uint v = 0u;
    for (int i = 0; i < bits; i++) {
        uint xi = (x >> uint(i)) & 1u;
        uint yi = (y >> uint(i)) & 1u;
        v = v * 4u + (2u * (xi ^ yi) + yi);
    }
    return float(v) / cells;
}
"""


def e_halcyon_dither(em, node, _i):
    kind = str(prop(node, 'pattern', 'BAYER4'))
    sizes = {'BAYER2': (1, 2), 'BAYER4': (2, 4), 'BAYER8': (3, 8)}
    if kind not in sizes:
        raise Unsupported('the halftone cell is a lookup table the frame '
                          'shader does not carry; the Bayer patterns '
                          'travel, halftone shades on the CPU')
    col, _t = em.tmp(VEC4, em.input(node, 'Color', VEC4))
    lv, _t = em.tmp(FLOAT, f'max({em.input(node, "Levels", FLOAT)}, 2.0)')
    strength, _t = em.tmp(FLOAT, em.input(node, 'Strength', FLOAT))
    if em.frame_mode and not em.secondary and \
            getattr(em, 'resolution', None) is not None:
        if '__bayer' not in em.once:
            em.once.add('__bayer')
            em.inline.append(_BAYER_GLSL)
        bits, n = sizes[kind]
        w, h = float(em.resolution[0]), float(em.resolution[1])
        pix, _t = em.tmp('ivec2', f'ivec2(int(vUV.x * {_c(w)}), '
                                  f'int(vUV.y * {_c(h)}))')
        # px % n by mask: n is a power of two, and the pixel is never
        # negative in the frame grid
        t, _t = em.tmp(FLOAT,
                       f'hal_bayer(int(uint({pix}.x) & {n - 1}u), '
                       f'int(uint({pix}.y) & {n - 1}u), '
                       f'{bits}, {_c(float(n * n))}) - 0.5')
    else:
        # ray hits shade with no pixel grid: nodeeval quantises undithered
        # there (t = 0), and this pass does exactly that
        t, _t = em.tmp(FLOAT, '0.0')
    step, _t = em.tmp(FLOAT, f'1.0 / max({lv} - 1.0, 1.0)')
    biased, _t = em.tmp(VEC3, f'{col}.rgb + vec3({t} * {strength} * {step})')
    q, _t = em.tmp(VEC3, f'roundEven(clamp({biased}, 0.0, 1.0) '
                         f'* ({lv} - 1.0)) / max({lv} - 1.0, 1.0)')
    return em.tmp(VEC4, f'vec4(clamp({q}, 0.0, 1.0), {col}.a)')


def e_halcyon_screen_info(em, node, index):
    outs = node.get('outputs') or []
    o = outs[index] if index < len(outs) else {}
    name = o.get('name')
    on_grid = (em.frame_mode and not em.secondary
               and getattr(em, 'resolution', None) is not None)
    if name == 'Screen UV':
        return em.tmp(VEC3, 'vec3(vUV, 0.0)' if on_grid
                      else 'vec3(0.0, 0.0, 0.0)')
    if name == 'Pixel':
        if not on_grid:
            return em.tmp(VEC3, 'vec3(0.0, 0.0, 0.0)')
        w, h = float(em.resolution[0]), float(em.resolution[1])
        return em.tmp(VEC3, f'vec3(floor(vUV.x * {_c(w)}), '
                            f'floor(vUV.y * {_c(h)}), 0.0)')
    if name == 'Facing':
        return em.tmp(FLOAT, 'abs(dot(normalize(hal_N), '
                             'normalize(hal_V)))')
    if name == 'Frame':
        em.frame_uniforms.add('hal_frame')
        return em.tmp(FLOAT, 'hal_frame')
    if name == 'Time':
        em.frame_uniforms.add('hal_time')
        return em.tmp(FLOAT, 'hal_time')
    raise Unsupported('view-space depth is not in the G-buffer; a material '
                      'reading Screen Info\'s Depth shades on the CPU')


def e_halcyon_pixelate(em, node, _i):
    p, _t = em.tmp(VEC3, tex_vector(em, node, 'uv'))
    parts = []
    for axis, sock in (('x', 'Pixels X'), ('y', 'Pixels Y'),
                       ('z', 'Pixels Z')):
        cnt, _t = em.tmp(FLOAT, em.input(node, sock, FLOAT))
        snapped, _t = em.tmp(
            FLOAT, f'({cnt} >= 1.0) ? (min(floor({p}.{axis} '
                   f'* max({cnt}, 1.0)), max({cnt}, 1.0) - 1.0) '
                   f'+ 0.5) / max({cnt}, 1.0) : {p}.{axis}')
        parts.append(snapped)
    return em.tmp(VEC3, f'vec3({parts[0]}, {parts[1]}, {parts[2]})')


def _scroll_time(em, node):
    """The scroll clock, with the stepped-time quantise baked in."""
    t = _pat_time(em, node)
    fps = int(prop(node, 'fps', 0) or 0)
    if t != '0.0' and fps > 0:
        t, _t = em.tmp(FLOAT, f'floor({t} * {_c(float(fps))}) '
                              f'/ {_c(float(fps))}')
    return t


def e_halcyon_scroll(em, node, _i):
    p, _t = em.tmp(VEC3, tex_vector(em, node, 'uv'))
    t = _scroll_time(em, node)
    sx = em.input(node, 'Scroll X', FLOAT)
    sy = em.input(node, 'Scroll Y', FLOAT)
    spin = em.input(node, 'Spin', FLOAT)
    ang, _t = em.tmp(FLOAT, f'{spin} * ({t} * 6.28318530717959)')
    ca, _t = em.tmp(FLOAT, f'cos({ang})')
    sa, _t = em.tmp(FLOAT, f'sin({ang})')
    dx, _t = em.tmp(FLOAT, f'{p}.x - 0.5')
    dy, _t = em.tmp(FLOAT, f'{p}.y - 0.5')
    return em.tmp(VEC3,
                  f'vec3(({dx} * {ca} - {dy} * {sa}) + 0.5 + {sx} * {t}, '
                  f'({dx} * {sa} + {dy} * {ca}) + 0.5 + {sy} * {t}, '
                  f'{p}.z)')


def e_halcyon_scanlines(em, node, _i):
    col, _t = em.tmp(VEC4, em.input(node, 'Color', VEC4))
    p, _t = em.tmp(VEC3, tex_vector(em, node, 'uv'))
    lines, _t = em.tmp(FLOAT, f'max({em.input(node, "Lines", FLOAT)}, 1.0)')
    dark, _t = em.tmp(FLOAT, f'clamp({em.input(node, "Darkness", FLOAT)}, '
                             f'0.0, 1.0)')
    thick, _t = em.tmp(FLOAT, f'clamp({em.input(node, "Thickness", FLOAT)},'
                              f' 0.0, 1.0)')
    t = _pat_time(em, node, animated=bool(prop(node, 'animate', False)))
    y, _t = em.tmp(FLOAT, f'{p}.y * {lines} - {t} * 6.0')
    on, _t = em.tmp(FLOAT, f'(({y} - floor({y})) < {thick}) ? 1.0 : 0.0')
    return em.tmp(VEC4, f'vec4({col}.rgb * (1.0 - {dark} * {on}), '
                        f'{col}.a)')


def _need_palette_fn(em, mode):
    """Inline the nearest-entry search for one named palette: unrolled,
    first-wins on ties exactly like argmin, distances summed r then g
    then b exactly as the evaluator's axis-2 sum."""
    from ..core.palette import NODE_PALETTES
    pal = NODE_PALETTES[mode]
    key = ('__pal', mode)
    fn = f'hal_pal_{mode.lower()}'
    if key not in em.once:
        em.once.add(key)
        lines = [f'vec4 {fn}(vec3 c)', '{',
                 '    float best = 1e9;',
                 '    float bi = 0.0;',
                 '    vec3 bc = vec3(0.0, 0.0, 0.0);',
                 '    vec3 e;', '    vec3 d;', '    float ds;']
        for k, entry in enumerate(pal):
            # a FIXED node palette is STRUCTURE, not a per-material
            # value: formatted exactly as the old class-level const did,
            # never marked for the R174 lifter
            _pe = list(entry)
            while len(_pe) < 3:
                _pe.append(1.0)
            lines.append('    e = vec3({:.8g}, {:.8g}, {:.8g});'.format(
                *_pe[:3]))
            lines.append('    d = c - e;')
            lines.append('    ds = d.x * d.x + d.y * d.y + d.z * d.z;')
            lines.append(f'    if (ds < best) '
                         f'{{ best = ds; bi = {_c(float(k))}; bc = e; }}')
        lines.append('    return vec4(bc, bi);')
        lines.append('}')
        em.inline.append('\n'.join(lines))
    return fn, len(pal)


def e_halcyon_palette(em, node, index):
    from ..core.palette import NODE_PALETTES
    col, _t = em.tmp(VEC4, em.input(node, 'Color', VEC4))
    mix, _t = em.tmp(FLOAT, f'clamp({em.input(node, "Mix", FLOAT)}, '
                            f'0.0, 1.0)')
    src, _t = em.tmp(VEC3, f'clamp({col}.rgb, 0.0, 1.0)')
    mode = str(prop(node, 'palette', 'EGA'))
    outs = node.get('outputs') or []
    o = outs[index] if index < len(outs) else {}
    if mode == 'RGB332':
        r, _t = em.tmp(FLOAT, f'roundEven({src}.x * 7.0)')
        g, _t = em.tmp(FLOAT, f'roundEven({src}.y * 7.0)')
        b, _t = em.tmp(FLOAT, f'roundEven({src}.z * 3.0)')
        if o.get('name') == 'Index':
            return em.tmp(FLOAT, f'({r} * 32.0 + {g} * 4.0 + {b}) / 255.0')
        snapped, _t = em.tmp(VEC3, f'vec3({r} / 7.0, {g} / 7.0, '
                                   f'{b} / 3.0)')
    else:
        if mode not in NODE_PALETTES:
            mode = 'EGA'
        fn, count = _need_palette_fn(em, mode)
        res, _t = em.tmp(VEC4, f'{fn}({src})')
        if o.get('name') == 'Index':
            return em.tmp(FLOAT,
                          f'{res}.w / {_c(float(max(count - 1, 1)))}')
        snapped, _t = em.tmp(VEC3, f'{res}.rgb')
    return em.tmp(VEC4, f'vec4({src} + ({snapped} - {src}) * {mix}, '
                        f'{col}.a)')


def e_halcyon_color_cycle(em, node, _i):
    fac = em.input(node, 'Fac', FLOAT)
    speed = em.input(node, 'Speed', FLOAT)
    steps, _t = em.tmp(FLOAT, em.input(node, 'Steps', FLOAT))
    t = _pat_time(em, node)
    phase, _t = em.tmp(FLOAT, f'{speed} * {t}')
    q, _t = em.tmp(FLOAT, f'max(floor({steps}), 1.0)')
    ph, _t = em.tmp(FLOAT, f'({steps} >= 1.0) '
                           f'? floor({phase} * {q}) / {q} : {phase}')
    out, _t = em.tmp(FLOAT, f'{fac} + {ph}')
    return em.tmp(FLOAT, f'{out} - floor({out})')


def e_halcyon_flipbook(em, node, _i):
    p, _t = em.tmp(VEC3, tex_vector(em, node, 'uv'))
    cols, _t = em.tmp(FLOAT,
                      f'max(floor({em.input(node, "Columns", FLOAT)}), '
                      f'1.0)')
    rows, _t = em.tmp(FLOAT,
                      f'max(floor({em.input(node, "Rows", FLOAT)}), 1.0)')
    rate = em.input(node, 'Rate', FLOAT)
    offset = em.input(node, 'Cell Offset', FLOAT)
    t = _pat_time(em, node)
    cell0, _t = em.tmp(FLOAT, f'floor({offset} + {t} * {rate})')
    total, _t = em.tmp(FLOAT, f'{cols} * {rows}')
    cell, _t = em.tmp(FLOAT,
                      f'{cell0} - {total} * floor({cell0} / {total})')
    cy, _t = em.tmp(FLOAT, f'floor({cell} / {cols})')
    cx, _t = em.tmp(FLOAT, f'{cell} - {cols} * {cy}')
    return em.tmp(VEC3,
                  f'vec3(({p}.x + {cx}) / {cols}, '
                  f'({p}.y + ({rows} - 1.0 - {cy})) / {rows}, {p}.z)')


def e_halcyon_uv_wave(em, node, _i):
    p, _t = em.tmp(VEC3, tex_vector(em, node, 'uv'))
    ax = em.input(node, 'Amplitude X', FLOAT)
    ay = em.input(node, 'Amplitude Y', FLOAT)
    freq = em.input(node, 'Frequency', FLOAT)
    speed = em.input(node, 'Speed', FLOAT)
    t = _pat_time(em, node)
    ph, _t = em.tmp(FLOAT, f'{t} * {speed}')
    return em.tmp(VEC3, (
        f'vec3({p}.x + sin(({p}.y * {freq} + {ph}) '
        f'* 6.28318530717959) * {ax}, '
        f'{p}.y + sin(({p}.x * {freq} + {ph} + 0.25) '
        f'* 6.28318530717959) * {ay}, {p}.z)'))


def e_halcyon_halftone(em, node, index):
    col, _t = em.tmp(VEC4, em.input(node, 'Color', VEC4))
    p, _t = em.tmp(VEC3, tex_vector(em, node, 'uv'))
    dots, _t = em.tmp(FLOAT, f'max({em.input(node, "Dots", FLOAT)}, '
                             f'1e-3)')
    ang, _t = em.tmp(FLOAT, f'{em.input(node, "Angle", FLOAT)} '
                            f'* 0.017453292519943295')
    ca, _t = em.tmp(FLOAT, f'cos({ang})')
    sa, _t = em.tmp(FLOAT, f'sin({ang})')
    # Rec.601 luma -- the NTSC weights, which is the period answer
    luma, _t = em.tmp(FLOAT, f'dot(clamp({col}.rgb, 0.0, 1.0), '
                             f'vec3(0.299, 0.587, 0.114))')
    gx, _t = em.tmp(FLOAT, f'({p}.x * {ca} - {p}.y * {sa}) * {dots}')
    gy, _t = em.tmp(FLOAT, f'({p}.x * {sa} + {p}.y * {ca}) * {dots}')
    fx, _t = em.tmp(FLOAT, f'{gx} - floor({gx}) - 0.5')
    fy, _t = em.tmp(FLOAT, f'{gy} - floor({gy}) - 0.5')
    d, _t = em.tmp(FLOAT, f'sqrt({fx} * {fx} + {fy} * {fy})')
    r, _t = em.tmp(FLOAT, f'0.70710678 * sqrt(max(1.0 - {luma}, 0.0))')
    ink, _t = em.tmp(FLOAT, f'({d} < {r}) ? 1.0 : 0.0')
    outs = node.get('outputs') or []
    o = outs[index] if index < len(outs) else {}
    if o.get('name') == 'Fac':
        return ink, FLOAT
    paper, _t = em.tmp(VEC4, em.input(node, 'Paper Color', VEC4))
    inkc, _t = em.tmp(VEC4, em.input(node, 'Ink Color', VEC4))
    return em.tmp(VEC4, f'vec4({paper}.rgb + ({inkc}.rgb - {paper}.rgb) '
                        f'* {ink}, {col}.a)')


def e_halcyon_threshold(em, node, _i):
    fac, _t = em.tmp(FLOAT, em.input(node, 'Fac', FLOAT))
    level, _t = em.tmp(FLOAT, em.input(node, 'Level', FLOAT))
    smooth, _t = em.tmp(FLOAT, em.input(node, 'Smooth', FLOAT))
    hard, _t = em.tmp(FLOAT, f'({fac} >= {level}) ? 1.0 : 0.0')
    tt, _t = em.tmp(FLOAT,
                    f'clamp(({fac} - ({level} - {smooth} * 0.5)) '
                    f'/ max({smooth}, 1e-6), 0.0, 1.0)')
    soft, _t = em.tmp(FLOAT, f'{tt} * {tt} * (3.0 - 2.0 * {tt})')
    return em.tmp(FLOAT, f'({smooth} > 1e-6) ? {soft} : {hard}')


def e_halcyon_quantize(em, node, _i):
    fac = em.input(node, 'Fac', FLOAT)
    s, _t = em.tmp(FLOAT, f'max(floor({em.input(node, "Steps", FLOAT)}), '
                          f'1.0)')
    q, _t = em.tmp(FLOAT, f'floor(clamp({fac}, 0.0, 1.0) * {s}) '
                          f'/ max({s} - 1.0, 1.0)')
    return em.tmp(FLOAT, f'clamp({q}, 0.0, 1.0)')


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
    # source REFLECTION projects the mirror direction, exactly
    # n_halcyon_matcap_uv -- BI's texco Refl
    if prop(node, 'source', 'NORMAL') == 'REFLECTION':
        p3, _t = em.tmp(VEC3, f'reflect(-hal_V, {n})')
    else:
        p3 = n
    scale, _t = em.tmp(FLOAT, em.input(node, 'Scale', FLOAT))
    # Centered + Offset, exactly `n_halcyon_matcap_uv`: origin-centred
    # output for sphere gradients, then the artist's shift
    centre = '0.0' if prop(node, 'centered', False) else '0.5'
    off, _t = em.tmp(VEC3, em.input(node, 'Offset', VEC3))
    return em.tmp(
        VEC3,
        f'vec3(dot({p3}, {right}) * 0.5 * {scale} + {centre}, '
        f'dot({p3}, {upv}) * 0.5 * {scale} + {centre}, 0.0) + {off}')


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


def e_halcyon_normal_map(em, node, _i):
    """The multi-flavour normal map: e_normal_map plus the DirectX flip."""
    col = em.input(node, 'Color', VEC4)
    strength = em.input(node, 'Strength', FLOAT)
    cv, _t = em.tmp(VEC4, col)
    if str(prop(node, 'map_type', 'OPENGL')) == 'DIRECTX':
        rgbv, _t = em.tmp(VEC3, f'vec3({cv}.r, 1.0 - {cv}.g, {cv}.b)')
    else:
        rgbv, _t = em.tmp(VEC3, f'{cv}.rgb')
    tn, _t = em.tmp(VEC3, f'{rgbv} * 2.0 - 1.0')
    n, _t = em.tmp(VEC3, 'normalize(hal_N)')
    if str(prop(node, 'space', 'TANGENT')) in ('OBJECT', 'WORLD'):
        out, _t = em.tmp(VEC3, f'normalize({tn})')
    else:
        up, _t = em.tmp(VEC3, f'(abs({n}.z) < 0.999) '
                              f'? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0)')
        t, _t = em.tmp(VEC3, f'normalize(cross({up}, {n}))')
        b, _t = em.tmp(VEC3, f'cross({n}, {t})')
        out, _t = em.tmp(VEC3, f'normalize({t} * {tn}.x + {b} * {tn}.y '
                               f'+ {n} * {tn}.z)')
    return em.tmp(VEC3, f'normalize({n} + ({out} - {n}) * {strength})')


def e_halcyon_normal_mix(em, node, _i):
    """Two normals into one, exactly as `nodeeval.n_halcyon_normal_mix`."""
    def _in(name):
        sock = next((s for s in node.get('inputs', ())
                     if s.get('name') == name), None)
        if sock and sock.get('link'):
            return em.tmp(VEC3, f'normalize({em.input(node, name, VEC3)})')[0]
        return em.tmp(VEC3, 'normalize(hal_N)')[0]

    n0, _t = em.tmp(VEC3, 'normalize(hal_N)')
    a = _in('Base')
    b = _in('Detail')
    f, _t = em.tmp(FLOAT,
                   f'clamp({em.input(node, "Factor", FLOAT)}, 0.0, 1.0)')
    mode = str(prop(node, 'mode', 'DETAIL'))
    if mode == 'MIX':
        out, _t = em.tmp(VEC3, f'{a} + ({b} - {a}) * {f}')
    elif mode == 'ADD':
        out, _t = em.tmp(VEC3, f'{a} + ({b} - {n0}) * {f}')
    else:
        axis, _t = em.tmp(VEC3, f'cross({n0}, {a})')
        s, _t = em.tmp(FLOAT, f'length({axis})')
        cth, _t = em.tmp(FLOAT, f'dot({n0}, {a})')
        ax, _t = em.tmp(VEC3, f'({s} > 1e-6) ? {axis} / max({s}, 1e-12) '
                              f': vec3(0.0)')
        rb, _t = em.tmp(VEC3,
                        f'{b} * {cth} + cross({ax}, {b}) * {s} '
                        f'+ {ax} * dot({ax}, {b}) * (1.0 - {cth})')
        rb2, _t = em.tmp(VEC3, f'({s} > 1e-6) ? normalize({rb}) '
                               f': (({cth} >= 0.0) ? {b} : -{b})')
        out, _t = em.tmp(VEC3, f'{a} + ({rb2} - {a}) * {f}')
    return em.tmp(VEC3, f'normalize({out})')


def e_halcyon_altitude_slope(em, node, index):
    """Bryce's terrain trio, off the fragment's own P and N.

    R202: when the Noise input is live (linked, or defaulted above 0)
    the three masks are wobbled by the same value noise the CPU uses
    -- hal_pt_vnoise IS patterns.value_noise. A node without the
    input, or with it at 0, emits exactly the old expressions.
    """
    n, _t = em.tmp(VEC3, 'normalize(hal_N)')
    noisy = any(s.get('name') == 'Noise'
                and (s.get('link')
                     or float(s.get('default') or 0.0) > 0.0)
                for s in node.get('inputs', ()))
    wob = None
    if noisy and index in (0, 1, 2):
        _need_prims(em)
        amt = em.input(node, 'Noise', FLOAT)
        nsc = em.input(node, 'Noise Scale', FLOAT)
        wob, _t = em.tmp(FLOAT,
                         f'(hal_pt_vnoise(hal_P * max({nsc}, 1e-4)) '
                         f'- 0.5) * max({amt}, 0.0)')
    if index == 0:
        if wob is None:
            return em.tmp(FLOAT, 'hal_P.z')
        mn0 = em.input(node, 'Minimum', FLOAT)
        mx0 = em.input(node, 'Maximum', FLOAT)
        mn0v, _t = em.tmp(FLOAT, mn0)
        span0, _t = em.tmp(FLOAT, f'(abs(({mx0}) - {mn0v}) < 1e-9) '
                                  f'? 1e-9 : (({mx0}) - {mn0v})')
        return em.tmp(FLOAT, f'(hal_P.z + {wob} * {span0})')
    if index == 1:
        mn = em.input(node, 'Minimum', FLOAT)
        mx = em.input(node, 'Maximum', FLOAT)
        mnv, _t = em.tmp(FLOAT, mn)
        span, _t = em.tmp(FLOAT, f'(abs(({mx}) - {mnv}) < 1e-9) '
                                 f'? 1e-9 : (({mx}) - {mnv})')
        base, _t = em.tmp(FLOAT,
                          f'clamp((hal_P.z - {mnv}) / {span}, 0.0, 1.0)')
        if wob is None:
            return base, FLOAT
        return em.tmp(FLOAT, f'clamp({base} + {wob}, 0.0, 1.0)')
    if index == 2:
        if wob is None:
            return em.tmp(FLOAT, f'(1.0 - abs({n}.z))')
        return em.tmp(FLOAT,
                      f'clamp((1.0 - abs({n}.z)) + {wob}, 0.0, 1.0)')
    return em.tmp(FLOAT,
                  f'(atan({n}.y, {n}.x) / 6.2831853071795864769 + 0.5)')


def e_halcyon_facing(em, node, index):
    ndv, _t = em.tmp(FLOAT, 'clamp(abs(dot(normalize(hal_N), '
                            'normalize(hal_V))), 0.0, 1.0)')
    if index == 1:
        return ndv, FLOAT
    if index == 0:
        power = em.input(node, 'Power', FLOAT)
        return em.tmp(FLOAT,
                      f'pow(1.0 - {ndv}, max({power}, 0.01))')
    ior, _t = em.tmp(FLOAT, f'max({em.input(node, "IOR", FLOAT)}, 1.0001)')
    f0, _t = em.tmp(FLOAT, f'(({ior} - 1.0) / ({ior} + 1.0)) '
                           f'* (({ior} - 1.0) / ({ior} + 1.0))')
    return em.tmp(FLOAT,
                  f'({f0} + (1.0 - {f0}) * pow(1.0 - {ndv}, 5.0))')


def e_halcyon_iridescent(em, node, index):
    """R202: the Iridescent node's GPU twin -- same hue wheel, same
    wavelength ratios, same noise lattice as the CPU."""
    ndv, _t = em.tmp(FLOAT, 'clamp(abs(dot(normalize(hal_N), '
                            'normalize(hal_V))), 0.0, 1.0)')
    t, _t2 = em.tmp(FLOAT, f'(1.0 - {ndv})')
    if index == 1:
        return t, FLOAT
    mode = str(prop(node, 'mode', 'SPECTRUM'))
    shift = em.input(node, 'Shift', FLOAT)
    scale = em.input(node, 'Scale', FLOAT)
    sat, _t3 = em.tmp(FLOAT,
                      f'clamp({em.input(node, "Saturation", FLOAT)}, '
                      f'0.0, 1.0)')

    def hue6(hexpr):
        v, _tt = em.tmp(VEC3,
                        f'clamp(abs(mod(vec3(({hexpr}) * 6.0) '
                        f'+ vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, '
                        f'0.0, 1.0)')
        return v

    if mode == 'PEARL':
        tint = em.input(node, 'Tint', VEC4)
        p, _t4 = em.tmp(FLOAT,
                        f'pow({t}, 1.0 / max({scale}, 0.05))')
        kiss = hue6(f'mod(({shift}) + {t} * 0.5, 1.0)')
        rgb, _t5 = em.tmp(
            VEC3,
            f'max((vec3(1.0) + (({tint}).rgb - vec3(1.0)) * {p}) '
            f'* (vec3(1.0) + 0.3 * {sat} * ({kiss} - vec3(0.5)) '
            f'* {p}), vec3(0.0))')
        return em.tmp(VEC4, f'vec4({rgb}, 1.0)')
    phase, _t6 = em.tmp(FLOAT, f'(({shift}) + {t} * ({scale}))')
    if mode == 'OIL':
        _need_prims(em)
        nsc = em.input(node, 'Noise Scale', FLOAT)
        phase, _t7 = em.tmp(
            FLOAT,
            f'({phase} + (hal_pt_vnoise(hal_P * max({nsc}, 1e-4)) '
            f'- 0.5) * 2.0)')
    if mode in ('THIN_FILM', 'OIL'):
        rgb, _t8 = em.tmp(
            VEC3,
            f'(vec3(0.5) + 0.5 * cos(9.42477796076938 * {phase} '
            f'* vec3(1.0, 1.282051282051282, 1.6091954022988506)))')
    else:
        rgb = hue6(f'mod({phase}, 1.0)')
    grey, _t9 = em.tmp(FLOAT,
                       f'(({rgb}.x + {rgb}.y + {rgb}.z) / 3.0)')
    out, _ta = em.tmp(VEC3,
                      f'(vec3({grey}) + ({rgb} - vec3({grey})) * {sat})')
    return em.tmp(VEC4, f'vec4({out}, 1.0)')


def e_halcyon_switch(em, node, _i):
    sw = em.input(node, 'Switch', FLOAT)
    a = em.input(node, 'A', VEC4)
    b = em.input(node, 'B', VEC4)
    return em.tmp(VEC4, f'((({sw}) > 0.5) ? ({b}) : ({a}))')


_WANG_GLSL = """
float hal_wang01(uint u)
{
    u = (u ^ 61u) ^ (u >> 16u);
    u = u * 9u;
    u = u ^ (u >> 4u);
    u = u * 668265261u;
    u = u ^ (u >> 15u);
    return float(u & 16777215u) / 16777216.0;
}
"""


def e_halcyon_random_per_object(em, node, index):
    """The object id (td.y, an exact integer float) hashed on the driver
    with the CPU's own uint32 Wang mix -- the SAME float either side."""
    if em.secondary:
        raise Unsupported('per-object random reads the object id, which '
                          'hit shading does not carry; the material '
                          'shades on the CPU in reflections')
    if '__wang' not in em.once:
        em.once.add('__wang')
        em.inline.append(_WANG_GLSL)
    seed = em.input(node, 'Seed', FLOAT)
    u, _t = em.tmp('uint',
                   f'uint(int(floor(td.y + 0.5)) '
                   f'+ int(floor(({seed}) + 0.5)) * 7919)')
    if index == 0:
        return em.tmp(FLOAT, f'hal_wang01({u})')
    r, _t = em.tmp(FLOAT, f'hal_wang01({u} ^ 1757225451u)')
    g, _t = em.tmp(FLOAT, f'hal_wang01({u} ^ 48610963u)')
    b, _t = em.tmp(FLOAT, f'hal_wang01({u} ^ 2524743835u)')
    return em.tmp(VEC4, f'vec4({r}, {g}, {b}, 1.0)')


def e_halcyon_levels(em, node, _i):
    col, _t = em.tmp(VEC4, em.input(node, 'Color', VEC4))
    black, _t = em.tmp(FLOAT, em.input(node, 'Black', FLOAT))
    white = em.input(node, 'White', FLOAT)
    gamma, _t = em.tmp(FLOAT, f'max({em.input(node, "Gamma", FLOAT)}, 1e-3)')
    omin, _t = em.tmp(FLOAT, em.input(node, 'Out Min', FLOAT))
    omax, _t = em.tmp(FLOAT, em.input(node, 'Out Max', FLOAT))
    span, _t = em.tmp(FLOAT, f'(abs(({white}) - {black}) < 1e-6) '
                             f'? 1e-6 : (({white}) - {black})')
    t, _t = em.tmp(VEC3, f'clamp(({col}.rgb - vec3({black})) / {span}, '
                         f'0.0, 1.0)')
    t2, _t = em.tmp(VEC3, f'pow({t}, vec3(1.0 / {gamma}))')
    return em.tmp(VEC4, f'vec4(vec3({omin}) + ({omax} - {omin}) * {t2}, '
                        f'{col}.a)')


def e_halcyon_smooth_step(em, node, _i):
    v = em.input(node, 'Value', FLOAT)
    a, _t = em.tmp(FLOAT, em.input(node, 'From Min', FLOAT))
    b = em.input(node, 'From Max', FLOAT)
    span, _t = em.tmp(FLOAT, f'(abs(({b}) - {a}) < 1e-9) '
                             f'? 1e-9 : (({b}) - {a})')
    t, _t = em.tmp(FLOAT, f'clamp((({v}) - {a}) / {span}, 0.0, 1.0)')
    interp = str(prop(node, 'interp', 'SMOOTH'))
    if interp == 'SMOOTHER':
        return em.tmp(FLOAT,
                      f'({t} * {t} * {t} * ({t} * ({t} * 6.0 - 15.0) '
                      f'+ 10.0))')
    if interp == 'SMOOTH':
        return em.tmp(FLOAT, f'({t} * {t} * (3.0 - 2.0 * {t}))')
    return t, FLOAT


def e_halcyon_channel_shuffle(em, node, _i):
    col, _t = em.tmp(VEC4, em.input(node, 'Color', VEC4))
    parts = []
    for i, key in enumerate(('out_r', 'out_g', 'out_b', 'out_a')):
        pick = str(prop(node, key, 'RGBA'[i]))
        if pick == 'ZERO':
            parts.append('0.0')
        elif pick == 'ONE':
            parts.append('1.0')
        else:
            parts.append(f'{col}.{pick.lower() if pick != "A" else "a"}')
    return em.tmp(VEC4, f'vec4({parts[0]}, {parts[1]}, {parts[2]}, '
                        f'{parts[3]})')


def e_halcyon_distance_mask(em, node, index):
    d, _t = em.tmp(FLOAT, 'length(hal_P - hal_eye)')
    if index == 1:
        return d, FLOAT
    start, _t = em.tmp(FLOAT, em.input(node, 'Start', FLOAT))
    end = em.input(node, 'End', FLOAT)
    span, _t = em.tmp(FLOAT, f'(abs(({end}) - {start}) < 1e-9) '
                             f'? 1e-9 : (({end}) - {start})')
    return em.tmp(FLOAT, f'clamp(({d} - {start}) / {span}, 0.0, 1.0)')


def e_halcyon_step_time(em, node, index):
    em.frame_uniforms.add('hal_frame')
    step, _t = em.tmp(FLOAT,
                      f'max({em.input(node, "Step Frames", FLOAT)}, 1.0)')
    held, _t = em.tmp(FLOAT, f'floor(hal_frame / {step}) * {step}')
    if index == 0:
        return held, FLOAT
    return em.tmp(FLOAT, f'((hal_frame - {held}) / {step})')


def e_halcyon_wave(em, node, _i):
    v = em.input(node, 'Value', FLOAT)
    freq = em.input(node, 'Frequency', FLOAT)
    ph = em.input(node, 'Phase', FLOAT)
    mn, _t = em.tmp(FLOAT, em.input(node, 'Minimum', FLOAT))
    mx = em.input(node, 'Maximum', FLOAT)
    t, _t = em.tmp(FLOAT, f'(({v}) * ({freq}) + ({ph}))')
    wave = str(prop(node, 'wave', 'SINE'))
    if wave == 'SQUARE':
        w, _t = em.tmp(FLOAT, f'((({t} - floor({t})) < 0.5) ? 1.0 : 0.0)')
    elif wave == 'TRIANGLE':
        w, _t = em.tmp(FLOAT, f'(1.0 - abs(2.0 * ({t} - floor({t})) - 1.0))')
    elif wave == 'SAW':
        w, _t = em.tmp(FLOAT, f'({t} - floor({t}))')
    else:
        w, _t = em.tmp(FLOAT,
                       f'(0.5 + 0.5 * sin({t} * 6.2831853071795864769))')
    return em.tmp(FLOAT, f'({mn} + (({mx}) - {mn}) * {w})')


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
    'HALCYON_LightMeterNode': 'it reads the lamp list at shading time; '
                              'the deferred pass would need the whole '
                              'light loop inside a node -- shades on '
                              'the CPU',
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
    'ShaderNodeVectorTransform': 'camera and object spaces need per-frame '
                                 'matrix uniforms the deferred pass does '
                                 'not bind yet',
    'ShaderNodeAmbientOcclusion': 'in-graph occlusion rays are CPU-only',
    'ShaderNodeBevel': 'in-graph geometry queries are CPU-only',
    'ShaderNodeLightFalloff': 'falloff lives on the lamps in this renderer',
    'ShaderNodeVolumeAbsorption': 'a volume node in a SURFACE chain; '
                                  'volumes march via Material Output > '
                                  'Volume (R222)',
    'ShaderNodeVolumeScatter': 'a volume node in a SURFACE chain; '
                               'volumes march via Material Output > '
                               'Volume (R222)',
    'ShaderNodeVolumePrincipled': 'a volume node in a SURFACE chain; '
                                  'volumes march via Material Output > '
                                  'Volume (R222)',
    'HALCYON_VolumeNode': 'a volume node in a SURFACE chain; volumes '
                          'march via Material Output > Volume (R222)',
    'ShaderNodeTexPointDensity': 'no volumetrics in this renderer',
    'ShaderNodeScript': 'OSL is not in this renderer; the Coded Shader '
                        'node is the native equivalent',
    'HALCYON_DepthCueNode': 'per-material distance fog wants the frame\'s '
                            'view matrix, which the deferred pass does not '
                            'bind; Render Properties\' own Fog rides the '
                            'readback on either device',
}


def e_halcyon_ramp(em, node, index):
    """The space-blended ramp, unrolled per stop pair.

    Positions and the space are props (baked); stop colours are sockets
    (may be linked). The conversion helpers live in one include; OKLCh
    and HSV take the short way round the hue circle exactly as the CPU.
    """
    _need_okramp(em)
    fac0 = em.input(node, 'Fac', FLOAT)
    fac, _t = em.tmp(FLOAT, f'clamp({fac0}, 0.0, 1.0)')
    n_stops = max(2, min(int(prop(node, 'stops', 2)), 6))
    pos = list(prop(node, 'positions', (0.0, 1.0, 0.5, 0.5, 0.5, 0.5)))
    space = str(prop(node, 'space', 'OKLAB'))
    ease = str(prop(node, 'easing', 'LINEAR'))
    stops = []
    for i in range(n_stops):
        cvar = em.input(node, f'Color {i + 1}', VEC4)
        stops.append((float(pos[i]), cvar))
    stops.sort(key=lambda s: s[0])
    sp = {'RGB': 0, 'OKLAB': 1, 'OKLCH': 2, 'HSV': 3}.get(space, 1)
    out, _t = em.tmp(VEC3, f'({stops[0][1]}).rgb')
    alpha, _t = em.tmp(FLOAT, f'({stops[0][1]}).a')
    for (p0, c0), (p1, c1) in zip(stops[:-1], stops[1:]):
        span = max(p1 - p0, 1e-6)
        t, _t2 = em.tmp(FLOAT,
                        f'clamp(({fac} - {p0!r}) / {span!r}, 0.0, 1.0)')
        if ease == 'SMOOTH':
            t2v, _t2 = em.tmp(FLOAT, f'{t} * {t} * (3.0 - 2.0 * {t})')
            t = t2v
        elif ease == 'CONSTANT':
            t2v, _t2 = em.tmp(FLOAT, f'({t} >= 1.0) ? 1.0 : 0.0')
            t = t2v
        blend, _t2 = em.tmp(VEC3, f'hal_ramp_blend(({c0}).rgb, '
                                  f'({c1}).rgb, {t}, {sp})')
        em.lines.append(f'    if ({fac} >= {p0!r}) {{')
        em.lines.append(f'        {out} = {blend};')
        em.lines.append(f'        {alpha} = ({c0}).a + '
                        f'(({c1}).a - ({c0}).a) * {t};')
        em.lines.append('    }')
    outs = node.get('outputs') or []
    o = outs[index] if index < len(outs) else {}
    if o.get('name') == 'Alpha':
        return alpha, FLOAT
    return em.tmp(VEC4, f'vec4({out}, {alpha})')


def _need_okramp(em):
    from .procedural import OKRAMP_GLSL
    if '__okramp' not in em.once:
        em.once.add('__okramp')
        em.inline.append(OKRAMP_GLSL)


def e_halcyon_blur(em, node, index):
    raise Unsupported(
        'Blur re-evaluates its input chain at shifted coordinates; that '
        're-run is CPU-only, so this material shades on the CPU')


# --------------------------------------------------- baked-LUT nodes (R206)
#
# ColorRamp and the curve nodes are all LUT nodes on the CPU: the export
# samples the ramp/curve into a 256-entry table (compat.sample_ramp /
# sample_curve) and nodeeval.lut_eval linearly interpolates it -- every
# interpolation mode, ease and HSV path is already baked into the table.
# The GLSL twins inline THE SAME table as a const array and reproduce
# lut_eval index-for-index: clamp to 0..1, scale by K-1, floor, lerp to
# the next entry. Parity is table-exact by construction.
#
# The field find behind this: 'Mountain' (an Altitude & Slope terrain
# graded through a ColorRamp) knocked every frame back to the CPU with
# "no GLSL emitter for ShaderNodeValToRGB".


def _emit_lut_f(em, values):
    """Declare float NAME[K] = float[K](...) and return (name, K)."""
    k = len(values)
    em._n += 1
    name = f'_v{em._n}'
    lits = ', '.join(f'{float(v):.8f}' for v in values)
    em.lines.append(f'    float {name}[{k}] = float[{k}]({lits});')
    return name, k


def _emit_lut_v4(em, rows):
    """Declare vec4 NAME[K] = vec4[K](...) and return (name, K)."""
    k = len(rows)
    em._n += 1
    name = f'_v{em._n}'
    lits = ', '.join(
        'vec4(' + ', '.join(f'{float(c):.8f}' for c in row[:4]) + ')'
        for row in rows)
    em.lines.append(f'    vec4 {name}[{k}] = vec4[{k}]({lits});')
    return name, k


def _lut_sample(em, lut, k, t, kind):
    """lut_eval's GLSL twin: clamp, scale by K-1, floor, lerp."""
    x, _t = em.tmp(FLOAT, f'clamp({t}, 0.0, 1.0) * {float(k - 1):.1f}')
    em._n += 1
    i0 = f'_v{em._n}'
    em.lines.append(f'    int {i0} = int(floor({x}));')
    em._n += 1
    i1 = f'_v{em._n}'
    em.lines.append(f'    int {i1} = min({i0} + 1, {k - 1});')
    return em.tmp(kind, f'mix({lut}[{i0}], {lut}[{i1}], '
                        f'{x} - float({i0}))')


def e_hair_info(em, node, index):
    """R208: the strand data rides the colour interpolant (hal_vcol)
    under the engine's one hair convention -- r intercept, g random,
    b length, a thickness -- and the graph root's 'strand' flag says
    whether this material IS hair. Both facts are batch constants, so
    the emitter reads them at plan time and the GLSL is just swizzles."""
    outs = node.get('outputs') or []
    o = (outs[index] if index < len(outs) else {}).get('name', '')
    strand = bool((em.graph or {}).get('strand'))
    if o == 'Tangent Normal':
        return em.tmp(VEC3, 'hal_N')
    if not strand:
        return em.tmp(FLOAT, '0.0')
    sw = {'Is Strand': None, 'Intercept': 'hal_vcol.r',
          'Length': 'hal_vcol.b', 'Thickness': 'hal_vcol.a',
          'Random': 'hal_vcol.g'}
    if o == 'Is Strand':
        return em.tmp(FLOAT, '1.0')
    return em.tmp(FLOAT, sw.get(o) or 'hal_vcol.r')


def e_val_to_rgb(em, node, index):
    """ShaderNodeValToRGB (the ColorRamp), from its baked 256-row LUT."""
    fac = em.input(node, 'Fac', FLOAT)
    lut = prop(node, 'lut')
    if not lut:
        # the CPU's no-table fallback: greyscale of the factor
        col, _t = em.tmp(VEC4, f'vec4({fac}, {fac}, {fac}, 1.0)')
        alpha, tk = em.tmp(FLOAT, '1.0')
    else:
        name, k = _emit_lut_v4(em, lut)
        col, _t = _lut_sample(em, name, k, fac, VEC4)
        alpha, tk = em.tmp(FLOAT, f'({col}).a')
    outs = node.get('outputs') or []
    o = outs[index] if index < len(outs) else {}
    if o.get('name') == 'Alpha':
        return alpha, FLOAT
    return col, VEC4


def e_float_curve(em, node, index):
    """ShaderNodeFloatCurve: val + (lut(val) - val) * factor."""
    val = em.input(node, 'Value', FLOAT)
    fac = em.input(node, 'Factor', FLOAT)
    lut = prop(node, 'lut')
    if not lut:
        return em.tmp(FLOAT, f'{val}')
    rows = [r[0] if isinstance(r, (list, tuple)) else r for r in lut]
    name, k = _emit_lut_f(em, rows)
    smp, _t = _lut_sample(em, name, k, val, FLOAT)
    return em.tmp(FLOAT, f'{val} + ({smp} - {val}) * ({fac})')


def e_rgb_curve(em, node, index):
    """ShaderNodeRGBCurve: combined curve, then each channel's own.

    The LUT is (256, 4): column 0 the combined C curve, columns 1..3
    the per-channel curves -- v = C(col.ch); out.ch = CH(v), exactly
    nodeeval.n_rgb_curve. Alpha passes through; Fac lerps at the end.
    """
    col = em.input(node, 'Color', VEC4)
    fac = em.input(node, 'Fac', FLOAT)
    lut = prop(node, 'lut')
    if not lut:
        return em.tmp(VEC4, f'{col}')
    name, k = _emit_lut_v4(em, lut)
    chans = []
    for ch, sw in enumerate(('r', 'g', 'b')):
        v, _t = _lut_sample(em, name, k, f'({col}).{sw}', VEC4)
        comb, _t = em.tmp(FLOAT, f'({v}).x')
        o, _t = _lut_sample(em, name, k, comb, VEC4)
        oc, _t = em.tmp(FLOAT, f'({o}).{"xyzw"[ch + 1]}')
        chans.append(oc)
    out, _t = em.tmp(VEC4, f'vec4({chans[0]}, {chans[1]}, {chans[2]}, '
                           f'({col}).a)')
    return em.tmp(VEC4, f'{col} + ({out} - {col}) * ({fac})')


# ------------------------------------------- the R216 families' GLSL twins

def _o_name(node, index):
    outs = node.get('outputs') or []
    return (outs[index] if index < len(outs) else {}).get('name', '')


def _time_of(em, node, gate='animate', default=True):
    if prop(node, gate, default):
        em.frame_uniforms.add('hal_time')
        return 'hal_time'
    return '0.0'


def e_u16_timer(em, node, index):
    em.frame_uniforms.add('hal_frame')
    nm = _o_name(node, index)
    if nm == 'Seconds':
        em.frame_uniforms.add('hal_time')
        return em.tmp(FLOAT, 'hal_time')
    if nm == 'Loop':
        lf = em.input(node, 'Loop Frames', FLOAT)
        return em.tmp(FLOAT, f'fract(hal_frame / max({lf}, 1.0))')
    return em.tmp(FLOAT, 'hal_frame')


def e_u16_oscillator(em, node, index):
    em.frame_uniforms.add('hal_time')
    t, _t = em.tmp(FLOAT, f'hal_time * {em.input(node, "Speed", FLOAT)}'
                          f' + {em.input(node, "Phase", FLOAT)}')
    wave = str(prop(node, 'wave', 'SINE'))
    if wave == 'SQUARE':
        w, _t = em.tmp(FLOAT, f'(fract({t}) < 0.5) ? 1.0 : 0.0')
    elif wave == 'TRIANGLE':
        w, _t = em.tmp(FLOAT, f'1.0 - abs(2.0 * fract({t}) - 1.0)')
    elif wave == 'SAW':
        w, _t = em.tmp(FLOAT, f'fract({t})')
    else:
        w, _t = em.tmp(FLOAT, f'sin({t} * 6.28318530717959) * 0.5 + 0.5')
    lo = em.input(node, 'Min', FLOAT)
    hi = em.input(node, 'Max', FLOAT)
    return em.tmp(FLOAT, f'{lo} + ({hi} - {lo}) * {w}')


def e_u16_counter(em, node, index):
    em.frame_uniforms.add('hal_frame')
    st = em.input(node, 'Frames Per Step', FLOAT)
    md = em.input(node, 'Modulo', FLOAT)
    off = em.input(node, 'Offset', FLOAT)
    return em.tmp(FLOAT, f'mod(floor(hal_frame / max({st}, 1.0)) '
                         f'+ {off}, max({md}, 1.0))')


def e_u16_pulse(em, node, index):
    em.frame_uniforms.add('hal_frame')
    per = em.input(node, 'Period', FLOAT)
    wid = em.input(node, 'Width', FLOAT)
    ph = em.input(node, 'Phase', FLOAT)
    return em.tmp(FLOAT, f'(mod(hal_frame - {ph}, max({per}, 1.0)) '
                         f'< {wid}) ? 1.0 : 0.0')


def e_u16_gate(em, node, index):
    v = em.input(node, 'Value', FLOAT)
    th = em.input(node, 'Threshold', FLOAT)
    so, _t = em.tmp(FLOAT, f'max({em.input(node, "Softness", FLOAT)}, '
                           '0.0)')
    tt, _t = em.tmp(FLOAT, f'clamp(({v} - ({th} - {so})) / '
                           f'max(2.0 * {so}, 1e-9), 0.0, 1.0)')
    out, _t = em.tmp(FLOAT, f'({so} > 0.0) ? ({tt} * {tt} * '
                            f'(3.0 - 2.0 * {tt})) '
                            f': (({v} >= {th}) ? 1.0 : 0.0)')
    if prop(node, 'invert', False):
        return em.tmp(FLOAT, f'1.0 - {out}')
    return out, FLOAT


def e_u16_selector(em, node, index):
    idx, _t = em.tmp(FLOAT, f'floor({em.input(node, "Index", FLOAT)})')
    if prop(node, 'wrap', True):
        i, _t = em.tmp(FLOAT, f'mod(mod({idx}, 4.0) + 4.0, 4.0)')
    else:
        i, _t = em.tmp(FLOAT, f'clamp({idx}, 0.0, 3.0)')
    c1 = em.input(node, 'Color 1', VEC4)
    c2 = em.input(node, 'Color 2', VEC4)
    c3 = em.input(node, 'Color 3', VEC4)
    c4 = em.input(node, 'Color 4', VEC4)
    return em.tmp(VEC4, f'({i} < 0.5) ? {c1} : (({i} < 1.5) ? {c2} : '
                        f'(({i} < 2.5) ? {c3} : {c4}))')


def e_u16_color_key(em, node, index):
    col, _t = em.tmp(VEC4, em.input(node, 'Color', VEC4))
    key, _t = em.tmp(VEC4, em.input(node, 'Key', VEC4))
    tol = em.input(node, 'Tolerance', FLOAT)
    so, _t = em.tmp(FLOAT, f'max({em.input(node, "Softness", FLOAT)}, '
                           '1e-6)')
    d, _t = em.tmp(FLOAT, f'distance({col}.rgb, {key}.rgb)')
    tt, _t = em.tmp(FLOAT, f'clamp(({d} - max({tol}, 0.0)) / {so}, '
                           '0.0, 1.0)')
    fac, _t = em.tmp(FLOAT, f'1.0 - {tt} * {tt} * (3.0 - 2.0 * {tt})')
    if _o_name(node, index) == 'Matte':
        return em.tmp(FLOAT, f'1.0 - {fac}')
    return fac, FLOAT


def e_u16_measure(em, node, index):
    mode = str(prop(node, 'mode', 'ORIGIN'))
    sc, _t = em.tmp(FLOAT, f'max({em.input(node, "Scale", FLOAT)}, '
                           '1e-9)')
    if mode == 'POINT':
        ref = em.input(node, 'Point', VEC3)
        d, _t = em.tmp(FLOAT, f'distance(hal_P, {ref})')
    elif mode == 'CAMERA':
        d, _t = em.tmp(FLOAT, 'distance(hal_P, hal_eye)')
    elif mode in ('AXIS_X', 'AXIS_Y', 'AXIS_Z'):
        d, _t = em.tmp(FLOAT, 'hal_P.' + mode[-1].lower())
    else:
        d, _t = em.tmp(FLOAT, 'distance(hal_P, hal_object_loc)')
    return em.tmp(FLOAT, f'{d} / {sc}')


def e_u16_step_ramp(em, node, index):
    f, _t = em.tmp(FLOAT, f'clamp({em.input(node, "Fac", FLOAT)}, '
                          '0.0, 1.0)')
    st, _t = em.tmp(FLOAT, f'max({em.input(node, "Steps", FLOAT)}, 1.0)')
    band, _t = em.tmp(FLOAT, f'min(floor({f} * {st}), {st} - 1.0)')
    if _o_name(node, index) == 'Band':
        return band, FLOAT
    q, _t = em.tmp(FLOAT, f'{band} / max({st} - 1.0, 1.0)')
    a = em.input(node, 'Color 1', VEC4)
    b = em.input(node, 'Color 2', VEC4)
    return em.tmp(VEC4, f'mix({a}, {b}, {q})')


def e_u16_wobble(em, node, index):
    _need_prims(em)
    em.frame_uniforms.add('hal_time')
    v = em.input(node, 'Value', FLOAT)
    amt = em.input(node, 'Amount', FLOAT)
    sp = em.input(node, 'Speed', FLOAT)
    seed = float(int(prop(node, 'seed', 0)) * 13.7)
    t, _t = em.tmp(FLOAT, f'hal_time * {sp} + {seed!r}')
    w, _t = em.tmp(FLOAT,
                   f'hal_pt_vnoise(vec3({t}, 0.37, 0.61))')
    return em.tmp(FLOAT, f'{v} + ({w} * 2.0 - 1.0) * {amt}')


def e_u16_frame_blend(em, node, index):
    em.frame_uniforms.add('hal_frame')
    s0 = em.input(node, 'Start Frame', FLOAT)
    e0 = em.input(node, 'End Frame', FLOAT)
    t, _t = em.tmp(FLOAT, f'clamp((hal_frame - {s0}) / '
                          f'max({e0} - {s0}, 1e-6), 0.0, 1.0)')
    if _o_name(node, index) == 'Fac':
        return t, FLOAT
    a = em.input(node, 'A', VEC4)
    b = em.input(node, 'B', VEC4)
    return em.tmp(VEC4, f'mix({a}, {b}, {t})')


def e_u16_blackbody(em, node, index):
    k, _t = em.tmp(FLOAT, f'clamp({em.input(node, "Kelvin", FLOAT)}, '
                          '1000.0, 12000.0)')
    t, _t = em.tmp(FLOAT, f'{k} / 100.0')
    r, _t = em.tmp(FLOAT, f'({t} <= 66.0) ? 1.0 : clamp(1.292936 * '
                          f'pow(max({t} - 60.0, 1e-3), -0.1332047), 0.0, 1.0)')
    g, _t = em.tmp(FLOAT, f'({t} <= 66.0) ? clamp(0.3900816 * '
                          f'log(max({t}, 1e-3)) - 0.6318414, 0.0, 1.0) '
                          f': clamp(1.129891 * pow(max({t} - 60.0, 1e-3), '
                          '-0.0755148), 0.0, 1.0)')
    b, _t = em.tmp(FLOAT, f'({t} >= 66.0) ? 1.0 : (({t} <= 19.0) ? 0.0 '
                          f': clamp(0.5432068 * log(max({t} - 10.0, '
                          '1e-3)) - 1.19625, 0.0, 1.0))')
    return em.tmp(VEC4, f'vec4({r}, {g}, {b}, 1.0)')


def e_u16_compare(em, node, index):
    a = em.input(node, 'A', FLOAT)
    b = em.input(node, 'B', FLOAT)
    eps, _t = em.tmp(FLOAT, f'max({em.input(node, "Epsilon", FLOAT)}, '
                            '0.0)')
    eq, _t = em.tmp(FLOAT, f'(abs({a} - {b}) <= {eps}) ? 1.0 : 0.0')
    nm = _o_name(node, index)
    if nm == 'Greater':
        return em.tmp(FLOAT, f'(({a} > {b}) && ({eq} < 0.5)) '
                             '? 1.0 : 0.0')
    if nm == 'Less':
        return em.tmp(FLOAT, f'(({a} < {b}) && ({eq} < 0.5)) '
                             '? 1.0 : 0.0')
    return eq, FLOAT


def e_u16_on_frame(em, node, index):
    em.frame_uniforms.add('hal_frame')
    s0 = em.input(node, 'Start Frame', FLOAT)
    e0 = em.input(node, 'End Frame', FLOAT)
    return em.tmp(FLOAT, f'((hal_frame >= {s0}) && (hal_frame <= {e0}))'
                         ' ? 1.0 : 0.0')


# --- the vector family ----------------------------------------------------


def _v16_uv(em, node):
    v = tex_vector(em, node, 'uv')
    out, _t = em.tmp(VEC3, v)
    return out


def e_v16_array(em, node, index):
    _need_prims(em)
    v = _v16_uv(em, node)
    ctr = em.input(node, 'Center', VEC3)
    size, _t = em.tmp(FLOAT, f'max({em.input(node, "Size", FLOAT)}, '
                             '1e-6)')
    rot = em.input(node, 'Rotation', FLOAT)
    jit = em.input(node, 'Jitter', FLOAT)
    mode = str(prop(node, 'mode', 'CIRCLE'))
    count = float(max(int(prop(node, 'count', 8)), 1))
    sides = float(max(int(prop(node, 'sides', 5)), 2))
    orient = bool(prop(node, 'orient', True))
    x, _t = em.tmp(FLOAT, f'{v}.x - {ctr}.x')
    y, _t = em.tmp(FLOAT, f'{v}.y - {ctr}.y')
    if mode == 'LINE':
        sp, _t = em.tmp(FLOAT, f'max({em.input(node, "Spacing", FLOAT)}'
                               ', 1e-6)')
        ca, _t = em.tmp(FLOAT, f'cos({rot})')
        sa, _t = em.tmp(FLOAT, f'sin({rot})')
        tl, _t = em.tmp(FLOAT, f'{x} * {ca} + {y} * {sa}')
        pp, _t = em.tmp(FLOAT, f'-{x} * {sa} + {y} * {ca}')
        idx, _t = em.tmp(FLOAT, f'clamp(floor({tl} / {sp} + 0.5), 0.0, '
                                f'{count - 1.0!r})')
        lx, _t = em.tmp(FLOAT, f'{tl} - {idx} * {sp}')
        ly, _t = em.tmp(FLOAT, pp)
        ang, _t = em.tmp(FLOAT, '0.0')
    elif mode == 'GRID':
        sp, _t = em.tmp(FLOAT, f'max({em.input(node, "Spacing", FLOAT)}'
                               ', 1e-6)')
        gi, _t = em.tmp(FLOAT, f'clamp(floor({x} / {sp} + 0.5), 0.0, '
                               f'{count - 1.0!r})')
        gj, _t = em.tmp(FLOAT, f'clamp(floor({y} / {sp} + 0.5), 0.0, '
                               f'{sides - 1.0!r})')
        lx, _t = em.tmp(FLOAT, f'{x} - {gi} * {sp}')
        ly, _t = em.tmp(FLOAT, f'{y} - {gj} * {sp}')
        idx, _t = em.tmp(FLOAT, f'{gj} * {count!r} + {gi}')
        ang, _t = em.tmp(FLOAT, '0.0')
    else:
        rad, _t = em.tmp(FLOAT, f'max({em.input(node, "Radius", FLOAT)}'
                                ', 1e-6)')
        th, _t = em.tmp(FLOAT, f'atan({y}, {x})')
        if mode == 'CIRCLE':
            step = 6.28318530717959 / count
            idx, _t = em.tmp(FLOAT, f'mod(floor({th} / {step!r} + 0.5) '
                                    f'+ {count!r}, {count!r})')
            aa, _t = em.tmp(FLOAT, f'{idx} * {step!r}')
            cx, _t = em.tmp(FLOAT, f'cos({aa}) * {rad}')
            cy, _t = em.tmp(FLOAT, f'sin({aa}) * {rad}')
            ang0, _t = em.tmp(FLOAT, f'{aa} + 1.5707963267949')
        elif mode == 'SQUARE':
            per, _t = em.tmp(FLOAT, f'fract({th} / 6.28318530717959 '
                                    '+ 0.5)')
            idx, _t = em.tmp(FLOAT, f'mod(floor({per} * {count!r} '
                                    f'+ 0.5), {count!r})')
            tp, _t = em.tmp(FLOAT, f'({idx} / {count!r}) * 4.0')
            side, _t = em.tmp(FLOAT, f'floor({tp})')
            ft, _t = em.tmp(FLOAT, f'{tp} - {side}')
            e0, _t = em.tmp(FLOAT, f'2.0 * {ft} - 1.0')
            cx, _t = em.tmp(FLOAT, f'(({side} < 0.5) ? 1.0 : '
                                   f'(({side} < 1.5) ? -{e0} : '
                                   f'(({side} < 2.5) ? -1.0 : {e0}))) '
                                   f'* {rad}')
            cy, _t = em.tmp(FLOAT, f'(({side} < 0.5) ? {e0} : '
                                   f'(({side} < 1.5) ? 1.0 : '
                                   f'(({side} < 2.5) ? -{e0} : -1.0))) '
                                   f'* {rad}')
            ang0, _t = em.tmp(FLOAT, f'atan({cy}, {cx}) '
                                     '+ 1.5707963267949')
        else:
            pts = sides if mode == 'POLYGON' else sides * 2.0
            step = 6.28318530717959 / pts
            idx, _t = em.tmp(FLOAT, f'mod(floor({th} / {step!r} + 0.5) '
                                    f'+ {pts!r}, {pts!r})')
            aa, _t = em.tmp(FLOAT, f'{idx} * {step!r}')
            if mode == 'STAR':
                inner = em.input(node, 'Inner', FLOAT)
                rr, _t = em.tmp(FLOAT, f'(mod({idx}, 2.0) > 0.5) ? '
                                       f'({rad} * clamp({inner}, 0.05, '
                                       f'1.0)) : {rad}')
            else:
                rr = rad
            cx, _t = em.tmp(FLOAT, f'cos({aa}) * {rr}')
            cy, _t = em.tmp(FLOAT, f'sin({aa}) * {rr}')
            ang0, _t = em.tmp(FLOAT, f'{aa} + 1.5707963267949')
        jx, _t = em.tmp(FLOAT, f'(hal_pt_hash3(int({idx}), 3, 11) '
                               f'- 0.5) * {jit} * {rad}')
        jy, _t = em.tmp(FLOAT, f'(hal_pt_hash3(int({idx}), 7, 23) '
                               f'- 0.5) * {jit} * {rad}')
        cx2, _t = em.tmp(FLOAT, f'{cx} + {jx}')
        cy2, _t = em.tmp(FLOAT, f'{cy} + {jy}')
        lx, _t = em.tmp(FLOAT, f'{x} - {cx2}')
        ly, _t = em.tmp(FLOAT, f'{y} - {cy2}')
        ang, _t = em.tmp(FLOAT, (ang0 if orient else '0.0'))
    if mode in ('LINE', 'GRID'):
        jlx, _t = em.tmp(FLOAT, f'(hal_pt_hash3(int({idx}), 3, 11) '
                                f'- 0.5) * {jit}')
        jly, _t = em.tmp(FLOAT, f'(hal_pt_hash3(int({idx}), 7, 23) '
                                f'- 0.5) * {jit}')
        lx, _t = em.tmp(FLOAT, f'{lx} + {jlx}')
        ly, _t = em.tmp(FLOAT, f'{ly} + {jly}')
    nm = _o_name(node, index)
    if nm == 'Index':
        return idx, FLOAT
    if nm == 'Random':
        return em.tmp(FLOAT, f'hal_pt_hash3(int({idx}), 0, 97)')
    aa2, _t = em.tmp(FLOAT, f'{ang} + {rot}' if mode not in
                     ('LINE', 'GRID') else f'{ang}')
    ca2, _t = em.tmp(FLOAT, f'cos(-{aa2})')
    sa2, _t = em.tmp(FLOAT, f'sin(-{aa2})')
    ox, _t = em.tmp(FLOAT, f'({lx} * {ca2} - {ly} * {sa2}) / {size} '
                           '+ 0.5')
    oy, _t = em.tmp(FLOAT, f'({lx} * {sa2} + {ly} * {ca2}) / {size} '
                           '+ 0.5')
    return em.tmp(VEC3, f'vec3({ox}, {oy}, 0.0)')


def e_v16_mirror_tile(em, node, index):
    v = _v16_uv(em, node)
    sc, _t = em.tmp(FLOAT, f'max({em.input(node, "Scale", FLOAT)}, '
                           '1e-9)')
    parts = []
    for comp, on in (('x', prop(node, 'axis_x', True)),
                     ('y', prop(node, 'axis_y', True))):
        q, _t = em.tmp(FLOAT, f'{v}.{comp} * {sc}')
        if on:
            h, _t = em.tmp(FLOAT, f'fract({q} * 0.5) * 2.0')
            r, _t = em.tmp(FLOAT, f'1.0 - abs({h} - 1.0)')
        else:
            r, _t = em.tmp(FLOAT, f'fract({q})')
        parts.append(r)
    return em.tmp(VEC3, f'vec3({parts[0]}, {parts[1]}, {v}.z)')


def e_v16_kaleidoscope(em, node, index):
    v = _v16_uv(em, node)
    ctr = em.input(node, 'Center', VEC3)
    sec, _t = em.tmp(FLOAT, f'max({em.input(node, "Sectors", FLOAT)}, '
                            '1.0)')
    base = em.input(node, 'Angle', FLOAT)
    x, _t = em.tmp(FLOAT, f'{v}.x - {ctr}.x')
    y, _t = em.tmp(FLOAT, f'{v}.y - {ctr}.y')
    r, _t = em.tmp(FLOAT, f'sqrt({x} * {x} + {y} * {y})')
    a, _t = em.tmp(FLOAT, f'atan({y}, {x}) - {base}')
    st, _t = em.tmp(FLOAT, f'6.28318530717959 / {sec}')
    k, _t = em.tmp(FLOAT, f'mod(mod({a}, 2.0 * {st}) + 2.0 * {st}, '
                          f'2.0 * {st})')
    fold, _t = em.tmp(FLOAT, f'(({k} > {st}) ? (2.0 * {st} - {k}) '
                             f': {k}) + {base}')
    return em.tmp(VEC3, f'vec3(cos({fold}) * {r} + {ctr}.x, '
                        f'sin({fold}) * {r} + {ctr}.y, {v}.z)')


def e_v16_polar(em, node, index):
    v = _v16_uv(em, node)
    ctr = em.input(node, 'Center', VEC3)
    nm = _o_name(node, index)
    if str(prop(node, 'direction', 'TO_POLAR')) == 'FROM_POLAR':
        a, _t = em.tmp(FLOAT, f'{v}.x * 6.28318530717959')
        r, _t = em.tmp(FLOAT, f'{v}.y')
        if nm == 'Radius':
            return r, FLOAT
        if nm == 'Angle':
            return em.tmp(FLOAT, f'{v}.x')
        return em.tmp(VEC3, f'vec3(cos({a}) * {r} + {ctr}.x, '
                            f'sin({a}) * {r} + {ctr}.y, {v}.z)')
    x, _t = em.tmp(FLOAT, f'{v}.x - {ctr}.x')
    y, _t = em.tmp(FLOAT, f'{v}.y - {ctr}.y')
    r, _t = em.tmp(FLOAT, f'sqrt({x} * {x} + {y} * {y})')
    a, _t = em.tmp(FLOAT, f'fract(atan({y}, {x}) / 6.28318530717959 '
                          '+ 1.0)')
    if nm == 'Radius':
        return r, FLOAT
    if nm == 'Angle':
        return a, FLOAT
    return em.tmp(VEC3, f'vec3({a}, {r}, {v}.z)')


def e_v16_twirl(em, node, index):
    v = _v16_uv(em, node)
    ctr = em.input(node, 'Center', VEC3)
    ang = em.input(node, 'Angle', FLOAT)
    rad, _t = em.tmp(FLOAT, f'max({em.input(node, "Radius", FLOAT)}, '
                            '1e-6)')
    x, _t = em.tmp(FLOAT, f'{v}.x - {ctr}.x')
    y, _t = em.tmp(FLOAT, f'{v}.y - {ctr}.y')
    r, _t = em.tmp(FLOAT, f'sqrt({x} * {x} + {y} * {y})')
    fall, _t = em.tmp(FLOAT, f'clamp(1.0 - {r} / {rad}, 0.0, 1.0)')
    a, _t = em.tmp(FLOAT, f'{ang} * {fall} * {fall}')
    ca, _t = em.tmp(FLOAT, f'cos({a})')
    sa, _t = em.tmp(FLOAT, f'sin({a})')
    return em.tmp(VEC3, f'vec3({x} * {ca} - {y} * {sa} + {ctr}.x, '
                        f'{x} * {sa} + {y} * {ca} + {ctr}.y, {v}.z)')


def e_v16_lens(em, node, index):
    v = _v16_uv(em, node)
    ctr = em.input(node, 'Center', VEC3)
    amt = em.input(node, 'Amount', FLOAT)
    x, _t = em.tmp(FLOAT, f'{v}.x - {ctr}.x')
    y, _t = em.tmp(FLOAT, f'{v}.y - {ctr}.y')
    f, _t = em.tmp(FLOAT, f'1.0 + {amt} * ({x} * {x} + {y} * {y})')
    return em.tmp(VEC3, f'vec3({x} * {f} + {ctr}.x, '
                        f'{y} * {f} + {ctr}.y, {v}.z)')


def e_v16_ripple_warp(em, node, index):
    v = _v16_uv(em, node)
    ctr = em.input(node, 'Center', VEC3)
    amp = em.input(node, 'Amplitude', FLOAT)
    fr = em.input(node, 'Frequency', FLOAT)
    sp = em.input(node, 'Speed', FLOAT)
    t = _time_of(em, node)
    x, _t = em.tmp(FLOAT, f'{v}.x - {ctr}.x')
    y, _t = em.tmp(FLOAT, f'{v}.y - {ctr}.y')
    r, _t = em.tmp(FLOAT, f'sqrt({x} * {x} + {y} * {y})')
    w, _t = em.tmp(FLOAT, f'sin(({r} * {fr} - {t} * {sp}) '
                          f'* 6.28318530717959) * {amp}')
    sf, _t = em.tmp(FLOAT, f'max({r}, 1e-9)')
    return em.tmp(VEC3, f'vec3({v}.x + ({x} / {sf}) * {w}, '
                        f'{v}.y + ({y} / {sf}) * {w}, {v}.z)')


def e_v16_wave_warp(em, node, index):
    v = _v16_uv(em, node)
    amp = em.input(node, 'Amplitude', FLOAT)
    wl, _t = em.tmp(FLOAT, f'max({em.input(node, "Wavelength", FLOAT)}'
                           ', 1e-6)')
    sp = em.input(node, 'Speed', FLOAT)
    t = _time_of(em, node)
    if str(prop(node, 'axis', 'X')) == 'Y':
        return em.tmp(VEC3, f'vec3({v}.x + sin(({v}.y / {wl} + {t} * '
                            f'{sp}) * 6.28318530717959) * {amp}, '
                            f'{v}.y, {v}.z)')
    return em.tmp(VEC3, f'vec3({v}.x, {v}.y + sin(({v}.x / {wl} '
                        f'+ {t} * {sp}) * 6.28318530717959) * {amp}, '
                        f'{v}.z)')


def e_v16_tile_random(em, node, index):
    _need_prims(em)
    v = _v16_uv(em, node)
    sc, _t = em.tmp(FLOAT, f'max({em.input(node, "Scale", FLOAT)}, '
                           '1e-9)')
    qx, _t = em.tmp(FLOAT, f'{v}.x * {sc}')
    qy, _t = em.tmp(FLOAT, f'{v}.y * {sc}')
    cx, _t = em.tmp(FLOAT, f'floor({qx})')
    cy, _t = em.tmp(FLOAT, f'floor({qy})')
    tid, _t = em.tmp(FLOAT, f'hal_pt_hash3(int({cx}), int({cy}), 53)')
    if _o_name(node, index) == 'Tile ID':
        return tid, FLOAT
    fx, _t = em.tmp(FLOAT, f'{qx} - {cx}')
    fy, _t = em.tmp(FLOAT, f'{qy} - {cy}')
    if prop(node, 'rotate', True):
        rs, _t = em.tmp(FLOAT, f'mod(floor(hal_pt_hash3(int({cx}), '
                               f'int({cy}), 11) * 4.0), 4.0)')
        ox, _t = em.tmp(FLOAT, f'{fx} - 0.5')
        oy, _t = em.tmp(FLOAT, f'{fy} - 0.5')
        fx, _t = em.tmp(FLOAT, f'(({rs} < 0.5) ? {ox} : (({rs} < 1.5) '
                               f'? -{oy} : (({rs} < 2.5) ? -{ox} : '
                               f'{oy}))) + 0.5')
        fy, _t = em.tmp(FLOAT, f'(({rs} < 0.5) ? {oy} : (({rs} < 1.5) '
                               f'? {ox} : (({rs} < 2.5) ? -{oy} : '
                               f'-{ox}))) + 0.5')
    if prop(node, 'flip', True):
        fs, _t = em.tmp(FLOAT, f'hal_pt_hash3(int({cx}), int({cy}), '
                               '29)')
        fx, _t = em.tmp(FLOAT, f'({fs} < 0.5) ? (1.0 - {fx}) : {fx}')
    return em.tmp(VEC3, f'vec3({fx}, {fy}, {v}.z)')


def e_v16_vector_snap(em, node, index):
    v = _v16_uv(em, node)
    st, _t = em.tmp(FLOAT, f'max({em.input(node, "Step", FLOAT)}, '
                           '1e-9)')
    fn = 'floor' if str(prop(node, 'mode', 'FLOOR')) == 'FLOOR' \
        else 'floor'
    if str(prop(node, 'mode', 'FLOOR')) == 'ROUND':
        return em.tmp(VEC3, f'vec3(floor({v}.x / {st} + 0.5) * {st}, '
                            f'floor({v}.y / {st} + 0.5) * {st}, {v}.z)')
    return em.tmp(VEC3, f'vec3({fn}({v}.x / {st}) * {st}, '
                        f'{fn}({v}.y / {st}) * {st}, {v}.z)')


def e_v16_shear(em, node, index):
    v = _v16_uv(em, node)
    xy = em.input(node, 'X By Y', FLOAT)
    yx = em.input(node, 'Y By X', FLOAT)
    return em.tmp(VEC3, f'vec3({v}.x + {v}.y * {xy}, '
                        f'{v}.y + {v}.x * {yx}, {v}.z)')


def e_v16_orbit(em, node, index):
    v = _v16_uv(em, node)
    rad = em.input(node, 'Radius', FLOAT)
    sp = em.input(node, 'Speed', FLOAT)
    ph = em.input(node, 'Phase', FLOAT)
    t = _time_of(em, node)
    a, _t = em.tmp(FLOAT, f'({t} * {sp} + {ph}) * 6.28318530717959')
    return em.tmp(VEC3, f'vec3({v}.x + cos({a}) * {rad}, '
                        f'{v}.y + sin({a}) * {rad}, {v}.z)')


def e_v16_region(em, node, index):
    v = _v16_uv(em, node)
    lo, _t = em.tmp(VEC3, em.input(node, 'Min', VEC3))
    hi, _t = em.tmp(VEC3, em.input(node, 'Max', VEC3))
    spx, _t = em.tmp(FLOAT, f'max({hi}.x - {lo}.x, 1e-9)')
    spy, _t = em.tmp(FLOAT, f'max({hi}.y - {lo}.y, 1e-9)')
    qx, _t = em.tmp(FLOAT, f'({v}.x - {lo}.x) / {spx}')
    qy, _t = em.tmp(FLOAT, f'({v}.y - {lo}.y) / {spy}')
    if _o_name(node, index) == 'Inside':
        return em.tmp(FLOAT, f'(({qx} >= 0.0) && ({qx} <= 1.0) && '
                             f'({qy} >= 0.0) && ({qy} <= 1.0)) '
                             '? 1.0 : 0.0')
    mode = str(prop(node, 'outside', 'CLIP'))
    if mode == 'WRAP':
        qx, _t = em.tmp(FLOAT, f'fract({qx})')
        qy, _t = em.tmp(FLOAT, f'fract({qy})')
    elif mode == 'MIRROR':
        hx, _t = em.tmp(FLOAT, f'fract({qx} * 0.5) * 2.0')
        hy, _t = em.tmp(FLOAT, f'fract({qy} * 0.5) * 2.0')
        qx, _t = em.tmp(FLOAT, f'1.0 - abs({hx} - 1.0)')
        qy, _t = em.tmp(FLOAT, f'1.0 - abs({hy} - 1.0)')
    else:
        qx, _t = em.tmp(FLOAT, f'clamp({qx}, 0.0, 1.0)')
        qy, _t = em.tmp(FLOAT, f'clamp({qy}, 0.0, 1.0)')
    return em.tmp(VEC3, f'vec3({lo}.x + {qx} * {spx}, '
                        f'{lo}.y + {qy} * {spy}, {v}.z)')


def e_v16_projector(em, node, index):
    v0 = tex_vector(em, node, 'generated')
    v, _t = em.tmp(VEC3, v0)
    sc, _t = em.tmp(FLOAT, f'max({em.input(node, "Scale", FLOAT)}, '
                           '1e-9)')
    axis = str(prop(node, 'axis', 'Z'))
    comp = {'X': ('y', 'z', 'x'), 'Y': ('x', 'z', 'y'),
            'Z': ('x', 'y', 'z')}[axis]
    u, _t = em.tmp(FLOAT, f'{v}.{comp[0]}')
    w, _t = em.tmp(FLOAT, f'{v}.{comp[1]}')
    h, _t = em.tmp(FLOAT, f'{v}.{comp[2]}')
    mode = str(prop(node, 'mode', 'PLANAR'))
    if mode == 'CYLINDER':
        a, _t = em.tmp(FLOAT, f'fract(atan({w}, {u}) '
                              '/ 6.28318530717959 + 1.0)')
        return em.tmp(VEC3, f'vec3({a} / {sc}, {h} / {sc}, 0.0)')
    if mode == 'SPHERE':
        r, _t = em.tmp(FLOAT, f'length(vec3({u}, {w}, {h}))')
        a, _t = em.tmp(FLOAT, f'fract(atan({w}, {u}) '
                              '/ 6.28318530717959 + 1.0)')
        el, _t = em.tmp(FLOAT, f'acos(clamp({h} / max({r}, 1e-9), '
                               '-1.0, 1.0)) / 3.14159265358979')
        return em.tmp(VEC3, f'vec3({a} / {sc}, (1.0 - {el}) / {sc}, '
                            '0.0)')
    if mode == 'BOX':
        ax_, _t = em.tmp(VEC3, f'abs({v})')
        pu, _t = em.tmp(FLOAT, f'({ax_}.x >= max({ax_}.y, {ax_}.z)) '
                               f'? {v}.y : (({ax_}.y >= {ax_}.z) '
                               f'? {v}.x : {v}.x)')
        pw, _t = em.tmp(FLOAT, f'({ax_}.x >= max({ax_}.y, {ax_}.z)) '
                               f'? {v}.z : (({ax_}.y >= {ax_}.z) '
                               f'? {v}.z : {v}.y)')
        return em.tmp(VEC3, f'vec3({pu} / {sc}, {pw} / {sc}, 0.0)')
    return em.tmp(VEC3, f'vec3({u} / {sc}, {w} / {sc}, 0.0)')


def e_v16_spin(em, node, index):
    v = _v16_uv(em, node)
    ctr = em.input(node, 'Center', VEC3)
    sp = em.input(node, 'Speed', FLOAT)
    base = em.input(node, 'Angle', FLOAT)
    t = _time_of(em, node)
    a, _t = em.tmp(FLOAT, f'{base} + {t} * {sp} * 6.28318530717959')
    ca, _t = em.tmp(FLOAT, f'cos({a})')
    sa, _t = em.tmp(FLOAT, f'sin({a})')
    x, _t = em.tmp(FLOAT, f'{v}.x - {ctr}.x')
    y, _t = em.tmp(FLOAT, f'{v}.y - {ctr}.y')
    return em.tmp(VEC3, f'vec3({x} * {ca} - {y} * {sa} + {ctr}.x, '
                        f'{x} * {sa} + {y} * {ca} + {ctr}.y, {v}.z)')



# ===================================================== R242: the Max study
# The GPU twins of nodeeval's n_mx_* handlers: each map calls its
# MAX_GLSL function with the same literal controls the CPU collapsed
# (a linked control refuses by name through _pat_scalar), the utilities
# are plain arithmetic, and the compound materials mix the sub-materials'
# colour chains per pixel exactly as e_mix_shader / e_add_shader do.

def _need_max(em, name):
    _need_pattern(em, 'mx_prims')
    _need_pattern(em, name)


def _mx_bool(node, name, default=False):
    return 1 if bool(prop(node, name, default)) else 0


def _mx_size_vec(em, node, name='Size'):
    """Max's 3D map coordinate exactly as `_mx_size_vec`: the vector over
    Size, Size a batch constant (0 -> 0.0001)."""
    size = float((next((s for s in node.get('inputs', ())
                        if s.get('name') == name), {}) or {}).get('default', 1.0))
    _pat_scalar(em, node, name, 1.0)              # refuses a linked Size
    if size == 0.0:
        size = 0.0001
    v = tex_vector(em, node, 'generated')
    p, _t = em.tmp(VEC3, f'{v} / {em.const(size, FLOAT)}')
    return p


def _mx_tile_vec(em, node):
    v = tex_vector(em, node, 'uv')
    p, _t = em.tmp(VEC3, f'{v} * {_pat_scalar(em, node, "Tiling", 1.0)}')
    return p


def _mx_out_name(node, index):
    outs = node.get('outputs') or []
    o = outs[index] if index < len(outs) else {}
    return o.get('name')


def e_mx_noise(em, node, index):
    _need_max(em, 'mx_noise')
    p = _mx_size_vec(em, node)
    kind = {'REGULAR': 0, 'FRACTAL': 1, 'TURBULENCE': 2}.get(
        str(prop(node, 'kind', 'REGULAR')), 0)
    return _pat_output(em, node, index, (
        f'hal_mx_noise({p}, {kind}, {_pat_scalar(em, node, "Levels", 3.0)}, '
        f'{_pat_scalar(em, node, "Low", 0.0)}, '
        f'{_pat_scalar(em, node, "High", 1.0)}, '
        f'{_pat_scalar(em, node, "Phase", 0.0)})'))


def e_mx_cellular(em, node, index):
    _need_max(em, 'mx_cellular')
    p = _mx_size_vec(em, node)
    chips = 1 if str(prop(node, 'chips', 'CIRCULAR')) == 'CHIPS' else 0
    uf, _t = em.tmp(VEC2, (
        f'hal_mx_cellular({p}, {chips}, '
        f'{_pat_scalar(em, node, "Spread", 0.5)}, '
        f'{_mx_bool(node, "fractal", False)}, '
        f'{em.const(float(prop(node, "iterations", 3)), FLOAT)}, '
        f'{_pat_scalar(em, node, "Roughness", 0.0)})'))
    name = _mx_out_name(node, index)
    if name == 'Fac':
        return em.tmp(FLOAT, f'clamp({uf}.x, 0.0, 1.0)')
    if name == 'Cell ID':
        return em.tmp(FLOAT, f'{uf}.y')
    return em.tmp(VEC4, (
        f'hal_mx_cellular_colors({uf}, {em.input(node, "Cell Color", VEC4)}, '
        f'{em.input(node, "Division Color 1", VEC4)}, '
        f'{em.input(node, "Division Color 2", VEC4)}, '
        f'{_pat_scalar(em, node, "Low", 0.0)}, '
        f'{_pat_scalar(em, node, "Mid", 0.5)}, '
        f'{_pat_scalar(em, node, "High", 1.0)}, '
        f'{_pat_scalar(em, node, "Variation", 0.0)})'))


def e_mx_smoke(em, node, index):
    _need_max(em, 'mx_smoke')
    p = _mx_size_vec(em, node)
    return _pat_output(em, node, index, (
        f'hal_mx_smoke({p}, {int(prop(node, "iterations", 5))}, '
        f'{_pat_scalar(em, node, "Phase", 0.0)}, '
        f'{_pat_scalar(em, node, "Exponent", 1.5)})'))


def e_mx_speckle(em, node, index):
    _need_max(em, 'mx_speckle')
    p = _mx_size_vec(em, node)
    return _pat_output(em, node, index, f'hal_mx_speckle({p})')


def e_mx_splat(em, node, index):
    _need_max(em, 'mx_splat')
    p = _mx_size_vec(em, node)
    return _pat_output(em, node, index, (
        f'hal_mx_splat({p}, {int(prop(node, "iterations", 4))}, '
        f'{_pat_scalar(em, node, "Threshold", 0.2)})'))


def e_mx_stucco(em, node, index):
    _need_max(em, 'mx_stucco')
    p = _mx_size_vec(em, node)
    return _pat_output(em, node, index, (
        f'hal_mx_stucco({p}, {_pat_scalar(em, node, "Thickness", 0.15)}, '
        f'{_pat_scalar(em, node, "Threshold", 0.57)})'))


def e_mx_marble(em, node, index):
    _need_max(em, 'mx_marble')
    p = _mx_size_vec(em, node)
    return _pat_output(em, node, index, (
        f'hal_mx_marble({p}, {_pat_scalar(em, node, "Vein Width", 0.025)})'))


def e_mx_perlin_marble(em, node, index):
    _need_max(em, 'mx_perlin_marble')
    p = _mx_size_vec(em, node)
    levels = int(round(float((next((s for s in node.get('inputs', ())
                                    if s.get('name') == 'Levels'), {}) or {}).get('default', 8.0))))
    _pat_scalar(em, node, 'Levels', 8.0)
    csp, _t = em.tmp(FLOAT, f'hal_mx_perlin_marble({p}, {levels})')
    if _mx_out_name(node, index) == 'Fac':
        return csp, FLOAT
    return em.tmp(VEC4, (
        f'hal_mx_perlin_marble_colors({csp}, {em.input(node, "Color 1", VEC4)}, '
        f'{em.input(node, "Color 2", VEC4)}, '
        f'{_pat_scalar(em, node, "Saturation 1", 85.0)} / 100.0, '
        f'{_pat_scalar(em, node, "Saturation 2", 70.0)} / 100.0)'))


def e_mx_wood(em, node, index):
    _need_max(em, 'mx_dentnoise')
    _need_pattern(em, 'mx_wood')
    p = _mx_size_vec(em, node, 'Grain Thickness')
    return _pat_output(em, node, index, (
        f'hal_mx_wood({p}, {_pat_scalar(em, node, "Radial Noise", 1.0)}, '
        f'{_pat_scalar(em, node, "Axial Noise", 1.0)})'))


def e_mx_dent(em, node, index):
    _need_max(em, 'mx_dentnoise')
    _need_pattern(em, 'mx_dent')
    p = _mx_size_vec(em, node)
    return _pat_output(em, node, index, (
        f'hal_mx_dent({p}, {_pat_scalar(em, node, "Strength", 20.0)}, '
        f'{int(prop(node, "iterations", 2))})'))


def e_mx_swirl(em, node, index):
    _need_max(em, 'mx_swirl')
    p = _mx_tile_vec(em, node)
    detail = int(round(float((next((s for s in node.get('inputs', ())
                                    if s.get('name') == 'Constant Detail'), {}) or {}).get('default', 4.0))))
    _pat_scalar(em, node, 'Constant Detail', 4.0)
    v, _t = em.tmp(FLOAT, (
        f'hal_mx_swirl({p}, {_pat_scalar(em, node, "Center X", -0.5)}, '
        f'{_pat_scalar(em, node, "Center Y", -0.5)}, '
        f'{_pat_scalar(em, node, "Twist", 1.0)}, '
        f'{_pat_scalar(em, node, "Swirl Intensity", 2.0)}, '
        f'{_pat_scalar(em, node, "Swirl Amount", 1.0)}, {detail}, '
        f'{_pat_scalar(em, node, "Color Contrast", 0.4)}, '
        f'{em.const(float(prop(node, "seed", 0)), FLOAT)})'))
    if _mx_out_name(node, index) == 'Fac':
        return em.tmp(FLOAT, f'clamp({v}, 0.0, 1.0)')
    base, _t = em.tmp(VEC4, em.input(node, 'Base Color', VEC4))
    sw, _t = em.tmp(VEC4, em.input(node, 'Swirl Color', VEC4))
    col, _t = em.tmp(VEC4, f'clamp({sw} * {v} + {base} * (1.0 - {v}), 0.0, 1.0)')
    return em.tmp(VEC4, f'vec4({col}.rgb, 1.0)')


_PLANET_COLS = ('Water 1', 'Water 2', 'Water 3', 'Land 1', 'Land 2',
                'Land 3', 'Land 4', 'Land 5')


def e_mx_planet(em, node, index):
    _need_max(em, 'mx_dentnoise')
    _need_pattern(em, 'mx_planet')
    p = _mx_size_vec(em, node, 'Continent Size')
    e, _t = em.tmp(FLOAT, (
        f'hal_mx_planet({p}, {_pat_scalar(em, node, "Island Factor", 0.5)})'))
    if _mx_out_name(node, index) == 'Elevation':
        return e, FLOAT
    cols = ', '.join(em.input(node, nm, VEC4) for nm in _PLANET_COLS)
    return em.tmp(VEC4, (
        f'hal_mx_planet_colors({e}, {cols}, '
        f'{_pat_scalar(em, node, "Ocean %", 60.0)}, '
        f'{_mx_bool(node, "blend", False)})'))


def e_mx_waves(em, node, index):
    _need_max(em, 'mx_waves')
    p = tex_vector(em, node, 'generated')
    return _pat_output(em, node, index, (
        f'hal_mx_waves({p}, {int(max(1, min(int(prop(node, "sets", 3)), 50)))}, '
        f'{_pat_scalar(em, node, "Wave Radius", 10.0)}, '
        f'{_pat_scalar(em, node, "Wave Len Min", 0.5)}, '
        f'{_pat_scalar(em, node, "Wave Len Max", 0.5)}, '
        f'{_pat_scalar(em, node, "Amplitude", 1.0)}, '
        f'{_pat_scalar(em, node, "Phase", 0.0)}, '
        f'{_mx_bool(node, "dist3d", True)}, {int(prop(node, "seed", 30159))})'))


def e_mx_checker(em, node, index):
    _need_max(em, 'mx_checker')
    p = _mx_tile_vec(em, node)
    return _pat_output(em, node, index, (
        f'hal_mx_checker({p}, {_pat_scalar(em, node, "Soften", 0.0)})'))


_TILE_PATTERNS = ('STACK', 'RUNNING', 'ENGLISH', 'FLEMISH')


def e_mx_tiles(em, node, index):
    _need_max(em, 'mx_tiles')
    p = _mx_tile_vec(em, node)
    pat = str(prop(node, 'pattern', 'RUNNING'))
    pat = _TILE_PATTERNS.index(pat) if pat in _TILE_PATTERNS else 1
    t3, _t = em.tmp(VEC3, (
        f'hal_mx_tiles({p}, {pat}, '
        f'{_pat_scalar(em, node, "Horizontal Count", 4.0)}, '
        f'{_pat_scalar(em, node, "Vertical Count", 4.0)}, '
        f'{_pat_scalar(em, node, "Horizontal Gap", 0.5)}, '
        f'{_pat_scalar(em, node, "Vertical Gap", 0.5)}, '
        f'{_pat_scalar(em, node, "Line Shift", 0.5)}, '
        f'{_pat_scalar(em, node, "Random Shift", 0.0)}, '
        f'{_pat_scalar(em, node, "Holes", 0.0)}, '
        f'{_pat_scalar(em, node, "Fade Variance", 0.05)}, '
        f'{_pat_scalar(em, node, "Color Variance", 0.0)}, '
        f'{int(prop(node, "seed", 33862))})'))
    name = _mx_out_name(node, index)
    if name == 'Fac':
        return em.tmp(FLOAT, f'{t3}.x')
    if name == 'Tile ID':
        return em.tmp(FLOAT, f'{t3}.y')
    tile, _t = em.tmp(VEC4, em.input(node, 'Tile Color', VEC4))
    grout, _t = em.tmp(VEC4, em.input(node, 'Grout Color', VEC4))
    tf, _t = em.tmp(VEC4, f'vec4({tile}.rgb * {t3}.z, {tile}.a)')
    return em.tmp(VEC4, f'clamp({grout} + ({tf} - {grout}) * {t3}.x, 0.0, 1.0)')


_GRAD_TYPES = ('FOUR_CORNER', 'BOX', 'DIAGONAL', 'LINEAR', 'NORMAL', 'PONG',
               'RADIAL', 'SPIRAL', 'SWEEP', 'TARTAN', 'MAPPED')


def e_mx_gradient_ramp(em, node, index):
    _need_max(em, 'mx_gradient_ramp')
    p = _mx_tile_vec(em, node)
    kind = str(prop(node, 'gradient_type', 'LINEAR'))
    k = _GRAD_TYPES.index(kind) if kind in _GRAD_TYPES else 3
    ndv, _t = em.tmp(FLOAT, 'dot(normalize(hal_N), normalize(hal_V))')
    u, _t = em.tmp(FLOAT, f'mod({p}.x, 1.0)')
    v, _t = em.tmp(FLOAT, f'mod({p}.y, 1.0)')
    a, _t = em.tmp(FLOAT, (
        f'hal_mx_gradient_ramp({u}, {v}, {k}, {ndv}, '
        f'{em.input(node, "Mapped", FLOAT)})'))
    if _mx_out_name(node, index) == 'Fac':
        return a, FLOAT
    pos = float((next((s for s in node.get('inputs', ())
                       if s.get('name') == 'Color 2 Position'), {}) or {}).get('default', 0.5))
    _pat_scalar(em, node, 'Color 2 Position', 0.5)
    pos = min(max(pos, 1e-4), 1.0 - 1e-4)
    posc = em.const(pos, FLOAT)
    c1, _t = em.tmp(VEC4, em.input(node, 'Color 1', VEC4))
    c2, _t = em.tmp(VEC4, em.input(node, 'Color 2', VEC4))
    c3, _t = em.tmp(VEC4, em.input(node, 'Color 3', VEC4))
    t1, _t = em.tmp(FLOAT, f'clamp({a} / {posc}, 0.0, 1.0)')
    t2, _t = em.tmp(FLOAT, f'clamp(({a} - {posc}) / (1.0 - {posc}), 0.0, 1.0)')
    return em.tmp(VEC4, f'({a} < {posc}) ? ({c1} + ({c2} - {c1}) * {t1}) '
                        f': ({c2} + ({c3} - {c2}) * {t2})')


def e_mx_falloff(em, node, index):
    _need_pattern(em, 'mx_prims')
    n, _t = em.tmp(VEC3, 'normalize(hal_N)')
    view, _t = em.tmp(VEC3, 'normalize(hal_V)')
    d = str(prop(node, 'direction', 'VIEW'))
    dv = {'WORLD_X': 'vec3(1.0, 0.0, 0.0)', 'WORLD_Y': 'vec3(0.0, 1.0, 0.0)',
          'WORLD_Z': 'vec3(0.0, 0.0, 1.0)'}.get(d, view)
    ndd, _t = em.tmp(FLOAT, f'dot({n}, {dv})')
    kind = str(prop(node, 'falloff_type', 'PERP_PARALLEL'))
    extrapolate = bool(prop(node, 'extrapolate', False))
    if kind == 'TOWARDS_AWAY':
        t, _t = em.tmp(FLOAT, f'1.0 - 0.5 * ({ndd} + 1.0)')
    elif kind == 'FRESNEL':
        _need_pattern(em, 'mx_fresnel')
        ior, _t = em.tmp(FLOAT, f'max({em.input(node, "IOR", FLOAT)}, 1.0)')
        t, _t = em.tmp(FLOAT, f'hal_mx_fresnel(dot({n}, {view}), {ior})')
    elif kind == 'SHADOW_LIGHT':
        raise Unsupported("Falloff's Shadow / Light reads the lamp list at "
                          'shading time, the Light Meter\'s road -- shades '
                          'on the CPU')
    elif kind == 'DISTANCE':
        near, _t = em.tmp(FLOAT, em.input(node, 'Near Distance', FLOAT))
        far, _t = em.tmp(FLOAT, em.input(node, 'Far Distance', FLOAT))
        dist, _t = em.tmp(FLOAT, 'length(hal_P - hal_eye)')
        span, _t = em.tmp(FLOAT, f'{far} - {near}')
        t, _t = em.tmp(FLOAT, f'({span} != 0.0) ? ({far} - {dist}) / '
                              f'(({span} != 0.0) ? {span} : 1.0) : 10000.0')
        if not extrapolate:
            t, _t = em.tmp(FLOAT, f'({dist} <= {near}) ? 1.0 : '
                                  f'(({dist} > {far}) ? 0.0 : {t})')
    else:
        t, _t = em.tmp(FLOAT, f'1.0 - abs({ndd})')
    if not (kind == 'DISTANCE' and extrapolate):
        t, _t = em.tmp(FLOAT, f'clamp({t}, 0.0, 1.0)')
    if _mx_out_name(node, index) == 'Fac':
        return t, FLOAT
    a, _t = em.tmp(VEC4, em.input(node, 'Front', VEC4))
    b, _t = em.tmp(VEC4, em.input(node, 'Side', VEC4))
    return em.tmp(VEC4, f'{a} + ({b} - {a}) * {t}')


def e_mx_mix(em, node, index):
    fac, _t = em.tmp(FLOAT, f'clamp({em.input(node, "Mix Amount", FLOAT)}, 0.0, 1.0)')
    if prop(node, 'use_curve', False):
        _need_pattern(em, 'mx_prims')
        fac, _t = em.tmp(FLOAT, (
            f'hal_mx_mixcurve({_pat_scalar(em, node, "Lower", 0.3)}, '
            f'{_pat_scalar(em, node, "Upper", 0.7)}, {fac})'))
    outs = node.get('outputs') or []
    o = outs[index] if index < len(outs) else {}
    if o.get('name') == 'Fac':
        return fac, FLOAT
    a, _t = em.tmp(VEC4, em.input(node, 'Color 1', VEC4))
    b, _t = em.tmp(VEC4, em.input(node, 'Color 2', VEC4))
    return em.tmp(VEC4, f'{a} + ({b} - {a}) * {fac}')


def e_mx_rgbtint(em, node, _i):
    c, _t = em.tmp(VEC4, em.input(node, 'Color', VEC4))
    r, _t = em.tmp(VEC4, em.input(node, 'R Tint', VEC4))
    g, _t = em.tmp(VEC4, em.input(node, 'G Tint', VEC4))
    b, _t = em.tmp(VEC4, em.input(node, 'B Tint', VEC4))
    return em.tmp(VEC4, f'vec4({r}.rgb * {c}.r + {g}.rgb * {c}.g '
                        f'+ {b}.rgb * {c}.b, {c}.a)')


def e_mx_output(em, node, index):
    c, _t = em.tmp(VEC4, em.input(node, 'Color', VEC4))
    level = em.input(node, 'RGB Level', FLOAT)
    offset = em.input(node, 'RGB Offset', FLOAT)
    amount = em.input(node, 'Output Amount', FLOAT)
    rgb, _t = em.tmp(VEC3, f'({c}.rgb * {level} + {offset} * {c}.a) * {amount}')
    alpha, _t = em.tmp(FLOAT, f'{c}.a * {amount}')
    if prop(node, 'invert', False):
        rgb, _t = em.tmp(VEC3, f'1.0 - {rgb}')
    if prop(node, 'clamp', False):
        rgb, _t = em.tmp(VEC3, f'clamp({rgb}, 0.0, 1.0)')
        alpha, _t = em.tmp(FLOAT, f'clamp({alpha}, 0.0, 1.0)')
    if prop(node, 'alpha_from_rgb', False):
        alpha, _t = em.tmp(FLOAT, f'({rgb}.r + {rgb}.g + {rgb}.b) / 3.0')
    if _mx_out_name(node, index) == 'Fac':
        return em.tmp(FLOAT, f'{rgb}.r * 0.2126 + {rgb}.g * 0.7152 '
                             f'+ {rgb}.b * 0.0722')
    return em.tmp(VEC4, f'vec4({rgb}, {alpha})')


def e_mx_mask(em, node, index):
    m, _t = em.tmp(FLOAT, f'clamp({em.input(node, "Mask", FLOAT)}, 0.0, 1.0)')
    if prop(node, 'invert_mask', False):
        m, _t = em.tmp(FLOAT, f'1.0 - {m}')
    if _mx_out_name(node, index) == 'Alpha':
        return m, FLOAT
    c, _t = em.tmp(VEC4, em.input(node, 'Map', VEC4))
    return em.tmp(VEC4, f'{c} * {m}')


def e_mx_rgbmultiply(em, node, index):
    a, _t = em.tmp(VEC4, em.input(node, 'Color 1', VEC4))
    b, _t = em.tmp(VEC4, em.input(node, 'Color 2', VEC4))
    mode = str(prop(node, 'alpha_from', 'MULTIPLY'))
    alpha = f'{a}.a' if mode == 'MAP1' else (f'{b}.a' if mode == 'MAP2'
                                             else f'{a}.a * {b}.a')
    outs = node.get('outputs') or []
    o = outs[index] if index < len(outs) else {}
    if o.get('name') == 'Alpha':
        return em.tmp(FLOAT, alpha)
    return em.tmp(VEC4, f'vec4({a}.rgb * {b}.rgb, {alpha})')


def e_mx_coords(em, node, _i):
    import numpy as _np
    src = str(prop(node, 'source', 'MAP_CHANNEL'))
    linked = any(s.get('name') == 'Vector' and s.get('link')
                 for s in node.get('inputs', ()))
    if linked:
        v = tex_vector(em, node, 'uv')
    elif src == 'OBJECT_XYZ':
        v = 'hal_generated'
    elif src == 'WORLD_XYZ':
        v = 'hal_P'
    elif src == 'SCREEN':
        if em.secondary:
            v = 'vec3(0.0)'
        else:
            v = 'vec3(vUV, 0.0)'
    else:
        v = 'vec3(hal_uv, 0.0)'
    v, _t = em.tmp(VEC3, v)
    u, _t = em.tmp(FLOAT, f'({v}.x + {_pat_scalar(em, node, "Offset U", 0.0)} '
                          f'- 0.5) * {_pat_scalar(em, node, "Tiling U", 1.0)} + 0.5')
    w, _t = em.tmp(FLOAT, f'({v}.y + {_pat_scalar(em, node, "Offset V", 0.0)} '
                          f'- 0.5) * {_pat_scalar(em, node, "Tiling V", 1.0)} + 0.5')
    # the angle's trig is baked here with NumPy's float32, as e_mapping does
    ang = 0.0
    for sk in node.get('inputs', ()):
        if sk.get('name') == 'Angle W':
            if sk.get('link'):
                raise Unsupported('the Coordinates node bakes its W angle; '
                                  'a per-pixel angle shades on the CPU')
            ang = float(sk.get('default', 0.0) or 0.0)
    a32 = _np.float32(ang) * _np.float32(0.0174532924)
    ca = em.const(float(_np.float32(_np.cos(a32))), FLOAT)
    sa = em.const(float(_np.float32(_np.sin(a32))), FLOAT)
    du, _t = em.tmp(FLOAT, f'{u} - 0.5')
    dw, _t = em.tmp(FLOAT, f'{w} - 0.5')
    u, _t = em.tmp(FLOAT, f'{du} * {ca} - {dw} * {sa} + 0.5')
    w, _t = em.tmp(FLOAT, f'{du} * {sa} + {dw} * {ca} + 0.5')
    if prop(node, 'mirror_u', False):
        t, _t = em.tmp(FLOAT, f'mod({u}, 2.0)')
        u, _t = em.tmp(FLOAT, f'({t} < 1.0) ? {t} : 2.0 - {t}')
    if prop(node, 'mirror_v', False):
        t, _t = em.tmp(FLOAT, f'mod({w}, 2.0)')
        w, _t = em.tmp(FLOAT, f'({t} < 1.0) ? {t} : 2.0 - {t}')
    return em.tmp(VEC3, f'vec3({u}, {w}, {v}.z)')


def _mx_default(node, name, fallback):
    return float((next((s for s in node.get('inputs', ())
                        if s.get('name') == name), {}) or {}).get('default', fallback))


def _mx_linked(node, name):
    return any(s.get('name') == name and s.get('link')
               for s in node.get('inputs', ()))


def e_mx_gradient(em, node, index):
    import numpy as _np
    _need_max(em, 'mx_gradient')
    p = _mx_tile_vec(em, node)
    u, _t = em.tmp(FLOAT, f'mod({p}.x, 1.0)')
    v, _t = em.tmp(FLOAT, f'mod({p}.y, 1.0)')
    amount = _mx_default(node, 'Noise Amount', 0.0)
    _pat_scalar(em, node, 'Noise Amount', 0.0)
    if amount > 0.0:
        size = _mx_default(node, 'Noise Size', 1.0)
        for nm, d in (('Noise Size', 1.0), ('Noise Phase', 0.0), ('Noise Levels', 4.0),
                      ('Threshold Low', 0.0), ('Threshold High', 1.0),
                      ('Threshold Smooth', 0.0)):
            _pat_scalar(em, node, nm, d)
        size1 = em.const(0.0 if size == 0.0 else float(_np.float32(20.0 / size)), FLOAT)
        q, _t = em.tmp(VEC3, (
            f'vec3({u} * {size1} + 1.0, {v} * {size1} + 1.0, '
            f'{em.const(_mx_default(node, "Noise Phase", 0.0), FLOAT)})'))
        kind = {'REGULAR': 0, 'FRACTAL': 1, 'TURBULENCE': 2}.get(
            str(prop(node, 'kind', 'REGULAR')), 0)
        levels = min(max(_mx_default(node, 'Noise Levels', 4.0), 1.0), 10.0)
        noise, _t = em.tmp(FLOAT, (
            f'hal_mx_gradient_noise({q}, {kind}, {em.const(levels, FLOAT)}, '
            f'{em.const(_mx_default(node, "Threshold Low", 0.0), FLOAT)}, '
            f'{em.const(_mx_default(node, "Threshold High", 1.0), FLOAT)}, '
            f'{em.const(_mx_default(node, "Threshold Smooth", 0.0), FLOAT)})'))
    else:
        amount = 0.0
        noise = '0.0'
    shape = 1 if str(prop(node, 'shape', 'LINEAR')) == 'RADIAL' else 0
    a, _t = em.tmp(FLOAT, f'hal_mx_gradient({u}, {v}, {shape}, '
                          f'{em.const(amount, FLOAT)}, {noise})')
    if _mx_out_name(node, index) == 'Fac':
        return a, FLOAT
    pos = min(max(_mx_default(node, 'Color 2 Position', 0.5), 0.0), 1.0)
    _pat_scalar(em, node, 'Color 2 Position', 0.5)
    c1, _t = em.tmp(VEC4, em.input(node, 'Color 1', VEC4))
    c2, _t = em.tmp(VEC4, em.input(node, 'Color 2', VEC4))
    c3, _t = em.tmp(VEC4, em.input(node, 'Color 3', VEC4))
    return em.tmp(VEC4, f'hal_mx_gradient_colors({a}, {em.const(pos, FLOAT)}, '
                        f'{c1}, {c2}, {c3})')


_BLEND_MODES = ('NORMAL', 'AVERAGE', 'ADDITION', 'SUBTRACT', 'DARKEN',
                'MULTIPLY', 'COLOR_BURN', 'LINEAR_BURN', 'LIGHTEN', 'SCREEN',
                'COLOR_DODGE', 'LINEAR_DODGE', 'SPOTLIGHT', 'SPOTLIGHT_BLEND',
                'OVERLAY', 'SOFT_LIGHT', 'HARD_LIGHT', 'PIN_LIGHT', 'HARD_MIX',
                'DIFFERENCE', 'EXCLUSION', 'HUE', 'SATURATION', 'COLOR', 'VALUE')


def e_mx_composite_map(em, node, index):
    _need_pattern(em, 'mx_prims')
    _need_pattern(em, 'mx_hsl')
    _need_pattern(em, 'mx_composite')
    res, _t = em.tmp(VEC4, 'vec4(0.0)')
    for k in range(1, 6):
        name = f'Layer {k}'
        if k > 1 and not _mx_linked(node, name):
            continue
        opacity = _mx_default(node, f'Opacity {k}', 100.0)
        _pat_scalar(em, node, f'Opacity {k}', 100.0)
        if opacity == 0.0:
            continue
        fg, _t = em.tmp(VEC4, em.input(node, name, VEC4))
        mask, _t = em.tmp(FLOAT, f'clamp({em.input(node, f"Mask {k}", FLOAT)}, 0.0, 1.0)')
        mode = str(prop(node, f'blend{k}', 'NORMAL')) if k > 1 else 'NORMAL'
        m = _BLEND_MODES.index(mode) if mode in _BLEND_MODES else 0
        res, _t = em.tmp(VEC4, f'hal_mx_comp_layer({res}, {fg}, '
                               f'{em.const(opacity, FLOAT)}, {mask}, {m})')
    res, _t = em.tmp(VEC4, f'hal_mx_comp_finish({res})')
    if _mx_out_name(node, index) == 'Alpha':
        return em.tmp(FLOAT, f'{res}.a')
    return res, VEC4


_REWIRE = ('RED', 'GREEN', 'BLUE', 'ALPHA', 'RED_INV', 'GREEN_INV',
           'BLUE_INV', 'ALPHA_INV', 'MONO', 'ONE', 'ZERO')
_CC_PRESETS = {'NORMAL': (0, 1, 2, 3), 'MONO': (8, 8, 8, 3),
               'INVERT': (4, 5, 6, 3)}


def e_mx_color_correction(em, node, index):
    _need_pattern(em, 'mx_prims')
    _need_pattern(em, 'mx_hsl')
    _need_pattern(em, 'mx_colorcorr')
    preset = str(prop(node, 'channels', 'NORMAL'))
    if preset in _CC_PRESETS:
        rw = _CC_PRESETS[preset]
    else:
        rw = []
        for key, dflt in (('rewire_r', 'RED'), ('rewire_g', 'GREEN'),
                          ('rewire_b', 'BLUE'), ('rewire_a', 'ALPHA')):
            v = str(prop(node, key, dflt))
            rw.append(_REWIRE.index(v) if v in _REWIRE else _REWIRE.index(dflt))
    adv = 1 if str(prop(node, 'lightness', 'STANDARD')) == 'ADVANCED' else 0
    c, _t = em.tmp(VEC4, em.input(node, 'Color', VEC4))
    tint, _t = em.tmp(VEC4, em.input(node, 'Hue Tint', VEC4))
    args = ', '.join(em.input(node, nm, FLOAT) for nm in ('Hue Shift', 'Saturation'))
    rest = ', '.join(em.input(node, nm, FLOAT) for nm in (
        'Strength',))
    light = ', '.join(em.input(node, nm, FLOAT) for nm in (
        'Brightness', 'Contrast', 'Gain', 'Gamma', 'Pivot', 'Lift'))
    out, _t = em.tmp(VEC4, (
        f'hal_mx_color_correction({c}, {rw[0]}, {rw[1]}, {rw[2]}, {rw[3]}, '
        f'{args}, {tint}.rgb, {rest}, {adv}, {light})'))
    if _mx_out_name(node, index) == 'Fac':
        return em.tmp(FLOAT, f'({out}.r + {out}.g + {out}.b) / 3.0')
    return out, VEC4


def e_mx_vertex_color(em, node, index):
    name = prop(node, 'layer_name', '')
    if name and name != getattr(em, 'color_name', ''):
        raise Unsupported(f"colour layer '{name}' is not in the G-buffer; "
                          f"the active layer '{em.color_name}' is")
    if str(prop(node, 'channel', 'VERTEX_COLOR')) == 'VERTEX_ALPHA':
        rgb, _t = em.tmp(VEC3, 'vec3(hal_vcol.a)')
    else:
        rgb, _t = em.tmp(VEC3, 'hal_vcol.rgb')
    sub = str(prop(node, 'sub_channel', 'ALL'))
    sw = {'RED': 'r', 'GREEN': 'g', 'BLUE': 'b'}.get(sub)
    if sw:
        rgb, _t = em.tmp(VEC3, f'vec3({rgb}.{sw})')
    if _mx_out_name(node, index) == 'Fac':
        return em.tmp(FLOAT, f'({rgb}.r + {rgb}.g + {rgb}.b) / 3.0')
    return em.tmp(VEC4, f'vec4({rgb}, 1.0)')


def e_mx_xyz_coords(em, node, _i):
    import numpy as _np
    src = str(prop(node, 'xyz_source', 'OBJECT_XYZ'))
    if _mx_linked(node, 'Vector'):
        v = tex_vector(em, node, 'generated')
    elif src == 'WORLD_XYZ':
        v = 'hal_P'
    elif src == 'MAP_CHANNEL':
        v = 'vec3(hal_uv, 0.0)'
    elif src == 'VERTEX_COLOR':
        v = 'hal_vcol.rgb'
    elif src == 'GENERATED':
        v = 'hal_generated'
    else:
        v = 'hal_object'
    v, _t = em.tmp(VEC3, v)
    offs, _t = em.tmp(VEC3, 'vec3({}, {}, {})'.format(
        *(_pat_scalar(em, node, f'Offset {ax}', 0.0) for ax in 'XYZ')))
    tile, _t = em.tmp(VEC3, 'vec3({}, {}, {})'.format(
        *(_pat_scalar(em, node, f'Tiling {ax}', 1.0) for ax in 'XYZ')))
    trig = []
    for ax in 'XYZ':
        _pat_scalar(em, node, f'Angle {ax}', 0.0)
        ang = _np.float32(_mx_default(node, f'Angle {ax}', 0.0)) * _np.float32(0.0174532924)
        trig.append((em.const(float(_np.float32(_np.cos(ang))), FLOAT),
                     em.const(float(_np.float32(_np.sin(ang))), FLOAT)))
    (cx, sx), (cy, sy), (cz, sz) = trig

    def turn(q):
        rx, _t = em.tmp(VEC3, f'vec3({q}.x, {q}.y * {cx} - {q}.z * {sx}, '
                              f'{q}.y * {sx} + {q}.z * {cx})')
        ry, _t = em.tmp(VEC3, f'vec3({rx}.z * {sy} + {rx}.x * {cy}, {rx}.y, '
                              f'{rx}.z * {cy} - {rx}.x * {sy})')
        rz, _t = em.tmp(VEC3, f'vec3({ry}.x * {cz} - {ry}.y * {sz}, '
                              f'{ry}.x * {sz} + {ry}.y * {cz}, {ry}.z)')
        return rz
    tv = turn(v)
    to = turn(offs)
    return em.tmp(VEC3, f'{tv} * {tile} + {to}')


def _mx_shader_chain(em, node, name):
    """A sub-material's colour chain, or None when the socket is bare."""
    for sk in node.get('inputs', ()):
        if sk.get('name') == name and sk.get('link'):
            return em.input(node, name, VEC4)
    return None


def _mx_mix_chains(em, a, b, fac):
    """mix(a, b, fac) with a bare side dropping out -- the CPU's closure
    normalises a lone material to full strength."""
    if a is None and b is None:
        return em.tmp(VEC4, 'vec4(0.0, 0.0, 0.0, 1.0)')
    if a is None:
        return em.tmp(VEC4, b)
    if b is None:
        return em.tmp(VEC4, a)
    return em.tmp(VEC4, f'mix({a}, {b}, {fac})')


def e_mx_blend(em, node, _i):
    fac, _t = em.tmp(FLOAT, f'clamp({em.input(node, "Mix Amount", FLOAT)}, 0.0, 1.0)')
    if prop(node, 'use_curve', False):
        _need_pattern(em, 'mx_prims')
        fac, _t = em.tmp(FLOAT, (
            f'hal_mx_mixcurve({_pat_scalar(em, node, "Lower", 0.25)}, '
            f'{_pat_scalar(em, node, "Upper", 0.75)}, {fac})'))
    return _mx_mix_chains(em, _mx_shader_chain(em, node, 'Material 1'),
                          _mx_shader_chain(em, node, 'Material 2'), fac)


def e_mx_doublesided(em, node, _i):
    t, _t = em.tmp(FLOAT, f'clamp({em.input(node, "Translucency", FLOAT)}, 0.0, 1.0)')
    # the stored face normal against the view -- the CPU's own test
    tn, _t = e_new_geometry(em, {'outputs': [{'name': 'True Normal'}]}, 0)
    facing, _t = em.tmp(FLOAT, f'(dot(normalize({tn}), normalize(hal_V)) > 0.0) '
                               f'? 1.0 : 0.0')
    # facing weight: 1 - t on a face toward the camera, t on one away
    wf, _t = em.tmp(FLOAT, f'({facing} > 0.5) ? (1.0 - {t}) : {t}')
    return _mx_mix_chains(em, _mx_shader_chain(em, node, 'Facing'),
                          _mx_shader_chain(em, node, 'Back'), f'(1.0 - {wf})')


def e_mx_topbottom(em, node, _i):
    _need_pattern(em, 'mx_prims')
    up, _t = em.tmp(FLOAT, '0.5 + 0.5 * normalize(hal_N).z')
    pos = _pat_scalar(em, node, 'Position', 0.5)
    blend = _pat_scalar(em, node, 'Blend', 0.0)
    tt, _t = em.tmp(FLOAT, f'hal_mx_mixcurve({pos} - max({blend}, 0.0) * 0.5, '
                           f'{pos} + max({blend}, 0.0) * 0.5, {up})')
    # Top carries weight tt: mix(bottom, top, tt)
    return _mx_mix_chains(em, _mx_shader_chain(em, node, 'Bottom'),
                          _mx_shader_chain(em, node, 'Top'), tt)


def e_mx_shellac(em, node, _i):
    k, _t = em.tmp(FLOAT, f'max({em.input(node, "Color Blend", FLOAT)}, 0.0)')
    base = _mx_shader_chain(em, node, 'Base')
    top = _mx_shader_chain(em, node, 'Shellac')
    if base is None and top is None:
        return em.tmp(VEC4, 'vec4(0.0, 0.0, 0.0, 1.0)')
    if top is None:
        return em.tmp(VEC4, base)
    if base is None:
        return em.tmp(VEC4, top)
    return em.tmp(VEC4, f'{base} + {top} * {k}')


def e_mx_composite(em, node, _i):
    result = _mx_shader_chain(em, node, 'Base')
    for i in (1, 2, 3, 4):
        layer = _mx_shader_chain(em, node, f'Material {i}')
        if layer is None:
            continue
        amt, _t = em.tmp(FLOAT, f'max({em.input(node, f"Amount {i}", FLOAT)}, 0.0)')
        if str(prop(node, f'mode{i}', 'MIX')) == 'ADD':
            if result is None:
                result, _t = em.tmp(VEC4, f'{layer} * min({amt}, 2.0)')
            else:
                result, _t = em.tmp(VEC4, f'{result} + {layer} * min({amt}, 2.0)')
        else:
            a, _t = em.tmp(FLOAT, f'min({amt}, 1.0)')
            result, _t = _mx_mix_chains(em, result, layer, a)
    if result is None:
        return em.tmp(VEC4, 'vec4(0.0, 0.0, 0.0, 1.0)')
    return em.tmp(VEC4, result)



def e_max_standard(em, node, _i):
    """Max's Standard material node, as the deferred pass needs it: its
    colour chain -- the Diffuse map lerped over the swatch by its Amount
    and, under a percentage Self-Illumination, dimmed by 1 - si exactly
    as n_max_standard dims the closure's colour (the glow itself is the
    probed emission constant). Every other socket is a surface field the
    probe harvests through closure_to_surface."""
    p = node.get('props', {})
    base = em.input(node, 'Diffuse Color', VEC4)
    linked = any((s.get('identifier') == 'Diffuse Color' or s.get('name') == 'Diffuse')
                 and s.get('link') for s in node.get('inputs', ()))
    amt = min(max(float(p.get('diffuse_map_amount', 100)), 0.0), 100.0) / 100.0
    if linked and amt < 1.0:
        sw = next((s.get('default') for s in node.get('inputs', ())
                   if s.get('identifier') == 'Diffuse Color'), None) or (0.588, 0.588, 0.588, 1.0)
        swc = em.const(tuple(float(v) for v in sw)[:4], VEC4)
        base, _t = em.tmp(VEC4, f'{swc} + {em.const(amt, FLOAT)} * ({base} - {swc})')
    if not p.get('self_illum_color'):
        si_sock = next((s for s in node.get('inputs', ())
                        if s.get('identifier') == 'Max Self-Illum'), None)
        if si_sock is not None and si_sock.get('link'):
            raise Unsupported("a map on Max's Self-Illumination percentage "
                              'dims the shading per pixel; the material '
                              'shades on the CPU')
        si = min(max(float((si_sock or {}).get('default') or 0.0), 0.0), 100.0) / 100.0
        if si > 0.0:
            base, _t = em.tmp(VEC4, f'vec4({base}.rgb * {em.const(1.0 - si, FLOAT)}, {base}.a)')
    fo = next((s for s in node.get('inputs', ())
               if s.get('identifier') == 'Max Falloff Amount'), None)
    if fo is not None and (fo.get('link') or float(fo.get('default') or 0.0) > 0.0):
        raise Unsupported("Max's Opacity Falloff varies the opacity by the "
                          'view angle per pixel; the material shades on the CPU')
    return em.tmp(VEC4, base)


def e_max_raytrace(em, node, _i):
    """Max's Raytrace material node: its Diffuse chain; the reflection,
    luminosity and transparency colours are probed surface constants."""
    return em.tmp(VEC4, em.input(node, 'Diffuse Color', VEC4))


MAX_EMITTERS = {
    'HALCYON_MaxNoiseNode': e_mx_noise,
    'HALCYON_MaxCellularNode': e_mx_cellular,
    'HALCYON_MaxSmokeNode': e_mx_smoke,
    'HALCYON_MaxSpeckleNode': e_mx_speckle,
    'HALCYON_MaxSplatNode': e_mx_splat,
    'HALCYON_MaxStuccoNode': e_mx_stucco,
    'HALCYON_MaxMarbleNode': e_mx_marble,
    'HALCYON_MaxPerlinMarbleNode': e_mx_perlin_marble,
    'HALCYON_MaxWoodNode': e_mx_wood,
    'HALCYON_MaxDentNode': e_mx_dent,
    'HALCYON_MaxGradientRampNode': e_mx_gradient_ramp,
    'HALCYON_MaxSwirlNode': e_mx_swirl,
    'HALCYON_MaxPlanetNode': e_mx_planet,
    'HALCYON_MaxWavesNode': e_mx_waves,
    'HALCYON_MaxCheckerNode': e_mx_checker,
    'HALCYON_MaxTilesNode': e_mx_tiles,
    'HALCYON_MaxFalloffNode': e_mx_falloff,
    'HALCYON_MaxMixNode': e_mx_mix,
    'HALCYON_MaxRGBTintNode': e_mx_rgbtint,
    'HALCYON_MaxOutputNode': e_mx_output,
    'HALCYON_MaxMaskNode': e_mx_mask,
    'HALCYON_MaxRGBMultiplyNode': e_mx_rgbmultiply,
    'HALCYON_MaxCoordsNode': e_mx_coords,
    'HALCYON_MaxGradientNode': e_mx_gradient,
    'HALCYON_MaxCompositeMapNode': e_mx_composite_map,
    'HALCYON_MaxColorCorrectionNode': e_mx_color_correction,
    'HALCYON_MaxVertexColorNode': e_mx_vertex_color,
    'HALCYON_MaxXYZCoordsNode': e_mx_xyz_coords,
    'HALCYON_MaxBlendNode': e_mx_blend,
    'HALCYON_MaxDoubleSidedNode': e_mx_doublesided,
    'HALCYON_MaxTopBottomNode': e_mx_topbottom,
    'HALCYON_MaxShellacNode': e_mx_shellac,
    'HALCYON_MaxCompositeNode': e_mx_composite,
    'HALCYON_MaxStandardNode': e_max_standard,
    'HALCYON_MaxRaytraceNode': e_max_raytrace,
}

EMITTERS = {
    'HALCYON_ShaderNode': e_halcyon_shader,
    'HALCYON_ConsoleShaderNode': e_console_shader,     # R252
    'HALCYON_AnimeShaderNode': e_anime_shader,
    'HALCYON_CartoonNode': e_cartoon_shader,
    'HALCYON_BIMaterialNode': e_bi_material,
    'HALCYON_CodeNode': e_code_node,

    'HALCYON_MatcapUVNode': e_matcap_uv,
    'HALCYON_RampNode': e_halcyon_ramp,
    'HALCYON_BlurNode': e_halcyon_blur,
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
    'ShaderNodeWireframe': e_wireframe,
    'ShaderNodeBsdfMetallic': e_bsdf_metallic,
    'ShaderNodeEeveeSpecular': e_bsdf_specular,
    'ShaderNodeBsdfTransparent': e_bsdf_transparent,
    'ShaderNodeMixShader': e_mix_shader,
    'ShaderNodeAddShader': e_add_shader,
    'ShaderNodeRGB': e_rgb,
    'ShaderNodeValue': e_value,
    'ShaderNodeMixRGB': e_mix_rgb,
    'ShaderNodeMix': e_mix,
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
    'ShaderNodeSeparateColor': e_separate_color,
    'ShaderNodeCombineColor': e_combine_color,
    'ShaderNodeTexChecker': e_checker,
    'ShaderNodeTexGradient': e_tex_gradient,
    'ShaderNodeTexMagic': e_tex_magic,
    'ShaderNodeTexWave': e_tex_wave,
    'HALCYON_BITextureNode': e_bi_texture,
    'HALCYON_BIInfluenceNode': e_bi_influence,
    'HALCYON_BIRGBBlendNode': e_bi_rgb_blend,
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
    'HALCYON_NoiseNode': e_pat_noise,
    'HALCYON_WaterNode': e_pat_water,
    'HALCYON_CausticsNode': e_pat_caustics,
    'HALCYON_GradientNode': e_pat_gradient_shaped,
    'HALCYON_CellsNode': e_pat_cells_tex,
    'HALCYON_StaticNode': e_pat_static,
    'HALCYON_FurTuftsNode': e_pat_fur_tufts,
    # R232: the 2D media
    'HALCYON_HatchingNode': e_md_hatching,
    'HALCYON_ScribbleNode': e_md_scribble,
    'HALCYON_StippleNode': e_md_stipple,
    'HALCYON_CharcoalNode': e_md_charcoal,
    'HALCYON_PaintStrokesNode': e_md_paint,
    'HALCYON_WashNode': e_md_wash,
    'HALCYON_PaperNode': e_md_paper,
    'HALCYON_TimerNode': e_u16_timer,
    'HALCYON_OscillatorNode': e_u16_oscillator,
    'HALCYON_CounterNode': e_u16_counter,
    'HALCYON_PulseNode': e_u16_pulse,
    'HALCYON_GateNode': e_u16_gate,
    'HALCYON_SelectorNode': e_u16_selector,
    'HALCYON_ColorKeyNode': e_u16_color_key,
    'HALCYON_MeasureNode': e_u16_measure,
    'HALCYON_StepRampNode': e_u16_step_ramp,
    'HALCYON_WobbleNode': e_u16_wobble,
    'HALCYON_FrameBlendNode': e_u16_frame_blend,
    'HALCYON_BlackbodyNode': e_u16_blackbody,
    'HALCYON_CompareNode': e_u16_compare,
    'HALCYON_OnFrameNode': e_u16_on_frame,
    'HALCYON_ArrayVecNode': e_v16_array,
    'HALCYON_MirrorTileNode': e_v16_mirror_tile,
    'HALCYON_KaleidoscopeNode': e_v16_kaleidoscope,
    'HALCYON_PolarNode': e_v16_polar,
    'HALCYON_TwirlNode': e_v16_twirl,
    'HALCYON_LensNode': e_v16_lens,
    'HALCYON_RippleWarpNode': e_v16_ripple_warp,
    'HALCYON_WaveWarpNode': e_v16_wave_warp,
    'HALCYON_TileRandomNode': e_v16_tile_random,
    'HALCYON_VectorSnapNode': e_v16_vector_snap,
    'HALCYON_ShearNode': e_v16_shear,
    'HALCYON_OrbitNode': e_v16_orbit,
    'HALCYON_RegionNode': e_v16_region,
    'HALCYON_ProjectorNode': e_v16_projector,
    'HALCYON_SpinNode': e_v16_spin,
    'HALCYON_PosterizeNode': e_halcyon_posterize,
    'HALCYON_NormalMapNode': e_halcyon_normal_map,
    'HALCYON_NormalMixNode': e_halcyon_normal_mix,
    'HALCYON_AltitudeSlopeNode': e_halcyon_altitude_slope,
    'HALCYON_FacingNode': e_halcyon_facing,
    'HALCYON_IridescentNode': e_halcyon_iridescent,
    'HALCYON_SwitchNode': e_halcyon_switch,
    'HALCYON_RandomPerObjectNode': e_halcyon_random_per_object,
    'HALCYON_LevelsNode': e_halcyon_levels,
    'HALCYON_SmoothStepNode': e_halcyon_smooth_step,
    'HALCYON_ChannelShuffleNode': e_halcyon_channel_shuffle,
    'HALCYON_DistanceMaskNode': e_halcyon_distance_mask,
    'HALCYON_StepTimeNode': e_halcyon_step_time,
    'HALCYON_WaveNode': e_halcyon_wave,
    'HALCYON_DitherNode': e_halcyon_dither,
    'HALCYON_ScreenInfoNode': e_halcyon_screen_info,
    'HALCYON_PixelateNode': e_halcyon_pixelate,
    'HALCYON_ScrollNode': e_halcyon_scroll,
    'HALCYON_ScanlinesNode': e_halcyon_scanlines,
    'HALCYON_PaletteNode': e_halcyon_palette,
    'HALCYON_ColorCycleNode': e_halcyon_color_cycle,
    'HALCYON_FlipbookNode': e_halcyon_flipbook,
    'HALCYON_UVWaveNode': e_halcyon_uv_wave,
    'HALCYON_HalftoneNode': e_halcyon_halftone,
    'HALCYON_ThresholdNode': e_halcyon_threshold,
    'HALCYON_QuantizeNode': e_halcyon_quantize,
    'ShaderNodeBsdfDiffuse': e_bsdf_diffuse,
    'ShaderNodeEmission': e_emission,
    'ShaderNodeFresnel': e_fresnel,
    'ShaderNodeTexImage': e_tex_image,
    'NodeReroute': e_reroute,
    # R206: the baked-LUT nodes -- ColorRamp knocked whole frames off
    # the GPU ("no GLSL emitter for ShaderNodeValToRGB")
    'ShaderNodeValToRGB': e_val_to_rgb,
    'ShaderNodeFloatCurve': e_float_curve,
    'ShaderNodeRGBCurve': e_rgb_curve,
    # R208: hair -- the strand convention rides the colour interpolant
    'ShaderNodeHairInfo': e_hair_info,
    # R242: the Max study
    **MAX_EMITTERS,
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


# ---- R251 material pack, wave 2 (MAT-B): period node emitters ----
# The GLSL twins of core/nodeeval.py's period evaluators: the same
# statements on vec3 with roundEven / floor / clamp, one op per em.tmp
# line where a rounding follows (no FMA across a quantisation), power-of-
# two divides only except the NV2A's proven /255, values through em._m.


def _mb_value(em, v):
    """A liftable VALUE written with NINE significant digits: `em._m` formats
    with %.8g, which does not round-trip every float32 (float32(1) -
    float32(0.9) = 0.1000000238 -> "0.10000002" -> 0.1000000164, one ULP
    off), and the MAT-B twins are claimed bitwise. The lifter parses the
    literal text, so the hal_mats texel carries the exact float32 too."""
    import numpy as _np
    try:
        f = float(_np.float32(v))
    except (TypeError, ValueError):
        f = 0.0
    s = f'{f:.9g}'
    if '.' not in s and 'e' not in s and 'inf' not in s and 'nan' not in s:
        s += '.0'
    if em.mark_values and f == f and abs(f) != float('inf'):
        return f'hal_MV({s})'
    return s

_R255_GLSL = '0.00392156886'


def _cb_q8_glsl(em, expr):
    return em.tmp(VEC3, f'roundEven(clamp({expr}, 0.0, 1.0) * 255.0)')[0]


def e_halcyon_combiner_stage(em, node, _i):
    """C028: the TEV / NV2A stage, integer arithmetic in floats."""
    A, _t = em.tmp(VEC4, em.input(node, 'A', VEC4))
    B, _t = em.tmp(VEC4, em.input(node, 'B', VEC4))
    C, _t = em.tmp(VEC4, em.input(node, 'C', VEC4))
    D, _t = em.tmp(VEC4, em.input(node, 'D', VEC4))
    if str(prop(node, 'hardware', 'TEV')) == 'NV2A':
        def nvmap(v, mapping):
            if mapping == 'UNSIGNED_INVERT':
                return em.tmp(VEC3, f'1.0 - clamp({v}.rgb, 0.0, 1.0)')[0]
            if mapping == 'EXPAND_NORMAL':
                m0 = em.tmp(VEC3, f'max({v}.rgb, 0.0)')[0]
                m1 = em.tmp(VEC3, f'2.0 * {m0}')[0]
                return em.tmp(VEC3, f'{m1} - 1.0')[0]
            if mapping == 'EXPAND_NEGATE':
                m0 = em.tmp(VEC3, f'max({v}.rgb, 0.0)')[0]
                m1 = em.tmp(VEC3, f'2.0 * {m0}')[0]
                m2 = em.tmp(VEC3, f'{m1} - 1.0')[0]
                return em.tmp(VEC3, f'-{m2}')[0]
            if mapping == 'HALF_BIAS_NORMAL':
                m0 = em.tmp(VEC3, f'max({v}.rgb, 0.0)')[0]
                return em.tmp(VEC3, f'{m0} - 0.5')[0]
            if mapping == 'HALF_BIAS_NEGATE':
                m0 = em.tmp(VEC3, f'max({v}.rgb, 0.0)')[0]
                m1 = em.tmp(VEC3, f'{m0} - 0.5')[0]
                return em.tmp(VEC3, f'-{m1}')[0]
            if mapping == 'SIGNED_IDENTITY':
                return em.tmp(VEC3, f'{v}.rgb')[0]
            if mapping == 'SIGNED_NEGATE':
                return em.tmp(VEC3, f'-{v}.rgb')[0]
            return em.tmp(VEC3, f'max({v}.rgb, 0.0)')[0]

        def q9(v):
            r = em.tmp(VEC3, f'roundEven({v} * 255.0)')[0]
            return em.tmp(VEC3, f'clamp({r}, -256.0, 255.0)')[0]
        qa = q9(nvmap(A, str(prop(node, 'map_a', 'UNSIGNED_IDENTITY'))))
        qb = q9(nvmap(B, str(prop(node, 'map_b', 'UNSIGNED_IDENTITY'))))
        qc = q9(nvmap(C, str(prop(node, 'map_c', 'UNSIGNED_IDENTITY'))))
        qd = q9(nvmap(D, str(prop(node, 'map_d', 'UNSIGNED_IDENTITY'))))
        pab_i = em.tmp(VEC3, f'{qa} * {qb}')[0]
        pab = em.tmp(VEC3, f'roundEven({pab_i} / 255.0)')[0]
        pcd_i = em.tmp(VEC3, f'{qc} * {qd}')[0]
        pcd = em.tmp(VEC3, f'roundEven({pcd_i} / 255.0)')[0]
        s = em.tmp(VEC3, f'{pab} + {pcd}')[0]
        sc = str(prop(node, 'nv_scale', 'X1'))
        if sc == 'X2':
            s = em.tmp(VEC3, f'{s} * 2.0')[0]
        elif sc == 'X4':
            s = em.tmp(VEC3, f'{s} * 4.0')[0]
        elif sc == 'HALF':
            s = em.tmp(VEC3, f'floor({s} / 2.0)')[0]
        if str(prop(node, 'nv_bias', 'NONE')) == 'MINUS_HALF':
            s = em.tmp(VEC3, f'{s} - 128.0')[0]
        o = em.tmp(VEC3, f'clamp({s}, -256.0, 255.0)')[0]
    else:
        a8 = _cb_q8_glsl(em, f'{A}.rgb')
        b8 = _cb_q8_glsl(em, f'{B}.rgb')
        c8 = _cb_q8_glsl(em, f'{C}.rgb')
        d0 = em.tmp(VEC3, f'roundEven({D}.rgb * 255.0)')[0]
        d10 = em.tmp(VEC3, f'clamp({d0}, -1024.0, 1023.0)')[0]
        op = str(prop(node, 'op', 'ADD'))
        if op in ('ADD', 'SUB'):
            c9a = em.tmp(VEC3, f'floor({c8} / 128.0)')[0]
            c9 = em.tmp(VEC3, f'{c8} + {c9a}')[0]
            q1 = em.tmp(VEC3, f'256.0 - {c9}')[0]
            p = em.tmp(VEC3, f'{a8} * {q1}')[0]
            q = em.tmp(VEC3, f'{b8} * {c9}')[0]
            s = em.tmp(VEC3, f'{p} + {q}')[0]
            lp = em.tmp(VEC3, f'floor({s} / 256.0)')[0]
            o = em.tmp(VEC3, f'{d10} + {lp}' if op == 'ADD' else f'{d10} - {lp}')[0]
            bias = str(prop(node, 'bias', 'ZERO'))
            if bias == 'ADD_HALF':
                o = em.tmp(VEC3, f'{o} + 128.0')[0]
            elif bias == 'SUB_HALF':
                o = em.tmp(VEC3, f'{o} - 128.0')[0]
            sc = str(prop(node, 'scale', 'X1'))
            if sc == 'X2':
                o = em.tmp(VEC3, f'{o} * 2.0')[0]
            elif sc == 'X4':
                o = em.tmp(VEC3, f'{o} * 4.0')[0]
            elif sc == 'HALF':
                o = em.tmp(VEC3, f'floor({o} / 2.0)')[0]
        else:
            gt = op.endswith('_GT')
            cmp = '>' if gt else '=='
            if op.startswith('COMP_R8'):
                sel = em.tmp(VEC3, f'({a8}.r {cmp} {b8}.r) ? {c8} : vec3(0.0)')[0]
            elif op.startswith('COMP_GR16'):
                pa0 = em.tmp(FLOAT, f'{a8}.g * 256.0')[0]
                pa = em.tmp(FLOAT, f'{pa0} + {a8}.r')[0]
                pb0 = em.tmp(FLOAT, f'{b8}.g * 256.0')[0]
                pb = em.tmp(FLOAT, f'{pb0} + {b8}.r')[0]
                sel = em.tmp(VEC3, f'({pa} {cmp} {pb}) ? {c8} : vec3(0.0)')[0]
            elif op.startswith('COMP_BGR24'):
                pa0 = em.tmp(FLOAT, f'{a8}.b * 65536.0')[0]
                pa1 = em.tmp(FLOAT, f'{a8}.g * 256.0')[0]
                pa2 = em.tmp(FLOAT, f'{pa0} + {pa1}')[0]
                pa = em.tmp(FLOAT, f'{pa2} + {a8}.r')[0]
                pb0 = em.tmp(FLOAT, f'{b8}.b * 65536.0')[0]
                pb1 = em.tmp(FLOAT, f'{b8}.g * 256.0')[0]
                pb2 = em.tmp(FLOAT, f'{pb0} + {pb1}')[0]
                pb = em.tmp(FLOAT, f'{pb2} + {b8}.r')[0]
                sel = em.tmp(VEC3, f'({pa} {cmp} {pb}) ? {c8} : vec3(0.0)')[0]
            else:                                               # COMP_RGB8
                sel = em.tmp(VEC3, f'vec3(({a8}.r {cmp} {b8}.r) ? {c8}.r : 0.0, '
                                   f'({a8}.g {cmp} {b8}.g) ? {c8}.g : 0.0, '
                                   f'({a8}.b {cmp} {b8}.b) ? {c8}.b : 0.0)')[0]
            o = em.tmp(VEC3, f'{d10} + {sel}')[0]
        if bool(prop(node, 'clamp', True)):
            o = em.tmp(VEC3, f'clamp({o}, 0.0, 255.0)')[0]
        else:
            o = em.tmp(VEC3, f'clamp({o}, -1024.0, 1023.0)')[0]
    return em.tmp(VEC4, f'vec4({o} * {_R255_GLSL}, {A}.a)')


EMITTERS.update({
    'HALCYON_CombinerStageNode': e_halcyon_combiner_stage,
})


# ---- MAT-B C023: SR Bump (PowerVR2) ----
def _sr_linked(node, name):
    sock = next((s for s in node.get('inputs', ()) if s.get('name') == name), None)
    return bool(sock and sock.get('link'))


def e_halcyon_sr_bump(em, node, index):
    """C023: the (S,R) intensity through the two raw table textures; K1,
    K2, K3, Q baked from the constant Light socket by the CPU's own
    `light_constants`; a linked Light refuses by name."""
    from ..core import srbump_tables as SRT
    light_linked = _sr_linked(node, 'Light')
    strength_linked = _sr_linked(node, 'Strength')
    if light_linked and strength_linked:
        raise Unsupported('SR Bump: Strength linked with a linked Light -- '
                          'per-pixel sqrt; shades on the CPU')
    if light_linked:
        raise Unsupported("SR Bump: Light linked -- the PVR2's light azimuth "
                          'is one per polygon (a per-pixel atan2); shades on '
                          'the CPU')
    key = ('__sr_tables',)
    if key not in em.once:
        em.once.add(key)
        em.samplers.append({'uniform': 'hal_sr_tab', 'image': '__sr_tables__',
                            'raw': True})
        em.samplers.append({'uniform': 'hal_sr_atan', 'image': '__sr_atan__',
                            'raw': True})
    if '__sr_decl' not in em.once:
        em.once.add('__sr_decl')
        em.inline.append('uniform sampler2D hal_sr_tab;\n'
                         'uniform sampler2D hal_sr_atan;\n')
    col, _t = em.tmp(VEC4, em.input(node, 'Color', VEC4))
    base, _t = em.tmp(VEC4, em.input(node, 'Base', VEC4))
    lsock = next((s for s in node.get('inputs', ()) if s.get('name') == 'Light'), None)
    lvec = list((lsock or {}).get('default') or (0.0, 0.0, 1.0))[:3]
    ssock = next((s for s in node.get('inputs', ()) if s.get('name') == 'Strength'), None)
    if strength_linked:
        # a linked Strength with an unlinked Light: sinT / cosT baked once,
        # K1 / K2 / K3 per pixel (no transcendental)
        K1c, K2c, K3c, Q = SRT.light_constants(lvec, 1.0)     # H = 1: K2 = sinT, K3 = cosT
        Hraw = em.input(node, 'Strength', FLOAT)
        Hv, _t = em.tmp(FLOAT, f'clamp({Hraw}, 0.0, 1.0)')
        K1, _t = em.tmp(FLOAT, f'1.0 - {Hv}')
        K2, _t = em.tmp(FLOAT, f'{_mb_value(em, K2c)} * {Hv}')
        K3, _t = em.tmp(FLOAT, f'{_mb_value(em, K3c)} * {Hv}')
    else:
        H = float((ssock or {}).get('default') or 0.0)
        K1c, K2c, K3c, Q = SRT.light_constants(lvec, H)
        K1, _t = em.tmp(FLOAT, _mb_value(em, K1c))
        K2, _t = em.tmp(FLOAT, _mb_value(em, K2c))
        K3, _t = em.tmp(FLOAT, _mb_value(em, K3c))
    Qv, _t = em.tmp(FLOAT, _mb_value(em, float(Q)))
    n8, _t = em.tmp(VEC3, f'roundEven(clamp({col}.rgb, 0.0, 1.0) * 255.0)')
    S, _t = em.tmp(FLOAT, f'texelFetch(hal_sr_tab, ivec2(int({n8}.b), 0), 0).r')
    R, _t = em.tmp(FLOAT, f'texelFetch(hal_sr_atan, ivec2(int({n8}.r), int({n8}.g)), 0).r')
    sinS, _t = em.tmp(FLOAT, f'texelFetch(hal_sr_tab, ivec2(int({S}) + 256, 0), 0).r')
    cosS, _t = em.tmp(FLOAT, f'texelFetch(hal_sr_tab, ivec2(int({S}) + 512, 0), 0).r')
    m0, _t = em.tmp(FLOAT, f'{R} - {Qv}')
    mf, _t = em.tmp(FLOAT, f'floor({m0} / 256.0)')
    m1, _t = em.tmp(FLOAT, f'256.0 * {mf}')
    m, _t = em.tmp(FLOAT, f'{m0} - {m1}')
    c256, _t = em.tmp(FLOAT, f'texelFetch(hal_sr_tab, ivec2(int({m}) + 768, 0), 0).r')
    p1, _t = em.tmp(FLOAT, f'{K2} * {sinS}')
    a, _t = em.tmp(FLOAT, f'{K1} + {p1}')
    c1, _t = em.tmp(FLOAT, f'{cosS} * {c256}')
    p2, _t = em.tmp(FLOAT, f'{K3} * {c1}')
    I0, _t = em.tmp(FLOAT, f'{a} + {p2}')
    I, _t = em.tmp(FLOAT, f'clamp({I0}, 0.0, 1.0)')
    if index == 0:
        return I, FLOAT
    if str(prop(node, 'blend', 'MULTIPLY')) == 'ADD':
        rgb, _t = em.tmp(VEC3, f'min({base}.rgb + vec3({I}), 1.0)')
    else:
        rgb, _t = em.tmp(VEC3, f'{base}.rgb * {I}')
    return em.tmp(VEC4, f'vec4({rgb}, {base}.a)')


EMITTERS.update({'HALCYON_SRBumpNode': e_halcyon_sr_bump})


# ---- MAT-B C135: Emboss Bump (DirectX 6), two stages ----
def e_halcyon_emboss_shift(em, node, _i):
    """C135 stage one: `inv` is a value baked on the CPU (1/size in float32,
    so the GPU never divides); a linked Texture Size refuses by name."""
    if _sr_linked(node, 'Texture Size'):
        raise Unsupported("Emboss Bump: Texture Size linked -- the shift's "
                          'reciprocal is baked once per material; shades on '
                          'the CPU')
    import numpy as _np
    ssock = next((s for s in node.get('inputs', ()) if s.get('name') == 'Texture Size'), None)
    size = max(float((ssock or {}).get('default') or 256.0), 1.0)
    inv = float(_np.float32(_np.float32(1.0) / _np.float32(size)))
    uv, _t = em.tmp(VEC3, tex_vector(em, node, 'uv'))
    L, _t = em.tmp(VEC3, em.input(node, 'Light', VEC3))
    off, _t = em.tmp(FLOAT, em.input(node, 'Offset', FLOAT))
    p, _t = em.tmp('vec2', f'{off} * {L}.xy')
    d, _t = em.tmp('vec2', f'{p} * {_mb_value(em, inv)}')
    return em.tmp(VEC3, f'vec3({uv}.xy + {d}, {uv}.z)')


def e_halcyon_emboss_bump(em, node, index):
    """C135 stage two: hd, b, clamp, m, the modulate-2x -- five statements."""
    h, _t = em.tmp(FLOAT, em.input(node, 'Height', FLOAT))
    hs, _t = em.tmp(FLOAT, em.input(node, 'Height Shifted', FLOAT))
    base, _t = em.tmp(VEC4, em.input(node, 'Base', VEC4))
    hd, _t = em.tmp(FLOAT, f'{h} - {hs}')
    b0, _t = em.tmp(FLOAT, f'0.5 + {hd}')
    b, _t = em.tmp(FLOAT, f'clamp({b0}, 0.0, 1.0)')
    if index == 0:
        return b, FLOAT
    m0, _t = em.tmp(FLOAT, f'2.0 * {b}')
    m, _t = em.tmp(FLOAT, f'min({m0}, 1.0)')
    rgb, _t = em.tmp(VEC3, f'{base}.rgb * {m}')
    return em.tmp(VEC4, f'vec4({rgb}, {base}.a)')


EMITTERS.update({'HALCYON_EmbossShiftNode': e_halcyon_emboss_shift,
                 'HALCYON_EmbossBumpNode': e_halcyon_emboss_bump})


# ---- MAT-B C099: Roughness (Imagine) ----
#: the Wang mix WITHOUT the 0..1 fold (the CPU hashes the pixel word once,
#: then folds the three lanes off the mixed word through hal_wang01)
_IMR_WANG_GLSL = """
uint hal_imr_wang(uint u)
{
    u = (u ^ 61u) ^ (u >> 16u);
    u = u * 9u;
    u = u ^ (u >> 4u);
    u = u * 668265261u;
    u = u ^ (u >> 15u);
    return u;
}
"""


def e_halcyon_imagine_roughness(em, node, _i):
    """C099: the same Wang hash of (pixel, seed[, frame]) on the driver; the
    Screen Info rule (the pixel exists only in the deferred frame); the
    seed is a plan-signature key, baked as a literal (`em.seed`)."""
    if not (em.frame_mode and not em.secondary
            and getattr(em, 'resolution', None) is not None):
        raise Unsupported('Roughness (Imagine) reads the pixel position, which '
                          'hit shading does not carry; shades on the CPU')
    if '__wang' not in em.once:
        em.once.add('__wang')
        em.inline.append(_WANG_GLSL)
    if '__imr_wang' not in em.once:
        em.once.add('__imr_wang')
        em.inline.append(_IMR_WANG_GLSL)
    nrm = em.input(node, 'Normal', VEC3) if _sr_linked(node, 'Normal') else 'hal_N'
    N, _t = em.tmp(VEC3, nrm)
    r0 = em.input(node, 'Roughness', FLOAT)
    r, _t = em.tmp(FLOAT, f'clamp({r0}, 0.0, 255.0)')
    w, h = float(em.resolution[0]), float(em.resolution[1])
    seed = int(getattr(em, 'seed', 0) or 0)
    shimmer = bool(prop(node, 'animate', False))
    base_salt = (seed * 7919) & 0xFFFFFFFF
    if shimmer:
        em.frame_uniforms.add('hal_frame')
        salt_expr = f'({base_salt}u + uint(hal_frame) * 104729u)'
    else:
        salt_expr = f'{base_salt}u'
    px, _t = em.tmp('uint', f'uint(floor(vUV.x * {_c(w)}))')
    py, _t = em.tmp('uint', f'uint(floor(vUV.y * {_c(h)}))')
    u, _t = em.tmp('uint', f'hal_imr_wang({px} + {py} * 65536u + {salt_expr})')
    x, _t = em.tmp(FLOAT, f'2.0 * hal_wang01({u} ^ 1757225451u) - 1.0')
    y, _t = em.tmp(FLOAT, f'2.0 * hal_wang01({u} ^ 48610963u) - 1.0')
    z, _t = em.tmp(FLOAT, f'2.0 * hal_wang01({u} ^ 2524743835u) - 1.0')
    nn, _t = em.tmp(VEC3, f'normalize({N})')
    k0, _t = em.tmp(FLOAT, f'{r} * 0.00392156886')
    k, _t = em.tmp(FLOAT, f'{k0} * 0.5')
    n2, _t = em.tmp(VEC3, f'{nn} + {k} * vec3({x}, {y}, {z})')
    return em.tmp(VEC3, f'({r} > 0.0) ? normalize({n2}) : {N}')


EMITTERS.update({'HALCYON_ImagineRoughnessNode': e_halcyon_imagine_roughness})


# ---- MAT-B C123: Env Chrome (Alias / Maya) ----
_EC_DEFAULTS_E = {'light_width': 0.5, 'light_depth': 0.1, 'light_width_gain': 1.0,
                  'light_width_offset': 0.0, 'light_depth_gain': 1.0,
                  'light_depth_offset': 0.0, 'grid_width': 0.1, 'grid_depth': 0.1,
                  'grid_width_gain': 1.0, 'grid_width_offset': 0.0,
                  'grid_depth_gain': 1.0, 'grid_depth_offset': 0.0,
                  'floor_altitude': -1.0}


def e_halcyon_env_chrome(em, node, _i):
    """C123: the showroom along reflect(I, N) -- every multiply-add split
    into two statements (numpy rounds them apart; no FMA may cross a
    floor or a compare), the two lerps as a + (b - a) * t, `?:` selects;
    the 13 parameters and the six colours are values."""
    def pv(name):
        v, _t = em.tmp(FLOAT, _mb_value(em, float(prop(node, name, _EC_DEFAULTS_E[name]))))
        return v
    pr = {k: pv(k) for k in _EC_DEFAULTS_E}
    cols = {}
    for name, key in (('Sky Color', 'sky'), ('Zenith Color', 'zen'), ('Light Color', 'light'),
                      ('Floor Color', 'floor'), ('Horizon Color', 'hor'), ('Grid Color', 'grid')):
        c4, _t = em.tmp(VEC4, em.input(node, name, VEC4))
        cols[key], _t = em.tmp(VEC3, f'{c4}.rgb')
    I, _t = em.tmp(VEC3, '-normalize(hal_V)')
    nrm = em.input(node, 'Normal', VEC3) if _sr_linked(node, 'Normal') else 'hal_N'
    N, _t = em.tmp(VEC3, f'normalize({nrm})')
    dn, _t = em.tmp(FLOAT, f'dot({N}, {I})')
    s2, _t = em.tmp(FLOAT, f'2.0 * {dn}')
    sn, _t = em.tmp(VEC3, f'{s2} * {N}')
    R, _t = em.tmp(VEC3, f'{I} - {sn}')
    # sky side
    rz, _t = em.tmp(FLOAT, f'max({R}.z, 1e-6)')
    t, _t = em.tmp(FLOAT, f'clamp({R}.z, 0.0, 1.0)')
    dz, _t = em.tmp(VEC3, f'{cols["zen"]} - {cols["sky"]}')
    dzt, _t = em.tmp(VEC3, f'{dz} * {t}')
    sky, _t = em.tmp(VEC3, f'{cols["sky"]} + {dzt}')
    px, _t = em.tmp(FLOAT, f'{R}.x / {rz}')
    pz, _t = em.tmp(FLOAT, f'{R}.y / {rz}')
    fx0, _t = em.tmp(FLOAT, f'{px} * {pr["light_width_gain"]}')
    fx1, _t = em.tmp(FLOAT, f'{fx0} + {pr["light_width_offset"]}')
    fx, _t = em.tmp(FLOAT, f'{fx1} - floor({fx1})')
    fz0, _t = em.tmp(FLOAT, f'{pz} * {pr["light_depth_gain"]}')
    fz1, _t = em.tmp(FLOAT, f'{fz0} + {pr["light_depth_offset"]}')
    fz, _t = em.tmp(FLOAT, f'{fz1} - floor({fz1})')
    up, _t = em.tmp(VEC3, f'(({fx} < {pr["light_width"]}) && ({fz} < {pr["light_depth"]})) ? {cols["light"]} : {sky}')
    # floor side
    rzn, _t = em.tmp(FLOAT, f'min({R}.z, -1e-6)')
    ax, _t = em.tmp(FLOAT, f'{R}.x * {pr["floor_altitude"]}')
    pxn, _t = em.tmp(FLOAT, f'{ax} / {rzn}')
    az, _t = em.tmp(FLOAT, f'{R}.y * {pr["floor_altitude"]}')
    pzn, _t = em.tmp(FLOAT, f'{az} / {rzn}')
    if bool(prop(node, 'real_floor', True)):
        dh, _t = em.tmp(FLOAT, f'{pr["floor_altitude"]} - hal_P.z')
        tt, _t = em.tmp(FLOAT, f'{dh} / {rzn}')
        mx, _t = em.tmp(FLOAT, f'{tt} * {R}.x')
        pxr, _t = em.tmp(FLOAT, f'hal_P.x + {mx}')
        mz, _t = em.tmp(FLOAT, f'{tt} * {R}.y')
        pzr, _t = em.tmp(FLOAT, f'hal_P.y + {mz}')
        px2, _t = em.tmp(FLOAT, f'({tt} >= 0.0) ? {pxr} : {pxn}')
        pz2, _t = em.tmp(FLOAT, f'({tt} >= 0.0) ? {pzr} : {pzn}')
    else:
        px2, pz2 = pxn, pzn
    nz, _t = em.tmp(FLOAT, f'-{R}.z')
    dfh, _t = em.tmp(VEC3, f'{cols["floor"]} - {cols["hor"]}')
    dft, _t = em.tmp(VEC3, f'{dfh} * {nz}')
    base, _t = em.tmp(VEC3, f'{cols["hor"]} + {dft}')
    gx0, _t = em.tmp(FLOAT, f'{px2} * {pr["grid_width_gain"]}')
    gx1, _t = em.tmp(FLOAT, f'{gx0} + {pr["grid_width_offset"]}')
    gx, _t = em.tmp(FLOAT, f'{gx1} - floor({gx1})')
    gz0, _t = em.tmp(FLOAT, f'{pz2} * {pr["grid_depth_gain"]}')
    gz1, _t = em.tmp(FLOAT, f'{gz0} + {pr["grid_depth_offset"]}')
    gz, _t = em.tmp(FLOAT, f'{gz1} - floor({gz1})')
    down, _t = em.tmp(VEC3, f'(({gx} < {pr["grid_width"]}) || ({gz} < {pr["grid_depth"]})) ? {cols["grid"]} : {base}')
    out, _t = em.tmp(VEC3, f'({R}.z >= 0.0) ? {up} : {down}')
    return em.tmp(VEC4, f'vec4({out}, 1.0)')


EMITTERS.update({'HALCYON_EnvChromeNode': e_halcyon_env_chrome})
