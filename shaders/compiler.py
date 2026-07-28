"""Compile shader source into a runnable Program.

    prog = compile_shader(src, lang='GLSL')
    outs, discard = prog.run(uniforms, inputs, n)

Programs are cached on (source, language) so re-rendering a frame does not
recompile, and the generated Python is kept on the Program for inspection --
useful when a shader misbehaves and you want to see what it actually became.
"""

import hashlib

import numpy as np

from . import builtins as rt
from .codegen import VARYINGS, check_recursion
from .gtypes import GType
from .lexer import ShaderError

_CACHE = {}
MAX_CACHE = 64


class Program:
    def __init__(self, source, lang, code, fn, outputs, uniforms, inputs, warnings):
        self.source = source
        self.lang = lang
        self.code = code
        self.fn = fn
        self.outputs = outputs          # [(glsl_name, py_name, GType)]
        self.uniforms = uniforms        # [(name, GType, default_ast)]
        self.inputs = inputs            # [(name, GType, binding)]
        self.warnings = list(warnings)
        self.error = None

    # ---------------------------------------------------------------- schema
    def uniform_schema(self):
        """What sockets the node editor should show for this shader."""
        out = []
        for name, t, default in self.uniforms:
            out.append({'name': name, 'type': t, 'kind': socket_kind(t),
                        'default': default_value(t, default)})
        return out

    def output_schema(self):
        out = []
        for gname, pyname, t in self.outputs:
            label = 'Result' if gname == '__return' else gname
            out.append({'name': label, 'key': gname, 'type': t,
                        'kind': socket_kind(t)})
        return out

    def input_names(self):
        return sorted({b for _, _, b in self.inputs})

    # ------------------------------------------------------------------- run
    def __getstate__(self):
        """Neither the interpreter's closure nor its tree pickles usefully, but
        the source they came from does. Workers recompile on arrival."""
        st = dict(self.__dict__)
        st.pop('fn', None)
        st.pop('interp', None)
        return st

    def __setstate__(self, state):
        self.__dict__.update(state)
        rebuilt = compile_shader(self.source, self.lang, use_cache=False)
        self.fn = rebuilt.fn
        self.interp = getattr(rebuilt, 'interp', None)

    def run(self, uniforms, inputs, n, ctx=None):
        if ctx is not None:
            rt.set_ctx(ctx)
        outs, discard = self.fn(rt, np, uniforms or {}, inputs or {}, int(n))
        fixed = {}
        for gname, pyname, t in self.outputs:
            v = outs.get(gname)
            fixed[gname] = rt.bc(v, n) if v is not None else None
        return fixed, discard


def socket_kind(t):
    if t.base == 'sampler':
        return 'IMAGE'
    if t.is_matrix:
        return 'MATRIX'
    if t.n == 1:
        return {'float': 'VALUE', 'int': 'INT', 'uint': 'INT',
                'bool': 'BOOL'}.get(t.base, 'VALUE')
    if t.n == 2:
        return 'VECTOR2'
    if t.n == 3:
        return 'VECTOR'
    return 'RGBA'


def default_value(t, default_ast):
    from .parser import _const_int
    if t.base == 'sampler':
        return None
    if t.is_matrix:
        return None
    n = t.comps
    if default_ast is not None:
        try:
            if default_ast[0] == 'num':
                v = float(default_ast[1].rstrip('uUfFhH'))
                return [v] * n
            if default_ast[0] == 'call':
                vals = []
                for a in default_ast[2]:
                    if a[0] == 'num':
                        vals.append(float(a[1].rstrip('uUfFhH')))
                    elif a[0] == 'un' and a[1] == '-' and a[2][0] == 'num':
                        vals.append(-float(a[2][1].rstrip('uUfFhH')))
                if vals:
                    if len(vals) == 1:
                        vals = vals * n
                    return (vals + [0.0] * n)[:n]
            if default_ast[0] == 'bool':
                return [1.0 if default_ast[1] else 0.0] * n
        except Exception:
            pass
    if n == 4:
        return [0.0, 0.0, 0.0, 1.0]
    return [0.0] * n


def source_key(src, lang, defines=None):
    h = hashlib.sha1()
    h.update(src.encode('utf8', 'replace'))
    h.update(lang.encode())
    if defines:
        h.update(repr(sorted(defines.items())).encode())
    return h.hexdigest()


def _make_runner(interp):
    """A closure over the interpreter, matching the old generated signature."""
    def __shader(_rt, _np, uniforms, inputs, n):
        return interp.run(uniforms or {}, inputs or {}, int(n))
    return __shader


def compile_shader(src, lang='GLSL', defines=None, use_cache=True):
    """Compile and return a Program. Raises ShaderError on failure."""
    key = source_key(src, lang, defines)
    if use_cache and key in _CACHE:
        return _CACHE[key]
    hlsl = lang.upper() in ('HLSL', 'CG', 'SHADERLAB')
    defs = dict(defines or {})
    defs.setdefault('HALCYON', ([], '1'))
    if hlsl:
        defs.setdefault('HLSL', ([], '1'))
    else:
        defs.setdefault('GLSL', ([], '1'))
    from .interp import Interpreter
    from .parser import parse

    # The backend builds a callable that walks the parsed tree. It used to
    # generate Python source and exec it, which an extension may not do.
    decls, structs = parse(src, hlsl=hlsl, defines=defs)
    check_recursion(decls)
    interp = Interpreter(decls, structs, hlsl=hlsl).validate()
    fn = _make_runner(interp)
    prog = Program(src, lang, None, fn, interp.output_schema_list(),
                   interp.uniform_schema_list(), interp.input_schema_list(),
                   interp.warnings)
    prog.interp = interp
    if use_cache:
        if len(_CACHE) > MAX_CACHE:
            _CACHE.clear()
        _CACHE[key] = prog
    return prog


def try_compile(src, lang='GLSL', defines=None):
    """Compile, returning (program, error_string)."""
    try:
        return compile_shader(src, lang, defines), None
    except ShaderError as e:
        return None, str(e)
    except RecursionError:
        return None, 'shader too deeply nested (recursive functions are not supported)'
    except Exception as e:                       # noqa: BLE001 - surfaced in the UI
        return None, f'{type(e).__name__}: {e}'


def clear_cache():
    _CACHE.clear()


DEFAULT_GLSL = """\
// Halcyon coded shader -- GLSL
// Uniforms become input sockets; `out` variables become output sockets.

uniform vec3  baseColor = vec3(0.85, 0.45, 0.2);
uniform float bands     = 6.0;
uniform float rimPower  = 2.5;

in vec3 vNormal;
in vec3 vView;

out vec4 Color;

void main() {
    vec3  n   = normalize(vNormal);
    float ndv = clamp(dot(n, normalize(vView)), 0.0, 1.0);

    // hard quantised terminator: the look of an 8-bit framebuffer
    float lam  = clamp(dot(n, normalize(vec3(0.4, -0.7, 0.6))), 0.0, 1.0);
    float step_ = floor(lam * bands) / bands;

    float rim = pow(1.0 - ndv, rimPower);
    vec3  col = baseColor * (0.25 + 0.75 * step_) + vec3(rim * 0.35);

    Color = vec4(col, 1.0);
}
"""

DEFAULT_HLSL = """\
// Halcyon coded shader -- HLSL dialect

uniform float3 Tint      = float3(0.6, 0.8, 1.0);
uniform float  Frequency = 24.0;

in float3 vPosition;
in float3 vNormal;

out float4 Color;

void main() {
    float3 n = normalize(vNormal);
    float  s = frac(vPosition.z * Frequency);
    float  stripe = step(0.5, s);
    float  d = saturate(dot(n, normalize(float3(0.3, -0.6, 0.7))));
    Color = float4(Tint * lerp(0.2, 1.0, d) * lerp(0.55, 1.0, stripe), 1.0);
}
"""
