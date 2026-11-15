"""Shader-compiler tests. Run with:  python3 -m halcyon.tests.test_shaders"""

import sys

import numpy as np

from ..shaders.compiler import try_compile

FAILS = []


def check(name, cond, extra=''):
    print(('  ok   ' if cond else '  FAIL ') + name + (('  ' + extra) if extra else ''))
    if not cond:
        FAILS.append(name)


def run(src, uniforms=None, inputs=None, n=8, lang='GLSL'):
    prog, err = try_compile(src, lang)
    if err:
        raise AssertionError(err)
    outs, discard = prog.run(uniforms or {}, inputs or {}, n)
    return outs, discard, prog


def test_divergent_if():
    src = """
    in vec3 vPosition;
    out vec4 Color;
    void main() {
        float x = vPosition.x;
        float y;
        if (x > 0.5) { y = 10.0; } else { y = -1.0; }
        Color = vec4(y, y, y, 1.0);
    }
    """
    p = np.zeros((8, 3), np.float32)
    p[:, 0] = np.linspace(0.0, 1.0, 8)
    outs, _, _ = run(src, inputs={'position': p})
    got = outs['Color'][:, 0]
    want = np.where(p[:, 0] > 0.5, 10.0, -1.0)
    check('divergent if/else', np.allclose(got, want), str(got))


def test_loop_break_continue():
    src = """
    in vec3 vPosition;
    out vec4 Color;
    void main() {
        float acc = 0.0;
        int limit = int(vPosition.x * 8.0);
        for (int i = 0; i < 8; i++) {
            if (i >= limit) break;
            if (i == 2) continue;
            acc += float(i);
        }
        Color = vec4(acc, 0.0, 0.0, 1.0);
    }
    """
    p = np.zeros((8, 3), np.float32)
    p[:, 0] = np.arange(8) / 8.0
    outs, _, _ = run(src, inputs={'position': p})
    want = []
    for lim in range(8):
        a = 0.0
        for i in range(8):
            if i >= lim:
                break
            if i == 2:
                continue
            a += i
        want.append(a)
    check('for + break + continue (per-lane trip counts)',
          np.allclose(outs['Color'][:, 0], want), str(outs['Color'][:, 0]))


def test_while_loop():
    src = """
    in vec3 vPosition;
    out vec4 Color;
    void main() {
        float v = vPosition.x;
        int steps = 0;
        while (v < 1.0) { v = v * 2.0 + 0.05; steps++; }
        Color = vec4(float(steps), v, 0.0, 1.0);
    }
    """
    p = np.zeros((6, 3), np.float32)
    p[:, 0] = np.array([0.02, 0.1, 0.3, 0.5, 0.9, 1.5], np.float32)
    outs, _, _ = run(src, inputs={'position': p}, n=6)
    want = []
    for v0 in p[:, 0]:
        v = float(v0)
        s = 0
        while v < 1.0:
            v = v * 2.0 + 0.05
            s += 1
        want.append(s)
    check('while with per-lane trip counts',
          np.allclose(outs['Color'][:, 0], want), str(outs['Color'][:, 0]))


def test_function_and_out_param():
    src = """
    float sq(float x) { return x * x; }
    void split(vec3 v, out float lo, out float hi) {
        lo = min(min(v.x, v.y), v.z);
        hi = max(max(v.x, v.y), v.z);
    }
    in vec3 vPosition;
    out vec4 Color;
    void main() {
        float lo; float hi;
        split(vPosition, lo, hi);
        Color = vec4(sq(vPosition.x), lo, hi, 1.0);
    }
    """
    p = np.array([[1., 2., 3.], [3., -1., 0.], [0.5, 0.5, 0.5]], np.float32)
    outs, _, _ = run(src, inputs={'position': p}, n=3)
    c = outs['Color']
    check('user function + out parameters',
          np.allclose(c[:, 0], p[:, 0] ** 2) and np.allclose(c[:, 1], p.min(1))
          and np.allclose(c[:, 2], p.max(1)), str(c))


def test_early_return():
    src = """
    float f(float x) {
        if (x < 0.0) return -1.0;
        if (x > 1.0) return 2.0;
        return x;
    }
    in vec3 vPosition;
    out vec4 Color;
    void main() { Color = vec4(f(vPosition.x), 0, 0, 1); }
    """
    p = np.zeros((5, 3), np.float32)
    p[:, 0] = np.array([-2.0, -0.1, 0.5, 1.0, 3.0], np.float32)
    outs, _, _ = run(src, inputs={'position': p}, n=5)
    want = [-1.0, -1.0, 0.5, 1.0, 2.0]
    check('early return inside a function',
          np.allclose(outs['Color'][:, 0], want), str(outs['Color'][:, 0]))


def test_struct_and_array():
    src = """
    struct Light { vec3 dir; float power; };
    uniform float gain = 2.0;
    in vec3 vNormal;
    out vec4 Color;
    void main() {
        Light l;
        l.dir = normalize(vec3(0.0, 0.0, 1.0));
        l.power = gain;
        float w[3];
        w[0] = 0.5; w[1] = 0.25; w[2] = 0.125;
        float acc = 0.0;
        for (int i = 0; i < 3; i++) acc += w[i];
        Color = vec4(dot(normalize(vNormal), l.dir) * l.power + acc, 0, 0, 1);
    }
    """
    nrm = np.tile(np.array([[0., 0., 1.]], np.float32), (4, 1))
    outs, _, _ = run(src, inputs={'normal': nrm}, n=4)
    check('structs + fixed-size arrays',
          np.allclose(outs['Color'][:, 0], 2.0 + 0.875), str(outs['Color'][:, 0]))


def test_matrix():
    src = """
    out vec4 Color;
    void main() {
        mat3 m = mat3(0.0, 1.0, 0.0,  -1.0, 0.0, 0.0,  0.0, 0.0, 1.0);
        vec3 v = vec3(1.0, 0.0, 0.0);
        vec3 r = m * v;
        Color = vec4(r, 1.0);
    }
    """
    outs, _, _ = run(src, n=2)
    check('mat3 construction and mat*vec',
          np.allclose(outs['Color'][0, :3], [0.0, 1.0, 0.0]), str(outs['Color'][0]))


def test_swizzle_write():
    src = """
    out vec4 Color;
    void main() {
        vec4 c = vec4(0.0);
        c.rgb = vec3(0.1, 0.2, 0.3);
        c.a = 1.0;
        c.xy = c.yx;
        Color = c;
    }
    """
    outs, _, _ = run(src, n=2)
    check('swizzle read + write',
          np.allclose(outs['Color'][0], [0.2, 0.1, 0.3, 1.0]), str(outs['Color'][0]))


def test_discard():
    src = """
    in vec2 vUV;
    out vec4 Color;
    void main() {
        if (vUV.x > 0.5) discard;
        Color = vec4(1.0);
    }
    """
    uv = np.zeros((6, 2), np.float32)
    uv[:, 0] = np.linspace(0, 1, 6)
    outs, disc, _ = run(src, inputs={'uv': uv}, n=6)
    check('discard mask', np.array_equal(np.asarray(disc), uv[:, 0] > 0.5), str(disc))


def test_preprocessor():
    src = """
    #define SCALE 3.0
    #define DOUBLE(x) ((x) * 2.0)
    #ifdef HALCYON
    #define OK 1.0
    #else
    #define OK 0.0
    #endif
    out vec4 Color;
    void main() { Color = vec4(DOUBLE(SCALE), OK, 0.0, 1.0); }
    """
    outs, _, _ = run(src, n=2)
    check('#define, function macros, #ifdef',
          np.allclose(outs['Color'][0, :2], [6.0, 1.0]), str(outs['Color'][0]))


def test_hlsl():
    src = """
    uniform float3 Tint = float3(1.0, 0.5, 0.25);
    in float3 vNormal;
    out float4 Color;
    void main() {
        float3 n = normalize(vNormal);
        float d = saturate(dot(n, float3(0.0, 0.0, 1.0)));
        float3 c = lerp(Tint * 0.2, Tint, d);
        Color = float4(saturate(c), 1.0);
    }
    """
    nrm = np.tile(np.array([[0., 0., 1.]], np.float32), (3, 1))
    outs, _, _ = run(src, inputs={'normal': nrm}, n=3, lang='HLSL')
    check('HLSL dialect (float3, saturate, lerp)',
          np.allclose(outs['Color'][0], [1.0, 0.5, 0.25, 1.0]), str(outs['Color'][0]))


def test_ternary_and_ops():
    src = """
    in vec3 vPosition;
    out vec4 Color;
    void main() {
        float x = vPosition.x;
        float y = x > 0.5 ? x * 2.0 : x * -1.0;
        int i = int(x * 4.0);
        float m = float(i % 3);
        Color = vec4(y, m, x >= 0.25 && x <= 0.75 ? 1.0 : 0.0, 1.0);
    }
    """
    p = np.zeros((8, 3), np.float32)
    p[:, 0] = np.linspace(0, 1, 8)
    outs, _, _ = run(src, inputs={'position': p})
    x = p[:, 0]
    want_y = np.where(x > 0.5, x * 2.0, -x)
    want_m = (x * 4.0).astype(np.int32) % 3
    ok = np.allclose(outs['Color'][:, 0], want_y) and \
        np.allclose(outs['Color'][:, 1], want_m)
    check('ternary, int modulo, chained logic', ok, str(outs['Color'][:, :2]))


def test_texture():
    from ..core.texture import Texture
    img = np.zeros((4, 4, 4), np.float32)
    img[:, :, 0] = np.arange(4)[None, :] / 3.0
    img[:, :, 3] = 1.0
    tex = Texture(img, filt='NEAREST')
    src = """
    uniform sampler2D tex;
    in vec2 vUV;
    out vec4 Color;
    void main() { Color = texture(tex, vUV); }
    """
    uv = np.stack([np.linspace(0.05, 0.95, 4), np.full(4, 0.5)], axis=1).astype(np.float32)
    outs, _, _ = run(src, uniforms={'tex': tex}, inputs={'uv': uv}, n=4)
    check('sampler2D binding + texture()',
          np.allclose(outs['Color'][:, 0], [0.0, 1 / 3, 2 / 3, 1.0], atol=1e-3),
          str(outs['Color'][:, 0]))


def test_error_reporting():
    prog, err = try_compile('out vec4 C; void main(){ C = nope(1.0); }')
    check('unknown function is reported, not crashed', prog is None and 'nope' in (err or ''),
          str(err))
    prog, err = try_compile('out vec4 C; void main(){ C = vec4(1.0) }')
    check('missing semicolon is reported', prog is None and 'line' in (err or '').lower(),
          str(err))


# ------------------------------------------------ names resolve at compile time
#
# R253 (review): the interpreter resolved a name when a lane reached it, so
# an identifier nothing declares compiled clean and failed when the shader
# ran -- or never, in a branch no lane took. shaders/names.py resolves every
# name before the first run; each case below compiled with error None.

_REPORTED = """uniform float gain = 1.0;

out vec4 Color;

void main() {
    Color = vec4(nothing, 0.05, gain, 1.0);
}
"""

#: (what, source, language, the whole error)
_NAME_ERRORS = [
    ('the reported shader', _REPORTED, 'GLSL',
     "line 6: undeclared identifier 'nothing'"),
    ('a function argument',
     'out vec4 Color;\nfloat f(float a) { return a * 2.0; }\nvoid main() {\n'
     '    Color = vec4(f(missing));\n}\n', 'GLSL',
     "line 4: undeclared identifier 'missing'"),
    ('a builtin argument',
     'out vec4 Color;\nvoid main() {\n    Color = vec4(sin(q));\n}\n', 'GLSL',
     "line 3: undeclared identifier 'q'"),
    ('a helper body',
     'out vec4 Color;\nfloat f(float a) { return a * b; }\nvoid main() {\n'
     '    Color = vec4(f(1.0));\n}\n', 'GLSL',
     "line 2: undeclared identifier 'b'"),
    ('an assignment target',
     'out vec4 Color;\nvoid main() {\n    Colour = vec4(1.0);\n'
     '    Color = vec4(1.0);\n}\n', 'GLSL',
     "line 3: undeclared identifier 'Colour'"),
    ('an array index',
     'out vec4 Color;\nvoid main() {\n    float a[3];\n    a[0] = 1.0;\n'
     '    Color = vec4(a[k]);\n}\n', 'GLSL',
     "line 5: undeclared identifier 'k'"),
    ('a ternary arm',
     'out vec4 Color;\nvoid main() {\n    Color = vec4(true ? 1.0 : zz);\n}\n',
     'GLSL', "line 3: undeclared identifier 'zz'"),
    ('a local read above its declaration',
     'out vec4 Color;\nvoid main() {\n    Color = vec4(x);\n'
     '    float x = 1.0;\n}\n', 'GLSL',
     "line 3: undeclared identifier 'x'"),
    ('a local read outside its block',
     'out vec4 Color;\nvoid main() {\n    { float x = 1.0; }\n'
     '    Color = vec4(x);\n}\n', 'GLSL',
     "line 4: undeclared identifier 'x'"),
    ('a loop variable read after its loop',
     'out vec4 Color;\nvoid main() {\n    float a = 0.0;\n'
     '    for (int i = 0; i < 3; i++) { a += 1.0; }\n'
     '    Color = vec4(a, float(i), 0.0, 1.0);\n}\n', 'GLSL',
     "line 5: undeclared identifier 'i'"),
    ("the caller's local read in the callee",
     'out vec4 Color;\nfloat f() { return secret; }\nvoid main() {\n'
     '    float secret = 3.0;\n    Color = vec4(f());\n}\n', 'GLSL',
     "line 2: undeclared identifier 'secret'"),
    ('a branch no lane takes',
     'uniform float gain = 1.0;\nout vec4 Color;\nvoid main() {\n'
     '    Color = vec4(1.0);\n    if (gain > 5.0) { Color = vec4(nothing); }\n'
     '}\n', 'GLSL', "line 5: undeclared identifier 'nothing'"),
    ('a function nothing calls',
     'out vec4 Color;\nfloat unused() { return nothing; }\nvoid main() {\n'
     '    Color = vec4(1.0);\n}\n', 'GLSL',
     "line 2: undeclared identifier 'nothing'"),
    ("a global's initialiser (it read zero without a word)",
     'const float TAU = 2.0 * PIE;\nout vec4 Color;\nvoid main() {\n'
     '    Color = vec4(TAU);\n}\n', 'GLSL',
     "line 1: undeclared identifier 'PIE'"),
    ("a global's initialiser reading the global BELOW it",
     'float a = b * 2.0;\nfloat b = 0.5;\nout vec4 Color;\nvoid main() {\n'
     '    Color = vec4(a, b, 0.0, 1.0);\n}\n', 'GLSL',
     "line 1: undeclared identifier 'b'"),
    ("a uniform's default",
     'uniform float g = nothing;\nout vec4 Color;\nvoid main() {\n'
     '    Color = vec4(g);\n}\n', 'GLSL',
     "line 1: undeclared identifier 'nothing'"),
    ('the HLSL dialect',
     'uniform float3 Tint = float3(0.6, 0.8, 1.0);\nout float4 Color;\n'
     'void main() {\n    Color = float4(Tint * nothing, 1.0);\n}\n', 'HLSL',
     "line 4: undeclared identifier 'nothing'"),
    ('a name a macro expands to, at the line that uses it',
     '#define LEVEL nothing\nout vec4 Color;\nvoid main() {\n'
     '    Color = vec4(LEVEL);\n}\n', 'GLSL',
     "line 4: undeclared identifier 'nothing'"),
    ('a varying the shader never declared (the error says what to write)',
     'out vec4 Color;\nvoid main() {\n    Color = vec4(vNormal, 1.0);\n}\n',
     'GLSL', "line 3: undeclared identifier 'vNormal' "
             "(declare it: in vec3 vNormal;)"),
    ('gl_FragColor with no out declared',
     'void main() {\n    gl_FragColor = vec4(1.0);\n}\n', 'GLSL',
     "line 2: undeclared identifier 'gl_FragColor' "
     "(declare it: out vec4 gl_FragColor;)"),
    ('a misspelled struct field',
     'struct S { float a; vec3 b; };\nout vec4 Color;\nvoid main() {\n'
     '    S s;\n    s.a = 1.0;\n    Color = vec4(s.aa);\n}\n', 'GLSL',
     "line 6: S has no field 'aa'"),
    ('a misspelled struct field, written',
     'struct S { float a; vec3 b; };\nout vec4 Color;\nvoid main() {\n'
     '    S s;\n    s.aa = 1.0;\n    Color = vec4(s.a);\n}\n', 'GLSL',
     "line 5: S has no field 'aa'"),
    ('a swizzle that is no swizzle',
     'out vec4 Color;\nvoid main() {\n    vec3 v = vec3(1.0);\n'
     '    Color = vec4(v.xq, 0.0, 1.0);\n}\n', 'GLSL',
     "line 4: bad swizzle 'xq'"),
    ("a swizzle past the vector's width (an IndexError when it ran)",
     'out vec4 Color;\nvoid main() {\n    vec2 v = vec2(1.0);\n'
     '    Color = vec4(v.xyz, 1.0);\n}\n', 'GLSL',
     "line 4: swizzle 'xyz' out of range for vec2"),
    ('the same, written',
     'out vec4 Color;\nvoid main() {\n    vec2 v = vec2(0.0);\n'
     '    v.z = 1.0;\n    Color = vec4(v, 0.0, 1.0);\n}\n', 'GLSL',
     "line 4: swizzle 'z' out of range for vec2"),
    ("a swizzle past a call's result",
     'out vec4 Color;\nvoid main() {\n    vec3 v = vec3(1.0);\n'
     '    Color = vec4(normalize(v).w);\n}\n', 'GLSL',
     "line 4: swizzle 'w' out of range for vec3"),
    ('an unknown function, with its line now',
     'out vec4 Color;\nvoid main() {\n    Color = vec9(1.0);\n}\n', 'GLSL',
     "line 3: unknown function 'vec9'"),
    ("an unknown function in a global's initialiser",
     'const float K = nope(1.0);\nout vec4 Color;\nvoid main() {\n'
     '    Color = vec4(K);\n}\n', 'GLSL',
     "line 1: unknown function 'nope'"),
]


def test_names_resolve_at_compile_time():
    for what, src, lang, want in _NAME_ERRORS:
        prog, err = try_compile(src, lang)
        check(f'compile error: {what}', prog is None and err == want,
              repr(err))


#: (what, source, language, uniforms, inputs, the expected Color row or None)
_FC = np.array([[3.5, 7.5, 0.25, 1.0]] * 4, np.float32)
_NAMES_FINE = [
    ('uniforms, an `in`, an out, a local, a parameter, a helper',
     'uniform float gain = 0.5;\nuniform vec3 tint = vec3(1.0, 0.5, 0.25);\n'
     'in vec3 vNormal;\nout vec4 Color;\n'
     'float half_of(float x) { float h = x * 0.5; return h; }\n'
     'void main() {\n    vec3 n = vNormal;\n'
     '    Color = vec4(tint * half_of(gain) + n * 0.0, 1.0);\n}\n',
     'GLSL', {}, {}, [0.25, 0.125, 0.0625, 1.0]),
    ('two uniforms in one declaration (they were dropped whole: no sockets, '
     'unknown identifiers when it ran)',
     'uniform float a = 0.5, b = 0.25;\nout vec4 Color;\n'
     'void main() { Color = vec4(a, b, 0.0, 1.0); }\n',
     'GLSL', {'b': np.float32(0.75)}, {}, [0.5, 0.75, 0.0, 1.0]),
    ('two plain globals in one declaration, the second reading the first',
     'float a = 0.5, b = a * 0.5;\nout vec4 Color;\n'
     'void main() { Color = vec4(a, b, 0.0, 1.0); }\n',
     'GLSL', {}, {}, [0.5, 0.25, 0.0, 1.0]),
    ("an HLSL cbuffer's members (dropped the same way) are uniforms",
     'cbuffer Params { float gain; float3 tint; };\nout float4 Color;\n'
     'void main() { Color = float4(tint * gain, 1.0); }\n',
     'HLSL', {'gain': np.float32(0.5),
              'tint': np.array([[0.2, 0.4, 0.8]], np.float32)}, {},
     [0.1, 0.2, 0.4, 1.0]),
    ('gl_FragCoord with no declaration, in main and in a helper (the root '
     'bound it; no function body could see it)',
     'out vec4 Color;\nfloat px() { return gl_FragCoord.x; }\n'
     'void main() { Color = vec4(px(), gl_FragCoord.yzw); }\n',
     'GLSL', {}, {'fragcoord': _FC}, [3.5, 7.5, 0.25, 1.0]),
    ('a constant built from the constant above it (it read ZERO: the '
     'initialiser ran in an empty scope and the failure was swallowed)',
     'const float PI = 3.0;\nconst float TAU = 2.0 * PI;\nout vec4 Color;\n'
     'void main() { Color = vec4(TAU, PI, 0.0, 1.0); }\n',
     'GLSL', {}, {}, [6.0, 3.0, 0.0, 1.0]),
    ("a uniform's default and a global reading what is declared above them",
     'const float K = 0.5;\nuniform float u = K;\nin vec3 vNormal;\n'
     'vec3 nn = vNormal * K;\nout vec4 Color;\n'
     'void main() { Color = vec4(nn.xy, u, 1.0); }\n',
     'GLSL', {}, {'normal': np.array([[0.2, 0.4, 0.8]] * 4, np.float32)},
     [0.1, 0.2, 0.5, 1.0]),
    ('a global and a function written BELOW the function that uses them '
     '(the run binds every global before main; a driver would not)',
     'out vec4 Color;\nvoid main() { Color = vec4(twice(k), k, 0.0, 1.0); }\n'
     'const float k = 0.25;\nfloat twice(float x) { return x * 2.0; }\n',
     'GLSL', {}, {}, [0.5, 0.25, 0.0, 1.0]),
    ('a parameter over a global of its name, a block local over both',
     'uniform float g = 0.25;\nout vec4 Color;\n'
     'float f(float g) { { float g = 0.75; } return g; }\n'
     'void main() { Color = vec4(f(0.5), g, 0.0, 1.0); }\n',
     'GLSL', {}, {}, [0.5, 0.25, 0.0, 1.0]),
    ('a declaration whose initialiser reads the OUTER name it shadows',
     'out vec4 Color;\nvoid main() {\n    float x = 0.5;\n'
     '    { float x = x * 0.5; Color = vec4(x); }\n}\n',
     'GLSL', {}, {}, [0.25, 0.25, 0.25, 0.25]),
    ('struct fields, read and written, through a function',
     'struct S { float a; vec3 b; };\n'
     'S mk(float x) { S s; s.a = x; s.b = vec3(x); return s; }\n'
     'out vec4 Color;\nvoid main() { Color = vec4(mk(0.25).b, mk(0.5).a); }\n',
     'GLSL', {}, {}, [0.25, 0.25, 0.25, 0.5]),
    ('swizzles: a scalar repeated, an array element, an ivec, a call result',
     'out vec4 Color;\nvoid main() {\n    float f = 0.5;\n    vec3 a[2];\n'
     '    a[0] = vec3(0.1, 0.2, 0.3);\n    ivec3 iv = ivec3(1, 2, 3);\n'
     '    vec3 v = vec3(0.0, 3.0, 4.0);\n'
     '    Color = vec4(f.xx.y, a[0].zy.x, float(iv.rg.g), '
     'normalize(v).zyx.x);\n}\n',
     'GLSL', {}, {}, [0.5, 0.3, 2.0, 0.8]),
    ('a swizzle the runtime is wide enough for though the type is not '
     '(pow is typed by its first argument and computed across the vector)',
     'out vec4 Color;\nvoid main() {\n    vec3 v = vec3(1.0, 2.0, 3.0);\n'
     '    Color = vec4(pow(2.0, v).xyz, 1.0);\n}\n',
     'GLSL', {}, {}, [2.0, 4.0, 8.0, 1.0]),
    ('a loop variable in its condition, step and body; a do-while',
     'out vec4 Color;\nvoid main() {\n    float a = 0.0;\n'
     '    for (int i = 0; i < 3; i++) { float s = float(i); a += s; }\n'
     '    int j = 0;\n    do { j++; } while (j < 4);\n'
     '    Color = vec4(a, float(j), 0.0, 1.0);\n}\n',
     'GLSL', {}, {}, [3.0, 4.0, 0.0, 1.0]),
    ('out and inout parameters written through',
     'void split(vec3 c, out float lum, inout vec3 t) {\n'
     '    lum = dot(c, vec3(0.25));\n    t *= 0.5;\n}\nout vec4 Color;\n'
     'void main() {\n    float l;\n    vec3 q = vec3(1.0);\n'
     '    split(vec3(1.0), l, q);\n    Color = vec4(q, l);\n}\n',
     'GLSL', {}, {}, [0.5, 0.5, 0.5, 0.75]),
    ('a macro, a switch, a const local, the HALCYON define',
     '#define GAIN 0.5\nout vec4 Color;\nvoid main() {\n    const float k = '
     'GAIN;\n    int m = 1;\n    float v = 0.0;\n    switch (m) { case 0: v = '
     '0.25; break; case 1: v = k; break; default: v = 1.0; }\n'
     '    Color = vec4(v, float(HALCYON), 0.0, 1.0);\n}\n',
     'GLSL', {}, {}, [0.5, 1.0, 0.0, 1.0]),
]


def test_declared_names_are_never_refused():
    """What compiles, resolves: every name the check accepts is one the run
    finds (no ShaderError out of run()), and nothing declared is refused."""
    from ..shaders.codegen import VARYINGS
    from ..shaders.compiler import DEFAULT_GLSL, DEFAULT_HLSL
    for what, src, lang, uniforms, inputs, want in _NAMES_FINE:
        prog, err = try_compile(src, lang)
        got = None
        if prog is not None:
            try:
                got = prog.run(uniforms, inputs, 4)[0]['Color'][0]
            except Exception as exc:                            # noqa: BLE001
                err = f'compiled, then {type(exc).__name__}: {exc}'
        check(f'resolves: {what}',
              got is not None and np.allclose(got, want, atol=1e-5),
              str(err if got is None else got))

    # every varying the table knows, DECLARED with its own type: compiles,
    # runs and reads the value handed in under its binding
    bad = []
    for name, (binding, t) in VARYINGS.items():
        comp = '' if t.n == 1 else '.x'
        src = (f'in {t.name()} {name};\nout vec4 Color;\n'
               f'void main() {{ Color = vec4(float({name}{comp})); }}\n')
        prog, err = try_compile(src)
        if prog is None:
            bad.append(f'{name}: {err}')
            continue
        value = np.full((4, t.n), 0.625, np.float32) if t.n > 1 \
            else np.full(4, 3 if t.base == 'int' else 0.625, np.float32)
        try:
            got = prog.run({}, {binding: value}, 4)[0]['Color'][0, 0]
        except Exception as exc:                                # noqa: BLE001
            bad.append(f'{name}: {type(exc).__name__}: {exc}')
            continue
        if abs(float(got) - float(np.ravel(value)[0])) > 1e-6:
            bad.append(f'{name}: read {got}')
    check(f'all {len(VARYINGS)} varyings resolve when declared and read '
          'their binding', not bad, '; '.join(bad))

    # the same names FREE: each is refused with the declaration to write --
    # gl_FragCoord alone is in scope without one, as GLSL defines it
    bad = []
    for name, (_binding, t) in VARYINGS.items():
        comp = '' if t.n == 1 else '.x'
        prog, err = try_compile(f'out vec4 Color;\nvoid main() {{ Color = '
                                f'vec4(float({name}{comp})); }}\n')
        want = None if name == 'gl_FragCoord' else \
            (f"line 2: undeclared identifier {name!r} "
             f"(declare it: in {t.name()} {name};)")
        if err != want:
            bad.append(f'{name}: {err}')
    check('an undeclared varying is refused with the declaration to write '
          '(gl_FragCoord is the one builtin)', not bad, '; '.join(bad))

    # the shipped starter shaders
    for label, src, lang in (('GLSL', DEFAULT_GLSL, 'GLSL'),
                             ('HLSL', DEFAULT_HLSL, 'HLSL')):
        prog, err = try_compile(src, lang)
        ran = None
        if prog is not None:
            n4 = np.tile(np.array([[0.0, 0.0, 1.0]], np.float32), (4, 1))
            ran = prog.run({}, {'normal': n4, 'view': n4, 'position': n4},
                           4)[0]['Color']
        check(f'the {label} starter shader compiles, warns of nothing and '
              'runs', ran is not None and not prog.warnings
              and bool(np.isfinite(ran).all()) and float(ran[0, 3]) == 1.0,
              str(err))


def test_a_name_error_never_reaches_the_run():
    """The other half of the rule: a program that compiled raises no name
    error however its lanes branch. Before the check, a lane reaching the
    name raised ShaderError out of run(); a lane mask that missed it ran
    clean, which is why the same source could shade one frame and break
    the next."""
    src = ('in vec3 vPosition;\nout vec4 Color;\nvoid main() {\n'
           '    Color = vec4(0.25);\n'
           '    if (vPosition.x > 0.5) { Color = vec4(nothing); }\n}\n')
    prog, err = try_compile(src)
    check('a name error behind a per-lane branch is a compile error',
          prog is None and err == "line 5: undeclared identifier 'nothing'",
          repr(err))
    fixed = src.replace('vec4(nothing)', 'vec4(vPosition.x)')
    p = np.zeros((8, 3), np.float32)
    p[:, 0] = np.linspace(0.0, 1.0, 8)
    outs, _, _ = run(fixed, inputs={'position': p})
    check('...and the corrected source takes the branch per lane',
          np.allclose(outs['Color'][:, 0], np.where(p[:, 0] > 0.5, p[:, 0],
                                                    0.25)),
          str(outs['Color'][:, 0]))


def main():
    from . import utf8_console
    utf8_console()
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, '__name__', '').startswith('test_'):
            try:
                fn()
            except Exception as e:                      # noqa: BLE001
                import traceback
                print('  FAIL ' + fn.__name__ + '  ' + repr(e))
                traceback.print_exc()
                FAILS.append(fn.__name__)
    print()
    print(f'{len(FAILS)} failure(s)' if FAILS else 'all shader tests passed')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
