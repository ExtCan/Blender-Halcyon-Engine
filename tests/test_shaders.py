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


# ------------------------------------------- what compiles, runs as written
#
# R253 (review): six things try_compile accepted (error None) that failed or
# went wrong when the program ran -- none of them a name error. Each case
# below compiled clean before; the comment over each test says what the run
# then did.

def _first(src, lang='GLSL', uniforms=None, inputs=None, key='Color', n=4):
    """(row 0 of one output, error) -- the error says which half failed."""
    prog, err = try_compile(src, lang)
    if prog is None:
        return None, f'compile: {err}'
    try:
        outs = prog.run(uniforms or {}, inputs or {}, n)[0]
    except Exception as exc:                                    # noqa: BLE001
        return None, f'compiled, then {type(exc).__name__}: {exc}'
    v = outs.get(key)
    return (None, f'no {key} in {sorted(outs)}') if v is None else (v[0], None)


def _refused(what, src, want, lang='GLSL'):
    prog, err = try_compile(src, lang)
    check(f'compile error: {what}', prog is None and err == want, repr(err))


_STRUCT = 'struct S { float a; vec3 b; };\nout vec4 Color;\n'


def test_struct_constructors():
    """`S s = S(0.5, vec3(0.1, 0.2, 0.3));` compiled and raised IndexError
    on the first field read: the parser registered S among the builtin
    types (gtypes.BY_NAME), where the interpreter found it ahead of its own
    struct branch and answered the first argument, typed S."""
    from ..shaders import gtypes
    before = dict(gtypes.BY_NAME)
    got, err = _first(_STRUCT + 'void main() {\n'
                      '    S s = S(0.5, vec3(0.1, 0.2, 0.3));\n'
                      '    Color = vec4(s.a, s.b);\n}\n')
    check('a struct constructor builds the struct (the reported shader)',
          got is not None and np.allclose(got, [0.5, 0.1, 0.2, 0.3]),
          str(err if got is None else got))
    check('...and the struct stays out of the table of builtin types',
          gtypes.BY_NAME == before and 'S' not in gtypes.BY_NAME,
          str(sorted(set(gtypes.BY_NAME) - set(before))))

    # a LATER shader that names a function S(a, b): with the struct still
    # in the global table its call was read as the constructor, and
    # answered `a`
    got, err = _first('out vec4 Color;\n'
                      'float S(float a, float b) { return a + b; }\n'
                      'void main() { Color = vec4(S(0.25, 0.5)); }\n')
    check('a struct of one shader is not a type of the next (its function '
          'S(a, b) is called)', got is not None and np.allclose(got, 0.75),
          str(err if got is None else got))

    fine = [
        ('each argument is converted to its field (an int, a scalar for a '
         'vector)',
         _STRUCT + 'void main() {\n    S s = S(2, 0.25);\n'
         '    Color = vec4(s.a * 0.25, s.b);\n}\n', [0.5, 0.25, 0.25, 0.25]),
        ('a struct inside a struct, and a swizzle of a constructed field',
         'struct In { vec2 p; float w; };\nstruct Out { In i; vec3 c; };\n'
         'out vec4 Color;\nvoid main() {\n'
         '    Out o = Out(In(vec2(0.25, 0.5), 2.0), vec3(1.0, 0.5, 0.0));\n'
         '    Color = vec4(o.i.p * o.i.w, Out(o.i, vec3(0.125)).c.zy);\n}\n',
         [0.5, 1.0, 0.125, 0.125]),
        ('returned from a function, passed to one, read without a variable',
         _STRUCT + 'S mk(float x) { return S(x, vec3(x * 2.0)); }\n'
         'float sum(S s) { return s.a + s.b.x; }\n'
         'void main() { Color = vec4(mk(0.25).b.xy, sum(mk(0.125)), '
         'S(0.75, vec3(0.0)).a); }\n', [0.5, 0.5, 0.375, 0.75]),
        ('a field of a constructed struct written afterwards',
         _STRUCT + 'void main() {\n    S s = S(0.5, vec3(0.25));\n'
         '    s.b.y = 1.0;\n    s.a += 0.25;\n'
         '    Color = vec4(s.b, s.a);\n}\n',
         [0.25, 1.0, 0.25, 0.75]),
        ('declared and written field by field, as before',
         _STRUCT + 'void main() {\n    S s;\n    s.a = 0.5;\n'
         '    s.b = vec3(0.1, 0.2, 0.3);\n    Color = vec4(s.a, s.b);\n}\n',
         [0.5, 0.1, 0.2, 0.3]),
    ]
    for what, src, want in fine:
        got, err = _first(src)
        check(f'struct: {what}',
              got is not None and np.allclose(got, want, atol=1e-6),
              str(err if got is None else got))

    # per lane: the constructor under a mask assigns only the lanes that
    # took the branch
    src = (_STRUCT + 'in vec3 vPosition;\nvoid main() {\n'
           '    S s = S(0.25, vec3(0.0));\n'
           '    if (vPosition.x > 0.5) { s = S(vPosition.x, vec3(1.0)); }\n'
           '    Color = vec4(s.a, s.b);\n}\n')
    p = np.zeros((8, 3), np.float32)
    p[:, 0] = np.linspace(0.0, 1.0, 8)
    outs, _, _ = run(src, inputs={'position': p})
    hot = p[:, 0] > 0.5
    check('struct: constructed per lane under a branch',
          np.allclose(outs['Color'][:, 0], np.where(hot, p[:, 0], 0.25))
          and np.allclose(outs['Color'][:, 1], np.where(hot, 1.0, 0.0)),
          str(outs['Color'][:, :2]))

    # one argument per field: the count is refused where it is written
    _refused('a struct constructor with too few arguments',
             _STRUCT + 'void main() {\n    S s = S(0.5);\n'
             '    Color = vec4(s.a);\n}\n',
             'line 4: S() takes 2 arguments (one per field), got 1')
    _refused('...with too many',
             _STRUCT + 'void main() {\n    S s = S(0.5, vec3(1.0), 2.0);\n'
             '    Color = vec4(s.a);\n}\n',
             'line 4: S() takes 2 arguments (one per field), got 3')
    _refused("...and the one-argument S(s) the scalar path answered by "
             "accident (no driver compiles it)",
             _STRUCT + 'void main() {\n    S s;\n    s.a = 0.5;\n'
             '    S t = S(s);\n    Color = vec4(t.a);\n}\n',
             'line 6: S() takes 2 arguments (one per field), got 1')


def test_comma_operator_and_every_operator_the_parser_knows():
    """`for (int i = 0; i < 3; i++, b += 1.0)` parsed to ('seq', l, r) and
    the interpreter had no case: 'unsupported expression' when it ran."""
    got, err = _first('out vec4 Color;\nvoid main() {\n    float b = 0.0;\n'
                      '    for (int i = 0; i < 3; i++, b += 1.0) { }\n'
                      '    Color = vec4(b);\n}\n')
    check('the comma operator in a for step (the reported shader)',
          got is not None and np.allclose(got, 3.0),
          str(err if got is None else got))
    fine = [
        ('in a for initialiser and a statement; its value is the right side',
         'out vec4 Color;\nvoid main() {\n    float a, b;\n    int i, j;\n'
         '    for (i = 0, j = 4; i < j; i++, j--) { }\n'
         '    a = 0.25, b = 0.5;\n    float c = (a += 0.25, a * 0.5);\n'
         '    Color = vec4(a, b, c, float(i + j));\n}\n',
         [0.5, 0.5, 0.25, 4.0]),
        ('three in a row, left to right',
         'out vec4 Color;\nvoid main() {\n    float a = 1.0;\n'
         '    float b = (a *= 2.0, a += 1.0, a * 0.25);\n'
         '    Color = vec4(a, b, 0.0, 1.0);\n}\n', [3.0, 0.75, 0.0, 1.0]),
        ('arguments stay arguments: f(a, b) is not a comma',
         'out vec4 Color;\nfloat sub(float a, float b) { return a - b; }\n'
         'void main() { Color = vec4(sub(0.75, 0.25), max(0.25, 0.125), '
         'vec2(1.0, 0.0)); }\n', [0.5, 0.25, 1.0, 0.0]),
    ]
    for what, src, want in fine:
        got, err = _first(src)
        check(f'comma: {what}',
              got is not None and np.allclose(got, want, atol=1e-6),
              str(err if got is None else got))

    # per lane: a loop whose trip count differs by lane, stepping two
    # variables through the comma
    src = ('in vec3 vPosition;\nout vec4 Color;\nvoid main() {\n'
           '    int limit = int(vPosition.x * 8.0);\n    float acc = 0.0;\n'
           '    for (int i = 0; i < limit; i++, acc += 0.5) { }\n'
           '    Color = vec4(acc, 0.0, 0.0, 1.0);\n}\n')
    p = np.zeros((8, 3), np.float32)
    p[:, 0] = np.arange(8) / 8.0
    outs, _, _ = run(src, inputs={'position': p})
    check('comma: per-lane trip counts step both variables',
          np.allclose(outs['Color'][:, 0], np.arange(8) * 0.5),
          str(outs['Color'][:, 0]))

    # the same gap, looked for everywhere: every binary and assignment
    # operator the parser accepts has a case in the interpreter. `^^` was
    # the other one without ('unsupported operator' when a lane got there)
    from ..shaders.parser import ASSIGN_OPS, BIN_PRECEDENCE
    bad = []
    for op in [o for level in BIN_PRECEDENCE for o in level]:
        src = (f'out vec4 Color;\nvoid main() {{\n    int a = 6;\n'
               f'    int b = 3;\n    Color = vec4(float(a {op} b));\n}}\n')
        got, err = _first(src)
        if got is None:
            bad.append(f'{op}: {err}')
    for op in ASSIGN_OPS:
        src = (f'out vec4 Color;\nvoid main() {{\n    int a = 6;\n'
               f'    a {op} 3;\n    Color = vec4(float(a));\n}}\n')
        got, err = _first(src)
        if got is None:
            bad.append(f'{op}: {err}')
    n_ops = sum(len(level) for level in BIN_PRECEDENCE) + len(ASSIGN_OPS)
    check(f'all {n_ops} binary and assignment operators the parser knows '
          'run', not bad, '; '.join(bad))
    got, err = _first('out vec4 Color;\nvoid main() {\n'
                      '    Color = vec4(float(true ^^ false), '
                      'float(true ^^ true), float(false ^^ false), '
                      'float(0.5 ^^ 0.0));\n}\n')
    check('^^ is exclusive or (its truth table)',
          got is not None and np.allclose(got, [1.0, 0.0, 0.0, 1.0]),
          str(err if got is None else got))


_F1 = 'out vec4 Color;\nfloat f(float a) { return a * 2.0; }\n'


def test_argument_counts_are_checked_at_compile_time():
    """With `float f(float a)`, both f(1.0, 2.0) and f() compiled and ran:
    the extra argument dropped, the missing one zero. The guard that
    refuses it (R235) lives in the generator nothing calls any more. A
    builtin with the wrong count raised only when a lane reached it."""
    from ..shaders.builtins import BUILTINS
    cases = [
        ('one argument too many (it was dropped)',
         _F1 + 'void main() { Color = vec4(f(1.0, 2.0)); }\n',
         'line 3: f() takes 1 argument, got 2', 'GLSL'),
        ('none where one is needed (it read zero)',
         _F1 + 'void main() { Color = vec4(f()); }\n',
         'line 3: f() takes 1 argument, got 0', 'GLSL'),
        ('in a function nothing calls',
         _F1 + 'float g() { return f(1.0, 2.0); }\n'
         'void main() { Color = vec4(f(1.0)); }\n',
         'line 3: f() takes 1 argument, got 2', 'GLSL'),
        ('two functions of the name, neither takes three',
         'out vec4 Color;\nfloat f(float a) { return a; }\n'
         'float f(float a, float b) { return a + b; }\n'
         'void main() {\n    Color = vec4(f(1.0, 2.0, 3.0));\n}\n',
         'line 5: f() takes 1 or 2 arguments, got 3', 'GLSL'),
        ('a function with no parameters handed one',
         'out vec4 Color;\nfloat one() { return 1.0; }\n'
         'void main() { Color = vec4(one(2.0)); }\n',
         'line 3: one() takes 0 arguments, got 1', 'GLSL'),
        ("the shader's own mix(a, b) called as the builtin's three (its "
         "function hides the builtin; the third was dropped)",
         'out vec4 Color;\nvec3 mix(vec3 a, vec3 b) { return a + b; }\n'
         'void main() {\n'
         '    Color = vec4(mix(vec3(0.0), vec3(1.0), 0.25), 1.0);\n}\n',
         'line 4: mix() takes 2 arguments, got 3', 'GLSL'),
        ('a builtin (it raised when a lane reached it)',
         'out vec4 Color;\nvoid main() {\n    Color = vec4(sin(1.0, 2.0));\n'
         '}\n', 'line 3: sin() takes 1 argument, got 2', 'GLSL'),
        ('a builtin in a branch no lane takes (it never raised)',
         'uniform float gain = 0.0;\nout vec4 Color;\nvoid main() {\n'
         '    Color = vec4(1.0);\n'
         '    if (gain > 5.0) { Color = vec4(clamp(gain, 0.0)); }\n}\n',
         'line 5: clamp() takes 3 arguments, got 2', 'GLSL'),
        ('a builtin that takes a range',
         'out vec4 Color;\nvoid main() {\n'
         '    Color = vec4(atan(1.0, 2.0, 3.0));\n}\n',
         'line 3: atan() takes 1 to 2 arguments, got 3', 'GLSL'),
        ('a parameter without a default left out (HLSL)',
         'out float4 Color;\n'
         'float f(float a, float b = 0.5) { return a + b; }\n'
         'void main() { Color = float4(f(), 0, 0, 1); }\n',
         'line 3: f() takes 1 to 2 arguments, got 0', 'HLSL'),
    ]
    for what, src, want, lang in cases:
        _refused(what, src, want, lang)

    fine = [
        ('the right count, and a function chosen by its count',
         'out vec4 Color;\nfloat f(float a) { return a; }\n'
         'float f(float a, float b) { return a + b; }\n'
         'void main() { Color = vec4(f(0.25), f(0.25, 0.5), 0.0, 1.0); }\n',
         'GLSL', [0.25, 0.75, 0.0, 1.0]),
        ('f(void) and a function with no parameters',
         'out vec4 Color;\nfloat one(void) { return 1.0; }\n'
         'float half_() { return 0.5; }\n'
         'void main() { Color = vec4(one(), half_(), 0.0, 1.0); }\n',
         'GLSL', [1.0, 0.5, 0.0, 1.0]),
        ('an HLSL default is READ when the argument is left out (it was '
         'parsed and dropped: the parameter read zero)',
         'static const float K = 0.25;\nout float4 Color;\n'
         'float f(float a, float b = 0.5, float c = K * 2.0) '
         '{ return a + b + c; }\n'
         'void main() { Color = float4(f(0.25), f(0.25, 0.25), '
         'f(0.25, 0.25, 0.25), 1); }\n',
         'HLSL', [1.25, 1.0, 0.75, 1.0]),
        ('constructors keep their own rules (a splat, a vector and a '
         'scalar, a cast)',
         'out vec4 Color;\nvoid main() {\n    vec3 v = vec3(0.5);\n'
         '    Color = vec4(v.xy, float(1), vec4(v, 0.25).w);\n}\n',
         'GLSL', [0.5, 0.5, 1.0, 0.25]),
    ]
    for what, src, lang, want in fine:
        got, err = _first(src, lang)
        check(f'counts: {what}',
              got is not None and np.allclose(got, want, atol=1e-6),
              str(err if got is None else got))

    # every builtin: each count of its range compiles, one under and one
    # over are refused by name, at compile time
    bad = []
    for name, (_rt, _rule, lo, hi) in sorted(BUILTINS.items()):
        for argc in range(max(lo - 1, 0), hi + 2):
            src = ('out vec4 Color;\nvoid main() {\n    Color = vec4(1.0);\n'
                   f'    {name}({", ".join(["1.0"] * argc)});\n}}\n')
            prog, err = try_compile(src)
            ok = lo <= argc <= hi
            if ok != (prog is not None) or (
                    not ok and not (err.startswith(f'line 4: {name}() takes ')
                                    and err.endswith(f', got {argc}'))):
                bad.append(f'{name}/{argc}: {err}')
    check(f'all {len(BUILTINS)} builtins take every count of their range '
          'and no other', not bad, '; '.join(bad[:6]))


_AB = ('out vec4 Color;\nvoid main() {\n    vec2 a = vec2(1.0);\n'
       '    vec3 b = vec3(2.0);\n')


def test_vector_widths_are_checked_at_compile_time():
    """`vec2 a; vec3 b; a + b` compiled and raised 'cannot combine vec2
    with vec3' from gtypes.promote when a lane ran the operator."""
    cases = [
        ('vec2 + vec3 (the reported shader)',
         _AB + '    Color = vec4(a + b, 1.0);\n}\n',
         'line 5: cannot combine vec2 with vec3'),
        ('the line is the operator\'s, not the statement\'s',
         _AB + '    Color = vec4(a\n        * b,\n        1.0);\n}\n',
         'line 6: cannot combine vec2 with vec3'),
        ('a comparison',
         _AB + '    Color = vec4(a == b ? 1.0 : 0.0);\n}\n',
         'line 5: cannot combine vec2 with vec3'),
        ('the two arms of a ternary',
         _AB + '    Color = vec4(true ? b : a, 1.0);\n}\n',
         'line 5: cannot combine vec3 with vec2'),
        ('a compound assignment',
         _AB + '    a += b;\n    Color = vec4(a, a);\n}\n',
         'line 5: cannot combine vec2 with vec3'),
        ('the arguments of min (NumPy raised on the two shapes)',
         _AB + '    Color = vec4(min(a, b), 1.0, 1.0);\n}\n',
         'line 5: cannot combine vec2 with vec3'),
        ('an ivec3 with a vec4, through a swizzle and a call',
         'out vec4 Color;\nvoid main() {\n    ivec3 i = ivec3(1);\n'
         '    vec4 v = vec4(1.0);\n'
         '    Color = vec4(normalize(v).wzyx - i.zyx, 1.0);\n}\n',
         'line 5: cannot combine vec4 with ivec3'),
        ('in a branch no lane takes',
         'uniform float gain = 0.0;\n' + _AB + '    Color = vec4(1.0);\n'
         '    if (gain > 5.0) { Color = vec4(a * b, 1.0); }\n}\n',
         'line 7: cannot combine vec2 with vec3'),
        ('in a function nothing calls',
         'vec3 unused(vec2 a, vec3 b) {\n    return a - b;\n}\n'
         'out vec4 Color;\nvoid main() { Color = vec4(1.0); }\n',
         'line 2: cannot combine vec2 with vec3'),
        ('the HLSL dialect (it never ran there either)',
         'out float4 Color;\nvoid main() {\n    float2 a = float2(1, 1);\n'
         '    float3 b = float3(2, 2, 2);\n'
         '    Color = float4(a / b, 1);\n}\n',
         'line 5: cannot combine vec2 with vec3'),
    ]
    for what, src, want in cases:
        _refused(what, src, want,
                 'HLSL' if 'float4' in src else 'GLSL')

    fine = [
        ('a scalar with any vector, a narrowing swizzle, a constructor',
         _AB + '    Color = vec4(a * 2.0 + b.xy, (vec3(a, 1.0) - b).z, '
         '(0.5 * b).x);\n}\n', [4.0, 4.0, -1.0, 1.0]),
        ('matrix * vector, vector * matrix, matrix * matrix',
         'out vec4 Color;\nvoid main() {\n    mat3 m = mat3(2.0);\n'
         '    vec3 v = vec3(0.25, 0.5, 0.75);\n'
         '    Color = vec4((m * v).xy, (v * m).z, (m * m * v).x);\n}\n',
         [0.5, 1.0, 1.5, 1.0]),
        ('same widths of different base types, a ternary of equal arms',
         'out vec4 Color;\nvoid main() {\n    ivec2 i = ivec2(1, 2);\n'
         '    vec2 v = vec2(0.5);\n    bool c = v.x < 1.0;\n'
         '    Color = vec4(v * i, c ? v + i : v - i);\n}\n',
         [0.5, 1.0, 1.5, 2.5]),
    ]
    for what, src, want in fine:
        got, err = _first(src)
        check(f'widths: {what}',
              got is not None and np.allclose(got, want, atol=1e-6),
              str(err if got is None else got))

    # where the interpreter's type is not the language's, nothing is judged:
    # a text a driver compiles must never be refused on this front-end's
    # own shortcut (every pass the engine emits is pre-flighted here)
    loose = [
        ('two functions of one name and count: the run takes the first, '
         'so f(vec3) is typed vec2 here',
         'out vec4 Color;\nvec2 f(vec2 x) { return x; }\n'
         'vec3 f(vec3 x) { return x; }\nvoid main() {\n'
         '    vec3 v = vec3(1.0);\n    Color = vec4(f(v) + v, 1.0);\n}\n'),
        ('a matrix times a scalar is typed as a vector here: a 2x3 matrix '
         'doubled, times a vec2, is a vec3 times a vec2 by that rule',
         'out vec4 Color;\nvoid main() {\n    vec3 c = vec3(1.0);\n'
         '    vec2 r = vec2(0.5);\n'
         '    vec3 w = (outerProduct(c, r) * 2.0) * r;\n'
         '    Color = vec4(w, 1.0);\n}\n'),
    ]
    for what, src in loose:
        prog, err = try_compile(src)
        check(f'widths, not judged: {what}', prog is not None, repr(err))

    # the rule itself, held against the run's own: for every pair of types
    # and every operator that goes through promote, the compile refuses
    # exactly when promote does, with promote's words
    from ..shaders.gtypes import BY_NAME, promote
    from ..shaders.lexer import ShaderError
    names = ('float', 'int', 'vec2', 'vec3', 'vec4', 'ivec2', 'ivec3',
             'uvec4', 'bvec2', 'mat2', 'mat3')
    bad, refused = [], 0
    for ta in names:
        for tb in names:
            try:
                promote(BY_NAME[ta], BY_NAME[tb])
                want = None
            except ShaderError as exc:
                want = f'line 5: {exc.msg}'
            for op in ('+', '-', '/', '%', '<', '==', '&'):
                src = (f'out vec4 Color;\nvoid main() {{\n    {ta} a;\n'
                       f'    {tb} b;\n    a {op} b;\n'
                       '    Color = vec4(1.0);\n}\n')
                prog, err = try_compile(src)
                refused += err is not None
                if err != want:
                    bad.append(f'{ta} {op} {tb}: {err}')
    check(f'{len(names) ** 2} type pairs x 7 operators: refused exactly '
          f'where the run\'s promote refuses ({refused} refusals)',
          not bad and refused > 0, '; '.join(bad[:6]))


_FRAG = np.array([[3.5, 7.5, 0.25, 1.0], [10.5, 2.5, 0.5, 1.0]], np.float32)


def test_entry_point_parameters_are_bound():
    """`void mainImage(out vec4 fragColor, in vec2 fragCoord)` compiled
    into a program with no outputs: the run called the entry with no
    arguments, so the `in` read zero and nothing the `out` was given left
    the call."""
    src = ('void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n'
           '    fragColor = vec4(fragCoord, 0.5, 1.0);\n}\n')
    prog, err = try_compile(src)
    check('mainImage: its out parameter is the output of the program',
          prog is not None and [o[0] for o in prog.outputs] == ['fragColor']
          and [(o['name'], o['kind']) for o in prog.output_schema()]
          == [('fragColor', 'RGBA')] and not prog.warnings, repr(err))
    if prog is not None:
        outs = prog.run({}, {'fragcoord': _FRAG, 'fragcoord2': _FRAG[:, :2]},
                        2)[0]
        check('mainImage: fragCoord is the fragment coordinate, per lane',
              np.allclose(outs['fragColor'],
                          [[3.5, 7.5, 0.5, 1.0], [10.5, 2.5, 0.5, 1.0]]),
              str(outs.get('fragColor')))
        outs = prog.run({}, {'fragcoord': _FRAG}, 2)[0]
        check('...and the xy of gl_FragCoord when that is all the caller '
              'hands over', np.allclose(outs['fragColor'][:, :2],
                                        _FRAG[:, :2]),
              str(outs.get('fragColor')))

    fine = [
        ('a Shadertoy body: fragCoord over iResolution, iTime',
         'in vec3 iResolution;\nin float iTime;\n'
         'void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n'
         '    vec2 uv = fragCoord / iResolution.xy;\n'
         '    fragColor = vec4(uv, iTime, 1.0);\n}\n', 'GLSL',
         {'fragcoord2': _FRAG[:, :2], 'time': np.float32(0.25),
          'resolution': np.array([[14.0, 10.0, 1.0]], np.float32)},
         'fragColor', [0.25, 0.75, 0.25, 1.0]),
        ('the parameter over an out global of its name; an inout starts '
         'at zero and is an output too',
         'out vec4 fragColor;\n'
         'void mainImage(out vec4 fragColor, inout float glow) {\n'
         '    glow += 0.5;\n    fragColor = vec4(glow);\n}\n', 'GLSL', {},
         'fragColor', [0.5, 0.5, 0.5, 0.5]),
        ('parameters named for what they bind: uv, normal, a vec3 of the '
         'four-wide vertex colour (its rgb)',
         'out vec4 Color;\nvoid main(in vec2 uv, in vec3 normal, '
         'in vec3 vColor) {\n'
         '    Color = vec4(uv.y, normal.z, vColor.gb);\n}\n', 'GLSL',
         {'uv': np.array([[0.25, 0.5]] * 2, np.float32),
          'normal': np.array([[0.0, 0.0, 0.75]] * 2, np.float32),
          'color': np.array([[0.1, 0.2, 0.3, 0.4]] * 2, np.float32)},
         'Color', [0.5, 0.75, 0.2, 0.3]),
        ('an HLSL entry that RETURNS its colour, found by name below a '
         'helper, its parameter bound by name',
         'float3 shade(float3 n) { return n * 0.5; }\n'
         'float4 PSMain(float2 uv : TEXCOORD0) : SV_Target {\n'
         '    return float4(shade(float3(uv, 1.0)), 1.0);\n}\n', 'HLSL',
         {'uv': np.array([[0.5, 1.0]] * 2, np.float32)},
         '__return', [0.25, 0.5, 0.5, 1.0]),
        ('a main that returns (the Result socket was declared and read '
         'nothing)',
         'vec4 main() { return vec4(0.25, 0.5, 0.75, 1.0); }\n', 'GLSL', {},
         '__return', [0.25, 0.5, 0.75, 1.0]),
    ]
    for what, src, lang, inputs, key, want in fine:
        got, err = _first(src, lang, inputs=inputs, key=key, n=2)
        check(f'entry: {what}',
              got is not None and np.allclose(got, want, atol=1e-6),
              str(err if got is None else got))

    prog, err = try_compile(fine[1][1])
    check('entry: an out parameter named like an out global is one output, '
          'and the inout is listed after it',
          prog is not None and [o[0] for o in prog.outputs]
          == ['fragColor', 'glow'], repr(err))
    prog, err = try_compile(fine[3][1], 'HLSL')
    check('entry: the returned value is the Result output',
          prog is not None and [(o['name'], o['key'], o['kind'])
                                for o in prog.output_schema()]
          == [('Result', '__return', 'RGBA')], repr(err))

    # nothing changes for the shape every other shader has
    from ..shaders.compiler import DEFAULT_GLSL, DEFAULT_HLSL
    plain = []
    for src, lang in ((DEFAULT_GLSL, 'GLSL'), (DEFAULT_HLSL, 'HLSL'),
                      ('out vec4 A;\nout float B;\nvoid main(void) '
                       '{ A = vec4(1.0); B = 0.5; }\n', 'GLSL')):
        prog, err = try_compile(src, lang)
        plain.append(None if prog is None else [o[0] for o in prog.outputs])
    check('entry: a void main() keeps exactly its declared outputs',
          plain == [['Color'], ['Color'], ['A', 'B']], str(plain))


def test_unknown_inputs_warn():
    """Nothing was ever warned of: Interpreter.warnings was never written
    to, so a declared input the renderer binds nothing to (a misspelled
    vNormals) read zero in silence and the node's warning box stayed
    empty."""
    from ..shaders.codegen import INPUT_KEYS, VARYINGS
    src = ('in vec3 vNormals;\nout vec4 Color;\n'
           'void main() { Color = vec4(vNormals, 1.0); }\n')
    prog, err = try_compile(src)
    n4 = np.tile(np.array([[0.0, 0.0, 1.0]], np.float32), (4, 1))
    got = None if prog is None else prog.run({}, {'normal': n4}, 4)[0]['Color']
    check('a misspelled input is warned of by name (the reported shader), '
          'and reads zero as the warning says',
          prog is not None and prog.warnings
          == ["unknown input 'vNormals': bound to zero"]
          and np.allclose(got, [0.0, 0.0, 0.0, 1.0]),
          repr(err if prog is None else prog.warnings))

    def warns(src, lang='GLSL'):
        prog, err = try_compile(src, lang)
        return err if prog is None else prog.warnings

    check('varying and attribute declarations are inputs too, each warned '
          'of once and in source order',
          warns('varying vec2 vTexel;\nattribute vec3 aSpin;\nin vec3 '
                'vNormal;\nout vec4 Color;\nvoid main() { Color = '
                'vec4(vTexel, aSpin.x, vNormal.x); }\n')
          == ["unknown input 'vTexel': bound to zero",
              "unknown input 'aSpin': bound to zero"],
          str(warns('varying vec2 vTexel;\nattribute vec3 aSpin;\nin vec3 '
                    'vNormal;\nout vec4 Color;\nvoid main() { Color = '
                    'vec4(vTexel, aSpin.x, vNormal.x); }\n')))

    # every name the table binds: declared, it warns of nothing
    noisy = []
    for name, (_binding, t) in VARYINGS.items():
        w = warns(f'in {t.name()} {name};\nout vec4 Color;\n'
                  'void main() { Color = vec4(1.0); }\n')
        if w:
            noisy.append(f'{name}: {w}')
    check(f'none of the {len(VARYINGS)} inputs the table binds is warned '
          'of', not noisy, '; '.join(noisy))

    # a name that is itself a key the renderer supplies reads that key (the
    # run looks a declared input up under its own name first), so "bound to
    # zero" would be false: no warning
    prog, err = try_compile('in vec3 position;\nout vec4 Color;\n'
                            'void main() { Color = vec4(position, 1.0); }\n')
    p = np.tile(np.array([[0.25, 0.5, 0.75]], np.float32), (4, 1))
    got = None if prog is None else prog.run({}, {'position': p},
                                             4)[0]['Color'][0]
    check("an input named for the renderer's own key reads it, unwarned",
          prog is not None and not prog.warnings
          and np.allclose(got, [0.25, 0.5, 0.75, 1.0]),
          repr(err if prog is None else (prog.warnings, got)))

    # an entry parameter is an input the same way
    check('an entry parameter nothing binds is warned of; fragCoord and uv '
          'are not',
          warns('void mainImage(out vec4 fragColor, in vec2 fragCoord, '
                'in vec3 wobble) {\n    fragColor = vec4(fragCoord, '
                'wobble.xy);\n}\n')
          == ["unknown input 'wobble': bound to zero"]
          and warns('out vec4 Color;\nvoid main(in vec2 uv) '
                    '{ Color = vec4(uv, 0.0, 1.0); }\n') == [],
          str(warns('void mainImage(out vec4 fragColor, in vec2 fragCoord, '
                    'in vec3 wobble) {\n    fragColor = vec4(fragCoord, '
                    'wobble.xy);\n}\n')))

    # the table and the renderer agree on the keys: what core/nodeeval hands
    # a program is exactly INPUT_KEYS less the one both roads leave at zero
    import types as _types
    from ..core import nodeeval
    z3 = np.zeros((2, 3), np.float32)
    ctx = _types.SimpleNamespace(
        n=2, px=np.zeros(2, np.int32), py=np.zeros(2, np.int32),
        depth=np.ones(2, np.float32), I=np.tile([[0.0, 0.0, -1.0]], (2, 1)),
        P=z3, N=z3 + 1.0, Ng=z3, T=None, uv=z3[:, :2], uv2=z3[:, :2],
        vcol=np.ones((2, 4), np.float32), generated=z3, width=4, height=4,
        backfacing=np.zeros(2, np.float32), camera_pos=np.zeros(3, np.float32),
        time=0.0, frame=1, random=np.zeros(2, np.float32))
    supplied = set(nodeeval._shader_inputs(ctx))
    check("the renderer supplies every key the warning counts as bound "
          "(bitangent alone reads zero on both roads)",
          supplied == set(INPUT_KEYS) - {'bitangent'},
          str(sorted(supplied ^ set(INPUT_KEYS))))


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
