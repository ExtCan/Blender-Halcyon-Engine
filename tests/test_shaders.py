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
        # this one stood among the loosely typed below: a matrix times a
        # scalar was typed a vector (and raised from einsum when run), so a
        # 2x3 matrix doubled, times a vec2, was a vec3 times a vec2 by that
        # rule and had to be stepped around. It is the matrix now, and runs
        ('a matrix times a scalar is the matrix: a 2x3 doubled, times a '
         'vec2, is a vec3',
         'out vec4 Color;\nvoid main() {\n    vec3 c = vec3(1.0);\n'
         '    vec2 r = vec2(0.5);\n'
         '    vec3 w = (outerProduct(c, r) * 2.0) * r;\n'
         '    Color = vec4(w, 1.0);\n}\n', [1.0, 1.0, 1.0, 1.0]),
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
        ('a matrix times a value typed one number that holds a vector '
         '(pow(2.0, v)): the run multiplies by the vector, so the product '
         'is a vec3 there and its swizzle, plus a vec2, is the run\'s',
         'out vec4 Color;\nvoid main() {\n    vec3 v = vec3(1.0);\n'
         '    vec2 r = vec2(0.5);\n'
         '    vec2 w = (mat3(2.0) * pow(2.0, v)).xy + r;\n'
         '    Color = vec4(w, 0.0, 1.0);\n}\n'),
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


# ----------------------- matrices, indexed writes, array fields, int division
#
# R253 (review): four more things try_compile accepted (error None) that
# failed, or answered another number, when the program ran. Each shader
# named "the reported shader" below is the text the defect was reproduced
# with; the comment over each test says what the run did with it.

def _body(body, head=''):
    return 'out vec4 Color;\n' + head + 'void main() {\n' + body + '}\n'


#: diag(1, 2, 3) written as nine numbers: the matrix every case below holds
_D9 = 'mat3(1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0)'
_DIAG = f'    mat3 m = {_D9};\n'

#: (what, declarations above main, statements that leave diag(1, 2, 3) in m)
_MATRIX_MADE = [
    ('nine numbers', '', _DIAG),
    ('three columns', '',
     '    mat3 m = mat3(vec3(1.0, 0.0, 0.0), vec3(0.0, 2.0, 0.0), '
     'vec3(0.0, 0.0, 3.0));\n'),
    ('declared, its columns written', '',
     '    mat3 m;\n    m[0] = vec3(1.0, 0.0, 0.0);\n'
     '    m[1] = vec3(0.0, 2.0, 0.0);\n    m[2] = vec3(0.0, 0.0, 3.0);\n'),
    ('declared, its components written', '',
     '    mat3 m;\n    m[0][0] = 1.0;\n    m[1][1] = 2.0;\n'
     '    m[2][2] = 3.0;\n'),
    ('the identity, its diagonal written through swizzles', '',
     '    mat3 m = mat3(1.0);\n    m[1].y = 2.0;\n    m[2].z = 3.0;\n'),
    ('returned by a function', f'mat3 diag() {{ return {_D9}; }}\n',
     '    mat3 m = diag();\n'),
    ('returned by a function that declares and fills it',
     'mat3 diag() {\n    mat3 r;\n    for (int i = 0; i < 3; i++) '
     '{ r[i][i] = float(i + 1); }\n    return r;\n}\n',
     '    mat3 m = diag();\n'),
    ('passed through a parameter', 'mat3 same(mat3 x) { return x; }\n',
     f'    mat3 m = same({_D9});\n'),
    ('written by an out parameter',
     f'void fill(out mat3 x) {{ x = {_D9}; }}\n',
     '    mat3 m;\n    fill(m);\n'),
    ('a struct field', 'struct T { mat3 m; float k; };\n',
     f'    T t;\n    t.m = {_D9};\n    mat3 m = t.m;\n'),
    ('a declared struct field, its columns written',
     'struct T { mat3 m; float k; };\n',
     '    T t;\n    t.m[0] = vec3(1.0, 0.0, 0.0);\n    t.m[1].y = 2.0;\n'
     '    t.m[2][2] = 3.0;\n    mat3 m = t.m;\n'),
    ('an array element', '',
     f'    mat3 a[2];\n    a[1] = {_D9};\n    mat3 m = a[1];\n'),
    ('a global constant', f'const mat3 G = {_D9};\n', '    mat3 m = G;\n'),
    ('a declared global, written in main', 'mat3 G;\n',
     '    G[0].x = 1.0;\n    G[1] = vec3(0.0, 2.0, 0.0);\n'
     '    G[2][2] = 3.0;\n    mat3 m = G;\n'),
    ('one arm of a ternary', '',
     f'    mat3 m = true ? {_D9} : mat3(0.0);\n'),
    ('a product', '', f'    mat3 m = {_D9} * mat3(1.0);\n'),
    ('transposed twice', '',
     f'    mat3 m = transpose(transpose({_D9}));\n'),
    ('cut from a mat4 that was built from it', '',
     f'    mat3 m = mat3(mat4({_D9}));\n'),
]

#: (what, statements reading m into Color, the Color of diag(1, 2, 3))
_MATRIX_USED = [
    ('m * v', '    Color = vec4(m * vec3(1.0), 1.0);\n', [1, 2, 3, 1]),
    ('v * m', '    Color = vec4(vec3(1.0, 0.5, 2.0) * m, 1.0);\n',
     [1, 1, 6, 1]),
    ('a column', '    Color = vec4(m[1], 1.0);\n', [0, 2, 0, 1]),
    ('a component', '    Color = vec4(m[0][0], m[1][1], m[2][2], m[1][2]);\n',
     [1, 2, 3, 0]),
    ('a swizzle of a column', '    Color = vec4(m[2].zyx, m[1].y);\n',
     [3, 0, 0, 2]),
    ('a column through a variable index',
     '    int i = 2;\n    Color = vec4(m[i], m[i - 1][i - 1]);\n',
     [0, 0, 3, 2]),
    ('a column written, then m * v',
     '    m[0] = vec3(0.5);\n    Color = vec4(m * vec3(1.0), 1.0);\n',
     [0.5, 2.5, 3.5, 1]),
    ('a component written, then m * v',
     '    m[2][0] = 4.0;\n    Color = vec4(m * vec3(1.0), 1.0);\n',
     [5, 2, 3, 1]),
    ('a swizzle of a column written',
     '    m[1].xz = vec2(7.0, 8.0);\n    Color = vec4(m[1], 1.0);\n',
     [7, 2, 8, 1]),
    ('a component incremented, a column added to',
     '    m[1][1]++;\n    m[2] += vec3(1.0);\n'
     '    Color = vec4(m[1].y, m[2]);\n', [3, 1, 1, 4]),
    ('m * m, its determinant',
     '    Color = vec4((m * m)[2], determinant(m));\n', [0, 0, 9, 6]),
    ('copied, the copy written',
     '    mat3 k = m;\n    k[1] = vec3(9.0);\n'
     '    Color = vec4(m[1].y, k[1].y, k[2].z, m[0].x);\n', [2, 9, 3, 1]),
]


def test_a_matrix_is_one_array_however_it_is_made():
    """`mat3 m = mat3(...); Color = vec4(m[1], 1.0);` compiled and raised
    ValueError: a constructed matrix is one (lanes, columns, rows) array
    and the column read indexed it as a list of columns -- the lane axis.
    With three lanes the shapes fitted and it answered lane 1's rows. A
    declared `mat3 m;` WAS a list of columns, and `m * v` handed it to
    mat_mul_vec: AttributeError."""
    reported = [
        ('a column of a constructed matrix',
         _body(_DIAG + '    Color = vec4(m[1], 1.0);\n'), [0, 2, 0, 1]),
        ('m[1][1]', _body(_DIAG + '    Color = vec4(m[1][1]);\n'),
         [2, 2, 2, 2]),
        ('m[1].xyz', _body(_DIAG + '    Color = vec4(m[1].xyz, 1.0);\n'),
         [0, 2, 0, 1]),
        ('m[0] = vec3(0.5) on a constructed matrix',
         _body(_DIAG + '    m[0] = vec3(0.5);\n'
               '    Color = vec4(m[0], m[1].y);\n'), [0.5, 0.5, 0.5, 2]),
        ('a declared matrix, its columns written, times a vector',
         _body('    mat3 m;\n    m[0] = vec3(1.0, 0.0, 0.0);\n'
               '    m[1] = vec3(0.0, 2.0, 0.0);\n'
               '    m[2] = vec3(0.0, 0.0, 3.0);\n'
               '    Color = vec4(m * vec3(1.0), 1.0);\n'), [1, 2, 3, 1]),
        ('mat3(...) * vec3, the road that worked',
         _body(_DIAG + '    Color = vec4(m * vec3(1.0), 1.0);\n'),
         [1, 2, 3, 1]),
    ]
    for what, src, want in reported:
        # three lanes is the count at which the wrong axis FITTED: the
        # column read answered [1, 0, 0] without a word
        bad = []
        for n in (1, 2, 3, 4, 7):
            got, err = _first(src, n=n)
            if got is None or not np.allclose(got, want, atol=1e-6):
                bad.append(f'{n} lanes: {err if got is None else got}')
        check(f'matrix: {what} (the reported shader, at 1, 2, 3, 4 and 7 '
              'lanes)', not bad, '; '.join(bad))

    # however the matrix came to be, every use of it reads the same matrix
    bad, ran = [], 0
    for made, head, make in _MATRIX_MADE:
        for used, use, want in _MATRIX_USED:
            for n in (3, 5):
                got, err = _first(_body(make + use, head), n=n)
                ran += 1
                if got is None or not np.allclose(got, want, atol=1e-6):
                    bad.append(f'{made} / {used} / {n} lanes: '
                               f'{err if got is None else got}')
    check(f'matrix: made {len(_MATRIX_MADE)} ways by used '
          f'{len(_MATRIX_USED)} ways, at 3 and 5 lanes ({ran} shaders): '
          'every one reads diag(1, 2, 3)', not bad,
          f'{len(bad)} wrong: ' + '; '.join(bad[:4]))

    # the one shape, wherever the value comes from: (lanes, columns, rows)
    from ..shaders.gtypes import BY_NAME
    src = ('out mat3 Made;\nout mat3 Declared;\nout mat4 Wide;\n'
           'out mat2 Small;\nout vec4 Color;\nvoid main() {\n'
           f'    Made = {_D9};\n    Declared[1] = vec3(0.0, 2.0, 0.0);\n'
           '    Wide[3] = vec4(1.0);\n    Small = mat2(2.0);\n'
           '    Color = vec4(1.0);\n}\n')
    prog, err = try_compile(src)
    outs = prog.run({}, {}, 5)[0] if prog else {}
    shapes = {k: getattr(v, 'shape', type(v).__name__)
              for k, v in outs.items()}
    zero = prog.interp.zero(BY_NAME['mat3'], 5) if prog else None
    check('matrix: a declared one and a constructed one are the same '
          '(lanes, columns, rows) float32 array',
          shapes == {'Made': (5, 3, 3), 'Declared': (5, 3, 3),
                     'Wide': (5, 4, 4), 'Small': (5, 2, 2),
                     'Color': (5, 4)}
          and isinstance(zero, np.ndarray) and zero.shape == (5, 3, 3)
          and all(outs[k].dtype == np.float32 for k in shapes)
          and np.array_equal(outs['Declared'][2],
                             [[0, 0, 0], [0, 2, 0], [0, 0, 0]])
          and np.array_equal(outs['Made'][4], np.diag([1.0, 2.0, 3.0])),
          str(err or shapes))

    # per lane: a matrix that differs by lane, written under a branch and
    # read through an index that differs by lane
    src = ('in vec3 vPosition;\nout vec4 Color;\nout vec4 Pick;\n'
           'void main() {\n    float x = vPosition.x;\n'
           '    mat3 m = mat3(x);\n    m[1] = vec3(x, 2.0, 0.25);\n'
           '    if (x > 0.5) { m[1][1] = 5.0; m[2] = vec3(x); }\n'
           '    int i = int(x * 3.0);\n'
           '    Color = vec4(m * vec3(1.0), m[1].y);\n'
           '    Pick = vec4(m[i], m[i][i]);\n}\n')
    p = np.zeros((8, 3), np.float32)
    p[:, 0] = np.arange(8) / 8.0
    outs, _, _ = run(src, inputs={'position': p})
    x = p[:, 0]
    hot = x > 0.5
    ref = np.zeros((8, 3, 3), np.float32)        # [lane, column, row]
    for c in range(3):
        ref[:, c, c] = x
    ref[:, 1] = np.stack([x, np.full(8, 2.0), np.full(8, 0.25)], 1)
    ref[hot, 1, 1] = 5.0
    ref[hot, 2] = x[hot, None]
    i = (x * 3.0).astype(int)
    lanes = np.arange(8)
    check('matrix: per lane -- built from a varying, written under a '
          'branch, multiplied',
          np.allclose(outs['Color'][:, :3], ref.sum(1))
          and np.allclose(outs['Color'][:, 3], ref[:, 1, 1]),
          str(outs['Color']))
    check('matrix: a variable index chooses the column per lane (it read '
          'column 0 for every lane)',
          np.allclose(outs['Pick'][:, :3], ref[lanes, i])
          and np.allclose(outs['Pick'][:, 3], ref[lanes, i, i])
          and len(set(i)) == 3, str(outs['Pick']))

    # a swizzle of a matrix is no driver's GLSL: refused where it is
    # written. The name check skipped the swizzle check for a matrix (and
    # for an array, on the same line), and the run swizzled the lane axis
    _refused('m.xyz on a matrix (it raised ValueError when run)',
             _body(_DIAG + '    Color = vec4(m.xyz, 1.0);\n'),
             "line 4: 'xyz' of mat3: a matrix is indexed first "
             "(m[0].xyz)")
    _refused('m.x, which answered column 0 typed float',
             _body(_DIAG + '    vec3 c = m.x;\n    Color = vec4(c, 1.0);\n'),
             "line 4: 'x' of mat3: a matrix is indexed first (m[0].x)")
    _refused('...in a branch no lane takes',
             _body(_DIAG + '    Color = vec4(1.0);\n'
                   '    if (Color.x > 2.0) {\n        Color = m.rgbr;\n'
                   '    }\n'),
             "line 6: 'rgbr' of mat3: a matrix is indexed first "
             "(m[0].rgbr)")
    _refused('...of a matrix a function returns',
             _body('    Color = vec4(ident().xy, 0.0, 1.0);\n',
                   'mat2 ident() { return mat2(1.0); }\n'),
             "line 4: 'xy' of mat2: a matrix is indexed first")
    _refused('a.x on an array (the same skipped check)',
             _body('    float a[3];\n    a[0] = 1.0;\n'
                   '    Color = vec4(a.x);\n'),
             "line 5: 'x' of float[3]: an array is indexed first "
             "(a[0].x)")
    _refused('a field of an array of structs (TypeError when run)',
             _body('    S a[2];\n    Color = vec4(a.k);\n',
                   'struct S { float k; };\n'),
             "line 5: 'k' of S[2]: an array is indexed first (a[0].k)")
    fine = [
        ('a swizzle of a column, of an element, of an element\'s field',
         _body(_DIAG + '    vec2 a[2];\n    a[1] = vec2(0.25, 0.5);\n'
               '    S s[2];\n    s[1].k = 0.75;\n'
               '    Color = vec4(m[2].z, a[1].yx, s[1].k);\n',
               'struct S { float k; };\n'), [3, 0.5, 0.25, 0.75]),
    ]
    for what, src, want in fine:
        got, err = _first(src)
        check(f'matrix: not refused -- {what}',
              got is not None and np.allclose(got, want, atol=1e-6),
              str(err if got is None else got))
    # where the type is the interpreter's shortcut and not the language's
    # (of two functions with one name and count the first is taken), the
    # member is left to the run: a driver compiles this text
    prog, err = try_compile(
        'out vec4 Color;\nmat3 f(float x) { return mat3(x); }\n'
        'vec3 f(vec3 x) { return x; }\nvoid main() {\n'
        '    Color = vec4(f(vec3(1.0)).xyz, 1.0);\n}\n')
    check('matrix: a member of a loosely typed value is not judged',
          prog is not None, repr(err))


def _lanes(count, limit):
    """`count` lanes of position: x counts 0, 1, 2 .. limit - 1 and round
    again, y is a quarter of the lane number."""
    p = np.zeros((count, 3), np.float32)
    p[:, 0] = np.arange(count) % limit
    p[:, 1] = np.arange(count) * 0.25
    return p


def test_writes_through_a_variable_index():
    """`float a[3]; for (int i = 0; i < 3; i++) { a[i] = float(i) * 0.25;
    }` compiled and raised ShaderError 'dynamic array writes are not
    supported': a write took a number written in the brackets and nothing
    else, where a read through a variable chose its element per lane."""
    reported = [
        ('an array filled in a loop',
         _body('    float a[3];\n    for (int i = 0; i < 3; i++) '
               '{ a[i] = float(i) * 0.25; }\n'
               '    Color = vec4(a[0], a[1], a[2], 1.0);\n'),
         [0, 0.25, 0.5, 1]),
        ('a component of a vector',
         _body('    vec3 v;\n    int i = 1;\n    v[i] = 0.5;\n'
               '    Color = vec4(v, 1.0);\n'), [0, 0.5, 0, 1]),
        ('a column of a matrix',
         _body('    mat3 m;\n    int i = 1;\n    m[i] = vec3(0.5);\n'
               '    Color = vec4(m[0].x, m[1].yz, m[2].z);\n'),
         [0, 0.5, 0.5, 0]),
    ]
    for what, src, want in reported:
        got, err = _first(src)
        check(f'indexed write: {what} (the reported shader)',
              got is not None and np.allclose(got, want, atol=1e-6),
              str(err if got is None else got))

    fine = [
        ('a compound assignment and ++ through a variable index',
         _body('    float a[3];\n    for (int i = 0; i < 3; i++) {\n'
               '        a[i] += float(i);\n        a[i] *= 2.0;\n'
               '        a[i]++;\n        --a[i];\n        a[i] -= 0.5;\n'
               '    }\n    Color = vec4(a[0], a[1], a[2], 1.0);\n'),
         [-0.5, 1.5, 3.5, 1]),
        ('a component of an element, an element of an element',
         _body('    vec2 a[3];\n    ivec2 b[2];\n'
               '    for (int i = 0; i < 3; i++) { a[i].y = float(i); '
               'a[i][0] = float(i) * 0.5; }\n'
               '    for (int i = 0; i < 2; i++) for (int j = 0; j < 2; '
               'j++) b[i][j] = i * 2 + j;\n'
               '    Color = vec4(a[2], a[1].x + a[1].y, '
               'float(b[0][1] + b[1][0] * 4 + b[1][1] * 16));\n'),
         [1, 2, 1.5, 57]),
        ('a component of a matrix, a swizzle of its column',
         _body('    mat2 m = mat2(0.0);\n'
               '    for (int c = 0; c < 2; c++) for (int r = 0; r < 2; '
               'r++) m[c][r] = float(c * 2 + r);\n'
               '    int k = 1;\n    m[k].yx += vec2(0.5, 0.25);\n'
               '    Color = vec4(m[0], m[1]);\n'), [0, 1, 2.25, 3.5]),
        ('an array inside a struct, a struct inside an array',
         _body('    S s;\n    S t[2];\n'
               '    for (int i = 0; i < 3; i++) { s.w[i] = float(i) * '
               '0.25; t[i % 2].w[i] = float(i + 1); }\n'
               '    Color = vec4(s.w[1] + s.w[2], t[0].w[0], t[1].w[1], '
               't[0].w[2]);\n', 'struct S { float w[3]; };\n'),
         [0.75, 1, 2, 3]),
        ('an out parameter given an element: written back to the element '
         'the call named',
         _body('    float a[3];\n    int i = 0;\n'
               '    step2(i, a[i]);\n    step2(i, a[i]);\n'
               '    Color = vec4(a[0], a[1], a[2], float(i));\n',
               'void step2(inout int k, out float v) { k += 1; '
               'v = float(k) * 0.25; }\n'), [0.25, 0.5, 0, 2]),
        ('the value of the assignment is the element written',
         _body('    float a[3];\n    float b[3];\n    int i = 2;\n'
               '    a[i] = b[i - 1] = 0.75;\n'
               '    Color = vec4(a[2], b[1], a[1], b[2]);\n'),
         [0.75, 0.75, 0, 0]),
        ('...and the column written',
         _body('    mat2 m = mat2(1.0);\n    mat2 k = mat2(0.0);\n'
               '    int i = 1;\n    k[i - 1] = m[i] = vec2(0.5, 0.25);\n'
               '    Color = vec4(k[0], m[1]);\n'), [0.5, 0.25, 0.5, 0.25]),
        ('the HLSL dialect',
         'out float4 Color;\nvoid main() {\n    float a[3];\n'
         '    float3 v = float3(0, 0, 0);\n'
         '    for (int i = 0; i < 3; i++) { a[i] = i * 0.5; v[i] = '
         'a[i] + 1; }\n    Color = float4(v, a[1]);\n}\n',
         [1, 1.5, 2, 0.5]),
        ('a column takes what a vector takes: a scalar fills it, an int '
         'vector is converted',
         'out float4 Color;\nvoid main() {\n'
         '    float3x3 m = float3x3(1, 0, 0, 0, 1, 0, 0, 0, 1);\n'
         '    int k = 2;\n    m[k] = 0.5;\n    m[k - 1] = int3(7, 8, 9);\n'
         '    Color = float4(m[2].xz, m[1].yz * 0.5);\n}\n',
         [0.5, 0.5, 4, 4.5]),
    ]
    for what, src, want in fine:
        got, err = _first(src, 'HLSL' if 'float4' in src else 'GLSL')
        check(f'indexed write: {what}',
              got is not None and np.allclose(got, want, atol=1e-6),
              str(err if got is None else got))

    # an index with a side effect moves once: the write names its element
    # again at every step on the way (`a[i].y = x` reads a[i], changes it
    # and writes a[i] back), and each used to be a fresh evaluation
    once = [
        ('a[i++] = x', '    a[i++].x = 0.5;\n    a[i++].x = 0.25;\n',
         [0.5, 0.25, 0, 2]),
        ('a[i++].y = x', '    a[i++].y = 0.5;\n    a[i++].y = 0.25;\n'
         '    a[0].x = a[0].y;\n    a[1].x = a[1].y;\n',
         [0.5, 0.25, 0, 2]),
        ('a[i++][1] = x', '    a[i++][1] = 0.5;\n    a[i++][1] = 0.25;\n'
         '    a[0].x = a[0].y;\n    a[1].x = a[1].y;\n',
         [0.5, 0.25, 0, 2]),
        ('a[i++].x += x', '    a[i++].x += 0.5;\n    a[i++].x += 0.25;\n',
         [0.5, 0.25, 0, 2]),
        ('a[++i].x++', '    a[++i].x++;\n', [0, 1, 0, 1]),
        ('++a[i++].x', '    ++a[i++].x;\n    --a[i++].x;\n',
         [1, -1, 0, 2]),
    ]
    for what, stmts, want in once:
        got, err = _first(_body(
            '    vec2 a[3];\n    int i = 0;\n' + stmts
            + '    Color = vec4(a[0].x, a[1].x, a[2].x, float(i));\n'))
        check(f'indexed write: {what} evaluates its index once',
              got is not None and np.allclose(got, want, atol=1e-6),
              str(err if got is None else got))

    # per lane: every lane writes the element ITS index names, under the
    # mask of the branch it is in
    src = ('in vec3 vPosition;\nout vec4 Color;\nout vec4 Vec;\n'
           'out vec4 Mat;\nvoid main() {\n'
           '    int k = int(vPosition.x);\n    float f = vPosition.y;\n'
           '    float a[4];\n    vec4 v = vec4(0.0);\n'
           '    mat2 m = mat2(0.0);\n'
           '    a[k] = 1.0 + f;\n    v[k] = 1.0 + f;\n'
           '    if (f > 1.0) { a[3 - k] += 8.0; v[3 - k] -= 8.0; }\n'
           '    m[k / 2][k % 2] = 1.0 + f;\n'
           '    if (k == 2) { m[k - 2] = vec2(f, -f); }\n'
           '    Color = vec4(a[0], a[1], a[2], a[3]);\n    Vec = v;\n'
           '    Mat = vec4(m[0], m[1]);\n}\n')
    p = _lanes(12, 4)
    outs, _, _ = run(src, inputs={'position': p}, n=12)
    k = p[:, 0].astype(int)
    f = p[:, 1]
    lanes = np.arange(12)
    ra = np.zeros((12, 4), np.float32)
    ra[lanes, k] = 1.0 + f
    rv = ra.copy()
    hot = f > 1.0
    ra[lanes[hot], 3 - k[hot]] += 8.0
    rv[lanes[hot], 3 - k[hot]] -= 8.0
    rm = np.zeros((12, 2, 2), np.float32)
    rm[lanes, k // 2, k % 2] = 1.0 + f
    two = k == 2
    rm[two, 0] = np.stack([f[two], -f[two]], 1)
    check('indexed write: per lane, into an array, a vector and a matrix, '
          'under a branch',
          np.array_equal(outs['Color'], ra) and np.array_equal(outs['Vec'], rv)
          and np.array_equal(outs['Mat'], rm.reshape(12, 4))
          and hot.any() and not hot.all(),
          str(outs['Color'][:5]))

    # past the end, a variable index is clamped as the read's is: what is
    # written through an index is what a read through it finds
    src = ('in vec3 vPosition;\nout vec4 Color;\nvoid main() {\n'
           '    int k = int(vPosition.x) - 2;\n    float a[3];\n'
           '    vec3 v = vec3(0.0);\n    a[k] = 1.0;\n    v[k] = 2.0;\n'
           '    Color = vec4(a[k], v[k], a[0] + a[1] * 2.0 + a[2] * 4.0, '
           'v.x + v.y * 2.0 + v.z * 4.0);\n}\n')
    p = _lanes(7, 7)
    outs, _, _ = run(src, inputs={'position': p}, n=7)
    where = np.clip(p[:, 0].astype(int) - 2, 0, 2)
    check('indexed write: an index past either end writes the element a '
          'read through it reads (-2 .. 4 into three elements)',
          np.array_equal(outs['Color'][:, 0], np.ones(7))
          and np.array_equal(outs['Color'][:, 1], np.full(7, 2.0))
          and np.array_equal(outs['Color'][:, 2], 2.0 ** where)
          and np.array_equal(outs['Color'][:, 3], 2.0 * 2.0 ** where),
          str(outs['Color']))

    # what a write still refuses, as it did -- by name now
    for what, src, want in (
            ('a number written past the size (it raised IndexError)',
             _body('    float a[3];\n    a[3] = 1.0;\n'
                   '    Color = vec4(a[2]);\n'),
             'index 3 is out of range for float[3]'),
            ('...past a vector\'s width',
             _body('    vec3 v = vec3(0.0);\n    v[3] = 1.0;\n'
                   '    Color = vec4(v, 1.0);\n'),
             'index 3 is out of range for vec3'),
            ('...past a matrix\'s columns',
             _body('    mat2 m = mat2(1.0);\n    m[2] = vec2(1.0);\n'
                   '    Color = vec4(m[1], 0.0, 1.0);\n'),
             'index 2 is out of range for mat2'),
            ('a variable index into a float: nothing to choose between',
             _body('    float f = 0.0;\n    int i = 1;\n    f[i] = 1.0;\n'
                   '    Color = vec4(f);\n'),
             'float is not an array, a vector or a matrix: a variable '
             'index cannot write to it')):
        got, err = _first(src)
        check(f'indexed write, still refused: {what}',
              got is None and err == f'compiled, then ShaderError: {want}',
              str(err if got is None else got))

    # the rule, held against a second implementation: seeded programs of
    # indexed reads and writes (constant and per-lane indices, in and out
    # of range, nested branches and loops), each lane computed here in
    # plain Python on plain lists
    bad, n_prog, n_writes = [], 120, 0
    for seed in range(n_prog):
        text, tree = _index_program(np.random.RandomState(1000 + seed))
        n_writes += text.count(' = ') + text.count('+= ')
        p = _lanes(10, 5)
        prog, err = try_compile(text)
        if prog is None:
            bad.append(f'seed {seed}: {err}')
            continue
        try:
            outs = prog.run({}, {'position': p}, 10)[0]
        except Exception as exc:                                # noqa: BLE001
            bad.append(f'seed {seed}: {type(exc).__name__}: {exc}')
            continue
        got = np.concatenate([outs['A'], outs['V'], outs['M']], 1)
        want = np.array([_index_reference(tree, int(k), float(f))
                         for k, f in p[:, :2]], np.float32)
        if not np.array_equal(got, want):
            lane = int(np.argwhere((got != want).any(1))[0, 0])
            bad.append(f'seed {seed} lane {lane}: {got[lane]} for '
                       f'{want[lane]}')
    check(f'indexed write: {n_prog} seeded programs ({n_writes} writes), '
          'every lane equal to a plain-Python run of the same program',
          not bad, f'{len(bad)} differ: ' + '; '.join(bad[:3]))


# -- the seeded programs of the test above: a small tree, written out as
# GLSL and run by `_index_reference` one lane at a time. Values are small
# multiples of 1/4, so every sum is exact in float32 and the two can be
# held to equality.

_IDX_SIZES = {'a': 4, 'v': 3, 'm': 2}


def _index_expr(rng, loops):
    """(GLSL text, f(env) -> int): an index, sometimes out of range."""
    kind = rng.randint(6)
    if kind == 0 and loops:
        name = loops[rng.randint(len(loops))]
        return name, lambda e: e[name]
    if kind == 1 and loops:
        name = loops[rng.randint(len(loops))]
        return f'({name} + k)', lambda e: e[name] + e['k']
    if kind == 2:
        c = int(rng.randint(4))
        return f'(k - {c})', lambda e: e['k'] - c
    if kind == 3:
        c = int(rng.randint(3))
        return f'(k * 2 - {c})', lambda e: e['k'] * 2 - c
    if kind == 4:
        return 'k', lambda e: e['k']
    c = int(rng.randint(2))            # a written number: in range for all
    return str(c), lambda e: c


def _index_value(rng, loops, depth=0):
    """(GLSL text, f(env) -> float): a value, sometimes an indexed read."""
    kind = rng.randint(7 if depth < 2 else 4)
    if kind == 0:
        c = int(rng.randint(1, 9))
        return f'{c * 0.25}', lambda e: c * 0.25
    if kind == 1:
        return 'f', lambda e: e['f']
    if kind == 2:
        return '(float(k) * 0.5)', lambda e: e['k'] * 0.5
    if kind == 3 and loops:
        name = loops[rng.randint(len(loops))]
        return f'float({name})', lambda e: float(e[name])
    if kind == 4:
        t, i = _index_expr(rng, loops)
        return f'a[{t}]', lambda e: e['a'][_clamp(i(e), 4)]
    if kind == 5:
        t, i = _index_expr(rng, loops)
        return f'v[{t}]', lambda e: e['v'][_clamp(i(e), 3)]
    if kind == 6:
        (t1, i1), (t2, i2) = _index_expr(rng, loops), _index_expr(rng, loops)
        return (f'm[{t1}][{t2}]',
                lambda e: e['m'][_clamp(i1(e), 2)][_clamp(i2(e), 2)])
    return '0.25', lambda e: 0.25


def _clamp(i, count):
    return max(0, min(int(i), count - 1))


def _index_block(rng, loops, depth, pad):
    """([GLSL lines], [statement closures]) of one block."""
    lines, stmts = [], []
    for _ in range(rng.randint(2, 5)):
        kind = rng.randint(9)
        if kind < 5:
            name = 'avm'[rng.randint(3)]
            (ti, fi), (tv, fv) = (_index_expr(rng, loops),
                                  _index_value(rng, loops))
            add = rng.randint(3) == 0
            op = '+=' if add else '='
            if name == 'm':
                tj, fj = _index_expr(rng, loops)
                lines.append(f'{pad}m[{ti}][{tj}] {op} {tv};')

                def put(e, fi=fi, fj=fj, fv=fv, add=add):
                    c, r, x = _clamp(fi(e), 2), _clamp(fj(e), 2), fv(e)
                    e['m'][c][r] = e['m'][c][r] + x if add else x
            else:
                size = _IDX_SIZES[name]
                lines.append(f'{pad}{name}[{ti}] {op} {tv};')

                def put(e, name=name, size=size, fi=fi, fv=fv, add=add):
                    j, x = _clamp(fi(e), size), fv(e)
                    e[name][j] = e[name][j] + x if add else x
            stmts.append(put)
        elif kind == 5:
            ti, fi = _index_expr(rng, loops)
            tv, fv = _index_value(rng, loops)
            lines.append(f'{pad}m[{ti}] = vec2({tv}, 0.75);')

            def put_col(e, fi=fi, fv=fv):
                e['m'][_clamp(fi(e), 2)] = [fv(e), 0.75]
            stmts.append(put_col)
        elif kind in (6, 7) and depth < 2:
            c = int(rng.randint(1, 4))
            use_f = rng.randint(2) == 0
            cond = f'f > {c * 0.5}' if use_f else f'k >= {c}'
            a_lines, a_stmts = _index_block(rng, loops, depth + 1,
                                            pad + '    ')
            b_lines, b_stmts = _index_block(rng, loops, depth + 1,
                                            pad + '    ')
            lines += ([f'{pad}if ({cond}) {{'] + a_lines
                      + [f'{pad}}} else {{'] + b_lines + [f'{pad}}}'])

            def branch(e, c=c, use_f=use_f, a=a_stmts, b=b_stmts):
                taken = e['f'] > c * 0.5 if use_f else e['k'] >= c
                for st in (a if taken else b):
                    st(e)
            stmts.append(branch)
        elif kind == 8 and depth < 2:
            name = 'ij'[len(loops)] if len(loops) < 2 else None
            if name is None:
                continue
            trips = int(rng.randint(1, 4))
            b_lines, b_stmts = _index_block(rng, loops + [name], depth + 1,
                                            pad + '    ')
            lines += ([f'{pad}for (int {name} = 0; {name} < {trips}; '
                       f'{name}++) {{'] + b_lines + [f'{pad}}}'])

            def loop(e, name=name, trips=trips, body=b_stmts):
                for t in range(trips):
                    e[name] = t
                    for st in body:
                        st(e)
            stmts.append(loop)
    return lines, stmts


def _index_program(rng):
    lines, stmts = _index_block(rng, [], 0, '    ')
    text = ('in vec3 vPosition;\nout vec4 A;\nout vec4 V;\nout vec4 M;\n'
            'void main() {\n    int k = int(vPosition.x);\n'
            '    float f = vPosition.y;\n'
            '    float a[4] = float[4](0.0, 0.0, 0.0, 0.0);\n'
            '    vec3 v = vec3(0.0);\n    mat2 m = mat2(0.0);\n'
            + '\n'.join(lines) + '\n'
            '    A = vec4(a[0], a[1], a[2], a[3]);\n    V = vec4(v, 1.0);\n'
            '    M = vec4(m[0], m[1]);\n}\n')
    return text, stmts


def _index_reference(stmts, k, f):
    env = {'k': k, 'f': f, 'a': [0.0] * 4, 'v': [0.0] * 3,
           'm': [[0.0, 0.0], [0.0, 0.0]]}
    for st in stmts:
        st(env)
    return env['a'] + env['v'] + [1.0] + env['m'][0] + env['m'][1]


def test_a_struct_field_keeps_its_array_size():
    """`struct S2 { float w[3]; }; S2 s; s.w[1] = 0.5; Color =
    vec4(s.w[1]);` compiled and raised IndexError: the field's array size
    -- the third element of the parser's field tuple -- was read by
    nothing, so the struct was built with ONE float for `w` and the name
    check typed it float."""
    got, err = _first('struct S2 { float w[3]; };\nout vec4 Color;\n'
                      'void main() {\n    S2 s;\n    s.w[1] = 0.5;\n'
                      '    Color = vec4(s.w[1]);\n}\n')
    check('struct array field: s.w[1] = 0.5 (the reported shader)',
          got is not None and np.allclose(got, 0.5),
          str(err if got is None else got))

    _S = 'struct S { float w[3]; vec3 p[2]; float k; };\n'
    fine = [
        ('every element is its own (w[1] read back what w[0] was given)',
         _body('    S s;\n    s.w[0] = 0.25;\n    s.w[1] = 0.5;\n'
               '    s.w[2] = 0.75;\n    s.k = 2.0;\n'
               '    Color = vec4(s.w[0], s.w[1], s.w[2], s.k);\n', _S),
         [0.25, 0.5, 0.75, 2]),
        ('an array of vectors: an element, its swizzle, its component '
         '(s.p[0].y was refused as a swizzle of a float)',
         _body('    S s;\n    s.p[1] = vec3(0.5);\n    s.p[0].y = 0.25;\n'
               '    s.p[1][2] = 0.75;\n'
               '    Color = vec4(s.p[1].xz, s.p[0].yx);\n', _S),
         [0.5, 0.75, 0.25, 0]),
        ('the constructor takes the array as one argument',
         _body('    S s = S(float[3](0.25, 0.5, 0.75), '
               'vec3[2](vec3(1.0), vec3(2.0)), 4.0);\n'
               '    Color = vec4(s.w[2], s.p[0].x, s.p[1].z, s.k);\n', _S),
         [0.75, 1, 2, 4]),
        ('copied whole, passed to a function, returned from one',
         _body('    S s = make(0.25);\n    S t = s;\n    t.w[1] = 9.0;\n'
               '    Color = vec4(s.w[1], t.w[1], total(s), t.p[1].y);\n',
               _S + 'S make(float x) {\n    S r;\n'
               '    for (int i = 0; i < 3; i++) { r.w[i] = x * float(i + '
               '1); }\n    r.p[1] = vec3(x);\n    return r;\n}\n'
               'float total(S s) { return s.w[0] + s.w[1] + s.w[2]; }\n'),
         [0.5, 9, 1.5, 0.25]),
        ('a struct inside a struct inside an array',
         _body('    Out o[2];\n    o[1].i.w[2] = 0.5;\n'
               '    o[0].i.w[2] = 0.25;\n    o[1].c[1] = vec3(0.75);\n'
               '    Color = vec4(o[1].i.w[2], o[0].i.w[2], o[1].c[1].z, '
               'o[1].i.w[1]);\n',
               'struct In { float w[3]; };\n'
               'struct Out { In i; vec3 c[2]; };\n'),
         [0.5, 0.25, 0.75, 0]),
        ('the field assigned whole: the value of the assignment is the '
         'array',
         _body('    S s;\n    S t;\n'
               '    float x = (s.w = float[3](0.25, 0.5, 0.75))[1];\n'
               '    t.w = s.w;\n    t.w[0] = 2.0;\n'
               '    Color = vec4(x, s.w[0], t.w[0], t.w[2]);\n', _S),
         [0.5, 0.25, 2, 0.75]),
        ('a field that is no array is as it was',
         _STRUCT + 'void main() {\n    S s = S(0.5, vec3(0.1, 0.2, '
         '0.3));\n    s.b.y = 1.0;\n    Color = vec4(s.a, s.b);\n}\n',
         [0.5, 0.1, 1.0, 0.3]),
        ('the HLSL dialect',
         'struct S { float w[3]; };\nout float4 Color;\nvoid main() {\n'
         '    S s;\n    s.w[2] = 0.5;\n    s.w[0] = 0.25;\n'
         '    Color = float4(s.w[0], s.w[1], s.w[2], 1);\n}\n',
         [0.25, 0, 0.5, 1]),
    ]
    for what, src, want in fine:
        got, err = _first(src, 'HLSL' if 'float4' in src else 'GLSL')
        check(f'struct array field: {what}',
              got is not None and np.allclose(got, want, atol=1e-6),
              str(err if got is None else got))

    # the two halves agree on what the field is: the value the struct is
    # built with and the type the name check gives it
    from ..shaders.interp import field_type
    from ..shaders.names import NameCheck, _Scope
    prog, err = try_compile(_body('    S s;\n    Color = vec4(s.k);\n', _S))
    zero = prog.interp.zero(prog.interp.fields('S')['w'], 4)
    built = prog.interp.zero(_struct_type('S'), 4)
    scope = _Scope()
    scope.names['s'] = _struct_type('S')
    typed = {f: NameCheck(prog.interp).expr(('member', ('var', 's', 0), f, 0),
                                            scope)[0].name()
             for f in ('w', 'p', 'k')}
    check('struct array field: built as three floats, two vec3 and a float; '
          'typed float[3], vec3[2], float',
          [np.shape(e) for e in built['w']] == [(4,)] * 3
          and [np.shape(e) for e in built['p']] == [(4, 3)] * 2
          and np.shape(built['k']) == (4,) and len(zero) == 3
          and typed == {'w': 'float[3]', 'p': 'vec3[2]', 'k': 'float'}
          and [field_type(f).name()
               for f in prog.interp.structs['S']] == ['float[3]', 'vec3[2]',
                                                      'float'],
          str((typed, {k: np.shape(v) for k, v in built.items()})))

    # per lane, under a branch
    src = (_S + 'in vec3 vPosition;\nout vec4 Color;\nvoid main() {\n'
           '    int k = int(vPosition.x);\n    S s;\n'
           '    s.w[k] = 1.0 + vPosition.y;\n'
           '    if (vPosition.y > 1.0) { s.w[2 - k] += 4.0; '
           's.p[k % 2].z = 0.5; }\n'
           '    Color = vec4(s.w[0], s.w[1], s.w[2], s.p[0].z + 2.0 * '
           's.p[1].z);\n}\n')
    p = _lanes(9, 3)
    outs, _, _ = run(src, inputs={'position': p}, n=9)
    k = p[:, 0].astype(int)
    lanes = np.arange(9)
    ref = np.zeros((9, 4), np.float32)
    ref[lanes, k] = 1.0 + p[:, 1]
    hot = p[:, 1] > 1.0
    ref[lanes[hot], 2 - k[hot]] += 4.0
    ref[hot, 3] = np.where(k[hot] % 2 == 0, 0.5, 1.0)
    check('struct array field: per lane, under a branch',
          np.array_equal(outs['Color'], ref) and hot.any() and not hot.all(),
          str(outs['Color']))

    # an array field is an array to the name check too
    _refused('a swizzle of an array field (it is indexed first)',
             _body('    S s;\n    Color = vec4(s.p.xyz, 1.0);\n', _S),
             "line 5: 'xyz' of vec3[2]: an array is indexed first")


def _struct_type(name):
    from ..shaders.interp import struct_type
    return struct_type(name)


def test_signed_integers_divide_to_an_integer():
    """`int a = 7; int b = 2; int c = a / b;` gave 3.5 and `-a / b` -3.5,
    with no error: the `/` of two ints kept NumPy's float quotient, typed
    int. A driver -- which compiles the same text when the shader is
    inlined in the deferred pass -- cuts the quotient toward zero: 3, -3.
    `uint` division was already right."""
    reported = [
        ('7 / 2 and -7 / 2',
         _body('    int a = 7;\n    int b = 2;\n    int c = a / b;\n'
               '    int d = -a / b;\n'
               '    Color = vec4(float(c), float(d), float(a / -b), '
               'float(-a / -b));\n'), [3, -3, -3, 3]),
        ('ivec2(7, 9) / 2',
         _body('    ivec2 e = ivec2(7, 9) / 2;\n'
               '    ivec2 g = 19 / ivec2(7, -9);\n'
               '    Color = vec4(e, g);\n'), [3, 4, 2, -2]),
        ('7 / 2 * 2 (it gave 7)',
         _body('    Color = vec4(float(7 / 2 * 2), float(7 - 7 / 2 * 2), '
               '1 / 2, float(1 / 2 + 1 / 2));\n'), [6, 1, 0, 0]),
    ]
    for what, src, want in reported:
        got, err = _first(src)
        check(f'int division: {what} (the reported shader)',
              got is not None and np.array_equal(got, want),
              str(err if got is None else got))

    fine = [
        ('a compound assignment, a vector divided by a vector',
         _body('    int a = 17;\n    a /= 5;\n'
               '    ivec3 v = ivec3(17, -17, 6) / ivec3(5, 5, -4);\n'
               '    Color = vec4(float(a), v);\n'), [3, 3, -3, -1]),
        ('a float on either side is a float quotient, as it was',
         _body('    int a = 7;\n    float h = 2.0;\n'
               '    Color = vec4(a / h, h / a, float(a) / 2.0, '
               'float(a / 2) / 2.0);\n'), [3.5, 2.0 / 7.0, 3.5, 1.5]),
        ('uint division is as it was',
         _body('    uint a = 7u;\n    uint big = 0xffffffffu;\n'
               '    Color = vec4(float(a / 2u), float(big / 0x10000u), '
               'float(a / 7u), float((a - 8u) / 0x80000000u));\n'),
         [3, 65535, 1, 1]),
        ('a quotient indexes, counts a loop and addresses a texel',
         _body('    float a[4] = float[4](0.25, 0.5, 0.75, 1.0);\n'
               '    int n = 7;\n    float acc = 0.0;\n'
               '    for (int i = 0; i < n / 2; i++) { acc += 1.0; }\n'
               '    int t = 29;\n    ivec2 at = ivec2(t % 8, t / 8);\n'
               '    Color = vec4(a[n / 2], acc, at);\n'),
         [1.0, 3, 5, 3]),
        ('the HLSL dialect',
         'out float4 Color;\nvoid main() {\n    int a = 7;\n'
         '    int2 v = int2(7, -9) / 2;\n'
         '    Color = float4(a / 2, -a / 2, v);\n}\n', [3, -3, 3, -4]),
    ]
    for what, src, want in fine:
        got, err = _first(src, 'HLSL' if 'float4' in src else 'GLSL')
        check(f'int division: {what}',
              got is not None and np.allclose(got, want, atol=1e-6),
              str(err if got is None else got))

    # the rule itself, on every pair of -12 .. 12: the quotient is cut
    # toward zero, lane by lane, and what it is stored as is an int
    src = ('in vec3 vPosition;\nout int Q;\nout ivec2 QV;\nout vec4 Color;\n'
           'void main() {\n    int a = int(vPosition.x);\n'
           '    int b = int(vPosition.y);\n    Q = a / b;\n'
           '    QV = ivec2(a, -a) / ivec2(b, 3);\n    int c = a;\n'
           '    c /= b;\n'
           '    Color = vec4(float(c), float(a / b * b + (a - a / b * b)), '
           'float(a) / float(b), 1.0);\n}\n')
    vals = [(a, b) for a in range(-12, 13) for b in range(-12, 13)]
    p = np.zeros((len(vals), 3), np.float32)
    p[:, :2] = vals
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('error')      # a zero divisor stays silent
        outs, _, _ = run(src, inputs={'position': p}, n=len(vals))
    a, b = np.array(vals).T
    live = b != 0
    cut = np.array([int(x / y) for x, y in np.array(vals)[live]])
    check(f'int division: all {int(live.sum())} pairs of -12 .. 12 with a '
          'divisor are cut toward zero (and the lanes that divide by zero '
          'raise no warning)',
          np.array_equal(outs['Q'][live], cut)
          and np.array_equal(outs['QV'][live, 0], cut)
          and np.array_equal(outs['QV'][:, 1],
                             [int(-x / 3) for x in a])
          and np.array_equal(outs['Color'][live, 0], cut)
          and np.array_equal(outs['Color'][live, 1], a[live])
          and np.allclose(outs['Color'][live, 2], a[live] / b[live])
          and (cut != np.floor(a[live] / b[live])).any()
          and (cut != np.round(a[live] / b[live])).any(),
          str(outs['Q'][:12]))

    # the cut is all that changes. A quotient that had no fraction is the
    # number it was, held as it was: a float vector built from it computes
    # at float32 as it did (held as an int32 the same chain would run in
    # float64 -- the 0.1 * 0.7 below rounds the other way there)
    got, err = _first(_body(
        '    int n = 6;\n    vec2 g = vec2(n / 2, n / 2) * 0.1 * 0.7;\n'
        '    Color = vec4(g, float(n / 2), float(n / 3 * 3));\n'))
    chain = np.float32(np.float32(3.0) * np.float32(0.1)) * np.float32(0.7)
    wide = np.float32(3.0 * float(np.float32(0.1)) * float(np.float32(0.7)))
    check('int division: an exact quotient moves nothing computed from it '
          '(vec2(n / 2, n / 2) * 0.1 * 0.7, bit for bit)',
          got is not None and chain != wide
          and np.array_equal(got, np.array([chain, chain, 3.0, 6.0],
                                           np.float32)),
          str(err if got is None else [float(v) for v in got]))

    # the ends of the range, compared as ints inside the shader (a float
    # output cannot hold them)
    got, err = _first(_body(
        '    int big = 2147483647;\n    int low = -2147483647 - 1;\n'
        '    Color = vec4(big / 2 == 1073741823, low / 2 == -1073741824, '
        'big / -big == -1 && low / big == -1, big / 3 * 3 == big - 1);\n'))
    check('int division: exact at the ends of the 32-bit range',
          got is not None and np.array_equal(got, [1, 1, 1, 1]),
          str(err if got is None else got))


# --------------------------- a matrix with a scalar, two matrices compared
#
# R253 (review): valid GLSL on a matrix that try_compile accepted (error
# None) and the run raised on -- or, at the lane count that happens to equal
# the matrix's rows, answered along the wrong axis without a word. Each
# shader named "the reported shader" is the text the defect was reproduced
# with.

#: (what, statements reading m into Color, the Color of diag(1, 2, 3)).
#: `k * vec3(1.0)` is the sum of k's columns, so every component is in it
_MATRIX_SCALED = [
    ('m * 2.0',
     '    mat3 k = m * 2.0;\n    Color = vec4(k * vec3(1.0), k[1][1]);\n',
     [2, 4, 6, 4]),
    ('2.0 * m',
     '    mat3 k = 2.0 * m;\n    Color = vec4(k * vec3(1.0), k[1][1]);\n',
     [2, 4, 6, 4]),
    ('m * 2, an int',
     '    mat3 k = m * 2;\n    Color = vec4(k * vec3(1.0), k[1][1]);\n',
     [2, 4, 6, 4]),
    ('m / 2.0',
     '    mat3 k = m / 2.0;\n    Color = vec4(k * vec3(1.0), k[1][1]);\n',
     [0.5, 1, 1.5, 1]),
    ('m / 4, an int',
     '    mat3 k = m / 4;\n    Color = vec4(k * vec3(1.0), k[1][1]);\n',
     [0.25, 0.5, 0.75, 0.5]),
    ('6.0 / (m + 1.0), the scalar on the left of a division',
     '    mat3 k = 6.0 / (m + 1.0);\n'
     '    Color = vec4(k[0][0], k[1][1], k[2][2], k[0][1]);\n',
     [3, 2, 1.5, 6]),
    ('m + 1.0',
     '    mat3 k = m + 1.0;\n    Color = vec4(k * vec3(1.0), k[1][1]);\n',
     [4, 5, 6, 3]),
    ('1.0 + m',
     '    mat3 k = 1.0 + m;\n    Color = vec4(k * vec3(1.0), k[1][1]);\n',
     [4, 5, 6, 3]),
    ('m - 1.0',
     '    mat3 k = m - 1.0;\n    Color = vec4(k * vec3(1.0), k[1][1]);\n',
     [-2, -1, 0, 1]),
    ('1.0 - m',
     '    mat3 k = 1.0 - m;\n    Color = vec4(k * vec3(1.0), k[1][1]);\n',
     [2, 1, 0, -1]),
    ('m *= 2.0',
     '    m *= 2.0;\n    Color = vec4(m * vec3(1.0), m[1][1]);\n',
     [2, 4, 6, 4]),
    ('m /= 2.0',
     '    m /= 2.0;\n    Color = vec4(m * vec3(1.0), m[1][1]);\n',
     [0.5, 1, 1.5, 1]),
    ('m += 1.0',
     '    m += 1.0;\n    Color = vec4(m * vec3(1.0), m[1][1]);\n',
     [4, 5, 6, 3]),
    ('m -= 1.0',
     '    m -= 1.0;\n    Color = vec4(m * vec3(1.0), m[1][1]);\n',
     [-2, -1, 0, 1]),
    ('m++, whose value is the matrix as it was',
     '    mat3 k = m++;\n    Color = vec4(m * vec3(1.0), k[1][1]);\n',
     [4, 5, 6, 2]),
    ('++m, whose value is the matrix as it is',
     '    mat3 k = ++m;\n    Color = vec4(m * vec3(1.0), k[1][1]);\n',
     [4, 5, 6, 3]),
    ('m--, then --m',
     '    m--;\n    mat3 k = --m;\n'
     '    Color = vec4(m * vec3(1.0), k[1][1]);\n', [-5, -4, -3, 0]),
    ('a chain: (m * 2.0 + 1.0) / 2.0 - 0.5 is m again',
     '    mat3 k = (m * 2.0 + 1.0) / 2.0 - 0.5;\n'
     '    Color = vec4(k * vec3(1.0), k[0][1]);\n', [1, 2, 3, 0]),
    ('the negative, scaled',
     '    mat3 k = -m * 2.0;\n    Color = vec4(k * vec3(1.0), k[1][1]);\n',
     [-2, -4, -6, -4]),
    ('a column, a component and a swizzle of a product',
     '    Color = vec4((m * 2.0)[1].xy, (2.0 * m)[2][2], '
     '(m / 2.0)[0].x);\n', [0, 4, 6, 0.5]),
    ('scaled, then times a vector, times a matrix, a vector times it',
     '    Color = vec4(((m * 2.0) * vec3(1.0)).xy, ((m * 2.0) * m)[2].z, '
     '(vec3(1.0) * (0.5 * m)).y);\n', [2, 4, 18, 1]),
    ('by a scalar a builtin answers',
     '    mat3 k = m * dot(vec3(1.0), vec3(1.0, 0.0, 1.0)) '
     '/ length(vec3(0.0, 2.0, 0.0));\n'
     '    Color = vec4(k * vec3(1.0), k[2][2]);\n', [1, 2, 3, 3]),
    ('m == m, m != m, and against what was computed from it',
     '    Color = vec4(m == m ? 1.0 : 0.0, m != m ? 1.0 : 0.0, '
     'm == m * 1.0 ? 1.0 : 0.0, m * 2.0 == m + m ? 1.0 : 0.0);\n',
     [1, 0, 1, 1]),
    ('equal to a copy until one component of the copy is written',
     '    mat3 k = m;\n    bool before = m == k;\n    k[2][1] = 0.5;\n'
     '    Color = vec4(before ? 1.0 : 0.0, m == k ? 1.0 : 0.0, '
     'm != k ? 1.0 : 0.0, k == m ? 1.0 : 0.0);\n', [1, 0, 1, 0]),
    ('compared in an if',
     '    mat3 k = m;\n    k[1][2] = 0.5;\n    Color = vec4(0.0);\n'
     '    if (m == k) { Color.x = 1.0; }\n'
     '    if (m != k) { Color.y = 1.0; }\n'
     '    if (m == m && !(m != m)) { Color.z = 1.0; }\n', [0, 1, 1, 0]),
]


def _lane_matrix(size, x, y):
    """A matrix that differs by lane and is symmetric nowhere, twice: the
    GLSL constructor, and the same numbers as [lane, column, row] -- filled
    one number at a time, so the reference owes nothing to broadcasting."""
    parts, ref = [], np.zeros((len(x), size, size), np.float32)
    for c in range(size):
        for r in range(size):
            k = c * size + r + 1
            parts.append(f'{k}.0 + ' + ('x' if (c + r) % 2 == 0 else 'y'))
            for lane in range(len(x)):
                ref[lane, c, r] = np.float32(k) + (
                    x[lane] if (c + r) % 2 == 0 else y[lane])
    return f'mat{size}(' + ', '.join(parts) + ')', ref


def _per_lane(fn, *arrays):
    """`fn` of each lane's own values, one lane at a time."""
    return np.stack([fn(*[a[i] for a in arrays])
                     for i in range(len(arrays[0]))])


def test_a_matrix_and_a_scalar_combine_component_by_component():
    """`mat3 k = m * 2.0;` compiled and raised ValueError from einsum: every
    `*` with a matrix on one side went to the linear product, the scalar
    handed over as if it were a vector (and the result typed a vector).
    `m / 2.0`, `m + 1.0` and `m - 1.0` raised 'operands could not be
    broadcast together': the scalar is (lanes,) and the matrix (lanes,
    columns, rows), which NumPy lines up with the rows -- so at three lanes
    of a mat3 they RAN, lane r's scalar on row r of every lane. `m == m`
    compared the first column of each side, a bool per lane and row."""
    reported = [
        ('mat3 k = m * 2.0',
         _body(_DIAG + '    mat3 k = m * 2.0;\n'
               '    Color = vec4(k * vec3(1.0), 1.0);\n'), [2, 4, 6, 1]),
        ('mat3 k = 2.0 * m',
         _body(_DIAG + '    mat3 k = 2.0 * m;\n'
               '    Color = vec4(k * vec3(1.0), 1.0);\n'), [2, 4, 6, 1]),
        ('m *= 2.0',
         _body(_DIAG + '    m *= 2.0;\n'
               '    Color = vec4(m * vec3(1.0), 1.0);\n'), [2, 4, 6, 1]),
        ('mat3 k = m / 2.0',
         _body(_DIAG + '    mat3 k = m / 2.0;\n'
               '    Color = vec4(k * vec3(1.0), 1.0);\n'), [0.5, 1, 1.5, 1]),
        ('m == m ? 1.0 : 0.0',
         _body(_DIAG + '    Color = vec4(m == m ? 1.0 : 0.0);\n'),
         [1, 1, 1, 1]),
    ]
    for what, src, want in reported:
        bad = []
        for n in (1, 3, 4, 7):
            got, err = _first(src, n=n)
            if got is None or not np.array_equal(got, want):
                bad.append(f'{n} lanes: {err if got is None else got}')
        check(f'matrix and scalar: {what} (the reported shader, at 1, 3, 4 '
              'and 7 lanes)', not bad, '; '.join(bad))

    # every form, on a matrix made every way a matrix is made
    bad, ran = [], 0
    for made, head, make in _MATRIX_MADE:
        for used, use, want in _MATRIX_SCALED:
            for n in (1, 3, 4, 7):
                got, err = _first(_body(make + use, head), n=n)
                ran += 1
                if got is None or not np.allclose(got, want, atol=1e-6):
                    bad.append(f'{made} / {used} / {n} lanes: '
                               f'{err if got is None else got}')
    check(f'matrix and scalar: {len(_MATRIX_SCALED)} forms on a matrix made '
          f'{len(_MATRIX_MADE)} ways, at 1, 3, 4 and 7 lanes ({ran} '
          'shaders)', not bad, f'{len(bad)} wrong: ' + '; '.join(bad[:4]))

    # per lane, and this is the check the lane counts are chosen for. The
    # scalar and the matrix both differ by lane, and the matrix is
    # symmetric nowhere, so an operator applied along another axis is
    # another number. The count at which the wrong axis FITS is the
    # matrix's own row count -- 2 lanes of a mat2, 3 of a mat3, 4 of a mat4
    # -- and there the old `/`, `+` and `-` answered without a word. Held
    # bit for bit: the same float32 operation, one lane at a time.
    f32 = np.float32
    for size in (2, 3, 4):
        mt = f'mat{size}'
        outs_decl = ''.join(f'out {mt} {name};\n' for name in (
            'Mul', 'Lmul', 'Div', 'Rdiv', 'Add', 'Radd', 'Sub', 'Rsub',
            'Comp', 'Step', 'Branch'))
        bad, counts = [], (1, 2, 3, 4, 7)
        for n in counts:
            x = (np.arange(n) + 1.0).astype(f32)
            y = (np.arange(n) * 0.5 + 0.75).astype(f32)
            make, m = _lane_matrix(size, x, y)
            src = ('in vec3 vPosition;\n' + outs_decl + 'out vec4 Color;\n'
                   'void main() {\n    float x = vPosition.x;\n'
                   '    float y = vPosition.y;\n'
                   f'    {mt} m = {make};\n'
                   '    Mul = m * x;\n    Lmul = y * m;\n    Div = m / y;\n'
                   '    Rdiv = x / m;\n    Add = m + x;\n    Radd = y + m;\n'
                   '    Sub = m - y;\n    Rsub = x - m;\n'
                   f'    {mt} c = m;\n    c *= x;\n    c += y;\n'
                   '    c /= 4.0;\n    c -= x;\n    Comp = c;\n'
                   f'    {mt} s = m;\n    s++;\n    ++s;\n    s--;\n'
                   '    Step = s;\n'
                   f'    {mt} b = m;\n'
                   '    if (x > 2.5) { b *= y; } else { b = b - x; }\n'
                   '    Branch = b;\n    Color = vec4(1.0);\n}\n')
            p = np.zeros((n, 3), f32)
            p[:, 0], p[:, 1] = x, y
            prog, err = try_compile(src)
            try:
                outs = prog.run({}, {'position': p}, n)[0]
            except Exception as exc:                            # noqa: BLE001
                bad.append(f'{n} lanes: {err or repr(exc)}')
                continue
            want = {
                'Mul': _per_lane(lambda a, s, t: a * s, m, x, y),
                'Lmul': _per_lane(lambda a, s, t: t * a, m, x, y),
                'Div': _per_lane(lambda a, s, t: a / t, m, x, y),
                'Rdiv': _per_lane(lambda a, s, t: s / a, m, x, y),
                'Add': _per_lane(lambda a, s, t: a + s, m, x, y),
                'Radd': _per_lane(lambda a, s, t: t + a, m, x, y),
                'Sub': _per_lane(lambda a, s, t: a - t, m, x, y),
                'Rsub': _per_lane(lambda a, s, t: s - a, m, x, y),
                'Comp': _per_lane(
                    lambda a, s, t: (a * s + t) / f32(4.0) - s, m, x, y),
                'Step': _per_lane(
                    lambda a, s, t: a + f32(1.0) + f32(1.0) - f32(1.0),
                    m, x, y),
                'Branch': _per_lane(
                    lambda a, s, t: a * t if s > 2.5 else a - s, m, x, y),
            }
            for name, ref in want.items():
                got = outs[name]
                if got.shape != (n, size, size) or got.dtype != f32 \
                        or not np.array_equal(got, ref):
                    bad.append(f'{name} at {n} lanes')
        check(f'matrix and scalar: {mt} per lane -- * / + - with the scalar '
              'on either side, the compound forms, ++ and --, under a '
              f'branch -- bit for bit at {", ".join(map(str, counts))} '
              f'lanes ({size} is where the row axis fitted)',
              not bad, '; '.join(bad[:6]))

    # the rule itself: a matrix with one number of any base, either side of
    # each operator, is that operator on every component -- and float32
    # whatever the number is (an int, left an int, carried the matrix into
    # float64: a precision no GPU has)
    diag = np.diag([1.0, 2.0, 3.0]).astype(f32)
    ops = {'*': np.multiply, '/': np.divide, '+': np.add, '-': np.subtract}
    bad, ran = [], 0
    for text, value in (('2.0', 2.0), ('2', 2.0), ('2u', 2.0),
                        ('true', 1.0)):
        for op, fn in ops.items():
            for left in (True, False):
                expr = f'(m + 1.0) {op} {text}' if left \
                    else f'{text} {op} (m + 1.0)'
                src = ('out mat3 K;\nout vec4 Color;\nvoid main() {\n'
                       + _DIAG + f'    K = {expr};\n'
                       f'    Color = vec4(({expr})[1].xyz, 1.0);\n}}\n')
                ref = fn(diag + f32(1.0), f32(value)) if left \
                    else fn(f32(value), diag + f32(1.0))
                prog, err = try_compile(src)
                for n in (3, 4):
                    ran += 1
                    try:
                        outs = prog.run({}, {}, n)[0]
                    except Exception as exc:                    # noqa: BLE001
                        bad.append(f'{expr}: {err or repr(exc)}')
                        continue
                    if outs['K'].dtype != f32 \
                            or outs['K'].shape != (n, 3, 3) \
                            or not np.array_equal(outs['K'][n - 1], ref) \
                            or not np.array_equal(outs['Color'][0, :3],
                                                  ref[1]):
                        bad.append(f'{expr} at {n} lanes: '
                                   f'{outs["K"].dtype} {outs["K"][n - 1]}')
    check('matrix and scalar: a float, an int, a uint and a bool on either '
          f'side of * / + - ({ran} runs): the operator on every component, '
          'a float32 (lanes, columns, rows) array, and the result indexed '
          'and swizzled as the matrix it is', not bad, '; '.join(bad[:4]))

    # wherever a matrix can stand
    fine = [
        ('returned by and passed to a function',
         _body(_DIAG + '    Color = vec4(twice(m) * vec3(1.0), '
               'twice(m / 2.0)[2].z);\n',
               'mat3 twice(mat3 x) { return x * 2.0; }\n'), [2, 4, 6, 3]),
        ('a struct field, by a field',
         _body(_DIAG + '    T t;\n    t.m = m;\n    t.k = 2.0;\n'
               '    t.m *= t.k;\n'
               '    Color = vec4(t.m * vec3(1.0), (t.m / t.k)[1].y);\n',
               'struct T { mat3 m; float k; };\n'), [2, 4, 6, 2]),
        ('an array element, written through a variable index',
         _body(_DIAG + '    mat3 a[2];\n    int i = 1;\n    a[i] = m;\n'
               '    a[i] *= 2.0;\n    a[i - 1] = a[i] / 4.0;\n'
               '    Color = vec4(a[1] * vec3(1.0), a[0][2][2]);\n'),
         [2, 4, 6, 1.5]),
        ('by a uniform and by a loop counter',
         _body(_DIAG + '    for (int i = 0; i < 3; i++) { m += float(i); }\n'
               '    Color = vec4((m * gain)[0], (gain * m)[1].y);\n',
               'uniform float gain = 0.5;\n'), [2, 1.5, 1.5, 2.5]),
        ('a mat4 and a mat2',
         _body('    mat4 a = mat4(2.0) * 0.5 + 1.0;\n'
               '    mat2 b = 3.0 - mat2(1.0) / 0.5;\n    b++;\n'
               '    Color = vec4(a[3].wx, b[0]);\n'), [2, 1, 2, 4]),
        ('the HLSL dialect',
         'out float4 Color;\nvoid main() {\n    float3x3 m = float3x3(1.0, '
         '0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0);\n'
         '    float3x3 k = m * 2.0;\n    k /= 4.0;\n'
         '    Color = float4(mul(k, float3(1.0, 1.0, 1.0)), '
         'k == m * 0.5 ? 1.0 : 0.0);\n}\n', [0.5, 1, 1.5, 1]),
        # what the run held to the VALUE, as it did: pow(2.0, v) is typed
        # float and holds a vector, and a matrix times it is the linear
        # product it always was
        ('a matrix times a value typed float that holds a vector is still '
         'matrix * vector',
         _body(_DIAG + '    vec3 v = vec3(1.0, 2.0, 3.0);\n'
               '    Color = vec4((m * pow(2.0, v)).zyx, 1.0);\n'),
         [24, 8, 2, 1]),
    ]
    for what, src, want in fine:
        bad = []
        for n in (1, 3, 4, 7):
            got, err = _first(src, 'HLSL' if 'float4' in src else 'GLSL',
                              n=n)
            if got is None or not np.allclose(got, want, atol=1e-6):
                bad.append(f'{n} lanes: {err if got is None else got}')
        check(f'matrix and scalar: {what}', not bad, '; '.join(bad))

    # typed as the language types it. The product was typed a vector and
    # marked loose, so nothing computed from it was judged -- and one thing
    # was misjudged: a column of it was a float, and its swizzle refused
    got, err = _first(_body(_DIAG + '    Color = vec4((m * 2.0)[1].xyz, '
                            '(2.0 * m)[2].z);\n'))
    check('matrix and scalar: (m * 2.0)[1].xyz compiles (it was refused: '
          "swizzle 'xyz' out of range for float)",
          got is not None and np.array_equal(got, [0, 4, 0, 6]),
          str(err if got is None else got))
    _refused('(m * 2.0).xyz, a swizzle of a matrix (it compiled, and raised '
             'from einsum when run)',
             _body(_DIAG + '    Color = vec4((m * 2.0).xyz, 1.0);\n'),
             "line 4: 'xyz' of mat3: a matrix is indexed first")
    _refused('a width computed from the product (nothing computed from it '
             'was judged)',
             _body(_DIAG + '    vec2 r = vec2(0.5);\n'
                   '    Color = vec4((m * 2.0) * vec3(1.0) + r, 1.0);\n'),
             'line 5: cannot combine vec3 with vec2')

    # what is not the language's is left as it was: to the run. An array is
    # the operand of no operator (`a * 2.0` of a `mat3 a[2]`, `m * w` of a
    # `float w[3]`). Typed by the rule above each would be a matrix and a
    # member of it refused -- where the text compiled and, in a branch no
    # lane takes, ran
    got, err = _first(_body(
        _DIAG + '    mat3 a[2];\n    float w[3];\n    Color = vec4(1.0);\n'
        '    if (Color.x > 2.0) {\n'
        '        Color = vec4((a * 2.0).x, (m * w).x, (w * m).y, 1.0);\n'
        '    }\n'))
    check('matrix and scalar: an array is neither (a mat3[2] times a float, '
          'a mat3 times a float[3]): not typed by this rule, and not '
          'refused where it compiled',
          got is not None and np.array_equal(got, [1, 1, 1, 1]),
          str(err if got is None else got))

    # a matrix handed in with one lane (a caller's uniform) against a
    # scalar with many, and the other way about
    src = ('uniform mat3 M;\nin vec3 vPosition;\nout mat3 K;\nout bool Same;'
           '\nout vec4 Color;\nvoid main() {\n    K = M * vPosition.x;\n'
           '    Same = M == M;\n    Color = vec4(M / 2.0 == 0.5 * M ? 1.0 '
           ': 0.0);\n}\n')
    p = _lanes(5, 5) + 1.0
    outs, _, _ = run(src, uniforms={'M': diag[None]}, inputs={'position': p},
                     n=5)
    check('matrix and scalar: one lane of matrix against five of scalar; '
          'the comparison answers a bool for every lane',
          outs['K'].shape == (5, 3, 3)
          and np.array_equal(outs['K'], _per_lane(lambda s: diag * s,
                                                  p[:, 0]))
          and outs['Same'].shape == (5,) and outs['Same'].dtype == bool
          and outs['Same'].all() and np.array_equal(outs['Color'][:, 0],
                                                    np.ones(5)),
          str(outs['K'].shape))

    # NOT this entry's: the Coded Shader node gives a `uniform mat3` a float
    # socket, so the uniform arrives as one float a lane. What a float
    # should mean there is undecided; until it is, a use of it is refused by
    # the run as it was -- never multiplied as if a (lanes,) array were a
    # matrix, which would answer a (lanes, 1, lanes) array without a word
    prog, err = try_compile('uniform mat3 M;\nout vec4 Color;\nvoid main() '
                            '{\n    mat3 k = M * 2.0;\n'
                            '    Color = vec4(k * vec3(1.0), 1.0);\n}\n')
    try:
        prog.run({'M': np.full(4, 0.5, f32)}, {}, 4)
        raised = None
    except Exception as exc:                                    # noqa: BLE001
        raised = exc
    check('matrix and scalar: a float handed in for a `uniform mat3` (the '
          'node\'s socket) still raises when it is used -- it is not taken '
          'for a matrix', raised is not None, repr(raised))


def test_two_matrices_compare_whole():
    """`Color = vec4(m == m ? 1.0 : 0.0);` compiled and raised ValueError:
    the comparison took component [0] of each side, which of a matrix is
    its first column -- a bool per lane and ROW, flattened. In an `if`,
    which cuts its condition to the lane count, that ran: `if (m == k)` was
    taken for two matrices that differed anywhere outside their first
    column (and read another lane's rows where they differed inside it)."""
    # one component of the copy differs, on the odd lanes alone, for each of
    # the nine in turn: `==` is all nine, `!=` any of them, per lane
    bad, ran = [], 0
    for c in range(3):
        for r in range(3):
            src = ('in vec3 vPosition;\nout vec4 Color;\nout bool Same;\n'
                   'void main() {\n    float x = vPosition.x;\n'
                   '    mat3 m = mat3(x, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, '
                   '8.0);\n    mat3 k = m;\n'
                   f'    if (mod(x, 2.0) > 0.5) {{ k[{c}][{r}] += 0.5; }}\n'
                   '    Same = m == k;\n'
                   '    Color = vec4(m == k ? 1.0 : 0.0, m != k ? 1.0 : 0.0,'
                   ' 0.0, Same ? 1.0 : 0.0);\n'
                   '    if (k == m) { Color.z = 1.0; }\n}\n')
            for n in (1, 3, 4, 7):
                p = np.zeros((n, 3), np.float32)
                p[:, 0] = np.arange(n) + 1.0
                ran += 1
                try:
                    outs, _, _ = run(src, inputs={'position': p}, n=n)
                except Exception as exc:                        # noqa: BLE001
                    bad.append(f'[{c}][{r}] at {n} lanes: {exc!r}')
                    continue
                same = (np.arange(n) + 1) % 2 == 0
                col = outs['Color']
                if outs['Same'].dtype != bool \
                        or not np.array_equal(outs['Same'], same) \
                        or not np.array_equal(col[:, 0], same) \
                        or not np.array_equal(col[:, 1], ~same) \
                        or not np.array_equal(col[:, 2], same) \
                        or not np.array_equal(col[:, 3], same):
                    bad.append(f'[{c}][{r}] at {n} lanes: {col[:, :2]}')
    check('matrix ==: a copy that differs in one component on the odd lanes, '
          f'each of the nine in turn, at 1, 3, 4 and 7 lanes ({ran} runs): '
          'equal where all nine are, in a ternary, a bool and an if',
          not bad, '; '.join(bad[:4]))

    fine = [
        ('a mat2 and a mat4; a product against its factors',
         _body('    mat4 a = mat4(2.0);\n    mat2 b = mat2(1.0, 2.0, 3.0, '
               '4.0);\n    mat2 t = transpose(b);\n'
               '    Color = vec4(a == mat4(2.0), a * 0.5 != mat4(1.0), '
               'b == t, transpose(t) == b && b != t);\n'), [1, 0, 0, 1]),
    ]
    for what, src, want in fine:
        bad = []
        for n in (1, 3, 4, 7):
            got, err = _first(src, n=n)
            if got is None or not np.array_equal(got, want):
                bad.append(f'{n} lanes: {err if got is None else got}')
        check(f'matrix ==: {what}', not bad, '; '.join(bad))



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
