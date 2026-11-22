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


# ------------------------------------ an array sized by a constant expression
#
# R253 (review): `const int N = 3; float a[N];` declared ONE float. The
# parser read a size with `_const_int`, which knew a number and answered 0
# -- "no brackets" -- for anything else: every index read the one float, a
# write past index 0 was refused by name (the indexed-write test above
# keeps that refusal loud), and a size nothing declares compiled. The
# shader named "the reported shader" below is the text the defect was
# reproduced with; the comment over each test says what the run did.

_N3 = 'const int N = 3;\n'

_SIZE_REPORTED = """const int N = 3;
out vec4 Color;
void main() {
    float a[N];
    a[0] = 0.5;
    Color = vec4(a[0], a[2], 0.0, 1.0);
}"""


def _sized(size, count, head='', pre=''):
    """A program whose array `a` is declared with `size` written in its
    brackets and used at every one of the `count` elements that is to
    mean: filled through a loop counter, the lane's own element changed,
    the whole summed with a weight per element, both ends read by number
    and one element through the lane's index."""
    return ('in vec3 vPosition;\nout vec4 Color;\n' + head
            + 'void main() {\n' + pre
            + '    int k = int(vPosition.x);\n'
            f'    float a[{size}];\n'
            f'    for (int i = 0; i < {count}; i++) '
            '{ a[i] = float(i + 1) * 0.25; }\n'
            '    a[k] += vPosition.y;\n    float s = 0.0;\n'
            f'    for (int i = 0; i < {count}; i++) '
            '{ s += a[i] * float(i + 1); }\n'
            f'    Color = vec4(a[0], a[{count - 1}], a[k], s);\n}}\n')


def _sized_reference(count, p):
    """What `_sized` computes for `count` elements, in NumPy. Every number
    is a small multiple of 1/4, so float32 holds each sum exactly and the
    two can be held to equality."""
    lanes = np.arange(len(p))
    k = p[:, 0].astype(int)
    weight = np.arange(1, count + 1, dtype=np.float32)
    a = np.tile(weight * np.float32(0.25), (len(p), 1))
    a[lanes, k] += p[:, 1]
    return np.stack([a[:, 0], a[:, count - 1], a[lanes, k],
                     (a * weight).sum(1, dtype=np.float32)], 1)


def _declared(prog):
    """{name: array size} of every local a program declares, read off its
    tree: 0 is no array at all, which is what a named size used to be."""
    out = {}

    def walk(node):
        if isinstance(node, tuple) and node and node[0] == 'decl':
            for name, arr, _init in node[2]:
                out[name] = arr
        if isinstance(node, (tuple, list)):
            for part in node:
                walk(part)

    for cands in prog.interp.funcs.values():
        for fn in cands:
            walk(fn[4])
    return out


#: (what, declarations above main, statements at the top of main, the size
#: as written, the number it is)
_SIZE_FORMS = [
    ('a const int', _N3, '', 'N', 3),
    ('N * 2', _N3, '', 'N * 2', 6),
    ('N + 1', _N3, '', 'N + 1', 4),
    ('parentheses: (N + 1) * 2', _N3, '', '(N + 1) * 2', 8),
    ('...and without them: N + 1 * 2', _N3, '', 'N + 1 * 2', 5),
    ('from the left: 9 - N - 1 and 16 / 2 / N', _N3, '',
     '9 - N - 1 + 16 / 2 / N', 7),
    ('the quotient of two ints, cut: N / 2 + 1', _N3, '', 'N / 2 + 1', 2),
    ('a remainder, a shift, the bit operators',
     _N3, '', '7 % N + (N << 1) - (6 & N | 1)', 4),
    ('unary minus and plus', _N3, '', '-N + +7', 4),
    ('two numbers and no name: 2 + 1', '', '', '2 + 1', 3),
    ('a constant made of constants',
     _N3 + 'const int K = N * 2 + 1;\n', '', 'K', 7),
    ('...three deep', _N3 + 'const int K = N * 2;\nconst int L = K - N;\n'
     'const int M = L + K / N;\n', '', 'M', 5),
    ('two constants of one declaration, the second made of the first',
     'const int A = 2, B = A + 3;\n', '    const int C = B - A, D = C * 2;\n',
     'B + D - A', 9),
    ('a const int of the function', '', '    const int L = 4;\n', 'L', 4),
    ('...made of a global one', _N3, '    const int L = N + 2;\n', 'L', 5),
    ('a const uint', 'const uint U = 3u;\n', '', 'U', 3),
    ('...plus a uint', 'const uint U = 3u;\n', '', 'U + 1u', 4),
    ('a #define of an expression', '#define M (2 + 1)\n', '', 'M', 3),
    ('a #define inside a constant',
     '#define W 2\nconst int N = W + 1;\n', '', 'N * W', 6),
    ('a ternary of constants', _N3, '', 'N > 2 ? N + 1 : 2', 4),
    ('a cast of a float constant',
     'const float R = 2.5;\n', '', 'int(R) + 1', 3),
    ('builtins of constants',
     _N3 + 'const float R = 2.5;\n', '', 'max(N, 4) + int(ceil(R))', 7),
    ('a component of a constant vector',
     'const ivec2 D = ivec2(2, 5);\n', '', 'D.y', 5),
    ('an element of a constant array',
     'const int T[2] = int[2](2, 4);\n', '', 'T[1]', 4),
    ('a field of a constant struct',
     'struct P { int n; float f; };\nconst P K = P(3, 1.0);\n', '', 'K.n',
     3),
    ('a constant declared with a named size itself',
     _N3 + 'const int T[N] = int[N](2, 4, 6);\n', '', 'T[N - 1]', 6),
]


def test_an_array_is_sized_by_a_constant_expression():
    """`const int N = 3; float a[N]; a[0] = 0.5; Color = vec4(a[0], a[2],
    0.0, 1.0);` compiled and answered (0.5, 0.5, 0, 1): `a` was a float,
    and `a[2]` read it. `a[1] = 0.5` raised `index 1 is out of range for
    float`, and the loop that fills the array `float is not an array, a
    vector or a matrix: a variable index cannot write to it`."""
    fill = '    for (int i = 0; i < N; i++) { a[i] = float(i) * 0.25; }\n'
    reported = [
        ('one element written, another read (it answered 0.5 for a[2])',
         _SIZE_REPORTED, [0.5, 0, 0, 1]),
        ('a write past index 0 (index 1 is out of range for float)',
         _N3 + _body('    float a[N];\n    a[1] = 0.5;\n'
                     '    Color = vec4(a[0], a[1], a[2], 1.0);\n'),
         [0, 0.5, 0, 1]),
        ('filled in a loop (float is not an array, a vector or a matrix)',
         _N3 + _body('    float a[N];\n' + fill
                     + '    Color = vec4(a[0], a[1], a[2], 1.0);\n'),
         [0, 0.25, 0.5, 1]),
    ]
    for what, src, want in reported:
        bad = []
        for n in (1, 2, 3, 4, 7):
            got, err = _first(src, n=n)
            if got is None or not np.array_equal(got, want):
                bad.append(f'{n} lanes: {err if got is None else got}')
        check(f'array size: {what} (the reported shader, at 1, 2, 3, 4 and '
              '7 lanes)', not bad, '; '.join(bad))

    # the rule: a size written as an expression is the size written as its
    # number. Each form is compiled both ways and run per lane (every lane
    # changes the element ITS index names); the two return the same
    # arrays, they are the arrays NumPy computes, and the tree holds the
    # number -- which a use of the first elements alone would not show
    for what, head, pre, size, count in _SIZE_FORMS:
        p = _lanes(2 * count + 1, count)
        named, err = try_compile(_sized(size, count, head, pre))
        plain, _e = try_compile(_sized(str(count), count, head, pre))
        if named is None:
            check(f'array size: {what}', False, repr(err))
            continue
        try:
            got = named.run({}, {'position': p}, len(p))[0]['Color']
            twin = plain.run({}, {'position': p}, len(p))[0]['Color']
        except Exception as exc:                                # noqa: BLE001
            check(f'array size: {what}', False,
                  f'{type(exc).__name__}: {exc}')
            continue
        check(f'array size: {what} -- a[{size}] is a[{count}]',
              _declared(named)['a'] == count == _declared(plain)['a']
              and got.dtype == twin.dtype and np.array_equal(got, twin)
              and np.array_equal(got, _sized_reference(count, p)),
              f'declared {_declared(named)["a"]}; {got[:3]}')

    # wherever a declarator stands: the suffix of a global, a uniform, a
    # parameter, a struct's field, a second declarator, a local of any
    # block. `{S}` is written N + 1 and then 4
    sites = [
        ('a global',
         _N3 + 'float g[{S}];\n' + _body(
             '    for (int i = 0; i <= N; i++) { g[i] = float(i) * 0.25; }\n'
             '    Color = vec4(g[0], g[1], g[2], g[3]);\n'),
         [0, 0.25, 0.5, 0.75],
         lambda prog: prog.interp.globals[-1][1].name()),
        ('a uniform',
         _N3 + 'uniform float w[{S}];\n' + _body(
             '    Color = vec4(w[0], w[1], w[2], w[3] + 1.0);\n'),
         [0, 0, 0, 1], lambda prog: prog.uniforms[0][1].name()),
        ('a parameter',
         _N3 + 'float total(float v[{S}]) {\n    float s = 0.0;\n'
         '    for (int i = 0; i <= N; i++) { s += v[i]; }\n'
         '    return s;\n}\n' + _body(
             '    float a[{S}];\n'
             '    for (int i = 0; i <= N; i++) { a[i] = float(i) * 0.25; }\n'
             '    Color = vec4(total(a), a[1], a[2], a[3]);\n'),
         [1.5, 0.25, 0.5, 0.75],
         lambda prog: prog.interp.funcs['total'][0][3][0][3]),
        ("a struct's field",
         _N3 + 'struct S { float w[{S}]; vec2 p[{S}]; };\n' + _body(
             '    S s;\n    for (int i = 0; i <= N; i++) { s.w[i] = '
             'float(i) * 0.25; s.p[i].y = float(i); }\n'
             '    Color = vec4(s.w[1], s.w[3], s.p[2].y, s.p[3].y);\n'),
         [0.25, 0.75, 2, 3],
         lambda prog: [t.name() for t in prog.interp.fields('S').values()]),
        ('every declarator of one declaration',
         _N3 + _body(
             '    float a[{S}], f = 2.0, b[{S}];\n'
             '    for (int i = 0; i <= N; i++) { a[i] = float(i); '
             'b[N - i] = float(i) * 0.25; }\n'
             '    Color = vec4(a[3], b[0], b[3], f);\n'),
         [3, 0.75, 0, 2], _declared),
        ('an array of vectors, an array of structs',
         _N3 + 'struct S { float k; };\n' + _body(
             '    vec3 p[{S}];\n    S t[{S}];\n'
             '    for (int i = 0; i <= N; i++) { p[i] = vec3(float(i)); '
             't[i].k = float(i) * 0.25; }\n'
             '    Color = vec4(p[1].x, p[3].z, t[2].k, t[3].k);\n'),
         [1, 3, 0.5, 0.75], _declared),
        ('a local of a loop, of a branch, of a helper',
         _N3 + 'float last() {\n    float c[{S}];\n    c[N] = 0.75;\n'
         '    return c[N];\n}\n' + _body(
             '    float r = 0.0;\n    float q = 0.0;\n'
             '    for (int j = 0; j < 2; j++) {\n        float a[{S}];\n'
             '        a[N] = float(j + 1);\n        r += a[N] + a[0];\n'
             '    }\n    if (r > 2.0) {\n        float b[{S}];\n'
             '        b[N - 1] = 0.5;\n        q = b[N - 1] + b[N];\n    }\n'
             '    Color = vec4(r, q, last(), 1.0);\n'),
         [3, 0.5, 0.75, 1], _declared),
    ]
    for what, src, want, shape in sites:
        named, err = try_compile(src.replace('{S}', 'N + 1'))
        plain, _e = try_compile(src.replace('{S}', '4'))
        got, err2 = _first(src.replace('{S}', 'N + 1'), n=3)
        twin, _e = _first(src.replace('{S}', '4'), n=3)
        check(f'array size: {what} sized N + 1 is the one sized 4',
              named is not None and got is not None
              and np.array_equal(got, want) and np.array_equal(got, twin)
              and shape(named) == shape(plain),
              str(err or err2 or (got, shape(named))))

    # a uniform array and its socket. The node gives `uniform float w[N]`
    # ONE float socket and the evaluator hands the run that socket's value,
    # a float per lane (core/nodeeval). While a named size declared a
    # float, every `w[i]` read it; now that it declares an array every
    # element reads it -- the same picture -- and so does an array sized
    # by a number, which read LANE i's value as element i. A list is the
    # array's own elements, as it was
    lanes = np.array([0.25, 0.5, 0.75, 1.0], np.float32)
    rgb = np.stack([lanes, lanes * 2, lanes * 4], 1)
    bad = []
    for size in ('N', '3'):
        prog, err = try_compile(
            _N3 + f'uniform float w[{size}];\nuniform vec3 p[{size}];\n'
            + _body('    float v[N];\n    v[1] = w[1] * 2.0;\n'
                    '    Color = vec4(w[0], v[1], w[2] + p[2].z, p[1].y);\n'))
        if prog is None:
            bad.append(f'[{size}]: {err}')
            continue
        fed = [
            ('a float per lane', {'w': lanes, 'p': rgb},
             np.stack([lanes, lanes * 2, lanes * 5, lanes * 2], 1)),
            ('one number', {'w': np.float32(0.5),
                            'p': np.array([[1.0, 2.0, 4.0]], np.float32)},
             [[0.5, 1, 4.5, 2]] * 4),
            ('a list of elements',
             {'w': [lanes * 0 + 0.25, lanes, lanes * 2], 'p': rgb},
             np.stack([lanes * 0 + 0.25, lanes * 2, lanes * 6, lanes * 2],
                      1)),
            ('nothing', {}, [[0, 0, 0, 0]] * 4),
        ]
        for what, uniforms, want in fed:
            try:
                got = prog.run(uniforms, {}, 4)[0]['Color']
            except Exception as exc:                            # noqa: BLE001
                bad.append(f'[{size}] {what}: {type(exc).__name__}: {exc}')
                continue
            if not np.array_equal(got, np.asarray(want, np.float32)):
                bad.append(f'[{size}] {what}: {got.tolist()}')
    check('array size: a uniform array reads its socket in every element, '
          'sized by a name or by a number (a float per lane, one number, '
          'a list of elements, nothing)', not bad, '; '.join(bad))

    # the size written in an array constructor is a size too, and then
    # informative: the array is as long as its arguments
    got, err = _first(_N3 + _body(
        '    float a[N] = float[N](0.25, 0.5, 0.75);\n'
        '    float b[] = float[N - 1](1.0, 2.0);\n'
        '    float c[] = float[2](3.0, 4.0);\n'
        '    Color = vec4(a[2], b[1], c[1], a[0]);\n'))
    check('array size: a constructor sized by a constant expression',
          got is not None and np.array_equal(got, [0.75, 2, 4, 0.25]),
          str(err if got is None else got))

    # the HLSL dialect (`static const`), and a `const` the shader spells
    # with the qualifiers the other way round
    for what, src in (
            ('HLSL, static const',
             'static const int N = 3;\nstatic const int K = N * 2;\n'
             'out float4 Color;\nvoid main() {\n    float a[K];\n'
             '    for (int i = 0; i < K; i++) { a[i] = i * 0.25; }\n'
             '    Color = float4(a[0], a[2], a[5], K);\n}\n'),
            ('HLSL, const static, sized in a helper',
             'const static int K = 6;\nout float4 Color;\n'
             'float last() {\n    float a[K];\n    a[K - 1] = 1.25;\n'
             '    a[2] = 0.5;\n    return a[5] + a[2] - a[0];\n}\n'
             'void main() {\n    Color = float4(0, 0.5, last() - 0.5, K);\n'
             '}\n')):
        got, err = _first(src, 'HLSL')
        check(f'array size: {what}',
              got is not None and np.array_equal(got, [0, 0.5, 1.25, 6]),
              str(err if got is None else got))
    got, err = _first(
        'cbuffer C { float gain; };\nstatic const int N = 3;\n'
        'out float4 Color;\nvoid main() {\n    const static int M = N * 2;\n'
        '    static const int L = M - 1;\n    float a[M];\n    float b[L];\n'
        '    a[5] = 0.5;\n    b[4] = 0.25;\n'
        '    Color = float4(a[5], b[4], gain, M + L);\n}\n', 'HLSL')
    check('array size: HLSL, a constant after a cbuffer, and `const '
          'static` / `static const` in a function',
          got is not None and np.array_equal(got, [0.5, 0.25, 0, 11]),
          str(err if got is None else got))

    # which constant a name is follows the scopes the run builds: the
    # nearest declaration above the brackets, whose own initialiser still
    # saw the outer name
    scopes = [
        ('the function\'s own N hides the global one',
         _N3 + _body('    const int N = 2;\n    float a[N];\n'
                     '    Color = vec4(float(N));\n'), {'a': 2}, 2),
        ('...from the statement after it: its initialiser saw the global',
         _N3 + _body('    const int N = N + 1;\n    float a[N];\n'
                     '    Color = vec4(float(N));\n'), {'a': 4}, 4),
        ('a block\'s N, and the global one again after the block',
         _N3 + _body('    { const int N = 5;\n      float a[N]; }\n'
                     '    float b[N];\n    Color = vec4(float(N));\n'),
         {'a': 5, 'b': 3}, 3),
        ('each function its own, and a loop its own',
         _N3 + 'float f() { const int N = 2; float c[N * 2]; return 1.0; }\n'
         + _body('    for (int i = 0; i < 2; i++) {\n'
                 '        const int N = 6;\n        float a[N];\n    }\n'
                 '    float b[N];\n    Color = vec4(float(N) + f());\n'),
         {'a': 6, 'b': 3, 'c': 4}, 4),
        ('a parameter of that name hides it in its own function alone',
         _N3 + 'float f(int N) { return float(N); }\n'
         + _body('    float a[N];\n    Color = vec4(f(1) + 3.0);\n'),
         {'a': 3}, 4),
        ('a constant of the head of a loop, in its body',
         _body('    float r = 0.0;\n'
               '    for (const int M = 2; r < 1.0; r += 1.0) {\n'
               '        float a[M * 2];\n    }\n'
               '    Color = vec4(r + 4.0);\n'), {'a': 4}, 5),
    ]
    for what, src, sizes, want in scopes:
        prog, err = try_compile(src)
        got, err2 = _first(src)
        check(f'array size: {what}',
              prog is not None and got is not None
              and {k: _declared(prog)[k] for k in sizes} == sizes
              and np.array_equal(got, [want] * 4),
              str(err or err2 or (_declared(prog), got)))

    # the two refusals that kept the named size from aliasing in silence
    # fire for what they name and nothing else: a number past the size is
    # said with the size the name resolved to (it said `for float`)
    got, err = _first(_N3 + _body('    float a[N * 2];\n    a[6] = 1.0;\n'
                                  '    Color = vec4(a[5]);\n'))
    check('array size: a number written past a named size is refused by '
          'the size it is',
          got is None and err == 'compiled, then ShaderError: index 6 is '
          'out of range for float[6]', str(err if got is None else got))


def test_a_size_that_is_not_a_constant_is_a_compile_error():
    """`float a[UNDECLARED];` compiled, and so did `uniform int count;
    float a[count];` -- each declared a float, as every size that was not a
    number did. A size is a constant expression of type int, at least 1:
    what is not is refused where it is written, whether or not a lane ever
    reaches the line -- a driver compiles none of these."""
    use = '    Color = vec4(a[0]);\n'
    count = 'uniform int count = 3;\n'
    no = 'array size is not a compile-time constant: '
    refused = [
        # ---- a name nothing above declares
        ('a size nothing declares (the reported shader)',
         _body('    float a[UNDECLARED];\n    a[0] = 0.5;\n' + use),
         "line 3: undeclared identifier 'UNDECLARED'"),
        ('...inside an expression',
         _N3 + _body('    float a[N * 2 + M];\n' + use),
         "line 4: undeclared identifier 'M'"),
        ('a constant declared below the size',
         _body('    float a[N];\n' + use) + _N3,
         "line 3: undeclared identifier 'N'"),
        ('a constant of a block that has closed',
         _body('    { const int M = 2; }\n    float a[M];\n' + use),
         "line 4: undeclared identifier 'M'"),
        ('...of an if arm without braces',
         _body('    if (true) const int M = 2;\n    float a[M];\n' + use),
         "line 4: undeclared identifier 'M'"),
        ('...of an else arm without braces',
         _body('    if (false) { } else const int M = 2;\n'
               '    float a[M];\n' + use),
         "line 4: undeclared identifier 'M'"),
        ('...of a loop',
         _body('    for (int i = 0; i < 2; i++) { const int M = 2; }\n'
               '    float a[M];\n' + use),
         "line 4: undeclared identifier 'M'"),
        ('...of the head of a loop, after the loop',
         _body('    for (const int M = 2; false; ) { }\n'
               '    float a[M];\n' + use),
         "line 4: undeclared identifier 'M'"),
        ('...of a loop body without braces: for, while, do',
         _body('    for (int i = 0; i < 2; i++) const int F = 2;\n'
               '    while (false) const int W = 2;\n'
               '    do const int D = 2; while (false);\n'
               '    float a[F + W + D];\n' + use),
         "line 6: undeclared identifier 'F'"),
        ('...of a while body without braces',
         _body('    while (false) const int W = 2;\n'
               '    float a[W];\n' + use),
         "line 4: undeclared identifier 'W'"),
        ('...of a do body without braces',
         _body('    do const int D = 2; while (false);\n'
               '    float a[D];\n' + use),
         "line 4: undeclared identifier 'D'"),
        ('...of another function',
         _body('    float a[M];\n' + use,
               'float f() { const int M = 2; return 1.0; }\n'),
         "line 4: undeclared identifier 'M'"),
        ('...of another case of a switch',
         _body('    int k = 1;\n    switch (k) {\n'
               '    case 0: const int M = 2; break;\n'
               '    case 1: float a[M]; break;\n    }\n'
               '    Color = vec4(1.0);\n'),
         "line 6: undeclared identifier 'M'"),
        ('...of a case, in the default after it',
         _body('    int k = 1;\n    switch (k) {\n'
               '    case 0: const int M = 2; break;\n'
               '    default: float a[M]; break;\n    }\n'
               '    Color = vec4(1.0);\n'),
         "line 6: undeclared identifier 'M'"),
        ('...written in a switch above its first case (the statement is '
         'dropped, and its name with it)',
         _body('    int k = 1;\n    switch (k) {\n'
               '        const int M = 2;\n    case 0: break;\n    }\n'
               '    float a[M];\n' + use),
         "line 8: undeclared identifier 'M'"),
        ('...of a switch, after it',
         _body('    int k = 1;\n    switch (k) {\n'
               '    default: const int M = 2; break;\n    }\n'
               '    float a[M];\n' + use),
         "line 7: undeclared identifier 'M'"),
        # ---- a name that is not a constant
        ('a uniform', _body('    float a[count];\n' + use, count),
         f"line 4: {no}'count' is a uniform"),
        ('...as the size of a global',
         count + 'float g[count];\n' + _body('    Color = vec4(g[0]);\n'),
         f"line 2: {no}'count' is a uniform"),
        ("...of a struct's field",
         count + 'struct S { float w[count * 2]; };\n'
         + _body('    Color = vec4(1.0);\n'),
         f"line 2: {no}'count' is a uniform"),
        ('...of a parameter',
         count + 'float f(float v[count]) { return v[0]; }\n'
         + _body('    Color = vec4(1.0);\n'),
         f"line 2: {no}'count' is a uniform"),
        ('an input',
         _body('    float a[int(vDepth) + 1];\n' + use, 'in float vDepth;\n'),
         f"line 4: {no}'vDepth' is an input"),
        ('gl_FragCoord',
         _body('    float a[int(gl_FragCoord.x) + 1];\n' + use),
         f"line 3: {no}'gl_FragCoord' is not a constant"),
        ('a global that is not const',
         _body('    float a[n];\n' + use, 'int n = 3;\n'),
         f"line 4: {no}'n' is not declared const"),
        ('a local that is not const',
         _body('    int n = 3;\n    float a[n];\n' + use),
         f"line 4: {no}'n' is not declared const"),
        ('a loop counter',
         _body('    for (int i = 1; i < 3; i++) { float a[i]; }\n'
               '    Color = vec4(1.0);\n'),
         f"line 3: {no}'i' is not declared const"),
        ('a parameter, const or not',
         _body('    Color = vec4(f(3));\n',
               'float f(const int n) { float a[n]; return a[0]; }\n'),
         f"line 2: {no}'n' is a parameter"),
        ('a const whose initialiser is not constant',
         _body('    const int n = count * 2;\n    float a[n];\n' + use,
               count),
         f"line 5: {no}the initialiser of 'n' is not constant ('count' is "
         'a uniform)'),
        ('...at one remove',
         _body('    float a[m + 1];\n' + use,
               count + 'const int n = count;\nconst int m = n * 2;\n'),
         f"line 6: {no}the initialiser of 'm' is not constant (the "
         "initialiser of 'n' is not constant ('count' is a uniform))"),
        ('a const with no initialiser',
         _body('    float a[n];\n' + use, 'const int n;\n'),
         f"line 4: {no}'n' has no initialiser"),
        ('a local that hides the constant: the size is not the outer 3',
         _N3 + _body('    int N = 2;\n    float a[N];\n' + use),
         f"line 5: {no}'N' is not declared const"),
        ('a parameter that hides it',
         _N3 + _body('    Color = vec4(f(1));\n',
                     'float f(int N) { float a[N]; return a[0]; }\n'),
         f"line 3: {no}'N' is a parameter"),
        ('an HLSL cbuffer\'s member, whatever it says',
         'cbuffer C { const int N = 3; };\nout float4 Color;\n'
         'void main() {\n    float a[N];\n    Color = float4(a[0]);\n}\n',
         f"line 4: {no}'N' is a cbuffer's member"),
        # ---- an expression that is not one
        ("the shader's own function",
         _body('    float a[three()];\n' + use,
               'int three() { return 3; }\n'),
         f"line 4: {no}three() is the shader's own function"),
        ('...declared above the size and defined below it',
         _body('    float a[three()];\n' + use, 'int three();\n')
         + 'int three() { return 3; }\n',
         f"line 4: {no}three() is the shader's own function"),
        ('...defined below the size, with nothing above it',
         _body('    float a[three()];\n' + use)
         + 'int three() { return 3; }\n',
         f'line 3: {no}three() is not declared above it'),
        ('...even with a builtin\'s name, defined below the size',
         _body('    float a[max(2, 3)];\n' + use)
         + 'int max(int a, int b) { return 7; }\n',
         f"line 3: {no}max() is the shader's own function"),
        ('a texture read',
         _body('    float a[int(texture(t, vec2(0.0)).x) + 1];\n' + use,
               'uniform sampler2D t;\n'),
         f"line 4: {no}texture() is not evaluated at compile time"),
        ('a builtin the language does not fold',
         _body('    float a[int(noise(vec3(0.5))) + 1];\n' + use),
         f'line 3: {no}noise() is not evaluated at compile time'),
        ('a ternary whose condition is not constant',
         _body('    float a[count > 2 ? 3 : 2];\n' + use, count),
         f"line 4: {no}'count' is a uniform"),
        ('an assignment',
         _body('    int n = 1;\n    float a[n = 3];\n' + use),
         f"line 4: {no}it assigns"),
        ('an increment',
         _N3 + _body('    float a[N++];\n' + use),
         f"line 4: {no}it writes (++ or --)"),
        ('...written in front', _N3 + _body('    float a[++N];\n' + use),
         f"line 4: {no}it writes (++ or --)"),
        ('a decrement', _N3 + _body('    float a[--N + 2];\n' + use),
         f"line 4: {no}it writes (++ or --)"),
        ('a comma', _N3 + _body('    float a[(1, N)];\n' + use),
         f"line 4: {no}it holds a comma"),
        # ---- a constant that is not an int
        ('a float constant',
         _body('    float a[R];\n' + use, 'const float R = 3.0;\n'),
         'line 4: array size must be an int, this is a float'),
        ('an int times a float',
         _N3 + _body('    float a[N * 1.5];\n' + use),
         'line 4: array size must be an int, this is a float'),
        ('a float constant given an int: it is the type it is declared',
         _body('    float a[R / 2 + 1];\n' + use, 'const float R = 5;\n'),
         'line 4: array size must be an int, this is a float'),
        ('a bool', _body('    float a[true];\n' + use),
         'line 3: array size must be an int, this is a bool'),
        ('a vector',
         _body('    float a[D];\n' + use, 'const ivec2 D = ivec2(2, 3);\n'),
         'line 4: array size must be an int, this is an ivec2'),
        ('a builtin of an int that answers a fraction',
         _body('    float a[sqrt(2)];\n' + use),
         'line 3: array size must be an int, this is 1.41421'),
        # ---- an int that is no size
        ('zero', _body('    float a[0];\n' + use),
         'line 3: array size must be at least 1, not 0'),
        ('a constant that is zero',
         _body('    float a[Z];\n' + use, 'const int Z = 0;\n'),
         'line 4: array size must be at least 1, not 0'),
        ('below zero', _N3 + _body('    float a[N - 5];\n' + use),
         'line 4: array size must be at least 1, not -2'),
        ('a division by zero', _body('    float a[1 / 0];\n' + use),
         'line 3: array size does not fit an int'),
        ('an array with nothing in it',
         _body('    float a[int(float[]())];\n' + use),
         'line 3: array size is not one number'),
        # ---- the size written in an array constructor is a size too
        ('a name nothing declares, in a constructor\'s brackets',
         _body('    float a[] = float[ZZ](1.0, 2.0);\n' + use),
         "line 3: undeclared identifier 'ZZ'"),
        ('a constructor size that is not a constant',
         _body('    int n = 2;\n    float a[] = float[n](1.0, 2.0);\n'
               + use),
         f"line 4: {no}'n' is not declared const"),
        # ---- what the name check refuses in the body, in the brackets
        ('a function nothing declares',
         _body('    float a[nope(3)];\n' + use),
         f'line 3: {no}nope() is not declared above it'),
        ('a call with the wrong count',
         _N3 + _body('    float a[max(N)];\n' + use),
         'line 4: max() takes 2 arguments, got 1'),
        ("a swizzle past the constant's width",
         _body('    float a[D.z];\n' + use, 'const ivec2 D = ivec2(2, 3);\n'),
         "line 4: swizzle 'z' out of range for ivec2"),
        ('a field the constant does not have',
         _body('    float a[K.m];\n' + use,
               'struct P { int n; };\nconst P K = P(3);\n'),
         "line 5: P has no field 'm'"),
        ('...in the initialiser of a constant the size names',
         _body('    float a[M];\n' + use, 'const ivec2 D = ivec2(2, 3);\n'
               'const int M = D.z + 1;\n'),
         "line 3: swizzle 'z' out of range for ivec2"),
    ]
    for what, src, want in refused:
        _refused(f'array size, {what}', src, want,
                 'HLSL' if 'float4' in src else 'GLSL')

    # refused where it is written, not where a lane arrives: in a helper
    # nothing calls, and behind a branch no lane takes
    _refused('array size, a non-constant size in a function nothing calls',
             _body('    Color = vec4(1.0);\n', count
                   + 'float unused() { float a[count]; return a[0]; }\n'),
             f"line 3: {no}'count' is a uniform")
    _refused('array size, ...behind a branch no lane takes',
             _body('    Color = vec4(1.0);\n'
                   '    if (count > 99) { float a[count]; Color = '
                   'vec4(a[0]); }\n', count),
             f"line 5: {no}'count' is a uniform")

    # what a number in the brackets does is what it did: an int, an int
    # with a suffix and (this front-end's own latitude) a float all read
    # as before; `[]` is sized by its initialiser. One spelling is new: a
    # size in hex failed the compile with Python's own TypeError
    numbers = [('3', 3), ('3u', 3), ('3.0', 3), ('3.9', 3), ('1e1', 10),
               ('4f', 4), ('0x10', 16), ('0x1Fu', 31)]
    bad = []
    for text, want in numbers:
        prog, err = try_compile(_body(
            f'    float a[{text}];\n    a[{want - 1}] = 0.5;\n'
            f'    Color = vec4(a[{want - 1}]);\n'))
        if prog is None or _declared(prog)['a'] != want or \
                not np.array_equal(prog.run({}, {}, 2)[0]['Color'][0],
                                   [0.5] * 4):
            bad.append(f'[{text}]: {err or _declared(prog)}')
    check('array size: a number in the brackets is the size it was '
          f'({", ".join(t for t, _w in numbers)})', not bad, '; '.join(bad))
    prog, err = try_compile(_body(
        '    float a[] = float[](0.25, 0.5, 0.75);\n    float b[];\n'
        '    Color = vec4(a[0], a[1], a[2], b[0] + 1.0);\n'))
    check('array size: [] is still sized by its initialiser',
          prog is not None and _declared(prog) == {'a': -1, 'b': -1}
          and np.array_equal(prog.run({}, {}, 2)[0]['Color'][0],
                             [0.25, 0.5, 0.75, 1]), repr(err))

    # the scopes the parser keeps cost no depth of Python's stack: a chain
    # of `else if`, and arms, loops and blocks inside one another, parse
    # as deep as they did. (Each scope was first opened through a helper
    # that called the parse -- one more frame a level -- and a chain of
    # 400 arms, which compiled, was "too deeply nested".)
    deep = 400
    arms = '\n    else '.join(f'if (i == {j}) c = {j}.0;' for j in range(deep))
    nests = [
        ('else if', 'in vec3 vPosition;\nout vec4 Color;\nvoid main() {\n'
         '    int i = int(vPosition.x);\n    float c = -1.0;\n    ' + arms
         + '\n    Color = vec4(c);\n}\n'),
        ('if', _body('    float c = 0.0;\n    ' + 'if (c < 1.0) ' * deep
                     + 'c = 1.0;\n    Color = vec4(c);\n')),
        ('for', _body('    float c = 0.0;\n    ' + ''.join(
            f'for (int i{j} = 0; i{j} < 1; i{j}++) ' for j in range(deep))
            + 'c += 1.0;\n    Color = vec4(c);\n')),
        ('while', _body('    float c = 0.0;\n    ' + 'while (c < 1.0) ' * deep
                        + 'c = 1.0;\n    Color = vec4(c);\n')),
        ('do', _body('    float c = 0.0;\n    ' + 'do ' * deep + 'c += 1.0;'
                     + ' while (false);' * deep + '\n    Color = vec4(c);\n')),
        ('{', _body('    float c = 0.0;\n    ' + '{ ' * deep + 'c = 1.0;'
                    + ' }' * deep + '\n    Color = vec4(c);\n')),
    ]
    bad = [f'{what}: {err}' for what, err in
           ((what, try_compile(src)[1]) for what, src in nests) if err]
    check(f'array size: {deep} levels of else if, if, for, while, do and '
          '{ } still compile', not bad, '; '.join(bad))
    short = '\n    else '.join(f'if (i == {j}) c = {j}.0;' for j in range(150))
    p = np.zeros((4, 3), np.float32)
    p[:, 0] = [0, 57, 149, 150]
    outs, _, _ = run('in vec3 vPosition;\nout vec4 Color;\nvoid main() {\n'
                     '    int i = int(vPosition.x);\n    float c = -1.0;\n    '
                     + short + '\n    Color = vec4(c);\n}\n',
                     inputs={'position': p}, n=4)
    check('array size: ...and a chain of 150 arms takes each lane to its '
          'own', np.array_equal(outs['Color'][:, 0], [0, 57, 149, -1]),
          str(outs['Color'][:, 0]))

    # nor do the constants a size names: one made of the one above it,
    # 3000 deep, is followed with a list
    prog, err = try_compile(
        'const int K0 = 1;\n' + ''.join(
            f'const int K{i} = K{i - 1} + 1;\n' for i in range(1, 3000))
        + _body('    Color = vec4(1.0);\n',
                'void sized() { float a[K2999 / 100]; }\n'))
    check('array size: a constant 3000 constants deep',
          prog is not None and _declared(prog)['a'] == 30, repr(err))

    # a shader that sizes its arrays with numbers computes nothing at
    # parse time, however many constants it declares: the constants are
    # recorded, and only a size that names one evaluates it
    from ..shaders import parser
    from ..shaders.compiler import DEFAULT_GLSL, DEFAULT_HLSL
    folds = []
    kept = parser.Parser.compute

    def counted(self, *args, **kwargs):
        folds.append(args[0])
        return kept(self, *args, **kwargs)

    numbered = _N3 + ('const float TAU = 6.2831853;\n'
                      'const vec3 K[2] = vec3[2](vec3(1.0), vec3(TAU));\n'
                      'uniform float w[4];\nstruct S { float f[2]; };\n'
                      'float sum(float v[3]) { return v[0] + v[2]; }\n'
                      ) + _body(
        '    float a[3];\n    const int M = N * 2;\n    a[2] = K[1].x;\n'
        '    Color = vec4(sum(a), float(M), w[3], 1.0);\n')
    parser.Parser.compute = counted
    try:
        quiet = []
        for src, lang in ((numbered, 'GLSL'), (DEFAULT_GLSL, 'GLSL'),
                          (DEFAULT_HLSL, 'HLSL')):
            quiet.append(try_compile(src + '\n// unfolded', lang)[1])
        none = len(folds)
        named = try_compile(numbered.replace('float a[3]', 'float a[N]')
                            .replace('float v[3]', 'float v[N]')
                            + '\n// folded')[1]
    finally:
        parser.Parser.compute = kept
    check('array size: nothing is computed for sizes written as numbers '
          '(and two sizes that name N compute three things: themselves, '
          'and N once)',
          quiet == [None] * 3 and none == 0 and named is None
          and len(folds) == 3, f'{quiet} {none} {named} {len(folds)}')

    # ...and in silence: what NumPy would warn of while a size is
    # computed is either no size at all (refused by name, above) or, as
    # here -- a float32 product past the range -- still the size it is
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        prog, err = try_compile(_body(
            '    float a[1e30 * 1e30 > 1.0 ? 3 : 2];\n' + use))
    check('array size: a size is computed without a NumPy warning',
          prog is not None and _declared(prog)['a'] == 3, repr(err))

    # the builtins a constant may call are the ones written down as such:
    # every one is a builtin, and what is left out is this list
    from ..shaders.builtins import BUILTINS, CONSTANT
    left_out = {'texture', 'texture2D', 'textureLod', 'texture2DLod',
                'textureGrad', 'texelFetch', 'textureSize', 'tex2D',
                'tex2Dlod', 'dFdx', 'dFdy', 'ddx', 'ddy', 'fwidth', 'clip',
                'noise', 'noise3', 'fbm', 'quantize', 'posterize',
                'dither4x4', 'hsv2rgb', 'rgb2hsv', 'luminance'}
    check('array size: the builtins a constant may call, and the ones it '
          'may not', CONSTANT <= set(BUILTINS)
          and set(BUILTINS) - CONSTANT == left_out,
          str(sorted((set(BUILTINS) - CONSTANT) ^ left_out)))


def _size_expr(rng, names, depth):
    """(GLSL text, value) of a seeded constant expression over the ints in
    `names` and small numbers -- every binary node in parentheses, the
    value computed here on Python ints the way the language defines it
    (the quotient cut toward zero). None for one the language leaves
    undefined: a zero divisor, or a remainder of anything negative."""
    kind = rng.randint(8) if depth > 0 else rng.randint(2)
    if kind == 0:
        v = int(rng.randint(1, 10))
        return str(v), v
    if kind == 1:
        name = sorted(names)[rng.randint(len(names))]
        return name, names[name]
    if kind == 2:
        sub = _size_expr(rng, names, depth - 1)
        return sub and (f'(-{sub[0]})', -sub[1])
    op = '+-*/%+-*'[rng.randint(8)]
    left = _size_expr(rng, names, depth - 1)
    right = _size_expr(rng, names, depth - 1)
    if left is None or right is None:
        return None
    (ta, a), (tb, b) = left, right
    if op == '/':
        if b == 0:
            return None
        v = abs(a) // abs(b) * (1 if (a < 0) == (b < 0) else -1)
    elif op == '%':
        if a < 0 or b <= 0:
            return None
        v = a % b
    else:
        v = {'+': a + b, '-': a - b, '*': a * b}[op]
    return f'({ta} {op} {tb})', v


def _size_case(rng):
    """One seeded shader: three constants, each made of the ones above it,
    and a size made of all three -- (source, the size, the constants)."""
    while True:
        names = {'A': int(rng.randint(1, 10))}
        lines = [f'const int A = {names["A"]};']
        for name in 'BC':
            made = _size_expr(rng, names, 2)
            if made is None or abs(made[1]) > 99:
                break
            names[name] = made[1]
            lines.append(f'const int {name} = {made[0]};')
        else:
            size = _size_expr(rng, names, 3)
            if size is not None and -40 <= size[1] <= 400:
                break
    text, value = size
    src = ('\n'.join(lines) + '\nout vec4 Color;\n'
           f'void sized() {{ float a[{text}]; }}\n'
           f'void main() {{\n    Color = vec4(float({text}), float(A), '
           'float(B), float(C));\n}\n')
    return src, value, names


def test_a_size_is_the_number_its_expression_runs_to():
    """The size in the brackets and the same expression in the body of the
    shader are one number: the parser computes a size with the
    interpreter's own arithmetic (interp.fold), and the two are held here
    against a third, plain Python on plain ints."""
    bad, sized, refused, n_prog = [], 0, 0, 300
    for seed in range(n_prog):
        src, value, names = _size_case(np.random.RandomState(4000 + seed))
        prog, err = try_compile(src)
        body = src.replace('void sized() {', '//')
        row = [value, names['A'], names['B'], names['C']]
        ran = [_first(body, n=n)[0] for n in (1, 4, 7)]
        if any(r is None or not np.array_equal(r, row) for r in ran):
            bad.append(f'seed {seed}: the body ran to {ran} for {row}')
        elif value < 1:
            refused += 1
            if prog is not None or err != ('line 5: array size must be at '
                                           f'least 1, not {value}'):
                bad.append(f'seed {seed}: {value} gave {err!r}')
        else:
            sized += 1
            if prog is None or _declared(prog)['a'] != value:
                bad.append(f'seed {seed}: {value} gave '
                           f'{err or _declared(prog)}')
    check(f'array size: {n_prog} seeded constant expressions -- the size '
          'declared, the value the body runs to (at 1, 4 and 7 lanes) and '
          f'plain Python agree ({sized} sizes, {refused} below 1 refused '
          'with their number)',
          not bad and sized > 150 and refused > 30,
          f'{len(bad)} differ: ' + '; '.join(bad[:3]))

    # a size is computed on more lanes than a vector or a matrix has
    # anything of, as a frame is. Two things this interpreter cannot do in
    # a frame -- a matrix divided by a scalar, HLSL's fmod of a vector and
    # a scalar -- go through on one or two lanes, where the shapes happen
    # to fit; computed on one lane the size would be a number the same
    # constant never runs to in a frame. So the size is refused -- or, the
    # day the interpreter can compute these, it is the number the body
    # runs to on seven lanes: the invariant, whichever holds
    from ..shaders.interp import FOLD_LANES
    cases = [
        ('a matrix divided by a scalar', 'GLSL',
         'const mat2 M = mat2(4.0) / 2.0;\n'
         'const int N = int(M[0][0]) + 1;\nout vec4 Color;\n',
         'void main() {\n    Color = vec4(float(N));\n}\n'),
        ('HLSL fmod of a vector and a scalar', 'HLSL',
         'static const float2 P = fmod(float2(7.0, 5.0), 4.0);\n'
         'static const int N = int(P.x) + 1;\nout float4 Color;\n',
         'void main() {\n    Color = float4(N, N, N, N);\n}\n'),
    ]
    for what, lang, head, main in cases:
        prog, err = try_compile(head + 'void sized() { float a[N]; }\n'
                                + main, lang)
        ran, err2 = _first(head + main, lang, n=FOLD_LANES)
        check(f'array size: {what} is no size, or the number a frame '
              'runs it to',
              FOLD_LANES > 4 and (
                  (prog is None and err.startswith(
                      'line 4: array size cannot be evaluated ('))
                  or (prog is not None and ran is not None
                      and _declared(prog)['a'] == ran[0])),
              str(err or err2 or (_declared(prog), ran)))



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
