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


def main():
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
