"""R251 (1.90.0) material pack, MAT-A: the period combiners' GLSL twins.

`combine_fns(model, consts)` returns the global function text a
vertex-rate pass needs for `model` (appended to the pass's `vlight_fns`,
before `void main()`), and `recombine_lines(model, consts, bake)` the
statements that replace `vec3 total = s.diffuse * hal_vl;` in the pass's
main body -- exactly that one line, verbatim, for every pre-1.90 model,
so the shipped Gouraud / FLAT shader text is unchanged to the byte.

Every function is core/combine.py's arithmetic written in the accepted
GLSL subset: floats with floor() for the integer steps (no int / int),
roundEven for np.rint, one op per statement wherever a rounding follows
(no FMA across a quantisation boundary), power-of-two divides, the
float32 reciprocal literals for the final scale, texelFetch only. The
simulator evaluates these bitwise the CPU; a driver's FMA contraction on
the corner interpolation is the shipped Gouraud seam's own class.
"""
from ..core import combine as CB

R255 = CB.R255_LIT
R31 = CB.R31_LIT
R63 = CB.R63_LIT


def _f(x):
    """material._f's twin, written here so this module imports nothing
    from gpu/material.py at load (material imports this module): a float
    literal a strict GLSL front-end cannot mistake for an int (%.9g, the
    float32 round-trip)."""
    s = f'{float(x):.9g}'
    if 'e' not in s and 'E' not in s and '.' not in s and 'inf' not in s             and 'nan' not in s:
        s += '.0'
    return s


# ------------------------------------------------------- the function texts

FN_LUM = (
    'float hal_lum(vec3 c)\n'
    '{\n'
    '    float l = 0.299 * c.r;\n'
    '    l = l + 0.587 * c.g;\n'
    '    l = l + 0.114 * c.b;\n'
    '    return l;\n'
    '}\n')

FN_PS1 = (
    'vec3 hal_cb_ps1(vec3 L, vec3 A)\n'
    '{\n'
    '    vec3 t8 = roundEven(clamp(A, 0.0, 1.0) * 255.0);\n'
    '    vec3 s8 = clamp(roundEven(L * 128.0), 0.0, 255.0);\n'
    '    vec3 v = t8 * s8;\n'
    '    vec3 o = floor(v / 128.0);\n'
    '    o = min(o, vec3(255.0));\n'
    f'    return o * {R255};\n'
    '}\n')

FN_SATURN = (
    'vec3 hal_cb_saturn(vec3 L, vec3 A)\n'
    '{\n'
    '    vec3 t5 = roundEven(clamp(A, 0.0, 1.0) * 31.0);\n'
    '    vec3 g5 = clamp(roundEven(L * 16.0), 0.0, 31.0);\n'
    '    vec3 o = t5 + g5;\n'
    '    o = o - 16.0;\n'
    '    o = clamp(o, 0.0, 31.0);\n'
    f'    return o * {R31};\n'
    '}\n')

FN_N64 = (
    'vec3 hal_cb_n64(vec3 L, vec3 A)\n'
    '{\n'
    '    vec3 t8 = roundEven(clamp(A, 0.0, 1.0) * 255.0);\n'
    '    vec3 s8 = clamp(roundEven(L * 255.0), 0.0, 255.0);\n'
    '    vec3 v = t8 * s8;\n'
    '    v = v + 128.0;\n'
    '    vec3 o = floor(v / 256.0);\n'
    f'    return o * {R255};\n'
    '}\n')

FN_DS = (
    'vec3 hal_cb_ds(vec3 L, vec3 A)\n'
    '{\n'
    '    vec3 m = clamp(A, 0.0, 1.0) * 63.0;\n'
    '    m = m + 0.5;\n'
    '    vec3 t6 = floor(m);\n'
    '    m = clamp(L, 0.0, 1.0) * 63.0;\n'
    '    m = m + 0.5;\n'
    '    vec3 v6 = floor(m);\n'
    '    vec3 v = (t6 + 1.0) * (v6 + 1.0);\n'
    '    v = v - 1.0;\n'
    '    vec3 o = floor(v / 64.0);\n'
    f'    return o * {R63};\n'
    '}\n')

FN_S22 = (
    'vec3 hal_cb_s22(vec3 L, vec3 A)\n'
    '{\n'
    '    vec3 t8 = roundEven(clamp(A, 0.0, 1.0) * 255.0);\n'
    '    vec3 s8 = clamp(roundEven(L * 64.0), 0.0, 255.0);\n'
    '    vec3 v = t8 * s8;\n'
    '    vec3 o = floor(v / 64.0);\n'
    '    o = min(o, vec3(255.0));\n'
    f'    return o * {R255};\n'
    '}\n')

FN_PS2HL = (
    'vec3 hal_cb_ps2hl(vec4 L, vec3 A)\n'
    '{\n'
    '    vec3 t8 = roundEven(clamp(A, 0.0, 1.0) * 255.0);\n'
    '    vec3 s8 = clamp(roundEven(L.rgb * 128.0), 0.0, 255.0);\n'
    '    vec3 v = t8 * s8;\n'
    '    vec3 m8 = floor(v / 128.0);\n'
    '    float a8 = clamp(roundEven(L.a * 128.0), 0.0, 255.0);\n'
    '    vec3 o = m8 + vec3(a8);\n'
    '    o = min(o, vec3(255.0));\n'
    f'    return o * {R255};\n'
    '}\n')

FN_D3DSPEC = (
    'vec3 hal_cb_d3dspec(vec4 L, vec3 A)\n'
    '{\n'
    '    vec3 p = A * L.rgb;\n'
    '    vec3 o = p + vec3(L.a);\n'
    '    return min(o, vec3(1.0));\n'
    '}\n')

FN_PCX = FN_LUM + (
    'vec3 hal_cb_pcx(vec3 c0, vec3 c1, vec3 c2, vec3 b, vec3 A)\n'
    '{\n'
    '    vec3 s = c0 + c1;\n'
    '    s = s + c2;\n'
    '    vec3 base = s * 0.333333343;\n'
    '    base = clamp(base, 0.0, 1.0);\n'
    '    float lb = max(hal_lum(base), 1e-4);\n'
    '    float i0 = clamp(hal_lum(c0) / lb, 0.0, 1.0);\n'
    '    float i1 = clamp(hal_lum(c1) / lb, 0.0, 1.0);\n'
    '    float i2 = clamp(hal_lum(c2) / lb, 0.0, 1.0);\n'
    '    float I = i0 * b.x;\n'
    '    I = I + i1 * b.y;\n'
    '    I = I + i2 * b.z;\n'
    '    vec3 p = A * base;\n'
    '    return p * I;\n'
    '}\n')

FN_LUMA64 = FN_LUM + (
    'vec3 hal_cb_luma64(vec3 L, vec3 A)\n'
    '{\n'
    '    float lum = hal_lum(L);\n'
    '    float l8 = floor(clamp(lum, 0.0, 1.0) * 255.0);\n'
    '    float lum6 = min(floor(l8 / 4.0), 63.0);\n'
    '    vec3 ca = clamp(A, 0.0, 1.0) * 31.0;\n'
    '    vec3 c5 = floor(ca + 0.5);\n'
    '    vec3 num = c5 * lum6;\n'
    '    num = num * 255.0;\n'
    '    vec3 q = num + 0.5;\n'
    '    vec3 o8 = floor(q / 1953.0);\n'
    f'    return o8 * {R255};\n'
    '}\n')

FN_DSTOON = (
    'vec3 hal_cb_dstoon(float cs, vec3 A, float hl)\n'
    '{\n'
    '    vec3 ta = clamp(A, 0.0, 1.0) * 63.0;\n'
    '    ta = ta + 0.5;\n'
    '    vec3 t6 = floor(ta);\n'
    '    vec3 v = (t6 + 1.0) * (cs + 1.0);\n'
    '    v = v - 1.0;\n'
    '    vec3 c6 = floor(v / 64.0);\n'
    '    vec3 c6h = min(c6 + vec3(cs), 63.0);\n'
    '    c6 = (hl > 0.5) ? c6h : c6;\n'
    f'    return c6 * {R63};\n'
    '}\n')


def _ramp_terms(ramp):
    return ' + '.join(f'((abs(c - {_f(k)}) < 0.5) ? {_f(v)} : 0.0)'
                      for k, v in enumerate(ramp))


FN_SHRAMP = (
    'float hal_sh_ramp(float cls, float c)\n'
    '{\n'
    '    float r = 0.0;\n'
    '    if (cls < 0.5) {\n'
    f'        r = {_ramp_terms(CB.RAMP_NORMAL)};\n'
    '    } else if (cls < 1.5) {\n'
    f'        r = {_ramp_terms(CB.RAMP_SHADOW)};\n'
    '    } else {\n'
    f'        r = {_ramp_terms(CB.RAMP_HIGHLIGHT)};\n'
    '    }\n'
    '    return r;\n'
    '}\n')

FN_VL4_TMPL = (
    'vec4 hal_vlight_fetch4(float i)\n'
    '{\n'
    '    int vi = int(i);\n'
    '    return texelFetch(hal_vlight, ivec2(vi % {vside}, vi / {vside}), '
    '0).rgba;\n'
    '}\n')

_FNS = {'PS1_MODULATE': FN_PS1, 'SATURN_ADD': FN_SATURN,
        'N64_COMBINE': FN_N64, 'S22_MODULATE': FN_S22,
        'PS2_HIGHLIGHT': FN_PS2HL, 'D3D_SEPARATE_SPEC': FN_D3DSPEC,
        'PCX_INTENSITY': FN_PCX, 'DS_TOON': FN_DSTOON,
        'DS_HIGHLIGHT': FN_DSTOON, 'MEGA_DRIVE_SH': FN_SHRAMP,
        'SUPERFX_PLOT': ''}


#: the models whose lines read the corner's ALPHA (the carried channel)
#: through hal_vlight_fetch4 -- the function is appended for them only,
#: so every pre-1.90 vertex-rate pass keeps its text to the byte
NEEDS_FETCH4 = frozenset({'PS2_HIGHLIGHT', 'D3D_SEPARATE_SPEC', 'DS_TOON',
                          'DS_HIGHLIGHT', 'MEGA_DRIVE_SH'})


def combine_fns(model, consts, vside):
    """The global GLSL a vertex-rate pass for `model` needs ('' for every
    pre-1.90 model): hal_vlight_fetch4 for the carrying models, then the
    combine's function."""
    kind = CB.LIGHTING_COMBINE.get(model)
    if kind == 'luma64':
        return FN_LUMA64
    if kind == 'ds':
        return FN_DS
    fns = ''
    if model in NEEDS_FETCH4:
        fns += FN_VL4_TMPL.replace('{vside}', str(int(vside)))
    return fns + _FNS.get(model, '')


# -------------------------------------------------------- the main-body lines

_PLAIN = ['    vec3 total = s.diffuse * hal_vl;']
_VL4 = ['    vec4 hal_vl4 = hal_vlight_fetch4(hal_vt) * f.bary.x',
        '        + hal_vlight_fetch4(hal_vt + 1.0) * f.bary.y',
        '        + hal_vlight_fetch4(hal_vt + 2.0) * f.bary.z;']
_CORNERS = ['    vec3 hal_c0 = hal_vlight_fetch(hal_vt);',
            '    vec3 hal_c1 = hal_vlight_fetch(hal_vt + 1.0);',
            '    vec3 hal_c2 = hal_vlight_fetch(hal_vt + 2.0);']


def recombine_lines(model, consts, bake):
    """The statements that produce `vec3 total` in a vertex-rate pass:
    `_PLAIN` verbatim for every pre-1.90 model, the machine's combine for
    a period model. Raises core.combine.Refusal for a DS toon table or a
    Super FX plot that cannot run (the probe already gated: this is the
    belt to its braces)."""
    from .material import _mv
    kind = CB.LIGHTING_COMBINE.get(model)
    if kind == 'luma64':
        return ['    vec3 total = hal_cb_luma64(hal_vl, s.diffuse);']
    if kind == 'ds':
        return ['    vec3 total = hal_cb_ds(hal_vl, s.diffuse);']
    if model == 'PS1_MODULATE':
        return ['    vec3 total = hal_cb_ps1(hal_vl, s.diffuse);']
    if model == 'SATURN_ADD':
        return ['    vec3 total = hal_cb_saturn(hal_vl, s.diffuse);']
    if model == 'N64_COMBINE':
        return ['    vec3 total = hal_cb_n64(hal_vl, s.diffuse);']
    if model == 'S22_MODULATE':
        return ['    vec3 total = hal_cb_s22(hal_vl, s.diffuse);']
    if model == 'PS2_HIGHLIGHT':
        return _VL4 + ['    vec3 total = hal_cb_ps2hl(hal_vl4, s.diffuse);']
    if model == 'D3D_SEPARATE_SPEC':
        return _VL4 + ['    vec3 total = hal_cb_d3dspec(hal_vl4, s.diffuse);']
    if model == 'PCX_INTENSITY':
        return _CORNERS + ['    vec3 total = hal_cb_pcx(hal_c0, hal_c1, '
                           'hal_c2, f.bary, s.diffuse);']
    if model in ('DS_TOON', 'DS_HIGHLIGHT'):
        tab = bake.get('__ds_table') if bake else None
        if tab is None:
            raise CB.Refusal('Toon Size linked: the DS table is one per '
                             'material')
        lines = list(_VL4)
        lines += [f'    float hal_dst{k} = {_mv(consts, float(v))};'
                  for k, v in enumerate(tab)]
        lines += ['    float hal_red = hal_vl4.a;',
                  '    float hal_ir = hal_red * 31.0;',
                  '    hal_ir = hal_ir + 0.5;',
                  '    float hal_idx = clamp(floor(hal_ir), 0.0, 31.0);',
                  '    float hal_cs = 0.0;']
        lines += [f'    hal_cs += (abs(hal_idx - {_f(k)}) < 0.5) ? '
                  f'hal_dst{k} : 0.0;' for k in range(32)]
        hl = '1.0' if model == 'DS_HIGHLIGHT' else '0.0'
        lines.append(f'    vec3 total = hal_cb_dstoon(hal_cs, s.diffuse, {hl});')
        return lines
    if model == 'MEGA_DRIVE_SH':
        return _VL4 + [
            '    float hal_cls = floor(hal_vl4.a + 0.5);',
            '    vec3 hal_c3 = roundEven(clamp(s.diffuse, 0.0, 1.0) * 7.0);',
            '    vec3 total = vec3(hal_sh_ramp(hal_cls, hal_c3.r), '
            'hal_sh_ramp(hal_cls, hal_c3.g), hal_sh_ramp(hal_cls, hal_c3.b))'
            f' * {R255};']
    if model == 'SUPERFX_PLOT':
        w, h = consts.get('resolution', (1.0, 1.0))
        ss = int(consts.get('plot_ss', 1) or 1)
        return _CORNERS + [
            f'    float hal_px = floor(vUV.x * {_f(w)});',
            '    hal_px = hal_px + 0.5;',
            f'    float hal_xo = floor(hal_px / {_f(ss)});',
            f'    float hal_py = floor(vUV.y * {_f(h)});',
            '    hal_py = hal_py + 0.5;',
            f'    float hal_yo = floor(hal_py / {_f(ss)});',
            '    float hal_ps = hal_xo + hal_yo;',
            '    float hal_par = hal_ps - 2.0 * floor(hal_ps / 2.0);',
            '    vec3 total = (hal_par > 0.5) ? hal_c1 : hal_c0;']
    return list(_PLAIN)
