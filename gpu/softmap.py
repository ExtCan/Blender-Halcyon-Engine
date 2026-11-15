"""GLSL for a SIZED lamp's shadow-map term: core/softmap.py, statement for
statement.

The CPU lookup (its module docstring is the algorithm) is emitted here with
every loop unrolled at build time: the tap tables are the float32 constants
`softmap.search_table` / `filter_table` hand BOTH roads, the lattice stride
is a comparison ladder, and the operations are written in the CPU's own
order so a tap lands on the same texel on either device.

What a fragment pays (F filter taps, B + 1 search taps; 16 and 13 at the
default Samples 4), in texel fetches:

    lit or umbra by the search          1 + (B + 1)               = 14
    decided by the probe                1 + 2 (B + 1)             = 27
    penumbra, one lattice               1 + 2 (B + 1) + 4 F       = 91
    penumbra, two lattices cross-faded  1 + 2 (B + 1) + 8 F       = 155

The three stages are real branches, so only penumbra fragments run the
filter; the simulator evaluates both sides of a branch and selects, which
gives the same value. The one extra fetch is the map's nearest depth: it
rides the G channel of the lamp's first atlas texel, so a moving caster
re-uploads the atlas (as it must) and never changes this text.
"""

import numpy as np

from ..core import softmap as SM


def _lit(x):
    """A float literal holding exactly the float32 the CPU computes with."""
    from .material import _f
    return _f(float(np.float32(x)))


def _vec3(v):
    return 'vec3({}, {}, {})'.format(*(_lit(c) for c in tuple(v)[:3]))


def _vec4(v):
    return 'vec4({}, {}, {}, {})'.format(*(_lit(c) for c in tuple(v)[:4]))


def shadow_function(i, meta):
    """`hal_shadow_vis{i}(P, N, L)` for a mapped lamp WITH a size (the
    meta carries 'soft': its shape and the map's axes)."""
    from .material import _f, _v3, _v4

    sp = meta['soft']
    shape = sp['shape']
    axes = sp['axes']
    faces = meta['faces']
    size = int(meta['size'])
    near, far, persp = meta['near'], meta['far'], bool(meta['persp'])
    grid_w = meta['grid'][0]
    bias = float(meta['bias'])
    density = float(meta['density'])
    origin = meta['origin']
    voff = int(meta.get('voff', 0))
    soft = max(float(meta['softness']), 0.0)
    ext = float(meta['extent'])
    cube = len(faces) > 1
    rect = shape['kind'] == 'RECT'
    F, B = SM.tap_counts(int(meta.get('samples', 1)))
    tab_s = SM.search_table(F, B, soft)
    tab_f = SM.filter_table(F, soft)
    nt, fn = _lit(len(tab_s)), _lit(len(tab_f))
    sq = np.float32(soft) * np.float32(soft)
    k = np.float32(size / (2.0 * ext))
    K = _lit(k)
    sz, sz1 = _lit(size), _lit(size - 1)
    fetch = 'texelFetch(hal_shadowpack, ivec2(int({}), int({})), 0).r'

    if persp:
        def plane(du, dv):
            return f'(1.0 / max(dr + gu * ({du}) + gv * ({dv}), dr))'
    else:
        def plane(du, dv):
            return f'min(dr + gu * ({du}) + gv * ({dv}), dr)'

    ctx = 'float u, float v, float dr, float gu, float gv, float sl, ' \
          'float cellx, float celly'
    cta = 'u, v, dr, gu, gv, sl, cellx, celly'
    L = ['uniform sampler2D hal_shadowpack;']
    if cube:
        # a depth texel of this face, continued past its edge onto the
        # neighbouring faces (`_CubeFetch`)
        ctx += ', vec3 sax, vec3 uax, vec3 fax'
        cta += ', sax, uax, fax'
        inv = _lit(np.float32(2.0 / size))
        L += [f'float hal_sm_occ{i}(float xi, float yi, float cellx, '
              'float celly, vec3 sax, vec3 uax, vec3 fax)',
              '{',
              '    float occ = 0.0;',
              f'    if (xi >= 0.0 && xi <= {sz1} && yi >= 0.0 && '
              f'yi <= {sz1}) {{',
              '        occ = ' + fetch.format('cellx + xi', 'celly + yi')
              + ';',
              '    } else {',
              f'        vec3 d = sax * ((xi + 0.5) * {inv} - 1.0) + '
              f'uax * ((yi + 0.5) * {inv} - 1.0) + fax;',
              '        vec3 a = abs(d);',
              '        int g = 0;',
              '        if (a.x >= a.y && a.x >= a.z) '
              '{ g = d.x >= 0.0 ? 0 : 1; }',
              '        else if (a.y >= a.z) { g = d.y >= 0.0 ? 2 : 3; }',
              '        else { g = d.z >= 0.0 ? 4 : 5; }',
              '        vec3 sg = vec3(1.0, 0.0, 0.0);',
              '        vec3 ug = vec3(0.0, 1.0, 0.0);',
              '        vec3 fg = vec3(0.0, 0.0, 1.0);',
              '        float gx = 0.0; float gy = 0.0;']
        for gi in range(len(faces)):
            s_g, u_g, f_g = axes[gi]
            L.append(f'        if (g == {gi}) {{ sg = {_vec3(s_g)}; '
                     f'ug = {_vec3(u_g)}; fg = {_vec3(f_g)}; '
                     f'gx = {_f((gi % grid_w) * size)}; '
                     f'gy = {_f((gi // grid_w) * size + voff)}; }}')
        L += ['        float md = d.x * fg.x + d.y * fg.y + d.z * fg.z;',
              '        float ds = d.x * sg.x + d.y * sg.y + d.z * sg.z;',
              '        float du = d.x * ug.x + d.y * ug.y + d.z * ug.z;',
              f'        float xg = clamp(floor((ds / md * 0.5 + 0.5) * '
              f'{sz}), 0.0, {sz1});',
              f'        float yg = clamp(floor((du / md * 0.5 + 0.5) * '
              f'{sz}), 0.0, {sz1});',
              '        occ = ' + fetch.format('gx + xg', 'gy + yg')
              + ' / md;',
              '    }',
              '    return occ;',
              '}']

        def texel(x, y):
            return [f'    float xi = {x};',
                    f'    float yi = {y};',
                    f'    float occ = hal_sm_occ{i}(xi, yi, cellx, celly, '
                    'sax, uax, fax);']
    else:
        def texel(x, y):
            return [f'    float xi = clamp({x}, 0.0, {sz1});',
                    f'    float yi = clamp({y}, 0.0, {sz1});',
                    '    float occ = '
                    + fetch.format('cellx + xi', 'celly + yi') + ';']

    # one search / probe tap: (blocked, blocked * depth)
    L += [f'vec2 hal_sm_blk{i}(float ox, float oy, {ctx})', '{'] + \
        texel('floor(u + ox)', 'floor(v + oy)') + \
        [f'    float pd = {plane("xi + 0.5 - u", "yi + 0.5 - v")};',
         '    float b = (occ < pd - sl) ? 1.0 : 0.0;',
         '    return vec2(b, b * occ);',
         '}']
    # one filter tap on the lattice of stride st: four plane-biased
    # compares, bilinear
    L += [f'float hal_sm_tap{i}(float px, float py, float st, {ctx})',
          '{',
          '    float gx0 = px / st - 0.5;',
          '    float gy0 = py / st - 0.5;',
          '    float x0 = floor(gx0);',
          '    float y0 = floor(gy0);',
          '    float fx = gx0 - x0;',
          '    float fy = gy0 - y0;',
          '    float hf = floor(st * 0.5);',
          '    float acc = 0.0;']
    for dy, wy in ((0.0, '(1.0 - fy)'), (1.0, 'fy')):
        for dx, wx in ((0.0, '(1.0 - fx)'), (1.0, 'fx')):
            L += ['    {'] + \
                texel(f'(x0 + {_f(dx)}) * st + hf',
                      f'(y0 + {_f(dy)}) * st + hf') + \
                [f'    float pd = {plane("xi + 0.5 - u", "yi + 0.5 - v")};',
                 f'    acc += ({wx} * {wy}) * '
                 '((pd - sl <= occ) ? 1.0 : 0.0);',
                 '    }']
    L += ['    return acc;', '}']

    L += [f'float hal_shadow_vis{i}(vec3 P, vec3 N, vec3 L)',
          '{',
          '    float ndl = clamp(dot(N, L), 0.0, 1.0);',
          f'    float sl = {_f(bias)} * (1.0 + 2.0 * (1.0 - ndl));',
          f'    float pdist = length(P - {_v3(origin)});']
    if persp:
        L.append(f'    float texel = 2.0 * {_f(ext)} * '
                 f'max(pdist, {_f(near)}) / {_f(size)};')
    else:
        L.append(f'    float texel = {_f(2.0 * ext / size)};')
    if meta.get('midpoint'):
        L.append('    float off_amt = 0.0;')
    else:
        # the normal offset, scaled down where the receiver plane serves
        # the compare exactly and whole only towards grazing light
        # (`softmap.offset_scale`); no blur factor, whatever the kernel
        L.append('    float off_amt = texel * (1.5 + 2.5 * '
                 'sqrt(max(1.0 - ndl * ndl, 0.0))) * '
                 f'({_lit(SM.OFFSET_BASE)} + {_lit(1.0 - SM.OFFSET_BASE)} * '
                 f'clamp(({_lit(SM.GRAZE_HI)} - ndl) / '
                 f'{_lit(SM.GRAZE_HI - SM.NDL_MIN)}, 0.0, 1.0));')
    L.append('    vec4 ph = vec4(P + N * off_amt, 1.0);')

    def face_consts(fi):
        s_ax, u_ax, f_ax = axes[fi]
        out = [f'sax = {_vec3(s_ax)};', f'uax = {_vec3(u_ax)};']
        if cube:
            out.append(f'fax = {_vec3(f_ax)};')
        if rect:
            e1, e2 = SM.rect_axes(shape, axes[fi])
            out += [f'e1 = {_vec3(e1)};', f'e2 = {_vec3(e2)};']
        return out

    L.append('    vec3 sax = vec3(1.0, 0.0, 0.0); '
             'vec3 uax = vec3(0.0, 1.0, 0.0);')
    if cube:
        L.append('    vec3 fax = vec3(0.0, 0.0, 1.0);')
    if rect:
        L.append('    vec3 e1 = vec3(0.0, 0.0, 0.0); '
                 'vec3 e2 = vec3(0.0, 0.0, 0.0);')
    if cube:
        L += ['    vec3 dvec = ph.xyz - ' + _v3(origin) + ';',
              '    vec3 avec = abs(dvec);',
              '    int face = 0;',
              '    if (avec.x >= avec.y && avec.x >= avec.z) '
              '{ face = dvec.x >= 0.0 ? 0 : 1; }',
              '    else if (avec.y >= avec.z) '
              '{ face = dvec.y >= 0.0 ? 2 : 3; }',
              '    else { face = dvec.z >= 0.0 ? 4 : 5; }',
              '    float clipx = 0.0; float clipy = 0.0;',
              '    float clipz = 0.0; float clipw = 1.0;',
              '    float cellx = 0.0; float celly = 0.0;']
        for fi, fc in enumerate(faces):
            vp = np.asarray(fc['vp'], np.float32)
            L += [f'    if (face == {fi}) {{',
                  f'        clipx = dot(ph, {_v4(vp[0])});',
                  f'        clipy = dot(ph, {_v4(vp[1])});',
                  f'        clipz = dot(ph, {_v4(vp[2])});',
                  f'        clipw = dot(ph, {_v4(vp[3])});',
                  f'        cellx = {_f((fi % grid_w) * size)};',
                  f'        celly = {_f((fi // grid_w) * size + voff)};'] + \
                 ['        ' + s for s in face_consts(fi)] + ['    }']
    else:
        vp = np.asarray(faces[0]['vp'], np.float32)
        L += [f'    float clipx = dot(ph, {_v4(vp[0])});',
              f'    float clipy = dot(ph, {_v4(vp[1])});',
              f'    float clipz = dot(ph, {_v4(vp[2])});',
              f'    float clipw = dot(ph, {_v4(vp[3])});',
              f'    float cellx = 0.0; float celly = {_f(voff)};'] + \
             ['    ' + s for s in face_consts(0)]
    eps = 1e-6
    L += [f'    float w = abs(clipw) < {_f(eps)} ? {_f(eps)} : clipw;',
          '    float nx = clipx / w;',
          '    float ny = clipy / w;',
          '    float nz = clipz / w;',
          '    bool inside = abs(nx) <= 1.0 && abs(ny) <= 1.0 && nz <= 1.0'
          + (' && clipw > 0.0;' if persp else ';'),
          '    float zc = clamp(nz, -1.0, 1.0);']
    if persp:
        L += [f'    float den = ({_f(far + near)}) - zc * {_f(far - near)};',
              f'    den = abs(den) < {_f(eps)} ? {_f(eps)} : den;',
              f'    float sdist = {_f(2.0 * near * far)} / den;']
    else:
        L.append(f'    float sdist = (zc * 0.5 + 0.5) * {_f(far - near)}'
                 f' + {_f(near)};')
    L += [f'    float u = (nx * 0.5 + 0.5) * {_f(size)};',
          f'    float v = (ny * 0.5 + 0.5) * {_f(size)};',
          '    float lit = 1.0;',
          '    if (inside) {',
          f'    float nd = max(ndl, {_lit(SM.NDL_MIN)});',
          '    float ns = N.x * sax.x + N.y * sax.y + N.z * sax.z;',
          '    float nu = N.x * uax.x + N.y * uax.y + N.z * uax.z;',
          # the map's nearest depth: the G channel of the lamp's first
          # atlas texel
          f'    float dmin = texelFetch(hal_shadowpack, ivec2(0, {voff}), '
          '0).g;']
    if persp:
        L += [f'    vec3 rel = ph.xyz - {_v3(origin)};',
              '    float deu = sqrt(rel.x * rel.x + rel.y * rel.y + '
              'rel.z * rel.z);',
              '    float cpl = -(deu * nd);',
              f'    float gu = ns / (cpl * {K});',
              f'    float gv = nu / (cpl * {K});',
              f'    float x = nx * {_lit(ext)};',
              f'    float y = ny * {_lit(ext)};',
              '    float dr = 1.0 / max(sdist, 1e-9);']
        if rect:
            L += [f'    float Ax = {K} * (e1.x - x * e1.z);',
                  f'    float Ay = {K} * (e1.y - y * e1.z);',
                  f'    float Bx = {K} * (e2.x - x * e2.z);',
                  f'    float By = {K} * (e2.y - y * e2.z);']
        else:
            kr = _lit(np.float32(float(k) * float(shape['R'])))
            L += ['    float s1 = 1.0 + sqrt(1.0 + x * x + y * y);',
                  f'    float Ax = {kr} * (1.0 + x * x / s1);',
                  f'    float Ay = {kr} * (x * y / s1);',
                  '    float Bx = Ay;',
                  f'    float By = {kr} * (1.0 + y * y / s1);']
        L.append('    float tmin = max(1.0 / max(dmin, 1e-9) - dr, 0.0);')
    else:
        kt = _lit(np.float32(float(k) * float(shape['tan'])))
        L += [f'    float gu = ns / (nd * {K});',
              f'    float gv = nu / (nd * {K});',
              '    float dr = sdist;',
              f'    float Ax = {kt}; float Ay = 0.0;',
              f'    float Bx = 0.0; float By = {kt};',
              '    float tmin = max(dr - dmin, 0.0);']
    grow = _lit(SM.search_grow(F))
    cap = _lit(np.float32(SM.CAP_FRAC * size))
    L += ['    float amax = max(sqrt(Ax * Ax + Ay * Ay), '
          'sqrt(Bx * Bx + By * By));',
          f'    float ts = min(tmin, {cap} / max(amax, 1e-12));']

    def search(t, acc_depth):
        out = [f'    float {t}g = {t} * {grow};']
        for a, b, pa, pb in tab_s:
            out += ['    {',
                    f'    vec2 r = hal_sm_blk{i}('
                    f'{t}g * ({_lit(a)} * Ax + {_lit(b)} * Bx) + {_lit(pa)}, '
                    f'{t}g * ({_lit(a)} * Ay + {_lit(b)} * By) + {_lit(pb)}, '
                    f'{cta});',
                    acc_depth,
                    '    }']
        return out

    def filt(acc, st):
        out = []
        for a, b, sa, sb in tab_f:
            out += ['    {',
                    f'    float px = u + fq * (tf * ({_lit(a)} * Ax + '
                    f'{_lit(b)} * Bx) + {_lit(sa)});',
                    f'    float py = v + fq * (tf * ({_lit(a)} * Ay + '
                    f'{_lit(b)} * By) + {_lit(sb)});',
                    f'    {acc} += hal_sm_tap{i}(px, py, {st}, {cta});',
                    '    }']
        return out

    # the blocker search; then, only where it is undecided, the probe;
    # then, only where THAT is undecided, the filter
    L += ['    float nb = 0.0; float dsum = 0.0;'] + \
        search('ts', '    nb += r.x; dsum += r.y;') + \
        [f'    if (nb >= {nt}) {{ lit = 0.0; }}',
         '    else if (nb > 0.0) {',
         '    float davg = dsum / max(nb, 1.0);']
    if persp:
        L.append('    float tp = min(max(1.0 / max(davg, 1e-9) - dr, 0.0), '
                 'ts);')
    else:
        L.append('    float tp = min(max(dr - davg, 0.0), ts);')
    L += ['    float nb2 = 0.0; float dsum2 = 0.0;'] + \
        search('tp', '    nb2 += r.x; dsum2 += r.y;') + \
        [f'    if (nb2 >= {nt}) {{ lit = 0.0; }}',
         '    else if (nb2 > 0.0) {',
         # the estimate again, from the casters the probe itself found
         '    float davg2 = dsum2 / max(nb2, 1.0);',
         ('    float tf = min(max(1.0 / max(davg2, 1e-9) - dr, 0.0), tp);'
          if persp else
          '    float tf = min(max(dr - davg2, 0.0), tp);'),
         # the lamp's penumbra and the Softness add in quadrature
         '    float rl = tf * amax;',
         f'    float rtex = sqrt(rl * rl + {_lit(sq)});',
         f'    float fq = rtex / max(rl + {_lit(soft)}, 1e-12);',
         f'    float q = max(2.0 * rtex / {fn}, 1.0);',
         '    float st0 = 1.0;']
    for lv in SM.STRIDES:
        L.append(f'    if (q >= {_f(lv)}) {{ st0 = {_f(lv)}; }}')
    L += [f'    float wb = clamp((q / st0 - {_lit(SM.SQRT2)}) / '
          f'{_lit(SM.SQRT2_SPAN)}, 0.0, 1.0);',
          '    float acc0 = 0.0;'] + filt('acc0', 'st0') + \
        [f'    lit = acc0 / {fn};',
         '    if (wb > 0.0) {',
         '    float st1 = st0 * 2.0;',
         '    float acc1 = 0.0;'] + filt('acc1', 'st1') + \
        [f'    lit = lit * (1.0 - wb) + (acc1 / {fn}) * wb;',
         '    }',
         '    }',
         '    }',
         '    }',
         f'    return 1.0 - (1.0 - lit) * {_f(density)};',
         '}']
    return '\n'.join(L) + '\n'
