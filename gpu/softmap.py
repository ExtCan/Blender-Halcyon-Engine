"""GLSL for a SIZED lamp's shadow-map term: core/softmap.py, statement for
statement.

The CPU lookup (its module docstring is the algorithm) is emitted here with
every loop unrolled at build time: the tap tables are the float32 constants
`softmap.search_table` / `filter_table` hand BOTH roads, the lattice stride
is a comparison ladder, and the operations are written in the CPU's own
order so a tap lands on the same texel on either device.

What a fragment pays (F filter taps, B + 1 search taps; 16 and 13 at the
default Samples 4), in texel fetches of the depth pack:

    lit: nothing nearer within reach    k   (the reach pyramid: 1 to
                                            SEARCH_STEPS, 2 to 3 mostly)
    lit or umbra by the search          k + (B + 1)               = 16
    decided by the probe                k + 2 (B + 1)             = 29
    penumbra, one lattice               k + 2 (B + 1) + 4 F       = 93
    penumbra, two lattices cross-faded  k + 2 (B + 1) + 8 F       = 157
    (the sums at k = 3; a strip lamp's footprint can cross-fade a
    second pair of lattices, finer across its short side: 16 F at most)

The stages are real branches, so only penumbra fragments run the filter
and a fragment the pyramid shows no nearer depth within reach of runs no
search at all; the simulator evaluates both sides of a branch and
selects, which gives the same value. The pyramid is rows of the depth
pack under the lamp's own cells (`softmap.pyr_layout`; gpu/shade's
`_shadow_meta` builds them, `_shadow_pack` stacks them). A lamp of size
0 under the user's Softness comes here too (`softmap.zero_shape`): its
footprint has no lamp term, the probe would read the search's own taps,
and is skipped (amax > 0).

WHAT IS A LITERAL AND WHAT IS A TEXEL. With `mb` -- the lamp's first tail
texel in hal_lights (`material.light_tail_layout`, `material.map_texels`)
-- every VALUE is fetched: the light-space rows, the origin, the clip
planes, Bias, Softness, Density, the lamp's size (k R, k tan, an Area's
half-axes per face), the map's axes and the radius cap; the nearest
depths that bound the search are texels of the pack itself (the reach
pyramid). A moved lamp, a dragged size or Softness and a moved caster then
re-upload texels and leave this text alone. What stays in the text is
structure: the map's size and atlas cells, the projection, the face count,
the tap count and the unit tap pattern. Without `mb` the same float32
values are printed as literals (a direct call, or the plan's literal road).
"""

import numpy as np

from ..core import softmap as SM


def _lit(x):
    """A float literal holding exactly the float32 the CPU computes with."""
    from .material import _f
    return _f(float(np.float32(x)))


def _vec3(v):
    return 'vec3({}, {}, {})'.format(*(_lit(c) for c in tuple(v)[:3]))


def shadow_function(i, meta, mb=None):
    """`hal_shadow_vis{i}(P, N, L)` for a mapped lamp that takes the
    per-receiver lookup (the meta carries 'soft': its shape and the
    map's axes). `mb` is the lamp's tail block in
    hal_lights, or None to print the values as literals."""
    from .material import MAP_FACE, MAP_HEAD, _f, _v3, _v4

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
    sz, sz1 = _lit(size), _lit(size - 1)
    fetch = 'texelFetch(hal_shadowpack, ivec2(int({}), int({})), 0).r'
    tx = mb is not None
    # the reach pyramid: one block per face, side by side under the
    # lamp's depth cells
    lay = SM.pyr_layout(size)
    pyr_y = voff + size * int(meta['grid'][1])

    def fb(fi):
        return mb + MAP_HEAD + fi * MAP_FACE

    if tx:
        # the header texels (`material.map_texels`), fetched once
        head = [f'    vec4 hal_m0 = hal_ltex({mb});',
                f'    vec4 hal_m1 = hal_ltex({mb + 1});',
                f'    vec4 hal_m2 = hal_ltex({mb + 2});',
                f'    vec4 hal_m3 = hal_ltex({mb + 3});']
        x_bias, x_origin = 'hal_m0.w', 'hal_m0.xyz'
        x_near, x_ext, x_texel = 'hal_m1.x', 'hal_m1.y', 'hal_m1.y'
        x_fpn, x_fmn, x_tnf = 'hal_m1.z', 'hal_m1.w', 'hal_m2.x'
        x_sv, x_dens = 'hal_m2.z', 'hal_m2.w'
        K, x_kr, x_cap = 'hal_m3.y', 'hal_m3.z', 'hal_m3.w'
        x_extl = 'hal_m1.y'

        def x_row(fi, r):
            return f'hal_ltex({fb(fi) + r})'

        def x_axis(fi, a):
            return f'hal_ltex({fb(fi) + 4 + a}).xyz'

        def x_rect(fi, a):
            return f'hal_ltex({fb(fi) + 7 + a}).xyz'
    else:
        head = []
        x_bias, x_origin = _f(bias), _v3(origin)
        x_near, x_ext = _f(near), _f(ext)
        x_texel = _f(2.0 * ext / size)
        x_fpn, x_fmn, x_tnf = (_f(far + near), _f(far - near),
                               _f(2.0 * near * far))
        x_sv, x_dens = _lit(soft), _f(density)
        K = _lit(k)
        if shape['kind'] == 'SPHERE':
            x_kr = _lit(np.float32(float(k) * float(shape['R'])))
        elif shape['kind'] == 'SUN':
            x_kr = _lit(np.float32(float(k) * float(shape['tan'])))
        else:
            x_kr = '0.0'
        x_cap = _lit(SM.cap_texels(size, ext, persp))
        x_extl = _lit(ext)

        def x_row(fi, r):
            return _v4(np.asarray(faces[fi]['vp'], np.float32)[r])

        def x_axis(fi, a):
            return _vec3(axes[fi][a])

        def x_rect(fi, a):
            return _vec3(SM.rect_axes(shape, axes[fi])[a])

    if persp:
        def plane(du, dv):
            return f'(1.0 / max(dr + gu * ({du}) + gv * ({dv}), dr))'
    else:
        def plane(du, dv):
            return f'min(dr + gu * ({du}) + gv * ({dv}), dr)'

    ctx = 'float u, float v, float dr, float gu, float gv, float sl, ' \
          'float cellx, float celly'
    cta = 'u, v, dr, gu, gv, sl, cellx, celly'
    # (no `uniform sampler2D hal_shadowpack;` here: gpu/material.assemble_frame
    # declares the one atlas sampler ONCE per pass, ahead of every mapped
    # lamp's function -- SHADOWPACK_DECL. A copy opening each sized lamp's
    # text made the raw pass declare it two or three times, the
    # redefinition a strict front-end refuses)
    L = []
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
            L.append(f'        if (g == {gi}) {{ sg = {x_axis(gi, 0)}; '
                     f'ug = {x_axis(gi, 1)}; fg = {x_axis(gi, 2)}; '
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

    # the nearest depth within r texels of (u, v): one cell of the reach
    # pyramid, at the level whose tile is the first at least r wide
    # (`softmap._reach_min`; a tile's width is a power of two, so the
    # division is exact on both roads)
    t0, n0, ox0, oy0 = lay['levels'][0]
    L += [f'float hal_sm_near{i}(float u, float v, float r, float pbx'
          + (', float dlow)' if cube else ')'),
          '{',
          f'    float T = {_f(float(t0))}; float ox = {_f(float(ox0))}; '
          f'float oy = {_f(float(oy0))}; float n1 = {_f(float(n0 - 1))};']
    for (tp_, _np, _ox, _oy), (t1, n1_, ox1, oy1) in zip(
            lay['levels'], lay['levels'][1:]):
        L.append(f'    if (r > {_f(float(tp_))}) {{ T = {_f(float(t1))}; '
                 f'ox = {_f(float(ox1))}; oy = {_f(float(oy1))}; '
                 f'n1 = {_f(float(n1_ - 1))}; }}')
    L += ['    float x = clamp(floor((u + 1.0) / T), 0.0, n1);',
          '    float y = clamp(floor((v + 1.0) / T), 0.0, n1);',
          '    float q = ' + fetch.format('pbx + ox + x',
                                          f'{_f(float(pyr_y))} + oy + y')
          + ';']
    if cube:
        # the whole cube's nearest depth bounds a tap too, divided by
        # no more than the tap's distance from the face's centre in
        # half-faces (`softmap._face`)
        half = _lit(np.float32(size * 0.5))
        L += [f'    float fo = max(abs(u - {half}) + r, '
              f'abs(v - {half}) + r) / {half};',
              '    q = max(q, dlow / max(fo, 1.0));']
    L += ['    return q;', '}']
    # one search / probe tap: (blocked, blocked * depth)
    L += [f'vec2 hal_sm_blk{i}(float ox, float oy, {ctx})', '{'] + \
        texel('floor(u + ox)', 'floor(v + oy)') + \
        [f'    float pd = {plane("xi + 0.5 - u", "yi + 0.5 - v")};',
         '    float b = (occ < pd - sl) ? 1.0 : 0.0;',
         '    return vec2(b, b * occ);',
         '}']
    # one filter tap on the lattice of strides (sx, sy) -- one stride
    # for a round footprint, a finer one across a strip's short side:
    # four plane-biased compares, bilinear
    L += [f'float hal_sm_tap{i}(float px, float py, float sx, float sy, '
          f'{ctx})',
          '{',
          '    float gx0 = px / sx - 0.5;',
          '    float gy0 = py / sy - 0.5;',
          '    float x0 = floor(gx0);',
          '    float y0 = floor(gy0);',
          '    float fx = gx0 - x0;',
          '    float fy = gy0 - y0;',
          '    float hx = floor(sx * 0.5);',
          '    float hy = floor(sy * 0.5);',
          '    float acc = 0.0;']
    for dy, wy in ((0.0, '(1.0 - fy)'), (1.0, 'fy')):
        for dx, wx in ((0.0, '(1.0 - fx)'), (1.0, 'fx')):
            L += ['    {'] + \
                texel(f'(x0 + {_f(dx)}) * sx + hx',
                      f'(y0 + {_f(dy)}) * sy + hy') + \
                [f'    float pd = {plane("xi + 0.5 - u", "yi + 0.5 - v")};',
                 f'    acc += ({wx} * {wy}) * '
                 '((pd - sl <= occ) ? 1.0 : 0.0);',
                 '    }']
    L += ['    return acc;', '}']

    grow = _lit(SM.search_grow(F))
    lamp = 'float Ax, float Ay, float Bx, float By'
    # the search pattern at one footprint t: (blocked taps, the sum of
    # their depths). Written once; the search and the probe both call it
    L += [f'vec2 hal_sm_find{i}(float t, {lamp}, {ctx})',
          '{',
          f'    float tg = t * {grow};',
          '    float nb = 0.0; float ds = 0.0;']
    if tx:
        # the pad (Softness + SEARCH_PAD) is a texel: each tap's share
        # is the float32 product `search_table` takes, in a statement
        # of its own ahead of the add
        L.append(f'    float pad = hal_ltex({mb + 3}).x;')
    for a, b, pa, pb in tab_s:
        if tx:
            L += ['    {',
                  f'    float pa = pad * {_lit(a)};',
                  f'    float pb = pad * {_lit(b)};',
                  f'    vec2 r = hal_sm_blk{i}('
                  f'tg * ({_lit(a)} * Ax + {_lit(b)} * Bx) + pa, '
                  f'tg * ({_lit(a)} * Ay + {_lit(b)} * By) + pb, '
                  f'{cta});',
                  '    nb += r.x; ds += r.y;',
                  '    }']
        else:
            L += ['    {',
                  f'    vec2 r = hal_sm_blk{i}('
                  f'tg * ({_lit(a)} * Ax + {_lit(b)} * Bx) + {_lit(pa)}, '
                  f'tg * ({_lit(a)} * Ay + {_lit(b)} * By) + {_lit(pb)}, '
                  f'{cta});',
                  '    nb += r.x; ds += r.y;',
                  '    }']
    L += ['    return vec2(nb, ds);', '}']
    # the F filter taps on one lattice. Written once; called for one
    # lattice, for the two that cross-fade, and for a strip's finer ones
    L += [f'float hal_sm_flt{i}(float tf, float fq, float sx, float sy, '
          f'{lamp}, {ctx})',
          '{',
          '    float acc = 0.0;']
    if tx:
        # the Softness is a texel: each tap's share is `filter_table`'s
        # float32 product
        L.append(f'    float sv = hal_ltex({mb + 2}).z;')
    for a, b, sa, sb in tab_f:
        if tx:
            L += ['    {',
                  f'    float sa = sv * {_lit(a)};',
                  f'    float sb = sv * {_lit(b)};',
                  f'    float px = u + fq * (tf * ({_lit(a)} * Ax + '
                  f'{_lit(b)} * Bx) + sa);',
                  f'    float py = v + fq * (tf * ({_lit(a)} * Ay + '
                  f'{_lit(b)} * By) + sb);',
                  f'    acc += hal_sm_tap{i}(px, py, sx, sy, {cta});',
                  '    }']
        else:
            L += ['    {',
                  f'    float px = u + fq * (tf * ({_lit(a)} * Ax + '
                  f'{_lit(b)} * Bx) + {_lit(sa)});',
                  f'    float py = v + fq * (tf * ({_lit(a)} * Ay + '
                  f'{_lit(b)} * By) + {_lit(sb)});',
                  f'    acc += hal_sm_tap{i}(px, py, sx, sy, {cta});',
                  '    }']
    L += ['    return acc;', '}']

    L += [f'float hal_shadow_vis{i}(vec3 P, vec3 N, vec3 L)',
          '{'] + head + [
          '    float ndl = clamp(dot(N, L), 0.0, 1.0);',
          f'    float sl = {x_bias} * (1.0 + 2.0 * (1.0 - ndl));',
          f'    float pdist = length(P - {x_origin});']
    if persp:
        L.append(f'    float texel = 2.0 * {x_ext} * '
                 f'max(pdist, {x_near}) / {_f(size)};')
    else:
        L.append(f'    float texel = {x_texel};')
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
        out = [f'sax = {x_axis(fi, 0)};', f'uax = {x_axis(fi, 1)};']
        if cube:
            out.append(f'fax = {x_axis(fi, 2)};')
        if rect:
            out += [f'e1 = {x_rect(fi, 0)};', f'e2 = {x_rect(fi, 1)};']
        return out

    L.append('    vec3 sax = vec3(1.0, 0.0, 0.0); '
             'vec3 uax = vec3(0.0, 1.0, 0.0);')
    if cube:
        L.append('    vec3 fax = vec3(0.0, 0.0, 1.0);')
    if rect:
        L.append('    vec3 e1 = vec3(0.0, 0.0, 0.0); '
                 'vec3 e2 = vec3(0.0, 0.0, 0.0);')
    if cube:
        L += ['    vec3 dvec = ph.xyz - ' + x_origin + ';',
              '    vec3 avec = abs(dvec);',
              '    int face = 0;',
              '    if (avec.x >= avec.y && avec.x >= avec.z) '
              '{ face = dvec.x >= 0.0 ? 0 : 1; }',
              '    else if (avec.y >= avec.z) '
              '{ face = dvec.y >= 0.0 ? 2 : 3; }',
              '    else { face = dvec.z >= 0.0 ? 4 : 5; }',
              '    float clipx = 0.0; float clipy = 0.0;',
              '    float clipz = 0.0; float clipw = 1.0;',
              '    float cellx = 0.0; float celly = 0.0;',
              '    float pbx = 0.0;']
        for fi, fc in enumerate(faces):
            L += [f'    if (face == {fi}) {{',
                  f'        clipx = dot(ph, {x_row(fi, 0)});',
                  f'        clipy = dot(ph, {x_row(fi, 1)});',
                  f'        clipz = dot(ph, {x_row(fi, 2)});',
                  f'        clipw = dot(ph, {x_row(fi, 3)});',
                  f'        cellx = {_f((fi % grid_w) * size)};',
                  f'        celly = {_f((fi // grid_w) * size + voff)};',
                  f'        pbx = {_f(float(fi * lay["w"]))};'] + \
                 ['        ' + s for s in face_consts(fi)] + ['    }']
    else:
        L += [f'    float clipx = dot(ph, {x_row(0, 0)});',
              f'    float clipy = dot(ph, {x_row(0, 1)});',
              f'    float clipz = dot(ph, {x_row(0, 2)});',
              f'    float clipw = dot(ph, {x_row(0, 3)});',
              f'    float cellx = 0.0; float celly = {_f(voff)};',
              '    float pbx = 0.0;'] + \
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
        L += [f'    float den = ({x_fpn}) - zc * {x_fmn};',
              f'    den = abs(den) < {_f(eps)} ? {_f(eps)} : den;',
              f'    float sdist = {x_tnf} / den;']
    else:
        L.append(f'    float sdist = (zc * 0.5 + 0.5) * {x_fmn}'
                 f' + {x_near};')
    L += [f'    float u = (nx * 0.5 + 0.5) * {_f(size)};',
          f'    float v = (ny * 0.5 + 0.5) * {_f(size)};',
          '    float lit = 1.0;',
          '    if (inside) {',
          f'    float nd = max(ndl, {_lit(SM.NDL_MIN)});',
          '    float ns = N.x * sax.x + N.y * sax.y + N.z * sax.z;',
          '    float nu = N.x * uax.x + N.y * uax.y + N.z * uax.z;']
    if persp:
        L += [f'    vec3 rel = ph.xyz - {x_origin};',
              '    float deu = sqrt(rel.x * rel.x + rel.y * rel.y + '
              'rel.z * rel.z);',
              '    float cpl = -(deu * nd);',
              f'    float gu = ns / (cpl * {K});',
              f'    float gv = nu / (cpl * {K});',
              f'    float x = nx * {x_extl};',
              f'    float y = ny * {x_extl};',
              '    float dr = 1.0 / max(sdist, 1e-9);']
        if rect:
            L += [f'    float Ax = {K} * (e1.x - x * e1.z);',
                  f'    float Ay = {K} * (e1.y - y * e1.z);',
                  f'    float Bx = {K} * (e2.x - x * e2.z);',
                  f'    float By = {K} * (e2.y - y * e2.z);']
        else:
            L += ['    float s1 = 1.0 + sqrt(1.0 + x * x + y * y);',
                  f'    float Ax = {x_kr} * (1.0 + x * x / s1);',
                  f'    float Ay = {x_kr} * (x * y / s1);',
                  '    float Bx = Ay;',
                  f'    float By = {x_kr} * (1.0 + y * y / s1);']
    else:
        L += [f'    float gu = ns / (nd * {K});',
              f'    float gv = nu / (nd * {K});',
              '    float dr = sdist;',
              f'    float Ax = {x_kr}; float Ay = 0.0;',
              f'    float Bx = 0.0; float By = {x_kr};']
    L += ['    float amax = max(sqrt(Ax * Ax + Ay * Ay), '
          'sqrt(Bx * Bx + By * By));',
          f'    float capt = {x_cap} / max(amax, 1e-12);']
    # the search's reach, tightened against the pyramid: from the cap,
    # up to SEARCH_STEPS times, the nearest depth within the current
    # reach and the reach that depth leaves (`softmap._face`, step 4a).
    # A step that does not move the bound ends them (the next would ask
    # the same cell), and so does a fragment with nothing nearer than
    # its own depth less the bias within reach: it is lit and runs no
    # search. A footprint with no lamp term asks one reach, the pad's
    if tx:
        L.append(f'    float padp = hal_ltex({mb + 3}).x + 1.0;')
    else:
        L.append(f'    float padp = {_lit(np.float32(np.float32(soft + SM.SEARCH_PAD) + np.float32(1.0)))};')
    t_of = ('min(max(1.0 / max(near, 1e-9) - dr, 0.0), capt)' if persp
            else 'min(max(dr - near, 0.0), capt)')
    near_x = ', dlow' if cube else ''
    if cube:
        dm = f'hal_ltex({mb + 4}).x' if tx else \
            _lit(np.float32(float(sp['d_min'])))
        L.append(f'    float dlow = {dm} * {_lit(SM._SEAM_SLACK)};')
    L += [f'    float r0 = (amax > 0.0 ? {x_cap} * {grow} : 0.0) + padp;',
          f'    float near = hal_sm_near{i}(u, v, r0, pbx{near_x});',
          f'    float ts = {t_of};',
          ('    float lim = 1.0 / dr - sl;' if persp
           else '    float lim = dr - sl;'),
          '    float more = amax > 0.0 ? 1.0 : 0.0;']
    for _step in range(1, SM.SEARCH_STEPS):
        L += ['    if (near < lim && more > 0.5) {',
              f'    float q = hal_sm_near{i}(u, v, '
              f'ts * {grow} * amax + padp, pbx{near_x});',
              '    more = q > near ? 1.0 : 0.0;',
              '    near = max(near, q);',
              f'    ts = {t_of};',
              '    }']

    # the blocker search; then, only where it is undecided, the probe;
    # then, only where THAT is undecided, the filter
    L += ['    if (near < lim) {',
          f'    vec2 fs = hal_sm_find{i}(ts, Ax, Ay, Bx, By, {cta});',
          '    float nb = fs.x; float dsum = fs.y;',
          f'    if (nb >= {nt}) {{ lit = 0.0; }}',
          '    else if (nb > 0.0) {',
          '    float davg = dsum / max(nb, 1.0);']
    if persp:
        L.append('    float tp = min(max(1.0 / max(davg, 1e-9) - dr, 0.0), '
                 'ts);')
    else:
        L.append('    float tp = min(max(dr - davg, 0.0), ts);')
    # a footprint with no lamp term (a lamp of size 0 under the user's
    # Softness) puts the probe's taps exactly on the search's: skipped
    L += ['    vec2 fp = fs;',
          '    if (amax > 0.0) {',
          f'    fp = hal_sm_find{i}(tp, Ax, Ay, Bx, By, {cta});',
          '    }',
          '    float nb2 = fp.x; float dsum2 = fp.y;',
          f'    if (nb2 >= {nt}) {{ lit = 0.0; }}',
         '    else if (nb2 > 0.0) {',
         # the estimate again, from the casters the probe itself found
         '    float davg2 = dsum2 / max(nb2, 1.0);',
         ('    float tf = min(max(1.0 / max(davg2, 1e-9) - dr, 0.0), tp);'
          if persp else
          '    float tf = min(max(dr - davg2, 0.0), tp);'),
         # the lamp's penumbra and the Softness add in quadrature
         '    float rl = tf * amax;']
    if tx:
        L += [f'    float sq = {x_sv} * {x_sv};',
              '    float rtex = sqrt(rl * rl + sq);']
    else:
        L += [f'    float sq = {_lit(sq)};',
              '    float rtex = sqrt(rl * rl + sq);']
    L += [f'    float fq = rtex / max(rl + {x_sv}, 1e-12);',
          f'    float q = max({_lit(SM.LATTICE)} * rtex / {fn}, 1.0);',
          '    float st0 = 1.0;']
    for lv in SM.STRIDES:
        L.append(f'    if (q >= {_f(lv)}) {{ st0 = {_f(lv)}; }}')
    L += [f'    float wb = clamp((q / st0 - {_lit(SM.SQRT2)}) / '
          f'{_lit(SM.SQRT2_SPAN)}, 0.0, 1.0);',
          # a strip's footprint: the kernel's blur radius along each
          # axis of the map, their ratio, and the divisor of the stride
          # across the short one (`softmap._filter`)
          '    float ex = tf * sqrt(Ax * Ax + Bx * Bx);',
          '    float ey = tf * sqrt(Ay * Ay + By * By);',
          '    float rx = sqrt(ex * ex + sq);',
          '    float ry = sqrt(ey * ey + sq);',
          '    float kq = max(rx, ry) / max(min(rx, ry), 1e-12);',
          '    float k0 = 1.0;']
    for lv in SM.ANISO_STEPS:
        L.append(f'    if (kq >= {_f(lv)}) {{ k0 = {_f(lv)}; }}')
    flt = f'hal_sm_flt{i}(tf, fq, {{}}, Ax, Ay, Bx, By, {cta})'

    def lattice(st, k):
        # (the stride across the short side, the call on that lattice)
        thin = f'max({st} / {k}, 1.0)'
        return flt.format(f'thx > 0.5 ? {thin} : {st}, '
                          f'thx > 0.5 ? {st} : {thin}')
    L += [f'    float wa = clamp((kq / k0 - {_lit(SM.ANISO_FROM)}) / '
          f'{_lit(2.0 - SM.ANISO_FROM)}, 0.0, 1.0);',
          '    float thx = rx <= ry ? 1.0 : 0.0;',
          f'    float acc0 = {lattice("st0", "k0")};',
          '    if (wa > 0.0) {',
          f'    float acc0f = {lattice("st0", "(k0 * 2.0)")};',
          '    acc0 = acc0 * (1.0 - wa) + acc0f * wa;',
          '    }',
          f'    lit = acc0 / {fn};',
          '    if (wb > 0.0) {',
          '    float st1 = st0 * 2.0;',
          f'    float acc1 = {lattice("st1", "k0")};',
          '    if (wa > 0.0) {',
          f'    float acc1f = {lattice("st1", "(k0 * 2.0)")};',
          '    acc1 = acc1 * (1.0 - wa) + acc1f * wa;',
          '    }',
          f'    lit = lit * (1.0 - wb) + (acc1 / {fn}) * wb;',
          '    }',
         '    }',
         '    }',
         '    }',
         '    }',
         f'    return 1.0 - (1.0 - lit) * {x_dens};',
         '}']
    return '\n'.join(L) + '\n'
