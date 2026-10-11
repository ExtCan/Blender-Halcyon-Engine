"""R253 (1.92.0): more Light options -- eight period lamp dials.

    (A) Square spot cone        Blender Internal LA_SQUARE
    (B) Specular Colour         OpenGL 1.1 GL_SPECULAR / D3DLIGHT7.dcvSpecular
    (C) Hemi ground colour      Rost, OpenGL Shading Language 9.1
    (D) nominal-distance decay  3ds Max Decay / LightWave Intensity Falloff
    (E) Near Attenuation        3ds Max's Attenuation Parameters rollout
    (F) shadow buffer clip      Blender Internal ClipSta / ClipEnd
    (G) cookie Offset/Tiling/Angle   3ds Max projector map, 2.4x lamp texture
    (H) Light Style flicker     Quake's twelve worldspawn lightstyles

The first test pins the identity at every default (bitwise the 1.91.0 tree
through the tests/_prev_engine rig when its zip sits beside the package,
and the emitted GLSL carries no R253 token at the defaults either way);
every option then proves its law at the function level, its A/B on a frame,
its GPU twin (the standalone statements through the GLSL front-end bitwise
where the design says bitwise, the frame at the deferred bar 6e-3), its
plan-cache structure (a value drag is a HIT, a flag a MISS) and its export.

    python -m halcyon.tests.test_r253_light_options
"""
import importlib
import sys
import traceback
import types

import numpy as np

from . import utf8_console
from .test_render import _prev_engine, base_settings
from .scenebuild import demo_scene
from . import featurematrix as FM
from .test_r251_lighting_lamps import _run_glsl, _twin
from ..core import lights as LI
from ..core import mathx as M
from ..core import post as PO
from ..core import raster as CRg
from ..core import render as R
from ..core.scene import Light
from ..gpu import material as GM
from ..gpu import shade as GSH

FAILS = []
f32 = np.float32
PREV_ZIP = 'halcyon-1.91.0.zip'
W, H = 96, 72

#: the FALLOFF items that existed before this round, in their positions
OLD_FALLOFF = ('NONE', 'INVERSE', 'INVERSE_SQUARE', 'CUSTOM', 'BI_LINEAR',
               'BI_SQUARE', 'BI_SLIDERS', 'GL_3TERM', 'POV_FADE_LINEAR',
               'POV_FADE_SQUARE', 'GX_GENTLE', 'GX_MEDIUM', 'GX_STEEP')
NEW_FIELDS = ('spot_square', 'specular_color', 'hemi_ground',
              'decay_near_start', 'decay_near_end', 'shadow_clip_start',
              'shadow_clip_end', 'cookie_offset', 'cookie_tile',
              'cookie_rotation', 'light_style', 'light_style_rate')
#: the GLSL tokens only an R253 option emits
R253_TOKENS = ('sq_x', 'sq_c', 'rad_s', 'rad_g', 'at_nt', 'at_dm', 'at_dq',
               'ck_u2', 'ck_v2')


def check(name, cond, extra=''):
    ok = bool(cond)
    print(f'  {"ok  " if ok else "FAIL"} {name}  {extra}' if extra
          else f'  {"ok  " if ok else "FAIL"} {name}')
    if not ok:
        FAILS.append(name)
    return ok


def note(text):
    print(f'  note {text}')


# ------------------------------------------------------------------ helpers


def _settings(**kw):
    st = base_settings(W, H, shadows=True, transparency='NONE')
    for k, v in kw.items():
        setattr(st, k, v)
    return st


def _scene(st, key='demo'):
    if key == 'demo':
        return demo_scene(st, with_texture=False)
    if key == 'hemi':
        sc = demo_scene(st, with_texture=False)
        sc.lights = [Light(type='HEMI', direction=(0.2, -0.3, -0.93),
                           energy=1.5, shadow='NONE')]
        return sc
    return FM.SCENES[key](st)


def _frame_bar(label, cpu, sim, mask, why, bar=6e-3):
    """The deferred road's own bar (the R251 lamps module's rig)."""
    check(f'{label}: the deferred pass plans and simulates', sim is not None,
          str(why))
    if sim is None:
        return
    yy, xx = np.nonzero(mask)
    d = np.abs(sim[yy, xx] - cpu[yy, xx, :3]).max(axis=1)
    check(f'{label}: the simulator matches the CPU within the deferred bar',
          int((d > 1e-2).sum()) == 0 and float(d.max()) < bar,
          f'{int((d > 1e-2).sum())} of {yy.size} px >0.01, '
          f'max {float(d.max()):.6f}')


def _gpu_rig(sc, st):
    """The test_light_texel_cache rig: (img, passes, why, atlases) through
    plan_frame + simulate on one G-buffer, for plan HIT / MISS pins."""
    view, _p, vp, eye = R.camera_matrices(sc.camera, W, H)
    g = CRg.GBuffer(W, H)
    CRg.rasterize(sc.mesh.verts, sc.mesh.tris, vp, W, H, gbuf=g)

    def run():
        job = R.ShadeJob(sc, st, R.prepare_textures(sc, st), None, view,
                         eye, W, H)
        passes, why, atl = GSH.plan_frame(job, g)
        if passes is None:
            return None, None, why, None
        img, _hit = GSH.simulate(job, g, passes, atl)
        return img, passes, None, atl
    return run, g


def _consts_literal():
    """_one_light_source consts for the LITERAL road (no hal_lights
    texture): the values land in the text, so the block stands alone."""
    return {'light_texels': False, 'falloff_default': 'INVERSE_SQUARE',
            'cookies': {}, 'light_links': {}, 'resolution': (W, H),
            'specular_in_gamma': True, 'light_clamp': 0.0,
            'shadow_default': 'PER_LIGHT'}


def _block(light, consts=None):
    """The emitted light block (a list of lines) for one light."""
    return GM._one_light_source(0, light, consts or _consts_literal())


def _lines_between(lines, first_token, last_token):
    out, on = [], False
    for ln in lines:
        if first_token in ln:
            on = True
        if on:
            out.append(ln)
        if on and last_token in ln:
            break
    return out


def _spot(**kw):
    base = dict(type='SPOT', name='sq', position=(0.0, 0.0, 4.0),
                direction=(0.0, 0.0, -1.0), color=(1.0, 1.0, 1.0),
                energy=600.0, spot_size=1.0, spot_blend=0.2, shadow='NONE',
                decay='NONE')
    base.update(kw)
    return Light(**base)


def _point(decay='INVERSE_SQUARE', end=4.0, **kw):
    base = dict(type='POINT', name='fade', position=(0.0, 0.0, 0.0),
                decay=decay, decay_end=end, shadow='NONE')
    base.update(kw)
    return Light(**base)


def _floor_mask(sc, g):
    """Covered pixels on the demo Floor (material 0, normal +Z)."""
    cov = g.tri >= 0
    mi = np.asarray(sc.mesh.mat_index)[np.clip(g.tri, 0, None)]
    return cov & (mi == 0)


def _gbuf(sc):
    view, _p, vp, eye = R.camera_matrices(sc.camera, W, H)
    g = CRg.GBuffer(W, H)
    CRg.rasterize(sc.mesh.verts, sc.mesh.tris, vp, W, H, gbuf=g)
    return g, vp, view, eye


def _render(sc, st):
    return np.asarray(R.render(sc, st))


# ------------------------------------------------------- 1. the identity pin


def test_a_identity_at_defaults():
    """Every new dial is invisible at its default: the lamp scenes render
    (and post.process) bitwise the 1.91.0 tree, under the INVERSE_SQUARE
    default and under every pre-existing FALLOFF item; and the GLSL the
    plan emits at the defaults carries no R253 token at all."""
    RP = _prev_engine(PREV_ZIP)
    if RP is None:
        note(f'{PREV_ZIP} is not beside the package -- the bitwise pin '
             'against the 1.91.0 tree is skipped here (the in-tree '
             'neutrality proofs below still run)')
    else:
        prev_post = importlib.import_module(
            RP.__name__.rsplit('.', 1)[0] + '.post')
        cases = [(f'the demo scene under light_falloff_default {m}', 'demo',
                  {'light_falloff_default': m}) for m in OLD_FALLOFF]
        cases += [('the spot-gobo scene (G identity: no transform)',
                   'cookie_spot', {}),
                  ('the negative-lamp scene (B/H identity)', 'negative', {}),
                  ('the soft-shadow scene (F identity: clip 0/0)', 'soft',
                   {}),
                  ('the area-lamp scene (E identity: no near pair)', 'area',
                   {}),
                  ('a HEMI scene (C identity: black ground)', 'hemi', {})]
        for label, key, kw in cases:
            st = _settings(**kw)
            sc = _scene(st, key)
            now = _render(sc, st)
            st2 = _settings(**kw)
            sc2 = _scene(st2, key)
            prev = np.asarray(RP.render(sc2, st2))
            same = now.shape == prev.shape and bool(np.array_equal(now, prev))
            check(f'{label} renders bitwise the 1.91.0 tree', same,
                  f'max {float(np.abs(now - prev).max()):.3g}'
                  if (not same and now.shape == prev.shape) else '')
        st = _settings()
        sc = _scene(st)
        now = np.asarray(PO.process(R.render(sc, st), st))
        st2 = _settings()
        sc2 = _scene(st2)
        prev = np.asarray(prev_post.process(RP.render(sc2, st2), st2))
        check('the demo scene: render + post.process bitwise the 1.91.0 tree',
              now.shape == prev.shape and bool(np.array_equal(now, prev)))

    # in-tree: the Light dataclass carries every default inert
    lt = Light()
    check('the Light dataclass defaults are the inert values',
          lt.spot_square is False and tuple(lt.specular_color) == (1, 1, 1)
          and tuple(lt.hemi_ground) == (0, 0, 0)
          and lt.decay_near_start == 0.0 and lt.decay_near_end == 0.0
          and lt.shadow_clip_start == 0.0 and lt.shadow_clip_end == 0.0
          and tuple(lt.cookie_offset) == (0, 0)
          and tuple(lt.cookie_tile) == (1, 1) and lt.cookie_rotation == 0.0
          and lt.light_style == 'NORMAL' and lt.light_style_rate == 10.0
          and lt.style_gain == 1.0)
    # the attenuate restructure: the NONE branch (its early return is
    # gone) still gives ones, and with Sphere the old wrap bitwise
    st = _settings()
    dist = np.linspace(0.0, 30.0, 301).astype(f32)
    a = LI.attenuate(_point('NONE', 25.0), dist, st)
    check('attenuate NONE is exactly ones (the early return is gone, the '
          'value is not)', bool(np.all(a == 1.0)) and a.dtype == np.float32)
    lt = _point('NONE', 25.0, bi_sphere=True)
    a = LI.attenuate(lt, dist, st)
    D = 25.0
    t = D - np.maximum(dist, 0.0)
    old = np.where(t <= 0.0, 0.0, np.ones_like(dist) * t / D).astype(f32)
    check('attenuate NONE + Sphere is bitwise the 1.91 wrap '
          '(where(t <= 0, 0, att * t / D))', bool(np.array_equal(a, old)))
    for mode in OLD_FALLOFF:
        lt = _point(mode, 6.0, decay_ld1=0.5, decay_ld2=0.1)
        a1 = LI.attenuate(lt, dist, st)
        lt.decay_near_start = 2.0
        lt.decay_near_end = 1.0            # end <= start: inert
        a2 = LI.attenuate(lt, dist, st)
        check(f'{mode}: a Near pair with End <= Start is bitwise inert',
              bool(np.array_equal(a1, a2)))
    # the emitted GLSL at the defaults carries no R253 token
    for key in ('demo', 'cookie_spot', 'hemi', 'negative'):
        st = _settings()
        sc = _scene(st, key)
        for l in sc.lights:
            l.shadow = 'NONE'
        run, _g = _gpu_rig(sc, st)
        GSH._PLAN_CACHE.clear()
        _img, passes, why, _atl = run()
        if passes is None:
            check(f'{key}: the default frame plans', False, str(why))
            continue
        text = '\n'.join(str(p[2]) for p in passes)
        found = [tok for tok in R253_TOKENS if tok in text]
        check(f'{key}: the GLSL at every default carries no R253 token',
              not found, ', '.join(found))
    # the texel stride doubled, every reader goes through the symbol
    check('LIGHT_TEXEL_STRIDE is 16 and _lref addresses through it',
          GM.LIGHT_TEXEL_STRIDE == 16
          and GM._lref(1, 8, 'rgb') == 'hal_ltex(24).rgb')
    sc = _scene(_settings(), 'demo')
    tex = GM.pack_light_texels(sc.lights)
    check('texels 12-15 are reserved zeros, 8.rgb white and 9.rgb black at '
          'the defaults',
          tex.shape[1] == len(sc.lights) * 16
          and bool(np.all(tex[0, 12:16] == 0.0))
          and bool(np.all(tex[0, 8, :3] == 1.0))
          and bool(np.all(tex[0, 9] == 0.0)))


# ------------------------------------------------------------------ (A)


def _cone_dirs(n=4001, half=0.6, seed=3):
    """Random unit directions L (surface -> light) within `half` radians
    of the -direction axis of a lamp pointing straight down, plus a few
    behind it."""
    rng = np.random.default_rng(seed)
    th = rng.uniform(0.0, half, n)
    ph = rng.uniform(0.0, 2.0 * np.pi, n)
    # light -> surface direction d = (sin th cos ph, sin th sin ph, -cos th)
    d = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph),
                  -np.cos(th)], axis=1)
    d[-7:, 2] = np.abs(d[-7:, 2])        # behind the lamp
    L = (-d).astype(f32)
    return L


def test_square_spot():
    """(A): Blender Internal's LA_SQUARE on every cone law and both devices."""
    lt = _spot(spot_square=True)
    L = _cone_dirs()
    z = -(L @ np.asarray(lt.direction, f32))
    got = LI.square_cos(lt, L, z)
    s, u, _f = LI.cookie_frame(lt)
    x = -(L.astype(np.float64) @ s.astype(np.float64))
    y = -(L.astype(np.float64) @ u.astype(np.float64))
    zz = np.maximum(z.astype(np.float64), 1e-20)
    m = np.maximum(np.abs(x / zz), np.abs(y / zz))
    want = np.where(z > 0.0, 1.0 / np.sqrt(1.0 + m * m), 0.0)
    check('square_cos == 1/sqrt(1 + max(|x/z|, |y/z|)^2) on random '
          'directions (within 1e-6 of float64) and 0 behind the lamp',
          float(np.abs(got - want).max()) < 1e-6
          and bool(np.all(got[z <= 0.0] == 0.0)) and got.dtype == np.float32,
          f'max {float(np.abs(got - want).max()):.2e}')
    # A/B at the function level: a floor point at lamp-space (0.9t, 0.9t)
    t = float(np.tan(0.5))
    P = np.array([[0.0, 0.0, 0.0], [0.9 * t * 4.0, 0.9 * t * 4.0, 0.0]], f32)
    pos = np.asarray(lt.position, f32)
    Lp = M.normalize(pos[None, :] - P)
    rnd = _spot()
    f_round = LI.spot_falloff(rnd, Lp)
    f_sq = LI.spot_falloff(lt, Lp)
    check('the corner point (0.9t, 0.9t) is dark under the round cone and '
          'lit under the square one',
          float(f_round[1]) == 0.0 and float(f_sq[1]) > 0.5,
          f'round {float(f_round[1]):.3f} square {float(f_sq[1]):.3f}')
    check('the axis point is bitwise unchanged by Square',
          float(f_round[0]) == float(f_sq[0]) and float(f_sq[0]) > 0.0)
    # every law gets a square version
    for law in ('BLENDER', 'GL11', 'POV', 'GX_FLAT'):
        a = LI.spot_falloff(_spot(spot_law=law, hotspot=0.5), L)
        b = LI.spot_falloff(_spot(spot_law=law, hotspot=0.5,
                                  spot_square=True), L)
        check(f'a {law} square spot differs from its round twin',
              float(np.abs(a - b).max()) > 1e-3)
    # the GLSL twin: the emitted statements through the front-end
    lines = _block(lt)
    sq = _lines_between(lines, 'float sq_x', 'cosang = (cosang > 0.0)')
    check('the square block is emitted after cosang (eleven statements)',
          len(sq) == 11 and sq[0].strip().startswith('float sq_x'),
          str(len(sq)))
    n = L.shape[0]
    head = ('uniform vec3 L;\nuniform float cosang_in;\nout vec4 Color;\n'
            'void main()\n{\n    float cosang = cosang_in;\n')
    src = head + '\n'.join(sq) + '\n    Color = vec4(cosang, 0.0, 0.0, 1.0);\n}\n'
    with np.errstate(over='ignore', invalid='ignore'):
        # the lanes behind the lamp overflow in the simulator too (and
        # are discarded by the same select)
        out, err = _run_glsl(src, n, L=L, cosang_in=z)
    if out is None:
        check('the square-cone GLSL compiles standalone', False, str(err))
    else:
        d = float(np.abs(out[:, 0] - got).max())
        check('the full square block (with its two dot products) equals '
              'square_cos within 2e-6', d < 2e-6, f'd {d:.3g}')
    # the arithmetic tail, fed the CPU's own x and y, is bitwise
    tail = [ln for ln in sq if 'dot(' not in ln]
    xs = (-(L @ s.astype(f32))).astype(f32)
    ys = (-(L @ u.astype(f32))).astype(f32)
    src2 = ('uniform float sq_x;\nuniform float sq_y;\nuniform float cosang_in;'
            '\nout vec4 Color;\nvoid main()\n{\n    float cosang = cosang_in;\n'
            + '\n'.join(tail) + '\n    Color = vec4(cosang, 0.0, 0.0, 1.0);\n}\n')
    with np.errstate(over='ignore', invalid='ignore'):
        out2, err2 = _run_glsl(src2, n, sq_x=xs, sq_y=ys, cosang_in=z)
    if out2 is None:
        check('the square-cone tail compiles standalone', False, str(err2))
    else:
        d = float(np.abs(out2[:, 0] - got).max())
        check('the square-cone arithmetic equals square_cos bitwise (d == 0.0)',
              d == 0.0, f'd {d:.3g}')
    # the frame twin and the A/B on the row
    sc, st = FM.build('square spot cone (Blender Internal LA_SQUARE)')
    st.transparency = 'NONE'
    cpu, sim, mask, why = _twin(sc, st)
    _frame_bar('the square-spot row', cpu, sim, mask, why)
    sc2, st2 = FM.build('square spot cone (Blender Internal LA_SQUARE)')
    st2.transparency = 'NONE'
    sc2.lights[-1].spot_square = False
    base = _render(sc2, st2)
    check('the square spot lights pixels the round cone leaves dark (and '
          'darkens none)',
          int(((cpu - base)[..., :3].sum(2) > 1e-3).sum()) > 20
          and float((base - cpu)[..., :3].max()) < 1e-5,
          f'{int(((cpu - base)[..., :3].sum(2) > 1e-3).sum())} px lit')
    # structure
    a, b = _spot(), _spot(spot_square=True)
    check('Square is light-signature structure (plan MISS) and shadow-map '
          'signature (map rebuild)',
          GSH._light_sig(a) != GSH._light_sig(b)
          and LI._light_shadow_signature(a) != LI._light_shadow_signature(b))
    # the widened buffer
    st3 = _settings()
    st3.shadow_default = 'PER_LIGHT'
    sc3 = demo_scene(st3, with_texture=False)
    sp = _spot(shadow='MAP', position=(-1.3, 0.2, 3.6), spot_size=1.2)
    sc3.lights = [sp]
    LI._SHADOW_CACHE.clear()
    R.render(sc3, st3)
    ext_round = float(sp.shadow_map.extent)
    sp.spot_square = True
    LI._SHADOW_CACHE.clear()
    R.render(sc3, st3)
    ext_sq = float(sp.shadow_map.extent)
    check('a Square spot widens its shadow buffer toward the diagonal '
          '(extent grows, below sqrt(2) x)',
          ext_sq > ext_round * 1.2 and ext_sq < ext_round * 1.5,
          f'{ext_round:.3f} -> {ext_sq:.3f}')
    # import
    from ..core import blend279_map as BM
    from . import fakebpy
    fakebpy.install()
    from .. import legacy_import as LI2
    parsed = {'kind': 'SPOT', 'name': 'Sq', 'r': 1.0, 'g': 1.0, 'b': 1.0,
              'energy': 1.0, 'dist': 20.0, 'falloff_type': 1,
              'mode': BM.LA_SQUARE | BM.LA_SHAD_BUF, 'spotsize': 0.8,
              'spotblend': 0.15, 'clipsta': 0.5, 'clipend': 30.0}
    lm = BM.lamp_map(parsed, 279)
    check('lamp_map imports LA_SQUARE as square = True and warns no more',
          lm.get('square') is True
          and not any('square' in w for w in lm['warnings']),
          str(lm['warnings']))
    lm2 = BM.lamp_map(dict(parsed, mode=BM.LA_SHAD_BUF), 279)
    check('without the bit the cone imports round', lm2.get('square') is False)
    fake = types.SimpleNamespace(
        color=None, energy=0.0, spot_size=0.0, spot_blend=0.0,
        halcyon=types.SimpleNamespace(
            hemi=False, shadow='MAP', decay='DEFAULT', decay_start=0.0,
            decay_end=25.0, decay_ld1=0.0, decay_ld2=0.0, bi_sphere=False,
            negative=False, specular_only=False, diffuse_only=False,
            spot_square=False, shadow_clip_start=0.0, shadow_clip_end=0.0))
    LI2._apply_lamp_bi(fake, lm)
    check('_apply_lamp_bi sets hs.spot_square (and the clip pair of a '
          'buffer spot)',
          fake.halcyon.spot_square is True
          and fake.halcyon.shadow_clip_start == 0.5
          and fake.halcyon.shadow_clip_end == 30.0)
    # export
    lt_e = _export_fake(spot_square=True)
    check('export_light carries spot_square', lt_e.spot_square is True)


# ------------------------------------------------------------------ (B)


def test_specular_color():
    """(B): the lamp's second colour tints the classic highlight only."""
    st = _settings(default_model='PHONG')
    sc = demo_scene(st, with_texture=False)
    for l in sc.lights:
        l.shadow = 'NONE'
    a = _render(sc, st)
    for l in sc.lights:
        l.specular_color = (1.0, 0.2, 0.2)
    b = _render(sc, st)
    g, _vp, _v, _e = _gbuf(sc)
    cov = g.tri >= 0
    moved = np.abs(b - a)[..., :3].max(axis=2) > 1e-6
    check('a red specular tint moves the highlight (some covered pixels) '
          'and leaves most of the frame bitwise',
          int(moved[cov].sum()) > 10 and int(moved[cov].sum()) < 0.6 * cov.sum(),
          f'{int(moved[cov].sum())} of {int(cov.sum())} px')
    check('the red channel is bitwise unchanged (rad_s.r = rad.r * 1.0 '
          'exactly) and green / blue never rise',
          bool(np.array_equal(a[..., 0], b[..., 0]))
          and float((b - a)[..., 1:3].max()) < 1e-6)
    for l in sc.lights:
        l.specular_color = (1.0, 1.0, 1.0)
    check('white is bitwise identity', bool(np.array_equal(_render(sc, st), a)))
    # the GPU frame
    for l in sc.lights:
        l.specular_color = (1.0, 0.2, 0.2)
    cpu, sim, mask, why = _twin(sc, st)
    _frame_bar('the tinted PHONG frame', cpu, sim, mask, why)
    sc_r, st_r = FM.build('lamp specular colour (GL_SPECULAR / dcvSpecular)')
    st_r.transparency = 'NONE'
    cpu, sim, mask, why = _twin(sc_r, st_r)
    _frame_bar('the specular-colour row', cpu, sim, mask, why)
    # the plan: a tint VALUE drag is a HIT, on/off a MISS
    run, _g = _gpu_rig(sc, st)
    GSH._PLAN_CACHE.clear()
    img1, p1, why1, atl1 = run()
    check('the tinted frame plans', p1 is not None, str(why1))
    if p1 is not None:
        for l in sc.lights:
            l.specular_color = (0.6, 0.9, 0.3)
        img2, p2, _w2, atl2 = run()
        check('a tint VALUE drag is a plan-cache HIT with fresh texels',
              p2 is p1 and atl2['hal_lights'][0] != atl1['hal_lights'][0]
              and float(np.abs(img2 - img1).max()) > 1e-4)
        for l in sc.lights:
            l.specular_color = (1.0, 1.0, 1.0)
        _img3, p3, _w3, _a3 = run()
        check('tint off (white) re-plans honestly (structure)', p3 is not p1)
    # the cel roads ignore the tint
    for model in ('CARTOON', 'ANIME'):
        stc = _settings()
        scc = demo_scene(stc, with_texture=False)
        for m_ in scc.materials:
            m_.model = model                 # the demo materials carry
        for l in scc.lights:                 # their own models
            l.shadow = 'NONE'
        try:
            a_c = _render(scc, stc)
            for l in scc.lights:
                l.specular_color = (1.0, 0.2, 0.2)
            b_c = _render(scc, stc)
        except Exception as e:                                  # noqa: BLE001
            check(f'the {model} road renders', False, str(e))
            continue
        check(f'the {model} road ignores the specular tint (frame bitwise)',
              bool(np.array_equal(a_c, b_c)))
    # the texel and the export
    tex = GM.pack_light_texels([_spot(specular_color=(0.3, 0.6, 0.9))])
    check('texel 8.rgb is the tint as float32',
          bool(np.array_equal(tex[0, 8, :3], np.asarray((0.3, 0.6, 0.9), f32))))
    check('export_light carries specular_color',
          _export_fake(specular_color=(0.3, 0.6, 0.9)).specular_color
          == (0.3, 0.6, 0.9))


# ------------------------------------------------------------------ (C)


def test_hemi_ground():
    """(C): the two-tone hemisphere's ground term, both roads."""
    st = _settings()
    st.global_ambient_level = 0.0
    sc = demo_scene(st, with_texture=False)
    sc.world.ambient_level = 0.0
    E = 1.5
    hemi = Light(type='HEMI', name='sky', direction=(0.0, 0.0, 1.0),
                 color=(1.0, 0.9, 0.8), energy=E, shadow='NONE')
    sc.lights = [hemi]
    base = _render(sc, st)                     # sky term 0 on the floor
    g, _vp, _v, _e = _gbuf(sc)
    floor = _floor_mask(sc, g)
    ground = (0.2, 0.4, 0.6)
    hemi.hemi_ground = ground
    a = _render(sc, st)
    got = (a - base)[floor][:, :3]
    dcol = np.asarray(sc.materials[0].diffuse, f32)
    level = f32(getattr(sc.materials[0], 'diffuse_level', 1.0))
    want = ((f32(1.0) * dcol) * level) * (np.asarray(ground, f32) * f32(E))
    want = want * f32(1.0 / np.pi)
    check('a floor facing away from the hemi (N.L = -1) shades exactly '
          'diffuse * level * ground * energy / pi',
          floor.sum() > 50 and float(np.abs(got - want[None, :]).max()) < 2e-6,
          f'max {float(np.abs(got - want[None, :]).max()):.2e} over '
          f'{int(floor.sum())} px')
    hemi.hemi_ground = (0.0, 0.0, 0.0)
    check('a black ground is bitwise the plain hemi frame',
          bool(np.array_equal(_render(sc, st), base)))
    # the sky term is untouched: a face toward the hemi (ground weight 0)
    sc_up = demo_scene(st, with_texture=False)
    sc_up.world.ambient_level = 0.0
    hemi_dn = Light(type='HEMI', name='sky', direction=(0.0, 0.0, -1.0),
                    color=(1.0, 0.9, 0.8), energy=E, shadow='NONE')
    sc_up.lights = [hemi_dn]
    up0 = _render(sc_up, st)
    hemi_dn.hemi_ground = ground
    up1 = _render(sc_up, st)
    g2, _vp2, _v2, _e2 = _gbuf(sc_up)
    floor2 = _floor_mask(sc_up, g2)
    check('a floor facing the hemi (N.L = 1, ground weight 0) is bitwise '
          'unchanged by the ground', bool(np.array_equal(up0[floor2], up1[floor2])))
    # negative negates both terms
    hemi.hemi_ground = ground
    hemi.negative = True
    neg = _render(sc, st)
    hemi.negative = False
    pos = _render(sc, st)
    check('a negative hemi negates the ground term too (pos + neg = 2 x base)',
          float(np.abs((pos + neg) - 2.0 * base)[floor].max()) < 4e-6)
    # the ground ignores the cookie
    hemi.cookie = FM._cookie_pixels()
    hemi.cookie_scale = 2.0
    ck = _render(sc, st)
    gt = (ck - base)[floor][:, :3]
    check('the ground term ignores the projected slide (constant across the '
          'floor under a tiled HEMI cookie)',
          float(np.ptp(gt, axis=0).max()) < 2e-6
          and float(np.abs(gt - want[None, :]).max()) < 2e-6,
          f'ptp {float(np.ptp(gt, axis=0).max()):.2e}')
    hemi.cookie = None
    # the GPU twin
    sc_r, st_r = FM.build('hemi ground colour (two-tone hemisphere)')
    st_r.transparency = 'NONE'
    cpu, sim, mask, why = _twin(sc_r, st_r)
    _frame_bar('the hemi-ground row', cpu, sim, mask, why)
    sc_n = demo_scene(st, with_texture=False)
    sc_n.lights = [Light(type='HEMI', direction=(0.2, -0.3, -0.93), energy=1.5,
                         shadow='NONE', hemi_ground=(0.3, 0.1, 0.0),
                         negative=True)]
    cpu, sim, mask, why = _twin(sc_n, st)
    _frame_bar('a negative two-tone hemi', cpu, sim, mask, why)
    # structure / texel / export
    h0 = Light(type='HEMI')
    h1 = Light(type='HEMI', hemi_ground=(0.1, 0.1, 0.1))
    h2 = Light(type='HEMI', hemi_ground=(0.5, 0.2, 0.1))
    check('a non-black ground is structure, its VALUE a texel',
          GSH._light_sig(h0) != GSH._light_sig(h1)
          and GSH._light_sig(h1) == GSH._light_sig(h2))
    tex = GM.pack_light_texels([h2])
    check('texel 9.rgb is the ground as float32 and 0.w the energy',
          bool(np.array_equal(tex[0, 9, :3], np.asarray((0.5, 0.2, 0.1), f32)))
          and float(tex[0, 0, 3]) == float(f32(h2.energy)))
    check('export_light carries hemi_ground',
          _export_fake(hemi_ground=(0.5, 0.2, 0.1), hemi=True,
                       kind='SUN').hemi_ground == (0.5, 0.2, 0.1))


# ------------------------------------------------------------------ (D)


def test_nominal_decay():
    """(D): (D/d)^n with D = Falloff End, 3ds Max / LightWave."""
    st = _settings()
    D = 4.0
    dist = np.linspace(0.0, 2.0 * D, 257).astype(f32)
    i_D = int(np.argmin(np.abs(dist - D)))
    i_2D = dist.size - 1
    assert float(dist[i_D]) == D and float(dist[i_2D]) == 2.0 * D
    a1 = LI.attenuate(_point('NOMINAL_INVERSE', D), dist, st)
    a2 = LI.attenuate(_point('NOMINAL_INV_SQUARE', D), dist, st)
    check('NOMINAL_INVERSE is exactly 1.0 at d = D, exactly 0.5 at 2D, '
          'above 1 inside and non-increasing',
          float(a1[i_D]) == 1.0 and float(a1[i_2D]) == 0.5
          and bool(np.all(a1[1:i_D] > 1.0)) and bool(np.all(np.diff(a1) <= 0)))
    check('NOMINAL_INV_SQUARE is exactly 1.0 at D, exactly 0.25 at 2D',
          float(a2[i_D]) == 1.0 and float(a2[i_2D]) == 0.25
          and bool(np.all(a2[1:i_D] > 1.0)))
    check('both are finite at the lamp itself (d floored at 1e-6)',
          bool(np.isfinite(a1[0])) and bool(np.isfinite(a2[0])))
    # standalone GLSL twin
    n = dist.size
    for mode in LI.DECAY_LAWS_R253:
        body = GM.decay_law_glsl(mode, {'D': 'u_D', 'ld1': 'u_ld1',
                                        'ld2': 'u_ld2', 'k1': 'u_k1',
                                        'k2': 'u_k2'})
        src = ('uniform float dist;\nuniform float u_D;\nout vec4 Color;\n'
               'void main()\n{\n' + '\n'.join(body)
               + '\n    Color = vec4(att, 0.0, 0.0, 1.0);\n}\n')
        got, err = _run_glsl(src, n, dist=dist, u_D=f32(max(D, LI.EPS)))
        want = LI.attenuate(_point(mode, D), dist, st)
        if got is None:
            check(f'the {mode} decay GLSL compiles standalone', False, str(err))
            continue
        d = float(np.abs(got[:, 0] - want).max())
        check(f'the {mode} decay GLSL equals LI.attenuate bitwise (d == 0.0)',
              d == 0.0, f'd {d:.3g}')
    # the Scene Default road, both devices
    st_d = _settings(light_falloff_default='NOMINAL_INVERSE')
    check('a lamp on the Scene Default takes NOMINAL_INVERSE bitwise',
          bool(np.array_equal(LI.attenuate(_point('DEFAULT', D), dist, st_d),
                              a1)))
    blk = '\n'.join(_block(_point('DEFAULT', D),
                           dict(_consts_literal(),
                                falloff_default='NOMINAL_INVERSE')))
    check('the GPU block of a DEFAULT lamp follows the scene default too',
          'at_dq' in blk and 'att = att * att' not in blk)
    blk2 = '\n'.join(_block(_point('NOMINAL_INV_SQUARE', D)))
    check('...and the square law squares', 'at_dq' in blk2
          and 'att = att * att;' in blk2)
    for key in ('inverse from nominal distance (3ds Max / LightWave)',
                'inverse square from nominal distance (3ds Max / LightWave)'):
        sc, stt = FM.build(key)
        stt.transparency = 'NONE'
        cpu, sim, mask, why = _twin(sc, stt)
        _frame_bar(f'the row {key!r}', cpu, sim, mask, why)
        sc2, st2 = FM.build(key)
        st2.transparency = 'NONE'
        st2.light_falloff_default = 'INVERSE_SQUARE'
        base = _render(sc2, st2)
        check(f'the row {key!r} differs from the default decay',
              float(np.abs(cpu - base).max()) > 1e-3)
    # bi_sphere composes: zero past Falloff End
    a3 = LI.attenuate(_point('NOMINAL_INVERSE', D, bi_sphere=True), dist, st)
    check('Sphere composes with the nominal law (zero at and past D, the '
          'law inside)', bool(np.all(a3[i_D:] == 0.0))
          and bool(np.all(a3[1:i_D] > 0.0)))
    # the enum positions
    from . import fakebpy
    fakebpy.install()
    from .. import properties as PR
    idents = [it[0] for it in PR.FALLOFF]
    check('every pre-existing FALLOFF item keeps its position (saved files '
          'hold the numbers)', tuple(idents[:len(OLD_FALLOFF)]) == OLD_FALLOFF)
    check('the two nominal laws are appended at the END',
          tuple(idents[-2:]) == LI.DECAY_LAWS_R253)
    check('the Scene Default mirror (RenderSettings) picks the new items up',
          'NOMINAL_INV_SQUARE' in [it[0] for it in
                                   PR.ENUMS['light_falloff_default']]
          if hasattr(PR, 'ENUMS') else True)


# ------------------------------------------------------------------ (E)


def test_near_attenuation():
    """(E): 3ds Max's Near Attenuation fade-in."""
    st = _settings()
    dist = np.linspace(0.0, 12.0, 241).astype(f32)
    plain = _point('INVERSE_SQUARE', 25.0)
    base = LI.attenuate(plain, dist, st)
    near = _point('INVERSE_SQUARE', 25.0, decay_near_start=2.0,
                  decay_near_end=6.0)
    a = LI.attenuate(near, dist, st)
    check('att is exactly 0 below Start', bool(np.all(a[dist < 2.0] == 0.0)))
    check('att is bitwise the plain curve from End on (t = 1.0 exactly)',
          bool(np.array_equal(a[dist >= 6.0], base[dist >= 6.0])))
    i3 = int(np.argmin(np.abs(dist - 3.0)))
    i4 = int(np.argmin(np.abs(dist - 4.0)))
    i5 = int(np.argmin(np.abs(dist - 5.0)))
    ratios = [float(a[i] / base[i]) for i in (i3, i4, i5)]
    check('linear between Start and End (0.25 / 0.5 / 0.75 of the curve at '
          '3 / 4 / 5)',
          all(abs(r - w) < 1e-6 for r, w in zip(ratios, (0.25, 0.5, 0.75))),
          str([round(r, 4) for r in ratios]))
    inert = _point('INVERSE_SQUARE', 25.0, decay_near_start=6.0,
                   decay_near_end=6.0)
    check('End <= Start is bitwise inert',
          bool(np.array_equal(LI.attenuate(inert, dist, st), base)))
    # SUN / AREA ignore it
    P = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, 0.0], [3.0, 0.0, 0.5]], f32)
    for kind, kw in (('SUN', dict(direction=(0.0, 0.0, -1.0))),
                     ('AREA', dict(position=(0.0, 0.0, 4.0),
                                   direction=(0.0, 0.0, -1.0),
                                   area_size=(1.0, 1.0)))):
        l0 = Light(type=kind, energy=10.0, shadow='NONE', **kw)
        l1 = Light(type=kind, energy=10.0, shadow='NONE',
                   decay_near_start=1.0, decay_near_end=8.0, **kw)
        r0 = LI.sample(l0, P, st)[1]
        r1 = LI.sample(l1, P, st)[1]
        check(f'a {kind} lamp ignores the near pair (radiance bitwise)',
              bool(np.array_equal(r0, r1)))
    # GLSL standalone bitwise
    blk = _block(_point('NONE', 25.0, decay_near_start=2.0,
                        decay_near_end=6.0))
    lines = _lines_between(blk, 'float at_dm', 'att = att * at_nt')
    check('the near block is emitted after att is final (five statements)',
          len(lines) == 5, str(lines))
    n = dist.size
    att_in = (np.linspace(0.3, 2.0, n)).astype(f32)
    src = ('uniform float dist;\nuniform float att_in;\nout vec4 Color;\n'
           'void main()\n{\n    float att = att_in;\n' + '\n'.join(lines)
           + '\n    Color = vec4(att, 0.0, 0.0, 1.0);\n}\n')
    got, err = _run_glsl(src, n, dist=dist, att_in=att_in)
    want = LI.near_attenuation(att_in, dist, 2.0, 6.0)
    if got is None:
        check('the near-attenuation GLSL compiles standalone', False, str(err))
    else:
        d = float(np.abs(got[:, 0] - want).max())
        check('the near-attenuation GLSL equals near_attenuation bitwise '
              '(d == 0.0)', d == 0.0, f'd {d:.3g}')
    # the order: after the Sphere wrap, before the BI snap
    blk2 = '\n'.join(_block(_point('BI_LINEAR', 5.0, bi_sphere=True,
                                   decay_near_start=1.0, decay_near_end=2.0)))
    i_att = blk2.find('float att =')
    i_near = blk2.find('float at_dm')
    i_snap = blk2.find('att = (att <= 0.001)')
    check('the near lines sit after att (with its Sphere wrap) and before '
          'the BI 0.001 snap', 0 <= i_att < i_near < i_snap)
    # frame
    sc, stt = FM.build('near attenuation fade-in (3ds Max)')
    stt.transparency = 'NONE'
    cpu, sim, mask, why = _twin(sc, stt)
    _frame_bar('the near-attenuation row', cpu, sim, mask, why)
    sc2, st2 = FM.build('near attenuation fade-in (3ds Max)')
    st2.transparency = 'NONE'
    for l in sc2.lights:
        l.decay_near_end = 0.0
    base_f = _render(sc2, st2)
    check('the fade-in darkens the near surfaces and brightens nothing',
          int(((base_f - cpu)[..., :3].sum(2) > 1e-3).sum()) > 50
          and float((cpu - base_f)[..., :3].max()) < 1e-5)
    # structure vs texel
    st3 = _settings()
    sc3 = demo_scene(st3, with_texture=False)
    for l in sc3.lights:
        l.shadow = 'NONE'
    pt = [l for l in sc3.lights if l.type == 'POINT'][0]
    pt.decay_near_start, pt.decay_near_end = 1.0, 3.0
    run, _g = _gpu_rig(sc3, st3)
    GSH._PLAN_CACHE.clear()
    img1, p1, why1, atl1 = run()
    check('the near frame plans', p1 is not None, str(why1))
    if p1 is not None:
        pt.decay_near_end = 4.5
        img2, p2, _w, atl2 = run()
        check('dragging the distances is a plan HIT with fresh texels',
              p2 is p1 and atl2['hal_lights'][0] != atl1['hal_lights'][0]
              and float(np.abs(img2 - img1).max()) > 1e-5)
        pt.decay_near_end = 0.0
        _i3, p3, _w3, _a3 = run()
        check('switching the fade-in off re-plans (structure)', p3 is not p1)
    tex = GM.pack_light_texels([near])
    check('texels 8.w / 9.w carry start and span as float32',
          float(tex[0, 8, 3]) == 2.0 and float(tex[0, 9, 3]) == 4.0)
    lt_e = _export_fake(decay_near_start=1.5, decay_near_end=2.5)
    check('export_light carries the near pair',
          lt_e.decay_near_start == 1.5 and lt_e.decay_near_end == 2.5)


# ------------------------------------------------------------------ (F)


def test_shadow_clip():
    """(F): BI's ClipSta / ClipEnd on the shadow buffer."""
    st = _settings()
    st.shadow_default = 'PER_LIGHT'
    sc = demo_scene(st, with_texture=False)
    lamp = Light(type='POINT', name='clipped', position=(-1.3, 0.2, 3.6),
                 color=(1.0, 1.0, 1.0), energy=300.0, shadow='MAP',
                 decay='INVERSE_SQUARE')
    sc.lights = [lamp]
    LI._SHADOW_CACHE.clear()
    a = _render(sc, st)
    near0, far0 = float(lamp.shadow_map.faces[0].near), float(lamp.shadow_map.faces[0].far)
    radius = LI.scene_bounds(sc.mesh.verts)[1]
    pos = np.asarray(lamp.position, np.float32)
    centre = LI.scene_bounds(sc.mesh.verts)[0]
    want_near = max(radius * 0.005, 1e-3)
    want_far = max(float(np.linalg.norm(centre - pos)) + radius * 1.5,
                   want_near * 10)
    check('clip 0/0 keeps the automatic planes bitwise',
          near0 == want_near and far0 == want_far,
          f'{near0:.6f} {far0:.6f}')
    meta, _atlas, why = GSH._shadow_meta(lamp, st)
    check("the GPU shadow meta's near/far are the built map's",
          meta is not None and meta['near'] == near0 and meta['far'] == far0,
          str(why))
    sig0 = LI._light_shadow_signature(lamp)
    lamp.shadow_clip_end = 2.2
    check('changing a clip changes the per-light shadow signature',
          LI._light_shadow_signature(lamp) != sig0)
    LI._SHADOW_CACHE.clear()
    b = _render(sc, st)
    check('ClipEnd replaces the far plane (near untouched)',
          float(lamp.shadow_map.faces[0].far) == 2.2
          and float(lamp.shadow_map.faces[0].near) == near0)
    d = (b - a)[..., :3].sum(2)
    check('a ClipEnd short of the floor drops the floor shadow (pixels '
          'brighten, none darken)',
          int((d > 1e-4).sum()) > 100 and int((d < -1e-4).sum()) == 0,
          f'+{int((d > 1e-4).sum())} / -{int((d < -1e-4).sum())} px')
    meta2, _a2, _w2 = GSH._shadow_meta(lamp, st)
    check("...and the GPU meta follows the clipped map",
          meta2 is not None and meta2['far'] == 2.2)
    lamp.shadow_clip_end = 0.0
    lamp.shadow_clip_start = 0.5
    LI._SHADOW_CACHE.clear()
    c = _render(sc, st)
    check('ClipSta replaces the near plane; with no caster that near the '
          'picture is unchanged',
          float(lamp.shadow_map.faces[0].near) == 0.5
          and float(lamp.shadow_map.faces[0].far) == far0
          and bool(np.array_equal(c, a)))
    lamp.shadow_clip_start = 3.0
    lamp.shadow_clip_end = 2.0
    LI._SHADOW_CACHE.clear()
    R.render(sc, st)
    check('an End inside Start falls back to the automatic far plane',
          float(lamp.shadow_map.faces[0].near) == 3.0
          and float(lamp.shadow_map.faces[0].far) == far0)
    lamp.shadow_clip_start = lamp.shadow_clip_end = 0.0
    # the SPOT buffer too
    sp = Light(type='SPOT', name='s', position=(-1.3, 0.2, 3.6),
               direction=(0.0, 0.0, -1.0), spot_size=1.2, energy=300.0,
               shadow='MAP', decay='INVERSE_SQUARE')
    sc.lights = [sp]
    LI._SHADOW_CACHE.clear()
    a2 = _render(sc, st)
    sp.shadow_clip_end = 2.2
    LI._SHADOW_CACHE.clear()
    b2 = _render(sc, st)
    d2 = (b2 - a2)[..., :3].sum(2)
    check('a SPOT buffer clips the same way',
          float(sp.shadow_map.far) == 2.2 and int((d2 > 1e-4).sum()) > 50
          and int((d2 < -1e-4).sum()) == 0)
    # per-light cache: the clipped lamp rebuilds, the other rides the cache
    sc.lights = [Light(type='SPOT', name='A', position=(2, -3, 6),
                       direction=(-0.3, 0.4, -0.85), energy=80.0,
                       shadow='MAP', spot_size=1.1, spot_blend=0.2),
                 Light(type='SPOT', name='B', position=(-4, -2, 5),
                       direction=(0.5, 0.3, -0.8), energy=60.0,
                       shadow='MAP', spot_size=0.9, spot_blend=0.2)]
    LI._SHADOW_CACHE.clear()
    R.render(sc, st)
    ma, mb = sc.lights[0].shadow_map, sc.lights[1].shadow_map
    sc.lights[0].shadow_clip_end = 9.0
    R.render(sc, st)
    check("clipping lamp A rebuilds A's map ONLY; B rides the cache",
          sc.lights[0].shadow_map is not ma and sc.lights[1].shadow_map is mb)
    # the GPU twin of the row
    sc_r, st_r = FM.build('shadow buffer clip end (BI ClipEnd)')
    st_r.transparency = 'NONE'
    LI._SHADOW_CACHE.clear()
    cpu, sim, mask, why = _twin(sc_r, st_r)
    _frame_bar('the clipped-map row', cpu, sim, mask, why)
    # import / export
    from ..core import blend279_map as BM
    lm = BM.lamp_map({'kind': 'SPOT', 'name': 'S', 'mode': BM.LA_SHAD_BUF,
                      'clipsta': 0.75, 'clipend': 40.0, 'spotsize': 0.8,
                      'spotblend': 0.1, 'energy': 1.0, 'dist': 20.0}, 279)
    check('lamp_map carries clipsta / clipend into clip_start / clip_end '
          'for a buffer spot',
          lm.get('clip_start') == 0.75 and lm.get('clip_end') == 40.0)
    lm2 = BM.lamp_map({'kind': 'SPOT', 'name': 'S', 'mode': BM.LA_SHAD_RAY,
                       'clipsta': 0.75, 'clipend': 40.0, 'spotsize': 0.8,
                       'spotblend': 0.1, 'energy': 1.0, 'dist': 20.0}, 279)
    check('a ray-shadowed spot carries no clip pair (no buffer to clip)',
          'clip_start' not in lm2)
    from ..core import blend279 as B279
    src = open(B279.__file__, encoding='utf-8').read()
    check("the 2.79 Lamp parser reads 'clipend' beside 'clipsta'",
          "'clipend': 'f'" in src.split('cameras = {}')[0].split("'LA'")[-1])
    lt_e = _export_fake(shadow_clip_start=0.3, shadow_clip_end=12.0)
    check('export_light carries the clip pair',
          lt_e.shadow_clip_start == 0.3 and lt_e.shadow_clip_end == 12.0)


# ------------------------------------------------------------------ (G)


def _stripes(w=8, h=8, period=2):
    """A w x h cookie with vertical stripes (columns alternate)."""
    ck = np.zeros((h, w, 4), f32)
    ck[:, :, 3] = 1.0
    cols = (np.arange(w) // (period // 2)) % 2
    ck[:, cols == 1, :3] = 1.0
    return ck


def _one_column(w=8, h=8):
    ck = np.zeros((h, w, 4), f32)
    ck[:, :, 3] = 1.0
    ck[:, 1, :3] = 1.0
    return ck


def _floor_grid(n=33, half=1.6):
    xs = np.linspace(-half, half, n, dtype=f32)
    X, Y = np.meshgrid(xs, xs)
    P = np.stack([X.ravel(), Y.ravel(), np.zeros(n * n, f32)], axis=1)
    return P, n


def _cookie_grid(lamp, cookie, **kw):
    lamp.cookie = cookie
    for k, v in kw.items():
        setattr(lamp, k, v)
    P, n = _floor_grid()
    pos = np.asarray(lamp.position, f32)
    L = M.normalize(pos[None, :] - P)
    return LI.cookie_factor(lamp, P, L)[:, 0].reshape(n, n)


def test_cookie_transform():
    """(G): the projector map's Offset / Tiling / Angle."""
    lamp = _spot(position=(0.0, 0.0, 2.0), spot_size=1.4)
    lamp.frame_x, lamp.frame_y = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)
    f0 = _cookie_grid(lamp, _stripes(), cookie_extend='REPEAT',
                      cookie_filter='CLOSEST')
    check('cookie_xform_on is False at the identity and the plain lookup '
          'is bitwise (no transform statement)',
          not LI.cookie_xform_on(lamp)
          and bool(np.array_equal(
              f0, _cookie_grid(lamp, _stripes(), cookie_offset=(0.0, 0.0),
                               cookie_tile=(1.0, 1.0), cookie_rotation=0.0))))
    vx0, vy0 = float(f0.var(axis=1).mean()), float(f0.var(axis=0).mean())
    check('vertical stripes vary along x only',
          vx0 > 0.1 and vy0 < 1e-6, f'{vx0:.3f} {vy0:.2e}')
    f90 = _cookie_grid(lamp, _stripes(), cookie_rotation=np.pi / 2)
    vx, vy = float(f90.var(axis=1).mean()), float(f90.var(axis=0).mean())
    check('a 90-degree Angle swaps the stripe axis',
          vy > 0.1 and vx < 1e-6, f'{vx:.2e} {vy:.3f}')
    lamp.cookie_rotation = 0.0
    f1 = _cookie_grid(lamp, _stripes(w=16, period=8), cookie_tile=(1.0, 1.0))
    f2 = _cookie_grid(lamp, _stripes(w=16, period=8), cookie_tile=(2.0, 1.0))
    row1 = f1[f1.shape[0] // 2]
    row2 = f2[f2.shape[0] // 2]
    c1 = int((np.diff(row1) != 0).sum())
    c2 = int((np.diff(row2) != 0).sum())
    check('Tiling 2 doubles the stripe count across the cone',
          c2 >= 2 * c1 - 1 and c1 >= 2, f'{c1} -> {c2}')
    lamp.cookie_tile = (1.0, 1.0)
    fa = _cookie_grid(lamp, _one_column(), cookie_offset=(0.0, 0.0))
    fb = _cookie_grid(lamp, _one_column(), cookie_offset=(0.5, 0.0))
    ra, rb = fa[fa.shape[0] // 2], fb[fb.shape[0] // 2]
    ia, ib = int(np.argmax(ra)), int(np.argmax(rb))
    # the slide spans the cone: 2 * height * tan(half) units wide over a
    # grid of 3.2 units / 32 steps -- half of it is 16.9 samples
    half_slide = 0.5 * (2.0 * 2.0 * np.tan(0.7)) / (3.2 / 32.0)
    check('Offset 0.5 slides the bright column by half the slide',
          ra.max() > 0.5 and rb.max() > 0.5
          and abs(abs(ib - ia) - half_slide) < 2.0,
          f'{ia} -> {ib} (half a slide is {half_slide:.1f} samples)')
    lamp.cookie_offset = (0.0, 0.0)
    fc1 = _cookie_grid(lamp, _stripes(), cookie_extend='CLIP',
                       cookie_tile=(1.0, 1.0))
    fc2 = _cookie_grid(lamp, _stripes(), cookie_extend='CLIP',
                       cookie_tile=(2.0, 2.0))
    check('CLIP still gates after the transform (a smaller slide darkens '
          'more of the cone)',
          int((fc2 == 0.0).sum()) > int((fc1 == 0.0).sum()) + 50,
          f'{int((fc1 == 0.0).sum())} -> {int((fc2 == 0.0).sum())}')
    lamp.cookie_extend = 'AUTO'
    lamp.cookie_tile = (1.0, 1.0)
    # the GLSL function opens with the transform only when live
    spec = {'kind': 'SPOT', 'w': 8, 'h': 8, 'extend': 'EXTEND',
            'filter': 'BILINEAR', 'xform': None}
    plain = GM._cookie_function(0, spec)
    spec['xform'] = tuple(float(v) for v in LI.cookie_xform_consts(
        _spot(cookie_rotation=0.3, cookie_tile=(2.0, 1.0),
              cookie_offset=(0.1, 0.0))))
    xf = GM._cookie_function(0, spec)
    check('the cookie function is the 1.91 text at the identity and opens '
          'with the fourteen transform statements when live',
          'ck_u' not in plain and xf.count('ck_') >= 14
          and xf.find('ck_u') < xf.find('float fx'))
    # GPU parity under the gobo bar, every kind
    ck = FM._cookie_pixels()
    rigs = {
        'SPOT': dict(position=(0.0, -4.0, 6.0), direction=(0.0, 0.45, -0.9),
                     energy=800.0, spot_size=1.0, spot_blend=0.2,
                     decay='INVERSE_SQUARE'),
        'SUN': dict(direction=(-0.5, 0.4, -0.75), energy=3.0),
        'POINT': dict(position=(0.0, -2.0, 5.0), energy=500.0,
                      decay='INVERSE_SQUARE'),
        'AREA': dict(position=(0.0, -3.0, 5.0), direction=(0.0, 0.4, -0.9),
                     energy=200.0, area_size=(2.0, 2.0)),
    }
    for kind, kw in rigs.items():
        for ext, filt in (('REPEAT', 'BILINEAR'), ('CLIP', 'CLOSEST')):
            st = _settings()
            st.shadows = False
            sc = demo_scene(st, with_texture=False)
            lp = Light(type=kind, name='gobo', color=(1.0, 1.0, 1.0),
                       shadow='NONE', **kw)
            lp.cookie = ck
            lp.cookie_extend, lp.cookie_filter = ext, filt
            lp.cookie_rotation = 0.6
            lp.cookie_tile = (1.5, 0.75)
            lp.cookie_offset = (0.2, -0.1)
            sc.lights = [lp]
            cpu, sim, mask, why = _twin(sc, st)
            _frame_bar(f'gobo {kind} {ext} {filt} transformed', cpu, sim,
                       mask, why)
    sc_r, st_r = FM.build('cookie offset / tiling / angle (projector map)')
    st_r.transparency = 'NONE'
    cpu, sim, mask, why = _twin(sc_r, st_r)
    _frame_bar('the cookie-transform row', cpu, sim, mask, why)
    sc2, st2 = FM.build('cookie offset / tiling / angle (projector map)')
    st2.transparency = 'NONE'
    sc2.lights[-1].cookie_rotation = 0.0
    sc2.lights[-1].cookie_tile = (1.0, 1.0)
    sc2.lights[-1].cookie_offset = (0.0, 0.0)
    check('the transform moves the projected picture',
          float(np.abs(cpu - _render(sc2, st2)).max()) > 1e-3)
    # structure and export
    a = _spot(cookie=ck)
    b = _spot(cookie=ck, cookie_rotation=0.2)
    c = _spot(cookie=ck, cookie_tile=(2.0, 2.0))
    check('the transform values are in the cookie plan signature',
          GSH._light_sig(a) != GSH._light_sig(b)
          and GSH._light_sig(a) != GSH._light_sig(c))
    lt_e = _export_fake(cookie_offset=(0.2, 0.1), cookie_tile=(2.0, 3.0),
                        cookie_rotation=0.4, with_cookie=True)
    check('export_light carries offset / tile / rotation with the cookie',
          lt_e.cookie_offset == (0.2, 0.1) and lt_e.cookie_tile == (2.0, 3.0)
          and lt_e.cookie_rotation == 0.4)


# ------------------------------------------------------------------ (H)


def test_light_style():
    """(H): Quake's lightstyles, stepped ten per second of scene time."""
    check('twelve styles, NORMAL first and the string m',
          len(LI.LIGHT_STYLES) == 12 and list(LI.LIGHT_STYLES)[0] == 'NORMAL'
          and LI.LIGHT_STYLES['NORMAL'] == 'm')
    check("every character of every string is a..z",
          all(all('a' <= c <= 'z' for c in s) for s in LI.LIGHT_STYLES.values()))
    check("'m' is exactly 1.0, 'a' 0.0, 'z' 25/12",
          LI.style_char_gain('m') == 1.0 and LI.style_char_gain('a') == 0.0
          and LI.style_char_gain('z') == 25.0 / 12.0)
    lt = Light(type='POINT', light_style='SLOW_STRONG_PULSE')
    s = LI.LIGHT_STYLES['SLOW_STRONG_PULSE']
    check('style_gain steps at 10 Hz: 0.0 and 0.09 read char 0, 0.1 char 1',
          LI.style_gain(lt, 0.0) == LI.style_char_gain(s[0])
          and LI.style_gain(lt, 0.09) == LI.style_char_gain(s[0])
          and LI.style_gain(lt, 0.1) == LI.style_char_gain(s[1]))
    check('...and wraps at the string length',
          LI.style_gain(lt, len(s) * 0.1) == LI.style_char_gain(s[0])
          and LI.style_gain(lt, (len(s) + 3) * 0.1) == LI.style_char_gain(s[3]))
    lt.light_style_rate = 20.0
    check('the Rate scales the step', LI.style_gain(lt, 0.05) == LI.style_char_gain(s[1]))
    check('NORMAL is 1.0 at any time',
          all(LI.style_gain(Light(type='POINT'), t) == 1.0
              for t in (0.0, 0.37, 12.9)))
    # NORMAL is bitwise identity at every frame
    st = _settings()
    sc = demo_scene(st, with_texture=False)
    a = _render(sc, st)
    sc.time = 0.73
    sc.frame = 18
    check('a NORMAL scene renders bitwise at another frame',
          bool(np.array_equal(_render(sc, st), a)))
    # FAST_STROBE at 0.1 ('a') zeroes the lamp -- per site
    def site(label, build, time_a=0.1):
        sc_s, st_s = build()
        sc_s.time = time_a
        LI._SHADOW_CACHE.clear()
        strobe = _render(sc_s, st_s)
        sc_0, st_0 = build()
        sc_0.time = time_a
        for l in sc_0.lights:
            if l.light_style != 'NORMAL':
                l.light_style = 'NORMAL'
                l.energy = 0.0
        LI._SHADOW_CACHE.clear()
        zero = _render(sc_0, st_0)
        check(f"{label}: FAST_STROBE at time 0.1 ('a') is bitwise the lamp "
              'at energy 0', bool(np.array_equal(strobe, zero)))
        sc_m, st_m = build()
        sc_m.time = 0.0                           # 'm'
        LI._SHADOW_CACHE.clear()
        m = _render(sc_m, st_m)
        sc_n, st_n = build()
        for l in sc_n.lights:
            l.light_style = 'NORMAL'
        LI._SHADOW_CACHE.clear()
        check(f"{label}: at time 0.0 ('m') it is bitwise the steady lamp "
              'and differs from the strobe',
              bool(np.array_equal(m, _render(sc_n, st_n)))
              and not np.array_equal(m, strobe))

    def b_point():
        st_ = _settings()
        sc_ = demo_scene(st_, with_texture=False)
        for l in sc_.lights:
            if l.type == 'POINT':
                l.light_style = 'FAST_STROBE'
        return sc_, st_

    def b_beam():
        st_ = _settings()
        st_.spot_cones = True
        st_.shadows = False
        sc_ = demo_scene(st_, with_texture=False)
        sp = Light(type='SPOT', name='beam', position=(0, 4.5, 2.2),
                   direction=(0, -1, -0.25), color=(1, 0.9, 0.7),
                   energy=800.0, spot_size=0.8, spot_blend=0.25,
                   volumetric=1.5, shadow='NONE')
        sp.light_style = 'FAST_STROBE'
        sc_.lights = list(sc_.lights) + [sp]
        return sc_, st_

    def b_screen():
        sc_, st_ = FM.build('Model 3 spotlight fog lobe')
        st_.transparency = 'NONE'
        sc_.lights[-1].light_style = 'FAST_STROBE'
        return sc_, st_

    def b_ambient():
        st_ = _settings()
        sc_ = demo_scene(st_, with_texture=False)
        amb = Light(type='POINT', name='amb', position=(0, 0, 3),
                    color=(0.4, 0.5, 0.9), energy=2.0, shadow='NONE')
        amb.ambient_only = True
        amb.light_style = 'FAST_STROBE'
        sc_.lights = list(sc_.lights) + [amb]
        return sc_, st_

    site('a plain POINT lamp', b_point)
    site('a volumetric beam', b_beam)
    site('the Model 3 screen spot and its fog lobe', b_screen)
    site('an ambient-only lamp', b_ambient)
    # the flare rides the gain at the function level
    st_f = _settings()
    sc_f = demo_scene(st_f, with_texture=False)
    fl = Light(type='POINT', name='flare', position=(0.0, 0.0, 2.5),
               energy=50.0, shadow='NONE', flare=0.8)
    fl.light_style = 'SLOW_STROBE'
    sc_f.lights = [fl]
    g, vp, _v, _e = _gbuf(sc_f)
    sc_f.time = 0.0                              # 'a'
    LI.apply_light_styles(sc_f)
    src_a = R._flare_sources(sc_f, st_f, g, vp)
    sc_f.time = 0.8                              # 'z'
    LI.apply_light_styles(sc_f)
    src_z = R._flare_sources(sc_f, st_f, g, vp)
    check('the lens flare flickers in sync (intensity x gain: 0 on an a, '
          '25/12 on a z)',
          len(src_a) == 1 and len(src_z) == 1
          and src_a[0]['intensity'] == 0.0
          and abs(src_z[0]['intensity'] - 0.8 * 25.0 / 12.0) < 1e-9,
          str([s_['intensity'] for s_ in src_a + src_z]))
    # the GPU: the energy texel is the CPU's own float32, the plan a HIT
    st = _settings()
    sc = demo_scene(st, with_texture=False)
    for l in sc.lights:
        l.shadow = 'NONE'
        l.light_style = 'SLOW_STRONG_PULSE'
    sc.time = 0.3
    run, _g = _gpu_rig(sc, st)
    GSH._PLAN_CACHE.clear()
    img1, p1, why1, atl1 = run()
    check('a styled frame plans', p1 is not None, str(why1))
    lights = LI.select_lights(sc.lights, st)
    job = R.ShadeJob(sc, st, {}, None, *R.camera_matrices(sc.camera, W, H)[::3], W, H)
    tex = GM.pack_light_texels(lights, job)
    ok = True
    for i, l in enumerate(lights):
        gain = LI.style_gain(l, sc.time)
        sign = -1.0 if l.negative else 1.0
        if l.type in ('SUN', 'HEMI'):
            want = f32(sign * (float(l.energy) * gain))
        else:
            want = f32(sign * (float(l.energy) * gain) / (4.0 * np.pi))
        ok = ok and float(tex[0, i * GM.LIGHT_TEXEL_STRIDE, 3]) == float(want) \
            and gain != 1.0
    check('the energy texel equals np.float32(sign * energy * gain / 4pi) '
          'bitwise (gain != 1)', ok)
    if p1 is not None:
        sc.time = 0.5
        img2, p2, _w2, atl2 = run()
        check('the next frame of the flicker is a plan-cache HIT with fresh '
              'texels and a moved picture',
              p2 is p1 and atl2['hal_lights'][0] != atl1['hal_lights'][0]
              and float(np.abs(img2 - img1).max()) > 1e-4)
        cpu2 = _render(sc, st)
        cov = _g.tri >= 0
        check('...at parity with the CPU frame',
              float(np.abs(img2[cov] - cpu2[cov][:, :3]).max()) < 6e-3)
    sc_r, st_r = FM.build('Quake light style (slow strong pulse)')
    st_r.transparency = 'NONE'
    cpu, sim, mask, why = _twin(sc_r, st_r)
    _frame_bar('the light-style row', cpu, sim, mask, why)
    s0 = GSH._light_sig(Light(type='POINT'))
    s1 = GSH._light_sig(Light(type='POINT', light_style='FLICKER_A',
                              light_style_rate=4.0))
    check('the style and its rate are not structure (value road only)', s0 == s1)
    lt_e = _export_fake(light_style='CANDLE_A', light_style_rate=7.0)
    check('export_light carries the style and rate',
          lt_e.light_style == 'CANDLE_A' and lt_e.light_style_rate == 7.0)


# ------------------------------------------------------- export and the UI


def _old_fake_hs(**extra):
    """The gobo test's fake hs (R219), WITHOUT the R253 fields."""
    hs = types.SimpleNamespace(
        decay='DEFAULT', decay_start=0.0, decay_end=25.0,
        decay_ld1=0.0, decay_ld2=0.0, bi_sphere=False,
        shadow='NONE', shadow_map_size=0, shadow_bias=0.0,
        shadow_softness=1.0, shadow_samples=4, shadow_density=1.0,
        shadow_color=(0.0, 0.0, 0.0), negative=False,
        diffuse_only=False, specular_only=False, ambient_only=False,
        hotspot=0.0, volumetric=0.0, volumetric_occlusion=False,
        flare=0.0, flare_scale=1.0, flare_streaks=6, flare_rings=1,
        flare_ghosts=6, caustics=0.0, caustics_scale=4.0,
        caustics_speed=1.0, exclude_mode='EXCLUDE',
        exclude_collection=None, cookie=None,
        cookie_strength=0.8, cookie_scale=10.0,
        cookie_extend='CLIP', cookie_filter='CUBIC')
    for k, v in extra.items():
        setattr(hs, k, v)
    return hs


def _export_fake(kind='SPOT', with_cookie=False, **fields):
    """export_light on a fake lamp whose hs carries `fields`."""
    from . import fakebpy
    fakebpy.install()
    from .. import export as EX
    from .. import compat
    hs = _old_fake_hs(**fields)
    if with_cookie:
        hs.cookie = types.SimpleNamespace(name_full='gobo.png')
    data = types.SimpleNamespace(type=kind, color=(1.0, 1.0, 1.0),
                                 energy=10.0, halcyon=hs,
                                 shadow_soft_size=0.25, spot_size=1.0,
                                 spot_blend=0.2, angle=0.01)
    ob = types.SimpleNamespace(name='Lamp', data=data)
    old = compat.image_pixels
    compat.image_pixels = lambda _im: np.ones((4, 4, 4), f32)
    try:
        return EX.export_light(ob, np.eye(4, dtype=np.float32))
    finally:
        compat.image_pixels = old


class _Layout:
    """A recording layout: enough of bpy's UILayout for a panel draw."""

    def __init__(self, rec):
        self.rec = rec
        self.active = True
        self.use_property_split = False

    def prop(self, data, name, **kw):
        self.rec.append(name)

    def column(self, **kw):
        return _Layout(self.rec)

    row = column
    box = column

    def separator(self, **kw):
        pass

    def label(self, **kw):
        self.rec.append('label:' + str(kw.get('text', ''))[:30])

    def template_ID(self, data, name, **kw):
        self.rec.append(name)

    def operator(self, idn, **kw):
        self.rec.append('op:' + idn)
        return types.SimpleNamespace(scope=None)


def test_export_and_ui():
    """Export with and without the new fields, the tooltip audit of the
    light panel (the node audit's sibling), the dataclass / PropertyGroup
    default agreement, and the panel drawing every new dial."""
    lt = _export_fake(
        spot_square=True, specular_color=(0.2, 0.4, 0.6),
        hemi_ground=(0.1, 0.2, 0.3), decay_near_start=1.0,
        decay_near_end=2.0, shadow_clip_start=0.5, shadow_clip_end=9.0,
        cookie_offset=(0.1, 0.2), cookie_tile=(2.0, 3.0),
        cookie_rotation=0.3, light_style='CANDLE_B', light_style_rate=5.0,
        with_cookie=True)
    check('export_light with a full fake hs carries every new field',
          lt.spot_square is True and lt.specular_color == (0.2, 0.4, 0.6)
          and lt.hemi_ground == (0.1, 0.2, 0.3)
          and lt.decay_near_start == 1.0 and lt.decay_near_end == 2.0
          and lt.shadow_clip_start == 0.5 and lt.shadow_clip_end == 9.0
          and lt.cookie_offset == (0.1, 0.2) and lt.cookie_tile == (2.0, 3.0)
          and lt.cookie_rotation == 0.3 and lt.light_style == 'CANDLE_B'
          and lt.light_style_rate == 5.0)
    lt2 = _export_fake(kind='POINT', with_cookie=True)
    d = Light()
    check("the gobo test's older fake hs (no new fields) still exports, at "
          'the dataclass defaults',
          all(getattr(lt2, f) == getattr(d, f) for f in NEW_FIELDS))
    # the fast path with halcyon=None
    from .. import export as EX
    inst_ob = types.SimpleNamespace(
        type='LIGHT', name='Lamp',
        data=types.SimpleNamespace(type='SUN', color=(1.0, 0.5, 0.25),
                                   energy=7.0, halcyon=None,
                                   shadow_soft_size=0.1))
    lt3 = EX.export_light(inst_ob, np.eye(4, dtype=np.float32))
    check('export_light with halcyon=None leaves every new field at its '
          'default', all(getattr(lt3, f) == getattr(d, f) for f in NEW_FIELDS))
    # the tooltip audit
    from . import fakebpy
    fakebpy.install()
    from .. import properties as PR
    ann = PR.HalcyonLightSettings.__annotations__
    thin = [n for n in NEW_FIELDS
            if len(str(ann[n].kw.get('description') or '')) < 40]
    check('every new light-panel property carries a tooltip of 40+ chars',
          not thin, ', '.join(thin))
    thin_i = [it[0] for it in PR.LIGHT_STYLE if len(it[2]) < 12]
    thin_i += [it[0] for it in PR.FALLOFF[-2:] if len(it[2]) < 12]
    check('every LIGHT_STYLE item and both new FALLOFF items carry a line '
          'of 12+ chars', not thin_i, ', '.join(thin_i))
    check('LIGHT_STYLE lists exactly the twelve LIGHT_STYLES keys, in order',
          [it[0] for it in PR.LIGHT_STYLE] == list(LI.LIGHT_STYLES))
    # dataclass vs PropertyGroup defaults
    mism = []
    for f in NEW_FIELDS:
        pd = ann[f].kw.get('default')
        dd = getattr(d, f)
        pd_t = tuple(pd) if isinstance(pd, (tuple, list)) else pd
        dd_t = tuple(dd) if isinstance(dd, (tuple, list)) else dd
        if pd_t != dd_t:
            mism.append(f'{f}: {pd_t!r} vs {dd_t!r}')
    check('the Light dataclass defaults equal the PropertyGroup defaults '
          'field by field', not mism, '; '.join(mism))
    # the panel draws every new dial, for every lamp type
    from .. import ui as UI
    defaults = {k: (tuple(v.kw.get('default')) if isinstance(
        v.kw.get('default'), (tuple, list)) else v.kw.get('default'))
        for k, v in ann.items()}
    seen_all = set()
    for kind, hemi in (('POINT', False), ('SUN', False), ('SUN', True),
                       ('SPOT', False), ('AREA', False)):
        hs = types.SimpleNamespace(**defaults)
        hs.hemi = hemi
        hs.cookie = None
        for dec in ('DEFAULT', 'CUSTOM', 'NOMINAL_INVERSE', 'GX_STEEP'):
            hs.decay = dec
            light = types.SimpleNamespace(type=kind, color=(1, 1, 1),
                                          energy=10.0, halcyon=hs)
            rec = []
            panel = UI.HALCYON_PT_light()
            panel.layout = _Layout(rec)
            ctx = types.SimpleNamespace(engine=UI.ENGINE, light=light,
                                        scene=None, object=None)
            try:
                UI.HALCYON_PT_light.draw(panel, ctx)
            except Exception as e:                              # noqa: BLE001
                check(f'the light panel draws for a {kind} lamp ({dec})',
                      False, repr(e))
                continue
            seen_all.update(rec)
            if kind == 'SPOT':
                check(f'a SPOT panel ({dec}) shows Square, the clip pair and '
                      'the near pair',
                      'spot_square' in rec and 'shadow_clip_start' in rec
                      and 'decay_near_start' in rec)
            if kind == 'AREA':
                check('an AREA panel shows no near pair, no Square, no clip',
                      'decay_near_start' not in rec and 'spot_square' not in rec
                      and 'shadow_clip_start' not in rec)
            if kind == 'SUN':
                check(f'a SUN panel (hemi {hemi}) shows the ground colour and '
                      'no near pair (directional lamps never attenuate)',
                      'hemi_ground' in rec and 'decay_near_start' not in rec)
    check('across the lamp kinds every new dial is drawn',
          all(f in seen_all for f in NEW_FIELDS),
          ', '.join(f for f in NEW_FIELDS if f not in seen_all))


def main():
    utf8_console()
    FAILS.clear()
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        print(t.__name__)
        try:
            t()
        except Exception:                                       # noqa: BLE001
            traceback.print_exc()
            FAILS.append(f'{t.__name__} raised')
    print()
    if FAILS:
        print(f'{len(FAILS)} failure(s): ' + ', '.join(FAILS))
        return 1
    print('all R253 light-option checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
