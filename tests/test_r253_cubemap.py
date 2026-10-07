"""R253 (1.92.0): the Cube Map world (CUBEMAP) -- the 1990s skybox.

Six square faces around the eye from one packed image (a horizontal or
vertical cross, a 6:1 or 1:6 strip) or six image slots named the OpenGL
(+X -X +Y -Y +Z -Z) or the Quake 2 / Half-Life (rt lf up dn bk ft) way;
NEAREST or bilinear INSIDE a face and never across a seam; the world's
own Rotation, Tint and Strength. Proved on BOTH roads: the face rule and
the seam law at the function level (core/sky.cube_face_uv / cube_sample
against the GLSL `hal_sky_sample_cube` through the simulator, bitwise on
4006 directions), the frame through test_r251_sky_camera's twin harness
(`gpu/sky.simulate` against `render._background_image`, d == 0.0), the
fake device, the refusal by name, the reflections' CPU road, the wiring
(fakebpy) and the neutrality of every new field at its default.

Runs alone::

    python -m halcyon.tests.test_r253_cubemap
"""
import contextlib
import io
import math
import re
import sys
import traceback

import numpy as np

from ..core import raster as CR
from ..core import render as R
from ..core import sky as SK
from ..core.scene import ImageBuffer, World
from ..core.texture import Texture
from ..gpu import shade as GSH
from ..gpu import sky as GSKY
from ..shaders.compiler import try_compile
from . import utf8_console
from .scenebuild import demo_scene
from .test_r251_sky_camera import (H, W, fake_device_checks,
                                   fake_device_render, settings, twin_d)

FAILS = []
f32 = np.float32


def check(name, cond, extra=''):
    ok = bool(cond)
    print(f'  {"ok  " if ok else "FAIL"} {name}  {extra}' if extra
          else f'  {"ok  " if ok else "FAIL"} {name}')
    if not ok:
        FAILS.append(name)
    return ok


# ------------------------------------------------------------- the fixtures

S = 8
SLOTS = ('px', 'nx', 'py', 'ny', 'pz', 'nz')
#: Blender-axis unit vectors in GL face order (+X -X +Y -Y +Z -Z of GL)
AX = {'+X': (1, 0, 0), '-X': (-1, 0, 0), '+Y': (0, 1, 0), '-Y': (0, -1, 0),
      '+Z': (0, 0, 1), '-Z': (0, 0, -1)}
#: the Blender direction each GL face looks along, by atlas face index
FACE_DIR = ['+X', '-X', '+Z', '-Z', '-Y', '+Y']
#: in-face axes of the canonical GL faces in Blender terms (table 8.19):
#: u (columns) rises toward the first, v (rows, bottom-up) toward the second
FACE_UV = {0: ('+Y', '+Z'), 1: ('-Y', '+Z'), 2: ('+X', '+Y'),
           3: ('+X', '-Y'), 4: ('+X', '+Z'), 5: ('-X', '+Z')}
#: Quake 2 (gl_warp.c): slot -> (Blender axis, s direction, t-top direction)
QUAKE = {0: ('+X', '-Y', '+Z'),      # rt
         1: ('+Y', '+X', '+Z'),      # lf
         2: ('+Z', '-Y', '-X'),      # up
         3: ('-Z', '-Y', '+X'),      # dn
         4: ('-X', '+Y', '+Z'),      # bk
         5: ('-Y', '-X', '+Z')}      # ft


def vec(name, toward=None, k=0.9):
    """A Blender direction: the axis `name`, leaning `k` toward `toward`."""
    d = np.asarray(AX[name], np.float64)
    if toward is not None:
        d = d + k * np.asarray(AX[toward], np.float64)
    return (d / np.linalg.norm(d)).astype(f32)


def cube_faces(n=S):
    """Six n x n RGBA faces: red = face / 5, green = column / (n-1),
    blue = row / (n-1) (row 0 at the bottom, Blender's image order)."""
    out = []
    for f in range(6):
        yy, xx = np.mgrid[0:n, 0:n]
        a = np.zeros((n, n, 4), f32)
        a[..., 0] = f / 5.0
        a[..., 1] = xx / (n - 1.0)
        a[..., 2] = yy / (n - 1.0)
        a[..., 3] = 1.0
        out.append(a)
    return out


FACES = cube_faces()
STACK = np.concatenate(FACES, axis=0)        # the canonical atlas (6S, S, 4)


def pack(layout, faces=None):
    """The six faces packed into a `layout` image (an ImageBuffer's pixel
    order, row 0 at the bottom), each cell in the GL image orientation
    less the layout's own turn (the VCROSS -Z is upside down)."""
    faces = FACES if faces is None else faces
    cols, rows, cells = SK.CUBE_LAYOUTS[layout]
    n = faces[0].shape[0]
    w, h = cols * n, rows * n
    px = np.zeros((h, w, 4), f32)
    px[..., 3] = 1.0
    for f, name in enumerate(SK.CUBE_FACES):
        cell = cells[name]
        c, r = int(cell[0]), int(cell[1])
        turns = int(cell[2]) if len(cell) > 2 else 0
        pic = np.rot90(faces[f], -turns) if turns else faces[f]
        px[h - (r + 1) * n:h - r * n, c * n:(c + 1) * n] = pic
    return px


def rig(ss=1, world=None, source='SINGLE', layout='HSTRIP', faces=None,
        **kw):
    """(scene, settings, textures): the demo scene under a CUBEMAP world
    built from `faces` (the gradient six by default) on the SINGLE road
    (`layout`), the SIX road, or with no image at all ('NONE')."""
    st = settings(ss=ss, **kw)
    sc = demo_scene(st, with_texture=False)
    sc.world.mode = 'CUBEMAP'
    faces = FACES if faces is None else faces
    textures = {}
    if source == 'SIX':
        sc.world.cube_source = 'SIX'
        for slot, pic in zip(SLOTS, faces):
            buf = ImageBuffer(name=f'cube_{slot}', pixels=pic,
                              colorspace='Linear')
            sc.images[buf.name] = buf
            setattr(sc.world, f'cube_image_{slot}', buf)
            textures[buf.name] = Texture(pic, name=buf.name,
                                         colorspace='Linear')
    elif source == 'SINGLE':
        px = pack(layout, faces)
        buf = ImageBuffer(name='cubesky', pixels=px, colorspace='Linear')
        sc.images['cubesky'] = buf
        sc.world.env_image = buf
        sc.world.cube_layout = layout
        textures['cubesky'] = Texture(px, name='cubesky',
                                      colorspace='Linear')
    for k, v in (world or {}).items():
        setattr(sc.world, k, v)
    return sc, st, textures


def world_for(px, layout='AUTO', **kw):
    """A bare World + textures for a packed image `px`."""
    w = World()
    w.mode = 'CUBEMAP'
    w.cube_layout = layout
    w.env_image = ImageBuffer(name='img', pixels=px, colorspace='Linear')
    for k, v in kw.items():
        setattr(w, k, v)
    return w, {'img': Texture(px, name='img', colorspace='Linear')}


def world_six(pics, conv='OPENGL', **kw):
    w = World()
    w.mode = 'CUBEMAP'
    w.cube_source = 'SIX'
    w.cube_convention = conv
    textures = {}
    for slot, pic in zip(SLOTS, pics):
        buf = ImageBuffer(name=f'six_{slot}', pixels=pic, colorspace='Linear')
        setattr(w, f'cube_image_{slot}', buf)
        textures[buf.name] = Texture(pic, name=buf.name, colorspace='Linear')
    for k, v in kw.items():
        setattr(w, k, v)
    return w, textures


def cpu_cube(atlas_px, dirs, filt):
    """The CPU sampler on the atlas: (N, 3) float32."""
    face, u, v = SK.cube_face_uv(dirs)
    return SK.cube_sample(atlas_px, face, u, v,
                          'NEAREST' if filt == 0 else 'BILINEAR')[:, :3]


_GLSL_PROG = {}


def glsl_cube(atlas_px, dirs, filt):
    """The GLSL `hal_sky_sample_cube` -- the shipped text, cut from
    gpu/sky.SOURCE -- through the simulator on per-lane directions, the
    params texel bound exactly as the pass binds it: (N, 3) float32 or
    None (the caller checks by name)."""
    if 'prog' not in _GLSL_PROG:
        m = re.search(r'vec3 hal_sky_sample_cube\(vec3 d, int filt\)\n\{.*?\n\}\n',
                      GSKY.SOURCE, re.S)
        fn = m.group(0) if m else ''
        wrap = ('uniform sampler2D hal_sky_env;\n' + GSKY._accessors()
                + '\nuniform vec3 dir;\nuniform int filt;\nout vec4 Color;\n'
                'vec4 hal_sky_fetch(int x, int y) { return texelFetch('
                'hal_sky_env, ivec2(x, y), 0); }\n' + fn
                + 'void main() { Color = vec4(hal_sky_sample_cube(dir, filt), '
                '1.0); }\n')
        prog, err = try_compile(wrap, 'GLSL')
        if prog is None:
            print('  [glsl_cube] compile failed:', err)
        _GLSL_PROG['prog'] = prog
    prog = _GLSL_PROG['prog']
    if prog is None:
        return None
    n = int(dirs.shape[0])
    params = GSKY.pack_params({'hal_sky_cube': (float(atlas_px.shape[1]),
                                                float(filt), 0.0)})
    uni = {'hal_sky_env': Texture(atlas_px, colorspace='Non-Color',
                                  filt='NEAREST', wrap='EXTEND'),
           'hal_sky_params': Texture(params, colorspace='Non-Color',
                                     filt='NEAREST', wrap='EXTEND'),
           'dir': np.asarray(dirs, f32), 'filt': np.full(n, filt, np.int32)}
    outs, _disc = prog.run(uni, {}, n)
    return np.asarray(outs['Color'], f32)[:, :3]


def captured(fn, *a, **kw):
    """(result, printed) with the once-set cleared first."""
    SK._CUBE_PRINTED.clear()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn(*a, **kw)
    return out, buf.getvalue()


# ===================================================== the face rule


def test_cube_face_rule():
    """The OpenGL 4.6 table 8.19 rule on the Z-up direction swapped to GL
    axes: Blender +X -X +Y -Y +Z -Z land on GL faces 0 1 5 4 2 3, the
    in-face axes read as the table says, the ties go X over Y over Z --
    on the CPU and through the shipped GLSL, bitwise."""
    axes = np.array([AX[k] for k in ('+X', '-X', '+Y', '-Y', '+Z', '-Z')], f32)
    face, u, v = SK.cube_face_uv(axes)
    check('Blender +X -X +Y -Y +Z -Z land on GL faces 0 1 5 4 2 3',
          face.tolist() == [0, 1, 5, 4, 2, 3], str(face.tolist()))
    check('an axis hits the centre of its face (u = v = 0.5)',
          np.array_equal(u, np.full(6, 0.5, f32))
          and np.array_equal(v, np.full(6, 0.5, f32)), f'{u} {v}')
    got = cpu_cube(STACK, axes, 0)
    check('NEAREST at the six axes returns the six faces\' own red (= face / 5)',
          np.array_equal(np.round(got[:, 0] * 5).astype(int),
                         np.array([0, 1, 5, 4, 2, 3])))
    # the in-face axes: lean 0.9 toward the u axis -> the last column, away
    # -> the first; the same for v (rows, bottom-up)
    for f in range(6):
        ua, va = FACE_UV[f]
        dirs = np.stack([vec(FACE_DIR[f], ua), vec(FACE_DIR[f], ua, -0.9),
                         vec(FACE_DIR[f], va), vec(FACE_DIR[f], va, -0.9)])
        fc, _u, _v = SK.cube_face_uv(dirs)
        c = cpu_cube(STACK, dirs, 0)
        check(f'face {f} ({FACE_DIR[f]}): u rises toward {ua}, v toward {va} '
              '(green = column, blue = row)',
              fc.tolist() == [f] * 4 and c[0, 1] == 1.0 and c[1, 1] == 0.0
              and c[2, 2] == 1.0 and c[3, 2] == 0.0,
              f'faces {fc.tolist()} g {c[:2, 1]} b {c[2:, 2]}')
    ties = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], f32)
    tf, _, _ = SK.cube_face_uv(ties / np.sqrt(2.0))
    check('the tie rule at (1,1,0) / (1,0,1) / (0,1,1) picks X, X, Y (faces 0, 0, 2)',
          tf.tolist() == [0, 0, 2], str(tf.tolist()))
    g = glsl_cube(STACK, ties / np.sqrt(2.0), 0)
    check('...and the GLSL picks the same faces',
          g is not None and np.round(g[:, 0] * 5).astype(int).tolist() == [0, 0, 2],
          str(None if g is None else g[:, 0]))
    # the function-level twin: 4000 random directions and the axes, both
    # filters, bitwise (the probe's 0.0, now the shipped text)
    rng = np.random.default_rng(3)
    d = rng.normal(size=(4000, 3)).astype(f32)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    d = np.concatenate([d, axes]).astype(f32)
    for filt, name in ((0, 'NEAREST'), (1, 'BILINEAR')):
        g = glsl_cube(STACK, d, filt)
        c = cpu_cube(STACK, d, filt)
        dm = float(np.abs(g - c).max()) if g is not None else -1.0
        check(f'{name}: the shipped GLSL sampler is bitwise the CPU on 4006 '
              'directions', g is not None and dm == 0.0, f'max {dm}')
    check('the face rule returns float32 u, v and int64 faces',
          u.dtype == np.float32 and v.dtype == np.float32
          and face.dtype == np.int64)


# ====================================================== the layouts


def test_cube_layouts():
    """The four single-image layouts cut to the same atlas; AUTO reads the
    aspect; a 5x3-cell image is refused to the solid colour, once, by
    size; the VCROSS -Z face is turned 180 degrees."""
    atlases = {}
    for lay in ('HCROSS', 'VCROSS', 'HSTRIP', 'VSTRIP'):
        w, tex = world_for(pack(lay), layout=lay)
        at = SK.cube_atlas(w, tex)
        atlases[lay] = None if at is None else np.asarray(at.pixels)
        check(f'{lay}: the atlas is the six faces stacked in GL order, bitwise',
              at is not None and at.pixels.shape == (6 * S, S, 4)
              and np.array_equal(at.pixels, STACK))
        wa, texa = world_for(pack(lay), layout='AUTO')
        ata = SK.cube_atlas(wa, texa)
        check(f'AUTO resolves the {lay} aspect ({pack(lay).shape[1]}x'
              f'{pack(lay).shape[0]}) to the same atlas',
              ata is not None and np.array_equal(ata.pixels, STACK)
              and SK.cube_layout_auto(pack(lay).shape[1], pack(lay).shape[0]) == lay)
    check('cube_layout_auto names 4:3 / 3:4 / 6:1 / 1:6 and nothing else',
          SK.cube_layout_auto(4000, 3000) == 'HCROSS'
          and SK.cube_layout_auto(300, 400) == 'VCROSS'
          and SK.cube_layout_auto(96, 16) == 'HSTRIP'
          and SK.cube_layout_auto(16, 96) == 'VSTRIP'
          and SK.cube_layout_auto(1024, 512) is None
          and SK.cube_layout_auto(0, 0) is None)
    # the VCROSS bottom cell is upside down in the file and righted here
    naive = pack('VCROSS')
    cols, rows, cells = SK.CUBE_LAYOUTS['VCROSS']
    c, r = cells['NZ'][0], cells['NZ'][1]
    h = rows * S
    naive[h - (r + 1) * S:h - r * S, c * S:(c + 1) * S] = FACES[5]   # NOT turned
    wn, texn = world_for(naive, layout='VCROSS')
    atn = SK.cube_atlas(wn, texn)
    check('the VCROSS -Z cell is read turned 180 degrees (a cell stored '
          'upright lands upside down in the atlas, a cell stored upside '
          'down lands upright)',
          atn is not None and np.array_equal(atn.pixels[5 * S:6 * S],
                                             np.rot90(FACES[5], 2))
          and not np.array_equal(atn.pixels[5 * S:6 * S], FACES[5]))
    # a 5x3-cell image: nothing -- the solid colour x strength, by name, once
    bad = np.ones((3 * S, 5 * S, 4), f32)
    wb, texb = world_for(bad, layout='AUTO', strength=1.5,
                         color=(0.2, 0.3, 0.4))
    atb, printed = captured(SK.cube_atlas, wb, texb)
    check('a 40x24 image under AUTO is refused (None) and the once-printed '
          'line names the size', atb is None and '[Halcyon] cube map:' in printed
          and '40x24' in printed, printed.strip())
    _atb2, printed2 = captured(lambda: (SK.cube_atlas(wb, texb),
                                        SK.cube_atlas(wb, texb)))
    check('...printed once per message, not per call',
          printed2.count('[Halcyon] cube map:') == 1, printed2.strip())
    wh, texh = world_for(bad, layout='HCROSS')
    ath, printedh = captured(SK.cube_atlas, wh, texh)
    check('a 40x24 image under HCROSS names the expected cell shape',
          ath is None and '4x3' in printedh and '40x24' in printedh,
          printedh.strip())
    dirs = np.array([AX['+X'], AX['+Z'], AX['-Y']], f32)
    got = SK.evaluate(wb, dirs, texb)
    want = (SK.solid(wb, dirs) * 1.5).astype(f32)
    check('...and the sky is the solid colour x strength, bitwise (hdri\'s '
          'rule)', np.array_equal(got, want))
    wn2 = World()
    wn2.mode = 'CUBEMAP'
    _at, printed_none = captured(SK.cube_atlas, wn2, {})
    check('no image at all is silent (hdri\'s rule: the solid colour, no '
          'complaint)', _at is None and printed_none == '')


# ===================================================== six images


def test_cube_six_images():
    """The SIX road: OPENGL equals the strip atlas bitwise; QUAKE2 lands
    rt lf up dn bk ft on +X +Y +Z -Z -X -Y and the gradient corners pin
    every face's s / t orientation (gl_warp.c); mixed sizes refuse."""
    w6, tex6 = world_six(FACES, 'OPENGL')
    at6 = SK.cube_atlas(w6, tex6)
    ws, texs = world_for(pack('HSTRIP'), layout='HSTRIP')
    ats = SK.cube_atlas(ws, texs)
    check('six OPENGL slots build the HSTRIP atlas bitwise',
          at6 is not None and ats is not None
          and np.array_equal(at6.pixels, ats.pixels))
    # Quake: slot i's picture has red = i / 5, green = s (columns), blue =
    # t with the picture's top at blue 1 (rows bottom-up)
    wq, texq = world_six(FACES, 'QUAKE2')
    atq = SK.cube_atlas(wq, texq)
    check('six QUAKE2 slots build an atlas', atq is not None)
    if atq is None:
        return
    for slot, (axis, s_dir, t_dir) in QUAKE.items():
        label = SK.CUBE_SLOT_LABELS['QUAKE2'][slot]
        dirs = np.stack([vec(axis), vec(axis, s_dir), vec(axis, s_dir, -0.9),
                         vec(axis, t_dir), vec(axis, t_dir, -0.9)])
        c = SK.evaluate(wq, dirs, texq)
        slot_hit = int(round(float(c[0, 0]) * 5))
        check(f'{label} lands on Blender {axis}', slot_hit == slot,
              f'slot {slot_hit}')
        check(f'{label}: s runs toward {s_dir} and the picture\'s top toward '
              f'{t_dir} (gl_warp.c st_to_vec, t = 1 - t)',
              c[1, 1] == 1.0 and c[2, 1] == 0.0 and c[3, 2] == 1.0
              and c[4, 2] == 0.0, f'g {c[1:3, 1]} b {c[3:5, 2]}')
    check('every Quake face is mirrored against GL\'s and up / dn turn (the '
          'derived table)',
          all(m for _f, _t, m in SK.CUBE_CONVENTIONS['QUAKE2'])
          and [t for _f, t, _m in SK.CUBE_CONVENTIONS['QUAKE2']] == [0, 0, 1, 3, 0, 0]
          and [f for f, _t, _m in SK.CUBE_CONVENTIONS['QUAKE2']]
          == ['PX', 'NZ', 'PY', 'NY', 'NX', 'PZ'])
    # two sizes: refused, once, naming both
    small = cube_faces(4)
    mixed = list(FACES)
    mixed[3] = small[3]
    wm, texm = world_six(mixed, 'OPENGL')
    atm, printed = captured(SK.cube_atlas, wm, texm)
    check('six faces of two sizes refuse to the solid colour and the line '
          'names both sizes', atm is None and '8x8' in printed
          and '4x4' in printed, printed.strip())
    rect = list(FACES)
    rect[0] = np.ones((8, 16, 4), f32)
    wr, texr = world_six(rect, 'OPENGL')
    atr, printedr = captured(SK.cube_atlas, wr, texr)
    check('a non-square face refuses by size', atr is None and '16x8' in printedr,
          printedr.strip())
    we, texe = world_six(FACES, 'QUAKE2')
    we.cube_image_py = None
    we.cube_image_nz = None
    ate, printede = captured(SK.cube_atlas, we, texe)
    check('an empty slot refuses and the line names it in the convention\'s '
          'words', ate is None and 'up' in printede and 'ft' in printede,
          printede.strip())
    dirs = np.array([AX['+X'], AX['-Z']], f32)
    got = SK.evaluate(we, dirs, texe)
    check('...and the sky is the solid colour (strength 1)',
          np.array_equal(got, SK.solid(we, dirs).astype(f32)))


# =============================================== per-face orientation


def test_cube_face_orient():
    """cube_face_rot[i] / cube_face_flip[i] permute the slot's texels
    exactly as np.rot90(face, k) then np.fliplr; the cache key follows
    them; the cache holds at most four atlases and serves a hit by
    identity."""
    SK._CUBE_CACHE.clear()
    rot = (1, 2, 3, 0, 1, 0)
    flip = (False, True, True, False, False, True)
    w, tex = world_for(pack('HCROSS'), layout='HCROSS', cube_face_rot=rot,
                       cube_face_flip=flip)
    at = SK.cube_atlas(w, tex)
    ok = at is not None
    for f in range(6):
        want = np.rot90(FACES[f], rot[f]) if rot[f] else FACES[f]
        if flip[f]:
            want = np.fliplr(want)
        ok &= bool(np.array_equal(at.pixels[f * S:(f + 1) * S], want)) if at else False
    check('each slab is np.rot90(face, k) then np.fliplr(...) as the dials '
          'say (array_equal)', ok)
    w0, tex0 = world_for(pack('HCROSS'), layout='HCROSS')
    at0 = SK.cube_atlas(w0, tex0)
    check('the atlas name (the GPU content key) changes with the dials',
          at0 is not None and at is not None and at0.name != at.name
          and at.name.endswith(':123010:011001'),
          f'{None if at is None else at.name} vs {None if at0 is None else at0.name}')
    check('a second call with the same dials is the cached object',
          SK.cube_atlas(w, tex) is at and SK.cube_atlas(w0, tex0) is at0)
    # the SIX road's dials act per SLOT (before the convention's own turn
    # they would be meaningless to a Quake file): rt turned once under
    # QUAKE2 is fliplr(rot90(rot90(pic, 0) mirrored, 1)) -> the slab
    wq, texq = world_six(FACES, 'QUAKE2', cube_face_rot=(1, 0, 0, 0, 0, 0))
    atq = SK.cube_atlas(wq, texq)
    want_rt = np.rot90(np.fliplr(FACES[0]), 1)
    check('under QUAKE2 a slot turn applies AFTER the convention\'s '
          'permutation (rt: rot90(fliplr(pic), 1) on the +X slab)',
          atq is not None and np.array_equal(atq.pixels[0:S], want_rt))
    SK._CUBE_CACHE.clear()
    for k in range(6):
        wk, texk = world_for(pack('HSTRIP'), layout='HSTRIP',
                             cube_face_rot=(k % 4, 0, 0, 0, 0, k // 4))
        SK.cube_atlas(wk, texk)
    check(f'the cache holds at most {SK.CUBE_CACHE_MAX} atlases',
          len(SK._CUBE_CACHE) == SK.CUBE_CACHE_MAX)
    # a recycled source id never serves a stale atlas: the hit re-checks
    # the source array by identity
    w1, tex1 = world_for(pack('HSTRIP'), layout='HSTRIP')
    a1 = SK.cube_atlas(w1, tex1)
    key = next(k for k, v in SK._CUBE_CACHE.items() if v[1] is a1)
    other = pack('HSTRIP')
    other[..., 0] = 1.0
    tex1['img'] = Texture(other, name='img', colorspace='Linear')
    SK._CUBE_CACHE[key] = (SK._CUBE_CACHE[key][0], a1)
    forged = dict(SK._CUBE_CACHE)
    SK._CUBE_CACHE.clear()
    SK._CUBE_CACHE.update({
        (key[0], key[1], key[2], key[3], key[4],
         ((key[5][0][0], id(tex1['img'].pixels), key[5][0][2], key[5][0][3]),)):
        forged[key]})
    a2 = SK.cube_atlas(w1, tex1)
    check('a cache entry whose source array is another object is rebuilt, '
          'not served', a2 is not a1 and float(a2.pixels[..., 0].min()) == 1.0)
    SK._CUBE_CACHE.clear()


# ============================================================ the seams


def test_cube_seams():
    """The seam law: NEAREST reads each face's own edge column and never
    a blend of two faces; BILINEAR's taps clamp inside the face."""
    d_x = np.array([[1.0, 1.0 - 1e-3, 0.0]], f32)       # +X face, toward +Y
    d_y = np.array([[1.0 - 1e-3, 1.0, 0.0]], f32)       # +Y_b face, toward +X
    fx, ux, _ = SK.cube_face_uv(d_x)
    fy, uy, _ = SK.cube_face_uv(d_y)
    check('(1, 1-1e-3, 0) is on the +X face at its +Y edge column and '
          '(1-1e-3, 1, 0) on the +Y face at its +X edge column',
          fx[0] == 0 and fy[0] == 5 and ux[0] > 0.999 and uy[0] < 0.001,
          f'{fx} {ux} {fy} {uy}')
    nx = cpu_cube(STACK, d_x, 0)[0]
    ny = cpu_cube(STACK, d_y, 0)[0]
    check('NEAREST: the two reads are each face\'s own edge texel (red 0 vs '
          '1, green 1 vs 0) and differ',
          np.array_equal(nx, STACK[0 * S + 4, 7, :3])
          and np.array_equal(ny, STACK[5 * S + 4, 0, :3])
          and not np.array_equal(nx, ny))
    # a sweep across the seam: every NEAREST sample IS one of the two edge
    # texels, never a value between them (no blend across faces)
    th = np.linspace(math.pi / 4 - 0.02, math.pi / 4 + 0.02, 401)
    sweep = np.stack([np.cos(th), np.sin(th), np.zeros_like(th)], 1).astype(f32)
    sw = cpu_cube(STACK, sweep, 0)
    is_a = np.all(sw == nx[None, :], axis=1)
    is_b = np.all(sw == ny[None, :], axis=1)
    check('a 401-sample sweep across the seam reads only the two edge '
          'texels (both appear, nothing in between)',
          bool(np.all(is_a | is_b)) and is_a.any() and is_b.any(),
          f'{int(is_a.sum())} / {int(is_b.sum())} / {int((~(is_a | is_b)).sum())}')
    g = glsl_cube(STACK, sweep, 0)
    check('...and the GLSL sweep is bitwise the CPU sweep',
          g is not None and np.array_equal(g, sw))
    # BILINEAR: the taps clamp to the face -- the +X edge column reads
    # green 1.0 exactly (never pulled toward the +Y face's green 0), the +Y
    # edge column green 0.0 exactly; the rows blend as the law says
    bx = cpu_cube(STACK, d_x, 1)[0]
    by = cpu_cube(STACK, d_y, 1)[0]
    # the law by hand (core/texture._sample_bilinear with the clamp):
    v = f32(0.5)
    fy_ = v * f32(S) - f32(0.5)
    ty = f32(fy_ - np.floor(fy_))
    want_x = STACK[0 * S + 3, 7, :3] + (STACK[0 * S + 4, 7, :3] - STACK[0 * S + 3, 7, :3]) * ty
    want_y = STACK[5 * S + 3, 0, :3] + (STACK[5 * S + 4, 0, :3] - STACK[5 * S + 3, 0, :3]) * ty
    check('BILINEAR at the +X edge: both column taps clamp to column 7 '
          '(green exactly 1.0), rows blend by the law',
          bx[1] == 1.0 and np.array_equal(bx, want_x.astype(f32)), str(bx))
    check('BILINEAR at the +Y edge: both column taps clamp to column 0 '
          '(green exactly 0.0)', by[1] == 0.0
          and np.array_equal(by, want_y.astype(f32)), str(by))
    gb = glsl_cube(STACK, np.concatenate([d_x, d_y]), 1)
    check('...and the GLSL bilinear edge reads are bitwise',
          gb is not None and np.array_equal(gb[0], bx) and np.array_equal(gb[1], by))
    # the whole-frame seam under BILINEAR: no sample anywhere blends two
    # faces' reds (red is constant per face, so a blend would land between)
    rng = np.random.default_rng(5)
    d = rng.normal(size=(5000, 3)).astype(f32)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    allb = cpu_cube(STACK, d, 1)
    reds = np.round(allb[:, 0] * 5)
    check('BILINEAR over 5000 random directions never blends two faces '
          '(every red is exactly a face\'s own)',
          bool(np.all(np.abs(allb[:, 0] * 5 - reds) < 1e-6)))


# =================================== rotation, tint, strength, environment


def test_cube_rotation_tint_strength():
    """World.rotation spins the skybox (the existing dial, the HDRI's
    sign), env_tint and strength multiply as hdri() does, and evaluate()
    returns the cube texel along ANY direction (an environment)."""
    w, tex = world_for(pack('HSTRIP'), layout='HSTRIP')
    wr, texr = world_for(pack('HSTRIP'), layout='HSTRIP', rotation=math.pi / 2)
    px = np.array([AX['+X']], f32)
    ny = np.array([AX['-Y']], f32)
    py = np.array([AX['+Y']], f32)
    a = SK.evaluate(wr, px, texr)
    b = SK.evaluate(w, ny, tex)
    c = SK.evaluate(w, py, tex)
    check('with rotation pi/2 the +X sample is the unrotated -Y sample '
          '(evaluate spins by -rotation: the HDRI sign), not the +Y one',
          np.array_equal(a, b) and not np.array_equal(a, c), f'{a} {b} {c}')
    # off-axis too: a direction leaning into the face
    lean = np.array([vec('+X', '+Z', 0.4), vec('+X', '+Y', 0.3)])
    lean_r = np.array([vec('-Y', '+Z', 0.4), vec('-Y', '+X', 0.3)])
    check('...and for leaning directions (same texels)',
          np.array_equal(SK.evaluate(wr, lean, texr), SK.evaluate(w, lean_r, tex)))
    tint = (0.5, 0.25, 1.0)
    wt, text = world_for(pack('HSTRIP'), layout='HSTRIP', env_tint=tint,
                         strength=1.7)
    rng = np.random.default_rng(11)
    d = rng.normal(size=(500, 3)).astype(f32)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    got = SK.evaluate(wt, d, text)
    raw = cpu_cube(STACK, d, 0)
    want = (raw * np.asarray(tint, f32)[None, :]).astype(f32) * 1.7
    check('env_tint then strength multiply bitwise as hdri() does',
          np.array_equal(got, want.astype(f32)),
          f'max {float(np.abs(got - want).max())}')
    wb, texb = world_for(pack('HSTRIP'), layout='HSTRIP', cube_filter='BILINEAR')
    gotb = SK.evaluate(wb, d, texb)
    from ..core import mathx as M
    dn = M.normalize(d)                  # evaluate()'s own normalise, first
    check('cube_filter BILINEAR evaluates through the bilinear law (on the '
          'road\'s own normalised direction)',
          np.array_equal(gotb, cpu_cube(STACK, dn, 1))
          and not np.array_equal(gotb, cpu_cube(STACK, dn, 0)))
    axes = np.array([AX[k] for k in ('+X', '-X', '+Y', '-Y', '+Z', '-Z')], f32)
    ev = SK.evaluate(w, axes, tex)
    check('SK.evaluate along the six axes returns the cube texels (an '
          'environment, unlike CYLINDER\'s flat colour)',
          np.round(ev[:, 0] * 5).astype(int).tolist() == [0, 1, 5, 4, 2, 3])
    check('strength False leaves the texel unscaled',
          np.array_equal(SK.evaluate(wt, d, text, strength=False),
                         (raw * np.asarray(tint, f32)[None, :]).astype(f32)))


# =========================================================== the GPU twin


def test_cube_gpu_twin():
    """The sky pass under CUBEMAP is bitwise the CPU sky at every uncovered
    pixel on three rigs and both filters; no image folds to MODE_FLAT;
    a too-tall atlas refuses by name; the fake device draws the simulator's
    pixels on the SINGLE and SIX roads; the content key follows the dials."""
    # NEAREST: bitwise. BILINEAR: the sampler is bitwise at the function
    # level (test_cube_face_rule, 4006 directions); on the FRAME the ray
    # build itself differs by an ulp at a dozen pixels under this NumPy
    # (the shipped HDRI / GRADIENT twins of test_gpu_sky_pass measure the
    # same 10-13 px on the pristine tree), which NEAREST never sees and a
    # lerp carries as ~1e-6 -- so BILINEAR's bar is 4e-6 with the pixel
    # count bounded (a wrong face or a crossed seam would be a whole texel)
    for filt, bar in (('NEAREST', 0.0), ('BILINEAR', 4e-6)):
        for label, ss, fast in (('ss 1', 1, True),
                                ('ss 2 under fast_background', 2, True),
                                ('ss 2 with fast_background off', 2, False)):
            sc, st, tex = rig(ss=ss, fast_background=fast,
                              world={'cube_filter': filt, 'strength': 1.2,
                                     'env_tint': (0.9, 0.8, 1.0)})
            d, why, plan = twin_d(sc, st, tex, ss)
            check(f'GPU twin, {filt} at {label}: the sky pass is '
                  + ('bitwise' if bar == 0.0 else f'within {bar} of')
                  + ' the CPU sky at every uncovered pixel',
                  d is not None and d <= bar,
                  str(why) if d is None else f'd {d}')
            check(f'...{filt} at {label}: the plan chose MODE_CUBE and the '
                  'cube texel is (S, filt, 0)',
                  plan is not None and plan['mode'] == GSKY.MODE_CUBE
                  and plan['params'].get('hal_sky_cube')
                  == (float(S), 0.0 if filt == 'NEAREST' else 1.0, 0.0),
                  str(None if plan is None else plan['params'].get('hal_sky_cube')))
    # under a World rotation: the CPU spins by float64 cos / sin
    # (sky._rotate_z), the texel carries them as float32 -- NEAREST holds
    # bitwise on these frames (no ray lands within an ulp of a texel
    # cliff), BILINEAR within 4e-6 (measured 1.7e-6 over four rotations
    # and three rigs: the rotation road's own ulp, the HDRI road's stance)
    for rot in (0.3, -0.7, 2.5):
        for filt, bar in (('NEAREST', 0.0), ('BILINEAR', 4e-6)):
            for label, ss, fast in (('ss 1', 1, True),
                                    ('ss 2 under fast_background', 2, True)):
                sc, st, tex = rig(ss=ss, fast_background=fast,
                                  world={'cube_filter': filt, 'rotation': rot,
                                         'strength': 1.2})
                d, why, plan = twin_d(sc, st, tex, ss)
                check(f'GPU twin, {filt} at {label} under rotation {rot}: '
                      + ('bitwise' if bar == 0.0 else f'within {bar}')
                      + ' (the rotation\'s own float32 cos / sin)',
                      d is not None and d <= bar,
                      str(why) if d is None else f'd {d}')
    # the pre-existing sign bug this twin found: the GPU HDRI sky spun the
    # other way under a non-zero Rotation since 1.89.0 (every uncovered
    # pixel differed); the texel now carries _rotate_z's own (c, s)
    st_h = settings()
    sc_h = demo_scene(st_h, with_texture=False)
    hb = ImageBuffer(name='hdr', pixels=pack('HSTRIP'), colorspace='Linear')
    sc_h.images['hdr'] = hb
    sc_h.world.mode = 'HDRI'
    sc_h.world.env_image = hb
    sc_h.world.env_filter = 'NEAREST'
    sc_h.world.rotation = 0.3
    d_h, why_h, plan_h = twin_d(sc_h, st_h, {'hdr': Texture(
        hb.pixels, name='hdr', colorspace='Linear')}, 1)
    check('the GPU HDRI sky under rotation 0.3 is bitwise the CPU\'s '
          '(pre-1.92 it spun the other way at every pixel)',
          d_h == 0.0 and plan_h is not None and plan_h['mode'] == GSKY.MODE_HDRI
          and plan_h['params']['hal_sky_rot'][1] < 0.0, str(why_h))
    sc, st, tex = rig()
    check('the refusal list takes CUBEMAP (no GPU refusal for the mode)',
          GSKY.refusal(sc, st) is None, str(GSKY.refusal(sc, st)))
    sc6, st6, tex6 = rig(source='SIX', world={'cube_convention': 'QUAKE2'})
    d6, why6, plan6 = twin_d(sc6, st6, tex6, 1)
    check('GPU twin on the SIX road under QUAKE2: bitwise, MODE_CUBE',
          d6 == 0.0 and plan6 is not None and plan6['mode'] == GSKY.MODE_CUBE,
          str(why6) if d6 is None else f'd {d6}')
    # no image: the solid colour x strength, MODE_FLAT, bitwise
    scn, stn, texn = rig(source='NONE', world={'strength': 1.5})
    dn, whyn, plann = twin_d(scn, stn, texn, 1)
    check('with no image the plan folds to MODE_FLAT and the twin holds '
          'bitwise', dn == 0.0 and plann is not None
          and plann['mode'] == GSKY.MODE_FLAT, str(whyn))
    _v, _p, vp, eye = R.camera_matrices(scn.camera, W, H)
    imgn = R._background_image(scn, stn, W, H, vp, eye, None, texn)
    exp = (SK.solid(scn.world, np.zeros((1, 3), f32)) * 1.5).astype(f32)[0]
    check('...and every CPU sky pixel is the World colour times strength',
          np.array_equal(imgn[..., :3], np.broadcast_to(exp, (H, W, 3))))
    # a face too tall for the driver: (None, why) by name, the CPU draws
    orig = SK.cube_atlas
    import types
    try:
        SK.cube_atlas = lambda world, textures: types.SimpleNamespace(width=3000)
        p, why = GSKY.plan(sc, st, W, H, vp, eye, tex, ss=1)
    finally:
        SK.cube_atlas = orig
    check('a 3000-px face plans (None, why) with the height-limit sentence',
          p is None and why == ('the cube map faces are 3000 px; the six-face '
                                "atlas would exceed the driver's texture height"),
          str(why))
    check('the height limit is the field\'s GL_MAX_TEXTURE_SIZE floor (16384)',
          GSKY.CUBE_ATLAS_MAX_HEIGHT == 16384 and 6 * 2730 <= 16384 < 6 * 2731)
    # the fake device: SINGLE and SIX
    for label, src in (('the SINGLE frame', 'SINGLE'), ('the SIX frame', 'SIX')):
        sc_f, st_f, _t = rig(source=src, world={'strength': 1.1, 'rotation': 0.2})
        cpu_f = np.asarray(R.render(sc_f, st_f))
        sc_g, st_g, _t2 = rig(source=src, world={'strength': 1.1, 'rotation': 0.2})
        got_f = fake_device_render(sc_g, st_g)
        fake_device_checks(label, sc_f, got_f, cpu_f)
    # the content key follows the orientation dials
    sa, sta, ta = rig()
    sb, stb, tb = rig(world={'cube_face_rot': (1, 0, 0, 0, 0, 0)})
    pa, _ = GSKY.plan(sa, sta, W, H, vp, eye, ta, ss=1)
    pb, _ = GSKY.plan(sb, stb, W, H, vp, eye, tb, ss=1)
    check('the plan\'s env_key differs between two orientations of the same '
          'source', pa is not None and pb is not None
          and pa['env_key'] != pb['env_key'] and pa['env_key'][0] == 'sky_env')
    check('the atlas rides the env sampler at (S, 6S): no new sampler, no '
          'new push constant', pa['params']['hal_sky_env_size']
          == (float(S), float(6 * S)) and GSKY.push(pa) == {'hal_sky_mode': 9}
          and set(GSKY.interface(GSKY.SOURCE)['samplers'])
          == {'hal_gb_ids', 'hal_sky_env', 'hal_sky_params', 'hal_sky_tab',
              'hal_sky_m7rows', 'hal_sky_m7map'})
    env = GSKY.env_pixels(pa)
    check('env_pixels hands the atlas bytes through unchanged',
          env is not None and np.array_equal(env, STACK))


# ======================================================== reflections


def test_cube_reflections():
    """A reflective material under a cube map takes the exact CPU-composite
    road on the GPU (`_env_world` -> ('CPU',)); on the CPU the mirror
    Ball reflects the skybox (the frame differs from the flat colour's at
    the covered pixels)."""
    sc, st, tex = rig()
    view, _proj, vp, eye = R.camera_matrices(sc.camera, W, H)
    job = R.ShadeJob(sc, st, {}, None, view, eye, W, H)
    spec, why = GSH._env_world(job)
    check('the deferred pass reflects the cube map by the exact CPU road',
          spec == ('CPU',) and why is None, f'{spec} {why}')
    st.raytrace = True
    st.ray_depth = 1
    sc.materials[1].reflect_level = 0.6          # the Ball becomes a mirror
    under_cube = np.asarray(R.render(sc, st))
    sc.world.mode = 'SOLID'
    under_flat = np.asarray(R.render(sc, st))
    g = CR.GBuffer(W, H)
    CR.rasterize(sc.mesh.verts, sc.mesh.tris, vp, W, H, gbuf=g)
    cov = g.tri >= 0
    dm = float(np.abs(under_cube - under_flat)[cov].max())
    check('a glossy demo-scene render under the cube map differs from one '
          'under the flat colour at the geometry (the skybox is reflected)',
          dm > 0.02, f'max {dm}')
    sc.world.mode = 'CUBEMAP'
    dirs = np.array([vec('+Z', '+X', 0.3), vec('-Y', '+Z', 0.2)], f32)
    ev = SK.evaluate(sc.world, dirs, tex)
    check('a reflected ray evaluates the cube texel, not the flat colour',
          np.array_equal(ev, cpu_cube(STACK, dirs, 0)[:, :3]))


# ============================================================== wiring


def test_cube_wiring():
    """fakebpy: the enum item is LAST, every World field mirrors into the
    settings group, the exporter skips and loads the six slots, the
    presets exclude every cube_* field, the panel draws the dials, the
    operator is registered, the path helper names the files, the icon
    is vetted."""
    import importlib
    import inspect
    import os

    from . import fakebpy
    from .blender_icons import ICONS as VERIFIED_ICONS
    fakebpy.install()
    P = importlib.import_module('halcyon.properties')
    importlib.reload(P)
    ui = importlib.import_module('halcyon.ui')
    from ..presets import skies as SKP
    items = P.HalcyonWorldSettings.__annotations__['mode'].kw['items']
    check("'CUBEMAP' is the LAST item of HalcyonWorldSettings.mode and of "
          'SKY.MODES (positional numbering, append-only)',
          items[-1][0] == 'CUBEMAP' and SK.MODES[-1] == 'CUBEMAP'
          and SK.MODES[:-1] == ('NODES', 'SOLID', 'GRADIENT', 'BANDS',
                                'STARFIELD', 'BRYCE', 'PHYSICAL', 'HDRI',
                                'PAINTED', 'CYLINDER', 'LW_GRADIENT'))
    import dataclasses
    cube_fields = [f.name for f in dataclasses.fields(World)
                   if f.name.startswith('cube_')]
    ann = P.HalcyonWorldSettings.__annotations__
    check('every cube_* World field has a HalcyonWorldSettings annotation '
          '(the generic export loop carries the dials)',
          len(cube_fields) == 12 and all(f in ann for f in cube_fields),
          str([f for f in cube_fields if f not in ann]))
    short = [f for f in cube_fields
             if len(str(ann[f].kw.get('description', '') or '')) < 40]
    check('every cube_* property carries a >= 40-char tooltip', not short,
          str(short))
    enum_short = []
    for f in ('mode', 'cube_source', 'cube_layout', 'cube_convention',
              'cube_filter'):
        for ident, _label, desc in ann[f].kw['items']:
            if (ident.startswith('CUBE') or f != 'mode') and len(desc) < 12:
                enum_short.append(f'{f}.{ident}')
    check('every cube enum item carries a >= 12-char description',
          not enum_short, str(enum_short))
    check('cube_face_rot / cube_face_flip are size-6 vector properties '
          '(a file-format promise)',
          ann['cube_face_rot'].kind == 'IntVectorProperty'
          and ann['cube_face_rot'].kw.get('size') == 6
          and ann['cube_face_flip'].kind == 'BoolVectorProperty'
          and ann['cube_face_flip'].kw.get('size') == 6)
    root = os.path.dirname(os.path.dirname(os.path.abspath(R.__file__)))
    esrc = open(os.path.join(root, 'export.py'), encoding='utf-8').read()
    check('export.py skips the six slots in the generic loop and loads them '
          'through _world_image',
          all(f"'cube_image_{s}'" in esrc for s in SLOTS)
          and "_world_image(hs, f'cube_image_{_slot}', images)" in esrc
          and 'def _world_image(hs, attr, images):' in esrc)
    check('presets.skies.EXCLUDED holds every cube_* field (no ImageBuffer '
          'in a sky file; a preset never clears a skybox)',
          all(f in SKP.EXCLUDED for f in cube_fields)
          and not any(f.startswith('cube_') for f in SKP.sky_fields()))
    src = inspect.getsource(ui.HALCYON_PT_world.draw)
    check('the World panel draws the layout, convention, filter, face turns '
          'and the Load Six Faces button',
          all(f"'{k}'" in src for k in ('cube_source', 'cube_layout',
                                         'cube_convention', 'cube_filter',
                                         'cube_face_rot', 'cube_face_flip'))
          and "operator('halcyon.cube_load_six'" in src
          and 'Face Orientation' in src)
    check('CUBEMAP is not in the show_ground list (an environment, like HDRI)',
          "'CUBEMAP'" not in src.split("if m in ('GRADIENT', 'BANDS'")[1].split(
              'col.separator()')[0])
    check('the sky operators the panel always drew are still reachable',
          all(op in src for op in ('halcyon.sky_preset', 'halcyon.sky_save',
                                   'halcyon.sky_load')))
    op = getattr(ui, 'HALCYON_OT_cube_load_six', None)
    check('HALCYON_OT_cube_load_six exists, is registered and has the file '
          'browser pattern', op is not None and op in ui.CLASSES
          and op.bl_idname == 'halcyon.cube_load_six'
          and 'fileselect_add' in inspect.getsource(op.invoke)
          and 'cube_set_from_path' in inspect.getsource(op.execute))
    icons = re.findall(r"icon='(\w+)'", src.split("m == 'CUBEMAP'")[1].split(
        'elif m ==')[0])
    check("the operator's icon is on the vetted list",
          icons and all(i in VERIFIED_ICONS for i in icons), str(icons))
    conv, paths = SK.cube_set_from_path('unit1_rt.tga')
    check("cube_set_from_path('unit1_rt.tga') -> ('QUAKE2', six paths)",
          conv == 'QUAKE2' and len(paths) == 6
          and paths['px'] == 'unit1_rt.tga' and paths['nx'] == 'unit1_lf.tga'
          and paths['py'] == 'unit1_up.tga' and paths['ny'] == 'unit1_dn.tga'
          and paths['pz'] == 'unit1_bk.tga' and paths['nz'] == 'unit1_ft.tga',
          str(paths))
    conv2, paths2 = SK.cube_set_from_path('posx.png')
    check("'posx.png' -> ('OPENGL', posx..negz)",
          conv2 == 'OPENGL' and paths2['nz'] == 'negz.png'
          and paths2['py'] == 'posy.png', str(paths2))
    conv3, why3 = SK.cube_set_from_path('sky.png')
    check("'sky.png' -> (None, reason)", conv3 is None and 'sky.png' in str(why3),
          str(why3))
    conv4, paths4 = SK.cube_set_from_path('/env/desertft.tga')
    conv5, paths5 = SK.cube_set_from_path('C:/sky/mountains_Left.PNG')
    conv6, paths6 = SK.cube_set_from_path('cube_NZ.exr')
    check('the Half-Life bare suffix, the _left family and _nz resolve with '
          'their case kept',
          conv4 == 'QUAKE2' and paths4['px'].endswith('desertrt.tga')
          and conv5 == 'OPENGL' and paths5['px'].endswith('mountains_Right.PNG')
          and conv6 == 'OPENGL' and paths6['py'] == 'cube_PY.exr',
          f'{paths4["px"]} {paths5["px"]} {paths6["py"]}')
    check('the console line names the mode', "9: 'cube map'" in open(
        os.path.join(root, 'core', 'render.py'), encoding='utf-8').read())
    check('CUBE_SLOT_LABELS name the slots in each convention\'s words',
          SK.CUBE_SLOT_LABELS['OPENGL'] == ('+X', '-X', '+Y', '-Y', '+Z', '-Z')
          and SK.CUBE_SLOT_LABELS['QUAKE2'] == ('rt', 'lf', 'up', 'dn', 'bk', 'ft'))


# ========================================================== neutrality


def test_cube_neutrality():
    """Every new field is inert at its default: a World() carries them,
    the PARAMS texel is appended last and never set by a pre-existing
    mode, an HDRI frame is still hdri()'s own evaluation along the frame's
    rays, and a GRADIENT / HDRI frame is bitwise the same whatever the
    cube dials hold."""
    w = World()
    check('a World() carries the new fields at their neutral defaults',
          w.cube_source == 'SINGLE' and w.cube_layout == 'AUTO'
          and w.cube_convention == 'OPENGL' and w.cube_filter == 'NEAREST'
          and w.cube_face_rot == (0, 0, 0, 0, 0, 0)
          and w.cube_face_flip == (False,) * 6
          and all(getattr(w, f'cube_image_{s}') is None for s in SLOTS))
    check("GSKY.PARAMS[-1][0] == 'hal_sky_cube' (appended at the end: every "
          'earlier texel keeps its index)',
          GSKY.PARAMS[-1][0] == 'hal_sky_cube'
          and GSKY.PARAM_INDEX['hal_sky_gfcol'] == len(GSKY.PARAMS) - 2)
    # an HDRI frame: bitwise hdri()'s own evaluation along the frame's rays
    st = settings()
    sc = demo_scene(st, with_texture=False)
    px = pack('HSTRIP')
    buf = ImageBuffer(name='hdr', pixels=px, colorspace='Linear')
    sc.images['hdr'] = buf
    sc.world.mode = 'HDRI'
    sc.world.env_image = buf
    sc.world.env_filter = 'NEAREST'      # the HDRI twin's own bitwise road
    tex = {'hdr': Texture(px, name='hdr', colorspace='Linear')}
    _v, _p, vp, eye = R.camera_matrices(sc.camera, W, H)
    img = R._background_image(sc, st, W, H, vp, eye, None, tex)
    from ..core import mathx as M
    yy, xx = np.mgrid[0:H, 0:W]
    yy, xx = yy.ravel(), xx.ravel()
    inv = np.linalg.inv(vp).astype(f32)
    nx = (xx.astype(f32) + 0.5) / W * 2.0 - 1.0
    ny = (yy.astype(f32) + 0.5) / H * 2.0 - 1.0
    pts = np.stack([nx, ny, np.ones(nx.size, f32), np.ones(nx.size, f32)], 1)
    wp = pts @ inv.T
    wp = wp[:, :3] / np.where(np.abs(wp[:, 3:4]) < 1e-9, 1e-9, wp[:, 3:4])
    dirs = M.normalize(wp - eye[None, :])
    want = SK.evaluate(sc.world, dirs, tex, eye=eye).reshape(H, W, 3)
    check("an HDRI frame is bitwise SK.hdri's own evaluation along "
          "_background_image's rays", np.array_equal(img[..., :3], want))
    d, why, plan = twin_d(sc, st, tex, 1)
    check('...and its GPU twin still holds bitwise under MODE_HDRI',
          d == 0.0 and plan is not None and plan['mode'] == GSKY.MODE_HDRI,
          str(why))
    # a pre-existing mode never sets the cube texel
    seen = []
    for mode in SK.MODES:
        if mode == 'CUBEMAP':
            continue
        sc_m = demo_scene(st, with_texture=False)
        sc_m.world.mode = mode
        if mode == 'HDRI':
            sc_m.images['hdr'] = buf
            sc_m.world.env_image = buf
        p, _why = GSKY.plan(sc_m, st, W, H, vp, eye, tex, ss=1)
        if p is not None and p['params'].get('hal_sky_cube') is not None:
            seen.append(mode)
        if p is not None:
            packed = GSKY.pack_params(p['params'])
            if packed[0, GSKY.PARAM_INDEX['hal_sky_cube']].any():
                seen.append(mode + ' (texel)')
    check('GSKY.plan for every pre-existing mode leaves hal_sky_cube unset '
          '(zeros): the texel is read only under mode 9', not seen, str(seen))
    # the dials are inert outside CUBEMAP: a GRADIENT frame and an HDRI
    # frame with every cube field changed are bitwise the default frames
    def frame(mode, dials):
        st_ = settings()
        sc_ = demo_scene(st_, with_texture=False)
        sc_.world.mode = mode
        if mode == 'HDRI':
            sc_.images['hdr'] = buf
            sc_.world.env_image = buf
        for k, v in dials.items():
            setattr(sc_.world, k, v)
        for slot, pic in zip(SLOTS, FACES):
            if dials:
                b6 = ImageBuffer(name=f'n_{slot}', pixels=pic, colorspace='Linear')
                sc_.images[b6.name] = b6
                setattr(sc_.world, f'cube_image_{slot}', b6)
        return np.asarray(R.render(sc_, st_))
    loud = {'cube_source': 'SIX', 'cube_layout': 'VCROSS',
            'cube_convention': 'QUAKE2', 'cube_filter': 'BILINEAR',
            'cube_face_rot': (1, 2, 3, 0, 1, 2),
            'cube_face_flip': (True, False, True, False, True, False)}
    for mode in ('GRADIENT', 'HDRI', 'SOLID'):
        a = frame(mode, {})
        b = frame(mode, loud)
        check(f'a {mode} frame is bitwise the same with every cube dial '
              'changed and six slots filled (the fields are read under '
              'CUBEMAP only)', np.array_equal(a, b),
              f'max {float(np.abs(a - b).max())}')
    # CYLINDER still sees the flat colour along a direction (its law), and
    # CUBEMAP shares env_image with it without changing it
    scc = demo_scene(st, with_texture=False)
    scc.images['hdr'] = buf
    scc.world.env_image = buf
    scc.world.mode = 'CYLINDER'
    dd = np.array([AX['+X']], f32)
    check('CYLINDER along a direction is still the flat colour',
          np.array_equal(SK.evaluate(scc.world, dd, tex), SK.solid(scc.world, dd)))


# ================== the deep re-read of the Console Emulation Shader road
#
# The 1.91.0 field crash: a PC fixed-function material on Modulate 4x
# reached the driver with hal_s8 / hal_t8 defined three times, the driver
# refused the pass and the application went down after the refusal. The
# simulator shrugs at everything a driver refuses, so this is the re-read's
# instrument: every PC fixed-function shading x texture op, every machine's
# defaults at two rates, and every option of every machine one at a time,
# through the GPU plan on the demo scene -- each assembled pass source
# checked for the whole class (a function defined twice, an HLSL-ism, a
# reserved-word identifier, a declaration left for CreateInfo to redeclare,
# an int / int division) and the frame held to the Gouraud-seam bar. The
# exhaustive sweep (654 combos, three rates) ran clean at R253; this keeps
# the field's combo and one of everything under the suite's time.


def test_zz_console_shader_driver_strictness():
    import inspect

    from ..core import console as CON
    from ..gpu import device as DEV
    from .test_r251_material import _sim_vs_cpu
    from .test_r252_console import _scene

    not_glsl = re.compile(
        r'(?<![\w.])(log10|saturate|lerp|frac|atan2|rsqrt|ddx|ddy|tex2D|'
        r'float[234]|int[234]|half[234]?|fmod|mul)\s*\(|\*\*|\bnp\.|\bmath\.')
    reserved = re.compile(r'\b(float|int|bool|vec[234])\s+'
                          r'(smooth|flat|sample|buffer|patch|precise|input|'
                          r'output|filter|active|common|partition)\s*[=;,)]')
    decl = re.compile(r'^(uniform|in|out)\s+\w[^;]*;$')
    intdiv = re.compile(r'(?<![\w.])(\d+)\s*/\s*(\d+)(?![\w.])')

    def combos():
        for t, _l, _c in CON.PC_TYPE_ITEMS:
            for op, _l2, _c2 in CON.PC_TEXTURE_OP_ITEMS:
                yield f'PC_FIXED {t} x {op}', {'console': 'PC_FIXED',
                                               'pc_type': t,
                                               'pc_texture_op': op}, 'VERTEX'
        for con, _lab, _d in CON.CONSOLE_ITEMS:
            yield con, {'console': con}, 'VERTEX'
            yield f'{con} at PIXEL', {'console': con}, 'PIXEL'
            for prop in CON.PANEL.get(con, ()):
                items = next((it for nm, it, _d in CON.ENUM_PROPS if nm == prop),
                             None)
                if items is None:
                    continue
                for ident, _l, _c in items:
                    yield (f'{con} {prop}={ident}',
                           {'console': con, prop: ident}, 'VERTEX')

    problems = []
    refused = []
    n = 0
    for label, props, rate in combos():
        n += 1
        sc, st = _scene(dict(props), w=32, h=24, shading_rate=rate)
        st.render_device = 'GPU'
        GSH._PLAN_CACHE.clear()
        d, nbad, _img, _cpu, passes, why = _sim_vs_cpu(sc, st)
        if passes is None:
            refused.append(f'{label}: {why}')
            continue
        if d is None:
            problems.append(f'{label}: simulate failed: {why}')
            continue
        srcs = [item for ps in passes
                for item in (ps if isinstance(ps, (tuple, list)) else (ps,))
                if isinstance(item, str) and 'void main' in item]
        if not srcs:
            problems.append(f'{label}: no pass source')
        for src in srcs:
            code = '\n'.join(ln.split('//', 1)[0] for ln in src.splitlines())
            dup = DEV.duplicate_definitions(src)
            if dup:
                problems.append(f'{label}: defines twice {dup}')
            hits = sorted(set(m.group(0) for m in not_glsl.finditer(code)))
            if hits:
                problems.append(f'{label}: HLSL-ism {hits}')
            if reserved.findall(code):
                problems.append(f'{label}: reserved-word identifier')
            left = [ln for ln in DEV.strip_declarations(src).splitlines()
                    if decl.match(ln.split('//', 1)[0].strip())
                    or re.search(r'\buniform\b', ln.split('//', 1)[0])]
            if left:
                problems.append(f'{label}: declaration left {left[:1]}')
            ih = intdiv.findall(code)
            if ih:
                problems.append(f'{label}: int / int {ih[:2]}')
        if not (d < 6e-3 and nbad == 0):
            problems.append(f'{label}: frame d {d} bad {nbad}')
    check(f'every planned console pass of {n} combos defines each function '
          'once, carries no HLSL-ism, reserved word, leftover declaration or '
          'int / int division, and holds the frame bar', not problems,
          '; '.join(problems[:4]))
    check(f'the {len(refused)} refused combos refuse by name (the Saturn mesh '
          'opacity, the Super FX palette gate)',
          refused and all(('SUPERFX' in r or 'MESH' in r)
                          and len(r.split(': ', 1)[1]) > 20 for r in refused),
          '; '.join(refused[:3]))
    # the device never hands a source the driver just refused to a second
    # GPU object: the 1.91.0 log reads "CreateInfo failed ... legacy
    # constructor also failed" before the application went down
    for fn in (DEV._compile_dynamic_miss, DEV._build):
        dsrc = inspect.getsource(fn)
        after = dsrc.split('create_from_info(info)', 1)[1]
        check(f'{fn.__name__}: a CreateInfo refusal returns at once (no '
              'legacy constructor on a source the driver refused)',
              'return None, ' in after.split('except Exception as exc:', 1)[1]
              .split('try:', 1)[0] and 'single hand-over' in after)


def main():
    utf8_console()
    FAILS.clear()
    names = sorted(n for n in globals() if n.startswith('test_'))
    for n in names:
        print(f'\n[{n}]')
        try:
            globals()[n]()
        except Exception:                                       # noqa: BLE001
            traceback.print_exc()
            FAILS.append(n + ' (exception)')
    print()
    print(f'{len(FAILS)} failure(s)' if FAILS else 'all R253 cube map checks passed')
    for f in FAILS:
        print('  FAIL', f)
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
