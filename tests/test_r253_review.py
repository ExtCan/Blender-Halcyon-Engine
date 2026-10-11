"""R253 review fixes: the adversarial re-read's confirmed findings, each
pinned where it bit.

(1) A singular object matrix (a zero scale on any axis) raised
    LinAlgError out of the whole frame on both Generated roads;
    `core/render.object_inverse` takes the pseudo-inverse and says so once.
(2) A console TYPE can pin the opacity below 1 with its socket hidden at
    1.0 (the Saturn mesh colour calculation): the alpha classifiers never
    saw it, so the half-transparent polygon stayed in the opaque pass.
    `core/console.pinned_alpha_reason` feeds both `material_see_through`
    and `export._alpha_reason`.
(3) The corner road's ALBEDO pass returned its raw fractional opacity:
    no punch-through clip, no Screen Door compare, no Transparency None.
    `ShadeJob._alpha_law` is the one law the pixel road already applied,
    now applied there too.
"""
import io
import sys
import traceback
import types
from contextlib import redirect_stdout

import numpy as np

from . import utf8_console
from .test_render import base_settings
from .scenebuild import demo_scene
from ..core import console as CON
from ..core import render as R
from ..core.scene import material_see_through
from .test_r252_console import _scene as console_scene, console_graph

FAILS = []
f32 = np.float32


def check(name, cond, extra=''):
    ok = bool(cond)
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f'  {extra}' if extra and not ok else ''))
    if not ok:
        FAILS.append(name)


def _zero_scale_scene(space):
    st = base_settings(48, 36, shadows=False, generated_space=space)
    sc = demo_scene(st, with_texture=False)
    m = np.eye(4, dtype=f32)
    m[2, 2] = 0.0
    sc.objects[1].matrix_world = m
    return sc, st


def test_singular_matrix_renders():
    R._SINGULAR_SAID.clear()
    for space in ('OBJECT', 'WORLD'):
        sc, st = _zero_scale_scene(space)
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                img = R.render(sc, st)
            ok = bool(np.isfinite(img).all())
        except Exception as exc:                                # noqa: BLE001
            img, ok = None, False
            print(f'  raised {type(exc).__name__}: {exc}')
        check(f'a zero-scale object renders a finite frame under generated_space {space}', ok)
        if space == 'OBJECT':
            check('and says so once by name', "object 'Ball': its matrix is singular" in buf.getvalue())
    # the helper itself: identity in, identity out (bitwise), singular -> finite
    eye = np.eye(4, dtype=f32)
    check('object_inverse(identity) is the identity bit for bit', bool(np.array_equal(R.object_inverse(eye), eye)))
    m = np.diag([2.0, 1.0, 0.0, 1.0]).astype(f32)
    inv = R.object_inverse(m, types.SimpleNamespace(name='flat'))
    check('object_inverse of a zero-scale matrix is finite and inverts the live axes',
          bool(np.isfinite(inv).all()) and abs(float(inv[0, 0]) - 0.5) < 1e-6 and abs(float(inv[1, 1]) - 1.0) < 1e-6
          and abs(float(inv[2, 2])) < 1e-6)
    check('None is the identity', bool(np.array_equal(R.object_inverse(None), eye)))


def test_console_pinned_opacity_is_alpha_evidence():
    res = CON.resolve({'console': 'SATURN', 'sat_type': 'MESH'})
    check('resolve pins the Saturn mesh opacity at 0.5', res.get('opacity') == 0.5)
    g = console_graph({'console': 'SATURN', 'sat_type': 'MESH'})
    why = CON.pinned_alpha_reason(g)
    check('pinned_alpha_reason names the machine, the type and the value',
          why is not None and 'Saturn' in why and 'mesh' in why.lower() and '0.500' in why, str(why))
    check('and is None for the Gouraud type and for a non-console graph',
          CON.pinned_alpha_reason(console_graph({'console': 'SATURN', 'sat_type': 'GOURAUD'})) is None
          and CON.pinned_alpha_reason(None) is None and CON.pinned_alpha_reason({'nodes': {}}) is None)
    sc, st = console_scene({'console': 'SATURN', 'sat_type': 'MESH'})
    check('material_see_through reads the pin (the split and the layer plan agree)',
          material_see_through(sc.materials[1]) == why)
    # the exporter's classifier (bpy-free: a material record with the graph)
    try:
        from . import fakebpy as _fb
        _fb.install()
        from .. import export as EX
        m = types.SimpleNamespace(opacity=1.0, graph=g)
        check('export._alpha_reason reads the pin before the hidden sockets',
              EX._alpha_reason(None, m) == why, str(EX._alpha_reason(None, m)))
    except ImportError as exc:
        check('export is importable under fakebpy', False, str(exc))
    # the picture: under Sorted the mesh polygon now blends with what is behind
    st.transparency = 'SORTED'
    mesh = np.asarray(R.render(sc, st))
    sc2, st2 = console_scene({'console': 'SATURN', 'sat_type': 'GOURAUD'})
    st2.transparency = 'SORTED'
    gour = np.asarray(R.render(sc2, st2))
    d = np.abs(mesh[:, :, :3] - gour[:, :, :3]).max(axis=2)
    check('under Sorted the Saturn mesh Ball differs from the Gouraud Ball (it reached the transparent pass)',
          int((d > 1e-3).sum()) > 100, str(int((d > 1e-3).sum())))


def test_corner_road_alpha_law():
    # Screen Door: the corner road's alpha is the compare's 0 / 1, as the pixel road's
    sc, st = console_scene({'console': 'SATURN', 'sat_type': 'MESH'})
    st.transparency = 'STIPPLE'
    img = np.asarray(R.render(sc, st))
    vals = set(np.round(img[:, :, 3].ravel(), 3).tolist())
    check('Screen Door: the Saturn mesh frame alpha is the compare bit (0 or 1), never 0.5',
          vals <= {0.0, 1.0} and 0.0 in vals and 1.0 in vals, str(sorted(vals))[:60])
    # the pixel road's own rule on the same frame, for the record
    st.transparency = 'NONE'
    img = np.asarray(R.render(sc, st))
    check('Transparency None: the corner road writes alpha 1.0', bool(np.all(img[:, :, 3] == 1.0)))
    # an opaque corner material is bitwise unchanged by the law (default neutrality)
    for tr in ('NONE', 'STIPPLE', 'SORTED'):
        sc_a, st_a = console_scene({'console': 'SATURN', 'sat_type': 'GOURAUD'})
        st_a.transparency = tr
        a = np.asarray(R.render(sc_a, st_a))
        check(f'an opaque corner material under {tr} keeps alpha 1.0 everywhere it covers',
              bool(np.all(a[:, :, 3][a[:, :, 3] > 0] == 1.0)))
    # the law is one method, used by both roads
    import inspect
    src = inspect.getsource(R.ShadeJob)
    check('the pixel road and the ALBEDO pass call the same _alpha_law',
          src.count('self._alpha_law(') >= 2 and 'def _alpha_law' in src)


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
    print('all R253 review-fix checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
