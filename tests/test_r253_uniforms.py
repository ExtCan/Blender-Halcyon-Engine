"""R253 (field): a uniform the driver stripped never fails a pass.

A real Blender 5.2.2 session on the OpenGL backend (Mesa llvmpipe)
printed, for every GPU frame:

    [Halcyon GPU] rasterising on the CPU: raster dispatch failed:
        ValueError: GPUShader.uniform_float: uniform hal_rtiles_w not found
    [Halcyon GPU] blended compositing fell back to per-pass readback:
        ValueError: GPUShader.uniform_float: uniform hal_cam_right not found
    [Halcyon GPU] shading on the CPU: drawing 'Material.001' failed:
        GPUShader.uniform_float: uniform hal_cam_right not found

A GL linker drops every uniform that does not reach the program's
output, Blender's uniform_float / uniform_int / uniform_bool raise
ValueError for a name the linked program no longer has, and Halcyon set
every uniform of a pass unconditionally -- so one stripped name failed
the whole pass and the frame fell back to the CPU road.

(1) `device.set_uniform` is the one door: the 'not found' ValueError is
    a skip, remembered per shader (the driver is asked once), and every
    other error is as loud as it was.
(2) The real `draw_fullscreen` / `draw_many` draw a pass whose shader
    refuses one name -- and with the old unconditional set the same
    fake raises the field's ValueError.
(3) The real `dispatch_compute` dispatches a kernel whose shader refuses
    one push constant.
(4) No compute kernel declares a uniform it never reads: the raster
    shell's `hal_rtiles_w` / `hal_rtiles_h` (dead since the tiles fetch
    became a texelFetch) are gone from the text, the CreateInfo and the
    dispatch, in all eight variants; `raster_on_device` sends exactly
    the push constants it declares.
(5) The demo frame on a GL-like driver that strips every unread uniform
    (tests/fakedevice.installed_gl: the REAL device draws in the loop)
    shades on the GPU road, bit for bit the frame of a driver that
    strips nothing -- and with the old set it prints the field's two
    shading lines word for word.
(6) The count is once per shader, the Debug breakdown's line reads
    'N uniforms stripped by the driver', reset() forgets it.
(7) Nothing else under gpu/ calls a GPUShader's uniform_* methods.
"""
import io
import os
import re
import sys
import traceback
import types
from contextlib import contextmanager, redirect_stdout

import numpy as np

from . import fakedevice, utf8_console
from .test_render import base_settings
from .scenebuild import demo_scene
from ..core import render as R
from ..gpu import craster as CRA
from ..gpu import device as DV
from ..gpu import rtrace as RT
from ..gpu import shade as GSH

FAILS = []
f32 = np.float32

#: the field log's own words, for the ValueError and for the two lines
#: the shading road printed
NOT_FOUND = 'GPUShader.uniform_float: uniform hal_cam_right not found'


def check(name, cond, extra=''):
    ok = bool(cond)
    print(f"  {'ok  ' if ok else 'FAIL'} {name}"
          + (f'  {extra}' if extra and not ok else ''))
    if not ok:
        FAILS.append(name)


class Stripping:
    """A GPUShader stand-in whose setters refuse the names in `stripped`
    with Blender's own ValueError, and record everything they were asked."""

    def __init__(self, stripped=(), fail=None):
        self.stripped = set(stripped)
        self.fail = dict(fail or {})        # name -> exception to raise
        self.asked = []                     # (method, name), every call
        self.values = {}
        self.samplers = {}
        self.images = {}
        self.binds = 0

    def bind(self):
        self.binds += 1

    def _set(self, method, name, value, store):
        self.asked.append((method, name))
        if name in self.fail:
            raise self.fail[name]
        if name in self.stripped:
            raise ValueError(f'GPUShader.{method}: uniform {name:.32s} '
                             'not found')
        store[name] = value

    def uniform_float(self, name, value):
        self._set('uniform_float', name, value, self.values)

    def uniform_int(self, name, value):
        self._set('uniform_int', name, value, self.values)

    def uniform_bool(self, name, value):
        self._set('uniform_bool', name, value, self.values)

    def uniform_sampler(self, name, tex):
        self._set('uniform_sampler', name, tex, self.samplers)

    def image(self, name, tex):
        self.images[name] = tex


def raw_set(shader, kind, name, value):
    """The door as it was before this round: every set unconditional."""
    getattr(shader, DV._UNIFORM_SETTERS[kind])(name, value)
    return True


@contextmanager
def clean_device():
    """The device's stripped memory empty for the block, restored after;
    `_main` in place (the suite is the main thread, as the marshal says)."""
    saved = dict(DV._STRIPPED)
    old_main, old_door = DV._main, DV.set_uniform
    DV._STRIPPED.clear()
    DV._main = lambda what, fn: fn()
    try:
        yield
    finally:
        DV._main, DV.set_uniform = old_main, old_door
        DV._STRIPPED.clear()
        DV._STRIPPED.update(saved)


class _Tex:
    """A GPUTexture a kernel writes: read() is its (h, w, 4) fill."""

    def __init__(self, size, fill=0.0):
        self.size = (int(size[0]), int(size[1]))
        self.fill = float(fill)

    def read(self):
        w, h = self.size
        return fakedevice._Buffer(np.full((h, w, 4), self.fill, f32))


@contextmanager
def fake_gl(dev=None, dispatched=None):
    """The gpu / gpu_extras.batch stand-ins the REAL device functions
    import, plus the compute half (GPUTexture, compute.dispatch); what
    the real functions leave in the device module is put back."""
    dev = dev if dev is not None else fakedevice.Device()
    mods = fakedevice._gl_modules(dev)
    made = []

    def _texture(size, format='RGBA32F', data=None):
        t = _Tex(size, 7.0)
        made.append(t)
        return t

    mods['gpu'].types = types.SimpleNamespace(Buffer=object,
                                              GPUTexture=_texture)
    mods['gpu'].compute = types.SimpleNamespace(
        dispatch=lambda sh, gx, gy, gz: (dispatched if dispatched is not None
                                         else []).append((sh, gx, gy, gz)))
    absent = object()
    saved_mods = {k: sys.modules.get(k, absent) for k in mods}
    saved_batches = dict(DV._STATE.get('batches', {}))
    saved_epoch = DV._STATE.get('flush_epoch', absent)
    saved_grave = list(DV._GRAVEYARD)
    saved_burst, saved_disp = dict(DV.LAST_BURST), dict(DV.LAST_DISPATCH)
    sys.modules.update(mods)
    DV._STATE['flush_epoch'] = 1000     # past flush_tick's forensics window
    try:
        yield dev, made
    finally:
        for k, v in saved_mods.items():
            if v is absent:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        DV._STATE['batches'] = saved_batches
        if saved_epoch is absent:
            DV._STATE.pop('flush_epoch', None)
        else:
            DV._STATE['flush_epoch'] = saved_epoch
        DV._GRAVEYARD[:] = saved_grave
        DV.LAST_BURST.clear()
        DV.LAST_BURST.update(saved_burst)
        DV.LAST_DISPATCH.clear()
        DV.LAST_DISPATCH.update(saved_disp)
        del fakedevice._BOUND[:]


# --------------------------------------------------------------- (1) the door

def test_the_door_skips_exactly_one_error():
    with clean_device():
        sh = Stripping(stripped={'hal_cam_right'})
        check('a uniform the program has is set and answers True',
              DV.set_uniform(sh, 'float', 'hal_eye', (1.0, 2.0, 3.0)) is True
              and sh.values == {'hal_eye': (1.0, 2.0, 3.0)})
        got = DV.set_uniform(sh, 'float', 'hal_cam_right', (1.0, 0.0, 0.0))
        check("the field's ValueError ('uniform hal_cam_right not found') "
              'is a skip: no exception, the answer is False',
              got is False and 'hal_cam_right' not in sh.values)
        asked = len(sh.asked)
        again = [DV.set_uniform(sh, 'float', 'hal_cam_right', (0.0, 1.0, 0.0))
                 for _ in range(5)]
        check('the refusal is remembered per shader: five more sets of the '
              'stripped name never reach the driver',
              again == [False] * 5 and len(sh.asked) == asked,
              f'{len(sh.asked) - asked} extra driver calls')
        check('...and it is counted ONCE (1 uniform, 1 shader)',
              DV.stripped_stats() == (1, 1)
              and DV.stripped_names() == ['hal_cam_right'],
              str(DV.stripped_stats()))
        other = Stripping()
        check('the memory is the SHADER\'s: another program is asked for the '
              'same name and takes it',
              DV.set_uniform(other, 'float', 'hal_cam_right', (1.0, 0.0, 0.0))
              is True and other.values and DV.stripped_stats() == (1, 1))

        # every setter kind goes through the same rule
        kinds = Stripping(stripped={'f', 'i', 'b', 's'})
        got = [DV.set_uniform(kinds, 'float', 'f', 1.0),
               DV.set_uniform(kinds, 'int', 'i', 3),
               DV.set_uniform(kinds, 'bool', 'b', [True]),
               DV.set_uniform(kinds, 'sampler', 's', object())]
        check('uniform_float, uniform_int, uniform_bool and uniform_sampler '
              'each skip a stripped name',
              got == [False] * 4
              and [m for m, _n in kinds.asked] == [
                  'uniform_float', 'uniform_int', 'uniform_bool',
                  'uniform_sampler'], str(kinds.asked))
        kept = Stripping()
        DV.set_uniform(kept, 'int', 'n', 3)
        DV.set_uniform(kept, 'bool', 'on', [True])
        tex = object()
        DV.set_uniform(kept, 'sampler', 'hal_gb_ids', tex)
        check('...and each hands a kept name to its own GPUShader method '
              'with the value untouched',
              kept.values == {'n': 3, 'on': [True]}
              and kept.samplers == {'hal_gb_ids': tex}
              and [m for m, _n in kept.asked] == [
                  'uniform_int', 'uniform_bool', 'uniform_sampler'])

        # Blender prints the name through %.32s: a long one arrives cut
        long_name = 'hal_' + 'x' * 40
        cut = Stripping(stripped={long_name})
        check('a name longer than 32 characters (Blender cuts it in the '
              'message) is still recognised as stripped',
              DV.set_uniform(cut, 'float', long_name, 1.0) is False
              and long_name in DV.stripped_names())


def test_every_other_error_stays_loud():
    with clean_device():
        cases = [
            ('a ValueError about the VALUE (a wrong length)',
             ValueError('GPUShader.uniform_float: invalid length 5, '
                        'expected 1..4, 9 or 16')),
            ('a TypeError (an int handed to a float uniform)',
             TypeError('GPUShader.uniform_float: expected a float')),
            ('a RuntimeError out of the driver',
             RuntimeError('driver said no')),
            ('a ReferenceError (a shader freed under the draw)',
             ReferenceError('GPUShader has been removed')),
        ]
        for label, exc in cases:
            sh = Stripping(fail={'hal_eye': exc})
            raised = None
            try:
                DV.set_uniform(sh, 'float', 'hal_eye', 1.0)
            except Exception as got:                            # noqa: BLE001
                raised = got
            check(f'{label} propagates, the same object',
                  raised is exc, repr(raised))
        check('none of them was counted as stripped',
              DV.stripped_stats() == (0, 0), str(DV.stripped_stats()))
        check('the test of the error itself: only a ValueError that says '
              'NOT FOUND is forgiven',
              DV.uniform_was_stripped(ValueError(NOT_FOUND))
              and DV.uniform_was_stripped(
                  ValueError('GPUShader.uniform_int: uniform hal_n not found'))
              and not DV.uniform_was_stripped(ValueError('invalid length'))
              and not DV.uniform_was_stripped(KeyError('x not found'))
              and not DV.uniform_was_stripped(RuntimeError(NOT_FOUND)))
        bad = None
        try:
            DV.set_uniform(Stripping(), 'vec9', 'hal_eye', 1.0)
        except KeyError as exc:
            bad = exc
        check('an unknown kind is a KeyError, not a silent skip',
              bad is not None)


# ---------------------------------------------------------- (2) the pass draws

def _one_pixel_pass(dev, strip):
    """A real compiled pass (the fake's front-end) that READS hal_tint and
    hal_gain and only declares the camera axis: colour = tint * gain."""
    src = ('in vec2 vUV;\n'
           'out vec4 Color;\n'
           'uniform vec3 hal_tint;\n'
           'uniform vec3 hal_cam_right;\n'
           'uniform float hal_gain;\n'
           'uniform sampler2D hal_src;\n'
           'void main()\n'
           '{\n'
           '    vec4 t = texture(hal_src, vUV);\n'
           '    Color = vec4(hal_tint * hal_gain + t.rgb, 1.0);\n'
           '}\n')
    spec = {'samplers': ['hal_src'], 'floats': ['hal_gain'], 'ints': [],
            'vec2': [], 'vec3': ['hal_tint', 'hal_cam_right']}
    inner, err = dev.compile_dynamic('HAL_STRIP_PROBE', src, spec)
    assert inner is not None, err
    gl = fakedevice.GLShader(inner, src)
    if not strip:
        gl.stripped = set()
    return gl


def test_a_stripped_uniform_still_draws_the_pass():
    uni = {'hal_tint': (0.25, 0.5, 1.0), 'hal_cam_right': (1.0, 0.0, 0.0),
           'hal_gain': 2.0}
    want = np.array([0.5, 1.0, 2.0, 1.0], f32)
    with clean_device(), fake_gl() as (dev, _made):
        sh = _one_pixel_pass(dev, strip=True)
        check('the fake program lost exactly the uniform its source never '
              'reads', sh.stripped == {'hal_cam_right'}, str(sh.stripped))
        bind = {'hal_src': dev.upload(np.zeros((2, 2, 4), f32))}
        tgt = dev.Target(4, 3)
        got = DV.draw_fullscreen(sh, uni, bind, tgt, clear=True)
        check('draw_fullscreen draws the pass whose shader refuses one '
              'uniform: the readback is tint * gain at every pixel',
              got is not None and got.shape == (3, 4, 4)
              and bool(np.array_equal(got, np.broadcast_to(want, (3, 4, 4)))),
              str(None if got is None else got[0, 0]))
        check('the uniforms the program HAS were all set, the stripped one '
              'refused once', sh.values == {'hal_tint': (0.25, 0.5, 1.0),
                                            'hal_gain': 2.0}
              and sh.refused == [('uniform_float', 'hal_cam_right')]
              and sh.draws == 1, f'{sh.values} {sh.refused}')

        # a burst: two passes into one target, one readback
        tgt2 = dev.Target(4, 3)
        uni2 = dict(uni, hal_gain=4.0)
        out = DV.draw_many([(sh, uni, bind, tgt2, 'NONE', True, None),
                            (sh, uni2, bind, tgt2, 'NONE', False,
                             (1, 0, 2, 3))], read=tgt2)
        inside = np.array([1.0, 2.0, 4.0, 1.0], f32)
        check('draw_many runs the whole burst (the second pass scissored) '
              'and reads it back',
              out is not None and bool(np.array_equal(out[:, 1:3],
                                                      np.broadcast_to(
                                                          inside, (3, 2, 4))))
              and bool(np.array_equal(out[:, 0], np.broadcast_to(
                  want, (3, 4))))
              and sh.draws == 3)
        check('three draws of the shader asked the driver about the '
              'stripped name ONCE', len(sh.refused) == 1, str(sh.refused))
        check('...and the count is 1 uniform in 1 shader',
              DV.stripped_stats() == (1, 1), str(DV.stripped_stats()))

        # the door as it was: the same fake, the field's exception
        DV.set_uniform = raw_set
        DV._STRIPPED.clear()
        raised = None
        try:
            DV.draw_fullscreen(sh, uni, bind, dev.Target(4, 3), clear=True)
        except ValueError as exc:
            raised = exc
        check("with the old unconditional set the same pass raises the "
              "field's own ValueError (the fake reproduces the driver)",
              raised is not None and str(raised) == NOT_FOUND, repr(raised))


def test_a_draw_restores_its_state_when_a_real_error_raises():
    """The skip must not swallow what the finally block is for: a loud
    error still leaves blend NONE and the scissor off."""
    with clean_device(), fake_gl() as (dev, _made):
        sh = _one_pixel_pass(dev, strip=True)
        boom = RuntimeError('driver said no')

        def _boom(name, value, _old=sh.uniform_float):
            if name == 'hal_gain':
                raise boom
            _old(name, value)
        sh.uniform_float = _boom
        bind = {'hal_src': dev.upload(np.zeros((2, 2, 4), f32))}
        raised = None
        try:
            DV.draw_fullscreen(sh, {'hal_cam_right': (1.0, 0.0, 0.0),
                                    'hal_gain': 2.0}, bind, dev.Target(4, 3),
                               blend='ALPHA_PREMULT', clear=True,
                               region=(0, 0, 2, 2))
        except RuntimeError as exc:
            raised = exc
        check('a driver error AFTER a skipped uniform still raises out of '
              'the draw', raised is boom, repr(raised))
        check('...with the blend back at NONE and the scissor off',
              fakedevice._GL['blend'] == 'NONE'
              and fakedevice._GL['scissor_on'] is False,
              str(fakedevice._GL))
        check('...and nothing was drawn', sh.draws == 0)


# ----------------------------------------------------- (3) the kernel dispatches

def test_a_stripped_push_constant_still_dispatches():
    dispatched = []
    with clean_device(), fake_gl(dispatched=dispatched) as (_dev, made):
        sh = Stripping(stripped={'hal_rtiles_w'})
        tex = object()
        out = DV.dispatch_compute(
            sh, 20, 9,
            uniforms={'hal_rc_side': 64.0, 'hal_rtiles_w': 2.0,
                      'hal_rw': 20.0, 'hal_rh': 9.0},
            samplers={'hal_rc': tex}, images=('hal_out_ids', 'hal_out_aux'))
        check("the field's raster line: a kernel whose shader refuses "
              "'hal_rtiles_w' dispatches (3 x 2 groups of 8 x 8 for 20 x 9)",
              dispatched == [(sh, 3, 2, 1)], str(dispatched))
        check('both images were bound and read back at the frame size',
              sorted(out) == ['hal_out_aux', 'hal_out_ids']
              and all(a.shape == (9, 20, 4) and float(a.min()) == 7.0
                      for a in out.values())
              and sorted(sh.images) == ['hal_out_aux', 'hal_out_ids']
              and len(made) == 2)
        check('every other push constant and the sampler were set',
              sh.values == {'hal_rc_side': 64.0, 'hal_rw': 20.0,
                            'hal_rh': 9.0} and sh.samplers == {'hal_rc': tex})
        check('the stripped push constant is counted once',
              DV.stripped_stats() == (1, 1)
              and DV.stripped_names() == ['hal_rtiles_w'])
        DV.dispatch_compute(sh, 20, 9, uniforms={'hal_rtiles_w': 2.0},
                            images=('hal_out_ids',))
        check('a second dispatch does not ask the driver again',
              [n for _m, n in sh.asked].count('hal_rtiles_w') == 1
              and DV.stripped_stats() == (1, 1) and len(dispatched) == 2)

        DV.set_uniform = raw_set
        DV._STRIPPED.clear()
        raised = None
        try:
            DV.dispatch_compute(sh, 20, 9, uniforms={'hal_rtiles_w': 2.0},
                                images=('hal_out_ids',))
        except ValueError as exc:
            raised = exc
        check('with the old unconditional set the dispatch raises the '
              "field's ValueError",
              raised is not None and str(raised) ==
              'GPUShader.uniform_float: uniform hal_rtiles_w not found',
              repr(raised))

        # an image is an OUTPUT: a kernel that lost one stays loud
        class NoImage(Stripping):
            def image(self, name, tex_):
                raise ValueError(f"Image '{name}' not found in shader")
        raised = None
        try:
            DV.dispatch_compute(NoImage(), 8, 8, images=('hal_out_ids',))
        except ValueError as exc:
            raised = exc
        check('a missing output IMAGE is not a skip: the dispatch raises',
              raised is not None and 'hal_out_ids' in str(raised))


# ------------------------------------------- (4) the kernels read what they declare

def _kernel_sources():
    out = []
    for lin in (False, True):
        for clear in (False, True):
            for cvg in (False, True):
                out.append((CRA.compute_variant_name(lin, clear, cvg),
                            CRA.compute_source(lin, clear, cvg)))
    for nm in sorted(dir(RT)):
        v = getattr(RT, nm)
        if isinstance(v, str) and 'void main' in v and 'COMPUTE' in nm:
            out.append((f'rtrace.{nm}', v))
    return out


def test_no_compute_kernel_declares_a_uniform_it_never_reads():
    srcs = _kernel_sources()
    check('the scan covers the eight raster variants and the ray kernels',
          sum(1 for n, _s in srcs if n.startswith('HAL_RASTER')) == 8
          and sum(1 for n, _s in srcs if n.startswith('rtrace.')) >= 2,
          str([n for n, _s in srcs]))
    dead = {n: fakedevice.unread_uniforms(s) for n, s in srcs}
    dead = {n: d for n, d in dead.items() if d}
    check('every uniform a compute kernel declares is read by its text '
          '(a GL linker has nothing to strip)', not dead, str(dead))
    check("the raster shell's dead tile-grid sizes are gone from every "
          'variant', not any('hal_rtiles_w' in s or 'hal_rtiles_h' in s
                             for n, s in srcs if n.startswith('HAL_RASTER')))
    check('...the tiles are still fetched by integer texel',
          all('texelFetch(hal_rtiles,' in s for n, s in srcs
              if n.startswith('HAL_RASTER')))
    # the CreateInfo's list against the text. (The text alone is not the
    # interface: _exact_core's fetcher swap carries hal_refer's
    # declaration away with the old fetcher, and the device declares
    # every push constant itself -- so the rule is READ, not declared.)
    unread, undeclared = {}, {}
    for name, src in srcs:
        if not name.startswith('HAL_RASTER'):
            continue
        body = re.sub(r'//[^\n]*', '', src)
        declared = [nm for kind, nm in fakedevice._DECL.findall(body)
                    if kind == 'float']
        for nm in CRA.RASTER_FLOATS:
            uses = len(re.findall(r'\b' + re.escape(nm) + r'\b', body)) \
                - declared.count(nm)
            if uses < 1:
                unread.setdefault(name, []).append(nm)
        extra = sorted(set(declared) - set(CRA.RASTER_FLOATS))
        if extra:
            undeclared[name] = extra
    check('every push constant in RASTER_FLOATS is READ by the text of all '
          'eight variants', not unread, str(unread))
    check('...and no variant declares a float the CreateInfo does not carry',
          not undeclared, str(undeclared))
    check('RASTER_FLOATS is 15 distinct push constants (60 bytes)',
          len(CRA.RASTER_FLOATS) == 15 and len(set(CRA.RASTER_FLOATS)) == 15)


def test_the_raster_sends_exactly_what_it_declares():
    """raster_on_device over a recording driver: the CreateInfo's floats
    and the dispatch's uniforms are one list, and a push constant the
    driver strips anyway does not fail the raster."""
    st = base_settings(48, 36)
    sc = demo_scene(st, with_texture=False)
    _view, _proj, vp, _eye = R.camera_matrices(sc.camera, 48, 36)
    seen = {}
    saved = {nm: getattr(DV, nm) for nm in ('compile_compute', 'upload')}

    def fake_compile(name, source, samplers=(), floats=(), images=()):
        seen.update(name=name, source=source, samplers=tuple(samplers),
                    floats=tuple(floats), images=tuple(images))
        # the driver strips one the text READS (dead code on some path is
        # a linker's call, not the text's): the door must take that too
        seen['shader'] = Stripping(stripped={'hal_refer'})
        return seen['shader'], None

    dispatched = []
    with clean_device(), fake_gl(dispatched=dispatched):
        DV.compile_compute = fake_compile
        DV.upload = lambda image: object()
        try:
            out, err = CRA.raster_on_device(sc.mesh, vp, 48, 36)
        finally:
            for nm, fn in saved.items():
                setattr(DV, nm, fn)
        check('the raster dispatch does not fail on a stripped push '
              "constant (no 'raster dispatch failed')",
              err is None and out is not None and len(dispatched) == 1,
              str(err))
        if out is not None:
            check('...and returns its two images at the frame size',
                  out['ids'].shape == (36, 48, 4)
                  and out['aux'].shape == (36, 48, 4))
        sh = seen.get('shader')
        asked = [n for m, n in sh.asked if m == 'uniform_float'] if sh else []
        check('the CreateInfo declares RASTER_FLOATS',
              seen.get('floats') == CRA.RASTER_FLOATS, str(seen.get('floats')))
        check('the dispatch sets every declared push constant once and '
              'nothing the CreateInfo does not declare',
              sorted(asked) == sorted(CRA.RASTER_FLOATS),
              str(sorted(set(asked) ^ set(CRA.RASTER_FLOATS))))
        check('neither carries the dead tile-grid sizes',
              'hal_rtiles_w' not in asked and 'hal_rtiles_h' not in asked
              and 'hal_rtiles_w' not in seen.get('floats', ()))
        check('the three data samplers are bound',
              sh is not None and sorted(sh.samplers) == [
                  'hal_rbins', 'hal_rc', 'hal_rtiles'])
        check('the stripped one is counted, once',
              DV.stripped_names() == ['hal_refer']
              and DV.stripped_stats() == (1, 1))


# ------------------------------------------------ (5) a frame on a GL-like driver

def _gpu_frame(installer, door=None, frames=1):
    """(images, printed, stats, names, shaders) of the demo frame on the
    GPU device under `installer` (a fakedevice context)."""
    st = base_settings(64, 48, render_device='GPU')
    R._GBUF_CACHE.clear()
    GSH._PLAN_CACHE.clear()
    buf = io.StringIO()
    imgs = []
    old_door = DV.set_uniform
    with installer() as dev:
        if door is not None:
            DV.set_uniform = door
        try:
            with redirect_stdout(buf):
                for _ in range(frames):
                    imgs.append(np.asarray(R.render(demo_scene(st), st)))
            stats, names = DV.stripped_stats(), DV.stripped_names()
            shaders = list(getattr(dev, 'gl_shaders', {}).values())
        finally:
            DV.set_uniform = old_door
    R._GBUF_CACHE.clear()
    GSH._PLAN_CACHE.clear()
    return imgs, buf.getvalue(), stats, names, shaders


def test_the_demo_frame_shades_on_a_driver_that_strips():
    plain, out_p, _s, _n, _sh = _gpu_frame(fakedevice.installed)
    gl, out_g, stats, names, shaders = _gpu_frame(fakedevice.installed_gl,
                                                  frames=2)
    mats = [s for s in shaders if s.name.startswith('HAL_MAT')]
    check('the demo frame plans three material passes and the sky on the '
          'GL-like driver', len(mats) == 3
          and any(s.name == 'HAL_SKY' for s in shaders),
          str([s.name for s in shaders]))
    axes = {'hal_cam_right', 'hal_cam_up', 'hal_cam_back'}
    check('that driver strips the camera axes from every material pass '
          '(no demo material fixes its key to the camera)',
          all(s.stripped == axes for s in mats),
          str([sorted(s.stripped) for s in mats]))
    check('the frame is SHADED ON THE GPU: no fallback to the per-pass '
          'readback, no CPU shading',
          'fell back to per-pass readback' not in out_g
          and 'shading on the CPU' not in out_g
          and '[Halcyon GPU] shade split' in out_g,
          '; '.join(l for l in out_g.splitlines() if 'Halcyon GPU' in l))
    check('...every material pass drew, each frame',
          all(s.draws == 2 for s in mats), str([s.draws for s in mats]))
    check('the frame is bit for bit the frame of a driver that strips '
          'nothing (a stripped uniform cannot change a pixel)',
          gl[0].shape == plain[0].shape
          and bool(np.array_equal(gl[0], plain[0])),
          str(float(np.abs(gl[0] - plain[0]).max())
              if gl[0].shape == plain[0].shape else gl[0].shape))
    check('the second frame is the same picture', bool(np.array_equal(
        gl[1], gl[0])))
    check('the count is ONCE PER SHADER: 9 uniforms in 3 shaders after two '
          'frames, the three axes by name',
          stats == (9, 3) and set(names) == axes, f'{stats} {names}')
    check('...and each program was asked about each stripped name once, '
          'in two frames', all(len(s.refused) == 3 for s in mats),
          str([len(s.refused) for s in mats]))

    # the door as it was: the field log, word for word
    _i, out_raw, _s2, _n2, _sh2 = _gpu_frame(fakedevice.installed_gl,
                                             door=raw_set)
    check("with the old unconditional set the same driver prints the "
          "field's burst line word for word",
          '[Halcyon GPU] blended compositing fell back to per-pass '
          f'readback: ValueError: {NOT_FOUND}' in out_raw,
          '; '.join(l for l in out_raw.splitlines() if 'Halcyon GPU' in l))
    check("...and the field's CPU line (drawing '<material>' failed)",
          re.search(r"\[Halcyon GPU\] shading on the CPU: drawing '[^']+' "
                    r'failed: ' + re.escape(NOT_FOUND), out_raw) is not None)
    check('the GL road leaves the device as it found it',
          not fakedevice._BOUND and fakedevice._GL['blend'] == 'NONE'
          and DV.draw_many.__module__.endswith('gpu.device'))


# ------------------------------------------------------- (6) the Debug breakdown

def test_the_breakdown_names_the_count():
    with clean_device():
        check('a driver that kept everything prints nothing',
              DV.stripped_note() == '' and DV.stripped_stats() == (0, 0))
        a = Stripping(stripped={'hal_cam_right'})
        DV.set_uniform(a, 'float', 'hal_cam_right', (1.0, 0.0, 0.0))
        check("one: '1 uniform stripped by the driver in 1 shader "
              "(hal_cam_right)'",
              DV.stripped_note().startswith(
                  '1 uniform stripped by the driver in 1 shader '
                  '(hal_cam_right): '), DV.stripped_note())
        b = Stripping(stripped={'hal_cam_right', 'hal_cam_up', 'hal_frame'})
        for nm in ('hal_cam_right', 'hal_cam_up', 'hal_frame',
                   'hal_cam_right'):
            DV.set_uniform(b, 'float', nm, 1.0)
        note = DV.stripped_note()
        check("several: 'N uniforms stripped by the driver in K shaders', "
              'each name once per shader, the distinct names sorted',
              note.startswith('4 uniforms stripped by the driver in 2 '
                              'shaders (hal_cam_right, hal_cam_up, '
                              'hal_frame): '), note)
        check('the line says what it means for the picture',
              'each set was skipped and the pass drew' in note)
        c = Stripping(stripped={f'hal_u{i}' for i in range(9)})
        for i in range(9):
            DV.set_uniform(c, 'float', f'hal_u{i}', 1.0)
        check('a long list is cut at six names with the rest counted',
              '(+6 more)' in DV.stripped_note()
              and DV.stripped_note().startswith('13 uniforms stripped by '
                                                'the driver in 3 shaders'),
              DV.stripped_note())

        # reset(): a new driver world links its own programs
        saved = {k: (dict(v) if isinstance(v, dict) else v)
                 for k, v in DV._STATE.items()}
        g0 = list(DV._GRAVEYARD)
        sg0 = list(DV._SCREEN_GRAVE)
        try:
            DV.reset()
            parked = [o for _e, _dl, o in DV._GRAVEYARD]
            check('reset() forgets the stripped names and parks the shaders '
                  'it held (never a direct free)',
                  DV.stripped_stats() == (0, 0) and DV.stripped_note() == ''
                  and all(any(o is s for o in parked) for s in (a, b, c)))
        finally:
            DV._STATE.clear()
            DV._STATE.update(saved)
            DV._GRAVEYARD[:] = g0
            DV._SCREEN_GRAVE[:] = sg0

    root = os.path.dirname(os.path.dirname(os.path.abspath(R.__file__)))
    esrc = open(os.path.join(root, 'engine.py'), encoding='utf-8').read()
    tail = esrc.split('ST.report(total=elapsed)', 1)
    check('the F12 Timing Breakdown prints the line under its table',
          len(tail) == 2 and '_gpu_stripped_note()' in tail[1][:400]
          and 'def _gpu_stripped_note' in esrc)
    ssrc = open(os.path.join(root, 'selftest.py'), encoding='utf-8').read()
    check("the self-test's frame breakdown prints it too",
          'stripped_note()' in ssrc.split('def frame_breakdown', 1)[1]
          .split('\ndef ', 1)[0])
    from . import fakebpy
    fakebpy.install()
    import importlib
    eng = importlib.import_module(__package__.rsplit('.', 1)[0] + '.engine')
    with clean_device():
        check('engine._gpu_stripped_note is empty on a clean device',
              eng._gpu_stripped_note() == '')
        DV.set_uniform(Stripping(stripped={'hal_cam_up'}), 'float',
                       'hal_cam_up', 1.0)
        check("...and the device's line once a uniform was stripped",
              eng._gpu_stripped_note() == DV.stripped_note()
              and eng._gpu_stripped_note().startswith(
                  '1 uniform stripped by the driver'))


# ------------------------------------------------------------- (7) the one door

def test_nothing_else_sets_a_uniform():
    root = os.path.dirname(os.path.dirname(os.path.abspath(R.__file__)))
    gdir = os.path.join(root, 'gpu')
    call = re.compile(r'\.uniform_(?:float|int|bool|sampler|block|'
                      r'vector_float|vector_int)\s*\(')
    hits = []
    for fn in sorted(os.listdir(gdir)):
        if not fn.endswith('.py'):
            continue
        for i, line in enumerate(open(os.path.join(gdir, fn),
                                      encoding='utf-8'), 1):
            code = line.split('#', 1)[0]
            if call.search(code):
                hits.append(f'gpu/{fn}:{i}')
    check('no file under gpu/ calls a GPUShader uniform_* method directly',
          not hits, str(hits))
    dsrc = open(os.path.join(gdir, 'device.py'), encoding='utf-8').read()
    door = dsrc.split('def set_uniform', 1)[1].split('\ndef ', 1)[0]
    check('device.set_uniform reaches the four setters by name '
          '(uniform_sampler, uniform_bool, uniform_int, uniform_float)',
          sorted(DV._UNIFORM_SETTERS.values()) == [
              'uniform_bool', 'uniform_float', 'uniform_int',
              'uniform_sampler']
          and 'getattr(shader, _UNIFORM_SETTERS[kind])(name, value)' in door)
    for fn_name in ('_draw_in_bound', '_dispatch_compute_impl'):
        body = dsrc.split(f'def {fn_name}', 1)[1].split('\ndef ', 1)[0]
        check(f'{fn_name} sets its samplers and uniforms through the door',
              body.count('set_uniform(shader, ') >= 2, str(body.count(
                  'set_uniform(shader, ')))
    check("the compute dispatch still binds its images directly (an "
          "output is never skipped)",
          'shader.image(k, tex)' in dsrc.split(
              'def _dispatch_compute_impl', 1)[1])


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
    print('all R253 stripped-uniform checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
