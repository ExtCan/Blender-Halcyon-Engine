"""A stand-in for gpu/device that runs draws through the GLSL front-end.

The driver road of a GPU pass (gpu/ink.apply, gpu/shade.shade_frame, the
post chain) is plumbing over the device module: uploads, cached uploads,
dynamic compiles, targets, a burst of draws and one readback. None of it
can run without Blender's gpu module, and every field defect in that
plumbing -- a uniform of the wrong kind, a sampler declared but never
bound, a pass sampling the target it draws into -- was invisible to a
suite that only ran the simulator. This module swaps in for the device's
functions and applies the DRIVER's rules: every declared uniform must be
in the CreateInfo spec, every spec sampler must be bound at the draw, an
int must arrive as an int and a float as a float, and no draw may read
its own target.

R250: the fake also keeps the driver's per-draw STATE and its target
POOL, because the passes now compose in one target and keep targets
alive across stages: `clear` zeroes the target first (full-frame, all
four channels), blend NONE overwrites, ALPHA_PREMULT composites
`src + dst * (1 - src.a)`, a scissor `region` writes only inside its
rectangle, a region readback returns the rectangle; `free()` returns a
target to a per-size pool and the next `Target(w, h)` hands back the
SAME object with its stale pixels (as the driver's offscreen pool does),
so a handle held past `free()` reads the next stage's scratch -- and a
draw that samples a freed target is refused by name. A lane the shader
`discard`s leaves its destination pixel untouched (the sky pass draws
LAST and discards where a material drew).

    with fakedevice.installed() as calls:
        got, why = GINK.apply(...)

R253 (field): `installed()` swaps the device's DRAW functions out, so the
one loop that talks to a GPUShader (device._draw_in_bound) never ran
headless -- and a GL driver that strips an unread uniform failed every
pass of every frame in the field without one check moving.
`installed_gl()` keeps this fake's compiles, uploads and targets and puts
the REAL device.draw_many / draw_fullscreen / read_target back in the
loop, over a stand-in `gpu` module whose shaders behave as the OpenGL
backend links them: a uniform the source declares and never reads is
gone from the program, and setting it raises Blender's own ValueError
('GPUShader.uniform_float: uniform hal_cam_right not found').

    with fakedevice.installed_gl() as dev:
        img = R.render(scene, settings)          # render_device='GPU'
        dev.gl_shaders                           # what each program lost
"""

import contextlib
import re

import numpy as np

from ..core.texture import Texture
from ..shaders.compiler import try_compile


class FakeTexture:
    """An uploaded texture. R254: a 2D (H, W) array is an R32F texture
    (gpu/device._upload_impl's rule) and keeps its one channel; `fmt`
    names what the driver would hold, for the tests that count bytes."""

    def __init__(self, arr):
        a = np.ascontiguousarray(np.asarray(arr, np.float32))
        if a.ndim == 3 and a.shape[2] == 3:
            a = np.concatenate([a, np.ones(a.shape[:2] + (1,), np.float32)], 2)
        self._arr = a
        self.fmt = 'R32F' if a.ndim == 2 else 'RGBA32F'

    @property
    def pixels(self):
        return self._arr

    @property
    def width(self):
        return self._arr.shape[1]

    @property
    def height(self):
        return self._arr.shape[0]


#: R253: the GL road's framebuffer binding (a stack: binds nest) and the
#: per-draw state gpu.state carries between the device's calls
_BOUND = []
_GL = {'blend': 'NONE', 'scissor': None, 'scissor_on': False}


class _Bind:
    """`with offscreen.bind():` -- the target becomes the framebuffer."""

    def __init__(self, target):
        self.target = target

    def __enter__(self):
        _BOUND.append(self.target)
        return self

    def __exit__(self, *exc):
        _BOUND.pop()
        return False


class _Buffer:
    """A readback as gpu.types.Buffer hands it over: flat floats, the
    buffer protocol (here __array__) and a settable `dimensions`."""

    def __init__(self, arr):
        self._a = np.ascontiguousarray(arr, np.float32).ravel()
        self.dimensions = int(self._a.size)

    def __array__(self, dtype=None, copy=None):
        return self._a if dtype is None else self._a.astype(dtype, copy=False)


class TargetTexture:
    """A target's colour texture: reads the target's CURRENT pixels."""

    def __init__(self, target):
        self.target = target

    def read(self):
        # R253: GPUTexture.read(), for the real device's readback
        return _Buffer(self.target.pixels)

    @property
    def pixels(self):
        return self.target.pixels

    @property
    def width(self):
        return self.target.width

    @property
    def height(self):
        return self.target.height


class Target:
    def __init__(self, width, height, fmt='RGBA32F'):
        self.width, self.height = int(width), int(height)
        self.pixels = np.zeros((self.height, self.width, 4), np.float32)
        self.freed = False
        self.reused = 0
        # the real Target's handle, so callers that test `offscreen is
        # None` (a released resident) read the fake the same way
        self.offscreen = self

    def free(self):
        self.freed = True
        self.offscreen = None

    # R253: the GPUOffScreen face (`offscreen` is this object), for the
    # GL road that runs the REAL device draws over the fake's targets
    def bind(self):
        return _Bind(self)

    @property
    def texture_color(self):
        return TargetTexture(self)


class Shader:
    def __init__(self, name, prog, spec):
        self.name, self.prog, self.spec = name, prog, spec


_DECL = re.compile(r'^\s*uniform\s+(\w+)\s+(\w+)\s*;', re.M)


class Device:
    def __init__(self):
        self.calls = {'compile': 0, 'upload': 0, 'cached': 0, 'draws': 0,
                      'targets': 0, 'reads': 0, 'replaced': 0}
        self.cache = {}
        self.stamps = {}      # R251 C015: key -> data stamp
        self.shaders = {}
        self.targets = []
        self.pool = {}
        self.pool_on = True

    def probe(self):
        return True, 'the fake device'

    def upload(self, image):
        self.calls['upload'] += 1
        return FakeTexture(image)

    def upload_cached(self, key, build, stamp=None):
        # R251 C015: gpu/device.upload_cached's stamp rule, mirrored --
        # a hit with a different stamp replaces the texture under the
        # SAME key (counted in calls['replaced']); None = the old road
        self.calls['cached'] += 1
        hit = self.cache.get(key)
        if hit is not None and stamp is not None and \
                self.stamps.get(key) != stamp:
            self.calls['replaced'] += 1
            hit = None
        if hit is None:
            hit = self.cache[key] = FakeTexture(build())
            if stamp is not None:
                self.stamps[key] = stamp
        return hit

    def compile_dynamic(self, name, fragment, spec):
        key = (name, hash(fragment))
        hit = self.shaders.get(key)
        if hit is not None:
            return hit, None
        self.calls['compile'] += 1
        prog, err = try_compile(fragment.replace('in vec2 vUV;',
                                                 'uniform vec2 vUV;'), 'GLSL')
        if prog is None:
            return None, f'{name}: {err}'
        declared = sum(spec.values(), [])
        for _kind, nm in _DECL.findall(fragment):
            if nm not in declared:
                return None, (f'{name}: uniform {nm} is declared but not in '
                              'the CreateInfo spec')
        sh = self.shaders[key] = Shader(name, prog, spec)
        return sh, None

    def compile_stage(self, name, fragment, vertex=None):
        """The post chain's static registry: its spec from stages.INTERFACE."""
        from ..gpu.stages import INTERFACE
        spec = {'samplers': [], 'floats': [], 'ints': [], 'vec2': [],
                'vec3': []}
        spec.update({k: list(v) for k, v in INTERFACE.get(name, {}).items()})
        return self.compile_dynamic(name, fragment, spec)

    def target_texture(self, target):
        return TargetTexture(target)

    def Target(self, width, height, fmt='RGBA32F'):
        key = (int(width), int(height), fmt)
        pool = self.pool.setdefault(key, [])
        if self.pool_on and pool:
            # the driver's pool hands the SAME offscreen back, stale
            # pixels and all: a handle held past free() sees them
            t = pool.pop()
            t.freed = False
            t.offscreen = t
            t.reused += 1
            return t
        self.calls['targets'] += 1
        t = Target(width, height, fmt)
        self.targets.append(t)
        return t

    def _free(self, target):
        if target.freed:
            return
        target.freed = True
        target.offscreen = None
        key = (target.width, target.height, 'RGBA32F')
        self.pool.setdefault(key, []).append(target)

    def _draw(self, shader, uniforms, samplers, target, blend='NONE',
              clear=False, region=None):
        spec = shader.spec
        w, h = target.width, target.height
        n = w * h
        yy, xx = np.mgrid[0:h, 0:w]
        u = {'vUV': np.stack([(xx.ravel() + 0.5) / w,
                              (yy.ravel() + 0.5) / h], 1).astype(np.float32)}
        for k, v in (samplers or {}).items():
            if k not in spec['samplers']:
                raise AssertionError(f'{shader.name}: sampler {k} not in spec')
            if isinstance(v, TargetTexture):
                if v.target is target:
                    raise AssertionError(f'{shader.name}: samples its own '
                                         'target')
                if v.target.freed:
                    raise AssertionError(f'{shader.name}: samples a freed '
                                         'target')
            if getattr(v, 'fmt', 'RGBA32F') == 'R32F':
                # one channel: the simulator's R32F sampler, (r, 0, 0, 1)
                from ..gpu.shade import RedTexture
                u[k] = RedTexture(v.pixels)
                continue
            u[k] = Texture(v.pixels, colorspace='Non-Color', filt='NEAREST',
                           wrap='EXTEND')
        for k in spec['samplers']:
            if k not in (samplers or {}):
                raise AssertionError(f'{shader.name}: sampler {k} unbound')
        for k, v in (uniforms or {}).items():
            if isinstance(v, bool):
                raise AssertionError(f'{shader.name}: bool uniform {k}')
            if isinstance(v, int):
                if k not in spec.get('ints', ()):
                    raise AssertionError(f'{shader.name}: int for {k}')
                u[k] = np.full(n, int(v), np.int32)
            elif isinstance(v, (tuple, list)):
                kind = 'vec2' if len(v) == 2 else 'vec3'
                if k not in spec.get(kind, ()):
                    raise AssertionError(f'{shader.name}: {kind} for {k}')
                u[k] = np.tile(np.asarray(v, np.float32)[None, :], (n, 1))
            else:
                if k not in spec.get('floats', ()):
                    raise AssertionError(f'{shader.name}: float for {k}')
                u[k] = np.full(n, float(v), np.float32)
        outs, discarded = shader.prog.run(u, {}, n)
        src = np.asarray(outs['Color'], np.float32).reshape(h, w, 4)
        if clear:
            # full-frame, scissor off, every channel: device._draw_in_bound
            target.pixels = np.zeros((h, w, 4), np.float32)
        dst = target.pixels
        if blend == 'ALPHA_PREMULT':
            out = src + dst * (1.0 - src[:, :, 3:4])
        else:
            out = src
        out = np.asarray(out, np.float32)
        # a discarded fragment writes nothing: the texel stands
        disc = np.asarray(discarded, bool).reshape(h, w)
        if disc.any():
            out = np.where(disc[:, :, None], dst, out)
        if region is not None:
            x, y, rw, rh = (int(v) for v in region)
            keep = dst.copy()
            keep[y:y + rh, x:x + rw] = out[y:y + rh, x:x + rw]
            out = keep
        target.pixels = np.ascontiguousarray(out, np.float32)

    def _read(self, read, read_region=None):
        self.calls['reads'] += 1
        if read_region is not None:
            x, y, rw, rh = (int(v) for v in read_region)
            return read.pixels[y:y + rh, x:x + rw].copy()
        return read.pixels.copy()

    def draw_many(self, draws, read=None, read_region=None):
        for shader, uniforms, samplers, target, blend, clear, region in draws:
            if target.freed:
                raise AssertionError(f'{shader.name}: draws into a freed target')
            self.calls['draws'] += 1
            self._draw(shader, uniforms, samplers, target, blend, clear, region)
        return self._read(read, read_region) if read is not None else None

    def draw_fullscreen(self, shader, uniforms, samplers, target, read=True,
                        blend='NONE', clear=False, region=None):
        if target.freed:
            raise AssertionError(f'{shader.name}: draws into a freed target')
        self.calls['draws'] += 1
        self._draw(shader, uniforms, samplers, target, blend, clear, region)
        return self._read(target) if read else None

    def read_target(self, target, region=None):
        if target.freed:
            raise AssertionError('reads a freed target')
        return self._read(target, region)


_NAMES = ('probe', 'upload', 'upload_cached', 'compile_dynamic',
          'compile_stage', 'target_texture', 'draw_many', 'draw_fullscreen',
          'read_target', 'Target')


@contextlib.contextmanager
def installed():
    """Swap the fake in for gpu/device's functions; yields the Device."""
    from ..gpu import device as real
    fake = Device()
    saved = {nm: getattr(real, nm) for nm in _NAMES}
    for nm in _NAMES:
        setattr(real, nm, getattr(fake, nm))
    # Target.free() must return the fake target to the fake's pool: the
    # real module's free() is a method on the real class, so the fake
    # class carries its own that reports to the device
    Target.free = lambda self, _dev=fake: _dev._free(self)
    try:
        yield fake
    finally:
        for nm, fn in saved.items():
            setattr(real, nm, fn)
        Target.free = _plain_free


def _plain_free(self):
    self.freed = True
    self.offscreen = None


Target.free = _plain_free


# ------------------------------------------------------------------ the GL road

def unread_uniforms(fragment, names=None):
    """The uniforms a source never READS, comments aside: of `names`, or
    (None) of the names the text itself declares.

    A name is read when it occurs anywhere but its own declaration. What
    a GL linker strips at the very least -- a real one also drops a name
    whose every use is dead code, so this is the floor, and it is
    deterministic: the fake strips exactly these. `names` is how a
    CreateInfo shader is asked: its interface is the SPEC (the device
    declares every push constant in the generated header, whether or
    not the text mentions it), so a spec name the text never touches is
    stripped exactly like one it declares and ignores."""
    body = re.sub(r'//[^\n]*', '', str(fragment or ''))
    body = re.sub(r'/\*.*?\*/', '', body, flags=re.S)
    declared = [nm for _kind, nm in _DECL.findall(body)]
    out = []
    for nm in (declared if names is None else names):
        uses = len(re.findall(r'\b' + re.escape(nm) + r'\b', body)) \
            - declared.count(nm)
        if uses < 1 and nm not in out:
            out.append(nm)
    return out


class GLShader:
    """A GPUShader as Blender's OpenGL backend links it.

    Wraps the fake's compiled Shader. The program keeps only the
    push constants its source reads: uniform_float / uniform_int for a
    stripped name raise the ValueError Blender raises (its own format,
    `%s: uniform %.32s not found`), recorded in `refused`. Samplers are
    simply kept: Blender's uniform_sampler carries no such refusal (the
    slot comes from the CreateInfo), and the device's door would skip
    one that did. Values persist between draws, as a program's uniforms
    do."""

    def __init__(self, inner, fragment):
        self.inner = inner
        self.name = inner.name
        self.spec = inner.spec
        # the CreateInfo's push constants (the spec), not the text's
        # declarations, are what the linker was handed
        pcs = [nm for kind in ('floats', 'ints', 'vec2', 'vec3')
               for nm in inner.spec.get(kind, ())]
        self.stripped = set(unread_uniforms(fragment, pcs))
        self.values = {}
        self.samplers = {}
        self.refused = []
        self.draws = 0

    def bind(self):
        pass

    def _set(self, method, name, value):
        if name in self.stripped:
            self.refused.append((method, name))
            raise ValueError(f'GPUShader.{method}: uniform {name:.32s} '
                             'not found')
        self.values[name] = value

    def uniform_float(self, name, value):
        self._set('uniform_float', name, value)

    def uniform_int(self, name, value):
        self._set('uniform_int', name, value)

    def uniform_bool(self, name, value):
        # the fake's own rule: no pass hands a bool over
        raise AssertionError(f'{self.name}: bool uniform {name}')

    def uniform_sampler(self, name, tex):
        self.samplers[name] = tex


class _GLBatch:
    """batch_for_shader's quad: draw() runs the pass through the fake's
    own front-end, into whatever target is bound, under the state the
    device set (blend, scissor) -- with the uniforms the PROGRAM holds,
    which is every one the device set and none it had to skip."""

    def __init__(self, dev):
        self.dev = dev

    def draw(self, shader):
        if not _BOUND:
            raise AssertionError(f'{shader.name}: draws with no '
                                 'framebuffer bound')
        target = _BOUND[-1]
        if target.freed:
            raise AssertionError(f'{shader.name}: draws into a freed target')
        self.dev.calls['draws'] += 1
        shader.draws += 1
        region = _GL['scissor'] if _GL['scissor_on'] else None
        # the clear already happened (the device clears through the
        # framebuffer before the draw), so _draw never clears here
        self.dev._draw(shader.inner, dict(shader.values),
                       dict(shader.samplers), target, _GL['blend'], False,
                       region)


class _GLFramebuffer:
    def __init__(self, dev, target):
        self.dev, self.target = dev, target

    def clear(self, color=(0.0, 0.0, 0.0, 0.0)):
        t = self.target
        px = np.zeros((t.height, t.width, 4), np.float32)
        px[:, :] = np.asarray(color, np.float32)
        t.pixels = px

    def read_color(self, x, y, w, h, channels, slot, fmt):
        self.dev.calls['reads'] += 1
        return _Buffer(self.target.pixels[y:y + h, x:x + w])


def _gl_modules(dev):
    """The `gpu` / `gpu_extras.batch` stand-ins the real device imports."""
    import types as _t
    gpu = _t.ModuleType('gpu')
    gpu.state = _t.SimpleNamespace(
        active_framebuffer_get=lambda: _GLFramebuffer(dev, _BOUND[-1]),
        blend_set=lambda mode: _GL.__setitem__('blend', mode),
        depth_test_set=lambda mode: None,
        scissor_set=lambda x, y, w, h: _GL.__setitem__(
            'scissor', (int(x), int(y), int(w), int(h))),
        scissor_test_set=lambda on: _GL.__setitem__('scissor_on', bool(on)))
    gpu.types = _t.SimpleNamespace(Buffer=object, GPUTexture=object)
    batch = _t.ModuleType('gpu_extras.batch')
    batch.batch_for_shader = lambda shader, kind, attrs: _GLBatch(dev)
    extras = _t.ModuleType('gpu_extras')
    extras.batch = batch
    return {'gpu': gpu, 'gpu_extras': extras, 'gpu_extras.batch': batch}


@contextlib.contextmanager
def installed_gl():
    """The fake with the REAL device draws in the loop, on a GL-like
    driver that strips unread uniforms; yields the Device, whose
    `gl_shaders` maps the compile key to each GLShader.

    Everything the real functions touch in the device module is put
    back on exit: the batch cache, the graveyard, the flush epoch, the
    burst instrument and the stripped-uniform memory (which starts
    EMPTY inside, so a test reads exactly what its own frames lost)."""
    import sys

    from ..gpu import device as real
    real_fns = {nm: getattr(real, nm)
                for nm in ('draw_many', 'draw_fullscreen', 'read_target')}
    absent = object()
    with installed() as fake:
        fake.gl_shaders = {}
        mods = _gl_modules(fake)
        saved_mods = {k: sys.modules.get(k, absent) for k in mods}
        saved_state = {k: (dict(real._STATE[k]) if k in real._STATE
                           else absent) for k in ('batches',)}
        saved_epoch = real._STATE.get('flush_epoch', absent)
        saved_grave = list(real._GRAVEYARD)
        saved_stripped = dict(real._STRIPPED)
        saved_burst = dict(real.LAST_BURST)
        saved_gl = dict(_GL)

        def _wrap(name, fragment, inner):
            key = (name, hash(fragment))
            gl = fake.gl_shaders.get(key)
            if gl is None or gl.inner is not inner:
                gl = fake.gl_shaders[key] = GLShader(inner, fragment)
            return gl

        def compile_dynamic(name, fragment, spec):
            inner, err = fake.compile_dynamic(name, fragment, spec)
            if inner is None:
                return None, err
            return _wrap(name, fragment, inner), None

        def compile_stage(name, fragment, vertex=None):
            inner, err = fake.compile_stage(name, fragment, vertex)
            if inner is None:
                return None, err
            return _wrap(name, fragment, inner), None

        try:
            sys.modules.update(mods)
            real._STRIPPED.clear()
            # past the forensics window: flush_tick writes its first
            # three completions to the crash log, a session's business
            real._STATE['flush_epoch'] = 1000
            _GL.update(blend='NONE', scissor=None, scissor_on=False)
            del _BOUND[:]
            real.compile_dynamic = compile_dynamic
            real.compile_stage = compile_stage
            for nm, fn in real_fns.items():
                setattr(real, nm, fn)
            yield fake
        finally:
            for k, v in saved_mods.items():
                if v is absent:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v
            for k, v in saved_state.items():
                if v is absent:
                    real._STATE.pop(k, None)
                else:
                    real._STATE[k] = v
            if saved_epoch is absent:
                real._STATE.pop('flush_epoch', None)
            else:
                real._STATE['flush_epoch'] = saved_epoch
            real._GRAVEYARD[:] = saved_grave
            real._STRIPPED.clear()
            real._STRIPPED.update(saved_stripped)
            real.LAST_BURST.clear()
            real.LAST_BURST.update(saved_burst)
            _GL.clear()
            _GL.update(saved_gl)
            del _BOUND[:]
