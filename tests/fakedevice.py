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
"""

import contextlib
import re

import numpy as np

from ..core.texture import Texture
from ..shaders.compiler import try_compile


class FakeTexture:
    def __init__(self, arr):
        a = np.ascontiguousarray(np.asarray(arr, np.float32))
        if a.ndim == 3 and a.shape[2] == 3:
            a = np.concatenate([a, np.ones(a.shape[:2] + (1,), np.float32)], 2)
        self._arr = a

    @property
    def pixels(self):
        return self._arr

    @property
    def width(self):
        return self._arr.shape[1]

    @property
    def height(self):
        return self._arr.shape[0]


class TargetTexture:
    """A target's colour texture: reads the target's CURRENT pixels."""

    def __init__(self, target):
        self.target = target

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
