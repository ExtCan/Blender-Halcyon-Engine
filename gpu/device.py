"""Blender GPU module access, with every failure mode handled.

This is the layer EEVEE sits on, and the only GPU route available to an add-on:
Cycles' device abstraction is C++ with precompiled kernels and is not exposed to
Python at all. What `gpu` gives us is GLSL shaders, offscreen framebuffers and
textures, which is enough to move the parallel parts of the post chain off the
CPU.

Everything here is written to fail into the CPU path rather than to fail. A GPU
that will not give a context, a driver that rejects a shader, a Blender build
without the module -- each returns a reason and the renderer carries on. The
engine has always worked without this and must continue to.
"""

import sys

_STATE = {'checked': False, 'available': False, 'reason': '', 'shaders': {}}


def reset():
    _STATE.update(checked=False, available=False, reason='')
    _STATE['shaders'].clear()


def probe():
    """(available, reason). Cheap after the first call."""
    if _STATE['checked']:
        return _STATE['available'], _STATE['reason']
    _STATE['checked'] = True
    try:
        import gpu                                              # noqa: F401
    except Exception as exc:                                    # noqa: BLE001
        _STATE['reason'] = f'the gpu module is unavailable ({exc})'
        return False, _STATE['reason']
    try:
        import gpu
        backend = getattr(gpu.platform, 'backend_type_get', lambda: 'UNKNOWN')()
        renderer = getattr(gpu.platform, 'renderer_get', lambda: '?')()
    except Exception as exc:                                    # noqa: BLE001
        _STATE['reason'] = f'the gpu module would not report a backend ({exc})'
        return False, _STATE['reason']
    _STATE['available'] = True
    _STATE['reason'] = f'{backend} on {renderer}'
    return True, _STATE['reason']


def describe():
    ok, why = probe()
    return ('GPU available: ' + why) if ok else ('GPU unavailable: ' + why)


def compile_stage(name, fragment, vertex=None):
    """Compile and cache one full-screen shader. Returns (shader, error)."""
    key = (name, hash(fragment))
    hit = _STATE['shaders'].get(key)
    if hit is not None:
        return hit, None
    ok, why = probe()
    if not ok:
        return None, why
    shader, err = _build(name, fragment, vertex)
    if shader is None:
        return None, err
    _STATE['shaders'][key] = shader
    return shader, None


def _build(name, fragment, vertex=None):
    """Build a shader, preferring CreateInfo.

    The legacy GPUShader(vertex, fragment) constructor exists only on the
    OpenGL backend. Blender defaults to Vulkan on Windows now, where it raises
    "cannot create 'GPUShader' instances" -- so CreateInfo is tried first and
    the old constructor is only a fallback for older builds.
    """
    import gpu

    from .stages import INTERFACE, body

    spec = INTERFACE.get(name)
    create_err = f'{name}: no CreateInfo path'
    if spec is not None and hasattr(gpu.types, 'GPUShaderCreateInfo'):
        try:
            iface = gpu.types.GPUStageInterfaceInfo('halcyon_' + name.lower())
            iface.smooth('VEC2', 'vUV')
            info = gpu.types.GPUShaderCreateInfo()
            info.vertex_in(0, 'VEC2', 'pos')
            info.vertex_in(1, 'VEC2', 'uv')
            info.vertex_out(iface)
            info.fragment_out(0, 'VEC4', 'Color')
            for i, samp in enumerate(spec.get('samplers', ())):
                info.sampler(i, 'FLOAT_2D', samp)
            kinds = {'floats': 'FLOAT', 'ints': 'INT', 'vec2': 'VEC2',
                     'vec3': 'VEC3'}
            for key_name, kind in kinds.items():
                for pc in spec.get(key_name, ()):
                    info.push_constant(kind, pc)
            info.vertex_source(VERTEX_CORE)
            info.fragment_source(body(name))
            return gpu.shader.create_from_info(info), None
        except Exception as exc:                                # noqa: BLE001
            create_err = f'{name}: CreateInfo failed: {exc}'

    try:
        return gpu.types.GPUShader(vertex or VERTEX, fragment), None
    except Exception as exc:                                    # noqa: BLE001
        return None, create_err + f'; legacy constructor also failed: {exc}'


VERTEX_CORE = """
void main()
{
    vUV = uv;
    gl_Position = vec4(pos, 0.0, 1.0);
}
"""

# legacy OpenGL path declares its own interface
VERTEX = """
in vec2 pos;
in vec2 uv;
out vec2 vUV;
void main()
{
    vUV = uv;
    gl_Position = vec4(pos, 0.0, 1.0);
}
"""


class Target:
    """An offscreen colour buffer to render a full-screen pass into."""

    def __init__(self, width, height, fmt='RGBA32F'):
        import gpu
        self.width = int(width)
        self.height = int(height)
        self.offscreen = gpu.types.GPUOffScreen(self.width, self.height,
                                                format=fmt)

    def free(self):
        try:
            self.offscreen.free()
        except Exception:                                       # noqa: BLE001
            pass


def upload(image):
    """(H,W,4) float32 -> a GPUTexture, bottom row first as everywhere else."""
    import gpu
    import numpy as np
    arr = np.ascontiguousarray(np.asarray(image, np.float32))
    h, w = arr.shape[:2]
    if arr.shape[2] == 3:
        arr = np.concatenate([arr, np.ones((h, w, 1), np.float32)], axis=2)
    buf = gpu.types.Buffer('FLOAT', h * w * 4, arr.ravel().tolist())
    # 32-bit float throughout: RGBA16F carries about eleven bits of mantissa,
    # and a gamma curve turns that into visible error near black
    return gpu.types.GPUTexture((w, h), format='RGBA32F', data=buf)


def draw_fullscreen(shader, uniforms, samplers, target):
    """Run one full-screen pass. Returns the result as (H,W,4) float32."""
    import gpu
    import numpy as np
    from gpu_extras.batch import batch_for_shader

    batch = batch_for_shader(
        shader, 'TRI_FAN',
        {'pos': ((-1, -1), (1, -1), (1, 1), (-1, 1)),
         'uv': ((0, 0), (1, 0), (1, 1), (0, 1))})

    with target.offscreen.bind():
        gpu.state.blend_set('NONE')
        gpu.state.depth_test_set('NONE')
        shader.bind()
        for k, v in (samplers or {}).items():
            shader.uniform_sampler(k, v)
        for k, v in (uniforms or {}).items():
            if isinstance(v, bool):
                shader.uniform_bool(k, [v])
            elif isinstance(v, int):
                shader.uniform_int(k, v)
            elif isinstance(v, (tuple, list)):
                shader.uniform_float(k, v)
            else:
                shader.uniform_float(k, float(v))
        batch.draw(shader)
        buf = target.offscreen.texture_color.read()
        buf.dimensions = target.width * target.height * 4
    return np.asarray(buf, np.float32).reshape(target.height, target.width, 4)
