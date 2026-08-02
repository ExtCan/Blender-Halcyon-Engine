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


def _main(what, fn):
    """The device boundary crossing: GL work runs on the main thread.

    When the render worker holds no GPU context (gpu/marshal.py), every
    genuine driver operation -- compile, upload, draw, dispatch, read --
    crosses here, and ONLY those: cache hits return on the calling
    thread, and the worker's CPU stretches (packing, ray building,
    composites, height chains) never block the interface at all. Each
    crossing is milliseconds, so the interface breathes between them.
    With marshalling off, or on the main thread already, fn runs in
    place; this is the whole cost.
    """
    from . import marshal
    return marshal.run_on_main(fn, what=what)


def reset():
    _STATE.update(checked=False, available=False, reason='')
    _STATE['shaders'].clear()


def probe():
    """(available, reason). Cheap after the first call."""
    if _STATE['checked']:
        return _STATE['available'], _STATE['reason']
    return _main('the GPU probe', _probe_impl)


def _probe_impl():
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

    def _miss():
        ok, why = probe()
        if not ok:
            return None, why
        shader, err = _build(name, fragment, vertex)
        if shader is None:
            return None, err
        _STATE['shaders'][key] = shader
        return shader, None

    return _main(f'compiling {name}', _miss)


def compile_dynamic(name, fragment, spec):
    """Compile a shader whose interface is given, not looked up.

    The post stages carry their interfaces in a static registry; a material's
    frame shader is generated per scene, so its spec -- samplers and the few
    push constants -- comes in as an argument. Cached on the source hash, so
    an unchanged scene compiles once however many frames it renders.
    """
    key = (name, hash(fragment))
    hit = _STATE['shaders'].get(key)
    if hit is not None:
        return hit, None
    return _main(f'compiling {name}', lambda: _compile_dynamic_miss(
        key, name, fragment, spec))


def _compile_dynamic_miss(key, name, fragment, spec):
    ok, why = probe()
    if not ok:
        return None, why

    import gpu

    create_err = f'{name}: no CreateInfo path'
    if hasattr(gpu.types, 'GPUShaderCreateInfo'):
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
            info.fragment_source(strip_declarations(fragment))
            shader = gpu.shader.create_from_info(info)
            _STATE['shaders'][key] = shader
            return shader, None
        except Exception as exc:                                # noqa: BLE001
            create_err = f'{name}: CreateInfo failed: {exc}'

    try:
        shader = gpu.types.GPUShader(VERTEX, fragment)
        _STATE['shaders'][key] = shader
        return shader, None
    except Exception as exc:                                    # noqa: BLE001
        return None, create_err + f'; legacy constructor also failed: {exc}'


def strip_declarations(fragment):
    """Remove global uniform/in/out declarations for a CreateInfo build.

    CreateInfo carries the interface itself, declares every resource in its
    own generated header, and the driver then rejects a source that declares
    the same name again. Only *global* declarations go: a `uniform` never
    appears inside a function body, and local variables are none of this
    function's business.

    The trailing-comment case is load-bearing. The first version required the
    line to *end* with the semicolon, so `uniform sampler2D x;  // note`
    slipped through, was declared twice, and the driver refused the whole
    shader -- the first deferred frame on real hardware died on exactly that,
    while Halcyon's own front-end shrugged at the redeclaration and every
    headless check passed.

    The multi-declaration case is load-bearing too. A joining bug once
    spliced two blocks into `uniform sampler2D hal_bump0;uniform sampler2D
    hal_shadow0;` on ONE line; the single-declaration pattern could not
    match it, both names reached CreateInfo twice, and every bump-node
    frame silently shaded on the CPU while the headless seams read zero.
    The assembler now joins blocks line-safely, and this function strips a
    line made of ANY number of whole declarations -- two independent walls.
    """
    import re
    decl = r'(?:uniform|in|out)\s+\w[^;]*?;'
    multi = re.compile(rf'^(?:{decl}\s*)+$')
    kept = []
    for line in fragment.splitlines():
        stripped = line.split('//', 1)[0].strip()
        if multi.match(stripped):
            continue
        kept.append(line)
    return '\n'.join(kept)


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
        self.width = int(width)
        self.height = int(height)

        def _make():
            import gpu
            return gpu.types.GPUOffScreen(self.width, self.height,
                                          format=fmt)

        self.offscreen = _main('an offscreen target', _make)

    def free(self):
        def _free():
            try:
                self.offscreen.free()
            except Exception:                                   # noqa: BLE001
                pass

        try:
            _main('freeing a target', _free)
        except Exception:                                       # noqa: BLE001
            pass


def target_texture(target):
    """A target's colour texture handle, fetched where the context lives."""
    return _main('reading a target handle',
                 lambda: target.offscreen.texture_color)


def upload_cached(key, build):
    """A GPUTexture cached across frames, keyed on the data's fingerprint.

    `build` is called only on a miss and must return the (H,W,3|4) float32
    array to upload -- lazily, so a cache hit skips the *packing* as well as
    the upload. The win this exists for: a point light's six-face shadow
    atlas is tens of megabytes, the maps are already cached across frames on
    the CPU side, and re-packing plus re-uploading them every frame was most
    of the deferred pass's warm cost.
    """
    cache = _STATE.setdefault('textures', {})
    hit = cache.get(key)
    if hit is not None:
        # move-to-end: eviction drops the least recently USED, so a
        # working set that fits never loses a member
        cache[key] = cache.pop(key)
        return hit
    tex = upload(build())
    if len(cache) >= 64:
        # drop the oldest half. The first version CLEARED the whole cache
        # past a cap of 24 -- and the moment a frame's working set crossed
        # the cap, every frame re-packed and re-uploaded everything it had
        # just evicted, shadow atlases included. A 60 ms section became
        # 355 ms on real hardware over ONE texture of growth.
        for k in list(cache)[:32]:
            del cache[k]
    cache[key] = tex
    return tex


def upload(image):
    """(H,W,4) float32 -> a GPUTexture, bottom row first as everywhere else."""
    return _main('a texture upload', lambda: _upload_impl(image))


def _upload_impl(image):
    import gpu
    import numpy as np
    arr = np.ascontiguousarray(np.asarray(image, np.float32))
    h, w = arr.shape[:2]
    if arr.shape[2] == 3:
        arr = np.concatenate([arr, np.ones((h, w, 1), np.float32)], axis=2)
    flat = np.ascontiguousarray(arr.ravel())
    try:
        # Blender's Buffer takes anything with the buffer protocol, and a
        # NumPy float32 array is one -- handing it over directly is a copy.
        # The old .tolist() built h*w*4 Python float objects first, which for
        # a G-buffer upload was a measurable slice of every deferred frame.
        buf = gpu.types.Buffer('FLOAT', flat.shape[0], flat)
    except (TypeError, ValueError):
        buf = gpu.types.Buffer('FLOAT', flat.shape[0], flat.tolist())
    # 32-bit float throughout: RGBA16F carries about eleven bits of mantissa,
    # and a gamma curve turns that into visible error near black
    return gpu.types.GPUTexture((w, h), format='RGBA32F', data=buf)


def draw_fullscreen(shader, uniforms, samplers, target, read=True,
                    blend='NONE', clear=False, region=None):
    """Run one full-screen pass. Returns (H,W,4) float32, or None if !read.

    `blend='ALPHA_PREMULT'` with `clear` on the first pass lets several
    material passes composite into one target on the GPU -- each pass writes
    colour only where its alpha is one, and untouched pixels keep what an
    earlier pass wrote -- so a frame costs one readback however many
    materials are in it, instead of one per material plus a NumPy merge.

    `region=(x, y, w, h)` scissors the DRAW to that rectangle: fragments
    outside it are never rasterised, which is what makes a sparse depth
    layer cost its own bounding box instead of the whole frame. The clear
    stays FULL-frame, deliberately -- every texel of the target is defined,
    so a neighbour fetch that lands outside the scissor reads a clean zero
    (and is masked off by the ids test anyway), never uninitialised memory.
    """
    return _main('a full-screen pass', lambda: _draw_fullscreen_impl(
        shader, uniforms, samplers, target, read, blend, clear, region))


def _draw_fullscreen_impl(shader, uniforms, samplers, target, read,
                          blend, clear, region=None):
    import gpu
    import numpy as np
    from gpu_extras.batch import batch_for_shader

    # TRI_STRIP, not TRI_FAN: the fan is deprecated and leaves in Blender
    # 6.0. The strip cuts the quad along the other diagonal, which changes
    # nothing here -- a fullscreen quad's varyings are affine, so the
    # interpolation is identical whichever way the quad is split
    batch = batch_for_shader(
        shader, 'TRI_STRIP',
        {'pos': ((-1, -1), (1, -1), (-1, 1), (1, 1)),
         'uv': ((0, 0), (1, 0), (0, 1), (1, 1))})

    with target.offscreen.bind():
        if clear:
            # clear with the scissor OFF: full-frame, every texel defined
            gpu.state.scissor_test_set(False)
            fb = gpu.state.active_framebuffer_get()
            fb.clear(color=(0.0, 0.0, 0.0, 0.0))
        gpu.state.blend_set(blend)
        gpu.state.depth_test_set('NONE')
        if region is not None:
            gpu.state.scissor_set(int(region[0]), int(region[1]),
                                  int(region[2]), int(region[3]))
            gpu.state.scissor_test_set(True)
        try:
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
        finally:
            gpu.state.blend_set('NONE')
            if region is not None:
                gpu.state.scissor_test_set(False)
        if not read:
            return None
        buf = target.offscreen.texture_color.read()
        buf.dimensions = target.width * target.height * 4
    return np.asarray(buf, np.float32).reshape(target.height, target.width, 4)


def read_target(target, region=None):
    """The target's colour buffer as (H,W,4) float32.

    With `region=(x, y, w, h)` only that rectangle crosses the bus, as a
    (h, w, 4) array -- a sparse depth layer's readback then costs its
    bounding box, not the frame. The region path reads through the bound
    framebuffer (`read_color`), the only API that takes a rectangle; the
    full-frame path keeps the proven texture read."""
    return _main('a framebuffer read',
                 lambda: _read_target_impl(target, region))


def _read_target_impl(target, region=None):
    import numpy as np
    if region is not None:
        import gpu
        x, y, w, h = (int(v) for v in region)
        with target.offscreen.bind():
            fb = gpu.state.active_framebuffer_get()
            buf = fb.read_color(x, y, w, h, 4, 0, 'FLOAT')
            buf.dimensions = w * h * 4
        return np.asarray(buf, np.float32).reshape(h, w, 4)
    buf = target.offscreen.texture_color.read()
    buf.dimensions = target.width * target.height * 4
    return np.asarray(buf, np.float32).reshape(target.height, target.width, 4)


def compile_compute(name, source, samplers=(), floats=(), images=()):
    """Compile a compute shader through CreateInfo, cached on the source.

    The API shape is the one the capability probe proved on hardware:
    local_group_size + image(WRITE) + compute_source + create_from_info.
    Inputs are samplers and float push constants, outputs are RGBA32F
    images -- the gpu module offers no storage buffers, and the G-buffer
    is texture-shaped anyway.
    """
    key = (name, hash(source))
    hit = _STATE['shaders'].get(key)
    if hit is not None:
        return hit, None
    return _main(f'compiling {name}', lambda: _compile_compute_miss(
        key, name, source, samplers, floats, images))


def _compile_compute_miss(key, name, source, samplers, floats, images):
    ok, why = probe()
    if not ok:
        return None, why
    import gpu
    try:
        info = gpu.types.GPUShaderCreateInfo()
        info.local_group_size(8, 8)
        for i, iname in enumerate(images):
            info.image(i, 'RGBA32F', 'FLOAT_2D', iname,
                       qualifiers={'WRITE'})
        for i, sname in enumerate(samplers):
            info.sampler(i, 'FLOAT_2D', sname)
        for pc in floats:
            info.push_constant('FLOAT', pc)
        info.compute_source(strip_declarations(source))
        shader = gpu.shader.create_from_info(info)
        _STATE['shaders'][key] = shader
        return shader, None
    except Exception as exc:                                    # noqa: BLE001
        return None, f'{name}: compute CreateInfo failed: {exc}'


def dispatch_compute(shader, width, height, uniforms=None, samplers=None,
                     images=None):
    """Bind, dispatch over a width x height grid, read the images back.

    Returns {image name: (H, W, 4) float32}. The group size is the 8x8 the
    shader was compiled with; the kernel guards the ragged edge itself.
    """
    return _main('a compute dispatch', lambda: _dispatch_compute_impl(
        shader, width, height, uniforms, samplers, images))


def _dispatch_compute_impl(shader, width, height, uniforms, samplers,
                           images):
    import gpu
    made = {}
    shader.bind()
    for k, tex in (samplers or {}).items():
        shader.uniform_sampler(k, tex)
    for k, v in (uniforms or {}).items():
        shader.uniform_float(k, float(v))
    for k in (images or ()):
        tex = gpu.types.GPUTexture((int(width), int(height)),
                                   format='RGBA32F')
        shader.image(k, tex)
        made[k] = tex
    gx = (int(width) + 7) // 8
    gy = (int(height) + 7) // 8
    gpu.compute.dispatch(shader, gx, gy, 1)
    out = {}
    import numpy as np
    for k, tex in made.items():
        buf = tex.read()
        # the buffer protocol, exactly as read_target: to_list() on a frame
        # of floats is the 316 ms mistake this project already made once
        try:
            buf.dimensions = int(width) * int(height) * 4
            arr = np.asarray(buf, np.float32)
        except Exception:                                       # noqa: BLE001
            arr = np.array(buf.to_list(), dtype=np.float32)
        out[k] = arr.reshape(int(height), int(width), 4)
    return out
