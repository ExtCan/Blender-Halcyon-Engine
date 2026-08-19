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
    # shaders, batches and cached textures go to the graveyard, not
    # straight to the driver -- a reset can arrive while queued commands
    # still reference them
    for sh in _STATE['shaders'].values():
        bury(sh)
    _STATE['shaders'].clear()
    for entry in _STATE.get('batches', {}).values():
        bury(entry[1])
    _STATE.get('batches', {}).clear()
    for tex in _STATE.get('textures', {}).values():
        bury(tex)
    _STATE.get('textures', {}).clear()
    for pool in _STATE.get('target_pool', {}).values():
        for off in pool:
            bury(off)
    _STATE.get('target_pool', {}).clear()
    # screen-parked objects: their redraw clock stops with the session,
    # so they retire through the wall-clock graveyard instead
    for _born, obj in _SCREEN_GRAVE:
        bury(obj)
    del _SCREEN_GRAVE[:]
    _STATE['draw_tick'] = 0


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


#: driver shader compilations this session: count and wall milliseconds.
#: The field's 33.9s cold frame hid ~29s of compilation inside buckets
#: labelled 'composite' and 'draw+read'; these counters let every split
#: line say 'compile Nx MMMM ms' instead. Sample with compile_stats(),
#: never reset mid-frame.
COMPILE_STATS = {'n': 0, 'ms': 0.0}


def compile_stats():
    return int(COMPILE_STATS['n']), float(COMPILE_STATS['ms'])


def _count_compile(t0):
    import time as _t
    COMPILE_STATS['n'] += 1
    COMPILE_STATS['ms'] += (_t.perf_counter() - t0) * 1000.0


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

    import time as _t
    t0 = _t.perf_counter()
    out = _main(f'compiling {name}', _miss)
    _count_compile(t0)
    return out


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
    import time as _t
    t0 = _t.perf_counter()
    out = _main(f'compiling {name}', lambda: _compile_dynamic_miss(
        key, name, fragment, spec))
    _count_compile(t0)
    return out


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
    """An offscreen colour buffer to render a full-screen pass into.

    POOLED for the whole session: offscreen DESTRUCTION is the one
    operation with no provably safe moment while a render graph can
    hold queued commands, and both field crash dumps died walking
    write barriers at a read flush -- the barrier class that belongs
    to render targets. A session touches only a handful of distinct
    sizes (draft, refine, F12, the post chain at frame size), so the
    pool is small and bounded; free() RETURNS the offscreen for reuse
    instead of destroying it, and only reset() sends the pool to the
    graveyard at teardown. Reuse is safe by construction: the offscreen
    stays alive, every stage fully overwrites it, and command order is
    preserved on the one queue.
    """

    _POOL_PER_SIZE = 8

    def __init__(self, width, height, fmt='RGBA32F'):
        self.width = int(width)
        self.height = int(height)
        self.fmt = fmt
        key = (self.width, self.height, fmt)
        pool = _STATE.setdefault('target_pool', {}).setdefault(key, [])
        if pool:
            self.offscreen = pool.pop()
            return

        def _make():
            import gpu
            return gpu.types.GPUOffScreen(self.width, self.height,
                                          format=fmt)

        self.offscreen = _main('an offscreen target', _make)

    def free(self):
        if self.offscreen is None:
            return
        key = (self.width, self.height, self.fmt)
        pool = _STATE.setdefault('target_pool', {}).setdefault(key, [])
        if len(pool) < self._POOL_PER_SIZE:
            pool.append(self.offscreen)
        else:
            # a pathological burst beyond the pool bound still never
            # destroys directly -- the graveyard's flush-epoch proof
            # decides when
            bury(self.offscreen)
        self.offscreen = None


def target_texture(target):
    """A target's colour texture handle, fetched where the context lives."""
    return _main('reading a target handle',
                 lambda: target.offscreen.texture_color)


# --------------------------------------------------- deferred GPU reclamation
#
# Blender 5.x's Vulkan backend RECORDS draw commands into a render graph
# and executes them later on a submission thread. Dropping the last Python
# reference to a GPUTexture (or freeing a GPUOffScreen) the moment it is
# replaced destroys the VkImage while queued commands may still hold
# barriers against it -- the field crash: EXCEPTION_ACCESS_VIOLATION inside
# nvoglv64.dll under vkCmdPipelineBarrier, on VKDevice::submission_runner,
# the instant a heavy scene entered rendered view. On OpenGL the same churn
# was harmless (GL defers deletion internally), which is why months of
# field use never saw it. So NOTHING here frees a live-ish GPU object
# directly any more: replaced and evicted objects are PARKED and released
# only after their in-flight window has safely passed, on the main thread.

_GRAVEYARD = []                # [(flush epoch at burial, deadline, object)]
_AFTERLIFE = 10.0              # wall-clock FALLBACK, for sessions where no
                               # readback ever ticks the flush epoch
_GRAVE_EMERGENCY = 512         # pathological-growth backstop


def bury(obj):
    """Park a GPU object instead of letting it die mid-flight."""
    if obj is None:
        return
    import time
    _GRAVEYARD.append((_STATE.get('flush_epoch', 0),
                       time.monotonic() + _AFTERLIFE, obj))


def flush_tick():
    """A GPU readback completed: everything recorded before it has
    EXECUTED, so every object buried before this moment is retirable.

    This is the ruler the graveyard measures with. The first version
    used TIME (one second), on the theory that queued commands flush
    within a swap. On Vulkan they do not: commands recorded early in a
    frame sit in the render graph until the frame's final readback
    flushes them all at once -- and the field's first draft frame,
    thick with shader compiles, ran LONGER than a second. A mid-frame
    collect() then freed post-chain targets buried earlier in the SAME
    frame, and the end-of-frame flush walked its barriers into
    destroyed images (the clean-install 5.1.1 crash: submission thread
    reading 0x20 in add_image_write_barriers while the main thread sat
    in OUR texture read). A returned read is the only PROOF of
    retirement; wall time is only the fallback for read-free sessions,
    where the swap chain drains the queue many times over.
    """
    _STATE['flush_epoch'] = _STATE.get('flush_epoch', 0) + 1
    if _STATE['flush_epoch'] <= 3:
        # the first few completions go to the crash-forensics log: a
        # fatal read WITHOUT any completion before it means the very
        # first offscreen readback is what dies; completions before it
        # mean the lifetime between flushes is the question
        try:
            from .. import fault_note
            fault_note('GPU readback completed '
                       f"(epoch {_STATE['flush_epoch']})", key='flush',
                       limit=3)
        except Exception:                                       # noqa: BLE001
            pass


def collect(force=False):
    """Release parked objects PROVEN retired (buried before the last
    completed readback), plus any past the wall-clock fallback.

    Runs where the GPU context lives (upload/draw moments and the
    viewport's persistent timer are the callers). Entries of the
    CURRENT epoch are never force-released by the emergency cap --
    they may still be referenced by the unflushed graph."""
    if not _GRAVEYARD:
        return
    import time
    now = time.monotonic()
    epoch = _STATE.get('flush_epoch', 0)
    keep = []
    for e, deadline, obj in _GRAVEYARD:
        if force or e < epoch or now >= deadline:
            fn = getattr(obj, 'free', None)    # GPUOffScreen frees; a
            if fn is not None:                 # GPUTexture just drops
                try:
                    fn()
                except Exception:                               # noqa: BLE001
                    pass
            continue
        keep.append((e, deadline, obj))
    if len(keep) > _GRAVE_EMERGENCY:
        del keep[:len(keep) - _GRAVE_EMERGENCY]
    _GRAVEYARD[:] = keep


#: objects whose LAST use was a SCREEN draw, waiting out the swapchain:
#: [(draw tick at burial, obj)]
_SCREEN_GRAVE = []
#: full redraws an object waits before release -- the swapchain runs 2-3
#: frames deep, so 8 completed view_draw calls put its last draw far
#: behind any frame still in flight
_SCREEN_AFTERLIFE = 8


def bury_screen(obj):
    """Park an object whose last use was drawn TO THE SCREEN (a parked-
    frame blit texture, a blit batch).

    Readback epochs cannot retire these: a completed readback proves OUR
    offscreen subgraph executed, but Vulkan's render graph prunes to the
    read's dependencies -- Blender's screen-draw stream is a different
    subgraph, and the field's GPU-device session died the moment epoch 1
    'proved' the CPU-phase blit textures retirable while their draws
    could still sit unflushed (two native access violations, no Python
    frames, right after 'GPU readback completed (epoch 1)'). What DOES
    retire a screen object is redraws: draw_tick() advances a clock at
    the end of every completed view_draw, and an object parked here
    frees only after _SCREEN_AFTERLIFE full redraws."""
    if obj is None:
        return
    _SCREEN_GRAVE.append((_STATE.get('draw_tick', 0), obj))


def draw_tick():
    """A view_draw completed: advance the screen clock, release what has
    waited out a full swapchain depth of redraws. Main thread only."""
    t = _STATE.get('draw_tick', 0) + 1
    _STATE['draw_tick'] = t
    if not _SCREEN_GRAVE:
        return
    keep = []
    for born, obj in _SCREEN_GRAVE:
        if t - born >= _SCREEN_AFTERLIFE:
            fn = getattr(obj, 'free', None)
            if fn is not None:
                try:
                    fn()
                except Exception:                               # noqa: BLE001
                    pass
            continue
        keep.append((born, obj))
    _SCREEN_GRAVE[:] = keep


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
    if len(cache) >= 96:
        # drop the oldest third INTO THE GRAVEYARD -- an evicted texture
        # can still be referenced by passes recorded THIS frame (the
        # real file's frame wants ~40 textures at once: 14 shadow
        # atlases, a dozen images, the G-buffer set), so eviction must
        # never destroy immediately. The first version CLEARED the whole
        # cache past a cap of 24 and re-uploaded everything every frame;
        # the second freed mid-frame and handed the Vulkan render graph
        # dead images. The cap is 96 so one heavy frame's working set
        # never triggers eviction at all.
        for k in list(cache)[:32]:
            bury(cache.pop(k))
    cache[key] = tex
    return tex


def upload(image):
    """(H,W,4) float32 -> a GPUTexture, bottom row first as everywhere else."""
    return _main('a texture upload', lambda: _upload_impl(image))


def _upload_impl(image):
    import gpu
    import numpy as np
    # NO collect() here. Uploads happen while a frame's commands are
    # being recorded (plan build runs one per image texture -- the real
    # file runs eleven), and freeing parked objects mid-recording is
    # the exact race the graveyard exists to prevent. The field pinned
    # it: GPU sessions died only on scenes with image textures. Parked
    # objects release on the persistent timer and the draw tick instead.
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
    tex = gpu.types.GPUTexture((w, h), format='RGBA32F', data=buf)
    # the SOURCE memory outlives the transfer: Vulkan RECORDS -- if the
    # backend consumes the staging data lazily at flush, `flat` and
    # `buf` dying at function exit hands the deferred upload freed
    # memory to read (garbage texels at best) and lets the allocator
    # reuse pages a transfer may still touch. They park until a
    # readback PROVES the graph that carried the upload executed
    bury((flat, buf))
    return tex


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


def draw_many(draws, read=None, read_region=None):
    """A SEQUENCE of full-screen passes in ONE main-thread crossing.

    `draws` is a list of (shader, uniforms, samplers, target, blend,
    clear, region) tuples -- draw_fullscreen's own arguments -- run
    back-to-back, optionally ending with a readback of `read` (a
    Target, with read_region as read_target takes it).

    R175: the field's 25-pass warm F12 spent ~370 ms in 'draw+read',
    ~14 ms per pass -- and each pass was its OWN marshal crossing: the
    render worker queued one draw, slept until the main loop's next
    pickup, woke, queued the next. Twenty-six round-trips of queue
    latency, none of it driver work. One crossing submits the same
    commands in the same order -- the driver sees an identical stream,
    so the pixels cannot move by a bit; only the sleeping stops. The
    main thread is not held longer than before: the draws are
    non-blocking submissions, and the readback (the only wait) already
    ran whole in a single crossing.
    """
    def _run():
        import time as _t
        t0 = _t.perf_counter()
        # R175b: consecutive draws into the SAME target share ONE
        # offscreen bind. The field measured the per-pass cost
        # unchanged after the crossings fused (~14 ms each inside one
        # crossing): the suspect is the bind/unbind bracket around
        # every draw -- on the Vulkan backend a framebuffer switch can
        # end the render pass and flush. Hoisting the bind issues the
        # same draws, same order, same per-draw state (blend, scissor,
        # clear) inside one activation; pixels cannot move.
        i = 0
        while i < len(draws):
            tgt = draws[i][3]
            j = i
            while j < len(draws) and draws[j][3] is tgt:
                j += 1
            with tgt.offscreen.bind():
                for (shader, uniforms, samplers, _t2, blend, clear,
                     region) in draws[i:j]:
                    _draw_in_bound(shader, uniforms, samplers, blend,
                                   clear, region)
            i = j
        t1 = _t.perf_counter()
        out = None
        if read is not None:
            out = _read_target_impl(read, read_region)
        t2 = _t.perf_counter()
        LAST_BURST.update(n=len(draws), draw_ms=(t1 - t0) * 1000.0,
                          read_ms=(t2 - t1) * 1000.0)
        return out
    return _main(f'a {len(draws)}-pass burst', _run)


#: R175b: where the last burst's milliseconds went, measured INSIDE the
#: crossing -- 'draws' is submission+state on the main thread, 'read'
#: is the readback wait (the GPU actually executing). One printed pair
#: tells whether a slow frame is submission overhead or real GPU work.
LAST_BURST = {'n': 0, 'draw_ms': 0.0, 'read_ms': 0.0}


def _fullscreen_batch(shader):
    """The 4-vertex quad batch for a shader, cached for the shader's life.

    A batch built per pass DIED at function exit -- and a no-read
    composite pass leaves its draw RECORDED in the Vulkan render graph,
    so the batch's vertex buffer was destroyed before the draw ever
    executed. A 16-material frame queued 16 draws over dead geometry and
    the barrier walk crashed inside the driver the moment the final
    readback flushed (the field's second crashlog: nvoglv64 access
    violation on the submission thread while OUR texture.read() sat on
    the main thread). The cache VALUE keeps a reference to the shader
    too, so the id() key can never be recycled while its entry lives;
    reset() buries entries instead of freeing them.
    """
    from gpu_extras.batch import batch_for_shader
    cache = _STATE.setdefault('batches', {})
    hit = cache.get(id(shader))
    if hit is not None:
        return hit[1]
    # TRI_STRIP, not TRI_FAN: the fan is deprecated and leaves in Blender
    # 6.0. The strip cuts the quad along the other diagonal, which changes
    # nothing here -- a fullscreen quad's varyings are affine, so the
    # interpolation is identical whichever way the quad is split
    batch = batch_for_shader(
        shader, 'TRI_STRIP',
        {'pos': ((-1, -1), (1, -1), (-1, 1), (1, 1)),
         'uv': ((0, 0), (1, 0), (0, 1), (1, 1))})
    cache[id(shader)] = (shader, batch)
    return batch


def _draw_in_bound(shader, uniforms, samplers, blend, clear, region=None):
    """One full-screen draw against the ALREADY BOUND framebuffer.

    Exactly the state bracket _draw_fullscreen_impl always ran between
    its bind and its read -- split out so draw_many can issue a run of
    same-target draws inside a single offscreen activation."""
    import gpu

    batch = _fullscreen_batch(shader)
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


def _draw_fullscreen_impl(shader, uniforms, samplers, target, read,
                          blend, clear, region=None):
    import numpy as np

    with target.offscreen.bind():
        _draw_in_bound(shader, uniforms, samplers, blend, clear, region)
        if not read:
            return None
        buf = target.offscreen.texture_color.read()
        buf.dimensions = target.width * target.height * 4
    flush_tick()                # the read returned: the graph executed
    # .copy(): np.asarray over the driver's Buffer is a VIEW of foreign
    # memory. These pixels travel to worker threads (the shading loop
    # fetches attributes from them), and a 1.35.10 field log caught a
    # worker taking an access violation inside exactly such a fetch.
    # The copy puts them on the Python heap, owned, thread-safe
    out = np.asarray(buf, np.float32).reshape(
        target.height, target.width, 4).copy()
    bury(buf)   # the Buffer's pages outlive any transfer that may still land
    return out


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
        flush_tick()            # the read returned: the graph executed
        # owned copy, not a view of the driver's buffer (see
        # _draw_fullscreen_impl); the Buffer itself parks so its pages
        # cannot be reused while a transfer could still touch them
        out = np.asarray(buf, np.float32).reshape(h, w, 4).copy()
        bury(buf)
        return out
    buf = target.offscreen.texture_color.read()
    buf.dimensions = target.width * target.height * 4
    flush_tick()                # the read returned: the graph executed
    out = np.asarray(buf, np.float32).reshape(
        target.height, target.width, 4).copy()
    bury(buf)
    return out


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
    import time as _t
    t0 = _t.perf_counter()
    out = _main(f'compiling {name}', lambda: _compile_compute_miss(
        key, name, source, samplers, floats, images))
    _count_compile(t0)
    return out


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
    import time as _time
    gx = (int(width) + 7) // 8
    gy = (int(height) + 7) // 8
    t0 = _time.perf_counter()
    gpu.compute.dispatch(shader, gx, gy, 1)
    t_dispatch = _time.perf_counter() - t0
    out = {}
    import numpy as np
    t0 = _time.perf_counter()
    for k, tex in made.items():
        buf = tex.read()
        # the buffer protocol, exactly as read_target: to_list() on a frame
        # of floats is the 316 ms mistake this project already made once
        try:
            buf.dimensions = int(width) * int(height) * 4
            arr = np.asarray(buf, np.float32)
        except Exception:                                       # noqa: BLE001
            arr = np.array(buf.to_list(), dtype=np.float32)
        # ONE owned copy, now: np.asarray over the driver's buffer is a
        # VIEW of foreign memory -- and ascontiguousarray of an already-
        # contiguous view returns the SAME view, so the old line here
        # never actually copied. .copy() does: the decoder's strided
        # slices then read the Python heap, and the worker threads that
        # fetch attributes from these arrays never touch driver memory
        # (a 1.35.10 field log caught a worker AV inside such a fetch)
        out[k] = arr.reshape(int(height), int(width), 4).copy()
        bury(buf)   # its pages outlive any transfer that may still land
    flush_tick()                # reads returned: the graph executed
    LAST_DISPATCH['dispatch_ms'] = t_dispatch * 1000.0
    LAST_DISPATCH['read_ms'] = (_time.perf_counter() - t0) * 1000.0
    return out


#: the last compute dispatch's device-side split: how long the kernel ran
#: vs how long reading the images back took. The raster split prints it so
#: "dispatch+read 920 ms" stops being one unattackable number
LAST_DISPATCH = {}
