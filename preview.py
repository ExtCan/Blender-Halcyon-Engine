"""The viewport preview's working half, free of bpy so it can be tested.

The engine's `view_update` exports the scene (main thread, where bpy access
is legal) and hands it here; `view_draw` reports what view is wanted and
blits whatever frame is newest. Everything BETWEEN those two -- the worker
thread, the render, the coalescing of a fast orbit down to the newest view,
the abort of a stale in-flight frame -- lives in this module and runs
against the bpy-free exported scene, which is exactly why the test suite
can drive it without Blender.

Rules the class keeps:

* One worker at a time, and IT FINISHES. The first field build aborted the
  in-flight render every time the camera moved -- and since the abort could
  only land between render stages, each render burned most of its work
  before dying, so during an orbit NOTHING ever completed and the preview
  only updated when the camera stopped. Now a moving camera renders DRAFTS
  (coarser by DRAFT_FACTOR) that run to completion and stream in, and the
  full-quality frame renders once the view rests. The only renders an
  abort may kill are a refine overtaken by new motion (its draft is
  already on screen) and anything belonging to a re-exported scene.
* No bpy off the main thread, ever. The worker touches only the exported
  scene, the settings copy, and NumPy. The one bpy import in this file is
  inside a guarded redraw request, and the guard is the point.
* No silent blanks. Every failure prints one `[Halcyon viewport]` line
  with the traceback, once per distinct reason -- a viewport that renders
  nothing must at least say why.
* The device switch means the viewport too. A worker thread has no GPU
  context -- the reason this path was pinned to the CPU for its whole
  life -- but that is the exact problem the F12 marshal solved: the
  worker renders, and each driver burst crosses to the main thread,
  which always has a context. The worker enables marshalling around its
  own render (the marshal refcounts, so an overlapping F12 cannot switch
  it off underneath), and TRY-holds the one-render-at-a-time pipeline
  lock: when an F12 or the self-test holds the driver, that one viewport
  frame renders on the CPU and says so once, because a draft nobody can
  see the difference on must never stall a final frame.
"""

import threading
import time

import numpy as np

from .core import post
from .core import render as core_render
from .gpu import marshal

#: the view is "in motion" for this long after its last change; drafts
#: render while it holds, the refine starts once it lapses
DRAFT_WINDOW = 0.35

#: drafts render at preview_scale * this -- a quarter of the pixels
DRAFT_FACTOR = 2

#: the most pixels a DRAFT may cost, whatever the pixel size: motion
#: frames exist to be fluid, and the divisor grows until they fit.
#: ~64k pixels is the 352x144-class frame the field machine renders at
#: a few frames a second through the full pipeline
DRAFT_BUDGET_PX = 64_000

#: how far past the window's lapse the armed recheck fires -- enough that
#: the clock has definitely crossed it, small enough to feel instant
RECHECK_SLACK = 0.05

# ---------------------------------------------------------------- redraws
#
# Requesting a redraw from a WORKER THREAD must touch NOTHING of bpy.
# 1.25.88 registered a bpy timer per completed frame from the worker;
# 1.25.89 poked one per GPU BURST -- a redraw storm the field described
# as "flashes violently", with Blender's context-state assert still
# flooding. All of it is now a pure-threading FLAG, and ONE persistent
# main-thread timer (registered from view_draw, which IS the main
# thread) polls it. Zero cross-thread bpy, zero timer churn.

_REDRAW = {'want': False, 'engine': None, 'timer': False,
           'flags': 0, 'draws': 0}


def request_redraw_flag(engine=None):
    """Safe from any thread: set the flag; the main-thread poll acts."""
    if engine is not None:
        _REDRAW['engine'] = engine
    _REDRAW['want'] = True
    _REDRAW['flags'] += 1


def ensure_redraw_timer():
    """Start the ONE polling timer. Call from the MAIN thread only
    (view_draw / view_update). Headless: quietly does nothing."""
    if _REDRAW['timer']:
        return
    try:
        import bpy as _bpy

        def _poll():
            import sys
            if getattr(sys, 'is_finalizing', lambda: False)():
                return None             # interpreter teardown: unregister
            if _REDRAW['want']:
                _REDRAW['want'] = False
                eng = _REDRAW.get('engine')
                if eng is not None:
                    try:
                        eng.tag_redraw()
                    except Exception:                           # noqa: BLE001
                        _REDRAW['engine'] = None
            # the ONE guaranteed main-thread heartbeat: parked GPU
            # objects (replaced blit textures, evicted uploads) release
            # here once their in-flight window has passed
            try:
                from .gpu import device as _dev
                _dev.collect()
            except BaseException:                               # noqa: BLE001
                pass
            return 0.03

        _bpy.app.timers.register(_poll, first_interval=0.03,
                                 persistent=True)
        _REDRAW['timer'] = True
    except Exception:                                           # noqa: BLE001
        _REDRAW['timer'] = False


def _black_measure(img, tile=24):
    """(black fraction, per-tile black map) of a frame -- the guard's eyes.

    The whole-frame fraction catches full blackouts; the TILE map catches
    a single material's region going dark, which can sit under any
    whole-frame threshold. Tiles smaller than one grid cell fall back to
    the fraction alone."""
    a = np.asarray(img, np.float32)
    bk = a[..., :3].max(axis=2) <= (1.0 / 255.0)
    frac = float(bk.mean())
    h, w = bk.shape
    th, tw = h // tile, w // tile
    if th < 1 or tw < 1:
        return frac, None
    tiles = bk[:th * tile, :tw * tile].reshape(
        th, tile, tw, tile).mean(axis=(1, 3))
    return frac, tiles.astype(np.float32)


class Cancelled(Exception):
    """Raised inside the worker when a newer view supersedes its render."""


def shape_settings(settings, w, h):
    """Turn a scene's render settings into viewport settings, in place.

    The look stays the F12 look -- dither, CRT, the whole post chain are
    the point of this engine -- and since 1.25.83 the DEVICE stays the
    F12 device too: the top-of-panel switch governs the viewport exactly
    as it governs a final render, with every per-frame refusal falling
    back by name through the same plan. What this strips is everything
    about HOW a final frame runs that makes no sense per redraw: worker
    processes (a pool per draft), anti-aliasing (drafts are quarter-res
    already), the stats firehose (one line per refine survives, behind
    the same Timing Breakdown toggle), and Blender-side sizing.
    """
    scale = max(int(getattr(settings, 'preview_scale', 1)), 1)
    settings.resolution_x = max(int(w) // scale, 4)
    settings.resolution_y = max(int(h) // scale, 4)
    if not getattr(settings, 'viewport_gpu', True):
        # the Debug bisect switch: viewport frames CPU, F12 untouched
        settings.render_device = 'CPU'
    settings.aa_mode = 'NONE'
    settings.aa_samples = 1
    settings.output_scale = 'NONE'
    settings.pixel_aspect_x = settings.pixel_aspect_y = 1.0
    settings.use_processes = False
    settings.progressive = False
    settings.motion_blur = False
    # stats: the full breakdown is per-frame console spam at draft rate;
    # remember the wish and print one line per completed refine instead
    settings._viewport_stats = bool(getattr(settings, 'show_stats', False))
    settings.show_stats = False
    settings._viewport = True         # quiets the per-frame depth report
    return settings


class Viewport:
    """Per-engine viewport state: one worker, one newest frame, one texture."""

    def __init__(self, clock=None):
        self.lock = threading.Lock()
        self.clock = clock or time.monotonic   # injectable, for the tests
        self.scene = None
        self.settings = None
        self.version = 0          # bumped by every view_update export
        self.wanted = None        # (key, camera, w, h) most recently drawn-for
        self.done_key = None      # (key, version, draft) the parked frame is of
        self.frame = None         # (H,W,4) float32, ready to blit
        self.abort = False
        self.busy = False
        self._inflight_draft = False
        self._last_change = -1e9  # when the wanted view or scene last moved
        self._tex = None
        self._tex_id = None
        self._said = set()
        self.last_engaged = None  # 'GPU'/'CPU': what the last frame used
        self._recheck_armed = False   # one pending window-lapse redraw, max
        self._black_prev = None   # black fraction of the last parked frame
        self._black_tiles_prev = None  # per-tile black map of the same
        self._black_key = None    # the view key those stats belong to
        self.guard_count = 0      # GPU frames the black guard re-shaded
        self._split_said = False  # first refine prints its split once

    def __del__(self):
        # leaving rendered view destroys the engine and this viewport;
        # the blit texture's last draw can still be queued in the Vulkan
        # render graph, so it parks in the screen grave (released after
        # later redraws, or swept to the wall-clock graveyard at reset),
        # never freed by direct GC. During INTERPRETER
        # FINALIZATION (Blender exit) this does nothing at all: imports
        # and even exception machinery are half-dismantled there, and an
        # exception escaping a __del__ at that point is exactly the
        # unraisable-error crash a field exit log showed.
        try:
            import sys
            if getattr(sys, 'is_finalizing', lambda: False)():
                return
            from .gpu import device as _dev
            _dev.bury_screen(self._tex)
            self._tex = None
            try:
                from . import fault_note
                # places 'the user left rendered view' on the timeline,
                # so a later fatal is not pinned on the viewport
                fault_note('viewport engine destroyed (left rendered '
                           'view)', key='vp-bye', limit=3)
            except Exception:                                   # noqa: BLE001
                pass
        except BaseException:                                   # noqa: BLE001
            pass

    # ------------------------------------------------------------- reporting
    def complain(self, what, detail=''):
        """Console-loud, once per distinct reason: a blank viewport must
        never be silent again."""
        if what in self._said:
            return
        self._said.add(what)
        print(f'[Halcyon viewport] {what}')
        if detail:
            print(detail)

    # ------------------------------------------------------------ main thread
    def set_scene(self, scene, settings):
        # material EXCLUSIVE light groups bind at scene arrival, so the
        # viewport's light loops (CPU draft and GPU frame alike) read
        # the same set F12 collects at render entry
        try:
            from .core.render import collect_exclusive_lights
            collect_exclusive_lights(scene)
        except Exception:                                       # noqa: BLE001
            pass
        with self.lock:
            self.scene = scene
            self.settings = settings
            self.version += 1
            # a re-export outdates anything mid-flight -- but a DRAFT
            # still runs to completion. Aborting drafts here is how
            # animation playback showed NOTHING: every frame's export
            # killed the previous frame's draft before it could park
            # (the R25 orbit bug, scene-version edition -- the field
            # named it: "running an animation in the viewport, it does
            # not update in realtime"). A one-export-stale draft parks
            # and streams; the version guard in kick() re-drafts the
            # newest export next, and refines it the moment the exports
            # stop. Only a REFINE dies here: a full-quality frame of an
            # outdated scene is expensive work nobody will look at.
            if self.busy and not self._inflight_draft:
                self.abort = True
            self._last_change = self.clock()   # an edit storm drafts too

    def has_scene(self):
        with self.lock:
            return self.scene is not None

    @staticmethod
    def _settings_sig(st):
        return tuple(sorted((k, repr(v)) for k, v in vars(st).items()
                            if not k.startswith('_')))

    def set_settings(self, settings):
        """Settings may have changed, the scene did not: keep the parked
        export, and re-render ONLY if the settings really differ.

        The cheap half of set_scene, for view_update calls whose
        depsgraph updates touch nothing renderable (the field runs
        add-ons that poke the depsgraph constantly; every poke was a
        full 100-object re-export followed by a draft+refine cascade).
        An identical-settings poke now costs nothing at all.
        """
        with self.lock:
            if self.settings is not None and \
                    self._settings_sig(settings) == \
                    self._settings_sig(self.settings):
                return
            self.settings = settings
            self.version += 1
            if self.busy and not self._inflight_draft:
                self.abort = True
            self._last_change = self.clock()

    def want(self, camera, w, h):
        """Record the view the draw side asked for.

        Animation-frame and data changes arrive as `set_scene` exports
        (version bumps); this records only what the draw side knows -- the
        camera and the region. A changed view marks the motion clock, and
        aborts ONLY an in-flight refine: its draft is already parked, so
        nothing on screen is lost, and the worker frees up for the next
        draft instead of finishing a full frame of a view nobody is at.
        A draft in flight always runs to completion -- killing those is
        the exact bug that made the preview update only at rest.
        """
        key = self._key(camera, w, h)
        with self.lock:
            changed = self.wanted is not None and self.wanted[0] != key
            self.wanted = (key, camera, w, h)
            if changed:
                self._last_change = self.clock()
                if self.busy and not self._inflight_draft:
                    self.abort = True

    def _arm_recheck(self, engine):
        """One redraw request for the moment the motion window lapses.

        Called (under the lock) when a parked draft declines to refine
        because the view is still inside the window. If the camera moves
        again first, the fired redraw finds a new window and re-arms with
        the new remainder -- it converges to firing once, just after the
        view truly rests. Never stacks: one armed recheck at a time.
        """
        if self._recheck_armed:
            return
        self._recheck_armed = True
        delay = max(DRAFT_WINDOW - (self.clock() - self._last_change),
                    0.0) + RECHECK_SLACK
        if not self._schedule_recheck(engine, delay):
            self._recheck_armed = False    # headless / no timers: no-op

    def _schedule_recheck(self, engine, delay):
        """The bpy half of the recheck, split out so the headless suite can
        capture the decision without a main loop. True if scheduled."""
        try:
            import bpy as _bpy
        except Exception:                                       # noqa: BLE001
            return False

        def _fire(vp=self, eng=engine):
            vp._recheck_armed = False
            try:
                if eng is not None:
                    eng.tag_redraw()   # -> view_draw -> kick: at rest now
            except Exception:                                   # noqa: BLE001
                pass
            return None                # run once

        try:
            _bpy.app.timers.register(_fire, first_interval=float(delay))
        except Exception:                                       # noqa: BLE001
            return False
        return True

    @staticmethod
    def _key(camera, w, h):
        cam = None
        if camera is not None:
            cam = (tuple(np.round(np.asarray(camera.matrix_world,
                                             np.float64).reshape(-1), 5)),
                   tuple(np.round(np.asarray(camera.projection,
                                             np.float64).reshape(-1), 5)),
                   camera.type)
        return (cam, int(w), int(h))

    def kick(self, engine=None):
        """Start a worker for the newest wanted view, if one is due.

        In motion (within DRAFT_WINDOW of the last change) the render is a
        draft; at rest it is full preview quality. A parked draft of the
        current view re-kicks once the view rests, so the picture sharpens
        the moment the orbit ends; a parked full frame of the current view
        kicks nothing, so a resting viewport costs zero.
        """
        with self.lock:
            if self.busy or self.scene is None or self.wanted is None:
                return False
            key, camera, w, h = self.wanted
            moving = (self.clock() - self._last_change) < DRAFT_WINDOW
            done = self.done_key
            if done is not None and done[0] == key and done[1] == self.version:
                if not done[2]:
                    return False     # parked full frame of this very view
                if moving:
                    # parked draft is enough while moving -- but Blender
                    # only redraws on EVENTS, and a GPU draft finishes in
                    # milliseconds, so its completion redraw lands INSIDE
                    # the motion window and this decline is the last word
                    # until the next input. The field named it: "doesn't
                    # always refine on stopping; pressing the middle mouse
                    # button usually fixes it" -- the MMB press was the
                    # missing event. Arm ONE timer for the window's lapse
                    # so the refine invites itself.
                    self._arm_recheck(engine)
                    return False
                # parked draft, view at rest: refine
            draft = moving
            self.busy = True
            self.abort = False
            self._inflight_draft = draft
            scene, settings, version = self.scene, self.settings, self.version
        worker = threading.Thread(
            target=self._render, name='halcyon-viewport',
            args=(engine, scene, settings, camera, w, h, key, version, draft),
            daemon=True)
        worker.start()
        return True

    # ---------------------------------------------------------- worker thread
    def _render(self, engine, scene, settings, camera, w, h, key, version,
                draft=False):
        holding = False
        try:
            # a fresh copy per frame: the stored settings object must not
            # accumulate this frame's resolution or device decisions
            stored = settings
            settings = stored.copy()
            settings._viewport = getattr(stored, '_viewport', True)
            settings._viewport_stats = getattr(stored, '_viewport_stats',
                                               False)
            scale = max(int(getattr(settings, 'preview_scale', 1)), 1)
            if draft:
                orbit = int(getattr(settings, 'orbit_scale', 0) or 0)
                if orbit > 0:
                    # the user chose their motion pixel size outright
                    scale = orbit
                else:
                    scale *= DRAFT_FACTOR
                    # Auto: a draft exists to keep MOTION fluid, so it
                    # gets a pixel BUDGET, not a fixed divisor. At
                    # Pixel Size X1 a half-res draft of a 1409x577
                    # region is 204k pixels -- measured at ~0.5-1 s a
                    # frame on the real file, a 1-2 fps slideshow the
                    # field called 'incredibly laggy'. The divisor
                    # grows until the draft fits the budget; the
                    # REFINE still honours the user's pixel size
                    # exactly, so rest quality is untouched
                    while (max(int(w) // scale, 4)
                           * max(int(h) // scale, 4)) > DRAFT_BUDGET_PX:
                        scale += 1
            settings.resolution_x = max(int(w) // scale, 4)
            settings.resolution_y = max(int(h) // scale, 4)
            if camera is not None:
                scene.camera = camera
            scene.settings = settings

            wants_gpu = str(getattr(settings, 'render_device',
                                    'CPU')).upper() == 'GPU'
            if wants_gpu:
                holding = marshal.PIPELINE.acquire(blocking=False)
                if holding:
                    marshal.enable()
                    # bursts cross to the main thread via the marshal's
                    # TIMER pump, between redraws -- never inside a draw
                    # callback. 1.25.89-.91 ran them inside view_draw and
                    # the whole scene flashed rapidly whenever bursts
                    # streamed (motion AND refines); the field called it
                    # a seizure risk. The pump model never flashed.
                else:
                    # an F12 or the self-test is on the driver: this one
                    # frame renders on the CPU rather than racing it
                    settings.render_device = 'CPU'
                    self.complain(
                        'the driver is busy with another render -- '
                        'viewport frames shade on the CPU until it '
                        'finishes')
            self.last_engaged = ('GPU' if wants_gpu and holding else 'CPU')

            def tick(_frac, _msg):
                if self.abort:
                    raise Cancelled()

            t0 = time.perf_counter()
            try:
                from . import fault_note
                fault_note(
                    f'viewport render started '
                    f'({settings.resolution_x}x{settings.resolution_y} '
                    f'device={getattr(settings, "render_device", "?")} '
                    f'{"draft" if draft else "refine"})', key='vp-start')
            except Exception:                                   # noqa: BLE001
                pass
            marshal.acct_reset()      # per-frame crossing/wait accounting
            img = core_render.render(scene, settings, progress=tick)
            try:
                from . import fault_note
                # splits the crash window: a fatal AFTER this line is in
                # post/guard/park on the worker's own arrays; a fatal
                # between 'readback completed' and this line is inside
                # the render's GPU stage or its return to the worker
                fault_note('render returned to the worker '
                           f'({self.last_engaged})', key='vp-back',
                           limit=3)
            except Exception:                                   # noqa: BLE001
                pass
            img = post.process(img, settings,
                               target_size=(settings.resolution_x,
                                            settings.resolution_y))
            # THE BLACK-FRAME GUARD (a field instrument): the field reports
            # materials randomly turning pure black / flashing, VIEWPORT
            # only, GPU device -- and the headless cadence stress runs
            # clean, so whatever it is lives in live driver state this
            # code cannot reproduce. So the viewport measures instead: a
            # GPU frame whose black fraction JUMPS against the previous
            # parked frame is re-shaded on the CPU (kept, so the flash
            # never reaches the screen) and counted, with one console
            # line naming the event. The line is the next instrument:
            # paste it. A legitimately dark scene converges (the guard
            # compares against what was last PARKED) and costs at most
            # one spurious CPU frame at a hard cut.
            if self.last_engaged == 'GPU':
                blk, tiles = _black_measure(img)
                prev = self._black_prev
                ptiles = self._black_tiles_prev
                flips = 0
                if ptiles is not None and tiles is not None and \
                        ptiles.shape == tiles.shape and \
                        self._black_key == key:
                    # a MATERIAL-sized blackout can sit under any
                    # whole-frame threshold: count TILES that flipped
                    # from mostly-lit to near-black. ONLY against a frame
                    # of the SAME VIEW -- across a moving camera, tiles
                    # flip legitimately as content crosses the frame, and
                    # an ungated tile guard would misfire on every orbit
                    flips = int(((ptiles < 0.5) & (tiles > 0.9)).sum())
                jumped = prev is not None and blk - prev > 0.20
                # THE FIRST-FRAME HOLE, closed: with no previous frame to
                # compare against, a 100%-black first GPU frame parked
                # unchallenged -- and a session that crashes or is closed
                # before frame two shows the user NOTHING, ever. The
                # field lived inside this hole for an entire arc: every
                # session was a first frame. A first GPU frame that is
                # near-TOTALLY black re-shades on the CPU like any other
                # guard hit; a legitimately empty scene re-shades once,
                # comes back just as black, and converges.
                first_black = prev is None and blk >= 0.97
                if jumped or flips >= 2 or first_black:
                    self.guard_count += 1
                    why = (f'{blk * 100.0:.0f}% black on the very first '
                           'GPU frame' if first_black else
                           f'{blk * 100.0:.0f}% black where the previous '
                           f'frame was {(prev or 0.0) * 100.0:.0f}%'
                           if jumped else
                           f'{flips} regions newly black')
                    print(f'[Halcyon viewport] a GPU frame came back '
                          f'{why} -- re-shaded on '
                          f'the CPU and kept (guard #{self.guard_count}, '
                          f'{"draft" if draft else "refine"} '
                          f'{settings.resolution_x}x'
                          f'{settings.resolution_y}). Paste this line.')
                    retry = settings.copy()
                    retry._viewport = getattr(settings, '_viewport', True)
                    retry._viewport_stats = False
                    retry.render_device = 'CPU'
                    scene.settings = retry
                    img = core_render.render(scene, retry, progress=tick)
                    img = post.process(img, retry,
                                       target_size=(retry.resolution_x,
                                                    retry.resolution_y))
                    self.last_engaged = 'CPU (guard)'
                    blk, tiles = _black_measure(img)
                self._black_prev = blk
                self._black_tiles_prev = tiles
                self._black_key = key
            else:
                self._black_prev, self._black_tiles_prev = \
                    _black_measure(img)
                self._black_key = key
            stats_on = getattr(settings, '_viewport_stats', False)
            if not draft and (stats_on or not self._split_said):
                # the split prints for the FIRST refine of a session
                # even without Show Stats: the one line that names the
                # slow stage must not depend on a toggle nobody found
                self._split_said = True
                if stats_on:
                    print(f'[Halcyon viewport] refine '
                          f'{settings.resolution_x}x'
                          f'{settings.resolution_y} '
                          f'in {time.perf_counter() - t0:.2f}s on '
                          f'{self.last_engaged} '
                          f'(census: {_REDRAW["flags"]} redraw flags, '
                          f'guard {self.guard_count})')
                # the field's performance instrument: where the frame's
                # milliseconds actually went. Same doctrine as the crash
                # milestones -- one pasted line beats a round of guessing
                try:
                    parts = []
                    from .gpu.shade import LAST_TIMINGS as _LT
                    for k in ('plan_ms', 'pack_upload_ms', 'draw_read_ms',
                              'reflect_ms', 'composite_ms'):
                        if _LT.get(k):
                            parts.append(f"{k[:-3]} {float(_LT[k]):.0f}")
                    if _LT.get('burst_draw_ms') or \
                            _LT.get('burst_read_ms'):
                        # R175b: submission vs GPU-execution halves of
                        # the draw burst, measured inside the crossing
                        parts.append(
                            f"burst "
                            f"{float(_LT.get('burst_draw_ms', 0.0)):.0f}"
                            f"+{float(_LT.get('burst_read_ms', 0.0)):.0f}")
                    if _LT.get('compile_ms'):
                        # the 45-second first refine, named: driver
                        # shader compilation, once per plan
                        parts.append(
                            f"COMPILE {int(_LT.get('compile_n', 0))}x "
                            f"{float(_LT['compile_ms']):.0f}")
                    from .gpu.craster import LAST_RASTER as _LR
                    rtot = sum(float(v) for v in _LR.values()
                               if isinstance(v, (int, float)))
                    if rtot:
                        parts.append(f'raster {rtot:.0f}')
                    # R171: whether THIS refine's rasterisation came from
                    # the G-buffer cache, and if not, which key component
                    # (or gate) broke it -- the field misses, the
                    # headless twin hits, and this word is the difference
                    from .core.render import _GBUF_STATS as _GS
                    if _GS.get('last'):
                        parts.append(f"gbuf {_GS['last']}")
                    # R172: the LAST export's split, viewport edition --
                    # F12 already prints it; the viewport's own
                    # view_update exports were invisible, and whether
                    # THEY hit the mesh cache is half the field question
                    from . import export as _EXP
                    _es = getattr(_EXP, 'EXPORT_SPLIT', None) or {}
                    if _es.get('total_ms'):
                        _d = ''
                        if _es.get('dirt_all'):
                            _d = f" dirt ALL({_es['dirt_all']})"
                        elif _es.get('dirt_n'):
                            _d = f" dirt {int(_es['dirt_n'])}u"
                        parts.append(
                            f"export meshes {int(_es.get('meshes', 0))}x "
                            f"{_es.get('mesh_ms', 0.0):.0f}"
                            f" (+{int(_es.get('cached', 0))} cached)"
                            + _d)
                    acct = marshal.acct()
                    if acct.get('crossings'):
                        wait = max(acct['wall_ms'] - acct['exec_ms'], 0.0)
                        parts.append(
                            f"marshal {acct['crossings']}x "
                            f"exec {acct['exec_ms']:.0f} wait {wait:.0f}")
                    if parts:
                        print('[Halcyon viewport] split (ms): '
                              + ', '.join(parts) + '. Paste this line.')
                except Exception:                               # noqa: BLE001
                    pass
            img = np.asarray(img, np.float32)
            if img.ndim != 3 or img.shape[2] not in (3, 4):
                raise ValueError(f'viewport frame has shape {img.shape}')
            if img.shape[2] == 3:
                img = np.concatenate(
                    [img, np.ones(img.shape[:2] + (1,), np.float32)], 2)
            img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
            with self.lock:
                self.frame = np.ascontiguousarray(img)
                self.done_key = (key, version, draft)
                self.busy = False
            try:
                from . import fault_note
                blk_pct = (self._black_prev or 0.0) * 100.0
                fault_note(
                    f'viewport frame parked '
                    f'({settings.resolution_x}x{settings.resolution_y} '
                    f'{"draft" if draft else "refine"} on '
                    f'{self.last_engaged}, {blk_pct:.0f}% black)',
                    key='vp-frame')
            except Exception:                                   # noqa: BLE001
                pass
        except Cancelled:
            with self.lock:
                self.busy = False     # the next draw kicks the newer view
        except Exception:                                       # noqa: BLE001
            import traceback
            with self.lock:
                self.busy = False
            self.complain('the viewport render failed',
                          traceback.format_exc())
        finally:
            if holding:
                marshal.disable()
                marshal.PIPELINE.release()
        self._request_redraw(engine)

    @staticmethod
    def _request_redraw(engine):
        """Ask the UI to draw again, from off the main thread.

        FLAG-ONLY since 1.25.90: the worker touches NOTHING of bpy --
        not tag_redraw (removed in .89), not even timers.register (a
        per-frame/per-burst registration from the worker was the last
        cross-thread bpy call standing, and the context-state flood
        outlived every other theory). The flag is read by ONE persistent
        main-thread poll started in view_draw.
        """
        if engine is None:
            return
        request_redraw_flag(engine)

    # ------------------------------------------------------------ draw thread
    def texture(self, gpu):
        """The newest frame as a GPUTexture, rebuilt only when it changed.

        The REPLACED texture is never dropped here: the previous redraw's
        display-space draw of it may still sit queued in Blender's Vulkan
        render graph, and destroying it mid-flight is a driver access
        violation at the next swap (the field crash, verbatim: nvoglv64
        under vkCmdPipelineBarrier the moment rendered view opened on a
        heavy scene -- a refine burst churns several of these a second).
        It parks in the SCREEN grave and releases only after a full
        swapchain depth of later redraws (device.draw_tick) -- readback
        epochs never retire it, because a readback proves our offscreen
        subgraph flushed, not Blender's screen-draw stream. And nothing
        collects HERE: this runs inside the draw callback, mid-recording,
        the worst possible moment to hand the driver a free().
        """
        from .gpu import device as _dev
        with self.lock:
            frame = self.frame
            fid = id(frame)
        if self._tex is not None and self._tex_id == fid:
            return self._tex
        ih, iw = frame.shape[:2]
        flat = np.ascontiguousarray(frame.ravel())
        try:
            buf = gpu.types.Buffer('FLOAT', flat.shape[0], flat)
        except (TypeError, ValueError):
            buf = gpu.types.Buffer('FLOAT', flat.shape[0], flat.tolist())
        _dev.bury_screen(self._tex)
        self._tex = gpu.types.GPUTexture((iw, ih), format='RGBA32F', data=buf)
        self._tex_id = fid
        # the upload's SOURCE rides the same redraw clock as the texture:
        # Vulkan records, and if the backend consumes staging data at
        # flush, `flat`/`buf` dying at return would hand the deferred
        # upload freed pages
        _dev.bury_screen((flat, buf))
        return self._tex

    def draw_placeholder(self, depsgraph):
        """First frame not ready (or nothing to render): flat dark fill, so
        entering rendered mode visibly DID something while the worker runs."""
        try:
            import gpu
            col = (0.02, 0.02, 0.025, 1.0)
            sc = getattr(depsgraph, 'scene', None)
            world = getattr(sc, 'world', None)
            if world is not None:
                try:
                    c = world.color
                    col = (float(c[0]), float(c[1]), float(c[2]), 1.0)
                except Exception:                               # noqa: BLE001
                    pass
            fb = gpu.state.active_framebuffer_get()
            fb.clear(color=col)
        except Exception:                                       # noqa: BLE001
            import traceback
            self.complain('the placeholder draw failed',
                          traceback.format_exc())
