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
"""

import threading
import time

import numpy as np

from .core import post
from .core import render as core_render

#: the view is "in motion" for this long after its last change; drafts
#: render while it holds, the refine starts once it lapses
DRAFT_WINDOW = 0.35

#: drafts render at preview_scale * this -- a quarter of the pixels
DRAFT_FACTOR = 2


class Cancelled(Exception):
    """Raised inside the worker when a newer view supersedes its render."""


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
        with self.lock:
            self.scene = scene
            self.settings = settings
            self.version += 1
            self.abort = True     # anything mid-flight is of the old scene
            self._last_change = self.clock()   # an edit storm drafts too

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
                    return False     # parked draft is enough while moving
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
        try:
            scale = max(int(getattr(settings, 'preview_scale', 1)), 1)
            if draft:
                scale *= DRAFT_FACTOR
            settings.resolution_x = max(int(w) // scale, 4)
            settings.resolution_y = max(int(h) // scale, 4)
            if camera is not None:
                scene.camera = camera
            scene.settings = settings

            def tick(_frac, _msg):
                if self.abort:
                    raise Cancelled()

            img = core_render.render(scene, settings, progress=tick)
            img = post.process(img, settings,
                               target_size=(settings.resolution_x,
                                            settings.resolution_y))
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
        except Cancelled:
            with self.lock:
                self.busy = False     # the next draw kicks the newer view
        except Exception:                                       # noqa: BLE001
            import traceback
            with self.lock:
                self.busy = False
            self.complain('the viewport render failed',
                          traceback.format_exc())
        self._request_redraw(engine)

    @staticmethod
    def _request_redraw(engine):
        """Ask the UI to draw again, from off the main thread.

        `tag_redraw` is what render engines call from their render threads;
        the timer is the documented thread-safe royal road. Both are
        harmless when the view already redrew, and both are guarded --
        in a headless test there is neither an engine nor a bpy.
        """
        if engine is None:
            return
        try:
            engine.tag_redraw()
        except Exception:                                       # noqa: BLE001
            pass
        try:
            import bpy as _bpy

            def _once(eng=engine):
                try:
                    eng.tag_redraw()
                except Exception:                               # noqa: BLE001
                    pass
                return None            # run once

            _bpy.app.timers.register(_once, first_interval=0.01)
        except Exception:                                       # noqa: BLE001
            pass

    # ------------------------------------------------------------ draw thread
    def texture(self, gpu):
        """The newest frame as a GPUTexture, rebuilt only when it changed."""
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
        self._tex = gpu.types.GPUTexture((iw, ih), format='RGBA32F', data=buf)
        self._tex_id = fid
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
