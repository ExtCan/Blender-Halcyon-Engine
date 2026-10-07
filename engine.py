"""The RenderEngine: final renders, tile reporting and the viewport preview."""

import time

import bpy
import numpy as np

from . import export
from .core import post
from .core import render as core_render
from .core import stats as ST

#: the slate's render-event clock: bumped once per final render, for the
#: session's lifetime. The rainbow escape scrolls on frame + THIS, so a
#: still re-rendered on one timeline frame moves too -- the field ran
#: that exact test twice and watched a timeline-only rainbow stand
#: still. It feeds ONLY the burn-in (via stamp_info); the picture's
#: determinism contract never sees it, exactly like %T's wall clock.
_RENDER_SERIAL = {'n': 0}


#: one-shot latches for the pin warnings, per session
_PIN_FAILED = {}


def _pin_display(scene):
    """Blender's color management is OFF under Halcyon.

    The engine is display-referred: its Display settings (exposure,
    gamma, presets, the CRT chain) are the ONLY grading. Any view
    transform regrades the output a second time -- AgX crushed the
    field's 2%% sheens into a flat black body, and 'Standard' turned
    out to be an sRGB encode for scene-linear data that washed every
    render grey. 'Raw' passes the numbers straight to the screen, and
    the engine pins it on every render and viewport update while it
    is the active engine, so no preset or new file drifts back."""
    scene = getattr(scene, 'original', None) or scene
    try:
        vs = scene.view_settings
        if getattr(vs, 'view_transform', None) != 'Raw':
            try:
                vs.view_transform = 'Raw'
            except (TypeError, ValueError):
                # a build whose OCIO config renamed/removed 'Raw': say
                # so ONCE, loudly -- the field's 'still extremely dark'
                # is exactly what an unpinned AgX looks like, and a
                # silent pass here hid it
                if not _PIN_FAILED.get('said'):
                    _PIN_FAILED['said'] = True
                    have = []
                    try:
                        prop = vs.bl_rna.properties['view_transform']
                        have = [i.identifier for i in prop.enum_items]
                    except Exception:                       # noqa: BLE001
                        pass
                    print("[Halcyon] WARNING: this Blender build refused "
                          "view_transform 'Raw'"
                          + (f' (available: {have})' if have else '')
                          + ' -- Blender will REGRADE every Halcyon '
                          'render. Pick the closest pass-through '
                          'transform manually in Render Properties > '
                          'Color Management.')
        if getattr(vs, 'look', 'None') != 'None':
            try:
                vs.look = 'None'
            except (TypeError, ValueError):
                pass
        if getattr(vs, 'exposure', 0.0) != 0.0:
            vs.exposure = 0.0
        if getattr(vs, 'gamma', 1.0) != 1.0:
            vs.gamma = 1.0
    except Exception:                                           # noqa: BLE001
        pass
    try:
        ds = scene.display_settings
        if getattr(ds, 'display_device', 'sRGB') != 'sRGB':
            ds.display_device = 'sRGB'
    except Exception:                                           # noqa: BLE001
        pass


class HalcyonRenderEngine(bpy.types.RenderEngine):
    bl_idname = 'HALCYON_RENDER'
    bl_label = "Halcyon"
    bl_use_preview = True
    bl_use_shading_nodes_custom = False
    bl_use_eevee_viewport = False
    bl_use_custom_freestyle = False
    bl_use_alembic_procedural = False
    # True so Blender binds a GPU context around render() -- the final F12
    # render runs on a render thread where gpu.* otherwise has no context
    # at all ("No active GPU context found") and every stage fell back to
    # the CPU. The self-test never saw it: operators run on the main
    # thread, which has one. This is the difference between the GPU port
    # working in a report and working on F12.
    #
    # …and holding that context froze the interface for the whole frame:
    # Blender cannot draw a window while the render thread owns the GPU,
    # so a 33-second frame read "Not Responding" for 33 seconds -- the
    # field named it twice. The default is now the viewport's own
    # architecture applied to F12: the render thread runs with NO context
    # and every GPU burst (compile, upload, draw, read back) is
    # marshalled to the main thread, which always has one, through
    # gpu/marshal.py. The interface breathes between bursts. The Debug
    # panel's "Hold GPU Context" restores the old behaviour (it can
    # start bursts a tick sooner), background renders keep it
    # automatically (no interface to freeze, and -b may give the main
    # loop no timers), and every marshalling failure falls back to the
    # CPU frame with the reason printed -- never a broken picture.
    bl_use_gpu_context = False

    def __init__(self, *args, **kwargs):
        try:
            super().__init__(*args, **kwargs)
        except TypeError:
            super().__init__()
        self._scene = None
        self._scene_key = None
        self._draw_data = None
        self._last_hash = None
        self._vp = None
        # R201: the animation clock. Blender builds one engine instance
        # per render JOB and calls render() once per frame of an
        # animation, so per-frame durations kept on the instance start
        # fresh with every animation and never leak across jobs
        self._anim_times = []

    def __del__(self):
        # Blender frees the engine's StructRNA before Python collects
        # the wrapper, and then ANY attribute access raises
        # ReferenceError ('StructRNA of type HalcyonRenderEngine has
        # been removed') as console noise. Teardown here is
        # best-effort: reach the abort flag if the wrapper still
        # works, stay silent if it does not.
        try:
            vp = getattr(self, '_vp', None)
            if vp is not None:
                vp.abort = True
        except BaseException:                                   # noqa: BLE001
            pass

    # ------------------------------------------------------------ final render
    def render(self, depsgraph):
        bscene = depsgraph.scene
        _pin_display(bscene)
        tw, th = self._target_size(bscene)
        preview = bool(getattr(self, 'is_preview', False))
        settings = _settings_from_scene(bscene, tw, th, preview)
        warnings = []

        # GPU context stewardship. `granted` is what Blender already did
        # for THIS render (it read the class attribute when the job
        # started); the class attribute is then re-synced from the
        # setting so the NEXT render follows it. Background renders
        # always hold: there is no interface to keep alive, and a
        # windowless main loop may never run a timer.
        background = bool(getattr(bpy.app, 'background', False))
        granted = bool(type(self).bl_use_gpu_context)
        hold = bool(getattr(settings, 'gpu_hold_context', False)) or \
            background
        type(self).bl_use_gpu_context = hold
        from .gpu import marshal as _marshal
        gpu_dev = str(getattr(settings, 'render_device',
                              'CPU')).upper() == 'GPU'
        marshalled = not granted and gpu_dev and not background
        # one render on the driver at a time: a GPU viewport frame
        # mid-flight finishes first (they are sub-second), then this
        # render owns the pipeline for its duration -- viewport frames
        # that arrive meanwhile render on the CPU and say so
        if gpu_dev:
            _marshal.PIPELINE.acquire()
        if marshalled:
            _marshal.enable()
        try:
            self._render_body(depsgraph, bscene, tw, th, preview,
                              settings, warnings)
        finally:
            if marshalled:
                _marshal.disable()
            if gpu_dev:
                _marshal.PIPELINE.release()

    def _render_body(self, depsgraph, bscene, tw, th, preview, settings,
                     warnings):

        from .version import version_string
        # wire= names the overlay's STATE, not just the mode: this header
        # used to print wire=ALL on every render, engaged or not, while
        # the field hunted a faint wireframe the overlay itself was drawing
        wire = settings.wire_mode if settings.render_wire else 'OFF'
        # R253: Blender's render border (Ctrl+B / Output > Format > Render
        # Region), in OUTPUT pixels -- None for a whole frame
        _rb = _border_rect(bscene, tw, th) if not preview else None
        print(f"[Halcyon] {version_string()} rendering "
              f"{tw}x{th}  pass={settings.debug_pass}"
              f"  wire={wire}"
              + (f"  region x{_rb[0]}..{_rb[2]} y{_rb[1]}..{_rb[3]} "
                 f"of {tw}x{th}" if _rb is not None else ""))
        # the display truth, every render: if Blender is still regrading
        # (the pin refused, or something re-set it between the pin and
        # now), the console says so INSTEAD of the picture silently
        # coming out dark or washed
        try:
            _orig = getattr(bscene, 'original', None) or bscene
            _vt = str(getattr(_orig.view_settings, 'view_transform', '?'))
            _lk = str(getattr(_orig.view_settings, 'look', 'None'))
            if _vt != 'Raw' or _lk not in ('None', 'NONE'):
                print(f"[Halcyon] WARNING: view transform is '{_vt}' "
                      f"(look '{_lk}') -- Blender is regrading this "
                      "render on top of Halcyon's own display "
                      "settings. It will look darker or washed "
                      "compared to the engine's true output.")
        except Exception:                                   # noqa: BLE001
            pass
        self.update_stats("Halcyon", "Exporting scene")
        ST.reset()
        ST.enable(bool(settings.show_stats))
        _apply_debug_prefs()
        export_started = time.time()
        try:
            with ST.track('export scene (Blender side)'):
                scene = export.export_scene(depsgraph, settings, warnings)
        except Exception as exc:                                # noqa: BLE001
            import traceback
            traceback.print_exc()
            if not preview:
                self.report({'ERROR'}, f"Halcyon export failed: {exc}")
            return
        if not preview:
            _es = getattr(export, 'EXPORT_SPLIT', None) or {}
            if _es.get('total_ms'):
                _cc = int(_es.get('cached', 0))
                _dup = int(_es.get('cached_dup', 0))
                _dirt = ''
                if _es.get('dirt_all'):
                    _dirt = f", dirt ALL({_es['dirt_all']})"
                elif _es.get('dirt_n'):
                    _dirt = f", dirt {int(_es['dirt_n'])}u"
                print('[Halcyon] export split: '
                      f"meshes {int(_es.get('meshes', 0))}x "
                      f"{_es.get('mesh_ms', 0.0):.0f} ms"
                      + (f" ({_cc} cached"
                         + (f", {_dup} dup" if _dup else '')
                         + f" {_es.get('cached_ms', 0.0):.0f} ms)"
                         if _cc else '') + _dirt + ', '
                      f"materials {int(_es.get('mats', 0))}x "
                      f"{_es.get('mat_ms', 0.0):.0f} ms, "
                      f"concat {_es.get('concat_ms', 0.0):.0f} ms, "
                      f"other {_es.get('other_ms', 0.0):.0f} ms")

        # the device plan needs the exported scene, so it runs after the export
        # rather than before it, where `scene` did not yet exist -- the
        # UnboundLocalError went straight into a bare except and the plan has
        # never once run
        try:
            from .gpu import capability as _cap
            dev, _stages, notes = _cap.plan(scene, settings)
            if str(settings.render_device).upper() == 'GPU':
                print(f"[Halcyon] device: {dev}")
                # a stage toggled off under the GPU device is worth one
                # plain line each -- the 10-second frame that looked broken
                # was exactly this, silently
                for name, label in (('gpu_raster', 'GPU Rasteriser'),
                                    ('gpu_shading', 'GPU Shading'),
                                    ('gpu_post', 'GPU Post Processing')):
                    if not getattr(settings, name, True):
                        print(f"[Halcyon]   {label} is OFF (Debug panel) -- "
                              "this stage runs on the CPU")
                for note in notes:
                    print(f"[Halcyon]   {note}")
        except Exception:                                       # noqa: BLE001
            pass

        # R180: the supersample bill, printed and REPORTED before the wait.
        # The field lost 38 minutes to Samples 16 (a 4x4 = 16x-pixel
        # internal frame) on a scene that had also fallen off the GPU --
        # and the only warnings were console lines. The cost of a sampling
        # choice belongs where the choice is made.
        if not preview:
            try:
                import numpy as _np
                _ssf = 1
                if str(settings.aa_mode) == 'SUPERSAMPLE':
                    _ssf = max(int(_np.round(_np.sqrt(
                        max(int(settings.aa_samples), 1)))), 1)
                if _ssf > 1:
                    print(f"[Halcyon] internal resolution "
                          f"{tw * _ssf}x{th * _ssf} -- Samples "
                          f"{int(settings.aa_samples)} supersamples "
                          f"{_ssf}x{_ssf} = {_ssf * _ssf}x the output "
                          f"pixels")
                if _ssf * _ssf >= 9:
                    self.report(
                        {'WARNING'},
                        f"Samples {int(settings.aa_samples)} renders "
                        f"{_ssf * _ssf}x the output pixels "
                        f"({tw * _ssf}x{th * _ssf} internally). The "
                        f"period default is 4 (a 2x2 grid)")
            except Exception:                                   # noqa: BLE001
                pass

        def on_progress(frac, msg):
            if self.test_break():
                raise _Cancelled()
            self.update_progress(float(frac))
            self.update_stats("Halcyon", msg)

        image = None
        # R230: shooting on twos or threes -- a held frame photographs
        # the key frame's cel again instead of rendering it; the film
        # stages in the post chain still run per frame
        _hold = int(getattr(settings, 'film_hold', 1) or 1)
        _hold_key = int(scene.frame)
        _held = None
        if _hold > 1 and not preview:
            _hold_key, _held = hold_lookup(
                str(getattr(bscene, 'name', '')), int(scene.frame),
                int(getattr(bscene, 'frame_start', 1)), _hold, (tw, th),
                settings)
            if _held is not None:
                image = _held['image']
                scene.last_depth = _held.get('depth')
                scene.last_shafts = _held.get('shafts')
                scene.last_flares = _held.get('flares')
                # R251 (C092): Super Black on a held frame floors by its
                # key frame's plane (off on held frames otherwise: the
                # alternating-roads bug class)
                scene.last_coverage = _held.get('coverage')
                settings._last_coverage = scene.last_coverage
                scene.last_cvg = _held.get('cvg')          # R251 C001
                scene.last_gel = _held.get('gel')        # R251 C134
                print(f"[Halcyon] shoot on {_hold}s: frame {int(scene.frame)}"
                      f" holds frame {_hold_key} (not rendered again)")
            elif _hold_key != int(scene.frame):
                print(f"[Halcyon] shoot on {_hold}s: frame {int(scene.frame)}"
                      f" would hold frame {_hold_key}, which this session "
                      "has not rendered at this size and these settings "
                      "-- rendered fresh (render the sequence from its "
                      "first frame for the holds)")
        mb_steps = int(getattr(settings, 'motion_steps', 0) or 0)
        if image is not None:
            pass                       # a held frame: nothing to render
        elif getattr(settings, 'motion_blur', False) and mb_steps > 1 \
                and not preview:
            # accumulation motion blur: the whole frame, re-exported and
            # re-rendered at N points across the shutter, averaged -- the
            # accumulation-buffer trails of the era, at the era's honest
            # price of N frames. Runs in-process (each step is its own
            # complete render; a pool inside a pass would nest pools).
            print(f"[Halcyon] motion blur: {mb_steps} steps across a "
                  f"{float(settings.motion_shutter):.2f}-frame shutter "
                  f"({mb_steps} full renders, "
                  f"{str(getattr(settings, 'motion_blur_mode', 'MEAN'))})")
            try:
                image, scene = self._render_motion_accumulated(
                    depsgraph, bscene, settings, warnings, on_progress)
            except _Cancelled:
                return
            except Exception as exc:                            # noqa: BLE001
                import traceback
                traceback.print_exc()
                if not preview:
                    self.report({'ERROR'},
                                f"Halcyon motion blur failed: {exc}")
                return
        gpu_whole_frame = str(getattr(settings, 'render_device',
                                      'CPU')).upper() == 'GPU' and \
            bool(getattr(settings, 'gpu_shading', False))
        if image is not None:
            pass                       # motion blur already rendered above
        elif settings.use_processes and not preview and \
                core_render.wanted_passes(settings):
            print("[Halcyon] extra render passes are on, so this frame renders "
                  "in-process: the worker pool sends back pixels, not buffers")
        elif settings.use_processes and not preview and gpu_whole_frame:
            # the pool splits the frame into bands, and a band shades on
            # the CPU -- the deferred pass is whole-frame by design. With
            # the GPU device on, pooling would silently throw the driver
            # away: the field's 33-second render was exactly this, with
            # Task Manager honestly reading 0% GPU. The faster machine
            # wins; the pool stays available by turning GPU Shading off.
            print("[Halcyon] worker pool skipped: GPU Shading renders "
                  "whole frames in-process (a pooled band would shade on "
                  "the CPU). Turn GPU Shading off to use the pool instead")
        elif settings.use_processes and not preview and \
                str(settings.aa_mode) == 'ADAPTIVE' and \
                int(settings.aa_samples) > 1:
            # adaptive AA flags its edge pixels off the WHOLE base
            # frame; a pooled band cannot see across its seam, so the
            # same scene would flag (and refine) different pixels
            # banded than whole -- and identical frames are the
            # contract. In-process it is: the refine passes only shade
            # a few percent of the frame anyway
            print("[Halcyon] worker pool skipped: Adaptive anti-aliasing "
                  "reads edges off the whole base frame, so the frame "
                  "renders in-process (base + edge refine passes)")
        elif settings.use_processes and not preview and \
                str(getattr(getattr(scene, 'camera', None), 'type', '')
                    ) == 'PANO':
            # the panorama is sixteen rotated strip renders resampled
            # onto one cylinder; a pooled band of the output has no
            # meaning before the stitch exists
            print("[Halcyon] worker pool skipped: the panorama camera "
                  "renders rotated strips in-process and stitches them")
        elif settings.use_processes and not preview and \
                int(getattr(settings, 'pano_parts', 1)) > 1 and \
                getattr(scene, 'camera', None) is not None and \
                str(getattr(scene.camera, 'type', 'PERSP')) == 'PERSP':
            # R251 C125: the strips are whole renders; a pooled band
            # would be the unstitched planar frame (the gate reads
            # band is None). The camera guard mirrors the render gate's
            print("[Halcyon] worker pool skipped: Pano Parts renders N "
                  "yawed strips in-process and butts them")
        elif settings.use_processes and not preview and \
                str(getattr(settings, 'stereo_mode', 'NONE')) != 'NONE':
            # each eye is its own whole frame; a pooled band would render
            # one mono slice
            print("[Halcyon] worker pool skipped: stereo renders each eye "
                  "as a whole frame in-process")
        elif settings.use_processes and not preview and \
                getattr(scene, 'halos', None):
            # a halo is splatted against the WHOLE frame's z-buffer in
            # frame coordinates; a pooled band's buffer starts at its
            # own y and every glow would land shifted
            print("[Halcyon] worker pool skipped: halo materials splat "
                  "against the whole frame's depth, so the frame "
                  "renders in-process")
        elif settings.use_processes and not preview and \
                any(float(getattr(l, 'flare', 0.0)) > 0.0
                    for l in (getattr(scene, 'lights', None) or ())):
            # a flare's visibility is sampled off the whole frame's
            # z-buffer, which a pooled band never holds
            print("[Halcyon] worker pool skipped: lens flares sample "
                  "their lamp's visibility off the whole frame's depth, "
                  "so the frame renders in-process")
        elif settings.use_processes and not preview and \
                str(getattr(getattr(scene, 'world', None), 'weather',
                            'NONE') or 'NONE') != 'NONE':
            # R200: weather particles splat in whole-frame coordinates,
            # exactly like the halos above
            print("[Halcyon] worker pool skipped: the weather overlay "
                  "splats in whole-frame coordinates, so the frame "
                  "renders in-process")
        elif settings.use_processes and not preview and \
                core_render.composite_reads_neighbours(scene, settings):
            # R251: the Fuzz / Thin Wall blend equations read the row
            # above or beside a pixel of the FINISHED frame, which a
            # pooled band never holds at its seam
            print("[Halcyon] worker pool skipped: the "
                  "Fuzz / Thin Wall blend reads neighbouring pixels "
                  "of the finished frame, so the frame renders in-process")
        elif settings.use_processes and not preview and \
                bool(getattr(settings, 'matte_glow', False)):
            # R251 C134: the optical printer's mattes are cut from the
            # WHOLE frame's material plane, which a pooled band never
            # holds (the pool sends back pixels, not buffers)
            print("[Halcyon] worker pool skipped: Matte Glow cuts its "
                  "mattes from the whole frame's material plane, so the "
                  "frame renders in-process")
        elif settings.use_processes and not preview:
            from .core import parallel as _par
            n = int(settings.process_count) or _resolve_cpus()
            key = (id(depsgraph), bscene.frame_current, tw, th)
            try:
                with ST.track('worker pool'):
                    image, why = _par.render_parallel(scene, settings, n,
                                                      scene_key=key,
                                                      progress=on_progress)
            except _Cancelled:
                return
            except Exception as exc:                            # noqa: BLE001
                image, why = None, f'{type(exc).__name__}: {exc}'
            if image is None and why:
                print(f"[Halcyon] process pool unavailable ({why}); "
                      f"rendering in this process")

        # R250: the frame the render leaves on the GPU is kept for the
        # post chain; this method releases it in its own finally below
        settings._keep_gpu_frame = True
        try:
            if image is None:
                image = core_render.render(scene, settings, progress=on_progress)
        except _Cancelled:
            _release_gpu_frame(settings)
            return
        except Exception as exc:                                # noqa: BLE001
            import traceback
            traceback.print_exc()
            _release_gpu_frame(settings)
            if not preview:
                self.report({'ERROR'}, f"Halcyon render failed: {exc}")
            return
        if _hold > 1 and not preview and _held is None \
                and _hold_key == int(scene.frame):
            # a freshly rendered KEY frame: the held frames after it
            # photograph this cel
            hold_store(str(getattr(bscene, 'name', '')), int(scene.frame),
                       (tw, th), settings, image,
                       getattr(scene, 'last_depth', None),
                       getattr(scene, 'last_shafts', None),
                       getattr(scene, 'last_flares', None),
                       gpu=bool(getattr(settings, '_frame_gpu_shaded',
                                        False)),
                       cvg=getattr(scene, 'last_cvg', None),   # R251 C001
                       coverage=getattr(scene, 'last_coverage', None))
            # R251 C134: the key frame's gel plane rides with its cel
            _HOLD_CACHE['last']['gel'] = getattr(scene, 'last_gel', None)
        elif _held is not None:
            # R250: a held frame photographs its key frame's cel again,
            # and its post chain takes the road that cel's shading took
            # -- the flag used to stay unset here, so every held frame
            # ran its post on the CPU in silence, alternating roads
            # frame by frame under the field's own 'shoot on twos'
            settings._frame_gpu_shaded = bool(_held.get('gpu', False))
            if settings._frame_gpu_shaded:
                _once(f"[Halcyon GPU] post on a held frame: the key frame's "
                      f"cel uploaded once (frame {int(scene.frame)} holds "
                      f"frame {_hold_key})")

        # R180: the GPU verdict, in the interface. A frame that fell to
        # the CPU used to say so in one console line; at a supersampled
        # resolution that silence cost the field 38 minutes. The reason
        # now lands in Blender's own report system, visible in the
        # status bar and the Info log without a console.
        if not preview:
            try:
                from .core.render import LAST_GPU_VERDICT as _V
                if _V.get('wanted') and not _V.get('engaged'):
                    _why = str(_V.get('why') or '').strip()
                    self.report(
                        {'WARNING'},
                        "GPU shading could not run this frame; it shaded "
                        "on the CPU"
                        + (f" -- {_why[:160]}" if _why else '')
                        + ". The console has the full reason")
            except Exception:                                   # noqa: BLE001
                pass

        self.update_stats("Halcyon", "Post processing")
        try:
            # what the burn-in tokens cannot learn from the core: the
            # render clock so far and the host version. %S therefore
            # reads "everything up to post", which is what a slate
            # written DURING the frame can honestly say.
            try:
                import bpy as _bpy
                _host = str(_bpy.app.version_string)
            except Exception:                                   # noqa: BLE001
                _host = '?'
            _RENDER_SERIAL['n'] += 1
            stamp_info = {'render_time': time.time() - export_started,
                          'blender': _host,
                          'scroll': _RENDER_SERIAL['n']}
            # R251: the scene token the CRTC state (signal_era.CRTC_STATE)
            # keys on, as hold_lookup keys on the scene name
            settings._scene_name = str(getattr(bscene, 'name', ''))
            with ST.track('post processing'):
                final = post.process(image, settings, frame=scene.frame,
                                     seed=settings.seed, target_size=(tw, th),
                                     allow_resize=False,
                                     key_frame=_hold_key,
                                     fps=float(getattr(scene, 'fps', 24.0)
                                               or 24.0),
                                     depth=getattr(scene, 'last_depth', None),
                                     shaft_sources=getattr(scene, 'last_shafts',
                                                           None),
                                     flare_sources=getattr(scene,
                                                           'last_flares',
                                                           None),
                                     stamp_info=stamp_info,
                                     cvg=getattr(scene, 'last_cvg', None),  # R251 C001
                                     coverage=getattr(scene, 'last_coverage', None),
                                     gel=getattr(scene, 'last_gel', None))
        except Exception as exc:                                # noqa: BLE001
            import traceback
            traceback.print_exc()
            if not preview:
                self.report({'WARNING'}, f"Post chain failed: {exc}")
            final = post.fit_to(np.clip(image, 0.0, 1.0), (tw, th))
        finally:
            _release_gpu_frame(settings)

        _extra = getattr(scene, 'last_passes', None)
        # R253: the Beauty pass IS the linear frame post was handed --
        # palette, dither, CRT and signal stages untouched (the render
        # resolution's frame; _deliver_passes fits it to Blender's buffer,
        # and the region crop below cuts it like every other pass)
        if getattr(settings, 'pass_beauty', False) and image is not None:
            _extra = dict(_extra or {})
            _extra['Beauty'] = np.asarray(image, np.float32)
        _size = None
        if _rb is not None:
            # R253: the crop happens LAST -- the post chain ran over the
            # full-frame canvas (zeros outside the rect) so every pattern
            # stage kept the frame's origin; Blender's result under
            # use_border is border-sized, and begin_result(0, 0, bw, bh)
            # is its contract (render_result_uncrop places it in the
            # full frame when Crop to Render Region is off)
            x0, y0, x1, y1 = _rb
            try:
                final = np.asarray(final, np.float32)
                if final.shape[0] != th or final.shape[1] != tw:
                    final = post.fit_to(final, (tw, th))
                final = np.ascontiguousarray(final[y0:y1, x0:x1])
                if _extra:
                    _cut = {}
                    for _nm, _buf in _extra.items():
                        _a = np.asarray(_buf, np.float32)
                        if _a.ndim == 3 and (_a.shape[0] != th
                                             or _a.shape[1] != tw):
                            _a = post.fit_to(_a, (tw, th))
                        _cut[_nm] = np.ascontiguousarray(_a[y0:y1, x0:x1]) \
                            if _a.ndim == 3 else _a
                    _extra = _cut
                _size = (x1 - x0, y1 - y0)
            except Exception:                                   # noqa: BLE001
                import traceback
                traceback.print_exc()
                _size = None
        with ST.track('deliver to Blender'):
            self._deliver(final, bscene, _extra, size=_size)

        # reporting during a preview render pushes UI work onto the preview
        # thread for a thumbnail nobody is reading
        if not preview:
            for w in set(warnings) | set(getattr(scene, 'unsupported', ()) or ()):
                self.report({'WARNING'}, str(w))
        elapsed = time.time() - export_started
        if not preview and getattr(self, 'is_animation', False):
            # R201: the animation ETA -- per-frame time and when the
            # whole run will be done, printed after every frame
            try:
                self._anim_times.append(float(elapsed))
                line = _anim_eta_line(
                    self._anim_times,
                    int(getattr(bscene, 'frame_start', 1)),
                    int(getattr(bscene, 'frame_end', 1)),
                    int(getattr(bscene, 'frame_step', 1) or 1),
                    int(getattr(bscene, 'frame_current', 1)))
                if line:
                    print(f"[Halcyon] {line}")
                    self.update_stats("Halcyon", line)
            except Exception:                                   # noqa: BLE001
                pass
        if settings.show_stats:
            ST.report(total=elapsed)
        # the one-line answer to "where did this frame go", printed for
        # EVERY render: the 33-second mystery frame had its breakdown
        # collected all along, gated behind a panel nobody had ticked
        tops = ST.top(3)
        if tops:
            line = ', '.join(f'{n} {t:.1f}s' for n, t in tops)
            tail = '' if settings.show_stats else \
                "  (the Debug panel's Timing Breakdown toggle prints " \
                'the full table)'
            print(f"[Halcyon] {elapsed:.1f}s -- top stages: {line}{tail}")
        self.update_stats("Halcyon", f"Done in {elapsed:.1f}s")
        self.update_progress(1.0)

    def _render_motion_accumulated(self, depsgraph, bscene, settings,
                                   warnings, on_progress):
        """N complete frames across the shutter, averaged.

        Each step moves Blender to a subframe with the official
        RenderEngine.frame_set, re-exports the evaluated scene (so object,
        camera and deformation motion all blur -- whatever animates,
        blurs), renders it fully, and the mean is the frame. Returns
        (image, center_scene): the center step's scene carries the depth
        and shaft data the post chain reads, so depth of field focuses on
        the middle of the shutter.
        """
        frame0 = int(bscene.frame_current)
        # R251 C090: the mode's slice count (Max's 32 cap, LW's 2x) and the
        # per-slice weights; the combine streams with ONE slice resident
        n = core_render.shutter_steps(settings)
        mode = str(getattr(settings, 'motion_blur_mode', 'MEAN'))
        wsum = None
        shutter = max(float(settings.motion_shutter), 0.0)
        offs = np.linspace(-shutter * 0.5, shutter * 0.5, n)
        center_k = int(np.argmin(np.abs(offs)))
        acc = None
        center_scene = None
        try:
            for k, off in enumerate(offs):
                base = int(np.floor(frame0 + off))
                sub = float(frame0 + off - base)
                self.frame_set(base, sub)
                scene_k = export.export_scene(depsgraph, settings, warnings)
                img = core_render.render(
                    scene_k, settings,
                    progress=on_progress if k == center_k else None)
                if mode == 'MEAN':
                    acc = img.astype(np.float64) if acc is None else acc + img
                else:
                    w = core_render.shutter_weight(settings, k, n, img.shape[0],
                                                   img.shape[1], frame_no=frame0,
                                                   seed=int(settings.seed))
                    acc = img * w[:, :, None] if acc is None \
                        else acc + img * w[:, :, None]
                    wsum = w if wsum is None else wsum + w
                if k == center_k:
                    center_scene = scene_k
                self.update_stats("Halcyon",
                                  f"Motion blur step {k + 1}/{n}")
                if self.test_break():
                    raise _Cancelled()
        finally:
            try:
                self.frame_set(frame0, 0.0)
            except Exception:                                   # noqa: BLE001
                pass
        return ((acc / float(n)) if mode == 'MEAN'
                else (acc / wsum[:, :, None])).astype(np.float32), center_scene

    # ------------------------------------------------------------------ passes
    def update_render_passes(self, scene=None, renderlayer=None):
        """Tell Blender which extra passes this render will contain.

        Without this the render result holds one pass, Combined, and everything
        else the compositor offers reads as black. Halcyon was force-enabling
        Blender's own Passes panel while delivering exactly one pass, which is
        the worst of both: the controls were there and none of them did
        anything.
        """
        self.register_pass(scene, renderlayer, "Combined", 4, "RGBA", 'COLOR')
        st = _settings_from_scene(scene, 1, 1) if scene is not None else None
        if st is None:
            return
        for name, chans, chan_id, kind in PASS_SPEC:
            if name in core_render.wanted_passes(st):
                self.register_pass(scene, renderlayer, name, chans, chan_id, kind)

    def _deliver_passes(self, result, buffers, w, h):
        """Write the extra passes, each fitted to Blender's own buffer."""
        try:
            passes = result.layers[0].passes
        except (IndexError, AttributeError):
            return
        for name, buf in buffers.items():
            try:
                target = passes[name]
            except (KeyError, TypeError):
                continue          # not enabled on this view layer
            arr = np.asarray(buf, np.float32)
            if arr.shape[0] != h or arr.shape[1] != w:
                arr = post.fit_to(arr, (w, h))
            chans = int(getattr(target, 'channels', arr.shape[2]) or arr.shape[2])
            if arr.shape[2] < chans:
                arr = np.concatenate(
                    [arr, np.zeros(arr.shape[:2] + (chans - arr.shape[2],),
                                   np.float32)], axis=2)
            elif arr.shape[2] > chans:
                arr = arr[:, :, :chans]
            flat = np.ascontiguousarray(np.nan_to_num(
                arr, nan=0.0, posinf=1e10, neginf=0.0).reshape(-1))
            try:
                target.rect.foreach_set(flat)
            except Exception:                                   # noqa: BLE001
                try:
                    target.rect = flat.reshape(-1, chans).tolist()
                except Exception:                               # noqa: BLE001
                    pass

    def _target_size(self, bscene):
        """The exact buffer size Blender has allocated for this render.

        `size_x`/`size_y` are what the engine was actually asked for, and are the
        only correct answer during a preview render, where the scene's own
        resolution has nothing to do with the thumbnail being drawn.
        """
        w = int(getattr(self, 'size_x', 0) or 0)
        h = int(getattr(self, 'size_y', 0) or 0)
        if w > 0 and h > 0:
            return w, h
        r = bscene.render
        pct = max(r.resolution_percentage, 1) / 100.0
        return max(int(r.resolution_x * pct), 1), max(int(r.resolution_y * pct), 1)

    def _deliver(self, final, bscene, extra=None, size=None):
        """Hand the finished image back through the render result.

        The buffer size is dictated by Blender, never by the image: writing more
        floats than `rect` holds overruns a C buffer and takes the process down
        with it. Post can legitimately resize (Pixel Scale, pixel aspect), so the
        result is fitted to the allocated size before a single value is written.

        R253: `size=(bw, bh)` is the render border's size -- the result
        Blender allocates under scene.render.use_border is border-sized
        (RE_InitState keeps winx, rectx = the border), and begin_result's
        own clamp plus the len(rect) guard below keep a wrong guess from
        ever overrunning it.
        """
        w, h = self._target_size(bscene)
        if size is not None:
            w, h = max(int(size[0]), 1), max(int(size[1]), 1)
        final = np.asarray(final, np.float32)
        if final.ndim != 3 or final.shape[2] not in (3, 4):
            return
        if final.shape[2] == 3:
            final = np.concatenate(
                [final, np.ones(final.shape[:2] + (1,), np.float32)], axis=2)
        if final.shape[0] != h or final.shape[1] != w:
            final = post.fit_to(final, (w, h))
        # scrub non-finite values only when any exist: nan_to_num
        # unconditionally rewrote 10M floats per delivered frame, and a
        # finished frame is finite in every ordinary render
        if not np.isfinite(final).all():
            final = np.nan_to_num(final, nan=0.0, posinf=1.0, neginf=0.0)

        result = self.begin_result(0, 0, w, h)
        try:
            layer = result.layers[0].passes["Combined"]
            if size is not None:
                # R253: the guard -- Blender's rect is the truth; if it
                # disagrees with the border size, say both and fit to
                # what it holds rather than scramble a resize
                try:
                    _have = int(len(layer.rect))
                except Exception:                               # noqa: BLE001
                    _have = -1
                if _have > 0 and _have != w * h * 4 and _have != w * h:
                    _px = _have // 4 if _have % 4 == 0 else _have
                    print(f'[Halcyon] render region: Blender allocated '
                          f'{_px} pixels for the result, the border is '
                          f'{w}x{h}={w * h}; the picture is fitted to the '
                          'allocation')
                    _tw, _th = self._target_size(bscene)
                    if _px == _tw * _th:
                        w, h = _tw, _th
                        final = post.fit_to(final, (w, h))
            if extra:
                self._deliver_passes(result, extra, w, h)
            # Both buffers are bottom-row-first: the rasteriser maps NDC y = -1
            # to row 0, and Blender's rect expects the bottom row first.
            # ONE flat float32 buffer: foreach_set's single-memcpy fast
            # path wants an unambiguous 1-D typed buffer; anything it
            # has to iterate is 10M Python float conversions
            flat = np.ascontiguousarray(final, np.float32).reshape(-1)
            expected = w * h * 4
            if flat.size != expected:                    # belt and braces
                flat = np.resize(flat, expected)
            try:
                layer.rect.foreach_set(flat)
            except Exception:                                   # noqa: BLE001
                layer.rect = flat.reshape(-1, 4).tolist()
            try:
                from . import fault_note
                fault_note(f'F12 render delivered ({w}x{h})', key='f12')
            except Exception:                                   # noqa: BLE001
                pass
        finally:
            self.end_result(result)

    # --------------------------------------------------------------- viewport
    #
    # The shape Blender's own engines use: view_update syncs data (main
    # thread, bpy access is legal), view_draw only DRAWS. The old path
    # rendered synchronously inside the draw callback -- the whole UI froze
    # for every frame, seconds at a time on a real scene, and any failure
    # left the viewport permanently blank. Rendering now happens on a
    # background thread against the bpy-free exported scene; every draw
    # blits the newest finished frame and kicks a fresh render only when
    # the camera, the region or the scene actually moved on.

    def view_update(self, context, depsgraph):
        vp = self._viewport()
        try:
            _pin_display(depsgraph.scene)
        except Exception:                                       # noqa: BLE001
            pass
        try:
            region = getattr(context, 'region', None)
            w = int(getattr(region, 'width', 0) or 0) or 640
            h = int(getattr(region, 'height', 0) or 0) or 480
            settings = _viewport_settings(depsgraph.scene, w, h)
            kind = _classify_updates(depsgraph) if vp.has_scene() \
                else 'full'
            if kind == 'none':
                # nothing renderable changed: keep the parked export.
                # Blender fires view_update for EVERY depsgraph poke --
                # selections, UI state, other add-ons' timers -- and the
                # field runs several add-ons that poke constantly. Each
                # poke was a full 100-object re-export (~2 s on the real
                # file) followed by a draft+refine cascade; the session
                # read as 'incredibly laggy' while standing still.
                # Settings edits still land (the cheap half), and a
                # same-settings poke costs nothing at all
                vp.set_settings(settings)
                self.tag_redraw()
                return
            if kind == 'lights':
                # a light moved or changed its values: re-export the
                # LIGHTS ONLY into a shallow next scene that shares the
                # meshes, materials and images with the parked export.
                # Dragging a lamp used to run the full main-thread
                # re-export (~2 s on the field's file) EVERY drag tick
                # -- 'Blender will freeze up' -- for data no lamp can
                # change. Sharing the objects also keeps every identity
                # cache warm downstream (BVH, textures, shadow maps).
                try:
                    scene = export.export_lights_into(vp.scene,
                                                      depsgraph)
                    vp.set_scene(scene, settings)
                    self.tag_redraw()
                    return
                except Exception:                               # noqa: BLE001
                    pass                    # fall through: full export
            scene = export.export_scene(depsgraph, settings)
            vp.set_scene(scene, settings)
            try:
                from . import fault_note
                fault_note('viewport export delivered to the worker',
                           key='vp-export')
            except Exception:                                   # noqa: BLE001
                pass
        except Exception:                                       # noqa: BLE001
            import traceback
            vp.complain('the viewport export failed',
                        traceback.format_exc())
        self.tag_redraw()

    def view_draw(self, context, depsgraph):
        import gpu

        # view_draw does NOTHING with the marshal queue. Bursts cross to
        # the main thread only in the pump's timer slices BETWEEN redraws
        # -- 1.25.89-.91 drained them here, mid-draw, and every redraw
        # that ran bursts mutated live driver state under Blender's
        # compositor: the whole scene flashed rapidly whenever bursts
        # streamed (camera motion AND refines). A time budget and a
        # viewport-rect fence each failed to tame it; the field called
        # it a seizure risk and the architecture was reverted. This
        # callback only blits the newest parked frame. The one thing it
        # does start is the persistent redraw-flag poll -- registered
        # HERE because view_draw is guaranteed main-thread.
        from . import preview as _preview
        _preview.ensure_redraw_timer()

        vp = self._viewport()
        region = context.region
        w, h = max(int(region.width), 4), max(int(region.height), 4)

        if vp.scene is None:
            # Blender promises a view_update before the first view_draw, but
            # a failed export leaves nothing to render -- say so on screen
            # rather than showing an eternal void
            vp.draw_placeholder(depsgraph)
            return

        # R253: the render rect (camera frame / Blender's border) joins
        # the wanted view; a failure here is a whole-region frame, never
        # a blank viewport
        try:
            _rect = _view_rect(context, depsgraph.scene, vp.settings)
        except Exception:                                       # noqa: BLE001
            import traceback
            vp.complain('the viewport render rect failed',
                        traceback.format_exc())
            _rect = None
        vp.want(_view_camera(context), w, h, rect=_rect)
        vp.kick(self)

        if vp.frame is None:
            vp.draw_placeholder(depsgraph)
            return
        try:
            tex = vp.texture(gpu)
            with vp.lock:
                _frect = vp.frame_rect
            if _frect is not None:
                # camera-frame mode: the parked frame is the rect's own
                # size -- clear the region to the placeholder colour
                # first, then blit at the rect's place
                vp.clear_region(depsgraph)
                _bx = int(round(float(_frect[0]) * w))
                _by = int(round(float(_frect[1]) * h))
                _bw = max(int(round(float(_frect[2]) * w)) - _bx, 1)
                _bh = max(int(round(float(_frect[3]) * h)) - _by, 1)
            else:
                _bx = _by = 0
                _bw, _bh = w, h
            self.bind_display_space_shader(depsgraph.scene)
            _draw_texture(tex, _bw, _bh, _bx, _by)
            self.unbind_display_space_shader()
            # the redraw completed: advance the screen grave's clock --
            # the ONLY thing that retires replaced blit textures and
            # evicted blit batches (readback epochs cannot; see
            # device.bury_screen)
            try:
                from .gpu import device as _dev
                _dev.draw_tick()
            except Exception:                                   # noqa: BLE001
                pass
            try:
                from . import fault_note
                fault_note('viewport frame drawn to screen',
                           key='vp-blit', limit=3)
            except Exception:                                   # noqa: BLE001
                pass
        except Exception:                                       # noqa: BLE001
            import traceback
            vp.complain('drawing the viewport frame failed',
                        traceback.format_exc())

    def _viewport(self):
        if getattr(self, '_vp', None) is None:
            from .preview import Viewport
            self._vp = Viewport()
        return self._vp


#: datablock type names whose update means the EXPORT is stale. Anything
#: else (Screen, WindowManager, Scene pokes with no geometry flag,
#: another add-on's property churn) re-renders at most, never re-exports.
_RENDER_ID_TYPES = frozenset((
    'Object', 'Mesh', 'Curve', 'SurfaceCurve', 'TextCurve', 'MetaBall',
    'Material', 'Light', 'World', 'Image', 'Armature', 'Lattice',
    'GreasePencil', 'Volume', 'PointCloud', 'Collection', 'Key',
    # R199 field find: editing the halo colour ramp -- a node UNLINKED
    # from any output -- updates only the ShaderNodeTree id; Blender
    # tags the owning Material only when shading is affected. The ramp
    # widget therefore never live-updated: the poke classified 'none'
    # and the parked export stood. Any node-tree edit re-exports now
    'ShaderNodeTree', 'NodeTree', 'GeometryNodeTree',
))


#: light DATA arrives as its concrete subclass -- type(uid).__name__ is
#: 'SunLight', never 'Light' -- so the generic name alone never matched
_LIGHT_ID_TYPES = frozenset((
    'Light', 'SunLight', 'PointLight', 'SpotLight', 'AreaLight',
))


def _anim_frame_cost(times):
    """The per-frame seconds an animation ETA should multiply by.

    R201: the first frame pays for what the rest reuse -- shadow maps,
    BVHs, mesh caches, compiled shaders -- so a plain mean drags the
    warm-up cost into every later prediction. The LOWER median of the
    last five frames forgets the warm-up the moment one steady frame
    exists, and still absorbs a single hiccup frame later in the run
    (one slow outlier among steady neighbours never becomes the
    estimate)."""
    if not times:
        return 0.0
    recent = sorted(times[-5:])
    return float(recent[(len(recent) - 1) // 2])


def _fmt_span(sec):
    """Seconds as a human span: 42s, 3m 05s, 1h 12m."""
    sec = max(float(sec), 0.0)
    if sec < 60.0:
        return f'{sec:.0f}s'
    m, s = divmod(int(round(sec)), 60)
    if m < 60:
        return f'{m}m {s:02d}s'
    h, m = divmod(m, 60)
    return f'{h}h {m:02d}m'


def _anim_eta_line(times, start, end, step, current, now=None):
    """One console line of animation progress and estimated completion.

    Pure of bpy and clock-injectable (`now` for the tests): the engine
    passes its per-frame durations and the scene's frame range, and
    gets back either "frame K/N in Xs -- ~R to go, done around H:MM"
    or the final tally line on the last frame."""
    step = max(int(step), 1)
    total = max((int(end) - int(start)) // step + 1, 1)
    done = min(max((int(current) - int(start)) // step + 1, 1), total)
    this = times[-1] if times else 0.0
    if done >= total:
        return (f'animation finished: {total} frame'
                f'{"s" if total != 1 else ""} in '
                f'{_fmt_span(sum(times))}')
    left = (total - done) * _anim_frame_cost(times)
    t_now = time.time() if now is None else float(now)
    clock = time.strftime('%H:%M', time.localtime(t_now + left))
    return (f'animation frame {done}/{total} in {this:.1f}s -- '
            f'~{_fmt_span(left)} to go, done around {clock}')


def _classify_updates(depsgraph):
    """'none' | 'lights' | 'full' -- what this depsgraph poke invalidates.

    'lights' means every render-relevant update is a lamp: light data
    (colour, energy, spot angle -- arriving as the concrete subclass,
    SunLight et al) or a light OBJECT's transform. Those re-export
    through the cheap lights-only path; anything else that touches the
    render re-exports fully, and pure UI churn not at all."""
    try:
        ups = list(getattr(depsgraph, 'updates', ()) or ())
    except Exception:                                           # noqa: BLE001
        return 'full'
    if not ups:
        # no update list at all: assume the worst, export
        return 'full'
    touched = False
    lights_only = True
    for u in ups:
        uid = getattr(u, 'id', None)
        tname = type(uid).__name__ if uid is not None else ''
        if tname in _LIGHT_ID_TYPES:
            touched = True
            continue
        if tname == 'Object' and getattr(uid, 'type', '') == 'LIGHT':
            touched = True
            continue
        if tname in _RENDER_ID_TYPES:
            touched = True
            lights_only = False
            continue
        if getattr(u, 'is_updated_geometry', False) or \
                getattr(u, 'is_updated_transform', False):
            touched = True
            lights_only = False
    if not touched:
        return 'none'
    return 'lights' if lights_only else 'full'


def _updates_touch_render(depsgraph):
    """True when a depsgraph update invalidates the exported scene."""
    return _classify_updates(depsgraph) != 'none'


#: R230: the frame hold -- the last KEY frame's rendered cel, so a held
#: frame photographs it again instead of rendering. One entry: sequences
#: render in order. Keyed on the scene, the key frame, the size and the
#: settings' fingerprint (the film stages' own dials excluded: they run
#: per frame either way).
_HOLD_CACHE = {}


def _hold_fingerprint(settings):
    import dataclasses
    # every film_* dial but the paint stages (misregister and bleed run
    # at render time, before the ink) and the transparency: the post
    # chain's film runs per frame either way
    keep = {'film_misregister', 'film_bleed', 'film_transparent'}
    skip = {'threads', 'process_count', 'use_processes'}
    parts = []
    for f in dataclasses.fields(settings):
        if f.name in skip or (f.name.startswith('film_')
                              and f.name not in keep):
            continue
        v = getattr(settings, f.name)
        parts.append((f.name, repr(v)))
    return hash(tuple(parts))


def hold_lookup(scene_name, frame, start, hold, size, settings):
    """(key_frame, cached) for `frame`: cached is the key frame's cel
    when this session rendered it at this size with these settings,
    else None. A key frame itself never looks anything up."""
    from .core.film import hold_key
    key = hold_key(frame, start, hold)
    if key == int(frame):
        return key, None
    entry = _HOLD_CACHE.get('last')
    if not entry:
        return key, None
    if entry['scene'] != scene_name or entry['frame'] != key \
            or entry['size'] != tuple(size) \
            or entry['fp'] != _hold_fingerprint(settings):
        return key, None
    return key, entry


def hold_store(scene_name, frame, size, settings, image, depth, shafts,
               flares, gpu=False, cvg=None, coverage=None):
    """Remember a key frame's cel (and the data the post chain reads)
    for the held frames that follow it. `gpu`: whether that cel's
    shading engaged the GPU (the held frames' post takes the same road)."""
    _HOLD_CACHE['last'] = {
        'scene': scene_name, 'frame': int(frame), 'size': tuple(size),
        'fp': _hold_fingerprint(settings), 'image': image, 'depth': depth,
        'shafts': shafts, 'flares': flares, 'gpu': bool(gpu),
        'cvg': cvg,                                     # R251 C001
        # R251 (C092): the key frame's coverage plane -- a held frame is
        # posted with a fresh settings object and no render
        'coverage': coverage}


_ONCE = set()


def _once(msg):
    if msg not in _ONCE:
        _ONCE.add(msg)
        print(msg)


def _release_gpu_frame(settings):
    """R250: whatever the render left resident on the GPU goes back to
    the pool; idempotent, and never an error."""
    try:
        from .gpu import frame as _FR
        _FR.release(settings)
    except Exception:                                           # noqa: BLE001
        pass


class _Cancelled(Exception):
    pass


def _apply_debug_prefs():
    """Push the developer preferences into the bpy-free core."""
    from .core import nodeeval
    try:
        p = bpy.context.preferences.addons[__package__].preferences
        nodeeval.STRICT = bool(p.debug_mode and p.strict_nodes)
    except Exception:                                           # noqa: BLE001
        nodeeval.STRICT = False


def _resolve_cpus():
    import os
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


SCALE_FACTOR = {'NONE': 1, '2X': 2, '3X': 3, '4X': 4,
                'THREEDO_2X': 2, 'GBA_MODE5': 0}   # R251: 0 = ask signal_era


def _settings_from_scene(bscene, target_w, target_h, preview=False):
    """Scene settings mapped onto the buffer Blender actually asked for.

    Pixel Scale is interpreted as an *internal* downscale: the output stays the
    size Blender wants, and the engine renders at 1/N of it and blows the result
    back up with nearest-neighbour. That is what makes the setting useful (a
    genuine chunky-pixel render filling a normal output) and it keeps the
    delivered buffer exactly the size Blender allocated.
    """
    hs = getattr(bscene, 'halcyon', None)
    if hs is None:
        from .core.settings import RenderSettings
        st = RenderSettings()
    else:
        st = hs.to_settings()

    # Blender's own Film > Transparent is where people look for this, so it
    # wins when it is on; the Halcyon toggle can also enable it independently.
    if getattr(bscene.render, 'film_transparent', False):
        st.film_transparent = True
    # R253: Blender's render border (Ctrl+B in camera view / Output >
    # Format > Render Region) rides the same derived road: the five
    # fields are scene.render's own, never a Halcyon property. Preview
    # thumbnails ignore it (a material ball has no border)
    st.use_border = False
    if not preview and bool(getattr(bscene.render, 'use_border', False)):
        try:
            st.border_min_x = float(bscene.render.border_min_x)
            st.border_min_y = float(bscene.render.border_min_y)
            st.border_max_x = float(bscene.render.border_max_x)
            st.border_max_y = float(bscene.render.border_max_y)
            st.use_border = True
        except (AttributeError, TypeError, ValueError):
            st.use_border = False
    n = SCALE_FACTOR.get(str(st.output_scale), 1)
    if str(st.output_scale) == 'GBA_MODE5':
        # R251: the Mode 5 bitmap the LCD reads through PA / PD
        from .core import signal_era as _SIG
        st.resolution_x, st.resolution_y = _SIG.gba_source_size(int(target_w), int(target_h))
    else:
        st.resolution_x = max(int(target_w) // n, 1)
        st.resolution_y = max(int(target_h) // n, 1)
    # Blender applies its own pixel aspect at display time; applying it here too
    # would double the stretch and change the buffer size
    st.pixel_aspect_x = st.pixel_aspect_y = 1.0
    # Halcyon's own thread count wins; 0 means "ask Blender", and Blender's AUTO
    # means "ask the machine"
    if int(st.threads) <= 0 and bscene.render.threads_mode != 'AUTO':
        st.threads = int(bscene.render.threads)

    if preview:
        # Thumbnails are tiny, generated in bulk, and drawn on a background
        # thread. Anything that costs seconds here is felt as a hang.
        st.aa_mode = 'NONE'
        st.aa_samples = 1
        st.shadows = False
        st.raytrace = False
        st.ambient_occlusion = False
        st.glow = st.star_filter = st.lens_flare = False
        st.jpeg_artifacts = False
        st.crt = False
        st.composite = False
        st.output_scale = 'NONE'
        st.resolution_x = min(max(int(target_w), 1), 256)
        st.resolution_y = min(max(int(target_h), 1), 256)
    return st


def _border_rect(bscene, tw, th):
    """R253: Blender's render border at OUTPUT size, (x0, y0, x1, y1), or
    None when scene.render.use_border is off or the rect is the whole
    frame. Takes the scene or its `render` struct. The rounding is the
    pipeline's own (render_init_from_main: disprect = border * winx,
    truncated) on both edges -- the region the engine CROPS to; the
    render rect core/render.region_pixels builds from the same fractions
    ceils its max edge, so the render is always a superset of this."""
    r = getattr(bscene, 'render', bscene)
    if not bool(getattr(r, 'use_border', False)):
        return None
    tw, th = max(int(tw), 1), max(int(th), 1)
    try:
        x0 = int(float(r.border_min_x) * tw)
        y0 = int(float(r.border_min_y) * th)
        x1 = int(float(r.border_max_x) * tw)
        y1 = int(float(r.border_max_y) * th)
    except (AttributeError, TypeError, ValueError):
        return None
    x0, y0 = min(max(x0, 0), tw - 1), min(max(y0, 0), th - 1)
    x1, y1 = min(max(x1, x0 + 1), tw), min(max(y1, y0 + 1), th)
    if (x0, y0, x1, y1) == (0, 0, tw, th):
        return None
    return (x0, y0, x1, y1)


def _view_rect(context, bscene, settings):
    """R253: the viewport's render rect as FRACTIONS of the region, or
    None -- `preview.region_rect(w, h, frame, border)` over:

    * the CAMERA FRAME, in camera view, when the Performance panel's
      Camera Frame Only is on: the camera's view_frame corners projected
      through view3d_utils onto the region (min / max of the four);
    * the BORDER: in camera view Blender's scene border (Ctrl+B writes
      scene.render.border_* there) mapped INTO the frame rect, the
      Cycles rule (BlenderSync::get_buffer_params); in a free view the
      space's own render border (space_data.use_render_border).
    """
    rv3d = getattr(context, 'region_data', None)
    region = getattr(context, 'region', None)
    if rv3d is None or region is None:
        return None
    w = max(int(getattr(region, 'width', 0) or 0), 4)
    h = max(int(getattr(region, 'height', 0) or 0), 4)
    in_camera = str(getattr(rv3d, 'view_perspective', '')) == 'CAMERA'
    frame = None
    border = None
    cam = getattr(bscene, 'camera', None)
    if in_camera and cam is not None:
        frame_rect = None
        try:
            from bpy_extras import view3d_utils as _v3u
            corners = cam.data.view_frame(scene=bscene)
            mw = cam.matrix_world
            pts = []
            for p in corners:
                q = _v3u.location_3d_to_region_2d(region, rv3d, mw @ p)
                if q is None:
                    pts = []
                    break
                pts.append((float(q[0]), float(q[1])))
            if pts:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                frame_rect = (min(xs) / w, min(ys) / h,
                              max(xs) / w, max(ys) / h)
        except Exception:                                       # noqa: BLE001
            frame_rect = None
        if bool(getattr(settings, 'viewport_camera_frame', False)):
            frame = frame_rect
        r = getattr(bscene, 'render', None)
        if r is not None and bool(getattr(r, 'use_border', False)):
            try:
                from .preview import border_in_frame
                border = border_in_frame(
                    frame_rect, (float(r.border_min_x), float(r.border_min_y),
                                 float(r.border_max_x), float(r.border_max_y)))
            except (AttributeError, TypeError, ValueError):
                border = None
    else:
        space = getattr(context, 'space_data', None)
        if space is not None and bool(getattr(space, 'use_render_border',
                                              False)):
            try:
                border = (float(space.render_border_min_x),
                          float(space.render_border_min_y),
                          float(space.render_border_max_x),
                          float(space.render_border_max_y))
            except (AttributeError, TypeError, ValueError):
                border = None
    from .preview import region_rect
    return region_rect(w, h, frame, border)


def _viewport_settings(bscene, w, h):
    """Render settings for an interactive preview of the region.

    The bpy half (reading the scene) lives here; the shaping (what a
    per-redraw render keeps and what it strips) lives in preview.py where
    the test suite can hold it. Since 1.25.83 the device rides through:
    the CPU/GPU switch governs the viewport exactly as it governs F12 --
    the worker thread borrows the same marshal the F12 render thread uses,
    so this path no longer pins render_device to the CPU.
    """
    from .preview import shape_settings
    return shape_settings(
        _settings_from_scene(bscene, max(w, 4), max(h, 4)), w, h)


def _view_camera(context):
    """The viewport's own camera as a Halcyon Camera, or None."""
    rv3d = context.region_data
    if rv3d is None:
        return None
    from .core.scene import Camera
    view_matrix = np.asarray(rv3d.view_matrix, np.float32)
    persp = bool(getattr(rv3d, 'is_perspective', True))
    space = getattr(context, 'space_data', None)
    return Camera(
        matrix_world=np.linalg.inv(view_matrix).astype(np.float32),
        projection=np.asarray(rv3d.window_matrix, np.float32),
        type='PERSP' if persp else 'ORTHO',
        clip_start=float(getattr(space, 'clip_start', 0.1) or 0.1),
        clip_end=float(getattr(space, 'clip_end', 1000.0) or 1000.0))


#: the viewport blit batches, cached per region size FOR THE SESSION. A
#: batch built per REDRAW died at function exit with its draw still
#: queued in Blender's Vulkan render graph (executed at the swap, later)
#: -- dead geometry in the barrier walk, the field's first crashlog. It
#: also explains why the crash never cared whether materials planned
#: onto the GPU: this blit runs on every rendered-view redraw
#: regardless. Burying the old batch on every size change was the
#: weaker successor of the same mistake -- draft and refine ALTERNATE
#: sizes several times a second, so the graveyard was fed screen-drawn
#: geometry continuously. Sizes in a session are few; every one is kept.
_BLIT = {'batches': {}}


def _draw_texture(tex, w, h, x=0, y=0):
    """Blit a texture over the region (R253: or at (x, y) with size
    (w, h) inside it -- the camera-frame rect; batches key on all four,
    the same session-long doctrine).

    The template idiom, with two amendments: TRI_STRIP instead of the
    deprecated TRI_FAN `draw_texture_2d` still carries (the fan leaves in
    Blender 6.0, and its warning was flooding consoles once per redraw),
    and premultiplied blending so a Film > Transparent frame composites
    over the viewport instead of overwriting it with black.
    """
    import gpu
    from gpu_extras.batch import batch_for_shader
    shader = gpu.shader.from_builtin('IMAGE')
    batches = _BLIT['batches']
    x, y = int(x), int(y)
    key = (w, h) if (x == 0 and y == 0) else (x, y, w, h)
    batch = batches.get(key)
    if batch is None:
        batch = batch_for_shader(
            shader, 'TRI_STRIP',
            {'pos': ((x, y), (x + w, y), (x, y + h), (x + w, y + h)),
             'texCoord': ((0, 0), (1, 0), (0, 1), (1, 1))})
        if len(batches) >= 16:
            # a continuously resized viewport: the eldest sizes park in
            # the SCREEN grave (redraw-clocked), never freed while their
            # draws can still be queued
            from .gpu import device as _dev
            for k in list(batches)[:8]:
                _dev.bury_screen(batches.pop(k))
        batches[key] = batch
    gpu.state.blend_set('ALPHA_PREMULT')
    try:
        shader.uniform_sampler('image', tex)
        batch.draw(shader)
    finally:
        gpu.state.blend_set('NONE')


#: (name, channels, channel ids, type) exactly as Blender names them, so a
#: Halcyon Z pass drops into a comp built for Cycles without rewiring
PASS_SPEC = (
    ("Depth", 1, "Z", 'VALUE'),
    ("Normal", 3, "XYZ", 'VECTOR'),
    ("Position", 3, "XYZ", 'VECTOR'),
    ("UV", 3, "UVA", 'VECTOR'),
    ("IndexOB", 1, "X", 'VALUE'),
    ("IndexMA", 1, "X", 'VALUE'),
    # R253: the compositing passes -- Blender Internal 2.79's own names
    # (render_result.c: Mist, Env, Diffuse, Spec, Shadow, AO, Emit, Color)
    # so a comp built for BI drops in; Beauty is the linear frame before
    # the post chain; Light00..Light07 the per-lamp split (a fixed slot
    # count, so registration needs no scene knowledge)
    ("Mist", 1, "Z", 'VALUE'),
    ("Env", 3, "RGB", 'COLOR'),
    ("Beauty", 4, "RGBA", 'COLOR'),
    ("Diffuse", 3, "RGB", 'COLOR'),
    ("Spec", 3, "RGB", 'COLOR'),
    ("Ambient", 3, "RGB", 'COLOR'),
    ("Emit", 3, "RGB", 'COLOR'),
    ("Shadow", 3, "RGB", 'COLOR'),
    ("AO", 3, "RGB", 'COLOR'),
    ("Color", 3, "RGB", 'COLOR'),
) + tuple((f"Light{i:02d}", 3, "RGB", 'COLOR')
          for i in range(core_render.LIGHT_PASS_SLOTS))


# ---------------------------------------------------------------- UI plumbing

# Blender marks its own engine-agnostic property panels by listing
# 'BLENDER_RENDER' in COMPAT_ENGINES. Discovering them beats maintaining a list
# by hand: the hand-written one was missing the material slot list, the UV map
# list, colour attributes, vertex groups and colour management, and it would
# have gone on rotting with every Blender release.
GENERIC_MARKER = 'BLENDER_RENDER'

# Panels for features this engine genuinely does not implement. Showing a
# control that silently does nothing is worse than not showing it.
EXCLUDED_PANELS = {
    'RENDER_PT_freestyle', 'RENDER_PT_freestyle_line_style',
    'VIEWLAYER_PT_freestyle', 'VIEWLAYER_PT_freestyle_lineset',
    'VIEWLAYER_PT_freestyle_style_modules',
    'VIEWLAYER_PT_freestyle_lineset_collection',
    'MATERIAL_PT_freestyle_line', 'RENDER_PT_gpencil',
    'RENDER_PT_simplify_greasepencil',
}

# Panels Blender does not mark as generic but that are safe and needed here.
FORCED_PANELS = {
    'MATERIAL_PT_context_material', 'MATERIAL_PT_viewport',
    'DATA_PT_context_mesh', 'DATA_PT_uv_texture', 'DATA_PT_vertex_colors',
    'DATA_PT_mesh_attributes', 'DATA_PT_vertex_groups', 'DATA_PT_shape_keys',
    'DATA_PT_normals', 'DATA_PT_customdata', 'DATA_PT_face_maps',
    'DATA_PT_context_light', 'DATA_PT_light', 'DATA_PT_EEVEE_light',
    'DATA_PT_spot', 'DATA_PT_area',
    'WORLD_PT_context_world', 'WORLD_PT_viewport_display',
    'RENDER_PT_color_management', 'RENDER_PT_color_management_curves',
}
# VIEWLAYER_PT_layer_passes is deliberately *not* forced: it lists Vector
# and Denoising Data, which this engine does not produce (R253: Mist, Env
# and the light components ARE produced now, under Blender Internal's own
# names). Halcyon has its own Passes panel offering exactly the passes it
# fills, which is the same rule EXCLUDED_PANELS exists for.

_patched = []


def _panel_classes():
    seen = set()
    stack = list(bpy.types.Panel.__subclasses__())
    while stack:
        cls = stack.pop()
        name = getattr(cls, '__name__', '')
        if name in seen:
            continue
        seen.add(name)
        stack.extend(cls.__subclasses__())
        yield name, cls


def enable_compatible_panels():
    """Let Halcyon use every stock property panel that is engine-agnostic."""
    global _patched
    _patched = []
    engine = HalcyonRenderEngine.bl_idname
    for name, cls in _panel_classes():
        if name in EXCLUDED_PANELS:
            continue
        compat = getattr(cls, 'COMPAT_ENGINES', None)
        if compat is None:
            continue
        if GENERIC_MARKER not in compat and name not in FORCED_PANELS:
            continue
        if getattr(cls, 'bl_space_type', '') != 'PROPERTIES':
            continue
        try:
            if engine not in compat:
                compat.add(engine)
                _patched.append(cls)
        except Exception:                                       # noqa: BLE001
            pass
    return _patched


def disable_compatible_panels():
    engine = HalcyonRenderEngine.bl_idname
    for cls in _patched:
        try:
            cls.COMPAT_ENGINES.discard(engine)
        except Exception:                                       # noqa: BLE001
            pass
    _patched.clear()


try:
    from bpy.app.handlers import persistent as _persistent
except Exception:                                               # noqa: BLE001
    def _persistent(f):
        return f


@_persistent
def _pin_display_handler(*_args):
    """Main-thread pin: render()/view_update see EVALUATED scenes,
    where writes may not stick. This runs on depsgraph updates and
    file loads and pins the ORIGINAL whenever Halcyon is active."""
    try:
        for scene in bpy.data.scenes:
            if getattr(scene.render, 'engine', '') == 'HALCYON_RENDER':
                _pin_display(scene)
    except Exception:                                           # noqa: BLE001
        pass


def register():
    bpy.utils.register_class(HalcyonRenderEngine)
    enable_compatible_panels()
    try:
        import bpy.app.handlers as _h
        if _pin_display_handler not in _h.depsgraph_update_post:
            _h.depsgraph_update_post.append(_pin_display_handler)
        if _pin_display_handler not in _h.load_post:
            _h.load_post.append(_pin_display_handler)
        _pin_display_handler()             # the already-open file too
    except Exception:                                           # noqa: BLE001
        pass


def unregister():
    try:
        from .core import parallel as _par
        _par.shutdown()
    except Exception:                                           # noqa: BLE001
        pass
    disable_compatible_panels()
    try:
        import bpy.app.handlers as _h
        for hl in (_h.depsgraph_update_post, _h.load_post):
            if _pin_display_handler in hl:
                hl.remove(_pin_display_handler)
    except Exception:                                           # noqa: BLE001
        pass
    try:
        bpy.utils.unregister_class(HalcyonRenderEngine)
    except Exception:                                           # noqa: BLE001
        pass
