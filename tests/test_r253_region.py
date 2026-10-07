"""R253 (camera-box): the render region -- Blender's Ctrl+B border for
F12 and the viewport, and the viewport's Camera Frame Only option.

One core mechanism (core/render.py `region_pixels` / `_region_keep` /
`_apply_region`): a full-frame G-buffer, shading masked to the rect plus
a context ring, zeros outside, the full-frame shape returned -- so the
post chain's pattern stages keep the frame's origin by construction and
the crop happens last (engine.py for F12, preview.py for the viewport).
Every heading here pins the rect BITWISE against the full frame's rect:
the plain frame, the supersampled frame, the ink / bump / radiosity /
edge / adaptive / Fuzz context rings, the see-through composite, the
pooled bands, the stereo and accumulate roads, the GLSL simulator's
region twin, the post chain over the zero-padded canvas, the viewport's
two modes and the engine's border-sized delivery.

    python -m halcyon.tests.test_r253_region
"""
import contextlib
import io
import os
import sys
import traceback

import numpy as np

from . import utf8_console
from .test_render import base_settings
from .scenebuild import demo_scene
from .featurematrix import SCENES, region_bump_scene
from ..core import parallel as PAR
from ..core import post
from ..core import render as R
from ..core.settings import RenderSettings
from ..gpu import shade as GSH

FAILS = []
f32 = np.float32

#: the design's rect: a quarter-ish window off the frame's centre
BORDER = dict(use_border=True, border_min_x=0.25, border_min_y=0.2,
              border_max_x=0.8, border_max_y=0.9)


def check(name, cond, extra=''):
    ok = bool(cond)
    print(f'  {"ok  " if ok else "FAIL"} {name}  {extra}' if extra
          else f'  {"ok  " if ok else "FAIL"} {name}')
    if not ok:
        FAILS.append(name)
    return ok


def _same(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return a.shape == b.shape and bool(np.array_equal(a, b))


def _bordered(st, **kw):
    stb = st.copy()
    for k, v in BORDER.items():
        setattr(stb, k, v)
    for k, v in kw.items():
        setattr(stb, k, v)
    return stb


def _inside_outside(reg, full, rect):
    """(max |reg - full| inside the rect, max |reg| outside it)."""
    x0, y0, x1, y1 = rect
    inside = float(np.abs(reg[y0:y1, x0:x1] - full[y0:y1, x0:x1]).max())
    m = np.ones(reg.shape[:2], bool)
    m[y0:y1, x0:x1] = False
    outside = float(np.abs(reg[m]).max()) if m.any() else 0.0
    return inside, outside


def _pin(label, sc_fn, st, w=160, h=120):
    """The bitwise pin: render(sc, border) has the full shape, equals
    render(sc, st)[rect] inside and is RGBA zero outside."""
    sc = sc_fn(st)
    full = R.render(sc, st)
    stb = _bordered(st)
    sc2 = sc_fn(stb)
    reg = R.render(sc2, stb)
    rect = R.region_pixels(stb, w, h)
    ok_shape = reg.shape == full.shape
    inside, outside = _inside_outside(reg, full, rect) if ok_shape \
        else (float('nan'), float('nan'))
    check(f'{label}: the region frame keeps the full shape, equals the '
          'full frame inside the rect bitwise and is zero outside',
          ok_shape and inside == 0.0 and outside == 0.0,
          f'shape {reg.shape} vs {full.shape}, inside {inside}, '
          f'outside {outside}, rect {rect}')
    return full, reg, rect


# -------------------------------------------------------------- the helper


def test_region_pixels_rounding():
    """region_pixels(): Blender's truncation on the min edge, a ceil on
    the max edge (a Pixel Scale render rect is a superset of the output
    crop), clamps, None for off / full / empty, a 1-px minimum."""
    st = RenderSettings()
    check('use_border off -> None', R.region_pixels(st, 160, 120) is None)
    st.use_border = True
    check('the whole frame -> None (the identity road)',
          R.region_pixels(st, 160, 120) is None)
    st.border_min_x, st.border_min_y = 0.25, 0.2
    st.border_max_x, st.border_max_y = 0.8, 0.9
    check('fractions -> output pixels (min truncated, max ceiled)',
          R.region_pixels(st, 160, 120) == (40, 24, 128, 108),
          str(R.region_pixels(st, 160, 120)))
    st.border_min_x, st.border_max_x = 0.333, 0.6667
    got = R.region_pixels(st, 100, 120)
    check('min 0.333*100 truncates to 33, max 0.6667*100 ceils to 67',
          got == (33, 24, 67, 108), str(got))
    st.border_min_x, st.border_max_x = 0.5, 0.5
    check('an empty rect -> None', R.region_pixels(st, 160, 120) is None)
    st.border_min_x, st.border_max_x = 0.999, 1.0
    got = R.region_pixels(st, 160, 120)
    check('a sliver at the right edge is at least one pixel wide, clamped',
          got is not None and got[2] - got[0] >= 1 and got[2] <= 160,
          str(got))
    st.border_min_x, st.border_max_x = -0.5, 1.5
    st.border_min_y, st.border_max_y = 0.2, 0.9
    got = R.region_pixels(st, 160, 120)
    check('out-of-range fractions clamp to the frame',
          got == (0, 24, 160, 108), str(got))
    st.border_min_y = float('nan')
    check('a non-finite fraction -> None (never a crash)',
          R.region_pixels(st, 160, 120) is None)
    st.border_min_y = 0.2
    st.use_border = False
    check('off again -> None', R.region_pixels(st, 160, 120) is None)
    # the keep mask and its box
    k = R._region_keep((40, 24, 128, 108), 320, 240, 2, 3)
    check('the keep mask is the rect at internal resolution plus the ring',
          k.shape == (240, 320) and bool(k[48 - 3:216 + 3, 80 - 3:256 + 3].all())
          and not k[48 - 4, 100] and not k[100, 80 - 4]
          and not k[216 + 3, 100] and not k[100, 256 + 3])
    check('the region box is the mask\'s bounds (x, y, w, h)',
          R._region_box(k) == (77, 45, 182, 174), str(R._region_box(k)))
    z = R._apply_region(np.ones((120, 160, 4), f32), (40, 24, 128, 108))
    check('_apply_region zeroes RGBA outside the rect and nothing inside',
          float(z[24:108, 40:128].min()) == 1.0
          and float(z[:24].max()) == 0.0 and float(z[108:].max()) == 0.0
          and float(z[:, :40].max()) == 0.0 and float(z[:, 128:].max()) == 0.0)
    zb = R._apply_region(np.ones((30, 160, 4), f32), (40, 24, 128, 108),
                         y_off=20)
    check('a band\'s y offset shifts the rows it zeroes',
          float(zb[:4].max()) == 0.0 and float(zb[4:, 40:128].min()) == 1.0)


# ----------------------------------------------------------- the core pin


def test_region_is_bitwise_the_full_frame_rect():
    """demo_scene at 160x120 with the design's border: every road the
    context ring exists for -- plain, 2x supersampled, a width-3 ink,
    a ShaderNodeBump material, EDGE and ADAPTIVE AA, radiosity spacing 2,
    the Fuzz / Thin Wall neighbour blends, a see-through A-buffer -- is
    bitwise the full frame's rect and zero outside."""
    W, H = 160, 120
    plain = lambda st: demo_scene(st)                           # noqa: E731
    _pin('plain', plain, base_settings(W, H))
    _pin('supersampled (2x2)', plain,
         base_settings(W, H, aa_mode='SUPERSAMPLE', aa_samples=4))
    _pin('supersampled, triangle filter', plain,
         base_settings(W, H, aa_mode='SUPERSAMPLE', aa_samples=4,
                       aa_filter='TRIANGLE', aa_filter_width=1.5))
    _pin('ink line of width 3 (the ink reach ring)', plain,
         base_settings(W, H, outline=True, outline_width=3))
    _pin('ShaderNodeBump material (the bump context ring)',
         region_bump_scene, base_settings(W, H, shadows=False))
    _pin('EDGE anti-aliasing (the edge tent ring)', plain,
         base_settings(W, H, aa_mode='EDGE'))
    _pin('ADAPTIVE anti-aliasing (the contrast window ring)', plain,
         base_settings(W, H, aa_mode='ADAPTIVE', aa_samples=4))
    _pin('radiosity spacing 2 (the interpolated grid ring)', plain,
         base_settings(W, H, radiosity=True, radiosity_spacing=2,
                       shadows=False))
    _pin('Thin Wall blend (composite_reads_neighbours ring)',
         SCENES['thin_wall'],
         base_settings(W, H, transparency='ABUFFER', shadows=False))
    # the Fuzz blend reads the finished frame row by row, layer upon
    # layer: no finite ring holds, so the frame renders whole and is cut
    # after, by name (the worker-pool gate's own precedent)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _pin('Fuzz blend equation (rendered whole, cropped by name)',
             SCENES['glass'],
             base_settings(W, H, transparency='ABUFFER',
                           blend_equation='FUZZ', shadows=False))
    check('the Fuzz refusal is printed by name, once per frame',
          buf.getvalue().count('render region: the Fuzz / Thin Wall blend '
                               'reads neighbouring pixels') == 1,
          buf.getvalue()[-300:])

    def see_through(st):
        sc = demo_scene(st)
        sc.materials[1].opacity = 0.4
        return sc
    _pin('see-through material (A-buffer composite)', see_through,
         base_settings(W, H, transparency='ABUFFER'))
    _pin('see-through material, sorted blend', see_through,
         base_settings(W, H, transparency='SORTED'))
    _pin('transparent film', plain,
         base_settings(W, H, film_transparent=True))
    # a second rect: the frame's corner (both edges on the frame edge)
    st = base_settings(W, H)
    sc = demo_scene(st)
    full = R.render(sc, st)
    stc = st.copy()
    stc.use_border = True
    stc.border_min_x, stc.border_min_y = 0.0, 0.0
    stc.border_max_x, stc.border_max_y = 0.4, 0.3
    reg = R.render(demo_scene(stc), stc)
    rect = R.region_pixels(stc, W, H)
    inside, outside = _inside_outside(reg, full, rect)
    check('a corner rect (0,0)-(0.4,0.3) pins the same way',
          rect == (0, 0, 64, 36) and inside == 0.0 and outside == 0.0,
          f'{rect} {inside} {outside}')
    # the identity: a border that covers the frame IS the full frame
    sti = st.copy()
    sti.use_border = True
    check('a full-frame border renders the full frame bitwise (identity)',
          _same(R.render(demo_scene(sti), sti), full))
    # default neutrality: the six new fields at their defaults
    st0 = base_settings(W, H)
    check('the six new fields default to the no-op (old scenes untouched)',
          st0.use_border is False and st0.border_min_x == 0.0
          and st0.border_min_y == 0.0 and st0.border_max_x == 1.0
          and st0.border_max_y == 1.0 and st0.viewport_camera_frame is False
          and R.region_pixels(st0, W, H) is None)


def test_region_outputs_stay_full_frame():
    """scene.last_depth, last_passes, last_coverage, last_cvg, last_shafts
    and last_flares keep the full frame's shape and values inside the
    rect (the post chain and the engine crop them with the picture)."""
    W, H = 160, 120
    st = base_settings(W, H, dof=True, pass_depth=True, pass_normal=True,
                       super_black=True)
    sc = demo_scene(st)
    R.render(sc, st)
    d_full = sc.last_depth.copy()
    p_full = {k: v.copy() for k, v in sc.last_passes.items()}
    c_full = sc.last_coverage.copy()
    sh_full, fl_full = list(sc.last_shafts), list(sc.last_flares)
    stb = _bordered(st)
    sc2 = demo_scene(stb)
    R.render(sc2, stb)
    x0, y0, x1, y1 = R.region_pixels(stb, W, H)
    check('last_depth keeps the full shape and equals the full frame\'s '
          'inside the rect (NaN at the sky on both)',
          sc2.last_depth.shape == d_full.shape
          and bool(np.array_equal(sc2.last_depth[y0:y1, x0:x1],
                                  d_full[y0:y1, x0:x1], equal_nan=True)))
    ok = set(sc2.last_passes) == set(p_full)
    for k, v in p_full.items():
        got = sc2.last_passes.get(k)
        ok = ok and got is not None and got.shape == v.shape \
            and _same(got[y0:y1, x0:x1], v[y0:y1, x0:x1])
    check('last_passes (Depth, Normal) keep the full shape and the rect\'s '
          'values', ok, str(sorted(sc2.last_passes)))
    check('last_coverage (Super Black) stays full-frame and equal inside',
          sc2.last_coverage is not None
          and sc2.last_coverage.shape == c_full.shape
          and _same(sc2.last_coverage[y0:y1, x0:x1], c_full[y0:y1, x0:x1]))
    check('last_cvg is absent on both (no coverage plane asked for)',
          sc.last_cvg is None and sc2.last_cvg is None)
    check('last_shafts / last_flares are the full frame\'s own',
          len(sc2.last_shafts) == len(sh_full)
          and len(sc2.last_flares) == len(fl_full))
    # the N64 coverage plane rides too
    stn = base_settings(W, H, raster_rules='N64')
    scn = demo_scene(stn)
    R.render(scn, stn)
    if getattr(scn, 'last_cvg', None) is not None:
        cv_full = scn.last_cvg.copy()
        stnb = _bordered(stn)
        scn2 = demo_scene(stnb)
        R.render(scn2, stnb)
        check('last_cvg (the N64 coverage plane) stays full-frame and '
              'equal inside the rect',
              scn2.last_cvg is not None and scn2.last_cvg.shape == cv_full.shape
              and _same(scn2.last_cvg[y0:y1, x0:x1], cv_full[y0:y1, x0:x1]))
    else:
        check('last_cvg: the N64 rules carry no plane here (skipped)', True)


def test_region_with_bands_and_the_pool():
    """A band of a region frame equals the full frame's rows x rect; the
    worker pool bands the rect's rows and returns the in-process region
    frame bitwise (skips with a reason when the pool is unavailable, as
    test_worker_pool does)."""
    W, H = 160, 120
    st = base_settings(W, H)
    sc = demo_scene(st)
    full = R.render(sc, st)
    stb = _bordered(st)
    x0, y0, x1, y1 = R.region_pixels(stb, W, H)
    band = R.render(demo_scene(stb), stb, band=(y0 + 5, y0 + 30))
    ref = R._apply_region(full[y0 + 5:y0 + 30].copy(), (x0, y0, x1, y1),
                          y_off=y0 + 5)
    check('a band of the region frame is the full frame\'s rows x rect, '
          'zero outside', _same(band, ref),
          f'{band.shape} max {float(np.abs(band - ref).max()) if band.shape == ref.shape else "shape"}')
    band2 = R.render(demo_scene(stb), stb, band=(0, 10))
    check('a band entirely outside the rect is all zero',
          band2.shape == (10, W, 4) and float(np.abs(band2).max()) == 0.0)
    # several bands rejoin to the whole region frame
    reg = R.render(demo_scene(stb), stb)
    rows = [(y, min(y + 25, H)) for y in range(0, H, 25)]
    joined = np.concatenate([R.render(demo_scene(stb), stb, band=b)
                             for b in rows], axis=0)
    check('bands of a region frame rejoin to the region frame bitwise',
          _same(joined, reg))
    # the pool (the rect's pixel count must clear the split gate: at
    # 640x480 the design's rect is 352x336)
    stp = base_settings(640, 480)
    scp = demo_scene(stp)
    stpb = _bordered(stp)
    gate_ok = PAR.find_interpreter() is not None
    if gate_ok:
        ref_p = R.render(demo_scene(stpb), stpb)
        img, err = PAR.render_parallel(scp, stpb, workers=3,
                                       scene_key='r253-region')
    else:
        img, err = None, 'no worker interpreter'
    if img is None:
        check('the worker pool renders a region (skipped: unavailable '
              'here)', True, str(err))
    else:
        check('the worker pool\'s region frame is the in-process region '
              'frame bitwise', _same(img, ref_p),
              f'{img.shape} vs {ref_p.shape}')
    # a rect too short to split declines with a reason, never a crash
    stps = _bordered(stp, border_min_y=0.5, border_max_y=0.52)
    _none, why = PAR.render_parallel(scp, stps, workers=4)
    check('a rect too short for the workers declines with a reason',
          _none is None and bool(why), str(why))
    PAR.shutdown()


def _post_kw(sc, st):
    return dict(frame=1, seed=st.seed,
                target_size=(st.resolution_x, st.resolution_y),
                allow_resize=False,
                coverage=getattr(sc, 'last_coverage', None),
                depth=getattr(sc, 'last_depth', None),
                shaft_sources=getattr(sc, 'last_shafts', None),
                flare_sources=getattr(sc, 'last_flares', None),
                cvg=getattr(sc, 'last_cvg', None),
                gel=getattr(sc, 'last_gel', None))


def test_region_post_chain_anchoring():
    """post.process over the zero-padded region canvas vs over the full
    frame: the per-pixel pattern stages (CRT mask / scanlines / vignette,
    16-bit + Bayer, noise and N64 noise dither, interlace, the display
    transform, Super Black) are bitwise over the rect -- the full-frame
    anchoring contract; the non-local stages (glow, NTSC composite)
    differ ONLY within their kernel radius of the rect edge and are
    bitwise deeper inside (the documented region-local tolerance)."""
    W, H = 160, 120
    base = base_settings(W, H)
    sc = demo_scene(base)
    full = R.render(sc, base)
    stb = _bordered(base)
    sc2 = demo_scene(stb)
    reg = R.render(sc2, stb)
    x0, y0, x1, y1 = R.region_pixels(stb, W, H)
    per_pixel = [
        ('CRT aperture mask + scanlines + vignette',
         dict(crt=True, crt_mask='APERTURE', crt_mask_strength=0.6,
              crt_scanlines=0.4, crt_vignette=0.3)),
        ('CRT slot mask', dict(crt=True, crt_mask='SLOT', crt_scanlines=0.2)),
        ('CRT shadow mask', dict(crt=True, crt_mask='SHADOW')),
        ('16-bit colour, Bayer 4x4', dict(color_depth='16', dither='BAYER4')),
        ('noise dither', dict(color_depth='15', dither='NOISE')),
        ('N64 RDP noise dither', dict(color_depth='15', dither='N64_NOISE')),
        ('interlace ODD field', dict(interlace='ODD')),
        ('display transform', dict(exposure=1.3, gamma=1.2, contrast=0.2,
                                   saturation=1.4, brightness=0.05)),
        ('Super Black', dict(super_black=True)),
        ('8-bit palette (fixed)', dict(color_depth='8', dither='BAYER2')),
    ]
    for label, ov in per_pixel:
        stp = base.copy()
        stq = stb.copy()
        for k, v in ov.items():
            setattr(stp, k, v)
            setattr(stq, k, v)
        if ov.get('super_black'):
            # the coverage plane is built at render time
            scp = demo_scene(stp)
            fullp = R.render(scp, stp)
            scq = demo_scene(stq)
            regp = R.render(scq, stq)
        else:
            scp, scq, fullp, regp = sc, sc2, full, reg
        a = post.process(fullp, stp, **_post_kw(scp, stp))
        b = post.process(regp, stq, **_post_kw(scq, stq))
        inside = float(np.abs(a[y0:y1, x0:x1] - b[y0:y1, x0:x1]).max()) \
            if a.shape == b.shape else float('nan')
        check(f'post {label}: bitwise over the rect (full-frame anchoring)',
              a.shape == b.shape and inside == 0.0,
              f'{a.shape} vs {b.shape} max {inside}')
    # the non-local stages: region-local within the kernel radius, and
    # deeper inside equal to within the running-sum blur's float
    # rounding (the box blur is a cumulative sum along the row: the zeros
    # left of the rect change the rounding of every later sum by an ulp,
    # which the 8-bit quantisation turns into at most one 1/255 step)
    non_local = [
        ('glow (radius 4, triple box = 12 px)',
         dict(glow=True, glow_radius=4.0, glow_threshold=0.3), 14,
         1.0 / 255.0 + 1e-6),
        ('NTSC composite (chroma blur, ringing)',
         dict(composite=True, composite_bleed=0.5, composite_ringing=0.3),
         12, 1e-4),
    ]
    for label, ov, margin, tol in non_local:
        stp = base.copy()
        stq = stb.copy()
        for k, v in ov.items():
            setattr(stp, k, v)
            setattr(stq, k, v)
        a = post.process(full, stp, **_post_kw(sc, stp))
        b = post.process(reg, stq, **_post_kw(sc2, stq))
        d = np.abs(a - b).max(axis=2)
        deep = d[y0 + margin:y1 - margin, x0 + margin:x1 - margin]
        rim = d[y0:y1, x0:x1]
        check(f'post {label}: within {tol:.2g} deeper than {margin} px '
              'inside the rect (region-local within the kernel radius; '
              'float rounding of the running sum beyond)',
              deep.size > 0 and float(deep.max()) <= tol,
              f'deep max {float(deep.max()) if deep.size else "?"}, rim max '
              f'{float(rim.max())}')
        check(f'post {label}: the rect edge IS where the stage differs '
              '(the tolerance is real, not vacuous)',
              float(rim.max()) > tol, f'rim max {float(rim.max())}')


def test_region_glsl_twin():
    """gpu/shade.simulate(job, gbuf, region=box) -- the headless twin of
    the scissored driver readback -- equals simulate(job, gbuf)[box]
    bitwise and is zero (hit False) outside; the box is the render's own
    region box (rect + ring at internal resolution)."""
    from .test_r251_material import _job_for
    st = base_settings(96, 72, shadows=False)
    st.transparency = 'NONE'
    sc = demo_scene(st, with_texture=False)
    g, job = _job_for(sc, st)
    GSH._PLAN_CACHE.clear()
    passes, why, atl = GSH.plan_frame(job, g)
    check('the demo plans onto the GPU (the twin has something to prove)',
          passes is not None, str(why))
    if passes is None:
        return
    full, hit_f = GSH.simulate(job, g, passes, atl)
    check('the full simulation runs', full is not None, str(hit_f))
    if full is None:
        return
    stb = _bordered(st)
    rect = R.region_pixels(stb, 96, 72)
    keep = R._region_keep(rect, 96, 72, 1, R._region_reach(sc, stb, 0, False))
    box = R._region_box(keep)
    part, hit_p = GSH.simulate(job, g, passes, atl, region=box)
    check('the region simulation runs', part is not None, str(hit_p))
    if part is None:
        return
    bx, by, bw, bh = box
    inside = float(np.abs(part[by:by + bh, bx:bx + bw]
                          - full[by:by + bh, bx:bx + bw]).max())
    m = np.ones((72, 96), bool)
    m[by:by + bh, bx:bx + bw] = False
    check('simulate(region) equals the full simulation inside the box '
          'bitwise', part.shape == full.shape and inside == 0.0,
          f'max {inside}')
    check('and is zero with hit False outside it',
          float(np.abs(part[m]).max()) == 0.0 and not hit_p[m].any()
          and bool((hit_p[~m] == hit_f[~m]).all()))
    check('the box covers the rect and its ring',
          bx <= rect[0] and by <= rect[1] and bx + bw >= rect[2]
          and by + bh >= rect[3])
    # a bump material: the height pre-pass runs over the box's lanes
    stb2 = _bordered(base_settings(96, 72, shadows=False))
    stb2.transparency = 'NONE'
    scb = region_bump_scene(stb2)
    g2, job2 = _job_for(scb, stb2)
    GSH._PLAN_CACHE.clear()
    passes2, why2, atl2 = GSH.plan_frame(job2, g2)
    if passes2 is None:
        check('the bump scene plans (skipped: refused by name)', True,
              str(why2))
        return
    full2, _h = GSH.simulate(job2, g2, passes2, atl2)
    keep2 = R._region_keep(rect, 96, 72, 1,
                           R._region_reach(scb, stb2, 0, True))
    box2 = R._region_box(keep2)
    part2, _h2 = GSH.simulate(job2, g2, passes2, atl2, region=box2)
    bx, by, bw, bh = box2
    ok = full2 is not None and part2 is not None and float(np.abs(
        part2[by:by + bh, bx:bx + bw] - full2[by:by + bh, bx:bx + bw]).max()) == 0.0
    check('the bump height pre-pass over the box lanes keeps the box '
          'bitwise (the +1 ring)', ok)


def test_region_gpu_device_falls_back_equal():
    """render_device GPU with no driver: the region frame equals the CPU
    region frame bitwise, and LAST_GPU_VERDICT names the driver's
    absence, never the region."""
    W, H = 160, 120
    st = base_settings(W, H)
    stb = _bordered(st)
    cpu = R.render(demo_scene(stb), stb)
    stg = _bordered(st)
    stg.render_device = 'GPU'
    gpu = R.render(demo_scene(stg), stg)
    check('the GPU device falls back onto the CPU region frame bitwise',
          _same(cpu, gpu))
    why = str(R.LAST_GPU_VERDICT.get('why', ''))
    check('the verdict names the refusal, not the region',
          R.LAST_GPU_VERDICT.get('wanted') is True
          and 'region' not in why.lower(), why)
    # the same with the ink + bump row the matrix homes
    stm = _bordered(base_settings(W, H, shadows=False, outline=True,
                                  outline_width=3))
    cpu2 = R.render(region_bump_scene(stm), stm)
    stm2 = stm.copy()
    stm2.render_device = 'GPU'
    gpu2 = R.render(region_bump_scene(stm2), stm2)
    check('ink + bump region: both devices agree bitwise', _same(cpu2, gpu2))


def test_region_pano_refuses_by_name():
    """A PANO camera and Pano Parts 3 with a border print the named line
    once, return the full shape, equal the un-bordered render inside the
    rect and are zero outside."""
    W, H = 128, 48
    st = base_settings(W, H)
    st.transparency = 'NONE'
    sc = demo_scene(st, with_texture=False)
    sc.camera.type = 'PANO'
    full = R.render(sc, st)
    stb = _bordered(st)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        reg = R.render(sc, stb)
    text = buf.getvalue()
    line = 'render region: the panorama camera renders whole and is ' \
           'cropped after the stitch'
    rect = R.region_pixels(stb, W, H)
    inside, outside = _inside_outside(reg, full, rect)
    check('PANO: the refusal is printed by name, once',
          text.count(line) == 1, text[:200])
    check('PANO: the full shape, equal inside the rect, zero outside',
          reg.shape == full.shape and inside == 0.0 and outside == 0.0,
          f'{inside} {outside}')
    # pano parts
    st3 = base_settings(96, 72)
    st3.transparency = 'NONE'
    st3.pano_parts = 3
    sc3 = demo_scene(st3, with_texture=False)
    full3 = R.render(sc3, st3)
    st3b = _bordered(st3)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        reg3 = R.render(sc3, st3b)
    rect3 = R.region_pixels(st3b, 96, 72)
    inside, outside = _inside_outside(reg3, full3, rect3)
    check('Pano Parts: the refusal is printed by name, once',
          buf.getvalue().count(line) == 1)
    check('Pano Parts: the full shape, equal inside the rect, zero outside',
          reg3.shape == full3.shape and inside == 0.0 and outside == 0.0,
          f'{inside} {outside}')


def test_region_stereo_and_motion():
    """Stereo SBS with a border = the packing of each eye's region frame
    (at zero eye distance: the mono region frame packed); ACCUMULATE 4
    samples with a border = the full accumulate's rect bitwise."""
    W, H = 96, 72
    st = base_settings(W, H)
    st.transparency = 'NONE'
    st.stereo_mode = 'SBS'
    st.stereo_eye_distance = 0.0
    st.stereo_convergence = 8.0
    stb = _bordered(st)
    sbs = R.render(demo_scene(st, with_texture=False), st)
    sbs_b = R.render(demo_scene(stb, with_texture=False), stb)
    mono_b = stb.copy()
    mono_b.stereo_mode = 'NONE'
    mono_reg = R.render(demo_scene(mono_b, with_texture=False), mono_b)
    packed = R._pack_stereo(mono_reg, mono_reg, 'SBS')
    check('SBS + border is the packing of the mono region frame (zero '
          'eye distance)', sbs_b.shape == sbs.shape and _same(sbs_b, packed),
          f'{sbs_b.shape} vs {packed.shape}')
    st.stereo_eye_distance = 0.12
    stb.stereo_eye_distance = 0.12
    sbs2 = R.render(demo_scene(st, with_texture=False), st)
    sbs2_b = R.render(demo_scene(stb, with_texture=False), stb)
    x0, y0, x1, y1 = R.region_pixels(stb, W, H)
    # the squeeze keeps every other column (source column 2c -> half
    # column c): the rect's columns [x0, x1) map to [ceil(x0/2),
    # ceil(x1/2)) in each half, bitwise
    hx0, hx1 = (x0 + 1) // 2, (x1 + 1) // 2
    half = W // 2
    ok = _same(sbs2_b[y0:y1, hx0:hx1], sbs2[y0:y1, hx0:hx1]) and \
        _same(sbs2_b[y0:y1, half + hx0:half + hx1],
              sbs2[y0:y1, half + hx0:half + hx1])
    check('SBS with parted eyes: each half equals the full SBS inside the '
          'squeezed rect bitwise', ok)
    outside = np.ones(sbs2_b.shape[:2], bool)
    outside[y0:y1, hx0:hx1] = False
    outside[y0:y1, half + hx0:half + hx1] = False
    check('and is zero outside both halves\' rects',
          float(np.abs(sbs2_b[outside]).max()) == 0.0)
    # accumulate
    sta = base_settings(W, H, aa_mode='ACCUMULATE', aa_samples=4)
    sta.transparency = 'NONE'
    acc = R.render(demo_scene(sta, with_texture=False), sta)
    stab = _bordered(sta)
    acc_b = R.render(demo_scene(stab, with_texture=False), stab)
    inside, outside_v = _inside_outside(acc_b, acc,
                                        R.region_pixels(stab, W, H))
    check('ACCUMULATE 4 samples: the region equals the full accumulate\'s '
          'rect bitwise, zero outside', inside == 0.0 and outside_v == 0.0,
          f'{inside} {outside_v}')


# -------------------------------------------------------------- viewport


def test_region_rect_helper():
    """preview.region_rect(w, h, frame, border) and border_in_frame:
    frame only, border only, the border mapped relative to the frame
    (Blender's camera-view rule), intersection, empty -> None, full ->
    None, Blender's truncation at the pixel level."""
    from ..preview import border_in_frame, region_rect
    check('nothing -> None', region_rect(200, 100) is None)
    got = region_rect(200, 100, frame=(0.25, 0.25, 0.75, 0.75))
    check('frame only: the rect, flagged frame_only',
          got == (0.25, 0.25, 0.75, 0.75, True), str(got))
    got = region_rect(200, 100, border=(0.25, 0.25, 0.75, 0.75))
    check('border only: the rect, not frame_only',
          got == (0.25, 0.25, 0.75, 0.75, False), str(got))
    b = border_in_frame((0.2, 0.1, 0.8, 0.9), (0.5, 0.0, 1.0, 0.5))
    check('border_in_frame maps the border relative to the frame',
          np.allclose(b, (0.5, 0.1, 0.8, 0.5)), str(b))
    check('border_in_frame with no frame is the border itself',
          border_in_frame(None, (0.1, 0.2, 0.3, 0.4)) == (0.1, 0.2, 0.3, 0.4))
    got = region_rect(200, 100, frame=(0.2, 0.1, 0.8, 0.9),
                      border=border_in_frame((0.2, 0.1, 0.8, 0.9),
                                             (0.5, 0.0, 1.0, 0.5)))
    check('frame + mapped border: the intersection, frame_only',
          got is not None and np.allclose(got[:4], (0.5, 0.1, 0.8, 0.5))
          and got[4] is True, str(got))
    check('a border outside the frame -> None (empty)',
          region_rect(200, 100, frame=(0.0, 0.0, 0.4, 0.4),
                      border=(0.5, 0.5, 1.0, 1.0)) is None)
    check('the whole region -> None', region_rect(200, 100,
                                                  border=(0.0, 0.0, 1.0, 1.0))
          is None)
    got = region_rect(200, 100, border=(0.333, 0.2, 0.6667, 0.9))
    check('fractions quantise to whole pixels with Blender\'s truncation',
          got is not None and got[:4] == (66 / 200, 20 / 100, 133 / 200,
                                          90 / 100), str(got))
    got1 = region_rect(200, 100, border=(0.3331, 0.2001, 0.66671, 0.9001))
    check('sub-pixel jitter maps to the same rect (a stable key)',
          got1 == got, f'{got1} vs {got}')
    check('an unclamped frame is clamped', region_rect(
        200, 100, frame=(-0.5, -0.5, 0.5, 0.5)) == (0.0, 0.0, 0.5, 0.5, True))
    check('a non-finite frame is ignored', region_rect(
        200, 100, frame=(float('nan'), 0, 1, 1)) is None)


def _vp_wait(vp):
    import time as _time
    for _ in range(600):
        with vp.lock:
            if not vp.busy:
                return vp.frame
        _time.sleep(0.02)
    return None


def test_viewport_border_mode():
    """Viewport.want(cam, 192, 144, rect=(0.25,0.25,0.75,0.75)) parks a
    region-sized refine whose rect equals render()+post.process of the
    same settings with use_border set, alpha 0 outside; a changed rect
    re-kicks, the same rect does not; rect None reproduces
    test_the_viewport_preview_works_headlessly's frames bitwise."""
    from ..core.scene import Camera
    from ..preview import Viewport, DRAFT_WINDOW, shape_settings

    st = base_settings(96, 72, shadows=False)
    st.preview_scale = 2
    sc = demo_scene(st, with_texture=False)
    t = {'now': 100.0}
    vp = Viewport(clock=lambda: t['now'])
    vp.set_scene(sc, st)
    view, proj, _vpm, _eye = R.camera_matrices(sc.camera, 96, 72)
    cam = Camera(matrix_world=np.linalg.inv(view).astype(np.float32),
                 projection=proj.astype(np.float32), type='PERSP')
    rect = (0.25, 0.25, 0.75, 0.75)
    vp.want(cam, 192, 144, rect=rect)
    check('a rect view kicks a draft', vp.kick() is True)
    frame = _vp_wait(vp)
    check('the draft parked at the region\'s draft size',
          frame is not None and frame.shape == (36, 48, 4),
          str(None if frame is None else frame.shape))
    t['now'] += DRAFT_WINDOW + 0.1
    check('at rest the rect view refines', vp.kick() is True)
    frame = _vp_wait(vp)
    check('the refine parked region-sized (border mode keeps the whole '
          'region)', frame is not None and frame.shape == (72, 96, 4)
          and vp.frame_rect is None,
          str(None if frame is None else frame.shape))
    if frame is None:
        return
    # the reference: the same settings road with use_border set
    sc.camera = cam
    ref_st = st.copy()
    shape_settings(ref_st, 192, 144)
    ref_st.resolution_x, ref_st.resolution_y = 96, 72
    ref_st.use_border = True
    ref_st.border_min_x, ref_st.border_min_y = rect[0], rect[1]
    ref_st.border_max_x, ref_st.border_max_y = rect[2], rect[3]
    ref_st._viewport = True
    img = R.render(sc, ref_st)
    ref = post.process(img, ref_st, frame=getattr(sc, 'frame', 0),
                       cvg=getattr(sc, 'last_cvg', None), coverage=None,
                       seed=ref_st.seed, target_size=(96, 72),
                       flare_sources=getattr(sc, 'last_flares', None))
    x0, y0, x1, y1 = R.region_pixels(ref_st, 96, 72)
    check('the parked frame IS render()+post of the bordered settings '
          'inside the rect', _same(frame[y0:y1, x0:x1], ref[y0:y1, x0:x1]),
          f'{(x0, y0, x1, y1)}')
    m = np.ones((72, 96), bool)
    m[y0:y1, x0:x1] = False
    check('and alpha 0 (RGBA zero) outside it',
          float(np.abs(frame[m]).max()) == 0.0)
    check('the rect\'s pixels are a real picture',
          float(frame[y0:y1, x0:x1, :3].std()) > 0.02)
    # the same rect again: nothing to do
    vp.want(cam, 192, 144, rect=rect)
    check('the same rect does not re-kick', vp.kick() is False)
    vp.want(cam, 192, 144, rect=(0.2, 0.2, 0.7, 0.7))
    check('a changed rect re-kicks', vp.kick() is True)
    _vp_wait(vp)
    # rect None: the pre-R253 frames, bitwise
    vp2 = Viewport(clock=lambda: t['now'])
    vp2.set_scene(sc, st)
    vp2.want(cam, 192, 144)
    vp2.kick()
    _vp_wait(vp2)
    t['now'] += DRAFT_WINDOW + 0.1
    vp2.kick()
    whole = _vp_wait(vp2)
    ref_st2 = st.copy()
    shape_settings(ref_st2, 192, 144)
    ref_st2.resolution_x, ref_st2.resolution_y = 96, 72
    ref_st2._viewport = True
    img2 = R.render(sc, ref_st2)
    ref2 = post.process(img2, ref_st2, frame=getattr(sc, 'frame', 0),
                        cvg=getattr(sc, 'last_cvg', None), coverage=None,
                        seed=ref_st2.seed, target_size=(96, 72),
                        flare_sources=getattr(sc, 'last_flares', None))
    check('rect None reproduces the whole-region frame bitwise (the '
          'pre-R253 road)', whole is not None and _same(whole, ref2)
          and vp2.frame_rect is None)
    check('the whole-region key is byte for byte the pre-R253 key',
          vp2._key(cam, 192, 144) == vp2._key(cam, 192, 144, None)
          and len(vp2._key(cam, 192, 144)) == 3)
    check('the rect\'s pixels equal the whole frame\'s rect (a window '
          'onto the region\'s render)',
          _same(frame[y0:y1, x0:x1], whole[y0:y1, x0:x1]))


def test_viewport_camera_frame_mode():
    """The frame_only rect parks a frame of the rect's pixel size with
    vp.frame_rect set, equal to the full viewport frame's rect before
    post and to post.process over the cropped frame; the black guard
    measures the rect only (a rect change on a GPU-marked frame does not
    count as a jump)."""
    from ..core.scene import Camera
    from ..preview import Viewport, shape_settings

    st = base_settings(96, 72, shadows=False)
    st.preview_scale = 1
    st.render_device = 'CPU'
    sc = demo_scene(st, with_texture=False)
    shape_settings(st, 96, 72)
    view, proj, _v, _e = R.camera_matrices(sc.camera, 96, 72)
    cam = Camera(matrix_world=np.linalg.inv(view).astype(np.float32),
                 projection=proj.astype(np.float32), type='PERSP')
    vp = Viewport()
    vp.set_scene(sc, st)
    vp.abort = False
    rect = (0.25, 0.25, 0.75, 0.75, True)
    key = vp._key(cam, 96, 72, rect)
    vp._render(None, sc, st, cam, 96, 72, key, vp.version, False, rect)
    frame = vp.frame
    check('camera-frame mode parks a frame of the rect\'s pixel size',
          frame is not None and frame.shape == (36, 48, 4),
          str(None if frame is None else frame.shape))
    check('and records where it goes on the region',
          vp.frame_rect is not None
          and np.allclose(vp.frame_rect, (0.25, 0.25, 0.75, 0.75)),
          str(vp.frame_rect))
    if frame is None:
        return
    # the reference: the bordered render cropped BEFORE post, post at the
    # cropped size
    sc.camera = cam
    ref_st = st.copy()
    ref_st._viewport = True
    ref_st.use_border = True
    ref_st.border_min_x, ref_st.border_min_y = 0.25, 0.25
    ref_st.border_max_x, ref_st.border_max_y = 0.75, 0.75
    img = R.render(sc, ref_st)
    x0, y0, x1, y1 = R.region_pixels(ref_st, 96, 72)
    crop = np.ascontiguousarray(img[y0:y1, x0:x1])
    ref = post.process(crop, ref_st, frame=getattr(sc, 'frame', 0),
                       cvg=None, coverage=None, seed=ref_st.seed,
                       target_size=(x1 - x0, y1 - y0),
                       flare_sources=getattr(sc, 'last_flares', None))
    check('the parked frame is post.process over the cropped render',
          _same(frame, ref))
    whole_st = st.copy()
    whole_st._viewport = True
    whole = R.render(sc, whole_st)
    check('the cropped render is the whole viewport frame\'s rect before '
          'post', _same(crop, whole[y0:y1, x0:x1]))
    # the black guard measures the rect only: GPU-marked frames (the
    # headless fallback) with a changing rect never read as a jump
    stg = st.copy()
    stg.render_device = 'GPU'
    vpg = Viewport()
    vpg.set_scene(sc, stg)
    vpg.abort = False
    r1 = (0.25, 0.25, 0.75, 0.75, False)
    r2 = (0.05, 0.05, 0.3, 0.3, False)
    vpg._render(None, sc, stg, cam, 96, 72, vpg._key(cam, 96, 72, r1),
                vpg.version, False, r1)
    vpg._render(None, sc, stg, cam, 96, 72, vpg._key(cam, 96, 72, r2),
                vpg.version, False, r2)
    vpg._render(None, sc, stg, cam, 96, 72, vpg._key(cam, 96, 72, rect),
                vpg.version, False, rect)
    check('a rect change on GPU-marked frames never trips the black guard',
          vpg.guard_count == 0, f'guard={vpg.guard_count}')
    check('the guard\'s black fraction is the rect\'s, not the zero '
          'outside\'s', vpg._black_prev is not None and vpg._black_prev < 0.5,
          str(vpg._black_prev))


# ---------------------------------------------------------------- engine


def test_engine_f12_border_delivery():
    """fakeblender: bscene.render carries use_border / border_* and the
    captured begin_result size is (bw, bh) with the delivered buffer
    equal to the full delivery's rect bitwise; use_border False delivers
    exactly today's buffer; extra passes come at the cropped size."""
    from . import fakeblender as FB
    props, engine = FB.install()
    full, fpasses, cap_f = FB.run_render(props, engine, pass_depth=True)
    check('a whole frame delivers at Blender\'s size (today\'s road)',
          full is not None and cap_f.get('size') == (120, 90)
          and full.shape == (90, 120, 4))
    if full is None:
        return

    def rig(eng, depsgraph, bscene):
        bscene.render.use_border = True
        bscene.render.border_min_x = 0.25
        bscene.render.border_min_y = 0.2
        bscene.render.border_max_x = 0.8
        bscene.render.border_max_y = 0.9
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        part, ppasses, cap_p = FB.run_render(props, engine, pass_depth=True,
                                             rig=rig)
    rect = engine._border_rect(
        type('S', (), {'render': type('R', (), dict(
            use_border=True, border_min_x=0.25, border_min_y=0.2,
            border_max_x=0.8, border_max_y=0.9))()})(), 120, 90)
    check('_border_rect truncates both edges like the pipeline',
          rect == (30, 18, 96, 81), str(rect))
    x0, y0, x1, y1 = rect
    check('the header line names the region',
          'region x30..96 y18..81 of 120x90' in buf.getvalue())
    check('begin_result is asked for the border size',
          cap_p.get('size') == (x1 - x0, y1 - y0), str(cap_p.get('size')))
    check('the delivered buffer is the full delivery\'s rect bitwise',
          part is not None and part.shape == (y1 - y0, x1 - x0, 4)
          and _same(part, full[y0:y1, x0:x1]),
          str(None if part is None else part.shape))
    dp = ppasses.get('Depth')
    df = fpasses.get('Depth')
    check('the Depth pass is delivered at the cropped size, the full '
          'pass\'s rect', dp is not None and df is not None
          and dp.shape == (y1 - y0, x1 - x0, 1)
          and _same(dp, df[y0:y1, x0:x1]))
    # the settings road: the five fields derive from scene.render
    depsgraph, bscene, hs, _m = FB.build_scene(props)
    st = engine._settings_from_scene(bscene, 120, 90)
    check('_settings_from_scene: no border by default',
          st.use_border is False)
    bscene.render.use_border = True
    bscene.render.border_min_x, bscene.render.border_min_y = 0.1, 0.2
    bscene.render.border_max_x, bscene.render.border_max_y = 0.7, 0.8
    st = engine._settings_from_scene(bscene, 120, 90)
    check('_settings_from_scene copies scene.render\'s border',
          st.use_border is True and (st.border_min_x, st.border_min_y,
                                     st.border_max_x, st.border_max_y)
          == (0.1, 0.2, 0.7, 0.8))
    stp = engine._settings_from_scene(bscene, 64, 64, preview=True)
    check('preview thumbnails ignore the border', stp.use_border is False)
    # a border + Pixel Scale: the render rect (at render size) is a
    # superset of the output crop
    st2 = engine._settings_from_scene(bscene, 120, 90)
    st2.output_scale = 'X2'
    n = engine.SCALE_FACTOR.get('X2', 2)
    rr = R.region_pixels(st2, 120 // n, 90 // n)
    ob = engine._border_rect(bscene, 120, 90)
    check('under Pixel Scale the render rect (x scale) covers the output '
          'crop', rr is not None and ob is not None
          and rr[0] * n <= ob[0] and rr[1] * n <= ob[1]
          and rr[2] * n >= ob[2] and rr[3] * n >= ob[3],
          f'render {rr} x{n}, output {ob}')
    # the delivery guard: a mismatched rect falls back to Blender's size
    eng = engine.HalcyonRenderEngine.__new__(engine.HalcyonRenderEngine)
    eng.size_x, eng.size_y = 120, 90
    got = {}

    class _Rect:
        def __init__(self, n):
            self.n = n

        def __len__(self):
            return self.n

        def foreach_set(self, flat):
            got['flat'] = np.asarray(flat, np.float32).copy()
    res = FB.FakeResult()
    pas = FB.FakePass('Combined', 4)
    pas.rect = _Rect(120 * 90 * 4)
    res.layers[0].passes['Combined'] = pas
    eng.begin_result = lambda x, y, w, h: got.update(size=(w, h)) or res
    eng.end_result = lambda r: None
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        eng._deliver(np.ones((63, 66, 4), f32), bscene, None, size=(66, 63))
    check('the len(rect) guard fits the picture to Blender\'s allocation '
          'and says so', got.get('size') == (66, 63)
          and got['flat'].size == 120 * 90 * 4
          and 'render region: Blender allocated' in buf.getvalue())


def test_settings_plumbing():
    """The five border fields are in properties._DERIVED_FIELDS (no
    generated property), to_settings leaves them at defaults, shape_
    settings forces use_border False, PRESERVED holds all six keys,
    apply_preset leaves them alone, the hold fingerprint changes with the
    border, the Performance panel draws the toggle, the tooltip is long."""
    from . import fakeblender as FB
    from ..presets.library import PRESERVED, apply_preset, PRESETS
    from ..preview import shape_settings
    props, engine = FB.install()
    five = {'use_border', 'border_min_x', 'border_min_y', 'border_max_x',
            'border_max_y'}
    check('the five border fields are derived (Blender\'s own panel is '
          'their UI)', five <= props._DERIVED_FIELDS)
    ann = getattr(props.HalcyonSettings, '__annotations__', {})
    check('no generated property for them',
          not (five & set(ann)), str(five & set(ann)))
    check('viewport_camera_frame IS a generated property with a long tooltip',
          'viewport_camera_frame' in ann
          and len(str(ann['viewport_camera_frame'].kw.get('description',
                                                          ''))) >= 40)
    hs = FB.live(props.HalcyonSettings)
    st = hs.to_settings()
    check('to_settings leaves the border at the no-op defaults',
          st.use_border is False and st.border_max_x == 1.0
          and st.border_max_y == 1.0 and st.border_min_x == 0.0)
    st.use_border = True
    st.border_min_x = 0.3
    shape_settings(st, 100, 100)
    check('shape_settings forces the viewport settings\' border off',
          st.use_border is False and st.border_min_x == 0.0)
    check('PRESERVED holds all six keys',
          (five | {'viewport_camera_frame'}) <= PRESERVED)
    st2 = RenderSettings()
    st2.use_border = True
    st2.border_min_x, st2.border_max_y = 0.3, 0.7
    st2.viewport_camera_frame = True
    apply_preset(st2, next(iter(PRESETS)))
    check('apply_preset leaves the region and the toggle alone',
          st2.use_border is True and st2.border_min_x == 0.3
          and st2.border_max_y == 0.7 and st2.viewport_camera_frame is True)
    check('no preset names the new keys',
          not any(k in p['settings'] for p in PRESETS.values()
                  for k in five | {'viewport_camera_frame'}))
    a = RenderSettings()
    b = RenderSettings()
    b.use_border = True
    b.border_min_x = 0.2
    check('the hold fingerprint changes with the border',
          engine._hold_fingerprint(a) != engine._hold_fingerprint(b))
    # the panel draws the toggle; the ui module reads the border
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, 'ui.py'), encoding='utf8') as fh:
        ui_src = fh.read()
    check('the Performance panel draws Camera Frame Only',
          "col.prop(hs, 'viewport_camera_frame')" in ui_src)
    check('the Output panel names the render region',
          'Render Region:' in ui_src and '_border_rect' in ui_src)
    # a RenderSettings copy carries the fields (the per-eye / per-pass
    # roads rely on it)
    c = b.copy()
    check('RenderSettings.copy() carries the border',
          c.use_border is True and c.border_min_x == 0.2)
    check('as_dict / apply round-trip the six fields',
          RenderSettings().apply(b.as_dict()).border_min_x == 0.2)


def test_docs_fragment_or_changelog():
    """The round's docs: the docs-dev fragment (pre-integration) or the
    CHANGELOG head (post-integration) names the render region and the
    PANO refusal; the version stamps stay in step (the stamp test keeps
    them)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    frag = os.path.join(root, 'docs-dev', 'r253', 'camera-box.md')
    text = ''
    if os.path.exists(frag):
        with open(frag, encoding='utf8') as fh:
            text = fh.read()
        check('the docs-dev fragment carries the three sections',
              '## CHANGELOG' in text and '## README' in text
              and '## Blender-only' in text)
    else:
        with open(os.path.join(root, 'CHANGELOG.md'), encoding='utf8') as fh:
            text = fh.read()
    low = text.lower()
    check('the docs name the render region, Camera Frame Only and the '
          'panorama refusal',
          'render region' in low and 'camera frame only' in low
          and 'panorama' in low)
    check('the docs state the region-local tolerance of the non-local '
          'stages', 'kernel radius' in low or 'region-local' in low)


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
    print('all R253 render region checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
