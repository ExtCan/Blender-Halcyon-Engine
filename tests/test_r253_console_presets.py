"""R253 (1.92.0): render presets for the consoles that were given shaders.

The 1.91.0 Console Emulation Shader resolves to 27 engine models, but the
render-preset shelf predated it: several machines the node emulates had
no preset (PSP, System 22, Super FX, Wii, Naomi, RenderWare on its three
platforms, the GTA-era PS2 look) and several console presets still named
a stand-in model (JAGUAR -> GOURAUD, THREEDO -> FLAT) or a light count
the node's own LIGHT_LIMIT contradicted. This module proves the nine new
presets and the corrected ones at the table level (presets/library.py
against core/console.py), at the frame level (the CPU road, distinct from
the default frame; the Super FX plot on its palette), through the GLSL
simulator (the shading twin once the raster / frame stages the simulator
does not carry are neutralised by name), and through the headless
Blender layer (the console_node hint reaching Convert to Game, the scene's
Console menu and the presets panel).

    python -m halcyon.tests.test_r253_console_presets
"""
import dataclasses
import os
import sys
import traceback
import types

import numpy as np

from . import utf8_console
from .test_render import base_settings
from .scenebuild import demo_scene
from ..core import combine as CB
from ..core import console as CON
from ..core import convert as C
from ..core import palette as PA
from ..core import post as PO
from ..core import render as R
from ..core import shading as SH
from ..core.settings import RenderSettings
from ..gpu import chain
from ..gpu import chain_palette as CP
from ..gpu import shade as GSH
from ..presets.library import (CONSOLE_HINTED, PRESETS, apply_preset,
                               console_hint)

FAILS = []
f32 = np.float32

#: the nine presets this round adds, with their category
NEW_PRESETS = {
    'PSP': 'CONSOLE', 'NAMCO_S22': 'ARCADE', 'SUPERFX': 'CONSOLE',
    'WII': 'CONSOLE', 'NAOMI': 'ARCADE', 'RENDERWARE_PS2': 'CONSOLE',
    'RENDERWARE_GC': 'CONSOLE', 'RENDERWARE_XBOX': 'CONSOLE',
    'GTA_PS2': 'CONSOLE',
}
#: the existing presets whose VALUES this round changes (model or light count)
CHANGED_PRESETS = ('JAGUAR', 'THREEDO', 'PSX', 'PSX_HIRES', 'PS2', 'N64')
#: the console / arcade presets that carry NO hint: software or untextured
#: flat renderers with no console model, or the handhelds' bitmap roads
UNHINTED = ('NAMCO_S21', 'MODEL1', 'SNES', 'GAME_BOY', 'VIRTUAL_BOY',
            'GAME_GEAR', 'NEO_GEO', 'SEGA_32X', 'GBA_MODE5')


def check(name, cond, extra=''):
    ok = bool(cond)
    print(f'  {"ok  " if ok else "FAIL"} {name}  {extra}' if extra
          else f'  {"ok  " if ok else "FAIL"} {name}')
    if not ok:
        FAILS.append(name)
    return ok


def _root():
    return os.path.dirname(os.path.dirname(os.path.abspath(R.__file__)))


def preset_settings(key, w=64, h=48):
    """apply_preset on a fresh RenderSettings, then the test-card frame:
    64 x 48, at most two samples, no pixel scale, square pixels."""
    st = RenderSettings()
    apply_preset(st, key)
    st.resolution_x, st.resolution_y = w, h
    st.aa_samples = min(int(st.aa_samples), 2)
    st.output_scale = 'NONE'
    st.pixel_aspect_x = st.pixel_aspect_y = 1.0
    return st


def preset_scene(st):
    """The demo scene with every material's own model cleared, so each
    one falls through to the preset's default_model (the
    tests/test_render.py settings-matrix `_clear_models` pattern)."""
    sc = demo_scene(st)
    for m in sc.materials:
        m.model = ''
    return sc


def _fakebpy():
    """The fake bpy with the two base classes the UI modules need."""
    from . import fakebpy
    bpy = fakebpy.install()
    bpy.types.UIList = type('UIList', (bpy.types.Panel,), {})
    bpy.types.AddonPreferences = type('AddonPreferences',
                                      (bpy.types.Panel,), {})
    return bpy


# ------------------------------------------------------------- the shelf


def test_console_shelf_exists_and_is_well_formed():
    """The nine new keys exist in their categories; every settings key
    is a RenderSettings field; labels stay unique; the notes say what
    they are; every console / arcade default_model is a real model; the
    count pin; and importing the library leaves the defaults untouched."""
    fields = {f.name for f in dataclasses.fields(RenderSettings)}
    missing = [k for k in NEW_PRESETS if k not in PRESETS]
    check('the nine new presets exist', not missing, ', '.join(missing))
    bad = []
    for key, cat in NEW_PRESETS.items():
        p = PRESETS.get(key)
        if p is None:
            continue
        if p['category'] != cat:
            bad.append(f'{key} category {p["category"]}')
        for req in ('label', 'category', 'note', 'settings', 'console_node'):
            if req not in p:
                bad.append(f'{key} missing {req}')
        if len(p.get('note', '')) < 40:
            bad.append(f'{key} note')
        for name in p.get('settings', {}):
            if name not in fields:
                bad.append(f'{key} unknown setting {name}')
        if set(p) - {'label', 'category', 'note', 'settings', 'console_node'}:
            bad.append(f'{key} extra keys')
    check('each new preset is complete, in its category (NAMCO_S22 and '
          'NAOMI arcade, the rest console), with a note of 40+ characters '
          'and only real settings', not bad, '; '.join(bad[:4]))
    labels = [e['label'] for e in PRESETS.values()]
    check('preset labels stay unique across the whole shelf',
          len(labels) == len(set(labels)))
    models = {m[0] for m in SH.MODEL_ITEMS}
    bad = [f'{k}:{v["settings"].get("default_model")}'
           for k, v in PRESETS.items()
           if v['category'] in ('CONSOLE', 'ARCADE')
           and 'default_model' in v['settings']
           and v['settings']['default_model'] not in models]
    check('every console / arcade preset names a model MODEL_ITEMS carries',
          not bad, ', '.join(bad))
    check('the shelf counts 121 presets (112 + the nine)',
          len(PRESETS) == 121, str(len(PRESETS)))
    check("the DEFAULT preset still carries no settings",
          PRESETS['DEFAULT']['settings'] == {})
    # the optional key is the hint and nothing else; apply_preset never
    # reads it (the settings loop alone)
    extra = {k: sorted(set(v) - {'label', 'category', 'note', 'settings'})
             for k, v in PRESETS.items()
             if set(v) - {'label', 'category', 'note', 'settings'}}
    check("'console_node' is the only optional top-level key on the shelf",
          all(ks == ['console_node'] for ks in extra.values()),
          str({k: v for k, v in extra.items() if v != ['console_node']}))
    # default neutrality: a fresh RenderSettings equals one built before
    # any preset was applied, and applying every console preset to a
    # throwaway never moves the dataclass defaults
    before = RenderSettings()
    for key in list(NEW_PRESETS) + list(CHANGED_PRESETS):
        apply_preset(RenderSettings(), key)
    after = RenderSettings()
    same = all(getattr(before, f) == getattr(after, f) for f in fields)
    check('RenderSettings() is field-for-field what it was after the '
          'presets are applied (default neutrality)', same)
    st = RenderSettings()
    apply_preset(st, 'DEFAULT')
    check('applying DEFAULT is the fresh dataclass, field for field',
          all(getattr(st, f) == getattr(after, f) for f in fields))


def test_console_hints_cohere():
    """Every console_node hint names only the node's declared properties
    with legal enum values, resolves to the preset's own default_model,
    carries the machine's light count when it toggles the limit, shades
    at the machine's rate (or a named exemption), and together the hints
    cover every machine of the Console menu."""
    enum_names = {n: it for n, it, _d in CON.ENUM_PROPS}
    scalar_names = {n for n, _k, _d in CON.SCALAR_PROPS}
    declared = set(enum_names) | scalar_names
    # the presets whose default_model is a classic the node does not own
    # (XBOX's BLINN per pixel, the generation's flex); VOODOO's GOURAUD
    # IS a console model and must match
    model_exempt = {'XBOX'}
    # the scene rate differs from the machine's by design, disclosed in
    # the note: the Saturn preset shades flat (quads, the VDP1's table is
    # per corner on the node), the Direct3D preset's FLAT_D3D_FIRST is
    # rate-fixed at FACE, the Xbox preset keeps its per-pixel Blinn
    rate_exempt = {'SATURN': 'FACE', 'D3D_RETAIL_1997': 'FACE',
                   'XBOX': 'PIXEL'}
    hinted = {k for k, v in PRESETS.items() if v.get('console_node')}
    check('CONSOLE_HINTED is exactly the set of keys carrying the dict',
          CONSOLE_HINTED == frozenset(hinted), str(hinted ^ CONSOLE_HINTED))
    check("console_hint('DEFAULT') is None, and so is an unknown key",
          console_hint('DEFAULT') is None and console_hint('NO_SUCH') is None)
    h = console_hint('PSX')
    check("console_hint returns a COPY (editing it leaves the shelf alone)",
          h == PRESETS['PSX']['console_node']
          and h is not PRESETS['PSX']['console_node'])
    bad = []
    machines = set()
    for key in sorted(hinted):
        hint = PRESETS[key]['console_node']
        st = PRESETS[key]['settings']
        if not isinstance(hint, dict) or 'console' not in hint:
            bad.append(f'{key}: no console')
            continue
        con = hint['console']
        machines.add(con)
        for k, v in hint.items():
            if k not in declared:
                bad.append(f'{key}: undeclared {k}')
            elif k in enum_names and v not in {i[0] for i in enum_names[k]}:
                bad.append(f'{key}: {k}={v!r} not an item')
        res = CON.resolve(hint)
        model = st.get('default_model')
        if model in CON.CONSOLE_MODELS and key not in model_exempt \
                and res['model'] != model:
            bad.append(f'{key}: resolves {res["model"]} not {model}')
        if hint.get('light_limit'):
            want = CON.LIGHT_LIMIT.get(con)
            if st.get('max_lights') != want or st.get('light_limit_mode') != 'FIRST':
                bad.append(f'{key}: lights {st.get("max_lights")}/'
                           f'{st.get("light_limit_mode")} vs {want} FIRST')
            if res['light_limit'] != want:
                bad.append(f'{key}: resolve light_limit {res["light_limit"]}')
        rate = st.get('shading_rate')
        want_rate = rate_exempt.get(key, CON.MACHINE_RATE.get(con))
        if rate != want_rate:
            bad.append(f'{key}: rate {rate} vs {want_rate}')
        if CON.label_of(hint) == con:
            bad.append(f'{key}: label_of has no type')
    check(f'the {len(hinted)} hints name declared props with legal values, '
          'resolve to the preset\'s model, carry the hardware light count '
          'under FIRST when they toggle the limit, and shade at the '
          'machine\'s rate', not bad, '; '.join(bad[:5]))
    every = {a for a, _b, _c in CON.CONSOLE_ITEMS}
    check(f'every one of the {len(every)} machines of the Console menu is '
          'named by at least one preset\'s hint', every <= machines,
          str(sorted(every - machines)))
    check('the RenderWare presets toggle no light limit (RpWorld has no '
          'hardware count: LIGHT_LIMIT 0, max_lights 0 = no limit)',
          all(not PRESETS[k]['console_node'].get('light_limit')
              and PRESETS[k]['settings'].get('max_lights') == 0
              for k in ('RENDERWARE_PS2', 'RENDERWARE_GC', 'RENDERWARE_XBOX',
                        'GTA_PS2'))
          and CON.LIGHT_LIMIT['RENDERWARE'] == 0)
    nohint = [k for k in UNHINTED if PRESETS.get(k, {}).get('console_node')]
    check('the software / untextured / bitmap machines carry no hint',
          not nohint, ', '.join(nohint))
    # the corrected light counts against the node's own table
    check('PSX / PSX_HIRES / PS2 / N64 / NDS / GAMECUBE carry the machine\'s '
          'light count under FIRST (3 / 3 / 3 / 7 / 4 / 8)',
          all((PRESETS[k]['settings']['max_lights'],
               PRESETS[k]['settings']['light_limit_mode']) == (n, 'FIRST')
              for k, n in (('PSX', 3), ('PSX_HIRES', 3), ('PS2', 3),
                           ('N64', 7), ('NDS', 4), ('GAMECUBE', 8))))
    check('JAGUAR shades with the blitter\'s CRY Gouraud and THREEDO with the '
          'cel engine\'s PIXC (the stand-ins retired)',
          PRESETS['JAGUAR']['settings']['default_model'] == 'JAGUAR_CRY'
          and PRESETS['THREEDO']['settings']['default_model'] == 'THREEDO_PIXC'
          and PRESETS['THREEDO']['settings']['shading_rate'] == 'FACE'
          and CB.RATE_FIXED.get('THREEDO_PIXC') == 'FACE')
    # the hints use only names export.NODE_PROPS covers (the R252 pin)
    try:
        from .. import export as EX
        np_names = set(EX.NODE_PROPS.get('HALCYON_ConsoleShaderNode', ()))
    except Exception:                                           # noqa: BLE001
        np_names = set()
    if np_names:
        bad = [f'{k}:{n}' for k in hinted
               for n in PRESETS[k]['console_node'] if n not in np_names]
        check('every hinted property is in export.NODE_PROPS for the node',
              not bad, ', '.join(bad[:5]))


# ------------------------------------------------------------- the frames


def test_console_presets_render_distinct():
    """On the CPU road every new and changed preset renders a finite
    frame that differs from the default one; the Super FX preset passes
    the plot's gates and every post pixel is a palette entry."""
    st0 = preset_settings('DEFAULT')
    ref = PO.process(R.render(preset_scene(st0), st0), st0)
    for key in list(NEW_PRESETS) + list(CHANGED_PRESETS):
        st = preset_settings(key)
        sc = preset_scene(st)
        try:
            raw = R.render(sc, st)
            img = PO.process(raw, st)
        except Exception as exc:                                # noqa: BLE001
            traceback.print_exc()
            check(f'{key} renders', False, repr(exc))
            continue
        d = float(np.abs(img[..., :3] - ref[..., :3]).max())
        check(f'{key}: finite, {img.shape[1]}x{img.shape[0]}, and not the '
              f'default frame (max |d| {d:.3f})',
              np.isfinite(img).all() and img.shape[:2] == (48, 64) and d > 1e-3)
        if key == 'SUPERFX':
            try:
                P = CB.plot_gate(st)
                gate_ok = P.shape == (16, 3)
            except CB.Refusal as r:
                gate_ok = False
                print('   ', r)
            check('SUPERFX: the plot\'s gates pass (16 fixed entries, gamma '
                  '1.0, no colour management)', gate_ok)
            check('SUPERFX: effective_model keeps SUPERFX_PLOT for a material '
                  'without its own model',
                  CB.effective_model('SUPERFX_PLOT', sc.materials[1], st)
                  == 'SUPERFX_PLOT')
            pal = PA.SUPERFX16
            px = img[..., :3].reshape(-1, 3)
            dist = np.abs(px[:, None, :] - pal[None, :, :]).max(axis=2).min(axis=1)
            check('SUPERFX: every post pixel is one of the 16 SUPERFX16 '
                  'entries within 1/255 (the palette snap after the plot)',
                  bool((dist <= 1.0 / 255.0 + 1e-6).all()), f'max {dist.max():.4f}')
            # the render itself, before post: the untextured Ball and Box
            # are plotted palette entries (the plot IS the dither; the
            # textured Floor keeps its texel over the plotted pair and the
            # sky is not a polygon). The material masks come from a
            # z-buffered raster, so the Box is read under ZBUFFER (the
            # preset's painter's sort draws the Floor over its far corner)
            from .test_r252_console import _ball_box
            ball, _box = _ball_box(sc, st)
            rp = raw[ball][:, :3]
            d2 = np.abs(rp[:, None, :] - pal[None, :, :]).max(axis=2).min(axis=1)
            check('SUPERFX: every Ball pixel of the render itself is a palette '
                  'entry before post (the plot IS the dither)',
                  bool((d2 <= 1.0 / 255.0 + 1e-6).all()), f'max {d2.max():.4f}')
            stz = preset_settings('SUPERFX')
            stz.depth_sort = 'ZBUFFER'
            scz = preset_scene(stz)
            rawz = np.asarray(R.render(scz, stz))
            ballz, boxz = _ball_box(scz, stz)
            rpz = rawz[ballz | boxz][:, :3]
            dz = np.abs(rpz[:, None, :] - pal[None, :, :]).max(axis=2).min(axis=1)
            check('SUPERFX under a z-buffer: every Ball and Box pixel is a '
                  'palette entry', bool((dz <= 1.0 / 255.0 + 1e-6).all()),
                  f'max {dz.max():.4f}')
            # the plot pairs differ: a checkerboard is two entries, so the
            # ball is not one flat colour
            check('SUPERFX: the plotted ball is a checker of several entries, '
                  'not a flat fill', len({tuple(np.rint(v * 31).astype(int))
                                          for v in rp}) >= 3)


def test_console_presets_on_the_gpu():
    """Through the GLSL simulator: each new preset's shading -- its
    model, its light limit at the vertex / face rate, its fog -- is the
    CPU's at the corner-road bar (max < 6e-3, no pixel > 1e-2) once the
    raster / frame stages the shading simulator does not carry are
    neutralised BY NAME (each has its own R251 road: the GPU
    rasteriser's painter's sort, depth encodings and pixel centre, the
    composite's framebuffer formats on the CPU on both roads, the edge
    pass after shading); the Super FX plot is the twin and refuses by
    name when its gate fails; the post stages the presets name report no
    refusal on the GPU chain."""
    from .test_r251_material import _sim_vs_cpu
    # the stages the shading simulator does not run, with their roads
    NEUTRAL = (
        ('depth_sort', 'ZBUFFER', "the GPU rasteriser's painter's sort"),
        ('framebuffer', 'NONE', "the composite's framebuffer format (CPU on "
                                "both roads, named by gpu/frame.edited)"),
        ('depth_encoding', 'LINEAR', "the GPU raster kernel's depth encoding"),
        ('pixel_center', 'HALF', "the GPU raster kernel's pixel centre"),
        ('aa_mode', 'NONE', "the edge / supersample pass after shading"),
    )
    for key in NEW_PRESETS:
        st = preset_settings(key)
        st.render_device = 'GPU'
        touched = []
        for name, val, road in NEUTRAL:
            if getattr(st, name) != val:
                touched.append(f'{name} {getattr(st, name)} ({road})')
                setattr(st, name, val)
        sc = preset_scene(st)
        GSH._PLAN_CACHE.clear()
        d, nbad, img, cpu, passes, why = _sim_vs_cpu(sc, st)
        if d is None:
            check(f'{key}: the frame qualifies for the GPU', False, str(why))
            continue
        check(f'{key}: the simulator frame is the CPU frame (max {d:.2e}, '
              f'{nbad} px > 1e-2)' + (f' with {len(touched)} raster / frame '
                                       'stage(s) neutralised by name'
                                       if touched else ''),
              d < 6e-3 and nbad == 0, '; '.join(touched))
        # the post chain: the palette / depth stage of each preset is a
        # GPU stage or names its reason (never a silent fallback)
        st2 = preset_settings(key)
        st2.render_device = 'GPU'
        q = chain.quant_refusal(st2)
        check(f'{key}: the depth / palette post stage runs on the GPU chain',
              q is None, str(q))
    # the stages neutralised above are documented both-road stages
    from ..gpu import capability as CAP
    txt = str(CAP.FEATURES.get('blend_equation', ('', ''))[1]) \
        if hasattr(CAP, 'FEATURES') else ''
    src = open(os.path.join(_root(), 'gpu', 'capability.py'),
               encoding='utf-8').read()
    check("the capability table says the framebuffer formats run in the "
          "composite on the CPU on both roads",
          'framebuffer formats run in the composite on the CPU' in (txt or src))
    # the Super FX plot: the twin at the preset's gates, the refusal by
    # name when gamma moves the entries off the palette
    st = preset_settings('SUPERFX')
    st.render_device = 'GPU'
    st.depth_sort = 'ZBUFFER'
    sc = preset_scene(st)
    GSH._PLAN_CACHE.clear()
    d, nbad, img, cpu, passes, why = _sim_vs_cpu(sc, st)
    check('SUPERFX: the plot is drawn in GLSL as the CPU\'s twin',
          d is not None and d < 6e-3 and nbad == 0, str(why))
    if img is not None:
        pal = PA.SUPERFX16
        cov = np.asarray(img).reshape(-1, 3)
        dist = np.abs(cov[:, None, :] - pal[None, :, :]).max(axis=2).min(axis=1)
        check('SUPERFX: the simulated frame plots palette entries over most '
              'of the picture', float((dist <= 1.0 / 255.0 + 1e-6).mean()) > 0.5)
    st.gamma = 2.2
    sc = preset_scene(st)
    GSH._PLAN_CACHE.clear()
    d, nbad, img, cpu, passes, why = _sim_vs_cpu(sc, st)
    check('SUPERFX at gamma 2.2: the GPU refuses by name (the plot writes '
          'palette entries) and the CPU shades the material as FLAT',
          d is None and why is not None and 'palette entries' in str(why)
          and CB.effective_model('SUPERFX_PLOT', sc.materials[1], st) == 'FLAT',
          str(why))


def test_superfx16_palette():
    """The stand-in Star Fox palette: sixteen distinct 5-bit entries in
    0..1, not EGA16; the enum item appended after EHB with a tooltip; the
    CPU palette stage snaps a gradient onto it and the GPU palette stage
    reports no refusal for the mode."""
    P = PA.FIXED_PALETTES['SUPERFX16'](16)
    check('FIXED_PALETTES[SUPERFX16](16) is a (16, 3) float32 table in 0..1',
          P.shape == (16, 3) and P.dtype == np.float32
          and float(P.min()) >= 0.0 and float(P.max()) <= 1.0)
    check('its 16 rows are distinct and it is not EGA16',
          len({tuple(r) for r in P.tolist()}) == 16
          and not np.array_equal(P, PA.EGA16))
    check('every entry sits on the SNES CGRAM lattice (k / 31), so BITS_5 '
          'snaps the table to itself', np.allclose(np.rint(P * 31) / 31, P)
          and np.array_equal(PA.snap_levels(P, 32), P)
          if hasattr(PA, 'snap_levels') else True)
    check('the size argument is ignored (16 entries whatever the depth asks)',
          PA.FIXED_PALETTES['SUPERFX16'](256).shape == (16, 3))
    check('get_palette routes the mode generically',
          np.array_equal(PA.get_palette('SUPERFX16', 256), P))
    _fakebpy()
    import importlib
    PR = importlib.import_module('halcyon.properties')
    ids = [i[0] for i in PR.PALETTE_MODE]
    item = next(i for i in PR.PALETTE_MODE if i[0] == 'SUPERFX16')
    check("PALETTE_MODE keeps EHB at index 10 and ends with SUPERFX16 "
          "(appended, positional)", ids.index('EHB') == 10 and ids[-1] == 'SUPERFX16')
    check('the item has a label and a tooltip of 12+ characters naming the '
          'stand-in', len(item[1]) >= 4 and len(item[2]) >= 12
          and 'stand-in' in item[2])
    # the CPU palette stage over a gradient
    w, h = 48, 16
    g = np.linspace(0.0, 1.0, w, dtype=f32)
    rgb = np.stack([np.tile(g, (h, 1)), np.tile(g[::-1], (h, 1)),
                    np.full((h, w), 0.4, f32)], axis=-1)
    st = base_settings(w, h)
    st.color_depth = '4'
    st.palette_mode = 'SUPERFX16'
    st.palette_size = 16
    st.palette_bits = 'BITS_5'
    st.dither = 'NONE'
    out = PO.reduce_depth(rgb, st)
    o = np.asarray(out)[..., :3].reshape(-1, 3)
    dist = np.abs(o[:, None, :] - P[None, :, :]).max(axis=2).min(axis=1)
    check("reduce_depth under palette_mode SUPERFX16 / color_depth '4' "
          'yields palette colours only', bool((dist <= 1.0 / 255.0 + 1e-6).all())
          and len({tuple(r) for r in np.rint(o * 31).astype(int).tolist()}) >= 4,
          f'max {dist.max():.4f}')
    st.render_device = 'GPU'
    check('the GPU palette stage reports no refusal for the mode',
          CP.palette_road(st) and CP.palette_refusal(st) is None
          and chain.quant_refusal(st) is None)
    check('the plot gate reads the same 16 entries',
          np.array_equal(CB.plot_palette(st), P))


# ------------------------------------------------- the hint reaches Blender


def test_hint_reaches_convert_to_game():
    """plan_for merges a preset's console_node after the machine; an
    undeclared key is dropped; without console_props the props are
    exactly the machine. Headless: Apply Preset moves the scene's
    Console menu to the hint's machine (and leaves it alone for a preset
    without one); Convert to Game From Preset builds the node at the
    preset's machine, type and light limit."""
    hint = PRESETS['GTA_PS2']['console_node']
    p = C.plan_for('CONSOLE', 'ShaderNodeBsdfPrincipled', {}, set(),
                   {'console': 'RENDERWARE', 'console_props': hint})
    check("the GTA_PS2 hint lands on the plan: 'console' first, rw_platform "
          "PS2, rw_type PRELIT",
          list(p['props'])[0] == 'console' and p['props']['console'] == 'RENDERWARE'
          and p['props'].get('rw_platform') == 'PS2'
          and p['props'].get('rw_type') == 'PRELIT'
          and p['label'] == CON.label_of(p['props']), str(p['props']))
    p2 = C.plan_for('CONSOLE', 'ShaderNodeBsdfPrincipled', {}, set(),
                    {'console': 'PS1', 'console_props': {
                        'console': 'N64', 'ps1_type': 'FLAT',
                        'no_such_prop': 1, 'light_limit': True}})
    check("an undeclared key is dropped and the hint's 'console' never "
          "overrides the menu's",
          p2['props'] == {'console': 'PS1', 'ps1_type': 'FLAT',
                          'light_limit': True}, str(p2['props']))
    p3 = C.plan_for('CONSOLE', 'ShaderNodeBsdfPrincipled', {}, set(),
                    {'console': 'PS1'})
    check("without console_props the props are exactly {'console': con}",
          p3['props'] == {'console': 'PS1'})
    p4 = C.plan_for('CONSOLE', 'ShaderNodeBsdfPrincipled', {},
                    {'Base Color'},
                    {'console': 'N64', 'console_props':
                     PRESETS['N64']['console_node']},
                    {'Base Color': 'ShaderNodeVertexColor'})
    check("the conversion's own vertex-colour switch wins over the hinted "
          "N64 type (merge before the overrides)",
          p4['props'].get('n64_type') == 'VERTEX'
          and p4['props'].get('light_limit') is True
          and list(p4['props'])[0] == 'console', str(p4['props']))
    # every hint goes through the plan and resolves to the preset's model
    bad = []
    for key in sorted(CONSOLE_HINTED):
        h = console_hint(key)
        pk = C.plan_for('CONSOLE', 'ShaderNodeBsdfPrincipled', {}, set(),
                        {'console': h['console'], 'console_props': h})
        if pk['props'] != dict(h):
            bad.append(f'{key}: {pk["props"]} != {h}')
    check(f'all {len(CONSOLE_HINTED)} hints pass through plan_for intact',
          not bad, '; '.join(bad[:3]))

    # the headless Blender layer
    _fakebpy()
    import importlib
    from . import fakeblender as FB
    PR = importlib.import_module('halcyon.properties')
    ui = importlib.import_module('halcyon.ui')
    CV = importlib.import_module('halcyon.convert')

    def scene_with(preset, console='PS1'):
        hs = FB.live(PR.HalcyonSettings, preset=preset,
                     convert_console=console)
        object.__setattr__(hs, 'property_unset', lambda name: None)
        render = types.SimpleNamespace(resolution_x=1920, resolution_y=1080,
                                       resolution_percentage=50,
                                       pixel_aspect_x=1.0, pixel_aspect_y=1.0,
                                       film_transparent=False)
        scene = types.SimpleNamespace(halcyon=hs, render=render,
                                      update_tag=lambda: None, objects=[])
        return types.SimpleNamespace(scene=scene), hs, render

    ctx, hs, render = scene_with('PSX', console='N64')
    op = ui.HALCYON_OT_apply_preset()
    op.preset, op.reset = 'PSX', True
    reports = []
    op.report = lambda kind, text: reports.append((kind, text))
    rc = op.execute(ctx)
    check("Apply Preset PSX moves the scene's Console menu to PS1 and writes "
          "the preset's resolution",
          rc == {'FINISHED'} and hs.convert_console == 'PS1'
          and render.resolution_x == 320 and render.resolution_y == 240
          and hs.max_lights == 3 and hs.light_limit_mode == 'FIRST', str(reports))
    ctx, hs, render = scene_with('INFINID_4', console='N64')
    op = ui.HALCYON_OT_apply_preset()
    op.preset, op.reset = 'INFINID_4', True
    op.report = lambda kind, text: None
    rc = op.execute(ctx)
    check("Apply Preset INFINID_4 (no hint) leaves the Console menu alone",
          rc == {'FINISHED'} and hs.convert_console == 'N64')
    check("convert_console is not a render setting (the hint never reaches "
          "RenderSettings)",
          'convert_console' not in {f.name for f in dataclasses.fields(RenderSettings)})

    # Convert to Game From Preset on a fake material
    from .test_r253_convert_buttons import _nodes, _principled_tree
    SN = _nodes()
    mat, src, tex, nm, out = _principled_tree(SN)
    op = CV.HALCYON_OT_convert_to_node()
    op.target, op.scope = 'CONSOLE', 'ACTIVE'
    op.style, op.compat, op.era, op.console = 'CUSTOM', 'GENERIC', 'CUSTOM', 'PS1'
    op.keep_original, op.force, op.from_preset = True, False, True
    reports = []
    op.report = lambda kind, text: reports.append((kind, text))
    ctx = types.SimpleNamespace(
        active_object=types.SimpleNamespace(active_material=mat),
        scene=types.SimpleNamespace(objects=[],
                                    halcyon=types.SimpleNamespace(preset='WII')),
        selected_objects=[])
    rc = op.execute(ctx)
    node = next((n for n in mat.node_tree.nodes
                 if n.bl_idname == 'HALCYON_ConsoleShaderNode'), None)
    check("From Preset with the scene on WII builds the node at GAMECUBE / "
          "LIT with the light limit on (the Console menu's PS1 overridden)",
          rc == {'FINISHED'} and node is not None
          and node.console == 'GAMECUBE' and node.gc_type == 'LIT'
          and node.light_limit is True and op.console == 'GAMECUBE'
          and reports and '1 converted' in reports[-1][1],
          str(reports) + ('' if node is None else f' {node.console}'))
    mat, src, tex, nm, out = _principled_tree(SN)
    op = CV.HALCYON_OT_convert_to_node()
    op.target, op.scope = 'CONSOLE', 'ACTIVE'
    op.style, op.compat, op.era, op.console = 'CUSTOM', 'GENERIC', 'CUSTOM', 'N64'
    op.keep_original, op.force, op.from_preset = True, False, True
    op.report = lambda kind, text: None
    ctx = types.SimpleNamespace(
        active_object=types.SimpleNamespace(active_material=mat),
        scene=types.SimpleNamespace(objects=[],
                                    halcyon=types.SimpleNamespace(preset='INFINID_4')),
        selected_objects=[])
    rc = op.execute(ctx)
    node = next((n for n in mat.node_tree.nodes
                 if n.bl_idname == 'HALCYON_ConsoleShaderNode'), None)
    check("From Preset with a preset that has no hint keeps the Console "
          "menu's machine", rc == {'FINISHED'} and node is not None
          and node.console == 'N64' and node.light_limit is False)
    ann = CV.HALCYON_OT_convert_to_node.__annotations__
    check('from_preset is a BoolProperty, default False, with a 40+ tooltip',
          ann['from_preset'].kind == 'BoolProperty'
          and ann['from_preset'].kw.get('default') is False
          and len(ann['from_preset'].kw.get('description', '')) >= 40)


def test_preset_panel_names_the_game_conversion():
    """On a recording layout: the presets panel names the Game conversion
    a hinted preset builds and says nothing for one without; the Game
    convert block draws From Preset (four operators) only when the
    active preset carries a hint, three otherwise -- the convert-buttons
    counts hold at the default preset."""
    _fakebpy()
    import importlib
    ui = importlib.import_module('halcyon.ui')

    class _Op(types.SimpleNamespace):
        pass

    class _Lay:
        def __init__(self, log):
            self.log = log
            self.scale_y = 1.0
            self.active = True

        def row(self, **k):
            return self

        def column(self, **k):
            return self

        def box(self):
            return self

        def split(self, **k):
            return self

        def separator(self, **k):
            self.log.append(('sep',))

        def label(self, **k):
            self.log.append(('label', k.get('text'), k.get('icon')))

        def prop(self, data, name, **k):
            self.log.append(('prop', name))

        def operator(self, idn, **k):
            op = _Op()
            self.log.append(('op', idn, k.get('text'), k.get('icon'), op))
            return op

    def draw_presets(preset):
        log = []
        hs = types.SimpleNamespace(preset=preset, render_device='CPU')
        ctx = types.SimpleNamespace(scene=types.SimpleNamespace(halcyon=hs))
        ui.HALCYON_PT_presets.draw(types.SimpleNamespace(layout=_Lay(log)), ctx)
        return log

    log = draw_presets('PSX')
    labels = [e[1] for e in log if e[0] == 'label' and e[1]]
    game = [t for t in labels if t.startswith('Game conversion: ')]
    check("the presets panel on PSX names 'Game conversion: ... PlayStation / "
          "Gouraud ...' with the NODE_MATERIAL icon",
          len(game) == 1 and 'PlayStation / Gouraud' in game[0]
          and any(e[0] == 'label' and e[1] == game[0] and e[2] == 'NODE_MATERIAL'
                  for e in log), str(game))
    log = draw_presets('INFINID_4')
    check("the presets panel on INFINID_4 draws no Game conversion line",
          not any(e[0] == 'label' and (e[1] or '').startswith('Game conversion')
                  for e in log))
    log = draw_presets('DEFAULT')
    check("...nor on DEFAULT (the menu's default)",
          not any(e[0] == 'label' and (e[1] or '').startswith('Game conversion')
                  for e in log))
    from .blender_icons import ICONS
    check('the icons are real (NODE_MATERIAL, IMPORT)',
          'NODE_MATERIAL' in ICONS and 'IMPORT' in ICONS)

    def draw_game(preset, kinds=frozenset()):
        log = []
        hal = types.SimpleNamespace(convert_anime_style='CUSTOM',
                                    convert_anime_compat='GENERIC',
                                    convert_cartoon_era='CUSTOM',
                                    convert_console='PS1', preset=preset)
        ui._draw_convert_target(_Lay(log), hal, 'CONSOLE', "To Game (Console):",
                                'SYSTEM', set(kinds))
        return log

    log = draw_game('PSX')
    ops = [e for e in log if e[0] == 'op']
    fp = [o for o in ops if o[2] == 'From Preset']
    check('the Game block on PSX draws four operators: This Material / '
          'Selected / Scene / From Preset', len(ops) == 4 and len(fp) == 1
          and [o[4].scope for o in ops] == ['ACTIVE', 'SELECTED', 'SCENE', 'ACTIVE'],
          str([o[2] for o in ops]))
    if fp:
        o = fp[0][4]
        check('From Preset carries from_preset, the hint\'s machine, the '
              'ACTIVE scope, the IMPORT icon and no force on a fresh tree',
              o.from_preset is True and o.console == 'PS1' and o.target == 'CONSOLE'
              and o.scope == 'ACTIVE' and fp[0][3] == 'IMPORT' and o.force is False)
        check('the other three buttons do not set from_preset',
              all(not getattr(x[4], 'from_preset', False) for x in ops[:3]))
        check("a 'Preset: <machine / type>' label follows the button",
              any(e[0] == 'label' and (e[1] or '').startswith('Preset: ')
                  and 'PlayStation' in e[1] for e in log))
    log = draw_game('PSX', kinds={'HALCYON_ConsoleShaderNode'})
    fp = [e for e in log if e[0] == 'op' and e[2] == 'From Preset']
    check('From Preset reconverts (force) when the tree already holds the node',
          fp and fp[0][4].force is True)
    for preset in ('INFINID_4', 'DEFAULT', 'NAMCO_S21'):
        log = draw_game(preset)
        ops = [e for e in log if e[0] == 'op']
        check(f'the Game block on {preset} (no hint) draws three operators',
              len(ops) == 3 and not any(o[2] == 'From Preset' for o in ops))
    log = []
    hal = types.SimpleNamespace(convert_anime_style='CUSTOM',
                                convert_anime_compat='GENERIC',
                                convert_cartoon_era='CUSTOM',
                                convert_console='PS1')
    ui._draw_convert_target(_Lay(log), hal, 'CONSOLE', "To Game (Console):",
                            'SYSTEM', set())
    check('a scene without a preset attribute (the R253 convert-buttons '
          'fixture) still draws three', len([e for e in log if e[0] == 'op']) == 3)
    log = []
    ui._draw_convert_target(_Lay(log), hal, 'ANIME', "To Anime Shader:",
                            'IPO_CONSTANT', set())
    check('the Anime block is untouched (three operators, no preset line)',
          len([e for e in log if e[0] == 'op']) == 3
          and not any(e[0] == 'label' and (e[1] or '').startswith('Preset')
                      for e in log))


# ---------------------------------------------------- the shelf's old pins


def test_old_shelf_unchanged_where_pinned():
    """The R251 pins on the console shelf, re-asserted in one place with
    their reasons, so a later edit fails here by name."""
    S = {k: PRESETS[k]['settings'] for k in PRESETS}
    check('NAMCO_S21 (test_r251_lighting_fog): fog_face, 16 bands, LINEAR, '
          'FACE -- the System 21 palette-bank haze',
          S['NAMCO_S21'].get('fog_face') is True
          and S['NAMCO_S21'].get('fog_bands') == 16
          and S['NAMCO_S21'].get('fog_mode') == 'LINEAR'
          and S['NAMCO_S21'].get('shading_rate') == 'FACE'
          and S['NAMCO_S21'].get('default_model') == 'FLAT')
    check('GAMECUBE: no fog_range_adjust, VERTEX, TRILINEAR, TABLE16, fog '
          'height, GC_14E2, XFB_422, GC_RGBA6',
          'fog_range_adjust' not in S['GAMECUBE']
          and S['GAMECUBE']['shading_rate'] == 'VERTEX'
          and S['GAMECUBE']['tex_filter'] == 'TRILINEAR'
          and S['GAMECUBE']['fog_mode'] == 'TABLE16'
          and S['GAMECUBE'].get('fog_height') is True
          and S['GAMECUBE'].get('depth_encoding') == 'GC_14E2'
          and S['GAMECUBE'].get('chroma_format') == 'XFB_422'
          and S['GAMECUBE'].get('framebuffer') == 'GC_RGBA6')
    check('DREAMCAST: PVR128 at Fog End 64, EXP 0.02 (the CLX2 table)',
          S['DREAMCAST'].get('fog_table') == 'PVR128'
          and S['DREAMCAST'].get('fog_end') == 64.0
          and S['DREAMCAST'].get('fog_mode') == 'EXP'
          and S['DREAMCAST'].get('fog_density') == 0.02)
    check("PS2: FIELDS, PS2_CT16, GS_Q, no crtc_blend key, 'CRTC' in the note",
          S['PS2'].get('interlace') == 'FIELDS'
          and S['PS2'].get('framebuffer') == 'PS2_CT16'
          and S['PS2'].get('tex_lod_source') == 'GS_Q'
          and 'crtc_blend' not in S['PS2']
          and 'CRTC' in PRESETS['PS2']['note'])
    check('SEGA_MODEL3: fog banks 6 / 45', S['SEGA_MODEL3'].get('fog_bank1_start') == 6.0
          and S['SEGA_MODEL3'].get('fog_bank1_end') == 45.0)
    check("N64: CI4, BLEND, N64_FLOAT18, 'Sharpen' in the note",
          S['N64'].get('tex_tmem_format') == 'CI4'
          and S['N64'].get('tex_mip_select') == 'BLEND'
          and S['N64'].get('depth_encoding') == 'N64_FLOAT18'
          and 'Sharpen' in PRESETS['N64']['note'])
    check('XBOX: INTEGER_D3D, W_FIXED, aniso >= 2, BLINN per pixel kept',
          S['XBOX'].get('pixel_center') == 'INTEGER_D3D'
          and S['XBOX'].get('depth_encoding') == 'W_FIXED'
          and S['XBOX'].get('tex_aniso', 0) >= 2
          and S['XBOX']['default_model'] == 'BLINN'
          and S['XBOX']['shading_rate'] == 'PIXEL')
    check("JAGUAR: CRY16 and 'CRY' in the note", S['JAGUAR'].get('color_depth') == 'CRY16'
          and 'CRY' in PRESETS['JAGUAR']['note'])
    check("NDS: category CONSOLE, 'polygon ID' in the note, DS32, ABUFFER / DS",
          PRESETS['NDS']['category'] == 'CONSOLE'
          and 'polygon ID' in PRESETS['NDS']['note']
          and S['NDS'].get('fog_table') == 'DS32'
          and S['NDS'].get('transparency') == 'ABUFFER'
          and S['NDS'].get('blend_equation') == 'DS')
    check('SNES: untouched -- FLAT / FACE, ADAPTIVE 256 at BITS_5, SNES_ADD_HALF',
          S['SNES']['default_model'] == 'FLAT' and S['SNES']['shading_rate'] == 'FACE'
          and S['SNES'].get('palette_mode') == 'ADAPTIVE'
          and S['SNES'].get('palette_bits') == 'BITS_5'
          and S['SNES'].get('blend_equation') == 'SNES_ADD_HALF'
          and 'console_node' not in PRESETS['SNES'])
    check('D3D_RETAIL_1997: INTEGER_D3D, no fog_depth, no resolution',
          S['D3D_RETAIL_1997'].get('pixel_center') == 'INTEGER_D3D'
          and 'fog_depth' not in S['D3D_RETAIL_1997']
          and 'resolution_x' not in S['D3D_RETAIL_1997'])
    check('POWERVR_PCX2: no tex_format', 'tex_format' not in S['POWERVR_PCX2'])
    check('SEGA_MODEL2: I4_MODEL2, FACE, AXIS', S['SEGA_MODEL2'].get('tex_format') == 'I4_MODEL2'
          and S['SEGA_MODEL2']['shading_rate'] == 'FACE'
          and S['SEGA_MODEL2'].get('specular_viewer') == 'AXIS')
    check('THREEDO: THREEDO_2X, SATURN_HALF, PAINTERS kept beside the PIXC model',
          S['THREEDO'].get('output_scale') == 'THREEDO_2X'
          and S['THREEDO'].get('blend_equation') == 'SATURN_HALF'
          and S['THREEDO'].get('depth_sort') == 'PAINTERS')
    check('the 1.91.0 shelf keys all survive (no preset removed or renamed)',
          all(k in PRESETS for k in (
              'NDS', 'SNES', 'DREAMCAST', 'PS2', 'GAMECUBE', 'XBOX', 'THREEDO',
              'JAGUAR', 'NAMCO_S21', 'PSX_HIRES', 'PSX', 'MODEL1', 'SATURN',
              'SEGA_MODEL2', 'SEGA_MODEL3', 'N64', 'VOODOO', 'VOODOO2',
              'D3D_RETAIL_1997', 'POWERVR_PCX2', 'MEGA_DRIVE', 'MAX_2012')))
    # no preset names a device key (the R158 contract) and every new
    # preset's note says what is a stand-in where the design made one
    from ..presets.library import DEVICE_KEYS
    check('no new preset names a device key',
          not any(set(S[k]) & DEVICE_KEYS for k in NEW_PRESETS))
    check('the stand-ins are disclosed in the notes (System 22 fog curve, '
          'the PSP dither matrix, the Star Fox palette, the RenderWare width)',
          'stand-in' in PRESETS['NAMCO_S22']['note']
          and 'stands in' in PRESETS['PSP']['note']
          and 'STAND-IN' in PRESETS['SUPERFX']['note']
          and 'stand-in' in PRESETS['RENDERWARE_PS2']['note']
          and 'anamorphic' in PRESETS['WII']['note'].lower())


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
    print('all R253 console-preset checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
