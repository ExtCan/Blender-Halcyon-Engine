"""File > Import > Legacy .blend: the classic-format append.

Two readers, each doing the half it is good at.

Blender's own append machinery (`bpy.data.libraries.load`) loads the
OBJECTS -- meshes with their custom normals, constraints, modifiers,
vertex groups, parenting, animation, every object type there is -- all
run through Blender's own version conversion, exactly as File > Append
would. Nothing that machinery understands is lost, because it is the
machinery.

And Halcyon's classic-format reader (core/blend279.py) recovers the one
thing that machinery DROPS: everything Blender Internal. The material
fields (shader pair, hardness, mirror, transparency), all eighteen
texture slots with their mappings and influences, the procedural
textures themselves -- Clouds, Wood, Marble, Magic, Blend, Stucci,
Noise, Musgrave, Voronoi, Distorted Noise -- with their colorbands,
noise bases and every parameter, rebuilt as Halcyon master-shader trees
whose BI Texture nodes run the original algorithms (core/bitex.py).
It also recovers the saved selection (so "Selected Objects Only" works
decades later), the old world as a Halcyon sky, and each lamp's falloff
onto Halcyon's own decay controls.

If Blender itself cannot append from a file (some pre-2.5 relics), the
importer falls back to rebuilding meshes, lamps and cameras directly
from the parsed file -- with a plain warning that constraints and
modifiers cannot survive that route.
"""

import os
import re

import bpy
from bpy.props import BoolProperty, FloatProperty, StringProperty

try:                                        # fakebpy has no bpy_extras
    from bpy_extras.io_utils import ImportHelper
except ImportError:                         # pragma: no cover
    class ImportHelper:
        pass

from . import templates
from .core import blend279, blend279_map


# --------------------------------------------------------------- materials


def _resolve_image(image, blend_dir, warnings, label, cache=None):
    """A legacy image slot -> a bpy image, or None.

    Packed textures come FIRST: 2.79 projects habitually packed their
    images into the .blend, so the pixels live in the file itself and
    nothing on disk needs to exist. The bytes are written to Blender's
    temp folder, loaded, and packed straight into the CURRENT file so
    the import owns its textures.

    Unpacked paths walk a ladder: the stored path resolved against the
    LEGACY file's folder ('//' is relative to it, not to the current
    file), the path as stored, the bare filename next to the legacy
    file, and the bare filename in a textures/ folder beside it --
    which is where moved projects usually keep them.

    `cache` spans the whole import: one classic image feeding four
    materials' worth of channel entries loads ONCE, not eleven times.
    """
    packed = image.get('packed')
    name = image.get('name') or 'image'
    key = ('packed', name, len(packed)) if packed \
        else ('path', image.get('path') or '')
    if cache is not None and key in cache:
        return cache[key]
    if packed:
        # pack the bytes STRAIGHT FROM MEMORY. The old route wrote the
        # file to the temp folder, loaded it back from disk, then
        # repacked -- and that disk round-trip was the one site raising
        # (handled) access violations in every field faulthandler log:
        # a freshly-written file in %TEMP% is exactly what antivirus
        # filters and memory-mapped loading fight over. Image.pack(data=)
        # skips the filesystem entirely; the temp-file route survives
        # only as the fallback for a bpy too old to have it.
        try:
            img = bpy.data.images.new(name, 1, 1)
            try:
                img.pack(data=bytes(packed), data_len=len(packed))
                img.source = 'FILE'         # pixels come from the pack
                img.filepath = f'//{name}'  # cosmetic; nothing on disk
            except TypeError:
                # no pack(data=) in this bpy: the old disk route
                bpy.data.images.remove(img)
                safe = ''.join(c if c.isalnum() or c in '._-' else '_'
                               for c in name) or 'packed_image'
                tmp = os.path.join(bpy.app.tempdir or blend_dir, safe)
                with open(tmp, 'wb') as fh:
                    fh.write(packed)
                img = bpy.data.images.load(tmp, check_existing=False)
                img.name = name
                try:
                    img.pack()              # the new file owns it now
                except Exception:                               # noqa: BLE001
                    pass
            if cache is not None:
                cache[key] = img
            return img
        except Exception as e:                                  # noqa: BLE001
            warnings.append(f'{label}: packed image {name!r} could not '
                            f'be extracted ({e})')
            return None
    path = image.get('path') or ''
    if not path:
        return None
    rel = path[2:] if path.startswith('//') else path
    tail = os.path.basename(rel.replace('\\', '/'))
    tried = []
    for cand in (os.path.join(blend_dir, rel), path,
                 os.path.join(blend_dir, tail),
                 os.path.join(blend_dir, 'textures', tail)):
        cand = os.path.normpath(cand)
        if cand in tried:
            continue
        tried.append(cand)
        try:
            if os.path.isfile(cand):
                img = bpy.data.images.load(cand, check_existing=True)
                if cache is not None:
                    cache[key] = img
                return img
        except Exception:                                       # noqa: BLE001
            continue
    warnings.append(f'{label}: image {path!r} not found (tried '
                    f'{len(tried)} locations); the node imports '
                    'without it')
    return None


def _convert_material(bmat, mdict, version, blend_dir, warnings,
                      img_cache=None):
    """Rebuild one APPENDED material's tree from its parsed BI data.

    Returns (bmat, number of texture entries that got an image). The
    import log states how many pictures made it -- the difference
    between 'the importer failed' and 'the classic file's image slots
    are empty' has cost a field round before.
    """
    spec = blend279_map.material_spec(mdict, version)
    warnings.extend(spec['warnings'])
    n_images = 0
    for entry in spec['textures']:
        image = entry.pop('image', None)
        # uv_layer stays ON the entry: build_spec wires a UV Map node
        # for slots that name a layer (the field's 'Blood' splatters
        # sampled the WRONG layout when the active layer stood in)
        if image is not None:
            img = _resolve_image(image, blend_dir, warnings,
                                 spec['name'], img_cache)
            if img is not None:
                entry.setdefault('props', {})['image'] = img
                n_images += 1
    templates.build_spec(bmat, spec)
    return bmat, n_images


def plan_objects(scene_data, only_selected, import_lights, import_cameras,
                 include_hidden=False):
    """Which parsed objects to request from the appender.

    Returns (names in file order, {name: parsed object}). Pure data ->
    testable without Blender. Objects on layers OUTSIDE the scene's
    visible mask are skipped by default: 2.79 never rendered them, and
    importing them lit the field's scenes with five hidden lamps and
    floated a hidden hand into frame.
    """
    names, per_name = [], {}
    scene_lay = int(scene_data.get('scene_lay') or 0)
    for ob in scene_data['objects']:
        if only_selected and not ob['selected']:
            continue
        if not include_hidden and scene_lay and \
                not (int(ob.get('layers') or 1) & scene_lay):
            continue
        if ob['kind'] == 'LAMP' and not import_lights:
            continue
        if ob['kind'] == 'CAMERA' and not import_cameras:
            continue
        if not ob['name'] or ob['name'] in per_name:
            continue
        per_name[ob['name']] = ob
        names.append(ob['name'])
    return names, per_name


def hidden_object_names(per_name, scene_lay):
    """The planned objects whose layers 2.79 had switched OFF.

    Pure data -> testable. Empty when the file carries no layer mask
    (pre-layer eras parse as 0: everything counts as visible)."""
    lay = int(scene_lay or 0)
    if not lay:
        return set()
    return {n for n, ob in per_name.items()
            if not (int(ob.get('layers') or 1) & lay)}


def base_name(name):
    """Strip Blender's .001-style suffix an append adds on collision."""
    return re.sub(r'\.\d{3,}$', '', name or '')


def material_name_candidates(name):
    """Appended-name -> classic-name candidates, most specific first.

    The appended name itself comes FIRST: classic files legitimately
    contain materials named 'Material.001' (Blender's own default
    names), and when the append had no collision the name arrives
    unchanged -- stripping unconditionally would look up 'Material'
    and convert the wrong material, or nothing (a real field failure:
    twelve default-named materials skipped in one file). Only after
    the exact name misses does one .NNN come off at a time, walking a
    collision rename back to the classic name.
    """
    seen = set()
    name = name or ''
    while name and name not in seen:
        seen.add(name)
        yield name
        stripped = re.sub(r'\.\d{3,}$', '', name)
        if stripped == name:
            return
        name = stripped


def find_parsed_material(name, by_name):
    """(BI dict, the classic name that matched) via the exact-first
    ladder, or (None, None)."""
    for cand in material_name_candidates(name):
        mdict = by_name.get(cand)
        if mdict is not None:
            return mdict, cand
    return None, None


def parsed_materials_by_name(materials):
    """{material name: BI dict} -- names are unique within one .blend
    (Blender enforces it), which makes name matching deterministic."""
    out = {}
    for mdict in materials.values():
        nm = mdict.get('name')
        if nm and nm not in out:
            out[nm] = mdict
    return out


def collect_slot_materials(objects):
    """Every distinct material on the given objects' slots, keyed by
    name_full so shared materials convert once."""
    seen = {}
    for ob in objects:
        for slot in getattr(ob, 'material_slots', None) or []:
            bm = getattr(slot, 'material', None)
            if bm is None:
                continue
            key = getattr(bm, 'name_full', None) or getattr(bm, 'name', '')
            if key and key not in seen:
                seen[key] = bm
    return seen


def match_slot_materials(appended, parsed, materials):
    """Pair an appended object's material slots with parsed BI data.

    Returns [(slot index, bpy material, BI dict), ...] for slots where
    the file carries Internal data. Matching is BY SLOT POSITION against
    the parser's matbits/colbits-resolved pointers -- names can collide
    and get .001 suffixes, slots cannot.
    """
    out = []
    slots = getattr(appended, 'material_slots', None) or []
    ptrs = parsed.get('mat_ptrs') or []
    for i, slot in enumerate(slots):
        bmat = getattr(slot, 'material', None)
        if bmat is None or i >= len(ptrs) or not ptrs[i]:
            continue
        mdict = materials.get(ptrs[i])
        if mdict is not None:
            out.append((i, bmat, mdict))
    return out


def plan_conversions(pairs, per_name, materials, by_name):
    """Decide which parsed BI material each appended material converts
    from, as pure data (no bpy calls -> testable headlessly).

    Route 1, slot pointers: the appender keeps slot order, so slot i of
    the appended object IS slot i of the parsed one, and the pointer
    resolved there names the material index-precisely -- immune to any
    rename. Proven against a real 2.79 file (152/152 slots). Route 2,
    names, for whatever the pointer walk did not reach: the exact name
    first, then one suffix strip at a time.

    Returns ([(key, bpy material, BI dict, route, stripped-to)],
    {'ptr_slots', 'parsed_with_ptrs', 'xray', 'unmatched'}).
    """
    order = []
    claimed = set()
    n_ptr_slots = 0
    n_parsed_with_ptrs = 0
    xray = []
    for name, ob in pairs:
        parsed = per_name.get(name)
        if parsed is None:
            continue
        ptrs = parsed.get('mat_ptrs') or []
        if any(ptrs):
            n_parsed_with_ptrs += 1
        matches = match_slot_materials(ob, parsed, materials)
        n_ptr_slots += len(matches)
        nslots = len(getattr(ob, 'material_slots', None) or [])
        if not matches and (nslots or any(ptrs)) and len(xray) < 5:
            xray.append(
                f'  pairing {name!r}: appended slots={nslots}, '
                f'parsed kind={parsed.get("kind")}, '
                f'parsed ptrs={[hex(p) for p in ptrs]}')
        for _i, bmat, mdict in matches:
            key = getattr(bmat, 'name_full', None) \
                or getattr(bmat, 'name', '')
            if not key or key in claimed:
                continue
            claimed.add(key)
            order.append((key, bmat, mdict, 'pointer', None))
    seen_mats = collect_slot_materials([ob for _n, ob in pairs])
    unmatched = []
    for key, bmat in seen_mats.items():
        if key in claimed:
            continue
        nm = getattr(bmat, 'name', key)
        mdict, matched = find_parsed_material(nm, by_name)
        if mdict is None:
            unmatched.append(nm)
            continue
        claimed.add(key)
        order.append((key, bmat, mdict, 'name',
                      None if matched == nm else matched))
    return order, {'ptr_slots': n_ptr_slots,
                   'parsed_with_ptrs': n_parsed_with_ptrs,
                   'xray': xray, 'unmatched': unmatched}


# ------------------------------------------------------- lamps and cameras


def _apply_lamp_bi(light, lm, warnings=None):
    """Everything Blender Internal knew about one lamp, onto the bpy
    light and its Halcyon settings.

    ONE function for BOTH import routes -- the parser build and the
    append enrichment -- so they cannot disagree. The append route
    used to keep Blender's own 2.79->5.x energy versioning and set a
    plain unbounded INVERSE falloff: point-family lamps came in
    several times darker than BI at working distances, which is
    exactly 'the lights are too dark' whenever a scene arrived
    through the appender. The file's raw values are the truth;
    Blender's watt conversion is not.

    Returns the applied strength. The set is VERIFIED by reading the
    property back: the field reported every imported sun at exactly
    1.0 -- Blender's own append default -- which is what this
    function's work looks like when it never lands. If the readback
    disagrees, that is now a named warning instead of a mystery.
    """
    hemi = lm['type'] == 'HEMI'
    try:
        light.color = lm['color']
    except (AttributeError, TypeError, ValueError):
        pass
    # BI-faithful intensity: contribution = lampcol * energy * visifac,
    # no unit conversions. The engine divides point radiance by 4*pi
    # and both lobes by pi, so the import pre-multiplies them back.
    if lm['type'] in ('SUN', 'HEMI'):
        want = lm['energy'] * 3.14159265
    else:
        want = lm['energy'] * 4.0 * 3.14159265 ** 2
    light.energy = want
    got = None
    try:
        got = float(light.energy)
    except (AttributeError, TypeError, ValueError):
        pass
    if warnings is not None and got is not None \
            and abs(got - want) > max(0.01, 0.001 * abs(want)):
        warnings.append(
            f"{lm.get('name', '?')}: energy {want:.4g} did not stick "
            f'(the light reads {got:.4g}) -- please report this line')
    if lm['type'] == 'SPOT':
        # lamp_map already speaks radians (pre-2.70 degrees converted)
        try:
            light.spot_size = lm['spot_size']
            light.spot_blend = lm['spot_blend']
        except (AttributeError, TypeError, ValueError):
            pass
    hs = getattr(light, 'halcyon', None)
    if hs is None:
        return want
    if hemi:
        hs.hemi = True
        hs.shadow = 'NONE'         # BI never shadowed hemi lamps
    else:
        # each lamp shadows exactly as 2.79 did: a spot's buffer maps,
        # the Ray bit traces, and a lamp with NEITHER casts no shadow
        # at all -- importing everything as MAP shadowed lamps BI
        # never shadowed and read as 'lights too dark' in the field
        try:
            hs.shadow = lm.get('shadow', 'NONE')
        except (AttributeError, TypeError, ValueError):
            pass
    if lm['type'] not in ('SUN', 'HEMI'):
        hs.decay = {'CONSTANT': 'NONE',
                    'INVERSE_LINEAR': 'BI_LINEAR',
                    'INVERSE_SQUARE': 'BI_SQUARE',
                    'SLIDERS': 'BI_SLIDERS'}.get(
                        lm.get('falloff'), 'BI_LINEAR')
        hs.decay_start = 0.0
        hs.decay_end = max(lm['distance'], 0.01)
        try:
            hs.decay_ld1 = float(lm.get('ld1') or 0.0)
            hs.decay_ld2 = float(lm.get('ld2') or 0.0)
            hs.bi_sphere = bool(lm.get('sphere'))
        except (AttributeError, TypeError, ValueError):
            pass
    # the lamp mode bits BI's loop honoured: Negative subtracts, No
    # Diffuse leaves only the highlight, No Specular only the diffuse
    try:
        if lm.get('negative'):
            hs.negative = True
        if lm.get('no_diffuse'):
            hs.specular_only = True
        if lm.get('no_specular'):
            hs.diffuse_only = True
    except (AttributeError, TypeError, ValueError):
        pass
    # R164: the lamp's shadow colour (lashdw tints the shadowed
    # diffuse; black = the classic full shadow)
    try:
        sc = lm.get('shadow_color')
        if sc is not None:
            hs.shadow_color = tuple(sc)
    except (AttributeError, TypeError, ValueError):
        pass
    return want


def _enrich_lamp(light, parsed_lamp, version, warnings):
    """An appended light, brought to exactly what the parser build
    produces: the classic file's own energy, colour, falloff, shadow
    rule and mode bits override whatever Blender's versioning made.

    Returns a one-line receipt ('Sun.002: file energy 3 -> 9.42') for
    the import log, so 'did the enrichment run' is never a guess."""
    lm = blend279_map.lamp_map(parsed_lamp, version)
    warnings.extend(lm['warnings'])
    applied = _apply_lamp_bi(light, lm, warnings)
    return (f"{lm.get('name', '?')}: file energy {lm['energy']:g} "
            f'-> {applied:.4g}')


def _apply_color_management(scene):
    """Blender's color management is DISABLED under Halcyon.

    Halcyon's OWN display chain (exposure, the view-transform curve,
    gamma, the preset looks, the CRT) is the whole grading pipeline --
    including, for 2.79 imports, the sRGB encode the file's 'Default'
    view applied (_apply_scene_pipeline). Any Blender view transform
    would regrade that output a SECOND time -- AgX darkened the
    field's lamps and crushed a black body's 2%% sheen; 'Standard'
    double-encoded and washed everything grey. 'Raw' is the only
    setting that shows the engine's output untouched, and the engine
    itself pins it while active."""
    try:
        scene.display_settings.display_device = 'sRGB'
    except Exception:                                           # noqa: BLE001
        pass
    try:
        vs = scene.view_settings
        try:
            vs.view_transform = 'Raw'
        except (TypeError, ValueError):
            vs.view_transform = 'Standard'   # a build without Raw
        try:
            vs.look = 'None'
        except (TypeError, ValueError):
            pass
        vs.exposure = 0.0
        vs.gamma = 1.0
    except Exception:                                           # noqa: BLE001
        pass


def _apply_scene_pipeline(scene, sdata, warnings):
    """The file's OWN pipeline onto Halcyon's settings.

    2.79 rendered BI scene-linear and showed it through the scene's
    view transform -- 'Default' is the sRGB display encode. Blender
    stays pinned to Raw (above); HALCYON runs the curve itself, plus
    the matching input end: sRGB-tagged textures linearize on load.
    DNA colours (materials, lamps, slots) are already linear and pass
    through untouched. Also applies the file's frame size and its
    transparent-sky flag (R_ALPHAPREMUL) -- the field's 960x960
    "white background" F12 was a transparent PNG on a white viewer."""
    sm, sw = blend279_map.scene_settings_map(sdata)
    warnings.extend(sw)
    hs = getattr(scene, 'halcyon', None)
    if hs is not None:
        for k in ('color_management', 'input_gamma_naive', 'exposure',
                  'gamma', 'film_transparent'):
            if k in sm:
                try:
                    setattr(hs, k, sm[k])
                except (TypeError, ValueError):
                    pass
    rd = getattr(scene, 'render', None)
    if rd is not None and 'res_x' in sm:
        try:
            rd.resolution_x = sm['res_x']
            rd.resolution_y = sm['res_y']
            rd.resolution_percentage = sm['res_pct']
        except (TypeError, ValueError, AttributeError):
            pass


def _apply_world(scene, wdict, warnings=None):
    wm = blend279_map.world_map(wdict, warnings)
    if wm is None:
        return
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new(wdict.get('name') or 'World')
        scene.world = world
    hw = getattr(world, 'halcyon', None)
    if hw is not None:
        hw.mode = 'GRADIENT'
        hw.horizon = wm['horizon']
        # a 2.79 world with Blend off was a solid horizon colour
        hw.zenith = wm['zenith'] if wm['blend'] else wm['horizon']
        if any(c > 0.0 for c in wm['ambient']):
            hw.ambient = wm['ambient']
            hw.ambient_level = 1.0
    hs = getattr(scene, 'halcyon', None)
    if hs is not None:
        # BI had exactly ONE ambient: the world's, times each
        # material's Amb slider. Halcyon ADDS the scene's global
        # ambient on top, and its cosy 0.05 default is a wash the
        # original render never had -- zero it so the world's own
        # ambient (set above, usually black in 2.79 files) is the
        # whole story, exactly as authored
        hs.global_ambient = (0.0, 0.0, 0.0)
    # R164: the Exposure panel travels onto the world settings
    if hw is not None:
        try:
            hw.exposure = float(wm.get('exposure', 0.0))
            hw.exposure_range = float(wm.get('exposure_range', 1.0))
        except (TypeError, ValueError, AttributeError):
            pass
    mist = wm.get('mist')
    if hs is not None and mist is not None:
        hs.fog = True
        hs.fog_mode = 'LINEAR'
        hs.fog_color = wm['horizon']
        hs.fog_start = mist['start']
        hs.fog_end = mist['start'] + mist['depth']
        fo = mist.get('falloff', 'LINEAR')
        if fo != 'LINEAR' and warnings is not None:
            warnings.append(
                f'World mist uses the {fo.title().replace("_", " ")} '
                'curve; the engine fog renders Linear')


# ------------------------------------------------------------ the fallback
#
# Only for files Blender's own loader refuses. Meshes, lamps and cameras
# rebuild from the parsed data; anything the parser does not carry
# (constraints, modifiers, other object types) is loudly absent.


def _build_mesh(name, geo):
    me = bpy.data.meshes.new(name)
    verts = [tuple(float(c) for c in v) for v in geo['verts']]
    me.from_pydata(verts, [], [tuple(p) for p in geo['polys']])
    uvl = geo.get('uv_loops')
    if uvl is not None and len(me.loops) == len(uvl):
        layer = me.uv_layers.new(name='UVMap')
        layer.data.foreach_set('uv', [float(x) for uv in uvl for x in uv])
    cols = geo.get('col_loops')
    if cols is not None and len(me.loops) == len(cols):
        attr = me.color_attributes.new(name='Col', type='FLOAT_COLOR',
                                       domain='CORNER')
        attr.data.foreach_set('color',
                              [float(x) for c in cols for x in c])
    ids = geo.get('mat_ids')
    if ids and len(ids) == len(me.polygons):
        me.polygons.foreach_set('material_index', ids)
    me.validate()
    me.update()
    return me


def _build_light(la, version):
    lm = blend279_map.lamp_map(la, version)
    # Blender has no HEMI light data since 2.8: a SUN carries it, and
    # the per-light Halcyon 'hemi' toggle tells the engine to shade
    # BI's wrap (both lobes, no shadows)
    light = bpy.data.lights.new(lm['name'],
                                'SUN' if lm['type'] == 'HEMI'
                                else lm['type'])
    # every BI lamp fact through the ONE shared function the append
    # enrichment also uses, so the two import routes cannot disagree
    lw = list(lm['warnings'])
    applied = _apply_lamp_bi(light, lm, lw)
    lw.append(f"{lm.get('name', '?')}: file energy {lm['energy']:g} "
              f'-> {applied:.4g}')
    return light, lw


def _build_camera(ca):
    cam = bpy.data.cameras.new(ca.get('name') or 'Camera')
    cam.lens = max(float(ca.get('lens') or 35.0), 1.0)
    cam.clip_start = max(float(ca.get('clipsta') or 0.1), 1e-4)
    cam.clip_end = max(float(ca.get('clipend') or 100.0), cam.clip_start)
    if ca.get('sensor_x'):
        cam.sensor_width = float(ca['sensor_x'])
    if int(ca.get('type') or 0) == 1:
        cam.type = 'ORTHO'
    return cam


def _fallback_import(context, filepath, per_name, materials, version,
                     blend_dir, coll, warnings, hidden=()):
    """Rebuild what the parser carries, for files the appender refused."""
    warnings.append(
        "Blender's own loader could not append from this file; geometry "
        'was rebuilt from the classic data directly. Meshes, lamps and '
        'cameras arrive; constraints, modifiers and other object types '
        'cannot survive this route')
    sc = blend279.read_legacy_scene(filepath, geometry=True)
    parsed_all = {o['name']: o for o in sc['objects']}
    mats = {}
    img_cache = {}
    imported = []
    try:
        from mathutils import Matrix
    except ImportError:                     # pragma: no cover (fakebpy)
        Matrix = None
    for name in per_name:
        ob = parsed_all.get(name)
        if ob is None:
            continue
        data = None
        if ob['kind'] == 'MESH' and ob.get('data'):
            data = _build_mesh(ob['data'].get('name') or name, ob['data'])
        elif ob['kind'] == 'LAMP' and ob.get('data'):
            data, lw = _build_light(ob['data'], version)
            warnings.extend(lw)
        elif ob['kind'] == 'CAMERA' and ob.get('data'):
            data = _build_camera(ob['data'])
        elif ob['kind'] != 'EMPTY':
            warnings.append(f'{name}: {ob["kind"].title()} objects need '
                            "Blender's own loader; skipped")
            continue
        new_ob = bpy.data.objects.new(name, data)
        if Matrix is not None and ob.get('matrix') is not None:
            new_ob.matrix_world = Matrix(
                [tuple(r) for r in ob['matrix']]).transposed()
        if ob['kind'] == 'MESH' and data is not None:
            for ptr in ob.get('mat_ptrs', []):
                mdict = materials.get(ptr)
                bmat = None
                if mdict is not None:
                    key = id(mdict)
                    if key not in mats:
                        mats[key], _ni = _convert_material(
                            bpy.data.materials.new(mdict.get('name')
                                                   or 'Material'),
                            mdict, version, blend_dir, warnings,
                            img_cache)
                    bmat = mats[key]
                data.materials.append(bmat)
        if name in hidden:
            try:
                new_ob.hide_render = True
                new_ob.hide_viewport = True
            except AttributeError:
                pass
        if ob.get('col') is not None:
            try:
                new_ob.color = tuple(ob['col'])
            except (AttributeError, TypeError, ValueError):
                pass
        if ob.get('smoothresh'):
            try:
                new_ob['halcyon_smoothresh'] = float(ob['smoothresh'])
            except (AttributeError, TypeError):
                pass
        coll.objects.link(new_ob)
        imported.append(new_ob)
    return imported, len(mats), len(img_cache)


# ----------------------------------------------------------------- logging


def safe_print(line):
    """A console line that can never abort an import.

    Windows consoles reject characters their codepage lacks; a print
    that raises mid-operator kills everything after it -- including
    the text-datablock log, the only record the user can send."""
    try:
        print(line)
    except Exception:                                           # noqa: BLE001
        try:
            print(line.encode('ascii', 'backslashreplace').decode('ascii'))
        except Exception:                                       # noqa: BLE001
            pass


def write_log_text(name, lines):
    """Every line into a text datablock, hardened; returns it or None.

    The old shape -- a loop of writes behind ONE fence -- meant a
    single unwritable line silently truncated everything after it:
    one field import log ends mid-list, exactly at the
    first line carrying a parsed object name, and the lamp receipts
    below it were never a guessable absence again. Now the whole
    payload writes in ONE call; if that fails, line-by-line with a
    PER-LINE fence that replaces only the offending line with a
    marker NAMING the exception. The terminator line always goes
    last, so 'no "log complete" at the bottom' now MEANS truncated.
    """
    try:
        text = bpy.data.texts.new(name)
    except Exception:                                           # noqa: BLE001
        return None
    done = f'log complete ({len(lines)} lines)'
    try:
        text.write('\n'.join(str(l) for l in lines) + f'\n{done}\n')
        return text
    except Exception:                                           # noqa: BLE001
        pass
    for l in lines:
        try:
            text.write(f'{l}\n')
        except Exception as e:                                  # noqa: BLE001
            try:
                text.write(f'<a line would not write: '
                           f'{type(e).__name__}: {e}>\n')
            except Exception:                                   # noqa: BLE001
                pass
    try:
        text.write(f'{done}\n')
    except Exception:                                           # noqa: BLE001
        pass
    return text


# ----------------------------------------------------- the shared lamp fix


def lamp_pools(sc):
    """Every lamp in a parsed classic file: ({object name: lamp data},
    {lamp datablock name: lamp data}). All OB spans in the FILE are
    included -- hidden layers and other scenes too, exactly the set
    the append browser offers."""
    by_obj, by_data = {}, {}
    for fob in sc['objects']:
        if fob['kind'] != 'LAMP' or fob.get('data') is None:
            continue
        if fob['name'] and fob['name'] not in by_obj:
            by_obj[fob['name']] = fob['data']
        dn = fob['data'].get('name')
        if dn and dn not in by_data:
            by_data[dn] = fob['data']
    return by_obj, by_data


def fix_lights_from_parsed(sc, targets, warnings, bare_lights=()):
    """The classic file's own BI values onto already-present lights.

    ONE core for every route that repairs lights after Blender's own
    machinery converted them -- the manual fixer operator and the
    automatic append watch -- so they cannot disagree. `targets` are
    light OBJECTS (matched by object name first, then their data
    name); `bare_lights` are Light datablocks that arrived without an
    object (a data-section append), matched by data name alone, and
    skipped when a target already covers them.

    Returns (receipt lines, unmatched names, by_obj, by_data).
    """
    version = int(sc['version']) if sc['version'].isdigit() else 279
    by_obj, by_data = lamp_pools(sc)
    receipts, unmatched = [], []
    if not by_obj and not by_data:
        return receipts, unmatched, by_obj, by_data
    seen_data = {}                  # id(light data) -> matched name
    for ob in targets:
        data_name = getattr(ob.data, 'name', '') or ''
        la, cand, how = _match_parsed_lamp(ob.name, data_name,
                                           by_obj, by_data)
        if la is None:
            unmatched.append(ob.name)
            continue
        key = id(ob.data)
        prev = seen_data.get(key)
        if prev is not None and prev != cand:
            warnings.append(
                f'{ob.name}: shares its light data with a lamp '
                f'already fixed from file lamp {prev!r}; the later '
                f'match {cand!r} wins')
        seen_data[key] = cand
        lm = blend279_map.lamp_map(la, version)
        warnings.extend(lm['warnings'])
        try:
            applied = _apply_lamp_bi(ob.data, lm, warnings)
        except Exception as e:                                  # noqa: BLE001
            import traceback
            traceback.print_exc()
            warnings.append(
                f'{ob.name}: lamp fix crashed ({type(e).__name__}: '
                f'{e}) -- the light keeps its current values; '
                'please report the console traceback')
            continue
        note = '' if cand == ob.name else \
            f' (matched file {how} {cand!r})'
        receipts.append(f"{ob.name}: file energy {lm['energy']:g} "
                        f'-> {applied:.4g}{note}')
    covered = {id(getattr(ob, 'data', None)) for ob in targets}
    for lt in bare_lights:
        if id(lt) in covered:
            continue                # its object was already the target
        name = getattr(lt, 'name', '') or ''
        la, cand, how = _match_parsed_lamp('', name, {}, by_data)
        if la is None:
            unmatched.append(name or '?')
            continue
        lm = blend279_map.lamp_map(la, version)
        warnings.extend(lm['warnings'])
        try:
            applied = _apply_lamp_bi(lt, lm, warnings)
        except Exception as e:                                  # noqa: BLE001
            import traceback
            traceback.print_exc()
            warnings.append(
                f'{name}: lamp fix crashed ({type(e).__name__}: '
                f'{e}) -- the light keeps its current values; '
                'please report the console traceback')
            continue
        note = '' if cand == name else \
            f' (matched file {how} {cand!r})'
        receipts.append(f"{name}: file energy {lm['energy']:g} "
                        f'-> {applied:.4g}{note}')
    return receipts, unmatched, by_obj, by_data


# ------------------------------------------------------------- the operator


class HALCYON_OT_append_legacy(bpy.types.Operator, ImportHelper):
    """Append from a classic .blend (2.79 and earlier) with the Blender
    Internal materials converted to Halcyon.

    Blender's own loader brings the objects -- constraints, normals,
    modifiers, parenting, animation, all of it -- and Halcyon reads the
    same file for what that loader drops: the Internal materials, whose
    procedural textures arrive on the original algorithms
    """

    bl_idname = 'halcyon.append_legacy'
    bl_label = "Import Legacy .blend"
    bl_options = {'REGISTER', 'UNDO'}

    filepath: StringProperty(subtype='FILE_PATH')
    filter_glob: StringProperty(default='*.blend', options={'HIDDEN'})

    only_selected: BoolProperty(
        name="Selected Objects Only", default=False,
        description="Import only the objects that were selected when the "
                    "old file was saved -- the selection is stored in the "
                    "file, so 'append what I had picked' still works "
                    "twenty-five years later")
    import_lights: BoolProperty(
        name="Lights", default=True,
        description="Import lamps with the classic file's OWN values: "
                    "energy through the Blender Internal conversion "
                    "(suns x pi, point-family x 4 pi squared), colour, "
                    "falloff, per-lamp shadow rule and mode bits -- "
                    "Blender's own watt conversion is overridden")
    include_hidden: BoolProperty(
        name="Hidden Layers", default=False,
        description="Also import objects on layers 2.79 had switched "
                    "off. They arrive hidden from the viewport and the "
                    "render -- exactly how 2.79 treated them -- so the "
                    "default F12 look is unchanged; unhide whichever "
                    "parts you want. Model packs often stash alternate "
                    "parts and extra light rigs on hidden layers")
    import_cameras: BoolProperty(
        name="Cameras", default=True,
        description="Import cameras with their lens, sensor and clip "
                    "ranges")
    import_world: BoolProperty(
        name="World as Halcyon Sky", default=False,
        description="Replace this scene's sky with the old file's world: "
                    "horizon and zenith as a Halcyon gradient, mist as "
                    "fog. Off by default because it changes the current "
                    "scene rather than adding to it")
    scale: FloatProperty(
        name="Scale", default=1.0, min=0.001, max=1000.0,
        description="Uniform scale applied to the imported objects "
                    "(parents scale their children with them). Old "
                    "scenes were often built at arbitrary sizes; 0.1 "
                    "shrinks a 10x-too-big one")

    def execute(self, context):
        # ---- read the classic file for everything Blender drops
        try:
            sc = blend279.read_legacy_scene(self.filepath, geometry=False)
        except blend279.BlendError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        except OSError as e:
            self.report({'ERROR'}, f'cannot read {self.filepath}: {e}')
            return {'CANCELLED'}

        version = int(sc['version']) if sc['version'].isdigit() else 279
        blend_dir = os.path.dirname(self.filepath)
        warnings = list(sc['warnings'])

        names, per_name = plan_objects(sc, self.only_selected,
                                       self.import_lights,
                                       self.import_cameras,
                                       include_hidden=self.include_hidden)
        if not names:
            self.report({'WARNING'},
                        'no matching objects in the file (try turning '
                        'Selected Objects Only off)')
            return {'CANCELLED'}
        # objects on layers 2.79 had switched off: importable (option
        # above), but they arrive HIDDEN -- 2.79 showed and rendered
        # neither, and ten of the field file's fifteen suns live there
        hidden_names = hidden_object_names(per_name, sc.get('scene_lay'))
        if self.include_hidden and hidden_names:
            warnings.append(
                f'{len(hidden_names)} hidden-layer objects imported '
                'HIDDEN from viewport and render, as 2.79 kept them '
                '(unhide to use): '
                + ', '.join(sorted(hidden_names)[:12])
                + ('...' if len(hidden_names) > 12 else ''))

        coll_name = os.path.splitext(
            os.path.basename(self.filepath))[0] or 'Legacy'
        coll = bpy.data.collections.new(f'{coll_name} (legacy)')
        context.scene.collection.children.link(coll)

        # ---- the actual append, through Blender's own machinery
        appended = []
        used_fallback = False
        try:
            with bpy.data.libraries.load(self.filepath) as (dfrom, dto):
                have = set(dfrom.objects)
                request = [n for n in names if n in have]
                for n in names:
                    if n not in have:
                        warnings.append(f'{n}: not offered by the '
                                        'appender; skipped')
                dto.objects = request
            pairs = [(n, ob) for n, ob in zip(request, dto.objects)
                     if ob is not None]
            appended = [ob for _n, ob in pairs]
        except Exception as e:                                  # noqa: BLE001
            print(f'[halcyon legacy import] appender failed: {e}')
            pairs = []

        n_mats = 0
        n_images = 0
        n_via_ptr = None                    # None -> fallback route
        if appended:
            for ob in appended:
                try:
                    coll.objects.link(ob)
                except RuntimeError:
                    pass
            # hidden-layer objects stay hidden, exactly as 2.79 kept
            # them: neither drawn nor rendered until the user unhides
            for name, ob in pairs:
                if name in hidden_names:
                    try:
                        ob.hide_render = True
                        ob.hide_viewport = True
                    except AttributeError:
                        pass
            # R164: the classic per-object fields the shader reads --
            # the object colour (MA_OBCOLOR) and the Auto Smooth angle
            # (the RAYBIAS terminator threshold, stashed as a custom
            # property because 5.x objects no longer carry it)
            for name, ob in pairs:
                parsed = per_name.get(name)
                if not parsed:
                    continue
                col = parsed.get('col')
                if col is not None:
                    try:
                        ob.color = tuple(col)
                    except (AttributeError, TypeError, ValueError):
                        pass
                sr = parsed.get('smoothresh')
                if sr:
                    try:
                        ob['halcyon_smoothresh'] = float(sr)
                    except (AttributeError, TypeError):
                        pass
            # dependencies the appender pulled in but nothing linked
            # (a parent outside a selected-only request, a constraint
            # target): give them a home so they are not orphans
            seen = set(appended)
            frontier = list(appended)
            while frontier:
                ob = frontier.pop()
                targets = [getattr(ob, 'parent', None)]
                for con in getattr(ob, 'constraints', []) or []:
                    targets.append(getattr(con, 'target', None))
                for mod in getattr(ob, 'modifiers', []) or []:
                    for attr in ('object', 'target', 'mirror_object',
                                 'offset_object', 'object_from',
                                 'object_to', 'origin', 'curve'):
                        targets.append(getattr(mod, attr, None))
                for t in targets:
                    if t is None or t in seen:
                        continue
                    seen.add(t)
                    frontier.append(t)
                    if not getattr(t, 'users_collection', None):
                        try:
                            coll.objects.link(t)
                            warnings.append(f'{t.name}: linked as a '
                                            'dependency')
                        except RuntimeError:
                            pass

            # ---- the materials. Slot pointers lead: index-precise and
            # immune to append renames (a classic file really does hold
            # both 'RedWire' and 'RedWire.001' as different materials --
            # only position tells them apart with certainty). Names
            # catch whatever the pointer walk missed, trying the EXACT
            # appended name before any suffix strip, because classic
            # files legitimately contain default names like
            # 'Material.001' that a strip would orphan.
            # the whole material stage is fenced: a crash here must
            # not starve the lamp enrichment or the scene pipeline
            # below it -- 'materials converted but every sun reads
            # Blender's default 1.0' is exactly what an unfenced
            # exception between the two stages produces
            try:
                by_name = parsed_materials_by_name(sc['materials'])
                plan, mstats = plan_conversions(pairs, per_name,
                                                sc['materials'], by_name)
            except Exception as e:                              # noqa: BLE001
                import traceback
                traceback.print_exc()
                warnings.append(
                    f'material planning crashed ({type(e).__name__}: '
                    f'{e}); materials left as appended -- please '
                    'report the console traceback')
                plan, mstats = [], {'unmatched': [],
                                    'parsed_with_ptrs': 0, 'xray': []}
            n_via_ptr = 0
            img_cache = {}
            for key, bmat, mdict, route, stripped in plan:
                if stripped is not None:
                    warnings.append(
                        f'{getattr(bmat, "name", key)}: matched classic '
                        f'material {stripped!r} after stripping the '
                        'append rename suffix')
                if mdict.get('use_nodes'):
                    warnings.append(
                        f"{mdict.get('name')}: was a Blender Internal "
                        'NODE material; the flat fallback values were '
                        'converted (BI node trees are not translated)')
                try:
                    _convert_material(bmat, mdict, version, blend_dir,
                                      warnings, img_cache)
                    n_mats += 1
                    if route == 'pointer':
                        n_via_ptr += 1
                except Exception as e:                          # noqa: BLE001
                    import traceback
                    traceback.print_exc()
                    warnings.append(
                        f"{mdict.get('name')}: conversion failed "
                        f'({type(e).__name__}: {e}); material left '
                        'as appended')
            n_images = len(img_cache)       # distinct pictures loaded
            for nm in mstats['unmatched']:
                warnings.append(f'{nm}: no BI material of this name in '
                                'the classic file; left as appended')

            # ---- pointer-path X-ray: when conversion had to ride on
            # names alone, the log shows exactly where the pointer walk
            # came up empty, so a field report is enough to debug it.
            if n_mats and n_via_ptr == 0:
                warnings.append(
                    'slot-pointer path matched nothing; conversion rode '
                    'on names alone -- please report with the lines '
                    'below')
                warnings.append(
                    f"  x-ray: {mstats['parsed_with_ptrs']} of "
                    f'{len(pairs)} parsed objects carry material '
                    'pointers')
                warnings.extend(mstats['xray'])

            # ---- lamp enrichment from the classic fields, with a
            # receipt per lamp in the import log: 'did it run' must
            # never be a guess (the field's every-sun-reads-1.0).
            # The count line is UNCONDITIONAL -- a truncated field
            # log had NEITHER receipt nor no-receipt warning, an
            # absence two branches cannot produce; now the stage
            # always leaves at least this line
            _kind_census = {}
            for n, _ob in pairs:
                k = per_name.get(n, {}).get('kind') or '?'
                _kind_census[k] = _kind_census.get(k, 0) + 1
            warnings.append(
                'lamp stage reached: '
                + str(_kind_census.get('LAMP', 0))
                + f' lamp object(s) among {len(pairs)} appended'
                # R170: the field's '0 lamp object(s) among 157' -- the
                # census and the options answer WHY in one line: either
                # the file's lamps were excluded by an option (Lights
                # off, saved selection, hidden layers) or their kind
                # parsed oddly. No more guessing.
                + ' (kinds: '
                + ', '.join(f'{k} {v}' for k, v
                            in sorted(_kind_census.items()))
                + f'; options: lights={self.import_lights} '
                  f'selected={self.only_selected} '
                  f'hidden={self.include_hidden})')
            lamp_lines = []
            for name, ob in pairs:
                parsed = per_name.get(name)
                if parsed and parsed['kind'] == 'LAMP' and parsed.get(
                        'data') is not None and ob.data is not None:
                    try:
                        lamp_lines.append(_enrich_lamp(
                            ob.data, parsed['data'], version, warnings))
                    except Exception as e:                      # noqa: BLE001
                        import traceback
                        traceback.print_exc()
                        warnings.append(
                            f'{name}: lamp enrichment crashed '
                            f'({type(e).__name__}: {e}) -- the light '
                            "keeps Blender's own converted values; "
                            'please report the console traceback')
            if lamp_lines:
                warnings.append('lamps enriched with the file\'s own '
                                'energies: ' + '; '.join(lamp_lines))
            elif any(per_name.get(n, {}).get('kind') == 'LAMP'
                     for n, _ob in pairs):
                warnings.append(
                    'NO lamp was enriched although lamps were appended '
                    '-- their strengths are Blender defaults; please '
                    'report this line')

            # ---- uniform scale on the roots (children follow parents)
            if abs(self.scale - 1.0) > 1e-9:
                try:
                    from mathutils import Matrix
                    S = Matrix.Scale(self.scale, 4)
                    for ob in appended:
                        if getattr(ob, 'parent', None) is None:
                            ob.matrix_world = S @ ob.matrix_world
                except ImportError:         # pragma: no cover (fakebpy)
                    pass
            imported = appended
        else:
            used_fallback = True
            imported, n_mats, n_images = _fallback_import(
                context, self.filepath, per_name, sc['materials'],
                version, blend_dir, coll, warnings,
                hidden=hidden_names)

        if self.import_world and sc.get('world'):
            _apply_world(context.scene, sc['world'], warnings)
        # Blender pinned to Raw, always -- even when the file has no
        # world; then the FILE's own pipeline onto Halcyon's settings
        _apply_color_management(context.scene)
        _apply_scene_pipeline(context.scene, sc, warnings)

        for ob in imported:
            try:
                ob.select_set(True)
            except (AttributeError, RuntimeError):
                pass
        if imported:
            try:
                context.view_layer.objects.active = imported[0]
            except (AttributeError, RuntimeError):
                pass

        n_parsed_mats = len(sc['materials'])
        n_textured = sum(1 for m in sc['materials'].values()
                         if m.get('slots'))
        route_note = f' ({n_via_ptr} via slot pointers, ' \
                     f'{n_mats - n_via_ptr} by name)' \
            if n_via_ptr is not None else ' (direct rebuild)'
        try:
            from .version import VERSION
            ver = '.'.join(str(v) for v in VERSION)
        except Exception:                                       # noqa: BLE001
            ver = '?'
        stage = (f"Halcyon {ver}: parsed {len(sc['objects'])} objects, "
                 f"{n_parsed_mats} BI materials ({n_textured} with "
                 f"texture slots); planned {len(names)}; appended "
                 f"{len(appended)}; converted {n_mats}{route_note}; "
                 f"{n_images} texture images resolved")
        try:
            from . import fault_note
            fault_note(f'legacy import completed ({stage})', key='import')
        except Exception:                                       # noqa: BLE001
            pass
        for w in [stage] + warnings:
            safe_print(f'[halcyon legacy import] {w}')
        # the log also lands in a text datablock, because on Windows the
        # console is hidden and warnings must not be. Hardened writer:
        # a field log truncated mid-list and took the
        # lamp receipts with it silently
        write_log_text(f'{coll_name} import log', [stage] + warnings)
        route = 'rebuilt directly (fallback)' if used_fallback \
            else "Blender's own append"
        self.report({'INFO'},
                    f'{coll_name}: {len(imported)} objects via {route}; '
                    f'{stage}. Full notes: text editor > '
                    f"'{coll_name} import log'")
        return {'FINISHED'}


def _match_parsed_lamp(ob_name, data_name, by_obj, by_data):
    """The parsed file lamp for one already-appended light, or
    (None, None, None).

    Object names lead (they are what the user picked in the append
    browser, and what Blender renames on collision), lamp-DATA names
    catch the rest (a light whose object was renamed by hand keeps its
    data name). Each pool tries the EXACT name before walking the
    .001 rename suffixes off one at a time -- the same exact-first
    ladder the material matcher uses, because classic files
    legitimately contain names like 'Lamp.001' of their own.

    Returns (parsed lamp dict, the file name that matched, which pool:
    'object' or 'lamp data'). Pure data -> testable without Blender.
    """
    for pool, label, nm in ((by_obj, 'object', ob_name),
                            (by_data, 'lamp data', data_name)):
        for cand in material_name_candidates(nm):
            la = pool.get(cand)
            if la is not None:
                return la, cand, label
    return None, None, None


class HALCYON_OT_fix_appended_lamps(bpy.types.Operator, ImportHelper):
    """Stamp the classic file's own Blender Internal lamp values onto
    lights already in this scene -- the repair for plain File > Append.

    Blender's own append machinery converts 2.79 lamp strengths to its
    modern units (every appended sun arrives at 1.0 W); this reads the
    original .blend again and applies the SAME conversion Halcyon's
    legacy importer uses -- energy (suns x pi, point-family x 4 pi
    squared), colour, falloff and its distances, the per-lamp shadow
    rule, the mode bits, the shadow colour, spot size and blend --
    onto every light whose name matches a lamp in the file, rename
    suffixes like .001 included. Nothing else in the scene is touched.
    """

    bl_idname = 'halcyon.fix_appended_lamps'
    bl_label = "Fix Appended Lamps (2.79)"
    bl_options = {'REGISTER', 'UNDO'}

    filepath: StringProperty(subtype='FILE_PATH')
    filter_glob: StringProperty(default='*.blend', options={'HIDDEN'})

    only_selected: BoolProperty(
        name="Selected Lights Only", default=False,
        description="Fix only the currently selected lights and leave "
                    "every other light in the scene untouched")
    apply_pipeline: BoolProperty(
        name="Scene Pipeline Too", default=False,
        description="Also apply the classic file's render pipeline to "
                    "this scene, exactly as the Halcyon importer would: "
                    "its view transform onto Halcyon's color management "
                    "(2.79's 'Default' view is the sRGB display "
                    "encode), the frame size, and the transparent-sky "
                    "flag. Off by default because it changes scene "
                    "settings beyond the lamps")

    def execute(self, context):
        try:
            sc = blend279.read_legacy_scene(self.filepath, geometry=False)
        except blend279.BlendError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        except OSError as e:
            self.report({'ERROR'}, f'cannot read {self.filepath}: {e}')
            return {'CANCELLED'}

        warnings = []

        # every lamp in the FILE, by object name and by datablock name.
        # Hidden layers included: the append browser offers those
        # objects too, and model packs stash light rigs on them.
        by_obj, by_data = lamp_pools(sc)
        if not by_obj and not by_data:
            self.report({'WARNING'},
                        f'{os.path.basename(self.filepath)}: no lamps '
                        'in the file; nothing to fix')
            return {'CANCELLED'}

        # the lights to fix: the scene's, or just the selection's
        pool = context.selected_objects if self.only_selected \
            else context.scene.objects
        targets = [ob for ob in pool
                   if getattr(ob, 'type', None) == 'LIGHT'
                   and getattr(ob, 'data', None) is not None]
        if not targets:
            self.report({'WARNING'},
                        'no lights selected (turn Selected Lights Only '
                        'off to fix the whole scene)'
                        if self.only_selected else
                        'no lights in the scene -- append the lamps '
                        'first, then run this')
            return {'CANCELLED'}

        receipts, unmatched, by_obj, by_data = fix_lights_from_parsed(
            sc, targets, warnings)
        if unmatched:
            warnings.append(
                f'{len(unmatched)} light(s) matched no lamp in the file '
                'and were left untouched: '
                + ', '.join(sorted(unmatched)[:12])
                + ('...' if len(unmatched) > 12 else ''))
            warnings.append(
                '  the file offers these lamp names: '
                + ', '.join(sorted(by_obj)[:12])
                + ('...' if len(by_obj) > 12 else ''))

        if self.apply_pipeline:
            _apply_color_management(context.scene)
            _apply_scene_pipeline(context.scene, sc, warnings)

        # the version-stamped log: 'did it run, from which build' must
        # never be a guess (the every-sun-reads-1.0 field mystery)
        try:
            from .version import VERSION
            ver = '.'.join(str(v) for v in VERSION)
        except Exception:                                       # noqa: BLE001
            ver = '?'
        base = os.path.splitext(
            os.path.basename(self.filepath))[0] or 'Legacy'
        stage = (f'Halcyon {ver}: {len(receipts)} of {len(targets)} '
                 f'scene lights fixed from the file\'s '
                 f'{len(by_obj)} lamps'
                 + (f'; {len(unmatched)} unmatched' if unmatched else '')
                 + ('; scene pipeline applied' if self.apply_pipeline
                    else ''))
        try:
            from . import fault_note
            fault_note(f'lamp fix completed ({stage})', key='lampfix')
        except Exception:                                       # noqa: BLE001
            pass
        for line in [stage] + receipts + warnings:
            safe_print(f'[halcyon lamp fix] {line}')
        write_log_text(f'{base} lamp fix log',
                       [stage] + receipts + warnings)
        if receipts:
            self.report({'INFO'},
                        f'{stage}. Receipts: text editor > '
                        f"'{base} lamp fix log'")
        else:
            self.report({'WARNING'},
                        f'{stage} -- no names matched; the log in '
                        f"text editor > '{base} lamp fix log' lists "
                        'what the file offers')
        return {'FINISHED'}


def menu_func(self, context):
    self.layout.operator(
        'halcyon.append_legacy',
        text="Legacy .blend (2.79 and earlier) — Halcyon materials")


CLASSES = (HALCYON_OT_append_legacy, HALCYON_OT_fix_appended_lamps)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    bpy.types.TOPBAR_MT_file_import.append(menu_func)


def unregister():
    try:
        bpy.types.TOPBAR_MT_file_import.remove(menu_func)
    except Exception:                                           # noqa: BLE001
        pass
    for c in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(c)
        except Exception:                                       # noqa: BLE001
            pass
