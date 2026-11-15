"""Thin shims over the parts of the Blender API that have moved recently.

Written so that a rename in a future 5.x point release degrades to a warning
rather than a stack trace at registration time.
"""

import bpy

VERSION = bpy.app.version


def at_least(*ver):
    return VERSION >= tuple(ver)


# ------------------------------------------------------------------- meshes


def loop_triangles(mesh):
    """Ensure and return the triangulated loops.

    `calc_loop_triangles` went away in 4.1 and the cache is built on access
    now, so its absence is expected. Anything else it may raise -- a mesh that
    is temporary, or read-only, or owned by the depsgraph rather than by us --
    is equally not fatal, because the property below is what we actually want.
    """
    try:
        mesh.calc_loop_triangles()
    except Exception:                                           # noqa: BLE001
        pass
    return mesh.loop_triangles


def ensure_normals(mesh):
    """Split (corner) normals, across the 4.1 API break.

    `calc_normals_split` was removed in 4.1; `corner_normals` is the modern
    accessor and is computed automatically.
    """
    for name in ('calc_normals_split', 'calc_normals'):
        fn = getattr(mesh, name, None)
        if fn is not None:
            try:
                fn()
            except Exception:                                   # noqa: BLE001
                pass
    return getattr(mesh, 'corner_normals', None)


def corner_normal_array(mesh, count):
    import numpy as np
    cn = getattr(mesh, 'corner_normals', None)
    out = np.zeros(count * 3, dtype=np.float32)
    if cn is not None and len(cn):
        try:
            cn.foreach_get('vector', out)
            return out.reshape(-1, 3)
        except Exception:                                       # noqa: BLE001
            pass
    loops = mesh.loops
    try:
        loops.foreach_get('normal', out)
        return out.reshape(-1, 3)
    except Exception:                                           # noqa: BLE001
        return None


def uv_layers(mesh):
    return getattr(mesh, 'uv_layers', None) or []


def color_layers(mesh):
    """Colour attributes, whatever they are called in this build."""
    for attr in ('color_attributes', 'vertex_colors'):
        layers = getattr(mesh, attr, None)
        if layers:
            return layers
    return []


def _edge_attr_bools(mesh, name, count):
    """A boolean edge attribute by name, or None. 4.x moved several
    edge flags (freestyle marks above all) into generic attributes."""
    import numpy as np
    try:
        attrs = getattr(mesh, 'attributes', None)
        a = attrs.get(name) if attrs is not None else None
        if a is None or getattr(a, 'domain', '') != 'EDGE':
            return None
        out = np.zeros(count, bool)
        a.data.foreach_get('value', out)
        return out
    except Exception:                                           # noqa: BLE001
        return None


def edge_ink_arrays(mesh):
    """R220: the artist's marked edges, as (indices (E,2) int32,
    flags (E,) uint8) -- or None when nothing is marked.

    Flag bits: 1 = UV seam, 2 = marked sharp, 4 = creased (> 0),
    8 = Freestyle edge mark. Each source is read down whichever road
    this Blender offers -- edge booleans, or the attribute the flag
    moved into -- and a mesh with no marks costs one boolean sweep and
    returns None, so unmarked scenes pay nothing.
    """
    import numpy as np
    edges = getattr(mesh, 'edges', None)
    n = len(edges) if edges is not None else 0
    if not n:
        return None
    flags = np.zeros(n, np.uint8)
    try:
        b = np.zeros(n, bool)
        edges.foreach_get('use_seam', b)
        flags |= b.astype(np.uint8) * 1
    except Exception:                                           # noqa: BLE001
        pass
    try:
        b = np.zeros(n, bool)
        edges.foreach_get('use_edge_sharp', b)
        flags |= b.astype(np.uint8) * 2
    except Exception:                                           # noqa: BLE001
        pass
    crease = None
    try:
        attrs = getattr(mesh, 'attributes', None)
        a = attrs.get('crease_edge') if attrs is not None else None
        if a is not None and getattr(a, 'domain', '') == 'EDGE':
            crease = np.zeros(n, np.float32)
            a.data.foreach_get('value', crease)
    except Exception:                                           # noqa: BLE001
        crease = None
    if crease is None:
        try:
            cl = getattr(mesh, 'edge_creases', None)
            if cl:
                crease = np.zeros(n, np.float32)
                cl[0].data.foreach_get('value', crease)
        except Exception:                                       # noqa: BLE001
            crease = None
    if crease is not None:
        flags |= (crease > 1e-6).astype(np.uint8) * 4
    fs = _edge_attr_bools(mesh, 'freestyle_edge', n)
    if fs is None:
        try:
            b = np.zeros(n, bool)
            edges.foreach_get('use_freestyle_mark', b)
            fs = b
        except Exception:                                       # noqa: BLE001
            fs = None
    if fs is not None:
        flags |= fs.astype(np.uint8) * 8
    if not flags.any():
        return None
    idx = np.empty(n * 2, np.int32)
    edges.foreach_get('vertices', idx)
    keep = flags != 0
    return idx.reshape(-1, 2)[keep], flags[keep]


# ------------------------------------------------------------------ objects


# object types that carry surfaces we can turn into triangles
GEOMETRY_TYPES = frozenset({'MESH', 'CURVE', 'SURFACE', 'FONT', 'META'})

# object types that plainly have geometry in them but that we cannot convert.
# Naming them is the difference between "unsupported, and here is what it was"
# and an object that silently is not in the picture.
UNCONVERTIBLE_TYPES = frozenset({'GPENCIL', 'GREASEPENCIL', 'CURVES',
                                 'POINTCLOUD', 'VOLUME'})

# types that convert by *building* a mesh rather than by handing over one the
# depsgraph already made. These are the ones the lifetime rules below exist
# for: a mesh object gives back geometry that outlives the call, whereas a
# text, curve, surface or metaball gives back a temporary that the next
# `to_mesh()` on the same object -- or any `to_mesh_clear()` -- destroys.
ALLOCATING_TYPES = frozenset({'CURVE', 'SURFACE', 'FONT', 'META'})


def uses_nodes(datablock):
    """`use_nodes` without the 6.0 deprecation flood.

    Blender 5.2 deprecates Material.use_nodes and World.use_nodes ahead of
    their 6.0 removal, and warns on EVERY read -- ten materials times every
    export made the console unreadable and buried real errors. The
    semantics still matter in 5.x (a legacy material with the toggle off
    renders from its flat settings), so the value is read, with only the
    deprecation warning suppressed. When the attribute finally goes,
    absent means True -- which is exactly what its removal will mean.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        return bool(getattr(datablock, 'use_nodes', True))


def enable_nodes(datablock):
    """Turn node mode on where the attribute still exists, silently.

    In 6.0 the attribute disappears because nodes become the only mode --
    at which point this is correctly a no-op.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        if hasattr(datablock, 'use_nodes') and not datablock.use_nodes:
            datablock.use_nodes = True


#: object type -> the SpaceView3D switch of the viewport's Object Types
#: Visibility popover that hides it
_VIEW_TYPE_SWITCH = {
    'MESH': 'show_object_viewport_mesh', 'CURVE': 'show_object_viewport_curve',
    'SURFACE': 'show_object_viewport_surf', 'META': 'show_object_viewport_meta',
    'FONT': 'show_object_viewport_font',
    'CURVES': 'show_object_viewport_curves',
    'POINTCLOUD': 'show_object_viewport_pointcloud',
    'VOLUME': 'show_object_viewport_volume',
    'GREASEPENCIL': 'show_object_viewport_grease_pencil',
    'GPENCIL': 'show_object_viewport_grease_pencil',
    'ARMATURE': 'show_object_viewport_armature',
    'LATTICE': 'show_object_viewport_lattice',
    'EMPTY': 'show_object_viewport_empty',
    'LIGHT': 'show_object_viewport_light',
    'LIGHT_PROBE': 'show_object_viewport_light_probe',
    'CAMERA': 'show_object_viewport_camera',
    'SPEAKER': 'show_object_viewport_speaker'}


def view_type_shown(space, kind):
    """The viewport's own switch for objects of type `kind` (True when
    the space has none)."""
    name = _VIEW_TYPE_SWITCH.get(str(kind))
    return True if name is None else bool(getattr(space, name, True))


def view_type_switches(space):
    """Every such switch of `space`, for a fingerprint."""
    return tuple(bool(getattr(space, name, True))
                 for name in sorted(set(_VIEW_TYPE_SWITCH.values())))


def in_view(inst, space):
    """False when the viewport `space` (a SpaceView3D) does not show this
    depsgraph instance; True with no space.

    R254: a viewport depsgraph instance's show_self stays True when the
    object's collection is hidden with the outliner's eye, when Local
    View leaves the object out, when a local collection or the viewport's
    object-type filter hides it. What changes is the per-viewport test
    Blender's own engines make (BKE_object_is_visible_in_viewport, here
    Object.visible_in_viewport_get).

    An INSTANCE follows its instancer, not its source: Blender hands its
    engines a copy of the instanced object that carries the instancer's
    visibility, so a library collection kept in the scene with its eye
    closed still shows through the Empty that instances it. Python sees
    the source's own answer on `inst.object` (measured in 5.2.2: False for
    such an instance), so the instancer (`inst.parent`) is asked instead,
    and the viewport's type switch is read for the instanced object's own
    type. An instancer hidden only by the switch of ITS type (Empties
    filtered out) still shows its instances, as in Blender. An object that
    cannot answer is shown."""
    if space is None:
        return True
    try:
        ob = inst.object
        if not getattr(inst, 'is_instance', False):
            return bool(ob.visible_in_viewport_get(space))
        if not view_type_shown(space, getattr(ob, 'type', '')):
            return False
        parent = getattr(inst, 'parent', None)
        if parent is None or parent.visible_in_viewport_get(space):
            return True
        return not view_type_shown(space, getattr(parent, 'type', ''))
    except Exception:                                           # noqa: BLE001
        return True


def evaluated_meshes(depsgraph, skip=None, space=None):
    """Yield (object, evaluated mesh, matrix, is_temporary), one at a time.

    `space`: the rendered viewport's SpaceView3D, or None (in_view).

    `skip(original, matrix)` -> True short-circuits an object BEFORE its
    depsgraph evaluation: the caller gets (original, None, matrix,
    False) and serves the geometry from its own cache (R171's export
    cache). The whole point is skipping `evaluated_get` + `to_mesh` --
    the measured cost of an unchanged 171-object export.

    THREE Blender rules govern this, and breaking any of them is a hard
    crash rather than an exception:

    * The instance iterator must not be live while `to_mesh()` runs, because
      converting allocates depsgraph-owned data and can invalidate it under
      us. So the instance list is snapshotted first.
    * NOTHING reached through a DepsgraphObjectInstance may outlive the
      iteration step -- not even the EVALUATED Object it points at. The
      old snapshot kept those evaluated pointers and called `to_mesh()`
      on them afterwards; the first conversion that allocates (a curve,
      a text) can reallocate evaluated storage and dangle the pointers
      held for every LATER item. It survived for months because a
      dangling pointer usually still reads intact memory -- until a
      render worker and a shadow-map pool churned the allocator hard
      enough to recycle the block, which is why the field crash landed
      only "while the renderer is actually running". The faulthandler
      log named this exact line. The snapshot therefore keeps each
      instance's ORIGINAL object -- a stable, bmain-owned ID -- and
      re-resolves the evaluated object through `evaluated_get()` at use
      time, so every `to_mesh()` runs on a pointer the CURRENT
      depsgraph just handed out.
    * A mesh from `to_mesh()` belongs to the object, not to the caller. The
      next `to_mesh()` on that same object frees it, and so does
      `to_mesh_clear()`. Converting every object up front and reading them
      afterwards -- which is what a `list()` around this generator does --
      therefore reads freed memory the moment one object appears twice, which
      is exactly what an instanced text or curve is.

    So each mesh is freed here, on the way to the next object, and the caller
    must consume this lazily and must not free anything itself. A mesh object
    survived the old ordering by luck: it hands back geometry the depsgraph
    already owns, so nothing was ever actually freed. Text, curves, surfaces
    and metaballs are the ones that allocate, and they are the ones it broke.
    """
    try:
        instances = []
        for i in depsgraph.object_instances:
            ob = i.object
            if ob is None or not i.show_self:
                continue
            if ob.type not in GEOMETRY_TYPES:
                continue
            if not in_view(i, space):
                continue
            # .original is the bmain-owned datablock: the ONE reference
            # that stays valid after this iteration step ends
            orig = getattr(ob, 'original', None) or ob
            instances.append((orig, i.matrix_world.copy()))
    except Exception:                                           # noqa: BLE001
        return
    live = None
    try:
        for orig, matrix_world in instances:
            if live is not None:
                free_mesh(live)
                live = None
            if skip is not None:
                try:
                    if skip(orig, matrix_world):
                        yield orig, None, matrix_world, False
                        continue
                except Exception:                               # noqa: BLE001
                    pass
            try:
                getter = getattr(orig, 'evaluated_get', None)
                ev = getter(depsgraph) if getter is not None else orig
                me = ev.to_mesh()
            except Exception:                                   # noqa: BLE001
                continue
            if me is None:
                continue
            live = ev
            yield ev, me, matrix_world, True
    finally:
        # runs on exhaustion, on an early break, and on close()
        if live is not None:
            free_mesh(live)


def unconvertible(depsgraph, space=None):
    """Names and types of visible objects we had to leave out."""
    out = []
    try:
        seen = set()
        for i in depsgraph.object_instances:
            ob = i.object
            if ob is None or not i.show_self:
                continue
            if not in_view(i, space):
                continue
            if ob.type in UNCONVERTIBLE_TYPES and ob.name not in seen:
                seen.add(ob.name)
                out.append((ob.name, ob.type))
    except Exception:                                           # noqa: BLE001
        pass
    return out


def free_mesh(ob):
    try:
        ob.to_mesh_clear()
    except Exception:                                           # noqa: BLE001
        pass


# ------------------------------------------------------------------ images


_PIXEL_CACHE = {}
_PIXEL_ORDER = []
_PIXEL_LIMIT = 24


def clear_image_cache():
    _PIXEL_CACHE.clear()
    _PIXEL_ORDER.clear()


def _file_stamp(image):
    """(st_mtime_ns, st_size) of the file an unpacked FILE image reads,
    or None (packed, generated, a path that is not one file).

    R254: the stamp is what moves when the file is rewritten on disk and
    Reload is pressed; nothing else on the Image does. Nanoseconds and
    the size, so a rewrite within the same second is seen."""
    try:
        if str(getattr(image, 'source', '')) != 'FILE' or \
                getattr(image, 'packed_file', None) is not None:
            return None
        raw = getattr(image, 'filepath_raw', '') or ''
        if not raw:
            return None
        import os
        path = bpy.path.abspath(raw, library=getattr(image, 'library', None))
        st = os.stat(path)
        return (int(st.st_mtime_ns), int(st.st_size))
    except Exception:                                           # noqa: BLE001
        return None


def _pixel_forget(name, keep=None):
    """Drop every cached generation of the image `name` but `keep`."""
    for k in [k for k in _PIXEL_CACHE if k[0] == name and k != keep]:
        _PIXEL_CACHE.pop(k, None)
        if k in _PIXEL_ORDER:
            _PIXEL_ORDER.remove(k)


def image_pixels(image):
    """(H,W,4) float32, bottom row first (Blender's own layout).

    Pulling pixels out of Blender copies the whole buffer through the Python
    API, which for a large texture is tens of megabytes per call. Doing that for
    every image on every frame of an animation is pure waste, since the pixels
    almost never change -- so the result is cached, keyed on everything that
    would alter it.

    R254: 'everything' was the name, size, source, path, the dirty flag and
    the colour space, and none of those moves when a file is rewritten and
    reloaded, or when an image that is already dirty is painted again -- a
    projector's gobo, or any texture, stayed as first read until Blender was
    restarted. Now a clean image is keyed on its file's stamp as well
    (_file_stamp), and a DIRTY image is read every time: Blender says
    nothing when its pixels change (a pixel write fires no depsgraph update),
    so reading them is the only way to know. When the fresh read equals the
    cached copy the cached ARRAY is returned, so the caches downstream that
    key on its identity stay warm; one generation per image is kept.
    """
    import numpy as np
    if image is None:
        return None
    key = None
    dirty = False
    try:
        src = getattr(image, 'source', 'FILE')
        if src not in ('SEQUENCE', 'MOVIE'):        # these change per frame
            dirty = bool(getattr(image, 'is_dirty', False))
            key = (image.name_full, tuple(image.size), src,
                   getattr(image, 'filepath_raw', ''),
                   'dirty' if dirty else _file_stamp(image),
                   getattr(image, 'colorspace_settings', None)
                   and image.colorspace_settings.name)
            if not dirty:
                hit = _PIXEL_CACHE.get(key)
                if hit is not None:
                    return hit
    except Exception:                                           # noqa: BLE001
        key = None
        dirty = False
    # foreach_get on an image with no pixel buffer reads from a null pointer
    if not getattr(image, 'has_data', True):
        try:
            image.update()
        except Exception:                                       # noqa: BLE001
            return None
        if not getattr(image, 'has_data', False):
            return None
    try:
        w, h = image.size
        if w == 0 or h == 0:
            return None
        buf = np.empty(w * h * 4, dtype=np.float32)
        image.pixels.foreach_get(buf)
        out = buf.reshape(h, w, 4)
        if key is not None:
            if dirty:
                prev = _PIXEL_CACHE.get(key)
                if prev is not None and prev.shape == out.shape and \
                        np.array_equal(prev, out):
                    return prev
            _pixel_forget(key[0])
            _PIXEL_CACHE[key] = out
            _PIXEL_ORDER.append(key)
            while len(_PIXEL_ORDER) > _PIXEL_LIMIT:
                _PIXEL_CACHE.pop(_PIXEL_ORDER.pop(0), None)
        return out
    except Exception:                                           # noqa: BLE001
        return None


# ------------------------------------------------------------- node add menu


def register_node_menu(draw_fn):
    """4.0+ removed nodeitems_utils; draw into the Add menu directly.

    R242: PREPENDED, so Halcyon is the first entry of the Add menu
    rather than the last (the field's ask); a Blender build without
    prepend on the menu class falls back to append."""
    import bpy as _bpy
    for menu_name in ('NODE_MT_add', 'NODE_MT_category_shader_output'):
        menu = getattr(_bpy.types, menu_name, None)
        if menu is not None:
            if hasattr(menu, 'prepend'):
                menu.prepend(draw_fn)
            else:
                menu.append(draw_fn)
            return menu_name
    return None


def unregister_node_menu(draw_fn, menu_name):
    import bpy as _bpy
    menu = getattr(_bpy.types, menu_name or '', None)
    if menu is not None:
        try:
            menu.remove(draw_fn)
        except Exception:                                       # noqa: BLE001
            pass


# ------------------------------------------------------------------- curves


def sample_curve(mapping, index, samples=256):
    """Bake a CurveMapping channel into a LUT."""
    import numpy as np
    try:
        mapping.update()
    except Exception:                                           # noqa: BLE001
        pass
    xs = np.linspace(0.0, 1.0, samples)
    curve = mapping.curves[index]
    out = np.empty(samples, np.float32)
    for i, x in enumerate(xs):
        try:
            out[i] = mapping.evaluate(curve, float(x))
        except Exception:                                       # noqa: BLE001
            try:
                out[i] = curve.evaluate(float(x))
            except Exception:                                   # noqa: BLE001
                out[i] = x
    return out


def sample_ramp(ramp, samples=256):
    import numpy as np
    out = np.empty((samples, 4), np.float32)
    for i in range(samples):
        out[i] = ramp.evaluate(i / (samples - 1))
    return out
