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
    """Ensure and return the triangulated loops."""
    try:
        mesh.calc_loop_triangles()
    except AttributeError:
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


# ------------------------------------------------------------------ objects


def evaluated_meshes(depsgraph):
    """Yield (object, evaluated mesh, matrix, is_temporary).

    The instance list is snapshotted first: `to_mesh()` and `to_mesh_clear()`
    allocate and free depsgraph-owned data, and doing that while the
    object_instances iterator is still live can invalidate it under us.
    """
    try:
        instances = [(i.object, i.matrix_world.copy(), i.show_self)
                     for i in depsgraph.object_instances]
    except Exception:                                           # noqa: BLE001
        return
    for ob, matrix_world, show_self in instances:
        if ob is None or ob.type not in {'MESH', 'CURVE', 'SURFACE', 'FONT',
                                         'META'}:
            continue
        if not show_self:
            continue
        try:
            me = ob.to_mesh()
        except Exception:                                       # noqa: BLE001
            continue
        if me is None:
            continue
        yield ob, me, matrix_world, True


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


def image_pixels(image):
    """(H,W,4) float32, bottom row first (Blender's own layout).

    Pulling pixels out of Blender copies the whole buffer through the Python
    API, which for a large texture is tens of megabytes per call. Doing that for
    every image on every frame of an animation is pure waste, since the pixels
    almost never change -- so the result is cached, keyed on everything that
    would alter it.
    """
    import numpy as np
    if image is None:
        return None
    key = None
    try:
        src = getattr(image, 'source', 'FILE')
        if src not in ('SEQUENCE', 'MOVIE'):        # these change per frame
            key = (image.name_full, tuple(image.size), src,
                   getattr(image, 'filepath_raw', ''),
                   bool(getattr(image, 'is_dirty', False)),
                   getattr(image, 'colorspace_settings', None)
                   and image.colorspace_settings.name)
            hit = _PIXEL_CACHE.get(key)
            if hit is not None:
                return hit
    except Exception:                                           # noqa: BLE001
        key = None
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
            _PIXEL_CACHE[key] = out
            _PIXEL_ORDER.append(key)
            while len(_PIXEL_ORDER) > _PIXEL_LIMIT:
                _PIXEL_CACHE.pop(_PIXEL_ORDER.pop(0), None)
        return out
    except Exception:                                           # noqa: BLE001
        return None


# ------------------------------------------------------------- node add menu


def register_node_menu(draw_fn):
    """4.0+ removed nodeitems_utils; append to the Add menu instead."""
    import bpy as _bpy
    for menu_name in ('NODE_MT_add', 'NODE_MT_category_shader_output'):
        menu = getattr(_bpy.types, menu_name, None)
        if menu is not None:
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
