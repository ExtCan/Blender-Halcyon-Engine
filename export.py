"""Turn an evaluated Blender scene into the plain dataclasses the renderer eats.

Nothing downstream of this file imports bpy. That boundary is what makes the
renderer testable headlessly, and it is why node trees are flattened into dicts
here rather than walked live during shading.
"""

import numpy as np

from . import compat
from .core.scene import (Camera, ImageBuffer, Light, Material, MeshData,
                         ObjectInfo, Scene, World)
from .shaders.compiler import try_compile

SOCKET_KIND = {
    'VALUE': 'VALUE', 'INT': 'VALUE', 'BOOLEAN': 'VALUE',
    'VECTOR': 'VECTOR', 'ROTATION': 'VECTOR',
    'RGBA': 'RGBA', 'SHADER': 'SHADER',
}


def socket_kind(sock):
    return SOCKET_KIND.get(getattr(sock, 'type', 'VALUE'), 'VALUE')


def socket_default(sock):
    kind = socket_kind(sock)
    if kind == 'SHADER':
        return None
    v = getattr(sock, 'default_value', None)
    if v is None:
        return None
    try:
        if hasattr(v, '__len__') and not isinstance(v, str):
            return [float(x) for x in v]
        return float(v)
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------- node property table

NODE_PROPS = {
    'ShaderNodeMixRGB': ('blend_type', 'use_clamp'),
    'ShaderNodeMix': ('data_type', 'blend_type', 'clamp_factor', 'clamp_result',
                      'factor_mode'),
    'ShaderNodeMath': ('operation', 'use_clamp'),
    'ShaderNodeVectorMath': ('operation',),
    'ShaderNodeMapping': ('vector_type',),
    'ShaderNodeVectorRotate': ('rotation_type', 'invert'),
    'ShaderNodeTexImage': ('interpolation', 'extension', 'projection',
                           'projection_blend'),
    'ShaderNodeTexEnvironment': ('interpolation', 'projection'),
    'ShaderNodeTexGradient': ('gradient_type',),
    'ShaderNodeTexSky': ('sky_type', 'sun_elevation', 'sun_rotation', 'turbidity',
                         'ground_albedo', 'sun_disc', 'sun_size', 'sun_intensity',
                         'altitude', 'air_density', 'dust_density',
                         'ozone_density'),
    'ShaderNodeTexWave': ('wave_type', 'wave_profile', 'bands_direction',
                          'rings_direction'),
    'ShaderNodeTexVoronoi': ('distance', 'feature', 'voronoi_dimensions'),
    'ShaderNodeTexMagic': ('turbulence_depth',),
    'ShaderNodeTexBrick': ('offset', 'offset_frequency', 'squash',
                           'squash_frequency'),
    'ShaderNodeTexNoise': ('noise_dimensions', 'noise_type', 'normalize'),
    'ShaderNodeTexMusgrave': ('musgrave_type', 'musgrave_dimensions'),
    'ShaderNodeBsdfGlossy': ('distribution',),
    'ShaderNodeBsdfAnisotropic': ('distribution',),
    'ShaderNodeBsdfGlass': ('distribution',),
    'ShaderNodeBsdfRefraction': ('distribution',),
    'ShaderNodeBsdfPrincipled': ('distribution', 'subsurface_method'),
    'ShaderNodeSubsurfaceScattering': ('falloff',),
    'ShaderNodeNormalMap': ('space', 'uv_map'),
    'ShaderNodeTangent': ('direction_type', 'axis', 'uv_map'),
    'ShaderNodeUVMap': ('uv_map', 'from_instancer'),
    'ShaderNodeAttribute': ('attribute_name', 'attribute_type'),
    'ShaderNodeVertexColor': ('layer_name',),
    'ShaderNodeBump': ('invert',),
    'ShaderNodeDisplacement': ('space',),
    'ShaderNodeVectorDisplacement': ('space',),
    'ShaderNodeMapRange': ('interpolation_type', 'data_type', 'clamp'),
    'ShaderNodeClamp': ('clamp_type',),
    'ShaderNodeSeparateColor': ('mode',),
    'ShaderNodeCombineColor': ('mode',),
    'ShaderNodeBsdfToon': ('component',),
    'ShaderNodeOutputMaterial': ('target',),
    'HALCYON_ShaderNode': ('model', 'toon_steps', 'wire_size'),
    'ShaderNodeTexGabor': ('gabor_type',),
    'HALCYON_CodeNode': ('language', 'as_surface', 'source_text'),
    # the retro utilities. Dither's pattern and Depth Cue's falloff were
    # MISSING from this table from the day those nodes shipped: the
    # evaluator read the prop, the exporter never copied it, and both
    # dropdowns silently rendered as their defaults. A test now walks the
    # DISPATCH handlers' _prop reads against this table so a node cannot
    # ship half-wired again.
    'HALCYON_DitherNode': ('pattern',),
    'HALCYON_DepthCueNode': ('mode',),
    'HALCYON_ScrollNode': ('animate', 'fps'),
    'HALCYON_ScanlinesNode': ('animate',),
    'HALCYON_PaletteNode': ('palette',),
    'HALCYON_ColorCycleNode': ('animate',),
    'HALCYON_FlipbookNode': ('animate',),
    'HALCYON_UVWaveNode': ('animate',),
}

try:
    from .nodes.pattern_nodes import NODE_PROPS as _PATTERN_PROPS
    NODE_PROPS.update(_PATTERN_PROPS)
except Exception:                                               # noqa: BLE001
    pass

RAMP_NODES = {'ShaderNodeValToRGB'}
CURVE_NODES = {'ShaderNodeRGBCurve': 4, 'ShaderNodeFloatCurve': 1,
               'ShaderNodeVectorCurve': 3}
VALUE_NODES = {'ShaderNodeRGB', 'ShaderNodeValue', 'ShaderNodeNormal'}


def _node_props(node, images, programs, warnings):
    props = {}
    for name in NODE_PROPS.get(node.bl_idname, ()):
        if hasattr(node, name):
            v = getattr(node, name)
            try:
                if hasattr(v, '__len__') and not isinstance(v, str):
                    v = [float(x) for x in v]
            except TypeError:
                pass
            props[name] = v

    img = getattr(node, 'image', None)
    if img is not None:
        key = img.name_full
        if key not in images:
            px = compat.image_pixels(img)
            if px is not None:
                cs = 'Non-Color'
                try:
                    cs = img.colorspace_settings.name
                except Exception:                               # noqa: BLE001
                    pass
                images[key] = ImageBuffer(name=key, pixels=px, colorspace=cs)
        props['image'] = key if key in images else None

    if node.bl_idname in RAMP_NODES and hasattr(node, 'color_ramp'):
        props['lut'] = compat.sample_ramp(node.color_ramp).tolist()

    n_curves = CURVE_NODES.get(node.bl_idname)
    if n_curves and hasattr(node, 'mapping'):
        cols = []
        for i in range(min(n_curves, len(node.mapping.curves))):
            cols.append(compat.sample_curve(node.mapping, i))
        if node.bl_idname == 'ShaderNodeRGBCurve' and len(cols) == 4:
            lut = np.stack([cols[3], cols[0], cols[1], cols[2]], axis=1)
        elif len(cols) == 1:
            lut = cols[0]
        else:
            lut = np.stack(cols, axis=1)
        props['lut'] = np.asarray(lut, np.float32).tolist()
        if node.bl_idname == 'ShaderNodeVectorCurve':
            props['range_min'] = [node.mapping.clip_min_x] * 3
            props['range_max'] = [node.mapping.clip_max_x] * 3

    if node.bl_idname in VALUE_NODES:
        outs = list(node.outputs)
        if outs and hasattr(outs[0], 'default_value'):
            props['value'] = socket_default(outs[0])
        if node.bl_idname == 'ShaderNodeNormal':
            props['direction'] = socket_default(outs[0]) if outs else [0, 0, 1]

    if node.bl_idname == 'HALCYON_CodeNode':
        prog, err = _compile_node(node)
        if prog is not None:
            programs[node.name] = prog
        elif err:
            warnings.append(f"{node.name}: {err}")
    return props


def _compile_node(node):
    src = ''
    text = getattr(node, 'source', None)
    if text is not None and hasattr(text, 'as_string'):
        src = text.as_string()
    if not src:
        src = getattr(node, 'source_text', '') or ''
    if not src.strip():
        return None, None
    lang = getattr(node, 'language', 'GLSL')
    return try_compile(src, lang)


# ------------------------------------------------------------- tree flatten


def serialize_tree(tree, images, programs, warnings, output_type='OUTPUT_MATERIAL',
                   depth=0):
    """Flatten a node tree into plain dicts. Groups recurse."""
    if tree is None or depth > 12:
        return None
    nodes = {}
    output_id = None
    group_output_id = None

    for node in tree.nodes:
        nid = node.name
        idname = node.bl_idname
        if idname == 'NodeFrame':
            continue
        if getattr(node, 'mute', False):
            idname = 'NodeMuted'
        entry = {
            'id': nid,
            'bl_idname': idname,
            'props': _node_props(node, images, programs, warnings)
            if idname != 'NodeMuted' else {},
            'inputs': [],
            'outputs': [],
        }
        for sock in node.inputs:
            link = None
            if sock.is_linked and sock.links:
                lk = sock.links[0]
                from_node = lk.from_node
                try:
                    oi = list(from_node.outputs).index(lk.from_socket)
                except ValueError:
                    oi = 0
                link = [from_node.name, oi]
            entry['inputs'].append({
                'name': sock.name,
                'identifier': getattr(sock, 'identifier', sock.name),
                'type': socket_kind(sock),
                'default': socket_default(sock),
                'link': link,
                'uniform': getattr(sock, 'halcyon_uniform', None),
                'is_image': getattr(sock, 'halcyon_is_image', False),
                'image': getattr(sock, 'halcyon_image_key', None),
            })
        for sock in node.outputs:
            entry['outputs'].append({
                'name': sock.name,
                'identifier': getattr(sock, 'identifier', sock.name),
                'type': socket_kind(sock),
                'key': getattr(sock, 'halcyon_key', None),
            })

        if idname == 'ShaderNodeGroup' and getattr(node, 'node_tree', None):
            sub = serialize_tree(node.node_tree, images, programs, warnings,
                                 output_type=None, depth=depth + 1)
            entry['group'] = sub

        if idname == 'NodeGroupOutput' and getattr(node, 'is_active_output', True):
            group_output_id = nid
        if idname in ('ShaderNodeOutputMaterial', 'ShaderNodeOutputWorld',
                      'ShaderNodeOutputLight'):
            if output_id is None or getattr(node, 'is_active_output', False):
                output_id = nid
        nodes[nid] = entry

    return {'nodes': nodes, 'output': output_id, 'group_output': group_output_id}


# ---------------------------------------------------------------- materials


def export_material(mat, images, warnings):
    m = Material(name=mat.name if mat else 'Default')
    if mat is None:
        return m
    hs = getattr(mat, 'halcyon', None)
    if hs is not None and hs.use_override:
        m.use_override = True
        m.model = hs.model
        m.diffuse = tuple(hs.diffuse)
        m.diffuse_level = hs.diffuse_level
        m.specular = tuple(hs.specular)
        m.specular_level = hs.specular_level
        m.glossiness = hs.glossiness
        m.ambient_level = hs.ambient_level
        m.emission = tuple(hs.emission)
        m.emission_level = hs.emission_level
        m.opacity = hs.opacity
        m.ior = hs.ior
        m.roughness = hs.roughness
        m.anisotropy = hs.anisotropy
        m.aniso_rotation = hs.aniso_rotation
        m.metallic = hs.metallic
        m.reflect_level = hs.reflect_level
        m.two_sided = hs.two_sided
        m.shadeless = hs.shadeless
        m.cast_shadow = hs.cast_shadow
        m.receive_shadow = hs.receive_shadow
        m.wire = hs.wire
    else:
        try:
            m.diffuse = tuple(mat.diffuse_color[:3])
            m.metallic = float(mat.metallic)
            m.roughness = float(mat.roughness)
            m.specular_level = float(getattr(mat, 'specular_intensity', 0.5))
        except Exception:                                       # noqa: BLE001
            pass
    if hs is not None:
        # outside the override branch on purpose: a material shaded as
        # Wireframe by its *node* never went through that branch, so its wire
        # width was stuck at the dataclass default with no way to change it
        m.wire_size = hs.wire_size
    if compat.uses_nodes(mat) and mat.node_tree:
        m.programs = {}
        m.graph = serialize_tree(mat.node_tree, images, m.programs, warnings)
    m.alpha_why = _alpha_reason(mat, m)
    m.has_alpha = m.alpha_why is not None
    return m


#: nodes whose presence in a tree IS alpha, whatever the sockets say
_ALPHA_NODES = {'ShaderNodeBsdfTransparent': 'a Transparent BSDF node',
                'ShaderNodeBsdfGlass': 'a Glass BSDF node',
                'ShaderNodeBsdfRefraction': 'a Refraction BSDF node',
                'ShaderNodeHoldout': 'a Holdout node'}


def _alpha_reason(mat, m):
    """Why this material needs the see-through pass -- or None.

    The rule that matters here: a blend MODE is a policy for handling
    alpha, not evidence that any alpha exists. This function used to
    return True for any `blend_method` other than OPAQUE, and the field
    found what that costs. A Sonic model imported from a game format
    arrived with all 25 of its materials carrying a non-opaque blend
    mode -- as importers routinely set -- so every one of its 1209
    triangles was classified see-through, the depth-buffered pass got
    NOTHING, and a solid character was rendered as a stack of A-buffer
    layers with culling off and no depth writes. Under Sorted Blend
    that is polygon-centroid ordering, which shows exactly the classic
    sorting errors: wedges of one surface punching through another.
    It read as broken depth for three rounds.

    So alpha must be USED, not merely permitted: an opacity below one,
    a transparent/glass/refraction/holdout node, or an Alpha socket
    that is linked or set below one. A material with none of those is
    opaque whatever its blend mode says -- and blending it would have
    blended with alpha 1.0 anyway, so nothing is lost by putting it
    back where hidden-surface removal happens.

    `blend_method` is deliberately not consulted. Halcyon's own alpha
    lives on the master shader's sockets and is independent of EEVEE's
    blend mode (the add-on's Glass template sets Opacity 0.12 with the
    blend mode untouched), so reading the mode in either direction
    would misclassify the engine's own presets.
    """
    if m.opacity < 0.999:
        return f'Opacity {float(m.opacity):.3f}'
    if not m.graph:
        return None
    for node in m.graph['nodes'].values():
        idn = node['bl_idname']
        if idn in _ALPHA_NODES:
            return _ALPHA_NODES[idn] + ' in its tree'
        if idn == 'ShaderNodeBsdfPrincipled':
            for s in node['inputs']:
                if s['name'] != 'Alpha':
                    continue
                if s['link'] is not None:
                    return 'its Principled Alpha socket is linked'
                if s['default'] is not None and float(s['default']) < 0.999:
                    return f'Principled Alpha {float(s["default"]):.3f}'
        if idn == 'HALCYON_ShaderNode':
            # the master shader's own alpha lives on its sockets, and the
            # add-on's templates put it there (Glass: Opacity 0.12) with
            # use_override off -- so m.opacity never sees it. Without this
            # check the engine's own glass presets exported as OPAQUE and
            # skipped the transparent pass entirely. Edge Opacity below
            # 1.0 is see-through at the silhouette even when Opacity is
            # 1.0, so it counts the same way.
            for s in node['inputs']:
                if s['name'] not in ('Opacity', 'Edge Opacity'):
                    continue
                if s['link'] is not None:
                    return f'its {s["name"]} socket is linked'
                if s['default'] is not None and float(s['default']) < 0.999:
                    return f'{s["name"]} {float(s["default"]):.3f}'
    return None


def _tree_has_alpha(mat, m):
    return _alpha_reason(mat, m) is not None


# -------------------------------------------------------------------- mesh


def _mesh_arrays(me, matrix, mat_offset, obj_index):
    """Per-corner (split) vertex buffer for one evaluated mesh."""
    tris = compat.loop_triangles(me)
    n_tris = len(tris)
    if n_tris == 0:
        return None
    compat.ensure_normals(me)

    n_loops = len(me.loops)
    n_verts = len(me.vertices)

    co = np.empty(n_verts * 3, np.float32)
    me.vertices.foreach_get('co', co)
    co = co.reshape(-1, 3)

    lv = np.empty(n_loops, np.int32)
    me.loops.foreach_get('vertex_index', lv)

    nrm = compat.corner_normal_array(me, n_loops)
    if nrm is None:
        vn = np.empty(n_verts * 3, np.float32)
        me.vertices.foreach_get('normal', vn)
        nrm = vn.reshape(-1, 3)[lv]

    uvs = np.zeros((n_loops, 2), np.float32)
    uvs2 = None
    uv_names = []
    layers = compat.uv_layers(me)
    if len(layers):
        buf = np.empty(n_loops * 2, np.float32)
        layers[0].data.foreach_get('uv', buf)
        uvs = buf.reshape(-1, 2)
        uv_names.append(str(getattr(layers[0], 'name', '') or ''))
        if len(layers) > 1:
            buf2 = np.empty(n_loops * 2, np.float32)
            layers[1].data.foreach_get('uv', buf2)
            uvs2 = buf2.reshape(-1, 2)
            uv_names.append(str(getattr(layers[1], 'name', '') or ''))

    cols = np.ones((n_loops, 4), np.float32)
    clayers = compat.color_layers(me)
    if len(clayers):
        try:
            lay = clayers[0]
            domain = getattr(lay, 'domain', 'CORNER')
            n_items = n_loops if domain == 'CORNER' else n_verts
            buf = np.empty(n_items * 4, np.float32)
            lay.data.foreach_get('color', buf)
            arr = buf.reshape(-1, 4)
            cols = arr if domain == 'CORNER' else arr[lv]
        except Exception:                                       # noqa: BLE001
            pass

    lt = np.empty(n_tris * 3, np.int32)
    tris.foreach_get('loops', lt)
    lt = lt.reshape(-1, 3)

    mat_idx = np.zeros(n_tris, np.int32)
    tris.foreach_get('material_index', mat_idx)

    smooth = np.zeros(n_tris, bool)
    try:
        poly_smooth = np.zeros(len(me.polygons), bool)
        me.polygons.foreach_get('use_smooth', poly_smooth)
        pidx = np.empty(n_tris, np.int32)
        tris.foreach_get('polygon_index', pidx)
        smooth = poly_smooth[pidx]
    except Exception:                                           # noqa: BLE001
        pass

    mw = np.asarray(matrix, np.float32)
    pos = co[lv] @ mw[:3, :3].T + mw[:3, 3]
    nmat = np.linalg.inv(mw[:3, :3]).T
    nn = nrm @ nmat.T
    ln = np.linalg.norm(nn, axis=1, keepdims=True)
    nn = nn / np.where(ln < 1e-12, 1.0, ln)

    fn = np.empty(n_tris * 3, np.float32)
    try:
        tris.foreach_get('normal', fn)
        fn = fn.reshape(-1, 3) @ nmat.T
        ln = np.linalg.norm(fn, axis=1, keepdims=True)
        fn = fn / np.where(ln < 1e-12, 1.0, ln)
    except Exception:                                           # noqa: BLE001
        e1 = pos[lt[:, 1]] - pos[lt[:, 0]]
        e2 = pos[lt[:, 2]] - pos[lt[:, 0]]
        fn = np.cross(e1, e2)
        ln = np.linalg.norm(fn, axis=1, keepdims=True)
        fn = fn / np.where(ln < 1e-12, 1.0, ln)

    return dict(verts=pos.astype(np.float32), normals=nn.astype(np.float32),
                uvs=uvs.astype(np.float32),
                uvs2=uvs2.astype(np.float32) if uvs2 is not None else None,
                uv_names=uv_names,
                colors=cols.astype(np.float32), tris=lt.astype(np.int32),
                mat_index=(mat_idx + mat_offset).astype(np.int32),
                obj_index=np.full(n_tris, obj_index, np.int32),
                face_normals=fn.astype(np.float32), smooth=smooth)


# ------------------------------------------------------------------- lights


def export_light(ob, matrix, unit_scale=1.0):
    la = ob.data
    hs = getattr(la, 'halcyon', None)
    mw = np.asarray(matrix, np.float32)
    pos = mw[:3, 3]
    direction = -mw[:3, 2]
    kind = {'POINT': 'POINT', 'SUN': 'SUN', 'SPOT': 'SPOT', 'AREA': 'AREA'}.get(
        la.type, 'POINT')
    lt = Light(type=kind, name=ob.name, position=tuple(pos),
               direction=tuple(direction), color=tuple(la.color[:3]),
               energy=float(la.energy),
               radius=float(getattr(la, 'shadow_soft_size', 0.0)))
    if kind == 'SUN':
        lt.energy = float(la.energy)
        lt.radius = float(getattr(la, 'angle', 0.0))
    if kind == 'SPOT':
        lt.spot_size = float(la.spot_size)
        lt.spot_blend = max(float(la.spot_blend), 1e-3)
    if kind == 'AREA':
        sx = float(getattr(la, 'size', 1.0))
        sy = float(getattr(la, 'size_y', sx)) if la.shape in ('RECTANGLE', 'ELLIPSE') \
            else sx
        lt.area_size = (sx, sy)
        lt.area_shape = la.shape
        lt.area_x = tuple(mw[:3, 0])
        lt.area_y = tuple(mw[:3, 1])
    if hs is not None:
        lt.decay = hs.decay
        lt.decay_start = hs.decay_start
        lt.decay_end = hs.decay_end
        lt.shadow = hs.shadow
        lt.shadow_map_size = hs.shadow_map_size
        lt.shadow_bias = hs.shadow_bias
        lt.shadow_softness = hs.shadow_softness
        lt.shadow_samples = hs.shadow_samples
        lt.shadow_density = hs.shadow_density
        lt.shadow_color = tuple(hs.shadow_color)
        lt.negative = hs.negative
        lt.diffuse_only = hs.diffuse_only
        lt.specular_only = hs.specular_only
        lt.ambient_only = hs.ambient_only
        lt.hotspot = hs.hotspot
        lt.volumetric = hs.volumetric
        lt.exclude_mode = getattr(hs, 'exclude_mode', 'EXCLUDE')
        coll = getattr(hs, 'exclude_collection', None)
        if coll is not None:
            lt.exclude_names = {o.name for o in coll.all_objects}
        ck = getattr(hs, 'cookie', None)
        if ck is not None and kind in ('SPOT', 'SUN'):
            px = compat.image_pixels(ck)
            if px is not None:
                lt.cookie = ImageBuffer(name=ck.name_full, pixels=px,
                                        colorspace='Non-Color')
                lt.cookie_strength = float(getattr(hs, 'cookie_strength',
                                                   1.0))
                lt.cookie_scale = float(getattr(hs, 'cookie_scale', 10.0))
    else:
        lt.shadow = 'MAP' if getattr(la, 'use_shadow', True) else 'NONE'
    if not getattr(la, 'use_shadow', True):
        lt.shadow = 'NONE'
    # the lamp's own axes, so a projected texture turns with the object
    lt.frame_x = tuple(mw[:3, 0] / max(float(np.linalg.norm(mw[:3, 0])),
                                       1e-9))
    lt.frame_y = tuple(mw[:3, 1] / max(float(np.linalg.norm(mw[:3, 1])),
                                       1e-9))
    return lt


# ------------------------------------------------------------------ the job


_KINDS = {'MESH': "Mesh", 'FONT': "Text", 'CURVE': "Curve",
          'SURFACE': "Surface", 'META': "Metaball", 'CURVES': "Hair curves",
          'POINTCLOUD': "Point cloud", 'VOLUME': "Volume",
          'GPENCIL': "Grease pencil", 'GREASEPENCIL': "Grease pencil"}


def _kind(ob):
    """What to call an object in a message, in the user's words not Blender's."""
    return _KINDS.get(getattr(ob, 'type', ''), "Object")


def export_scene(depsgraph, settings, warnings=None):
    """Evaluated depsgraph -> Scene."""
    warnings = warnings if warnings is not None else []
    bscene = depsgraph.scene
    images = {}
    materials = []
    mat_lookup = {}
    objects = []
    lights = []
    parts = []

    # consumed lazily on purpose: the generator frees each converted mesh on
    # the way to the next one, and wrapping it in list() would read every one
    # of them after it had been freed. See compat.evaluated_meshes.
    for ob, me, matrix, temp in compat.evaluated_meshes(depsgraph):
        try:
            slots = list(me.materials) if me.materials else [None]
            for slot in slots:
                key = slot.name_full if slot else '__default__'
                if key not in mat_lookup:
                    mat_lookup[key] = len(materials)
                    materials.append(export_material(slot, images, warnings))
            remap = np.array([mat_lookup[(s.name_full if s else '__default__')]
                              for s in slots], np.int32)
            obj_index = len(objects)
            data = _mesh_arrays(me, matrix, 0, obj_index)
        except Exception as exc:                                # noqa: BLE001
            # one object that will not convert is not a reason to lose the
            # frame. Say which one it was, in the console and in the UI, and
            # carry on with the rest of the scene.
            import traceback
            traceback.print_exc()
            warnings.append(f"{_kind(ob)} '{getattr(ob, 'name', '?')}' could "
                            f"not be exported ({type(exc).__name__}: {exc}) "
                            f"and is missing from the render")
            continue
        if data is None:
            if getattr(ob, 'type', 'MESH') in compat.ALLOCATING_TYPES:
                # the usual reason a text object is invisible: it converted,
                # but to outlines rather than to faces
                warnings.append(
                    f"{_kind(ob)} '{ob.name}' has no faces to render. Give it "
                    f"a Fill Mode (Object Data ▸ Geometry) or an Extrude, or "
                    f"convert it to a mesh")
            continue
        idx = np.clip(data['mat_index'], 0, len(remap) - 1)
        data['mat_index'] = remap[idx]
        parts.append(data)
        mw = np.asarray(matrix, np.float32)
        objects.append(ObjectInfo(
            name=ob.name, location=tuple(mw[:3, 3]), matrix_world=mw,
            color=tuple(ob.color) if hasattr(ob, 'color') else (1, 1, 1, 1),
            index=int(getattr(ob, 'pass_index', 0)),
            random=float(abs(hash(ob.name)) % 10000) / 10000.0,
            visible_camera=getattr(ob, 'visible_camera', True),
            cast_shadow=getattr(ob, 'visible_shadow', True),
            holdout=getattr(ob, 'is_holdout', False)))

    for name, kind in compat.unconvertible(depsgraph):
        warnings.append(f"{_KINDS.get(kind, kind.title())} '{name}' is not a "
                        f"surface Halcyon can render. Convert it to a mesh "
                        f"to include it")

    for inst in depsgraph.object_instances:
        ob = inst.object
        if ob is None or ob.type != 'LIGHT' or not inst.show_self:
            continue
        lights.append(export_light(ob, inst.matrix_world,
                                   bscene.unit_settings.scale_length))

    # light linking is authored by object name; resolve to the indices the
    # renderer works in, now that every object has one
    name_to_index = {o.name: i for i, o in enumerate(objects)}
    for lt in lights:
        names = getattr(lt, 'exclude_names', None)
        if names:
            lt.exclude_objects = tuple(sorted(
                name_to_index[n] for n in names if n in name_to_index))

    mesh = _concat(parts)

    cam_ob = bscene.camera
    camera = Camera()
    if cam_ob is not None:
        cam = cam_ob.data
        camera.matrix_world = np.asarray(cam_ob.matrix_world, np.float32)
        camera.type = cam.type
        camera.lens = float(getattr(cam, 'lens', 50.0))
        camera.sensor = float(getattr(cam, 'sensor_width', 36.0))
        camera.clip_start = float(cam.clip_start)
        camera.clip_end = float(cam.clip_end)
        camera.ortho_scale = float(getattr(cam, 'ortho_scale', 6.0))
        camera.shift_x = float(cam.shift_x)
        camera.shift_y = float(cam.shift_y)
        camera.dof = bool(getattr(cam.dof, 'use_dof', False))
        camera.projection = _projection_from_camera(cam_ob, depsgraph, settings)

    world = World()
    bw = bscene.world
    if bw is not None:
        hs = getattr(bw, 'halcyon', None)
        try:
            world.color = tuple(bw.color[:3])
        except Exception:                                       # noqa: BLE001
            pass
        if hs is not None:
            import dataclasses as _dc
            for f in _dc.fields(World):
                if f.name in ('graph', 'env_image', 'mist', 'mist_start',
                              'mist_depth', 'mist_color', 'mist_falloff',
                              'mist_intensity'):
                    continue
                if hasattr(hs, f.name):
                    v = getattr(hs, f.name)
                    setattr(world, f.name, tuple(v)
                            if isinstance(f.default, tuple) else v)
            img = getattr(hs, 'env_image', None)
            if img is not None:
                key = img.name_full
                if key not in images:
                    px = compat.image_pixels(img)
                    if px is not None:
                        images[key] = ImageBuffer(name=key, pixels=px,
                                                  colorspace='Linear')
                world.env_image = images.get(key)
        if compat.uses_nodes(bw) and bw.node_tree:
            world.graph = serialize_tree(bw.node_tree, images, {}, warnings)
    if bscene.render.use_freestyle:
        warnings.append("Freestyle is not supported; use the Wireframe shader instead")

    sc = Scene(mesh=mesh, materials=materials or [Material()], objects=objects,
               lights=lights, camera=camera, world=world, settings=settings,
               frame=bscene.frame_current, fps=bscene.render.fps,
               time=bscene.frame_current / max(bscene.render.fps, 1))
    sc.images = images
    sc.warnings = warnings
    return sc


def _projection_from_camera(cam_ob, depsgraph, settings):
    """Blender's own projection matrix, so framing matches the viewport exactly.

    `calc_matrix_camera` is C code that dereferences the depsgraph it is handed.
    Passing None segfaults the whole process, and a segfault cannot be caught by
    try/except -- so the guard has to happen before the call, not around it.
    The previous version fished a depsgraph out of `scene.view_layers[0]`, which
    is NULL during a material preview render. That was the crash.
    """
    if depsgraph is None or cam_ob is None:
        return None
    if not hasattr(cam_ob, 'calc_matrix_camera'):
        return None
    w = max(int(settings.resolution_x), 1)
    h = max(int(settings.resolution_y), 1)
    try:
        m = cam_ob.calc_matrix_camera(depsgraph, x=w, y=h,
                                      scale_x=float(settings.pixel_aspect_x),
                                      scale_y=float(settings.pixel_aspect_y))
        return np.asarray(m, np.float32)
    except Exception:                                           # noqa: BLE001
        return None


def _concat(parts):
    mesh = MeshData()
    if not parts:
        mesh.verts = np.zeros((0, 3), np.float32)
        mesh.tris = np.zeros((0, 3), np.int32)
        mesh.face_normals = np.zeros((0, 3), np.float32)
        mesh.normals = np.zeros((0, 3), np.float32)
        mesh.uvs = np.zeros((0, 2), np.float32)
        mesh.colors = np.zeros((0, 4), np.float32)
        mesh.mat_index = np.zeros(0, np.int32)
        mesh.obj_index = np.zeros(0, np.int32)
        mesh.smooth = np.zeros(0, bool)
        return mesh
    base = 0
    V, N, U, U2, C, T, MI, OI, FN, S = ([] for _ in range(10))
    has_uv2 = any(p['uvs2'] is not None for p in parts)
    for p in parts:
        V.append(p['verts'])
        N.append(p['normals'])
        U.append(p['uvs'])
        if has_uv2:
            U2.append(p['uvs2'] if p['uvs2'] is not None else p['uvs'])
        C.append(p['colors'])
        T.append(p['tris'] + base)
        MI.append(p['mat_index'])
        OI.append(p['obj_index'])
        FN.append(p['face_normals'])
        S.append(p['smooth'])
        base += p['verts'].shape[0]
    mesh.verts = np.concatenate(V)
    mesh.normals = np.concatenate(N)
    mesh.uvs = np.concatenate(U)
    mesh.uvs2 = np.concatenate(U2) if has_uv2 else None
    # layer NAMES, first-seen: the UV Map node resolves by name on both
    # devices. Two sets travel -- the period dual-texture budget -- and
    # a mesh whose objects disagree about names keeps the first pair.
    mesh.uv_names = next((p['uv_names'] for p in parts
                          if p.get('uv_names')), [])
    mesh.colors = np.concatenate(C)
    mesh.tris = np.concatenate(T)
    mesh.mat_index = np.concatenate(MI)
    mesh.obj_index = np.concatenate(OI)
    mesh.face_normals = np.concatenate(FN)
    mesh.smooth = np.concatenate(S)
    return mesh
