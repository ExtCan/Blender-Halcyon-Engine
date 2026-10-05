"""Turn an evaluated Blender scene into the plain dataclasses the renderer eats.

Nothing downstream of this file imports bpy. That boundary is what makes the
renderer testable headlessly, and it is why node trees are flattened into dicts
here rather than walked live during shading.
"""

import re

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
    'HALCYON_AnimeShaderNode': ('compat', 'tones', 'use_vertex_ao',
                                'emission_alpha', 'rim_blend',
                                'matcap_mode',
                                # R229: the 80s additions' menus
                                'airbrush_side', 'line_source',
                                # R238: the cel's light
                                'light_source', 'rim_mode', 'rim_side',
                                'smooth_shape',
                                # R239: the SDF face shadow's frame
                                'face_forward', 'face_up',
                                # R241: the hair pass's wave shape
                                'hair_shine_shape'),
    # R225: the volume master's law, container shape and voxel lattice
    'HALCYON_VolumeNode': ('model', 'shape', 'voxels'),
    # R228: the cartoon master's shadow mode (the Era menu is a preset
    # applicator: it writes the sockets, which serialize by themselves)
    'HALCYON_CartoonNode': ('shadow_mode', 'rim_blend',
                            # R238: the cel's light and the airbrush
                            'light_source', 'rim_mode', 'rim_side',
                            'smooth_shape', 'airbrush_side',
                            # R241: the hair pass's wave shape
                            'hair_shine_shape'),
    'HALCYON_ShaderNode': ('model', 'toon_steps', 'wire_size',
                           'fresnel_blend', 'rim_blend', 'matcap_mode',
                           'faceted'),
    'HALCYON_BIMaterialNode': (
        'diff_shader', 'spec_shader', 'shadeless',
        # the BI panel round: sorted by panel
        'use_cubic', 'use_tangent_v',
        'use_transparency', 'transp_mode',
        'use_mirror',
        'use_ramp_dif', 'ramp_dif_input', 'ramp_dif_blend',
        'ramp_dif_factor',
        'use_ramp_spec', 'ramp_spec_input', 'ramp_spec_blend',
        'ramp_spec_factor',
        'use_mist', 'vcol_paint', 'vcol_light',
        'light_group', 'light_group_exclusive',
        'shadow_receive', 'shadow_cast', 'shadow_cast_only',
        'shadow_only',
        # R164: terminator fix + object colour
        'sbias', 'raybias', 'use_obcolor',
        # R162: the Subsurface Scattering panel
        'sss_enable', 'sss_scale', 'sss_radius', 'sss_color',
        'sss_ior', 'sss_error', 'sss_colfac', 'sss_texfac',
        'sss_front', 'sss_back'),
    'HALCYON_RampNode': ('space', 'stops', 'positions', 'easing'),
    'HALCYON_BlurNode': ('taps',),
    'ShaderNodeTexGabor': ('gabor_type',),
    'HALCYON_CodeNode': ('language', 'as_surface', 'source_text'),
    # the retro utilities. Dither's pattern and Depth Cue's falloff were
    # MISSING from this table from the day those nodes shipped: the
    # evaluator read the prop, the exporter never copied it, and both
    # dropdowns silently rendered as their defaults. A test now walks the
    # DISPATCH handlers' _prop reads against this table so a node cannot
    # ship half-wired again.
    'HALCYON_DitherNode': ('pattern',),
    'HALCYON_NormalMapNode': ('space', 'map_type'),
    'HALCYON_NormalMixNode': ('mode',),
    # R203 field find: the Iridescent node shipped WITHOUT this entry,
    # so its Type never reached the renderer -- every mode rendered as
    # Rainbow Sweep, which also deadened Tint (Pearl's) and Noise
    # Scale (Oil's). A guard test now audits every node's annotations
    # against this table so the class of bug is closed, not the case
    'HALCYON_IridescentNode': ('mode',),
    'HALCYON_SmoothStepNode': ('interp',),
    'HALCYON_ChannelShuffleNode': ('out_r', 'out_g', 'out_b', 'out_a'),
    'HALCYON_WaveNode': ('wave',),
    'HALCYON_BIInfluenceNode': ('blend', 'tex_rgb', 'rgbtoint',
                                'negative', 'alphamix', 'calc_alpha',
                                'neg_alpha'),
    'HALCYON_BIRGBBlendNode': ('blend', 'tex_rgb', 'rgbtoint',
                               'negative', 'alphamix', 'map_alpha',
                               'calc_alpha', 'neg_alpha'),
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
    from .nodes.shader_nodes import FAMILY_NODE_PROPS as _FAM_PROPS
    NODE_PROPS.update(_FAM_PROPS)
except Exception:                                               # noqa: BLE001
    pass

try:
    from .nodes.bitex_node import BI_NODE_PROPS as _BI_PROPS
    NODE_PROPS['HALCYON_BITextureNode'] = _BI_PROPS
except Exception:                                               # noqa: BLE001
    pass

try:
    # R242: the 3DS Max shelf's menus travel with their nodes
    from .nodes.max_nodes import NODE_PROPS as _MAX_PROPS
    NODE_PROPS.update(_MAX_PROPS)
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

    if node.bl_idname == 'HALCYON_BITextureNode':
        if '__bitex_tables__' not in images:
            from .core.bitex_tables import table_pixels
            images['__bitex_tables__'] = ImageBuffer(
                name='__bitex_tables__', pixels=table_pixels(),
                colorspace='Non-Color')
        stops = getattr(node, 'stops', None)
        if stops is not None and len(stops):
            props['coba'] = [(float(s.position), float(s.color[0]),
                              float(s.color[1]), float(s.color[2]),
                              float(s.color[3])) for s in stops]

    if node.bl_idname == 'HALCYON_BIMaterialNode':
        # the ramps' stops, position-sorted as colorband_eval expects,
        # and the ipo enum mapped to do_colorband's integer
        ipo_map = {'LINEAR': 0, 'EASE': 1, 'B_SPLINE': 2,
                   'CARDINAL': 3, 'CONSTANT': 4}
        for prefix, which, attr in (('ramp_dif', 'dif', 'dif_stops'),
                                    ('ramp_spec', 'spec', 'spec_stops')):
            stops = None
            if hasattr(node, 'ramp_stops'):
                # the gradient widget, when the artist has one, IS the
                # band; the collection is the fallback
                try:
                    stops = node.ramp_stops(which)
                except Exception:                               # noqa: BLE001
                    stops = None
            if not stops:
                coll = getattr(node, attr, None)
                if coll is not None and len(coll):
                    stops = sorted(
                        (float(s.position), float(s.color[0]),
                         float(s.color[1]), float(s.color[2]),
                         float(s.color[3])) for s in coll)
            if stops:
                props[f'{prefix}_stops'] = [tuple(s) for s in stops]
            if hasattr(node, 'ramp_ipo'):
                try:
                    props[f'{prefix}_ipo'] = int(node.ramp_ipo(which))
                    continue
                except Exception:                               # noqa: BLE001
                    pass
            props[f'{prefix}_ipo'] = ipo_map.get(
                str(getattr(node, f'{prefix}_ipo', 'LINEAR')), 0)
        # the Light Group resolved to lamp names NOW, in bpy's world:
        # the engine never sees collections, only the name list
        gname = str(getattr(node, 'light_group', '') or '')
        if gname:
            try:
                coll = bpy.data.collections.get(gname)
            except Exception:                                   # noqa: BLE001
                coll = None
            names = []
            if coll is not None:
                for ob in coll.all_objects:
                    if getattr(ob, 'type', '') == 'LIGHT':
                        names.append(ob.name)
            props['light_group_lights'] = sorted(names)
            if not names:
                warnings.append(
                    f"BI Material light group '{gname}': no lamps "
                    "found (missing collection or no lights in it) -- "
                    "the material will receive NO direct light")

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


#: an identifier carrying a suffix ('Shader_001', 'A_Color', 'Socket_3',
#: 'From_Min_FLOAT3', 'Color.001') is a DISAMBIGUATOR, not a name
_IDENT_DECORATED = re.compile(r'[._]')


def canonical_socket_name(sock):
    """The socket name Halcyon's evaluators and emitters key on.

    R226 field find ("the Blender default textures just make the
    volume vanish"): Blender 4/5 renamed many DISPLAY names -- the
    texture nodes' "Fac" outputs are "Factor" now, so are the Fresnel,
    Wireframe and Attribute factors and the Mix Shader's factor input
    -- while keeping each socket's IDENTIFIER, the Python-API-stable
    name, exactly as it was. Every Halcyon evaluator and GLSL emitter
    keys its outputs by the classic names, and the graph was
    serialized by display name: under 5.x a Noise Texture's Factor
    output resolved to NOTHING, read zero, and a volume fed by it
    vanished (a surface fed by it went black, a mix went to its
    second input). A plain identifier that differs from the display
    name IS the classic name: use it. A decorated identifier is a
    disambiguator ('Shader_001', the Mix node's 'A_Color', a group's
    'Socket_3'): those keep the display name, which the evaluators
    match by name OR identifier for inputs.
    """
    name = str(getattr(sock, 'name', '') or '')
    ident = str(getattr(sock, 'identifier', '') or '')
    if ident and ident != name and not _IDENT_DECORATED.search(ident):
        return ident
    return name


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
                'name': canonical_socket_name(sock),
                'label': sock.name,
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
                'name': canonical_socket_name(sock),
                'label': sock.name,
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
        # R219: the SURFACE FLAGS ride with every material, node-shaded or
        # overridden. They lived inside the override branch above, so a
        # NODE material -- which is what every fur pelt wears -- exported
        # with the dataclass defaults no matter what the panel said:
        # cast_shadow stayed True and sixteen stacked shells blanketed
        # their own lower layers and the body in cast shadow ("the lower
        # layers of the pelt are black, and the original model is very
        # dark"), and alpha_mode stayed BLEND so the R211 punch-through
        # never rode the z-pass in the field. These are flags about how
        # the surface meets the frame, not shading parameters -- the
        # override toggle was never meant to gate them.
        m.cast_shadow = hs.cast_shadow
        m.receive_shadow = hs.receive_shadow
        # R251 C134: the gel of the Tron printer's backlit matte
        m.glow_gel = tuple(float(c) for c in
                           getattr(hs, 'glow_gel', (0.0, 0.0, 0.0)))
        # R251 C020/C036: the material's triangles as an authored volume
        m.volume_role = str(getattr(hs, 'volume_role', 'NONE'))
        m.polygon_id = int(getattr(hs, 'polygon_id', 0))
        m.shadow_alpha = int(getattr(hs, 'shadow_alpha', 16))
        m.alpha_mode = str(getattr(hs, 'alpha_mode', 'BLEND'))
        m.alpha_clip = float(getattr(hs, 'alpha_clip', 0.5))
        # R251: the per-material blend equation rides the same road
        m.blend_mode = str(getattr(hs, 'blend_mode', 'INHERIT') or 'INHERIT')
        # R251 C126: Blender 2.4x's Zoffs / ZInvert
        m.z_offset = float(getattr(hs, 'z_offset', 0.0))
        m.z_invert = bool(getattr(hs, 'z_invert', False))
        # R251 C095: Max's Thin Wall Refraction Thickness Offset
        m.thin_wall_offset = float(getattr(hs, 'thin_wall_offset', 0.5))
        # R251 C101: Imagine's Fog Length
        m.fog_length = float(getattr(hs, 'fog_length', 1.0))
        # R220: per-material ink rides the same unconditional road
        m.ink_mode = str(getattr(hs, 'ink_mode', 'INHERIT') or 'INHERIT')
        m.ink_use_color = bool(getattr(hs, 'ink_use_color', False))
        m.ink_color = tuple(getattr(hs, 'ink_color', (0.0, 0.0, 0.0)))
        m.ink_width = int(getattr(hs, 'ink_width', 0))
        # R239: the Guilty Gear vertex-colour line control
        m.ink_vc = str(getattr(hs, 'ink_vc', 'OFF') or 'OFF')
        # R233: the cel or the painting
        m.paint_mode = str(getattr(hs, 'paint_mode', 'CEL') or 'CEL')
        if getattr(hs, 'halo', False):
            # R191: BI's halo material -- the whole panel in one spec.
            # Counts are 0 when their toggle is off, exactly the 2.79
            # mode-bit-gates-count arrangement
            m.halo = {
                'size': float(hs.halo_size),
                'hardness': int(hs.halo_hardness),
                'add': float(hs.halo_add),
                'alpha': float(hs.halo_alpha),
                'color': tuple(hs.halo_color),
                'seed': int(hs.halo_seed),
                'rings': int(hs.halo_ring_count) if hs.halo_rings else 0,
                'lines': int(hs.halo_line_count) if hs.halo_lines else 0,
                'star_points': int(hs.halo_star_tips) if hs.halo_star
                               else 0,
                'ring_color': tuple(hs.halo_ring_color),
                'line_color': tuple(hs.halo_line_color),
                'xalpha': bool(hs.halo_xalpha),
                'soft': bool(hs.halo_soft),
                'shaded': bool(hs.halo_shaded),
                'puno': bool(hs.halo_puno),
                # R194: the expansion kit -- all neutral at defaults
                'shape': str(getattr(hs, 'halo_shape', 'DISC')),
                'line_width': float(getattr(hs, 'halo_line_width', 1.0)),
                'ring_width': float(getattr(hs, 'halo_ring_width', 1.0)),
                'gradient': bool(getattr(hs, 'halo_gradient', False)),
                'color2': tuple(getattr(hs, 'halo_color2',
                                        (0.0, 0.0, 0.0))),
                'rand_hue': float(getattr(hs, 'halo_rand_hue', 0.0)),
                'rand_sat': float(getattr(hs, 'halo_rand_sat', 0.0)),
                'rand_val': float(getattr(hs, 'halo_rand_val', 0.0)),
                'pulse': float(getattr(hs, 'halo_pulse', 0.0)),
                'flicker': float(getattr(hs, 'halo_flicker', 0.0)),
                'spin': float(getattr(hs, 'halo_spin', 0.0)),
                'anim_speed': float(getattr(hs, 'halo_anim_speed', 1.0)),
                'aspect': float(getattr(hs, 'halo_aspect', 1.0)),
                'rotation': float(getattr(hs, 'halo_rotation', 0.0)),
                # R198: the energy kit
                'noise': float(getattr(hs, 'halo_noise', 0.0)),
                'noise_scale': float(getattr(hs, 'halo_noise_scale',
                                             4.0)),
                'bolts': int(getattr(hs, 'halo_bolts', 0)),
                'bolt_width': float(getattr(hs, 'halo_bolt_width', 1.0)),
                'rays': int(getattr(hs, 'halo_rays', 0)),
                'ray_sharp': float(getattr(hs, 'halo_ray_sharp', 8.0)),
                'rings_even': bool(getattr(hs, 'halo_rings_even',
                                           False)),
                'gradient_type': str(getattr(hs, 'halo_gradient_type',
                                             'RADIAL')),
                'gradient_noise': float(getattr(hs,
                                                'halo_gradient_noise',
                                                0.0)),
                # R200: master HSV shift and the per-effect clocks --
                # all exactly neutral at 0 / 1 / 1 and x1.0
                'hue_shift': float(getattr(hs, 'halo_hue_shift', 0.0)),
                'sat_shift': float(getattr(hs, 'halo_sat_shift', 1.0)),
                'val_shift': float(getattr(hs, 'halo_val_shift', 1.0)),
                'pulse_speed': float(getattr(hs, 'halo_pulse_speed',
                                             1.0)),
                'flicker_speed': float(getattr(hs,
                                               'halo_flicker_speed',
                                               1.0)),
                'noise_speed': float(getattr(hs, 'halo_noise_speed',
                                             1.0)),
                'bolt_speed': float(getattr(hs, 'halo_bolt_speed',
                                            1.0)),
                'grad_noise_speed': float(getattr(
                    hs, 'halo_grad_noise_speed', 1.0)),
            }
            # R200: Ray/Bolt Colour ship only once the artist has SET
            # them -- untouched, the key stays absent and the core
            # keeps riding the Line Colour, so every existing scene
            # renders exactly as before the dials existed
            for pk, sk in (('ray_color', 'halo_ray_color'),
                           ('bolt_color', 'halo_bolt_color')):
                try:
                    if hs.is_property_set(sk):
                        m.halo[pk] = tuple(getattr(hs, sk))
                except Exception:                               # noqa: BLE001
                    pass
            # the image halo: the picture rides the scene's image pool
            himg = getattr(hs, 'halo_image', None)
            if himg is not None and \
                    str(getattr(hs, 'halo_shape', '')) == 'IMAGE':
                try:
                    key = himg.name_full
                    if key not in images:
                        px = compat.image_pixels(himg)
                        if px is not None:
                            images[key] = ImageBuffer(name=key,
                                                      pixels=px)
                    m.halo['image'] = images.get(key)
                except Exception:                               # noqa: BLE001
                    pass
            # the colour ramp, sampled into a small LUT (the widget
            # lives on the hidden '__halo_ramp' node)
            try:
                if getattr(hs, 'halo_gradient', False) and \
                        compat.uses_nodes(mat) and mat.node_tree:
                    rn = mat.node_tree.nodes.get('__halo_ramp')
                    if rn is not None:
                        cr = rn.color_ramp
                        m.halo['ramp'] = [tuple(cr.evaluate(i / 31.0))
                                          for i in range(32)]
            except Exception:                                   # noqa: BLE001
                pass
    if compat.uses_nodes(mat) and mat.node_tree and \
            not (hs is not None and hs.use_override):
        # a material with Override on shades from the panel's own fields --
        # THE WHOLE PANEL, not just the model. Serialising the tree anyway
        # meant the graph won every field except `model` and the Material
        # tab's sliders moved nothing: the "tab that doesn't do anything".
        # With the graph withheld, the core's graphless-constants path
        # shades from exactly the values the panel shows, on both devices,
        # and alpha falls to the override's own Opacity above.
        m.programs = {}
        m.graph = serialize_tree(mat.node_tree, images, m.programs, warnings)
    # R208: a material marked Hair Geometry (the fur-shell material, or
    # any custom hair mesh) shades with the strand convention -- the
    # colour layer carries intercept/random/length/thickness and Hair
    # Info reads it. The flag rides the graph so both devices see it.
    if hs is not None and getattr(hs, 'strand', False):
        m.strand = True
        if m.graph:
            m.graph['strand'] = True
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
        if idn == 'HALCYON_BIMaterialNode':
            # the BI node's transparency panel gates everything: OFF,
            # the material is opaque whatever Alpha says (n_bi_material
            # forces opacity 1), so the alpha predicate must agree or
            # the layer plan peels an opaque surface
            props = node.get('props', {})
            if props.get('shadow_only'):
                # the catcher's alpha IS the shadow, transparency
                # panel or no transparency panel (as BI had it)
                return 'it is a Shadows Only catcher'
            if props.get('shadow_cast_only'):
                # invisible to the camera = alpha 0 on the camera pass
                return 'it is Cast Only (camera-invisible)'
            if not props.get('use_transparency'):
                continue
            for s in node['inputs']:
                sname = s.get('identifier') or s['name']
                if sname == 'Transp Fresnel':
                    if s['link'] is not None:
                        return 'its Transparency Fresnel is linked'
                    if s['default'] is not None and \
                            abs(float(s['default'])) > 0.0:
                        return ('its Transparency Fresnel '
                                f'{float(s["default"]):.3f} makes alpha '
                                'view-dependent')
                if sname != 'Opacity' and s['name'] != 'Alpha':
                    continue
                if s['link'] is not None:
                    return f'its {s["name"]} socket is linked'
                if s['default'] is not None and float(s['default']) < 0.999:
                    return f'{s["name"]} {float(s["default"]):.3f}'
            continue
        if idn in ('HALCYON_MaxStandardNode', 'HALCYON_MaxRaytraceNode'):
            # R243: Max's materials -- Opacity is a percentage, Falloff
            # fades it by the view angle, Raytrace's Transparency colour
            # is its see-through amount
            for s in node['inputs']:
                sname = s.get('identifier') or s['name']
                if sname == 'Max Opacity':
                    if s['link'] is not None:
                        return 'its Opacity socket is linked'
                    if s['default'] is not None and float(s['default']) < 99.9:
                        return f'Opacity {float(s["default"]):.1f} percent'
                elif sname == 'Max Falloff Amount':
                    if s['link'] is not None or (
                            s['default'] is not None and float(s['default']) > 0.0):
                        return 'its Opacity Falloff fades the surface'
                elif sname == 'Max Transparency':
                    if s['link'] is not None:
                        return 'its Transparency socket is linked'
                    d = s['default']
                    if d is not None and max(float(v) for v in list(d)[:3]) > 1e-3:
                        return 'Raytrace Transparency above black'
            continue
        if idn == 'HALCYON_ShaderNode':
            # the master shader's own alpha lives on its sockets, and the
            # add-on's templates put it there (Glass: Opacity 0.12) with
            # use_override off -- so m.opacity never sees it. Without this
            # check the engine's own glass presets exported as OPAQUE and
            # skipped the transparent pass entirely. Edge Opacity below
            # 1.0 is see-through at the silhouette even when Opacity is
            # 1.0, so it counts the same way.
            for s in node['inputs']:
                sname = s.get('identifier') or s['name']
                if sname not in ('Opacity', 'Edge Opacity') and \
                        s['name'] not in ('Opacity', 'Edge Opacity'):
                    continue
                if s['link'] is not None:
                    return f'its {s["name"]} socket is linked'
                if s['default'] is not None and float(s['default']) < 0.999:
                    return f'{s["name"]} {float(s["default"]):.3f}'
    return None


def _tree_has_alpha(mat, m):
    return _alpha_reason(mat, m) is not None


# -------------------------------------------------------------------- mesh


def _check_fur_strand_health(ob, me, data, warnings):
    """R216: "The shells are still black." Say WHY, at export, by name.

    A pelt built by 1.55.0..1.57.x carries a corrupt mesh in the .blend
    -- the fast-path builder zeroed the strand colour layer, and a mesh
    is DATA: upgrading the add-on does not heal what is already built.
    At intercept 0 every shell passes its alpha test and wears the root
    colour, which is exactly a dense black pelt. This check reads the
    pelt the renderer is about to wear and names the failure in the
    console, so a field paste pins the cause in one line:
    - the strand layer is MISSING (never built / stripped),
    - the strand layer is SHADOWED (another colour attribute sits at
      index 0, so the export reads the wrong layer as strand data), or
    - the strand layer reads ALL ZEROS (the pre-1.58 corrupt build).
    The cure for all three is the same and is printed with the warning:
    Rebuild From Source on 1.58+, which builds via from_pydata and
    verifies the layer by reading it back.
    """
    try:
        if not getattr(getattr(ob, 'halcyon_fur', None), 'is_fur', False):
            return
        if not isinstance(data, dict):
            return
        layers = compat.color_layers(me)
        names = [str(getattr(a, 'name', '') or '') for a in layers]
        why = None
        if 'HalcyonFur' not in names:
            why = ("its strand colour layer 'HalcyonFur' is MISSING from "
                   "the mesh")
        elif names[0] != 'HalcyonFur':
            why = (f"its strand colour layer is SHADOWED -- "
                   f"'{names[0]}' sits first and the export reads that "
                   f"as strand data")
        else:
            cols = data.get('colors')
            if cols is not None and cols.size and \
                    float(np.abs(np.asarray(cols)[:, :3]).max()) < 1e-6:
                why = ('its strand colour layer reads ALL ZEROS -- the '
                       'corrupt pre-1.58 mesh build, still in the .blend')
        # R219: a pelt whose tuft material CASTS is its own eclipse --
        # sixteen stacked shells drop cast shadow on every layer beneath
        # them and on the body, and the field reads it as "the lower
        # layers of the pelt are black, and the original model is very
        # dark" no matter how strong the light. The builder turns the
        # flag off (the era's shell fur was never in a shadow map), but
        # a hand-made tuft material, or one edited back on, still can.
        try:
            for slot_mat in [s.material for s in
                             getattr(ob, 'material_slots', [])
                             if getattr(s, 'material', None) is not None]:
                mh = getattr(slot_mat, 'halcyon', None)
                if mh is not None and getattr(mh, 'strand', False) and \
                        getattr(mh, 'cast_shadow', True):
                    cmsg = (f"fur pelt '{getattr(ob, 'name', '?')}': its "
                            f"tuft material "
                            f"'{getattr(slot_mat, 'name', '?')}' has Cast "
                            f"Shadows ON, so the stacked shells blanket "
                            f"their own lower layers and the body in cast "
                            f"shadow -- a black pelt under any light. "
                            f"Untick Cast Shadows on that material, or "
                            f"press Rebuild From Source")
                    print(f'[Halcyon] {cmsg}')
                    warnings.append(cmsg)
                    break
        except Exception:                                       # noqa: BLE001
            pass
        if why is None:
            return
        msg = (f"fur pelt '{getattr(ob, 'name', '?')}': {why}. At "
               f"intercept 0 every shell passes its alpha test and "
               f"wears the root colour: a dense BLACK pelt. Select the "
               f"pelt and press Rebuild From Source (Object Properties "
               f"> Halcyon Fur), or nudge any fur dial, to re-grow it "
               f"verified")
        print(f'[Halcyon] {msg}')
        warnings.append(msg)
    except Exception:                                           # noqa: BLE001
        pass


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
    layers = list(compat.uv_layers(me))
    if len(layers):
        # the RENDER-ACTIVE layer is the primary -- BI's unnamed slots
        # sample it, and layer ORDER lied on the field's meshes (the
        # edit-active head layer sampled a white region of the fur map)
        prim = next((i for i, l in enumerate(layers)
                     if getattr(l, 'active_render', False)), 0)
        order = [layers[prim]] + [l for i, l in enumerate(layers)
                                  if i != prim]
        buf = np.empty(n_loops * 2, np.float32)
        order[0].data.foreach_get('uv', buf)
        uvs = buf.reshape(-1, 2)
        uv_names.append(str(getattr(order[0], 'name', '') or ''))
        if len(order) > 1:
            buf2 = np.empty(n_loops * 2, np.float32)
            order[1].data.foreach_get('uv', buf2)
            uvs2 = buf2.reshape(-1, 2)
            uv_names.append(str(getattr(order[1], 'name', '') or ''))

    cols = np.ones((n_loops, 4), np.float32)
    color_name = ''
    clayers = compat.color_layers(me)
    if len(clayers):
        try:
            lay = clayers[0]
            color_name = str(getattr(lay, 'name', '') or '')
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

    # R220: the artist's marked edges (Sharp, creases, Freestyle edge
    # marks -- UV seams deliberately excluded, they mark unwrapping,
    # not looks), folded into a per-TRIANGLE bitmask: bit k set = the
    # edge opposite corner k is marked. The ink pass measures each
    # pixel's exact distance to its own triangle's marked slots, so
    # hidden-line removal is the z-buffer's own verdict and the mask
    # concatenates like any other per-tri array. A mesh with no marks
    # exports None and costs one boolean sweep.
    ink_mask = None
    try:
        ie = compat.edge_ink_arrays(me)
        if ie is not None:
            eidx, eflags = ie
            drawn = (eflags & (2 | 4 | 8)) != 0
            if drawn.any():
                eidx = eidx[drawn].astype(np.int64)
                stride = np.int64(max(n_verts, 1))
                key = np.minimum(eidx[:, 0], eidx[:, 1]) * stride \
                    + np.maximum(eidx[:, 0], eidx[:, 1])
                ov = lv[lt].astype(np.int64)          # (T,3) original ids

                def _pk(a, b):
                    return np.minimum(a, b) * stride + np.maximum(a, b)

                ink_mask = (
                    np.isin(_pk(ov[:, 1], ov[:, 2]), key).astype(np.uint8)
                    | (np.isin(_pk(ov[:, 2], ov[:, 0]), key)
                       .astype(np.uint8) << 1)
                    | (np.isin(_pk(ov[:, 0], ov[:, 1]), key)
                       .astype(np.uint8) << 2))
                if not ink_mask.any():
                    ink_mask = None
    except Exception:                                           # noqa: BLE001
        ink_mask = None

    return dict(verts=pos.astype(np.float32), normals=nn.astype(np.float32),
                uvs=uvs.astype(np.float32),
                uvs2=uvs2.astype(np.float32) if uvs2 is not None else None,
                uv_names=uv_names, color_name=color_name,
                colors=cols.astype(np.float32), tris=lt.astype(np.int32),
                mat_index=(mat_idx + mat_offset).astype(np.int32),
                obj_index=np.full(n_tris, obj_index, np.int32),
                face_normals=fn.astype(np.float32), smooth=smooth,
                ink_tri_mask=ink_mask)


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
        if hs is not None and getattr(hs, 'hemi', False):
            # BI's Hemi rides a Sun since 2.8 removed the type; the
            # engine shades the wrap and skips shadows
            lt.type = 'HEMI'
            lt.shadow = 'NONE'
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
            lt.area_gamma = float(getattr(hs, 'area_gamma', 1.0) or 1.0)
    if hs is not None:
        lt.decay = hs.decay
        lt.decay_start = hs.decay_start
        lt.decay_end = hs.decay_end
        lt.decay_ld1 = getattr(hs, 'decay_ld1', 0.0)
        lt.decay_ld2 = getattr(hs, 'decay_ld2', 0.0)
        lt.bi_sphere = getattr(hs, 'bi_sphere', False)
        lt.gx_ref_brite = float(getattr(hs, 'gx_ref_brite', 0.5) or 0.5)
        lt.shadow = hs.shadow
        lt.shadow_map_size = hs.shadow_map_size
        lt.shadow_bias = hs.shadow_bias
        lt.shadow_map_depth = str(getattr(hs, 'shadow_map_depth', 'INHERIT'))
        lt.shadow_softness = hs.shadow_softness
        lt.shadow_samples = hs.shadow_samples
        lt.shadow_density = hs.shadow_density
        lt.shadow_color = tuple(hs.shadow_color)
        lt.negative = hs.negative
        lt.only_shadow = bool(getattr(hs, 'only_shadow', False))
        lt.diffuse_only = hs.diffuse_only
        lt.specular_only = hs.specular_only
        lt.ambient_only = hs.ambient_only
        lt.hotspot = hs.hotspot
        lt.spot_law = str(getattr(hs, 'spot_law', 'BLENDER') or 'BLENDER')
        lt.spot_exponent = float(getattr(hs, 'spot_exponent', 0.0) or 0.0)
        lt.screen_spot = bool(getattr(hs, 'screen_spot', False))
        lt.volumetric = hs.volumetric
        lt.volumetric_occlusion = getattr(hs, 'volumetric_occlusion', False)
        lt.flare = float(getattr(hs, 'flare', 0.0))
        lt.flare_scale = float(getattr(hs, 'flare_scale', 1.0))
        lt.flare_streaks = int(getattr(hs, 'flare_streaks', 6))
        lt.flare_rings = int(getattr(hs, 'flare_rings', 1))
        lt.flare_ghosts = int(getattr(hs, 'flare_ghosts', 6))
        lt.caustics = float(getattr(hs, 'caustics', 0.0))
        lt.caustics_scale = float(getattr(hs, 'caustics_scale', 4.0))
        lt.caustics_speed = float(getattr(hs, 'caustics_speed', 1.0))
        lt.exclude_mode = getattr(hs, 'exclude_mode', 'EXCLUDE')
        coll = getattr(hs, 'exclude_collection', None)
        if coll is not None:
            lt.exclude_names = {o.name for o in coll.all_objects}
        ck = getattr(hs, 'cookie', None)
        if ck is not None:
            # R219: every lamp kind projects (the old gate let only
            # SPOT and SUN through -- "Gobos only work with Sun")
            px = compat.image_pixels(ck)
            if px is not None:
                lt.cookie = ImageBuffer(name=ck.name_full, pixels=px,
                                        colorspace='Non-Color')
                lt.cookie_strength = float(getattr(hs, 'cookie_strength',
                                                   1.0))
                lt.cookie_scale = float(getattr(hs, 'cookie_scale', 10.0))
                lt.cookie_extend = str(getattr(hs, 'cookie_extend',
                                               'AUTO') or 'AUTO')
                lt.cookie_filter = str(getattr(hs, 'cookie_filter',
                                               'BILINEAR') or 'BILINEAR')
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


#: R170: where the last export_scene's milliseconds went (meshes,
#: materials, concat, other) -- the engine prints it under the F12
#: breakdown so the field's next paste names the export wall
EXPORT_SPLIT = {}

# --------------------------------------------------- the mesh export cache
#
# R171. The field's warm F12 was 74% export, and the export split named it:
# "meshes 171x 332 ms" -- 171 unchanged objects pushed through
# `evaluated_get` + `to_mesh` + eleven foreach_get walks per frame, to
# rebuild arrays identical to the last frame's. The cache keeps each
# object's finished `_mesh_arrays` output keyed on (data session_uid,
# depsgraph mode, world matrix bytes), and a depsgraph_update_post handler
# marks data stale the moment Blender reports ANY update touching it --
# so an edit of any kind re-exports the object live, and an unchanged
# object costs a dict lookup.
#
# Exactness rules, in force:
# * The key's identity is the OBJECT (session_uid; name as the stub
#   fallback -- names are unique in bmain), NOT the data-block: modifier
#   stacks are per-object, so two objects sharing one mesh can evaluate
#   to DIFFERENT geometry at the same matrix. Each entry remembers its
#   data-block's uid separately so a direct data edit (a Mesh update)
#   still drops every user object's entries even if the depsgraph did
#   not list all the users.
# * The key carries the depsgraph MODE: render and viewport depsgraphs
#   evaluate DIFFERENT meshes from the same datablock (show_render vs
#   show_viewport modifiers, render-level subdivision), so the two sides
#   warm separately and can never serve each other a wrong mesh.
# * mat_index and obj_index are REBUILT on every reuse (the stored entry
#   keeps the pre-remap slot indices and the slot NAMES): material list
#   order and object numbering come out identical to a fully live export.
# * ObjectInfo fields are snapshotted from the EVALUATED object at store
#   time -- a driven color or pass_index reads the driven value, and any
#   change dirties the object anyway.
# * The update handler is deliberately pessimistic: only ID types that
#   provably cannot alter another object's evaluated geometry are ignored
#   (materials, images, lamps, worlds, cameras, shader/compositor trees);
#   an Object or geometry-data update dirties that data everywhere it is
#   instanced, and ANY unrecognised update flushes the whole cache.
#   Undo, redo, file load and frame changes flush it whole too.
# * A stored slot name that no longer names a material (renamed since the
#   store) sends that object down the RETRY path: a live re-export, in
#   the exact list position the cache reserved for it.
# Without handlers (a bpy stub, a failed registration) the cache never
# arms and every export runs the full live path, exactly as before.

_MESH_CACHE = {}          # (object token, mode, matrix bytes) -> entry
_MESH_CACHE_CAP = 8192    # count backstop; the real bounds are below
_MESH_CACHE_BUDGET = 512 * 1024 * 1024   # bytes of cached arrays, LRU
_MESH_CACHE_BYTES = 0     # running total of entry array bytes
_DIRTY_OBJS = set()       # object tokens whose entries must be dropped
_DIRTY_DATA = set()       # data session_uids whose entries must be dropped
_DIRTY_ALL = False
_DIRTY_ALL_BY = None      # the ID type name that last set _DIRTY_ALL
_CACHE_ARMED = False

#: ID type names whose updates cannot change any object's evaluated mesh
#: geometry, our matrices, or the ObjectInfo fields we serve. Material
#: edits re-export every frame anyway (materials are never cached);
#: images/textures that feed a modifier re-evaluate their OBJECT too, and
#: that update is the one that dirties. Scene is here as of R173: its one
#: geometry lever (Simplify) is fingerprinted directly at every export
#: (_scene_geo_sig below), which catches it MORE reliably than update
#: reports. Collection and ViewLayer joined in R178: the field's
#: viewport printed 'dirt ALL(Collection)' on EVERY refine -- Blender
#: tags collections on ordinary edits, and each tag was a full
#: 171-object re-export. What a collection or view-layer toggle CAN
#: change about a cached object -- its evaluated flags: holdout,
#: visibility, colour -- is re-read FRESH from the evaluated object on
#: every reuse now (~1 ms for the whole scene), and geometry that
#: depends on a collection (a node-tree collection input) re-evaluates
#: its OBJECT, whose own update dirties it -- the same dependency
#: propagation Image and Texture already trust. Everything not named
#: here still flushes the whole cache -- unknown means dirty.
_IRRELEVANT_IDS = frozenset({
    'Material', 'Image', 'Texture', 'ImageTexture', 'Light', 'World',
    'Camera', 'Scene', 'Collection', 'ViewLayer', 'ShaderNodeTree',
    'CompositorNodeTree', 'TextureNodeTree', 'Brush', 'Palette',
    'Screen', 'WindowManager', 'WorkSpace'})

#: geometry datablock types whose session_uid IS a cache key prefix
_GEOM_DATA_IDS = frozenset({'Mesh', 'Curve', 'SurfaceCurve', 'TextCurve',
                            'MetaBall'})


def _persistent(fn):
    """bpy.app.handlers.persistent where it exists, identity elsewhere."""
    try:
        import bpy
        return bpy.app.handlers.persistent(fn)
    except Exception:                                           # noqa: BLE001
        return fn


def _obj_token(ob):
    """The object's stable identity: session_uid, or its unique name."""
    uid = getattr(ob, 'session_uid', None)
    if uid is not None:
        return int(uid)
    name = getattr(ob, 'name', None)
    return ('n', name) if name else None


@_persistent
def _watch_updates(scene, depsgraph):
    """depsgraph_update_post: turn Blender's update report into dirt."""
    global _DIRTY_ALL, _DIRTY_ALL_BY
    if not _CACHE_ARMED:
        return
    try:
        for u in depsgraph.updates:
            idb = u.id
            idb = getattr(idb, 'original', None) or idb
            t = type(idb).__name__
            if t == 'Object':
                # any Object update -- transform, geometry, a modifier,
                # shading, even a selection tick -- dirties THAT object:
                # the arrays bake its modifier stack, its matrix and the
                # ObjectInfo snapshot, so the cheap answer is the
                # correct one
                tok = _obj_token(idb)
                if tok is not None:
                    _DIRTY_OBJS.add(tok)
                else:
                    _DIRTY_ALL = True
                    _DIRTY_ALL_BY = t
                    return
                continue
            if t in _IRRELEVANT_IDS:
                continue
            uid = getattr(idb, 'session_uid', None)
            if t in _GEOM_DATA_IDS and uid is not None:
                # a direct data edit: drop EVERY user object's entries
                # via the data uid each entry remembers
                _DIRTY_DATA.add(int(uid))
            else:
                # Scene (simplify!), Collection (holdout!), node groups,
                # shape keys, actions, anything new in a future Blender:
                # correctness first
                _DIRTY_ALL = True
                _DIRTY_ALL_BY = t
                return
    except Exception:                                           # noqa: BLE001
        _DIRTY_ALL = True
        _DIRTY_ALL_BY = 'error'


@_persistent
def _flush_mesh_cache(*_args):
    """undo/redo/load: identities survive but contents moved wholesale."""
    global _DIRTY_ALL, _DIRTY_ALL_BY
    _DIRTY_ALL = True
    _DIRTY_ALL_BY = 'flush'


#: last (frame, subframe) seen per scene name, for the handler below
_LAST_FRAME = {}


@_persistent
def _on_frame_change(scene, depsgraph=None):
    """frame_change_post: flush only when the frame actually MOVED.

    R173, from the field's own instrument: every still F12 fires
    frame_change on the CURRENT frame (the render pipeline re-sets the
    frame it is about to draw), and the old blanket flush wiped both
    pools before every render -- 'dirt ALL(flush)' on every warm F12,
    150 objects re-exported for nothing. A re-evaluation of the SAME
    (frame, subframe) has identical time inputs, so it cannot change
    any object the depsgraph does not list -- whatever it DOES list
    walks through the ordinary targeted dirt. A real frame step (an
    animation render, playback, scrubbing) still flushes whole,
    exactly as before.
    """
    global _DIRTY_ALL, _DIRTY_ALL_BY
    if not _CACHE_ARMED:
        return
    try:
        key = getattr(scene, 'name', '') or ''
        fr = int(getattr(scene, 'frame_current', 0))
        sub = float(getattr(scene, 'frame_subframe', 0.0) or 0.0)
        last = _LAST_FRAME.get(key)
        _LAST_FRAME[key] = (fr, sub)
        while len(_LAST_FRAME) > 16:
            _LAST_FRAME.pop(next(iter(_LAST_FRAME)))
        if last == (fr, sub):
            if depsgraph is not None:
                _watch_updates(scene, depsgraph)
            return
    except Exception:                                           # noqa: BLE001
        pass
    _DIRTY_ALL = True
    _DIRTY_ALL_BY = 'frame'


#: the Scene fields that can change EVALUATED MESH GEOMETRY -- the
#: Simplify block. Fingerprinted at every export instead of trusting
#: Scene update reports: a change flushes both pools deterministically,
#: whether or not an update ever fired.
_SCENE_GEO_SIG = None


def _scene_geo_sig(scene):
    r = getattr(scene, 'render', None)
    if r is None:
        return ()
    try:
        return (bool(getattr(r, 'use_simplify', False)),
                int(getattr(r, 'simplify_subdivision', 6)),
                int(getattr(r, 'simplify_subdivision_render', 6)),
                float(getattr(r, 'simplify_child_particles', 1.0)),
                float(getattr(r, 'simplify_child_particles_render', 1.0)),
                float(getattr(r, 'simplify_volumes', 1.0)))
    except Exception:                                           # noqa: BLE001
        return None


def _entry_bytes(ent):
    n = 0
    try:
        for v in ent['data'].values():
            if hasattr(v, 'nbytes'):
                n += int(v.nbytes)
        n += int(ent['raw_mat'].nbytes)
    except Exception:                                           # noqa: BLE001
        pass
    return n


def _cache_pop(key):
    global _MESH_CACHE_BYTES
    ent = _MESH_CACHE.pop(key, None)
    if ent is not None:
        _MESH_CACHE_BYTES = max(
            _MESH_CACHE_BYTES - ent.get('bytes', 0), 0)
    return ent


def _cache_store(key, ent):
    """Insert under the byte budget and the count backstop, LRU."""
    global _MESH_CACHE_BYTES
    ent['bytes'] = _entry_bytes(ent)
    if key in _MESH_CACHE:
        _cache_pop(key)
    while _MESH_CACHE and (
            len(_MESH_CACHE) >= _MESH_CACHE_CAP
            or _MESH_CACHE_BYTES + ent['bytes'] > _MESH_CACHE_BUDGET):
        _cache_pop(next(iter(_MESH_CACHE)))
    _MESH_CACHE[key] = ent
    _MESH_CACHE_BYTES += ent['bytes']


def _cache_scrub():
    """Apply the dirt collected since the last export, then forget it.

    Returns (all_by, n_objs, n_data) -- what the dirt WAS, for the
    split instrument: the field's next paste can then say whether live
    re-exports came from real edits or from something spamming updates.
    """
    global _DIRTY_ALL, _DIRTY_ALL_BY
    took = (_DIRTY_ALL_BY if _DIRTY_ALL else None,
            len(_DIRTY_OBJS), len(_DIRTY_DATA))
    if _DIRTY_ALL:
        for k in list(_MESH_CACHE):
            _cache_pop(k)
        _DIRTY_OBJS.clear()
        _DIRTY_DATA.clear()
        _DIRTY_ALL = False
        _DIRTY_ALL_BY = None
        return took
    if _DIRTY_OBJS or _DIRTY_DATA:
        doomed = [k for k, e in _MESH_CACHE.items()
                  if k[0] in _DIRTY_OBJS
                  or e.get('data_uid') in _DIRTY_DATA]
        for k in doomed:
            _cache_pop(k)
        _DIRTY_OBJS.clear()
        _DIRTY_DATA.clear()
    return took


def _material_by_key(key, depsgraph):
    """The material whose name_full is `key`, evaluated -- or None.

    The live path exports the EVALUATED material a mesh slot points at;
    resolving a cached slot name must land on the same thing, so the
    bmain datablock found by name is pushed through evaluated_get.
    name_full is unique across a session (locals by name, linked
    qualified by library), which is exactly why it is the stored key.
    """
    try:
        import bpy
        mats = getattr(bpy.data, 'materials', None)
        if mats is None:
            return None
        for m in mats:
            if getattr(m, 'name_full', None) == key:
                try:
                    getter = getattr(m, 'evaluated_get', None)
                    return getter(depsgraph) if getter is not None else m
                except Exception:                               # noqa: BLE001
                    return m
    except Exception:                                           # noqa: BLE001
        pass
    return None


def _empty_part():
    """A zero-triangle part: keeps later objects' baked obj_index true."""
    z = np.zeros
    return dict(verts=z((0, 3), np.float32), normals=z((0, 3), np.float32),
                uvs=z((0, 2), np.float32), uvs2=None, uv_names=[],
                colors=z((0, 4), np.float32), tris=z((0, 3), np.int32),
                mat_index=z(0, np.int32), obj_index=z(0, np.int32),
                face_normals=z((0, 3), np.float32),
                smooth=z(0, bool))


def _smoke_grid(ob, warnings=None):
    """R223: a fluid DOMAIN's density voxels, read defensively.

    Mantaflow exposes the cached grids as flat float sequences on the
    domain settings; a domain whose cache is not baked (or a build that
    hides the attribute) reads None, with the reason in the warnings --
    the volume marcher then runs on the node chain alone.
    """
    try:
        for mod in getattr(ob, 'modifiers', ()) or ():
            if getattr(mod, 'type', '') != 'FLUID':
                continue
            if str(getattr(mod, 'fluid_type', '')) != 'DOMAIN':
                continue
            ds = getattr(mod, 'domain_settings', None)
            if ds is None:
                return None
            res = tuple(int(v) for v in
                        getattr(ds, 'domain_resolution', ()) or ())
            if len(res) != 3 or min(res) < 1:
                return None
            try:
                dens = np.asarray(list(ds.density_grid), np.float32)
            except Exception:                                   # noqa: BLE001
                dens = None
            want = res[0] * res[1] * res[2]
            if dens is None or dens.size != want:
                if warnings is not None:
                    warnings.append(
                        f"smoke domain '{getattr(ob, 'name', '?')}': no "
                        f"readable density grid (is the sim baked?) -- "
                        f"the volume marches on its node chain alone")
                return None
            flame = None
            try:
                fl = np.asarray(list(ds.flame_grid), np.float32)
                if fl.size == want:
                    flame = fl
            except Exception:                                   # noqa: BLE001
                flame = None
            return {'res': res, 'density': dens, 'flame': flame}
    except Exception:                                           # noqa: BLE001
        return None
    return None


def _info_snapshot(ob):
    """The ObjectInfo fields, read from the evaluated object."""
    return dict(
        name=ob.name,
        color=tuple(ob.color) if hasattr(ob, 'color') else (1, 1, 1, 1),
        index=int(getattr(ob, 'pass_index', 0)),
        random=float(abs(hash(ob.name)) % 10000) / 10000.0,
        visible_camera=getattr(ob, 'visible_camera', True),
        cast_shadow=getattr(ob, 'visible_shadow', True),
        holdout=getattr(ob, 'is_holdout', False),
        smoothresh=float(ob.get('halcyon_smoothresh', 0.0))
        if hasattr(ob, 'get') else 0.0,
        smoke_grid=_smoke_grid(ob))


def _info_object(info, matrix):
    mw = np.asarray(matrix, np.float32)
    return ObjectInfo(
        name=info['name'], location=tuple(mw[:3, 3]), matrix_world=mw,
        color=info['color'], index=info['index'], random=info['random'],
        visible_camera=info['visible_camera'],
        cast_shadow=info['cast_shadow'], holdout=info['holdout'],
        smoothresh=info['smoothresh'],
        smoke_grid=info.get('smoke_grid'))


def register():
    """Arm the cache: without every handler in place it stays off."""
    global _CACHE_ARMED
    if _CACHE_ARMED:
        return
    try:
        import bpy
        h = bpy.app.handlers
        h.depsgraph_update_post.append(_watch_updates)
        h.frame_change_post.append(_on_frame_change)
        for name in ('undo_post', 'redo_post', 'load_post'):
            getattr(h, name).append(_flush_mesh_cache)
    except Exception:                                           # noqa: BLE001
        unregister()
        return
    _CACHE_ARMED = True


def unregister():
    global _CACHE_ARMED, _DIRTY_ALL, _DIRTY_ALL_BY, _MESH_CACHE_BYTES
    global _SCENE_GEO_SIG
    _CACHE_ARMED = False
    _MESH_CACHE.clear()
    _MESH_CACHE_BYTES = 0
    _DIRTY_OBJS.clear()
    _DIRTY_DATA.clear()
    _DIRTY_ALL = False
    _DIRTY_ALL_BY = None
    _LAST_FRAME.clear()
    _SCENE_GEO_SIG = None
    try:
        import bpy
        h = bpy.app.handlers
    except Exception:                                           # noqa: BLE001
        return
    for fn, names in ((_watch_updates, ('depsgraph_update_post',)),
                      (_on_frame_change, ('frame_change_post',)),
                      (_flush_mesh_cache,
                       ('undo_post', 'redo_post', 'load_post'))):
        for name in names:
            try:
                hooks = getattr(h, name)
                if fn in hooks:
                    hooks.remove(fn)
            except Exception:                                   # noqa: BLE001
                pass

_KINDS = {'MESH': "Mesh", 'FONT': "Text", 'CURVE': "Curve",
          'SURFACE': "Surface", 'META': "Metaball", 'CURVES': "Hair curves",
          'POINTCLOUD': "Point cloud", 'VOLUME': "Volume",
          'GPENCIL': "Grease pencil", 'GREASEPENCIL': "Grease pencil"}


def _kind(ob):
    """What to call an object in a message, in the user's words not Blender's."""
    return _KINDS.get(getattr(ob, 'type', ''), "Object")


def export_lights_into(parked, depsgraph):
    """A next Scene with FRESH lights and everything else SHARED.

    A lamp drag fires view_update every tick, and each tick used to run
    the full main-thread re-export -- every mesh through foreach_get,
    every material through the serializer -- for data no lamp can
    change (~2 s per tick on the field's file: 'Blender will freeze
    up'). This walks depsgraph.object_instances for the LIGHTS only and
    shallow-copies the parked export around them, so the mesh, material,
    image and world objects keep their identities and every downstream
    identity-keyed cache (BVH, prepared textures, shadow signatures)
    stays warm."""
    import copy
    bscene = depsgraph.scene
    lights = []
    for inst in depsgraph.object_instances:
        ob = inst.object
        if ob is None or ob.type != 'LIGHT' or not inst.show_self:
            continue
        lights.append(export_light(ob, inst.matrix_world,
                                   bscene.unit_settings.scale_length))
    name_to_index = {o.name: i
                     for i, o in enumerate(parked.objects or [])}
    for lt in lights:
        names = getattr(lt, 'exclude_names', None)
        if names:
            lt.exclude_objects = tuple(sorted(
                name_to_index[n] for n in names if n in name_to_index))
    scene = copy.copy(parked)
    scene.lights = lights
    return scene


def _collect_halo_points(me, matrix, slots):
    """R191: the raw VERTICES of a mesh wearing halo materials.

    make_render_halos reads mvert directly -- every vertex, loose ones
    included (the classic halo workflow is a mesh of nothing but
    vertices) -- once per halo-material slot, so a mesh with two halo
    slots glows twice, exactly as 2.79 did. Returns None when no slot
    is a halo, else per-slot dicts of world positions (and world
    normals when the material's Vertex Normal scaling asks)."""
    halo_slots = [si for si, s in enumerate(slots)
                  if s is not None
                  and getattr(getattr(s, 'halcyon', None), 'halo', False)]
    if not halo_slots:
        return None
    try:
        n = len(me.vertices)
    except Exception:                                           # noqa: BLE001
        return None
    if n == 0:
        return None
    co = np.empty(n * 3, np.float32)
    me.vertices.foreach_get('co', co)
    mw = np.asarray(matrix, np.float32)
    pos = (co.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3]).astype(np.float32)
    out = []
    for si in halo_slots:
        hs = getattr(slots[si], 'halcyon', None)
        entry = {'slot': si, 'pos': pos}
        if getattr(hs, 'halo_puno', False):
            no = np.empty(n * 3, np.float32)
            try:
                me.vertices.foreach_get('normal', no)
                nw = no.reshape(-1, 3) @ np.linalg.inv(mw[:3, :3])
                nw /= np.maximum(np.linalg.norm(nw, axis=1),
                                 1e-9)[:, None]
                entry['normals'] = nw.astype(np.float32)
            except Exception:                                   # noqa: BLE001
                pass
        out.append(entry)
    return out


def _halo_groups_from(points, remap, materials):
    """Slot-local halo points -> final scene groups with running seeds."""
    groups = []
    for entry in points or ():
        si = int(entry['slot'])
        mi = int(remap[min(si, len(remap) - 1)])
        m = materials[mi] if 0 <= mi < len(materials) else None
        spec = getattr(m, 'halo', None) if m is not None else None
        if not spec:
            continue
        n = entry['pos'].shape[0]
        base = int(spec.get('seed', 0))
        groups.append({'mat': mi, 'pos': entry['pos'],
                       # seed1 + running vertex index, mod 256 at use
                       # exactly as RE_inithalo stores seed % 256
                       'seeds': (base + np.arange(n, dtype=np.int64)) % 256,
                       'normals': entry.get('normals')})
    return groups


def _strand_material(mat, mat_lookup, materials, images, warnings):
    """The hair CLONE of a material: same graph, marked as strand.

    Hair ribbons carry their strand data in the colour layer; the Hair
    Info node needs to know the difference between hair geometry and a
    mesh that merely has vertex paint. The material clone carries
    `strand=True` and its GRAPH root carries `'strand': True` -- the
    CPU evaluator and the GLSL emitter both read the graph flag, so
    the whole thing travels inside data that already flows to both
    devices. The original material is untouched: a mesh wearing it
    renders exactly as before.
    """
    key = ((mat.name_full if mat is not None else '__default__')
           + '::strand')
    if key in mat_lookup:
        return mat_lookup[key]
    exported = export_material(mat, images, warnings)
    try:
        exported.strand = True
        if getattr(exported, 'graph', None):
            exported.graph['strand'] = True
        exported.name = (getattr(exported, 'name', 'Material')
                         + '.strand')
    except Exception:                                           # noqa: BLE001
        pass
    mat_lookup[key] = len(materials)
    materials.append(exported)
    return mat_lookup[key]


def _collect_hair_parts(depsgraph, bscene, mat_lookup, materials, images,
                        objects, parts, warnings):
    """R208: hair renders. Particle-system hair and Curves objects both
    become camera-facing ribbon meshes (core.geometry.hair_ribbons --
    the primitive BI rendered strands as), wearing a strand-flagged
    clone of the emitter's material so its full node graph runs on the
    hair, Hair Info included. Every bpy touch is getattr-guarded: an
    API this Blender does not carry contributes nothing and breaks
    nothing.
    """
    from .core import geometry as _geo

    cam = getattr(bscene, 'camera', None)
    try:
        eye = tuple(np.asarray(cam.matrix_world, np.float32)[:3, 3]) \
            if cam is not None else (0.0, 0.0, 0.0)
    except Exception:                                           # noqa: BLE001
        eye = (0.0, 0.0, 0.0)

    def _push_part(strands, widths, uv_roots, mat, ob, label):
        if not strands:
            return
        try:
            root_w, tip_w = widths
            v, t, n, uv, c = _geo.hair_ribbons(
                strands, eye, root_width=root_w, tip_width=tip_w,
                uv_roots=uv_roots,
                seed=abs(hash(label)) % 65536)
            if not len(t):
                return
            mi = _strand_material(mat, mat_lookup, materials, images,
                                  warnings)
            obj_index = len(objects)
            fn = np.zeros((t.shape[0], 3), np.float32)
            a = v[t[:, 1]] - v[t[:, 0]]
            b = v[t[:, 2]] - v[t[:, 0]]
            fx = np.cross(a, b)
            ln = np.sqrt((fx * fx).sum(1, keepdims=True))
            fn = (fx / np.maximum(ln, 1e-12)).astype(np.float32)
            parts.append({
                'verts': v, 'normals': n, 'uvs': uv, 'uvs2': None,
                'colors': c, 'tris': t,
                'mat_index': np.full(t.shape[0], mi, np.int32),
                'obj_index': np.full(t.shape[0], obj_index, np.int32),
                'face_normals': fn,
                'smooth': np.ones(t.shape[0], bool),
                'uv_names': []})
            info = _info_snapshot(ob)
            objects.append(_info_object(info, np.eye(4,
                                                     dtype=np.float32)))
        except Exception as exc:                                # noqa: BLE001
            warnings.append(f"hair on '{getattr(ob, 'name', '?')}' "
                            f"could not be exported "
                            f"({type(exc).__name__}: {exc})")

    for inst in depsgraph.object_instances:
        ob = inst.object
        if ob is None or not getattr(inst, 'show_self', True):
            continue

        # ---- legacy particle-system hair ----
        for ps in (getattr(ob, 'particle_systems', None) or ()):
            try:
                st_ = ps.settings
                if str(getattr(st_, 'type', '')) != 'HAIR':
                    continue
                step = int(getattr(st_, 'render_step', 3))
                steps = (1 << max(min(step, 7), 1)) + 1
                n_par = len(getattr(ps, 'particles', ()) or ())
                n_child = len(getattr(ps, 'child_particles', ()) or ())
                total = n_par + n_child \
                    if str(getattr(st_, 'child_type', 'NONE')) != 'NONE' \
                    and n_child else n_par
                if total == 0:
                    continue
                strands = []
                uv_roots = []
                mod = next((m for m in getattr(ob, 'modifiers', ())
                            if getattr(m, 'type', '')
                            == 'PARTICLE_SYSTEM'
                            and getattr(m, 'particle_system', None)
                            and m.particle_system.name == ps.name),
                           None)
                for i in range(total):
                    pts = []
                    for k in range(steps):
                        co = ps.co_hair(ob, particle_no=i, step=k)
                        pts.append((co[0], co[1], co[2]))
                    P = np.asarray(pts, np.float32)
                    if not np.isfinite(P).all() or \
                            float(np.abs(np.diff(P, axis=0)).sum()) \
                            < 1e-9:
                        continue
                    strands.append(P)
                    uvr = None
                    if mod is not None and i < n_par:
                        try:
                            uvr = tuple(ps.uv_on_emitter(
                                mod, particle=ps.particles[i],
                                particle_no=i, uv_no=0))[:2]
                        except Exception:                       # noqa: BLE001
                            uvr = None
                    uv_roots.append(uvr if uvr is not None
                                    else (float(i) / max(total, 1), 0.0))
                if not strands:
                    continue
                rscale = float(getattr(st_, 'radius_scale', 0.01)
                               or 0.01)
                root_w = float(getattr(st_, 'root_radius', 1.0)
                               or 1.0) * rscale
                tip_w = float(getattr(st_, 'tip_radius', 0.0)
                              or 0.0) * rscale
                mslot = int(getattr(st_, 'material', 1)) - 1
                slots = list(getattr(ob, 'material_slots', ()) or ())
                mat = slots[mslot].material \
                    if 0 <= mslot < len(slots) else None
                _push_part(strands, (root_w, tip_w), uv_roots, mat, ob,
                           f'{ob.name}/{ps.name}')
            except Exception as exc:                            # noqa: BLE001
                warnings.append(
                    f"hair system on '{getattr(ob, 'name', '?')}' could "
                    f"not be exported ({type(exc).__name__}: {exc})")

        # ---- Curves-object hair (the sculptable kind) ----
        if getattr(ob, 'type', '') == 'CURVES':
            try:
                cd = ob.data
                pos_attr = cd.attributes.get('position')
                npts = len(pos_attr.data)
                co = np.empty(npts * 3, np.float32)
                pos_attr.data.foreach_get('vector', co)
                co = co.reshape(-1, 3)
                mw = np.asarray(inst.matrix_world, np.float32)
                co = (co @ mw[:3, :3].T + mw[:3, 3]).astype(np.float32)
                ncur = len(getattr(cd, 'curves', ()) or ())
                offs = np.empty(ncur + 1, np.int32)
                try:
                    cd.curve_offset_data.foreach_get('value', offs)
                except Exception:                               # noqa: BLE001
                    counts = npts // max(ncur, 1)
                    offs = (np.arange(ncur + 1) * counts).astype(np.int32)
                strands = [co[offs[i]:offs[i + 1]]
                           for i in range(ncur)
                           if offs[i + 1] - offs[i] >= 2]
                rad = 0.005
                try:
                    r_attr = cd.attributes.get('radius')
                    if r_attr is not None and len(r_attr.data):
                        rr = np.empty(len(r_attr.data), np.float32)
                        r_attr.data.foreach_get('value', rr)
                        rad = float(np.median(rr)) or 0.005
                except Exception:                               # noqa: BLE001
                    pass
                slots = list(getattr(ob, 'material_slots', ()) or ())
                mat = slots[0].material if slots else None
                _push_part(strands, (rad * 2.0, rad * 0.4), None, mat,
                           ob, f'{ob.name}/curves')
            except Exception as exc:                            # noqa: BLE001
                warnings.append(
                    f"hair curves '{getattr(ob, 'name', '?')}' could "
                    f"not be exported ({type(exc).__name__}: {exc})")


def _collect_extra_halos(depsgraph, mat_lookup, materials, images,
                         halo_groups, warnings):
    """Particles and point clouds as halos, API-guarded.

    BI rendered particles through the same halo road when their
    material was a halo. Blender 5.x may or may not still carry the
    legacy particle API, so every touch is getattr-guarded: if the
    API is gone this collects nothing and says nothing. Point-cloud
    objects contribute their points the same way."""
    import numpy as _np
    for inst in depsgraph.object_instances:
        ob = inst.object
        if ob is None or not getattr(inst, 'show_self', True):
            continue
        mw = _np.asarray(inst.matrix_world, _np.float32)
        # ---- legacy particle systems ----
        for ps in (getattr(ob, 'particle_systems', None) or ()):
            try:
                st_ = ps.settings
                mslot = int(getattr(st_, 'material', 1)) - 1
                slots = list(getattr(ob, 'material_slots', ()) or ())
                mat = slots[mslot].material if 0 <= mslot < len(slots) \
                    else None
                hs = getattr(mat, 'halcyon', None) if mat else None
                if not getattr(hs, 'halo', False):
                    continue
                locs = [tuple(p.location) for p in ps.particles
                        if getattr(p, 'alive_state', 'ALIVE') == 'ALIVE']
                if not locs:
                    continue
                key = mat.name_full
                if key not in mat_lookup:
                    mat_lookup[key] = len(materials)
                    materials.append(export_material(mat, images,
                                                     warnings))
                mi = mat_lookup[key]
                spec = materials[mi].halo or {}
                pos = _np.asarray(locs, _np.float32)
                n = pos.shape[0]
                base = int(spec.get('seed', 0))
                halo_groups.append({
                    'mat': mi, 'pos': pos,
                    'seeds': (base + _np.arange(n, dtype=_np.int64))
                             % 256,
                    'normals': None})
            except Exception:                                   # noqa: BLE001
                continue
        # ---- point clouds ----
        if getattr(ob, 'type', '') == 'POINTCLOUD':
            try:
                slots = list(getattr(ob, 'material_slots', ()) or ())
                mat = slots[0].material if slots else None
                hs = getattr(mat, 'halcyon', None) if mat else None
                if not getattr(hs, 'halo', False):
                    continue
                attr = ob.data.attributes.get('position')
                n = len(attr.data)
                co = _np.empty(n * 3, _np.float32)
                attr.data.foreach_get('vector', co)
                pos = (co.reshape(-1, 3) @ mw[:3, :3].T
                       + mw[:3, 3]).astype(_np.float32)
                key = mat.name_full
                if key not in mat_lookup:
                    mat_lookup[key] = len(materials)
                    materials.append(export_material(mat, images,
                                                     warnings))
                mi = mat_lookup[key]
                spec = materials[mi].halo or {}
                base = int(spec.get('seed', 0))
                halo_groups.append({
                    'mat': mi, 'pos': pos,
                    'seeds': (base + _np.arange(n, dtype=_np.int64))
                             % 256,
                    'normals': None})
            except Exception:                                   # noqa: BLE001
                continue


def export_scene(depsgraph, settings, warnings=None):
    """Evaluated depsgraph -> Scene."""
    import time as _time
    warnings = warnings if warnings is not None else []
    bscene = depsgraph.scene
    images = {}
    materials = []
    mat_lookup = {}
    objects = []
    lights = []
    parts = []
    halo_groups = []
    _t_all = _time.perf_counter()
    _sp = {'mesh_ms': 0.0, 'mat_ms': 0.0, 'meshes': 0, 'mats': 0,
           'cached': 0, 'cached_ms': 0.0, 'cached_dup': 0}

    # R171/R172: the mesh export cache -- see the block above EXPORT_SPLIT
    # R173: Simplify is fingerprinted here, not trusted to update reports
    global _SCENE_GEO_SIG, _DIRTY_ALL, _DIRTY_ALL_BY
    _sig = _scene_geo_sig(bscene)
    if _CACHE_ARMED and _SCENE_GEO_SIG is not None \
            and _sig != _SCENE_GEO_SIG:
        _DIRTY_ALL = True
        _DIRTY_ALL_BY = 'simplify'
    _SCENE_GEO_SIG = _sig
    _dirt_all_by, _dirt_objs, _dirt_data = _cache_scrub()
    _sp['dirt_all'] = _dirt_all_by or ''
    _sp['dirt_n'] = _dirt_objs + _dirt_data
    _dg_mode = str(getattr(depsgraph, 'mode', 'VIEWPORT'))
    _cache_on = _CACHE_ARMED and not _DIRTY_ALL
    _touched = set()          # keys stored or reused THIS export
    retries = []

    def _cache_key(orig, matrix):
        tok = _obj_token(orig)
        if tok is None:
            return None
        try:
            return (tok, _dg_mode,
                    np.asarray(matrix, np.float32).tobytes())
        except Exception:                                       # noqa: BLE001
            return None

    def _skip(orig, matrix):
        if not _cache_on:
            return False
        k = _cache_key(orig, matrix)
        return k is not None and k in _MESH_CACHE

    # consumed lazily on purpose: the generator frees each converted mesh on
    # the way to the next one, and wrapping it in list() would read every one
    # of them after it had been freed. See compat.evaluated_meshes.
    for ob, me, matrix, temp in compat.evaluated_meshes(depsgraph,
                                                        skip=_skip):
        if me is None:
            # the skip path: this object was never evaluated -- serve its
            # arrays from the cache. Materials still export FRESH every
            # frame (they are cheap and they are what the user tweaks
            # between renders); only the per-mesh foreach_get walls skip.
            k = _cache_key(getattr(ob, 'original', None) or ob, matrix)
            ent = _MESH_CACHE.get(k) if k is not None else None
            if ent is None:                     # cannot happen; stay safe
                continue
            _MESH_CACHE[k] = _MESH_CACHE.pop(k)             # LRU touch
            _t0 = _time.perf_counter()
            ok = True
            for key in ent['slot_keys']:
                if key in mat_lookup:
                    continue
                m_db = None
                if key != '__default__':
                    m_db = _material_by_key(key, depsgraph)
                    if m_db is None:
                        ok = False
                        break
                mat_lookup[key] = len(materials)
                materials.append(export_material(m_db, images, warnings))
                _sp['mats'] += 1
            _sp['mat_ms'] += (_time.perf_counter() - _t0) * 1000.0
            obj_index = len(objects)
            if not ok:
                # a stored slot name no longer names a material (renamed
                # since the store): the only exact answer is the live
                # path, run after the loop in the position reserved here
                retries.append((len(parts), obj_index, ob, matrix))
                parts.append(None)
                objects.append(None)
                continue
            _t0 = _time.perf_counter()
            remap = np.array([mat_lookup[kk] for kk in ent['slot_keys']],
                             np.int32)
            data = dict(ent['data'])
            data['uv_names'] = list(data['uv_names'])
            raw = ent['raw_mat']
            data['mat_index'] = remap[np.clip(raw, 0, len(remap) - 1)]
            data['obj_index'] = np.full(len(raw), obj_index, np.int32)
            parts.append(data)
            if ent.get('halos'):
                halo_groups += _halo_groups_from(ent['halos'], remap,
                                                 materials)
            # R178: the ObjectInfo fields are read FRESH from the
            # evaluated object -- an evaluated_get is a pointer lookup,
            # and six RNA reads per object cost ~1 ms for a whole
            # scene. This is what lets Collection and ViewLayer updates
            # be ignored above: a holdout toggle, a colour edit or a
            # visibility flag lands here every frame regardless of any
            # dirt. The stored snapshot remains the fallback for
            # environments without evaluated_get.
            info = ent['info']
            try:
                getter = getattr(ob, 'evaluated_get', None)
                _ev = getter(depsgraph) if getter is not None else ob
                info = _info_snapshot(_ev)
            except Exception:                                   # noqa: BLE001
                pass
            objects.append(_info_object(info, matrix))
            _sp['cached_ms'] += (_time.perf_counter() - _t0) * 1000.0
            _sp['cached'] += 1
            if k in _touched:
                # the same object at the same matrix TWICE in one frame
                # (an instance overlapping its original): a same-frame
                # dup -- the instrument separates these from real warmth
                _sp['cached_dup'] += 1
            _touched.add(k)
            continue
        try:
            slots = list(me.materials) if me.materials else [None]
            _t0 = _time.perf_counter()
            for slot in slots:
                key = slot.name_full if slot else '__default__'
                if key not in mat_lookup:
                    mat_lookup[key] = len(materials)
                    materials.append(export_material(slot, images, warnings))
                    _sp['mats'] += 1
            _sp['mat_ms'] += (_time.perf_counter() - _t0) * 1000.0
            remap = np.array([mat_lookup[(s.name_full if s else '__default__')]
                              for s in slots], np.int32)
            obj_index = len(objects)
            _t0 = _time.perf_counter()
            data = _mesh_arrays(me, matrix, 0, obj_index)
            _check_fur_strand_health(ob, me, data, warnings)
            halo_pts = _collect_halo_points(me, matrix, slots)
            _sp['mesh_ms'] += (_time.perf_counter() - _t0) * 1000.0
            _sp['meshes'] += 1
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
        if halo_pts:
            halo_groups += _halo_groups_from(halo_pts, remap, materials)
        if data is None:
            if halo_pts:
                # a mesh of nothing but vertices wearing a halo
                # material IS a render -- the classic sparkle object
                continue
            if getattr(ob, 'type', 'MESH') in compat.ALLOCATING_TYPES:
                # the usual reason a text object is invisible: it converted,
                # but to outlines rather than to faces
                warnings.append(
                    f"{_kind(ob)} '{ob.name}' has no faces to render. Give it "
                    f"a Fill Mode (Object Data ▸ Geometry) or an Extrude, or "
                    f"convert it to a mesh")
            continue
        info = _info_snapshot(ob)
        if _cache_on:
            # store BEFORE the in-place remap below: raw_mat is the
            # pre-remap slot-index array, and the shared 'data' arrays
            # exclude the two that are rebuilt per reuse
            _orig = getattr(ob, 'original', None) or ob
            _ck = _cache_key(_orig, matrix)
            if _ck is not None:
                _cache_store(_ck, {
                    'data': {kk: vv for kk, vv in data.items()
                             if kk not in ('mat_index', 'obj_index')},
                    'raw_mat': data['mat_index'].copy(),
                    'slot_keys': [(s.name_full if s else '__default__')
                                  for s in slots],
                    'data_uid': getattr(getattr(_orig, 'data', None),
                                        'session_uid', None),
                    'halos': halo_pts,
                    'info': info})
                _touched.add(_ck)
        idx = np.clip(data['mat_index'], 0, len(remap) - 1)
        data['mat_index'] = remap[idx]
        parts.append(data)
        objects.append(_info_object(info, matrix))

    # the RETRY pass: cached entries whose stored slot names went
    # unresolvable (a material rename). One live re-export each, one at a
    # time AFTER the generator has closed (its finally already freed the
    # last mesh, so the to_mesh lifetime rules hold), into the exact
    # parts/objects positions the loop reserved -- object numbering and
    # every later object's baked obj_index stay identical to a fully
    # live export.
    for i_part, obj_index, r_ob, r_matrix in retries:
        data = None
        info = None
        ev = None
        try:
            getter = getattr(r_ob, 'evaluated_get', None)
            ev = getter(depsgraph) if getter is not None else r_ob
            me = ev.to_mesh()
            r_halo_pts = None
            if me is not None:
                slots = list(me.materials) if me.materials else [None]
                _t0 = _time.perf_counter()
                for slot in slots:
                    key = slot.name_full if slot else '__default__'
                    if key not in mat_lookup:
                        mat_lookup[key] = len(materials)
                        materials.append(
                            export_material(slot, images, warnings))
                        _sp['mats'] += 1
                _sp['mat_ms'] += (_time.perf_counter() - _t0) * 1000.0
                remap = np.array(
                    [mat_lookup[(s.name_full if s else '__default__')]
                     for s in slots], np.int32)
                _t0 = _time.perf_counter()
                data = _mesh_arrays(me, r_matrix, 0, obj_index)
                r_halo_pts = _collect_halo_points(me, r_matrix, slots)
                if r_halo_pts:
                    halo_groups += _halo_groups_from(r_halo_pts, remap,
                                                     materials)
                _sp['mesh_ms'] += (_time.perf_counter() - _t0) * 1000.0
                _sp['meshes'] += 1
                info = _info_snapshot(ev)
                if data is not None:
                    if _cache_on:
                        # refresh the entry so the NEXT frame resolves
                        _rorig = getattr(ev, 'original', None) or r_ob
                        _ck = _cache_key(_rorig, r_matrix)
                        if _ck is not None:
                            _cache_store(_ck, {
                                'data': {kk: vv for kk, vv in data.items()
                                         if kk not in ('mat_index',
                                                       'obj_index')},
                                'raw_mat': data['mat_index'].copy(),
                                'slot_keys': [(s.name_full if s
                                               else '__default__')
                                              for s in slots],
                                'data_uid': getattr(
                                    getattr(_rorig, 'data', None),
                                    'session_uid', None),
                                'info': info})
                            _touched.add(_ck)
                    idx = np.clip(data['mat_index'], 0, len(remap) - 1)
                    data['mat_index'] = remap[idx]
        except Exception as exc:                                # noqa: BLE001
            import traceback
            traceback.print_exc()
            warnings.append(f"{_kind(r_ob)} '{getattr(r_ob, 'name', '?')}' "
                            f"could not be exported ({type(exc).__name__}: "
                            f"{exc}) and is missing from the render")
        finally:
            if ev is not None:
                compat.free_mesh(ev)
        if info is None:
            info = _info_snapshot(r_ob)
        parts[i_part] = data if data is not None else _empty_part()
        objects[obj_index] = _info_object(info, r_matrix)

    # R172: the stale sweep. Keys nobody visits again -- an instancer
    # empty dragged through fifty matrices, a scene emptied out -- used
    # to pile into a fixed cap whose LRU then evicted the OTHER pool
    # (the field's viewport churn silently wiped the F12 pool: '21
    # cached' where 171 should have been). Now each export sweeps ITS
    # OWN mode's untouched entries once they outnumber the live set 2:1
    # -- the other mode's pool is never touched, and the 2x hysteresis
    # spares alternating view layers the worst of it.
    if _cache_on and _touched:
        _mode_keys = [k for k in _MESH_CACHE if k[1] == _dg_mode]
        if len(_mode_keys) > 2 * len(_touched):
            for k in _mode_keys:
                if k not in _touched:
                    _cache_pop(k)

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

    # R191: particles and point clouds wearing halo materials
    try:
        _collect_extra_halos(depsgraph, mat_lookup, materials, images,
                             halo_groups, warnings)
    except Exception:                                           # noqa: BLE001
        pass

    # R208: hair -- particle-system strands and Curves objects become
    # camera-facing ribbons wearing a strand-flagged material clone
    try:
        _collect_hair_parts(depsgraph, bscene, mat_lookup, materials,
                            images, objects, parts, warnings)
    except Exception:                                           # noqa: BLE001
        pass

    _t0 = _time.perf_counter()
    mesh = _concat([p for p in parts if p is not None])
    _sp['concat_ms'] = (_time.perf_counter() - _t0) * 1000.0

    # R251 C020/C036: the authored shadow volumes leave the surface mesh
    # HERE, once, on the merged mesh with its FINAL global mat_index (the
    # per-object cache stores data without mat_index and remaps at every
    # export, so a split at the _mesh_arrays sites would freeze the role)
    shadow_volumes = _split_volumes(mesh, materials, objects)

    # R191: a halo material draws NO faces -- 2.79 turned the mesh into
    # halos and never converted its geometry. Drop the soup triangles
    # wearing a halo material (vertices stay; unreferenced is harmless)
    if halo_groups and mesh is not None and \
            getattr(mesh, 'tris', None) is not None and len(mesh.tris):
        _is_halo = np.array([1 if getattr(m, 'halo', None) else 0
                             for m in materials], bool)
        if _is_halo.any():
            _keep = ~_is_halo[np.clip(mesh.mat_index, 0,
                                      len(materials) - 1)]
            if not _keep.all():
                mesh.tris = mesh.tris[_keep]
                mesh.mat_index = mesh.mat_index[_keep]
                mesh.obj_index = mesh.obj_index[_keep]
                if getattr(mesh, 'face_normals', None) is not None:
                    mesh.face_normals = mesh.face_normals[_keep]
                if getattr(mesh, 'smooth', None) is not None:
                    mesh.smooth = mesh.smooth[_keep]

    cam_ob = bscene.camera
    camera = Camera()
    if cam_ob is not None:
        cam = cam_ob.data
        # R205: strip object scale from the camera at the source. A
        # camera under a scaled import rig ($$$DUMMY chains) carries the
        # parents' unit scale in matrix_world; the picture composes the
        # same but fog, clip and depth of field read distances
        # multiplied by the inverse scale -- the field's pagoda drowned
        # in fog colour on F12 while the viewport (whose camera-view
        # matrix Blender keeps rigid) looked right. Blender's renderers
        # ignore camera scale; Halcyon does too, saying so once.
        _raw_mw = np.asarray(cam_ob.matrix_world, np.float32)
        from .core import mathx as _mx
        camera.matrix_world = _mx.rigid_camera_matrix(_raw_mw)
        if camera.matrix_world is not _raw_mw:
            _sc = np.sqrt((_raw_mw[:3, :3] ** 2).sum(axis=0))
            print(f"[Halcyon] camera '{getattr(cam_ob, 'name', '?')}' "
                  f"carries object scale ~{float(_sc.mean()):.4g} from "
                  "its parents; ignored for rendering (fog, clip and "
                  "depth of field read true distances), exactly as "
                  "Blender's own renderers do")
        camera.type = cam.type
        camera.lens = float(getattr(cam, 'lens', 50.0))
        camera.sensor = float(getattr(cam, 'sensor_width', 36.0))
        camera.clip_start = float(cam.clip_start)
        camera.clip_end = float(cam.clip_end)
        camera.ortho_scale = float(getattr(cam, 'ortho_scale', 6.0))
        camera.shift_x = float(cam.shift_x)
        camera.shift_y = float(cam.shift_y)
        camera.dof = bool(getattr(cam.dof, 'use_dof', False))
        # R251 C098: the lens-pass road reads the camera's own f-number
        camera.fstop = float(getattr(cam.dof, 'aperture_fstop', 2.8))
        camera.focus_distance = float(getattr(cam.dof, 'focus_distance', 5.0))
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
                              'mist_intensity', 'ground_image'):
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
            # R251 C048: the Mode 7 map, an image datablock like env_image
            img7 = getattr(hs, 'ground_image', None)
            if img7 is not None:
                key = img7.name_full
                if key not in images:
                    px = compat.image_pixels(img7)
                    if px is not None:
                        images[key] = ImageBuffer(name=key, pixels=px,
                                                  colorspace='Linear')
                world.ground_image = images.get(key)
            # R251 C038: the rear-plane depth bitmap -- the red channel of
            # a float Z pass in Non-Color space (an 8-bit or colour-managed
            # image is not a distance)
            imgd = getattr(hs, 'backdrop_depth_image', None)
            if imgd is not None:
                _cs = getattr(getattr(imgd, 'colorspace_settings', None),
                              'name', '')
                if not bool(getattr(imgd, 'is_float', False)) or \
                        str(_cs) != 'Non-Color':
                    print('[Halcyon] Backdrop Depth ignored: the image must '
                          'be a float Z pass in Non-Color space (an 8-bit '
                          'or colour-managed image is not a distance)')
                else:
                    pxd = compat.image_pixels(imgd)
                    if pxd is not None:
                        world.backdrop_depth = np.ascontiguousarray(
                            np.asarray(pxd, np.float32)[:, :, 0])
                        world.backdrop_offset = (
                            int(getattr(hs, 'backdrop_offset_x', 0)),
                            int(getattr(hs, 'backdrop_offset_y', 0)))
            # R203: the material-ground road -- the picked material's
            # tree serialized onto the world, programs beside it, so
            # the bpy-free plane can wear it
            gm = getattr(hs, 'ground_material', None)
            if gm is not None and str(world.ground_mode) == 'MATERIAL':
                try:
                    if compat.uses_nodes(gm) and gm.node_tree:
                        progs = {}
                        world.ground_graph = serialize_tree(
                            gm.node_tree, images, progs, warnings)
                        world.ground_programs = progs
                except Exception:                               # noqa: BLE001
                    warnings.append(
                        f"Ground material '{gm.name}' could not "
                        "serialize; the plane falls back to Solid")
        if compat.uses_nodes(bw) and bw.node_tree:
            world.graph = serialize_tree(bw.node_tree, images, {}, warnings)
    if bscene.render.use_freestyle:
        warnings.append("Freestyle is not supported; use the Wireframe shader instead")

    # a UV Map node referencing a layer name outside the merged pair:
    # if some object carries that name as its SECOND layer, promote it
    # into slot 1 -- each object's uvs2 rows are its own second layer,
    # so the by-name lookup lands on the right DATA (the field's
    # 'Blood' splatters, present on one head mesh only)
    try:
        pairs = getattr(mesh, 'uv_name_pairs', None) or []
        have = set(mesh.uv_names or ())
        wanted = set()
        for m in materials:
            for node in ((getattr(m, 'graph', None) or {})
                         .get('nodes', {})).values():
                if node.get('bl_idname') == 'ShaderNodeUVMap':
                    nm = (node.get('props') or {}).get('uv_map')
                    if nm:
                        wanted.add(str(nm))
        for nm in sorted(wanted - have):
            if any(len(p) > 1 and p[1] == nm for p in pairs):
                names = list(mesh.uv_names or [''])
                while len(names) < 2:
                    names.append('')
                names[1] = nm
                mesh.uv_names = names
                break
    except Exception:                                           # noqa: BLE001
        pass

    sc = Scene(mesh=mesh, materials=materials or [Material()], objects=objects,
               shadow_volumes=shadow_volumes,
               lights=lights, camera=camera, world=world, settings=settings,
               frame=bscene.frame_current, fps=bscene.render.fps,
               time=bscene.frame_current / max(bscene.render.fps, 1))
    sc.halos = halo_groups or None
    sc.images = images
    sc.warnings = warnings
    # R170: the export split -- at 0.36s of a 0.75s warm F12, "export"
    # is a number and a split is a diagnosis (engine prints it)
    _sp['total_ms'] = (_time.perf_counter() - _t_all) * 1000.0
    _sp['other_ms'] = max(_sp['total_ms'] - _sp['mesh_ms']
                          - _sp['mat_ms'] - _sp.get('concat_ms', 0.0)
                          - _sp.get('cached_ms', 0.0), 0.0)
    EXPORT_SPLIT.clear()
    EXPORT_SPLIT.update(_sp)
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


def _split_volumes(mesh, materials, objects):
    """R251 C020/C036: pull every volume-role material's triangles OUT of
    the merged mesh into Scene.shadow_volumes entries (one per (object,
    material) pair, object order then material order) and compact the
    corner arrays to the corners the kept triangles reference -- so a
    volume never touches the surface mesh, not even its bounds (an
    unreferenced far corner would coarsen every shadow map's texel).
    Returns the list; the mesh is edited in place."""
    vols = []
    if mesh is None or getattr(mesh, 'tris', None) is None or \
            not len(mesh.tris) or not materials or \
            getattr(mesh, 'mat_index', None) is None:
        return vols
    roles = np.array([str(getattr(m, 'volume_role', 'NONE') or 'NONE')
                      for m in materials])
    mi_all = np.clip(mesh.mat_index, 0, len(materials) - 1)
    vol = roles[mi_all] != 'NONE'
    if not vol.any():
        return vols
    oi_all = mesh.obj_index if getattr(mesh, 'obj_index', None) is not None \
        else np.zeros(len(mesh.tris), np.int32)
    pairs = sorted({(int(o), int(m))
                    for o, m in zip(oi_all[vol], mi_all[vol])})
    for oi, mi in pairs:
        sel = vol & (oi_all == oi) & (mi_all == mi)
        t = mesh.tris[sel]
        used, inv = np.unique(t.ravel(), return_inverse=True)
        mat = materials[mi]
        name = objects[oi].name if objects and 0 <= oi < len(objects) \
            else f'object {oi}'
        vols.append({
            'name': str(name),
            'verts': np.ascontiguousarray(mesh.verts[used], np.float32),
            'tris': inv.reshape(t.shape).astype(np.int32),
            'role': str(mat.volume_role),
            'polygon_id': int(getattr(mat, 'polygon_id', 0)),
            'alpha': int(getattr(mat, 'shadow_alpha', 16)),
            'color': tuple(float(v) for v in
                           (getattr(mat, 'diffuse', None) or (0.8, 0.8, 0.8))[:3]),
        })
    keep = ~vol
    for name in ('tris', 'mat_index', 'obj_index', 'face_normals', 'smooth',
                 'ink_tri_mask'):
        a = getattr(mesh, name, None)
        if a is not None and len(a) == keep.size:
            setattr(mesh, name, a[keep])
    keep_v = np.zeros(mesh.verts.shape[0], bool)
    if len(mesh.tris):
        keep_v[mesh.tris.ravel()] = True
    idx = np.cumsum(keep_v) - 1
    if len(mesh.tris):
        mesh.tris = idx[mesh.tris].astype(np.int32)
    for name in ('verts', 'normals', 'uvs', 'uvs2', 'colors'):
        a = getattr(mesh, name, None)
        if a is not None and len(a) == keep_v.size:
            setattr(mesh, name, a[keep_v])
    return vols


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
    # Every part's pair is kept alongside so the scene build can put a
    # NODE-REFERENCED second name (the field's 'Blood') into the slot.
    mesh.uv_names = next((p['uv_names'] for p in parts
                          if p.get('uv_names')), [])
    mesh.uv_name_pairs = [tuple(p['uv_names']) for p in parts
                          if p.get('uv_names')]
    # R246: the colour layer's name, first-seen, the same way
    mesh.color_name = next((p['color_name'] for p in parts
                            if p.get('color_name')), None)
    mesh.colors = np.concatenate(C)
    mesh.tris = np.concatenate(T)
    mesh.mat_index = np.concatenate(MI)
    mesh.obj_index = np.concatenate(OI)
    mesh.face_normals = np.concatenate(FN)
    mesh.smooth = np.concatenate(S)
    # R220: the marked-edge tri masks ride alongside like smooth --
    # per-tri, so no rebasing; .get() keeps parts from a pre-R220 mesh
    # cache legal, and a part without marks contributes zeros
    IM = [p.get('ink_tri_mask') for p in parts]
    if any(m is not None for m in IM):
        mesh.ink_tri_mask = np.concatenate(
            [m if m is not None
             else np.zeros(p['tris'].shape[0], np.uint8)
             for m, p in zip(IM, parts)])
    return mesh
