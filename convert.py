"""Convert existing materials to the Halcyon Shader.

The point is that this is not a reset button: whatever was feeding the source
BSDF -- image textures, ramps, whole node networks -- is relinked onto the
equivalent input of the Halcyon Shader. A material with a texture in Base Color
comes out with that same texture in Diffuse Color.
"""

import bpy
from bpy.props import (BoolProperty, EnumProperty, IntProperty,
                       StringProperty)
from bpy.types import Operator

from . import compat
from .core.convert import (BI_NODE, MASTER_NODE, TARGET_NODES, bi_plan,
                           plan, plan_for)

OUTPUT_NODES = ('ShaderNodeOutputMaterial',)


def _active_output(tree):
    best = None
    for node in tree.nodes:
        if node.bl_idname in OUTPUT_NODES:
            if getattr(node, 'is_active_output', False):
                return node
            best = best or node
    return best


def _surface_source(out_node):
    """The node feeding Surface, stepping through reroutes and mix shaders."""
    if out_node is None:
        return None, []
    notes = []
    sock = out_node.inputs.get('Surface')
    seen = 0
    while sock is not None and sock.is_linked and seen < 16:
        seen += 1
        node = sock.links[0].from_node
        if node.bl_idname == 'NodeReroute':
            sock = node.inputs[0]
            continue
        if node.bl_idname in ('ShaderNodeMixShader', 'ShaderNodeAddShader'):
            # follow the heavier branch; a single master shader cannot carry two
            inputs = [s for s in node.inputs if s.type == 'SHADER']
            pick = inputs[-1] if inputs else None
            if node.bl_idname == 'ShaderNodeMixShader' and len(inputs) == 2:
                try:
                    fac = node.inputs['Fac'].default_value
                except (KeyError, AttributeError):
                    fac = 0.5
                pick = inputs[1] if fac >= 0.5 else inputs[0]
            notes.append(f"{node.bl_idname.replace('ShaderNode', '')} collapsed "
                         "to its dominant branch")
            if pick is None or not pick.is_linked:
                return None, notes
            sock = pick
            continue
        return node, notes
    return None, notes


def _gather(node):
    values, links = {}, set()
    for sock in node.inputs:
        if sock.is_linked:
            links.add(sock.name)
        else:
            v = getattr(sock, 'default_value', None)
            if v is not None:
                try:
                    values[sock.name] = list(v) if hasattr(v, '__len__') \
                        and not isinstance(v, str) else v
                except TypeError:
                    pass
    return values, links


def _gather_sources(node):
    """R253: {socket name: bl_idname of the node feeding it} for the
    linked inputs -- what plan_for's vertex-colour detection reads
    (a Vertex Color node into Base Color is the console's own material
    source, not a texture chain). _gather stays as it is."""
    out = {}
    for sock in node.inputs:
        if not getattr(sock, 'is_linked', False):
            continue
        try:
            out[sock.name] = sock.links[0].from_node.bl_idname
        except (AttributeError, IndexError):
            pass
    return out


def convert_material(mat, model='AUTO', keep_original=True, force=False):
    """Rebuild `mat` around a Halcyon Shader. Returns (ok, message)."""
    if mat is None:
        return False, "no material"
    from . import compat
    compat.enable_nodes(mat)
    tree = mat.node_tree
    if tree is None:
        return False, f"{mat.name}: no node tree"

    existing = [n for n in tree.nodes if n.bl_idname == MASTER_NODE]
    if existing and not force:
        return False, f"{mat.name}: already uses the Halcyon Shader"

    out = _active_output(tree)
    if out is None:
        out = tree.nodes.new('ShaderNodeOutputMaterial')
        out.location = (400, 0)

    source, notes = _surface_source(out)
    if source is None:
        values, links = {}, set()
        src_id = 'ShaderNodeBsdfPrincipled'
        notes.append("no source shader found; started from defaults")
    else:
        values, links = _gather(source)
        src_id = source.bl_idname

    p = plan(src_id, values, links, model)
    notes.extend(p['notes'])

    master = tree.nodes.new(MASTER_NODE)
    master.model = p['model']
    if source is not None:
        master.location = (source.location.x, source.location.y)
        master.label = f"from {source.bl_label}"
    else:
        master.location = (out.location.x - 260, out.location.y)

    carried = 0
    for target, alias in p['pairs']:
        dst = master.inputs.get(target)
        if dst is None or source is None:
            continue
        src_sock = source.inputs.get(alias)
        if src_sock is None:
            continue
        if src_sock.is_linked:
            try:
                tree.links.new(src_sock.links[0].from_socket, dst)
                carried += 1
            except Exception:                                   # noqa: BLE001
                pass
        elif hasattr(dst, 'default_value') and hasattr(src_sock, 'default_value'):
            try:
                sv = src_sock.default_value
                if hasattr(dst.default_value, '__len__') and hasattr(sv, '__len__'):
                    n = min(len(dst.default_value), len(sv))
                    for i in range(n):
                        dst.default_value[i] = sv[i]
                elif hasattr(dst.default_value, '__len__'):
                    for i in range(3):
                        dst.default_value[i] = float(sv)
                elif hasattr(sv, '__len__'):
                    dst.default_value = float(sv[0])
                else:
                    dst.default_value = float(sv)
                carried += 1
            except (TypeError, ValueError, IndexError):
                pass

    for name, value in p['extras'].items():
        sock = master.inputs.get(name)
        if sock is not None and not sock.is_linked and hasattr(sock, 'default_value'):
            try:
                sock.default_value = value
            except (TypeError, ValueError):
                pass

    # links the plan marked as needing a scale -- emission strength being
    # folded into a LINKED emission colour goes through a real multiply
    # node, so the strength applies exactly instead of being a console note
    for target, spec in (p.get('scale_links') or {}).items():
        dst = master.inputs.get(target)
        if dst is None or not dst.is_linked or source is None:
            continue
        try:
            upstream = dst.links[0].from_socket
            mul, fac, in_a, in_b, out_c = _new_multiply(tree)
            if mul is None:
                continue
            mul.location = (master.location.x - 200, master.location.y - 260)
            mul.label = "Emission Strength"
            fac.default_value = 1.0
            tree.links.new(upstream, in_a)
            kind, val = spec
            wired = False
            if kind == 'LINK':
                s_sock = source.inputs.get(val)
                if s_sock is not None and s_sock.is_linked:
                    tree.links.new(s_sock.links[0].from_socket, in_b)
                    wired = True
                elif s_sock is not None:
                    val = float(getattr(s_sock, 'default_value', 1.0) or 0.0)
            if not wired:
                in_b.default_value = (float(val), float(val), float(val), 1.0)
            tree.links.new(out_c, dst)   # replaces the direct link
        except Exception:                                       # noqa: BLE001
            pass

    try:
        tree.links.new(master.outputs['Surface'], out.inputs['Surface'])
    except Exception as exc:                                    # noqa: BLE001
        return False, f"{mat.name}: could not link output ({exc})"

    if source is not None:
        if keep_original:
            source.location = (source.location.x, source.location.y - 340)
            source.label = "replaced by Halcyon Shader"
            source.mute = True
        else:
            tree.nodes.remove(source)

    master.refresh_sockets()
    mat.halcyon.use_override = False
    msg = f"{mat.name}: {p['model']}, {carried} inputs carried"
    if notes:
        msg += " (" + "; ".join(notes[:2]) + ")"
    return True, msg


def _carry(tree, source, node, pairs, extras, scale_links):
    """R253: land a plan on `node` -- the pairs loop, the extras loop and
    the scale_links multiply block of convert_material, parameterised on
    the node so the Anime / Cartoon / Console conversions share the
    master's surgery. Returns the carried count. convert_material keeps
    its own copy this round (its pinned strings stay where they are)."""
    carried = 0
    for target, alias in pairs:
        dst = node.inputs.get(target)
        if dst is None or source is None:
            continue
        src_sock = source.inputs.get(alias)
        if src_sock is None:
            continue
        if src_sock.is_linked:
            try:
                tree.links.new(src_sock.links[0].from_socket, dst)
                carried += 1
            except Exception:                                   # noqa: BLE001
                pass
        elif hasattr(dst, 'default_value') and hasattr(src_sock, 'default_value'):
            try:
                sv = src_sock.default_value
                if hasattr(dst.default_value, '__len__') and hasattr(sv, '__len__'):
                    n = min(len(dst.default_value), len(sv))
                    for i in range(n):
                        dst.default_value[i] = sv[i]
                elif hasattr(dst.default_value, '__len__'):
                    for i in range(3):
                        dst.default_value[i] = float(sv)
                elif hasattr(sv, '__len__'):
                    dst.default_value = float(sv[0])
                else:
                    dst.default_value = float(sv)
                carried += 1
            except (TypeError, ValueError, IndexError):
                pass

    for name, value in extras.items():
        sock = node.inputs.get(name)
        if sock is None or sock.is_linked or not hasattr(sock, 'default_value'):
            continue
        try:
            # element-wise for colours (a 3-tuple onto an RGBA socket
            # must not be refused), straight for floats
            if hasattr(sock.default_value, '__len__') and hasattr(value, '__len__'):
                n = min(len(sock.default_value), len(value))
                for i in range(n):
                    sock.default_value[i] = value[i]
            elif hasattr(sock.default_value, '__len__'):
                for i in range(3):
                    sock.default_value[i] = float(value)
            else:
                sock.default_value = float(value[0]) \
                    if hasattr(value, '__len__') else float(value)
        except (TypeError, ValueError, IndexError):
            pass

    # links the plan marked as needing a scale -- emission strength being
    # folded into a LINKED emission colour goes through a real multiply
    # node, so the strength applies exactly instead of being a console note
    for target, spec in (scale_links or {}).items():
        dst = node.inputs.get(target)
        if dst is None or not dst.is_linked or source is None:
            continue
        try:
            upstream = dst.links[0].from_socket
            mul, fac, in_a, in_b, out_c = _new_multiply(tree)
            if mul is None:
                continue
            mul.location = (node.location.x - 200, node.location.y - 260)
            mul.label = "Emission Strength"
            fac.default_value = 1.0
            tree.links.new(upstream, in_a)
            kind, val = spec
            wired = False
            if kind == 'LINK':
                s_sock = source.inputs.get(val)
                if s_sock is not None and s_sock.is_linked:
                    tree.links.new(s_sock.links[0].from_socket, in_b)
                    wired = True
                elif s_sock is not None:
                    val = float(getattr(s_sock, 'default_value', 1.0) or 0.0)
            if not wired:
                in_b.default_value = (float(val), float(val), float(val), 1.0)
            tree.links.new(out_c, dst)   # replaces the direct link
        except Exception:                                       # noqa: BLE001
            pass
    return carried


#: R253: the labels the skip message and the panel use per target
_TARGET_LABELS = {'ANIME': "Anime Shader", 'CARTOON': "Cartoon Shader",
                  'CONSOLE': "Console Emulation Shader"}


def convert_material_to(mat, target, choice=None, keep_original=True,
                        force=False):
    """R253: rebuild `mat` around the Anime Shader, the Cartoon Shader or
    the Console Emulation Shader (`target` in TARGET_NODES). Returns
    (ok, message).

    The same relink-not-reset contract as convert_material: whatever fed
    the source's Base Color arrives in Diffuse Color / Paint Color, the
    normal chain in Normal, the alpha in Opacity, the emission colour and
    strength in Self-Illumination / Emission Strength; the constants go
    through plan_for's mapping. The source may be any Blender shader or
    one of the engine's own nodes -- the master included -- and carries
    by socket name. `choice` is the scene's menu for the target: the
    anime Style and Compatibility, the cartoon Era, the console machine.
    """
    if mat is None:
        return False, "no material"
    if target not in TARGET_NODES:
        return False, f"{mat.name}: unknown target {target!r}"
    compat.enable_nodes(mat)
    tree = mat.node_tree
    if tree is None:
        return False, f"{mat.name}: no node tree"
    node_id = TARGET_NODES[target]
    label = _TARGET_LABELS[target]

    existing = [n for n in tree.nodes if n.bl_idname == node_id]
    if existing and not force:
        # 'already uses' is what the operator counts as skipped
        return False, f"{mat.name}: already uses the {label}"

    out = _active_output(tree)
    if out is None:
        out = tree.nodes.new('ShaderNodeOutputMaterial')
        out.location = (400, 0)

    source, notes = _surface_source(out)
    if source is None:
        values, links, sources = {}, set(), {}
        src_id = 'ShaderNodeBsdfPrincipled'
        notes.append("no source shader found; started from defaults")
    else:
        values, links = _gather(source)
        sources = _gather_sources(source)
        src_id = source.bl_idname

    p = plan_for(target, src_id, values, links, choice, sources)
    notes.extend(p['notes'])

    node = tree.nodes.new(node_id)
    # the props in plan order: compat then style (the style's update
    # writes sockets, which the carried values then override), era,
    # the console first and its type switches after it
    for k, v in p['props'].items():
        try:
            setattr(node, k, v)
        except (AttributeError, TypeError, ValueError):
            pass
    # Blender fires the Style / Era update on setattr; the explicit call
    # is idempotent and is what the headless fake exercises
    try:
        if 'style' in p['props'] and callable(getattr(node, 'apply_style', None)):
            node.apply_style(p['props']['style'])
        if 'era' in p['props'] and callable(getattr(node, 'apply_era', None)):
            node.apply_era(p['props']['era'])
    except Exception:                                           # noqa: BLE001
        pass
    if source is not None:
        node.location = (source.location.x, source.location.y)
        node.label = f"from {source.bl_label}"
    else:
        node.location = (out.location.x - 260, out.location.y)

    carried = _carry(tree, source, node, p['pairs'], p['extras'],
                     p.get('scale_links'))

    try:
        tree.links.new(node.outputs['Surface'], out.inputs['Surface'])
    except Exception as exc:                                    # noqa: BLE001
        return False, f"{mat.name}: could not link output ({exc})"

    if source is not None:
        if keep_original:
            source.location = (source.location.x, source.location.y - 340)
            source.label = f"replaced by {getattr(node, 'bl_label', label)}"
            source.mute = True
        else:
            tree.nodes.remove(source)

    # the console hides the sockets its machine ignores (a linked one
    # stays visible; a carried constant on a hidden socket is unused)
    try:
        if callable(getattr(node, 'refresh_sockets', None)):
            node.refresh_sockets()
    except Exception:                                           # noqa: BLE001
        pass
    mat.halcyon.use_override = False
    msg = f"{mat.name}: {p['label']}, {carried} inputs carried"
    if notes:
        msg += " (" + "; ".join(notes[:2]) + ")"
    return True, msg


class HALCYON_OT_convert_to_node(Operator):
    """R253: Convert to Anime / Cartoon / Game -- one operator, a target
    enum, the three scopes. The panel pushes the scene's menu choices
    in through the string properties (as `model` rides on the master
    conversion), so no enum is duplicated on the operator."""

    bl_idname = 'halcyon.convert_to_node'
    bl_label = "Convert to Anime / Cartoon / Game"
    bl_description = ("Rebuild materials around the Anime Shader, the "
                      "Cartoon Shader or the Console Emulation Shader, "
                      "with textures, normal chains, alpha and emission "
                      "relinked")
    bl_options = {'REGISTER', 'UNDO'}

    target: EnumProperty(name="Target", default='ANIME', items=(
        ('ANIME', "Anime Shader",
         "The anime/cel master: tone bands, game-texture decodes, ramps"),
        ('CARTOON', "Cartoon Shader",
         "The Western cel: paint, one shadow tone, an era preset"),
        ('CONSOLE', "Game (Console Emulation)",
         "A period machine's light unit and combiner, by console")),
        description="Which of the engine's masters the materials are "
                    "rebuilt around")
    scope: EnumProperty(name="Scope", default='ACTIVE', items=(
        ('ACTIVE', "Active Material", "Only the active material slot"),
        ('SELECTED', "Selected Objects", "Every material on the selection"),
        ('SCENE', "Whole Scene", "Every material in the scene")),
        description="Which materials the conversion walks")
    style: StringProperty(name="Anime Style", default='CUSTOM')
    compat: StringProperty(name="Anime Compatibility", default='GENERIC')
    era: StringProperty(name="Cartoon Era", default='CUSTOM')
    console: StringProperty(name="Console", default='PS1')
    keep_original: BoolProperty(
        name="Keep Original Shader", default=True,
        description="Mute the old shader and move it aside instead of "
                    "deleting it, so the conversion can be inspected")
    force: BoolProperty(
        name="Reconvert", default=False,
        description="Convert again even if the material already uses the "
                    "target node")

    def execute(self, context):
        mats = _materials_for(context, self.scope)
        if not mats:
            self.report({'WARNING'}, "No materials found for that scope")
            return {'CANCELLED'}
        choice = {'style': self.style, 'compat': self.compat,
                  'era': self.era, 'console': self.console}
        done, skipped, failed = 0, 0, 0
        for mat in mats:
            try:
                ok, msg = convert_material_to(mat, self.target, choice,
                                              self.keep_original, self.force)
            except Exception as exc:                            # noqa: BLE001
                import traceback
                traceback.print_exc()
                ok, msg = False, f"{mat.name}: {exc}"
            print("[Halcyon convert]", msg)
            if ok:
                done += 1
            elif 'already uses' in msg:
                skipped += 1
            else:
                failed += 1
        parts = [f"{done} converted"]
        if skipped:
            parts.append(f"{skipped} already converted")
        if failed:
            parts.append(f"{failed} failed")
        self.report({'INFO' if not failed else 'WARNING'},
                    "Halcyon: " + ", ".join(parts) + " (details in console)")
        return {'FINISHED'}


def _new_multiply(tree):
    """A colour-multiply node on whichever Mix node this Blender has.

    Returns (node, fac_socket, a_socket, b_socket, out_socket) or a None
    node. The modern ShaderNodeMix hides its per-type sockets behind
    identifiers, so they are found by identifier first and by socket type
    as the fallback; the legacy ShaderNodeMixRGB is the second try for
    older builds.
    """
    try:
        n = tree.nodes.new('ShaderNodeMix')
        n.data_type = 'RGBA'
        n.blend_type = 'MULTIPLY'
        a = b = None
        for s in n.inputs:
            ident = getattr(s, 'identifier', '')
            if ident in ('A_Color', 'A'):
                a = s
            elif ident in ('B_Color', 'B'):
                b = s
        if a is None or b is None:
            cols = [s for s in n.inputs if getattr(s, 'type', '') == 'RGBA']
            if len(cols) >= 2:
                a, b = cols[0], cols[1]
        out = None
        for s in n.outputs:
            if getattr(s, 'type', '') == 'RGBA':
                out = s
                break
        fac = n.inputs[0]
        if a is not None and b is not None and out is not None:
            return n, fac, a, b, out
        tree.nodes.remove(n)
    except Exception:                                           # noqa: BLE001
        pass
    try:
        n = tree.nodes.new('ShaderNodeMixRGB')
        n.blend_type = 'MULTIPLY'
        return (n, n.inputs['Fac'], n.inputs['Color1'], n.inputs['Color2'],
                n.outputs['Color'])
    except Exception:                                           # noqa: BLE001
        return None, None, None, None, None


def _materials_for(context, scope):
    seen, out = set(), []

    def add(mat):
        if mat is not None and mat.name_full not in seen:
            seen.add(mat.name_full)
            out.append(mat)

    if scope == 'ACTIVE':
        ob = context.active_object
        if ob is not None and ob.active_material is not None:
            add(ob.active_material)
        elif getattr(context, 'material', None) is not None:
            add(context.material)
    elif scope == 'SELECTED':
        for ob in context.selected_objects:
            for slot in getattr(ob, 'material_slots', []):
                add(slot.material)
    else:
        for ob in context.scene.objects:
            for slot in getattr(ob, 'material_slots', []):
                add(slot.material)
    return out


class HALCYON_OT_convert_materials(Operator):
    """Rebuild materials around the Halcyon Shader, relinking their textures"""

    bl_idname = 'halcyon.convert_materials'
    bl_label = "Convert to Halcyon Shader"
    bl_options = {'REGISTER', 'UNDO'}

    scope: EnumProperty(name="Scope", default='ACTIVE', items=(
        ('ACTIVE', "Active Material", "Just the active material"),
        ('SELECTED', "Selected Objects", "Every material on the selected objects"),
        ('SCENE', "Whole Scene", "Every material used in the scene")))
    model: StringProperty(name="Model", default='AUTO')
    keep_original: BoolProperty(
        name="Keep Original Shader", default=True,
        description="Mute the old shader and move it aside instead of deleting "
                    "it, so the conversion can be inspected")
    force: BoolProperty(
        name="Reconvert", default=False,
        description="Convert again even if the material already uses the "
                    "Halcyon Shader")

    def execute(self, context):
        mats = _materials_for(context, self.scope)
        if not mats:
            self.report({'WARNING'}, "No materials found for that scope")
            return {'CANCELLED'}
        done, skipped, failed = 0, 0, 0
        for mat in mats:
            try:
                ok, msg = convert_material(mat, self.model, self.keep_original,
                                           self.force)
            except Exception as exc:                            # noqa: BLE001
                import traceback
                traceback.print_exc()
                ok, msg = False, f"{mat.name}: {exc}"
            print("[Halcyon convert]", msg)
            if ok:
                done += 1
            elif 'already uses' in msg:
                skipped += 1
            else:
                failed += 1
        parts = [f"{done} converted"]
        if skipped:
            parts.append(f"{skipped} already converted")
        if failed:
            parts.append(f"{failed} failed")
        self.report({'INFO' if not failed else 'WARNING'},
                    "Halcyon: " + ", ".join(parts) + " (details in console)")
        return {'FINISHED'}


def convert_material_to_bi(mat, keep_original=True, force=False):
    """Rebuild `mat` around the BI Material node. Returns (ok, message).

    The same relink-not-reset contract as convert_material: whatever fed
    the source shader's Base Color arrives in the BI node's Color, the
    normal chain arrives in Normal, and the constants go through
    bi_plan's 2.79-shaped mapping.
    """
    if mat is None:
        return False, "no material"
    compat.enable_nodes(mat)
    tree = mat.node_tree
    if tree is None:
        return False, f"{mat.name}: no node tree"

    existing = [n for n in tree.nodes if n.bl_idname == BI_NODE]
    if existing and not force:
        return False, f"{mat.name}: already uses the BI Material node"

    out = _active_output(tree)
    if out is None:
        out = tree.nodes.new('ShaderNodeOutputMaterial')
        out.location = (400, 0)

    source, notes = _surface_source(out)
    if source is not None and source.bl_idname == BI_NODE and not force:
        return False, f"{mat.name}: already uses the BI Material node"
    if source is None:
        values, links = {}, set()
        src_id = 'ShaderNodeBsdfPrincipled'
        notes.append("no source shader found; started from defaults")
    else:
        values, links = _gather(source)
        src_id = source.bl_idname

    p = bi_plan(src_id, values, links)
    notes.extend(p['notes'])

    node = tree.nodes.new(BI_NODE)
    for k, v in p['props'].items():
        try:
            setattr(node, k, v)
        except (AttributeError, TypeError, ValueError):
            pass
    if source is not None:
        node.location = (source.location.x, source.location.y)
        node.label = f"from {source.bl_label}"
    else:
        node.location = (out.location.x - 260, out.location.y)

    carried = 0
    for name, value in p['sockets'].items():
        sock = node.inputs.get(name)
        if sock is None or not hasattr(sock, 'default_value'):
            continue
        try:
            if hasattr(sock.default_value, '__len__') and \
                    hasattr(value, '__len__'):
                n = min(len(sock.default_value), len(value))
                for i in range(n):
                    sock.default_value[i] = value[i]
            elif hasattr(sock.default_value, '__len__'):
                for i in range(3):
                    sock.default_value[i] = float(value)
            else:
                sock.default_value = float(value[0]) \
                    if hasattr(value, '__len__') else float(value)
            carried += 1
        except (TypeError, ValueError, IndexError):
            pass
    for bi_name, alias in p['links']:
        dst = node.inputs.get(bi_name)
        src_sock = source.inputs.get(alias) if source is not None else None
        if dst is None or src_sock is None or not src_sock.is_linked:
            continue
        try:
            tree.links.new(src_sock.links[0].from_socket, dst)
            carried += 1
        except Exception:                                       # noqa: BLE001
            pass

    try:
        tree.links.new(node.outputs['Surface'], out.inputs['Surface'])
    except Exception as exc:                                    # noqa: BLE001
        return False, f"{mat.name}: could not link output ({exc})"

    if source is not None:
        if keep_original:
            source.location = (source.location.x, source.location.y - 340)
            source.label = "replaced by BI Material"
            source.mute = True
        else:
            tree.nodes.remove(source)

    node.refresh_sockets()
    mat.halcyon.use_override = False
    msg = (f"{mat.name}: BI {p['props'].get('diff_shader', 'LAMBERT')}"
           f"/{p['props'].get('spec_shader', 'COOKTORR')}, "
           f"{carried} inputs carried")
    if notes:
        msg += " (" + "; ".join(notes[:2]) + ")"
    return True, msg


class HALCYON_OT_convert_to_bi(Operator):
    """Convert materials to the Blender Internal material node.

    The 2.79 look, chosen deliberately: the BI Material node shades with
    Blender Internal's own diffuse/specular pairs, so a scene aiming at
    the 2.79 render gets the exact model instead of a translation onto
    the master shader.
    """

    bl_idname = 'halcyon.convert_to_bi'
    bl_label = "Convert to Blender Internal"
    bl_description = ("Rebuild materials around the BI Material node -- "
                      "Blender Internal's diffuse and specular shaders, "
                      "with textures and normal chains relinked")
    bl_options = {'REGISTER', 'UNDO'}

    scope: EnumProperty(name="Scope", default='ACTIVE', items=(
        ('ACTIVE', "Active Material", "Only the active material slot"),
        ('SELECTED', "Selected Objects", "Every material on the selection"),
        ('SCENE', "Whole Scene", "Every material in the scene"),
    ))
    keep_original: BoolProperty(
        name="Keep Original Shader", default=True,
        description="Mute the old shader and move it aside instead of "
                    "deleting it, so the conversion can be inspected")
    force: BoolProperty(
        name="Reconvert", default=False,
        description="Convert again even if the material already uses the "
                    "BI Material node")

    def execute(self, context):
        mats = _materials_for(context, self.scope)
        if not mats:
            self.report({'WARNING'}, "No materials found for that scope")
            return {'CANCELLED'}
        done, skipped, failed = 0, 0, 0
        for mat in mats:
            try:
                ok, msg = convert_material_to_bi(mat, self.keep_original,
                                                 self.force)
            except Exception as exc:                            # noqa: BLE001
                import traceback
                traceback.print_exc()
                ok, msg = False, f"{mat.name}: {exc}"
            print("[Halcyon convert]", msg)
            if ok:
                done += 1
            elif 'already uses' in msg:
                skipped += 1
            else:
                failed += 1
        parts = [f"{done} converted"]
        if skipped:
            parts.append(f"{skipped} already converted")
        if failed:
            parts.append(f"{failed} failed")
        self.report({'INFO' if not failed else 'WARNING'},
                    "Halcyon: " + ", ".join(parts) + " (details in console)")
        return {'FINISHED'}


class HALCYON_OT_material_new(Operator):
    """New material, born as a Halcyon Shader.

    The panel's New button routes here instead of `material.new`, so a
    material made under this engine STARTS on the master shader instead
    of needing a convert step afterwards.
    """

    bl_idname = 'halcyon.material_new'
    bl_label = "New Halcyon Material"
    bl_description = ("Create a new material already built around the "
                      "Halcyon Shader, and put it in the active slot")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        mat = bpy.data.materials.new(name="Material")
        ob = context.object
        if ob is not None and hasattr(ob, 'material_slots'):
            if not len(ob.material_slots):
                try:
                    bpy.ops.object.material_slot_add()
                except Exception:                               # noqa: BLE001
                    pass
            try:
                ob.material_slots[ob.active_material_index].material = mat
            except Exception:                                   # noqa: BLE001
                pass
        ok, msg = convert_material(mat, keep_original=False, force=True)
        if not ok:
            self.report({'WARNING'}, msg)
        return {'FINISHED'}


def _pristine_default(mat):
    """True only for a factory-fresh material nobody has touched.

    The bar is deliberately strict -- exactly the two default nodes, the
    Principled sockets at their factory values, nothing linked, nothing
    animated -- because the handler below rewrites what this returns True
    for, and rewriting anything a person has edited is vandalism.
    """
    try:
        # compat.uses_nodes: the same read without the 6.0 deprecation
        # warning -- this line alone printed it 25 times per legacy
        # import in the field console
        if not compat.uses_nodes(mat) or mat.node_tree is None:
            return False
        if getattr(mat, 'library', None) is not None or \
                getattr(mat, 'override_library', None) is not None:
            return False
        if getattr(mat.node_tree, 'animation_data', None) is not None or \
                getattr(mat, 'animation_data', None) is not None:
            return False
        nodes = list(mat.node_tree.nodes)
        if len(nodes) != 2:
            return False
        kinds = sorted(n.bl_idname for n in nodes)
        if kinds != ['ShaderNodeBsdfPrincipled', 'ShaderNodeOutputMaterial']:
            return False
        p = next(n for n in nodes
                 if n.bl_idname == 'ShaderNodeBsdfPrincipled')
        for s in p.inputs:
            if s.is_linked:
                return False
        checks = (('Metallic', 0.0), ('Roughness', 0.5), ('Alpha', 1.0))
        for name, want in checks:
            s = p.inputs.get(name)
            if s is not None and abs(float(s.default_value) - want) > 1e-6:
                return False
        bc = p.inputs.get('Base Color')
        if bc is not None:
            v = tuple(bc.default_value)[:3]
            if any(abs(c - 0.8) > 1e-6 for c in v):
                return False
        return True
    except Exception:                                           # noqa: BLE001
        return False


def _halcyon_default_material(scene, depsgraph):
    """depsgraph_update_post: fresh default materials become Halcyon.

    Registered by name so the self-test's handler audit can tell this
    apart from a foreign add-on's watcher, and guarded three ways: only
    under this engine, only on materials `_pristine_default` vouches for,
    and self-terminating (a converted tree stops being pristine, so the
    update this conversion causes matches nothing).
    """
    try:
        if scene is None or scene.render.engine != 'HALCYON_RENDER':
            return
        for upd in depsgraph.updates:
            mid = getattr(upd, 'id', None)
            if mid is None or not isinstance(mid, bpy.types.Material):
                continue
            mat = bpy.data.materials.get(mid.name)
            if mat is not None and _pristine_default(mat):
                convert_material(mat, keep_original=False, force=True)
    except Exception:                                           # noqa: BLE001
        pass


class HALCYON_OT_bake_lightmap(Operator):
    """Bake the active object's lighting into a UV-space image."""

    bl_idname = 'halcyon.bake_lightmap'
    bl_label = "Bake Lightmap"
    bl_description = ("Bake this object's lighting (or ambient "
                      "occlusion) into its active UV layout through the "
                      "Halcyon CPU core -- same lamps, shadows and "
                      "materials as the frame, view pinned along the "
                      "normal, the way lightmaps were made")
    bl_options = {'REGISTER', 'UNDO'}

    mode: EnumProperty(name="Bake", default='COMBINED', items=(
        ('COMBINED', "Combined Lighting",
         "Full direct lighting with shadows, ambient and emission"),
        ('AO', "Ambient Occlusion",
         "The occlusion term alone, white where open sky reaches"),
    ))
    size: EnumProperty(name="Size", default='512', items=(
        ('128', "128", ""), ('256', "256", ""), ('512', "512", ""),
        ('1024', "1024", ""), ('2048', "2048", "")))
    margin: bpy.props.IntProperty(
        name="Margin", default=4, min=0, max=32,
        description="Texels of island-edge dilation, so bilinear "
                    "lookups never bleed the empty background")

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return ob is not None and getattr(ob, 'type', '') in (
            'MESH', 'CURVE', 'SURFACE', 'FONT', 'META')

    def execute(self, context):
        import numpy as np

        from . import export as EX
        from .core import render as core_render
        from .core.settings import RenderSettings

        ob = context.active_object
        hs = getattr(context.scene, 'halcyon', None)
        st = hs.to_settings() if hs is not None else RenderSettings()
        size = int(self.size)
        st.resolution_x = st.resolution_y = size
        depsgraph = context.evaluated_depsgraph_get()
        warnings = []
        sc = EX.export_scene(depsgraph, st, warnings)
        obj_index = None
        for i, info in enumerate(sc.objects or ()):
            if getattr(info, 'name', None) == ob.name:
                obj_index = i
                break
        if obj_index is None:
            self.report({'ERROR'},
                        f"'{ob.name}' exported no renderable faces")
            return {'CANCELLED'}
        img, why = core_render.bake_lightmap(
            sc, st, obj_index=obj_index, size=size, mode=self.mode,
            margin=int(self.margin))
        if img is None:
            self.report({'ERROR'}, f"Bake failed: {why}")
            return {'CANCELLED'}
        suffix = '_ao' if self.mode == 'AO' else '_lightmap'
        name = f'{ob.name}{suffix}'
        bimg = bpy.data.images.get(name)
        if bimg is None:
            bimg = bpy.data.images.new(name, size, size, alpha=True,
                                       float_buffer=True)
        elif tuple(bimg.size) != (size, size):
            bimg.scale(size, size)
        # engine images are bottom-row-first -- Blender's pixel order
        bimg.pixels.foreach_set(
            np.ascontiguousarray(img, np.float32).ravel())
        bimg.update()
        self.report({'INFO'},
                    f"Baked {self.mode.title()} for '{ob.name}' into "
                    f"image '{name}' ({size}x{size})")
        return {'FINISHED'}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)


class HALCYON_OT_palette_table(Operator):
    """R202: turn any image into a colour-palette table -- a new image
    with ONE PIXEL PER DISTINCT COLOUR, laid out as a grid. Feed the
    table to Render Properties > Colour > Palette Image (Custom mode)
    and the whole render is forced through those colours"""

    bl_idname = 'halcyon.palette_table'
    bl_label = "Make Palette Table"
    bl_options = {'REGISTER', 'UNDO'}

    max_colors: IntProperty(
        name="Max Colours", default=256, min=1, max=4096,
        description="Distinct colours beyond this are median-cut down "
                    "to fit; 256 is the classic VGA table")
    sort: EnumProperty(
        name="Sort", default='LUMA',
        description="The order the table's pixels are laid out in",
        items=[('LUMA', "Dark to Light",
                "Sorted by luminance -- the classic palette strip"),
               ('HUE', "Around the Wheel",
                "Sorted by hue, greys at the end"),
               ('FREQ', "Most Used First",
                "Sorted by how often each colour appears in the "
                "source image")])
    columns: IntProperty(
        name="Columns", default=0, min=0, max=4096,
        description="Table width in pixels; 0 picks the squarest "
                    "grid, and the colour count makes a strip")

    @classmethod
    def poll(cls, context):
        sp = getattr(context, 'space_data', None)
        return getattr(sp, 'image', None) is not None or \
            getattr(context, 'edit_image', None) is not None

    def execute(self, context):
        import numpy as np

        from .core.palette import palette_from_pixels, \
            palette_table_layout
        sp = getattr(context, 'space_data', None)
        src = getattr(sp, 'image', None) or \
            getattr(context, 'edit_image', None)
        if src is None:
            self.report({'ERROR'}, "No image open in this editor")
            return {'CANCELLED'}
        px = compat.image_pixels(src)
        if px is None:
            self.report({'ERROR'},
                        f"'{src.name}' has no readable pixels")
            return {'CANCELLED'}
        cols = palette_from_pixels(px, self.max_colors, self.sort)
        n = len(cols)
        cw, chh = palette_table_layout(n, self.columns)
        name = f"{src.name} Palette"
        img = bpy.data.images.new(name, width=cw, height=chh,
                                  alpha=True, float_buffer=False)
        buf = np.zeros((chh * cw, 4), np.float32)
        buf[:n, :3] = cols
        buf[:n, 3] = 1.0
        # top row first, the way a palette strip reads: flip rows into
        # Blender's bottom-first buffer
        grid = buf.reshape(chh, cw, 4)[::-1].reshape(-1)
        img.pixels.foreach_set(grid)
        img.update()
        try:
            if sp is not None and hasattr(sp, 'image'):
                sp.image = img
        except Exception:                                       # noqa: BLE001
            pass
        self.report({'INFO'},
                    f"'{name}': {n} colour{'s' if n != 1 else ''} as a "
                    f"{cw}x{chh} table; pick it as the Palette Image "
                    f"to force the render through it")
        return {'FINISHED'}


def _draw_palette_table_menu(self, context):
    self.layout.separator()
    self.layout.operator('halcyon.palette_table', icon='COLOR')


CLASSES = (HALCYON_OT_convert_materials, HALCYON_OT_convert_to_bi,
           HALCYON_OT_material_new, HALCYON_OT_bake_lightmap,
           HALCYON_OT_palette_table,
           # R253: Convert to Anime / Cartoon / Game
           HALCYON_OT_convert_to_node)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    menu = getattr(bpy.types, 'IMAGE_MT_image', None)
    if menu is not None and hasattr(menu, 'append'):
        menu.append(_draw_palette_table_menu)
    try:
        hs = bpy.app.handlers.depsgraph_update_post
        if _halcyon_default_material not in hs:
            hs.append(_halcyon_default_material)
    except Exception:                                           # noqa: BLE001
        pass


def unregister():
    menu = getattr(bpy.types, 'IMAGE_MT_image', None)
    if menu is not None and hasattr(menu, 'remove'):
        try:
            menu.remove(_draw_palette_table_menu)
        except Exception:                                       # noqa: BLE001
            pass
    try:
        hs = bpy.app.handlers.depsgraph_update_post
        while _halcyon_default_material in hs:
            hs.remove(_halcyon_default_material)
    except Exception:                                           # noqa: BLE001
        pass
    for c in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(c)
        except Exception:                                       # noqa: BLE001
            pass
