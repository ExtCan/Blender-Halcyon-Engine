"""Convert existing materials to the Halcyon Shader.

The point is that this is not a reset button: whatever was feeding the source
BSDF -- image textures, ramps, whole node networks -- is relinked onto the
equivalent input of the Halcyon Shader. A material with a texture in Base Color
comes out with that same texture in Diffuse Color.
"""

import bpy
from bpy.props import BoolProperty, EnumProperty, StringProperty
from bpy.types import Operator

from .core.convert import MASTER_NODE, plan

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


CLASSES = (HALCYON_OT_convert_materials,)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(c)
        except Exception:                                       # noqa: BLE001
            pass
