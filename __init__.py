"""Halcyon -- a from-scratch render engine for Blender 5.2 that targets the
output of mid-to-late 1990s home-computer 3D software.

Everything under halcyon/core/ and halcyon/shaders/ is free of any bpy import,
so the renderer, the shader compiler and the post chain can all be run and
tested without Blender at all:

    python3 -m halcyon.tests.run_all
"""

bl_info = {
    "name": "Halcyon Render Engine",
    "author": "Built by Claude with help from Mr. Emotiman",
    "version": (1, 25, 66),
    "blender": (5, 1, 0),
    "location": "Render Properties > Render Engine > Halcyon",
    "description": "Scanline/raytrace hybrid engine reproducing mid-to-late "
                   "1990s CGI, with classic shading models and coded GLSL/HLSL "
                   "shader nodes",
    "warning": "",
    "doc_url": "",
    "category": "Render",
}

_MODULES = ()


def _import_modules():
    from . import (compat, convert, engine, export, objects, properties,
                   selftest, templates, ui)
    from .nodes import shader_nodes
    return (properties, shader_nodes, convert, templates, objects, engine,
            selftest, ui)


def register():
    global _MODULES
    _MODULES = _import_modules()
    for mod in _MODULES:
        if hasattr(mod, 'register'):
            mod.register()


def unregister():
    for mod in reversed(_MODULES):
        if hasattr(mod, 'unregister'):
            try:
                mod.unregister()
            except Exception:                                   # noqa: BLE001
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    register()
