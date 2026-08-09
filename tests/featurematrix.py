"""The feature x device matrix: one row per feature, shared by two provers.

The headless suite renders every row on BOTH devices and demands the
pictures agree EXACTLY -- with no driver present, the GPU device must fall
back onto the very same CPU code, so any difference is a hole in the
switch or the fallback plumbing. The self-test renders the same rows on a
REAL driver and prints each row's CPU-vs-GPU difference with which stages
actually engaged, which is the field's answer to "does every feature work
with the GPU" -- either the driver reproduces the CPU picture within the
deferred bar, or the feature routes to the CPU by name and the picture is
exact. Both outcomes are the switch working; wrong pixels are the only
failure.

Rows are (key, label, settings overrides, scene builder name). Builders
are bpy-free. Resolution stays small: the matrix is about coverage, and
sixty small frames beat six big ones.
"""

import numpy as np

from ..core.scene import Light
from .scenebuild import demo_scene


def _cookie_pixels():
    ck = np.zeros((8, 8, 4), np.float32)
    ck[:, :, 3] = 1.0
    ck[::2, ::2, :3] = 1.0
    ck[1::2, 1::2, :3] = 1.0
    ck[:, :, 1] *= 0.3
    ck[0, :, 2] = 1.0
    return ck


def _sc_demo(st):
    return demo_scene(st, with_texture=False)


def _sc_textured(st):
    return demo_scene(st, with_texture=True)


def _sc_mirror(st):
    sc = demo_scene(st, with_texture=False)
    sc.materials[2].reflect_level = 0.5
    return sc


def _sc_matcap_image(st):
    """The documented matcap workflow -- an image through Matcap
    Coordinates into the Matcap socket. The field's 'Eyes' material:
    the shape that refused for a whole 640x640 frame."""
    from ..core.scene import ImageBuffer, Material
    from .scenebuild import checker_image
    sc = demo_scene(st, with_texture=False)
    sc.images['eyes'] = ImageBuffer(name='eyes', pixels=checker_image())
    ins = [
        _sk('Diffuse Color', 'RGBA', [0.6, 0.6, 0.6, 1.0]),
        _sk('Vertex Color', 'RGBA', [1, 1, 1, 1]),
        _sk('Vertex Color Mix', 'VALUE', 0.0),
        _sk('Diffuse Level', 'VALUE', 1.0),
        _sk('Specular Color', 'RGBA', [1, 1, 1, 1]),
        _sk('Specular Level', 'VALUE', 0.4),
        _sk('Glossiness', 'VALUE', 24.0),
        _sk('Roughness', 'VALUE', 0.3),
        _sk('Ambient', 'VALUE', 1.0),
        _sk('Self-Illumination', 'RGBA', [0, 0, 0, 1]),
        _sk('Opacity', 'VALUE', 1.0),
        _sk('IOR', 'VALUE', 1.45),
        _sk('Anisotropy', 'VALUE', 0.0),
        _sk('Anisotropic Rotation', 'VALUE', 0.0),
        _sk('Metalness', 'VALUE', 0.0),
        _sk('Soften', 'VALUE', 0.0),
        _sk('Reflection', 'VALUE', 0.0),
        _sk('Translucency', 'VALUE', 0.0),
        _sk('Toon Size', 'VALUE', 0.5),
        _sk('Toon Smooth', 'VALUE', 0.05),
        _sk('Matcap', 'RGBA', [0, 0, 0, 1], ['tex', 0]),
        _sk('Matcap Blend', 'VALUE', 0.55),
    ]
    sc.materials[1] = Material(name='Eyes', index=1, graph={
        'output': 'out', 'nodes': {
            'muv': {'id': 'muv', 'bl_idname': 'HALCYON_MatcapUVNode',
                    'props': {},
                    'inputs': [_sk('Scale', 'VALUE', 1.0)],
                    'outputs': [{'name': 'Vector', 'type': 'VECTOR'},
                                {'name': 'Facing', 'type': 'VALUE'}]},
            'tex': {'id': 'tex', 'bl_idname': 'ShaderNodeTexImage',
                    'props': {'image': 'eyes',
                              'interpolation': 'Closest'},
                    'inputs': [_sk('Vector', 'VECTOR', [0, 0, 0],
                                   ['muv', 0])],
                    'outputs': [{'name': 'Color', 'type': 'RGBA'},
                                {'name': 'Alpha', 'type': 'VALUE'}]},
            'hal': {'id': 'hal', 'bl_idname': 'HALCYON_ShaderNode',
                    'props': {'model': 'PHONG', 'toon_steps': 2},
                    'inputs': ins,
                    'outputs': [{'name': 'Surface', 'type': 'SHADER'}]},
            'out': {'id': 'out',
                    'bl_idname': 'ShaderNodeOutputMaterial', 'props': {},
                    'inputs': [_sk('Surface', 'SHADER', None,
                                   ['hal', 0]),
                               _sk('Displacement', 'VECTOR', [0, 0, 0])],
                    'outputs': []}}})
    return sc


def _sc_glass(st):
    sc = demo_scene(st, with_texture=False)
    sc.materials[1].opacity = 0.5
    sc.materials[2].opacity = 0.6
    sc.materials[2].reflect_level = 0.5
    return sc


def _sc_cookie_spot(st):
    sc = demo_scene(st, with_texture=False)
    spot = Light(type='SPOT', name='proj', position=(0.0, -4.0, 6.0),
                 direction=(0.0, 0.45, -0.9), color=(1.0, 1.0, 1.0),
                 energy=800.0, spot_size=1.0, spot_blend=0.2,
                 shadow='NONE', decay='INVERSE_SQUARE')
    spot.cookie = _cookie_pixels()
    sc.lights = list(sc.lights) + [spot]
    return sc


def _sc_cookie_sun(st):
    sc = demo_scene(st, with_texture=False)
    sun = Light(type='SUN', name='clouds', direction=(-0.5, 0.4, -0.75),
                color=(1.0, 0.97, 0.9), energy=3.0, shadow='NONE')
    sun.cookie = _cookie_pixels()
    sun.cookie_scale = 3.0
    sc.lights = list(sc.lights) + [sun]
    return sc


def _sc_area(st):
    sc = demo_scene(st, with_texture=False)
    sc.lights = list(sc.lights) + [
        Light(type='AREA', name='panel', position=(2.0, -2.0, 5.0),
              color=(0.9, 0.9, 1.0), energy=500.0, area_size=(2.0, 1.0),
              area_shape='RECTANGLE', shadow='NONE')]
    return sc


def _sc_negative(st):
    sc = demo_scene(st, with_texture=False)
    neg = Light(type='POINT', name='anti', position=(1.5, -1.5, 3.0),
                color=(1.0, 1.0, 1.0), energy=300.0, shadow='NONE')
    neg.negative = True
    sc.lights = list(sc.lights) + [neg]
    return sc


def _sc_soft(st):
    sc = demo_scene(st, with_texture=False)
    for l in sc.lights:
        if l.type == 'POINT':
            l.radius = 0.4
    return sc


def _sc_bryce(st):
    sc = demo_scene(st, with_texture=False)
    try:
        from ..presets.skies import apply_sky
        if getattr(sc, 'world', None) is not None:
            apply_sky(sc.world, 'BRYCE_DEFAULT')
    except Exception:                                           # noqa: BLE001
        pass
    return sc


def _sc_ortho(st):
    sc = demo_scene(st, with_texture=False)
    sc.camera.type = 'ORTHO'
    return sc


def _sk(name, tp, default, link=None):
    return {'name': name, 'type': tp, 'default': default, 'link': link}


def _one_bsdf_graph(idname, inputs, props=None):
    """A raw single-BSDF graph, exactly as the exporter would carry it."""
    return {'output': 'out', 'nodes': {
        'bsdf': {'id': 'bsdf', 'bl_idname': idname, 'props': props or {},
                 'inputs': inputs,
                 'outputs': [{'name': 'BSDF', 'type': 'SHADER'}]},
        'out': {'id': 'out', 'bl_idname': 'ShaderNodeOutputMaterial',
                'props': {},
                'inputs': [_sk('Surface', 'SHADER', None, ['bsdf', 0]),
                           _sk('Displacement', 'VECTOR', [0, 0, 0])],
                'outputs': []}}}


def _sc_metallic_node(st):
    sc = demo_scene(st, with_texture=False)
    sc.materials[1].graph = _one_bsdf_graph(
        'ShaderNodeBsdfMetallic',
        [_sk('Base Color', 'RGBA', [0.9, 0.6, 0.2, 1.0]),
         _sk('Roughness', 'VALUE', 0.35)])
    return sc


def _sc_specular_node(st):
    sc = demo_scene(st, with_texture=False)
    sc.materials[1].graph = _one_bsdf_graph(
        'ShaderNodeEeveeSpecular',
        [_sk('Base Color', 'RGBA', [0.2, 0.5, 0.8, 1.0]),
         _sk('Specular', 'RGBA', [1.0, 0.9, 0.7, 1.0]),
         _sk('Roughness', 'VALUE', 0.25),
         _sk('Emissive Color', 'RGBA', [0.0, 0.0, 0.0, 1.0]),
         _sk('Transparency', 'VALUE', 0.0)])
    return sc


def _sc_noise_node(st):
    sc = demo_scene(st, with_texture=False)
    sc.materials[1].graph = {'output': 'out', 'nodes': {
        'noise': {'id': 'noise', 'bl_idname': 'HALCYON_NoiseNode',
                  'props': {'kind': 'RIDGED', 'octaves': 5},
                  'inputs': [_sk('Vector', 'VECTOR', [0, 0, 0]),
                             _sk('Scale', 'VALUE', 4.0),
                             _sk('Lacunarity', 'VALUE', 2.0),
                             _sk('Gain', 'VALUE', 0.5),
                             _sk('Color 1', 'RGBA', [0.1, 0.05, 0.3, 1.0]),
                             _sk('Color 2', 'RGBA', [1.0, 0.9, 0.6, 1.0])],
                  'outputs': [{'name': 'Color', 'type': 'RGBA'},
                              {'name': 'Fac', 'type': 'VALUE'}]},
        'bsdf': {'id': 'bsdf', 'bl_idname': 'ShaderNodeBsdfDiffuse',
                 'props': {},
                 'inputs': [_sk('Color', 'RGBA', [1, 1, 1, 1],
                                ['noise', 0]),
                            _sk('Roughness', 'VALUE', 0.0)],
                 'outputs': [{'name': 'BSDF', 'type': 'SHADER'}]},
        'out': {'id': 'out', 'bl_idname': 'ShaderNodeOutputMaterial',
                'props': {},
                'inputs': [_sk('Surface', 'SHADER', None, ['bsdf', 0]),
                           _sk('Displacement', 'VECTOR', [0, 0, 0])],
                'outputs': []}}}
    return sc


def _sc_cells_palette_node(st):
    sc = demo_scene(st, with_texture=False)
    sc.materials[1].graph = {'output': 'out', 'nodes': {
        'cells': {'id': 'cells', 'bl_idname': 'HALCYON_CellsNode',
                  'props': {'feature': 'CELL'},
                  'inputs': [_sk('Vector', 'VECTOR', [0, 0, 0]),
                             _sk('Scale', 'VALUE', 5.0),
                             _sk('Randomness', 'VALUE', 1.0),
                             _sk('Color 1', 'RGBA', [0.9, 0.2, 0.1, 1.0]),
                             _sk('Color 2', 'RGBA', [0.1, 0.5, 0.9, 1.0])],
                  'outputs': [{'name': 'Color', 'type': 'RGBA'},
                              {'name': 'Fac', 'type': 'VALUE'},
                              {'name': 'Cell ID', 'type': 'VALUE'}]},
        'pal': {'id': 'pal', 'bl_idname': 'HALCYON_PaletteNode',
                'props': {'palette': 'EGA'},
                'inputs': [_sk('Color', 'RGBA', [0.8, 0.8, 0.8, 1.0],
                               ['cells', 0]),
                           _sk('Mix', 'VALUE', 1.0)],
                'outputs': [{'name': 'Color', 'type': 'RGBA'},
                            {'name': 'Index', 'type': 'VALUE'}]},
        'bsdf': {'id': 'bsdf', 'bl_idname': 'ShaderNodeBsdfDiffuse',
                 'props': {},
                 'inputs': [_sk('Color', 'RGBA', [1, 1, 1, 1], ['pal', 0]),
                            _sk('Roughness', 'VALUE', 0.0)],
                 'outputs': [{'name': 'BSDF', 'type': 'SHADER'}]},
        'out': {'id': 'out', 'bl_idname': 'ShaderNodeOutputMaterial',
                'props': {},
                'inputs': [_sk('Surface', 'SHADER', None, ['bsdf', 0]),
                           _sk('Displacement', 'VECTOR', [0, 0, 0])],
                'outputs': []}}}
    return sc


def _sc_retro_chain_node(st):
    """Pixelate -> UV Scroll -> Marble, then Scanlines over the colour:
    four of the utility nodes in one chain, the way a user wires them."""
    sc = demo_scene(st, with_texture=False)
    sc.materials[1].graph = {'output': 'out', 'nodes': {
        'pix': {'id': 'pix', 'bl_idname': 'HALCYON_PixelateNode',
                'props': {},
                'inputs': [_sk('Vector', 'VECTOR', [0, 0, 0]),
                           _sk('Pixels X', 'VALUE', 24.0),
                           _sk('Pixels Y', 'VALUE', 24.0),
                           _sk('Pixels Z', 'VALUE', 0.0)],
                'outputs': [{'name': 'Vector', 'type': 'VECTOR'}]},
        'scr': {'id': 'scr', 'bl_idname': 'HALCYON_ScrollNode',
                'props': {'animate': True, 'fps': 15},
                'inputs': [_sk('Vector', 'VECTOR', [0, 0, 0], ['pix', 0]),
                           _sk('Scroll X', 'VALUE', 0.2),
                           _sk('Scroll Y', 'VALUE', 0.05),
                           _sk('Spin', 'VALUE', 0.1)],
                'outputs': [{'name': 'Vector', 'type': 'VECTOR'}]},
        'marble': {'id': 'marble', 'bl_idname': 'HALCYON_MarbleNode',
                   'props': {'octaves': 4, 'axis': 'X'},
                   'inputs': [_sk('Vector', 'VECTOR', [0, 0, 0],
                                  ['scr', 0]),
                              _sk('Scale', 'VALUE', 3.0),
                              _sk('Turbulence', 'VALUE', 1.0),
                              _sk('Veins', 'VALUE', 1.0),
                              _sk('Sharpness', 'VALUE', 1.0),
                              _sk('Color 1', 'RGBA', [0.9, 0.9, 0.85, 1]),
                              _sk('Color 2', 'RGBA', [0.2, 0.15, 0.3, 1])],
                   'outputs': [{'name': 'Color', 'type': 'RGBA'},
                               {'name': 'Fac', 'type': 'VALUE'}]},
        'scan': {'id': 'scan', 'bl_idname': 'HALCYON_ScanlinesNode',
                 'props': {'animate': False},
                 'inputs': [_sk('Color', 'RGBA', [0.8, 0.8, 0.8, 1.0],
                                ['marble', 0]),
                            _sk('Vector', 'VECTOR', [0, 0, 0]),
                            _sk('Lines', 'VALUE', 48.0),
                            _sk('Darkness', 'VALUE', 0.5),
                            _sk('Thickness', 'VALUE', 0.5)],
                 'outputs': [{'name': 'Color', 'type': 'RGBA'}]},
        'bsdf': {'id': 'bsdf', 'bl_idname': 'ShaderNodeBsdfDiffuse',
                 'props': {},
                 'inputs': [_sk('Color', 'RGBA', [1, 1, 1, 1],
                                ['scan', 0]),
                            _sk('Roughness', 'VALUE', 0.0)],
                 'outputs': [{'name': 'BSDF', 'type': 'SHADER'}]},
        'out': {'id': 'out', 'bl_idname': 'ShaderNodeOutputMaterial',
                'props': {},
                'inputs': [_sk('Surface', 'SHADER', None, ['bsdf', 0]),
                           _sk('Displacement', 'VECTOR', [0, 0, 0])],
                'outputs': []}}}
    return sc


def _sc_static_dither_node(st):
    """TV Static through the Ordered Dither node -- an in-scene dead
    channel, quantised at shading time."""
    sc = demo_scene(st, with_texture=False)
    sc.materials[1].graph = {'output': 'out', 'nodes': {
        'tv': {'id': 'tv', 'bl_idname': 'HALCYON_StaticNode',
               'props': {'animate': True},
               'inputs': [_sk('Vector', 'VECTOR', [0, 0, 0]),
                          _sk('Scale', 'VALUE', 48.0),
                          _sk('Color 1', 'RGBA', [0.02, 0.02, 0.02, 1.0]),
                          _sk('Color 2', 'RGBA', [0.9, 0.9, 0.9, 1.0])],
               'outputs': [{'name': 'Color', 'type': 'RGBA'},
                           {'name': 'Fac', 'type': 'VALUE'}]},
        'dith': {'id': 'dith', 'bl_idname': 'HALCYON_DitherNode',
                 'props': {'pattern': 'BAYER8'},
                 'inputs': [_sk('Color', 'RGBA', [0.8, 0.8, 0.8, 1.0],
                                ['tv', 0]),
                            _sk('Levels', 'VALUE', 4.0),
                            _sk('Strength', 'VALUE', 1.0)],
                 'outputs': [{'name': 'Color', 'type': 'RGBA'}]},
        'bsdf': {'id': 'bsdf', 'bl_idname': 'ShaderNodeBsdfDiffuse',
                 'props': {},
                 'inputs': [_sk('Color', 'RGBA', [1, 1, 1, 1],
                                ['dith', 0]),
                            _sk('Roughness', 'VALUE', 0.0)],
                 'outputs': [{'name': 'BSDF', 'type': 'SHADER'}]},
        'out': {'id': 'out', 'bl_idname': 'ShaderNodeOutputMaterial',
                'props': {},
                'inputs': [_sk('Surface', 'SHADER', None, ['bsdf', 0]),
                           _sk('Displacement', 'VECTOR', [0, 0, 0])],
                'outputs': []}}}
    return sc


def _sc_flipbook_wave_node(st):
    """UV Wave -> Flipbook -> Marble: animated-coordinate utilities."""
    sc = demo_scene(st, with_texture=False)
    sc.materials[1].graph = {'output': 'out', 'nodes': {
        'wave': {'id': 'wave', 'bl_idname': 'HALCYON_UVWaveNode',
                 'props': {'animate': True},
                 'inputs': [_sk('Vector', 'VECTOR', [0, 0, 0]),
                            _sk('Amplitude X', 'VALUE', 0.03),
                            _sk('Amplitude Y', 'VALUE', 0.02),
                            _sk('Frequency', 'VALUE', 6.0),
                            _sk('Speed', 'VALUE', 1.0)],
                 'outputs': [{'name': 'Vector', 'type': 'VECTOR'}]},
        'flip': {'id': 'flip', 'bl_idname': 'HALCYON_FlipbookNode',
                 'props': {'animate': True},
                 'inputs': [_sk('Vector', 'VECTOR', [0, 0, 0],
                                ['wave', 0]),
                            _sk('Columns', 'VALUE', 4.0),
                            _sk('Rows', 'VALUE', 4.0),
                            _sk('Rate', 'VALUE', 8.0),
                            _sk('Cell Offset', 'VALUE', 2.0)],
                 'outputs': [{'name': 'Vector', 'type': 'VECTOR'}]},
        'marble': {'id': 'marble', 'bl_idname': 'HALCYON_MarbleNode',
                   'props': {'octaves': 4, 'axis': 'X'},
                   'inputs': [_sk('Vector', 'VECTOR', [0, 0, 0],
                                  ['flip', 0]),
                              _sk('Scale', 'VALUE', 4.0),
                              _sk('Turbulence', 'VALUE', 1.0),
                              _sk('Veins', 'VALUE', 1.0),
                              _sk('Sharpness', 'VALUE', 1.0),
                              _sk('Color 1', 'RGBA', [0.9, 0.85, 0.7, 1]),
                              _sk('Color 2', 'RGBA', [0.25, 0.1, 0.1, 1])],
                   'outputs': [{'name': 'Color', 'type': 'RGBA'},
                               {'name': 'Fac', 'type': 'VALUE'}]},
        'bsdf': {'id': 'bsdf', 'bl_idname': 'ShaderNodeBsdfDiffuse',
                 'props': {},
                 'inputs': [_sk('Color', 'RGBA', [1, 1, 1, 1],
                                ['marble', 0]),
                            _sk('Roughness', 'VALUE', 0.0)],
                 'outputs': [{'name': 'BSDF', 'type': 'SHADER'}]},
        'out': {'id': 'out', 'bl_idname': 'ShaderNodeOutputMaterial',
                'props': {},
                'inputs': [_sk('Surface', 'SHADER', None, ['bsdf', 0]),
                           _sk('Displacement', 'VECTOR', [0, 0, 0])],
                'outputs': []}}}
    return sc


def _sc_halftone_chain_node(st):
    """Cells -> Quantize -> Threshold driving a mix over a Halftone of
    the same cells: the value utilities in one graph."""
    sc = demo_scene(st, with_texture=False)
    cells_outs = [{'name': 'Color', 'type': 'RGBA'},
                  {'name': 'Fac', 'type': 'VALUE'},
                  {'name': 'Cell ID', 'type': 'VALUE'}]
    sc.materials[1].graph = {'output': 'out', 'nodes': {
        'cells': {'id': 'cells', 'bl_idname': 'HALCYON_CellsNode',
                  'props': {'feature': 'F1'},
                  'inputs': [_sk('Vector', 'VECTOR', [0, 0, 0]),
                             _sk('Scale', 'VALUE', 4.0),
                             _sk('Randomness', 'VALUE', 1.0),
                             _sk('Color 1', 'RGBA', [0.1, 0.1, 0.1, 1.0]),
                             _sk('Color 2', 'RGBA', [0.9, 0.85, 0.8, 1.0])],
                  'outputs': cells_outs},
        'quant': {'id': 'quant', 'bl_idname': 'HALCYON_QuantizeNode',
                  'props': {},
                  'inputs': [_sk('Fac', 'VALUE', 0.5, ['cells', 1]),
                             _sk('Steps', 'VALUE', 4.0)],
                  'outputs': [{'name': 'Fac', 'type': 'VALUE'}]},
        'thresh': {'id': 'thresh', 'bl_idname': 'HALCYON_ThresholdNode',
                   'props': {},
                   'inputs': [_sk('Fac', 'VALUE', 0.5, ['quant', 0]),
                              _sk('Level', 'VALUE', 0.45),
                              _sk('Smooth', 'VALUE', 0.1)],
                   'outputs': [{'name': 'Fac', 'type': 'VALUE'}]},
        'half': {'id': 'half', 'bl_idname': 'HALCYON_HalftoneNode',
                 'props': {},
                 'inputs': [_sk('Color', 'RGBA', [0.5, 0.5, 0.5, 1.0],
                                ['cells', 0]),
                            _sk('Vector', 'VECTOR', [0, 0, 0]),
                            _sk('Dots', 'VALUE', 20.0),
                            _sk('Angle', 'VALUE', 45.0),
                            _sk('Ink Color', 'RGBA',
                                [0.05, 0.05, 0.05, 1.0]),
                            _sk('Paper Color', 'RGBA',
                                [0.95, 0.93, 0.88, 1.0])],
                 'outputs': [{'name': 'Color', 'type': 'RGBA'},
                             {'name': 'Fac', 'type': 'VALUE'}]},
        'mix': {'id': 'mix', 'bl_idname': 'ShaderNodeMixRGB',
                'props': {'blend_type': 'MIX'},
                'inputs': [_sk('Fac', 'VALUE', 0.0, ['thresh', 0]),
                           _sk('Color1', 'RGBA', [0.2, 0.3, 0.7, 1.0],
                               ['half', 0]),
                           _sk('Color2', 'RGBA', [0.9, 0.6, 0.2, 1.0])],
                'outputs': [{'name': 'Color', 'type': 'RGBA'}]},
        'bsdf': {'id': 'bsdf', 'bl_idname': 'ShaderNodeBsdfDiffuse',
                 'props': {},
                 'inputs': [_sk('Color', 'RGBA', [1, 1, 1, 1],
                                ['mix', 0]),
                            _sk('Roughness', 'VALUE', 0.0)],
                 'outputs': [{'name': 'BSDF', 'type': 'SHADER'}]},
        'out': {'id': 'out', 'bl_idname': 'ShaderNodeOutputMaterial',
                'props': {},
                'inputs': [_sk('Surface', 'SHADER', None, ['bsdf', 0]),
                           _sk('Displacement', 'VECTOR', [0, 0, 0])],
                'outputs': []}}}
    return sc


GEOMETRY_OUTS = [{'name': 'Position', 'type': 'VECTOR'},
                 {'name': 'Normal', 'type': 'VECTOR'},
                 {'name': 'Tangent', 'type': 'VECTOR'},
                 {'name': 'True Normal', 'type': 'VECTOR'},
                 {'name': 'Incoming', 'type': 'VECTOR'},
                 {'name': 'Parametric', 'type': 'VECTOR'},
                 {'name': 'Backfacing', 'type': 'VALUE'},
                 {'name': 'Pointiness', 'type': 'VALUE'},
                 {'name': 'Random Per Island', 'type': 'VALUE'}]


def _sc_geometry_node(st):
    """Geometry's Tangent x Incoming, absolute value, as the colour --
    two of the outputs whose first GPU mapping was silently wrong."""
    sc = demo_scene(st, with_texture=False)
    sc.materials[1].graph = {'output': 'out', 'nodes': {
        'geo': {'id': 'geo', 'bl_idname': 'ShaderNodeNewGeometry',
                'props': {}, 'inputs': [],
                'outputs': [dict(o) for o in GEOMETRY_OUTS]},
        'mul': {'id': 'mul', 'bl_idname': 'ShaderNodeVectorMath',
                'props': {'operation': 'MULTIPLY'},
                # both operand sockets are display-named 'Vector', exactly
                # as Blender exports them -- the emitter resolves the
                # second by counting same-named sockets
                'inputs': [_sk('Vector', 'VECTOR', [0, 0, 0], ['geo', 2]),
                           _sk('Vector', 'VECTOR', [0, 0, 0],
                               ['geo', 4])],
                'outputs': [{'name': 'Vector', 'type': 'VECTOR'}]},
        'absn': {'id': 'absn', 'bl_idname': 'ShaderNodeVectorMath',
                 'props': {'operation': 'ABSOLUTE'},
                 'inputs': [_sk('Vector', 'VECTOR', [0, 0, 0],
                                ['mul', 0])],
                 'outputs': [{'name': 'Vector', 'type': 'VECTOR'}]},
        'bsdf': {'id': 'bsdf', 'bl_idname': 'ShaderNodeBsdfDiffuse',
                 'props': {},
                 'inputs': [_sk('Color', 'RGBA', [1, 1, 1, 1],
                                ['absn', 0]),
                            _sk('Roughness', 'VALUE', 0.0)],
                 'outputs': [{'name': 'BSDF', 'type': 'SHADER'}]},
        'out': {'id': 'out', 'bl_idname': 'ShaderNodeOutputMaterial',
                'props': {},
                'inputs': [_sk('Surface', 'SHADER', None, ['bsdf', 0]),
                           _sk('Displacement', 'VECTOR', [0, 0, 0])],
                'outputs': []}}}
    return sc


def _sc_wireframe_node(st):
    sc = demo_scene(st, with_texture=False)
    sc.materials[1].graph = {'output': 'out', 'nodes': {
        'wire': {'id': 'wire', 'bl_idname': 'ShaderNodeWireframe',
                 'props': {'use_pixel_size': False},
                 'inputs': [_sk('Size', 'VALUE', 0.08)],
                 'outputs': [{'name': 'Fac', 'type': 'VALUE'}]},
        'mix': {'id': 'mix', 'bl_idname': 'ShaderNodeMixRGB',
                'props': {'blend_type': 'MIX'},
                'inputs': [_sk('Fac', 'VALUE', 0.0, ['wire', 0]),
                           _sk('Color1', 'RGBA', [0.15, 0.2, 0.6, 1.0]),
                           _sk('Color2', 'RGBA', [1.0, 1.0, 1.0, 1.0])],
                'outputs': [{'name': 'Color', 'type': 'RGBA'}]},
        'bsdf': {'id': 'bsdf', 'bl_idname': 'ShaderNodeBsdfDiffuse',
                 'props': {},
                 'inputs': [_sk('Color', 'RGBA', [1, 1, 1, 1], ['mix', 0]),
                            _sk('Roughness', 'VALUE', 0.0)],
                 'outputs': [{'name': 'BSDF', 'type': 'SHADER'}]},
        'out': {'id': 'out', 'bl_idname': 'ShaderNodeOutputMaterial',
                'props': {},
                'inputs': [_sk('Surface', 'SHADER', None, ['bsdf', 0]),
                           _sk('Displacement', 'VECTOR', [0, 0, 0])],
                'outputs': []}}}
    return sc


SCENES = {
    'demo': _sc_demo, 'textured': _sc_textured, 'mirror': _sc_mirror,
    'matcap_image': _sc_matcap_image,
    'glass': _sc_glass, 'cookie_spot': _sc_cookie_spot,
    'cookie_sun': _sc_cookie_sun, 'area': _sc_area,
    'negative': _sc_negative, 'soft': _sc_soft, 'bryce': _sc_bryce,
    'metallic_node': _sc_metallic_node, 'specular_node': _sc_specular_node,
    'wireframe_node': _sc_wireframe_node,
    'noise_node': _sc_noise_node,
    'cells_palette_node': _sc_cells_palette_node,
    'retro_chain_node': _sc_retro_chain_node,
    'static_dither_node': _sc_static_dither_node,
    'flipbook_wave_node': _sc_flipbook_wave_node,
    'halftone_chain_node': _sc_halftone_chain_node,
    'geometry_node': _sc_geometry_node,
    'ortho': _sc_ortho,
}

#: (key, settings overrides, scene name). Labels are the keys.
ROWS = [
    ('baseline PHONG + map shadows', {}, 'demo'),
    # ------------------------------------------------ the 18 models, forced
    ('model LAMBERT', {'force_model': 'LAMBERT'}, 'demo'),
    ('model GOURAUD', {'force_model': 'GOURAUD'}, 'demo'),
    ('model FLAT', {'force_model': 'FLAT'}, 'demo'),
    ('model PHONG', {'force_model': 'PHONG'}, 'demo'),
    ('model BLINN_PHONG', {'force_model': 'BLINN_PHONG'}, 'demo'),
    ('model BLINN', {'force_model': 'BLINN'}, 'demo'),
    ('model COOK_TORRANCE', {'force_model': 'COOK_TORRANCE'}, 'demo'),
    ('model OREN_NAYAR', {'force_model': 'OREN_NAYAR'}, 'demo'),
    ('model MINNAERT', {'force_model': 'MINNAERT'}, 'demo'),
    ('model WARD', {'force_model': 'WARD'}, 'demo'),
    ('model ANISOTROPIC', {'force_model': 'ANISOTROPIC'}, 'demo'),
    ('model METAL', {'force_model': 'METAL'}, 'demo'),
    ('model STRAUSS', {'force_model': 'STRAUSS'}, 'demo'),
    ('model MULTI_LAYER', {'force_model': 'MULTI_LAYER'}, 'demo'),
    ('model TOON', {'force_model': 'TOON'}, 'demo'),
    ('model TRANSLUCENT', {'force_model': 'TRANSLUCENT'}, 'demo'),
    ('model CONSTANT', {'force_model': 'CONSTANT'}, 'demo'),
    ('model WIREFRAME', {'force_model': 'WIREFRAME'}, 'demo'),
    # ------------------------------------------------------- shading rates
    ('scene rate VERTEX (Gouraud)', {'shading_rate': 'VERTEX'}, 'demo'),
    ('scene rate FACE (flat)', {'shading_rate': 'FACE'}, 'demo'),
    ('normal source SPLIT', {'normal_source': 'SPLIT'}, 'demo'),
    ('normal source FACE', {'normal_source': 'FACE'}, 'demo'),
    ('one-sided lighting', {'two_sided_lighting': False}, 'demo'),
    ('specular in linear', {'specular_in_gamma': False}, 'demo'),
    ('unclamped specular', {'clamp_specular': False}, 'demo'),
    # ------------------------------------------------------------- lights
    ('light limit 2 brightest', {'max_lights': 2}, 'demo'),
    ('light limit nearest', {'max_lights': 2,
                             'light_limit_mode': 'NEAREST'}, 'demo'),
    ('inverse falloff', {'light_falloff_default': 'INVERSE'}, 'demo'),
    ('light clamp', {'light_clamp': 0.4}, 'demo'),
    ('area light', {}, 'area'),
    ('negative light', {}, 'negative'),
    ('spot gobo (projected texture)', {}, 'cookie_spot'),
    ('sun cloud cookie', {}, 'cookie_sun'),
    # ------------------------------------------------------------ shadows
    ('shadows off', {'shadows': False}, 'demo'),
    ('ray shadows', {'shadow_default': 'RAY'}, 'demo'),
    ('per-light shadow modes', {'shadow_default': 'PER_LIGHT'}, 'demo'),
    ('soft shadows (radius + 8 taps)', {'shadow_samples': 8}, 'soft'),
    ('soft RAY shadows', {'shadow_default': 'RAY',
                          'shadow_samples': 4}, 'soft'),
    ('ambient occlusion', {'ambient_occlusion': True,
                           'ao_samples': 4}, 'demo'),
    # -------------------------------------------------------- ray tracing
    ('traced reflection', {'raytrace': True, 'ray_depth': 1}, 'mirror'),
    ('traced recursion depth 2', {'raytrace': True, 'ray_depth': 2},
     'mirror'),
    ('env reflection (no rays)', {'env_reflection': True}, 'mirror'),
    ('rich world (Bryce) reflection', {'raytrace': True, 'ray_depth': 1},
     'bryce'),
    # ------------------------------------------------- raw node graphs
    ('node Metallic BSDF (raw graph)', {}, 'metallic_node'),
    ('node Specular BSDF (raw graph)', {}, 'specular_node'),
    ('node Fractal Noise (ridged)', {}, 'noise_node'),
    ('node Cells + Hardware Palette', {}, 'cells_palette_node'),
    ('node Pixelate + Scroll + Scanlines', {}, 'retro_chain_node'),
    ('node TV Static + Ordered Dither', {}, 'static_dither_node'),
    ('node Flipbook + UV Wave', {}, 'flipbook_wave_node'),
    ('node Halftone + Threshold + Quantize', {}, 'halftone_chain_node'),
    ('node Geometry (Tangent x Incoming)', {}, 'geometry_node'),
    ('radiosity (one bounce)', {'radiosity': True, 'radiosity_samples': 4,
                                'radiosity_distance': 4.0}, 'demo'),
    ('radiosity full-rate (spacing 1)',
     {'radiosity': True, 'radiosity_samples': 4, 'radiosity_distance': 4.0,
      'radiosity_spacing': 1}, 'demo'),
    ('blurry reflections', {'raytrace': True, 'ray_depth': 1,
                            'reflection_blur': 8.0,
                            'reflection_blur_samples': 4}, 'mirror'),
    ('burn-in stamp', {'watermark': 'SHOT 12 %F'}, 'demo'),
    ('image-driven matcap (per-pixel)', {}, 'matcap_image'),
    ('node Wireframe (cel ink)', {}, 'wireframe_node'),
    # ----------------------------------------------------------- textures
    ('texture NEAREST', {'tex_filter': 'NEAREST'}, 'textured'),
    ('texture BILINEAR', {'tex_filter': 'BILINEAR'}, 'textured'),
    ('texture TRILINEAR + mips', {'tex_filter': 'TRILINEAR',
                                  'tex_mipmap': True}, 'textured'),
    ('texture N64 3-point', {'tex_filter': 'N64_3POINT'}, 'textured'),
    ('mip bias sharp', {'tex_filter': 'TRILINEAR', 'tex_mipmap': True,
                        'tex_mip_bias': -1.0}, 'textured'),
    ('anisotropy 4x', {'tex_filter': 'TRILINEAR', 'tex_mipmap': True,
                       'tex_aniso': 4}, 'textured'),
    ('texture quantise 16', {'tex_quantize': 16}, 'textured'),
    ('texture size cap 64', {'tex_max_size': 64}, 'textured'),
    ('affine mapping (PS1 warp)', {'tex_perspective': False}, 'textured'),
    ('affine + subdivision', {'tex_perspective': False,
                              'tex_affine_subdiv': 8}, 'textured'),
    # ------------------------------------------------------- transparency
    ('transparency NONE', {'transparency': 'NONE'}, 'glass'),
    ('screen-door stipple', {'transparency': 'STIPPLE'}, 'glass'),
    ('sorted glass layers', {'transparency': 'SORTED'}, 'glass'),
    ('A-buffer glass', {'transparency': 'ABUFFER'}, 'glass'),
    ('ray-traced glass layers', {'transparency': 'SORTED',
                                 'raytrace': True, 'ray_depth': 1},
     'glass'),
    ('binary alpha', {'transparency': 'SORTED', 'alpha_bits': 1}, 'glass'),
    # ---------------------------------------------------------------- fog
    ('fog LINEAR', {'fog': True, 'fog_mode': 'LINEAR', 'fog_start': 3.0,
                    'fog_end': 12.0}, 'demo'),
    ('fog EXP2', {'fog': True, 'fog_mode': 'EXP2',
                  'fog_density': 0.12}, 'demo'),
    ('fog TABLE16 (fixed-function)', {'fog': True, 'fog_mode': 'TABLE16',
                                      'fog_start': 3.0, 'fog_end': 12.0},
     'demo'),
    ('per-vertex fog', {'fog': True, 'fog_mode': 'LINEAR',
                        'fog_start': 3.0, 'fog_end': 12.0,
                        'fog_vertex': True}, 'demo'),
    ('height fog (ground mist)', {'fog': True, 'fog_mode': 'EXP',
                                  'fog_density': 0.1, 'fog_height': True,
                                  'fog_height_top': 0.5,
                                  'fog_height_falloff': 1.5}, 'demo'),
    # -------------------------------------------------------------- depth
    ("Painter's sort", {'depth_sort': 'PAINTERS'}, 'demo'),
    ('z-buffer no write', {'depth_sort': 'ZBUFFER_NOWRITE'}, 'demo'),
    ('16-bit z-buffer', {'depth_precision': 16}, 'demo'),
    ('vertex snapping (PS1)', {'vertex_snap': True,
                               'vertex_snap_grid': 1.0}, 'demo'),
    ('fixed-point subpixel', {'subpixel_precision': 'FIXED_4'}, 'demo'),
    ('integer subpixel', {'subpixel_precision': 'INTEGER'}, 'demo'),
    ('backface culling', {'backface_cull': True}, 'demo'),
    ('orthographic camera', {}, 'ortho'),
    # ----------------------------------------------------------------- AA
    ('supersample 4x', {'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4},
     'demo'),
    ('edge antialias (flicker filter)', {'aa_mode': 'EDGE'}, 'demo'),
    ('accumulation AA', {'aa_mode': 'ACCUMULATE', 'aa_samples': 4},
     'demo'),
    # --------------------------------------------------------------- post
    ('ordered dither BAYER4', {'dither': 'BAYER4',
                               'color_depth': '16'}, 'demo'),
    ('error diffusion FLOYD', {'dither': 'FLOYD',
                               'color_depth': '16'}, 'demo'),
    ('15-bit colour', {'color_depth': '15'}, 'demo'),
    ('8-bit adaptive palette', {'color_depth': '8',
                                'palette_mode': 'ADAPTIVE'}, 'demo'),
    ('VGA 256 palette', {'color_depth': '8',
                         'palette_mode': 'VGA256'}, 'demo'),
    ('HAM8 (Amiga)', {'color_depth': 'HAM8'}, 'demo'),
    ('1-bit + halftone', {'color_depth': '1', 'dither': 'HALFTONE'},
     'demo'),
    ('glow / bloom', {'glow': True, 'glow_threshold': 0.5,
                      'glow_intensity': 0.8}, 'demo'),
    ('star filter', {'star_filter': True, 'glow_threshold': 0.5,
                     'star_intensity': 0.8}, 'demo'),
    ('lens flare', {'lens_flare': True, 'flare_intensity': 0.7}, 'demo'),
    ('light shafts', {'shaft_threshold': 0.4, 'shaft_length': 0.4},
     'demo'),
    ('depth of field', {'dof': True, 'dof_focus': 6.0,
                        'dof_amount': 2.0}, 'demo'),
    ('lens distortion + CA', {'lens_distortion': 0.3,
                              'chromatic_aberration': 2.0}, 'demo'),
    ('CRT (mask + scanlines + curve)', {'crt': True, 'crt_scanlines': 0.4,
                                        'crt_mask': 'APERTURE',
                                        'crt_curvature': 0.1,
                                        'crt_vignette': 0.3}, 'demo'),
    ('NTSC composite', {'composite': True, 'composite_bleed': 0.7},
     'demo'),
    ('interlace FIELDS', {'interlace': 'FIELDS'}, 'demo'),
    ('JPEG artefacts', {'jpeg_artifacts': True, 'jpeg_quality': 40},
     'demo'),
    ('display transform', {'exposure': 1.3, 'gamma': 1.2,
                           'contrast': 0.2, 'saturation': 1.4,
                           'brightness': 0.05}, 'demo'),
    ('transparent film', {'film_transparent': True}, 'demo'),
    # ---------------------------------------------------------- wire, misc
    ('wireframe overlay ALL', {'render_wire': True}, 'demo'),
    ('wireframe CREASE (cel ink)', {'render_wire': True,
                                    'wire_mode': 'CREASE'}, 'demo'),
    ('debug pass DEPTH', {'debug_pass': 'DEPTH'}, 'demo'),
    ('aux passes (depth + normal)', {'pass_depth': True,
                                     'pass_normal': True}, 'demo'),
]


def build(key, w=96, h=72):
    """(scene, settings) for one row, ready to render on either device."""
    from ..core.settings import RenderSettings

    for k, overrides, scene_name in ROWS:
        if k == key:
            st = RenderSettings()
            st.resolution_x, st.resolution_y = w, h
            st.aa_samples = 1
            st.shadows = True
            st.threads = 1
            for name, val in overrides.items():
                setattr(st, name, val)
            return SCENES[scene_name](st), st
    raise KeyError(key)


def keys():
    return [r[0] for r in ROWS]
