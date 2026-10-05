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

from ..core.scene import ImageBuffer, Light
from .scenebuild import checker_image, demo_scene


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


# ---- R251 sky-camera (SKY): the two pixel-addressed backdrops and the
# LightWave gradient, on synthetic images so nothing ships

def _cyl_image():
    """A 32x16 RGB image with a distinct hue per column and a distinct
    value per row: any column or row error moves a pixel."""
    px = np.zeros((16, 32, 4), np.float32)
    xs = np.arange(32, dtype=np.float32) / 31.0
    ys = np.arange(16, dtype=np.float32) / 15.0
    px[..., 0] = xs[None, :]
    px[..., 1] = ys[:, None]
    px[..., 2] = (1.0 - xs)[None, :] * 0.5 + 0.25
    px[..., 3] = 1.0
    return px


def _sc_doom_sky(st):
    """R251 C056: the demo scene under a Doom cylinder sky (the FM
    camera sees the sky above the floor plane)."""
    from ..core.scene import ImageBuffer
    sc = demo_scene(st, with_texture=False)
    buf = ImageBuffer(name='cylsky', pixels=_cyl_image(), colorspace='Linear')
    sc.images['cylsky'] = buf
    sc.world.mode = 'CYLINDER'
    sc.world.env_image = buf
    return sc


def _sc_lw_sky(st):
    """R251 C100: the demo scene under LightWave's gradient backdrop, the
    camera LEVELLED (the demo camera looks 18.5 deg down with a 17.8 deg
    half-FOV, so the frame would show the ground half alone)."""
    import dataclasses
    from .scenebuild import look_at_matrix
    sc = demo_scene(st, with_texture=False)
    sc.world.mode = 'LW_GRADIENT'
    sc.camera = dataclasses.replace(
        sc.camera, matrix_world=look_at_matrix((5.2, -6.4, 3.6),
                                               (0.0, -0.2, 3.6)))
    return sc


def _m7_image():
    """A 64x64 map: an 8x8 checker of saturated hues with a red
    one-texel border."""
    px = np.zeros((64, 64, 4), np.float32)
    hues = np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0),
                     (0, 1, 1), (1, 0, 1), (1, 1, 1), (0.5, 0.5, 0.5)],
                    np.float32)
    yy, xx = np.mgrid[0:64, 0:64]
    cell = ((yy // 8) * 8 + (xx // 8)) % 8
    px[..., :3] = hues[cell]
    px[0, :, :3] = (1, 0, 0)
    px[-1, :, :3] = (1, 0, 0)
    px[:, 0, :3] = (1, 0, 0)
    px[:, -1, :3] = (1, 0, 0)
    px[..., 3] = 1.0
    return px


def _sc_matte_glow(st):
    """R251 C134: the demo scene with a Glow Gel on the ball (the Tron
    printer's matte is cut from the ball's material pixels)."""
    sc = demo_scene(st, with_texture=False)
    sc.materials[1].glow_gel = (0.2, 0.9, 1.0)
    return sc


def _sc_mode7(st):
    """R251 C048: the demo scene MINUS its floor plane (the analytic
    floor is visible) under a GRADIENT sky with the Mode 7 floor on."""
    from ..core.scene import ImageBuffer
    from .scenebuild import _mesh_concat, cube, sphere
    sc = demo_scene(st, with_texture=False)
    mesh = _mesh_concat([
        sphere(centre=(-1.3, 0.2, 1.0), radius=1.0, mat=1, obj=1),
        cube(centre=(1.4, -0.4, 0.9), size=1.8, mat=2, obj=2),
    ])
    smooth = np.zeros(mesh.tris.shape[0], bool)
    smooth[mesh.mat_index == 1] = True
    mesh.smooth = smooth
    sc.mesh = mesh
    sc.world.mode = 'GRADIENT'
    sc.world.ground_plane = True
    sc.world.ground_mode = 'MODE7'
    sc.world.ground_height = 0.0
    sc.world.mode7_texel_size = 0.5
    sc.world.ground_image = ImageBuffer(name='m7', pixels=_m7_image(),
                                        colorspace='Linear')
    return sc


def _sc_textured(st):
    return demo_scene(st, with_texture=True)


def _sc_mirror(st):
    sc = demo_scene(st, with_texture=False)
    sc.materials[2].reflect_level = 0.5
    return sc


def _sc_textured_white(st):
    """R251: the demo checker with one pure white square, so Model 2's 0xF
    hole marker has a texel to fire on."""
    sc = demo_scene(st, with_texture=True)
    px = checker_image()
    px[0:8, 0:8, :3] = 1.0
    sc.images['checker'] = ImageBuffer(name='checker', pixels=px)
    return sc


def _sc_textured_blocks(st):
    """R251: a 5-texel checker (64 // 12 = 5 per square) under a two-axis
    brightness ramp (0.55..1.0 along x and along y), so every 4x4 DXT block
    and every 2x2 VQ block straddles a square edge AND holds more than two
    colours: the interpolants fire (a two-colour block decodes to its
    endpoints alone -- measured) and the 1024 distinct 2x2 blocks exceed
    the 256-entry codebook, so VQ must quantise."""
    sc = demo_scene(st, with_texture=True)
    px = checker_image(size=64, squares=12)
    ramp = np.linspace(0.55, 1.0, 64, dtype=np.float32)
    px[:, :, :3] *= (ramp[None, :, None] * ramp[:, None, None])
    sc.images['checker'] = ImageBuffer(name='checker', pixels=px)
    return sc


def _sc_textured_mag(st):
    """R251: an 8x8 two-square checker under REPEAT, bilinear -- it tiles
    four times across the floor and is MAGNIFIED on 2692 of 2724 floor
    pixels at 96x72 (m = 0.17..1.10), with real level-0 / level-1
    differences at every block edge: the scene for C080's terraces and
    C008's sharpen."""
    sc = demo_scene(st, with_texture=True)
    sc.images['checker'] = ImageBuffer(name='checker',
                                       pixels=checker_image(size=8, squares=2))
    sc.materials[0].graph['nodes']['tex']['props']['interpolation'] = 'Linear'
    return sc


def _sc_textured_extend(st):
    """R251: an Extend-mode, bilinear 8x8 checker whose [0,1]^2 sits ON the
    floor, so the GL_CLAMP half-texel seam spans pixels. The floor's four
    vertices come first in _mesh_concat (scenebuild :370-373) with UV * 4
    (:162); scaled to [0,1] the texture ends 11/4 units from the corner and
    the seam (8 texels over 11 units ~ 0.7 unit) is several pixels wide."""
    sc = demo_scene(st, with_texture=True)
    sc.mesh.uvs[:4] *= 0.25
    sc.images['checker'] = ImageBuffer(name='checker', pixels=checker_image(size=8, squares=2))
    node = sc.materials[0].graph['nodes']['tex']
    node['props']['extension'] = 'EXTEND'
    node['props']['interpolation'] = 'Linear'
    return sc


def _sc_textured_keyed(st):
    """R251: a bilinear checker with a transparent disc, for the chroma-key fringe."""
    sc = demo_scene(st, with_texture=True)
    px = checker_image(size=16, squares=4)
    yy, xx = np.mgrid[0:16, 0:16]
    px[(xx - 7.5) ** 2 + (yy - 7.5) ** 2 < 16.0, 3] = 0.0
    sc.images['checker'] = ImageBuffer(name='checker', pixels=px)
    sc.materials[0].graph['nodes']['tex']['props']['interpolation'] = 'Linear'
    return sc


def _sc_textured_keyed_alpha(st):
    """R251: the keyed checker with its Alpha WIRED, so the chroma-key range
    moves the frame (the keyed floor pixels composite over the world)."""
    sc = _sc_textured_keyed(st)
    nodes = sc.materials[0].graph['nodes']
    nodes['bsdf'] = {'id': 'bsdf', 'bl_idname': 'ShaderNodeBsdfPrincipled',
                     'props': {},
                     'inputs': [{'name': 'Base Color', 'type': 'RGBA',
                                 'default': [0.8, 0.8, 0.8, 1.0], 'link': ['tex', 0]},
                                {'name': 'Alpha', 'type': 'VALUE', 'default': 1.0,
                                 'link': ['tex', 1]},
                                {'name': 'Metallic', 'type': 'VALUE', 'default': 0.0, 'link': None},
                                {'name': 'Roughness', 'type': 'VALUE', 'default': 0.5, 'link': None},
                                {'name': 'IOR', 'type': 'VALUE', 'default': 1.45, 'link': None},
                                {'name': 'Specular', 'type': 'VALUE', 'default': 0.5, 'link': None},
                                {'name': 'Transmission', 'type': 'VALUE', 'default': 0.0, 'link': None},
                                {'name': 'Normal', 'type': 'VECTOR', 'default': [0, 0, 0], 'link': None}],
                     'outputs': [{'name': 'BSDF', 'type': 'SHADER'}]}
    sc.materials[0].has_alpha = True                  # R:5680: the see-through flag the exporter sets
    sc.materials[0].alpha_why = 'Image Texture Alpha'
    return sc


def _sc_shafts(st):
    # light shafts gate on a light's volumetric value; the demo lights
    # carry none, so a row on the plain demo scene would smear nothing
    # and prove nothing (found by the settings audit)
    sc = demo_scene(st, with_texture=False)
    sc.lights[1].volumetric = 2.0
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


def _gradient_opacity_graph(model='LAMBERT'):
    """R251 C031: a master graph whose Opacity is a 0..1 gradient across
    the object (Texture Coordinate > Generated > Separate XYZ > X)."""
    ins = [_sk('Diffuse Color', 'RGBA', [0.85, 0.2, 0.15, 1.0]),
           _sk('Vertex Color', 'RGBA', [1, 1, 1, 1]),
           _sk('Vertex Color Mix', 'VALUE', 0.0),
           _sk('Diffuse Level', 'VALUE', 1.0),
           _sk('Specular Color', 'RGBA', [1, 1, 1, 1]),
           _sk('Specular Level', 'VALUE', 0.4),
           _sk('Glossiness', 'VALUE', 24.0),
           _sk('Roughness', 'VALUE', 0.3),
           _sk('Ambient', 'VALUE', 1.0),
           _sk('Self-Illumination', 'RGBA', [0, 0, 0, 1]),
           _sk('Opacity', 'VALUE', 1.0, ['sep', 0]),
           _sk('IOR', 'VALUE', 1.45),
           _sk('Anisotropy', 'VALUE', 0.0),
           _sk('Anisotropic Rotation', 'VALUE', 0.0),
           _sk('Metalness', 'VALUE', 0.0),
           _sk('Soften', 'VALUE', 0.0),
           _sk('Reflection', 'VALUE', 0.0),
           _sk('Translucency', 'VALUE', 0.0),
           _sk('Toon Size', 'VALUE', 0.5),
           _sk('Toon Smooth', 'VALUE', 0.05)]
    return {'output': 'out', 'nodes': {
        'tc': {'id': 'tc', 'bl_idname': 'ShaderNodeTexCoord', 'props': {},
               'inputs': [],
               'outputs': [{'name': 'Generated', 'type': 'VECTOR'},
                           {'name': 'Normal', 'type': 'VECTOR'},
                           {'name': 'UV', 'type': 'VECTOR'},
                           {'name': 'Object', 'type': 'VECTOR'},
                           {'name': 'Camera', 'type': 'VECTOR'},
                           {'name': 'Window', 'type': 'VECTOR'},
                           {'name': 'Reflection', 'type': 'VECTOR'}]},
        'sep': {'id': 'sep', 'bl_idname': 'ShaderNodeSeparateXYZ',
                'props': {},
                'inputs': [_sk('Vector', 'VECTOR', [0, 0, 0], ['tc', 0])],
                'outputs': [{'name': 'X', 'type': 'VALUE'},
                            {'name': 'Y', 'type': 'VALUE'},
                            {'name': 'Z', 'type': 'VALUE'}]},
        'hal': {'id': 'hal', 'bl_idname': 'HALCYON_ShaderNode',
                'props': {'model': model}, 'inputs': ins,
                'outputs': [{'name': 'Surface', 'type': 'SHADER'}]},
        'out': {'id': 'out', 'bl_idname': 'ShaderNodeOutputMaterial',
                'props': {},
                'inputs': [_sk('Surface', 'SHADER', None, ['hal', 0]),
                           _sk('Displacement', 'VECTOR', [0, 0, 0])],
                'outputs': []}}}


def _sc_clip_blend(st):
    """R251 C031 (PS2 GS AFAIL): the pane scene's pane with a gradient
    alpha under Alpha Mode Clip+Blend at threshold 0.5 -- its solid
    half resolves in the z-buffer, the soft remainder blends."""
    sc = _sc_pane(st)
    m = sc.materials[1]
    m.opacity = 1.0
    m.graph = _gradient_opacity_graph()
    m.has_alpha = True
    m.alpha_why = 'Opacity linked (a gradient)'
    m.alpha_mode = 'CLIP_BLEND'
    m.alpha_clip = 0.5
    return sc


def _sc_clip_hard(st):
    """R251 C031's OFF tile: the same gradient-alpha pane under plain
    Alpha Clip at 0.5 (the hard cut-out; no row -- a collage reference)."""
    sc = _sc_clip_blend(st)
    sc.materials[1].alpha_mode = 'CLIP'
    return sc


def _tilted_pane(z, tilt, size, mat, obj):
    """A `plane` sheared in z along x (the TR `tilted()` shape)."""
    from .scenebuild import plane
    V, N, UV, T, _m, _o = plane(z=z, size=size, mat=mat, obj=obj)
    V = V.copy()
    V[:, 2] += V[:, 0] * tilt
    return (V, N, UV, T, mat, obj)


def _sc_thin_wall(st):
    """R251 C095 (3ds Max Thin Wall Refraction): the checker floor and
    the demo geometry plus a 30-degree tilted pane (a fourth material,
    Blend Mode THIN_WALL, opacity 0.2, IOR 1.5) -- the picture beneath
    it jogs along its projected normal; the OFF tile is the same pane
    at Alpha Over."""
    from ..core.scene import Material, ObjectInfo
    from .scenebuild import _mesh_concat
    sc = demo_scene(st, with_texture=True)
    m = sc.mesh
    base = (m.verts, m.normals, m.uvs, m.tris, 0, 0)
    pane = _tilted_pane(1.6, float(np.tan(np.radians(30.0))), 3.2, 3, 3)
    merged = _mesh_concat([base, pane])
    n0 = m.tris.shape[0]
    merged.mat_index[:n0] = m.mat_index
    merged.obj_index[:n0] = m.obj_index
    smooth = np.zeros(merged.tris.shape[0], bool)
    smooth[:n0] = m.smooth
    merged.smooth = smooth
    sc.mesh = merged
    sc.materials.append(Material(name='Pane', index=3, model='PHONG',
                                 diffuse=(0.7, 0.8, 0.9),
                                 specular_level=0.2, opacity=0.2,
                                 ior=1.5, blend_mode='THIN_WALL',
                                 thin_wall_offset=0.5))
    sc.objects.append(ObjectInfo(name='Pane', index=3, location=(0, 0, 1.6),
                                 matrix_world=np.eye(4, dtype=np.float32)))
    return sc


def _sc_thin_wall_alpha(st):
    """R251 C095's OFF tile: the same tilted pane under Alpha Over (no
    row -- a collage reference)."""
    sc = _sc_thin_wall(st)
    sc.materials[3].blend_mode = 'ALPHA'
    return sc


def _sc_imagine_fog(st):
    """R251 C101 (Imagine Fog Length): the demo scene's Box as a fog
    object -- a closed cube of depth 1.8 under Blend Mode IMAGINE_FOG,
    Fog Length 4.0, the Constant (shadeless) model for Imagine's unlit
    colour."""
    sc = demo_scene(st, with_texture=False)
    m = sc.materials[2]
    m.model = 'CONSTANT'
    m.blend_mode = 'IMAGINE_FOG'
    m.fog_length = 4.0
    return sc


def _sc_zoffs_decal(st):
    """R251 C126 (Blender 2.4x Zoffs): a shadeless decal quad 1e-3
    BEHIND the floor at opacity 0.5 with Z Offset 0.01 -- captured and
    drawn by the offset; without it the raster never collects it."""
    from ..core.scene import Material, ObjectInfo
    from .scenebuild import _mesh_concat, plane
    sc = demo_scene(st, with_texture=False)
    m = sc.mesh
    base = (m.verts, m.normals, m.uvs, m.tris, 0, 0)
    decal = plane(z=-1e-3, size=3.0, mat=3, obj=3)
    V = decal[0].copy()
    V[:, 0] += 1.2
    V[:, 1] += 2.2
    merged = _mesh_concat([base, (V,) + decal[1:]])
    n0 = m.tris.shape[0]
    merged.mat_index[:n0] = m.mat_index
    merged.obj_index[:n0] = m.obj_index
    smooth = np.zeros(merged.tris.shape[0], bool)
    smooth[:n0] = m.smooth
    merged.smooth = smooth
    sc.mesh = merged
    sc.materials.append(Material(name='Decal', index=3, model='CONSTANT',
                                 diffuse=(0.95, 0.85, 0.1), opacity=0.5,
                                 z_offset=0.01))
    sc.objects.append(ObjectInfo(name='Decal', index=3,
                                 location=(1.2, 2.2, -1e-3),
                                 matrix_world=np.eye(4, dtype=np.float32)))
    return sc


def _sc_zinvert_shell(st):
    """R251 C126 (Blender 2.4x ZInvert): the DS shell geometry (a second
    sphere around the Ball, one material) with Invert Z Depth on -- the
    far inner walls composite in front of the near ones."""
    sc = _sc_ds_shell(st)
    sc.materials[1].z_invert = True
    return sc


def _sc_env_hole(st):
    """R251 C126 (Blender 2.4x Env): the Ball as an Env hole -- the
    world along the view ray, alpha 0, nothing behind it."""
    sc = demo_scene(st, with_texture=False)
    sc.materials[1].blend_mode = 'ENV_HOLE'
    return sc


def _sc_glass(st):
    sc = demo_scene(st, with_texture=False)
    sc.materials[1].opacity = 0.5
    sc.materials[2].opacity = 0.6
    sc.materials[2].reflect_level = 0.5
    return sc


def _sc_pane(st, two=False):
    """R251 (transparency pack): the law-probe scene. The glass scene's Ball
    and Box are CLOSED meshes rasterised into both faces, so no pixel of
    theirs holds exactly one fragment; here the Ball and Box take the
    Floor's material and materials[1] is an OPEN quad hovering above the
    Ball's top -- one fragment per pixel it covers (299 at 96x72). With
    `two=True` a second pane (materials[2]) sits under it in projection
    for the two-deep chains. No row: a probe scene, not a feature."""
    from .scenebuild import _mesh_concat, cube, plane, sphere
    from ..core.scene import ObjectInfo
    sc = demo_scene(st, with_texture=False)
    parts = [plane(z=0.0, size=11.0, mat=0, obj=0),
             sphere(centre=(-1.3, 0.2, 1.0), radius=1.0, mat=0, obj=1),
             cube(centre=(1.4, -0.4, 0.9), size=1.8, mat=0, obj=2),
             plane(z=2.2, size=3.0, mat=1, obj=3)]
    sc.objects.append(ObjectInfo(name='Pane', index=3, location=(0, 0, 2.2),
                                 matrix_world=np.eye(4, dtype=np.float32)))
    if two:
        V, N, UV, T, mi, oi = plane(z=1.2, size=4.0, mat=2, obj=4)
        V = V.copy()
        V[:, 0] += -1.9
        V[:, 1] += 2.3
        parts.append((V, N, UV, T, mi, oi))
        sc.objects.append(ObjectInfo(name='Pane2', index=4,
                                     location=(-1.9, 2.3, 1.2),
                                     matrix_world=np.eye(4, dtype=np.float32)))
    mesh = _mesh_concat(parts)
    smooth = np.zeros(mesh.tris.shape[0], bool)
    smooth[mesh.obj_index == 1] = True
    mesh.smooth = smooth
    sc.mesh = mesh
    sc.materials[1].opacity = 0.5
    sc.materials[2].opacity = 0.6
    return sc


def _sc_ds_shell(st):
    """R251 (transparency pack): a second, larger sphere around the Ball with
    the Ball's material AND object index -- a pixel where the shells
    overlap holds four fragments of one polygon ID (the DS same-ID-once
    rule's scene). `mesh.smooth` is rebuilt for the concatenated mesh."""
    from .scenebuild import _mesh_concat, sphere
    sc = demo_scene(st, with_texture=False)
    m = sc.mesh
    shell = sphere(centre=(-1.3, 0.2, 1.0), radius=1.3, mat=1, obj=1)
    base = (m.verts, m.normals, m.uvs, m.tris, 0, 0)
    merged = _mesh_concat([base, shell])
    # _mesh_concat fills mat/obj per part from scalars: restore the demo
    # mesh's own per-triangle indices for its part
    n0 = m.tris.shape[0]
    merged.mat_index[:n0] = m.mat_index
    merged.obj_index[:n0] = m.obj_index
    smooth = np.zeros(merged.tris.shape[0], bool)
    smooth[merged.mat_index == 1] = True
    merged.smooth = smooth
    sc.mesh = merged
    sc.materials[1].opacity = 0.5
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


def _sc_only_shadow(st):
    """R251 F014: the demo scene plus one POINT lamp that only casts
    its shadow (Blender Internal's LA_ONLYSHADOW)."""
    sc = demo_scene(st, with_texture=False)
    osh = Light(type='POINT', name='shadower', position=(1.5, -1.5, 3.0),
                color=(1.0, 1.0, 1.0), energy=300.0, shadow='MAP')
    osh.only_shadow = True
    sc.lights = list(sc.lights) + [osh]
    return sc


def _sc_spot_law(st, law, exponent=0.0):
    """R251 F012: the demo scene plus one SPOT at (2, -2, 4) aimed at
    the origin, spot_size 1.0, hotspot 0.5, no shadow, under a cone law."""
    sc = demo_scene(st, with_texture=False)
    pos = np.array([2.0, -2.0, 4.0], np.float32)
    d = -pos / np.float32(np.linalg.norm(pos))
    spot = Light(type='SPOT', name='cone', position=tuple(float(v) for v in pos),
                 direction=tuple(float(v) for v in d), color=(1.0, 0.95, 0.85),
                 energy=600.0, spot_size=1.0, spot_blend=0.2, hotspot=0.5,
                 shadow='NONE', decay='INVERSE_SQUARE')
    spot.spot_law = law
    spot.spot_exponent = float(exponent)
    sc.lights = list(sc.lights) + [spot]
    return sc


def _sc_spot_pov(st):
    return _sc_spot_law(st, 'POV')


def _sc_spot_gl(st):
    return _sc_spot_law(st, 'GL11', 8.0)


def _sc_spot_gx_ring(st):
    return _sc_spot_law(st, 'GX_RING1')


def _sc_decay_law(st, end, ld1=0.0, ld2=0.0, rb=0.5):
    """R251 F013: the demo POINT lamp on the Scene Default falloff with
    the law's distance / sliders / ref_brite; the ROW sets the law as the
    scene default, so the reference (INVERSE_SQUARE) is the same lamp."""
    sc = demo_scene(st, with_texture=False)
    for l in sc.lights:
        if l.type == 'POINT':
            l.decay = 'DEFAULT'
            l.decay_end = float(end)
            l.decay_ld1 = float(ld1)
            l.decay_ld2 = float(ld2)
            l.gx_ref_brite = float(rb)
    return sc


def _sc_decay_pov(st):
    return _sc_decay_law(st, 4.0)


def _sc_decay_gl(st):
    return _sc_decay_law(st, 25.0, ld1=0.5, ld2=0.1)


def _sc_decay_gx(st):
    return _sc_decay_law(st, 5.0, rb=0.3)


def _sc_screen_spot(st):
    """R251 F015: the demo scene plus one Model 3 screen spotlight --
    a SPOT at (0, -3, 3) toward the origin, energy 200, Falloff End 10."""
    sc = demo_scene(st, with_texture=False)
    pos = np.array([0.0, -3.0, 3.0], np.float32)
    d = -pos / np.float32(np.linalg.norm(pos))
    spot = Light(type='SPOT', name='headlight',
                 position=tuple(float(v) for v in pos),
                 direction=tuple(float(v) for v in d), color=(1.0, 0.9, 0.7),
                 energy=200.0, spot_size=1.0, decay_end=10.0, shadow='NONE')
    spot.screen_spot = True
    sc.lights = list(sc.lights) + [spot]
    return sc


def _sc_facing_quad(st):
    """R251 F011: one quad squarely facing the camera (its normal IS
    the camera axis view[2, :3]) under one SUN 30 degrees off the
    axis, PHONG through the demo's Ball material -- under the fixed
    camera-axis viewer every pixel sees the same N, L and viewer, so
    the highlight is constant across the quad; per pixel it slides."""
    from ..core import render as _R
    from .scenebuild import _mesh_concat
    sc = demo_scene(st, with_texture=False)
    view, _p, _vp, eye = _R.camera_matrices(sc.camera, 96, 72)
    right = np.asarray(view[0, :3], np.float32)
    up = np.asarray(view[1, :3], np.float32)
    axis = np.asarray(view[2, :3], np.float32)
    axis = axis / np.float32(np.linalg.norm(axis))
    c = np.asarray(eye, np.float32) - axis * np.float32(6.0)
    V = np.stack([c - right * 3.0 - up * 3.0, c + right * 3.0 - up * 3.0,
                  c + right * 3.0 + up * 3.0, c - right * 3.0 + up * 3.0]
                 ).astype(np.float32)
    N = np.tile(axis[None, :], (4, 1)).astype(np.float32)
    UV = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    T = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
    mesh = _mesh_concat([(V, N, UV, T, 1, 1)])
    mesh.face_normals = np.tile(axis[None, :], (2, 1)).astype(np.float32)
    sc.mesh = mesh
    d = -(axis + right * np.float32(0.577))
    d = d / np.float32(np.linalg.norm(d))
    sc.lights = [Light(type='SUN', name='Key', direction=tuple(float(v) for v in d),
                       color=(1.0, 1.0, 1.0), energy=4.0, shadow='NONE')]
    return sc


def _sc_pitched(st):
    """R251 C058: the demo scene with its camera pitched 16.5 degrees
    UP, so the Y-shear row is not vacuous (a level camera passes
    through bitwise)."""
    from .scenebuild import look_at_matrix
    sc = demo_scene(st, with_texture=False)
    sc.camera.matrix_world = look_at_matrix((5.2, -6.4, 1.0),
                                            (0.0, -0.2, 3.4))
    return sc


def _sc_wide_lens(st):
    """R251 C098: the demo scene at f/1.4 (the f-number lives on the
    camera); at the demo's f/2.8 and this size the lens offsets stay
    under a pixel and the rows would be near-vacuous."""
    sc = demo_scene(st, with_texture=False)
    sc.camera.fstop = 1.4
    return sc


def _sc_ortho(st):
    sc = demo_scene(st, with_texture=False)
    sc.camera.type = 'ORTHO'
    return sc


def _sc_near_wall(st):
    """R251 C027: the demo camera slid along its view axis until the
    floor plane crosses clip_start inside the frame -- CLIP cuts the
    floor at the near plane, REJECT drops its straddling triangle whole
    (the PS2 / PS1 vanishing floor). The slide is computed from the
    camera's own clip_start, so the row is non-vacuous at 96x72."""
    sc = demo_scene(st, with_texture=False)
    cam = sc.camera
    mw = np.asarray(cam.matrix_world, np.float64).copy()
    fwd = -mw[:3, 2]
    fwd = fwd / np.linalg.norm(fwd)
    eye = mw[:3, 3].copy()
    m = sc.mesh
    fv = m.verts[np.unique(m.tris[m.mat_index == 0])].astype(np.float64)
    depth = (fv - eye[None, :]) @ fwd
    slide = float(depth.min()) - 0.5 * float(cam.clip_start)
    mw[:3, 3] = eye + fwd * slide
    cam.matrix_world = mw.astype(np.float32)
    return sc


def _sc_depth_fight_pair(st):
    """R251 C007/C026/C075: a red sheet 5e-4 units above the floor --
    resolved cleanly by a 24-bit linear buffer (2e-5 units at that
    depth), fought over by the period encodings wherever their step at
    that depth is coarser than the gap (the fill's tie rule hands an
    equal code to the floor, the lower id). Measured at 96x72: the
    sheet keeps 2591 px under LINEAR-24, 1114 under N64_FLOAT18, 604
    under GC_14E2, 822 under W_FIXED-16, 2145 under VOODOO_W16. The
    fight is the encoding's own picture."""
    from .scenebuild import _mesh_concat, cube, plane, sphere
    from ..core.scene import ObjectInfo
    sc = demo_scene(st, with_texture=False)
    mesh = _mesh_concat([
        plane(z=0.0, size=11.0, mat=0, obj=0),
        sphere(centre=(-1.3, 0.2, 1.0), radius=1.0, mat=1, obj=1),
        cube(centre=(1.4, -0.4, 0.9), size=1.8, mat=2, obj=2),
        plane(z=5e-4, size=9.0, mat=1, obj=3),
    ])
    smooth = np.zeros(mesh.tris.shape[0], bool)
    smooth[mesh.obj_index == 1] = True
    mesh.smooth = smooth
    sc.mesh = mesh
    sc.objects = list(sc.objects) + [
        ObjectInfo(name='Sheet', index=3, location=(0, 0, 5e-4),
                   matrix_world=np.eye(4, dtype=np.float32))]
    return sc


def _sc_quantized_ship(st):
    """R251 C012: the demo's sphere and cube at a third of their size
    on the floor -- a small 'ship' whose vertices a coarse world
    lattice (4 units per unit) crunches visibly at 96x72: the sphere's
    rings collapse to steps and the cube's corners land on the grid."""
    from .scenebuild import _mesh_concat, cube, plane, sphere
    sc = demo_scene(st, with_texture=False)
    mesh = _mesh_concat([
        plane(z=0.0, size=11.0, mat=0, obj=0),
        sphere(centre=(-0.9, 0.3, 0.35), radius=0.35, mat=1, obj=1),
        cube(centre=(0.7, -0.6, 0.3), size=0.6, mat=2, obj=2),
        sphere(centre=(0.2, 1.1, 0.5), radius=0.5, segs=12, rings=8,
               mat=1, obj=1),
    ])
    smooth = np.zeros(mesh.tris.shape[0], bool)
    smooth[mesh.mat_index == 1] = True
    mesh.smooth = smooth
    sc.mesh = mesh
    return sc


def _sc_backdrop_depth(st):
    """R251 C038: the demo scene behind a slanted backdrop wall -- a Z
    pass ramping from 2.0 units (left) to 30.0 (right) at the frame's
    own size, so the sphere and the floor pass behind the near side and
    stand in front of the far side (non-vacuous at 96x72)."""
    sc = demo_scene(st, with_texture=False)
    h = max(int(getattr(st, 'resolution_y', 72)), 1)
    w = max(int(getattr(st, 'resolution_x', 96)), 1)
    ramp = np.linspace(2.0, 30.0, w, dtype=np.float32)
    sc.world.backdrop_depth = np.ascontiguousarray(
        np.tile(ramp[None, :], (h, 1)).astype(np.float32))
    sc.world.backdrop_offset = (0, 0)
    return sc


def _sc_coverage_edge(st):
    """R251 C001: the demo scene with a raised sheet whose long slanted
    edges run across the frame -- rows of partial-coverage pixels the
    VI blends and the divot medians, visible at 96x72."""
    from .scenebuild import _mesh_concat, cube, plane, sphere
    sc = demo_scene(st, with_texture=False)
    mesh = _mesh_concat([
        plane(z=0.0, size=11.0, mat=0, obj=0),
        sphere(centre=(-1.3, 0.2, 1.0), radius=1.0, mat=1, obj=1),
        cube(centre=(1.4, -0.4, 0.9), size=1.8, mat=2, obj=2),
        plane(z=1.6, size=3.2, mat=2, obj=2),
    ])
    smooth = np.zeros(mesh.tris.shape[0], bool)
    smooth[mesh.mat_index == 1] = True
    mesh.smooth = smooth
    sc.mesh = mesh
    return sc


def _sc_ordering_table_stack(st):
    """R251 C004: three equal sheets stacked 0.06 units apart in front
    of the objects, stepped sideways along the screen's horizontal so
    all three show and their mean depths differ ONLY by the stack
    (blue lowest and added first, red, grey on top). A centroid
    Painter's sort draws the top sheet on top; the ordering table at 256
    entries to 40 units (a 0.16-unit bucket) lands all three in ONE
    bucket and the first-added -- the lowest -- ends on top: the PS1's
    bucket-order sort error. At 4096 entries they sort correctly."""
    from .scenebuild import _mesh_concat, cube, plane, sphere
    from ..core.scene import ObjectInfo
    sc = demo_scene(st, with_texture=False)
    mw = np.asarray(sc.camera.matrix_world, np.float64)
    fwd = -mw[:3, 2] / np.linalg.norm(mw[:3, 2])
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    right = right / np.linalg.norm(right)

    def sheet(i, z, mat, obj):
        V, N, UV, T, mi, oi = plane(z=z, size=1.4, mat=mat, obj=obj)
        V = V.copy()
        V[:, 0] += np.float32(-0.7 + 0.3 * i * right[0])
        V[:, 1] += np.float32(-2.5 + 0.3 * i * right[1])
        return (V, N, UV, T, mi, oi)
    mesh = _mesh_concat([
        plane(z=0.0, size=11.0, mat=0, obj=0),
        sphere(centre=(-1.3, 0.2, 1.0), radius=1.0, mat=1, obj=1),
        cube(centre=(1.4, -0.4, 0.9), size=1.8, mat=2, obj=2),
        sheet(0, 0.50, 2, 3),
        sheet(1, 0.56, 1, 4),
        sheet(2, 0.62, 0, 5),
    ])
    smooth = np.zeros(mesh.tris.shape[0], bool)
    smooth[mesh.obj_index == 1] = True
    mesh.smooth = smooth
    sc.mesh = mesh
    sc.objects = list(sc.objects) + [
        ObjectInfo(name=f'Sheet{i}', index=i, location=(-0.7, -2.5, z),
                   matrix_world=np.eye(4, dtype=np.float32))
        for i, z in ((3, 0.50), (4, 0.56), (5, 0.62))]
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


def _sc_light_linked(st):
    """Light linking, both modes at once: a magenta point EXCLUDED from
    object 1, a cyan point lighting ONLY object 2 -- the per-object
    masks that routed the whole frame to the CPU until the td.y ladder
    carried light_surface's isin() decision into the light loop."""
    sc = demo_scene(st, with_texture=False)
    ex = Light(type='POINT', name='not-the-ball',
               position=(-2.0, -2.5, 4.0), color=(1.0, 0.2, 1.0),
               energy=400.0, shadow='NONE')
    ex.exclude_objects = (1,)
    ex.exclude_mode = 'EXCLUDE'
    on = Light(type='POINT', name='only-the-box',
               position=(2.5, -2.0, 3.5), color=(0.2, 1.0, 1.0),
               energy=400.0, shadow='NONE')
    on.exclude_objects = (2,)
    on.exclude_mode = 'ONLY'
    sc.lights = list(sc.lights) + [ex, on]
    return sc


def _sc_geometry_aux_node(st):
    """Geometry's True Normal (abs) scaled by Random Per Island -- the
    two outputs that refused until the hal_triaux per-tri texture baked
    the CPU's own stored normals and sin-fract randoms."""
    sc = demo_scene(st, with_texture=False)
    sc.materials[1].graph = {'output': 'out', 'nodes': {
        'geo': {'id': 'geo', 'bl_idname': 'ShaderNodeNewGeometry',
                'props': {}, 'inputs': [],
                'outputs': [dict(o) for o in GEOMETRY_OUTS]},
        'absn': {'id': 'absn', 'bl_idname': 'ShaderNodeVectorMath',
                 'props': {'operation': 'ABSOLUTE'},
                 'inputs': [_sk('Vector', 'VECTOR', [0, 0, 0],
                                ['geo', 3])],
                 'outputs': [{'name': 'Vector', 'type': 'VECTOR'}]},
        'scal': {'id': 'scal', 'bl_idname': 'ShaderNodeVectorMath',
                 'props': {'operation': 'SCALE'},
                 'inputs': [_sk('Vector', 'VECTOR', [0, 0, 0],
                                ['absn', 0]),
                            _sk('Scale', 'VALUE', 1.0, ['geo', 8])],
                 'outputs': [{'name': 'Vector', 'type': 'VECTOR'}]},
        'bsdf': {'id': 'bsdf', 'bl_idname': 'ShaderNodeBsdfDiffuse',
                 'props': {},
                 'inputs': [_sk('Color', 'RGBA', [1, 1, 1, 1],
                                ['scal', 0]),
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


def _sc_planar_half(st):
    """R251 C052: the SUN's planar shadow at half density, the POINT
    lamp casting none."""
    sc = demo_scene(st, with_texture=False)
    sc.lights[0].shadow_density = 0.5
    sc.lights[1].shadow = 'NONE'
    return sc


def _sc_planar_raised(st):
    """R251 C052: the demo rebuilt with its floor at Z = 0.4 (ball and
    box lifted with it), so the raised-plane row moves the picture."""
    from .scenebuild import cube, plane, sphere, _mesh_concat
    sc = demo_scene(st, with_texture=False)
    mesh = _mesh_concat([
        plane(z=0.4, size=11.0, mat=0, obj=0),
        sphere(centre=(-1.3, 0.2, 1.4), radius=1.0, mat=1, obj=1),
        cube(centre=(1.4, -0.4, 1.3), size=1.8, mat=2, obj=2),
    ])
    smooth = np.zeros(mesh.tris.shape[0], bool)
    smooth[mesh.mat_index == 1] = True
    mesh.smooth = smooth
    sc.mesh = mesh
    return sc


def _volume_cube(centre, size, role, color=(0.0, 0.0, 0.0), alpha=16,
                 polygon_id=1, name='Volume'):
    from .scenebuild import cube
    v, _n, _uv, t, _m, _o = cube(centre=centre, size=size)
    return {'name': name, 'verts': v, 'tris': t, 'role': role,
            'polygon_id': int(polygon_id), 'alpha': int(alpha),
            'color': tuple(color)}


def _sc_modvol(st):
    """R251 C020: a Dreamcast INCLUDE modifier volume around the ball's
    base (the PowerVR parity shadow at the DC Shadow Scale)."""
    sc = demo_scene(st, with_texture=False)
    sc.shadow_volumes.append(_volume_cube((-1.3, 0.2, 0.5), 1.4,
                                          'DC_INCLUDE', name='Modifier'))
    return sc


def _sc_modvol_exclude(st):
    """R251 C020: the same volume as EXCLUDE -- the whole frame outside
    it takes the scale."""
    sc = demo_scene(st, with_texture=False)
    sc.shadow_volumes.append(_volume_cube((-1.3, 0.2, 0.5), 1.4,
                                          'DC_EXCLUDE', name='Modifier'))
    return sc


def _sc_ds_shadow(st):
    """R251 C036: a DS shadow polygon volume around the ball, the ball
    carrying the volume's polygon ID (self-exclusion)."""
    sc = demo_scene(st, with_texture=False)
    sc.materials[1].polygon_id = 1
    sc.shadow_volumes.append(_volume_cube((-1.3, 0.2, 0.3), 1.6,
                                          'DS_SHADOW', color=(0, 0, 0),
                                          alpha=16, polygon_id=1,
                                          name='Shadow'))
    return sc


def _sc_pov_finish(st, frame=None, **sockets):
    """R251 (LIGHT-B2): the demo scene with the ball's material on the
    master shader and the named POV finish sockets set -- Brilliance,
    Crand, Metallic (POV) -- a white Sun as the demo's key. `frame` sets
    the scene's frame number (crand per frame)."""
    sc = demo_scene(st, with_texture=False)
    diffuse = list(sockets.pop('diffuse', (0.85, 0.2, 0.15))) + [1.0]
    # every socket the master reads without a fallback (a missing socket
    # reads as zero: Opacity 0 would refuse, Diffuse Level 0 go black)
    ins = [
        _sk('Diffuse Color', 'RGBA', diffuse),
        _sk('Vertex Color', 'RGBA', [1, 1, 1, 1]),
        _sk('Vertex Color Mix', 'VALUE', 0.0),
        _sk('Diffuse Level', 'VALUE', 1.0),
        _sk('Specular Color', 'RGBA', [1, 1, 1, 1]),
        _sk('Specular Level', 'VALUE', 0.9),
        _sk('Glossiness', 'VALUE', 48.0),
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
    ]
    for name, val in sockets.items():
        ins.append(_sk(name, 'VALUE', float(val)))
    sc.materials[1].graph = _one_bsdf_graph('HALCYON_ShaderNode', ins,
                                            {'model': 'PHONG',
                                             'toon_steps': 2})
    if frame is not None:
        sc.frame = int(frame)
    return sc


def _sc_pov_finish_plain(st):
    return _sc_pov_finish(st)


def _sc_pov_finish_plain_red(st):
    return _sc_pov_finish(st, diffuse=(1.0, 0.2, 0.1))


def _sc_pov_finish_brilliance(st):
    return _sc_pov_finish(st, **{'Brilliance': 2.0})


def _sc_pov_finish_crand(st):
    return _sc_pov_finish(st, **{'Crand': 0.3})


def _sc_pov_finish_crand_f3(st):
    return _sc_pov_finish(st, frame=3, **{'Crand': 0.3})


def _sc_pov_finish_metallic(st):
    return _sc_pov_finish(st, diffuse=(1.0, 0.2, 0.1),
                          **{'Metallic (POV)': 1.0})


# ---- R251 (RAST-B): the AA resolve and wire-road scenes -----------------

def _sc_hot_edge(st):
    """C094: an overbright object on the dark demo world -- its edges are
    where an unclipped sample bleeds a halo through the AA filter."""
    sc = demo_scene(st, with_texture=False)
    sc.materials[1].emission = (1.0, 0.95, 0.8)
    sc.materials[1].emission_level = 6.0
    return sc


def _sc_elite_ship(st):
    """C063: a concave arrangement -- a cube BEHIND a nearer wall, on the
    floor -- where Elite's rule shows the hidden cube's edges through the
    wall and a depth-tested overlay cannot."""
    from .scenebuild import _mesh_concat, cube, plane
    sc = demo_scene(st, with_texture=False)
    sc.mesh = _mesh_concat([
        plane(z=0.0, size=11.0, mat=0, obj=0),
        cube(centre=(2.0, -2.6, 1.2), size=2.4, mat=2, obj=1),   # the wall
        cube(centre=(-0.6, 1.2, 0.9), size=1.8, mat=1, obj=2),   # behind it
    ])
    sc.world.color = (0.0, 0.0, 0.0)
    sc.world.sky_blend = False
    return sc


def _sc_beam_vectors(st):
    """C055: a black world with a low-poly ball, a cube and the floor's
    rim -- every edge a crease past 25 degrees, so every edge is a stroke
    and the strokes cross where the ball overlaps the cube."""
    from .scenebuild import _mesh_concat, cube, plane, sphere
    sc = demo_scene(st, with_texture=False)
    sc.mesh = _mesh_concat([
        plane(z=0.0, size=11.0, mat=0, obj=0),
        sphere(centre=(-1.0, 0.4, 1.1), radius=1.2, segs=8, rings=4, mat=1,
               obj=1, smooth=False),
        cube(centre=(1.4, -0.4, 0.9), size=1.8, mat=2, obj=2),
    ])
    sc.world.color = (0.0, 0.0, 0.0)
    sc.world.sky_blend = False
    return sc


# ---- R251 material pack, wave 2 (MAT-B): the period node scenes ----
def _matb_master_ins(diffuse=(0.85, 0.2, 0.15), **over):
    """The master shader's input list (the _sc_pov_finish shape); `over`
    sets VALUE sockets by name."""
    ins = [
        _sk('Diffuse Color', 'RGBA', list(diffuse) + [1.0]),
        _sk('Vertex Color', 'RGBA', [1, 1, 1, 1]),
        _sk('Vertex Color Mix', 'VALUE', 0.0),
        _sk('Diffuse Level', 'VALUE', 1.0),
        _sk('Specular Color', 'RGBA', [1, 1, 1, 1]),
        _sk('Specular Level', 'VALUE', 0.9),
        _sk('Glossiness', 'VALUE', 48.0),
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
    ]
    for k, v in over.items():
        ins.append(_sk(k, 'VALUE', float(v)))
    return ins


def _matb_out(nodes, last):
    nodes['bsdf'] = {'id': 'bsdf', 'bl_idname': 'ShaderNodeBsdfDiffuse', 'props': {},
                     'inputs': [_sk('Color', 'RGBA', [1, 1, 1, 1], [last, 0]),
                                _sk('Roughness', 'VALUE', 0.0)],
                     'outputs': [{'name': 'BSDF', 'type': 'SHADER'}]}
    nodes['out'] = {'id': 'out', 'bl_idname': 'ShaderNodeOutputMaterial', 'props': {},
                    'inputs': [_sk('Surface', 'SHADER', None, ['bsdf', 0]),
                               _sk('Displacement', 'VECTOR', [0, 0, 0])],
                    'outputs': []}
    return {'output': 'out', 'nodes': nodes}


def _sc_combiner_node(st, hardware='TEV', **props):
    """C028: checker -> A, a constant B, Facing -> C, D = 0 through one
    TEV ADD stage (bias +0.5, scale x2, clamp) into a diffuse BSDF."""
    sc = demo_scene(st, with_texture=True)
    p = {'hardware': hardware, 'op': 'ADD', 'bias': 'ADD_HALF', 'scale': 'X2',
         'clamp': True, 'map_a': 'UNSIGNED_IDENTITY', 'map_b': 'UNSIGNED_IDENTITY',
         'map_c': 'UNSIGNED_IDENTITY', 'map_d': 'UNSIGNED_IDENTITY',
         'nv_scale': 'X1', 'nv_bias': 'NONE'}
    p.update(props)
    nodes = {
        'tex': {'id': 'tex', 'bl_idname': 'ShaderNodeTexImage',
                'props': {'image': 'checker', 'interpolation': 'Closest'},
                'inputs': [_sk('Vector', 'VECTOR', [0, 0, 0])],
                'outputs': [{'name': 'Color', 'type': 'RGBA'}, {'name': 'Alpha', 'type': 'VALUE'}]},
        'facing': {'id': 'facing', 'bl_idname': 'HALCYON_FacingNode', 'props': {},
                   'inputs': [_sk('Power', 'VALUE', 1.0), _sk('IOR', 'VALUE', 1.45)],
                   'outputs': [{'name': 'Facing', 'type': 'VALUE'}, {'name': 'Incidence', 'type': 'VALUE'},
                               {'name': 'Fresnel', 'type': 'VALUE'}]},
        'comb': {'id': 'comb', 'bl_idname': 'HALCYON_CombinerStageNode', 'props': p,
                 'inputs': [_sk('A', 'RGBA', [0, 0, 0, 1], ['tex', 0]),
                            _sk('B', 'RGBA', [0.9, 0.3, 0.2, 1]),
                            _sk('C', 'RGBA', [0, 0, 0, 1], ['facing', 0]),
                            _sk('D', 'RGBA', [0, 0, 0, 1])],
                 'outputs': [{'name': 'Color', 'type': 'RGBA'}]},
    }
    sc.materials[1].graph = _matb_out(nodes, 'comb')
    return sc


def _sc_combiner_nv2a_node(st):
    return _sc_combiner_node(st, hardware='NV2A', map_a='EXPAND_NORMAL')


def _matb_nmap(sc):
    """The scenebuild normal-map image (add_normal_mapped_ball's recipe):
    a 10x10 random tangent-space map whose decoded z stays >= 0.5."""
    rng = np.random.default_rng(7)
    nm = rng.random((10, 10, 4)).astype(np.float32)
    nm[:, :, 2] = 0.75 + nm[:, :, 2] * 0.25
    nm[:, :, 3] = 1.0
    sc.images['nmap'] = ImageBuffer(name='nmap', pixels=nm, colorspace='Non-Color')
    return sc


def _sc_srbump_node(st, light=(0.6, 0.0, 0.8), strength=0.7, blend='MULTIPLY'):
    """C023: the nmap image (Closest) -> Color, a constant Light, the
    checker -> Base through the SR Bump node into a diffuse BSDF."""
    sc = _matb_nmap(demo_scene(st, with_texture=False))
    sc.images['checker'] = ImageBuffer(name='checker', pixels=checker_image())
    nodes = {
        'ntex': {'id': 'ntex', 'bl_idname': 'ShaderNodeTexImage',
                 'props': {'image': 'nmap', 'interpolation': 'Closest'},
                 'inputs': [_sk('Vector', 'VECTOR', [0, 0, 0])],
                 'outputs': [{'name': 'Color', 'type': 'RGBA'}, {'name': 'Alpha', 'type': 'VALUE'}]},
        'tex': {'id': 'tex', 'bl_idname': 'ShaderNodeTexImage',
                'props': {'image': 'checker', 'interpolation': 'Closest'},
                'inputs': [_sk('Vector', 'VECTOR', [0, 0, 0])],
                'outputs': [{'name': 'Color', 'type': 'RGBA'}, {'name': 'Alpha', 'type': 'VALUE'}]},
        'srb': {'id': 'srb', 'bl_idname': 'HALCYON_SRBumpNode', 'props': {'blend': blend},
                'inputs': [_sk('Color', 'RGBA', [0.5, 0.5, 1.0, 1.0], ['ntex', 0]),
                           _sk('Light', 'VECTOR', list(light)),
                           _sk('Strength', 'VALUE', float(strength)),
                           _sk('Base', 'RGBA', [0.8, 0.8, 0.8, 1.0], ['tex', 0])],
                'outputs': [{'name': 'Intensity', 'type': 'VALUE'}, {'name': 'Color', 'type': 'RGBA'}]},
    }
    g = _matb_out(nodes, 'srb')
    g['nodes']['bsdf']['inputs'][0]['link'] = ['srb', 1]
    sc.materials[1].graph = g
    return sc


def _sc_emboss_node(st, light=(0.7071, 0.7071, 0.0), interpolation='Closest', size_link=None):
    """C135: the checker as the height map, sampled twice (at the UV and
    at the Emboss Shift node's Shifted UV), both into the Emboss Bump
    node over a constant Base, into a diffuse BSDF. The nodes'
    interpolation decides the filter (the per-node filter wins)."""
    sc = demo_scene(st, with_texture=False)
    px = checker_image()
    sc.images['checker'] = ImageBuffer(name='checker', pixels=px)
    nodes = {
        'shift': {'id': 'shift', 'bl_idname': 'HALCYON_EmbossShiftNode', 'props': {},
                  'inputs': [_sk('Vector', 'VECTOR', [0, 0, 0]),
                             _sk('Light', 'VECTOR', list(light)),
                             _sk('Offset', 'VALUE', 1.0),
                             _sk('Texture Size', 'VALUE', float(px.shape[1]), size_link)],
                  'outputs': [{'name': 'Shifted UV', 'type': 'VECTOR'}]},
        'htex': {'id': 'htex', 'bl_idname': 'ShaderNodeTexImage',
                 'props': {'image': 'checker', 'interpolation': interpolation},
                 'inputs': [_sk('Vector', 'VECTOR', [0, 0, 0])],
                 'outputs': [{'name': 'Color', 'type': 'RGBA'}, {'name': 'Alpha', 'type': 'VALUE'}]},
        'htex2': {'id': 'htex2', 'bl_idname': 'ShaderNodeTexImage',
                  'props': {'image': 'checker', 'interpolation': interpolation},
                  'inputs': [_sk('Vector', 'VECTOR', [0, 0, 0], ['shift', 0])],
                  'outputs': [{'name': 'Color', 'type': 'RGBA'}, {'name': 'Alpha', 'type': 'VALUE'}]},
        'emb': {'id': 'emb', 'bl_idname': 'HALCYON_EmbossBumpNode', 'props': {},
                'inputs': [_sk('Height', 'VALUE', 0.5, ['htex', 0]),
                           _sk('Height Shifted', 'VALUE', 0.5, ['htex2', 0]),
                           _sk('Base', 'RGBA', [0.8, 0.7, 0.6, 1.0])],
                'outputs': [{'name': 'Factor', 'type': 'VALUE'}, {'name': 'Color', 'type': 'RGBA'}]},
    }
    if size_link is not None:
        nodes['val'] = {'id': 'val', 'bl_idname': 'ShaderNodeValue', 'props': {'value': float(px.shape[1])},
                        'inputs': [], 'outputs': [{'name': 'Value', 'type': 'VALUE'}]}
    g = _matb_out(nodes, 'emb')
    g['nodes']['bsdf']['inputs'][0]['link'] = ['emb', 1]
    sc.materials[1].graph = g
    return sc


def _sc_roughness_node(st, roughness=180.0, animate=False, frame=None):
    """C099: Roughness (Imagine) into the master shader's Normal (PHONG,
    glossiness 48) on the ball."""
    sc = demo_scene(st, with_texture=False)
    ins = _matb_master_ins() + [_sk('Normal', 'VECTOR', [0, 0, 0], ['rough', 0])]
    sc.materials[1].graph = {'output': 'out', 'nodes': {
        'rough': {'id': 'rough', 'bl_idname': 'HALCYON_ImagineRoughnessNode',
                  'props': {'animate': bool(animate)},
                  'inputs': [_sk('Normal', 'VECTOR', [0, 0, 0]),
                             _sk('Roughness', 'VALUE', float(roughness))],
                  'outputs': [{'name': 'Normal', 'type': 'VECTOR'}]},
        'hal': {'id': 'hal', 'bl_idname': 'HALCYON_ShaderNode',
                'props': {'model': 'PHONG', 'toon_steps': 2},
                'inputs': ins, 'outputs': [{'name': 'Surface', 'type': 'SHADER'}]},
        'out': {'id': 'out', 'bl_idname': 'ShaderNodeOutputMaterial', 'props': {},
                'inputs': [_sk('Surface', 'SHADER', None, ['hal', 0]),
                           _sk('Displacement', 'VECTOR', [0, 0, 0])],
                'outputs': []}}}
    if frame is not None:
        sc.frame = int(frame)
    return sc


def _sc_envchrome_node(st, **props):
    """C123: Env Chrome (Maya's defaults) into the master shader's Matcap
    (Add, blend 1.0) on the PHONG ball -- Maya's Reflected Color slot."""
    sc = demo_scene(st, with_texture=False)
    ins = _matb_master_ins() + [
        _sk('Matcap', 'RGBA', [0, 0, 0, 1], ['env', 0]),
        _sk('Matcap Blend', 'VALUE', 1.0)]
    sc.materials[1].graph = {'output': 'out', 'nodes': {
        'env': {'id': 'env', 'bl_idname': 'HALCYON_EnvChromeNode', 'props': dict(props),
                'inputs': [_sk('Normal', 'VECTOR', [0, 0, 0]),
                           _sk('Sky Color', 'RGBA', [0.55, 0.62, 0.78, 1.0]),
                           _sk('Zenith Color', 'RGBA', [0.15, 0.22, 0.48, 1.0]),
                           _sk('Light Color', 'RGBA', [1.0, 1.0, 1.0, 1.0]),
                           _sk('Floor Color', 'RGBA', [0.18, 0.18, 0.18, 1.0]),
                           _sk('Horizon Color', 'RGBA', [0.5, 0.5, 0.5, 1.0]),
                           _sk('Grid Color', 'RGBA', [0.04, 0.04, 0.04, 1.0])],
                'outputs': [{'name': 'Color', 'type': 'RGBA'}]},
        'hal': {'id': 'hal', 'bl_idname': 'HALCYON_ShaderNode',
                'props': {'model': 'PHONG', 'toon_steps': 2, 'matcap_mode': 'ADD'},
                'inputs': ins, 'outputs': [{'name': 'Surface', 'type': 'SHADER'}]},
        'out': {'id': 'out', 'bl_idname': 'ShaderNodeOutputMaterial', 'props': {},
                'inputs': [_sk('Surface', 'SHADER', None, ['hal', 0]),
                           _sk('Displacement', 'VECTOR', [0, 0, 0])],
                'outputs': []}}}
    return sc


def _sc_gradient_world(st):
    """R251 LIGHT-A2 (F008 / F009): the demo scene under a GRADIENT
    world whose horizon and zenith differ strongly, the camera pitched
    UP (the demo camera looks down: every one of its rays falls below
    the horizon, where a GRADIENT world is one colour), so a per-pixel
    backdrop target and an elevation-fogged sky both show."""
    from .scenebuild import look_at_matrix
    sc = demo_scene(st, with_texture=False)
    if getattr(sc, 'world', None) is not None:
        sc.world.mode = 'GRADIENT'
        sc.world.horizon = (0.8, 0.5, 0.3)
        sc.world.zenith = (0.2, 0.3, 0.7)
    sc.camera.matrix_world = look_at_matrix((5.2, -6.4, 1.0),
                                            (0.0, -0.2, 3.4))
    return sc


def _sc_fog_materials_plain(st):
    """R251 LIGHT-A2 (F006): _sc_fog_materials with every fog socket at
    its default -- the collage's OFF tile (the same materials, no dial)."""
    sc = _sc_fog_materials(st)
    for m in sc.materials:
        g = getattr(m, 'graph', None) or {}
        for node in (g.get('nodes') or {}).values():
            for ins in node.get('inputs') or ():
                if ins.get('name') in ('Fog Burn-Through', 'Fog Bias',
                                       'Fog Bank'):
                    ins['default'] = 0.0
    return sc


def _sc_fog_materials(st):
    """R251 LIGHT-A2 (F006): the demo scene with the ball on the master
    shader at Fog Burn-Through 1 (never fogged) and the box at Fog
    Bias 0.5 on Fog Bank 1; the floor keeps the scene's curve."""
    sc = _sc_pov_finish(st, **{'Fog Burn-Through': 1.0})
    ins = [
        _sk('Diffuse Color', 'RGBA', [0.3, 0.55, 0.85, 1.0]),
        _sk('Vertex Color', 'RGBA', [1, 1, 1, 1]),
        _sk('Vertex Color Mix', 'VALUE', 0.0),
        _sk('Diffuse Level', 'VALUE', 1.0),
        _sk('Specular Color', 'RGBA', [1, 1, 1, 1]),
        _sk('Specular Level', 'VALUE', 0.5),
        _sk('Glossiness', 'VALUE', 32.0),
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
        _sk('Fog Bias', 'VALUE', 0.5),
        _sk('Fog Bank', 'VALUE', 1.0),
    ]
    sc.materials[2].graph = _one_bsdf_graph('HALCYON_ShaderNode', ins,
                                            {'model': 'PHONG',
                                             'toon_steps': 2})
    return sc


SCENES = {
    # R251 material pack, wave 2 (MAT-B): the period node scenes
    'combiner_node': _sc_combiner_node,
    'combiner_nv2a_node': _sc_combiner_nv2a_node,
    'srbump_node': _sc_srbump_node,
    'emboss_node': _sc_emboss_node,
    'roughness_node': _sc_roughness_node,
    'envchrome_node': _sc_envchrome_node,
    'demo_plain': _sc_demo,      # the untextured demo (the node rows' OFF tile)
    # R251 (LIGHT-B2): the POV finish scenes (the two plain ones are the
    # collage's OFF tiles: the same master material with the socket off)
    'pov_finish_plain': _sc_pov_finish_plain,
    'pov_finish_plain_red': _sc_pov_finish_plain_red,
    'pov_finish_brilliance': _sc_pov_finish_brilliance,
    'pov_finish_crand': _sc_pov_finish_crand,
    'pov_finish_crand_f3': _sc_pov_finish_crand_f3,
    'pov_finish_metallic': _sc_pov_finish_metallic,
    'demo': _sc_demo, 'textured': _sc_textured, 'mirror': _sc_mirror,
    # R251 shadow pack
    'planar_half': _sc_planar_half, 'planar_raised': _sc_planar_raised,
    'modvol': _sc_modvol, 'modvol_exclude': _sc_modvol_exclude,
    'ds_shadow': _sc_ds_shadow,
    'shafts': _sc_shafts,
    'matcap_image': _sc_matcap_image,
    'glass': _sc_glass, 'cookie_spot': _sc_cookie_spot,
    # R251 (transparency pack): the law-probe pane and the DS shell
    'pane': _sc_pane, 'ds_shell': _sc_ds_shell,
    # R251 (transparency pack, wave 2)
    'clip_blend': _sc_clip_blend, 'thin_wall': _sc_thin_wall,
    'imagine_fog': _sc_imagine_fog, 'zoffs_decal': _sc_zoffs_decal,
    'zinvert_shell': _sc_zinvert_shell, 'env_hole': _sc_env_hole,
    'clip_hard': _sc_clip_hard, 'thin_wall_alpha': _sc_thin_wall_alpha,
    # the plain (untextured) demo scene under a key tools/r251_collage.py
    # does not intercept ('demo' there is the TEXTURED demo): the OFF tile
    # of zoffs_decal / env_hole / imagine_fog
    'demo_plain': _sc_demo,
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
    'geometry_aux_node': _sc_geometry_aux_node,
    'light_linked': _sc_light_linked,
    'ortho': _sc_ortho,
    'pitched': _sc_pitched,
    'wide_lens': _sc_wide_lens,
    # R251 (RAST-B)
    'hot_edge': _sc_hot_edge, 'elite_ship': _sc_elite_ship,
    'beam_vectors': _sc_beam_vectors,
    # R251 LIGHT-B1 (F011)
    'facing_quad': _sc_facing_quad,
    'only_shadow': _sc_only_shadow,
    'spot_pov': _sc_spot_pov, 'spot_gl': _sc_spot_gl,
    'spot_gx_ring': _sc_spot_gx_ring,
    'decay_pov': _sc_decay_pov, 'decay_gl': _sc_decay_gl,
    'decay_gx': _sc_decay_gx,
    'screen_spot': _sc_screen_spot,
    # R251 raster pack (RAST-A1)
    'near_wall': _sc_near_wall,
    'depth_fight_pair': _sc_depth_fight_pair,
    'ordering_table_stack': _sc_ordering_table_stack,
    # R251 raster pack (RAST-A2)
    'quantized_ship': _sc_quantized_ship,
    'backdrop_depth': _sc_backdrop_depth,
    'coverage_edge': _sc_coverage_edge,
    # R251 texture pack (TEX-1, TEX-2)
    'textured_white': _sc_textured_white,
    'textured_blocks': _sc_textured_blocks,
    'textured_mag': _sc_textured_mag,
    'textured_extend': _sc_textured_extend,
    'textured_keyed': _sc_textured_keyed,
    'textured_keyed_alpha': _sc_textured_keyed_alpha,
    # R251 sky-camera (SKY)
    'doom_sky': _sc_doom_sky, 'lw_sky': _sc_lw_sky, 'mode7': _sc_mode7,
    # R251 LIGHT-A2 (F006 / F008 / F009)
    'gradient_world': _sc_gradient_world,
    'fog_materials': _sc_fog_materials,
    'fog_materials_plain': _sc_fog_materials_plain,
    # R251 post-signal (SIG-3)
    'matte_glow': _sc_matte_glow,
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
    # R251 (LIGHT-B2): the console light units -- the VERTEX rows route
    # every material to the corner road (the machines' own rate), the
    # per-pixel rows exercise the GLSL twins
    ('model GX_LIGHT', {'force_model': 'GX_LIGHT',
                        'shading_rate': 'VERTEX'}, 'demo'),
    ('model GX_LIGHT (per pixel)', {'force_model': 'GX_LIGHT'}, 'demo'),
    ('model SEGA_MODEL2', {'force_model': 'SEGA_MODEL2'}, 'demo'),
    ('model SEGA_MODEL3', {'force_model': 'SEGA_MODEL3'}, 'demo'),
    ('model DS_FIXED', {'force_model': 'DS_FIXED', 'max_lights': 4,
                        'shading_rate': 'VERTEX'}, 'demo'),
    ('model DS_FIXED (per pixel)', {'force_model': 'DS_FIXED',
                                    'max_lights': 4}, 'demo'),
    # R251 material pack (MAT-A): the period combiners -- each a shading
    # RATE plus the machine's combine, on the textured scene where the
    # texel x light arithmetic shows; the two Sega / DS rows are the
    # combine halves of the lighting pack's items
    ('model FLAT_GL_LAST', {'force_model': 'FLAT_GL_LAST'}, 'demo'),
    ('model FLAT_D3D_FIRST', {'force_model': 'FLAT_D3D_FIRST'}, 'demo'),
    ('model PS1_MODULATE', {'force_model': 'PS1_MODULATE',
                            'shading_rate': 'VERTEX'}, 'textured'),
    ('model SATURN_ADD', {'force_model': 'SATURN_ADD',
                          'shading_rate': 'FACE'}, 'textured'),
    ('model N64_COMBINE', {'force_model': 'N64_COMBINE',
                           'shading_rate': 'VERTEX'}, 'textured'),
    ('model DS_FIXED (textured)', {'force_model': 'DS_FIXED',
                                   'shading_rate': 'VERTEX'}, 'textured'),
    ('model S22_MODULATE', {'force_model': 'S22_MODULATE',
                            'shading_rate': 'FACE'}, 'textured'),
    ('model PS2_HIGHLIGHT', {'force_model': 'PS2_HIGHLIGHT',
                             'shading_rate': 'VERTEX'}, 'textured'),
    ('model D3D_SEPARATE_SPEC', {'force_model': 'D3D_SEPARATE_SPEC',
                                 'shading_rate': 'VERTEX'}, 'textured'),
    ('model PCX_INTENSITY', {'force_model': 'PCX_INTENSITY',
                             'shading_rate': 'VERTEX'}, 'textured'),
    ('model DS_TOON', {'force_model': 'DS_TOON',
                       'shading_rate': 'VERTEX'}, 'textured'),
    ('model DS_HIGHLIGHT', {'force_model': 'DS_HIGHLIGHT',
                            'shading_rate': 'VERTEX'}, 'textured'),
    ('model SEGA_MODEL2 (textured)', {'force_model': 'SEGA_MODEL2'},
     'textured'),
    ('model SEGA_MODEL3 (textured)', {'force_model': 'SEGA_MODEL3'},
     'textured'),
    ('model MEGA_DRIVE_SH', {'force_model': 'MEGA_DRIVE_SH'}, 'textured'),
    ('model SUPERFX_PLOT', {'force_model': 'SUPERFX_PLOT',
                            'palette_mode': 'EGA16', 'color_depth': '4',
                            'dither': 'NONE'}, 'demo'),
    # R251 (LIGHT-B2): POV-Ray's finish dials on the master shader
    ('POV brilliance 2 (master shader)', {}, 'pov_finish_brilliance'),
    ('POV crand 0.3 (master shader)', {}, 'pov_finish_crand'),
    ('POV crand per frame', {'crand_per_frame': True},
     'pov_finish_crand_f3'),
    ('POV metallic highlight (master shader)', {}, 'pov_finish_metallic'),
    # ------------------------------------------------------- shading rates
    ('scene rate VERTEX (Gouraud)', {'shading_rate': 'VERTEX'}, 'demo'),
    ('scene rate FACE (flat)', {'shading_rate': 'FACE'}, 'demo'),
    # R251 C119 (MAT-B): the REYES micropolygon grid at 4 px of area
    ('REYES shading rate 4', {'shading_rate_area': 4.0}, 'textured'),
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
    ('light linking (exclude + only)', {}, 'light_linked'),
    # ------------------------------------------------ R251 LIGHT-B1
    ('fixed camera-axis viewer (GL 1.1 / Sega Model / DS)',
     {'specular_viewer': 'AXIS', 'default_model': 'PHONG',
      'normal_source': 'FACE'}, 'facing_quad'),
    ('only shadow lamp (Blender LA_ONLYSHADOW)', {}, 'only_shadow'),
    ('spot law POV (hotspot + Hermite)', {}, 'spot_pov'),
    ('spot law OpenGL 1.1 (exponent 8)', {}, 'spot_gl'),
    ('spot law GX ring (GameCube)', {}, 'spot_gx_ring'),
    ('POV fade_distance / fade_power 2',
     {'light_falloff_default': 'POV_FADE_SQUARE'}, 'decay_pov'),
    ('OpenGL 1.1 three-term attenuation',
     {'light_falloff_default': 'GL_3TERM'}, 'decay_gl'),
    ('GX distance attenuation STEEP (GameCube)',
     {'light_falloff_default': 'GX_STEEP'}, 'decay_gx'),
    ('Model 3 screen-space spotlight', {}, 'screen_spot'),
    ('Model 3 spotlight fog lobe',
     {'fog': True, 'fog_mode': 'LINEAR', 'fog_start': 3.0, 'fog_end': 12.0,
      'fog_spot': 1.0}, 'screen_spot'),
    # ------------------------------------------------------------ shadows
    ('shadows off', {'shadows': False}, 'demo'),
    ('ray shadows', {'shadow_default': 'RAY'}, 'demo'),
    ('per-light shadow modes', {'shadow_default': 'PER_LIGHT'}, 'demo'),
    ('soft shadows (radius + 8 taps)', {'shadow_samples': 8}, 'soft'),
    ('soft RAY shadows', {'shadow_default': 'RAY',
                          'shadow_samples': 4}, 'soft'),
    # R251 shadow pack
    ('midpoint shadow map (Woo / Classic-Halfway)',
     {'shadow_map_depth': 'MIDPOINT'}, 'demo'),
    ('planar shadows (Blinn / Model 1)', {'shadow_default': 'PLANAR'},
     'demo'),
    ('planar shadows, half blob', {'shadow_default': 'PLANAR'},
     'planar_half'),
    ('planar shadows, raised floor', {'shadow_default': 'PLANAR',
                                      'planar_plane_z': 0.4},
     'planar_raised'),
    ('Dreamcast modifier volume (inside)', {}, 'modvol'),
    ('Dreamcast modifier volume (outside)', {'modvol_scale': 64},
     'modvol_exclude'),
    ('DS shadow polygon (ID self-exclusion)', {'force_model': 'CONSTANT'},
     'ds_shadow'),
    ('ambient occlusion', {'ambient_occlusion': True,
                           'ao_samples': 4}, 'demo'),
    # -------------------------------------------------------- ray tracing
    ('traced reflection', {'raytrace': True, 'ray_depth': 1}, 'mirror'),
    ('traced recursion depth 2', {'raytrace': True, 'ray_depth': 2},
     'mirror'),
    ('env reflection (no rays)', {'env_reflection': True}, 'mirror'),
    ('rich world (Bryce) reflection', {'raytrace': True, 'ray_depth': 1},
     'bryce'),
    # R251 sky-camera (SKY): pixel-addressed backdrops, bitwise twins
    ('sky CYLINDER (Doom)', {}, 'doom_sky'),
    ('sky LW_GRADIENT (LightWave)', {}, 'lw_sky'),
    ('ground MODE7 (SNES)', {}, 'mode7'),
    # ------------------------------------------------- raw node graphs
    ('node Metallic BSDF (raw graph)', {}, 'metallic_node'),
    ('node Specular BSDF (raw graph)', {}, 'specular_node'),
    ('node Fractal Noise (ridged)', {}, 'noise_node'),
    ('node Cells + Hardware Palette', {}, 'cells_palette_node'),
    ('node Pixelate + Scroll + Scanlines', {}, 'retro_chain_node'),
    # R251 material pack, wave 2 (MAT-B): the period nodes
    ('node combiner TEV', {}, 'combiner_node'),
    ('node combiner NV2A', {}, 'combiner_nv2a_node'),
    ('node SR bump (Dreamcast)', {'tex_filter': 'NEAREST'}, 'srbump_node'),
    ('node emboss bump (DX6)', {}, 'emboss_node'),
    ('node Imagine roughness', {'seed': 3}, 'roughness_node'),
    ('node env chrome (Alias/Maya)', {}, 'envchrome_node'),
    ('node TV Static + Ordered Dither', {}, 'static_dither_node'),
    ('node Flipbook + UV Wave', {}, 'flipbook_wave_node'),
    ('node Halftone + Threshold + Quantize', {}, 'halftone_chain_node'),
    ('node Geometry (Tangent x Incoming)', {}, 'geometry_node'),
    ('node Geometry TrueN x island random', {},
     'geometry_aux_node'),
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
    # R251 texture pack (TEX-1, wave 1)
    ('texel format RGB565 (Glide)', {'tex_format': 'RGB565'}, 'textured'),
    ('texel format ARGB4444 (PCX)', {'tex_format': 'ARGB4444'}, 'textured'),
    ('texel format NCC YIQ422 (3dfx)', {'tex_format': 'YIQ422'}, 'textured'),
    ('texel format I4 Model 2 (0xF hole)', {'tex_format': 'I4_MODEL2'}, 'textured_white'),
    ('texel format A3I5 (DS)', {'tex_format': 'A3I5'}, 'textured'),
    ('N64 TMEM RGBA16 budget', {'tex_tmem_format': 'RGBA16'}, 'textured'),
    ('N64 TMEM CI4 budget', {'tex_tmem_format': 'CI4'}, 'textured'),
    ('DXT1 blocks (S3TC)', {'tex_compress': 'DXT1'}, 'textured_blocks'),
    ('DXT1 NV2A 16-bit decode (Xbox)', {'tex_compress': 'DXT1_NV2A'}, 'textured_blocks'),
    ('CMPR blocks (GameCube)', {'tex_compress': 'CMPR_GC'}, 'textured_blocks'),
    ('VQ codebook (Dreamcast)', {'tex_compress': 'VQ_DC'}, 'textured_blocks'),
    ('bilinear 4-bit fraction (Voodoo1)', {'tex_filter': 'BILINEAR',
                                          'tex_frac_bits': 'BITS_4'}, 'textured_mag'),
    ('texture POV normalised distance', {'tex_filter': 'POV_NORMDIST'}, 'textured'),
    # R251 texture pack (TEX-2, wave 2)
    ('GL_CLAMP border seam (OpenGL 1.1)', {'tex_filter': 'BILINEAR',
                                          'tex_clamp_mode': 'GL_CLAMP'}, 'textured_extend'),
    ('chroma key after filtering (Glide)', {'tex_filter': 'BILINEAR', 'tex_frac_bits': 'BITS_4',
                                           'tex_colorkey': True}, 'textured_keyed'),
    ('N64 3-point mip blend', {'tex_filter': 'N64_3POINT', 'tex_mipmap': True,
                               'tex_mip_select': 'BLEND'}, 'textured'),
    ('nearest mip level (GL 1.1)', {'tex_filter': 'BILINEAR', 'tex_mipmap': True,
                                    'tex_mip_select': 'NEAREST_LEVEL'}, 'textured'),
    ('Voodoo1 LOD dither', {'tex_filter': 'BILINEAR', 'tex_mipmap': True,
                            'tex_mip_select': 'DITHER_VOODOO'}, 'textured'),
    ('per-polygon mip level (Riva 128)', {'tex_filter': 'BILINEAR', 'tex_mipmap': True,
                                          'tex_mip_select': 'NEAREST_LEVEL',
                                          'tex_lod_source': 'TRIANGLE'}, 'textured'),
    ('PS2 LOD from Q, nearest level', {'tex_filter': 'BILINEAR', 'tex_mipmap': True,
                                       'tex_mip_select': 'NEAREST_LEVEL',
                                       'tex_lod_source': 'GS_Q', 'tex_lod_k': -2.0}, 'textured'),
    ('PS2 LOD from Q, trilinear', {'tex_filter': 'TRILINEAR', 'tex_mipmap': True,
                                   'tex_lod_source': 'GS_Q', 'tex_lod_k': -2.0}, 'textured'),
    ('N64 sharpen (G_TD_SHARPEN)', {'tex_filter': 'N64_3POINT', 'tex_mipmap': True,
                                    'tex_lod_sharpen': True}, 'textured_mag'),
    ('texture Summed Area (3DS/Max)', {'tex_filter': 'SUMMED_AREA'}, 'textured'),
    ('Summed Area footprint x4', {'tex_filter': 'SUMMED_AREA', 'tex_mip_bias': 2.0}, 'textured'),
    # ------------------------------------------------------- transparency
    ('transparency NONE', {'transparency': 'NONE'}, 'glass'),
    ('screen-door stipple', {'transparency': 'STIPPLE'}, 'glass'),
    ('sorted glass layers', {'transparency': 'SORTED'}, 'glass'),
    ('A-buffer glass', {'transparency': 'ABUFFER'}, 'glass'),
    ('ray-traced glass layers', {'transparency': 'SORTED',
                                 'raytrace': True, 'ray_depth': 1},
     'glass'),
    ('binary alpha', {'transparency': 'SORTED', 'alpha_bits': 1}, 'glass'),
    # R251 (transparency pack): the blend unit's fixed equations, the
    # framebuffer format at the blend, the DS composite rules
    ('blend PS1_ADD (PlayStation)', {'transparency': 'SORTED',
                                     'blend_equation': 'PS1_ADD'}, 'glass'),
    ('blend SATURN_HALF (Saturn VDP1)', {'transparency': 'ABUFFER',
                                         'blend_equation': 'SATURN_HALF'},
     'glass'),
    ('blend THREEDO_XOR (3DO PIXC)', {'transparency': 'SORTED',
                                      'blend_equation': 'THREEDO_XOR'},
     'glass'),
    ('blend SNES_ADD_HALF (SNES colour math)',
     {'transparency': 'ABUFFER', 'blend_equation': 'SNES_ADD_HALF'},
     'glass'),
    ('blend GBA (BLDALPHA sixteenths)', {'transparency': 'SORTED',
                                         'blend_equation': 'GBA'}, 'glass'),
    ('stipple COLUMNS (Mega Drive mesh)', {'transparency': 'STIPPLE',
                                           'stipple_pattern': 'COLUMNS'},
     'glass'),
    ('blend DS (Nintendo DS)', {'transparency': 'ABUFFER',
                                'blend_equation': 'DS'}, 'glass'),
    ('DS auto-sort + depth write', {'transparency': 'ABUFFER',
                                    'translucent_order': 'Y_SORT',
                                    'translucent_depth_write': True},
     'glass'),
    ('DS same-ID once', {'transparency': 'ABUFFER',
                         'blend_equation': 'DS'}, 'ds_shell'),
    ('framebuffer PS2_CT16 (GS 16-bit)', {'transparency': 'SORTED',
                                          'framebuffer': 'PS2_CT16'},
     'glass'),
    ('framebuffer GC_RGBA6 (Flipper EFB)', {'transparency': 'ABUFFER',
                                            'framebuffer': 'GC_RGBA6'},
     'glass'),
    ('framebuffer VOODOO_565_4X4', {'transparency': 'ABUFFER',
                                    'framebuffer': 'VOODOO_565_4X4'},
     'glass'),
    ('framebuffer VOODOO_565_2X2, no subtract',
     {'transparency': 'SORTED', 'framebuffer': 'VOODOO_565_2X2',
      'fb_dither_subtract': False}, 'glass'),
    ('framebuffer on an opaque frame', {'transparency': 'NONE',
                                        'framebuffer': 'PS2_CT16'}, 'demo'),
    ('framebuffer PS2_CT16, DTHE off (truncation)',
     {'transparency': 'SORTED', 'framebuffer': 'PS2_CT16',
      'fb_dither': False}, 'glass'),
    ('blend FUZZ (Doom Spectre)', {'transparency': 'ABUFFER',
                                   'blend_equation': 'FUZZ'}, 'glass'),
    # R251 transparency pack, wave 2 (TRANS-2)
    ('stipple N64_NOISE (RDP random alpha compare)',
     {'transparency': 'STIPPLE', 'stipple_pattern': 'N64_NOISE'}, 'glass'),
    ('alpha CLIP_BLEND (PS2 two-pass)', {'transparency': 'ABUFFER'},
     'clip_blend'),
    ('z_offset decal (Blender 2.4x Zoffs)', {'transparency': 'ABUFFER'},
     'zoffs_decal'),
    ('z_invert shell (Blender 2.4x ZInvert)', {'transparency': 'ABUFFER'},
     'zinvert_shell'),
    ('env hole (Blender 2.4x Env)', {}, 'env_hole'),
    ('blend THIN_WALL (Max Thin Wall Refraction)', {'transparency': 'ABUFFER'},
     'thin_wall'),
    ('blend IMAGINE_FOG (Imagine Fog Length)', {'transparency': 'ABUFFER'},
     'imagine_fog'),
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
    # R251 lighting F000-F003 / F007 / F022: the fog twin's rows
    ('fog EXP', {'fog': True, 'fog_mode': 'EXP', 'fog_density': 0.08},
     'demo'),
    ('fog bands 4', {'fog': True, 'fog_mode': 'LINEAR', 'fog_start': 3.0,
                     'fog_end': 12.0, 'fog_bands': 4}, 'demo'),
    ('fog hyperbolic 1/z (PS1 GTE)', {'fog': True, 'fog_mode': 'GTE_1Z',
                                      'fog_start': 3.0, 'fog_end': 12.0,
                                      'shading_rate': 'VERTEX'}, 'demo'),
    ('fog hyperbolic 1/z (PS1 GTE, per pixel)',
     {'fog': True, 'fog_mode': 'GTE_1Z', 'fog_start': 3.0,
      'fog_end': 12.0}, 'demo'),
    ('Voodoo 64-entry fog table', {'fog': True, 'fog_mode': 'EXP',
                                   'fog_density': 0.06,
                                   'fog_table': 'VOODOO64'}, 'demo'),
    ('Voodoo2 fog dither', {'fog': True, 'fog_mode': 'EXP',
                            'fog_density': 0.06, 'fog_table': 'VOODOO64',
                            'fog_dither': True}, 'demo'),
    ('PowerVR2 128-entry fog table', {'fog': True, 'fog_mode': 'EXP',
                                      'fog_density': 0.04, 'fog_end': 60.0,
                                      'fog_table': 'PVR128'}, 'demo'),
    ('DS 32-entry fog table', {'fog': True, 'fog_mode': 'LINEAR',
                               'fog_start': 3.0, 'fog_end': 12.0,
                               'fog_table': 'DS32'}, 'demo'),
    ('Direct3D z-fog', {'fog': True, 'fog_mode': 'LINEAR',
                        'fog_depth': 'Z', 'fog_start': 0.9,
                        'fog_end': 1.0}, 'demo'),
    ('Direct3D z-fog (orthographic)', {'fog': True, 'fog_mode': 'LINEAR',
                                       'fog_depth': 'Z', 'fog_start': 0.03,
                                       'fog_end': 0.07}, 'ortho'),
    # R251 lighting LIGHT-A2: F004 / F005 / F006 / F008 / F009 / F010
    ('GameCube fog range adjust', {'fog': True, 'fog_mode': 'TABLE16',
                                   'fog_start': 3.0, 'fog_end': 12.0,
                                   'fog_range_adjust': True}, 'demo'),
    ('per-polygon fog (Namco System 21)', {'fog': True, 'fog_mode': 'LINEAR',
                                          'fog_start': 3.0, 'fog_end': 12.0,
                                          'fog_face': True, 'fog_bands': 16},
     'demo'),
    ('per-material fog burn/bias (Model 3 / System 22)',
     {'fog': True, 'fog_mode': 'LINEAR', 'fog_start': 3.0, 'fog_end': 12.0},
     'fog_materials'),
    ('fog bank 1 (System 22)', {'fog': True, 'fog_mode': 'LINEAR',
                                'fog_start': 3.0, 'fog_end': 12.0,
                                'fog_bank1_start': 1.0, 'fog_bank1_end': 5.0},
     'fog_materials'),
    ('fog ambient 0.5 (Model 3 fogAmbient)', {'fog': True, 'fog_mode': 'LINEAR',
                                             'fog_start': 3.0, 'fog_end': 12.0,
                                             'fog_ambient': 0.5}, 'demo'),
    ('LightWave backdrop fog', {'fog': True, 'fog_mode': 'LINEAR',
                                'fog_start': 2.0, 'fog_end': 9.0,
                                'fog_color_source': 'BACKDROP'},
     'gradient_world'),
    ('POV ground fog (atan integral)', {'fog': True, 'fog_mode': 'GROUND',
                                        'fog_density': 0.15,
                                        'fog_ground_offset': -0.5,
                                        'fog_ground_alt': 1.5,
                                        'fog_color': (0.75, 0.75, 0.8)},
     'demo'),
    ('POV ground fog, the sky fogged by elevation',
     {'fog': True, 'fog_mode': 'GROUND', 'fog_density': 0.15,
      'fog_ground_offset': -0.5, 'fog_ground_alt': 1.5,
      'fog_color': (0.75, 0.75, 0.8)}, 'gradient_world'),
    ('POV turbulent fog', {'fog': True, 'fog_mode': 'EXP', 'fog_density': 0.1,
                           'fog_turbulence': 1.0}, 'demo'),
    ('POV turbulent fog, depth 1.0', {'fog': True, 'fog_mode': 'EXP',
                                      'fog_density': 0.1, 'fog_turbulence': 1.0,
                                      'fog_turb_depth': 1.0}, 'demo'),
    # -------------------------------------------------------------- depth
    ("Painter's sort", {'depth_sort': 'PAINTERS'}, 'demo'),
    ("Painter's sort, nearest-vertex key",
     {'depth_sort': 'PAINTERS', 'painters_key': 'NEAREST'}, 'demo'),
    ('16-bit z-buffer', {'depth_precision': 16}, 'demo'),
    ('vertex snapping (PS1)', {'vertex_snap': True,
                               'vertex_snap_grid': 1.0}, 'demo'),
    ('fixed-point subpixel', {'subpixel_precision': 'FIXED_4'}, 'demo'),
    ('integer subpixel', {'subpixel_precision': 'INTEGER'}, 'demo'),
    ('backface culling', {'backface_cull': True}, 'demo'),
    ('orthographic camera', {}, 'ortho'),
    # ------------------------------------------------- camera (R251)
    ('camera y-shear (Heretic / Build)', {'camera_yshear': True},
     'pitched'),
    ('camera pano parts x3 (Blender 2.4 Xparts)', {'pano_parts': 3},
     'demo'),
    ('stereo parallax layers (Virtual Boy)',
     {'stereo_mode': 'SBS', 'stereo_parallax_layers': True,
      'stereo_parallax_max': 8, 'stereo_eye_distance': 0.8,
      'stereo_convergence': 4.0}, 'demo'),
    ('camera lens DOF (accumulation buffer)',
     {'dof': True, 'dof_method': 'LENS_ACCUMULATE',
      'dof_lens_pattern': 'HALTON_DISC', 'dof_lens_samples': 4,
      'dof_focus': 2.0}, 'wide_lens'),
    ('camera lens DOF (Max spiral)',
     {'dof': True, 'dof_method': 'LENS_ACCUMULATE',
      'dof_lens_pattern': 'MAX_SPIRAL', 'dof_lens_samples': 4,
      'dof_focus': 2.0}, 'wide_lens'),
    # ------------------------------------- R251 raster pack (RAST-A1)
    ('integer pixel centres (Direct3D 9)',
     {'pixel_center': 'INTEGER_D3D'}, 'demo'),
    ('whole-triangle near rejection (PS2 VU1)',
     {'near_clip_mode': 'REJECT'}, 'near_wall'),
    ("Painter's ordering table (PS1)",
     {'depth_sort': 'PAINTERS', 'painters_key': 'ORDERING_TABLE'}, 'demo'),
    ("Painter's ordering table, far wall, 256 entries",
     {'depth_sort': 'PAINTERS', 'painters_key': 'ORDERING_TABLE',
      'ot_far': 8.0, 'ot_length': 256}, 'demo'),
    ("Painter's ordering table, stacked sheets, 256 entries",
     {'depth_sort': 'PAINTERS', 'painters_key': 'ORDERING_TABLE',
      'ot_length': 256}, 'ordering_table_stack'),
    ('N64 18-bit floating z-buffer',
     {'depth_encoding': 'N64_FLOAT18'}, 'demo'),
    ('N64 18-bit floating z-buffer, fighting sheet',
     {'depth_encoding': 'N64_FLOAT18'}, 'depth_fight_pair'),
    ('GameCube 14e2 compressed z',
     {'depth_encoding': 'GC_14E2'}, 'depth_fight_pair'),
    ('W-buffer, 16-bit fixed (Xbox)',
     {'depth_encoding': 'W_FIXED', 'depth_precision': 16},
     'depth_fight_pair'),
    ('Voodoo 16-bit floating W-buffer',
     {'depth_encoding': 'VOODOO_W16'}, 'depth_fight_pair'),
    # R251 raster pack (RAST-A2): C012 the console vertex formats (units
    # 8 so the crunch shows at 96x72; 'textured' so the UV grid moves
    # texels), the crunch rig
    ('vertex quantisation (PS1: 16-bit, 8-bit uv)',
     {'vertex_quantize': 'PS1', 'vertex_units': 8.0}, 'textured'),
    ('vertex quantisation (N64: 8-bit normals)',
     {'vertex_quantize': 'N64', 'vertex_units': 8.0}, 'demo'),
    ('vertex quantisation (PS1) on the crunch rig',
     {'vertex_quantize': 'PS1', 'vertex_units': 4.0}, 'quantized_ship'),
    # C038 the DS rear-plane depth bitmap (a World image, not a setting)
    ('rear-plane depth bitmap (DS)', {}, 'backdrop_depth'),
    # C001 N64 coverage AA + the VI's blend and divot (build sets
    # aa_samples 1, so the rows are non-vacuous)
    ('N64 coverage AA + VI filter', {'n64_coverage_aa': True}, 'demo'),
    ('N64 coverage AA, no divot',
     {'n64_coverage_aa': True, 'n64_divot': False}, 'demo'),
    ('N64 coverage AA on the coverage-edge rig',
     {'n64_coverage_aa': True}, 'coverage_edge'),
    # ----------------------------------------------------------------- AA
    ('supersample 4x', {'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4},
     'demo'),
    ('edge antialias (flicker filter)', {'aa_mode': 'EDGE'}, 'demo'),
    ('accumulation AA', {'aa_mode': 'ACCUMULATE', 'aa_samples': 4},
     'demo'),
    # R251 (RAST-A2): C127 REYES's jittered sample positions
    ('jittered sample positions (REYES)',
     {'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4,
      'aa_sample_pattern': 'JITTER'}, 'demo'),
    ('jittered sample position at 1 sample (a per-pixel offset)',
     {'aa_mode': 'NONE', 'aa_sample_pattern': 'JITTER'}, 'demo'),
    # R251 (RAST-B): C094 LightWave's clamp before the filter (the flat
    # ambient pushes samples past 1.0 so the row is non-vacuous), C122
    # Blender 2.41's gamma-2 sample blend
    ('LightWave limit dynamic range (clamp before AA)',
     {'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4, 'aa_clamp_samples': True,
      'global_ambient': (1.0, 1.0, 1.0), 'global_ambient_level': 3.0},
     'demo'),
    ('LightWave limit dynamic range on a hot edge (GAUSS, ss 2)',
     {'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4, 'aa_filter': 'GAUSS',
      'aa_clamp_samples': True}, 'hot_edge'),
    ('Blender 2.41 gamma-2 OSA blend',
     {'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4, 'aa_gamma_blend': True},
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
    # R251 (post-palette): the era colour roads
    ('VGA 256 + BAYER4 (GPU palette snap)',
     {'color_depth': '8', 'palette_mode': 'VGA256', 'dither': 'BAYER4'},
     'demo'),
    ('palette registers 3-bit (Atari ST)',
     {'color_depth': '4', 'palette_mode': 'ADAPTIVE', 'palette_size': 16,
      'palette_bits': 'BITS_3'}, 'demo'),
    ('Extra Half-Brite (Amiga)', {'palette_mode': 'EHB'}, 'demo'),
    ('CRY16 colour (Jaguar)', {'color_depth': 'CRY16'}, 'demo'),
    ('YJK colour (MSX2+)', {'color_depth': 'YJK'}, 'demo'),
    # R251 (post-palette, wave 2): attribute cells, the per-scanline
    # palette, Super Black (the 64 row is super_black_threshold's home),
    # Video Color Check (the PAL row is video_system's and
    # video_ire_limit's home)
    ('attribute cells ZX Spectrum',
     {'attribute_cells': 'ZX_SPECTRUM', 'dither': 'BAYER8'}, 'demo'),
    ('attribute cells C64 multicolour', {'attribute_cells': 'C64_MULTI'},
     'demo'),
    ('per-scanline palette (Spectrum 512)',
     {'scanline_palette': 'SPECTRUM_512'}, 'demo'),
    ('Super Black 15 (3D Studio / Max)',
     {'super_black': True, 'exposure': 0.1}, 'demo'),
    ('Super Black 64 (3D Studio / Max)',
     {'super_black': True, 'super_black_threshold': 64, 'exposure': 0.1},
     'demo'),
    ('video colour check FLAG_BLACK (Max)',
     {'video_color_check': 'FLAG_BLACK', 'saturation': 2.0}, 'demo'),
    ('video colour check SCALE_SAT (Max)',
     {'video_color_check': 'SCALE_SAT', 'saturation': 2.0}, 'demo'),
    ('video colour check PAL at 110 IRE (Max)',
     {'video_color_check': 'FLAG_BLACK', 'video_system': 'PAL',
      'video_ire_limit': 'IRE_110', 'saturation': 2.0}, 'demo'),
    ('1-bit + halftone', {'color_depth': '1', 'dither': 'HALFTONE'},
     'demo'),
    ('glow / bloom', {'glow': True, 'glow_threshold': 0.5,
                      'glow_intensity': 0.8}, 'demo'),
    ('star filter', {'star_filter': True, 'glow_threshold': 0.5,
                     'star_intensity': 0.8}, 'demo'),
    ('lens flare', {'lens_flare': True, 'flare_intensity': 0.7}, 'demo'),
    ('light shafts', {'shaft_threshold': 0.4, 'shaft_length': 0.4},
     'shafts'),
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
    # R251 post-signal (SIG-1): the machine's scan-out stages
    ('N64 VI dedither + gamma dither',
     {'color_depth': '15', 'dither': 'BAYER2', 'vi_dither_filter': True,
      'vi_gamma': 'GAMMA_DITHER'}, 'demo'),
    ('GameCube copy-filter deflicker', {'copy_filter': 'DEFLICKER'},
     'demo'),
    ('PS2 CRTC blend against BGCOLOR',
     {'crtc_blend': 'BG_COLOR', 'crtc_alpha': 160,
      'crtc_bg_color': (0.1, 0.0, 0.2)}, 'demo'),
    ('3dfx 22-bit scan-out filter',
     {'color_depth': '16', 'dither': 'BAYER4', 'video_filter': 'VOODOO1'},
     'demo'),
    ('3dfx Voodoo2 look-ahead filter',
     {'color_depth': '16', 'dither': 'BAYER4', 'video_filter': 'VOODOO2'},
     'demo'),
    ('3DO 2x interpolated, blue at 4 bits (fit_to keeps the sites)',
     {'output_scale': 'THREEDO_2X', 'color_depth': '15',
      'dither': 'BAYER2'}, 'demo'),
    ('GBA Mode 5 affine stretch',
     {'output_scale': 'GBA_MODE5', 'color_depth': '15'}, 'demo'),
    # R251 post-signal (SIG-2): chroma siting, the cable, the PAL
    # receiver, the tape
    ('DV 4:1:1 chroma, held', {'chroma_format': 'Y411'}, 'demo'),
    ('MPEG-1 4:2:0 chroma, linear',
     {'chroma_format': 'Y420_MPEG1', 'chroma_upsample': 'LINEAR'}, 'demo'),
    ('GameCube XFB 4:2:2', {'chroma_format': 'XFB_422'}, 'demo'),
    ('S-Video cable', {'signal': 'SVIDEO'}, 'demo'),
    ('RF modulator over composite',
     {'signal': 'RF', 'composite': True, 'rf_beat': 0.5, 'rf_snow': 0.05,
      'rf_ghost': 0.3}, 'demo'),
    ('PAL delay-line decoder + crawl',
     {'pal_decoder': 'DELAY_LINE', 'pal_crawl': 0.6}, 'demo'),
    ('Simple PAL, Hanover bars',
     {'pal_decoder': 'SIMPLE', 'pal_phase_error': 20.0}, 'demo'),
    ('VHS tape, third generation',
     {'tape': 'VHS', 'tape_generations': 3, 'tape_noise': 0.03}, 'demo'),
    # R251 post-signal (SIG-3): the codecs and the optical printer
    ('Tron matte glow (gel on the ball)',
     {'matte_glow': True, 'matte_glow_radius': 12.0}, 'matte_glow'),
    ('MPEG-1 intra blocks, Video CD',
     {'mpeg1': True, 'mpeg1_qscale': 12, 'mpeg1_gop': 0}, 'demo'),
    ('MPEG-1 B picture in a group of 15',
     {'mpeg1': True, 'mpeg1_qscale': 8, 'mpeg1_gop': 15}, 'demo'),
    ('Smacker 4x4 blocks over a 256 palette',
     {'smacker': True, 'smacker_quality': 0.5}, 'demo'),
    ('display transform', {'exposure': 1.3, 'gamma': 1.2,
                           'contrast': 0.2, 'saturation': 1.4,
                           'brightness': 0.05}, 'demo'),
    ('transparent film', {'film_transparent': True}, 'demo'),
    # ---------------------------------------------------------- wire, misc
    ('wireframe overlay ALL', {'render_wire': True}, 'demo'),
    ('wireframe CREASE (cel ink)', {'render_wire': True,
                                    'wire_mode': 'CREASE'}, 'demo'),
    # R251 (RAST-B): C063 Elite's rule (8.0: the demo camera is 7.6 from
    # the cube, 9.0 from the floor, 9.6 from the ball -- floor and ball
    # collapse to dots, the cube keeps its edges), C055 the vector beam
    # (a colour the 1-bit channels keep; the DVG row homes both dials)
    ('Elite wireframe rule (no depth test, dots past 8 units)',
     {'render_wire': True, 'wire_mode': 'ELITE', 'wire_dot_distance': 8.0},
     'demo'),
    ('Elite through-lines on a concave hull',
     {'render_wire': True, 'wire_mode': 'ELITE',
      'wire_color': (1.0, 1.0, 1.0)}, 'elite_ship'),
    ('vector monitor beam (AVG)',
     {'render_wire': True, 'wire_mode': 'BEAM',
      'wire_color': (1.0, 1.0, 1.0)}, 'beam_vectors'),
    ('vector monitor beam (DVG, fat spot)',
     {'render_wire': True, 'wire_mode': 'BEAM', 'beam_machine': 'DVG',
      'beam_sigma': 1.4, 'wire_color': (0.85, 1.0, 0.9)}, 'demo'),
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
