"""Build test scenes without Blender, so the renderer can be exercised headlessly."""

import numpy as np

from ..core.scene import Camera, ImageBuffer, Light, Material, MeshData, ObjectInfo, Scene, World
from ..core.settings import RenderSettings


def _mesh_concat(parts):
    verts, norms, uvs, cols, tris, mats, objs = [], [], [], [], [], [], []
    base = 0
    for v, n, uv, t, mi, oi in parts:
        verts.append(v)
        norms.append(n)
        uvs.append(uv)
        cols.append(np.ones((v.shape[0], 4), np.float32))
        tris.append(t + base)
        mats.append(np.full(t.shape[0], mi, np.int32))
        objs.append(np.full(t.shape[0], oi, np.int32))
        base += v.shape[0]
    V = np.concatenate(verts).astype(np.float32)
    T = np.concatenate(tris).astype(np.int32)
    e1 = V[T[:, 1]] - V[T[:, 0]]
    e2 = V[T[:, 2]] - V[T[:, 0]]
    fn = np.cross(e1, e2)
    ln = np.linalg.norm(fn, axis=1, keepdims=True)
    fn = fn / np.where(ln < 1e-12, 1.0, ln)
    m = MeshData()
    m.verts = V
    m.normals = np.concatenate(norms).astype(np.float32)
    m.uvs = np.concatenate(uvs).astype(np.float32)
    m.colors = np.concatenate(cols).astype(np.float32)
    m.tris = T
    m.mat_index = np.concatenate(mats).astype(np.int32)
    m.obj_index = np.concatenate(objs).astype(np.int32)
    m.face_normals = fn.astype(np.float32)
    m.smooth = np.zeros(T.shape[0], bool)
    return m


def cube(centre=(0, 0, 0), size=1.0, mat=0, obj=0):
    c = np.asarray(centre, np.float32)
    h = size * 0.5
    faces = [
        ((0, 0, 1), [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]),
        ((0, 0, -1), [(-1, 1, -1), (1, 1, -1), (1, -1, -1), (-1, -1, -1)]),
        ((1, 0, 0), [(1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1)]),
        ((-1, 0, 0), [(-1, -1, 1), (-1, 1, 1), (-1, 1, -1), (-1, -1, -1)]),
        ((0, 1, 0), [(-1, 1, 1), (1, 1, 1), (1, 1, -1), (-1, 1, -1)]),
        ((0, -1, 0), [(-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)]),
    ]
    V, N, UV, T = [], [], [], []
    base = 0
    for nrm, quad in faces:
        for p in quad:
            V.append(c + np.asarray(p, np.float32) * h)
            N.append(nrm)
        UV += [(0, 0), (1, 0), (1, 1), (0, 1)]
        T.append([base, base + 1, base + 2])
        T.append([base, base + 2, base + 3])
        base += 4
    return (np.array(V, np.float32), np.array(N, np.float32),
            np.array(UV, np.float32), np.array(T, np.int32), mat, obj)


def sphere(centre=(0, 0, 0), radius=1.0, segs=24, rings=16, mat=0, obj=0, smooth=True):
    c = np.asarray(centre, np.float32)
    V, N, UV, T = [], [], [], []
    for i in range(rings + 1):
        v = i / rings
        phi = v * np.pi
        for j in range(segs + 1):
            u = j / segs
            th = u * 2 * np.pi
            d = np.array([np.sin(phi) * np.cos(th), np.sin(phi) * np.sin(th),
                          np.cos(phi)], np.float32)
            V.append(c + d * radius)
            N.append(d)
            UV.append((u, 1.0 - v))
    for i in range(rings):
        for j in range(segs):
            a = i * (segs + 1) + j
            b = a + segs + 1
            T.append([a, b, a + 1])
            T.append([a + 1, b, b + 1])
    return (np.array(V, np.float32), np.array(N, np.float32),
            np.array(UV, np.float32), np.array(T, np.int32), mat, obj)


def plane(z=0.0, size=10.0, mat=0, obj=0):
    h = size * 0.5
    V = np.array([[-h, -h, z], [h, -h, z], [h, h, z], [-h, h, z]], np.float32)
    N = np.tile(np.array([[0, 0, 1.0]], np.float32), (4, 1))
    UV = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32) * 4.0
    T = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
    return (V, N, UV, T, mat, obj)


def checker_image(size=64, a=(0.9, 0.9, 0.85), b=(0.15, 0.2, 0.45), squares=8):
    img = np.zeros((size, size, 4), np.float32)
    img[:, :, 3] = 1.0
    step = max(size // squares, 1)
    yy, xx = np.mgrid[0:size, 0:size]
    m = ((xx // step + yy // step) % 2) == 0
    img[m, :3] = np.asarray(a, np.float32)
    img[~m, :3] = np.asarray(b, np.float32)
    return img


def look_at_matrix(eye, target, up=(0, 0, 1)):
    eye = np.asarray(eye, np.float32)
    target = np.asarray(target, np.float32)
    up = np.asarray(up, np.float32)
    f = target - eye
    f /= np.linalg.norm(f)
    if abs(float(np.dot(f, up))) > 0.999:
        up = np.array([0, 1.0, 0], np.float32)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[:3, 0] = s
    m[:3, 1] = u
    m[:3, 2] = -f
    m[:3, 3] = eye
    return m


def demo_scene(settings=None, with_texture=True):
    """A cube, a sphere and a floor under two lights -- the classic test card."""
    st = settings or RenderSettings()
    mats = [
        Material(name='Floor', index=0, model='LAMBERT', diffuse=(0.55, 0.55, 0.6),
                 specular_level=0.0, ambient_level=0.6),
        Material(name='Ball', index=1, model='PHONG', diffuse=(0.85, 0.2, 0.15),
                 specular=(1.0, 1.0, 1.0), specular_level=0.9, glossiness=48.0),
        Material(name='Box', index=2, model='BLINN_PHONG', diffuse=(0.2, 0.45, 0.85),
                 specular=(1.0, 1.0, 0.9), specular_level=0.6, glossiness=18.0,
                 anisotropy=0.55, aniso_rotation=0.2),
    ]
    mesh = _mesh_concat([
        plane(z=0.0, size=11.0, mat=0, obj=0),
        sphere(centre=(-1.3, 0.2, 1.0), radius=1.0, mat=1, obj=1),
        cube(centre=(1.4, -0.4, 0.9), size=1.8, mat=2, obj=2),
    ])
    idx0 = 0
    smooth = np.zeros(mesh.tris.shape[0], bool)
    smooth[mesh.mat_index == 1] = True
    mesh.smooth = smooth

    objs = [ObjectInfo(name='Floor', index=0, location=(0, 0, 0),
                       matrix_world=np.eye(4, dtype=np.float32)),
            ObjectInfo(name='Ball', index=1, location=(-1.3, 0.2, 1.0),
                       matrix_world=np.eye(4, dtype=np.float32), color=(1, .4, .4, 1)),
            ObjectInfo(name='Box', index=2, location=(1.4, -0.4, 0.9),
                       matrix_world=np.eye(4, dtype=np.float32))]

    lights = [
        Light(type='SUN', name='Key', direction=(-0.62, 0.45, -0.45),
              color=(1.0, 0.96, 0.88), energy=6.0, shadow='MAP', shadow_bias=0.02),
        Light(type='POINT', name='Fill', position=(3.5, -3.0, 3.0),
              color=(0.45, 0.6, 1.0), energy=600.0, shadow='MAP',
              decay='INVERSE_SQUARE', shadow_bias=0.03),
    ]

    cam = Camera(matrix_world=look_at_matrix((5.2, -6.4, 3.6), (0.0, -0.2, 0.9)),
                 lens=42.0, sensor=36.0, clip_start=0.1, clip_end=200.0)

    world = World(color=(0.06, 0.07, 0.11), ambient=(0.06, 0.07, 0.10),
                  sky_blend=True, horizon=(0.10, 0.09, 0.12),
                  zenith=(0.05, 0.08, 0.18))

    sc = Scene(mesh=mesh, materials=mats, objects=objs, lights=lights,
               camera=cam, world=world, settings=st)
    sc.images = {}
    if with_texture:
        buf = ImageBuffer(name='checker', pixels=checker_image())
        sc.images['checker'] = buf
        mats[0].graph = {
            'output': 'out',
            'nodes': {
                'tex': {'id': 'tex', 'bl_idname': 'ShaderNodeTexImage',
                        'props': {'image': 'checker', 'interpolation': 'Closest'},
                        'inputs': [{'name': 'Vector', 'type': 'VECTOR',
                                    'default': [0, 0, 0], 'link': None}],
                        'outputs': [{'name': 'Color', 'type': 'RGBA'},
                                    {'name': 'Alpha', 'type': 'VALUE'}]},
                'bsdf': {'id': 'bsdf', 'bl_idname': 'ShaderNodeBsdfDiffuse',
                         'props': {},
                         'inputs': [{'name': 'Color', 'type': 'RGBA',
                                     'default': [0.8, 0.8, 0.8, 1.0],
                                     'link': ['tex', 0]},
                                    {'name': 'Roughness', 'type': 'VALUE',
                                     'default': 0.0, 'link': None},
                                    {'name': 'Normal', 'type': 'VECTOR',
                                     'default': [0, 0, 0], 'link': None}],
                         'outputs': [{'name': 'BSDF', 'type': 'SHADER'}]},
                'out': {'id': 'out', 'bl_idname': 'ShaderNodeOutputMaterial',
                        'props': {},
                        'inputs': [{'name': 'Surface', 'type': 'SHADER',
                                    'default': None, 'link': ['bsdf', 0]},
                                   {'name': 'Displacement', 'type': 'VECTOR',
                                    'default': [0, 0, 0], 'link': None}],
                        'outputs': []},
            },
        }
    return sc
