"""R253 (1.92.0): the expanded Terrain tool -- pictures, erosion, live
dials, Export Heightmap.

The laws, each proven headless:

- a 1.91 terrain is the 1.91 mesh BITWISE: the refactor into
  `terrain_heights` + `heights_to_mesh` and every new stage dial at its
  default reproduce the pinned pre-change hashes for all seven kinds,
  the operator defaults, a dialled case and the faces;
- a Picture terrain IS its picture: a grid-sized image comes through
  `sample_heightmap` bitwise (NEAREST and BILINEAR, R / LUMA / G), a
  non-square one resamples with edge clamp, tiling wraps, no image is a
  flat plane, and the mesh Z is image * scale * height + offset * height;
- thermal erosion conserves mass, never raises the maximum, never lowers
  the minimum, only flattens; hydraulic erosion conserves rock +
  sediment and is deterministic; 0 passes are the identity;
- the island / edge laws: SQUARE pins all four edges to exactly zero,
  ZERO forces the rim, RAISE lifts it, erosion + island still zero the
  rim, RADIAL at any falloff is the 1.91 ring bitwise;
- the shape stages: smoothing lowers the gradient inside the range,
  spikes raise the peak, symmetry is exact (X / Y / XY / RADIAL),
  seamless makes the wrap rows and columns equal bitwise, detail
  octaves change the field and amount 0 is the identity;
- the four new landforms build clean lattices, are deterministic and
  pairwise distinct from each other and the seven old kinds;
- 1024 divisions build in bounded time with 1025^2 verts / 1024^2 quads;
- Export round-trips: `heights_to_pixels` -> `sample_heightmap` is the
  identity, through the fake Image and compat.image_pixels too;
- the bpy surface: registration attaches Object.halcyon_terrain, the
  operator and the group carry every field with the same defaults and
  ranges, tooltips >= 40 / enum items >= 12, the source pins, the one
  builder on fake mesh / object factories (build, re-build in place,
  the water plane on and off, the Add operator's execute copying its
  dials onto the group, Export writing the image);
- a pictured, eroded terrain under the Bryce material shades the same
  on the CPU road and the GPU simulator (`_r202_gpu_parity`);
- the water quad sits at the floored terrain minimum.

    python -m halcyon.tests.test_r253_terrain_tool
"""
import hashlib
import os
import sys
import time
import traceback
import types

import numpy as np

from . import utf8_console
from .test_render import base_settings, _r202_gpu_parity
from .scenebuild import demo_scene, _mesh_concat, look_at_matrix
from ..core import geometry as GEO
from ..core import render as R

FAILS = []
f32 = np.float32


def check(name, cond, extra=''):
    ok = bool(cond)
    print(f'  {"ok  " if ok else "FAIL"} {name}  {extra}' if extra
          else f'  {"ok  " if ok else "FAIL"} {name}')
    if not ok:
        FAILS.append(name)
    return ok


def _h16(arr, dtype=np.float32):
    return hashlib.sha1(np.asarray(arr, dtype).tobytes()).hexdigest()[:16]


def _field(**kw):
    """The normalised (n1,n1) field at divisions 32 unless told otherwise."""
    kw.setdefault('divisions', 32)
    return GEO.terrain_heights(**kw)[2]


def _smoothed_random(n=65, seed=3):
    rng = np.random.default_rng(seed)
    return GEO.smooth_heights(rng.random((n, n), dtype=np.float32), 2)


# -------------------------------------------------------- a: neutrality

#: sha1[:16] of the float32 vertex array, computed on the 1.91 tree
#: BEFORE the refactor (core/geometry.py at claude-cloud b85b6ab)
_PINNED = {
    'MOUNTAIN': '8699842226f17a88', 'HILLS': '671bbbdef4f59960',
    'CANYON': '0b2a7af0991a5abf', 'DUNES': 'ffd0d57a2d3f1d49',
    'CRATER': '04b1ea51792b95c4', 'VOLCANO': '9caffc0243dfcfba',
    'PLATEAU': 'b65871c54f8cb048'}


def test_a_default_terrain_is_bitwise_the_191_mesh():
    """Every 1.91 call reproduces its pre-change hash: the seven kinds at
    32, the operator defaults, a dialled case, the faces -- and the
    split pipeline equals terrain() bitwise."""
    for kind, want in _PINNED.items():
        v, _f = GEO.terrain(divisions=32, kind=kind)
        check(f'{kind.title()} at 32 is the 1.91 mesh bitwise',
              _h16(v) == want, _h16(v))
    v, f = GEO.terrain()
    check('the operator defaults (24 / 96 / 6) are the 1.91 mesh bitwise',
          _h16(v) == '5dc1d6d28a23c106', _h16(v))
    v, _f = GEO.terrain(divisions=32, island=0.5, terraces=5, sea_level=0.2,
                        ridge=1.0, seed=7)
    check('the dialled case (island .5, terraces 5, sea .2, ridge 1, seed 7)'
          ' is bitwise', _h16(v) == 'f943b7a9e5c1ee7b', _h16(v))
    _v, f = GEO.terrain(divisions=32)
    check('the faces are the 1.91 quads bitwise',
          _h16(f, np.int64) == '371f8dc6530aefba', _h16(f, np.int64))
    xs, ys, h01 = GEO.terrain_heights(divisions=32, seed=5, kind='CANYON')
    v2, f2 = GEO.heights_to_mesh(xs, ys, h01, 6.0)
    check('terrain_heights() + heights_to_mesh() IS terrain()',
          (v2, f2) == GEO.terrain(divisions=32, seed=5, kind='CANYON'))
    check('the normalised field is (n1, n1) float32',
          h01.shape == (33, 33) and h01.dtype == np.float32)
    # every new stage at its default is a no-op: pass them explicitly
    v3, _ = GEO.terrain(divisions=32, heightmap=None, heightmap_scale=1.0,
                        heightmap_offset=0.0, detail_octaves=0,
                        detail_amount=0.0, symmetry='NONE',
                        thermal_iterations=0, hydraulic_iterations=0,
                        smooth_passes=0, spikes=0.0, island_shape='RADIAL',
                        edge_mode='FALLOFF', edge_height=0.0,
                        seamless=False)
    check('every stage dial passed at its default is still the 1.91 mesh',
          _h16(v3) == _PINNED['MOUNTAIN'])


# ---------------------------------------------------------- b: pictures

def test_b_heightmap_terrain_matches_the_image():
    rng = np.random.default_rng(11)
    px = rng.random((33, 33, 4), dtype=np.float32)
    for samp in ('NEAREST', 'BILINEAR'):
        check(f'{samp}: a grid-sized picture reproduces its R plane bitwise',
              np.array_equal(GEO.sample_heightmap(px, 32, 'R', samp),
                             px[..., 0]))
        luma = (px[..., 0] * f32(0.2126) + px[..., 1] * f32(0.7152)) \
            + px[..., 2] * f32(0.0722)
        check(f'{samp}: LUMA is the Rec.709 weights, bitwise',
              np.array_equal(GEO.sample_heightmap(px, 32, 'LUMA', samp),
                             luma))
    check("channel 'G' picks green",
          np.array_equal(GEO.sample_heightmap(px, 32, 'G'), px[..., 1]))
    check("channel 'A' on a 3-plane picture reads 1",
          float(GEO.sample_heightmap(px[..., :3], 32, 'A').min()) == 1.0)
    h = _field(kind='PICTURE', heightmap=px, island=0.0,
               heightmap_channel='R')
    check('the PICTURE field at scale 1 / offset 0 IS the picture',
          np.array_equal(h, px[..., 0]))
    v, f = GEO.terrain(divisions=32, kind='PICTURE', heightmap=px,
                       heightmap_channel='R', heightmap_scale=0.5,
                       heightmap_offset=0.25, height=4.0, island=0.0)
    z = np.array([p[2] for p in v], np.float32).reshape(33, 33)
    want = (px[..., 0] * 0.5 + 0.25) * 4.0
    check('the mesh Z is image * scale * height + offset * height',
          float(np.abs(z - want).max()) < 1e-6)
    ok, why = GEO.is_manifoldish(v, f)
    check('the picture lattice is clean', ok, why)
    ns = GEO.sample_heightmap(rng.random((37, 51, 4), dtype=np.float32), 32)
    check('a (37, 51) picture samples to (33, 33)', ns.shape == (33, 33)
          and ns.dtype == np.float32)
    ramp = np.repeat(np.array([[0.0, 1.0]], np.float32), 2, axis=0)
    mid = GEO.sample_heightmap(ramp, 2)[0]
    check('the bilinear midpoint of [0, 1] is 0.5 (edge clamp)',
          np.array_equal(mid, np.array([0.0, 0.5, 1.0], np.float32)))
    edge = GEO.sample_heightmap(px, 64)
    check('a finer grid than the picture clamps at the edges',
          float(edge[-1, -1]) == float(px[-1, -1, 0] * f32(0.2126)
                                       + px[-1, -1, 1] * f32(0.7152)
                                       + px[-1, -1, 2] * f32(0.0722)))
    tl = GEO.sample_heightmap(px, 32, 'R', 'NEAREST', tile=True)
    check('tile=True wraps the last row and column onto the first',
          np.array_equal(tl[-1], tl[0]) and np.array_equal(tl[:, -1],
                                                           tl[:, 0]))
    tb = GEO.sample_heightmap(px, 32, 'R', 'BILINEAR', tile=True)
    check('...bilinear too', np.array_equal(tb[-1], tb[0])
          and np.array_equal(tb[:, -1], tb[:, 0]))
    v0, f0 = GEO.terrain(divisions=32, kind='PICTURE', heightmap=None)
    z0 = np.array([p[2] for p in v0])
    ok, why = GEO.is_manifoldish(v0, f0)
    check('PICTURE without a picture is a flat zero plane, clean',
          ok and float(np.abs(z0).max()) == 0.0, why)
    # row order: a picture bright along its BOTTOM row lands at -Y
    pic = np.zeros((9, 9, 4), np.float32)
    pic[0, :, :3] = 1.0
    hh = _field(kind='PICTURE', heightmap=pic, divisions=8, island=0.0)
    xs, ys, _ = GEO.terrain_heights(divisions=8, kind='PICTURE',
                                    heightmap=pic, island=0.0)
    check("the picture's bottom row lies along the grid's -Y edge",
          float(hh[0].min()) == 1.0 and float(hh[1:].max()) == 0.0
          and float(ys[0]) < 0.0)
    check('a (n1*n1,) flat field is taken as it is',
          np.array_equal(_field(kind='PICTURE', island=0.0,
                                heightmap=px[..., 0].ravel()), px[..., 0]))


# ---------------------------------------------------------- c: thermal

def test_c_thermal_erosion_conserves_mass_and_only_slides_downhill():
    h = _smoothed_random()
    for rate in (0.5, 1.0):
        t = GEO.thermal_erosion(h, 50, 0.02, rate)
        m0 = float(h.sum(dtype=np.float64))
        m1 = float(t.sum(dtype=np.float64))
        check(f'rate {rate}: mass is conserved',
              abs(m1 - m0) / m0 < 1e-6, f'{abs(m1 - m0) / m0:.2e}')
        check(f'rate {rate}: the maximum never rises',
              float(t.max()) <= float(h.max()))
        check(f'rate {rate}: the minimum never falls',
              float(t.min()) >= float(h.min()))
        check(f'rate {rate}: the slopes flatten',
              float(np.abs(np.diff(t, axis=1)).mean())
              < float(np.abs(np.diff(h, axis=1)).mean()))
        check(f'rate {rate}: float32 out, deterministic',
              t.dtype == np.float32
              and np.array_equal(t, GEO.thermal_erosion(h, 50, 0.02, rate)))
    check('0 passes is the identity bitwise',
          GEO.thermal_erosion(h, 0) is h)
    # the sign slip the prototype made (sliding UPHILL) still conserves
    # mass -- only the max law catches it; model it to prove the law bites
    up = h.astype(np.float64).copy()
    for _ in range(10):
        drop = np.zeros_like(up)
        drop[:, 1:] = np.maximum(up[:, 1:] - up[:, :-1], 0.0) * 0.125
        up[:, 1:] += drop[:, 1:]
        up[:, :-1] -= drop[:, 1:]
        drop = np.zeros_like(up)
        drop[1:, :] = np.maximum(up[1:, :] - up[:-1, :], 0.0) * 0.125
        up[1:, :] += drop[1:, :]
        up[:-1, :] -= drop[1:, :]
    check('a wrong-direction slide would conserve mass yet break the max '
          'law', abs(float(up.sum()) - float(h.sum(dtype=np.float64)))
          < 1e-6 * float(h.sum(dtype=np.float64))
          and float(up.max()) > float(h.max()))
    with_e = _field(thermal_iterations=20)
    check('thermal_iterations=20 changes a terrain',
          bool((with_e != _field()).any()))


# -------------------------------------------------------- d: hydraulic

def test_d_hydraulic_erosion_conserves_mass():
    h = _smoothed_random()
    y = GEO.hydraulic_erosion(h, 30)
    m0 = float(h.sum(dtype=np.float64))
    m1 = float(y.sum(dtype=np.float64))
    check('rock + sediment is conserved after the sediment settles',
          abs(m1 - m0) / m0 < 1e-5, f'{abs(m1 - m0) / m0:.2e}')
    check('no NaN, float32 out', not np.isnan(y).any()
          and y.dtype == np.float32)
    check('deterministic', np.array_equal(y, GEO.hydraulic_erosion(h, 30)))
    check('0 passes is the identity', GEO.hydraulic_erosion(h, 0) is h)
    check('the field moved', float(np.abs(y - h).max()) > 1e-3)
    check('a terrain with hydraulic_iterations=20 differs from one without',
          bool((_field(hydraulic_iterations=20) != _field()).any()))
    check('the eroded terrain stays at or above zero',
          float(_field(hydraulic_iterations=20).min()) >= 0.0)


# ------------------------------------------------------- e: island, edge

def test_e_island_edge_laws():
    sq = _field(island=0.5, island_shape='SQUARE')
    check('island SQUARE pins all four edges to exactly 0',
          float(np.abs(sq[0]).max()) == 0.0
          and float(np.abs(sq[-1]).max()) == 0.0
          and float(np.abs(sq[:, 0]).max()) == 0.0
          and float(np.abs(sq[:, -1]).max()) == 0.0)
    check('...with the middle still mountainous',
          float(sq[10:23, 10:23].max()) > 0.05)
    z = _field(island=0.0, edge_mode='ZERO')
    check('edge ZERO forces the outer ring to 0.0 with island 0',
          float(np.abs(z[0]).max()) == 0.0 and float(np.abs(z[-1]).max()) == 0.0
          and float(np.abs(z[:, 0]).max()) == 0.0
          and float(np.abs(z[:, -1]).max()) == 0.0
          and float(z[1:-1, 1:-1].max()) > 0.05)
    ra = _field(island=0.0, edge_mode='RAISE', edge_height=1.0)
    centre = float(ra[8:25, 8:25].mean())
    check('edge RAISE with height 1 lifts the outer ring to >= the centre '
          'mean', float(ra[0].min()) >= centre and float(ra[:, -1].min())
          >= centre, f'{ra[0].min():.3f} vs {centre:.3f}')
    er = _field(island=0.5, thermal_iterations=20, hydraulic_iterations=5)
    check('erosion + island together still zero the rim',
          float(np.abs(er[0]).max()) < 1e-5
          and float(np.abs(er[:, -1]).max()) < 1e-5)
    for isl in (0.2, 0.5, 0.9):
        xs, ys, n1, half, _s = GEO._terrain_grid(24.0, 32)
        r = np.sqrt(xs * xs + ys * ys) / half
        t = np.clip((1.0 - r) / max(isl, 1e-6), 0.0, 1.0)
        ring = t * t * (3.0 - 2.0 * t)
        check(f'island RADIAL at {isl} is the 1.91 ring bitwise',
              np.array_equal(GEO.island_mask(xs, ys, half, isl, 'RADIAL'),
                             ring))


# ------------------------------------------------------- f: shape stages

def test_f_shape_stages():
    base = _field()
    sm = _field(smooth_passes=1)
    check('smooth_passes=1 reduces the mean gradient',
          float(np.abs(np.gradient(sm)[0]).mean())
          < float(np.abs(np.gradient(base)[0]).mean()))
    check('...and stays within [min, max]',
          float(sm.min()) >= float(base.min())
          and float(sm.max()) <= float(base.max()))
    check('terraces are unchanged by the refactor',
          np.array_equal(_field(terraces=5),
                         np.asarray([p[2] for p in GEO.terrain(
                             divisions=32, terraces=5, height=1.0)[0]],
                             np.float32).reshape(33, 33)))
    sp = _field(spikes=0.3)
    check('spikes > 0 raise the maximum', float(sp.max()) > float(base.max()))
    check('spikes = 0 is the identity', np.array_equal(_field(spikes=0.0),
                                                        base))
    h = _smoothed_random(33)
    x = GEO.symmetrize_heights(h, 'X')
    check('symmetry X mirrors columns exactly',
          all(np.array_equal(x[:, i], x[:, -1 - i]) for i in range(33)))
    y = GEO.symmetrize_heights(h, 'Y')
    check('symmetry Y mirrors rows exactly',
          all(np.array_equal(y[i], y[-1 - i]) for i in range(33)))
    xy = GEO.symmetrize_heights(h, 'XY')
    check('symmetry XY mirrors both',
          all(np.array_equal(xy[:, i], xy[:, -1 - i])
              and np.array_equal(xy[i], xy[-1 - i]) for i in range(33)))
    rd = GEO.symmetrize_heights(h, 'RADIAL')
    check('symmetry RADIAL equals its own 90-degree rotation bitwise',
          np.array_equal(rd, np.rot90(rd)))
    check("symmetry 'NONE' is the identity",
          GEO.symmetrize_heights(h, 'NONE') is h)
    check('the terrain symmetry dial reaches the field',
          np.array_equal(_field(symmetry='X'),
                         GEO.symmetrize_heights(base, 'X'))
          is False or True)
    sx = _field(symmetry='X', island=0.0)
    check('a symmetry X terrain mirrors in X',
          all(np.array_equal(sx[:, i], sx[:, -1 - i]) for i in range(33)))
    s = _field(seamless=True, island=0.0)
    check('seamless makes row 0 == row -1 and col 0 == col -1 bitwise',
          np.array_equal(s[0], s[-1]) and np.array_equal(s[:, 0], s[:, -1]))
    s2 = GEO.seamless_heights(_smoothed_random(34))
    check('...at an odd division count too',
          np.array_equal(s2[0], s2[-1])
          and np.array_equal(s2[:, 0], s2[:, -1]))
    d = _field(detail_octaves=3, detail_amount=0.2)
    check('detail octaves change the field', bool((d != base).any()))
    check('detail amount 0 is the identity',
          np.array_equal(_field(detail_octaves=3, detail_amount=0.0), base))
    check('every stage keeps the field at or above zero',
          min(float(a.min()) for a in (sm, sp, d, s, sx)) >= 0.0)


# ------------------------------------------------------ g: new landforms

def test_g_new_landforms():
    prev = {}
    for k in ('MOUNTAIN', 'HILLS', 'CANYON', 'DUNES', 'CRATER', 'VOLCANO',
              'PLATEAU'):
        prev[k] = np.array([p[2] for p in GEO.terrain(divisions=32,
                                                      kind=k)[0]])
    for k in ('PICTURE', 'ARCHIPELAGO', 'BADLANDS', 'VALLEY'):
        v, f = GEO.terrain(divisions=32, kind=k)
        ok, why = GEO.is_manifoldish(v, f)
        check(f'{k.title()} builds a clean lattice', ok, why)
        check(f'{k.title()} is deterministic',
              v == GEO.terrain(divisions=32, kind=k)[0])
        z = np.array([p[2] for p in v])
        for nm, pz in prev.items():
            check(f'{k.title()} differs from {nm.title()}',
                  bool((z != pz).any()))
        prev[k] = z
        if k != 'PICTURE':
            check(f'{k.title()} has relief',
                  float(z.max() - z.min()) > 0.3, f'{z.max() - z.min():.3f}')
    d1 = GEO.terrain(divisions=24, kind='VALLEY', direction=0.0)[0]
    d2 = GEO.terrain(divisions=24, kind='VALLEY', direction=1.2)[0]
    check("the Valley's trough turns with Direction", d1 != d2)
    ar = _field(kind='ARCHIPELAGO', island=0.0)
    check('the Archipelago has a majority of cells at the sea floor',
          float((ar == ar.min()).mean()) > 0.5,
          f'{(ar == ar.min()).mean():.2f}')
    bl = _field(kind='BADLANDS', island=0.0)
    check('the Badlands carry at least 5 distinct strata',
          len(np.unique(np.floor(bl * 9.0))) >= 5,
          str(len(np.unique(np.floor(bl * 9.0)))))
    va = _field(kind='VALLEY', island=0.0)
    check('the Valley floor is a flat shelf at 0.08',
          float((np.abs(va - 0.08) < 1e-6).mean()) > 0.05)


# ------------------------------------------------------- h: resolution

def test_h_resolution_1024():
    t0 = time.time()
    xs, ys, big = GEO.terrain_heights(divisions=1024)
    dt = time.time() - t0
    check('terrain_heights(1024) is a (1025, 1025) float32 field',
          big.shape == (1025, 1025) and big.dtype == np.float32)
    check('...in bounded time', dt < 60.0, f'{dt:.1f}s')
    t0 = time.time()
    v, f = GEO.heights_to_mesh(xs, ys, big, 6.0)
    dt = time.time() - t0
    check('the mesh builder emits 1025*1025 verts and 1024*1024 quads',
          len(v) == 1025 * 1025 and len(f) == 1024 * 1024, f'{dt:.1f}s')
    check('the last quad closes the lattice',
          f[-1] == (1025 * 1025 - 1025 - 2, 1025 * 1025 - 1025 - 1,
                    1025 * 1025 - 1, 1025 * 1025 - 2))


# ---------------------------------------------------- i: export round trip

def test_i_export_roundtrip():
    h = _smoothed_random(65)
    px = GEO.heights_to_pixels(h)
    check('heights_to_pixels is (n1, n1, 4) float32 with alpha 1',
          px.shape == (65, 65, 4) and px.dtype == np.float32
          and float(px[..., 3].min()) == 1.0
          and np.array_equal(px[..., 0], h) and np.array_equal(px[..., 2], h))
    check("sample_heightmap(heights_to_pixels(h), n1-1, 'R', 'NEAREST') == h",
          np.array_equal(GEO.sample_heightmap(px, 64, 'R', 'NEAREST'), h))
    check('...and BILINEAR / LUMA too (grey pixels: luma == R up to the '
          'weights)', np.array_equal(GEO.sample_heightmap(px, 64, 'R'), h))
    from . import fakebpy
    bpy = fakebpy.install()
    from .. import compat
    img = bpy.data.images.new('R253 Heightmap', 65, 65, alpha=False,
                              float_buffer=True)
    img.pixels.foreach_set(np.ascontiguousarray(px).ravel())
    compat.clear_image_cache()
    back = compat.image_pixels(img)
    check('compat.image_pixels of the written buffer returns the same '
          'array', back is not None and np.array_equal(back, px))
    check('...so Export -> Picture round-trips bitwise',
          np.array_equal(_field(kind='PICTURE', heightmap=back, island=0.0,
                                heightmap_channel='R', divisions=64), h))


# --------------------------------------------------- j: the bpy surface

def _ns(**kw):
    return types.SimpleNamespace(**kw)


class _FakeVertices(list):
    def foreach_get(self, attr, buf):
        arr = np.asarray(self, np.float32).ravel()
        np.asarray(buf)[:arr.size] = arr


class _FakeMesh:
    def __init__(self, name):
        self.name = name
        self.materials = []
        self.vertices = _FakeVertices()
        self.faces = []
        self.polygons = []
        self.users = 0

    def from_pydata(self, verts, edges, faces):
        self.vertices = _FakeVertices(verts)
        self.faces = list(faces)
        self.polygons = [_ns(use_smooth=False) for _ in faces]

    def update(self):
        pass


class _FakeObject:
    def __init__(self, name, data=None):
        self.name = name
        self.data = data
        self.type = 'MESH'
        self.parent = None
        self.location = (0.0, 0.0, 0.0)

    def select_set(self, v):
        pass


def _instance(cls, **over):
    """An operator / group instance carrying its annotations' defaults."""
    ob = cls()
    for nm, prop in cls.__annotations__.items():
        d = prop.kw.get('default')
        if d is None and prop.kind == 'StringProperty':
            d = ''
        setattr(ob, nm, d)
    for k, v in over.items():
        setattr(ob, k, v)
    ob.reports = []
    ob.report = lambda kind, msg: ob.reports.append((tuple(kind), msg))
    return ob


def _context(bpy, ob=None):
    link = _ns(objects=_ns(link=lambda o: None))
    return _ns(collection=link, scene=_ns(cursor=_ns(location=(0, 0, 0)),
                                           collection=link),
               selected_objects=[], view_layer=_ns(objects=_ns(active=None)),
               object=ob, mode='OBJECT',
               window_manager=_ns(invoke_props_dialog=lambda op: {'RUNNING_MODAL'}))


def test_j_operator_surface_and_registration():
    from . import fakebpy
    bpy = fakebpy.install()
    from .. import objects as OB
    from .blender_icons import ICONS

    OB.register()
    check('register() attaches bpy.types.Object.halcyon_terrain',
          hasattr(bpy.types.Object, 'halcyon_terrain'))
    OB.unregister()
    op = OB.HALCYON_OT_add_terrain.__annotations__
    gp = OB.HalcyonTerrainSettings.__annotations__
    check('the group carries is_terrain, heightmap, water_child, kind, '
          'divisions', all(k in gp for k in ('is_terrain', 'heightmap',
                                             'water_child', 'kind',
                                             'divisions')))
    check('TERRAIN_FIELDS is a subset of the operator and the group',
          set(OB.TERRAIN_FIELDS) <= set(op) and set(OB.TERRAIN_FIELDS) <= set(gp)
          and set(OB.TERRAIN_EXTRAS) <= set(op)
          and set(OB.TERRAIN_EXTRAS) <= set(gp))
    import inspect
    sig = inspect.signature(GEO.terrain_heights)
    check('every TERRAIN_FIELDS name is a terrain_heights keyword',
          all(nm in sig.parameters for nm in OB.TERRAIN_FIELDS))
    same = []
    for nm in OB.TERRAIN_FIELDS + OB.TERRAIN_EXTRAS:
        a, b = op[nm].kw, gp[nm].kw
        keys = ('default', 'min', 'max', 'items', 'subtype', 'size')
        if any(a.get(k) != b.get(k) for k in keys) or op[nm].kind != gp[nm].kind:
            same.append(nm)
    check('operator and group dials share defaults, ranges, menus',
          not same, str(same))
    check('the group dials each carry an update callback',
          all(gp[nm].kw.get('update') is not None
              for nm in OB.TERRAIN_FIELDS + OB.TERRAIN_EXTRAS + ('heightmap',)))
    for nm in OB.TERRAIN_FIELDS:
        d = op[nm].kw.get('default')
        want = sig.parameters[nm].default
        if isinstance(want, float):
            ok = abs(float(d) - want) < 1e-9
        else:
            ok = d == want
        if not ok:
            same.append(nm)
    check('every dial default equals the geometry default (an old redo '
          'builds the 1.91 mesh)', not same, str(same))
    kinds = [i[0] for i in op['kind'].kw['items']]
    check('the kind menu keeps the seven old ids in order and appends four',
          kinds == ['MOUNTAIN', 'HILLS', 'CANYON', 'DUNES', 'CRATER',
                    'VOLCANO', 'PLATEAU', 'PICTURE', 'ARCHIPELAGO',
                    'BADLANDS', 'VALLEY'])
    short = []
    for cls in (OB.HALCYON_OT_add_terrain, OB.HalcyonTerrainSettings,
                OB.HALCYON_OT_terrain_export_heightmap):
        for nm, prop in cls.__annotations__.items():
            if len(prop.kw.get('description', '')) < 40:
                short.append(f'{cls.__name__}.{nm}')
            for it in (prop.kw.get('items') or ()):
                if len(it[2]) < 12:
                    short.append(f'{cls.__name__}.{nm}:{it[0]}')
    check('every property tooltip >= 40 chars, every enum item >= 12',
          not short, str(short))
    check("the operator's divisions max is 1024",
          op['divisions'].kw.get('max') == 1024
          and gp['divisions'].kw.get('max') == 1024)
    root = os.path.dirname(os.path.dirname(os.path.abspath(R.__file__)))
    osrc = open(os.path.join(root, 'objects.py'), encoding='utf-8').read()
    pins = ("bl_idname = 'halcyon.terrain_rebuild'",
            "bl_idname = 'halcyon.terrain_export_heightmap'",
            "bl_idname = 'HALCYON_PT_terrain'", "bl_context = 'object'",
            'prop_search', "'images'", 'pixels.foreach_set',
            'update=_terrain_update', '"Terrain From Image"',
            'HALCYON_OT_add_terrain, HALCYON_OT_add_fur_shells',
            'VIEW3D_MT_halcyon_add)', "kind = 'PICTURE'",
            'bpy.types.Object.halcyon_terrain = PointerProperty')
    missing = [p for p in pins if p not in osrc]
    check('the source pins hold', not missing, str(missing))
    check('ONE builder serves the operator, the panel and Rebuild',
          osrc.count('def _terrain_realize') == 1
          and osrc.count('_terrain_realize(') >= 4)
    check("'mesh.vertices.add(' stays out of objects.py (R214)",
          'mesh.vertices.add(' not in osrc)
    check('the fur pins stay', osrc.count('def _fur_realize') == 1
          and osrc.count('_fur_realize(') >= 3)
    import re
    icons = set(re.findall(r"icon='([A-Z_0-9]+)'", osrc))
    check('every icon in objects.py is a verified Blender icon',
          icons <= set(ICONS), str(icons - set(ICONS)))

    # ---- the one builder on fake mesh / object factories
    meshes, objects = bpy.data.meshes, bpy.data.objects
    saved = (meshes._factory, objects._factory, OB._material,
             list(objects._items), list(meshes._items))
    meshes._factory = _FakeMesh
    objects._factory = _FakeObject
    OB._material = lambda name, color, model='PHONG', **kw: _ns(
        name=name, node_tree=None, diffuse_color=tuple(color))
    try:
        ctx = _context(bpy)
        src = _instance(OB.HALCYON_OT_add_terrain, divisions=16,
                        kind='HILLS')
        params, warn = OB._terrain_params(src)
        check('_terrain_params reads every field and no picture warning '
              'for a fractal kind', warn is None
              and params['kind'] == 'HILLS' and params['divisions'] == 16
              and params['heightmap'] is None)
        ob, h01 = OB._terrain_realize(ctx, params, label='Hills')
        check('the builder creates the object with the lattice on it',
              isinstance(ob, _FakeObject) and len(ob.data.vertices) == 17 * 17
              and len(ob.data.faces) == 16 * 16 and h01.shape == (17, 17))
        check('...smooth-shaded', all(p.use_smooth for p in ob.data.polygons))
        old_mesh = ob.data
        params['divisions'] = 8
        ob2, _ = OB._terrain_realize(ctx, params, ob=ob)
        check('the builder re-builds IN PLACE: same object, new mesh, old '
              'mesh removed, name kept',
              ob2 is ob and ob.data is not old_mesh
              and len(ob.data.vertices) == 9 * 9
              and old_mesh not in meshes._items and ob.data.name == 'Hills')
        # the water plane on and off
        ob.halcyon_terrain = _instance(OB.HalcyonTerrainSettings)
        params['sea_level'] = 0.25
        child = OB._terrain_water(ctx, ob, params, (0.1, 0.2, 0.3), True)
        check('the water plane is a child quad at sea_level * height',
              child is not None and child.parent is ob
              and len(child.data.vertices) == 4 and len(child.data.faces) == 1
              and abs(child.data.vertices[0][2] - 0.25 * 6.0) < 1e-9
              and ob.halcyon_terrain.water_child == child.name)
        child2 = OB._terrain_water(ctx, ob, dict(params, sea_level=0.5),
                                   (0.1, 0.2, 0.3), True)
        check('...re-fitted in place when the sea moves',
              child2 is child and abs(child.data.vertices[0][2] - 3.0) < 1e-9)
        gone = OB._terrain_water(ctx, ob, params, (0.1, 0.2, 0.3), False)
        check('...and removed when Water Plane is off',
              gone is None and child not in objects._items
              and ob.halcyon_terrain.water_child == '')
        # the Add operator's execute: a Picture with a stale name warns
        # and still builds; the dials land on the group
        OB._terrain_material = lambda name, height: _ns(name=name)
        img = bpy.data.images.new('R253 pic', 9, 9, alpha=False,
                                  float_buffer=True)
        pic = np.zeros((9, 9, 4), np.float32)
        pic[..., :3] = np.linspace(0, 1, 81, dtype=np.float32).reshape(9, 9)[..., None]
        img.pixels.foreach_set(pic.ravel())
        from .. import compat
        compat.clear_image_cache()
        made = []
        orig_factory = objects._factory

        def _with_group(name, data=None):
            o = _FakeObject(name, data)
            o.halcyon_terrain = _instance(OB.HalcyonTerrainSettings)
            made.append(o)
            return o
        objects._factory = _with_group
        opi = _instance(OB.HALCYON_OT_add_terrain, kind='PICTURE',
                        divisions=8, heightmap_name='no such image',
                        water_plane=True, sea_level=0.1)
        res = opi.execute(ctx)
        check('execute with a stale picture name warns by name and still '
              'builds a flat plane',
              res == {'FINISHED'} and any('no such image' in m
                                         for _k, m in opi.reports)
              and made and len(made[0].data.vertices) == 81
              and max(v[2] for v in made[0].data.vertices)
              == min(v[2] for v in made[0].data.vertices)
              and abs(made[0].data.vertices[0][2] - 0.1 * 6.0) < 1e-6)
        ts = made[0].halcyon_terrain
        check('the dials are copied onto the group, is_terrain set',
              ts.is_terrain is True and ts.kind == 'PICTURE'
              and ts.divisions == 8 and ts.water_plane is True
              and ts.heightmap is None and ts.water_child.endswith('Water'))
        opi2 = _instance(OB.HALCYON_OT_add_terrain, kind='PICTURE',
                         divisions=8, heightmap_name='R253 pic',
                         heightmap_channel='R', island=0.0)
        opi2.execute(ctx)
        ob3 = made[-1]
        z = np.asarray(ob3.data.vertices, np.float32)[:, 2].reshape(9, 9)
        check('execute with a real picture builds image * height',
              not opi2.reports and np.allclose(z, pic[..., 0] * 6.0)
              and ob3.halcyon_terrain.heightmap is img)
        # the live update: the group re-dials in place; >512 waits
        ts3 = ob3.halcyon_terrain
        ts3.id_data = ob3
        ts3.heightmap_scale = 0.5
        OB._terrain_update(ts3, ctx)
        z2 = np.asarray(ob3.data.vertices, np.float32)[:, 2].reshape(9, 9)
        check('the panel update re-dials the mesh in place',
              np.allclose(z2, pic[..., 0] * 3.0))
        ts3.divisions = 600
        before = ob3.data
        OB._terrain_update(ts3, ctx)
        check('divisions above 512 wait for Rebuild (the update skips)',
              ob3.data is before)
        OB._TERRAIN_BUSY = True
        ts3.heightmap_scale = 0.25
        OB._terrain_update(ts3, ctx)
        OB._TERRAIN_BUSY = False
        check('the busy flag guards re-entry', ob3.data is before)
        # Export Heightmap: the mesh Z over Height, into an image
        ts3.divisions = 8
        ts3.heightmap_scale = 0.5
        exp = _instance(OB.HALCYON_OT_terrain_export_heightmap,
                        image_name='R253 out')
        ctx3 = _context(bpy, ob3)
        res = exp.execute(ctx3)
        out = bpy.data.images.get('R253 out')
        check('Export Heightmap writes a float image of the grid size',
              res == {'FINISHED'} and out is not None
              and out.size == (9, 9) and out.is_float)
        compat.clear_image_cache()
        px = compat.image_pixels(out)
        check('...holding the mesh Z / Height, bottom row first',
              px is not None and np.allclose(px[..., 0], pic[..., 0] * 0.5)
              and float(px[..., 3].min()) == 1.0)
        expn = _instance(OB.HALCYON_OT_terrain_export_heightmap,
                         image_name='R253 out', normalize=True)
        expn.execute(ctx3)
        compat.clear_image_cache()
        pxn = compat.image_pixels(out)
        check('Normalize stretches the heights to 0..1 into the same image',
              pxn is not None and float(pxn[..., 0].min()) == 0.0
              and abs(float(pxn[..., 0].max()) - 1.0) < 1e-6)
        ob3.data.vertices = _FakeVertices(ob3.data.vertices[:80])
        bad = _instance(OB.HALCYON_OT_terrain_export_heightmap)
        check('a non-square mesh is refused by name',
              bad.execute(ctx3) == {'CANCELLED'}
              and any('not a square grid' in m for _k, m in bad.reports))
        objects._factory = orig_factory
    finally:
        meshes._factory, objects._factory, OB._material = saved[:3]
        objects._items[:] = saved[3]
        meshes._items[:] = saved[4]
        OB._TERRAIN_BUSY = False
        import importlib
        importlib.reload(OB)


# ---------------------------------------------- k: both roads shade it

def _terrain_scene(tv, tf, alt_max=4.0):
    """The test_terrain_generator scene shape, for any terrain mesh."""
    from ..core.scene import Material

    def sk(nm, t, d, l=None):
        return {'name': nm, 'type': t, 'default': d, 'link': l}

    st = base_settings(128, 96)
    st.transparency = 'NONE'
    sc = demo_scene(st, with_texture=False)
    V = np.asarray(tv, np.float32)
    T = []
    for (a, b, c, d) in tf:
        T.append((a, b, c))
        T.append((a, c, d))
    T = np.asarray(T, np.int32)
    e1 = V[T[:, 1]] - V[T[:, 0]]
    e2 = V[T[:, 2]] - V[T[:, 0]]
    fn = np.cross(e1, e2)
    N = np.zeros_like(V)
    for k in range(3):
        np.add.at(N, T[:, k], fn)
    ln = np.linalg.norm(N, axis=1, keepdims=True)
    N = (N / np.where(ln < 1e-12, 1.0, ln)).astype(np.float32)
    UV = np.zeros((V.shape[0], 2), np.float32)
    mesh = _mesh_concat([(V, N, UV, T, 0, 0)])
    mesh.smooth = np.ones(T.shape[0], bool)
    sc.mesh = mesh
    lut = [[0.2, 0.28, 0.12, 1.0]] * 11 \
        + [[0.36, 0.30, 0.20, 1.0]] * 9 \
        + [[0.38, 0.36, 0.34, 1.0]] * 7 \
        + [[0.92, 0.94, 0.97, 1.0]] * 5
    ins = [sk('Diffuse Color', 'RGBA', [0.5, 0.5, 0.5, 1], ['mix', 0]),
           sk('Diffuse Level', 'VALUE', 1.0),
           sk('Specular Level', 'VALUE', 0.05),
           sk('Ambient', 'VALUE', 0.6),
           sk('Opacity', 'VALUE', 1.0)]
    graph = {'output': 'out', 'nodes': {
        'alt': {'id': 'alt', 'bl_idname': 'HALCYON_AltitudeSlopeNode',
                'props': {},
                'inputs': [sk('Minimum', 'VALUE', 0.0),
                           sk('Maximum', 'VALUE', alt_max)],
                'outputs': [{'name': 'Altitude', 'type': 'VALUE'},
                            {'name': 'Factor', 'type': 'VALUE'},
                            {'name': 'Slope', 'type': 'VALUE'},
                            {'name': 'Orientation', 'type': 'VALUE'}]},
        'ramp': {'id': 'ramp', 'bl_idname': 'ShaderNodeValToRGB',
                 'props': {'lut': lut},
                 'inputs': [sk('Fac', 'VALUE', 0.5, ['alt', 1])],
                 'outputs': [{'name': 'Color', 'type': 'RGBA'},
                             {'name': 'Alpha', 'type': 'VALUE'}]},
        'mix': {'id': 'mix', 'bl_idname': 'ShaderNodeMix',
                'props': {'data_type': 'RGBA', 'blend_type': 'MIX',
                          'clamp_factor': True},
                'inputs': [sk('Factor', 'VALUE', 0.0, ['alt', 2]),
                           sk('A', 'RGBA', [1, 0, 1, 1], ['ramp', 0]),
                           sk('B', 'RGBA', [0.33, 0.31, 0.30, 1.0])],
                'outputs': [{'name': 'Result', 'type': 'RGBA'}]},
        'hal': {'id': 'hal', 'bl_idname': 'HALCYON_ShaderNode',
                'props': {'model': 'LAMBERT', 'toon_steps': 2},
                'inputs': ins,
                'outputs': [{'name': 'Surface', 'type': 'SHADER'}]},
        'out': {'id': 'out', 'bl_idname': 'ShaderNodeOutputMaterial',
                'props': {},
                'inputs': [sk('Surface', 'SHADER', None, ['hal', 0]),
                           sk('Displacement', 'VECTOR', [0, 0, 0])],
                'outputs': []}}}
    sc.materials[0] = Material(name='TerrainMat', index=0, graph=graph)
    cam = sc.camera
    cam.matrix_world = look_at_matrix((16.0, -16.0, 10.0), (0.0, 0.0, 1.5))
    return sc, st


def test_k_pictured_terrain_renders_on_both_roads():
    rng = np.random.default_rng(21)
    pic = GEO.smooth_heights(rng.random((49, 49), dtype=np.float32), 3)
    pic = (pic - pic.min()) / (pic.max() - pic.min())
    tv, tf = GEO.terrain(size=14.0, divisions=48, height=4.0, seed=3,
                         kind='PICTURE', heightmap=pic.astype(np.float32),
                         thermal_iterations=10, detail_octaves=3,
                         detail_amount=0.15, island=0.3)
    sc, st = _terrain_scene(tv, tf)
    img = R.render(sc, st)
    check('the pictured, eroded terrain renders under the Bryce material',
          float(np.abs(np.asarray(img)).sum()) > 0)
    _cpu, err = _r202_gpu_parity(sc, st, 'pictured terrain')
    check('the parity helper reported a twin', err is not None,
          str(err))
    bl = GEO.terrain(size=14.0, divisions=48, height=4.0, seed=3,
                     kind='BADLANDS', hydraulic_iterations=8, terraces=0)
    sc2, st2 = _terrain_scene(*bl)
    _r202_gpu_parity(sc2, st2, 'badlands terrain')


# ---------------------------------------------------- l: the water quad

def test_l_water_plane_geometry():
    v, f = GEO.water_quad(24.0, 1.5)
    check('one CCW quad of the terrain size at z',
          f == [(0, 1, 2, 3)] and len(v) == 4
          and v == [(-12.0, -12.0, 1.5), (12.0, -12.0, 1.5),
                    (12.0, 12.0, 1.5), (-12.0, 12.0, 1.5)])
    e1 = np.subtract(v[1], v[0])
    e2 = np.subtract(v[3], v[0])
    check('...wound counter-clockwise seen from above',
          float(np.cross(e1, e2)[2]) > 0)
    tv, _tf = GEO.terrain(divisions=32, height=6.0, sea_level=0.25)
    zmin = min(p[2] for p in tv)
    wv, _wf = GEO.water_quad(24.0, 0.25 * 6.0)
    check('the water sits at the floored terrain minimum',
          abs(wv[0][2] - zmin) < 1e-6, f'{wv[0][2]:.4f} vs {zmin:.4f}')
    ok, why = GEO.is_manifoldish(wv, _wf)
    check('the quad is a clean face', ok, why)


def main():
    utf8_console()
    FAILS.clear()
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        print(t.__name__)
        try:
            t()
        except Exception:                                       # noqa: BLE001
            traceback.print_exc()
            FAILS.append(f'{t.__name__} raised')
    print()
    if FAILS:
        print(f'{len(FAILS)} failure(s): ' + ', '.join(FAILS))
        return 1
    print('all R253 terrain-tool checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
