"""Volume containers -- real ray-marched volumes, and their cel cousins.

R222. Until now "volumetrics" meant the screen-space kinds: light
shafts smeared from bright pixels, the visible beam cones, Height Fog.
This module is the missing third thing -- an actual volume: a mesh
whose material ends in a VOLUME closure becomes a container, and the
renderer marches every camera ray through the container's bounding box
(the era's volume gizmos were boxes and spheres -- 3D Studio's Volume
Fog and Combustion, POV-Ray's media, Bryce's clouds all shaped a
procedural density inside a simple bound), accumulating absorption and
single scatter from the scene's own lamps.

What makes it Halcyon rather than a physics paper:

- The DENSITY IS A NODE CHAIN. Whatever feeds the volume node's
  Density is evaluated by the renderer's own graph evaluator at every
  march sample, with Generated coordinates spanning the container box
  -- so a Bozo/Marble/Cells chain shapes clouds exactly the way the
  era's procedural volumes worked, and an animated chain animates.
- The lamps are the scene's own lamp loop: falloffs, spot cones,
  cookies/gobos, light linking and per-lamp shadows all arrive through
  the same `lights.sample`/`visibility` roads every surface uses.
- Fixed-count midpoint marching, no jitter: banding at low step counts
  is the era's own slicing artefact (the cones module's doctrine), and
  every pixel is a pure function of its ray -- bands, workers and
  refine passes reproduce it bit for bit.
- The STYLIZED dials live on the Halcyon Volume node: Edge Threshold /
  Softness cut hard anime cloud edges out of soft density fields,
  Bands posterizes the scattered light into cel steps, and Shadow
  Tint applies the anime rule to volumes -- a shadowed region of the
  volume takes a COLOUR, never just darkness.

Computed on the CPU over the shared G-buffer in the render tail on
BOTH devices (the outline doctrine): the devices cannot disagree.
Traced reflection rays still see a container as an empty surface --
a named limit on the ledger, not a silent one.

R225 -- the field's morning after the first volumes: "there should be
multiple volume shader models, like the regular master shader. Also
there's no voxel count, all volumes are cubes regardless of the
model." Three answers on the Halcyon Volume node:

- MODEL: eight scattering laws (VOLUME_MODEL_ITEMS). Henyey-Greenstein
  (the default, the old behaviour), Uniform, a dual-lobe cloud phase,
  POV-Ray's three media types -- Rayleigh, Mie haze, Mie murky -- and
  two UNLIT models from 3D Studio's atmospherics: Volume Fog (the fog
  colour composited by density, no lamps) and Combustion (the Fire
  Effect: an inner-to-outer colour ramp by density, self-lit). The lit
  phases are all normalised to integrate to one over the sphere, so
  Color stays the single-scatter albedo whichever law is picked
  (POV-Ray's own factors are unnormalised and its media brighten by
  type; the shapes are POV's, the energy is consistent).
- SHAPE: the container marches the MESH now (per-container BVH,
  entry and exit by its own faces, up to four solid spans for concave
  shapes; a mesh that IS its bounding box -- every fog cube -- takes
  the slab road bitwise). BOX keeps the old gizmo; SPHERE and
  CYLINDER are the other two 3D Studio atmospheric gizmos, fitted to
  the bound.
- VOXELS: the density and colour chains (and a smoke grid) are read
  at the centres of an N^3 lattice over the bound instead of at the
  sample point -- the blocky voxel volume of the era's grid
  volumetrics, and a stylized look in its own right. 0 is continuous.
"""

import numpy as np

from . import lights as LI
from . import mathx as M

EPS = 1e-6

#: node types that can END a volume chain, and how to read them.
VOLUME_NODES = ('HALCYON_VolumeNode', 'ShaderNodeVolumePrincipled',
                'ShaderNodeVolumeScatter', 'ShaderNodeVolumeAbsorption')

#: R225: the Halcyon Volume node's Model menu -- (identifier, label,
#: tooltip). The evaluator's phase/emission laws key on the identifier;
#: the node's EnumProperty reads this table, so the two cannot drift.
VOLUME_MODEL_ITEMS = (
    ('HG', "Henyey-Greenstein",
     "The standard single-lobe phase: Anisotropy steers the scatter "
     "forward (+) or back (-), 0 is isotropic. The default, and the "
     "law every earlier Halcyon volume marched"),
    ('UNIFORM', "Uniform",
     "Isotropic scatter whatever Anisotropy says -- POV-Ray's media "
     "type 1, the flattest, most even fog"),
    ('DUAL', "Dual Lobe (Clouds)",
     "Two Henyey-Greenstein lobes averaged: a forward lobe at "
     "Anisotropy and a back lobe at half of it, negated -- the "
     "cloud phase of the real-time era, silver lining and back-glow "
     "in one. Set Anisotropy near 0.8"),
    ('RAYLEIGH', "Rayleigh",
     "The 1 + cos^2 phase of molecular scattering (POV-Ray media "
     "type 4): a soft symmetric glow, forward and back alike -- thin "
     "air, blue distance"),
    ('HAZE', "Mie Haze",
     "POV-Ray's mie_haze (type 2): a strong forward peak on an even "
     "base, 1 + 9((1 + cos)/2)^8. Sunlit haze that blooms around a "
     "lamp seen through it"),
    ('MURKY', "Mie Murky",
     "POV-Ray's mie_murky (type 3): a very sharp forward spike, "
     "1 + 50((1 + cos)/2)^32. Thick, dirty air -- a lamp shows as a "
     "tight halo, everything else stays dim"),
    ('FOG', "Volume Fog (Unlit)",
     "3D Studio's Volume Fog atmospheric: the fog COLOUR composited "
     "by density, no lamps at all -- exactly Color * (1 - "
     "transmittance). Bands still steps it; Absorption darkens it"),
    ('COMBUSTION', "Combustion (Fire)",
     "3D Studio's Fire Effect: self-lit, no lamps. The flame colour "
     "ramps from Color (the OUTER, sparse colour) to Emission Color "
     "(the INNER, dense colour) by density, glowing as bright as the "
     "density; Absorption sets how much the flames hide what is "
     "behind them (0 = pure additive fire). Feed Density a Cells or "
     "Bozo chain for the flames"),
)

#: R225: the container Shape menu.
VOLUME_SHAPE_ITEMS = (
    ('MESH', "Mesh",
     "March the container's own faces: entry and exit by ray-casting "
     "the mesh, so a sphere is a sphere and a torus a torus (up to "
     "four solid spans per ray; face winding tells inside from "
     "outside, so recalculate normals outward if a shape inverts). A "
     "mesh that IS its bounding box takes the box road"),
    ('BOX', "Box",
     "The bounding box of the mesh, the era's gizmo (every Halcyon "
     "volume before 1.69 marched this)"),
    ('SPHERE', "Sphere",
     "The ellipsoid inscribed in the bounding box -- 3D Studio's "
     "sphere gizmo. Fitted in world axes: rotate the mesh and the "
     "bound grows around it; pick Mesh for a rotated shape"),
    ('CYLINDER', "Cylinder",
     "The Z-axis cylinder inscribed in the bounding box -- 3D "
     "Studio's cylinder gizmo, capped at the box's top and bottom"),
)

#: the models that never ask the lamp loop (self-lit atmospherics)
UNLIT_MODELS = ('FOG', 'COMBUSTION')

#: R225: the largest Voxels lattice the node offers
MAX_VOXELS = 512


def volume_output(graph):
    """The volume node a material's OUTPUT actually links, or None.

    A material is a volume container when its Material Output's Volume
    input is linked (mixing several volume closures with shader-mix
    nodes is not resolved -- the first volume node upstream wins, and
    the tooltip says so).
    """
    if not isinstance(graph, dict):
        return None
    nodes = graph.get('nodes') or {}
    out_id = graph.get('output')
    out = nodes.get(out_id) if out_id is not None else None
    if out is None:
        for nd in nodes.values():
            if nd.get('bl_idname') == 'ShaderNodeOutputMaterial':
                out = nd
                break
    if out is None:
        return None
    link = None
    for s in out.get('inputs', ()):
        if s.get('name') == 'Volume':
            link = s.get('link')
            break
    if not link:
        return None
    seen = set()
    nid = link[0]
    while nid is not None and nid not in seen:
        seen.add(nid)
        nd = nodes.get(nid)
        if nd is None:
            return None
        if nd.get('bl_idname') in VOLUME_NODES:
            return nd
        # walk through pass-through shader nodes (Mix/Add): first input
        nxt = None
        for s in nd.get('inputs', ()):
            if s.get('link'):
                nxt = s['link'][0]
                break
        nid = nxt
    return None


def material_is_volume(mat):
    """True when the material shades as a VOLUME container (its output
    links a volume chain). Such a material draws no surface."""
    return volume_output(getattr(mat, 'graph', None)) is not None


def scene_volumes(scene):
    """[(mat_index, obj_index, volume_node, graph, bbox_min, bbox_max,
    tri_ids)] for every container the frame carries. The bound is the
    AABB of the triangles wearing the material -- the era's gizmo --
    split PER OBJECT when the mesh carries object ids (R223: two fog
    boxes sharing one material are two containers, not one joint bound
    marching the empty air between them). R225: `tri_ids` indexes the
    container's own triangles in scene.mesh.tris, for the Mesh shape."""
    out = []
    mesh = getattr(scene, 'mesh', None)
    mats = getattr(scene, 'materials', None) or []
    if mesh is None or mesh.mat_index is None or mesh.verts is None or \
            not len(mats):
        return out
    for vi, m in enumerate(mats):
        g = getattr(m, 'graph', None)
        vn = volume_output(g)
        if vn is None:
            continue
        tsel = np.nonzero(mesh.mat_index == vi)[0]
        if tsel.size == 0:
            continue
        if mesh.obj_index is not None:
            groups = [(int(oi), tsel[mesh.obj_index[tsel] == oi])
                      for oi in np.unique(mesh.obj_index[tsel])]
        else:
            groups = [(-1, tsel)]
        for oi, sub in groups:
            vids = np.unique(mesh.tris[sub].reshape(-1))
            pts = mesh.verts[vids]
            out.append((vi, oi, vn, g,
                        pts.min(axis=0).astype(np.float32),
                        pts.max(axis=0).astype(np.float32),
                        sub.astype(np.int64)))
    return out


def node_model(vnode):
    """The Halcyon Volume node's Model (other volume nodes: HG)."""
    if vnode.get('bl_idname') != 'HALCYON_VolumeNode':
        return 'HG'
    m = str((vnode.get('props') or {}).get('model', 'HG') or 'HG').upper()
    return m if m in {k for k, _l, _d in VOLUME_MODEL_ITEMS} else 'HG'


def node_shape(vnode):
    """The container Shape (Blender's own volume nodes: MESH)."""
    s = str((vnode.get('props') or {}).get('shape', 'MESH') or 'MESH').upper()
    return s if s in {k for k, _l, _d in VOLUME_SHAPE_ITEMS} else 'MESH'


def node_voxels(vnode):
    """The Voxels lattice size, 0 for continuous."""
    try:
        v = int((vnode.get('props') or {}).get('voxels', 0) or 0)
    except (TypeError, ValueError):
        v = 0
    return int(np.clip(v, 0, MAX_VOXELS))


def _is_axis_box(pts, lo, hi):
    """True when the container's vertices are exactly the eight corners
    of its bound (a fog cube): the Mesh shape takes the slab road, so
    every box container marches bitwise what it marched before."""
    if pts.shape[0] < 8:
        return False
    at_lo = pts == lo[None, :]
    at_hi = pts == hi[None, :]
    if not (at_lo | at_hi).all():
        return False
    # split vertices (a cube exported with UV seams carries 24) are
    # still eight corners
    keys = (at_hi.astype(np.int32) * np.array([1, 2, 4], np.int32)).sum(axis=1)
    return np.unique(keys).size == 8


def _quadratic_span(a, b, c):
    """Real roots of a t^2 + b t + c as (t0, t1, hit), vectorised."""
    disc = b * b - 4.0 * a * c
    hit = disc > 0.0
    sq = np.sqrt(np.maximum(disc, 0.0))
    safe_a = np.where(np.abs(a) < 1e-12, 1e-12, a)
    r0 = (-b - sq) / (2.0 * safe_a)
    r1 = (-b + sq) / (2.0 * safe_a)
    return np.minimum(r0, r1), np.maximum(r0, r1), hit


def _sphere_span(origin, dirs, lo, hi):
    """Ray/ellipsoid inscribed in the bound: unit sphere in the box's
    normalised frame."""
    c = (lo + hi) * 0.5
    r = np.maximum((hi - lo) * 0.5, 1e-6)
    o = ((origin - c) / r)[None, :].astype(np.float64)
    d = (dirs / r[None, :]).astype(np.float64)
    a = (d * d).sum(axis=1)
    b = 2.0 * (o * d).sum(axis=1)
    cc = (o * o).sum(axis=1) - 1.0
    t0, t1, hit = _quadratic_span(a, b, cc)
    t0 = np.maximum(t0, 0.0)
    return t0.astype(np.float32), t1.astype(np.float32), \
        hit & (t1 > t0 + EPS)


def _cylinder_span(origin, dirs, lo, hi):
    """Ray/Z cylinder inscribed in the bound: an ellipse in xy (the
    box's normalised frame) capped by the z slab."""
    c = (lo + hi) * 0.5
    r = np.maximum((hi - lo) * 0.5, 1e-6)
    o = ((origin - c) / r)[None, :].astype(np.float64)
    d = (dirs / r[None, :]).astype(np.float64)
    a = d[:, 0] ** 2 + d[:, 1] ** 2
    b = 2.0 * (o[:, 0] * d[:, 0] + o[:, 1] * d[:, 1])
    cc = o[:, 0] ** 2 + o[:, 1] ** 2 - 1.0
    axial = a < 1e-12
    t0, t1, hit = _quadratic_span(np.where(axial, 1.0, a),
                                  np.where(axial, 0.0, b),
                                  np.where(axial, -1.0, cc))
    # a ray running along the axis: inside the ellipse forever, or
    # missing it forever
    inside_xy = cc <= 0.0
    t0 = np.where(axial, -1e30, t0)
    t1 = np.where(axial, 1e30, t1)
    hit = np.where(axial, inside_xy, hit)
    dz = np.where(np.abs(d[:, 2]) < 1e-12, 1e-12, d[:, 2])
    za = (-1.0 - o[:, 2]) / dz
    zb = (1.0 - o[:, 2]) / dz
    t0 = np.maximum(np.maximum(t0, np.minimum(za, zb)), 0.0)
    t1 = np.minimum(t1, np.maximum(za, zb))
    return t0.astype(np.float32), t1.astype(np.float32), \
        hit & (t1 > t0 + EPS)


#: the Mesh shape keeps this many solid spans per pixel: a concave
#: container (a torus seen edge-on is two) marches all of them
MESH_MAX_SPANS = 4


def _mesh_spans(cverts, ctris, vp, w, h, eye, inv, pix, tmax, t1_box):
    """Solid spans of a closed container mesh along the camera rays.

    Depth peeling by the engine's own rasteriser: every fragment of
    the container's faces is collected (an A-buffer with no depth
    test), sorted per pixel by distance, and read as a state machine
    -- a front face opens a span, the next back face closes it; a
    same-facing run collapses to its first fragment (the two faces at
    a shared edge both cover the edge pixel), so facings alternate. A
    pixel whose nearest fragment is a BACK face starts inside the
    container: its first span opens at 0. A span left open by the
    surface the ray ends on (or by a hole in the mesh) closes at that
    surface, bounded by the box exit.

    The camera rays ARE the pixels, which is why rasterising beats
    casting here: the whole frame's entries and exits cost one small
    raster pass, not two million BVH walks.

    `pix` are the flat pixel indices the march is working on, `tmax`
    and `t1_box` their surface distances and slab exits. Returns
    (len(pix), MESH_MAX_SPANS, 2): entry/exit distances along the
    normalised ray, inf where absent.
    """
    from . import raster as RA
    S = MESH_MAX_SPANS
    npx = pix.size
    spans = np.full((npx, S, 2), np.inf, np.float32)
    if ctris.shape[0] == 0 or npx == 0:
        return spans
    fl = RA.FragmentList()
    RA.rasterize(cverts, ctris, vp, w, h, cull='NONE', snap=0.0,
                 depth_bits=32, frags=fl, depth_write=False,
                 depth_test=False)
    px, py, tri, zz, _bary, _front = fl.finish()
    if px.size == 0:
        return spans
    # the fragment's distance along its pixel's ray: unproject as the
    # march unprojects the surface depth, so the two agree exactly
    fx = (px.astype(np.float32) + 0.5) / w * 2.0 - 1.0
    fy = (py.astype(np.float32) + 0.5) / h * 2.0 - 1.0
    hp = np.stack([fx, fy, zz.astype(np.float32),
                   np.ones(px.size, np.float32)], axis=1) @ inv.T
    P = hp[:, :3] / np.where(np.abs(hp[:, 3:4]) < 1e-9, 1e-9, hp[:, 3:4])
    rel = P - eye[None, :]
    t = np.linalg.norm(rel, axis=1).astype(np.float32)
    # facing from the face's own winding against the ray (not the
    # raster's flag, whose sign convention is the G-buffer's business):
    # a face whose geometric normal points along the ray is seen from
    # behind -- the ray is leaving the container through it
    e1 = cverts[ctris[:, 1]] - cverts[ctris[:, 0]]
    e2 = cverts[ctris[:, 2]] - cverts[ctris[:, 0]]
    fn = np.cross(e1, e2)
    front = np.einsum('ij,ij->i', fn[tri], rel) < 0.0
    fpix = py.astype(np.int64) * w + px.astype(np.int64)
    # only the pixels the march holds
    want = np.zeros(w * h, bool)
    want[pix] = True
    keep = want[fpix]
    if not keep.any():
        return spans
    fpix, t, front = fpix[keep], t[keep], front[keep]
    o = np.lexsort((t, fpix))
    fpix, t, front = fpix[o], t[o], front[o]
    # collapse same-facing runs within a pixel to their first fragment
    same_pix = np.concatenate([[False], fpix[1:] == fpix[:-1]])
    same_face = np.concatenate([[False], front[1:] == front[:-1]])
    k1 = ~(same_pix & same_face)
    fpix, t, front = fpix[k1], t[k1], front[k1]
    starts = np.concatenate([[True], fpix[1:] != fpix[:-1]])
    idx = np.arange(fpix.size)
    gstart = np.maximum.accumulate(np.where(starts, idx, 0))
    inside0 = ~front[gstart]                  # nearest fragment is a back
    slot = (idx - gstart) + inside0.astype(np.int64)
    # slot parity: even = entry of span slot//2, odd = its exit
    where = np.full(w * h, -1, np.int64)
    where[pix] = np.arange(npx)
    row = where[fpix]
    ok = slot < 2 * S
    flat = spans.reshape(npx, 2 * S)
    flat[row[ok], slot[ok]] = t[ok]
    # an inside start opens its first span at the eye
    ins = starts & inside0
    flat[row[ins], 0] = 0.0
    spans = flat.reshape(npx, S, 2)
    # spans that begin past the surface are behind it: gone
    ent = spans[:, :, 0]
    behind = np.isfinite(ent) & (ent >= tmax[:, None])
    spans[behind] = np.inf
    # a span with an entry but no exit closes at the ray's end
    open_ = np.isfinite(spans[:, :, 0]) & ~np.isfinite(spans[:, :, 1])
    if open_.any():
        end = np.minimum(tmax, t1_box())
        spans[:, :, 1] = np.where(open_, end[:, None], spans[:, :, 1])
    return spans


def container_spans(shape, mesh, tri_ids, eye, dirs, tmax, lo, hi,
                    frame=None):
    """(t0, t1, live, spans) for one container: the march segment per
    ray (first entry to last exit, clipped by the surface) and, for a
    Mesh container with several solid spans, the per-span table the
    march masks samples against (None when one span covers it).

    `frame` = (vp, w, h, inv, pix) is the Mesh shape's raster frame:
    the view-projection, the frame size, its inverse and the flat
    pixel indices the rays belong to. Without it a Mesh container
    falls back to its box (the analytic shapes never need it).
    """
    if shape == 'MESH':
        vids = np.unique(mesh.tris[tri_ids].reshape(-1))
        pts = mesh.verts[vids]
        if _is_axis_box(pts, lo, hi) or frame is None:
            shape = 'BOX'
    if shape == 'BOX':
        t0, t1, hit = _slab(eye, dirs, lo, hi)
    elif shape == 'SPHERE':
        t0, t1, hit = _sphere_span(eye, dirs, lo, hi)
    elif shape == 'CYLINDER':
        t0, t1, hit = _cylinder_span(eye, dirs, lo, hi)
    else:
        vp, w, h, inv, pix = frame
        remap = np.full(int(vids.max()) + 1, -1, np.int64)
        remap[vids] = np.arange(vids.size)
        ctris = remap[mesh.tris[tri_ids]].astype(np.int32)
        # the slab exit is only read to close a span the mesh left
        # open (a hole, or the surface): computed on demand
        spans = _mesh_spans(pts.astype(np.float32), ctris, vp, w, h, eye,
                            inv, pix, tmax,
                            lambda: _slab(eye, dirs, lo, hi)[1])
        ent = spans[:, :, 0]
        has = np.isfinite(ent)
        t0 = np.where(has[:, 0], ent[:, 0], 0.0).astype(np.float32)
        ex = np.where(has, spans[:, :, 1], -np.inf)
        t1 = np.minimum(ex.max(axis=1), tmax).astype(np.float32)
        live = has[:, 0] & (t1 > t0 + EPS)
        multi = has[:, 1:].any(axis=1)
        return t0, t1, live, (spans if multi.any() else None)
    t1 = np.minimum(t1, tmax)
    live = hit & (t1 > t0 + EPS)
    return t0, t1, live, None


def _grid_sample(grid, res, gen):
    """Trilinear sample of a smoke grid at container coordinates.

    `grid` is the flat Mantaflow layout (x fastest: i + j*rx + k*rx*ry),
    `res` (rx, ry, rz), `gen` (n,3) in [0,1]^3 across the domain box.
    """
    rx, ry, rz = (max(int(v), 1) for v in res)
    vol3 = np.asarray(grid, np.float32).reshape(rz, ry, rx)
    gx = np.clip(gen[:, 0], 0.0, 1.0) * (rx - 1)
    gy = np.clip(gen[:, 1], 0.0, 1.0) * (ry - 1)
    gz = np.clip(gen[:, 2], 0.0, 1.0) * (rz - 1)
    x0 = np.floor(gx).astype(np.int64)
    y0 = np.floor(gy).astype(np.int64)
    z0 = np.floor(gz).astype(np.int64)
    tx = (gx - x0).astype(np.float32)
    ty = (gy - y0).astype(np.float32)
    tz = (gz - z0).astype(np.float32)
    x1 = np.minimum(x0 + 1, rx - 1)
    y1 = np.minimum(y0 + 1, ry - 1)
    z1 = np.minimum(z0 + 1, rz - 1)
    c000 = vol3[z0, y0, x0]
    c100 = vol3[z0, y0, x1]
    c010 = vol3[z0, y1, x0]
    c110 = vol3[z0, y1, x1]
    c001 = vol3[z1, y0, x0]
    c101 = vol3[z1, y0, x1]
    c011 = vol3[z1, y1, x0]
    c111 = vol3[z1, y1, x1]
    a = c000 + (c100 - c000) * tx
    b = c010 + (c110 - c010) * tx
    c = c001 + (c101 - c001) * tx
    d = c011 + (c111 - c011) * tx
    e = a + (b - a) * ty
    f = c + (d - c) * ty
    return (e + (f - e) * tz).astype(np.float32)


def _slab(origin, dirs, lo, hi):
    """Ray/AABB: (t0, t1, hit) per ray, era-plain slab test."""
    inv = 1.0 / np.where(np.abs(dirs) < 1e-12,
                         np.where(dirs < 0, -1e-12, 1e-12), dirs)
    ta = (lo[None, :] - origin[None, :]) * inv
    tb = (hi[None, :] - origin[None, :]) * inv
    tmin = np.minimum(ta, tb).max(axis=1)
    tmax = np.maximum(ta, tb).min(axis=1)
    t0 = np.maximum(tmin, 0.0)
    t1 = tmax
    return t0, t1, (t1 > t0 + EPS)


def _hg_phase(g, cos_t):
    """Henyey-Greenstein, isotropic at g = 0 (1/4pi folded)."""
    g = np.clip(g, -0.95, 0.95)
    one = np.float32(1.0 / (4.0 * np.pi))
    denom = 1.0 + g * g - 2.0 * g * cos_t
    return one * (1.0 - g * g) / np.maximum(denom, 1e-4) ** 1.5


#: R225: the lit models' phase normalisations -- each law integrates
#: to exactly 1 over the sphere, closed form:
#:   Rayleigh  int (1 + cos^2) dw = 16 pi / 3
#:   Mie haze  int (1 + 9 u^8)  dw = 8 pi          (u = (1 + cos) / 2)
#:   Mie murky int (1 + 50 u^32) dw = 332 pi / 33
_RAYLEIGH_NORM = np.float32(3.0 / (16.0 * np.pi))
_HAZE_NORM = np.float32(1.0 / (8.0 * np.pi))
_MURKY_NORM = np.float32(33.0 / (332.0 * np.pi))


def phase(model, g, cos_t):
    """The scattering phase of a lit model at cos(theta) between the
    light direction and the view ray, (n,) -- see VOLUME_MODEL_ITEMS.
    Every law integrates to one over the sphere."""
    if model == 'UNIFORM':
        return np.full(np.shape(cos_t), 1.0 / (4.0 * np.pi), np.float32)
    if model == 'DUAL':
        return 0.5 * (_hg_phase(g, cos_t) + _hg_phase(-0.5 * g, cos_t))
    if model == 'RAYLEIGH':
        return (_RAYLEIGH_NORM * (1.0 + cos_t * cos_t)).astype(np.float32)
    if model == 'HAZE':
        u = np.clip(0.5 * (1.0 + cos_t), 0.0, 1.0)
        return (_HAZE_NORM * (1.0 + 9.0 * u ** 8)).astype(np.float32)
    if model == 'MURKY':
        u = np.clip(0.5 * (1.0 + cos_t), 0.0, 1.0)
        return (_MURKY_NORM * (1.0 + 50.0 * u ** 32)).astype(np.float32)
    return _hg_phase(g, cos_t)


def voxel_snap(gen, voxels):
    """Generated coordinates snapped to the centres of an N^3 lattice
    over the bound (N = 0: unchanged). Half-open cells: gen 1.0 lands
    in the last cell, never a phantom N-th one."""
    if voxels <= 0:
        return gen
    nv = np.float32(voxels)
    cell = np.floor(np.clip(gen, 0.0, 1.0 - 1e-6) * nv)
    return ((cell + np.float32(0.5)) / nv).astype(np.float32)


def _read(ev, node, name, kind, default, n):
    """A socket that may be missing from an old graph reads its
    default here, never zero -- volumes are new, graphs are old."""
    from .nodeeval import coerce
    for s in node.get('inputs', ()):
        if s.get('name') == name:
            return coerce(ev.input(node, name, kind), kind, n)
    if kind == 'RGBA':
        d = np.zeros((n, 4), np.float32)
        d[:] = np.asarray(list(default) + [1.0] * (4 - len(default)),
                          np.float32)[None, :]
        return d
    return np.full(n, float(default), np.float32)


class VolumeFrame:
    """R226: what a volume chain may ask about the frame it marches in.

    The R222 sampler handed the graph evaluator a bare context -- the
    sample's position, Generated coordinates and the settings -- so a
    Texture Coordinate node's Window and Camera outputs read zero, an
    Object output read world space, the clock stood at frame 1 (an
    animated density chain froze), and an Object Info node knew
    nothing. Every camera ray IS a pixel, so the march carries the
    pixel identity and the eye down to each sample: a drawn cloud
    projected from the camera through the fog (Window), object-space
    chains that follow a moving container (Object), scrolling and
    clocked chains that animate.
    """
    __slots__ = ('eye', 'width', 'height', 'time', 'frame', 'obj',
                 'view_matrix')

    def __init__(self, eye=None, width=0, height=0, time=0.0, frame=1,
                 obj=None, view_matrix=None):
        self.eye = None if eye is None else np.asarray(eye, np.float32)
        self.width = int(width)
        self.height = int(height)
        self.time = float(time)
        self.frame = int(frame)
        self.obj = obj
        self.view_matrix = view_matrix


def _fill_volume_context(ctx, P, lo, hi, frame, pix):
    """The per-sample context a volume chain reads (see VolumeFrame)."""
    n = P.shape[0]
    ctx.is_volume = True
    ctx.object_loc = np.broadcast_to(((lo + hi) * 0.5)[None, :],
                                     (n, 3)).astype(np.float32).copy()
    if frame is None:
        return
    if frame.eye is not None:
        ctx.camera_pos = frame.eye
        ctx.I = M.normalize(P - frame.eye[None, :]).astype(np.float32)
    if pix is not None and frame.width > 0:
        ctx.px = (pix % frame.width).astype(np.int64)
        ctx.py = (pix // frame.width).astype(np.int64)
        ctx.width = frame.width
        ctx.height = frame.height
    ctx.time = frame.time
    ctx.frame = frame.frame
    ctx.view_matrix = frame.view_matrix
    ob = frame.obj
    if ob is not None:
        mw = getattr(ob, 'matrix_world', None)
        if mw is not None:
            try:
                mw = np.asarray(mw, np.float64).reshape(4, 4)
                inv = np.linalg.inv(mw).astype(np.float32)
                ctx.object_matrix_inv = np.broadcast_to(
                    inv[None, :, :], (n, 4, 4))
                ctx.object_loc = np.broadcast_to(
                    mw[:3, 3].astype(np.float32)[None, :], (n, 3)).copy()
            except (np.linalg.LinAlgError, ValueError):
                pass
        col = getattr(ob, 'color', None)
        if col is not None:
            try:
                c4 = np.asarray(list(col)[:4], np.float32)
                if c4.size == 4:
                    ctx.object_color = np.broadcast_to(c4[None, :],
                                                       (n, 4)).copy()
            except (TypeError, ValueError):
                pass
        ctx.object_index = np.full(n, float(getattr(ob, 'index', 0) or 0),
                                   np.float32)
        ctx.object_random = np.full(
            n, float(getattr(ob, 'random', 0.0) or 0.0), np.float32)


def _sample_volume(vnode, graph, textures, settings, P, lo, hi,
                   voxels=0, frame=None, pix=None):
    """Evaluate one volume node at world points P.

    Returns dict with sigma (n,3) extinction, scatter (n,3) albedo x
    density, emission (n,3), g (n,), the model, the stylized dials,
    `gen`, the Generated coordinate the chains were read at (voxel-
    snapped when Voxels is set -- the grid sampler reads the same one),
    and `problems`: the evaluator's named failures (a node that raised,
    an output that resolved to nothing) so the march can print them
    once instead of shading a silent zero.

    `frame` is a VolumeFrame (eye, frame size, clock, container object)
    and `pix` the flat pixel index of each sample's ray: together they
    give the chain Window/Camera/Object coordinates, the clock and the
    Object Info fields (R226).
    """
    from .nodeeval import RGBA, VALUE, GraphEvaluator, ShadeContext
    n = P.shape[0]
    ctx = ShadeContext(n)
    size = np.maximum(hi - lo, 1e-6)
    gen = ((P - lo[None, :]) / size[None, :]).astype(np.float32)
    if voxels > 0:
        # R225: every chain reads at the voxel's centre -- the point's
        # own position included, so a Bozo fed world coordinates
        # blocks up exactly like one fed Generated
        gen = voxel_snap(gen, voxels)
        P = (lo[None, :] + gen * size[None, :]).astype(np.float32)
    ctx.P = P.astype(np.float32)
    ctx.generated = gen
    ctx.uv = gen[:, :2].astype(np.float32)
    _fill_volume_context(ctx, ctx.P, lo, hi, frame, pix)
    ctx.settings = settings
    ev = GraphEvaluator(graph, ctx, images=textures or {})
    kind = vnode.get('bl_idname')
    model = node_model(vnode)
    density = np.maximum(_read(ev, vnode, 'Density', VALUE, 1.0, n), 0.0)
    out = {'bands': 0.0, 'tint': None, 'tint_amount': 0.0, 'gen': gen,
           'model': model}
    if kind == 'ShaderNodeVolumeAbsorption':
        color = np.clip(_read(ev, vnode, 'Color', RGBA,
                              (0.8, 0.8, 0.8), n)[:, :3], 0.0, 1.0)
        out['sigma'] = density[:, None] * (1.0 - color)
        out['scatter'] = np.zeros((n, 3), np.float32)
        out['emission'] = np.zeros((n, 3), np.float32)
        out['g'] = np.zeros(n, np.float32)
        out['problems'] = ev.unsupported | ev.unresolved
        return out
    color = np.clip(_read(ev, vnode, 'Color', RGBA,
                          (0.8, 0.8, 0.8), n)[:, :3], 0.0, 1.0)
    g = np.clip(_read(ev, vnode, 'Anisotropy', VALUE, 0.0, n),
                -0.95, 0.95)
    emission = np.zeros((n, 3), np.float32)
    e_col = None
    if kind in ('ShaderNodeVolumePrincipled', 'HALCYON_VolumeNode'):
        e_str = np.maximum(_read(ev, vnode, 'Emission Strength', VALUE,
                                 0.0, n), 0.0)
        e_col = np.clip(_read(ev, vnode, 'Emission Color', RGBA,
                              (1.0, 1.0, 1.0), n)[:, :3], 0.0, None)
        emission = e_col * e_str[:, None]
    if kind == 'HALCYON_VolumeNode':
        # the stylized dials -- and the hard anime cloud edge, cut out
        # of the soft density field BEFORE lighting sees it
        th = _read(ev, vnode, 'Edge Threshold', VALUE, 0.0, n)
        soft = np.maximum(_read(ev, vnode, 'Edge Softness', VALUE,
                                0.25, n), 1e-4)
        use = th > 1e-6
        if use.any():
            t = np.clip((density - (th - soft)) / (2.0 * soft), 0.0, 1.0)
            edge = t * t * (3.0 - 2.0 * t)
            density = np.where(use, edge * np.maximum(th, 1.0), density)
        absorb = np.maximum(_read(ev, vnode, 'Absorption', VALUE,
                                  0.0, n), 0.0)
        if model == 'COMBUSTION':
            # 3D Studio's Fire Effect: the flames hide what is behind
            # them only as much as Absorption says (0: additive fire)
            out['sigma'] = (density * absorb)[:, None] * \
                np.ones((1, 3), np.float32)
        else:
            out['sigma'] = (density + absorb)[:, None] * \
                np.ones((1, 3), np.float32)
        out['bands'] = float(np.asarray(
            _read(ev, vnode, 'Bands', VALUE, 0.0, n)).reshape(-1)[0])
        out['tint'] = np.clip(_read(ev, vnode, 'Shadow Tint', RGBA,
                                    (0.35, 0.3, 0.5), n)[:, :3],
                              0.0, 1.0)
        out['tint_amount'] = np.clip(np.asarray(_read(
            ev, vnode, 'Tint Amount', VALUE, 0.0, n)), 0.0, 1.0)
    else:
        out['sigma'] = density[:, None] * np.ones((1, 3), np.float32)
    if model == 'FOG':
        # 3D Studio's Volume Fog: the fog colour composited by density
        # -- integrating colour * density * T along the ray IS
        # colour * (1 - transmittance), the atmospheric's own blend
        out['scatter'] = np.zeros((n, 3), np.float32)
        emission = emission + color * density[:, None]
    elif model == 'COMBUSTION':
        # the Fire Effect's colour ramp: Outer (Color) where the
        # density is sparse, Inner (Emission Color) where it is
        # dense, glowing as bright as the density -- 3DS's own
        # "Density sets the opacity and brightness"
        f = np.clip(density, 0.0, 1.0)[:, None]
        inner = e_col if e_col is not None else np.ones((n, 3), np.float32)
        flame = color + (inner - color) * f
        out['scatter'] = np.zeros((n, 3), np.float32)
        emission = emission + flame * density[:, None]
    else:
        out['scatter'] = color * density[:, None]
    out['emission'] = emission.astype(np.float32)
    out['g'] = g
    out['problems'] = ev.unsupported | ev.unresolved
    return out


def march(img, scene, st, gbuf, vp, eye, w, h, bvh=None, sel_mask=None,
          textures=None, view=None):
    """Composite every volume container over the finished frame.

    Per pixel: reconstruct the ray, clip the container's slab segment
    by the surface depth, midpoint-march `volume_steps` samples,
    accumulate rgb transmittance (Beer-Lambert, per channel -- an
    Absorption volume tints what shows through it) and single scatter
    from every lamp through the standard sample/visibility roads.
    Deterministic per pixel; sel_mask restricts to a refine pass's
    pixels with identical values.
    """
    vols = scene_volumes(scene)
    if not vols:
        return img
    steps = int(np.clip(getattr(st, 'volume_steps', 48), 4, 256))
    shadows_on = bool(getattr(st, 'volume_shadows', True)) and \
        bool(getattr(st, 'shadows', True))
    inv = np.linalg.inv(np.asarray(vp, np.float64)).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    nx = (xx.ravel().astype(np.float32) + 0.5) / w * 2.0 - 1.0
    ny = (yy.ravel().astype(np.float32) + 0.5) / h * 2.0 - 1.0
    zz = gbuf.depth.reshape(-1).astype(np.float32)
    sel = None
    pix = np.arange(w * h, dtype=np.int64)
    if sel_mask is not None:
        sel = np.asarray(sel_mask, bool).reshape(-1)
        if not sel.any():
            return img
        nx, ny, zz = nx[sel], ny[sel], zz[sel]
        pix = np.nonzero(sel)[0].astype(np.int64)
    npx = nx.size
    one = np.ones(npx, np.float32)
    far = np.stack([nx, ny, one, one], axis=1) @ inv.T
    far = far[:, :3] / np.where(np.abs(far[:, 3:4]) < 1e-9, 1e-9,
                                far[:, 3:4])
    dirs = M.normalize(far - eye[None, :])
    covered = np.isfinite(zz) & (zz < 1e11)
    dist = np.full(npx, 1e9, np.float32)
    if covered.any():
        hit_h = np.stack([nx[covered], ny[covered], zz[covered],
                          one[covered]], axis=1) @ inv.T
        hit = hit_h[:, :3] / np.where(np.abs(hit_h[:, 3:4]) < 1e-9,
                                      1e-9, hit_h[:, 3:4])
        dist[covered] = np.linalg.norm(hit - eye[None, :], axis=1)

    lights_all = [l for l in (getattr(scene, 'lights', None) or ())
                  if str(getattr(l, 'type', '')).upper() != 'AMBIENT']
    # soft ray shadows draw whole-lane scalars: each (volume, step,
    # light) triple gets its OWN stream derived arithmetically from the
    # seed, so the draws are a pure function of the triple -- a band
    # that skips empty steps, an early-out, a refine pass: all read the
    # same numbers as the whole frame. Never a shared sequential stream.
    base_seed = int(getattr(st, 'seed', 0) or 0)

    def _vol_rng(vol_i, step_k, light_i):
        s = (base_seed * 2654435761 + vol_i * 97911
             + step_k * 8191 + light_i * 131071 + 12043) & 0x7fffffff
        return np.random.RandomState(s)
    objects = getattr(scene, 'objects', None) or []
    mesh = getattr(scene, 'mesh', None)
    mats = getattr(scene, 'materials', None) or []
    # R226: what the chains may ask about the frame -- the eye, the
    # pixel grid, the clock -- and where their failures are told
    view_m = None if view is None else np.asarray(view, np.float32)
    clock = (float(getattr(scene, 'time', 0.0) or 0.0),
             int(getattr(scene, 'frame', 1) or 1))
    problems = {}
    T_total = np.ones((npx, 3), np.float32)
    scatter_total = np.zeros((npx, 3), np.float32)
    for _vi, _oi, vnode, graph, lo, hi, tri_ids in vols:
        # R223: a fluid-sim smoke grid on the container's OBJECT is a
        # density field -- the sim's own voxels, trilinear, multiplying
        # whatever the node chain says (Density 1 = the raw sim)
        grid_spec = None
        ob = objects[_oi] if 0 <= _oi < len(objects) else None
        if ob is not None:
            grid_spec = getattr(ob, 'smoke_grid', None)
            if not (isinstance(grid_spec, dict)
                    and grid_spec.get('density') is not None
                    and grid_spec.get('res') is not None):
                grid_spec = None
        vframe = VolumeFrame(eye=eye, width=w, height=h, time=clock[0],
                             frame=clock[1], obj=ob, view_matrix=view_m)
        shape = node_shape(vnode)
        voxels = node_voxels(vnode)
        model = node_model(vnode)
        # R225: the container's SHAPE decides the march segment -- the
        # mesh's own faces by default, the era's gizmos on request
        t0, t1, live0, spans = container_spans(
            shape, mesh, tri_ids, eye, dirs, dist, lo, hi,
            frame=(vp, w, h, inv, pix))
        if not live0.any():
            continue
        idx = np.nonzero(live0)[0]
        seg0 = t0[idx]
        seg1 = t1[idx]
        d_idx = dirs[idx]
        n_live = idx.size
        dt = (seg1 - seg0) / steps
        T = np.ones((n_live, 3), np.float32)
        acc = np.zeros((n_live, 3), np.float32)
        bands = 0.0
        # a concave Mesh container: the rays that cross MORE than one
        # solid span mask their samples against the span table -- a
        # sample in the gap between two spans is empty air, skipped
        # rather than evaluated (identical to a zero density there).
        # Single-span rays never pay for the test.
        mrows = None
        if spans is not None:
            sp_idx = spans[idx]
            mrows = np.nonzero(np.isfinite(sp_idx[:, 1, 0]))[0]
            sp_multi = sp_idx[mrows]
            if mrows.size == 0:
                mrows = None
        for k in range(steps):
            tmid = seg0 + (k + 0.5) * dt
            P = eye[None, :] + d_idx * tmid[:, None]
            # the early-out: a fully absorbed lane stops paying for
            # evaluation (peak-performance rule; values identical, the
            # lane's remaining contribution is provably < 1e-4)
            liveT = T.max(axis=1) > 1e-4
            if mrows is not None:
                tm = tmid[mrows][:, None]
                inside = ((tm >= sp_multi[:, :, 0])
                          & (tm < sp_multi[:, :, 1])).any(axis=1)
                liveT[mrows] &= inside
            if not liveT.any():
                if mrows is None:
                    break
                continue
            Pv = P[liveT]
            sm = _sample_volume(vnode, graph, textures, st, Pv, lo, hi,
                                voxels=voxels, frame=vframe,
                                pix=pix[idx][liveT])
            bands = sm['bands'] or bands
            if sm['problems']:
                problems.setdefault(_vi, set()).update(sm['problems'])
            if grid_spec is not None:
                gfac = _grid_sample(grid_spec['density'],
                                    grid_spec['res'], sm['gen'])
                sm['sigma'] = sm['sigma'] * gfac[:, None]
                sm['scatter'] = sm['scatter'] * gfac[:, None]
                sm['emission'] = sm['emission'] * gfac[:, None]
            sig = sm['sigma'] * dt[liveT, None]
            step_T = np.exp(-sig)
            sc_alb = sm['scatter']
            add = np.zeros((Pv.shape[0], 3), np.float32)
            if sc_alb.max() > 0.0 and lights_all and \
                    model not in UNLIT_MODELS:
                dv = d_idx[liveT]
                for li, light in enumerate(lights_all):
                    L, rad, ldist = LI.sample(light, Pv, st)
                    if shadows_on:
                        vis = LI.visibility(
                            light, Pv, L, L, ldist, st, bvh,
                            rng=_vol_rng(_vi, k, li))
                    else:
                        vis = np.ones(Pv.shape[0], np.float32)
                    if sm['tint'] is not None and \
                            np.max(sm['tint_amount']) > 0.0:
                        # the anime rule for volumes: a shadowed region
                        # takes a COLOUR, never just darkness
                        shade = sm['tint'] + \
                            (1.0 - sm['tint']) * vis[:, None]
                        amt = sm['tint_amount'][:, None]
                        vis3 = vis[:, None] * (1.0 - amt) + shade * amt
                    else:
                        vis3 = vis[:, None]
                    ph = phase(model, sm['g'],
                               np.einsum('ij,ij->i', L, dv))
                    add += rad * vis3 * (ph * dt[liveT])[:, None] \
                        * sc_alb
            add = add + sm['emission'] * dt[liveT, None]
            acc[liveT] += T[liveT] * add
            T[liveT] = T[liveT] * step_T
        if bands and bands >= 2.0:
            # cel steps quantize the LUMINANCE and keep the hue -- a
            # per-channel posterize fringes rainbow at every band edge
            # (channels cross their steps at different depths). Round
            # to the nearest step, so low haze lands on its band
            # instead of vanishing into zero.
            b = float(bands)
            lum = (acc[:, 0] * np.float32(0.2126)
                   + acc[:, 1] * np.float32(0.7152)
                   + acc[:, 2] * np.float32(0.0722))
            q = np.floor(np.clip(lum, 0.0, 4.0) * b + 0.5) / b
            scale = q / np.maximum(lum, 1e-6)
            acc = acc * scale[:, None]
        Tw = np.ones((npx, 3), np.float32)
        aw = np.zeros((npx, 3), np.float32)
        Tw[idx] = T
        aw[idx] = acc
        scatter_total = scatter_total * Tw + aw
        T_total = T_total * Tw

    if problems:
        # the evaluator's failures, BY NAME, once per frame per
        # container material -- a chain that raised or an output that
        # resolved to nothing reads zero, and a zero density is an
        # invisible volume: the field's "the textures make the volume
        # vanish" must never be silent again
        for _vi in sorted(problems):
            mname = getattr(mats[_vi], 'name', f'material {_vi}') \
                if _vi < len(mats) else f'material {_vi}'
            msg = (f"volume '{mname}': its chain has problems -- "
                   + '; '.join(sorted(problems[_vi])))
            print('[Halcyon] ' + msg)
            try:
                lst = getattr(scene, 'unsupported', None)
                if isinstance(lst, list) and msg not in lst:
                    lst.append(msg)
            except Exception:                                   # noqa: BLE001
                pass

    changed = (T_total.min(axis=1) < 1.0 - 1e-6) | \
        (scatter_total.max(axis=1) > 1e-6)
    if not changed.any():
        return img
    if sel is None:
        flat = img.reshape(-1, 4)
    else:
        flat = img.reshape(-1, 4)
        full_T = np.ones((flat.shape[0], 3), np.float32)
        full_s = np.zeros((flat.shape[0], 3), np.float32)
        full_T[sel] = T_total
        full_s[sel] = scatter_total
        T_total, scatter_total = full_T, full_s
    a_vol = 1.0 - T_total.mean(axis=1)
    flat[:, :3] = flat[:, :3] * T_total + scatter_total
    flat[:, 3] = np.clip(a_vol + flat[:, 3] * (1.0 - a_vol), 0.0, 1.0)
    return img
