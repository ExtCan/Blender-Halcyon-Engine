"""R251 shadow pack: the shadow-MASK road (planar polygons, Dreamcast
modifier volumes, Nintendo DS shadow polygons).

Three period shadows are not lighting terms: they are regions DRAWN onto
the frame -- a caster's triangles projected onto the floor plane (Blinn
1988 / Sega Model 1), a stencil-parity region times the PowerVR2's
FPU_SHAD_SCALE (Dreamcast), a translucent volume polygon keyed on the
polygon ID (DS POLYGON_ATTR mode 3) -- that darken the pixel's finished
colour whatever the lamps do. So they share ONE mechanism:

  * `build` decides everything ONCE on the CPU, on BOTH device roads,
    right after the G-buffer is final (after the R211 punch-through
    promotion): a bit pack at the internal size -- `bits[..., 0]` the
    PLANAR lamps whose projected polygon covers the pixel AND whose
    receiver test holds (bit i = lamp index i), `bits[..., 1]` the
    Dreamcast stencil bit, `bits[..., 2]` the DS volumes drawn on the
    pixel (bit k = volume k) -- plus its own int32 copy of the opaque
    winner's triangle id, and the literals the apply rule needs.
  * `apply` is ONE float32 numpy rule, one op per statement, the same
    code in the same order on both roads (the CPU frame and the GPU
    readback), applied by `apply_after_readback` over the finished
    opaque frame before the wireframe, the ink, the halos and the
    transparent composite. Bitwise across devices by construction; it
    uploads nothing and refuses nothing.

Every mask raster takes the frame's own snap, near-plane epsilon, band
scissor AND depth grid (`depth_bits = st.depth_precision`): the fragments
and the G-buffer then sit on one grid, an equal depth IS equal, and the
strict compares below decide every coplanar tie as named. A scissor drops
only triangles wholly outside the band and the fill of a kept triangle is
scissor-blind, so a pixel's bit is the same in every band by the law that
makes the beauty G-buffer band-invariant.

Named tie rules: a planar receiver is ON the plane within `eps` (1e-4 of
the scene radius); a volume face within the A-buffer depth tolerance of
the surface (`raster.ABUF_DEPTH_TOL_REL` / `_ABS`, the engine's own
"modeled contact" band of ~30 float32 ULPs) is AT the surface and NOT a
crossing in front of it -- the coplanar cap of a volume standing on the
floor -- which is also the DS depth test (LESS) that such a cap fails.
Measured, not assumed: on one 24-bit grid the cap's and the floor's
float32 interpolations still straddled a quantisation step at 1 of ~600
contact pixels (a strict `<` on the grid alone would have drawn a
speckle there); the band lands the tie the same way under any rounding,
as the A-buffer's own keep test does.
"""

import numpy as np

from . import lights as LI
from . import raster

#: the pack's capacities: bit i of a float32 integer plane, exact below 2^24
MAX_LAMPS = 24
MAX_VOLUMES = 24

_F1_255 = np.float32(1.0 / 255.0)
_F1_63 = np.float32(1.0 / 63.0)


def nearer(fz, zs):
    """THE tie rule of the volume roads: a volume fragment at `fz` is in
    front of the surface at `zs` only beyond the A-buffer depth tolerance
    (`zs - (|zs| * REL + ABS)`); within it the fragment is AT the surface
    (a coplanar cap, a modeled contact) -- not a crossing for the PowerVR
    parity, a failed LESS test for the DS polygon."""
    zs = np.asarray(zs, np.float32)
    tol = np.abs(zs) * raster.ABUF_DEPTH_TOL_REL + raster.ABUF_DEPTH_TOL_ABS
    return np.asarray(fz, np.float32) < (zs - tol)


def _resolved_mode(light, st):
    return light.shadow if st.shadow_default == 'PER_LIGHT' else \
        st.shadow_default


def _lamp_name(light):
    return str(getattr(light, 'name', '') or 'lamp')


# ------------------------------------------------------------------ C052

def planar_mask(mesh, cast_tri, light, z0, vp, W, H, snap, near_eps, gbuf,
                eps, scissor, depth_bits):
    """C052: one lamp's projected shadow polygon AND the receiver test.

    Casters (the same set the maps honour) are projected along the lamp
    onto the plane Z = z0 and rasterised with the frame's own rasteriser;
    `cover` is the union of the polygons (order-free). The receiver test
    holds where the G-buffer's surface point lies ON the plane within
    `eps`. Returns the (H, W) bool mask, or zeros (with a printed reason)
    when the lamp cannot project."""
    verts = np.asarray(mesh.verts, np.float32)
    tris = mesh.tris if cast_tri is None else mesh.tris[cast_tri]
    z0 = np.float32(z0)
    eps = np.float32(eps)
    h = verts[:, 2] - z0
    kind = str(getattr(light, 'type', 'POINT')).upper()
    if kind == 'SUN':
        d = np.asarray(light.direction, np.float32)
        Lv = -d / np.float32(max(float(np.linalg.norm(d)), 1e-12))
        Lv = Lv.astype(np.float32)
        if float(Lv[2]) <= 1e-6:
            print(f"[Halcyon] planar shadows: lamp '{_lamp_name(light)}' "
                  f'sits below the plane Z={float(z0):g}; it casts no '
                  'floor shadow')
            return np.zeros((H, W), bool)
        t = h / Lv[2]
        tl = t[:, None] * Lv[None, :]
        Pp = verts - tl
        ok = np.ones(verts.shape[0], bool)
    else:
        Q = np.asarray(light.position, np.float32)
        if float(Q[2] - z0) <= float(eps):
            print(f"[Halcyon] planar shadows: lamp '{_lamp_name(light)}' "
                  f'sits below the plane Z={float(z0):g}; it casts no '
                  'floor shadow')
            return np.zeros((H, W), bool)
        ok = verts[:, 2] < Q[2] - eps
        s = (z0 - Q[2]) / (verts[ok, 2] - Q[2])
        d = verts[ok] - Q[None, :]
        ds = d * s[:, None]
        Pp = verts.copy()                 # excluded corners keep a finite value
        Pp[ok] = Q[None, :] + ds
    corner_ok = (h > eps) & ok
    keep = corner_ok[tris].all(axis=1)
    cover = np.zeros((H, W), bool)
    if keep.any():
        gb = raster.GBuffer(W, H)
        raster.rasterize(Pp.astype(np.float32), tris[keep], vp, W, H,
                         cull='NONE', snap=snap, depth_bits=depth_bits,
                         gbuf=gb, near_eps=near_eps, scissor=scissor)
        cover = gb.mask()
    # the receiver test, decided once: the surface point's world Z from
    # the G-buffer, float32 in a fixed order
    ti = np.clip(gbuf.tri, 0, mesh.tris.shape[0] - 1)
    t3 = mesh.tris[ti]
    b = gbuf.bary
    Pz = verts[t3[..., 0], 2] * b[..., 0]
    Pz = Pz + verts[t3[..., 1], 2] * b[..., 1]
    Pz = Pz + verts[t3[..., 2], 2] * b[..., 2]
    recv = (gbuf.tri >= 0) & (np.abs(Pz - z0) <= eps)
    return cover & recv


planar_shadow = planar_mask


# ------------------------------------------------------------- volumes

def _volume_fragments(vol, vp, W, H, snap, near_eps, scissor, depth_bits,
                      flat_depth=None):
    """Every fragment of an authored volume at the frame's pixel centres
    (the transparent raster's real call shape): (px, py, zndc, front).

    `front` is the CULLER's facing (`cull='BACK'` keeps the front faces,
    RA:463-466), captured as two culled rasters whose union is the
    unculled capture (a cull drops whole triangles; the fill of a kept
    one is the same). Measured, not assumed: the fragment list's own
    `front` flag (`ar < 0.0`, RA:572) is False on every outward face of
    scenebuild's cube and on all 1200 box pixels of the demo G-buffer --
    the opposite sign of the culler's `area > 0.0` -- so the DS rule
    reads the culler, the convention the convex cross-check test uses."""
    verts = np.asarray(vol['verts'], np.float32)
    tris = np.asarray(vol['tris'], np.int32)
    parts = []
    for cull, is_front in (('BACK', True), ('FRONT', False)):
        gb = raster.GBuffer(W, H)         # depth inf: every fragment kept
        fl = raster.FragmentList()
        raster.rasterize(verts, tris, vp, W, H, cull=cull, snap=snap,
                         depth_bits=depth_bits, gbuf=gb, frags=fl,
                         depth_write=False, near_eps=near_eps,
                         scissor=scissor, flat_depth=flat_depth)
        px, py, _t, fz, _b, _ff = fl.finish()
        parts.append((px, py, fz, np.full(px.size, is_front, bool)))
    px = np.concatenate([p[0] for p in parts])
    py = np.concatenate([p[1] for p in parts])
    fz = np.concatenate([p[2] for p in parts])
    ffront = np.concatenate([p[3] for p in parts])
    return px, py, fz, ffront


def _dc_stencil(vol, frags, gbuf):
    """C020: the PowerVR2 stencil parity of one modifier volume -- the
    count of its faces nearer than the pixel's surface (beyond the tie
    band, `nearer`), mod 2; an EXCLUDE volume flips the bit over the
    whole frame."""
    px, py, fz, _front = frags
    H, W = gbuf.tri.shape
    cnt = np.zeros((H, W), np.int32)
    if px.size:
        zs = gbuf.zndc[py, px]
        infront = nearer(fz, zs)
        np.add.at(cnt, (py[infront], px[infront]), 1)
    par = cnt & 1
    if str(vol.get('role', 'DC_INCLUDE')) == 'DC_EXCLUDE':
        par = par ^ 1
    return par


modvol_apply = _dc_stencil


def _ds_cover(vol, frags, gbuf, pid_px):
    """C036: where one DS shadow polygon draws -- the depth-fail mask set
    per failing back-face fragment (melonDS's rule), the draw where a
    front face passes, the polygon-ID self-exclusion."""
    px, py, fz, ffront = frags
    H, W = gbuf.tri.shape
    mask = np.zeros((H, W), bool)
    draw = np.zeros((H, W), bool)
    if px.size:
        zs = gbuf.zndc[py, px]
        pass_ = nearer(fz, zs)               # the DS LESS test, tie band
        back = ~ffront
        m = back & ~pass_
        mask[py[m], px[m]] = True
        d = ffront & pass_
        draw[py[d], px[d]] = True
    return mask & draw & (gbuf.tri >= 0) & \
        (pid_px != int(vol.get('polygon_id', 0)))


ds_shadow_apply = _ds_cover


# --------------------------------------------------------------- build

def build(scene, st, gbuf, vp, snap, near_eps, cast_tri, scissor=None, view=None, eye=None):
    """The bake: (pack, params), or (None, None) when no PLANAR lamp and
    no authored volume exists (then no pixel is touched and the frame is
    bitwise the release before this road)."""
    if not bool(getattr(st, 'shadows', True)):
        # the render's master Shadows switch: off, no polygon and no
        # volume is drawn (the machine's frame without its shadows)
        return None, None
    mesh = scene.mesh
    lights = list(getattr(scene, 'lights', None) or ())
    vols = list(getattr(scene, 'shadow_volumes', None) or ())
    planar = []
    if mesh is not None and \
            getattr(mesh, 'tris', None) is not None and len(mesh.tris):
        idx = {id(l): i for i, l in enumerate(lights)}
        for l in LI.select_lights(lights, st):
            if str(getattr(l, 'type', '')).upper() in ('HEMI', 'AMBIENT'):
                continue
            if getattr(l, 'shadow', 'MAP') == 'NONE':
                continue
            if _resolved_mode(l, st) != 'PLANAR':
                continue
            planar.append((idx[id(l)], l))
        planar.sort(key=lambda p: p[0])
    if not planar and not vols:
        return None, None
    H, W = gbuf.tri.shape
    bits = np.zeros((H, W, 3), np.float32)
    params = {'planar': [], 'modvol': None, 'ds': []}
    depth_bits = int(getattr(st, 'depth_precision', 24) or 24)

    if planar:
        _c, radius = LI.scene_bounds(np.asarray(mesh.verts, np.float32))
        eps = np.float32(1e-4) * np.float32(max(1.0, float(radius)))
        z0 = np.float32(getattr(st, 'planar_plane_z', 0.0))
        over = False
        for i, l in planar:
            if i >= MAX_LAMPS:
                if not over:
                    print(f'[Halcyon] planar shadows: {MAX_LAMPS} lamps per '
                          f"frame; lamp '{_lamp_name(l)}' and later cast "
                          'none')
                    over = True
                continue
            m = planar_mask(mesh, cast_tri, l, z0, vp, W, H, snap, near_eps,
                            gbuf, eps, scissor, depth_bits)
            if m.any():
                bits[..., 0][m] += np.float32(float(1 << i))
            col = np.asarray(getattr(l, 'shadow_color', (0, 0, 0)),
                             np.float32)[:3]
            params['planar'].append(
                (int(i), tuple(np.float32(v) for v in col),
                 np.float32(getattr(l, 'shadow_density', 1.0))))

    if vols:
        flat = str(getattr(st, 'depth_sort', 'ZBUFFER')) == 'PAINTERS' \
            and view is not None and eye is not None
        if flat:
            from types import SimpleNamespace
            from .render import polygon_depths
        pid_px = None
        g = np.zeros((H, W), np.int32)
        any_dc = False
        k = 0
        over = False
        for vol in vols:
            role = str(vol.get('role', 'NONE'))
            if role not in ('DC_INCLUDE', 'DC_EXCLUDE', 'DS_SHADOW'):
                continue
            fd = None
            if flat:
                fd = polygon_depths(
                    SimpleNamespace(verts=np.asarray(vol['verts'], np.float32),
                                    tris=np.asarray(vol['tris'], np.int32)),
                    view, eye, getattr(st, 'painters_key', 'CENTROID'))
            frags = _volume_fragments(vol, vp, W, H, snap, near_eps, scissor,
                                      depth_bits, flat_depth=fd)
            if role == 'DS_SHADOW':
                if k >= MAX_VOLUMES:
                    if not over:
                        print(f'[Halcyon] shadow volumes: {MAX_VOLUMES} per '
                              f"frame; volume '{vol.get('name', '')}' and "
                              'later are not drawn')
                        over = True
                    continue
                if pid_px is None:
                    mats = list(getattr(scene, 'materials', None) or ())
                    pid_of = np.array(
                        [int(getattr(m, 'polygon_id', 0)) for m in mats]
                        or [0], np.int32)
                    mi = getattr(mesh, 'mat_index', None)
                    if mi is None or not len(mi):
                        pid_px = np.zeros((H, W), np.int32)
                    else:
                        ti = np.clip(gbuf.tri, 0, len(mi) - 1)
                        pid_px = pid_of[np.clip(mi[ti], 0, pid_of.size - 1)]
                sel = _ds_cover(vol, frags, gbuf, pid_px)
                if sel.any():
                    bits[..., 2][sel] += np.float32(float(1 << k))
                cs6 = tuple(int(v) for v in np.rint(
                    np.clip(np.asarray(vol.get('color', (0, 0, 0)),
                                       np.float64)[:3], 0.0, 1.0) * 63.0))
                a = int(min(max(int(vol.get('alpha', 16)), 1), 30))
                params['ds'].append((int(k), cs6, a))
                k += 1
            else:
                g = g ^ _dc_stencil(vol, frags, gbuf)
                any_dc = True
        if any_dc:
            g = g & (gbuf.tri >= 0).astype(np.int32)
            bits[..., 1] = g.astype(np.float32)
            params['modvol'] = int(min(max(int(getattr(st, 'modvol_scale',
                                                       128)), 0), 255))
    if not params['planar'] and params['modvol'] is None and \
            not params['ds']:
        return None, None
    return (bits, gbuf.tri.copy()), params


def params_sig(params):
    """The literals as a hashable (for a future plan signature)."""
    if not params:
        return None
    return (tuple((int(i), tuple(float(v) for v in c), float(d))
                  for i, c, d in params['planar']),
            params['modvol'],
            tuple((int(k), tuple(int(v) for v in c), int(a))
                  for k, c, a in params['ds']))


# --------------------------------------------------------------- apply

def _bit(v, k):
    """Bit k of a float32 integer plane: three exact operations (a
    power-of-two multiply, a floor, an exact small-integer subtract) --
    never `mod` (a GLSL divide) and never `>>`."""
    t = np.floor(v * np.float32(2.0 ** -k))
    return t - np.float32(2.0) * np.floor(t * np.float32(0.5))


def apply(rgb, px, py, tri, pack, params):
    """The per-pixel rule, float32, one op per statement, in the fixed
    order: Dreamcast scale, PLANAR lamps in lamp order, DS volumes in
    volume order. `own` (the pixel is still the opaque winner the bake
    saw) compares int32 against int32."""
    bits, ptri = pack
    rgb = np.array(rgb, np.float32, copy=True)
    if rgb.ndim != 2 or rgb.shape[0] == 0:
        return rgb
    own = ptri[py, px] == np.asarray(tri, np.int32)
    r = bits[py, px, 0]
    g = bits[py, px, 1]
    b = bits[py, px, 2]
    scale = params.get('modvol')
    if scale is not None:
        m = own & (g > np.float32(0.5))
        if m.any():
            c8 = np.clip(rgb[m], np.float32(0.0), np.float32(1.0))
            c8 = c8 * np.float32(255.0)
            c8 = np.rint(c8)
            c8 = c8 * np.float32(scale)
            c8 = c8 * np.float32(0.00390625)
            c8 = np.floor(c8)
            rgb[m] = c8 * _F1_255
    for i, colour, dens in params.get('planar', ()):
        bit = _bit(r, int(i))
        t = bit * np.float32(dens)
        t = np.where(own, t, np.float32(0.0)).astype(np.float32)
        col = np.asarray(colour, np.float32).reshape(1, 3)
        d = col - rgb
        pm = d * t[:, None]
        rgb = rgb + pm
    for k, cs6, a in params.get('ds', ()):
        bit = _bit(b, int(k))
        m = own & (bit > np.float32(0.5))
        if not m.any():
            continue
        a = int(a)
        c6 = np.clip(rgb[m], np.float32(0.0), np.float32(1.0))
        c6 = c6 * np.float32(63.0)
        c6 = np.rint(c6)
        c6 = c6 * np.float32(31 - a)
        c6 = c6 + np.asarray([np.float32(int(v) * (a + 1)) for v in cs6],
                             np.float32).reshape(1, 3)
        c6 = c6 * np.float32(0.03125)
        c6 = np.floor(c6)
        rgb[m] = c6 * _F1_63
    return rgb


def apply_frame(out, gbuf, hit, st):
    """The rule over a whole rgb frame at the covered pixels `hit` (the
    readback form; `own` reads the G-buffer's triangle ids)."""
    pack = getattr(st, '_mask_pack', None)
    if pack is None:
        return out
    py, px = np.nonzero(hit)
    if py.size == 0:
        return out
    out = np.array(out, np.float32, copy=True)
    tri = gbuf.tri[py, px]
    out[py, px, :3] = apply(out[py, px, :3], px, py, tri, pack,
                            st._mask_params)
    return out


def apply_after_readback(img, job, st):
    """The ONE call render() makes on both roads once the opaque frame is
    final (the CPU shade or the GPU readback): every covered pixel is the
    opaque winner there, so `own` holds by construction and the pack's
    own triangle plane says which pixels are covered. The resident GPU
    frame is released by name first (the CPU edits the picture)."""
    pack = getattr(st, '_mask_pack', None)
    if pack is None or img is None:
        return img
    bits, ptri = pack
    if img.shape[0] != ptri.shape[0] or img.shape[1] != ptri.shape[1]:
        print('[Halcyon] shadow mask: the pack does not match the frame '
              f'({ptri.shape[1]}x{ptri.shape[0]} vs '
              f'{img.shape[1]}x{img.shape[0]}); not applied')
        return img
    try:
        from ..gpu import frame as _FR
        _FR.edited(st, 'shadow mask',
                   'planar polygons / modifier volumes / DS shadow polygons')
    except Exception:                                           # noqa: BLE001
        pass
    py, px = np.nonzero(ptri >= 0)
    if py.size == 0:
        return img
    img = np.array(img, np.float32, copy=True)
    img[py, px, :3] = apply(img[py, px, :3], px, py, ptri[py, px], pack,
                            st._mask_params)
    return img
