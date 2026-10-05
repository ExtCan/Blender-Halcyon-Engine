"""The cel's light (R238): the key a drawing is lit by, the shadows a
drawing casts, and the rim a drawing carries -- for the Cartoon and
Anime shaders.

A hand-drawn cel is not lit by lamps in a room. The key is decided per
shot -- usually from the screen's upper left -- and every character is
lit by it the same way whatever the set does; its cast shadows are
DRAWN shapes (the hair's shadow across the brow, the chin's on the
neck, the feet's on the ground), crisp and close; and the character's
silhouette carries a rim of light where it turns from the key or the
backlight. The 3D anime pipelines rebuilt each of these on purpose:
the character light (a direction fixed to the camera or the world, not
the scene's lamps), screen-space shadows (a short depth march toward
the light: what is nearer along the way shadows the pixel), and the
depth rim (a neighbour a few pixels away that is farther is a
silhouette, and the rim sits inside it). This module is those three.

The SHADING LIGHT lives in the shading: each cel material names its
key (the scene's lamps, as before; a direction fixed to the camera;
a direction fixed to the world) and the lamp loop follows it, folding
the scene's cast shadows in (`render.light_surface`, `gpu/material`).

The SCREEN SHADOW and the DEPTH RIM are a whole-frame FIELD computed
once per frame from the G-buffer on the CPU, before shading, and read
per pixel by both devices (the radiosity field's road): `compute`
gives the field, `reach_rows` names how far past a band it reads, and
the consumers guard on the field's own depth so a transparent layer
over the opaque surface does not read the surface's shadow.

Sizes are given in pixels at 1080 lines and scale with the internal
frame's height, as the ink's widths do.
"""

import numpy as np

from . import lines as LN
from . import mathx as M

CEL_LIGHT_ITEMS = (
    ('SCENE', "Scene Lamps",
     "The scene's lamps light the cel as before: every lamp its own "
     "tone verdict, the strongest wins on the Cartoon Shader, the tints "
     "sum on the Anime Shader"),
    ('CAMERA', "Camera Key",
     "One key light fixed to the camera -- Light Azimuth and Elevation "
     "about the screen (positive azimuth from the screen's left, "
     "positive elevation from above), so the character is lit the same "
     "way on screen wherever the camera goes: the drawing's light. The "
     "scene's lamps still cast their shadows onto it"),
    ('WORLD', "World Key",
     "One key light fixed in the world -- Azimuth counter-clockwise "
     "from +X, Elevation above the horizon -- whatever the scene's "
     "lamps do; their cast shadows still land"),
)

CEL_RIM_MODE_ITEMS = (
    ('FRESNEL', "Fresnel Rim",
     "The rim by the view angle: the silhouette's grazing surfaces "
     "brighten by Rim Power, everywhere the surface turns away"),
    ('SCREEN', "Depth Rim",
     "The anime rim: a band Rim Width pixels wide just inside the "
     "silhouette, wherever the surface Rim Width pixels along the "
     "screen normal is farther -- a crisp line of light that keeps its "
     "width whatever the form, on the lit side, the shadow side or "
     "both"),
)

CEL_RIM_SIDE_ITEMS = (
    ('LIT', "Lit Side", "The rim faces the key: the light catching the edge"),
    ('SHADOW', "Shadow Side",
     "The rim faces away from the key: the backlight's edge"),
    ('BOTH', "Both", "The rim all the way around the silhouette"),
)

CEL_SHAPE_ITEMS = (
    ('SPHERE', "Sphere",
     "Shadow Smoothing bends the shading normal toward a sphere around "
     "the object's bounds: the terminator sweeps as one round shape"),
    ('CYLINDER', "Cylinder",
     "...toward an upright cylinder through the object's bounds: the "
     "terminator runs straight down a limb or a torso"),
    ('CAMERA', "Facing the Camera",
     "...toward the camera: the whole surface reads as one flat plane "
     "facing the viewer, lit as one -- the anime face, with only the "
     "cast and screen shadows on it"),
)

LIGHT_CODE = {'SCENE': 0.0, 'CAMERA': 1.0, 'WORLD': 2.0}
RIM_MODE_CODE = {'FRESNEL': 0.0, 'SCREEN': 1.0}
RIM_SIDE_CODE = {'LIT': 0.0, 'SHADOW': 1.0, 'BOTH': 2.0}
SHAPE_CODE = {'SPHERE': 0.0, 'CYLINDER': 1.0, 'CAMERA': 2.0}

#: the march's sample distances in pixels at 1080 lines: every pixel
#: out to forty (a hair's shadow right under it, a collar's on the
#: chest, a body's on the ground, found at the pixel), every second
#: pixel beyond -- a sparser march tears a long shadow's far edge into
#: teeth where the ray grazes a silhouette (the 2-px parity of the
#: sample against a thin band of occluding pixels)
_MARCH_BASE = tuple(float(i) for i in range(1, 41)) \
    + tuple(float(42 + 2 * i) for i in range(0, 44))

#: the screen shadow's tolerance, in pixels of world size at the
#: pixel's depth: an occluder must be nearer than the ray by this much
#: at the pixel, growing by _SS_BIAS_GROW per pixel marched (a
#: silhouette grazed far along the march is not a shadow), plus the
#: receiver's own slope over _SS_SLOPE_PX pixels -- the march samples
#: the depth plane eroded to its nearest neighbour (a silhouette a
#: pixel fat, so a ray grazing it cannot alternate between hit and
#: miss), and a grazing receiver's own neighbours must not count
_SS_BIAS_PX = 1.5
_SS_BIAS_GROW = 0.25
_SS_SLOPE_PX = 2.0
_SS_SLOPE_MAX = 20.0
#: an occluder farther in front of the ray than this does not shadow
#: (it is not between the point and the light for all the march knows
#: -- the screen-space shadow's own limit): the larger of this many
#: march lengths and this fraction of the pixel's depth (a character
#: a body deep at eight metres shadows its own ground; a foreground
#: figure five metres in front of a far wall does not), fading out
#: over the last third of it rather than cutting (a cut tore the far
#: edge of a long shadow into teeth where the gap crossed it)
_SS_THICK = 2.5
_SS_THICK_DEPTH = 0.25
#: the depth rim's step, in pixels of world size per pixel of width
#: (plus twice the surface's own slope over the width): a neighbour
#: along the depth's gradient that is farther than the surface itself
#: could carry it is a silhouette, a steep flat face is not
_RIM_STEP_PX = 3.0


def scale_of(h):
    """The frame's size factor: sizes are given at 1080 lines."""
    return float(h) / 1080.0


def key_vector(mode, az_deg, el_deg):
    """The key's unit direction TOWARD the light in its own frame, from
    the dials, float32: CAMERA about the screen (x right, y up, z
    toward the viewer; positive azimuth swings the light to the
    screen's LEFT, positive elevation above), WORLD about the world
    (azimuth counter-clockwise from +X, elevation above the horizon).
    SCENE has no key of its own (zero)."""
    az = np.float32(np.radians(float(az_deg)))
    el = np.float32(np.radians(float(el_deg)))
    ca, sa = np.float32(np.cos(az)), np.float32(np.sin(az))
    ce, se = np.float32(np.cos(el)), np.float32(np.sin(el))
    code = int(round(float(mode))) if not isinstance(mode, str) \
        else int(LIGHT_CODE.get(mode.upper(), 0.0))
    if code == 1:
        v = np.array([-sa * ce, se, ca * ce], np.float32)
    elif code == 2:
        v = np.array([ca * ce, sa * ce, se], np.float32)
    else:
        return np.zeros(3, np.float32)
    ln = float(np.sqrt(float((v * v).sum())))
    return (v / np.float32(max(ln, 1e-9))).astype(np.float32)


def camera_axes(camera):
    """The camera's (right, up, back) unit axes in the world, float32,
    from its rigid matrix (object scale stripped, as every metric
    consumer does). No camera: the world's own axes, looking down -Y."""
    if camera is None or getattr(camera, 'matrix_world', None) is None:
        return (np.array([1.0, 0.0, 0.0], np.float32),
                np.array([0.0, 0.0, 1.0], np.float32),
                np.array([0.0, 1.0, 0.0], np.float32))
    mw = np.asarray(M.rigid_camera_matrix(camera.matrix_world), np.float32)
    axes = []
    for c in range(3):
        a = mw[:3, c].astype(np.float32)
        ln = float(np.sqrt(float((a * a).sum())))
        axes.append((a / np.float32(max(ln, 1e-9))).astype(np.float32))
    return tuple(axes)


def world_key(mode, cel_dir, camera):
    """The key's world direction: CAMERA composes the camera-frame
    vector on the camera's axes (right * x + up * y + back * z, in that
    order, float32 -- the GPU's three multiply-adds), WORLD is the
    vector itself."""
    code = int(round(float(mode)))
    d = np.asarray(cel_dir, np.float32)
    if code == 1:
        r, u, b = camera_axes(camera)
        v = (r * d[0] + u * d[1] + b * d[2]).astype(np.float32)
        ln = float(np.sqrt(float((v * v).sum())))
        return (v / np.float32(max(ln, 1e-9))).astype(np.float32)
    ln = float(np.sqrt(float((d * d).sum())))
    if ln < 1e-9:
        return np.array([0.0, 0.0, 1.0], np.float32)
    return (d / np.float32(ln)).astype(np.float32)


# ---------------------------------------------------- the materials' dials


def _node_of(graph):
    """The cel master node of a material graph (the Anime or the
    Cartoon Shader feeding the output), or None."""
    if not graph:
        return None
    nodes = graph.get('nodes', {}) or {}
    for nd in nodes.values():
        if nd.get('bl_idname') in ('HALCYON_AnimeShaderNode',
                                   'HALCYON_CartoonNode'):
            return nd
    return None


def _sock(node, name, default):
    for s in node.get('inputs', ()) or ():
        if s.get('name') == name:
            v = s.get('default')
            if v is None:
                return default
            try:
                return float(v if not isinstance(v, (list, tuple)) else v[0])
            except (TypeError, ValueError):
                return default
    return default


def material_params(mat):
    """The cel dials of one material, read from its master node's
    sockets and menus (a linked socket reads its default: these dials
    are per material, never per pixel), or None for a material with no
    cel master. Keys: light, dir (the key's own-frame vector), ss,
    ss_len, rim, rim_mode, rim_width, rim_side."""
    nd = _node_of(getattr(mat, 'graph', None))
    if nd is None:
        return None
    p = nd.get('props', {}) or {}
    light = LIGHT_CODE.get(str(p.get('light_source', 'SCENE')).upper(), 0.0)
    az = _sock(nd, 'Light Azimuth', 35.0)
    el = _sock(nd, 'Light Elevation', 30.0)
    return {
        'light': float(light),
        'dir': key_vector(light, az, el),
        'ss': float(np.clip(_sock(nd, 'Screen Shadow', 0.0), 0.0, 1.0)),
        'ss_len': float(max(_sock(nd, 'Screen Shadow Length', 24.0), 1.0)),
        'rim': float(max(_sock(nd, 'Rim Amount', 0.0), 0.0)),
        'rim_mode': float(RIM_MODE_CODE.get(
            str(p.get('rim_mode', 'FRESNEL')).upper(), 0.0)),
        'rim_width': float(max(_sock(nd, 'Rim Width', 4.0), 0.5)),
        'rim_side': float(RIM_SIDE_CODE.get(
            str(p.get('rim_side', 'LIT')).upper(), 0.0)),
    }


def wants_field(params):
    """Whether a material's dials read the field at all."""
    if params is None:
        return False
    return params['ss'] > 0.0 or (params['rim_mode'] > 0.5
                                  and params['rim'] > 1e-4)


def field_on(scene, st):
    """Whether any material of the scene reads the field."""
    for m in getattr(scene, 'materials', None) or ():
        if wants_field(material_params(m)):
            return True
    return False


def reach_rows(scene, st, rh):
    """How many rows past a band the field reads: the longest march and
    the widest rim among the materials that use them, at the internal
    height `rh`, plus two."""
    sc = scale_of(rh)
    reach = 0.0
    for m in getattr(scene, 'materials', None) or ():
        p = material_params(m)
        if p is None:
            continue
        if p['ss'] > 0.0:
            reach = max(reach, p['ss_len'] * sc)
        if p['rim_mode'] > 0.5 and p['rim'] > 1e-4:
            reach = max(reach, p['rim_width'] * sc)
    return int(np.ceil(reach)) + 2 if reach > 0.0 else 0


# --------------------------------------------------------------- the field


def view_depth_plane(gbuf, proj):
    """The G-buffer's depth as a VIEW depth (positive, scene units):
    the stored z is clip.z / clip.w, inverted through the projection
    matrix's own third and fourth rows -- perspective or orthographic,
    whichever the camera is -- so it matches the depth every shading
    context carries to the z-buffer's own grid. Infinite where nothing
    was hit."""
    pm = np.asarray(proj, np.float64)
    zn = np.asarray(gbuf.depth, np.float64)
    cov = gbuf.mask()
    zc = np.where(cov, zn, 0.0)
    den = zc * pm[3, 2] - pm[2, 2]
    den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    zv = (pm[2, 3] - zc * pm[3, 3]) / den
    out = np.abs(zv).astype(np.float32)
    out[~cov] = np.inf
    return out


def _view_points(proj, px, py, z, W, H):
    """View-space positions (x, y, -z) of pixel centres (px, py) at
    view depth z, float32, through the projection's own terms (a
    camera shift rides p02 / p12); perspective or orthographic."""
    pm = np.asarray(proj, np.float32)
    xn = (px + np.float32(0.5)) * np.float32(2.0 / W) - np.float32(1.0)
    yn = (py + np.float32(0.5)) * np.float32(2.0 / H) - np.float32(1.0)
    if abs(float(pm[3, 2])) > 1e-9:
        # clip.w = -z_v = z: x_ndc = (p00 x_v + p02 z_v + p03) / z.
        # R251 C098: the lens shear puts a clip translation in
        # pm[., 3]; subtracting an exact 0.0 is bitwise the old form,
        # and _screen_of already carries the same term
        xv = ((xn + pm[0, 2]) * z - pm[0, 3]) / pm[0, 0]
        yv = ((yn + pm[1, 2]) * z - pm[1, 3]) / pm[1, 1]
    else:
        xv = (xn - pm[0, 3]) / pm[0, 0]
        yv = (yn - pm[1, 3]) / pm[1, 1]
    return xv.astype(np.float32), yv.astype(np.float32), (-z).astype(np.float32)


def _screen_of(proj, xv, yv, zv, W, H):
    """Pixel coordinates of view-space points, float32."""
    pm = np.asarray(proj, np.float32)
    cx = pm[0, 0] * xv + pm[0, 1] * yv + pm[0, 2] * zv + pm[0, 3]
    cy = pm[1, 0] * xv + pm[1, 1] * yv + pm[1, 2] * zv + pm[1, 3]
    cw = pm[3, 0] * xv + pm[3, 1] * yv + pm[3, 2] * zv + pm[3, 3]
    cw = np.where(np.abs(cw) < np.float32(1e-9), np.float32(1e-9), cw)
    xs = (cx / cw * np.float32(0.5) + np.float32(0.5)) * np.float32(W)
    ys = (cy / cw * np.float32(0.5) + np.float32(0.5)) * np.float32(H)
    return xs.astype(np.float32), ys.astype(np.float32)


def compute(scene, mesh, gbuf, view, proj, vp, camera, settings):
    """The field for one frame: dict(ss=(H, W) float32 shadow term in
    0..1, rim=(H, W) float32 rim mask in 0..1, depth=(H, W) float32 the
    G-buffer's own view depth for the consumers' guard), or None when
    no material reads it. Every pixel's march and rim follow ITS
    material's dials and its key (the scene's key lamp, the camera key
    or the world key), so one field serves every cel material at once.
    Everything is read off the depth plane and the pixel grid in view
    space -- no vertex gathers: the frame pays a few dozen array
    operations plus one gather per march step."""
    mats = getattr(scene, 'materials', None) or []
    table = [material_params(m) for m in mats]
    if not any(wants_field(p) for p in table):
        return None
    f32 = np.float32
    H, W = gbuf.tri.shape
    sc = scale_of(H)
    cov = gbuf.mask()
    ss_out = np.zeros(H * W, f32)
    rim_out = np.zeros(H * W, f32)
    depth = view_depth_plane(gbuf, proj)
    idx = np.nonzero(cov.reshape(-1))[0]
    if idx.size == 0:
        return {'ss': ss_out.reshape(H, W), 'rim': rim_out.reshape(H, W),
                'depth': depth}
    tri = gbuf.tri.reshape(-1)[idx]
    mi = np.asarray(mesh.mat_index, np.int64)[np.where(tri >= 0, tri, 0)] \
        if getattr(mesh, 'mat_index', None) is not None \
        else np.zeros(idx.size, np.int64)
    mi = np.clip(mi, 0, max(len(table) - 1, 0))
    # the per-pixel dials from the per-material table
    nm = max(len(table), 1)
    t_ss = np.zeros(nm, f32)
    t_len = np.zeros(nm, f32)
    t_rim = np.zeros(nm, f32)
    t_rw = np.zeros(nm, f32)
    t_side = np.zeros(nm, f32)
    t_light = np.zeros(nm, f32)
    t_dir = np.zeros((nm, 3), f32)
    rot = np.asarray(view, f32)[:3, :3]
    for k, p in enumerate(table):
        if p is None:
            continue
        t_ss[k] = p['ss']
        t_len[k] = p['ss_len'] * sc
        t_rim[k] = p['rim'] if p['rim_mode'] > 0.5 else 0.0
        t_rw[k] = p['rim_width'] * sc
        t_side[k] = p['rim_side']
        t_light[k] = p['light']
        # the key in VIEW space (the march lives there)
        t_dir[k] = rot @ world_key(p['light'], p['dir'], camera)
    active = (t_ss[mi] > 0.0) | (t_rim[mi] > 1e-4)
    if not np.any(active):
        return {'ss': ss_out.reshape(H, W), 'rim': rim_out.reshape(H, W),
                'depth': depth}
    sel = np.nonzero(active)[0]
    idx = idx[sel]
    mi = mi[sel]
    dflat = depth.reshape(-1)
    z = dflat[idx]
    px = (idx % W).astype(f32)
    py = (idx // W).astype(f32)
    xv, yv, zv = _view_points(proj, px, py, z, W, H)
    pws = LN.pixel_world_size(proj, z, H).astype(f32)
    # the key per pixel, in view space: the scene's key lamp for SCENE
    # materials (a sun's one direction, a point lamp's from every
    # pixel), the material's own fixed key otherwise
    Lv = t_dir[mi].astype(f32)
    scene_px = t_light[mi] < 0.5
    if np.any(scene_px):
        key = LN.key_light(scene)
        if key is None:
            Lv[scene_px] = np.array([0.0, 0.0, 1.0], f32)
        else:
            kind = str(getattr(key, 'type', 'POINT')).upper()
            if kind in ('SUN', 'HEMI'):
                d = np.asarray(getattr(key, 'direction', (0.0, 0.0, -1.0)),
                               f32)
                ln = float(np.linalg.norm(d))
                Lw = (-d / ln) if ln > 1e-9 else np.array([0, 0, 1], f32)
                Lv[scene_px] = rot @ Lw.astype(f32)
            else:
                pos = np.asarray(getattr(key, 'position', (0.0, 0.0, 0.0)),
                                 f32)
                vm = np.asarray(view, f32)
                pos_v = vm[:3, :3] @ pos + vm[:3, 3]
                sp = np.nonzero(scene_px)[0]
                dvec = np.stack([pos_v[0] - xv[sp], pos_v[1] - yv[sp],
                                 pos_v[2] - zv[sp]], 1)
                ln = np.sqrt((dvec * dvec).sum(1))
                Lv[sp] = dvec / np.maximum(ln, f32(1e-9))[:, None]
    # the light's screen direction per pixel: the pixel and the pixel
    # moved one pixel's world size toward the light, projected
    xs0, ys0 = _screen_of(proj, xv, yv, zv, W, H)
    xs1, ys1 = _screen_of(proj, xv + Lv[:, 0] * pws, yv + Lv[:, 1] * pws,
                          zv + Lv[:, 2] * pws, W, H)
    dx = xs1 - xs0
    dy = ys1 - ys0
    dz = (-(Lv[:, 2] * pws)).astype(f32)         # depth = -z_v
    ln = np.maximum(np.abs(dx), np.abs(dy))
    ok = ln > f32(1e-6)
    inv = np.where(ok, f32(1.0) / np.where(ok, ln, f32(1.0)), f32(0.0))
    sx = (dx * inv).astype(f32)
    sy = (dy * inv).astype(f32)
    sz = (dz * inv).astype(f32)                  # depth change per pixel
    fx = px + f32(0.5)
    fy = py + f32(0.5)
    # the receiver's own depth slope per pixel, the smaller of its
    # one-sided differences each way (a silhouette's jump is not the
    # surface's slope), capped -- the march's bias and the rim's
    # threshold both allow for it
    dpad = np.pad(depth, 1, mode='edge')
    slope_plane = np.zeros((H, W), f32)      # the smaller one-sided step
    steep_plane = np.zeros((H, W), f32)      # the larger one
    dfin = np.where(np.isfinite(dpad), dpad, f32(1e9))
    for axis in (0, 1):
        fwd = np.abs(np.diff(dfin, axis=axis))
        if axis == 0:
            a = fwd[:H, 1:W + 1]
            b = fwd[1:H + 1, 1:W + 1]
        else:
            a = fwd[1:H + 1, :W]
            b = fwd[1:H + 1, 1:W + 1]
        np.maximum(slope_plane, np.minimum(a, b), out=slope_plane)
        np.maximum(steep_plane, np.maximum(a, b), out=steep_plane)
    slope_all = np.minimum(slope_plane.reshape(-1)[idx],
                           f32(_SS_SLOPE_MAX) * pws)
    steep_all = np.minimum(steep_plane.reshape(-1)[idx],
                           f32(_SS_SLOPE_MAX) * pws)
    # ---- the screen shadow: the march toward the light
    px_len = t_len[mi]
    px_ssa = t_ss[mi]
    marching = (px_ssa > 0.0) & ok
    if np.any(marching):
        mi_m = np.nonzero(marching)[0]
        # the eroded depth plane: every pixel's nearest of its 3x3
        # neighbourhood (edges clamped), the silhouettes a pixel fat --
        # padded by the longest march with "nothing hit", so a step off
        # the frame reads no occluder and needs no clipping
        dmin = depth.copy()
        for oy in (0, 1, 2):
            for ox in (0, 1, 2):
                if oy == 1 and ox == 1:
                    continue
                np.minimum(dmin, dpad[oy:oy + H, ox:ox + W], out=dmin)
        max_step = int(np.ceil(float(px_len[mi_m].max()))) + 2
        WP = W + 2 * max_step
        dbig = np.full((H + 2 * max_step, WP), np.inf, f32)
        dbig[max_step:max_step + H, max_step:max_step + W] = dmin
        dbig_flat = dbig.reshape(-1)
        pw = pws[mi_m]
        slope = slope_all[mi_m]
        # per-pixel constants of the march, float32, compacted once
        mfx = (fx[mi_m] + f32(max_step)).astype(f32)
        mfy = (fy[mi_m] + f32(max_step)).astype(f32)
        msx = sx[mi_m]
        msy = sy[mi_m]
        mz = z[mi_m]
        msz = sz[mi_m]
        mlen = px_len[mi_m]
        bias0 = (f32(_SS_BIAS_PX) * pw + f32(_SS_SLOPE_PX) * slope).astype(f32)
        biasg = (f32(_SS_BIAS_GROW) * pw).astype(f32)
        thick = np.maximum(f32(_SS_THICK) * mlen * pw,
                           f32(_SS_THICK_DEPTH) * mz).astype(f32)
        inv_fade = (f32(3.0) / thick).astype(f32)
        occl = np.zeros(mi_m.size, f32)
        len_max = float(mlen.max())
        len_min = float(mlen.min())
        # the march, in place: one set of buffers, no temporaries (a
        # fresh array per operation paid page faults on every step)
        m = mi_m.size
        tf = np.empty(m, f32)
        tg = np.empty(m, f32)
        tw = np.empty(m, f32)
        tb = np.empty(m, f32)
        # int32 indices: the padded plane is well inside 2^31 texels,
        # and every coordinate is positive (the pad), so the float to
        # int copy truncates exactly as floor would
        gxi = np.empty(m, np.int32)
        gyi = np.empty(m, np.int32)
        WP32 = np.int32(WP)
        for d in _MARCH_BASE:
            step = f32(d * sc)
            if len_max < float(step) - 1e-6:
                break
            np.multiply(msx, step, out=tf)
            np.add(tf, mfx, out=tf)
            gxi[...] = tf
            np.multiply(msy, step, out=tf)
            np.add(tf, mfy, out=tf)
            gyi[...] = tf
            np.multiply(gyi, WP32, out=gyi)
            np.add(gyi, gxi, out=gyi)
            np.take(dbig_flat, gyi, out=tg)          # the occluder's depth
            np.multiply(msz, step, out=tf)
            np.add(tf, mz, out=tf)
            np.subtract(tf, tg, out=tf)              # the gap
            np.multiply(biasg, f32(d), out=tb)
            np.add(tb, bias0, out=tb)                # the bias
            np.subtract(thick, tf, out=tw)
            np.multiply(tw, inv_fade, out=tw)
            np.clip(tw, 0.0, 1.0, out=tw)            # the fade
            np.greater(tf, tb, out=tg)               # nearer than the bias
            np.multiply(tw, tg, out=tw)
            if len_min < float(step) - 1e-6:
                np.greater_equal(mlen, step - f32(1e-6), out=tg)
                np.multiply(tw, tg, out=tw)
            np.maximum(occl, tw, out=occl)
        ss_out[idx[mi_m]] = (occl * px_ssa[mi_m]).astype(f32)
    # ---- the depth rim: the neighbour along the depth's own gradient
    px_rw = t_rw[mi]
    px_rima = t_rim[mi]
    rimming = px_rima > 1e-4
    if np.any(rimming):
        ri = np.nonzero(rimming)[0]
        # the outward direction: the depth plane's gradient (central
        # differences, a silhouette's jump dominating exactly where the
        # rim lives), unit in pixels
        finite = dfin
        gx_p = (finite[1:H + 1, 2:W + 2] - finite[1:H + 1, :W]).reshape(-1)[idx[ri]]
        gy_p = (finite[2:H + 2, 1:W + 1] - finite[:H, 1:W + 1]).reshape(-1)[idx[ri]]
        gl = np.sqrt(gx_p * gx_p + gy_p * gy_p)
        gok = gl > f32(1e-12)
        nx = np.where(gok, gx_p / np.where(gok, gl, f32(1.0)), f32(0.0))
        ny = np.where(gok, gy_p / np.where(gok, gl, f32(1.0)), f32(0.0))
        wpx = np.maximum(px_rw[ri], f32(0.5))
        gx = np.floor(fx[ri] + nx * wpx).astype(np.int64)
        gy = np.floor(fy[ri] + ny * wpx).astype(np.int64)
        outside = (gx < 0) | (gx >= W) | (gy < 0) | (gy >= H)
        gx = np.clip(gx, 0, W - 1)
        gy = np.clip(gy, 0, H - 1)
        gd = dflat[gy * W + gx]
        # farther than the surface's own steepest step could carry it
        # over the width: a silhouette, not a steep face and not the
        # crease between two faces (the background and the sky are
        # infinitely far and always count)
        thr = wpx * (f32(_RIM_STEP_PX) * pws[ri] + f32(_SS_SLOPE_PX) * steep_all[ri])
        far = outside | (~np.isfinite(gd)) | ((gd - z[ri]) > thr)
        # the side: the outward direction against the key's screen
        # direction; a key straight down the view axis has no screen
        # direction and both sides read as the whole silhouette
        dots = nx * sx[ri] + ny * sy[ri]
        side = t_side[mi[ri]]
        lit = np.clip((dots + f32(0.1)) * f32(2.5), 0.0, 1.0)
        lit = lit * lit * (f32(3.0) - f32(2.0) * lit)
        sh = np.clip((-dots + f32(0.1)) * f32(2.5), 0.0, 1.0)
        sh = sh * sh * (f32(3.0) - f32(2.0) * sh)
        k_side = np.where(side > 1.5, f32(1.0), np.where(side > 0.5, sh, lit))
        k_side = np.where(ok[ri], k_side, f32(1.0)).astype(f32)
        rim_out[idx[ri]] = (far.astype(f32) * k_side).astype(f32)
    return {'ss': ss_out.reshape(H, W), 'rim': rim_out.reshape(H, W),
            'depth': depth}


def lookup(field, spx, spy, depth_ctx):
    """The field at pixels (spx, spy) for surfaces at view depth
    `depth_ctx`: (ss, rim) flat float32, zero where the surface is not
    the G-buffer's own (a transparent layer over it, a hit off it)."""
    if field is None or spx is None or spy is None:
        return None, None
    H, W = field['ss'].shape
    x = np.clip(np.asarray(spx, np.int64), 0, W - 1)
    y = np.clip(np.asarray(spy, np.int64), 0, H - 1)
    ss = field['ss'][y, x]
    rim = field['rim'][y, x]
    if depth_ctx is not None:
        gd = field['depth'][y, x]
        d = np.asarray(depth_ctx, np.float32)
        own = np.abs(gd - d) <= np.float32(1e-3) * np.maximum(d, np.float32(1e-3))
        ss = np.where(own, ss, 0.0).astype(np.float32)
        rim = np.where(own, rim, 0.0).astype(np.float32)
    return ss, rim
