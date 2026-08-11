"""Carrying the CPU's G-buffer to the GPU so shading can happen there.

Shading is about 71% of a frame; rasterising is about 9%. So the GPU port does
not need a GPU rasteriser to be worth doing -- it needs the *shading* to move.
The CPU already produces everything shading needs (triangle IDs, barycentric
coordinates, depth), and that is a fixed-size buffer per pixel, which is exactly
the shape of data a full-screen fragment pass consumes.

This is the same mechanism the post stages use, and those are measured working
on real hardware. A GPU rasteriser is a harder, less verifiable, lower-value
piece of work, and sequencing it first was habit rather than evidence.

What is verifiable without a GPU: the packing, and the GLSL that reconstructs
attributes from it. Both are checked against the CPU's own reconstruction.
What is not: the upload and the draw.
"""

import numpy as np

#: layout of the packed attribute texture, one texel per vertex-attribute slot.
#: Slot 3 held a tangent that was never written; the vertex COLOUR lives there
#: now, all four components, which is what unlocked painted-vertex materials
POSITION, NORMAL, UV0, COLOR = 0, 1, 2, 3
SLOTS = 4


def pack_ids(gbuf):
    """Triangle ID and barycentrics as an (H, W, 4) float32 texture.

    The ID goes in the alpha channel as a float. A float32 carries integers
    exactly up to 2**24, which is 16.7 million triangles -- far past anything
    this renderer will rasterise, and it avoids needing an integer texture
    format, which is one fewer thing to get wrong at bind time.
    """
    h, w = gbuf.tri.shape
    out = np.zeros((h, w, 4), np.float32)
    bary = gbuf.bary
    out[:, :, 0] = bary[:, :, 0]
    out[:, :, 1] = bary[:, :, 1]
    out[:, :, 2] = 1.0 - bary[:, :, 0] - bary[:, :, 1]
    out[:, :, 3] = gbuf.tri.astype(np.float32)
    return out


def pack_ids_lin(gbuf):
    """The SCREEN-LINEAR barycentrics as a second ids texture.

    Affine texture mode (the PS1 warp) interpolates uv by these instead
    of the perspective-correct set -- the CPU's attributes() picks
    `bary_lin` for uv exactly when tex_perspective is off, and the
    deferred pass re-interpolates its uv from this texture the same way.
    The channels carry the rasteriser's OWN values verbatim (all three,
    no recomputation): the warp must be the CPU's warp bit for bit at
    the interpolation inputs.
    """
    h, w = gbuf.tri.shape
    out = np.zeros((h, w, 4), np.float32)
    if gbuf.bary_lin is not None:
        out[:, :, :3] = gbuf.bary_lin
    out[:, :, 3] = gbuf.tri.astype(np.float32)
    return out


def pack_attributes(mesh, slot_count=SLOTS, respect_smooth=False):
    """Per-triangle vertex attributes as a texture the shader can index.

    Laid out as (tri_count * 3 * slot_count) texels in a 2D texture, so a
    fragment reads its three corners' attributes by computing texel offsets
    from the triangle ID rather than following any pointer.

    `respect_smooth` writes each flat triangle's *face* normal into all three
    of its corners, which is what the CPU's attribute interpolation resolves
    to for flat shading -- interpolating three identical normals is that face
    normal at every barycentric coordinate. Without it a flat-shaded cube
    comes back smooth from the GPU and nothing else in the picture is wrong,
    which is the most confusing possible way to disagree.
    """
    tris = np.asarray(mesh.tris, np.int32)
    n = tris.shape[0]
    texels = n * 3 * slot_count
    side = int(np.ceil(np.sqrt(max(texels, 1))))
    out = np.zeros((side * side, 4), np.float32)

    verts = np.asarray(mesh.verts, np.float32)
    normals = getattr(mesh, 'normals', None)
    uvs = getattr(mesh, 'uvs', None)

    flat = None
    if respect_smooth:
        smooth = getattr(mesh, 'smooth', None)
        fn = getattr(mesh, 'face_normals', None)
        if smooth is not None and fn is not None:
            flat = ~np.asarray(smooth, bool)
            fn = np.asarray(fn, np.float32)

    colors = getattr(mesh, 'colors', None)
    for corner in range(3):
        idx = tris[:, corner]
        base = (np.arange(n) * 3 + corner) * slot_count
        out[base + POSITION, :3] = verts[idx]
        if normals is not None:
            nrm = np.asarray(normals, np.float32)[idx]
            if flat is not None and flat.any():
                nrm = np.where(flat[:, None], fn, nrm)
            out[base + NORMAL, :3] = nrm
        if uvs is not None:
            uv = np.asarray(uvs, np.float32)
            out[base + UV0, :2] = uv[idx] if uv.ndim == 2 else uv[tris][:, corner]
        # the second UV map rides the SAME texel's free half: period
        # dual-texture hardware carried exactly two UV sets, and so
        # does the attribute layout -- no new slot, no new fetch
        uvs2 = getattr(mesh, 'uvs2', None)
        if uvs2 is not None:
            uv2 = np.asarray(uvs2, np.float32)
            out[base + UV0, 2:4] = uv2[idx] if uv2.ndim == 2 \
                else uv2[tris][:, corner]
        # an unpainted mesh reads as white, exactly as ShadeJob.attributes
        # answers when mesh.colors is None
        if colors is not None:
            out[base + COLOR] = np.asarray(colors, np.float32)[idx]
        else:
            out[base + COLOR] = 1.0
    return out.reshape(side, side, 4), side


def pack_tri_data(mesh):
    """Per-triangle data (not per-corner): material and object index.

    One texel per triangle, so a fragment can ask which material covered it
    and the per-material passes can keep to their own pixels.
    """
    tris = np.asarray(mesh.tris, np.int32)
    n = tris.shape[0]
    side = int(np.ceil(np.sqrt(max(n, 1))))
    out = np.zeros((side * side, 4), np.float32)
    mats = getattr(mesh, 'mat_index', None)
    objs = getattr(mesh, 'obj_index', None)
    if mats is not None:
        out[:n, 0] = np.asarray(mats, np.float32)
    if objs is not None:
        out[:n, 1] = np.asarray(objs, np.float32)
    return out.reshape(side, side, 4), side


def pack_tri_aux(mesh):
    """The per-triangle auxiliary texel: (face normal, per-tri random).

    rgb = the CPU's OWN normalize(mesh.face_normals) -- the very values
    ctx.Ng carries, baked rather than recomputed, so Normal Source FACE
    and the Geometry node's True Normal read the same bits on either
    device. a = the CPU's own per-tri random (render._hash1 of the tri
    index) -- the sin-fract hash a driver would decorrelate, so it is
    computed HERE, once, by the same NumPy code the CPU evaluator runs.
    """
    from ..core import mathx as M
    from ..core.render import _hash1
    tris = np.asarray(mesh.tris, np.int32)
    n = tris.shape[0]
    side = int(np.ceil(np.sqrt(max(n, 1))))
    out = np.zeros((side * side, 4), np.float32)
    fn = getattr(mesh, 'face_normals', None)
    if fn is not None:
        out[:n, :3] = M.normalize(np.asarray(fn, np.float32))
    out[:n, 3] = _hash1(np.arange(n, dtype=np.float32))
    return out.reshape(side, side, 4), side


GLSL = """
// --- G-buffer reconstruction -------------------------------------------
// A fragment knows which triangle covered it and where inside that triangle
// it landed. Everything else is interpolation, which is what the CPU path
// does too -- this is the same arithmetic, moved.
//
// Every read in here is texelFetch with INTEGER coordinates. These are
// data textures -- a triangle id, an attribute atlas -- and Blender's
// Python gpu module offers no sampler-state control, so texture() rides
// whatever filter the backend happens to bind. A filtered tap of an ID
// channel blends two unrelated triangle numbers into a third at every
// screen-space edge the 2x2 kernel straddles: the wrong triangle's
// corners are then interpolated, and the lighting kinks one pixel wide
// along EVERY visible edge -- a faint wireframe over the whole frame.
// Small frames land the uv exactly on texel centres and dodge it, which
// is why the parity suites and Run Self Test never saw it; at a 7200 px
// internal frame (1440 out, Supersample 24) the float32 uv is off-centre
// by ulps and the lottery pays out. texelFetch has no filter to ride:
// exact at every size, and byte-identical where texture() was right.

uniform sampler2D hal_gb_ids;        // rgb = barycentric, a = triangle id
uniform sampler2D hal_gb_attrs;      // packed per-corner attributes
uniform float     hal_attr_side;     // width of the attribute texture
uniform float     hal_slot_count;

vec4 hal_fetch_attr(float tri, int corner, int slot)
{
    int side = int(hal_attr_side);
    int index = (int(tri) * 3 + corner) * int(hal_slot_count) + slot;
    return texelFetch(hal_gb_attrs, ivec2(index % side, index / side), 0);
}

// Interpolate one attribute slot across the covering triangle.
vec3 hal_interp(float tri, vec3 bary, int slot)
{
    vec3 a = hal_fetch_attr(tri, 0, slot).xyz;
    vec3 b = hal_fetch_attr(tri, 1, slot).xyz;
    vec3 c = hal_fetch_attr(tri, 2, slot).xyz;
    return a * bary.x + b * bary.y + c * bary.z;
}

// All four components, for the slots that carry them -- the vertex colour
// keeps its alpha.
vec4 hal_interp4(float tri, vec3 bary, int slot)
{
    vec4 a = hal_fetch_attr(tri, 0, slot);
    vec4 b = hal_fetch_attr(tri, 1, slot);
    vec4 c = hal_fetch_attr(tri, 2, slot);
    return a * bary.x + b * bary.y + c * bary.z;
}

uniform sampler2D hal_gb_tris;       // one texel per triangle: (mat, obj, -, -)
uniform float     hal_tri_side;

vec4 hal_tri_data(float tri)
{
    int side = int(hal_tri_side);
    int t = int(tri);
    return texelFetch(hal_gb_tris, ivec2(t % side, t / side), 0);
}

struct HalcyonFragment {
    float tri;
    vec3  bary;
    vec3  P;
    vec3  N;
    vec2  uv;
    vec2  uv2;
    bool  covered;
};

HalcyonFragment hal_read_gbuffer(vec2 screen_uv)
{
    // uv arrives at texel centres (x+0.5)/W, so int() of uv*size floors to
    // the pixel with half a texel of tolerance -- against LINEAR's blend
    // threshold of ulps at a 7200 px frame
    ivec2 gbsz = textureSize(hal_gb_ids, 0);
    ivec2 gbpx = ivec2(clamp(screen_uv * vec2(gbsz),
                             vec2(0.0), vec2(gbsz) - vec2(1.0)));
    vec4 ids = texelFetch(hal_gb_ids, gbpx, 0);
    HalcyonFragment f;
    f.tri = ids.a;
    f.bary = ids.rgb;
    f.covered = ids.a >= 0.0;
    if (!f.covered) {
        f.P = vec3(0.0);
        f.N = vec3(0.0, 0.0, 1.0);
        f.uv = vec2(0.0);
        f.uv2 = vec2(0.0);
        return f;
    }
    f.P = hal_interp(f.tri, f.bary, 0);
    f.N = normalize(hal_interp(f.tri, f.bary, 1));
    f.uv = hal_interp(f.tri, f.bary, 2).xy;
    f.uv2 = hal_interp4(f.tri, f.bary, 2).zw;
    return f;
}
"""


def cpu_reconstruct(mesh, gbuf, slot):
    """What the shader above should produce, computed on the CPU.

    Kept here rather than in the test so the two stay side by side: if the
    interpolation ever changes on one side, the mismatch is a diff away.
    """
    tris = np.asarray(mesh.tris, np.int32)
    cov = gbuf.tri >= 0
    ids = np.where(cov, gbuf.tri, 0)
    b0 = gbuf.bary[:, :, 0]
    b1 = gbuf.bary[:, :, 1]
    b2 = 1.0 - b0 - b1

    source = {0: np.asarray(mesh.verts, np.float32),
              1: np.asarray(getattr(mesh, 'normals', mesh.verts), np.float32)}[slot]
    corners = tris[ids]
    out = (source[corners[:, :, 0]] * b0[:, :, None]
           + source[corners[:, :, 1]] * b1[:, :, None]
           + source[corners[:, :, 2]] * b2[:, :, None])
    return np.where(cov[:, :, None], out, 0.0).astype(np.float32)
