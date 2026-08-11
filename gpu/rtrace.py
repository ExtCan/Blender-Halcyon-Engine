"""Ray tracing on the GPU, starting where the CPU starts: the BVH.

The traversal is STACKLESS -- a threaded BVH. The front-end that verifies
every kernel headlessly supports divergent while-loops but not dynamic
array writes, so instead of a traversal stack each node carries two links
precomputed at pack time: `first` (the child visited first) and `miss`
(where to go when this subtree is done or skipped). One scalar `node`
variable walks the whole tree.

The links encode the CPU's EXACT visit order. `occluded()` pushes left
then right and pops last-in-first-out, so the right subtree is walked
first: `first` is the right child, the right child's `miss` is its left
sibling, and the left child's `miss` is the parent's. Any-hit results
never depend on order, but closest-hit ties do -- and order turned out
to be necessary, not sufficient: when two surfaces COINCIDE (a box
resting on the floor), the winner comes down to the last bit of two
almost-equal t values, and the driver's rounding is not NumPy's. Those
rays the kernel flags (id -2) and the wrappers re-resolve on
bvh.intersect itself; see INTERSECT_GLSL's note for the measured
anatomy.

Everything numerical mirrors bvh.py to the operation: the slab test with
its sign-losing 1e12 fallback for near-zero direction components, and
Möller-Trumbore with the exact epsilon set (det 1e-12, u,v >= -1e-6,
u+v <= 1+1e-6, t > 1e-6).
"""

import numpy as np


def thread_links(bvh):
    """(first, miss) int32 arrays: the CPU pop order as two links per node.

    Right subtree first -- exactly the LIFO order `occluded` walks -- and
    a leaf's `first` is -1.
    """
    n = bvh.n_nodes
    first = np.full(n, -1, np.int32)
    miss = np.full(n, -1, np.int32)
    if n == 0:
        return first, miss
    # iterative right-first DFS, assigning misses as we descend
    stack = [(0, -1)]
    while stack:
        node, node_miss = stack.pop()
        miss[node] = node_miss
        left = int(bvh.left[node])
        right = int(bvh.right[node])
        if left < 0:
            continue                       # leaf: first stays -1
        if right < 0:
            # the degenerate split: one child only; visit it, then miss
            first[node] = left
            stack.append((left, node_miss))
            continue
        first[node] = right                # right subtree walks first
        stack.append((right, left))        # after right, the left sibling
        stack.append((left, node_miss))    # after left, the parent's miss
    return first, miss


def _square(texels):
    return int(np.ceil(np.sqrt(max(texels, 1))))


def pack_bvh(bvh):
    """The tree as two textures.

    nodes: 3 texels each -- (bmin, 0), (bmax, 0), (start, count, first,
    miss). `first` < 0 marks a leaf. tris: 3 texels each, REORDERED by
    `bvh.order` so every leaf's range is contiguous -- (v0, original id),
    (e1, 0), (e2, 0). The spare channels are reserved for the cast flags
    the shadow integration will need.
    """
    n = bvh.n_nodes
    first, miss = thread_links(bvh)
    n_side = _square(n * 3)
    nodes = np.zeros((n_side * n_side, 4), np.float32)
    base = np.arange(n) * 3
    nodes[base, :3] = bvh.bmin[:n]
    nodes[base + 1, :3] = bvh.bmax[:n]
    nodes[base + 2, 0] = bvh.start[:n]
    nodes[base + 2, 1] = bvh.count[:n]
    nodes[base + 2, 2] = first
    nodes[base + 2, 3] = miss
    nodes = nodes.reshape(n_side, n_side, 4)

    order = bvh.order
    t = order.shape[0]
    t_side = _square(t * 3)
    tris = np.zeros((t_side * t_side, 4), np.float32)
    tbase = np.arange(t) * 3
    tris[tbase, :3] = bvh.v0[order]
    tris[tbase, 3] = order.astype(np.float32)
    tris[tbase + 1, :3] = bvh.e1[order]
    tris[tbase + 2, :3] = bvh.e2[order]
    tris = tris.reshape(t_side, t_side, 4)
    return nodes, n_side, tris, t_side


#: the traversal, shared by every wrapper that will ever use it
TRAVERSE_GLSL = """
uniform sampler2D hal_bvh;           // 3 texels per node
uniform float hal_bvh_side;
uniform sampler2D hal_btris;         // 3 texels per triangle, leaf order
uniform float hal_btris_side;

// texture() with half-texel centring, as the proven ray-shadow passes
// always fetched. A texelFetch conversion rode one round on suspicion
// of row-boundary misreads (the RAY texture's real, caught-on-hardware
// scar) -- and the glass-mirror flips stayed BIT-IDENTICAL, which is
// an acquittal: the tree fetches were never misreading. The RAY
// fetches in the compute wrappers keep texelFetch, because theirs is
// the layout that actually misread on hardware (95 caught flips);
// the tree keeps the filtered form the fast path has always measured
// 0 px against the CPU with. (1.30.6 note: the G-buffer's filtered ID
// reads WERE misreading -- the faint-wireframe defect -- and went
// texelFetch with the rest of the deferred pass. The tree stays as
// measured: the conversion bought zero correctness there and cost the
// field driver real milliseconds; if a ray row ever disagrees at a
// bigger tree side, this comment is where to look first.)
vec4 hal_bvh_texel(float index)
{
    float x = mod(index, hal_bvh_side);
    float y = floor(index / hal_bvh_side);
    return texture(hal_bvh, (vec2(x, y) + vec2(0.5)) / hal_bvh_side);
}

vec4 hal_btri_texel(float index)
{
    float x = mod(index, hal_btris_side);
    float y = floor(index / hal_btris_side);
    return texture(hal_btris, (vec2(x, y) + vec2(0.5)) / hal_btris_side);
}

// any-hit query, exactly bvh.occluded(): 1.0 the moment anything sits
// between t=1e-6 and tmax along the ray, 0.0 otherwise
float hal_bvh_occluded(vec3 org, vec3 dir, float tmax)
{
    vec3 inv;
    inv.x = (abs(dir.x) < 1e-12) ? 1e12 : 1.0 / dir.x;
    inv.y = (abs(dir.y) < 1e-12) ? 1e12 : 1.0 / dir.y;
    inv.z = (abs(dir.z) < 1e-12) ? 1e12 : 1.0 / dir.z;
    float node = 0.0;
    float guard = 0.0;
    while (node > -0.5 && guard < 100000.0) {
        guard = guard + 1.0;
        vec4 a = hal_bvh_texel(node * 3.0);
        vec4 b = hal_bvh_texel(node * 3.0 + 1.0);
        vec4 c = hal_bvh_texel(node * 3.0 + 2.0);
        vec3 t0 = (a.xyz - org) * inv;
        vec3 t1 = (b.xyz - org) * inv;
        float tn = max(max(min(t0.x, t1.x), min(t0.y, t1.y)),
                       min(t0.z, t1.z));
        float tf = min(min(max(t0.x, t1.x), max(t0.y, t1.y)),
                       max(t0.z, t1.z));
        if (!(tf >= max(tn, 0.0) && tn <= tmax)) {
            node = c.w;                       // miss: skip the subtree
            continue;
        }
        if (c.z < -0.5) {
            // a leaf: Moller-Trumbore over its contiguous range, the
            // CPU's exact epsilons
            float start = c.x;
            float count = c.y;
            for (int i = 0; i < int(count); i++) {
                float tb = (start + float(i)) * 3.0;
                vec3 v0 = hal_btri_texel(tb).xyz;
                vec3 e1 = hal_btri_texel(tb + 1.0).xyz;
                vec3 e2 = hal_btri_texel(tb + 2.0).xyz;
                vec3 p = cross(dir, e2);
                float det = dot(e1, p);
                if (abs(det) <= 1e-12) { continue; }
                float inv_det = 1.0 / det;
                vec3 tvec = org - v0;
                float u = dot(tvec, p) * inv_det;
                vec3 q = cross(tvec, e1);
                float v = dot(dir, q) * inv_det;
                float t = dot(e2, q) * inv_det;
                if (u >= -1e-6 && v >= -1e-6 && u + v <= 1.000001
                        && t > 1e-6 && t < tmax) {
                    return 1.0;
                }
            }
            node = c.w;
        } else {
            node = c.z;                       // first child: the right one
        }
    }
    return 0.0;
}
"""

#: closest-hit, exactly bvh.intersect(): needs TRAVERSE_GLSL's texel
#: fetchers in the same source. Returns (original tri id or -1, t, u, v);
#: t stays at tmax on a miss, exactly as the CPU leaves best_t.
#:
#: The tie story is the whole reason the traversal is threaded the way it
#: is. The CPU pops nodes LIFO (right subtree first) and keeps a hit only
#: when strictly closer; within a leaf, argmin keeps the FIRST of equal
#: minima. A sequential walk in the SAME order with the SAME strict `<`
#: reproduces both -- but only down to the last bit of t, and the last
#: bit is the driver's. The glass-mirror hunt measured the failure
#: exactly: a box RESTING on the floor puts two coplanar triangles at
#: the same depth, 22106 of the frame's 141478 sweep rays hit both
#: within 1e-6 relative t, and the driver's own rounding (fused
#: multiply-adds the front-end cannot reproduce, `precise` ignored)
#: picked the other surface on 1925 of them -- floor on one device,
#: glass on the other, deterministic and unfixable by arithmetic
#: edits, which two bit-identical rounds proved. So the kernel now
#: NAMES the ties instead of guessing: it tracks the two nearest
#: accepted hits, and when they land within 1e-5 relative (measured
#: valley: real ties sit under 1e-6, the next distinct surface beyond
#: 1e-3), it returns id -2.0 -- "this decision is inside float noise"
#: -- and the wrappers re-resolve those rays on bvh.intersect itself,
#: the reference. Ray routing, exactly the layer-routing doctrine:
#: route, never guess. The slab prune loosens by the same window so a
#: tying candidate in a barely-pruned subtree still registers. The
#: original id rides texel 0's w channel (float32 is id-exact to 2^24
#: triangles).
INTERSECT_GLSL = """
vec4 hal_bvh_intersect(vec3 org, vec3 dir, float tmax)
{
    vec3 inv;
    inv.x = (abs(dir.x) < 1e-12) ? 1e12 : 1.0 / dir.x;
    inv.y = (abs(dir.y) < 1e-12) ? 1e12 : 1.0 / dir.y;
    inv.z = (abs(dir.z) < 1e-12) ? 1e12 : 1.0 / dir.z;
    float best_id = -1.0;
    float best_t = tmax;
    float best_u = 0.0;
    float best_v = 0.0;
    float second_t = 1e30;
    float node = 0.0;
    float guard = 0.0;
    while (node > -0.5 && guard < 100000.0) {
        guard = guard + 1.0;
        vec4 a = hal_bvh_texel(node * 3.0);
        vec4 b = hal_bvh_texel(node * 3.0 + 1.0);
        vec4 c = hal_bvh_texel(node * 3.0 + 2.0);
        vec3 t0 = (a.xyz - org) * inv;
        vec3 t1 = (b.xyz - org) * inv;
        float tn = max(max(min(t0.x, t1.x), min(t0.y, t1.y)),
                       min(t0.z, t1.z));
        float tf = min(min(max(t0.x, t1.x), max(t0.y, t1.y)),
                       max(t0.z, t1.z));
        // the slab prunes against the LIVE best -- loosened by the tie
        // window, so a subtree whose nearest hit can still TIE is
        // visited and second_t stays honest
        if (!(tf >= max(tn, 0.0) && tn <= best_t * 1.00001)) {
            node = c.w;
            continue;
        }
        if (c.z < -0.5) {
            float start = c.x;
            float count = c.y;
            for (int i = 0; i < int(count); i++) {
                float tb = (start + float(i)) * 3.0;
                vec4 tv0 = hal_btri_texel(tb);
                vec3 e1 = hal_btri_texel(tb + 1.0).xyz;
                vec3 e2 = hal_btri_texel(tb + 2.0).xyz;
                vec3 p = cross(dir, e2);
                float det = dot(e1, p);
                if (abs(det) <= 1e-12) { continue; }
                float inv_det = 1.0 / det;
                vec3 tvec = org - tv0.xyz;
                float u = dot(tvec, p) * inv_det;
                vec3 q = cross(tvec, e1);
                float v = dot(dir, q) * inv_det;
                float t = dot(e2, q) * inv_det;
                if (u >= -1e-6 && v >= -1e-6 && u + v <= 1.000001
                        && t > 1e-6) {
                    if (t < best_t) {
                        second_t = best_t;
                        best_t = t;
                        best_id = tv0.w;
                        best_u = u;
                        best_v = v;
                    } else if (t < second_t) {
                        second_t = t;
                    }
                }
            }
            node = c.w;
        } else {
            node = c.z;
        }
    }
    // two accepted surfaces inside the noise window: the winner is a
    // coin flip this device must not call. -2.0 asks the CPU.
    if (best_id > -0.5 && second_t <= best_t * 1.00001) {
        return vec4(-2.0, best_t, best_u, best_v);
    }
    return vec4(best_id, best_t, best_u, best_v);
}
"""

#: fragment-style wrapper the NumPy front-end runs: one ray per lane
OCCLUDE_FRAGMENT = TRAVERSE_GLSL + """
uniform vec3 hal_rorg;
uniform vec3 hal_rdir;
uniform float hal_rtmax;
out vec4 Color;

void main()
{
    Color = vec4(hal_bvh_occluded(hal_rorg, hal_rdir, hal_rtmax),
                 0.0, 0.0, 1.0);
}
"""

#: compute wrapper the driver runs: rays packed two texels each.
#:
#: The ray fetches are texelFetch with INTEGER coordinates, and that is a
#: field scar, not a style choice: sampling the rays texture through
#: texture() with computed normalized coordinates misread row-boundary-
#: adjacent texels on real hardware (RTX 5060 Ti, Vulkan) -- once per row
#: of a 245-wide layout, deterministically, while 283-wide read clean.
#: texelFetch bypasses the sampler entirely; integer addressing is exact
#: by specification. The BVH texel helpers stay texture(): they are shared
#: with the fragment paths, where they are measured at zero across every
#: shadow and kernel section.
OCCLUDE_COMPUTE = TRAVERSE_GLSL + """
uniform sampler2D hal_rays;          // 2 texels per ray: (org,tmax)(dir,0)
uniform float hal_rays_side;
uniform float hal_ray_count;
uniform float hal_out_w;

void main()
{
    ivec2 xy = ivec2(gl_GlobalInvocationID.xy);
    int idx = xy.y * int(hal_out_w) + xy.x;
    if (idx >= int(hal_ray_count)) { return; }
    int side = int(hal_rays_side);
    int rb = idx * 2;
    vec4 ra = texelFetch(hal_rays, ivec2(rb % side, rb / side), 0);
    int rb1 = rb + 1;
    vec4 rd = texelFetch(hal_rays, ivec2(rb1 % side, rb1 / side), 0);
    float hit = hal_bvh_occluded(ra.xyz, rd.xyz, ra.w);
    imageStore(hal_out_hits, xy, vec4(hit, 0.0, 0.0, 1.0));
}
"""


#: front-end wrapper for the closest-hit query
INTERSECT_FRAGMENT = TRAVERSE_GLSL + INTERSECT_GLSL + """
uniform vec3 hal_rorg;
uniform vec3 hal_rdir;
uniform float hal_rtmax;
out vec4 Color;

void main()
{
    Color = hal_bvh_intersect(hal_rorg, hal_rdir, hal_rtmax);
}
"""

#: compute wrapper: rays two texels each, out image (id, t, u, v).
#: texelFetch for the rays, for the exact reason OCCLUDE_COMPUTE documents
#: -- this kernel is where the row-boundary misreads were CAUGHT: 95
#: deterministic hit flips at side 245, every one an off-by-one-texel ray
#: read, every one gone from the fragment path that takes rays as
#: uniforms.
INTERSECT_COMPUTE = TRAVERSE_GLSL + INTERSECT_GLSL + """
uniform sampler2D hal_rays;          // 2 texels per ray: (org,tmax)(dir,0)
uniform float hal_rays_side;
uniform float hal_ray_count;
uniform float hal_out_w;

void main()
{
    ivec2 xy = ivec2(gl_GlobalInvocationID.xy);
    int idx = xy.y * int(hal_out_w) + xy.x;
    if (idx >= int(hal_ray_count)) { return; }
    int side = int(hal_rays_side);
    int rb = idx * 2;
    vec4 ra = texelFetch(hal_rays, ivec2(rb % side, rb / side), 0);
    int rb1 = rb + 1;
    vec4 rd = texelFetch(hal_rays, ivec2(rb1 % side, rb1 / side), 0);
    imageStore(hal_out_hits, xy, hal_bvh_intersect(ra.xyz, rd.xyz, ra.w));
}
"""


#: how many rays the LAST closest-hit call re-resolved on the CPU
#: because the kernel flagged a tie (id -2): the self-test prints it,
#: and a nonzero count on a frame NAMES coincident contact geometry
LAST_TIE_ROUTED = 0


def _resolve_ties(bvh, org, dirs, tmax, ids, t, u, v):
    """Rays the kernel flagged as noise-window ties get the CPU's answer.

    The kernel returns id -2.0 when its two nearest accepted hits sit
    within 1e-5 relative t of each other -- a decision the driver's
    last-bit arithmetic cannot make reproducibly (the glass-mirror
    floor/box-bottom contact plane: 1925 deterministic flips). Those
    rays, and only those, re-run through bvh.intersect itself, so the
    GPU path returns the reference's own winner by construction. Route,
    never guess."""
    global LAST_TIE_ROUTED
    tie = ids == -2
    LAST_TIE_ROUTED = int(tie.sum())
    if not tie.any():
        return ids, t, u, v
    ci, ct, cu, cv = bvh.intersect(org[tie], dirs[tie], tmax[tie])
    ids = ids.copy()
    t = np.asarray(t).copy()
    u = np.asarray(u).copy()
    v = np.asarray(v).copy()
    ids[tie] = ci
    t[tie] = ct
    u[tie] = cu
    v[tie] = cv
    return ids, t, u, v


def simulate_intersect(bvh, org, dirs, tmax):
    """Closest-hit through the front-end. Returns (ids int32, t, u, v)."""
    from ..core.texture import Texture
    from ..shaders.compiler import try_compile

    nodes, n_side, tris, t_side = pack_bvh(bvh)
    prog, err = try_compile(INTERSECT_FRAGMENT, 'GLSL')
    if prog is None:
        raise RuntimeError(f'closest-hit kernel does not compile: {err}')
    n = org.shape[0]
    tmax = np.asarray(tmax, np.float32)
    if tmax.ndim == 0:
        tmax = np.full(n, float(tmax), np.float32)
    uni = {
        'hal_bvh': Texture(nodes, colorspace='Non-Color', filt='NEAREST',
                           wrap='EXTEND'),
        'hal_btris': Texture(tris, colorspace='Non-Color', filt='NEAREST',
                             wrap='EXTEND'),
        'hal_bvh_side': np.full(n, float(n_side), np.float32),
        'hal_btris_side': np.full(n, float(t_side), np.float32),
        'hal_rorg': org.astype(np.float32),
        'hal_rdir': dirs.astype(np.float32),
        'hal_rtmax': tmax,
    }
    got = prog.run(uni, {}, n)[0]['Color']
    return _resolve_ties(bvh, org, dirs, tmax,
                         np.round(got[:, 0]).astype(np.int32), got[:, 1],
                         got[:, 2], got[:, 3])


def intersect_on_device(bvh, org, dirs, tmax):
    """The same query on the real driver.

    Returns ((ids, t, u, v), why) -- ids -1 on a miss, t left at tmax.
    """
    from . import device

    nodes, n_side, tris, t_side = pack_bvh(bvh)
    n = org.shape[0]
    tmax = np.asarray(tmax, np.float32)
    if tmax.ndim == 0:
        tmax = np.full(n, float(tmax), np.float32)
    r_side = _square(n * 2)
    rays = np.zeros((r_side * r_side, 4), np.float32)
    rb = np.arange(n) * 2
    rays[rb, :3] = org
    rays[rb, 3] = tmax
    rays[rb + 1, :3] = dirs
    rays = rays.reshape(r_side, r_side, 4)

    ow = 256
    oh = (n + ow - 1) // ow
    shader, err = device.compile_compute(
        'HAL_RT_INTERSECT', INTERSECT_COMPUTE,
        samplers=('hal_bvh', 'hal_btris', 'hal_rays'),
        floats=('hal_bvh_side', 'hal_btris_side', 'hal_rays_side',
                'hal_ray_count', 'hal_out_w'),
        images=('hal_out_hits',))
    if shader is None:
        return None, err
    try:
        out = device.dispatch_compute(
            shader, ow, oh,
            uniforms={'hal_bvh_side': float(n_side),
                      'hal_btris_side': float(t_side),
                      'hal_rays_side': float(r_side),
                      'hal_ray_count': float(n),
                      'hal_out_w': float(ow)},
            samplers={'hal_bvh': device.upload(nodes),
                      'hal_btris': device.upload(tris),
                      'hal_rays': device.upload(rays)},
            images=('hal_out_hits',))
    except Exception as exc:                                    # noqa: BLE001
        return None, f'closest-hit dispatch failed: ' \
                     f'{type(exc).__name__}: {exc}'
    flat = out['hal_out_hits'].reshape(-1, 4)[:n]
    return _resolve_ties(bvh, org, dirs, tmax,
                         np.round(flat[:, 0]).astype(np.int32), flat[:, 1],
                         flat[:, 2], flat[:, 3]), None


def intersect_frame(bvh, org, dirs, tmax=1e30):
    """Closest-hit for a frame's reflection rays, with the BVH cached.

    `intersect_on_device` is the self-test tool: it packs and uploads the
    tree fresh on every call, which is the right shape for a measurement
    and the wrong one for a render loop. This one rides the same
    content-fingerprint upload cache as every atlas, so the warm cost is
    the rays and the readback. Returns ((ids, t, u, v), why).
    """
    from . import device

    ok, why = device.probe()
    if not ok:
        return None, why
    n = org.shape[0]
    tmax = np.asarray(tmax, np.float32)
    if tmax.ndim == 0:
        tmax = np.full(n, float(tmax), np.float32)
    r_side = _square(n * 2)
    rays = np.zeros((r_side * r_side, 4), np.float32)
    rb = np.arange(n) * 2
    rays[rb, :3] = org
    rays[rb, 3] = tmax
    rays[rb + 1, :3] = dirs
    rays = rays.reshape(r_side, r_side, 4)

    # the compute trace gets its OWN cached copies of the tree, never the
    # texture objects the ray-shadow FRAGMENT passes sample. The first
    # reflected frame on real hardware flipped 95 hit ids that the same
    # rays through fresh textures (the self-test kernel) never flip, with
    # the shared fragment-then-compute texture objects as the one variable
    # every other experiment failed to eliminate -- cross-stage image
    # state is exactly the kind of thing a driver may cache per usage.
    # Two copies of the tree in VRAM is the price; the trees are small.
    key = bvh_fingerprint(bvh)
    packed = {}

    def _pack():
        if 'data' not in packed:
            packed['data'] = pack_bvh(bvh)
        return packed['data']

    try:
        tex_nodes = device.upload_cached(('bvh_nodes_compute',) + key,
                                         lambda: _pack()[0])
        tex_tris = device.upload_cached(('bvh_tris_compute',) + key,
                                        lambda: _pack()[2])
    except Exception as exc:                                    # noqa: BLE001
        return None, f'uploading the BVH failed: {type(exc).__name__}: {exc}'

    ow = 256
    oh = (n + ow - 1) // ow
    shader, err = device.compile_compute(
        'HAL_RT_INTERSECT', INTERSECT_COMPUTE,
        samplers=('hal_bvh', 'hal_btris', 'hal_rays'),
        floats=('hal_bvh_side', 'hal_btris_side', 'hal_rays_side',
                'hal_ray_count', 'hal_out_w'),
        images=('hal_out_hits',))
    if shader is None:
        return None, err
    try:
        out = device.dispatch_compute(
            shader, ow, oh,
            uniforms={'hal_bvh_side': float(tex_nodes.width),
                      'hal_btris_side': float(tex_tris.width),
                      'hal_rays_side': float(r_side),
                      'hal_ray_count': float(n),
                      'hal_out_w': float(ow)},
            samplers={'hal_bvh': tex_nodes,
                      'hal_btris': tex_tris,
                      'hal_rays': device.upload(rays)},
            images=('hal_out_hits',))
    except Exception as exc:                                    # noqa: BLE001
        return None, f'reflection trace failed: {type(exc).__name__}: {exc}'
    flat = out['hal_out_hits'].reshape(-1, 4)[:n]
    return _resolve_ties(bvh, org, dirs, tmax,
                         np.round(flat[:, 0]).astype(np.int32), flat[:, 1],
                         flat[:, 2], flat[:, 3]), None


def simulate_occluded(bvh, org, dirs, tmax):
    """The kernel through the front-end: one boolean per ray, no driver."""
    from ..core.texture import Texture
    from ..shaders.compiler import try_compile

    nodes, n_side, tris, t_side = pack_bvh(bvh)
    prog, err = try_compile(OCCLUDE_FRAGMENT, 'GLSL')
    if prog is None:
        raise RuntimeError(f'occlusion kernel does not compile: {err}')
    n = org.shape[0]
    tmax = np.asarray(tmax, np.float32)
    if tmax.ndim == 0:
        tmax = np.full(n, float(tmax), np.float32)
    uni = {
        'hal_bvh': Texture(nodes, colorspace='Non-Color', filt='NEAREST',
                           wrap='EXTEND'),
        'hal_btris': Texture(tris, colorspace='Non-Color', filt='NEAREST',
                             wrap='EXTEND'),
        'hal_bvh_side': np.full(n, float(n_side), np.float32),
        'hal_btris_side': np.full(n, float(t_side), np.float32),
        'hal_rorg': org.astype(np.float32),
        'hal_rdir': dirs.astype(np.float32),
        'hal_rtmax': tmax,
    }
    got = prog.run(uni, {}, n)[0]['Color']
    return got[:, 0] > 0.5


def occluded_on_device(bvh, org, dirs, tmax):
    """The same query on the real driver. Returns (hits (N,) bool, why)."""
    from . import device

    nodes, n_side, tris, t_side = pack_bvh(bvh)
    n = org.shape[0]
    tmax = np.asarray(tmax, np.float32)
    if tmax.ndim == 0:
        tmax = np.full(n, float(tmax), np.float32)
    r_side = _square(n * 2)
    rays = np.zeros((r_side * r_side, 4), np.float32)
    rb = np.arange(n) * 2
    rays[rb, :3] = org
    rays[rb, 3] = tmax
    rays[rb + 1, :3] = dirs
    rays = rays.reshape(r_side, r_side, 4)

    ow = 256
    oh = (n + ow - 1) // ow
    shader, err = device.compile_compute(
        'HAL_RT_OCCLUDE', OCCLUDE_COMPUTE,
        samplers=('hal_bvh', 'hal_btris', 'hal_rays'),
        floats=('hal_bvh_side', 'hal_btris_side', 'hal_rays_side',
                'hal_ray_count', 'hal_out_w'),
        images=('hal_out_hits',))
    if shader is None:
        return None, err
    try:
        out = device.dispatch_compute(
            shader, ow, oh,
            uniforms={'hal_bvh_side': float(n_side),
                      'hal_btris_side': float(t_side),
                      'hal_rays_side': float(r_side),
                      'hal_ray_count': float(n),
                      'hal_out_w': float(ow)},
            samplers={'hal_bvh': device.upload(nodes),
                      'hal_btris': device.upload(tris),
                      'hal_rays': device.upload(rays)},
            images=('hal_out_hits',))
    except Exception as exc:                                    # noqa: BLE001
        return None, f'occlusion dispatch failed: {type(exc).__name__}: {exc}'
    hits = out['hal_out_hits'][:, :, 0].reshape(-1)[:n] > 0.5
    return hits, None


def bvh_fingerprint(bvh):
    """A cheap content key, in the upload cache's strided-sum idiom."""
    v = bvh.verts
    stride = max(1, v.shape[0] // 512)
    return (int(bvh.n_nodes), int(bvh.tris.shape[0]),
            round(float(v[::stride].sum()), 3))


def bvh_atlas_entries(bvh):
    """(entries, consts): the two BVH textures in the shadow-atlas idiom.

    `entries` maps sampler name -> (cache key, build); the pack runs once
    however many lights share it. `consts` carries the texture sides the
    traversal needs as baked floats.
    """
    key = bvh_fingerprint(bvh)
    packed = {}

    def _pack():
        if 'data' not in packed:
            packed['data'] = pack_bvh(bvh)
        return packed['data']

    nodes, n_side, tris, t_side = _pack()
    entries = {
        'hal_bvh': (('bvh_nodes',) + key, lambda: _pack()[0]),
        'hal_btris': (('bvh_tris',) + key, lambda: _pack()[2]),
    }
    consts = {'hal_bvh_side': float(n_side), 'hal_btris_side': float(t_side)}
    return entries, consts
