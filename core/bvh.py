"""Bounding volume hierarchy and ray casting.

Built for numpy's strengths -- and rebuilt when the profiler showed the old
shape was numpy's weakness. The first traversal walked the tree node-by-node
in Python with the rays batched per node: correct, but on a half-million-
triangle scene one shadow pass made ~600k tiny `_slab` calls and 1.6M
sub-millisecond reductions, and 83% of a CPU frame was interpreter overhead
around small arrays.

The traversal is now LEVEL-SYNCHRONOUS: the frontier is a flat array of
(node, ray) pairs, every pair alive this level is slab-tested in ONE
vectorised batch, inner pairs expand to their two children by concatenation,
and every leaf pair explodes into flat (ray, triangle) items tested by ONE
Moller-Trumbore call over the whole level. A wave over the same tree visits
the same (node, ray) pairs and accepts hits by the same epsilons -- only the
schedule changes, ~40 large array ops per query instead of thousands of
small ones.

Closest-hit ties are resolved ORDER-FREE: among equal-t hits the lowest
triangle id wins -- the raster's own named tie rule -- so the answer is a
pure function of the candidate set, not of traversal order. (The old DFS
let whichever leaf it popped first keep an exact tie. The GPU kernel is
unaffected: it routes every near-tie back here by design -- route, never
guess -- and outside its noise window a strict minimum is unique.)
"""

import numpy as np

LEAF_SIZE = 8
MAX_DEPTH = 40

#: cap on (node, ray) pairs processed per wave iteration -- the overflow
#: carries to the next iteration, bounding peak memory on incoherent rays
WAVE_CHUNK = 2_000_000

#: nodes larger than this split by SAH; smaller ones take the median cut.
#: The top of the tree is where a bad cut costs every ray; near the leaves
#: the two heuristics give near-identical trees and SAH's extra per-node
#: arithmetic is pure build time (it is Python-call-bound at 160k nodes)
SAH_THRESHOLD = 64


#: R177: bump when ANY build-affecting code changes (_build, _new_node,
#: the constants above, the split arithmetic) -- the disk cache key
#: carries it, so an old session's tree can never masquerade as the new
#: algorithm's. The suite pins that a cache round-trip is bit-identical.
BUILD_VERSION = 1

#: only meshes at least this big cache to disk: small builds are
#: millisecond-fast and the suite makes hundreds of them
CACHE_MIN_TRIS = 20_000

#: prune bounds for the on-disk cache directory
CACHE_MAX_FILES = 8
CACHE_MAX_BYTES = 768 * 1024 * 1024


class BVH:
    def __init__(self, verts, tris, leaf_size=LEAF_SIZE):
        self._derive(verts, tris)
        n = len(self.tris)
        cap = max(4, 4 * n)
        self.bmin = np.zeros((cap, 3), np.float32)
        self.bmax = np.zeros((cap, 3), np.float32)
        self.left = np.full(cap, -1, np.int32)
        self.right = np.full(cap, -1, np.int32)
        self.start = np.zeros(cap, np.int32)
        self.count = np.zeros(cap, np.int32)
        self.order = np.arange(n, dtype=np.int32)
        self.n_nodes = 0
        self.leaf_size = leaf_size
        if n:
            self._build(0, n, 0)
        self._finalize()

    def _derive(self, verts, tris):
        """The per-triangle arrays that follow from the inputs alone.

        Shared by the builder and the disk-cache loader: recomputing
        them runs the same numpy expressions over the same inputs in
        the same order, so the loaded object is bit-identical to the
        built one everywhere -- only the TREE arrays travel on disk."""
        self.verts = np.ascontiguousarray(verts, dtype=np.float32)
        self.tris = np.ascontiguousarray(tris, dtype=np.int32)
        self.v0 = self.verts[self.tris[:, 0]]
        self.e1 = self.verts[self.tris[:, 1]] - self.v0
        self.e2 = self.verts[self.tris[:, 2]] - self.v0
        p = self.verts[self.tris]                      # (T,3,3)
        self.tmin = p.min(axis=1)
        self.tmax = p.max(axis=1)
        self.centroid = (self.tmin + self.tmax) * 0.5

    # ------------------------------------------------ the disk cache

    @staticmethod
    def _cache_digest(verts, tris):
        import hashlib
        h = hashlib.blake2b(digest_size=20)
        h.update(np.ascontiguousarray(verts, np.float32).tobytes())
        h.update(np.ascontiguousarray(tris, np.int32).tobytes())
        h.update(f'{BUILD_VERSION}/{LEAF_SIZE}/{SAH_THRESHOLD}/'
                 f'{MAX_DEPTH}'.encode())
        return h.hexdigest()

    @staticmethod
    def _cache_dir():
        import os
        import tempfile
        d = os.path.join(tempfile.gettempdir(), 'halcyon_bvh_cache')
        os.makedirs(d, exist_ok=True)
        return d

    @classmethod
    def cached(cls, verts, tris, leaf_size=LEAF_SIZE):
        """Build, or load the identical tree a previous session built.

        R177: the field's cold F12 spent 2.5 s (49% of the frame)
        rebuilding a tree that is a pure function of the mesh -- every
        session, same file, same tree. The tree arrays now persist
        under a content digest (verts + tris bytes + every build
        constant + BUILD_VERSION): a reopened session LOADS the exact
        arrays the build produced, so the pixels cannot move by a bit.
        Any load problem -- missing, truncated, wrong shapes -- falls
        back to a fresh build silently; the cache is never
        load-bearing. Small meshes skip the disk entirely.
        """
        import os
        import time as _t
        n = len(tris)
        if n < CACHE_MIN_TRIS:
            return cls(verts, tris, leaf_size)
        try:
            digest = cls._cache_digest(verts, tris)
            path = os.path.join(cls._cache_dir(), digest + '.npz')
        except Exception as exc:                            # noqa: BLE001
            print(f'[Halcyon] BVH: cache unavailable '
                  f'({type(exc).__name__}: {exc}); building')
            return cls(verts, tris, leaf_size)
        t0 = _t.perf_counter()
        got = cls._cache_load(path, verts, tris, leaf_size)
        if got is not None:
            print(f'[Halcyon] BVH: loaded from cache in '
                  f'{(_t.perf_counter() - t0) * 1000.0:.0f} ms '
                  f'({n:,} tris)')
            return got
        t0 = _t.perf_counter()
        built = cls(verts, tris, leaf_size)
        t_build = _t.perf_counter() - t0
        ok, why = built._cache_save(path)
        # this line is the instrument: a session that BUILDS says
        # whether the next one will load -- and if not, names why
        print(f'[Halcyon] BVH: built in {t_build:.2f} s ({n:,} tris); '
              + ('cached for the next session' if ok
                 else f'CACHE SAVE FAILED ({why}) -- the next session '
                      'builds again'))
        return built

    @classmethod
    def _cache_load(cls, path, verts, tris, leaf_size):
        import os
        if not os.path.exists(path):
            return None
        try:
            with np.load(path, allow_pickle=False) as z:
                tree = {k: z[k] for k in
                        ('bmin', 'bmax', 'left', 'right', 'start',
                         'count', 'order')}
            if tree['order'].size != len(tris) or \
                    tree['bmin'].shape[0] != tree['left'].size:
                return None
            self = cls.__new__(cls)
            self._derive(verts, tris)
            for k, v in tree.items():
                setattr(self, k, np.ascontiguousarray(v))
            self.n_nodes = int(tree['left'].size)
            self.leaf_size = leaf_size
            self._finalize()
            try:
                os.utime(path)          # LRU freshness for the pruner
            except Exception:                               # noqa: BLE001
                pass
            return self
        except Exception:                                   # noqa: BLE001
            return None

    def _cache_save(self, path):
        """(ok, why): write the tree atomically; never raises."""
        import os
        try:
            # the temp name ENDS in .npz so numpy writes exactly this
            # file (it appends .npz to any other suffix -- the classic
            # savez trap: an mkstemp handle would leave the real data
            # stranded under a mangled name and move an empty file)
            tmp = f'{path}.tmp{os.getpid()}.npz'
            np.savez(tmp, bmin=self.bmin, bmax=self.bmax,
                     left=self.left, right=self.right,
                     start=self.start, count=self.count,
                     order=self.order)
            os.replace(tmp, path)
            self._cache_prune(os.path.dirname(path))
            return True, ''
        except Exception as exc:                            # noqa: BLE001
            return False, f'{type(exc).__name__}: {exc}'

    @staticmethod
    def _cache_prune(d):
        import os
        try:
            entries = []
            for f in os.listdir(d):
                if not f.endswith('.npz') or '.tmp' in f:
                    continue
                p = os.path.join(d, f)
                st = os.stat(p)
                entries.append((st.st_mtime, st.st_size, p))
            entries.sort(reverse=True)          # newest first
            total = 0
            for i, (_m, sz, p) in enumerate(entries):
                total += sz
                if i >= CACHE_MAX_FILES or total > CACHE_MAX_BYTES:
                    try:
                        os.remove(p)
                    except Exception:                       # noqa: BLE001
                        pass
        except Exception:                                   # noqa: BLE001
            pass

    def _finalize(self):
        """Trim the build-time over-allocation and lay the node bounds out
        as six flat per-axis arrays.

        The R167 field profile put 8 of a 23-second frame inside
        `_slab_pairs`: the (M,3) gathers and the 3-wide `.max(axis=1)`
        reductions are numpy's weakest shape (45k generic ufunc
        reductions of three elements each). The same test over
        contiguous per-axis columns is plain elementwise arithmetic --
        same values, same operation order, identical bits -- at a
        fraction of the interpreter overhead."""
        nn = self.n_nodes
        for name in ('bmin', 'bmax'):
            setattr(self, name, np.ascontiguousarray(getattr(self, name)[:nn]))
        for name in ('left', 'right', 'start', 'count'):
            setattr(self, name, np.ascontiguousarray(getattr(self, name)[:nn]))
        self._bx0 = np.ascontiguousarray(self.bmin[:, 0])
        self._by0 = np.ascontiguousarray(self.bmin[:, 1])
        self._bz0 = np.ascontiguousarray(self.bmin[:, 2])
        self._bx1 = np.ascontiguousarray(self.bmax[:, 0])
        self._by1 = np.ascontiguousarray(self.bmax[:, 1])
        self._bz1 = np.ascontiguousarray(self.bmax[:, 2])
        #: per-(sun-)direction Moller precomputation cache, small and
        #: keyed by the direction's exact bytes (see _dir_tables)
        self._dir_cache = {}

    def _new_node(self):
        i = self.n_nodes
        self.n_nodes += 1
        if i >= len(self.bmin):
            for name in ('bmin', 'bmax'):
                a = getattr(self, name)
                setattr(self, name, np.vstack([a, np.zeros_like(a)]))
            for name in ('left', 'right', 'start', 'count'):
                a = getattr(self, name)
                setattr(self, name, np.concatenate([a, np.full_like(a, -1)]))
        return i

    def _build(self, start, count, depth):
        """Binned SAH split, median fallback.

        The first build split every node at the centroid median. On a field
        scene -- a 1600-unit ground plane under seven stacked landscapes --
        the median produces siblings whose boxes overlap almost entirely,
        and every ray pays for both subtrees all the way down: the profiler
        measured shadow rays touching hundreds of triangles each. The
        surface-area heuristic puts the cut where (left area x left count +
        right area x right count) is smallest, which is the standard fix
        (Goldsmith/Salmon 1987 by way of every production tracer since).
        Same arrays, same leaf contents, same intersection arithmetic --
        every hit is identical, only the number of boxes a ray visits
        changes -- and the GPU kernels traverse whatever tree is packed,
        so the driver's ray passes ride the same improvement.
        """
        node = self._new_node()
        idx = self.order[start:start + count]
        self.bmin[node] = self.tmin[idx].min(axis=0)
        self.bmax[node] = self.tmax[idx].max(axis=0)
        self.start[node] = start
        self.count[node] = count
        if count <= self.leaf_size or depth >= MAX_DEPTH:
            self.left[node] = -1
            self.right[node] = -1
            return node
        c = self.centroid[idx]
        clo = c.min(axis=0)
        ext = c.max(axis=0) - clo
        axis = int(np.argmax(ext))
        if ext[axis] <= 1e-12:
            self.left[node] = -1
            return node

        mid = None
        K = 16
        if count > SAH_THRESHOLD:
            # bin by centroid along the widest axis
            b = np.minimum((c[:, axis] - clo[axis]) *
                           (K / ext[axis]), K - 1).astype(np.int32)
            nbin = np.bincount(b, minlength=K)
            # per-bin bounds of the member triangles' boxes
            blo = np.full((K, 3), np.inf, np.float32)
            bhi = np.full((K, 3), -np.inf, np.float32)
            np.minimum.at(blo, b, self.tmin[idx])
            np.maximum.at(bhi, b, self.tmax[idx])
            # prefix/suffix accumulation of counts and bounds
            nl = np.cumsum(nbin)[:-1]
            nr = count - nl
            llo = np.minimum.accumulate(blo, axis=0)[:-1]
            lhi = np.maximum.accumulate(bhi, axis=0)[:-1]
            rlo = np.minimum.accumulate(blo[::-1], axis=0)[::-1][1:]
            rhi = np.maximum.accumulate(bhi[::-1], axis=0)[::-1][1:]

            def area(lo, hi):
                d = np.maximum(hi - lo, 0.0)
                return d[:, 0] * d[:, 1] + d[:, 1] * d[:, 2] + \
                    d[:, 2] * d[:, 0]

            cost = np.where((nl > 0) & (nr > 0),
                            area(llo, lhi) * nl + area(rlo, rhi) * nr,
                            np.inf)
            best = int(np.argmin(cost))
            if np.isfinite(cost[best]):
                mask = b <= best
                mid = int(nl[best])
                self.order[start:start + count] = np.concatenate(
                    [idx[mask], idx[~mask]])
        if mid is None or mid <= 0 or mid >= count:
            # degenerate spread: fall back to the median cut
            mid = count // 2
            part = np.argpartition(c[:, axis], mid)
            self.order[start:start + count] = idx[part]
        l = self._build(start, mid, depth + 1)
        r = self._build(start + mid, count - mid, depth + 1)
        self.left[node] = l
        self.right[node] = r
        self.count[node] = 0
        return node

    # ------------------------------------------------------------- traversal
    @staticmethod
    def _slab(org, inv, bmin, bmax, tmax):
        t0 = (bmin - org) * inv
        t1 = (bmax - org) * inv
        tn = np.minimum(t0, t1).max(axis=1)
        tf = np.maximum(t0, t1).min(axis=1)
        return (tf >= np.maximum(tn, 0.0)) & (tn <= tmax)

    def _leaf_items(self, ln, lr, cast=None):
        """Explode leaf (node, ray) pairs into flat (ray, tri) item arrays."""
        counts = self.count[ln].astype(np.int64)
        total = int(counts.sum())
        if total == 0:
            return None, None
        ray_it = np.repeat(lr, counts)
        base = np.repeat(self.start[ln].astype(np.int64), counts)
        offs = np.arange(total, dtype=np.int64) - \
            np.repeat(np.cumsum(counts) - counts, counts)
        tri_it = self.order[base + offs]
        if cast is not None:
            ck = cast[tri_it]
            ray_it = ray_it[ck]
            tri_it = tri_it[ck]
            if ray_it.size == 0:
                return None, None
        return ray_it, tri_it

    def occluded(self, org, dirs, tmax, mask=None, cast=None):
        """Any-hit query. Returns a bool array (N,).

        (A near-to-far sub-batched variant -- sort each wave by slab entry,
        re-cull between sub-batches -- rode this bench and LOST: the sort
        and the 4x call multiplication cost more than the skipped leaves
        saved, 1.15s -> 1.79s on the field scene's shadow rays. Plain
        waves, measured, kept.)

        R167: rays that all share ONE direction -- a sun's shadow rays,
        the field frame's whole cost -- take a specialised lane: the
        slab test broadcasts three scalar inverses instead of gathering
        an (N,3) table, and the Moller-Trumbore terms that depend only
        on (direction, triangle) -- the P vector and the determinant --
        are computed once per (direction, tree) and gathered per item.
        The arithmetic expressions and their operation order are
        UNCHANGED, so every accepted hit is bit-identical; only the
        redundancy is gone.
        """
        n = len(org)
        hit = np.zeros(n, bool)
        if self.n_nodes == 0 or n == 0:
            return hit
        active = np.arange(n) if mask is None else np.nonzero(mask)[0]
        if active.size == 0:
            return hit
        d = np.asarray(dirs, np.float32)
        uniform = d.ndim == 2 and n > 8 and bool((d == d[0]).all())
        tmax = np.asarray(tmax, np.float32)
        if tmax.ndim == 0:
            tmax = np.full(n, float(tmax), np.float32)

        ox = np.ascontiguousarray(org[:, 0])
        oy = np.ascontiguousarray(org[:, 1])
        oz = np.ascontiguousarray(org[:, 2])
        if uniform:
            d0 = d[0]
            iv3 = np.where(np.abs(d0) < 1e-12, 1e12,
                           1.0 / np.where(np.abs(d0) < 1e-12, 1.0, d0))
            ivx, ivy, ivz = iv3[0], iv3[1], iv3[2]
            dt = self._dir_tables(d0)
        else:
            inv = np.where(np.abs(d) < 1e-12, 1e12,
                           1.0 / np.where(np.abs(d) < 1e-12, 1.0, d))
            ivx = np.ascontiguousarray(inv[:, 0])
            ivy = np.ascontiguousarray(inv[:, 1])
            ivz = np.ascontiguousarray(inv[:, 2])
            dt = None

        pn = np.zeros(active.size, np.int32)          # frontier: node ids
        pr = active.astype(np.int64)                  # frontier: ray ids
        while pn.size:
            m = min(pn.size, WAVE_CHUNK)
            nsel, rsel = pn[:m], pr[:m]
            rest_n, rest_r = pn[m:], pr[m:]
            live = ~hit[rsel]
            nsel, rsel = nsel[live], rsel[live]
            if nsel.size == 0:
                pn, pr = rest_n, rest_r
                continue
            o0, o1, o2 = ox[rsel], oy[rsel], oz[rsel]
            if uniform:
                keep = self._slab_cols(o0, o1, o2, ivx, ivy, ivz,
                                       tmax[rsel], nsel)
            else:
                keep = self._slab_cols(o0, o1, o2, ivx[rsel], ivy[rsel],
                                       ivz[rsel], tmax[rsel], nsel)
            nsel, rsel = nsel[keep], rsel[keep]
            if nsel.size == 0:
                pn, pr = rest_n, rest_r
                continue
            lf = self.left[nsel] < 0
            ray_it, tri_it = self._leaf_items(nsel[lf], rsel[lf], cast)
            if ray_it is not None:
                if uniform:
                    t, _u, _v, ok = self._moller_flat_dir(
                        org[ray_it], d0, tri_it, dt)
                else:
                    t, _u, _v, ok = self._moller_flat(org[ray_it],
                                                      d[ray_it], tri_it)
                ok &= (t > 1e-6) & (t < tmax[ray_it])
                hit[ray_it[ok]] = True
            inn, inr = nsel[~lf], rsel[~lf]
            pn = np.concatenate([rest_n, self.left[inn], self.right[inn]])
            pr = np.concatenate([rest_r, inr, inr])
        return hit

    def _slab_cols(self, ox, oy, oz, ivx, ivy, ivz, tlimit, nodes):
        """The slab test, unrolled per axis over contiguous columns.

        Exactly `_slab_pairs`' arithmetic: numpy's 3-wide `.max(axis=1)`
        reduces ((x, y), z) pairwise, and the explicit nesting below is
        that same order -- identical bits, none of the reduction
        machinery. `iv*` may be scalars (the uniform-direction lane) or
        per-ray columns; broadcasting changes no rounding."""
        b0 = self._bx0[nodes]
        b1 = self._bx1[nodes]
        t0 = (b0 - ox) * ivx
        t1 = (b1 - ox) * ivx
        tn = np.minimum(t0, t1)
        tf = np.maximum(t0, t1)
        b0 = self._by0[nodes]
        b1 = self._by1[nodes]
        t0 = (b0 - oy) * ivy
        t1 = (b1 - oy) * ivy
        tn = np.maximum(tn, np.minimum(t0, t1))
        tf = np.minimum(tf, np.maximum(t0, t1))
        b0 = self._bz0[nodes]
        b1 = self._bz1[nodes]
        t0 = (b0 - oz) * ivz
        t1 = (b1 - oz) * ivz
        tn = np.maximum(tn, np.minimum(t0, t1))
        tf = np.minimum(tf, np.maximum(t0, t1))
        return (tf >= np.maximum(tn, 0.0)) & (tn <= tlimit)

    def _slab_pairs(self, org, inv, tlimit, nodes, rays):
        """The slab test over a whole wave of (node, ray) pairs at once."""
        return self._slab_cols(org[rays, 0], org[rays, 1], org[rays, 2],
                               inv[rays, 0], inv[rays, 1], inv[rays, 2],
                               tlimit[rays], nodes)

    def _dir_tables(self, d0):
        """Per-triangle Moller terms that depend only on the direction:
        P = d x e2 (component-wise, the same expressions `_moller_flat`
        writes out), det = e1 . P, and its guarded inverse. One
        computation per (direction, tree) serves every wave and every
        chunk of a sun's shadow pass; keyed by the direction's exact
        bytes, capped small."""
        key = d0.tobytes()
        hitc = self._dir_cache.get(key)
        if hitc is not None:
            return hitc
        e1 = self.e1
        e2 = self.e2
        dx, dy, dz = d0[0], d0[1], d0[2]
        px = dy * e2[:, 2] - dz * e2[:, 1]
        py = dz * e2[:, 0] - dx * e2[:, 2]
        pz = dx * e2[:, 1] - dy * e2[:, 0]
        det = e1[:, 0] * px + e1[:, 1] * py + e1[:, 2] * pz
        ok = np.abs(det) > 1e-12
        inv_det = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        tab = (px, py, pz, inv_det, ok)
        if len(self._dir_cache) > 8:
            self._dir_cache.clear()
        self._dir_cache[key] = tab
        return tab

    def _moller_flat_dir(self, org, d0, tri_it, tab):
        """`_moller_flat` for a single shared direction: P, det and the
        guarded inverse arrive precomputed per triangle (same
        expressions, same order -- identical bits); the per-item work
        is the T vector, the Q cross product and three dot products."""
        px = tab[0][tri_it]
        py = tab[1][tri_it]
        pz = tab[2][tri_it]
        inv_det = tab[3][tri_it]
        ok = tab[4][tri_it]
        v0 = self.v0[tri_it]
        e1 = self.e1[tri_it]
        e2 = self.e2[tri_it]
        tv = org - v0
        u = (tv[:, 0] * px + tv[:, 1] * py + tv[:, 2] * pz) * inv_det
        qx = tv[:, 1] * e1[:, 2] - tv[:, 2] * e1[:, 1]
        qy = tv[:, 2] * e1[:, 0] - tv[:, 0] * e1[:, 2]
        qz = tv[:, 0] * e1[:, 1] - tv[:, 1] * e1[:, 0]
        dx, dy, dz = d0[0], d0[1], d0[2]
        v = (dx * qx + dy * qy + dz * qz) * inv_det
        t = (e2[:, 0] * qx + e2[:, 1] * qy + e2[:, 2] * qz) * inv_det
        ok = ok & (u >= -1e-6) & (v >= -1e-6) & (u + v <= 1.0 + 1e-6)
        return t, u, v, ok

    def intersect(self, org, dirs, tmax, mask=None):
        """Closest-hit query. Returns (tri_id (N,) int32 or -1, t, u, v)."""
        n = len(org)
        best_t = np.asarray(tmax, np.float32)
        if best_t.ndim == 0:
            best_t = np.full(n, float(tmax), np.float32)
        else:
            best_t = best_t.copy()
        best_id = np.full(n, -1, np.int32)
        best_u = np.zeros(n, np.float32)
        best_v = np.zeros(n, np.float32)
        if self.n_nodes == 0 or n == 0:
            return best_id, best_t, best_u, best_v
        active = np.arange(n) if mask is None else np.nonzero(mask)[0]
        if active.size == 0:
            return best_id, best_t, best_u, best_v
        d = dirs
        inv = np.where(np.abs(d) < 1e-12, 1e12, 1.0 / np.where(np.abs(d) < 1e-12, 1.0, d))
        pn = np.zeros(active.size, np.int32)
        pr = active.astype(np.int64)
        while pn.size:
            m = min(pn.size, WAVE_CHUNK)
            nsel, rsel = pn[:m], pr[:m]
            rest_n, rest_r = pn[m:], pr[m:]
            keep = self._slab_pairs(org, inv, best_t, nsel, rsel)
            nsel, rsel = nsel[keep], rsel[keep]
            if nsel.size == 0:
                pn, pr = rest_n, rest_r
                continue
            lf = self.left[nsel] < 0
            ray_it, tri_it = self._leaf_items(nsel[lf], rsel[lf])
            if ray_it is not None:
                t, u, v, ok = self._moller_flat(org[ray_it], d[ray_it], tri_it)
                # non-strict against the current best so an exact tie
                # reaches the tie rule instead of vanishing at the gate
                ok &= (t > 1e-6) & (t <= best_t[ray_it])
                if ok.any():
                    ri = ray_it[ok]
                    ti = t[ok]
                    ii = tri_it[ok].astype(np.int64)
                    ui = u[ok]
                    vi = v[ok]
                    # order-free reduction: min over (t, tri id) per ray.
                    # lexsort keys run last-is-primary; 'first occurrence
                    # per ray' after the sort IS that lexicographic min
                    o = np.lexsort((ii, ti, ri))
                    ri, ti, ii, ui, vi = ri[o], ti[o], ii[o], ui[o], vi[o]
                    first = np.unique(ri, return_index=True)[1]
                    cr, ct = ri[first], ti[first]
                    ci, cu, cv = ii[first], ui[first], vi[first]
                    # strictly closer always wins; an exact tie falls to
                    # the lowest triangle id, but never displaces the
                    # tmax sentinel (id -1 means 'no hit yet')
                    prev_t = best_t[cr]
                    prev_i = best_id[cr]
                    better = (ct < prev_t) | \
                        ((ct == prev_t) & (prev_i >= 0) & (ci < prev_i))
                    sel = cr[better]
                    best_t[sel] = ct[better]
                    best_id[sel] = ci[better].astype(np.int32)
                    best_u[sel] = cu[better]
                    best_v[sel] = cv[better]
            inn, inr = nsel[~lf], rsel[~lf]
            pn = np.concatenate([rest_n, self.left[inn], self.right[inn]])
            pr = np.concatenate([rest_r, inr, inr])
        return best_id, best_t, best_u, best_v

    def _moller_flat(self, org, dirs, tri_it):
        """Moller-Trumbore over flat (ray, triangle) ITEMS: all inputs (M,3)
        or (M,), outputs (M,). The same arithmetic as `_moller`, without the
        (rays x leaf) broadcast -- one item is one ray against one triangle,
        which is exactly the shape the wave traversal's leaf explosion
        produces. The cross products are written out component-wise:
        np.cross on broadcast views was 70 seconds of a profiled frame."""
        v0 = self.v0[tri_it]
        e1 = self.e1[tri_it]
        e2 = self.e2[tri_it]
        px = dirs[:, 1] * e2[:, 2] - dirs[:, 2] * e2[:, 1]
        py = dirs[:, 2] * e2[:, 0] - dirs[:, 0] * e2[:, 2]
        pz = dirs[:, 0] * e2[:, 1] - dirs[:, 1] * e2[:, 0]
        det = e1[:, 0] * px + e1[:, 1] * py + e1[:, 2] * pz
        ok = np.abs(det) > 1e-12
        inv_det = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        tv = org - v0
        u = (tv[:, 0] * px + tv[:, 1] * py + tv[:, 2] * pz) * inv_det
        qx = tv[:, 1] * e1[:, 2] - tv[:, 2] * e1[:, 1]
        qy = tv[:, 2] * e1[:, 0] - tv[:, 0] * e1[:, 2]
        qz = tv[:, 0] * e1[:, 1] - tv[:, 1] * e1[:, 0]
        v = (dirs[:, 0] * qx + dirs[:, 1] * qy + dirs[:, 2] * qz) * inv_det
        t = (e2[:, 0] * qx + e2[:, 1] * qy + e2[:, 2] * qz) * inv_det
        ok &= (u >= -1e-6) & (v >= -1e-6) & (u + v <= 1.0 + 1e-6)
        return t, u, v, ok

    def _moller(self, org, dirs, tri_ids):
        """(R,3),(R,3),(K,) -> t,u,v,ok each (R,K)."""
        v0 = self.v0[tri_ids][None, :, :]
        e1 = self.e1[tri_ids][None, :, :]
        e2 = self.e2[tri_ids][None, :, :]
        r, k = org.shape[0], len(tri_ids)
        d = np.broadcast_to(dirs[:, None, :], (r, k, 3))
        e1b = np.broadcast_to(e1, (r, k, 3))
        e2b = np.broadcast_to(e2, (r, k, 3))
        p = np.cross(d, e2b)
        det = np.einsum('rkc,rkc->rk', e1b, p)
        ok = np.abs(det) > 1e-12
        inv_det = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        tvec = org[:, None, :] - v0
        u = np.einsum('rkc,rkc->rk', tvec, p) * inv_det
        q = np.cross(tvec, e1b)
        v = np.einsum('rkc,rkc->rk', d, q) * inv_det
        t = np.einsum('rkc,rkc->rk', e2b, q) * inv_det
        ok &= (u >= -1e-6) & (v >= -1e-6) & (u + v <= 1.0 + 1e-6)
        return t, u, v, ok


def make_bvh(mesh, cast_mask=None):
    if mesh is None or mesh.tris is None or len(mesh.tris) == 0:
        return None
    return BVH.cached(mesh.verts, mesh.tris)
