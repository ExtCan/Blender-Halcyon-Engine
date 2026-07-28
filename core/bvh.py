"""Bounding volume hierarchy and ray casting.

Built for numpy's strengths: the tree is walked node-by-node in Python while
every ray that reaches a node is tested as one vectorised batch. Leaf tests are
Moller-Trumbore over (rays x leaf triangles).
"""

import numpy as np

LEAF_SIZE = 8
MAX_DEPTH = 40


class BVH:
    def __init__(self, verts, tris, leaf_size=LEAF_SIZE):
        self.verts = np.ascontiguousarray(verts, dtype=np.float32)
        self.tris = np.ascontiguousarray(tris, dtype=np.int32)
        n = len(self.tris)
        self.v0 = self.verts[self.tris[:, 0]]
        self.e1 = self.verts[self.tris[:, 1]] - self.v0
        self.e2 = self.verts[self.tris[:, 2]] - self.v0

        p = self.verts[self.tris]                      # (T,3,3)
        self.tmin = p.min(axis=1)
        self.tmax = p.max(axis=1)
        self.centroid = (self.tmin + self.tmax) * 0.5

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
        ext = c.max(axis=0) - c.min(axis=0)
        axis = int(np.argmax(ext))
        if ext[axis] <= 1e-12:
            self.left[node] = -1
            return node
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

    def occluded(self, org, dirs, tmax, mask=None, cast=None):
        """Any-hit query. Returns a bool array (N,)."""
        n = len(org)
        hit = np.zeros(n, bool)
        if self.n_nodes == 0 or n == 0:
            return hit
        active = np.arange(n) if mask is None else np.nonzero(mask)[0]
        if active.size == 0:
            return hit
        d = dirs
        inv = np.where(np.abs(d) < 1e-12, 1e12, 1.0 / np.where(np.abs(d) < 1e-12, 1.0, d))
        tmax = np.asarray(tmax, np.float32)
        if tmax.ndim == 0:
            tmax = np.full(n, float(tmax), np.float32)

        stack = [(0, active)]
        while stack:
            node, rays = stack.pop()
            if rays.size == 0:
                continue
            live = ~hit[rays]
            rays = rays[live]
            if rays.size == 0:
                continue
            keep = self._slab(org[rays], inv[rays], self.bmin[node], self.bmax[node],
                              tmax[rays])
            rays = rays[keep]
            if rays.size == 0:
                continue
            if self.left[node] < 0:
                tri_ids = self.order[self.start[node]:self.start[node] + self.count[node]]
                if cast is not None:
                    tri_ids = tri_ids[cast[tri_ids]]
                    if tri_ids.size == 0:
                        continue
                t, u, v, ok = self._moller(org[rays], dirs[rays], tri_ids)
                ok &= (t > 1e-6) & (t < tmax[rays][:, None])
                any_hit = ok.any(axis=1)
                hit[rays[any_hit]] = True
            else:
                stack.append((int(self.left[node]), rays))
                stack.append((int(self.right[node]), rays))
        return hit

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
        stack = [(0, active)]
        while stack:
            node, rays = stack.pop()
            if rays.size == 0:
                continue
            keep = self._slab(org[rays], inv[rays], self.bmin[node], self.bmax[node],
                              best_t[rays])
            rays = rays[keep]
            if rays.size == 0:
                continue
            if self.left[node] < 0:
                tri_ids = self.order[self.start[node]:self.start[node] + self.count[node]]
                t, u, v, ok = self._moller(org[rays], dirs[rays], tri_ids)
                t = np.where(ok & (t > 1e-6), t, np.inf)
                k = np.argmin(t, axis=1)
                rows = np.arange(len(rays))
                tt = t[rows, k]
                better = tt < best_t[rays]
                sel = rays[better]
                best_t[sel] = tt[better]
                best_id[sel] = tri_ids[k[better]]
                best_u[sel] = u[rows, k][better]
                best_v[sel] = v[rows, k][better]
            else:
                stack.append((int(self.left[node]), rays))
                stack.append((int(self.right[node]), rays))
        return best_id, best_t, best_u, best_v

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
    return BVH(mesh.verts, mesh.tris)
