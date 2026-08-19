"""Blender Internal subsurface scattering, transcribed from 2.79.

Source of truth: sss.c (scatter_settings_new, the dipole Rd with its
lookup tables, the octree build/aggregate/traversal, compute_radiance),
rendercore.c's shade_sample_sss (the pre-pass point area and colour) and
shadeinput.c's shade_input_calc_viewco (the pixel-footprint
derivatives) -- all fetched verbatim in R162 and archived under
blender279src/. The application block in shade_lamp_loop lives in
core/render.py's light loop.

The arithmetic is BI's, kept quirks and all:

- ``Fdr = -1.440f/ior*ior + ...`` -- the shipping source has NO
  parentheses, so the first term parses ``(-1.440/ior)*ior = -1.440``.
  2.79 rendered with that for its whole life; so does this port.
- The secant solve for the reduced albedo, the two Rd lookup tables
  (squared-distance to rr=100, plain distance to r=10000), the
  8-point/15-depth octree with the single-subnode collapse, the
  rad-weighted node positions, the ``area+backarea > error*dist``
  acceptance, and the front/back MAX2 combine are all structural
  transcriptions.
- Points and queries live in CAMERA space (shi->co was), so the octree
  splits -- and therefore the exact aggregation pattern -- follow the
  camera exactly as BI's did.

One documented divergence: NumPy accumulates the per-query sums in
vectorised batches, so float addition ORDER differs from the C's
sequential depth-first walk. The result is deterministic here (fixed
traversal order, fixed dtype) but can differ from a byte-compare with
2.79 in the last float digits -- the same class of difference every
``.sum()`` in the engine already carries.
"""

import numpy as np

RD_TABLE_RANGE = 100.0
RD_TABLE_RANGE_2 = 10000.0
RD_TABLE_SIZE = 10000
MAX_OCTREE_NODE_POINTS = 8
MAX_OCTREE_DEPTH = 15

_F = np.float32


def _f_rd(alpha_, a, ro):
    sq = np.sqrt(3.0 * (1.0 - alpha_))
    return (alpha_ / 2.0) * (1.0 + np.exp((-4.0 / 3.0) * a * sq)) \
        * np.exp(-sq) - ro


def _compute_reduced_albedo(a, ro):
    """The C's secant iteration, scalar and verbatim."""
    tolerance = 1e-8
    xn_1, xn = 0.0, 1.0
    fxn = _f_rd(xn, a, ro)
    fxn_1 = _f_rd(xn_1, a, ro)
    for _i in range(20):
        fsub = (fxn - fxn_1)
        if abs(fsub) < tolerance:
            break
        d = ((xn - xn_1) / fsub) * fxn
        if abs(d) < tolerance:
            break
        xn_1 = xn
        fxn_1 = fxn
        xn = xn - d
        if xn > 1.0:
            xn = 1.0
        if xn_1 > 1.0:
            xn_1 = 1.0
        fxn = _f_rd(xn, a, ro)
    if xn <= 0.0:
        xn = 0.00001
    return float(xn)


class ChannelSettings:
    """scatter_settings_new, one colour channel."""

    def __init__(self, refl, radius, ior, reflfac, frontweight,
                 backweight):
        ior = float(ior)
        self.eta = ior
        # the shipping 2.79 line has NO parentheses: (-1.440/ior)*ior
        self.fdr = -1.440 / ior * ior + 0.710 / ior + 0.668 \
            + 0.0636 * ior
        self.a = (1.0 + self.fdr) / (1.0 - self.fdr)
        self.ld = max(float(radius), 1e-6)
        self.ro = min(float(refl), 0.99)
        self.color = self.ro * float(reflfac) + (1.0 - float(reflfac))
        self.alpha_ = _compute_reduced_albedo(self.a, self.ro)
        self.sigma = 1.0 / self.ld
        self.sigma_t_ = self.sigma / np.sqrt(3.0 * (1.0 - self.alpha_))
        self.sigma_s_ = self.alpha_ * self.sigma_t_
        self.sigma_a = self.sigma_t_ - self.sigma_s_
        self.d = 1.0 / (3.0 * self.sigma_t_)
        self.zr = 1.0 / self.sigma_t_
        self.zv = self.zr + 4.0 * self.a * self.d
        self.frontweight = float(frontweight)
        self.backweight = float(backweight)
        self._build_rd_table()

    def rd_rsquare(self, rr):
        """The dipole, on SQUARED distance (vectorised)."""
        rr = np.asarray(rr, np.float64)
        sr = np.sqrt(rr + self.zr * self.zr)
        sv = np.sqrt(rr + self.zv * self.zv)
        rdr = self.zr * (1.0 + self.sigma * sr) \
            * np.exp(-self.sigma * sr) / (sr * sr * sr)
        rdv = self.zv * (1.0 + self.sigma * sv) \
            * np.exp(-self.sigma * sv) / (sv * sv * sv)
        return ((1.0 / (4.0 * np.pi)) * (rdr + rdv)).astype(_F)

    def rd(self, r):
        r = np.asarray(r, np.float64)
        return self.rd_rsquare(r * r)

    def _build_rd_table(self):
        i = np.arange(RD_TABLE_SIZE + 1, dtype=np.float64)
        # tableRd is indexed by SQUARED distance: Rd(sqrt(rr))
        rr = i * (RD_TABLE_RANGE / RD_TABLE_SIZE)
        self.table_rd = self.rd(np.sqrt(rr))
        # tableRd2 by plain distance, out to 10000
        r2 = i * (RD_TABLE_RANGE_2 / RD_TABLE_SIZE)
        self.table_rd2 = self.rd(r2)


def approximate_rd_rgb(ss3, rr):
    """approximate_Rd_rgb vectorised: (Q,) squared distances -> (Q,3).

    Branches exactly as the C: beyond RANGE_2^2 (or past the table end)
    the true dipole; above RANGE the distance table; else the
    squared-distance table; linear interpolation between entries."""
    rr = np.asarray(rr, np.float64)
    q = rr.shape[0]
    out = np.empty((q, 3), _F)
    big = rr > (RD_TABLE_RANGE_2 * RD_TABLE_RANGE_2)
    mid = (~big) & (rr > RD_TABLE_RANGE)
    low = ~(big | mid)

    if np.any(mid):
        r = np.sqrt(rr[mid])
        indexf = r * (RD_TABLE_SIZE / RD_TABLE_RANGE_2)
        index = indexf.astype(np.int64)
        t = (indexf - index).astype(_F)
        ok = (index >= 0) & (index < RD_TABLE_SIZE)
        idx = np.clip(index, 0, RD_TABLE_SIZE - 1)
        for c, ss in enumerate(ss3):
            v = ss.table_rd2[idx] * (1.0 - t) + ss.table_rd2[idx + 1] * t
            out[np.nonzero(mid)[0], c] = np.where(
                ok, v, ss.rd_rsquare(rr[mid]))
    if np.any(low):
        indexf = rr[low] * (RD_TABLE_SIZE / RD_TABLE_RANGE)
        index = indexf.astype(np.int64)
        t = (indexf - index).astype(_F)
        ok = (index >= 0) & (index < RD_TABLE_SIZE)
        idx = np.clip(index, 0, RD_TABLE_SIZE - 1)
        for c, ss in enumerate(ss3):
            v = ss.table_rd[idx] * (1.0 - t) + ss.table_rd[idx + 1] * t
            out[np.nonzero(low)[0], c] = np.where(
                ok, v, ss.rd_rsquare(rr[low]))
    if np.any(big):
        for c, ss in enumerate(ss3):
            out[np.nonzero(big)[0], c] = ss.rd_rsquare(rr[big])
    return out


class ScatterTree:
    """The octree, built and traversed with BI's exact rules."""

    def __init__(self, ss3, scale, error, co, color, area):
        """co (N,3) camera-space, color (N,3), area (N,) SIGNED --
        negative marks a back-layer point, exactly the tile pass."""
        scale = max(float(scale), 1e-9)
        self.ss = list(ss3)
        self.scale = scale
        self.error = max(float(error), 1e-6)
        co = np.asarray(co, _F) / _F(scale)
        area_in = np.asarray(area, _F)
        self.pco = co
        self.prad = np.asarray(color, _F).copy()
        self.parea = np.abs(area_in) / _F(scale * scale)
        self.pback = area_in < 0.0
        self.n = co.shape[0]
        # node storage (lists appended during the recursive build)
        self.children = []      # (8,) int arrays, -1 empty; None = leaf
        self.split = []         # (3,) float or None
        self.pstart = []        # leaf slice into the permuted order
        self.pcount = []
        self.agg_co = []
        self.agg_rad = []
        self.agg_brad = []
        self.agg_area = []
        self.agg_barea = []
        self.order = np.arange(self.n, dtype=np.int64)
        if self.n:
            lo = co.min(axis=0)
            hi = co.max(axis=0)
            mid = (lo + hi) * 0.5
            size = (hi - lo) * 0.5
            self.root = self._create(0, self.n, mid, size, 0)
        else:
            self.root = -1
        self._finalize()

    # ---- build: create_octree_node verbatim (incl. the one-subnode
    # collapse and the depth-15 / 8-point leaf rule) ----
    def _new_node(self):
        self.children.append(None)
        self.split.append(None)
        self.pstart.append(0)
        self.pcount.append(0)
        self.agg_co.append(None)
        self.agg_rad.append(None)
        self.agg_brad.append(None)
        self.agg_area.append(0.0)
        self.agg_barea.append(0.0)
        return len(self.children) - 1

    def _create(self, start, count, mid, size, depth):
        node = self._new_node()
        self._fill(node, start, count, np.asarray(mid, _F),
                   np.asarray(size, _F), depth)
        self._aggregate(node)
        return node

    def _fill(self, node, start, count, mid, size, depth):
        while True:
            if count <= MAX_OCTREE_NODE_POINTS or depth == MAX_OCTREE_DEPTH:
                self.pstart[node] = start
                self.pcount[node] = count
                return
            idx = self.order[start:start + count]
            co = self.pco[idx]
            sub = ((co[:, 0] >= mid[0]).astype(np.int64)
                   + (co[:, 1] >= mid[1]).astype(np.int64) * 2
                   + (co[:, 2] >= mid[2]).astype(np.int64) * 4)
            counts = np.bincount(sub, minlength=8)
            used = np.nonzero(counts)[0]
            subsize = size * _F(0.5)
            if used.size <= 1:
                # all points in ONE subnode: descend the SAME node into
                # that subnode's cell (no split recorded), depth+1
                usedi = int(used[0]) if used.size else 0
                mid = _subnode_middle(usedi, mid, subsize)
                size = subsize
                depth += 1
                continue
            # partition the order slice by subnode (stable, the C's
            # offset walk keeps first-seen order within each bucket)
            perm = np.argsort(sub, kind='stable')
            self.order[start:start + count] = idx[perm]
            self.split[node] = mid.copy()
            kids = np.full(8, -1, np.int64)
            off = start
            for i in range(8):
                c = int(counts[i])
                if c > 0:
                    kids[i] = self._create(
                        off, c, _subnode_middle(i, mid, subsize),
                        subsize, depth + 1)
                    off += c
            self.children[node] = kids
            self.pstart[node] = 0
            self.pcount[node] = 0
            return

    # ---- aggregation: sum_leaf_radiance / sum_branch_radiance ----
    def _aggregate(self, node):
        if self.children[node] is None:
            idx = self.order[self.pstart[node]:
                             self.pstart[node] + self.pcount[node]]
            co = self.pco[idx]
            rad = self.prad[idx]
            area = self.parea[idx]
            back = self.pback[idx]
            radw = area * np.abs(rad.sum(axis=1))
            totrad = float(radw.sum())
            f = ~back
            farea = float(area[f].sum())
            barea = float(area[back].sum())
            fr = (rad[f] * area[f, None]).sum(axis=0)
            br = (rad[back] * area[back, None]).sum(axis=0)
            if farea > 1e-16:
                fr = fr / farea
            if barea > 1e-16:
                br = br / barea
            if totrad > 1e-16:
                nco = (co * radw[:, None]).sum(axis=0) / totrad
            else:
                nco = co.mean(axis=0) if idx.size else np.zeros(3, _F)
            self.agg_co[node] = nco.astype(_F)
            self.agg_rad[node] = fr.astype(_F)
            self.agg_brad[node] = br.astype(_F)
            self.agg_area[node] = farea
            self.agg_barea[node] = barea
        else:
            kids = [k for k in self.children[node] if k >= 0]
            a = np.array([self.agg_area[k] for k in kids], _F)
            ba = np.array([self.agg_barea[k] for k in kids], _F)
            r = np.stack([self.agg_rad[k] for k in kids])
            br = np.stack([self.agg_brad[k] for k in kids])
            cos = np.stack([self.agg_co[k] for k in kids])
            radw = a * np.abs(r.sum(axis=1)) + ba * np.abs(br.sum(axis=1))
            totrad = float(radw.sum())
            farea = float(a.sum())
            barea = float(ba.sum())
            fr = (r * a[:, None]).sum(axis=0)
            bbr = (br * ba[:, None]).sum(axis=0)
            if farea > 1e-16:
                fr = fr / farea
            if barea > 1e-16:
                bbr = bbr / barea
            if totrad > 1e-16:
                nco = (cos * radw[:, None]).sum(axis=0) / totrad
            else:
                nco = cos.mean(axis=0)
            self.agg_co[node] = nco.astype(_F)
            self.agg_rad[node] = fr.astype(_F)
            self.agg_brad[node] = bbr.astype(_F)
            self.agg_area[node] = farea
            self.agg_barea[node] = barea

    # ---- traversal: traverse_octree + add_radiance + compute_radiance,
    # vectorised over (query, node) PAIR LISTS, level by level. The
    # rules are the C's exactly; only the batching differs (and with it
    # the float accumulation ORDER -- the documented divergence).
    def _finalize(self):
        """Flatten the built tree into arrays the sampler gathers."""
        m = len(self.children)
        self.f_children = np.full((m, 8), -1, np.int64)
        self.f_leaf = np.zeros(m, bool)
        self.f_split = np.zeros((m, 3), _F)
        # a depth-15 leaf keeps EVERY remaining point (the C's cap), so
        # the padded width is the largest leaf, not the 8-point rule
        maxp = 1
        for i in range(m):
            if self.children[i] is None:
                maxp = max(maxp, self.pcount[i])
        self.f_pts = np.full((m, maxp), -1, np.int64)
        for i in range(m):
            if self.children[i] is None:
                self.f_leaf[i] = True
                idx = self.order[self.pstart[i]:
                                 self.pstart[i] + self.pcount[i]]
                self.f_pts[i, :idx.size] = idx
            else:
                self.f_children[i] = self.children[i]
                self.f_split[i] = self.split[i]
        self.f_aco = np.stack([c if c is not None else np.zeros(3, _F)
                               for c in self.agg_co]) if m else \
            np.zeros((0, 3), _F)
        self.f_arad = np.stack([c if c is not None else np.zeros(3, _F)
                                for c in self.agg_rad]) if m else \
            np.zeros((0, 3), _F)
        self.f_abrad = np.stack([c if c is not None else np.zeros(3, _F)
                                 for c in self.agg_brad]) if m else \
            np.zeros((0, 3), _F)
        self.f_aarea = np.asarray(self.agg_area, _F) if m else \
            np.zeros(0, _F)
        self.f_abarea = np.asarray(self.agg_barea, _F) if m else \
            np.zeros(0, _F)
        # padded point attributes (last row = zero pad for id -1)
        self.f_pco = np.vstack([self.pco, np.zeros((1, 3), _F)]) \
            if self.n else np.zeros((1, 3), _F)
        self.f_prad = np.vstack([self.prad, np.zeros((1, 3), _F)]) \
            if self.n else np.zeros((1, 3), _F)
        self.f_parea = np.concatenate([self.parea, np.zeros(1, _F)]) \
            if self.n else np.zeros(1, _F)
        self.f_pback = np.concatenate([self.pback, np.zeros(1, bool)]) \
            if self.n else np.zeros(1, bool)
        # stacked channel tables for one-pass RGB lookups
        self._t1 = np.stack([s.table_rd for s in self.ss], axis=1)
        self._t2 = np.stack([s.table_rd2 for s in self.ss], axis=1)
        self._zr = np.array([s.zr for s in self.ss], np.float64)
        self._zv = np.array([s.zv for s in self.ss], np.float64)
        self._sg = np.array([s.sigma for s in self.ss], np.float64)

    def _rd_exact_rgb(self, rr):
        rr = np.asarray(rr, np.float64)[:, None]
        sr = np.sqrt(rr + self._zr[None, :] ** 2)
        sv = np.sqrt(rr + self._zv[None, :] ** 2)
        rdr = self._zr * (1.0 + self._sg * sr) \
            * np.exp(-self._sg * sr) / (sr * sr * sr)
        rdv = self._zv * (1.0 + self._sg * sv) \
            * np.exp(-self._sg * sv) / (sv * sv * sv)
        return ((1.0 / (4.0 * np.pi)) * (rdr + rdv)).astype(_F)

    def _rd_rgb(self, rr):
        """approximate_Rd_rgb, all three channels in one pass."""
        rr = np.asarray(rr, np.float64)
        out = np.empty((rr.shape[0], 3), _F)
        big = rr > (RD_TABLE_RANGE_2 * RD_TABLE_RANGE_2)
        mid = (~big) & (rr > RD_TABLE_RANGE)
        low = ~(big | mid)
        if np.any(low):
            li = np.nonzero(low)[0]
            indexf = rr[li] * (RD_TABLE_SIZE / RD_TABLE_RANGE)
            index = indexf.astype(np.int64)
            t = (indexf - index).astype(_F)[:, None]
            idx = np.clip(index, 0, RD_TABLE_SIZE - 1)
            out[li] = self._t1[idx] * (1.0 - t) + self._t1[idx + 1] * t
        if np.any(mid):
            mi = np.nonzero(mid)[0]
            r = np.sqrt(rr[mi])
            indexf = r * (RD_TABLE_SIZE / RD_TABLE_RANGE_2)
            index = indexf.astype(np.int64)
            t = (indexf - index).astype(_F)[:, None]
            ok = index < RD_TABLE_SIZE
            idx = np.clip(index, 0, RD_TABLE_SIZE - 1)
            v = self._t2[idx] * (1.0 - t) + self._t2[idx + 1] * t
            if not ok.all():
                v[~ok] = self._rd_exact_rgb(rr[mi][~ok])
            out[mi] = v
        if np.any(big):
            bi = np.nonzero(big)[0]
            out[bi] = self._rd_exact_rgb(rr[bi])
        return out

    def sample(self, qco, chunk=8192):
        """(Q,3) camera-space points -> (Q,3) scattered radiance.

        Queries are independent, so they stream through in chunks: the
        pair list a pathological scene can fan out (a hierarchy whose
        node areas dwarf error*dist descends everywhere, exactly as
        BI's recursive walk did) stays bounded in memory."""
        qco = np.asarray(qco, _F)
        q = qco.shape[0]
        if q > chunk:
            out = np.empty((q, 3), _F)
            for s in range(0, q, chunk):
                out[s:s + chunk] = self.sample(qco[s:s + chunk], chunk)
            return out
        qco = qco / _F(self.scale)
        rad = np.zeros((q, 3), _F)
        brad = np.zeros((q, 3), _F)
        rdsum = np.zeros((q, 3), _F)
        brdsum = np.zeros((q, 3), _F)
        if self.root < 0 or self.n == 0 or q == 0:
            return rad
        error = _F(self.error)
        qn = np.arange(q, dtype=np.int64)
        nd = np.full(q, self.root, np.int64)
        sf = np.ones(q, bool)
        while qn.size:
            leaf_m = self.f_leaf[nd]
            if np.any(leaf_m):
                lq = qn[leaf_m]
                ln = nd[leaf_m]
                pts = self.f_pts[ln]                      # (P, K)
                valid = pts >= 0
                pid = np.where(valid, pts, self.n)        # pad row
                d = qco[lq][:, None, :] - self.f_pco[pid]
                rr = (d * d).sum(axis=2)                  # (P, K)
                vi = np.nonzero(valid)
                rd = self._rd_rgb(rr[vi])                 # (V, 3)
                pv = pid[vi]
                prd = rd * self.f_parea[pv][:, None]
                contrib = self.f_prad[pv] * prd
                qrep = lq[vi[0]]
                bk = self.f_pback[pv]
                if np.any(~bk):
                    np.add.at(rad, qrep[~bk], contrib[~bk])
                    np.add.at(rdsum, qrep[~bk], prd[~bk])
                if np.any(bk):
                    np.add.at(brad, qrep[bk], contrib[bk])
                    np.add.at(brdsum, qrep[bk], prd[bk])
            bm = ~leaf_m
            if not np.any(bm):
                break
            bq = qn[bm]
            bn = nd[bm]
            bs = sf[bm]
            split = self.f_split[bn]
            sub = ((qco[bq, 0] >= split[:, 0]).astype(np.int64)
                   + (qco[bq, 1] >= split[:, 1]).astype(np.int64) * 2
                   + (qco[bq, 2] >= split[:, 2]).astype(np.int64) * 4)
            kids = self.f_children[bn]                    # (P, 8)
            valid = kids >= 0
            k = np.where(valid, kids, 0)
            containing = bs[:, None] \
                & (sub[:, None] == np.arange(8)[None, :])
            d = qco[bq][:, None, :] - self.f_aco[k]
            rr = (d * d).sum(axis=2)                      # (P, 8)
            tot_area = self.f_aarea[k] + self.f_abarea[k]
            descend = tot_area > error * rr
            agg_m = valid & ~containing & ~descend
            if np.any(agg_m):
                ai = np.nonzero(agg_m)
                ak = k[ai]
                rd = self._rd_rgb(rr[ai])
                qrep = bq[ai[0]]
                fa = self.f_aarea[ak]
                fm = fa > 0.0
                if np.any(fm):
                    frd = rd[fm] * fa[fm][:, None]
                    np.add.at(rad, qrep[fm],
                              self.f_arad[ak[fm]] * frd)
                    np.add.at(rdsum, qrep[fm], frd)
                ba = self.f_abarea[ak]
                bmk = ba > 0.0
                if np.any(bmk):
                    brd = rd[bmk] * ba[bmk][:, None]
                    np.add.at(brad, qrep[bmk],
                              self.f_abrad[ak[bmk]] * brd)
                    np.add.at(brdsum, qrep[bmk], brd)
            go = valid & (containing | descend)
            gi = np.nonzero(go)
            qn = bq[gi[0]]
            nd = k[gi]
            sf = containing[gi]

        # compute_radiance's tail, verbatim
        fw = _F(self.ss[0].frontweight)
        bw = _F(self.ss[0].backweight)
        rad = rad * fw
        brad = brad * bw
        backrad = rad + brad
        backrdsum = rdsum + brdsum
        out = rad.copy()
        for c in range(3):
            col = _F(self.ss[c].color)
            m = rdsum[:, c] > 1e-16
            out[m, c] = col * rad[m, c] / rdsum[m, c]
            mb = backrdsum[:, c] > 1e-16
            bb = np.where(mb, col * backrad[:, c]
                          / np.where(mb, backrdsum[:, c], 1.0),
                          backrad[:, c])
            out[:, c] = np.maximum(out[:, c], bb)
        return out.astype(_F)

    # ---- GPU packing: the tree as a data texture -------------------
    #
    # The kernels support divergent while-loops but not dynamic array
    # writes (the BVH note in gpu/rtrace.py), so the traversal is
    # STACKLESS: nodes linearised in preorder with hit/miss links --
    # next_hit descends into the subtree, next_miss skips it -- child
    # order 0..7, which is the C's own visit order. The 'self' chain
    # (SUBNODE_INDEX matches along the path, which forces descent past
    # the error criterion) becomes a per-node box: the intersection of
    # every ancestor split's half-spaces, >= on the upper side exactly
    # like SUBNODE_INDEX. Collapsed cells add no split and therefore no
    # half-space, matching the C, where the collapse is invisible to
    # the traversal.
    GPU_TEX_W = 2048

    def pack_gpu(self, view_rows=None):
        """Flatten to a (H, GPU_TEX_W, 4) float32 texture.

        Layout (texel indices):
          0: [n_nodes, n_points, scale, error]
          1: [frontweight, backweight, 0, 0]
          2: [color_r, color_g, color_b, 0]
          3..5: per channel [zr, zv, sigma, 0] (the exact-dipole
                fallthrough past the far table)
          6: [NODES_OFF, PTS_OFF, TAB1_OFF, TAB2_OFF]
          7..9: the world->camera rows [r.xyz, t] -- the shader maps
                its world-space P into the tree's camera space with
                these, so the pass SOURCE stays camera-independent
                (the plan cache deliberately excludes the camera; the
                texture is rebuilt with each pre-pass anyway)
          TAB1: RD_TABLE_SIZE+1 texels, rgb = the three channels' Rd
          TAB2: likewise for the far table
          NODES: 6 texels per node --
            0 [agg_co.xyz, is_leaf]
            1 [rad.rgb, area]
            2 [backrad.rgb, backarea]
            3 [next_hit, next_miss, pstart, pcount]
            4 [selfbox_lo.xyz, 0]
            5 [selfbox_hi.xyz, 0]
          PTS: 2 texels per point --
            0 [co.xyz, signed_area]   (negative area = back point)
            1 [rad.rgb, 0]
        """
        big = np.float32(3.0e38)
        m = len(self.children)
        if self.root < 0 or m == 0:
            m = 0
        # preorder order + links + selfboxes
        order = []
        next_hit = {}
        next_miss = {}
        boxes = {}

        def walk(node, lo, hi, miss):
            order.append(node)
            boxes[node] = (lo.copy(), hi.copy())
            next_miss[node] = miss
            kids = self.children[node]
            if kids is None:
                next_hit[node] = miss
                return
            real = [int(k) for k in kids if k >= 0]
            split = self.split[node]
            # child i's half-spaces: bit set -> [split, hi), else
            # [lo, split) on that axis
            child_boxes = {}
            for i in range(8):
                k = int(kids[i])
                if k < 0:
                    continue
                clo = lo.copy()
                chi = hi.copy()
                for ax, bit in ((0, 1), (1, 2), (2, 4)):
                    if i & bit:
                        clo[ax] = max(clo[ax], split[ax])
                    else:
                        chi[ax] = min(chi[ax], split[ax])
                child_boxes[k] = (clo, chi)
            next_hit[node] = real[0]
            for j, k in enumerate(real):
                child_miss = real[j + 1] if j + 1 < len(real) else miss
                clo, chi = child_boxes[k]
                walk(k, clo, chi, child_miss)

        if m:
            walk(self.root, np.full(3, -big, np.float32),
                 np.full(3, big, np.float32), -1)
        remap = {node: i for i, node in enumerate(order)}
        nn = len(order)
        npts = self.n
        tab_n = RD_TABLE_SIZE + 1
        tab1_off = 10
        tab2_off = tab1_off + tab_n
        nodes_off = tab2_off + tab_n
        pts_off = nodes_off + nn * 6
        total = pts_off + npts * 2
        w = self.GPU_TEX_W
        h = (total + w - 1) // w
        data = np.zeros((h * w, 4), np.float32)
        data[0] = [nn, npts, self.scale, self.error]
        data[1] = [self.ss[0].frontweight, self.ss[0].backweight, 0, 0]
        data[2] = [self.ss[0].color, self.ss[1].color,
                   self.ss[2].color, 0]
        for c in range(3):
            data[3 + c] = [self.ss[c].zr, self.ss[c].zv,
                           self.ss[c].sigma, 0.0]
        data[6] = [nodes_off, pts_off, tab1_off, tab2_off]
        vr = np.asarray(view_rows, np.float32).reshape(3, 4) \
            if view_rows is not None else np.hstack(
                [np.eye(3, dtype=np.float32),
                 np.zeros((3, 1), np.float32)])
        data[7:10] = vr
        data[tab1_off:tab1_off + tab_n, :3] = self._t1
        data[tab2_off:tab2_off + tab_n, :3] = self._t2
        for i, node in enumerate(order):
            base = nodes_off + i * 6
            leaf = self.children[node] is None
            data[base, :3] = self.agg_co[node]
            data[base, 3] = 1.0 if leaf else 0.0
            data[base + 1, :3] = self.agg_rad[node]
            data[base + 1, 3] = self.agg_area[node]
            data[base + 2, :3] = self.agg_brad[node]
            data[base + 2, 3] = self.agg_barea[node]
            nh = next_hit[node]
            nm = next_miss[node]
            data[base + 3] = [remap.get(nh, -1) if nh >= 0 else -1,
                              remap.get(nm, -1) if nm is not None
                              and nm >= 0 else -1,
                              self.pstart[node] if leaf else 0,
                              self.pcount[node] if leaf else 0]
            lo, hi = boxes[node]
            data[base + 4, :3] = lo
            data[base + 5, :3] = hi
        if npts:
            # points in ORDER-array sequence, so leaf [pstart, pcount)
            # ranges are contiguous runs
            perm = self.order
            signed = self.parea[perm] * np.where(self.pback[perm],
                                                 -1.0, 1.0)
            data[pts_off:pts_off + 2 * npts:2, :3] = self.pco[perm]
            data[pts_off:pts_off + 2 * npts:2, 3] = signed
            data[pts_off + 1:pts_off + 1 + 2 * npts:2, :3] = \
                self.prad[perm]
        return data.reshape(h, w, 4)

    def sample_brute(self, qco):
        """Reference: the direct O(N*Q) sum with NO tree approximation
        (every point exact) -- what the traversal converges to as
        error -> 0. Used by the tests."""
        qco = np.asarray(qco, _F) / _F(self.scale)
        q = qco.shape[0]
        rad = np.zeros((q, 3), _F)
        brad = np.zeros((q, 3), _F)
        rdsum = np.zeros((q, 3), _F)
        brdsum = np.zeros((q, 3), _F)
        for p in range(self.n):
            d = qco - self.pco[p][None, :]
            rr = (d * d).sum(axis=1)
            rd = approximate_rd_rgb(self.ss, rr)
            prd = rd * _F(self.parea[p])
            contrib = self.prad[p][None, :] * prd
            if self.pback[p]:
                brad += contrib
                brdsum += prd
            else:
                rad += contrib
                rdsum += prd
        fw = _F(self.ss[0].frontweight)
        bw = _F(self.ss[0].backweight)
        rad = rad * fw
        brad = brad * bw
        backrad = rad + brad
        backrdsum = rdsum + brdsum
        out = rad.copy()
        for c in range(3):
            col = _F(self.ss[c].color)
            m = rdsum[:, c] > 1e-16
            out[m, c] = col * rad[m, c] / rdsum[m, c]
            mb = backrdsum[:, c] > 1e-16
            bb = np.where(mb, col * backrad[:, c]
                          / np.where(mb, backrdsum[:, c], 1.0),
                          backrad[:, c])
            out[:, c] = np.maximum(out[:, c], bb)
        return out.astype(_F)


def _subnode_middle(i, mid, subsize):
    x, y, z = i & 1, i & 2, i & 4
    return np.array([mid[0] + (subsize[0] if x else -subsize[0]),
                     mid[1] + (subsize[1] if y else -subsize[1]),
                     mid[2] + (subsize[2] if z else -subsize[2])], _F)


def settings_for(params):
    """The three per-channel ScatterSettings, from the BI material
    fields, exactly sss_create_tree_mat's calls."""
    col = params.get('color', (1.0, 1.0, 1.0))
    rad = params.get('radius', (1.0, 1.0, 1.0))
    ior = float(params.get('ior', 1.3))
    cfac = float(params.get('colfac', 1.0))
    fw = float(params.get('front', 1.0))
    bw = float(params.get('back', 1.0))
    return [ChannelSettings(float(col[c]), float(rad[c]), ior, cfac,
                            fw, bw) for c in range(3)]


# ---- the pre-pass pixel footprint: shade_sample_sss's area, via
# shade_input_calc_viewco's exact derivatives, in camera space ----

def pixel_areas(co_cam, nor_cam, v1_cam, sx, sy, proj, width, height,
                ortho):
    """Per-sample area = min(|dxco|*|dyco|, 2*orthoarea), verbatim.

    co_cam: (N,3) camera-space sample positions; nor_cam: (N,3) face
    normals (camera space); v1_cam: (N,3) one vertex of each face;
    sx, sy: pixel centres; proj: the projection matrix; ortho: bool.
    The 'ortho area' is the SAME construction with the face normal
    replaced by the normalised view ray (the C swaps facenor and
    recomputes)."""
    n = co_cam.shape[0]
    if n == 0:
        return np.zeros(0, _F)
    p00 = float(proj[0, 0])
    p11 = float(proj[1, 1])
    if ortho:
        fx = 2.0 / (width * p00)
        fy = 2.0 / (height * p11)

        def deriv(nrm):
            nz = nrm[:, 2]
            safe = np.where(np.abs(nz) > 1e-20, nz, 1.0)
            dx = np.zeros((n, 3), _F)
            dx[:, 0] = fx
            dx[:, 2] = np.where(np.abs(nz) > 1e-20,
                                -(nrm[:, 0] * fx) / safe, 0.0)
            dy = np.zeros((n, 3), _F)
            dy[:, 1] = fy
            dy[:, 2] = np.where(np.abs(nz) > 1e-20,
                                -(nrm[:, 1] * fy) / safe, 0.0)
            return dx, dy

        dxa, dya = deriv(nor_cam)
        area = np.linalg.norm(dxa, axis=1) * np.linalg.norm(dya, axis=1)
        view = np.zeros((n, 3), _F)
        view[:, 2] = -1.0
        dxo, dyo = deriv(view)
        orthoarea = np.linalg.norm(dxo, axis=1) \
            * np.linalg.norm(dyo, axis=1)
        return np.minimum(area, 2.0 * orthoarea).astype(_F)

    # perspective: the viewplane ray through each pixel centre and its
    # constant per-pixel increments
    ndc_x = 2.0 * (np.asarray(sx, np.float64) / width) - 1.0
    ndc_y = 2.0 * (np.asarray(sy, np.float64) / height) - 1.0
    view = np.stack([ndc_x / p00, ndc_y / p11,
                     -np.ones(n)], axis=1).astype(np.float64)
    viewdx = np.array([2.0 / (width * p00), 0.0, 0.0])
    viewdy = np.array([0.0, 2.0 / (height * p11), 0.0])

    def deriv(nrm, v1):
        dface = (nrm * v1).sum(axis=1)
        div = (nrm * view).sum(axis=1)
        safe = np.where(np.abs(div) > 1e-20, div, 1.0)
        fac = np.where(np.abs(div) > 1e-20, dface / safe, 0.0)
        co = fac[:, None] * view
        du = div - (nrm * viewdx[None, :]).sum(axis=1)
        dv = div - (nrm * viewdy[None, :]).sum(axis=1)
        du = np.where(np.abs(du) > 1e-20, du, 1.0)
        dv = np.where(np.abs(dv) > 1e-20, dv, 1.0)
        u = dface / du
        v = dface / dv
        dxco = co - (view - viewdx[None, :]) * u[:, None]
        dyco = co - (view - viewdy[None, :]) * v[:, None]
        return (np.linalg.norm(dxco, axis=1)
                * np.linalg.norm(dyco, axis=1))

    area = deriv(nor_cam.astype(np.float64), v1_cam.astype(np.float64))
    vhat = view / np.linalg.norm(view, axis=1, keepdims=True)
    orthoarea = deriv(vhat, v1_cam.astype(np.float64))
    return np.minimum(area, 2.0 * orthoarea).astype(_F)
