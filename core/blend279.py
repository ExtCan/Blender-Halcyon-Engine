"""Read classic .blend files (2.79 and earlier) without Blender.

Modern Blender appends the OBJECTS from an old file fine -- and silently
drops everything that made a Blender Internal material what it was: the
diffuse/specular shader pair, hardness, mirror, the eighteen texture slots
with their mappings and influences. Those fields still sit in the file; the
loader just stopped reading them. So Halcyon reads them itself.

This module is bpy-free. It parses the classic format directly -- 12-byte
header, 4- or 8-byte pointers, either endianness, optional gzip -- walks
the file's own DNA to compute every struct offset (so a 2.49 file with its
24-byte ID names decodes as correctly as a 2.79 one), and returns plain
dicts. The .blend format has carried its own schema since the beginning;
reading it this way is what the format was designed for.

The same discipline as the renderer's own scene parsing: old pointers are
only unique per file SESSION, so blocks are resolved through a pointer map
scoped to the ID block they follow, with a global map as fallback.
"""

import gzip
import struct

import numpy as np

#: Blender Internal texture type codes (DNA_texture_types.h, stable
#: from 2.3x through 2.79)
TEX_TYPES = {1: 'CLOUDS', 2: 'WOOD', 3: 'MARBLE', 4: 'MAGIC', 5: 'BLEND',
             6: 'STUCCI', 7: 'NOISE', 8: 'IMAGE', 9: 'PLUGIN', 10: 'ENVMAP',
             11: 'MUSGRAVE', 12: 'VORONOI', 13: 'DISTNOISE'}

#: MTex.mapto influence bits
MAP_COL, MAP_NORM, MAP_COLSPEC, MAP_COLMIR = 1, 2, 4, 8
MAP_REF, MAP_SPEC, MAP_EMIT, MAP_ALPHA = 16, 32, 64, 128
MAP_HAR, MAP_RAYMIRR, MAP_TRANSLU = 256, 512, 1024
MAP_AMB, MAP_DISPLACE, MAP_WARP = 2048, 4096, 8192

#: MTex.texco coordinate-source bits
# MTex.texco, 2.79 DNA_texture_types.h. The first cut of this table had
# GLOB/OBJECT/ORCO scrambled (UV=16 was right by luck alone): REFL-coord
# env maps -- the field's chrome and matcaps -- routed to Object space
# and sampled garbage, OBJECT slots landed in Generated, ORCO in Global.
TEXCO_ORCO = 1
TEXCO_REFL = 2
TEXCO_NORM = 4
TEXCO_GLOB = 8
TEXCO_UV = 16
TEXCO_OBJECT = 32
TEXCO_WINDOW = 1024
TEXCO_STRESS = 8192
TEXCO_TANGENT = 16384

#: Material.mode bits (the ones the mapping consumes)
MA_SHLESS, MA_WIRE, MA_VERTEXCOLP = 4, 8, 128
MA_ZTRANSP = 0x40
MA_RAYTRANSP = 0x20000
MA_RAYMIRROR = 0x40000
MA_ONLYSHADOW = 0x400
# the BI panel round's bits (2.79 DNA_material_types.h `mode`)
MA_TRACEBLE = 1
MA_SHADOW = 2            # receive shadows
MA_TRANSP = 0x10000      # the Transparency CHECKBOX (method bits above)
MA_VERTEXCOL = 16        # vertex color LIGHT (paint is 128 above)
MA_ONLYCAST = 0x2000
MA_NOMIST = 0x4000
MA_RAMP_COL = 0x100000
MA_RAMP_SPEC = 0x200000
MA_TANGENT_V = 0x4000000
MA_GROUP_NOLAY = 0x10000000   # light-group Exclusive
# shade_flag bits
MA_CUBIC = 1
# mode2 bits
MA_CASTSHADOW = 1

OB_MESH, OB_LAMP_279, OB_CAMERA_279 = 1, 10, 11

#: every classic object type the appender can bring across; the parser
#: lists them all so selection and slot data are known for each
OB_KINDS = {0: 'EMPTY', 1: 'MESH', 2: 'CURVE', 3: 'SURF', 4: 'FONT',
            5: 'META', 10: 'LAMP', 11: 'CAMERA', 12: 'SPEAKER',
            22: 'LATTICE', 25: 'ARMATURE'}

LA_TYPES = {0: 'POINT', 1: 'SUN', 2: 'SPOT', 3: 'HEMI', 4: 'AREA'}


class BlendError(ValueError):
    pass


class Blend279:
    """One parsed classic .blend: blocks, DNA, and typed field access."""

    def __init__(self, path_or_bytes):
        if isinstance(path_or_bytes, (bytes, bytearray)):
            raw = bytes(path_or_bytes)
        else:
            with open(path_or_bytes, 'rb') as fh:
                raw = fh.read()
        if raw[:2] == b'\x1f\x8b':                      # old Compress = gzip
            raw = gzip.decompress(raw)
        if raw[:7] != b'BLENDER':
            raise BlendError('not a .blend file')
        self.raw = raw
        ptr_c, end_c = chr(raw[7]), chr(raw[8])
        if ptr_c not in '_-' or end_c not in 'vV':
            raise BlendError('unrecognised .blend header flags')
        self.psize = 4 if ptr_c == '_' else 8
        self.end = '<' if end_c == 'v' else '>'
        self.version = raw[9:12].decode('ascii', 'replace')
        if self.version.isdigit() and int(self.version) >= 280:
            raise BlendError(
                f'this is a {self.version[0]}.{self.version[1:]} file; the '
                'legacy importer reads 2.79 and earlier (a modern file '
                'appends normally -- it has no Internal materials to save)')
        self._walk_blocks()
        self._parse_dna()
        self._map_pointers()

    # ------------------------------------------------------------- low level

    def _u4(self, off):
        return struct.unpack_from(self.end + 'I', self.raw, off)[0]

    def _ptr_at(self, off):
        fmt = 'I' if self.psize == 4 else 'Q'
        return struct.unpack_from(self.end + fmt, self.raw, off)[0]

    def _walk_blocks(self):
        self.blocks = []                 # (code, old_ptr, data_off, len, sdna, count)
        off = 12
        head = 16 + self.psize
        n = len(self.raw)
        while off + head <= n:
            code = self.raw[off:off + 4].rstrip(b'\x00').decode('ascii',
                                                                'replace')
            blen = self._u4(off + 4)
            optr = self._ptr_at(off + 8)
            sdna = self._u4(off + 8 + self.psize)
            count = self._u4(off + 12 + self.psize)
            off += head
            if code == 'ENDB':
                break
            self.blocks.append((code, optr, off, blen, sdna, count))
            off += blen
        else:
            raise BlendError('no ENDB terminator')

    def _parse_dna(self):
        dna = next((b for b in self.blocks if b[0] == 'DNA1'), None)
        if dna is None:
            raise BlendError('no DNA1 block')
        raw, off = self.raw, dna[2]

        def expect(tag):
            nonlocal off
            while raw[off:off + 4] != tag:      # sections are 4-aligned
                off += 1
            off += 4

        expect(b'SDNA')
        expect(b'NAME')
        n_names = self._u4(off)
        off += 4
        names = []
        for _ in range(n_names):
            e = raw.index(b'\x00', off)
            names.append(raw[off:e].decode('ascii', 'replace'))
            off = e + 1
        expect(b'TYPE')
        n_types = self._u4(off)
        off += 4
        types = []
        for _ in range(n_types):
            e = raw.index(b'\x00', off)
            types.append(raw[off:e].decode('ascii', 'replace'))
            off = e + 1
        expect(b'TLEN')
        tlens = list(struct.unpack_from(f'{self.end}{n_types}H', raw, off))
        off += 2 * n_types
        expect(b'STRC')
        n_str = self._u4(off)
        off += 4
        structs = []
        sidx = {}
        for _ in range(n_str):
            t, nf = struct.unpack_from(self.end + 'HH', raw, off)
            off += 4
            fields = []
            for _f in range(nf):
                ft, fn = struct.unpack_from(self.end + 'HH', raw, off)
                off += 4
                fields.append((ft, fn))
            sidx[types[t]] = len(structs)
            structs.append((t, fields))
        self.names, self.types, self.tlens = names, types, tlens
        self.structs, self.sidx = structs, sidx
        self.tlen_by_name = dict(zip(types, tlens))
        self._off_cache = {}

    def field_size(self, type_i, fname):
        if fname.startswith('(*'):
            return self.psize
        dims = 1
        n = fname
        while '[' in n:
            a, b = n.index('['), n.index(']')
            dims *= int(n[a + 1:b])
            n = n[:a] + n[b + 1:]
        if n.startswith('*'):
            return self.psize * dims
        return self.tlens[type_i] * dims

    def offsets(self, sname):
        """{field: (offset, decl, type_name)} for a struct, from this
        file's OWN DNA -- which is what makes a 2.49 file decode as
        correctly as a 2.79 one."""
        hit = self._off_cache.get(sname)
        if hit is not None:
            return hit
        si = self.sidx.get(sname)
        if si is None:
            self._off_cache[sname] = {}
            return {}
        _t, fields = self.structs[si]
        out = {}
        off = 0
        for (ti, ni) in fields:
            fn = self.names[ni]
            base = fn
            while '[' in base:
                a, b = base.index('['), base.index(']')
                base = base[:a] + base[b + 1:]
            base = base.lstrip('*').replace('(', '').replace(')', '')
            out.setdefault(base, (off, fn, self.types[ti]))
            off += self.field_size(ti, fn)
        self._off_cache[sname] = out
        return out

    def slen(self, sname):
        """sizeof(struct), from the file's DNA (tlens is indexed by TYPE
        index; sidx by struct-list position -- never mix them)."""
        return self.tlens[self.structs[self.sidx[sname]][0]]

    def _map_pointers(self):
        """Old-pointer -> block, scoped: DATA blocks bind to the ID block
        they follow, because old pointers are only unique per allocation
        LIFETIME and files genuinely reuse them across datablocks.

        Three maps, in trust order. `spans` scopes DATA to its owning ID
        block. `id_map` holds ONLY ID blocks: every datablock is alive at
        save time, so ID pointers never collide with each other -- but a
        DATA block written from a reused temporary CAN carry the same old
        pointer as an ID block, and a first-wins global map would then
        shadow the ID behind garbage. Cross-datablock hops (Material ->
        Tex -> Image) resolve through id_map for exactly that reason.
        `global_map` is the last resort."""
        self.spans = {}
        self.id_map = {}
        self.global_map = {}
        cur = None
        for blk in self.blocks:
            code = blk[0]
            if code not in ('DATA', 'DNA1', 'REND', 'TEST', 'GLOB'):
                cur = blk
                self.spans.setdefault((code, blk[1]), {})[blk[1]] = blk
                self.id_map.setdefault(blk[1], blk)
            elif code == 'DATA' and cur is not None:
                self.spans[(cur[0], cur[1])][blk[1]] = blk
            self.global_map.setdefault(blk[1], blk)

    def resolve(self, ptr, span=None, id_block=False):
        """ptr -> block. id_block=True is for pointers that name another
        DATABLOCK (a Tex, an Image): those look in the ID-only map first,
        so a stray DATA block sharing the pointer cannot shadow them."""
        if not ptr:
            return None
        if id_block:
            hit = self.id_map.get(ptr)
            if hit is not None:
                return hit
        if span is not None:
            hit = span.get(ptr)
            if hit is not None:
                return hit
        return self.global_map.get(ptr)

    # ----------------------------------------------------------- field reads

    def f32(self, off, n=1):
        return np.frombuffer(self.raw, self.end + 'f4', n, off).astype(
            np.float32)

    def i16(self, off):
        return struct.unpack_from(self.end + 'h', self.raw, off)[0]

    def i32(self, off):
        return struct.unpack_from(self.end + 'i', self.raw, off)[0]

    def int_at(self, off, tname):
        """An integer read at the width the file's DNA declares -- fields
        genuinely changed width across eras (Base.flag short->int,
        Camera.type char) and the caller shouldn't have to know."""
        size = self.tlen_by_name.get(tname, 4)
        fmt = {1: 'b', 2: 'h', 4: 'i', 8: 'q'}.get(size, 'i')
        if tname.startswith('u'):
            fmt = fmt.upper()
        return struct.unpack_from(self.end + fmt, self.raw, off)[0]

    def array(self, base, stride, count, off, dtype, ncomp):
        """A vectorised strided field read: `count` structs of `stride`
        bytes starting at `base`, taking `ncomp` items of `dtype` at
        struct-relative `off` from each."""
        w = np.dtype(dtype).itemsize * ncomp
        rows = np.frombuffer(self.raw, np.uint8, count * stride,
                             base).reshape(count, stride)
        return rows[:, off:off + w].copy().view(self.end + dtype)

    def read(self, sname, doff, span, spec):
        """Read named fields from a struct instance at `doff`.

        spec: {field: kind} with kind one of 'f', 'f3', 'f4', 'f16', 'i',
        'h', 'ptr', or ('str', n). Missing fields (older files) come back
        as None rather than raising -- the caller decides the default.
        Integer kinds ('i'/'h') read at the width the file's DNA declares,
        not the width the caller guessed.
        """
        offs = self.offsets(sname)
        out = {}
        for field, kind in spec.items():
            ent = offs.get(field)
            if ent is None:
                out[field] = None
                continue
            o = doff + ent[0]
            if kind == 'f':
                out[field] = float(self.f32(o)[0])
            elif kind == 'f3':
                out[field] = tuple(float(x) for x in self.f32(o, 3))
            elif kind == 'f4':
                out[field] = tuple(float(x) for x in self.f32(o, 4))
            elif kind == 'f16':
                out[field] = self.f32(o, 16).reshape(4, 4)
            elif kind in ('i', 'h'):
                out[field] = self.int_at(o, ent[2])
            elif kind == 'ptr':
                out[field] = self._ptr_at(o)
            elif isinstance(kind, tuple) and kind[0] == 'str':
                s = self.raw[o:o + kind[1]]
                out[field] = s.split(b'\x00')[0].decode('utf-8', 'replace')
        return out

    def id_name(self, sname, doff):
        """The datablock name, minus its two-letter code prefix."""
        ido = self.offsets('ID')
        ent = ido.get('name')
        if ent is None:
            return ''
        size = 66 if '[66]' in ent[1] else (24 if '[24]' in ent[1] else 66)
        s = self.raw[doff + ent[0]:doff + ent[0] + size]
        nm = s.split(b'\x00')[0].decode('utf-8', 'replace')
        return nm[2:] if len(nm) > 2 else nm


# =========================================================== scene extraction


def _mesh_geometry(bf, span, doff):
    """verts, normals, polys (list of index tuples), per-poly material ids,
    uv loops, colour loops -- MPoly/MLoop era with MFace fallback."""
    m = bf.read('Mesh', doff, span, {
        'totvert': 'i', 'totpoly': 'i', 'totloop': 'i', 'totface': 'i',
        'mvert': 'ptr', 'mpoly': 'ptr', 'mloop': 'ptr', 'mloopuv': 'ptr',
        'mloopcol': 'ptr', 'mface': 'ptr', 'mtface': 'ptr', 'tface': 'ptr',
        'mat': 'ptr', 'totcol': 'h'})
    nv = int(m['totvert'] or 0)
    if nv <= 0 or not m['mvert']:
        return None
    vb = bf.resolve(m['mvert'], span)
    if vb is None:
        return None
    VO = bf.offsets('MVert')
    vsize = bf.slen('MVert')
    verts = bf.array(vb[2], vsize, nv, VO['co'][0], 'f4', 3)
    if 'no' in VO:
        norms = bf.array(vb[2], vsize, nv, VO['no'][0], 'i2', 3).astype(
            np.float32) / 32767.0
    else:
        norms = np.zeros((nv, 3), np.float32)

    polys, mat_ids = [], []
    uv_loops = None
    col_loops = None

    if m['mpoly'] and m['mloop'] and (m['totpoly'] or 0) > 0:
        npo, nlo = int(m['totpoly']), int(m['totloop'] or 0)
        pb = bf.resolve(m['mpoly'], span)
        lb = bf.resolve(m['mloop'], span)
        if pb is None or lb is None:
            return None
        PO = bf.offsets('MPoly')
        psz = bf.slen('MPoly')
        LO = bf.offsets('MLoop')
        lsz = bf.slen('MLoop')
        loops = bf.array(lb[2], lsz, nlo, LO['v'][0], 'i4', 1)[:, 0]
        starts = bf.array(pb[2], psz, npo, PO['loopstart'][0], 'i4', 1)[:, 0]
        counts = bf.array(pb[2], psz, npo, PO['totloop'][0], 'i4', 1)[:, 0]
        if 'mat_nr' in PO:
            mdt = 'i2' if bf.tlen_by_name.get(PO['mat_nr'][2], 2) == 2 \
                else 'i4'
            mat_ids = [int(x) for x in
                       bf.array(pb[2], psz, npo, PO['mat_nr'][0],
                                mdt, 1)[:, 0]]
        else:
            mat_ids = [0] * npo
        for i in range(npo):
            ls, tl = int(starts[i]), int(counts[i])
            polys.append(tuple(int(v) for v in loops[ls:ls + tl]))
        # EVERY loop-UV layer, by NAME, through CustomData ldata -- one
        # pointer (Mesh.mloopuv) only carries the edit-active layer, and
        # the field's blood splatters live on a layer named 'Blood'.
        # The render-active index sits on the first layer of the type.
        uv_layers = []
        uv_render = None
        LD = bf.offsets('Mesh').get('ldata')
        if LD is not None and 'CustomDataLayer' in bf.sidx and \
                'CustomData' in bf.sidx:
            CO2 = bf.offsets('CustomData')
            cbase = doff + LD[0]
            lay_ptr = bf._ptr_at(cbase + CO2['layers'][0])
            tot = int(bf.int_at(cbase + CO2['totlayer'][0],
                                CO2['totlayer'][2]) or 0)
            lb2 = bf.resolve(lay_ptr, span) if lay_ptr else None
            if lb2 is not None and 0 < tot <= 512:
                CLO = bf.offsets('CustomDataLayer')
                clsz = bf.slen('CustomDataLayer')
                UO = bf.offsets('MLoopUV')
                usz = bf.slen('MLoopUV')
                act_rnd = None
                for i in range(tot):
                    o = lb2[2] + i * clsz
                    typ = int(bf.int_at(o + CLO['type'][0],
                                        CLO['type'][2]) or 0)
                    if typ != 16:                      # CD_MLOOPUV
                        continue
                    if act_rnd is None:
                        act_rnd = int(bf.int_at(
                            o + CLO['active_rnd'][0],
                            CLO['active_rnd'][2]) or 0) \
                            if 'active_rnd' in CLO else 0
                    nm_raw = bf.raw[o + CLO['name'][0]:
                                    o + CLO['name'][0] + 64]
                    nm = nm_raw.split(b'\x00')[0].decode('utf-8',
                                                         'replace')
                    dp = bf._ptr_at(o + CLO['data'][0])
                    db2 = bf.resolve(dp, span) if dp else None
                    if db2 is None:
                        continue
                    uv_layers.append(
                        (nm, bf.array(db2[2], usz, nlo,
                                      UO['uv'][0], 'f4', 2)))
                if uv_layers:
                    idx = act_rnd if act_rnd is not None and \
                        0 <= act_rnd < len(uv_layers) else 0
                    uv_render = uv_layers[idx][0]
                    uv_loops = uv_layers[idx][1]
        if uv_loops is None and m['mloopuv'] and 'MLoopUV' in bf.sidx:
            ub = bf.resolve(m['mloopuv'], span)
            if ub is not None:
                UO = bf.offsets('MLoopUV')
                usz = bf.slen('MLoopUV')
                uv_loops = bf.array(ub[2], usz, nlo, UO['uv'][0], 'f4', 2)
        if m['mloopcol'] and 'MLoopCol' in bf.sidx:
            cb = bf.resolve(m['mloopcol'], span)
            if cb is not None:
                csz = bf.slen('MLoopCol')
                col_loops = np.frombuffer(
                    bf.raw, np.uint8, nlo * csz, cb[2]
                ).reshape(nlo, csz)[:, :4].astype(np.float32) / 255.0
    elif m['mface'] and (m['totface'] or 0) > 0 and 'MFace' in bf.sidx:
        # <= 2.62, and 2.4x: quads and tris in MFace, v4 == 0 marks a tri
        nf = int(m['totface'])
        fb = bf.resolve(m['mface'], span)
        if fb is None:
            return None
        FO = bf.offsets('MFace')
        fsz = bf.slen('MFace')
        vs = np.stack([bf.array(fb[2], fsz, nf, FO[k][0], 'i4', 1)[:, 0]
                       for k in ('v1', 'v2', 'v3', 'v4')], axis=1)
        if 'mat_nr' in FO:
            ms = bf.array(fb[2], fsz, nf, FO['mat_nr'][0], 'i1', 1)[:, 0]
        else:
            ms = np.zeros(nf, np.int8)
        # face UVs: MTFace (2.5x) or TFace (2.4x), uv[4][2] either way
        tb, TO, tsz = None, None, 0
        for fld, stru in (('mtface', 'MTFace'), ('tface', 'TFace')):
            p = m.get(fld)
            if p and stru in bf.sidx and 'uv' in bf.offsets(stru):
                blk = bf.resolve(p, span)
                if blk is not None:
                    tb, TO = blk, bf.offsets(stru)
                    tsz = bf.slen(stru)
                    break
        uvf = bf.array(tb[2], tsz, nf, TO['uv'][0], 'f4', 8).reshape(
            nf, 4, 2) if tb is not None else None
        uv_loops = [] if uvf is not None else None
        for i in range(nf):
            v4 = int(vs[i, 3])
            poly = tuple(int(v) for v in (vs[i, :4] if v4 else vs[i, :3]))
            polys.append(poly)
            mat_ids.append(int(ms[i]))
            if uv_loops is not None:
                uv_loops.extend(tuple(u) for u in uvf[i, :len(poly)])
        if uv_loops is not None:
            uv_loops = np.asarray(uv_loops, np.float32) \
                if uv_loops else None
    else:
        return None

    # material pointer array on the mesh
    mat_ptrs = []
    if m['mat'] and (m['totcol'] or 0) > 0:
        ab = bf.resolve(m['mat'], span)
        if ab is not None:
            for i in range(int(m['totcol'])):
                mat_ptrs.append(bf._ptr_at(ab[2] + i * bf.psize))
    out = {'verts': verts, 'normals': norms, 'polys': polys,
           'mat_ids': mat_ids, 'uv_loops': uv_loops,
           'col_loops': col_loops, 'mat_ptrs': mat_ptrs}
    try:
        if uv_layers:
            out['uv_layers'] = uv_layers
            out['uv_render'] = uv_render
    except NameError:
        pass                       # the MFace fallback path has no ldata
    return out


def _colorband(bf, span, ptr):
    """A Tex's ColorBand block -> (stops, ipotype, hue_ipo, color_mode).

    Stops come back position-sorted as (pos, r, g, b, a); missing era
    fields (ipotype_hue, color_mode) default to the classic behaviour.
    """
    blk = bf.resolve(ptr, span)
    if blk is None or 'ColorBand' not in bf.sidx:
        return None
    doff = blk[2]
    cb = bf.read('ColorBand', doff, span, {
        'tot': 'h', 'ipotype': 'h', 'ipotype_hue': 'h', 'color_mode': 'h'})
    CO = bf.offsets('ColorBand')
    DO = bf.offsets('CBData')
    if 'data' not in CO or 'r' not in DO or 'pos' not in DO:
        return None
    dsz = bf.slen('CBData')
    tot = max(0, min(int(cb['tot'] or 0), 32))
    stops = []
    base = doff + CO['data'][0]
    for i in range(tot):
        o = base + i * dsz
        stops.append((float(bf.f32(o + DO['pos'][0])[0]),
                      float(bf.f32(o + DO['r'][0])[0]),
                      float(bf.f32(o + DO['g'][0])[0]),
                      float(bf.f32(o + DO['b'][0])[0]),
                      float(bf.f32(o + DO['a'][0])[0])))
    stops.sort(key=lambda s: s[0])
    return {'stops': stops, 'ipotype': int(cb['ipotype'] or 0),
            'ipotype_hue': int(cb['ipotype_hue'] or 0),
            'color_mode': int(cb['color_mode'] or 0)}


def _material(bf, span, doff):
    mt = bf.read('Material', doff, span, {
        'r': 'f', 'g': 'f', 'b': 'f',
        'specr': 'f', 'specg': 'f', 'specb': 'f',
        'mirr': 'f', 'mirg': 'f', 'mirb': 'f',
        'har': 'h', 'spec': 'f', 'ref': 'f', 'alpha': 'f', 'emit': 'f',
        'amb': 'f', 'translucency': 'f', 'ray_mirror': 'f',
        'roughness': 'f', 'darkness': 'f', 'ang': 'f',
        'fresnel_mir': 'f', 'fresnel_mir_i': 'f',
        'diff_shader': 'h', 'spec_shader': 'h', 'mode': 'i',
        # the BI panel round: the rest of the material panel
        'refrac': 'f', 'rms': 'f',
        'fresnel_tra': 'f', 'fresnel_tra_i': 'f',
        'spectra': 'f', 'filter': 'f',
        'shade_flag': 'h', 'mode2': 'i',
        'rampin_col': 'h', 'rampblend_col': 'h', 'rampfac_col': 'f',
        'rampin_spec': 'h', 'rampblend_spec': 'h', 'rampfac_spec': 'f',
        'ramp_col': 'ptr', 'ramp_spec': 'ptr', 'group': 'ptr',
        'use_nodes': 'h', 'mtex': 'ptr',
        # R164: the shadow-bias terminator fix (shade_one_light)
        'sbias': 'f',
        # R162: the Subsurface Scattering panel (sss.c reads these)
        'sss_radius': 'f3', 'sss_col': 'f3', 'sss_error': 'f',
        'sss_scale': 'f', 'sss_ior': 'f', 'sss_colfac': 'f',
        'sss_texfac': 'f', 'sss_front': 'f', 'sss_back': 'f',
        'sss_flag': 'h'})
    mt['name'] = bf.id_name('Material', doff)

    # the material's own colorbands (the texture ones follow the Tex)
    for key in ('ramp_col', 'ramp_spec'):
        if mt.get(key):
            mt[f'{key}_band'] = _colorband(bf, span, mt[key])

    # the Light Group: Group -> gobject list -> lamp OBJECT names.
    # Resolved to names HERE, in .blend land, so the engine and the
    # node prop both get plain strings.
    if mt.get('group'):
        gb = bf.resolve(mt['group'], span, id_block=True)
        if gb is not None:
            gspan = bf.spans.get(('GR', gb[1]), span)
            mt['group_name'] = bf.id_name('Group', gb[2])
            GO = bf.offsets('Group')
            lamp_names = []
            ent = GO.get('gobject')
            if ent is not None:
                BO = bf.offsets('GroupObject')
                p = bf._ptr_at(gb[2] + ent[0])     # ListBase.first
                seen = 0
                while p and seen < 4096:
                    blk = bf.resolve(p, gspan)
                    if blk is None:
                        break
                    ob_ptr = bf._ptr_at(blk[2] + BO['ob'][0])
                    ob = bf.resolve(ob_ptr, gspan, id_block=True)
                    if ob is not None:
                        ot = bf.read('Object', ob[2],
                                     bf.spans.get(('OB', ob[1]), gspan),
                                     {'type': 'h'})
                        if int(ot.get('type') or 0) == 10:   # OB_LAMP
                            lamp_names.append(
                                bf.id_name('Object', ob[2]))
                    p = bf._ptr_at(blk[2] + BO['next'][0])
                    seen += 1
            mt['group_lights'] = sorted(lamp_names)
    mt['param'] = tuple(float(x) for x in bf.f32(
        doff + bf.offsets('Material')['param'][0], 4)) \
        if 'param' in bf.offsets('Material') else (0.5, 0.1, 0.5, 0.1)

    # the eighteen texture slots: *mtex[18] is an ARRAY of pointers
    slots = []
    MO = bf.offsets('Material')
    ent = MO.get('mtex')
    if ent is not None and '[' in ent[1]:
        count = int(ent[1][ent[1].index('[') + 1:ent[1].index(']')])
        for i in range(count):
            p = bf._ptr_at(doff + ent[0] + i * bf.psize)
            blk = bf.resolve(p, span)
            if blk is None:
                continue
            slot = bf.read('MTex', blk[2], span, {
                'texco': 'h', 'mapto': 'h', 'maptoneg': 'h',
                'blendtype': 'h', 'ofs': 'f3', 'size': 'f3', 'tex': 'ptr',
                'colfac': 'f', 'norfac': 'f', 'specfac': 'f',
                'alphafac': 'f', 'emitfac': 'f', 'difffac': 'f',
                'raymirrfac': 'f', 'translfac': 'f', 'hardfac': 'f',
                # the value channels' blend TARGET (DVar) and the image
                # projection for 3D coords (flat/cube/tube/sphere)
                'def_var': 'f', 'mapping': 'h', 'ambfac': 'f',
                'colspecfac': 'f', 'mirrfac': 'f', 'dispfac': 'f',
                'varfac': 'f',
                # the slot's own colour: BI's `tcol` whenever an
                # INTENSITY texture drives a colour channel (the texture
                # supplies only the factor; the colour comes from here),
                # and the flags (stencil/negative/RGBtoIntensity)
                'r': 'f', 'g': 'f', 'b': 'f', 'texflag': 'h',
                # the placement axes (x/y/z or 0=constant); virtually
                # always 1,2,3 -- warn on anything else rather than
                # silently sampling a swizzled axis
                'projx': 'h', 'projy': 'h', 'projz': 'h',
                'uvname': ('str', 64)})
            tex = None
            if slot['tex']:
                # the Tex is its own ID block; resolve globally, then
                # read through ITS span for the image pointer
                tb = bf.resolve(slot['tex'], span, id_block=True)
                if tb is not None:
                    tspan = bf.spans.get(('TE', tb[1]), span)
                    tx = bf.read('Tex', tb[2], tspan, {
                        'type': 'h', 'stype': 'h', 'noisesize': 'f',
                        'turbul': 'f', 'noisedepth': 'h', 'noisetype': 'h',
                        'noisebasis': 'h', 'noisebasis2': 'h',
                        'bright': 'f', 'contrast': 'f', 'saturation': 'f',
                        'rfac': 'f', 'gfac': 'f', 'bfac': 'f',
                        'flag': 'h', 'imaflag': 'h',
                        'mg_H': 'f', 'mg_lacunarity': 'f',
                        'mg_octaves': 'f', 'mg_offset': 'f', 'mg_gain': 'f',
                        'ns_outscale': 'f', 'dist_amount': 'f',
                        'vn_w1': 'f', 'vn_w2': 'f', 'vn_w3': 'f',
                        'vn_w4': 'f', 'vn_mexp': 'f', 'vn_distm': 'h',
                        'vn_coltype': 'h', 'ima': 'ptr', 'coba': 'ptr'})
                    if tx['coba']:
                        tx['colorband'] = _colorband(bf, tspan, tx['coba'])
                    tx['name'] = bf.id_name('Tex', tb[2])
                    tx['kind'] = TEX_TYPES.get(int(tx['type'] or 0), 'NONE')
                    if tx['ima']:
                        ib = bf.resolve(tx['ima'], tspan, id_block=True)
                        if ib is not None:
                            im = bf.read('Image', ib[2], span, {
                                'name': ('str', 1024),
                                'packedfile': 'ptr'})
                            tx['image_path'] = im['name']
                            tx['image_name'] = bf.id_name('Image', ib[2])
                            if im.get('packedfile'):
                                # packed textures live INSIDE the .blend:
                                # PackedFile{size, seek, *data} then the
                                # raw file bytes as a DATA block
                                ispan = bf.spans.get(('IM', ib[1]), span)
                                pb = bf.resolve(im['packedfile'], ispan)
                                if pb is not None:
                                    pf = bf.read('PackedFile', pb[2],
                                                 ispan, {'size': 'i',
                                                         'data': 'ptr'})
                                    db = bf.resolve(pf.get('data'), ispan)
                                    size = int(pf.get('size') or 0)
                                    if db is not None and 0 < size \
                                            <= db[3]:
                                        o = db[2]
                                        tx['packed'] = bytes(
                                            bf.raw[o:o + size])
                    tex = tx
            if tex is not None:
                slot['tex'] = tex
                slots.append(slot)
    mt['slots'] = slots
    return mt


def read_legacy_scene(path_or_bytes, geometry=True):
    """Everything the importer needs, as plain data.

    geometry=False skips vertex/loop extraction (the appender-based
    importer lets Blender load geometry; only materials, slots,
    selection, lamps, cameras and the world are read then).

    Returns {'version', 'objects': [...], 'materials': {old_ptr: {...}},
    'world': {...} or None, 'warnings': [...]}. Each object carries its
    name, type, world matrix, selection flag, layer bits, and its payload
    (mesh geometry / lamp / camera fields).
    """
    bf = Blend279(path_or_bytes)
    warnings = []

    # selection: every scene's base list, flag & 1
    selected = set()
    scene_lay = 0
    scene_cm = None
    scene_render = None

    def _cstr(off, n=64):
        b = bf.raw[off:off + n]
        i = b.find(b'\x00')
        return b[:i if i >= 0 else n].decode('utf-8', 'replace')

    for key in [k for k in bf.spans if k[0] == 'SC']:
        span = bf.spans[key]
        doff = span[key[1]][2]
        SCO = bf.offsets('Scene')
        # the scene's visible-layer mask: 2.79 renders only objects on
        # these layers (the field's hidden lamps and floating hands)
        lent = SCO.get('lay')
        if lent is not None:
            scene_lay |= int(bf.int_at(doff + lent[0], lent[2]) or 0)
        # ---- the scene's OWN color management (2.5+ files only): what
        # transform the user's F12 actually applied. 2.79's 'Default'
        # view is the sRGB display encode of a scene-linear render --
        # the difference between a BI frame and the same arithmetic
        # shown raw ("I think the Gamma might be different")
        if scene_cm is None and 'view_settings' in SCO \
                and 'display_settings' in SCO:
            try:
                vso = doff + SCO['view_settings'][0]
                VSO = bf.offsets('ColorManagedViewSettings')
                dso = doff + SCO['display_settings'][0]
                DSO = bf.offsets('ColorManagedDisplaySettings')
                cm = {'display_device':
                      _cstr(dso + DSO['display_device'][0]),
                      'view_transform':
                      _cstr(vso + VSO['view_transform'][0])}
                if 'look' in VSO:
                    cm['look'] = _cstr(vso + VSO['look'][0])
                for fname in ('exposure', 'gamma'):
                    e = VSO.get(fname)
                    if e is not None:
                        cm[fname] = float(struct.unpack_from(
                            bf.end + 'f', bf.raw, vso + e[0])[0])
                scene_cm = cm
            except (KeyError, ValueError, IndexError):
                scene_cm = None
        # ---- the embedded RenderData: the file's own frame size and
        # sky-alpha mode (alphamode 1 = the F12 background is
        # TRANSPARENT -- the field's "white background" was a
        # transparent PNG on a white viewer)
        if scene_render is None and 'r' in SCO:
            try:
                ro = doff + SCO['r'][0]
                RO = bf.offsets('RenderData')

                def _rint(fname):
                    e = RO.get(fname)
                    if e is None:
                        return None
                    return int(bf.int_at(ro + e[0], e[2]) or 0)
                scene_render = {'xsch': _rint('xsch'),
                                'ysch': _rint('ysch'),
                                'size': _rint('size'),
                                'alphamode': _rint('alphamode'),
                                'osa': _rint('osa'),
                                'mode': _rint('mode')}
            except (KeyError, ValueError, IndexError):
                scene_render = None
        ent = SCO.get('base')
        if ent is None:
            continue
        first = bf._ptr_at(doff + ent[0])          # ListBase.first
        BO = bf.offsets('Base')
        p = first
        seen = 0
        while p and seen < 100000:
            blk = bf.resolve(p, span)
            if blk is None:
                break
            flag = bf.int_at(blk[2] + BO['flag'][0], BO['flag'][2])
            obp = bf._ptr_at(blk[2] + BO['object'][0])
            if flag & 1 and obp:
                selected.add(obp)
            p = bf._ptr_at(blk[2] + BO['next'][0])
            seen += 1

    materials = {}
    for key in [k for k in bf.spans if k[0] == 'MA']:
        span = bf.spans[key]
        materials[key[1]] = _material(bf, span, span[key[1]][2])

    meshes = {}
    if geometry:
        for key in [k for k in bf.spans if k[0] == 'ME']:
            span = bf.spans[key]
            geo = _mesh_geometry(bf, span, span[key[1]][2])
            if geo is not None:
                geo['name'] = bf.id_name('Mesh', span[key[1]][2])
                meshes[key[1]] = geo
    else:
        for key in [k for k in bf.spans if k[0] == 'ME']:
            span = bf.spans[key]
            meshes[key[1]] = {'name': bf.id_name('Mesh', span[key[1]][2])}

    # material slot arrays live on the DATA (Mesh/Curve/MetaBall alike):
    # {data old_ptr: [material old_ptrs]}
    data_slots = {}
    for code, sname in (('ME', 'Mesh'), ('CU', 'Curve'), ('MB', 'MetaBall')):
        for key in [k for k in bf.spans if k[0] == code]:
            span = bf.spans[key]
            doff = span[key[1]][2]
            m = bf.read(sname, doff, span, {'mat': 'ptr', 'totcol': 'h'})
            ptrs = []
            if m['mat'] and (m['totcol'] or 0) > 0:
                ab = bf.resolve(m['mat'], span)
                if ab is not None:
                    ptrs = [bf._ptr_at(ab[2] + i * bf.psize)
                            for i in range(int(m['totcol']))]
            data_slots[key[1]] = ptrs

    lamps = {}
    for key in [k for k in bf.spans if k[0] == 'LA']:
        span = bf.spans[key]
        doff = span[key[1]][2]
        la = bf.read('Lamp', doff, span, {
            'type': 'h', 'r': 'f', 'g': 'f', 'b': 'f', 'energy': 'f',
            'dist': 'f', 'spotsize': 'f', 'spotblend': 'f', 'mode': 'i',
            'area_size': 'f', 'area_sizey': 'f', 'clipsta': 'f',
            'falloff_type': 'h', 'att1': 'f', 'att2': 'f',
            # R164: the lamp's SHADOW COLOUR -- lashdw in
            # shade_one_light tints the shadowed part of the light
            'shdwr': 'f', 'shdwg': 'f', 'shdwb': 'f', 'k': 'f'})
        la['name'] = bf.id_name('Lamp', doff)
        la['kind'] = LA_TYPES.get(int(la['type'] or 0), 'POINT')
        lamps[key[1]] = la

    cameras = {}
    for key in [k for k in bf.spans if k[0] == 'CA']:
        span = bf.spans[key]
        doff = span[key[1]][2]
        ca = bf.read('Camera', doff, span, {
            'lens': 'f', 'clipsta': 'f', 'clipend': 'f', 'type': 'h',
            'sensor_x': 'f'})
        ca['name'] = bf.id_name('Camera', doff)
        cameras[key[1]] = ca

    world = None
    for key in [k for k in bf.spans if k[0] == 'WO']:
        span = bf.spans[key]
        doff = span[key[1]][2]
        world = bf.read('World', doff, span, {
            'horr': 'f', 'horg': 'f', 'horb': 'f',
            'zenr': 'f', 'zeng': 'f', 'zenb': 'f',
            'ambr': 'f', 'ambg': 'f', 'ambb': 'f',
            'skytype': 'h', 'misi': 'f', 'miststa': 'f', 'mistdist': 'f',
            'misthi': 'f', 'mistype': 'h',
            # the Gather panel: WO_AMB_OCC/WO_ENV_LIGHT/WO_INDIRECT_LIGHT
            # live in mode; the energies and the env colour choice decide
            # how much neutral fill 2.79 added on top of the lamps
            'mode': 'h', 'aoenergy': 'f', 'ao_env_energy': 'f',
            'aomix': 'h', 'aocolor': 'h', 'aomode': 'h',
            'ao_gather_method': 'h', 'aosamp': 'h', 'aodist': 'f',
            'exp': 'f', 'range': 'f'})
        world['name'] = bf.id_name('World', doff)
        break

    objects = []
    for key in [k for k in bf.spans if k[0] == 'OB']:
        span = bf.spans[key]
        doff = span[key[1]][2]
        ob = bf.read('Object', doff, span, {
            'type': 'h', 'obmat': 'f16', 'data': 'ptr', 'mat': 'ptr',
            'matbits': 'ptr', 'colbits': 'i', 'totcol': 'i', 'lay': 'i',
            # R164: the object colour (MA_OBCOLOR modulates by it) and
            # the auto-smooth angle the RAYBIAS terminator fix reads
            'col': 'f4', 'smoothresh': 'f'})
        ob['name'] = bf.id_name('Object', doff)
        ob['selected'] = key[1] in selected
        ot = int(ob['type'] or 0)
        kind = OB_KINDS.get(ot, f'TYPE{ot}')
        payload = None
        if kind == 'MESH':
            payload = meshes.get(ob['data'])
        elif kind == 'LAMP':
            payload = lamps.get(ob['data'])
        elif kind == 'CAMERA':
            payload = cameras.get(ob['data'])

        # material slots: the data's array, overridden per slot by the
        # object's WHERE the file says the slot is object-linked --
        # matbits (2.5x, a byte per slot) or colbits (2.4x, a bit per
        # slot). Files predating both fall back to non-null-wins.
        slot_ptrs = list(data_slots.get(ob['data'], []))
        totcol = int(ob['totcol'] or 0)
        if ob['mat'] and totcol > 0:
            ab = bf.resolve(ob['mat'], span)
            if ab is not None:
                bits = None
                if ob.get('matbits'):
                    bb = bf.resolve(ob['matbits'], span)
                    if bb is not None:
                        bits = bf.raw[bb[2]:bb[2] + totcol]
                colbits = ob.get('colbits')
                for i in range(totcol):
                    p = bf._ptr_at(ab[2] + i * bf.psize)
                    if bits is not None and i < len(bits):
                        use_ob = bits[i] != 0
                    elif colbits is not None:
                        use_ob = bool(colbits & (1 << min(i, 31)))
                    else:
                        use_ob = bool(p)
                    if i < len(slot_ptrs):
                        if use_ob and p:
                            slot_ptrs[i] = p
                    else:
                        slot_ptrs.append(p if use_ob and p else
                                         (slot_ptrs[i]
                                          if i < len(slot_ptrs) else 0))
        objects.append({'name': ob['name'], 'kind': kind,
                        'type_code': ot,
                        'matrix': ob['obmat'], 'selected': ob['selected'],
                        'layers': int(ob['lay'] or 1),
                        'data': payload, 'mat_ptrs': slot_ptrs,
                        # R164: MA_OBCOLOR's colour and the RAYBIAS
                        # terminator fix's auto-smooth threshold
                        'col': ob.get('col'),
                        'smoothresh': ob.get('smoothresh')})

    return {'version': bf.version, 'pointer_size': bf.psize,
            'endian': bf.end, 'objects': objects, 'materials': materials,
            'world': world, 'warnings': warnings,
            # the union of every scene's visible layers (0 = unknown:
            # treat everything as visible)
            'scene_lay': scene_lay,
            # the scene's own view transform + frame settings (None on
            # pre-2.5 files, which had no OCIO color management)
            'scene_cm': scene_cm, 'scene_render': scene_render}


def diagnose(path, out=print):
    """A plain-text parse summary -- run without Blender:

        python3 -m halcyon.core.blend279 old_file.blend

    This is the first thing to reach for when an import behaves oddly:
    it shows exactly what the reader found, stage by stage.
    """
    sc = read_legacy_scene(path)
    out(f"version 2.{sc['version'][1:]} | {sc['pointer_size']*8}-bit "
        f"pointers | {'little' if sc['endian'] == '<' else 'big'}-endian")
    out(f"objects: {len(sc['objects'])}")
    for ob in sc['objects']:
        sel = '*' if ob['selected'] else ' '
        mats = ', '.join(
            (sc['materials'].get(p) or {}).get('name', '<none>')
            if p else '<empty>' for p in ob['mat_ptrs']) or '-'
        out(f"  {sel} {ob['name']:24s} {ob['kind']:8s} slots: {mats}")
    out(f"materials: {len(sc['materials'])}")
    for ptr, m in sc['materials'].items():
        out(f"  {m['name']:24s} diff={m['diff_shader']} "
            f"spec={m['spec_shader']} har={m['har']} "
            f"rgb=({m['r']:.2f},{m['g']:.2f},{m['b']:.2f}) "
            f"mode=0x{(m['mode'] or 0):x} use_nodes={m.get('use_nodes')} "
            f"slots={len(m['slots'])}")
        for s in m['slots']:
            t = s['tex'] or {}
            out(f"      slot texco={s['texco']} mapto={s['mapto']} "
                f"kind={t.get('kind')} name={t.get('name')!r} "
                f"image={t.get('image_path')!r} "
                f"band={'yes' if t.get('colorband') else 'no'}")
    w = sc['world']
    if w:
        out(f"world: hor=({w['horr']:.2f},{w['horg']:.2f},{w['horb']:.2f}) "
            f"mist={w['misi']}")
    for warn in sc['warnings']:
        out(f"warning: {warn}")
    return sc


if __name__ == '__main__':
    import sys
    diagnose(sys.argv[1])
