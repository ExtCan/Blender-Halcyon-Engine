"""Write synthetic classic .blend files, for testing the legacy importer.

These are REAL .blend files in the classic encoding: 12-byte header, block
headers of code/len/old-pointer/SDNA-index/count, a genuine DNA1 catalogue
(SDNA/NAME/TYPE/TLEN/STRC sections), and struct data laid out at the offsets
that DNA declares. The parser under test computes every offset from the
file's own DNA -- which is exactly the property these fixtures exercise, so
the fixture DNA is self-consistent by construction (authentic 2.79 struct
and field names, realistic ordering) rather than a byte-copy of Blender's.
A parser that hardcodes offsets fails these files; a DNA-driven one cannot
tell the difference.

Two eras are written:

* build_279(): MPoly/MLoop mesh, name[66] IDs, 18-slot materials with the
  full 2.5x-era per-channel factor set, radians spot sizes -- plus a
  DELIBERATE old-pointer collision between two DATA blocks under different
  ID owners, because real files reuse heap addresses and the reader must
  scope its pointer maps.
* build_249(): name[24] IDs, MFace/TFace mesh (quads and tris, v4==0 marks
  a tri), the short 2.4x MTex without per-channel factors, no Material
  param[4], degree spot sizes -- the fields whose absence or width the
  reader has to take from the file, not from assumption.
"""

import struct as _st

_SCALARS = [('char', 1), ('uchar', 1), ('short', 2), ('ushort', 2),
            ('int', 4), ('long', 4), ('ulong', 4), ('float', 4),
            ('double', 8), ('int64_t', 8), ('uint64_t', 8), ('void', 0)]


class DNA:
    """A DNA catalogue being built: types, names, structs, and the same
    offset arithmetic the reader uses (sizes accumulate in field order;
    pointers are psize; arrays multiply)."""

    def __init__(self, psize=4, end='<'):
        self.psize, self.end = psize, end
        self.types, self.tlens, self.tmap = [], [], {}
        self.names, self.nmap = [], {}
        self.structs = []               # (type_idx, [(type_idx, name_idx)])
        self.smap = {}                  # struct name -> index in structs
        self._offs = {}                 # struct name -> {base: (off, decl, t)}
        for t, ln in _SCALARS:
            self._t(t, ln)

    def _t(self, tname, tlen=None):
        i = self.tmap.get(tname)
        if i is None:
            i = self.tmap[tname] = len(self.types)
            self.types.append(tname)
            self.tlens.append(tlen or 0)
        elif tlen is not None:
            self.tlens[i] = tlen
        return i

    def _n(self, decl):
        i = self.nmap.get(decl)
        if i is None:
            i = self.nmap[decl] = len(self.names)
            self.names.append(decl)
        return i

    def _fsize(self, tname, decl):
        if decl.startswith('(*'):
            return self.psize
        dims, s = 1, decl
        while '[' in s:
            a, b = s.index('['), s.index(']')
            dims *= int(s[a + 1:b])
            s = s[:a] + s[b + 1:]
        if s.startswith('*'):
            return self.psize * dims
        return self.tlens[self.tmap[tname]] * dims

    def struct(self, sname, fields):
        ti = self._t(sname)
        fl, offs, off = [], {}, 0
        for tname, decl in fields:
            fti = self._t(tname)
            base = decl
            while '[' in base:
                a, b = base.index('['), base.index(']')
                base = base[:a] + base[b + 1:]
            base = base.lstrip('*').replace('(', '').replace(')', '')
            offs.setdefault(base, (off, decl, tname))
            off += self._fsize(tname, decl)
            fl.append((fti, self._n(decl)))
        self.tlens[ti] = off
        self.smap[sname] = len(self.structs)
        self.structs.append((ti, fl))
        self._offs[sname] = offs

    def tlen(self, sname):
        return self.tlens[self.tmap[sname]]

    def offs(self, sname):
        return self._offs[sname]

    def payload(self):
        e, out = self.end, bytearray()
        out += b'SDNA'
        out += b'NAME' + _st.pack(e + 'i', len(self.names))
        for n in self.names:
            out += n.encode() + b'\x00'
        while len(out) % 4:
            out += b'\x00'
        out += b'TYPE' + _st.pack(e + 'i', len(self.types))
        for t in self.types:
            out += t.encode() + b'\x00'
        while len(out) % 4:
            out += b'\x00'
        out += b'TLEN' + _st.pack(f'{e}{len(self.tlens)}H', *self.tlens)
        while len(out) % 4:
            out += b'\x00'
        out += b'STRC' + _st.pack(e + 'i', len(self.structs))
        for ti, fl in self.structs:
            out += _st.pack(e + 'HH', ti, len(fl))
            for fti, ni in fl:
                out += _st.pack(e + 'HH', fti, ni)
        return bytes(out)


class S:
    """Data for `count` instances of one struct, with typed field setters
    that place values at the DNA-declared offsets."""

    def __init__(self, dna, sname, count=1):
        self.dna, self.sname = dna, sname
        self.size = dna.tlen(sname)
        self.count = count
        self.buf = bytearray(self.size * count)

    def _off(self, field, i):
        return i * self.size + self.dna.offs(self.sname)[field][0]

    def place(self, off, data):
        self.buf[off:off + len(data)] = data

    def f(self, field, vals, i=0):
        if not isinstance(vals, (list, tuple)):
            vals = (vals,)
        flat = []
        for v in vals:
            flat.extend(v if isinstance(v, (list, tuple)) else [v])
        self.place(self._off(field, i),
                   _st.pack(f'{self.dna.end}{len(flat)}f', *flat))

    def n(self, field, val, i=0):
        tname = self.dna.offs(self.sname)[field][2]
        size = self.dna.tlens[self.dna.tmap[tname]]
        fmt = {1: 'b', 2: 'h', 4: 'i', 8: 'q'}[size]
        if tname.startswith('u'):
            fmt = fmt.upper()
        self.place(self._off(field, i), _st.pack(self.dna.end + fmt, val))

    def n3(self, field, vals, i=0):
        tname = self.dna.offs(self.sname)[field][2]
        size = self.dna.tlens[self.dna.tmap[tname]]
        fmt = {1: 'b', 2: 'h', 4: 'i'}[size]
        self.place(self._off(field, i),
                   _st.pack(f'{self.dna.end}{len(vals)}{fmt}', *vals))

    def p(self, field, val, i=0, elem=0):
        fmt = 'I' if self.dna.psize == 4 else 'Q'
        self.place(self._off(field, i) + elem * self.dna.psize,
                   _st.pack(self.dna.end + fmt, val))

    def s(self, field, text, i=0):
        self.place(self._off(field, i), text.encode() + b'\x00')

    def id_name(self, name):
        """Write 'id.name' -- the embedded ID struct sits at offset 0."""
        ido = self.dna.offs('ID')['name']
        self.place(self.dna.offs(self.sname)['id'][0] + ido[0],
                   name.encode() + b'\x00')

    def lb(self, field, first, last, i=0):
        """An embedded ListBase: first and last pointers."""
        fmt = 'I' if self.dna.psize == 4 else 'Q'
        o = self._off(field, i)
        self.place(o, _st.pack(self.dna.end + fmt * 2, first, last))


class BlendFile:
    def __init__(self, dna, version=b'279'):
        self.dna, self.version = dna, version
        self.chunks = []
        self._next = 0x00100000

    def alloc(self):
        p = self._next
        self._next += 0x80
        return p

    def add(self, code, data, sname=None, count=1, old=None):
        if isinstance(data, S):
            count = data.count
            sname = sname or data.sname
            data = bytes(data.buf)
        old = self.alloc() if old is None else old
        e = self.dna.end
        pfmt = 'I' if self.dna.psize == 4 else 'Q'
        sdna = self.dna.smap.get(sname, 0) if sname else 0
        head = code.encode().ljust(4, b'\x00') \
            + _st.pack(e + 'I', len(data)) \
            + _st.pack(e + pfmt, old) \
            + _st.pack(e + 'II', sdna, count)
        self.chunks.append(head + data)
        return old

    def tobytes(self):
        hdr = b'BLENDER' \
            + (b'_' if self.dna.psize == 4 else b'-') \
            + (b'v' if self.dna.end == '<' else b'V') + self.version
        e = self.dna.end
        pfmt = 'I' if self.dna.psize == 4 else 'Q'
        dna = self.dna.payload()
        dna_head = b'DNA1' + _st.pack(e + 'I', len(dna)) \
            + _st.pack(e + pfmt, self.alloc()) + _st.pack(e + 'II', 0, 1)
        endb = b'ENDB' + _st.pack(e + 'I', 0) \
            + _st.pack(e + pfmt, 0) + _st.pack(e + 'II', 0, 0)
        return hdr + b''.join(self.chunks) + dna_head + dna + endb


# ------------------------------------------------------------------ 2.79 DNA


def dna_279(psize=4, end='<'):
    d = DNA(psize, end)
    d.struct('ID', [
        ('void', '*next'), ('void', '*prev'), ('ID', '*newid'),
        ('Library', '*lib'), ('char', 'name[66]'), ('short', 'flag'),
        ('short', 'tag'), ('int', 'us'), ('int', 'icon_id'),
        ('IDProperty', '*properties')])
    d.struct('ListBase', [('void', '*first'), ('void', '*last')])
    d.struct('FileGlobal', [
        ('char', 'subvstr[4]'), ('short', 'subversion'),
        ('short', 'minversion'), ('short', 'minsubversion'),
        ('char', 'pad[2]'), ('bScreen', '*curscreen'),
        ('Scene', '*curscene'), ('int', 'fileflags'), ('int', 'globalf'),
        ('uint64_t', 'build_commit_timestamp'), ('char', 'build_hash[16]'),
        ('char', 'filename[1024]')])
    d.struct('Base', [
        ('Base', '*next'), ('Base', '*prev'), ('int', 'lay'),
        ('int', 'selcol'), ('int', 'flag'), ('short', 'sx'),
        ('short', 'sy'), ('Object', '*object')])
    # the 2.79 scene carries its OCIO settings and RenderData embedded;
    # the parser reads what the user's F12 actually applied (R159)
    d.struct('ColorManagedViewSettings', [
        ('int', 'flag'), ('int', 'pad'), ('char', 'look[64]'),
        ('char', 'view_transform[64]'), ('float', 'exposure'),
        ('float', 'gamma')])
    d.struct('ColorManagedDisplaySettings', [
        ('char', 'display_device[64]')])
    d.struct('RenderData', [
        ('int', 'mode'), ('short', 'osa'), ('short', 'size'),
        ('short', 'alphamode'), ('short', 'pad'),
        ('int', 'xsch'), ('int', 'ysch')])
    d.struct('Scene', [
        ('ID', 'id'), ('AnimData', '*adt'), ('Object', '*camera'),
        ('World', '*world'), ('Scene', '*set'), ('ListBase', 'base'),
        ('Base', '*basact'), ('Object', '*obedit'), ('float', 'cursor[3]'),
        ('int', 'lay'), ('int', 'layact'), ('int', 'pad1'),
        ('RenderData', 'r'),
        ('ColorManagedViewSettings', 'view_settings'),
        ('ColorManagedDisplaySettings', 'display_settings')])
    d.struct('Object', [
        ('ID', 'id'), ('AnimData', '*adt'), ('short', 'type'),
        ('short', 'partype'), ('int', 'par1'), ('int', 'par2'),
        ('int', 'par3'), ('char', 'parsubstr[64]'), ('Object', '*parent'),
        ('Object', '*track'), ('Ipo', '*ipo'), ('BoundBox', '*bb'),
        ('void', '*data'), ('ListBase', 'defbase'),
        ('ListBase', 'modifiers'), ('int', 'mode'),
        ('int', 'restore_mode'), ('Material', '**mat'),
        ('char', '*matbits'), ('int', 'totcol'), ('int', 'actcol'),
        ('float', 'loc[3]'), ('float', 'dloc[3]'), ('float', 'size[3]'),
        ('float', 'rot[3]'), ('float', 'quat[4]'),
        ('float', 'obmat[4][4]'), ('float', 'parentinv[4][4]'),
        ('int', 'lay'), ('short', 'flag'), ('short', 'colbits')])
    d.struct('MVert', [
        ('float', 'co[3]'), ('short', 'no[3]'), ('char', 'flag'),
        ('char', 'bweight')])
    d.struct('MPoly', [
        ('int', 'loopstart'), ('int', 'totloop'), ('short', 'mat_nr'),
        ('char', 'flag'), ('char', 'pad')])
    d.struct('MLoop', [('int', 'v'), ('int', 'e')])
    d.struct('MLoopUV', [('float', 'uv[2]'), ('int', 'flag')])
    d.struct('MLoopCol', [
        ('uchar', 'r'), ('uchar', 'g'), ('uchar', 'b'), ('uchar', 'a')])
    d.struct('MFace', [
        ('int', 'v1'), ('int', 'v2'), ('int', 'v3'), ('int', 'v4'),
        ('char', 'pad'), ('char', 'mat_nr'), ('char', 'edcode'),
        ('char', 'flag')])
    d.struct('MTFace', [
        ('float', 'uv[4][2]'), ('Image', '*tpage'), ('char', 'flag'),
        ('char', 'transp'), ('short', 'mode'), ('short', 'tile'),
        ('short', 'unwrap')])
    d.struct('Mesh', [
        ('ID', 'id'), ('AnimData', '*adt'), ('BoundBox', '*bb'),
        ('Ipo', '*ipo'), ('Key', '*key'), ('Material', '**mat'),
        ('MSelect', '*mselect'), ('MPoly', '*mpoly'),
        ('MTexPoly', '*mtpoly'), ('MLoop', '*mloop'),
        ('MLoopUV', '*mloopuv'), ('MLoopCol', '*mloopcol'),
        ('MFace', '*mface'), ('MTFace', '*mtface'), ('TFace', '*tface'),
        ('MVert', '*mvert'), ('MEdge', '*medge'),
        ('MDeformVert', '*dvert'), ('MCol', '*mcol'),
        ('Mesh', '*texcomesh'), ('int', 'totvert'), ('int', 'totedge'),
        ('int', 'totface'), ('int', 'totselect'), ('int', 'totpoly'),
        ('int', 'totloop'), ('short', 'totcol'), ('short', 'flag'),
        ('float', 'smoothresh')])
    d.struct('MTex', [
        ('short', 'texco'), ('short', 'mapto'), ('short', 'maptoneg'),
        ('short', 'blendtype'), ('Object', '*object'), ('Tex', '*tex'),
        ('char', 'uvname[64]'), ('char', 'projx'), ('char', 'projy'),
        ('char', 'projz'), ('char', 'mapping'), ('float', 'ofs[3]'),
        ('float', 'size[3]'), ('float', 'rot'), ('short', 'texflag'),
        ('short', 'colormodel'), ('short', 'pmapto'),
        ('short', 'pmaptoneg'), ('short', 'normapspace'),
        ('short', 'which_output'), ('float', 'r'), ('float', 'g'),
        ('float', 'b'), ('float', 'k'), ('float', 'def_var'),
        ('float', 'rt'), ('float', 'colfac'), ('float', 'varfac'),
        ('float', 'norfac'), ('float', 'dispfac'), ('float', 'warpfac'),
        ('float', 'colspecfac'), ('float', 'mirrfac'),
        ('float', 'alphafac'), ('float', 'difffac'), ('float', 'specfac'),
        ('float', 'emitfac'), ('float', 'hardfac'),
        ('float', 'raymirrfac'), ('float', 'translfac'),
        ('float', 'ambfac')])
    d.struct('Material', [
        ('ID', 'id'), ('AnimData', '*adt'), ('short', 'material_type'),
        ('short', 'flag'), ('float', 'r'), ('float', 'g'), ('float', 'b'),
        ('float', 'specr'), ('float', 'specg'), ('float', 'specb'),
        ('float', 'mirr'), ('float', 'mirg'), ('float', 'mirb'),
        ('float', 'ambr'), ('float', 'ambb'), ('float', 'ambg'),
        ('float', 'amb'), ('float', 'emit'), ('float', 'ang'),
        ('float', 'spectra'), ('float', 'ray_mirror'), ('float', 'alpha'),
        ('float', 'ref'), ('float', 'spec'), ('float', 'zoffs'),
        ('float', 'add'), ('float', 'translucency'),
        ('float', 'fresnel_mir'), ('float', 'fresnel_mir_i'),
        ('float', 'fresnel_tra'), ('float', 'fresnel_tra_i'),
        ('float', 'filter'), ('short', 'ray_depth'),
        ('short', 'ray_depth_tra'), ('short', 'har'), ('char', 'seed1'),
        ('char', 'seed2'), ('float', 'gloss_mir'), ('float', 'gloss_tra'),
        ('short', 'samp_gloss_mir'), ('short', 'samp_gloss_tra'),
        ('int', 'mode'), ('int', 'mode_l'), ('short', 'flarec'),
        ('short', 'starc'), ('short', 'linec'), ('short', 'ringc'),
        ('short', 'pr_lamp'), ('short', 'pr_texture'),
        ('short', 'diff_shader'), ('short', 'spec_shader'),
        ('float', 'roughness'), ('float', 'refrac'), ('float', 'param[4]'),
        ('float', 'rms'), ('float', 'darkness'), ('short', 'texco'),
        ('short', 'mapto'), ('MTex', '*mtex[18]'),
        ('bNodeTree', '*nodetree'), ('Ipo', '*ipo'), ('Group', '*group')])
    d.struct('Tex', [
        ('ID', 'id'), ('AnimData', '*adt'), ('float', 'noisesize'),
        ('float', 'turbul'), ('float', 'bright'), ('float', 'contrast'),
        ('float', 'saturation'), ('float', 'rfac'), ('float', 'gfac'),
        ('float', 'bfac'), ('float', 'filtersize'),
        ('short', 'noisedepth'), ('short', 'noisetype'),
        ('short', 'noisebasis'), ('short', 'noisebasis2'),
        ('short', 'imaflag'), ('short', 'flag'), ('short', 'type'),
        ('short', 'stype'), ('float', 'cropxmin'), ('float', 'cropymin'),
        ('float', 'cropxmax'), ('float', 'cropymax'), ('Ipo', '*ipo'),
        ('Image', '*ima'), ('ColorBand', '*coba'), ('EnvMap', '*env')])
    d.struct('PackedFile', [
        ('int', 'size'), ('int', 'seek'), ('void', '*data')])
    d.struct('Image', [
        ('ID', 'id'), ('char', 'name[1024]'), ('PackedFile', '*packedfile'),
        ('int', 'ok'), ('int', 'flag'), ('short', 'source'),
        ('short', 'type')])
    d.struct('CBData', [
        ('float', 'r'), ('float', 'g'), ('float', 'b'), ('float', 'a'),
        ('float', 'pos'), ('int', 'cur')])
    d.struct('ColorBand', [
        ('short', 'tot'), ('short', 'cur'), ('char', 'ipotype'),
        ('char', 'ipotype_hue'), ('char', 'color_mode'), ('char', 'pad'),
        ('CBData', 'data[32]')])
    d.struct('Curve', [
        ('ID', 'id'), ('AnimData', '*adt'), ('BoundBox', '*bb'),
        ('Material', '**mat'), ('ListBase', 'nurb'), ('short', 'totcol'),
        ('short', 'flag'), ('float', 'size[3]')])
    d.struct('Lamp', [
        ('ID', 'id'), ('AnimData', '*adt'), ('short', 'type'),
        ('short', 'flag'), ('int', 'mode'), ('short', 'colormodel'),
        ('short', 'totex'), ('float', 'r'), ('float', 'g'), ('float', 'b'),
        ('float', 'k'), ('float', 'shdwr'), ('float', 'shdwg'),
        ('float', 'shdwb'), ('float', 'shdwpad'), ('float', 'energy'),
        ('float', 'dist'), ('float', 'spotsize'), ('float', 'spotblend'),
        ('float', 'haint'), ('float', 'att1'), ('float', 'att2'),
        ('float', 'clipsta'), ('float', 'clipend'), ('float', 'area_size'),
        ('float', 'area_sizey'), ('float', 'area_sizez')])
    d.struct('Camera', [
        ('ID', 'id'), ('AnimData', '*adt'), ('char', 'type'),
        ('char', 'dtx'), ('short', 'flag'), ('float', 'passepartalpha'),
        ('float', 'clipsta'), ('float', 'clipend'), ('float', 'lens'),
        ('float', 'ortho_scale'), ('float', 'drawsize'),
        ('float', 'sensor_x'), ('float', 'sensor_y'), ('float', 'shiftx'),
        ('float', 'shifty')])
    d.struct('World', [
        ('ID', 'id'), ('AnimData', '*adt'), ('short', 'colormodel'),
        ('short', 'totex'), ('short', 'texact'), ('short', 'mistype'),
        ('float', 'horr'), ('float', 'horg'), ('float', 'horb'),
        ('float', 'zenr'), ('float', 'zeng'), ('float', 'zenb'),
        ('float', 'ambr'), ('float', 'ambg'), ('float', 'ambb'),
        ('float', 'exposure'), ('float', 'exp'), ('float', 'range'),
        ('float', 'linfac'), ('float', 'logfac'), ('float', 'gravity'),
        ('float', 'activityBoxRadius'), ('short', 'skytype'),
        ('short', 'mode'), ('float', 'misi'), ('float', 'miststa'),
        ('float', 'mistdist'), ('float', 'misthi')])
    return d


# ------------------------------------------------------- the 2.79 scene file


#: the packed-image payload build_279 embeds (tests assert extraction)
PACKED_PAYLOAD = (b'\x89PNG-fixture-payload-' * 3)[:60]

CUBE_VERTS = [(-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
              (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1)]
CUBE_QUADS = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
              (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
CUBE_MAT_NR = [0, 0, 0, 0, 0, 1]
QUAD_UV = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]


def build_279(psize=4, end='<'):
    """The standard fixture. Returns (file bytes, ground-truth dict)."""
    d = dna_279(psize, end)
    bf = BlendFile(d, b'279')

    # -- forward pointer allocations we need to cross-reference
    p_scene, p_world, p_cam_ob, p_cam = bf.alloc(), bf.alloc(), bf.alloc(), \
        bf.alloc()
    p_cube_ob, p_plane_ob, p_lamp_ob = bf.alloc(), bf.alloc(), bf.alloc()
    p_rig_ob, p_path_ob = bf.alloc(), bf.alloc()
    p_cube_me, p_plane_me, p_lamp = bf.alloc(), bf.alloc(), bf.alloc()
    p_skin, p_chrome = bf.alloc(), bf.alloc()
    p_tex_img, p_tex_cloud, p_image = bf.alloc(), bf.alloc(), bf.alloc()
    p_bases = [bf.alloc() for _ in range(6)]
    # a second deliberate collision: the SECOND Base DATA block shares its
    # old pointer with the SkinTex TE datablock written much later. A
    # first-wins global map would resolve Material->Tex onto Base garbage;
    # the ID-block map must win for datablock hops while the scene's own
    # span still finds its Base.
    p_bases[1] = p_tex_img
    # the deliberate collision: the cube's MVert DATA and Skin's first
    # MTex DATA share one old pointer, in different ID spans
    P_COLLIDE = 0x0F000000

    bf.add('REND', b'\x00' * 72)
    glob = S(d, 'FileGlobal')
    glob.s('subvstr', '279')
    glob.n('subversion', 0)
    glob.p('curscene', p_scene)
    glob.s('filename', '/tmp/fixture279.blend')
    bf.add('GLOB', glob)

    # ----- Scene + its Base list (DATA blocks in the scene's span)
    sc = S(d, 'Scene')
    sc.id_name('SCScene')
    sc.p('camera', p_cam_ob)
    sc.p('world', p_world)
    sc.lb('base', p_bases[0], p_bases[-1])
    sc.n('lay', 1)
    # the scene's own pipeline, as a real 2.79 default file carries it:
    # 'Default' view on an sRGB display = linear render, sRGB encode
    # on display. exposure 0.25 STOPS exercises the 2^e mapping.
    SCO = d.offs('Scene')
    VSO = d.offs('ColorManagedViewSettings')
    DSO = d.offs('ColorManagedDisplaySettings')
    vso = SCO['view_settings'][0]
    sc.place(vso + VSO['look'][0], b'None\x00')
    sc.place(vso + VSO['view_transform'][0], b'Default\x00')
    sc.place(vso + VSO['exposure'][0], _st.pack(d.end + 'f', 0.25))
    sc.place(vso + VSO['gamma'][0], _st.pack(d.end + 'f', 1.0))
    sc.place(SCO['display_settings'][0] + DSO['display_device'][0],
             b'sRGB\x00')
    ro = SCO['r'][0]
    RO = d.offs('RenderData')
    sc.place(ro + RO['xsch'][0], _st.pack(d.end + 'i', 1920))
    sc.place(ro + RO['ysch'][0], _st.pack(d.end + 'i', 1920))
    sc.place(ro + RO['size'][0], _st.pack(d.end + 'h', 50))
    sc.place(ro + RO['alphamode'][0], _st.pack(d.end + 'h', 1))
    sc.place(ro + RO['osa'][0], _st.pack(d.end + 'h', 16))
    bf.add('SC', sc, old=p_scene)
    base_obs = [(p_cube_ob, 1), (p_plane_ob, 0), (p_lamp_ob, 1),
                (p_cam_ob, 1), (p_rig_ob, 1), (p_path_ob, 0)]
    for i, (obp, flag) in enumerate(base_obs):
        ba = S(d, 'Base')
        ba.p('next', p_bases[i + 1] if i + 1 < len(base_obs) else 0)
        ba.p('prev', p_bases[i - 1] if i else 0)
        ba.n('lay', 1)
        ba.n('flag', flag)                    # SELECT = 1
        ba.p('object', obp)
        bf.add('DATA', ba, old=p_bases[i])

    # ----- objects
    def matrix(tx, ty, tz, s=1.0):
        # C storage order: obmat[3][0..2] is the translation row
        return [[s, 0, 0, 0], [0, s, 0, 0], [0, 0, s, 0], [tx, ty, tz, 1]]

    cube = S(d, 'Object')
    cube.id_name('OBCube')
    cube.n('type', 1)                         # OB_MESH
    cube.p('data', p_cube_me)
    p_cube_slots = bf.alloc()
    cube.p('mat', p_cube_slots)
    cube.n('totcol', 2)
    p_cube_bits = bf.alloc()
    cube.p('matbits', p_cube_bits)
    cube.f('obmat', matrix(1.0, 2.0, 3.0))
    cube.n('lay', 1)
    bf.add('OB', cube, old=p_cube_ob)
    # object-level slots hold Chrome in BOTH, but matbits says only
    # slot 1 is object-linked -- slot 0 must stay the mesh's Skin
    slots_data = _st.pack(
        (d.end + ('I' if psize == 4 else 'Q') * 2), p_chrome, p_chrome)
    bf.add('DATA', slots_data, old=p_cube_slots)
    bf.add('DATA', bytes([0, 1]), old=p_cube_bits)

    plane = S(d, 'Object')
    plane.id_name('OBPlane')
    plane.n('type', 1)
    plane.p('data', p_plane_me)
    plane.f('obmat', matrix(0.0, 0.0, -1.0))
    plane.n('lay', 1)
    bf.add('OB', plane, old=p_plane_ob)

    lamp_ob = S(d, 'Object')
    lamp_ob.id_name('OBSpot')
    lamp_ob.n('type', 10)                     # OB_LAMP
    lamp_ob.p('data', p_lamp)
    lamp_ob.f('obmat', matrix(4.0, 1.0, 6.0))
    lamp_ob.n('lay', 1)
    bf.add('OB', lamp_ob, old=p_lamp_ob)

    rig = S(d, 'Object')
    rig.id_name('OBRig')
    rig.n('type', 0)                          # OB_EMPTY
    rig.f('obmat', matrix(0.0, 5.0, 0.0))
    rig.n('lay', 1)
    bf.add('OB', rig, old=p_rig_ob)

    p_curve = bf.alloc()
    path = S(d, 'Object')
    path.id_name('OBPath')
    path.n('type', 2)                         # OB_CURVE
    path.p('data', p_curve)
    path.f('obmat', matrix(-3.0, 0.0, 0.0))
    path.n('lay', 1)
    bf.add('OB', path, old=p_path_ob)
    cu = S(d, 'Curve')
    cu.id_name('CUPath')
    p_cu_slots = bf.alloc()
    cu.p('mat', p_cu_slots)
    cu.n('totcol', 1)
    bf.add('CU', cu, old=p_curve)
    bf.add('DATA', _st.pack(d.end + ('I' if psize == 4 else 'Q'),
                            p_chrome), old=p_cu_slots)

    cam_ob = S(d, 'Object')
    cam_ob.id_name('OBCamera')
    cam_ob.n('type', 11)                      # OB_CAMERA
    cam_ob.p('data', p_cam)
    cam_ob.f('obmat', matrix(7.0, -7.0, 5.0))
    cam_ob.n('lay', 1)
    bf.add('OB', cam_ob, old=p_cam_ob)

    # ----- the cube mesh (MPoly/MLoop era) + its arrays
    nv, npo = len(CUBE_VERTS), len(CUBE_QUADS)
    nlo = 4 * npo
    me = S(d, 'Mesh')
    me.id_name('MECubeMesh')
    p_mpoly, p_mloop, p_mluv, p_mlcol = (bf.alloc(), bf.alloc(), bf.alloc(),
                                         bf.alloc())
    p_me_slots = bf.alloc()
    me.p('mvert', P_COLLIDE)
    me.p('mpoly', p_mpoly)
    me.p('mloop', p_mloop)
    me.p('mloopuv', p_mluv)
    me.p('mloopcol', p_mlcol)
    me.p('mat', p_me_slots)
    me.n('totvert', nv)
    me.n('totpoly', npo)
    me.n('totloop', nlo)
    me.n('totcol', 2)
    bf.add('ME', me, old=p_cube_me)

    mv = S(d, 'MVert', nv)
    for i, co in enumerate(CUBE_VERTS):
        mv.f('co', [float(c) for c in co], i)
        mv.n3('no', [18918 if c > 0 else -18918 for c in co], i)
    bf.add('DATA', mv, old=P_COLLIDE)
    mp = S(d, 'MPoly', npo)
    ml = S(d, 'MLoop', nlo)
    mu = S(d, 'MLoopUV', nlo)
    mc = S(d, 'MLoopCol', nlo)
    li = 0
    for i, quad in enumerate(CUBE_QUADS):
        mp.n('loopstart', li, i)
        mp.n('totloop', 4, i)
        mp.n('mat_nr', CUBE_MAT_NR[i], i)
        for k, v in enumerate(quad):
            ml.n('v', v, li)
            mu.f('uv', QUAD_UV[k], li)
            mc.n('r', min(255, v * 30), li)
            mc.n('g', 60, li)
            mc.n('b', 200, li)
            mc.n('a', 255, li)
            li += 1
    bf.add('DATA', mp, old=p_mpoly)
    bf.add('DATA', ml, old=p_mloop)
    bf.add('DATA', mu, old=p_mluv)
    bf.add('DATA', mc, old=p_mlcol)
    me_slots = _st.pack((d.end + ('I' if psize == 4 else 'Q') * 2),
                        p_skin, p_skin)
    bf.add('DATA', me_slots, sname='Link', old=p_me_slots)

    # ----- the plane mesh: 4 verts, one quad, no UVs, no material
    pm = S(d, 'Mesh')
    pm.id_name('MEPlaneMesh')
    p_pv, p_pp, p_pl = bf.alloc(), bf.alloc(), bf.alloc()
    pm.p('mvert', p_pv)
    pm.p('mpoly', p_pp)
    pm.p('mloop', p_pl)
    pm.n('totvert', 4)
    pm.n('totpoly', 1)
    pm.n('totloop', 4)
    bf.add('ME', pm, old=p_plane_me)
    pv = S(d, 'MVert', 4)
    for i, co in enumerate([(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)]):
        pv.f('co', [float(c) for c in co], i)
        pv.n3('no', [0, 0, 32767], i)
    bf.add('DATA', pv, old=p_pv)
    pp = S(d, 'MPoly', 1)
    pp.n('loopstart', 0)
    pp.n('totloop', 4)
    bf.add('DATA', pp, old=p_pp)
    pl = S(d, 'MLoop', 4)
    for i, v in enumerate((0, 1, 2, 3)):
        pl.n('v', v, i)
    bf.add('DATA', pl, old=p_pl)

    # ----- materials
    skin = S(d, 'Material')
    skin.id_name('MASkin')
    skin.f('r', 0.8)
    skin.f('g', 0.55)
    skin.f('b', 0.45)
    skin.f('specr', 1.0)
    skin.f('specg', 0.95)
    skin.f('specb', 0.9)
    skin.f('mirr', 1.0)
    skin.f('mirg', 1.0)
    skin.f('mirb', 1.0)
    skin.f('amb', 1.0)
    skin.f('emit', 0.0)
    skin.f('ang', 1.0)
    skin.f('ray_mirror', 0.0)
    skin.f('alpha', 1.0)
    skin.f('ref', 0.8)
    skin.f('spec', 0.4)
    skin.f('translucency', 0.1)
    skin.f('roughness', 0.9)
    skin.n('har', 50)
    skin.n('mode', 0)
    skin.n('diff_shader', 1)                  # OREN-NAYAR
    skin.n('spec_shader', 2)                  # BLINN
    skin.f('param', (0.5, 0.1, 0.5, 0.1))
    p_mtex0, p_mtex5 = P_COLLIDE, bf.alloc()  # slot 0 COLLIDES with mvert
    skin.p('mtex', p_mtex0, elem=0)
    skin.p('mtex', p_mtex5, elem=5)           # sparse: slots 1-4 empty
    bf.add('MA', skin, old=p_skin)
    t0 = S(d, 'MTex')
    t0.n('texco', 16)                         # TEXCO_UV
    t0.n('mapto', 1)                          # MAP_COL
    t0.n('blendtype', 0)                      # MIX
    t0.p('tex', p_tex_img)
    t0.s('uvname', 'UVMap')
    t0.f('ofs', (0.0, 0.0, 0.0))
    t0.f('size', (1.0, 1.0, 1.0))
    t0.f('colfac', 0.85)
    t0.f('norfac', 0.5)
    bf.add('DATA', t0, old=p_mtex0)
    t5 = S(d, 'MTex')
    t5.n('texco', 1)                          # TEXCO_ORCO (DNA value)
    t5.n('mapto', 2)                          # MAP_NORM
    t5.n('blendtype', 0)
    t5.p('tex', p_tex_cloud)
    t5.f('ofs', (0.0, 0.25, 0.0))
    t5.f('size', (2.0, 2.0, 2.0))
    t5.f('norfac', 0.6)
    bf.add('DATA', t5, old=p_mtex5)

    chrome = S(d, 'Material')
    chrome.id_name('MAChrome')
    chrome.f('r', 0.6)
    chrome.f('g', 0.65)
    chrome.f('b', 0.7)
    chrome.f('specr', 1.0)
    chrome.f('specg', 1.0)
    chrome.f('specb', 1.0)
    chrome.f('mirr', 0.9)
    chrome.f('mirg', 0.95)
    chrome.f('mirb', 1.0)
    chrome.f('amb', 1.0)
    chrome.f('ang', 1.45)
    chrome.f('ray_mirror', 0.75)
    chrome.f('alpha', 1.0)
    chrome.f('ref', 0.3)
    chrome.f('spec', 1.5)
    chrome.n('har', 180)
    chrome.n('mode', 0x40000)                 # MA_RAYMIRROR
    chrome.n('diff_shader', 2)                # TOON
    chrome.n('spec_shader', 0)                # COOKTORR
    chrome.f('param', (0.6, 0.05, 0.4, 0.1))
    bf.add('MA', chrome, old=p_chrome)

    # ----- textures + image
    ti = S(d, 'Tex')
    ti.id_name('TESkinTex')
    ti.n('type', 8)                           # TEX_IMAGE
    # the 2.79 default_tex imaflag: INTERPOL|USEALPHA|MIPMAP -- real
    # files carry it, and the alpha law reads it (R158)
    ti.n('imaflag', 7)
    ti.p('ima', p_image)
    ti.f('noisesize', 0.25)
    bf.add('TE', ti, old=p_tex_img)
    tc = S(d, 'Tex')
    tc.id_name('TEBumps')
    tc.n('type', 1)                           # TEX_CLOUDS
    tc.f('noisesize', 0.35)
    tc.f('turbul', 6.0)
    tc.n('noisedepth', 3)
    tc.n('noisebasis', 0)
    tc.n('flag', 1)                           # TEX_COLORBAND
    tc.f('bright', 1.1)
    tc.f('contrast', 0.9)
    tc.f('saturation', 1.0)
    tc.f('rfac', 1.0)
    tc.f('gfac', 1.0)
    tc.f('bfac', 1.0)
    p_coba = bf.alloc()
    tc.p('coba', p_coba)
    bf.add('TE', tc, old=p_tex_cloud)
    cb = S(d, 'ColorBand')
    cb.n('tot', 3)
    cb.n('ipotype', 1)                        # EASE
    CB = d.offs('ColorBand')['data'][0]
    dsz = d.tlen('CBData')
    for i, (pos, r, g, b, a) in enumerate([(0.0, 0.1, 0.0, 0.3, 1.0),
                                           (0.5, 0.9, 0.6, 0.1, 0.8),
                                           (1.0, 1.0, 1.0, 0.9, 1.0)]):
        base = CB + i * dsz
        cb.place(base, _st.pack(d.end + '5f', r, g, b, a, pos))
    bf.add('DATA', cb, old=p_coba)
    im = S(d, 'Image')
    im.id_name('IMtex.png')
    im.s('name', '//textures/tex.png')
    p_pf, p_pfdata = bf.alloc(), bf.alloc()
    im.p('packedfile', p_pf)
    bf.add('IM', im, old=p_image)
    PACKED = PACKED_PAYLOAD
    pf = S(d, 'PackedFile')
    pf.n('size', len(PACKED))
    pf.p('data', p_pfdata)
    bf.add('DATA', pf, old=p_pf)
    bf.add('DATA', PACKED, old=p_pfdata)

    # ----- lamp, camera, world
    la = S(d, 'Lamp')
    la.id_name('LASpot')
    la.n('type', 2)                           # SPOT
    la.f('r', 1.0)
    la.f('g', 0.9)
    la.f('b', 0.8)
    la.f('energy', 1.5)
    la.f('dist', 25.0)
    la.f('spotsize', 0.7853982)               # radians in 2.70+
    la.f('spotblend', 0.15)
    la.n('mode', 1)
    bf.add('LA', la, old=p_lamp)
    ca = S(d, 'Camera')
    ca.id_name('CACamera')
    ca.n('type', 0)                           # PERSP (char in 2.79)
    ca.f('lens', 35.0)
    ca.f('clipsta', 0.1)
    ca.f('clipend', 100.0)
    ca.f('sensor_x', 32.0)
    bf.add('CA', ca, old=p_cam)
    wo = S(d, 'World')
    wo.id_name('WOWorld')
    wo.f('horr', 0.05)
    wo.f('horg', 0.07)
    wo.f('horb', 0.12)
    wo.f('zenr', 0.4)
    wo.f('zeng', 0.5)
    wo.f('zenb', 0.8)
    wo.f('ambr', 0.1)
    wo.f('ambg', 0.1)
    wo.f('ambb', 0.1)
    wo.n('skytype', 3)                        # BLEND | REAL
    wo.f('misi', 0.2)
    wo.f('miststa', 5.0)
    wo.f('mistdist', 40.0)
    bf.add('WO', wo, old=p_world)

    truth = {
        'materials': {'skin': p_skin, 'chrome': p_chrome},
        'collide_ptr': P_COLLIDE,
        'cube_slots': [p_skin, p_chrome],     # matbits: only slot 1 is
        'curve_slots': [p_chrome],            # object-linked
    }
    return bf.tobytes(), truth


# ------------------------------------------------ the 2.4x-style era variant


def dna_249(psize=4, end='<'):
    d = DNA(psize, end)
    d.struct('ID', [
        ('void', '*next'), ('void', '*prev'), ('ID', '*newid'),
        ('Library', '*lib'), ('char', 'name[24]'), ('short', 'us'),
        ('short', 'flag')])
    d.struct('ListBase', [('void', '*first'), ('void', '*last')])
    d.struct('Base', [
        ('Base', '*next'), ('Base', '*prev'), ('int', 'lay'),
        ('int', 'selcol'), ('short', 'flag'), ('short', 'sx'),
        ('short', 'sy'), ('short', 'pad'), ('Object', '*object')])
    d.struct('Scene', [
        ('ID', 'id'), ('Object', '*camera'), ('World', '*world'),
        ('Scene', '*set'), ('Image', '*ima'), ('ListBase', 'base'),
        ('Base', '*basact'), ('float', 'cursor[3]'), ('int', 'lay')])
    d.struct('Object', [
        ('ID', 'id'), ('short', 'type'), ('short', 'partype'),
        ('int', 'par1'), ('int', 'par2'), ('int', 'par3'),
        ('char', 'parsubstr[32]'), ('Object', '*parent'),
        ('Object', '*track'), ('Ipo', '*ipo'), ('void', '*data'),
        ('Material', '**mat'), ('int', 'totcol'), ('int', 'actcol'),
        ('int', 'colbits'),
        ('float', 'loc[3]'), ('float', 'dloc[3]'), ('float', 'size[3]'),
        ('float', 'rot[3]'), ('float', 'quat[4]'),
        ('float', 'obmat[4][4]'), ('int', 'lay')])
    d.struct('MVert', [
        ('float', 'co[3]'), ('short', 'no[3]'), ('char', 'flag'),
        ('char', 'mat_nr')])
    d.struct('MFace', [
        ('int', 'v1'), ('int', 'v2'), ('int', 'v3'), ('int', 'v4'),
        ('char', 'pad'), ('char', 'mat_nr'), ('char', 'edcode'),
        ('char', 'flag')])
    d.struct('TFace', [
        ('void', '*tpage'), ('float', 'uv[4][2]'), ('int', 'col[4]'),
        ('char', 'flag'), ('char', 'transp'), ('short', 'mode'),
        ('short', 'tile'), ('short', 'unwrap')])
    d.struct('Mesh', [
        ('ID', 'id'), ('BoundBox', '*bb'), ('Ipo', '*ipo'), ('Key', '*key'),
        ('Material', '**mat'), ('MFace', '*mface'), ('TFace', '*tface'),
        ('MVert', '*mvert'), ('MEdge', '*medge'), ('MCol', '*mcol'),
        ('int', 'totvert'), ('int', 'totedge'), ('int', 'totface'),
        ('short', 'totcol'), ('short', 'flag'), ('float', 'smoothresh')])
    d.struct('MTex', [
        ('short', 'texco'), ('short', 'mapto'), ('short', 'maptoneg'),
        ('short', 'blendtype'), ('Object', '*object'), ('Tex', '*tex'),
        ('char', 'uvname[24]'), ('char', 'projx'), ('char', 'projy'),
        ('char', 'projz'), ('char', 'mapping'), ('float', 'ofs[3]'),
        ('float', 'size[3]'), ('short', 'texflag'), ('short', 'colormodel'),
        ('short', 'pmapto'), ('short', 'pmaptoneg'), ('float', 'r'),
        ('float', 'g'), ('float', 'b'), ('float', 'k'),
        ('float', 'def_var'), ('float', 'colfac'), ('float', 'norfac'),
        ('float', 'varfac')])
    d.struct('Material', [
        ('ID', 'id'), ('short', 'colormodel'), ('short', 'flag'),
        ('float', 'r'), ('float', 'g'), ('float', 'b'), ('float', 'specr'),
        ('float', 'specg'), ('float', 'specb'), ('float', 'mirr'),
        ('float', 'mirg'), ('float', 'mirb'), ('float', 'ambr'),
        ('float', 'ambb'), ('float', 'ambg'), ('float', 'amb'),
        ('float', 'emit'), ('float', 'ang'), ('float', 'spectra'),
        ('float', 'ray_mirror'), ('float', 'alpha'), ('float', 'ref'),
        ('float', 'spec'), ('float', 'zoffs'), ('float', 'add'),
        ('float', 'translucency'), ('float', 'fresnel_mir'),
        ('float', 'fresnel_mir_i'), ('short', 'har'), ('char', 'seed1'),
        ('char', 'seed2'), ('int', 'mode'), ('short', 'flarec'),
        ('short', 'starc'), ('short', 'diff_shader'),
        ('short', 'spec_shader'), ('float', 'roughness'),
        ('float', 'refrac'), ('float', 'rms'), ('float', 'darkness'),
        ('short', 'texco'), ('short', 'mapto'), ('MTex', '*mtex[10]'),
        ('Ipo', '*ipo')])
    d.struct('Tex', [
        ('ID', 'id'), ('float', 'noisesize'), ('float', 'turbul'),
        ('float', 'bright'), ('float', 'contrast'), ('float', 'rfac'),
        ('float', 'gfac'), ('float', 'bfac'), ('float', 'filtersize'),
        ('short', 'noisedepth'), ('short', 'noisetype'), ('short', 'type'),
        ('short', 'stype'), ('Ipo', '*ipo'), ('Image', '*ima')])
    d.struct('Image', [
        ('ID', 'id'), ('char', 'name[160]'), ('int', 'ok'), ('int', 'flag')])
    d.struct('Lamp', [
        ('ID', 'id'), ('short', 'type'), ('short', 'flag'), ('int', 'mode'),
        ('short', 'colormodel'), ('short', 'totex'), ('float', 'r'),
        ('float', 'g'), ('float', 'b'), ('float', 'k'), ('float', 'energy'),
        ('float', 'dist'), ('float', 'spotsize'), ('float', 'spotblend'),
        ('float', 'haint'), ('float', 'att1'), ('float', 'att2'),
        ('float', 'clipsta'), ('float', 'clipend')])
    d.struct('Camera', [
        ('ID', 'id'), ('short', 'type'), ('short', 'flag'),
        ('float', 'passepartalpha'), ('float', 'clipsta'),
        ('float', 'clipend'), ('float', 'lens'), ('float', 'ortho_scale'),
        ('float', 'drawsize')])
    d.struct('World', [
        ('ID', 'id'), ('short', 'colormodel'), ('short', 'totex'),
        ('short', 'texact'), ('short', 'mistype'), ('float', 'horr'),
        ('float', 'horg'), ('float', 'horb'), ('float', 'zenr'),
        ('float', 'zeng'), ('float', 'zenb'), ('float', 'ambr'),
        ('float', 'ambg'), ('float', 'ambb'), ('short', 'skytype'),
        ('short', 'mode'), ('float', 'misi'), ('float', 'miststa'),
        ('float', 'mistdist')])
    return d


def build_249(psize=4, end='<'):
    """A 2.4x-era file: name[24] IDs, MFace/TFace mesh with a quad AND a
    tri, degree spot size, no per-channel MTex factors, no param[4]."""
    d = dna_249(psize, end)
    bf = BlendFile(d, b'249')

    p_scene, p_world = bf.alloc(), bf.alloc()
    p_ob, p_me, p_ma = bf.alloc(), bf.alloc(), bf.alloc()
    p_lamp_ob, p_lamp = bf.alloc(), bf.alloc()
    p_tex, p_img = bf.alloc(), bf.alloc()
    p_base = [bf.alloc(), bf.alloc()]

    sc = S(d, 'Scene')
    sc.id_name('SCScene')
    sc.p('world', p_world)
    sc.lb('base', p_base[0], p_base[-1])
    bf.add('SC', sc, old=p_scene)
    for i, (obp, flag) in enumerate([(p_ob, 1), (p_lamp_ob, 1)]):
        ba = S(d, 'Base')
        ba.p('next', p_base[i + 1] if i == 0 else 0)
        ba.n('lay', 1)
        ba.n('flag', flag)
        ba.p('object', obp)
        bf.add('DATA', ba, old=p_base[i])

    ob = S(d, 'Object')
    ob.id_name('OBOld')
    ob.n('type', 1)
    ob.p('data', p_me)
    ob.f('obmat', [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
                   [-2.0, 0.0, 1.0, 1]])
    ob.n('lay', 1)
    bf.add('OB', ob, old=p_ob)

    lob = S(d, 'Object')
    lob.id_name('OBLamp')
    lob.n('type', 10)
    lob.p('data', p_lamp)
    lob.f('obmat', [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
                    [0.0, 0.0, 8.0, 1]])
    lob.n('lay', 1)
    bf.add('OB', lob, old=p_lamp_ob)

    # a quad and a tri: 5 verts
    me = S(d, 'Mesh')
    me.id_name('MEOldMesh')
    p_mv, p_mf, p_tf, p_slots = bf.alloc(), bf.alloc(), bf.alloc(), \
        bf.alloc()
    me.p('mvert', p_mv)
    me.p('mface', p_mf)
    me.p('tface', p_tf)
    me.p('mat', p_slots)
    me.n('totvert', 5)
    me.n('totface', 2)
    me.n('totcol', 1)
    bf.add('ME', me, old=p_me)
    mv = S(d, 'MVert', 5)
    for i, co in enumerate([(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0),
                            (0, 2, 0)]):
        mv.f('co', [float(c) for c in co], i)
        mv.n3('no', [0, 0, 32767], i)
    bf.add('DATA', mv, old=p_mv)
    mf = S(d, 'MFace', 2)
    mf.n('v1', 0, 0)
    mf.n('v2', 1, 0)
    mf.n('v3', 2, 0)
    mf.n('v4', 3, 0)
    mf.n('mat_nr', 0, 0)
    mf.n('v1', 3, 1)                          # the tri: v4 == 0
    mf.n('v2', 2, 1)
    mf.n('v3', 4, 1)
    mf.n('v4', 0, 1)
    mf.n('mat_nr', 0, 1)
    bf.add('DATA', mf, old=p_mf)
    tf = S(d, 'TFace', 2)
    # uv[4][2] is one field: write all 8 floats per face
    tf.f('uv', [c for uv in QUAD_UV for c in uv], 0)
    tf.f('uv', [c * 0.5 for uv in QUAD_UV for c in uv], 1)
    bf.add('DATA', tf, old=p_tf)
    bf.add('DATA', _st.pack(d.end + ('I' if psize == 4 else 'Q'), p_ma),
           sname='Link', old=p_slots)

    ma = S(d, 'Material')
    ma.id_name('MAOldSkin')
    ma.f('r', 0.7)
    ma.f('g', 0.3)
    ma.f('b', 0.2)
    ma.f('specr', 1.0)
    ma.f('specg', 1.0)
    ma.f('specb', 1.0)
    ma.f('amb', 0.5)
    ma.f('ang', 1.0)
    ma.f('alpha', 0.65)
    ma.f('ref', 0.9)
    ma.f('spec', 0.6)
    ma.n('har', 96)
    ma.n('mode', 0x40)                        # MA_ZTRANSP
    ma.n('diff_shader', 0)                    # LAMBERT
    ma.n('spec_shader', 1)                    # PHONG
    p_mtex = bf.alloc()
    ma.p('mtex', p_mtex, elem=2)
    bf.add('MA', ma, old=p_ma)
    mt = S(d, 'MTex')
    mt.n('texco', 1)                          # ORCO (DNA value)
    mt.n('mapto', 1)                          # MAP_COL
    mt.n('blendtype', 1)                      # MUL
    mt.p('tex', p_tex)
    mt.f('size', (3.0, 3.0, 3.0))
    mt.f('colfac', 1.0)
    bf.add('DATA', mt, old=p_mtex)
    te = S(d, 'Tex')
    te.id_name('TEMarb')
    te.n('type', 3)                           # MARBLE
    te.f('noisesize', 0.6)
    te.f('turbul', 5.0)
    te.n('noisedepth', 2)
    bf.add('TE', te, old=p_tex)

    la = S(d, 'Lamp')
    la.id_name('LAKey')
    la.n('type', 2)                           # SPOT
    la.f('r', 1.0)
    la.f('g', 1.0)
    la.f('b', 1.0)
    la.f('energy', 1.0)
    la.f('dist', 20.0)
    la.f('spotsize', 45.0)                    # DEGREES before 2.70
    la.f('spotblend', 0.3)
    bf.add('LA', la, old=p_lamp)

    wo = S(d, 'World')
    wo.id_name('WOWorld')
    wo.f('horr', 0.3)
    wo.f('horg', 0.4)
    wo.f('horb', 0.6)
    wo.f('zenr', 0.05)
    wo.f('zeng', 0.05)
    wo.f('zenb', 0.15)
    bf.add('WO', wo, old=p_world)

    return bf.tobytes(), {'material': p_ma}


def build_modern_stub():
    """A minimal file claiming to be 3.00 -- the version guard's food."""
    d = dna_279()
    bf = BlendFile(d, b'300')
    return bf.tobytes()
