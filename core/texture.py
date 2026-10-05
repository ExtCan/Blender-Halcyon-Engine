"""Texture storage and sampling.

Point sampling is the default because that is what mid-90s software rasterisers
actually did; bilinear, trilinear and the N64's 3-point filter are all here too.
Images are stored (H,W,4) float32 with row 0 at the *bottom*, matching Blender's
pixel order and UV convention.

R251 (the texture pack): the prep-time storage laws of the accelerator era --
the card's texel format (`store_format`: Glide / Direct3D / PowerVR PCX /
Sega Model 2 / Nintendo DS / 3dfx NCC), the Nintendo 64's 4 KB TMEM budget
(`fit_tmem`), block compression (`block_compress`: DXT1, the Xbox NV2A's
16-bit decode, GameCube CMPR, the Dreamcast's VQ codebook) -- run ONCE per
prepared texture, so both devices sample one array and every law is bitwise
by construction. The sample-time dials travel as one `opts` dict
(`sample_opts`): the Voodoo's 4/8-bit texel fraction (`frac_bits`) here, the
wave-2 roads (clamp mode, chroma key, mip select, LOD source, sharpen) on the
same object. POV-Ray's `interpolate 4` is a filter (`_sample_normdist`).
"""

import numpy as np

WRAP_MODES = ('REPEAT', 'EXTEND', 'CLIP', 'MIRROR', 'BORDER')   # BORDER is internal (C077, wave 2); the node enum is unchanged
PYRAMID_FILTERS = ('NEAREST', 'BILINEAR', 'N64_3POINT', 'TRILINEAR')   # filters with a per-level tap set
FOOTPRINT_TEX_FILTERS = ('TRILINEAR', 'SUMMED_AREA')                    # filters that read the footprint by themselves

#: the sample-time dials of R251 at their neutral values: `sample_opts` of a
#: default RenderSettings IS this dict, and `opts=None` means it too
SAMPLE_OPTS_NEUTRAL = dict(frac_bits=0, clamp_mode='EDGE', colorkey=False,
                           colorkey_range=0, mip_select='FILTER',
                           lod_sharpen=False, lod_source='DERIVATIVE',
                           lod_k16=0, lod_l=0, aniso=1)


def sample_opts(settings):
    """The sample-time dials of R251 as one dict; every default is neutral
    (SAMPLE_OPTS_NEUTRAL reproduces 1.89.0 bit for bit). Three rules live
    HERE and nowhere else, so both devices read one decision:
      (1) the level roads need Mipmaps: with tex_mipmap False, mip_select is
          FILTER, lod_source DERIVATIVE and lod_sharpen False -- the greyed
          UI rows are honestly inert (build_mips builds on demand, so
          without this rule they would act with Mipmaps off);
      (2) the filter decides the pyramid: the three level dials act only for
          PYRAMID_FILTERS (CUBIC, POV_NORMDIST and SUMMED_AREA have no
          pyramid and read them as neutral);
      (3) anisotropy is off under any level road: aniso = tex_aniso only
          when lod_source == 'DERIVATIVE' and mip_select == 'FILTER' and not
          lod_sharpen (no card had both; the GPU reads this same function
          through _tex_opts);
      (4) a per-polygon level is ONE level (catalogue C079: "no per-pixel
          LOD and no blend"): under lod_source == 'TRIANGLE', FILTER and
          BLEND read as NEAREST_LEVEL; the Voodoo's DITHER_VOODOO stays (a
          dithered pick of the polygon's level is what that card did)."""
    g = lambda k, d: getattr(settings, k, d)                      # noqa: E731
    filt = str(g('tex_filter', 'NEAREST'))
    o = dict(SAMPLE_OPTS_NEUTRAL)
    o['frac_bits'] = {'FLOAT': 0, 'BITS_4': 4, 'BITS_8': 8}.get(
        str(g('tex_frac_bits', 'FLOAT')), 0)
    o['clamp_mode'] = str(g('tex_clamp_mode', 'EDGE'))
    o['colorkey'] = bool(g('tex_colorkey', False))
    o['colorkey_range'] = int(g('tex_colorkey_range', 0))
    if bool(g('tex_mipmap', False)) and filt in PYRAMID_FILTERS:   # rules (1) and (2)
        o['mip_select'] = str(g('tex_mip_select', 'FILTER'))
        o['lod_sharpen'] = bool(g('tex_lod_sharpen', False))
        o['lod_source'] = str(g('tex_lod_source', 'DERIVATIVE'))
        # half-to-even, the named tie
        o['lod_k16'] = int(np.round(float(g('tex_lod_k', 0.0)) * 16.0))
        o['lod_l'] = int(g('tex_lod_l', 0))
        if filt == 'TRILINEAR' and o['mip_select'] == 'BLEND':
            # Blend IS what Trilinear does: one picture, one dial position
            o['mip_select'] = 'FILTER'
        if o['lod_source'] == 'TRIANGLE' and o['mip_select'] in ('FILTER', 'BLEND'):
            o['mip_select'] = 'NEAREST_LEVEL'                    # rule (4)
    level_road = (o['lod_source'] != 'DERIVATIVE' or o['mip_select'] != 'FILTER'
                  or o['lod_sharpen'])
    o['aniso'] = 1 if level_road else int(g('tex_aniso', 1) or 1)   # rule (3)
    return o


def footprint_wanted(settings):
    """True when a sampler will read the screen footprint: the derivative
    gate of ShadeJob.context and the GPU's `wants_fp` are this one
    predicate."""
    filt = str(getattr(settings, 'tex_filter', 'NEAREST'))
    if filt in FOOTPRINT_TEX_FILTERS:
        return True
    o = sample_opts(settings)
    return filt in PYRAMID_FILTERS and (o['lod_sharpen'] or o['mip_select'] != 'FILTER')


# ------------------------------------------------- R251 wave 2 (TEX-2) laws
def colorkey_threshold(rng, decoded):
    """C083: the chroma key's compare value for `tex_colorkey_range` levels
    (Voodoo2 chromaRange; 0 = the Voodoo1 / Direct3D exact match). The
    hardware compared BYTES: c*255 < range + 0.5 <=> round(c*255) <= range.
    On a texture the pipeline decoded to linear (`decoded`), the same
    byte's decoded value is the threshold, so the compare of the hardware
    becomes the compare of the linear texel. One float32 scalar."""
    e = (int(rng) + 0.5) / 255.0
    if decoded:
        from .mathx import srgb_to_linear
        return np.float32(srgb_to_linear(np.array([e], np.float32))[0])
    return np.float32(e)


#: C088: 1/1 .. 1/256 as the CPU's own float32 reciprocals; both roads
#: read this table (the GPU as the hal_recip256 atlas), never a divide
RECIP256 = (np.float32(1.0) / np.arange(1, 257, dtype=np.float32)).astype(np.float32)


def summed_area_table(pixels):
    """C088: Crow's summed-area table of the 16-bit texel values (np.round
    half-to-even, clipped to 0..65535), cumulative over rows then columns
    modulo 2^32, split into two EXACT float32 planes (hi = S >> 16,
    lo = S & 65535), each < 2^16. The GPU atlas holds the same two planes
    side by side; a box is four reads and a uint32 wrap on both roads."""
    t16 = np.round(np.asarray(pixels, np.float32) * np.float32(65535.0))
    t16 = np.clip(t16, 0.0, 65535.0).astype(np.uint64)
    s = t16.cumsum(axis=0).cumsum(axis=1)
    s32 = s & np.uint64(0xFFFFFFFF)
    hi = (s32 >> np.uint64(16)).astype(np.float32)
    lo = (s32 & np.uint64(65535)).astype(np.float32)
    return hi, lo


def gs_lod16(depth, k16, l):
    """C022: the PlayStation 2 GS TEX1 rule, LOD = (log2(1/|Q|) << L) + K in
    7.4 fixed point, on the float32 bit pattern of `depth` (= |view z| =
    1/|Q|): the exponent is the whole part, the top four mantissa bits the
    GS's four fractional bits. Integer arithmetic, then one exact scale by
    1/16 (|lod16| < 2^24): the sampler's floor / frac recover level =
    lod16 >> 4 and frac = (lod16 & 15) / 16. Both devices read this
    function (the GPU through the uploaded hal_lodq field)."""
    bits = np.ascontiguousarray(np.asarray(depth, np.float32)).view(np.int32)
    e = ((bits >> 23) & 255) - 127
    m4 = (bits >> 19) & 15
    lod16 = ((e * 16 + m4) << int(l)) + int(k16)
    return lod16.astype(np.float32) * np.float32(0.0625)


def clamp9(v8):
    """C008: angrylion's 9-bit combiner clamp on an integer-valued array:
    the low nine bits are the two's-complement pattern; bit 8 set means
    out of range and bit 7 says which way (300 -> 255, 400 -> 0, -10 -> 0,
    -200 -> 255). Returns 0..255 int32."""
    w9 = np.asarray(v8, np.int32) & 0x1FF
    return np.where(w9 & 0x100, np.where(w9 & 0x80, 0, 255), w9).astype(np.int32)


def _bayer4_16():
    from .dither import BAYER4
    return np.round(np.asarray(BAYER4, np.float32) * 16.0).astype(np.float32)


#: C072: the ordinary 4x4 Bayer matrix as exact integers 0..15 (DI.BAYER4 * 16),
#: indexed [py & 3, px & 3]; the GLSL twin writes it as bit arithmetic
BAYER4_16 = _bayer4_16()


# ------------------------------------------------------------ texel laws
#: bits per texel of the N64's TMEM formats (C013)
TMEM_BPT = {'RGBA16': 16, 'RGBA32': 32, 'CI8': 8, 'CI4': 4, 'IA16': 16,
            'IA8': 8, 'IA4': 4, 'I8': 8, 'I4': 4}
#: the texel law each TMEM format stores at (three INTERNAL names beyond the
#: C074 enum: RGBA8888, IA31, I4A)
TMEM_INTERNAL = {'RGBA16': 'ARGB1555', 'RGBA32': 'RGBA8888', 'CI8': 'ARGB1555',
                 'CI4': 'ARGB1555', 'IA16': 'AI88', 'IA8': 'AI44', 'IA4': 'IA31',
                 'I8': 'I8', 'I4': 'I4A'}
#: every texel format `store_format` accepts (the C074 enum + the internal three)
TEXEL_FORMATS = ('RGB565', 'ARGB1555', 'ARGB4444', 'RGB332', 'RGB5550', 'I8', 'A8',
                 'AI44', 'AI88', 'I4', 'I4_MODEL2', 'A3I5', 'A5I3', 'YIQ422',
                 'AYIQ8422', 'RGBA8888', 'IA31', 'I4A')
COMPRESS_MODES = ('DXT1', 'DXT1_NV2A', 'CMPR_GC', 'VQ_DC')


def _k(v, b):
    """Quantise 0..1 float32 to b bits: np.round = half-to-even, the named tie."""
    return np.round(np.asarray(v, np.float32) * np.float32(2 ** b - 1)).astype(np.int32)


def _repl(k, b):
    """The chip's expansion of a b-bit value to a byte by bit replication."""
    if b == 8:
        return k
    if b == 6:
        return 4 * k + k // 16
    if b == 5:
        return 8 * k + k // 4
    if b == 4:
        return 17 * k
    if b == 3:
        return 36 * k + k // 2
    if b == 2:
        return 85 * k
    if b == 1:
        return 255 * k
    raise ValueError(b)


def _q(v, b):
    """Round to b bits and expand back: one float32 division by 255."""
    return _repl(_k(v, b), b).astype(np.float32) / np.float32(255)


def _luma601(e):
    """Rec.601 luma on the ENCODED value, float32, in this order."""
    return ((np.float32(0.299) * e[..., 0] + np.float32(0.587) * e[..., 1])
            + np.float32(0.114) * e[..., 2]).astype(np.float32)


def _ncc_fit(e8):
    """3dfx NCC (narrow-channel compression): a 16-level luma ramp plus a
    4+4 chroma vocabulary, fitted once per image in float64 (Halcyon's own
    deterministic fit -- TexUS's fitter was never published), then decoded
    by integer table lookups exactly as the card's GuNccTable did."""
    R8 = e8[..., 0].astype(np.float64)
    G8 = e8[..., 1].astype(np.float64)
    B8 = e8[..., 2].astype(np.float64)
    Y = 0.299 * R8 + 0.587 * G8 + 0.114 * B8
    I = 0.596 * R8 - 0.274 * G8 - 0.322 * B8
    Q = 0.211 * R8 - 0.523 * G8 + 0.312 * B8
    ymin, ymax = float(Y.min()), float(Y.max())
    imin, imax = float(I.min()), float(I.max())
    qmin, qmax = float(Q.min()), float(Q.max())
    ytab = np.array([min(max(round(ymin + j * (ymax - ymin) / 15), 0), 255)
                     for j in range(16)], np.int64)
    ilev = np.array([imin + j * (imax - imin) / 3 for j in range(4)], np.float64)
    qlev = np.array([qmin + j * (qmax - qmin) / 3 for j in range(4)], np.float64)

    def _cl(x):
        return min(max(round(x), -256), 255)

    irgb = np.array([(_cl(0.956 * l), _cl(-0.272 * l), _cl(-1.106 * l)) for l in ilev],
                    np.int64)
    qrgb = np.array([(_cl(0.621 * l), _cl(-0.647 * l), _cl(1.703 * l)) for l in qlev],
                    np.int64)
    # independent nearest, ties lowest (argmin returns the first minimum)
    yi = np.argmin(np.abs(Y[..., None] - ytab[None, None, :].astype(np.float64)), axis=-1)
    ii = np.argmin(np.abs(I[..., None] - ilev[None, None, :]), axis=-1)
    qi = np.argmin(np.abs(Q[..., None] - qlev[None, None, :]), axis=-1)
    rgb8 = np.clip(ytab[yi][..., None] + irgb[ii] + qrgb[qi], 0, 255)
    table = np.clip(ytab[:, None, None, None] + irgb[None, :, None, :]
                    + qrgb[None, None, :, :], 0, 255).reshape(-1, 3)
    return rgb8.astype(np.int32), yi, table


class Texture:
    __slots__ = ('pixels', 'width', 'height', 'mips', 'name', 'colorspace', 'wrap',
                 'filter', 'quantized',
                 'format', 'compress', 'sat', 'prep')  # R251: C074 / C024 / C088 / the prep tag (A12.13)

    def __init__(self, pixels, name='', colorspace='sRGB', wrap='REPEAT',
                 filt='NEAREST'):
        px = np.asarray(pixels, dtype=np.float32)
        if px.ndim == 2:
            px = px[:, :, None]
        if px.shape[2] == 1:
            px = np.repeat(px, 4, axis=2)
            px[:, :, 3] = 1.0
        elif px.shape[2] == 3:
            px = np.concatenate([px, np.ones(px.shape[:2] + (1,), np.float32)], axis=2)
        self.pixels = np.ascontiguousarray(px)
        self.height, self.width = self.pixels.shape[:2]
        self.mips = None
        self.name = name
        self.colorspace = colorspace
        self.wrap = wrap
        self.filter = filt
        self.quantized = False
        self.format = 'NONE'
        self.compress = 'NONE'
        self.sat = None
        self.prep = None

    # ------------------------------------------------------------- authoring
    def to_linear(self):
        if self.colorspace == 'sRGB':
            from .mathx import srgb_to_linear
            # decode the CONTIGUOUS RGBA buffer and put alpha back,
            # instead of slicing a strided (H,W,3) view: the slice
            # forced a copy and strided arithmetic on multi-megapixel
            # images, and alpha restored afterwards is the same bits
            # as alpha never touched
            rgba = self.pixels
            out = srgb_to_linear(rgba)
            out[:, :, 3] = rgba[:, :, 3]
            self.pixels = np.ascontiguousarray(out)
            self.colorspace = 'Linear'
        return self

    def clamp_size(self, max_size):
        """Downsample to a hardware-style texture budget (box filter)."""
        if not max_size or (self.width <= max_size and self.height <= max_size):
            return self
        while self.width > max_size or self.height > max_size:
            h = max(1, self.height // 2)
            w = max(1, self.width // 2)
            self.pixels = _box_half(self.pixels)
            self.height, self.width = self.pixels.shape[:2]
            if self.height == h and self.width == w and (h == 1 and w == 1):
                break
        self.mips = None
        return self

    def quantize(self, colors):
        """Reduce the texture to N colours -- 90s texture memory was tiny."""
        if not colors or colors >= 256 * 256:
            return self
        from .palette import quantize_image
        rgb = self.pixels[:, :, :3]
        out, _pal = quantize_image(rgb, colors, method='MEDIAN_CUT', dither='NONE')
        self.pixels = np.concatenate([out, self.pixels[:, :, 3:]], axis=2)
        self.quantized = True
        self.mips = None
        return self

    def build_mips(self):
        if self.mips is not None:
            return self.mips
        mips = [self.pixels]
        cur = self.pixels
        while cur.shape[0] > 1 or cur.shape[1] > 1:
            cur = _box_half(cur)
            mips.append(cur)
        self.mips = mips
        return mips

    # ------------------------------------------------- R251 storage laws
    def _encoded_rgb(self, encoded):
        """(e, a): the encoded 0..1 bytes-as-float32 and the alpha plane."""
        rgb = self.pixels[:, :, :3]
        a = self.pixels[:, :, 3]
        if encoded:
            return np.ascontiguousarray(rgb, np.float32), a
        from .mathx import linear_to_srgb
        return np.ascontiguousarray(linear_to_srgb(rgb), np.float32), a

    def _store_rgb(self, rgb_e, a, encoded):
        """Put the quantised ENCODED rgb (0..1 float32) and alpha back, decoding
        with the pipeline's own srgb_to_linear when the image was decoded."""
        rgb = rgb_e
        if not encoded:
            from .mathx import srgb_to_linear
            rgb = srgb_to_linear(np.ascontiguousarray(rgb_e, np.float32))
        px = np.empty(self.pixels.shape, np.float32)
        px[:, :, :3] = rgb
        px[:, :, 3] = a
        self.pixels = np.ascontiguousarray(px)
        self.mips = None
        self.sat = None

    def store_format(self, fmt, encoded, keyed=False):
        """C074: store the texture at the card's texel format -- rounded once,
        as the Glide / Direct3D drivers converted, expanded by bit
        replication as the chip fetched. `encoded` says whether the
        pixels are still the file's bytes (True) or were decoded to linear
        (then the law re-encodes, quantises and decodes with the
        pipeline's own converters). One prepared texture for both devices.
        `keyed` (C083): under a format with NO alpha plane the driver's
        conversion of cut-outs to the key colour happens here, from the
        source alpha this law is about to drop."""
        if fmt not in TEXEL_FORMATS:
            raise ValueError(f'unknown texel format {fmt!r}')
        e, a = self._encoded_rgb(encoded)
        a = np.asarray(a, np.float32)
        r, g, b = e[:, :, 0], e[:, :, 1], e[:, :, 2]
        one = np.ones(a.shape, np.float32)
        if fmt == 'RGB565':
            out = np.stack([_q(r, 5), _q(g, 6), _q(b, 5)], axis=-1)
            aa = one
        elif fmt == 'ARGB1555':
            out = np.stack([_q(r, 5), _q(g, 5), _q(b, 5)], axis=-1)
            aa = np.where(a >= 0.5, np.float32(1.0), np.float32(0.0)).astype(np.float32)
        elif fmt == 'ARGB4444':
            out = np.stack([_q(r, 4), _q(g, 4), _q(b, 4)], axis=-1)
            aa = _q(a, 4)
        elif fmt == 'RGB332':
            out = np.stack([_q(r, 3), _q(g, 3), _q(b, 2)], axis=-1)
            aa = one
        elif fmt == 'RGB5550':
            out = np.stack([_q(r, 5), _q(g, 5), _q(b, 5)], axis=-1)
            aa = one
        elif fmt == 'RGBA8888':
            out = np.stack([_q(r, 8), _q(g, 8), _q(b, 8)], axis=-1)
            aa = _q(a, 8)
        elif fmt == 'A8':
            out = np.ones(e.shape, np.float32)
            aa = _q(a, 8)
        elif fmt in ('YIQ422', 'AYIQ8422'):
            e8 = np.stack([_k(r, 8), _k(g, 8), _k(b, 8)], axis=-1)
            rgb8, _yi, _table = _ncc_fit(e8)
            out = rgb8.astype(np.float32) / np.float32(255)
            aa = one if fmt == 'YIQ422' else _q(a, 8)
        else:
            lum = _luma601(e)
            if fmt == 'I8':
                i = _q(lum, 8)
                aa = i
            elif fmt == 'AI44':
                i = _q(lum, 4)
                aa = _q(a, 4)
            elif fmt == 'AI88':
                i = _q(lum, 8)
                aa = _q(a, 8)
            elif fmt == 'I4':
                i = _q(lum, 4)
                aa = one
            elif fmt == 'I4_MODEL2':
                i = _q(lum, 4)
                # the 0xF marker: white texels are holes, as on the board
                aa = np.where(_k(lum, 4) == 15, np.float32(0.0),
                              np.float32(1.0)).astype(np.float32)
            elif fmt == 'I4A':
                i = _q(lum, 4)
                aa = i                       # the N64's I formats read alpha from intensity
            elif fmt == 'A3I5':
                i = _q(lum, 5)
                aa = _q(a, 3)
            elif fmt == 'A5I3':
                i = _q(lum, 3)
                aa = _q(a, 5)
            elif fmt == 'IA31':
                i = _q(lum, 3)
                aa = np.where(a >= 0.5, np.float32(1.0), np.float32(0.0)).astype(np.float32)
            else:                            # pragma: no cover -- guarded above
                raise ValueError(fmt)
            out = np.repeat(i[:, :, None], 3, axis=2)
        if keyed and aa is one:
            # C083: the cut-out is the only record of the hole and this
            # format carries no alpha -- afterwards `colorkey_prepare`
            # has nothing left to find (the VOODOO preset keyed NOTHING
            # on RGB565 until this line). Black is 0 on either side of
            # the decode, so the stored texel is the key exactly.
            out = np.where((a < np.float32(0.5))[:, :, None],
                           np.float32(0.0), out).astype(np.float32)
        self._store_rgb(np.ascontiguousarray(out, np.float32),
                        np.ascontiguousarray(aa, np.float32), encoded)
        self.format = fmt
        return self

    def fit_tmem(self, fmt, mips, encoded):
        """C013: fit the texture into the Nintendo 64's 4 KB TMEM by format
        -- rows padded to 64-bit words, colour-index formats confined to the
        lower 2 KB, the mip chain counted when `mips` -- by the largest
        halving that fits, then the format's own texel law (the palette
        formats through the median cut, the TLUT at 5551)."""
        if fmt not in TMEM_BPT:
            raise ValueError(f'unknown TMEM format {fmt!r}')
        w, h = int(self.width), int(self.height)
        while tmem_bytes(w, h, fmt, mips) > tmem_budget(fmt) and (w > 1 or h > 1):
            w, h = max(1, w // 2), max(1, h // 2)
        while (int(self.width), int(self.height)) != (w, h):
            self.pixels = _box_half(self.pixels)
            self.height, self.width = self.pixels.shape[:2]
        if fmt == 'CI8':
            self.quantize(256)
        elif fmt == 'CI4':
            self.quantize(16)
        self.store_format(TMEM_INTERNAL[fmt], encoded)
        self.mips = None
        return self

    def block_compress(self, mode, encoded):
        """C024: bake the era's block compression -- DXT1 (S3TC), the Xbox
        NV2A's 16-bit decode, GameCube CMPR, the Dreamcast's VQ codebook --
        into the texels, once, from the encoded bytes."""
        if mode not in COMPRESS_MODES:
            raise ValueError(f'unknown block compression {mode!r}')
        e, a = self._encoded_rgb(encoded)
        E = np.stack([_k(e[:, :, 0], 8), _k(e[:, :, 1], 8), _k(e[:, :, 2], 8)],
                     axis=-1)                                     # int32 (H,W,3)
        A = np.asarray(a, np.float32) >= 0.5
        if mode == 'VQ_DC':
            rgb8, alpha = _vq_encode(E, _k(a, 8))
        else:
            rgb8, alpha = _dxt1_encode(E, A, mode)
        self._store_rgb(rgb8.astype(np.float32) / np.float32(255),
                        alpha.astype(np.float32), encoded)
        self.compress = mode
        return self

    def colorkey_prepare(self):
        """C083, the prep half: the driver's conversion of alpha cut-outs
        to the key colour -- a keyed format carried no alpha, so a texel
        below one half alpha is stored as OPAQUE black; a texel at or above
        one half keeps its own alpha (a 4444 texture under chromakey kept
        its sixteen levels). The filter then bleeds black into every keyed
        edge, and the sample-time compare (`sample`) tests the RESULT.
        A format without an alpha plane (RGB565, RGB332, RGB5550, YIQ422,
        I4) has already converted its holes in `store_format(keyed=True)`:
        this pass then finds none there and changes nothing."""
        px = np.ascontiguousarray(self.pixels, np.float32).copy()
        hole = px[:, :, 3] < 0.5
        px[hole, :3] = 0.0
        px[hole, 3] = 1.0
        self.pixels = px
        self.mips = None
        self.sat = None
        return self

    def build_sat(self):
        """C088: the summed-area table (`summed_area_table`), built on
        demand and kept on the texture like the mip chain."""
        if self.sat is None:
            self.sat = summed_area_table(self.pixels)
        return self.sat

    # -------------------------------------------------------------- sampling
    def sample(self, u, v, filt=None, wrap=None, lod=None, aniso=1,
               duv=None, dvv=None, bias=0.0, opts=None, depth=None, px=None,
               py=None):
        """`opts=None` is neutral: with the old inputs the dispatch is
        1.89.0's. `opts` (from `sample_opts`) carries the R251 dials;
        `depth` and `px`/`py` are the wave-2 level roads' inputs (the GS's
        Q, the Voodoo's dither pixel).

        The wave-2 roads, in order: GL_CLAMP maps an Extend wrap to the
        internal BORDER (C077); Summed Area is its own filter (C088); the
        GS rule replaces the level from `depth` (C022); a Mip Level Select
        picks or blends whole levels with the filter's own taps (C072, the
        per-polygon level of C079 arriving as `lod`); Sharpen extrapolates
        under magnification (C008); the chroma key discards after the
        filter (C083). Every road rides the footprint (`duv is not None`),
        as the GPU's emission site does."""
        filt = filt or self.filter
        wrap = wrap or self.wrap
        u = np.asarray(u, np.float32)
        v = np.asarray(v, np.float32)
        o = opts if opts is not None else SAMPLE_OPTS_NEUTRAL
        fb = int(o['frac_bits'])
        if wrap == 'EXTEND' and o['clamp_mode'] == 'GL_CLAMP':
            wrap = 'BORDER'                                          # C077
        if filt == 'SUMMED_AREA':
            out = self._sample_sat(u, v, duv, dvv, wrap, float(bias))  # C088
            return self._colorkey_tail(out, o)
        pyramid = filt in PYRAMID_FILTERS
        fp = duv is not None
        if pyramid and fp and o['lod_source'] == 'GS_Q' and depth is not None \
                and (filt == 'TRILINEAR' or o['mip_select'] != 'FILTER'):
            lod = gs_lod16(depth, o['lod_k16'], o['lod_l'])            # C022
        if pyramid and fp and o['mip_select'] != 'FILTER':
            out = self._sample_level_select(u, v, filt, wrap, lod, duv, dvv,
                                            float(bias), o, px, py)     # C072 / C079
        elif filt == 'TRILINEAR' and duv is not None and int(aniso) > 1:
            out = self._sample_aniso(u, v, duv, dvv, float(bias), wrap,
                                     int(aniso), frac_bits=fb)
        elif filt == 'TRILINEAR' and lod is not None:
            out = self._sample_trilinear(u, v, lod, wrap, frac_bits=fb)
        elif filt == 'NEAREST':
            out = self._sample_nearest(self.pixels, u, v, wrap)
        elif filt == 'N64_3POINT':
            out = self._sample_3point(self.pixels, u, v, wrap)
        elif filt == 'CUBIC':
            out = self._sample_bicubic(self.pixels, u, v, wrap)
        elif filt == 'POV_NORMDIST':
            out = self._sample_normdist(self.pixels, u, v, wrap)
        else:
            out = self._sample_bilinear(self.pixels, u, v, wrap, frac_bits=fb)
        if pyramid and fp and o['lod_sharpen']:
            out = self._sharpen(out, u, v, filt, wrap, duv, dvv, fb)   # C008
        return self._colorkey_tail(out, o)

    # ------------------------------------------- R251 wave-2 sample roads
    def _colorkey_tail(self, out, o):
        """C083, the sample half: after the filter (and after CLIP's
        zeroing) a sample below the threshold on every channel is keyed
        out -- the verdict DISCARDS (alpha 0); a survivor keeps its own
        alpha. Strict compare: ties at exactly (range + 0.5)/255 survive."""
        if not o['colorkey']:
            return out
        thr = colorkey_threshold(int(o['colorkey_range']), self.colorspace == 'Linear')
        kill = (out[:, 0] < thr) & (out[:, 1] < thr) & (out[:, 2] < thr)
        out = np.array(out, np.float32, copy=True)
        out[:, 3] = np.where(kill, np.float32(0.0), out[:, 3])
        return out

    def _level_filter(self, filt, frac_bits):
        """The filter's own per-level sampler F(img, u, v, wrap) for the
        level roads (C072 / C008): TRILINEAR's is bilinear."""
        if filt == 'NEAREST':
            return self._sample_nearest
        if filt == 'N64_3POINT':
            return self._sample_3point

        def _bil(img, u, v, wrap, _fb=int(frac_bits)):
            return self._sample_bilinear(img, u, v, wrap, frac_bits=_fb)
        return _bil

    def _sample_level_select(self, u, v, filt, wrap, lod, duv, dvv, bias, o,
                             px, py):
        """C072: pick or blend whole mip levels with the filter's own taps.
        lod -> clamp -> select -> per-level filter -> (BLEND) lerp; every
        step exact in float32 (integers < 2^24, power-of-two scales), so
        the GPU's hal_pick is bitwise on the same CPU-decided lod.
          BLEND          the RDP's lod_frac lerp over the filter's taps
                         (delegates to _sample_trilinear: BILINEAR + BLEND
                         IS Trilinear, structurally)
          NEAREST_LEVEL  GL 1.1: q = ceil(lambda + 1/2) - 1 (a tie at .5
                         goes DOWN, written with floor)
          DITHER_VOODOO  MAME's 8.8 fixed lod + (bayer4 << 4) >> 8 on the
                         sample's screen pixel."""
        mips = self.build_mips()
        n1 = np.float32(len(mips) - 1)
        F = self._level_filter(filt, int(o['frac_bits']))
        if lod is None:
            lod = compute_lod(duv, dvv, self.width, self.height, bias)
        lc = np.clip(np.asarray(lod, np.float32), np.float32(0.0), n1)
        sel = o['mip_select']
        if sel == 'BLEND':
            return self._sample_trilinear(u, v, lc, wrap, frac_bits=int(o['frac_bits']), F=F)
        if sel == 'DITHER_VOODOO':
            if px is None or py is None:
                d16 = np.zeros(u.shape[0], np.float32)
            else:
                d16 = BAYER4_16[np.asarray(py, np.int64) & 3, np.asarray(px, np.int64) & 3]
            lod8 = np.floor(lc * np.float32(256.0))
            dd = d16 * np.float32(16.0)
            ls = lod8 + dd
            lq = ls * np.float32(0.00390625)
            lv = np.floor(lq)
        else:                                                  # NEAREST_LEVEL
            lq = lc + np.float32(0.5)
            lc2 = np.float32(0.0) - np.floor(np.float32(0.0) - lq)
            lv = lc2 - np.float32(1.0)
        lv = np.clip(lv, np.float32(0.0), n1).astype(np.int32)
        out = np.zeros((u.shape[0], 4), np.float32)
        for lvl in np.unique(lv):
            m = lv == lvl
            out[m] = F(mips[int(lvl)], u[m], v[m], wrap)
        return out

    def _sharpen(self, out, u, v, filt, wrap, duv, dvv, frac_bits):
        """C008: the RDP's G_TD_SHARPEN. Under magnification (fewer than
        one texel per pixel: m = max |ds|,|dt| < 1) the LOD fraction is
        NEGATIVE, lod_frac = m - 1, so (TEXEL1 - TEXEL0) * frac + TEXEL0
        extrapolates level 0 AWAY from level 1 -- most at full
        magnification, none at 1:1 -- and the 9-bit combiner clamp
        saturates the overshoot. Halcyon lerps its float texels, rounds to
        8 bits (half-to-even) and clamps, one multiply by the shared
        float32 1/255 on both roads. A one-level texture is untouched."""
        mips = self.build_mips()
        if len(mips) < 2:
            return out
        w = np.float32(self.width)
        h = np.float32(self.height)
        ax = np.abs(duv[:, 0]) * w
        ay = np.abs(duv[:, 1]) * w
        bx = np.abs(dvv[:, 0]) * h
        by = np.abs(dvv[:, 1]) * h
        mx = np.maximum(ax, ay)
        my = np.maximum(bx, by)
        m = np.maximum(mx, my)
        mag = m < np.float32(1.0)
        if not mag.any():
            return out
        F = self._level_filter(filt, frac_bits)
        um, vm = u[mag], v[mag]
        t0 = F(mips[0], um, vm, wrap)
        t1 = F(mips[1], um, vm, wrap)
        f = (m[mag] - np.float32(1.0))[:, None]
        d = t1 - t0
        df = d * f
        vv = t0 + df
        v255 = vv * np.float32(255.0)
        v8 = np.round(v255).astype(np.int32)
        w9 = v8 & 0x1FF
        out8 = np.where(w9 & 0x100, np.where(w9 & 0x80, 0, 255), w9)
        out = np.array(out, np.float32, copy=True)
        out[mag] = out8.astype(np.float32) * np.float32(1.0 / 255.0)
        return out

    def _sample_sat(self, u, v, duv, dvv, wrap, bias):
        """C088: Crow's summed-area box over the footprint rectangle --
        half-extents hu, hv = half the sum of the axis's two screen
        extents in texels (at least half a texel, at most 127: the 2^32
        window), scaled by 2^bias; the CONTINUOUS coordinate wraps
        (REPEAT, the continuous MIRROR with no "- 1"), the box clamps at
        the texture edge, four uint32 reads, a wrap-around difference,
        three correctly-rounded multiplies (two reciprocals from RECIP256,
        one by 1/65535), no division. Without a footprint the box of one
        texel IS the nearest texel."""
        if duv is None:
            return self._sample_nearest(self.pixels, u, v, wrap)
        hi, lo = self.build_sat()
        w, h = int(self.width), int(self.height)
        W = np.float32(w)
        H = np.float32(h)
        scale = np.float32(2.0 ** float(bias))
        half = np.float32(0.5)
        cap = np.float32(127.0)

        def _extent(d, dim):
            a = np.abs(np.asarray(d[:, 0], np.float32))
            b = np.abs(np.asarray(d[:, 1], np.float32))
            s = a + b
            hm = s * half
            hw = hm * dim
            hs = hw * scale
            hu = np.maximum(hs, half)
            return np.minimum(hu, cap)
        hu = _extent(duv, W)
        hv = _extent(dvv, H)
        uc = u * W
        vc = v * H
        if wrap == 'REPEAT':
            uc = uc - np.floor(uc / W) * W
            vc = vc - np.floor(vc / H) * H
        elif wrap == 'MIRROR':
            W2 = np.float32(2.0 * w)
            H2 = np.float32(2.0 * h)
            mu = uc - np.floor(uc / W2) * W2
            uc = np.where(mu < W, mu, W2 - mu).astype(np.float32)
            mv = vc - np.floor(vc / H2) * H2
            vc = np.where(mv < H, mv, H2 - mv).astype(np.float32)
        zero = np.float32(0.0)
        one = np.float32(1.0)
        x0 = np.floor(uc - hu)
        nx1 = zero - (uc + hu)
        fx1 = np.floor(nx1)
        x1 = (zero - fx1) - one
        y0 = np.floor(vc - hv)
        ny1 = zero - (vc + hv)
        fy1 = np.floor(ny1)
        y1 = (zero - fy1) - one
        x0 = np.clip(x0, 0, w - 1).astype(np.int64)
        x1 = np.clip(x1, 0, w - 1).astype(np.int64)
        y0 = np.clip(y0, 0, h - 1).astype(np.int64)
        y1 = np.clip(y1, 0, h - 1).astype(np.int64)

        def S(x, y):
            valid = (x >= 0) & (y >= 0)
            xc = np.maximum(x, 0)
            yc = np.maximum(y, 0)
            val = hi[yc, xc].astype(np.uint32) * np.uint32(65536) \
                + lo[yc, xc].astype(np.uint32)
            return np.where(valid[:, None], val, np.uint32(0)).astype(np.uint32)
        A = S(x1, y1)
        B = S(x0 - 1, y1)
        C = S(x1, y0 - 1)
        D = S(x0 - 1, y0 - 1)
        with np.errstate(over='ignore'):
            AB = A - B
            ABC = AB - C
            box = ABC + D                                   # uint32 wrap
        qh = box >> np.uint32(16)
        ql = box & np.uint32(65535)
        fh = qh.astype(np.float32) * np.float32(65536.0)
        f = fh + ql.astype(np.float32)
        m1 = f * RECIP256[x1 - x0][:, None]
        m2 = m1 * RECIP256[y1 - y0][:, None]
        out = (m2 * np.float32(1.0 / 65535.0)).astype(np.float32)
        oob = self._oob(u, v, wrap)
        if oob is not None:
            out[oob] = 0.0
        return out

    # -------------------------------------------------------------- internals
    @staticmethod
    def _wrap_index(i, n, mode):
        if mode == 'REPEAT':
            return np.mod(i, n)
        if mode == 'MIRROR':
            period = 2 * n
            m = np.mod(i, period)
            return np.where(m < n, m, period - 1 - m)
        return np.clip(i, 0, n - 1)

    def _oob(self, u, v, wrap):
        if wrap != 'CLIP':
            return None
        return (u < 0.0) | (u > 1.0) | (v < 0.0) | (v > 1.0)

    def _sample_nearest(self, img, u, v, wrap):
        h, w = img.shape[:2]
        x = np.floor(u * w).astype(np.int64)
        y = np.floor(v * h).astype(np.int64)
        oob = self._oob(u, v, wrap)
        x = self._wrap_index(x, w, wrap)
        y = self._wrap_index(y, h, wrap)
        out = img[y, x]
        if oob is not None:
            out = out.copy()
            out[oob] = 0.0
        return out

    @staticmethod
    def _border_taps(x0, y0, w, h, c00, c10, c01, c11):
        """C077 GL_CLAMP: the coordinate was clamped, so x0 lies in
        [-1, w-1] and x0+1 in [0, w]; a tap past the last texel reads the
        GL default border (0, 0, 0, 0). Per tap, per axis, integer
        compares -- no tie rule."""
        vx0 = x0 >= 0
        vx1 = (x0 + 1) <= w - 1
        vy0 = y0 >= 0
        vy1 = (y0 + 1) <= h - 1
        z = np.float32(0.0)
        c00 = np.where((vx0 & vy0)[:, None], c00, z).astype(np.float32)
        c10 = np.where((vx1 & vy0)[:, None], c10, z).astype(np.float32)
        c01 = np.where((vx0 & vy1)[:, None], c01, z).astype(np.float32)
        c11 = np.where((vx1 & vy1)[:, None], c11, z).astype(np.float32)
        return c00, c10, c01, c11

    def _sample_bilinear(self, img, u, v, wrap, frac_bits=0):
        h, w = img.shape[:2]
        if wrap == 'BORDER':
            # C077: GL_CLAMP clamps the COORDINATE (float32); the taps
            # still straddle the edge and read the border below
            u = np.clip(u, np.float32(0.0), np.float32(1.0))
            v = np.clip(v, np.float32(0.0), np.float32(1.0))
        fx = u * w - 0.5
        fy = v * h - 0.5
        x0 = np.floor(fx).astype(np.int64)
        y0 = np.floor(fy).astype(np.int64)
        tx = (fx - x0).astype(np.float32)[:, None]
        ty = (fy - y0).astype(np.float32)[:, None]
        if frac_bits:
            # C080: the Voodoo1's bilinear unit kept the top four bits of the
            # texel fraction after the half-texel shift (the Voodoo2 eight):
            # tx in [0,1), tx*n exact (a power-of-two scale), floor exact,
            # *inv exact -> k/n. No tie rule needed
            n = np.float32(2 ** frac_bits)
            inv = np.float32(1.0 / 2 ** frac_bits)
            tx = np.floor(tx * n) * inv
            ty = np.floor(ty * n) * inv
        oob = self._oob(u, v, wrap)
        x0w = self._wrap_index(x0, w, wrap)
        x1w = self._wrap_index(x0 + 1, w, wrap)
        y0w = self._wrap_index(y0, h, wrap)
        y1w = self._wrap_index(y0 + 1, h, wrap)
        c00 = img[y0w, x0w]
        c10 = img[y0w, x1w]
        c01 = img[y1w, x0w]
        c11 = img[y1w, x1w]
        if wrap == 'BORDER':
            c00, c10, c01, c11 = self._border_taps(x0, y0, w, h, c00, c10, c01, c11)
        top = c00 + (c10 - c00) * tx
        bot = c01 + (c11 - c01) * tx
        out = top + (bot - top) * ty
        if oob is not None:
            out = out.copy()
            out[oob] = 0.0
        return out

    def _sample_normdist(self, img, u, v, wrap):
        """C111: POV-Ray's `interpolate 4` -- the four texels around the
        sample weighted by the inverse SQUARED distance to each centre,
        normalised (imageutil.cpp Interp() / norm_dist()): a cusp at every
        centre, texels as soft blobs with sharp middles. float32
        throughout, one operation per line where the order matters; the
        GLSL twin in gpu/material writes the same statements."""
        h, w = img.shape[:2]
        xp = u * w + 0.5                     # POV's +0.5: texel i's centre is xp = i + 1
        yp = v * h + 0.5
        ix = np.floor(xp).astype(np.int64)
        iy = np.floor(yp).astype(np.int64)
        p = (xp - ix).astype(np.float32)
        q = (yp - iy).astype(np.float32)
        wx0 = self._wrap_index(ix, w, wrap)
        wx1 = self._wrap_index(ix - 1, w, wrap)
        wy0 = self._wrap_index(iy, h, wrap)
        wy1 = self._wrap_index(iy - 1, h, wrap)
        c0 = img[wy0, wx0]
        c1 = img[wy0, wx1]
        c2 = img[wy1, wx0]
        c3 = img[wy1, wx1]
        pm = np.float32(1.0) - p
        qm = np.float32(1.0) - q
        pm2 = pm * pm
        qm2 = qm * qm
        p2 = p * p
        q2 = q * q
        d0 = pm2 + qm2
        d1 = p2 + qm2
        d2 = pm2 + q2
        d3 = p2 + q2
        eps = np.float32(1e-12)              # the float32 the GLSL front-end rounds its literal to
        one = np.float32(1.0)
        w0 = one / np.maximum(d0, eps)
        w1 = one / np.maximum(d1, eps)
        w2 = one / np.maximum(d2, eps)
        w3 = one / np.maximum(d3, eps)
        s01 = w0 + w1
        s012 = s01 + w2
        s = s012 + w3
        a0 = c0 * w0[:, None]
        a1 = c1 * w1[:, None]
        a2 = c2 * w2[:, None]
        a3 = c3 * w3[:, None]
        t01 = a0 + a1
        t012 = t01 + a2
        t = t012 + a3
        out = (t / s[:, None]).astype(np.float32)
        oob = self._oob(u, v, wrap)
        if oob is not None:
            out = out.copy()
            out[oob] = 0.0
        return out

    def _sample_bicubic(self, img, u, v, wrap):
        """R219: bicubic B-spline, the smooth 4x4 lookup.

        Uniform cubic B-spline weights -- every weight is non-negative
        and the four sum to one exactly, so the result never overshoots
        the texel range (a Catmull-Rom would, and a light's projected
        factor must never ring negative). Texel centres and wrap
        arithmetic are _sample_bilinear's own; the GLSL mirror in
        gpu/material writes out the same sixteen fetches and the same
        weight polynomials so both devices filter with the same float
        math.
        """
        h, w = img.shape[:2]
        fx = u * w - 0.5
        fy = v * h - 0.5
        x0 = np.floor(fx).astype(np.int64)
        y0 = np.floor(fy).astype(np.int64)
        tx = (fx - x0).astype(np.float32)
        ty = (fy - y0).astype(np.float32)
        oob = self._oob(u, v, wrap)

        def _wts(t):
            t2 = t * t
            t3 = t2 * t
            return ((1.0 - 3.0 * t + 3.0 * t2 - t3) * np.float32(1 / 6),
                    (4.0 - 6.0 * t2 + 3.0 * t3) * np.float32(1 / 6),
                    (1.0 + 3.0 * t + 3.0 * t2 - 3.0 * t3)
                    * np.float32(1 / 6),
                    t3 * np.float32(1 / 6))

        wx = _wts(tx)
        wy = _wts(ty)
        xi = [self._wrap_index(x0 + k - 1, w, wrap) for k in range(4)]
        out = None
        for j in range(4):
            yj = self._wrap_index(y0 + j - 1, h, wrap)
            row = None
            for i in range(4):
                c = img[yj, xi[i]] * wx[i][:, None]
                row = c if row is None else row + c
            row = row * wy[j][:, None]
            out = row if out is None else out + row
        if oob is not None:
            out[oob] = 0.0
        return out.astype(np.float32)

    def _sample_3point(self, img, u, v, wrap):
        """Nintendo 64 3-point (triangular) filter: cheaper, visibly different."""
        h, w = img.shape[:2]
        if wrap == 'BORDER':
            u = np.clip(u, np.float32(0.0), np.float32(1.0))     # C077
            v = np.clip(v, np.float32(0.0), np.float32(1.0))
        fx = u * w - 0.5
        fy = v * h - 0.5
        x0 = np.floor(fx).astype(np.int64)
        y0 = np.floor(fy).astype(np.int64)
        tx = (fx - x0).astype(np.float32)
        ty = (fy - y0).astype(np.float32)
        x0w = self._wrap_index(x0, w, wrap)
        x1w = self._wrap_index(x0 + 1, w, wrap)
        y0w = self._wrap_index(y0, h, wrap)
        y1w = self._wrap_index(y0 + 1, h, wrap)
        c00 = img[y0w, x0w]
        c10 = img[y0w, x1w]
        c01 = img[y1w, x0w]
        c11 = img[y1w, x1w]
        if wrap == 'BORDER':
            c00, c10, c01, c11 = self._border_taps(x0, y0, w, h, c00, c10, c01, c11)
        upper = (tx + ty) > 1.0
        a = np.where(upper[:, None], c11, c00)
        s = np.where(upper[:, None], 1.0 - ty[:, None], tx[:, None])
        t = np.where(upper[:, None], 1.0 - tx[:, None], ty[:, None])
        b = np.where(upper[:, None], c01, c10)
        c = np.where(upper[:, None], c10, c01)
        out = a + (b - a) * s + (c - a) * t
        oob = self._oob(u, v, wrap)
        if oob is not None:
            out = out.copy()
            out[oob] = 0.0
        return out

    def _sample_aniso(self, u, v, duv, dvv, bias, wrap, max_aniso, frac_bits=0):
        """Hardware-style N-tap anisotropic filtering.

        The pixel's screen-x and screen-y footprints in texel units pick
        a MAJOR and a minor axis; the mip level follows the MINOR axis
        (so the texture keeps its detail along the stretch -- the whole
        point of anisotropy on a grazing floor), and `max_aniso`
        trilinear taps average along the major axis in UV space.
        Deterministic and vectorised: every pixel takes the same tap
        count, weights uniform -- the era's box approximation of EWA,
        honestly, rather than EWA itself.
        """
        tw, th = float(self.width), float(self.height)
        vx2 = (duv[:, 0] * tw) ** 2 + (dvv[:, 0] * th) ** 2
        vy2 = (duv[:, 1] * tw) ** 2 + (dvv[:, 1] * th) ** 2
        lx = np.sqrt(vx2)
        ly = np.sqrt(vy2)
        major_x = lx >= ly
        major = np.where(major_x, lx, ly)
        minor = np.maximum(np.where(major_x, ly, lx), 1e-6)
        ratio = np.clip(major / minor, 1.0, float(max(max_aniso, 1)))
        lod = (np.log2(np.maximum(major / ratio, 1e-6)) + bias) \
            .astype(np.float32)
        ax_u = np.where(major_x, duv[:, 0], duv[:, 1]).astype(np.float32)
        ax_v = np.where(major_x, dvv[:, 0], dvv[:, 1]).astype(np.float32)
        n_taps = max(int(max_aniso), 1)
        acc = np.zeros((u.shape[0], 4), np.float32)
        for k in range(n_taps):
            t = np.float32((k + 0.5) / n_taps - 0.5)
            acc += self._sample_trilinear(u + ax_u * t, v + ax_v * t,
                                          lod, wrap, frac_bits=frac_bits)
        return (acc / np.float32(n_taps)).astype(np.float32)

    def _sample_trilinear(self, u, v, lod, wrap, frac_bits=0, F=None):
        """Two levels lerped by the LOD fraction, `a + (b - a) * frac`.
        `F` (C072 BLEND) is the filter's own per-level sampler; None is
        1.89.0's bilinear -- ONE body, so BILINEAR + BLEND is Trilinear
        structurally (A12.11)."""
        mips = self.build_mips()
        lod = np.clip(np.asarray(lod, np.float32), 0.0, len(mips) - 1.0)
        l0 = np.floor(lod).astype(np.int32)
        frac = (lod - l0)[:, None]
        out = np.zeros((u.shape[0], 4), np.float32)
        for lvl in np.unique(l0):
            m = l0 == lvl
            if F is None:
                a = self._sample_bilinear(mips[int(lvl)], u[m], v[m], wrap,
                                          frac_bits=frac_bits)
                b = self._sample_bilinear(mips[min(int(lvl) + 1, len(mips) - 1)],
                                          u[m], v[m], wrap, frac_bits=frac_bits)
            else:
                a = F(mips[int(lvl)], u[m], v[m], wrap)
                b = F(mips[min(int(lvl) + 1, len(mips) - 1)], u[m], v[m], wrap)
            out[m] = a + (b - a) * frac[m]
        return out


# ------------------------------------------------------------- C013 budget
def tmem_budget(fmt):
    """The N64's TMEM bytes a format may use: 2 KB for colour-index formats
    (the TLUT lives above), 4 KB otherwise."""
    return 2048 if fmt in ('CI4', 'CI8') else 4096


def tmem_bytes(w, h, fmt, mips=False):
    """Bytes of one texture in TMEM: rows padded to 8-byte (64-bit) words;
    with `mips`, the whole level chain down to 1x1 inclusive."""
    bpt = TMEM_BPT[fmt]

    def one(ww, hh):
        return hh * ((ww * bpt + 63) // 64) * 8

    total = one(w, h)
    if mips:
        ww, hh = w, h
        while ww > 1 or hh > 1:
            ww, hh = max(1, ww // 2), max(1, hh // 2)
            total += one(ww, hh)
    return total


# ------------------------------------------------------------- C024 blocks
def _expand565(r5, g6, b5):
    """5:6:5 -> bytes by bit replication (integer)."""
    return np.stack([(r5 << 3) | (r5 >> 2), (g6 << 2) | (g6 >> 4),
                     (b5 << 3) | (b5 >> 2)], axis=-1)


def _pack565(T):
    """Integer RGB triples -> packed 5:6:5 (half-to-even rounding) and the
    5/6/5 channel values."""
    r5 = np.rint(T[..., 0] * 31 / 255).astype(np.int64)
    g6 = np.rint(T[..., 1] * 63 / 255).astype(np.int64)
    b5 = np.rint(T[..., 2] * 31 / 255).astype(np.int64)
    return (r5 << 11) | (g6 << 5) | b5


def _unpack565(v):
    return (v >> 11) & 31, (v >> 5) & 63, v & 31


def _dxt1_encode(E, A, mode):
    """The DXT1 family per 4x4 block (raster order of blocks and texels,
    i = y*4 + x): S3's min/max-luminance endpoints, the reference (DXT1),
    NV2A 16-bit (DXT1_NV2A) or CMPR 3/8-5/8 (CMPR_GC) interpolants, index
    by nearest colour (ties lowest), decode. Returns (rgb8 int (H,W,3),
    alpha float32 (H,W))."""
    H, W = E.shape[:2]
    Hp, Wp = -(-H // 4) * 4, -(-W // 4) * 4
    Ep = np.pad(E, ((0, Hp - H), (0, Wp - W), (0, 0)), mode='edge')
    Ap = np.pad(A, ((0, Hp - H), (0, Wp - W)), mode='edge')
    Hb, Wb = Hp // 4, Wp // 4
    T = Ep.reshape(Hb, 4, Wb, 4, 3).transpose(0, 2, 1, 3, 4).reshape(Hb, Wb, 16, 3).astype(np.int64)
    O = Ap.reshape(Hb, 4, Wb, 4).transpose(0, 2, 1, 3).reshape(Hb, Wb, 16)
    mode3 = ~O.all(axis=-1)
    none_opaque = ~O.any(axis=-1)
    luma = 0.299 * T[..., 0] + 0.587 * T[..., 1] + 0.114 * T[..., 2]     # float64 on the ints
    lmin = np.where(O, luma, np.inf)
    lmax = np.where(O, luma, -np.inf)
    iA = np.argmin(lmin, axis=-1)
    iB = np.argmax(lmax, axis=-1)
    tA = np.take_along_axis(T, iA[..., None, None], axis=2)[..., 0, :]
    tB = np.take_along_axis(T, iB[..., None, None], axis=2)[..., 0, :]
    vA = _pack565(tA)
    vB = _pack565(tB)
    vA = np.where(none_opaque, 0, vA)
    vB = np.where(none_opaque, 0, vB)
    c0 = np.where(mode3, np.minimum(vA, vB), np.maximum(vA, vB))
    c1 = np.where(mode3, np.maximum(vA, vB), np.minimum(vA, vB))
    q0 = _unpack565(c0)
    q1 = _unpack565(c1)
    P0 = _expand565(*q0)
    P1 = _expand565(*q1)
    if mode == 'DXT1':
        P2_4 = (2 * P0 + P1 + 1) // 3
        P3_4 = (P0 + 2 * P1 + 1) // 3
        P2_3 = (P0 + P1 + 1) // 2
    elif mode == 'DXT1_NV2A':
        P2_4 = _expand565(*[(2 * a + b) // 3 for a, b in zip(q0, q1)])
        P3_4 = _expand565(*[(a + 2 * b) // 3 for a, b in zip(q0, q1)])
        P2_3 = _expand565(*[(a + b) // 2 for a, b in zip(q0, q1)])
    else:                                                          # CMPR_GC
        P2_4 = (5 * P0 + 3 * P1) >> 3
        P3_4 = (3 * P0 + 5 * P1) >> 3
        P2_3 = (P0 + P1) >> 1
    m3 = mode3[..., None]
    P2 = np.where(m3, P2_3, P2_4)
    P3 = np.where(m3, 0, P3_4)
    P = np.stack([P0, P1, P2, P3], axis=2)                         # (Hb, Wb, 4, 3)
    d = ((T[:, :, :, None, :] - P[:, :, None, :, :]) ** 2).sum(axis=-1)   # (Hb, Wb, 16, 4)
    d = np.where((m3 & np.array([False, False, False, True]))[:, :, None, :], np.iinfo(np.int64).max, d)
    idx = np.argmin(d, axis=-1)                                    # ties -> lowest index
    idx = np.where(mode3[..., None] & ~O, 3, idx)
    dec = np.take_along_axis(P, idx[..., None], axis=2)              # (Hb,Wb,16,3)
    alpha = np.where(mode3[..., None] & (idx == 3), 0.0, 1.0)
    rgb8 = dec.reshape(Hb, Wb, 4, 4, 3).transpose(0, 2, 1, 3, 4).reshape(Hp, Wp, 3)[:H, :W]
    al = alpha.reshape(Hb, Wb, 4, 4).transpose(0, 2, 1, 3).reshape(Hp, Wp)[:H, :W]
    return rgb8.astype(np.int32), al.astype(np.float32)


def _vq_nearest_exact(X, C, chunk=2048):
    """argmin of the plain squared distance (sum over 16 dims of (x-c)^2),
    ties lowest: the law the spec names, exact in float64."""
    out = np.empty(X.shape[0], np.int64)
    for s in range(0, X.shape[0], chunk):
        x = X[s:s + chunk]
        d = ((x[:, None, :] - C[None, :, :]) ** 2).sum(axis=-1)
        out[s:s + chunk] = np.argmin(d, axis=1)
    return out


def _vq_encode(E, A8):
    """The Dreamcast's VQ: 256 codebook entries of 2x2 texels, fitted by
    Halcyon's own seeded LBG (the SDK's quantiser is unpublished) on at
    most 4096 training blocks, then every block assigned to its nearest
    ORIGINAL code and decoded from the 16-bit-quantised codebook.
    Returns (rgb8 int (H,W,3), alpha float32 (H,W))."""
    H, W = E.shape[:2]
    Hp, Wp = -(-H // 2) * 2, -(-W // 2) * 2
    px = np.concatenate([E, A8[..., None]], axis=-1)
    px = np.pad(px, ((0, Hp - H), (0, Wp - W), (0, 0)), mode='edge')
    Hb, Wb = Hp // 2, Wp // 2
    # (Hb, Wb, 2y, 2x, 4) -> 16-vectors of texels (0,0),(1,0),(0,1),(1,1)
    X = px.reshape(Hb, 2, Wb, 2, 4).transpose(0, 2, 1, 3, 4).reshape(Hb * Wb, 16).astype(np.float64)
    n = X.shape[0]
    stride = -(-n // 4096)
    train = X[::stride]
    C = train.mean(axis=0, keepdims=True)

    def lloyd(C, iters):
        for _ in range(iters):
            a = _vq_nearest_exact(train, C)
            for k in range(C.shape[0]):
                m = a == k
                if m.any():
                    C[k] = train[m].mean(axis=0)
        return C

    while C.shape[0] < 256:
        C = np.concatenate([C + 0.5, C - 0.5], axis=0)
        C = lloyd(C, 2)
    C = lloyd(C, 16)
    opaque = bool((A8 == 255).all())
    # quantise each code's four texels to the DC's 16-bit format
    Cq = C.reshape(256, 4, 4)
    cf = (Cq.astype(np.float32) / np.float32(255)).astype(np.float32)
    if opaque:
        dq = np.stack([_q(cf[..., 0], 5), _q(cf[..., 1], 6), _q(cf[..., 2], 5),
                       np.ones(cf.shape[:2], np.float32)], axis=-1)
    else:
        dq = np.stack([_q(cf[..., 0], 4), _q(cf[..., 1], 4), _q(cf[..., 2], 4),
                       _q(cf[..., 3], 4)], axis=-1)
    dq8 = np.rint(dq * 255.0).astype(np.int64)                    # exact: dq is k/255
    idx = _vq_nearest_exact(X, C)
    dec = dq8[idx].reshape(Hb, Wb, 2, 2, 4).transpose(0, 2, 1, 3, 4).reshape(Hp, Wp, 4)[:H, :W]
    rgb8 = dec[..., :3].astype(np.int32)
    alpha = (dec[..., 3].astype(np.float32) / np.float32(255)).astype(np.float32)
    return rgb8, alpha


def _box_half(img):
    h, w = img.shape[:2]
    hh, ww = max(1, h // 2), max(1, w // 2)
    if h >= 2 and w >= 2:
        return (img[0:hh * 2:2, 0:ww * 2:2] + img[1:hh * 2:2, 0:ww * 2:2] +
                img[0:hh * 2:2, 1:ww * 2:2] + img[1:hh * 2:2, 1:ww * 2:2]) * 0.25
    if h >= 2:
        return (img[0:hh * 2:2] + img[1:hh * 2:2]) * 0.5
    return (img[:, 0:ww * 2:2] + img[:, 1:ww * 2:2]) * 0.5


def compute_lod(du, dv, width, height, bias=0.0):
    """Screen-space UV derivatives -> mip level."""
    dx = np.sqrt((du[:, 0] * width) ** 2 + (du[:, 1] * height) ** 2)
    dy = np.sqrt((dv[:, 0] * width) ** 2 + (dv[:, 1] * height) ** 2)
    rho = np.maximum(np.maximum(dx, dy), 1e-6)
    return np.log2(rho) + bias


def env_sphere_uv(d):
    """Mirror-ball / sphere-map lookup (the 90s way to fake reflections)."""
    m = 2.0 * np.sqrt(np.maximum(d[:, 0] ** 2 + d[:, 1] ** 2 + (d[:, 2] + 1.0) ** 2, 1e-8))
    return d[:, 0] / m + 0.5, d[:, 1] / m + 0.5


def env_equirect_uv(d):
    u = np.arctan2(d[:, 1], -d[:, 0]) / (2.0 * np.pi) + 0.5
    v = np.arctan2(d[:, 2], np.sqrt(np.maximum(d[:, 0] ** 2 + d[:, 1] ** 2, 1e-12))) / np.pi + 0.5
    return u.astype(np.float32), v.astype(np.float32)
