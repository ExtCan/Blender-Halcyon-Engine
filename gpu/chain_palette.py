"""R251: the era colour roads of the resident post chain (post-palette pack).

Chain functions in the map's shape `def <name>(frame, st, frame_no=0,
seed=0, **_k)` + `<name>_refusal(st)`, reached through `chain.quant` /
`chain.quant_refusal` (the existing 'quant' door of post.process): the
palette snap (P0), the ordered dither to bits (P1), Extra Half-Brite
(C054), CRY16 (C011) and YJK (C059). Every table a stage reads is built
on the CPU, cached by its bytes and uploaded once (`device.upload_cached`);
every uniform is pre-rounded with `float(np.float32(x))`; every int
uniform is a Python `int`.

IMPORT DIRECTION: `chain.py` imports this module at its BOTTOM (after
`quant` closes) and this module reads `chain` LAZILY inside each function
(`from . import chain as CH` at call time) -- a half-initialised `chain`
is never touched at import, and the self test's `chain.ENABLED = ...` lift
reaches the gates below.

GATE ORDER for a stage that needs the frame's own pixels for its tables
(the adaptive palette under Lock Palette, the Extra Half-Brite 32): the
three try_stage checks -- `name in CH.ENABLED`, `CH.available(st)`,
`device.compile_stage` -- run FIRST through `gate()`, and only then the
peek / `frame.down` / fit / `draw()`. A refusal never costs a readback.
"""

import numpy as np

from . import device
from .stages_palette import STAGES
from ..core import dither as DI
from ..core import palette as PA
from ..core import palette_era as PE

#: the pixels are never read for a fixed palette mode: post._palette_for
#: takes an array for its signature only
_DUMMY = np.zeros((1, 1, 3), np.float32)


# ------------------------------------------------------------- the doors


def gate(names, st):
    """The try_stage checks for every stage of `names` (a str or a list),
    BEFORE any readback: the compiled shader (or the list of them), or
    None after try_stage's own warning texts."""
    from . import chain as CH
    one = isinstance(names, str)
    names = [names] if one else list(names)
    shaders = []
    for name in names:
        if name not in CH.ENABLED:
            return None
        ok, why = CH.available(st)
        if not ok:
            CH._warn(f'{name} on the CPU: {why}')
            return None
        shader, err = device.compile_stage(name, STAGES[name])
        if shader is None:
            CH._warn(f'{name} on the CPU: {err}')
            return None
        shaders.append(shader)
    return shaders[0] if one else shaders


def draw(name, frame, shader, uniforms, extra_binds=None):
    """The try_stage tail: record the draw on the frame, or None (the CPU
    function runs) with the reason printed once."""
    from . import chain as CH
    try:
        frame.draw(name, shader, uniforms, {'source': 'source'}, extra_binds)
        return frame
    except Exception as exc:                                    # noqa: BLE001
        CH._warn(f'{name} fell back to the CPU: {type(exc).__name__}: {exc}')
        return None


# ----------------------------------------------------------- the tables


def pal_image(pal):
    """A palette as a (1, N, 4) float32 image: rgb the palette's own
    float32 values, alpha 1."""
    p = np.asarray(pal, np.float32).reshape(-1, 3)
    img = np.ones((1, p.shape[0], 4), np.float32)
    img[0, :, :3] = p
    return img


def tile_image(m):
    """An ordered threshold matrix as an (n, n, 4) image, .r = the CPU's
    float32 matrix (row 0 = the picture's bottom row, as np.tile lays it
    from row 0 and the upload is bottom row first)."""
    m = np.asarray(m, np.float32)
    img = np.zeros(m.shape + (4,), np.float32)
    img[:, :, 0] = m
    img[:, :, 3] = 1.0
    return img


_ZERO_TILE = np.zeros((1, 1, 4), np.float32)


def _tile_texture(kind):
    """The tile texture for `kind` (a DI.ORDERED key) and its mask; the
    cached 1 x 1 zero image and mask 0 under no dither (every declared
    sampler must be bound)."""
    m = DI.ORDERED.get(str(kind))
    if m is None:
        return device.upload_cached(('palette_tile', 'NONE'),
                                    lambda: _ZERO_TILE), 0
    return device.upload_cached(('palette_tile', str(kind)),
                                lambda: tile_image(m)), int(m.shape[0] - 1)


def _lut255():
    from . import chain as CH
    return device.upload_cached(('quant_lut', 8, 8, 8),
                                lambda: CH.quant_lut((8, 8, 8)))


# ------------------------------------------------------ P0: the palette


def palette_road(st):
    """post.reduce_depth's palette-road condition (P:348-349)."""
    depth = str(st.color_depth)
    return depth in ('8', '4') or str(st.palette_mode) != 'ADAPTIVE'


def palette_refusal(st):
    """Why the palette road stays on the CPU this frame, or None when the
    PALETTE stage is exactly what the CPU would run."""
    kind = str(st.dither)
    if kind in DI.KERNELS:
        return f'the {kind} dither runs on the CPU'
    if kind == 'NOISE':
        return 'the NOISE dither runs on the CPU'
    mode = str(st.palette_mode)
    if mode == 'CUSTOM' and not getattr(st, 'palette_colors', None):
        return ('the Custom palette has no image: it is built from this '
                'frame on the CPU')
    if mode == 'ADAPTIVE' and not getattr(st, 'palette_lock', True):
        return ('the adaptive palette is rebuilt from every frame (Lock '
                'Palette is off)')
    return None


def palette_size(st):
    depth = str(st.color_depth)
    size = int(st.palette_size)
    if depth == '4':
        size = min(size, 16)
    elif depth == '8':
        size = min(size, 256)
    return size


def _ordered_uniforms(frame, st, pal):
    """The dither transports shared by PALETTE and EHB."""
    kind = str(st.dither)
    tile, mask = _tile_texture(kind)
    h, w = frame.height, frame.width
    on = 1 if kind in DI.ORDERED else 0
    return ({'resolution': (float(w), float(h)),
             'strength': float(np.float32(float(st.dither_strength))),
             'spacing': float(np.float32(DI._palette_spacing(pal))),
             'tile_mask': int(mask), 'dither_on': int(on)},
            tile)


def palette(frame, st, frame_no=0, seed=0, **_k):
    """The palette snap on the GPU: the CPU's own inverse colormap and
    palette as textures. Gate first; then the table (peeked from the lock
    cache, or built from ONE readback and cached); then the draw."""
    from ..core import post as PO
    shader = gate('PALETTE', st)
    if shader is None:
        return None
    size = palette_size(st)
    mode = str(st.palette_mode)
    if mode != 'ADAPTIVE' or (mode == 'CUSTOM'
                              and getattr(st, 'palette_colors', None)):
        pal = PO._palette_for(st, size, _DUMMY, seed)
    else:
        key = ('ADAPTIVE', int(size), str(st.palette_method), int(seed))
        peeked = PE.cached_peek(key)
        if peeked is not None:
            pal = PE.snap_registers(peeked, st)
        else:
            rgb = frame.down('palette', 'the adaptive palette is built from '
                             'this frame once (Lock Palette)')
            pal = PO._palette_for(st, size, rgb, seed)
    pal = np.ascontiguousarray(np.asarray(pal, np.float32).reshape(-1, 3))
    key_b = pal.tobytes()
    try:
        icm_tex = device.upload_cached(
            ('icm', key_b),
            lambda: PA.icm_index_image(PA.get_inverse_colormap(pal)))
        pal_tex = device.upload_cached(('pal', key_b), lambda: pal_image(pal))
        uniforms, tile = _ordered_uniforms(frame, st, pal)
    except Exception as exc:                                    # noqa: BLE001
        from . import chain as CH
        CH._warn(f'PALETTE on the CPU: {exc}')
        return None
    return draw('PALETTE', frame, shader, uniforms,
                extra_binds={'icm': icm_tex, 'pal': pal_tex, 'tile': tile})


# ------------------------------------------- P1: ordered dither to bits


def ordered_road(st):
    """post.reduce_depth's DI.ordered_bits condition: the bit-depth road
    with an ordered dither."""
    from ..core.post import DEPTH_BITS
    depth = str(st.color_depth)
    return (depth in DEPTH_BITS and depth not in ('HAM6', 'HAM8', '1')
            and not palette_road(st) and str(st.dither) in DI.ORDERED)


def ordered(frame, st, frame_no=0, seed=0, **_k):
    """The ordered dither to a bit depth on the GPU: the tile texture and
    the CPU's own level table (chain.quant_lut). No cold readback."""
    from . import chain as CH
    from ..core.post import DEPTH_BITS
    bits = DEPTH_BITS.get(str(st.color_depth), (8, 8, 8))
    kind = str(st.dither)
    try:
        tile, mask = _tile_texture(kind)
        lut_tex = device.upload_cached(('quant_lut',) + tuple(bits),
                                       lambda: CH.quant_lut(bits))
    except Exception as exc:                                    # noqa: BLE001
        CH._warn(f'ORDERED on the CPU: {exc}')
        return None
    h, w = frame.height, frame.width
    uniforms = {'resolution': (float(w), float(h)),
                'levels': tuple(float((1 << b) - 1) for b in bits),
                'strength': float(np.float32(float(st.dither_strength))),
                'tile_mask': int(mask)}
    return CH.try_stage('ORDERED', frame, st, uniforms,
                        extra_binds={'tile': tile, 'lut': lut_tex})


# ------------------------------------------------ C054: Extra Half-Brite


def ehb_refusal(st):
    kind = str(st.dither)
    if kind in DI.KERNELS:
        return f'the {kind} dither runs on the CPU'
    if kind == 'NOISE':
        return 'the NOISE dither runs on the CPU'
    if not getattr(st, 'palette_lock', True):
        return ('the Extra Half-Brite registers are refitted from every '
                'frame (Lock Palette is off)')
    return None


def ehb(frame, st, frame_no=0, seed=0, **_k):
    """Extra Half-Brite on the GPU: the 64-way argmin table and the 64
    registers as textures. Gate first; the 32 peeked from the lock cache or
    fitted from ONE readback; then the draw."""
    shader = gate('EHB', st)
    if shader is None:
        return None
    key = PE.ehb_key(st, seed)
    P32 = PE.cached_peek(key)
    if P32 is None:
        rgb = frame.down('palette', 'the Extra Half-Brite registers are '
                         'fitted from this frame once (Lock Palette)')
        P32 = PE.ehb_fit(rgb, st, seed)
    _q32, q64, P64 = PE.ehb_registers(P32)
    q64 = np.ascontiguousarray(q64)
    try:
        lut_tex = device.upload_cached(('ehb_lut', q64.tobytes()),
                                       lambda: PE.ehb_lut_image(q64))
        pal_tex = device.upload_cached(('pal', P64.tobytes()),
                                       lambda: pal_image(P64))
        uniforms, tile = _ordered_uniforms(frame, st, P64)
    except Exception as exc:                                    # noqa: BLE001
        from . import chain as CH
        CH._warn(f'EHB on the CPU: {exc}')
        return None
    return draw('EHB', frame, shader, uniforms,
                extra_binds={'ehb_lut': lut_tex, 'pal': pal_tex,
                             'tile': tile})


# ---------------------------------------------------------- C011: CRY16


def cry16_refusal(st):
    """None always: the road is integer end to end (kept for the shape
    and the self test's table)."""
    return None


def cry16(frame, st, frame_no=0, seed=0, **_k):
    from . import chain as CH
    try:
        imgs = PE.cry_images()
        binds = {k: device.upload_cached((k,), (lambda _v=v: _v))
                 for k, v in imgs.items()}
        binds['lut255'] = _lut255()
    except Exception as exc:                                    # noqa: BLE001
        CH._warn(f'CRY16 on the CPU: {exc}')
        return None
    h, w = frame.height, frame.width
    return CH.try_stage('CRY16', frame, st,
                        {'resolution': (float(w), float(h))},
                        extra_binds=binds)


# ------------------------------------------------------------ C059: YJK


def yjk_refusal(st):
    """None always: the road is integer end to end."""
    return None


def yjk(frame, st, frame_no=0, seed=0, **_k):
    from . import chain as CH
    try:
        lut = _lut255()
    except Exception as exc:                                    # noqa: BLE001
        CH._warn(f'YJK on the CPU: {exc}')
        return None
    h, w = frame.height, frame.width
    return CH.try_stage('YJK', frame, st,
                        {'resolution': (float(w), float(h))},
                        extra_binds={'lut255': lut})


# ------------------------------------------------------ the era routing

#: era owner -> (chain function, refusal)
_ERA = {'CRY16': (cry16, cry16_refusal), 'YJK': (yjk, yjk_refusal),
        'EHB': (ehb, ehb_refusal)}


def era_refusal(st):
    """The chain-side twin of palette_era.reduce_depth_era's precedence:
    (handled, why) -- handled False when no era mode is selected (the
    1.89.0 roads decide); handled True with why None for an era GPU stage,
    or a string naming why that era road runs on the CPU this frame."""
    owner = PE.era_mode(st)
    if owner is None:
        return False, None
    return True, _ERA[owner][1](st)


def era_road(st):
    """True exactly when era_refusal is (True, None)."""
    handled, why = era_refusal(st)
    return bool(handled and why is None)


def quant_era(frame, st, **_k):
    owner = PE.era_mode(st)
    if owner is None:
        return None
    return _ERA[owner][0](frame, st, **_k)


# ======================================================================
# ---- wave 2 (PAL-2): the two video-out stages between DISPLAY and the
# halftone (post.process calls them by name through gpu/chain.py's
# `legalise` / `superblack` aliases), then attribute cells and the
# per-scanline refusal
# ======================================================================

# ------------------------------------------- C093: Video Color Check


def legalise_refusal(st):
    """None always: every mode draws (kept for the shape)."""
    return None


def legalise(frame, st, frame_no=0, seed=0, **_k):
    """The composite-envelope legaliser on the GPU: the CPU's own two
    float32 limits as uniforms. No table, no readback."""
    from . import chain as CH
    mode = PE.LEGAL_MODES.get(str(getattr(st, 'video_color_check', 'NONE')))
    # a future item must never draw the identity silently; post.process
    # gates NONE
    assert mode in (1, 2, 3), mode
    hi, lo = PE.legal_limits(st)
    h, w = frame.height, frame.width
    return CH.try_stage('LEGALISE', frame, st,
                        {'resolution': (float(w), float(h)),
                         'mode': int(mode),
                         'hi': float(hi), 'lo': float(lo)})


# ------------------------------------------------- C092: Super Black


def superblack_refusal(st):
    """None always: the no-coverage case is a named SKIP on both roads
    (post.process), not a refusal."""
    return None


def superblack(frame, st, frame_no=0, seed=0, **_k):
    """Super Black on the GPU: the coverage plane packed 64 pixels a texel
    and uploaded for this frame (never cached: it is the frame's own),
    the floor the CPU's float32 quotient."""
    from . import chain as CH
    cov = getattr(frame, 'coverage', None)
    if cov is None:
        return None                      # never reached: post.process gates
    cov = np.asarray(cov, bool)
    if cov.shape != (frame.height, frame.width):
        CH._warn('SUPERBLACK on the CPU: the coverage plane is not the '
                 "frame's size")
        return None
    if 'SUPERBLACK' not in CH.ENABLED:
        return None                      # before the upload is spent
    try:
        tex = device.upload(PE.pack_coverage(cov))
    except Exception as exc:                                    # noqa: BLE001
        CH._warn(f'SUPERBLACK on the CPU: {exc}')
        return None
    h, w = frame.height, frame.width
    return CH.try_stage('SUPERBLACK', frame, st,
                        {'resolution': (float(w), float(h)),
                         'threshold': float(PE.super_black_threshold(st))},
                        extra_binds={'coverage': tex})


# ------------------------------------------------ C050: attribute cells


def cells_refusal(st):
    """Why the attribute cells stay on the CPU this frame, or None."""
    mode = str(getattr(st, 'attribute_cells', 'NONE'))
    if mode == 'C64_MULTI' and not getattr(st, 'palette_lock', True):
        return ('the C64 multicolour background is recounted from every '
                'frame (Lock Palette is off)')
    kind = str(getattr(st, 'dither', 'NONE'))
    if kind in DI.ORDERED and PE.cells_dither(st) is None:
        return f'the {kind} pattern is not a square tile: it runs on the CPU'
    return None


def cells(frame, st, frame_no=0, seed=0, **_k):
    """Attribute cells on the GPU, two passes orchestrated as chain.ntsc
    is: CELLS_FIT into a cell-sized target (one texel per cell: the chosen
    set), CELLS_SNAP at the frame's size. Gate BOTH stages first; only
    then the C64 multicolour background (peeked from the lock cache, or
    counted from ONE readback and cached); then the two draws."""
    from . import chain as CH
    mode = str(st.attribute_cells)
    shaders = gate(('CELLS_FIT', 'CELLS_SNAP'), st)
    if shaders is None:
        return None
    fit_sh, snap_sh = shaders
    bg = 0
    if mode == 'C64_MULTI':
        bg = PE.cached_peek(PE.cells_bg_key(seed))
        if bg is None:
            rgb = frame.down('cells', 'the multicolour background is counted '
                             'from this frame once (Lock Palette)')
            bg = PE.cells_bg(rgb, st, seed)
        bg = int(bg)
    src_target = None
    moved = False
    try:
        mach, set_idx = PE.cells_sets(mode, bg)
        sets_tex = device.upload_cached(
            ('cells_sets', mode, bg), lambda: PE.cells_sets_image(mode, bg))
        kind = PE.cells_dither(st)
        tile, mask = _tile_texture(kind if kind is not None else 'NONE')
        lut = _lut255()
        cw, ch = PE.CELL_SIZE[mode]
        w, h = int(frame.width), int(frame.height)
        cells_x, cells_y = -(-w // cw), -(-h // ch)
        dith = {'strength': float(np.float32(float(st.dither_strength))),
                'spacing': float(np.float32(PE.cells_spacing(mode))),
                'tile_mask': int(mask),
                'dither_on': int(1 if kind is not None else 0)}
        # the source picture is read by BOTH passes: the fit draws into
        # its own cell-sized target and the frame's target stays as it is
        src_tex = frame.source()
        src_target = frame.target
        frame.target = None
        moved = True
        cell_t = device.Target(cells_x, cells_y)
        out_t = device.Target(w, h)
        fit_u = dict(dith, resolution=(float(cells_x), float(cells_y)),
                     frame_size=(float(w), float(h)),
                     cell_w=int(cw), cell_h=int(ch),
                     n_sets=int(set_idx.shape[0]),
                     n_colors=int(set_idx.shape[1]))
        snap_u = dict(dith, resolution=(float(w), float(h)),
                      cell_shift_x=int(cw.bit_length() - 1),
                      cell_shift_y=int(ch.bit_length() - 1),
                      n_colors=int(set_idx.shape[1]))
        draws = [(fit_sh, fit_u,
                  {'source': src_tex, 'sets': sets_tex, 'tile': tile},
                  cell_t, 'NONE', False, None),
                 (snap_sh, snap_u,
                  {'source': src_tex,
                   'cells': device.target_texture(cell_t),
                   'sets': sets_tex, 'tile': tile, 'lut255': lut},
                  out_t, 'NONE', False, None)]
        frame.pending.extend(draws)
        frame.retired.extend([cell_t] + ([src_target]
                                         if src_target is not None else []))
        frame.target = out_t
        frame._source_tex = None
        frame.stages.append('CELLS')
        return frame
    except Exception as exc:                                    # noqa: BLE001
        if moved and frame.target is None:
            frame.target = src_target       # the picture is still there
        CH._warn(f'CELLS fell back to the CPU: {type(exc).__name__}: {exc}')
        return None


# --------------------------------------- C060: per-scanline palettes


def scanline_refusal(st):
    """Always a string: the per-scanline palette is CPU only, by name."""
    mode = str(getattr(st, 'scanline_palette', 'NONE'))
    return (f'the {mode} per-scanline palette is fitted row by row on the '
            'CPU (a median cut per line'
            + (', then a sequential HAM encode along it'
               if mode == 'SHAM' else '') + ')')


# the era routing, extended: cells > scanline > CRY16 > YJK > EHB
_ERA['CELLS'] = (cells, cells_refusal)
_ERA['SCANLINE'] = (None, scanline_refusal)
