"""The post chain on the GPU, resident: texture in, texture out.

Until 1.88.0 every GPU post stage began with an upload of the NumPy frame
and ended with a readback (NTSC four times over), and the CPU stages
between them ran on the array in between. Now the chain carries a
`Frame`: the picture as a pooled target on the GPU, uploaded ONCE (or
inherited from the render -- the frame the shading, the ink and the
resolve already left there, gpu/frame.py), drawn stage to stage in
ping-pong targets, and read back ONCE at the end -- or at the first
CPU-only stage that is active, by name, after which the CPU stage runs
and the chain uploads again for what follows.

Only stages listed in `stages.ENABLED` are attempted, and that list is set
by measured agreement with the CPU path rather than by intent. If a stage
is not enabled, or the GPU refuses it, the CPU function runs instead and
the reason is printed once. `chain.try_stage` is the one door every stage
knocks on; the self test and the gate test count it.
"""

import numpy as np

from ..core import dither as DI
from . import device
from .stages import CM_MODES, ENABLED, MASK_KINDS, STAGES

_WARNED = set()

#: what the last post chain did, for the console and the field test:
#: 'resident' (started from the render's own target), 'stages' (the GPU
#: stages that drew), 'readbacks' [(stage, why)] (CPU stages that forced
#: the frame down), 'uploads', 'final_read'
LAST = {}


def _warn(msg):
    if msg not in _WARNED:
        _WARNED.add(msg)
        print(f'[Halcyon GPU] {msg}')


def available(settings):
    # defense in depth: the caller's gate checks the device too, but a
    # second door that cannot open on the CPU device costs nothing
    if str(getattr(settings, 'render_device', 'CPU')).upper() != 'GPU':
        return False, 'the render device is CPU'
    if not getattr(settings, 'gpu_post', False):
        return False, 'GPU post processing is off'
    ok, why = device.probe()
    return ok, why


class Frame:
    """The picture on the GPU during the post chain.

    `target` is a pooled device.Target holding the current picture (or
    None while the picture is on the CPU); `rgb` is the CPU array when
    it is down. `up()` puts a CPU picture on the GPU (one upload),
    `down(stage, why)` reads it back by name. Stages call `draw(name,
    shader, uniforms, samplers)`: a new pooled target receives the draw,
    the previous one is freed (never the one a sampler reads), and the
    burst is a single crossing -- no readback until someone asks.
    """

    def __init__(self, rgb=None, target=None, width=None, height=None,
                 resident=False):
        self.rgb = rgb
        self.target = target
        self.width = int(width if width is not None else rgb.shape[1])
        self.height = int(height if height is not None else rgb.shape[0])
        self.resident = bool(resident)
        self.pending = []           # draws recorded, not yet submitted
        self.retired = []           # targets a recorded draw still reads
        self.stages = []
        self.readbacks = []
        self.uploads = 0
        self._source_tex = None

    @property
    def on_gpu(self):
        return self.target is not None

    def up(self):
        """The CPU picture onto the GPU (one upload). The upload IS a
        texture, so the next stage samples it and draws into a fresh
        target: no copy draw."""
        if self._source_tex is not None:
            return self._source_tex
        img = np.asarray(self.rgb, np.float32)
        if img.shape[2] == 3:
            img = np.concatenate([img, np.ones(img.shape[:2] + (1,),
                                               np.float32)], 2)
        self._source_tex = device.upload(img)
        self.uploads += 1
        return self._source_tex

    def source(self):
        """The sampler for the next stage: the current target's texture,
        or the uploaded texture when the picture just came up."""
        if self.target is not None:
            return device.target_texture(self.target)
        tex = getattr(self, '_source_tex', None)
        if tex is None:
            tex = self.up()
        return tex

    def flush(self, read=False):
        """Submit the recorded draws in one crossing; optionally read the
        current target back. The targets those draws read are returned
        to the pool only now, after the burst -- a Target made before
        this could otherwise pop one of them and alias a picture a
        recorded draw still samples."""
        out = None
        if self.pending or read:
            out = device.draw_many(self.pending, read=self.target if read
                                   else None)
            self.pending = []
        for t in self.retired:
            t.free()
        self.retired = []
        return out

    def draw(self, name, shader, uniforms, samplers, extra_binds=None):
        """Record one stage: `samplers` maps names to 'source' (this
        frame's current picture) or textures."""
        src = self.source()
        binds = {}
        for k, v in (samplers or {}).items():
            binds[k] = src if v == 'source' else v
        if extra_binds:
            binds.update(extra_binds)
        tgt = device.Target(self.width, self.height)
        self.pending.append((shader, dict(uniforms), binds, tgt, 'NONE',
                             False, None))
        prev = self.target
        self.target = tgt
        if prev is not None:
            # freed after the burst that samples it has run (flush)
            self.retired.append(prev)
        self._source_tex = None
        self.stages.append(name)

    def down(self, stage, why):
        """Read the picture back for a CPU stage, by name."""
        if self.target is None:
            return self.rgb
        got = self.flush(read=True)
        if got is None:
            got = device.read_target(self.target)
        self.target.free()
        self.target = None
        self._source_tex = None
        self.rgb = np.ascontiguousarray(got[:, :, :3], np.float32)
        self.readbacks.append((str(stage), str(why)))
        return self.rgb

    def finish(self):
        """The final readback (H, W, 3) float32 and the target's release."""
        if self.target is None:
            return self.rgb
        got = self.flush(read=True)
        if got is None:
            got = device.read_target(self.target)
        self.target.free()
        self.target = None
        self._source_tex = None
        self.rgb = np.ascontiguousarray(got[:, :, :3], np.float32)
        return self.rgb

    def release(self):
        self.pending = []
        for t in self.retired:
            try:
                t.free()
            except Exception:                                   # noqa: BLE001
                pass
        self.retired = []
        if self.target is not None:
            try:
                self.target.free()
            except Exception:                                   # noqa: BLE001
                pass
            self.target = None
        self._source_tex = None


def try_stage(name, frame, settings, uniforms, extra_binds=None):
    """Run one stage on the GPU over `frame` (a Frame). Returns the Frame,
    or None to mean "use the CPU one"."""
    if name not in ENABLED:
        return None
    ok, why = available(settings)
    if not ok:
        _warn(f'{name} on the CPU: {why}')
        return None
    shader, err = device.compile_stage(name, STAGES[name])
    if shader is None:
        _warn(f'{name} on the CPU: {err}')
        return None
    try:
        frame.draw(name, shader, uniforms, {'source': 'source'}, extra_binds)
        return frame
    except Exception as exc:                                    # noqa: BLE001
        _warn(f'{name} fell back to the CPU: {type(exc).__name__}: {exc}')
        return None


# ------------------------------------------------------------ the stages


def display(frame, st, **_k):
    return try_stage('DISPLAY', frame, st, {
        'exposure': float(st.exposure), 'brightness': float(st.brightness),
        'contrast': float(st.contrast), 'saturation': float(st.saturation),
        'gamma': max(float(st.gamma), 1e-3),
        'cm_mode': CM_MODES.get(str(st.color_management), 0)})


def lens(frame, st, **_k):
    """Validated, unwired: post.process runs lens_distortion on the CPU
    (the CPU order is CRT then lens; wiring it here would change that
    order or duplicate the warp). Kept for the self test's table."""
    if abs(float(st.lens_distortion)) < 1e-5 and \
            abs(float(st.chromatic_aberration)) < 1e-5:
        return None
    h, w = frame.height, frame.width
    return try_stage('LENS', frame, st, {
        'distortion': float(st.lens_distortion),
        'aberration': float(st.chromatic_aberration),
        'edges': 1.0 if st.lens_vignette_edges else 0.0,
        'resolution': (float(w), float(h))})


def ntsc(frame, st, frame_no=0, **_k):
    """Composite chroma bleed as the CPU does it: three blur draws, one mix.

    The CPU's triple box re-pads the frame edge before every pass, so the
    only way to match it exactly is to *run* three passes -- an intermediate
    YIQ target ping-pongs through NTSC_BLUR and the final draw reassembles.
    Dot crawl is frame-dependent and stays on the CPU; a frame using it falls
    back whole, because half a composite artefact is worse than either half.
    R250: the three blurs ping-pong on the GPU with no readback between.
    """
    if not getattr(st, 'composite', False):
        return None
    if float(getattr(st, 'composite_dot_crawl', 0.0)) > 0.0:
        # the chain records this as 'refused by name (the console says
        # why)': it has to say it
        _warn('NTSC on the CPU: dot crawl is frame-dependent and stays on '
              'the CPU, and a frame using it falls back whole')
        return None
    if 'NTSC' not in ENABLED:
        return None
    ok, why = available(st)
    if not ok:
        _warn(f'NTSC on the CPU: {why}')
        return None
    blur_sh, err = device.compile_stage('NTSC_BLUR', STAGES['NTSC_BLUR'])
    final_sh, err2 = device.compile_stage('NTSC', STAGES['NTSC'])
    if blur_sh is None or final_sh is None:
        _warn(f'NTSC on the CPU: {err or err2}')
        return None
    try:
        h, w = frame.height, frame.width
        bleed = float(st.composite_bleed)
        ri = max(int(round(w / 320.0 * 6.0 * bleed)), 1)
        rq = max(int(round(w / 320.0 * 12.0 * bleed)), 1)
        if max(ri, rq) > 96:
            _warn('NTSC on the CPU: the chroma radius exceeds the shader '
                  'loop bound at this resolution')
            return None
        # the source picture must survive the three blurs: it is read
        # again by the combine, so the blurs draw into their own pair of
        # targets and the frame's own target stays as it is
        src_tex = frame.source()
        src_target = frame.target
        frame.target = None            # the blurs must not free it
        a = device.Target(w, h)
        b = device.Target(w, h)
        cur_tex = src_tex
        draws = []
        targets = [a, b]
        for step in range(3):
            tgt = targets[step % 2]
            draws.append((blur_sh,
                          {'ri': float(ri), 'rq': float(rq), 'ry': 2.0,
                           'to_yiq': 1.0 if step == 0 else 0.0,
                           'resolution': (float(w), float(h))},
                          {'source': cur_tex}, tgt, 'NONE', False, None))
            cur_tex = device.target_texture(tgt)
        out_t = device.Target(w, h)
        draws.append((final_sh, {'ringing': float(st.composite_ringing)},
                      {'source': src_tex, 'blurred': cur_tex}, out_t,
                      'NONE', False, None))
        frame.pending.extend(draws)
        frame.retired.extend([a, b] + ([src_target] if src_target is not None
                                       else []))
        frame.target = out_t
        frame._source_tex = None
        frame.stages.append('NTSC')
        return frame
    except Exception as exc:                                    # noqa: BLE001
        _warn(f'NTSC fell back to the CPU: {type(exc).__name__}: {exc}')
        return None


def crt(frame, st, **_k):
    if not st.crt:
        return None
    if st.crt_curvature > 0.0 or st.crt_bloom > 0.0:
        return None            # those stages are not ported; keep it consistent
    h, w = frame.height, frame.width
    return try_stage('CRT', frame, st, {
        'scanlines': float(st.crt_scanlines),
        'mask_strength': float(st.crt_mask_strength),
        'mask_kind': int(MASK_KINDS.get(st.crt_mask, 0)),
        'vignette': float(st.crt_vignette),
        'resolution': (float(w), float(h))})


# ------------------------------------------------------ R250: the film


def film_refusal(st, h):
    """Why the linear-light film stages stay on the CPU this frame, or
    None when the GPU GRAIN stage (grain <= 1.2 px, no clumps, the
    flicker scalar) is exactly what the CPU would run."""
    from ..core import film as FILM
    from ..core import wear as WEAR
    if not FILM.film_on(st):
        return 'no film stage is on'
    if FILM.process_on(st):
        return 'the colour process runs on the CPU'
    if str(getattr(st, 'film_grade', 'NONE') or 'NONE').upper() != 'NONE':
        return 'the stock grade runs on the CPU'
    if float(getattr(st, 'film_softness', 0.0)) > 1e-3:
        return 'the rostrum softness runs on the CPU'
    if float(getattr(st, 'film_weave', 0.0)) > 0.0:
        return 'the gate weave runs on the CPU'
    if float(getattr(st, 'film_dust', 0.0)) > 0.0:
        return 'the dust runs on the CPU'
    if float(getattr(st, 'film_hairs', 0.0)) > 0.0:
        return 'the gate hairs run on the CPU'
    if float(getattr(st, 'film_scratches', 0.0)) > 0.0:
        return 'the scratches run on the CPU'
    if float(getattr(st, 'film_reel', 0.0)) > 0.0:
        return 'the cue marks run on the CPU'
    if WEAR.grain_on(st):
        g, size_px, clump, chroma = WEAR.grain_params(st, h)
        if size_px > 1.2:
            return (f'Grain Size {size_px:.2f} px blurs the sheets on the '
                    'CPU (the GPU grain covers grains to 1.2 px)')
        if clump > 1e-3:
            return 'Grain Clump blurs a coarse sheet on the CPU'
    return None


def grain_uniforms(st, h, w, frame_no, seed):
    """The GRAIN stage's dials, every constant rounded as the CPU rounds
    it (wear.grain_plain / _white_sheets / grain_sigma; film.flicker)."""
    from ..core import film as FILM
    from ..core import wear as WEAR
    if WEAR.grain_on(st):
        g, size_px, clump, chroma = WEAR.grain_params(st, h)
        amp = float(np.float32(g * float(WEAR._GRAIN_K) * min(size_px, 1.0)))
        ch = float(np.clip(chroma, 0.0, 1.0))
        mode = 0 if ch <= 0.0 else (1 if ch >= 1.0 else 2)
    else:
        amp, ch, mode = 0.0, 0.0, 0
    f = float(np.clip(getattr(st, 'film_flicker', 0.0), 0.0, 1.0))
    if f > 0.0:
        k = 1.0 + f * 0.12 * (2.0 * FILM._frame_noise(frame_no, seed, 21)
                              - 1.0)
    else:
        k = 1.0
    return {'resolution': (float(w), float(h)),
            'key_a': int(WEAR._sheet_key(frame_no, seed, 3)),
            'key_b': int(WEAR._sheet_key(frame_no, seed, 4)),
            'chroma_mode': int(mode),
            'mix_a': float(np.float32(np.sqrt(1.0 - ch))),
            'mix_b': float(np.float32(np.sqrt(ch))),
            'amp': amp,
            'tri_scale': float(np.float32(np.sqrt(6.0) / 4096.0)),
            'tri_off': float(np.float32(np.sqrt(6.0))),
            'u_scale': float(np.float32(np.sqrt(3.0) * 2.0 / 256.0)),
            'u_off': float(np.float32(np.sqrt(3.0))),
            'a_coef': float(np.float32(-1.0 / 2.4)),
            'ln10': float(np.float32(np.log(10.0))),
            'flicker': float(np.float32(k))}


def film(frame, st, frame_no=0, seed=0, **_k):
    """The linear-light film stages: GRAIN (+ flicker) on the GPU when
    they are all the CPU would run; None (with the reason recorded by
    the caller) otherwise."""
    why = film_refusal(st, frame.height)
    if why:
        return None
    return try_stage('GRAIN', frame, st,
                     grain_uniforms(st, frame.height, frame.width,
                                    int(frame_no), int(seed)))


# ----------------------------------------------------- R250: the depth


def quant_refusal(st):
    """Why reduce_depth stays on the CPU this frame, or None when it is
    palette.snap_bits (bit depth, no dither, no palette)."""
    from ..core.post import DEPTH_BITS
    depth = str(st.color_depth)
    kind = str(st.dither)
    # R251: the era roads first -- cells > scanline > CRY16 > YJK > EHB
    handled, why = CP.era_refusal(st)
    if handled:
        return why
    if depth in ('HAM6', 'HAM8'):
        return 'HAM encoding runs on the CPU'
    if depth == '1':
        return '1-bit thresholding runs on the CPU'
    if CP.palette_road(st):
        return CP.palette_refusal(st)     # R251 P0: the PALETTE stage
    if depth not in DEPTH_BITS:
        return f"colour depth '{depth}' runs on the CPU"
    if kind in DI.ORDERED:
        return None                        # R251 P1: the ORDERED stage
    if kind != 'NONE':
        return f'the {kind} dither runs on the CPU'
    return None


_QUANT_LUTS = {}


def quant_lut(bits):
    """The per-channel level values as the CPU computes them
    (k / levels in float32), one texel per index: the GPU fetches the
    CPU's own quotient bits instead of dividing."""
    key = tuple(int(b) for b in bits)
    hit = _QUANT_LUTS.get(key)
    if hit is None:
        n = max((1 << b) for b in key)
        lut = np.zeros((1, n, 4), np.float32)
        for ch, b in enumerate(key):
            levels = float((1 << b) - 1)
            idx = np.arange(1 << b, dtype=np.float32)
            lut[0, :1 << b, ch] = idx / levels
        lut[0, :, 3] = 1.0
        hit = _QUANT_LUTS[key] = lut
    return hit


def quant(frame, st, **_k):
    """Framebuffer quantisation without a dither: palette.snap_bits."""
    if quant_refusal(st):
        return None
    if CP.era_road(st):
        return CP.quant_era(frame, st, **_k)
    if CP.palette_road(st):
        return CP.palette(frame, st, **_k)
    if CP.ordered_road(st):
        return CP.ordered(frame, st, **_k)
    from ..core.post import DEPTH_BITS
    bits = DEPTH_BITS.get(str(st.color_depth), (8, 8, 8))
    levels = tuple(float((1 << b) - 1) for b in bits)
    try:
        lut_tex = device.upload_cached(('quant_lut',) + tuple(bits),
                                       lambda: quant_lut(bits))
    except Exception as exc:                                    # noqa: BLE001
        _warn(f'QUANT on the CPU: {exc}')
        return None
    h, w = frame.height, frame.width
    return try_stage('QUANT', frame, st,
                     {'resolution': (float(w), float(h)), 'levels': levels},
                     extra_binds={'lut': lut_tex})


# R251: the post-signal pack's stages (gpu/chain_signal.py); post.process finds them by name
from .chain_signal import (vi_filter, copy_filter, crtc_blend, video_filter,  # noqa: E402,F401
                           crtc_state)
# R251 C001 (raster pack): the N64 VI's coverage blend and divot (gpu/chain_vi.py)
from .chain_vi import n64vi, n64vi_refusal  # noqa: E402,F401
# R251 SIG-2: the tape, the cable, the PAL receiver, the chroma siting (gpu/chain_tape.py)
from .chain_tape import (chroma_site, tape, cable_chroma, pal_decode,  # noqa: E402,F401
                         cable_rf)

# R251 SIG-3: the codec / optical-printer stages (gpu/chain_codec.py)
from .chain_codec import matte_glow, mpeg1, smacker  # noqa: E402,F401


from . import chain_palette as CP  # noqa: E402  R251: imported last -- chain_palette reads this module lazily
# R251 (PAL-2): the doors post._gpu_stage's getattr(chain, name) finds
legalise = CP.legalise        # C093 Video Color Check
superblack = CP.superblack    # C092 Super Black
