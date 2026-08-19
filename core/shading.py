"""Reflectance models.

Every model here is the actual published formulation, evaluated per fragment on
flat numpy arrays. Gouraud and Flat are *interpolation* rates rather than
reflectance models -- they are handled by the renderer's shading-rate machinery
(vertex-rate and face-rate shading respectively) and reuse whichever reflectance
model the material asks for, which is historically correct.

Each model returns (diffuse_weight, specular_weight):
  diffuse_weight  (N,)  or (N,3)
  specular_weight (N,)  or (N,3)
both already including the N.L cosine term where the model calls for it.
"""

import numpy as np

from . import mathx as M

EPS = 1e-6

DIFFUSE_MODELS = ('LAMBERT', 'OREN_NAYAR', 'MINNAERT', 'TOON_DIFFUSE', 'FUJII')
SPECULAR_MODELS = ('PHONG', 'BLINN_PHONG', 'BLINN', 'COOK_TORRANCE', 'WARD',
                   'ANISOTROPIC', 'TOON_SPEC', 'METAL', 'STRAUSS', 'NONE')

MODEL_ITEMS = (
    ('LAMBERT', "Lambert",
     "Pure diffuse, no highlight at all. Brightness depends only on the angle "
     "to the light, so surfaces read as chalk or matte paint. The oldest model "
     "there is (1760) and the default for anything that should not shine"),
    ('GOURAUD', "Gouraud",
     "A shading RATE, not a reflectance model: lighting is evaluated once per "
     "vertex and the colour interpolated across the triangle (1971). Gives the "
     "banding and the sliding highlights of console-era hardware. Coarse "
     "geometry shows it most"),
    ('FLAT', "Flat / Faceted",
     "A shading RATE: one lighting evaluation for the whole polygon, so every "
     "face is a single flat colour. The look of very early hardware and of "
     "un-smoothed low-polygon models"),
    ('PHONG', "Phong",
     "Interpolated normals with a (R.V)^n highlight (1975). The workhorse of "
     "1990s software -- Infini-D, 3D Studio, LightWave. A tight round "
     "highlight that blows out to white readily"),
    ('BLINN_PHONG', "Blinn-Phong",
     "Phong's highlight computed from the half-vector instead (1977). Broader "
     "and softer than Phong at the same Glossiness, and better behaved at "
     "grazing angles. What most hardware actually implemented"),
    ('BLINN', "Blinn (microfacet)",
     "Torrance-Sparrow microfacets with Fresnel, as 3D Studio MAX shipped it. "
     "The highlight brightens toward the edges of a surface, which reads as "
     "polished plastic or coated metal"),
    ('COOK_TORRANCE', "Cook-Torrance",
     "Beckmann microfacet distribution with geometric masking (1982). The "
     "physically-grounded option: Roughness drives it rather than Glossiness, "
     "and metals keep their colour in the highlight"),
    ('OREN_NAYAR', "Oren-Nayar",
     "Rough diffuse (1994). Surfaces stay bright toward their edges instead of "
     "falling off, which is what makes clay, plaster, dust and unglazed "
     "ceramic look right. Driven by Roughness"),
    ('MINNAERT', "Minnaert",
     "Diffuse with darkening or limb-brightening controlled by Roughness "
     "(1941). Originally for the Moon; useful for velvet, dusty surfaces and "
     "anything with a soft rim"),
    ('WARD', "Ward",
     "Anisotropic Gaussian on the slope distribution (1992). A physically "
     "grounded stretched highlight -- the model to reach for on hair, satin "
     "and machined metal when the shape of the streak matters"),
    ('ANISOTROPIC', "Anisotropic",
     "3D Studio's elliptical highlight: a Blinn lobe given two exponents so it "
     "stretches. Brushed metal and vinyl records. With Anisotropy at 0 it is "
     "Blinn-Phong, which is exactly what the original did"),
    ('METAL', "Metal",
     "The period metal shader: the highlight takes the diffuse colour instead "
     "of the light's, so gold stays gold in its reflection. Diffuse is "
     "suppressed. Gives chrome and brass without needing reflections"),
    ('STRAUSS', "Strauss",
     "Parameterised by metalness and glossiness rather than by lobes (1990). "
     "An early attempt at the controls PBR later settled on, and the easiest "
     "of the metal models to dial in"),
    ('MULTI_LAYER', "Multi-Layer",
     "Two independent specular lobes: a tight one over a broad one. Car paint, "
     "lacquer, and anything with a clear coat over a base"),
    ('TOON', "Toon",
     "Diffuse quantised into bands with a hard-edged highlight. Toon Size sets "
     "where the terminator falls and Toon Smooth how sharp it is. Cel "
     "animation looks"),
    ('TRANSLUCENT', "Translucent",
     "Lambert plus a back-side lobe, so light coming from behind shows "
     "through. Paper, leaves, lampshades, thin fabric. Driven by Translucency, "
     "and identical to Lambert while that is 0"),
    ('CONSTANT', "Constant / Shadeless",
     "Unlit: emits its Diffuse Colour flat, ignoring every light in the scene. "
     "For skyboxes, UI elements, self-lit panels and anything that must not "
     "receive shading"),
    ('WIREFRAME', "Wireframe",
     "Draws the triangle edges only and leaves the rest of the surface "
     "see-through. Width comes from the material's Wire Size"),
    ('BI_COOKTORR', "CookTorr (Blender Internal)",
     "Blender Internal's default highlight, transcribed from 2.79: a "
     "half-vector lobe raised to Glossiness (Hardness), divided by "
     "(0.1 + N.V) so it brightens toward grazing view. The renderer "
     "classic .blend files were lit for; legacy imports use it"),
    ('BI_PHONG', "Phong (Blender Internal)",
     "Blender Internal's Phong, transcribed from 2.79 -- which was "
     "always the HALF-VECTOR lobe pow(N.H, Hardness), not the "
     "reflection-vector Phong of the textbooks. Legacy imports use it "
     "for materials that chose Phong"),
    ('BI_BLINN', "Blinn (Blender Internal)",
     "Blender Internal's Blinn, transcribed from 2.79: Torrance-"
     "Sparrow geometry with BI's own refraction-index Fresnel and a "
     "Gaussian half-angle lobe whose width comes from Hardness. IOR "
     "is BI's Refr slider. Legacy imports use it for Blinn materials"),
)


class Surface:
    """Per-fragment material parameters, all broadcast to (N,) or (N,3)."""

    __slots__ = ('n', 'diffuse', 'specular', 'glossiness', 'roughness', 'metallic',
                 'ior', 'anisotropy', 'aniso_rot', 'soften', 'ambient', 'emission',
                 'opacity', 'diffuse_level', 'specular_level', 'translucency',
                 'toon_size', 'toon_smooth', 'toon_steps', 'reflect', 'model',
                 'tangent', 'bitangent', 'backfacing',
                 'fresnel', 'fresnel_power', 'fresnel_color',
                 'rim', 'rim_power', 'rim_color',
                 'matcap', 'matcap_blend', 'reflect_color',
                 'edge_opacity', 'backface_color', 'backface_mix',
                 'sheen', 'sheen_color', 'sheen_roughness', 'refraction',
                 # the BI material node's own controls: the specular
                 # Toon pair, the diffuse-Fresnel pair, WardIso's Slope
                 'toon_size2', 'toon_smooth2', 'bi_fresnel',
                 'bi_fresnel_fac', 'bi_slope',
                 # the BI panel round: transparency Fresnel/Blend and
                 # spectra, mirror Fresnel/Blend, the ray-transparency
                 # IOR and Filter, Cubic and Tangent shading, the
                 # shadow flags, the mist gate -- and `bi`, one python
                 # object per batch carrying the non-numeric material
                 # extras (ramp specs, light group)
                 'bi_transp_fresnel', 'bi_transp_blend', 'bi_spectra',
                 'bi_mir_fresnel', 'bi_mir_blend', 'ray_ior',
                 'bi_ray_filter', 'bi_cubic', 'bi_tangent',
                 'shadow_receive', 'cast_only', 'shadows_only',
                 'use_mist', 'bi')

    def __init__(self, n):
        self.n = n
        f32 = np.float32
        one = np.ones(n, f32)
        self.diffuse = np.full((n, 3), 0.8, f32)
        self.specular = np.ones((n, 3), f32)
        self.glossiness = one * 25.0
        self.roughness = one * 0.3
        self.metallic = np.zeros(n, f32)
        self.ior = one * 1.45
        self.anisotropy = np.zeros(n, f32)
        self.aniso_rot = np.zeros(n, f32)
        self.soften = np.zeros(n, f32)
        self.ambient = one.copy()
        self.emission = np.zeros((n, 3), f32)
        self.opacity = one.copy()
        self.diffuse_level = one.copy()
        self.specular_level = one * 0.5
        self.translucency = np.zeros(n, f32)
        self.toon_size = one * 0.5
        self.toon_smooth = one * 0.05
        self.toon_steps = one * 2.0
        self.toon_size2 = one * 0.5      # the BI node's spec Toon pair
        self.toon_smooth2 = one * 0.1
        self.bi_fresnel = one * 0.1      # BI diffuse-Fresnel grad/fac
        self.bi_fresnel_fac = one * 0.5
        self.bi_slope = one * 0.1        # BI WardIso Slope (rms)
        # ---- the BI panel round's fields; every default reproduces the
        # engine's behaviour before the field existed
        self.bi_transp_fresnel = np.zeros(n, f32)   # 0 = plain alpha
        self.bi_transp_blend = one * 1.25
        self.bi_spectra = np.zeros(n, f32)
        self.bi_mir_fresnel = np.zeros(n, f32)      # 0 = flat mirror
        self.bi_mir_blend = one * 1.25
        self.ray_ior = one * 1.45        # refraction bend (master: = ior)
        self.bi_ray_filter = one.copy()  # 1 = full diffuse tint (master)
        self.bi_cubic = np.zeros(n, f32)
        self.bi_tangent = np.zeros(n, f32)
        self.shadow_receive = one.copy()
        self.cast_only = np.zeros(n, f32)
        self.shadows_only = np.zeros(n, f32)
        self.use_mist = one.copy()
        self.bi = None                   # per-batch material extras
        self.reflect = np.zeros(n, f32)
        self.tangent = None
        self.bitangent = None
        self.backfacing = np.zeros(n, bool)
        # artistic terms applied after the reflectance model, so they behave the
        # same whichever one is chosen
        self.fresnel = np.zeros(n, np.float32)
        self.fresnel_power = np.full(n, 3.0, np.float32)
        self.fresnel_color = np.ones((n, 3), np.float32)
        self.rim = np.zeros(n, np.float32)
        self.rim_power = np.full(n, 3.0, np.float32)
        self.rim_color = np.ones((n, 3), np.float32)
        self.matcap = np.zeros((n, 3), np.float32)
        self.matcap_blend = np.zeros(n, np.float32)
        self.reflect_color = np.ones((n, 3), np.float32)
        self.edge_opacity = np.ones(n, np.float32)
        self.backface_color = np.zeros((n, 3), np.float32)
        self.backface_mix = np.zeros(n, np.float32)
        # a velvet lobe, added in the light loop rather than inside a model:
        # the packages that had this offered it on top of whichever shader was
        # picked, exactly like the terms above
        self.sheen = np.zeros(n, np.float32)
        self.sheen_color = np.ones((n, 3), np.float32)
        self.sheen_roughness = np.full(n, 0.3, np.float32)
        # how much of the ray traced through a transparent surface is kept
        self.refraction = np.ones(n, np.float32)
        self.model = 'PHONG'


def fresnel_schlick(cos_t, f0):
    c = np.clip(1.0 - cos_t, 0.0, 1.0)
    c5 = c * c * c * c * c
    if np.ndim(f0) == 2:
        return f0 + (1.0 - f0) * c5[:, None]
    return f0 + (1.0 - f0) * c5


def fresnel_dielectric(cos_t, ior):
    """Exact unpolarised dielectric Fresnel -- what the raytracers of the era used."""
    ci = np.clip(cos_t, 0.0, 1.0)
    eta = np.where(ci > 0, ior, 1.0 / np.maximum(ior, EPS))
    st2 = (1.0 - ci * ci) / np.maximum(eta * eta, EPS)
    tir = st2 > 1.0
    ct = np.sqrt(np.maximum(1.0 - st2, 0.0))
    rs = (eta * ci - ct) / np.maximum(eta * ci + ct, EPS)
    rp = (ci - eta * ct) / np.maximum(ci + eta * ct, EPS)
    f = 0.5 * (rs * rs + rp * rp)
    return np.where(tir, 1.0, np.clip(f, 0.0, 1.0))


def _gloss_to_alpha(gloss):
    """Phong exponent -> microfacet roughness (Walter et al. mapping)."""
    return np.sqrt(2.0 / np.maximum(gloss + 2.0, EPS))


def _soften(spec, ndl, amount):
    """3D Studio's 'Soften': fades the highlight where N.L approaches zero."""
    if not np.any(amount > 0):
        return spec
    s = np.clip(ndl * 3.0, 0.0, 1.0)
    fade = 1.0 - amount + amount * s
    return spec * fade


# ----------------------------------------------------------------- diffuse


def diffuse_lambert(ndl, **_):
    return np.maximum(ndl, 0.0)


def diffuse_oren_nayar(ndl, ndv, l, v, n, roughness, **_):
    """2.79's OrenNayar_Diff, verbatim (R155): nv clamps at 0 (View_A
    caps at pi/2), the projected-vector cosine floors at 0, and the
    smaller angle is scaled by 0.95 before tan -- the C's own guard
    against the tangent shooting to infinity."""
    s2 = roughness * roughness
    a = 1.0 - 0.5 * s2 / (s2 + 0.33)
    b = 0.45 * s2 / (s2 + 0.09)
    nl = np.clip(ndl, -1.0, 1.0)
    nv = np.maximum(np.clip(ndv, -1.0, 1.0), 0.0)
    ti = np.arccos(np.clip(nl, -1.0, 1.0))
    tr = np.arccos(np.clip(nv, -1.0, 1.0))
    alpha = np.maximum(ti, tr)
    beta = np.minimum(ti, tr) * 0.95
    lp = l - n * nl[:, None]
    vp = v - n * nv[:, None]
    cos_dphi = np.clip(M.dot(M.normalize(lp), M.normalize(vp)), -1.0, 1.0)
    return (np.maximum(nl, 0.0) * (a + b * np.maximum(cos_dphi, 0.0) *
                                   np.sin(alpha) * np.tan(beta))
            ).astype(np.float32)


def diffuse_minnaert(ndl, ndv, darkness=1.0, **_):
    nl = np.maximum(ndl, 0.0)
    nv = np.maximum(ndv, EPS)
    k = np.maximum(darkness, 0.0)
    return nl * np.power(np.maximum(nl * nv, EPS), k - 1.0) * nv


def diffuse_toon(ndl, size, smooth, steps=2.0, **_):
    """Cel shading: the light ramp cut into `steps` flat tones.

    `steps` counts tones, not edges, so two is the familiar lit/unlit cel and
    the edges for more than that are spread evenly across the lit range. At two
    this is bit-identical to the single-step version it replaces -- the loop
    runs once and reproduces the old expression exactly -- which matters
    because two is the default and every toon render made before this used it.

    The parameter was accepted and ignored for four releases: it reached the
    node, the exporter and this signature, and then nothing read it.
    """
    nl = np.clip(ndl, 0.0, 1.0)
    ang = np.arccos(np.clip(nl, 0.0, 1.0)) / (np.pi * 0.5)
    lim = np.clip(1.0 - size, 0.0, 1.0)
    sm = np.maximum(smooth, 1e-4)
    edges = np.maximum(np.round(np.asarray(steps, np.float32)) - 1.0, 1.0)
    top = int(np.max(edges)) if np.size(edges) else 1
    if top <= 1:
        return np.clip((lim + sm - ang) / sm, 0.0, 1.0)
    acc = np.zeros_like(nl, np.float32)
    for i in range(1, top + 1):
        band = np.clip((lim * (np.float32(i) / edges) + sm - ang) / sm, 0.0, 1.0)
        acc = acc + np.where(i <= edges, band, 0.0)
    return (acc / edges).astype(np.float32)


def diffuse_fujii(ndl, ndv, roughness, **_):
    """Energy-conserving 'Fujii' Oren-Nayar variant (cheap, well behaved)."""
    s = roughness
    fl = 1.0 / (np.pi * (1.0 + 0.5 * s))
    return np.maximum(ndl, 0.0) * (1.0 + s * (1.0 - 0.5 * np.maximum(ndl, 0)) *
                                   (1.0 - 0.5 * np.maximum(ndv, 0))) * fl * np.pi


# ---------------------------------------------------------------- specular


def spec_phong(ndl, rdv, gloss, **_):
    return np.where(ndl > 0, M.safe_pow(np.maximum(rdv, 0.0), gloss), 0.0)


def spec_blinn_phong(ndl, ndh, gloss, **_):
    return np.where(ndl > 0, M.safe_pow(np.maximum(ndh, 0.0), gloss * 4.0), 0.0)


def spec_blinn(ndl, ndv, ndh, vdh, gloss, ior, **_):
    """Blinn 1977 / Torrance-Sparrow: D * G * F / (4 N.V)."""
    a = _gloss_to_alpha(gloss)
    a2 = np.maximum(a * a, 1e-6)
    nh = np.maximum(ndh, 0.0)
    d = np.exp((nh * nh - 1.0) / np.maximum(a2 * nh * nh, EPS)) / \
        (np.pi * a2 * np.maximum(nh ** 4, EPS))
    g = np.minimum(1.0, np.minimum(2.0 * nh * np.maximum(ndv, 0) / np.maximum(vdh, EPS),
                                   2.0 * nh * np.maximum(ndl, 0) / np.maximum(vdh, EPS)))
    f0 = ((ior - 1.0) / np.maximum(ior + 1.0, EPS)) ** 2
    f = fresnel_schlick(np.maximum(vdh, 0.0), f0)
    out = d * g * f / (4.0 * np.maximum(ndv, EPS))
    return np.where(ndl > 0, np.maximum(out, 0.0), 0.0)


def spec_cook_torrance(ndl, ndv, ndh, vdh, roughness, ior, **_):
    m = np.maximum(roughness, 0.02)
    m2 = m * m
    nh = np.maximum(ndh, EPS)
    nh2 = nh * nh
    d = np.exp((nh2 - 1.0) / np.maximum(m2 * nh2, EPS)) / \
        (np.pi * m2 * np.maximum(nh2 * nh2, EPS))
    g = np.minimum(1.0, np.minimum(2.0 * nh * np.maximum(ndv, 0) / np.maximum(vdh, EPS),
                                   2.0 * nh * np.maximum(ndl, 0) / np.maximum(vdh, EPS)))
    f = fresnel_dielectric(np.maximum(vdh, 0.0), ior)
    out = d * g * f / (np.pi * np.maximum(ndv, EPS) * np.maximum(ndl, EPS))
    return np.where(ndl > 0, np.clip(out * np.maximum(ndl, 0), 0.0, 64.0), 0.0)


def spec_ward(ndl, ndv, h, n, t, b, ax, ay, **_):
    hdn = np.maximum(M.dot(h, n), EPS)
    hdt = M.dot(h, t)
    hdb = M.dot(h, b)
    ax = np.maximum(ax, 0.005)
    ay = np.maximum(ay, 0.005)
    ex = -((hdt / ax) ** 2 + (hdb / ay) ** 2) / np.maximum(hdn * hdn, EPS)
    denom = 4.0 * np.pi * ax * ay * np.sqrt(np.maximum(np.maximum(ndl, EPS) *
                                                       np.maximum(ndv, EPS), EPS))
    out = np.exp(ex) / np.maximum(denom, EPS)
    return np.where(ndl > 0, np.clip(out * np.maximum(ndl, 0), 0.0, 64.0), 0.0)


def spec_aniso_blinn(ndl, ndh, h, n, t, b, gloss, aniso, **_):
    """Elliptical Blinn highlight, as 3D Studio's Anisotropic shader.

    Distinct from Ward: this stretches a Blinn cosine lobe by giving it two
    exponents, rather than evaluating a Gaussian on the slope distribution.
    At anisotropy 0 the two exponents coincide and it reduces to Blinn-Phong,
    which is exactly what the original did.
    """
    ht = M.dot(h, t)
    hb = M.dot(h, b)
    hn = np.clip(ndh, -1.0, 1.0)
    denom = np.maximum(1.0 - hn * hn, 1e-6)
    a = np.clip(aniso, -0.95, 0.95)
    nu = np.maximum(gloss * (1.0 + a), 0.5)
    nv = np.maximum(gloss * (1.0 - a), 0.5)
    e = (nu * ht * ht + nv * hb * hb) / denom
    lobe = np.power(np.clip(hn, 0.0, 1.0), np.clip(e, 0.0, 8192.0))
    norm = np.sqrt((nu + 1.0) * (nv + 1.0)) / (8.0 * np.pi)
    return np.where(ndl > 0.0, lobe * norm * 8.0, 0.0).astype(np.float32)


def bi_spec_pow(inp, gloss):
    """2.79's spec(): the integer-bit square-multiply power, verbatim.

    Not pow(x, n): b1 = x*x floors at 0.01 before the bit ladder, b1
    zeroes below 0.001 twice on the way up, an EVEN hardness drops the
    x^1 factor, and bit 256 squares once more. Hardness itself is
    shi->har -- a SHORT -- so the per-pixel float chain truncates to an
    integer here, exactly where the C's assignment did. The 0.01 floor
    is visible: it brightens the dim tail of low-hardness highlights
    (the porcelain range) over what a plain power gives."""
    inp = np.asarray(inp, np.float32)
    hard = np.asarray(np.floor(np.asarray(gloss, np.float32)),
                      np.int32)
    x = np.clip(inp, 0.0, 1.0)
    out = np.where((hard & 1) == 0, np.float32(1.0), x)
    b1 = np.maximum(x * x, np.float32(0.01))
    out = np.where((hard & 2) != 0, out * b1, out)
    b1 = b1 * b1
    out = np.where((hard & 4) != 0, out * b1, out)
    b1 = b1 * b1
    out = np.where((hard & 8) != 0, out * b1, out)
    b1 = b1 * b1
    out = np.where((hard & 16) != 0, out * b1, out)
    b1 = b1 * b1
    b1 = np.where(b1 < 0.001, np.float32(0.0), b1)
    out = np.where((hard & 32) != 0, out * b1, out)
    b1 = b1 * b1
    out = np.where((hard & 64) != 0, out * b1, out)
    b1 = b1 * b1
    out = np.where((hard & 128) != 0, out * b1, out)
    b1 = np.where(b1 < 0.001, np.float32(0.0), b1)
    out = np.where((hard & 256) != 0, out * (b1 * b1), out)
    return np.where(inp >= 1.0, np.float32(1.0),
                    np.where(inp <= 0.0, np.float32(0.0),
                             out)).astype(np.float32)


def spec_bi_cooktorr(ndl, ndv, ndh, gloss, **_):
    """Blender Internal's CookTorr_Spec, verbatim from 2.79 (R155).

    spec(N.H, hardness) / (0.1 + N.V) -- the divisor is the whole
    character: the same highlight brightens up to 10x toward grazing
    view. The C has NO N.L gate (spec can sit past the terminator,
    a quirk BI shipped for its whole life) and no upper clamp; nh < 0
    returns 0, nv < 0 clamps to 0."""
    nv = np.maximum(ndv, 0.0)
    out = bi_spec_pow(ndh, gloss) / (0.1 + nv)
    return np.where(ndh < 0.0, np.float32(0.0), out).astype(np.float32)


def spec_bi_phong(ndl, ndh, gloss, **_):
    """Blender Internal's Phong_Spec, verbatim from 2.79 (R155): the
    HALF-vector lobe through spec(); rslt <= 0 returns 0, and there is
    no N.L gate in the C."""
    return np.where(ndh > 0.0, bi_spec_pow(ndh, gloss),
                    np.float32(0.0)).astype(np.float32)


def spec_bi_blinn(ndl, ndv, ndh, vdh, gloss, ior, **_):
    """Blender Internal's Blinn_Spec, transcribed term for term.

    Hardness maps onto a Gaussian half-angle width exactly as BI did
    (sqrt(1/hard) under 100, 10/hard above), the geometry term is
    Torrance-Sparrow's, and the Fresnel is BI's own refraction-index
    form. Returns 0 below refrac 1, as BI did."""
    refrac = np.maximum(np.asarray(ior, np.float32), 0.0)
    # spec_power arrives as (float)shi->har -- an INT-truncated short
    sp = np.maximum(np.floor(np.asarray(gloss, np.float32)), 1.0)
    spow = np.where(sp < 100.0, np.sqrt(1.0 / sp), 10.0 / sp)
    nh = np.maximum(ndh, 0.0)
    nv = np.maximum(ndv, 0.01)
    nl = np.maximum(ndl, 0.0)
    vh = np.maximum(vdh, 0.01)
    # the C's geometry pick is a STRICT-compare chain: g stays 0.0 on
    # ties (verbatim R155) -- a<b&&a<c, elif b<a&&b<c, elif c<a&&c<b
    a = np.ones_like(nh)
    b = 2.0 * nh * nv / vh
    c = 2.0 * nh * nl / vh
    g = np.where((a < b) & (a < c), a,
                 np.where((b < a) & (b < c), b,
                          np.where((c < a) & (c < b), c,
                                   np.float32(0.0))))
    p = np.sqrt(np.maximum(refrac * refrac + vh * vh - 1.0, 0.0))
    f = ((p - vh) ** 2 / (p + vh) ** 2) * \
        (1.0 + ((vh * (p + vh) - 1.0) ** 2
                / (vh * (p - vh) + 1.0) ** 2))
    ang = np.arccos(np.clip(nh, -1.0, 1.0))
    out = f * g * np.exp(-(ang * ang)
                         / np.maximum(2.0 * spow * spow, 1e-8))
    # verbatim gates: refrac < 1 -> 0, nl <= 0.01 -> 0, nh < 0 -> 0,
    # negative result -> 0; NO upper clamp in the C
    out = np.where((refrac >= 1.0) & (ndl > 0.01) & (ndh >= 0.0),
                   out, 0.0)
    return np.maximum(out, 0.0).astype(np.float32)


def bi_fresnel_fac(t1, grad, fac):
    """Blender Internal's fresnel_fac(), transcribed: t1 is view.vn."""
    fac = np.asarray(fac, np.float32)
    t2 = np.where(t1 > 0.0, 1.0 + t1, 1.0 - t1)
    t2 = grad + (1.0 - grad) * M.safe_pow(t2, fac)
    out = np.clip(t2, 0.0, 1.0)
    return np.where(fac == 0.0, 1.0, out).astype(np.float32)


def diffuse_bi_fresnel(ndl, grad, fac, **_):
    """BI's Fresnel diffuse, transcribed: fresnel_fac(lv, vn, ...) with
    lv pointing LAMP to surface, so view.vn = -N.L. It REPLACES the
    cosine outright -- the classic BI look where grazing lights glow."""
    return bi_fresnel_fac(-np.asarray(ndl, np.float32), grad, fac)


def diffuse_bi_toon(ndl, size, smooth, **_):
    """BI's Toon_Diff, transcribed: a hard angular band on acos(N.L)."""
    ang = np.arccos(np.clip(ndl, -1.0, 1.0))
    sm = np.asarray(smooth, np.float32)
    ramp = np.where(sm <= 0.0, 0.0,
                    1.0 - (ang - size) / np.maximum(sm, 1e-9))
    out = np.where(ang < size, 1.0,
                   np.where(ang >= size + sm, 0.0, ramp))
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def diffuse_bi_minnaert(ndl, ndv, darkness, **_):
    """BI's Minnaert_Diff, verbatim (R155), BOTH branches: darkness <=
    1 darkens toward the rim via pow(max(nv*nl, 0.1), dark-1); above 1
    it brightens the rim via pow(1.001 - nv, dark-1) -- 1.001, the C's
    own constant."""
    nl = np.maximum(ndl, 0.0)
    nv = np.maximum(ndv, 0.0)
    dk = np.asarray(darkness, np.float32)
    low = nl * M.safe_pow(np.maximum(nv * nl, 0.1), dk - 1.0)
    high = nl * M.safe_pow(np.maximum(1.001 - nv, 1e-6), dk - 1.0)
    return np.where(dk <= 1.0, low, high).astype(np.float32)


def spec_bi_toon(ndl, ndh, size, smooth, **_):
    """BI's Toon_Spec, verbatim: the angular band on acos(N.H). The C
    has no N.L gate -- the band shows wherever the half-vector allows,
    exactly like the other BI speculars."""
    ang = np.arccos(np.clip(ndh, -1.0, 1.0))
    sm = np.asarray(smooth, np.float32)
    ramp = np.where(sm <= 0.0, 0.0,
                    1.0 - (ang - size) / np.maximum(sm, 1e-9))
    out = np.where(ang < size, 1.0,
                   np.where(ang >= size + sm, 0.0, ramp))
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def spec_bi_wardiso(ndl, ndv, ndh, rms, **_):
    """BI's WardIso_Spec, verbatim from 2.79 (R155): an isotropic
    Gaussian on tan(acos(N.H)) with the Slope (rms) width. The C
    CLAMPS nl/nv/nh to 0.001 -- it never gates, so even a backfacing
    light keeps a (vanishing) term -- and applies no upper clamp."""
    nh = np.maximum(ndh, 0.001)
    nv = np.maximum(ndv, 0.001)
    nl = np.maximum(ndl, 0.001)
    alpha = np.maximum(np.asarray(rms, np.float32), 0.001)
    angle = np.tan(np.arccos(np.clip(nh, -1.0, 1.0)))
    out = nl * (1.0 / (4.0 * np.pi * alpha * alpha)) * \
        (np.exp(-(angle * angle) / (alpha * alpha))
         / np.sqrt(np.maximum(nv * nl, 1e-8)))
    return out.astype(np.float32)


#: the BI material node's shader menus, in DNA order: the model string
#: 'BI_MATRIX_{d}_{s}' carries one digit from each
BI_DIFF_ORDER = ('LAMBERT', 'OREN_NAYAR', 'TOON', 'MINNAERT', 'FRESNEL')
BI_SPEC_ORDER = ('COOKTORR', 'PHONG', 'BLINN', 'TOON', 'WARDISO')


def bi_matrix_terms(model, surf, n, l, v, ndl, ndv, ndh, vdh):
    """(diffuse, specular scalar) for a BI material node's shader pair.

    The node keeps Blender Internal's diffuse and specular menus
    INDEPENDENT -- the 5x5 matrix one collapsed 'model' never could --
    and every branch is the transcribed 2.79 formula. Field packing:
    roughness carries Oren-Nayar roughness or Minnaert darkness (the
    diffuse menu chooses one), bi_slope carries WardIso's Slope,
    toon_size2/toon_smooth2 the specular Toon pair, glossiness the
    Hardness, ior the Refr slider."""
    di = int(model[10])
    si = int(model[12])
    if di == 1:
        dif = diffuse_oren_nayar(ndl, ndv, l, v, n, surf.roughness)
    elif di == 2:
        dif = diffuse_bi_toon(ndl, surf.toon_size, surf.toon_smooth)
    elif di == 3:
        dif = diffuse_bi_minnaert(ndl, ndv, surf.roughness)
    elif di == 4:
        dif = diffuse_bi_fresnel(ndl, surf.bi_fresnel,
                                 surf.bi_fresnel_fac)
    else:
        dif = np.maximum(ndl, 0.0)
    if si == 1:
        spec = spec_bi_phong(ndl, ndh, surf.glossiness)
    elif si == 2:
        spec = spec_bi_blinn(ndl, ndv, ndh, vdh, surf.glossiness,
                             surf.ior)
    elif si == 3:
        spec = spec_bi_toon(ndl, ndh, surf.toon_size2, surf.toon_smooth2)
    elif si == 4:
        spec = spec_bi_wardiso(ndl, ndv, ndh, surf.bi_slope)
    else:
        spec = spec_bi_cooktorr(ndl, ndv, ndh, surf.glossiness)
    return dif, spec


def bi_cubic(dif):
    """BI's Cubic Interpolation: smoothstep on the diffuse term.

    Transcribed with 2.79's own guard -- only values strictly inside
    (0, 1) are reshaped, so a Fresnel diffuse sitting at exactly 1.0
    or a negative pre-clamp term passes through untouched."""
    dif = np.asarray(dif, np.float32)
    inside = (dif > 0.0) & (dif < 1.0)
    return np.where(inside, 3.0 * dif * dif - 2.0 * dif * dif * dif,
                    dif).astype(np.float32)


def bi_tangent_normal(t, l):
    """BI's Tangent Shading: the per-light fake normal.

    2.79 builds cross(lv, tang) then cross(tang, that) -- algebraically
    L - T*(T.L), the light direction stripped of its along-strand
    component -- and shades with it in place of the surface normal.
    Returns the normalized fake normal (falls back to L where the
    light runs exactly along the tangent)."""
    t = np.asarray(t, np.float32)
    l = np.asarray(l, np.float32)
    n_eff = l - t * M.dot(t, l)[:, None]
    length = np.sqrt(np.maximum((n_eff * n_eff).sum(1), 1e-18))
    return (n_eff / length[:, None]).astype(np.float32)


#: BI ramp_blend() mode names, in MA_RAMP_* DNA order (material.c)
BI_RAMP_BLEND_ORDER = ('MIX', 'ADD', 'MULT', 'SUB', 'SCREEN', 'DIV',
                       'DIFF', 'DARK', 'LIGHT', 'OVERLAY', 'DODGE',
                       'BURN', 'HUE', 'SAT', 'VAL', 'COLOR', 'SOFT',
                       'LINEAR')

#: BI ramp input names, in MA_RAMP_IN_* DNA order
BI_RAMP_INPUT_ORDER = ('SHADER', 'ENERGY', 'NORMAL', 'RESULT')


def bi_ramp_blend(mode, col, fac, rampcol):
    """2.79's ramp_blend() (blenkernel material.c), vectorized.

    col (N,3) is blended toward rampcol (N,3) by fac (N,); every mode
    is the C transcribed, including the per-channel conditionals and
    the achromatic guards on HUE/COLOR (a grey ramp colour leaves the
    base untouched) and SAT (a grey base keeps its grey)."""
    col = np.asarray(col, np.float32).copy()
    rc = np.asarray(rampcol, np.float32)
    fac = np.asarray(fac, np.float32)
    if fac.ndim == 1:
        fac = fac[:, None]
    facm = 1.0 - fac
    if mode == 'MIX':
        return (facm * col + fac * rc).astype(np.float32)
    if mode == 'ADD':
        return (col + fac * rc).astype(np.float32)
    if mode == 'MULT':
        return (col * (facm + fac * rc)).astype(np.float32)
    if mode == 'SCREEN':
        return (1.0 - (facm + fac * (1.0 - rc)) *
                (1.0 - col)).astype(np.float32)
    if mode == 'OVERLAY':
        low = col * (facm + 2.0 * fac * rc)
        high = 1.0 - (facm + 2.0 * fac * (1.0 - rc)) * (1.0 - col)
        return np.where(col < 0.5, low, high).astype(np.float32)
    if mode == 'SUB':
        return (col - fac * rc).astype(np.float32)
    if mode == 'DIV':
        # per channel: only where the ramp colour is nonzero
        out = facm * col + fac * col / np.where(rc != 0.0, rc, 1.0)
        return np.where(rc != 0.0, out, col).astype(np.float32)
    if mode == 'DIFF':
        return (facm * col + fac * np.abs(col - rc)).astype(np.float32)
    if mode == 'DARK':
        tmp = rc + (1.0 - rc) * facm
        return np.minimum(col, tmp).astype(np.float32)
    if mode == 'LIGHT':
        return np.maximum(col, fac * rc).astype(np.float32)
    if mode == 'DODGE':
        tmp = 1.0 - fac * rc
        lifted = np.where(tmp <= 0.0, 1.0,
                          np.minimum(col / np.where(tmp <= 0.0, 1.0, tmp),
                                     1.0))
        return np.where(col != 0.0, lifted, col).astype(np.float32)
    if mode == 'BURN':
        tmp = facm + fac * rc
        burned = np.where(tmp <= 0.0, 0.0,
                          np.clip(1.0 - (1.0 - col) /
                                  np.where(tmp <= 0.0, 1.0, tmp), 0.0, 1.0))
        return burned.astype(np.float32)
    if mode in ('HUE', 'SAT', 'VAL', 'COLOR'):
        rh, rs, rv = M.rgb_to_hsv(col[:, 0], col[:, 1], col[:, 2])
        ch, cs, cv = M.rgb_to_hsv(rc[:, 0], rc[:, 1], rc[:, 2])
        f1 = fac[:, 0]
        fm1 = 1.0 - f1
        if mode == 'HUE':
            tr, tg, tb = M.hsv_to_rgb(ch, rs, rv)
            mixed = (facm * col +
                     fac * np.stack([tr, tg, tb], 1)).astype(np.float32)
            return np.where((cs != 0.0)[:, None], mixed,
                            col).astype(np.float32)
        if mode == 'SAT':
            tr, tg, tb = M.hsv_to_rgb(rh, fm1 * rs + f1 * cs, rv)
            out = np.stack([tr, tg, tb], 1)
            return np.where((rs != 0.0)[:, None], out,
                            col).astype(np.float32)
        if mode == 'VAL':
            tr, tg, tb = M.hsv_to_rgb(rh, rs, fm1 * rv + f1 * cv)
            return np.stack([tr, tg, tb], 1).astype(np.float32)
        # COLOR: ramp hue+sat over base value, achromatic-guarded
        tr, tg, tb = M.hsv_to_rgb(ch, cs, rv)
        mixed = (facm * col +
                 fac * np.stack([tr, tg, tb], 1)).astype(np.float32)
        return np.where((cs != 0.0)[:, None], mixed, col).astype(np.float32)
    if mode == 'SOFT':
        scr = 1.0 - (1.0 - rc) * (1.0 - col)
        return (facm * col +
                fac * ((1.0 - col) * rc * col +
                       col * scr)).astype(np.float32)
    if mode == 'LINEAR':
        return (col + fac * np.where(rc > 0.5, 2.0 * (rc - 0.5),
                                     2.0 * rc - 1.0)).astype(np.float32)
    return col.astype(np.float32)


def spec_toon(ndl, rdv, size, smooth, **_):
    ang = np.arccos(np.clip(rdv, -1.0, 1.0)) / (np.pi * 0.5)
    lim = np.clip(1.0 - size, 0.0, 1.0)
    sm = np.maximum(smooth, 1e-4)
    return np.where(ndl > 0, np.clip((lim + sm - ang) / sm, 0.0, 1.0), 0.0)


def spec_strauss(ndl, ndv, rdv, h, n, l, v, smoothness, metalness, transparency, **_):
    """Strauss 1990, as shipped in 3D Studio MAX."""
    s = np.clip(smoothness, 0.0, 1.0)
    m = np.clip(metalness, 0.0, 1.0)
    t = np.clip(transparency, 0.0, 1.0)
    h_ = 3.0 / np.maximum(1.0 - s, 1e-3)
    rn = (1.0 - t) - (1.0 - s) ** 3 * (1.0 - t)
    kf, kg = 1.12, 1.01
    fnl = _strauss_f(np.arccos(np.clip(ndl, -1, 1)) / (np.pi * 0.5), kf)
    gnl = _strauss_g(np.arccos(np.clip(ndl, -1, 1)) / (np.pi * 0.5), kg)
    gnv = _strauss_g(np.arccos(np.clip(ndv, -1, 1)) / (np.pi * 0.5), kg)
    j = fnl * gnl * gnv
    rj = np.minimum(1.0, rn + (rn + 0.1) * j)
    rs = M.safe_pow(np.maximum(-rdv, 0.0), h_) * rj
    return np.where(ndl > 0, rs, 0.0), rn, m


def _strauss_f(x, k):
    return (1.0 / (x - k) ** 2 - 1.0 / (k * k)) / (1.0 / (1.0 - k) ** 2 - 1.0 / (k * k))


def _strauss_g(x, k):
    return (1.0 / (k - 1.0) ** 2 - 1.0 / (x - k) ** 2) / \
           (1.0 / (k - 1.0) ** 2 - 1.0 / (k * k))


# ------------------------------------------------------------------ driver


def evaluate(model, surf, n, l, v, ndl_raw=None):
    """Evaluate one light for `model`.

    n, l, v: (N,3) unit vectors. l points from surface *toward* the light,
    v points from surface toward the eye.
    Returns (diffuse (N,), specular (N,3)).
    """
    ndl = M.dot(n, l) if ndl_raw is None else ndl_raw
    ndv = M.dot(n, v)
    h = M.normalize(l + v)
    ndh = M.dot(n, h)
    vdh = M.dot(v, h)
    r = M.reflect(-l, n)
    rdv = M.dot(r, v)
    gloss = surf.glossiness
    zero = np.zeros_like(ndl)

    if model in ('CONSTANT', 'WIREFRAME'):
        return zero, np.zeros((surf.n, 3), np.float32)

    if isinstance(model, str) and model.startswith('BI_MATRIX_'):
        # the BI material node: independent diffuse and specular menus,
        # every branch a transcribed 2.79 formula
        if np.any(surf.bi_tangent > 0.5) and surf.tangent is not None:
            # Tangent Shading: 2.79 swaps the surface normal for
            # cross(tang, cross(lv, tang)) -- the light direction
            # stripped of its along-tangent component -- per light,
            # for BOTH lobes
            n_t = bi_tangent_normal(surf.tangent, l)
            on = (surf.bi_tangent > 0.5)[:, None]
            n_eff = np.where(on, n_t, n)
            ndl = M.dot(n_eff, l)
            ndv = M.dot(n_eff, v)
            ndh = M.dot(n_eff, h)
            n_use = n_eff
        else:
            n_use = n
        dif, spec = bi_matrix_terms(model, surf, n_use, l, v,
                                    ndl, ndv, ndh, vdh)
        if np.any(surf.translucency > 0.0):
            # BI's translucency: the SAME diffuse shader, evaluated
            # through the flipped normal, scaled by the slider -- for
            # every shader, not just a dedicated model
            dif_back, _sb = bi_matrix_terms(model, surf, -n_use, l, v,
                                            -ndl, -ndv, -ndh, vdh)
            dif = dif + np.clip(surf.translucency, 0.0, 1.0) * dif_back
        if np.any(surf.bi_cubic > 0.5):
            dif = np.where(surf.bi_cubic > 0.5, bi_cubic(dif), dif)
        spec = _soften(spec, ndl, surf.soften)
        return dif, spec[:, None] * surf.specular

    # ---- diffuse term
    if model == 'OREN_NAYAR':
        dif = diffuse_oren_nayar(ndl, ndv, l, v, n, surf.roughness)
    elif model == 'MINNAERT':
        dif = diffuse_minnaert(ndl, ndv, 1.0 + surf.roughness * 2.0)
    elif model == 'TOON':
        dif = diffuse_toon(ndl, surf.toon_size, surf.toon_smooth, surf.toon_steps)
    elif model == 'TRANSLUCENT':
        dif = np.maximum(ndl, 0.0) + np.maximum(-ndl, 0.0) * surf.translucency
    elif model == 'STRAUSS':
        dif = np.maximum(ndl, 0.0)
    else:
        dif = np.maximum(ndl, 0.0)

    # ---- specular term
    if model in ('LAMBERT', 'OREN_NAYAR', 'MINNAERT', 'TRANSLUCENT'):
        spec = zero
        spec_col = surf.specular
    elif model == 'PHONG':
        spec = spec_phong(ndl, rdv, gloss)
        spec_col = surf.specular
    elif model in ('BLINN_PHONG', 'GOURAUD', 'FLAT'):
        spec = spec_blinn_phong(ndl, ndh, gloss)
        spec_col = surf.specular
    elif model == 'BLINN':
        spec = spec_blinn(ndl, ndv, ndh, vdh, gloss, surf.ior)
        spec_col = surf.specular
    elif model == 'COOK_TORRANCE':
        spec = spec_cook_torrance(ndl, ndv, ndh, vdh, surf.roughness, surf.ior)
        spec_col = surf.specular
    elif model == 'WARD':
        t, b = _aniso_frame(surf, n)
        rough = np.maximum(surf.roughness, 0.02)
        aniso = np.clip(surf.anisotropy, -0.99, 0.99)
        ax = rough * (1.0 + aniso)
        ay = rough * (1.0 - aniso)
        spec = spec_ward(ndl, ndv, h, n, t, b, ax, ay)
        spec_col = surf.specular
    elif model == 'ANISOTROPIC':
        t, b = _aniso_frame(surf, n)
        spec = spec_aniso_blinn(ndl, ndh, h, n, t, b, gloss, surf.anisotropy)
        spec_col = surf.specular
    elif model == 'METAL':
        # 3D Studio's Metal: highlight takes the diffuse colour, no white sheen
        spec = spec_cook_torrance(ndl, ndv, ndh, vdh,
                                  np.maximum(1.0 / np.maximum(gloss, 1.0), 0.02),
                                  surf.ior)
        spec_col = surf.diffuse
    elif model == 'TOON':
        spec = spec_toon(ndl, rdv, surf.toon_size * 0.5, surf.toon_smooth)
        spec_col = surf.specular
    elif model == 'STRAUSS':
        smooth = np.clip(gloss / 100.0, 0.0, 1.0)
        spec, rn, met = spec_strauss(ndl, ndv, rdv, h, n, l, v, smooth,
                                     surf.metallic, 1.0 - surf.opacity)
        white = np.ones((surf.n, 3), np.float32)
        cs = white + met[:, None] * (1.0 - _strauss_fresnel(ndl)[:, None]) * \
            (surf.diffuse - white)
        return dif * rn, spec[:, None] * cs
    elif model == 'BI_COOKTORR':
        spec = spec_bi_cooktorr(ndl, ndv, ndh, gloss)
        spec_col = surf.specular
    elif model == 'BI_PHONG':
        spec = spec_bi_phong(ndl, ndh, gloss)
        spec_col = surf.specular
    elif model == 'BI_BLINN':
        spec = spec_bi_blinn(ndl, ndv, ndh, vdh, gloss, surf.ior)
        spec_col = surf.specular
    elif model == 'MULTI_LAYER':
        s1 = spec_blinn_phong(ndl, ndh, gloss)
        s2 = spec_blinn_phong(ndl, ndh, np.maximum(gloss * 0.15, 1.0))
        spec = s1 + s2 * 0.35
        spec_col = surf.specular
    else:
        spec = spec_blinn_phong(ndl, ndh, gloss)
        spec_col = surf.specular

    spec = _soften(spec, ndl, surf.soften)
    if np.ndim(spec) == 1:
        spec = spec[:, None] * spec_col
    return dif, spec


def _strauss_fresnel(ndl):
    return np.clip(1.0 - np.abs(ndl), 0.0, 1.0)


def _aniso_frame(surf, n):
    if surf.tangent is not None:
        t = surf.tangent
        b = surf.bitangent if surf.bitangent is not None else M.cross(n, t)
    else:
        t, b = M.orthonormal_basis(n)
    rot = surf.aniso_rot
    if np.any(np.abs(rot) > 1e-6):
        a = rot * 2.0 * np.pi
        ca, sa = np.cos(a)[:, None], np.sin(a)[:, None]
        t2 = t * ca + b * sa
        b2 = -t * sa + b * ca
        t, b = t2, b2
    return M.normalize(t), M.normalize(b)


def ambient_term(surf, model):
    if model in ('CONSTANT', 'WIREFRAME'):
        return np.ones((surf.n, 3), np.float32)
    return surf.ambient[:, None] * np.ones((1, 3), np.float32)
