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
                 'edge_opacity', 'backface_color', 'backface_mix')

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
    s2 = roughness * roughness
    a = 1.0 - 0.5 * s2 / (s2 + 0.33)
    b = 0.45 * s2 / (s2 + 0.09)
    nl = np.clip(ndl, -1.0, 1.0)
    nv = np.clip(ndv, -1.0, 1.0)
    ti = np.arccos(np.clip(nl, -1.0, 1.0))
    tr = np.arccos(np.clip(nv, -1.0, 1.0))
    alpha = np.maximum(ti, tr)
    beta = np.minimum(ti, tr)
    lp = l - n * nl[:, None]
    vp = v - n * nv[:, None]
    cos_dphi = np.clip(M.dot(M.normalize(lp), M.normalize(vp)), -1.0, 1.0)
    return np.maximum(nl, 0.0) * (a + b * np.maximum(cos_dphi, 0.0) *
                                  np.sin(alpha) * np.tan(beta))


def diffuse_minnaert(ndl, ndv, darkness=1.0, **_):
    nl = np.maximum(ndl, 0.0)
    nv = np.maximum(ndv, EPS)
    k = np.maximum(darkness, 0.0)
    return nl * np.power(np.maximum(nl * nv, EPS), k - 1.0) * nv


def diffuse_toon(ndl, size, smooth, steps=2.0, **_):
    nl = np.clip(ndl, 0.0, 1.0)
    ang = np.arccos(np.clip(nl, 0.0, 1.0)) / (np.pi * 0.5)
    lim = np.clip(1.0 - size, 0.0, 1.0)
    sm = np.maximum(smooth, 1e-4)
    return np.clip((lim + sm - ang) / sm, 0.0, 1.0)


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
