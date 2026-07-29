"""The reflectance models as GLSL.

Written to compile under both Blender's GPUShader and Halcyon's own GLSL
front-end, because that is what makes them checkable: the same source is run
through NumPy and compared against `core/shading.py` model by model. Nothing
here is trusted until that comparison passes.

Each model is a function returning `vec4(diffuse, specular.rgb)`, matching what
`shading.evaluate` returns on the CPU.

`HalcyonSurface` mirrors the fields of `core.shading.Surface` that any model
reads. Members are set by the generated material shader before the call.
"""

COMMON = """
struct HalcyonSurface {
    vec3  diffuse;
    vec3  specular;
    float diffuse_level;
    float specular_level;
    float glossiness;
    float roughness;
    float metallic;
    float anisotropy;
    float aniso_rot;
    float soften;
    float ior;
    float translucency;
    float toon_size;
    float toon_smooth;
    float toon_steps;
    float opacity;
    vec3  tangent;
    vec3  bitangent;
};

float hal_saturate(float x) { return clamp(x, 0.0, 1.0); }

vec3 hal_rgb2hsv(vec3 c)
{
    float mx = max(c.r, max(c.g, c.b));
    float mn = min(c.r, min(c.g, c.b));
    float d = mx - mn;
    float h = 0.0;
    if (d > 1e-8) {
        if (mx == c.r)      { h = mod((c.g - c.b) / d, 6.0); }
        else if (mx == c.g) { h = (c.b - c.r) / d + 2.0; }
        else                { h = (c.r - c.g) / d + 4.0; }
        h = h / 6.0;
    }
    return vec3(h, mx > 1e-8 ? d / mx : 0.0, mx);
}

vec3 hal_hsv2rgb(vec3 c)
{
    float h = fract(c.x) * 6.0;
    float f = fract(h);
    float p = c.z * (1.0 - c.y);
    float q = c.z * (1.0 - c.y * f);
    float t = c.z * (1.0 - c.y * (1.0 - f));
    int i = int(floor(h));
    if (i == 0) { return vec3(c.z, t, p); }
    if (i == 1) { return vec3(q, c.z, p); }
    if (i == 2) { return vec3(p, c.z, t); }
    if (i == 3) { return vec3(p, q, c.z); }
    if (i == 4) { return vec3(t, p, c.z); }
    return vec3(c.z, p, q);
}

float hal_gloss_to_alpha(float gloss)
{
    // the same 2/a^2 - 2 relation the CPU path inverts
    return sqrt(2.0 / (max(gloss, 0.5) + 2.0));
}

float hal_fresnel_schlick(float cos_t, float f0)
{
    float m = clamp(1.0 - cos_t, 0.0, 1.0);
    return f0 + (1.0 - f0) * m * m * m * m * m;
}

float hal_fresnel_dielectric(float cos_t, float ior)
{
    float c = abs(cos_t);
    float g2 = ior * ior - 1.0 + c * c;
    if (g2 < 0.0) { return 1.0; }
    float g = sqrt(g2);
    float a = (g - c) / (g + c);
    float b = (c * (g + c) - 1.0) / (c * (g - c) + 1.0);
    return 0.5 * a * a * (1.0 + b * b);
}

float hal_soften(float spec, float ndl, float amount)
{
    if (amount <= 0.0) { return spec; }
    float t = hal_saturate(ndl / max(amount, 1e-4));
    return spec * (t * t * (3.0 - 2.0 * t));
}
"""

DIFFUSE = """
float hal_diffuse_lambert(float ndl) { return max(ndl, 0.0); }

float hal_diffuse_oren_nayar(float ndl, float ndv, vec3 l, vec3 v, vec3 n,
                             float roughness)
{
    float s2 = roughness * roughness;
    float A = 1.0 - 0.5 * s2 / (s2 + 0.33);
    float B = 0.45 * s2 / (s2 + 0.09);
    vec3 lp = normalize(l - n * ndl);
    vec3 vp = normalize(v - n * ndv);
    float cosphi = max(dot(lp, vp), 0.0);
    float ti = acos(clamp(ndl, -1.0, 1.0));
    float tr = acos(clamp(ndv, -1.0, 1.0));
    float alpha = max(ti, tr);
    float beta = min(ti, tr);
    return max(ndl, 0.0) * (A + B * cosphi * sin(alpha) * tan(beta));
}

float hal_diffuse_minnaert(float ndl, float ndv, float darkness)
{
    float nl = max(ndl, 0.0);
    float nv = max(ndv, 1e-6);
    float k = max(darkness, 0.0);
    return nl * pow(max(nl * nv, 1e-6), k - 1.0) * nv;
}

// angle-based, not a smoothstep on the cosine: the terminator lands where the
// artist asked rather than wherever the cosine happens to cross
float hal_diffuse_toon(float ndl, float size, float smoothness, float steps)
{
    float nl = clamp(ndl, 0.0, 1.0);
    float ang = acos(clamp(nl, 0.0, 1.0)) / (3.14159265 * 0.5);
    float lim = clamp(1.0 - size, 0.0, 1.0);
    float sm = max(smoothness, 1e-4);
    float edges = max(floor(steps + 0.5) - 1.0, 1.0);
    float acc = 0.0;
    for (int i = 1; i <= 16; i++) {
        if (float(i) > edges) break;
        acc += clamp((lim * (float(i) / edges) + sm - ang) / sm, 0.0, 1.0);
    }
    return acc / edges;
}
"""

SPECULAR = """
float hal_spec_phong(float ndl, float rdv, float gloss)
{
    if (ndl <= 0.0) { return 0.0; }
    return pow(max(rdv, 0.0), max(gloss, 0.5));
}

float hal_spec_blinn_phong(float ndl, float ndh, float gloss)
{
    // the half-vector lobe is four times tighter than the reflection lobe at
    // the same exponent, which is why the CPU path multiplies by four
    if (ndl <= 0.0) { return 0.0; }
    return pow(max(ndh, 0.0), max(gloss * 4.0, 0.5));
}

float hal_spec_blinn(float ndl, float ndv, float ndh, float vdh, float gloss,
                     float ior)
{
    if (ndl <= 0.0 || ndv <= 0.0) { return 0.0; }
    float a = hal_gloss_to_alpha(gloss);
    float a2 = max(a * a, 1e-6);
    float nh = max(ndh, 0.0);
    float nh2 = nh * nh;
    float D = exp((nh2 - 1.0) / max(a2 * nh2, 1e-6))
              / (3.14159265 * a2 * max(nh2 * nh2, 1e-6));
    float G = min(1.0, min(2.0 * nh * max(ndv, 0.0) / max(vdh, 1e-6),
                           2.0 * nh * max(ndl, 0.0) / max(vdh, 1e-6)));
    float f0 = (ior - 1.0) / max(ior + 1.0, 1e-6);
    f0 = f0 * f0;
    float F = hal_fresnel_schlick(max(vdh, 0.0), f0);
    return D * G * F / (4.0 * max(ndv, 1e-6));
}

float hal_spec_cook_torrance(float ndl, float ndv, float ndh, float vdh,
                             float roughness, float ior)
{
    if (ndl <= 0.0 || ndv <= 0.0) { return 0.0; }
    float m = max(roughness, 1e-3);
    float m2 = m * m;
    float c2 = ndh * ndh;
    float t2 = (c2 - 1.0) / max(c2 * m2, 1e-8);
    float D = exp(t2) / max(3.14159265 * m2 * c2 * c2, 1e-8);
    float G = min(1.0, min(2.0 * ndh * ndv / max(vdh, 1e-6),
                           2.0 * ndh * ndl / max(vdh, 1e-6)));
    float F = hal_fresnel_dielectric(vdh, max(ior, 1.0001));
    return D * G * F / max(3.14159265 * ndv, 1e-6);
}

float hal_spec_ward(float ndl, float ndv, vec3 h, vec3 n, vec3 t, vec3 b,
                    float ax, float ay)
{
    if (ndl <= 0.0 || ndv <= 0.0) { return 0.0; }
    float hdn = max(dot(h, n), 1e-6);
    float axc = max(ax, 0.005);
    float ayc = max(ay, 0.005);
    float ht = dot(h, t) / axc;
    float hb = dot(h, b) / ayc;
    float e = -(ht * ht + hb * hb) / max(hdn * hdn, 1e-6);
    float denom = 4.0 * 3.14159265 * axc * ayc
                  * sqrt(max(max(ndl, 1e-6) * max(ndv, 1e-6), 1e-6));
    return clamp(exp(e) / max(denom, 1e-6) * max(ndl, 0.0), 0.0, 64.0);
}

float hal_spec_aniso_blinn(float ndl, float ndh, vec3 h, vec3 n, vec3 t, vec3 b,
                           float gloss, float aniso)
{
    if (ndl <= 0.0) { return 0.0; }
    float ht = dot(h, t);
    float hb = dot(h, b);
    float hn = clamp(ndh, -1.0, 1.0);
    float denom = max(1.0 - hn * hn, 1e-6);
    float a = clamp(aniso, -0.95, 0.95);
    float nu = max(gloss * (1.0 + a), 0.5);
    float nv = max(gloss * (1.0 - a), 0.5);
    float e = (nu * ht * ht + nv * hb * hb) / denom;
    float lobe = pow(clamp(hn, 0.0, 1.0), clamp(e, 0.0, 8192.0));
    float norm = sqrt((nu + 1.0) * (nv + 1.0)) / (8.0 * 3.14159265);
    return lobe * norm * 8.0;
}

float hal_strauss_f(float x, float k)
{
    float a = x - k;
    return (1.0 / (a * a) - 1.0 / (k * k))
           / (1.0 / ((1.0 - k) * (1.0 - k)) - 1.0 / (k * k));
}

float hal_strauss_g(float x, float k)
{
    float a = k - 1.0;
    float b = x - k;
    return (1.0 / (a * a) - 1.0 / (b * b))
           / (1.0 / (a * a) - 1.0 / (k * k));
}

// returns vec2(specular, rn) -- rn is needed for the metal tint
vec2 hal_spec_strauss(float ndl, float ndv, float rdv, float smoothness,
                      float transparency)
{
    float sm = clamp(smoothness, 0.0, 1.0);
    float t = clamp(transparency, 0.0, 1.0);
    float hh = 3.0 / max(1.0 - sm, 1e-3);
    float omt = 1.0 - t;
    float oms = 1.0 - sm;
    float rn = omt - oms * oms * oms * omt;
    float kf = 1.12;
    float kg = 1.01;
    float half_pi = 3.14159265 * 0.5;
    float anl = acos(clamp(ndl, -1.0, 1.0)) / half_pi;
    float anv = acos(clamp(ndv, -1.0, 1.0)) / half_pi;
    float j = hal_strauss_f(anl, kf) * hal_strauss_g(anl, kg)
              * hal_strauss_g(anv, kg);
    float rj = min(1.0, rn + (rn + 0.1) * j);
    float rs = pow(max(-rdv, 0.0), hh) * rj;
    return vec2(ndl > 0.0 ? rs : 0.0, rn);
}

float hal_spec_toon(float ndl, float rdv, float size, float smoothness)
{
    if (ndl <= 0.0) { return 0.0; }
    float ang = acos(clamp(rdv, -1.0, 1.0)) / (3.14159265 * 0.5);
    float lim = clamp(1.0 - size, 0.0, 1.0);
    float sm = max(smoothness, 1e-4);
    return clamp((lim + sm - ang) / sm, 0.0, 1.0);
}
"""

GLSL = COMMON + DIFFUSE + SPECULAR

#: models expressed as a call the generated shader can make. Each returns
#: vec4(diffuse, specular.rgb).
DISPATCH = """
vec4 hal_evaluate(int model, HalcyonSurface s, vec3 n, vec3 l, vec3 v)
{
    float ndl = dot(n, l);
    float ndv = dot(n, v);
    vec3 h = normalize(l + v);
    float ndh = dot(n, h);
    float vdh = dot(v, h);
    vec3 r = reflect(-l, n);
    float rdv = dot(r, v);

    float d = 0.0;
    float sp = 0.0;
    vec3 tint = vec3(1.0);

    if (model == 0) {                       // LAMBERT
        d = hal_diffuse_lambert(ndl);
    } else if (model == 1 || model == 2) {  // GOURAUD, FLAT
        // shading RATES, not models: the maths is Blinn-Phong, what differs is
        // how often the shader is invoked, which the caller decides
        d = hal_diffuse_lambert(ndl);
        sp = hal_spec_blinn_phong(ndl, ndh, s.glossiness);
    } else if (model == 3) {                // PHONG
        d = hal_diffuse_lambert(ndl);
        sp = hal_spec_phong(ndl, rdv, s.glossiness);
    } else if (model == 4) {                // BLINN_PHONG
        d = hal_diffuse_lambert(ndl);
        sp = hal_spec_blinn_phong(ndl, ndh, s.glossiness);
    } else if (model == 5) {                // BLINN
        d = hal_diffuse_lambert(ndl);
        sp = hal_spec_blinn(ndl, ndv, ndh, vdh, s.glossiness, s.ior);
    } else if (model == 6) {                // COOK_TORRANCE
        d = hal_diffuse_lambert(ndl);
        sp = hal_spec_cook_torrance(ndl, ndv, ndh, vdh, s.roughness, s.ior);
    } else if (model == 7) {                // OREN_NAYAR
        d = hal_diffuse_oren_nayar(ndl, ndv, l, v, n, s.roughness);
    } else if (model == 8) {                // MINNAERT
        d = hal_diffuse_minnaert(ndl, ndv, 1.0 + s.roughness * 2.0);
    } else if (model == 9) {                // WARD
        // driven by roughness, not by the Phong exponent: Ward is a Gaussian
        // on the slope distribution and takes its widths directly
        d = hal_diffuse_lambert(ndl);
        float rough = max(s.roughness, 0.02);
        float an = clamp(s.anisotropy, -0.99, 0.99);
        sp = hal_spec_ward(ndl, ndv, h, n, s.tangent, s.bitangent,
                           rough * (1.0 + an), rough * (1.0 - an));
    } else if (model == 10) {               // ANISOTROPIC
        d = hal_diffuse_lambert(ndl);
        sp = hal_spec_aniso_blinn(ndl, ndh, h, n, s.tangent, s.bitangent,
                                  s.glossiness, s.anisotropy);
    } else if (model == 11) {               // METAL
        d = hal_diffuse_lambert(ndl);
        sp = hal_spec_cook_torrance(ndl, ndv, ndh, vdh,
                                    max(1.0 / max(s.glossiness, 1.0), 0.02),
                                    s.ior);
        tint = s.diffuse;
    } else if (model == 12) {               // STRAUSS
        float smooth_s = clamp(s.glossiness / 100.0, 0.0, 1.0);
        vec2 st = hal_spec_strauss(ndl, ndv, rdv, smooth_s, 1.0 - s.opacity);
        // the diffuse is scaled by rn, and the metal tint uses a plain
        // 1 - |N.L| falloff rather than the Strauss F term
        d = hal_diffuse_lambert(ndl) * st.y;
        sp = st.x;
        float m = clamp(s.metallic, 0.0, 1.0);
        float fr = clamp(1.0 - abs(ndl), 0.0, 1.0);
        tint = vec3(1.0) + m * (1.0 - fr) * (s.diffuse - vec3(1.0));
        // Strauss returns before the soften pass on the CPU
        return vec4(d, tint * sp);
    } else if (model == 13) {               // MULTI_LAYER
        d = hal_diffuse_lambert(ndl);
        sp = hal_spec_blinn_phong(ndl, ndh, s.glossiness)
             + hal_spec_blinn_phong(ndl, ndh, max(s.glossiness * 0.15, 1.0))
               * 0.35;
    } else if (model == 14) {               // TOON
        d = hal_diffuse_toon(ndl, s.toon_size, s.toon_smooth, s.toon_steps);
        sp = hal_spec_toon(ndl, rdv, s.toon_size * 0.5, s.toon_smooth);
    } else if (model == 15) {               // TRANSLUCENT
        d = hal_diffuse_lambert(ndl)
            + max(-ndl, 0.0) * clamp(s.translucency, 0.0, 1.0);
    } else if (model == 16) {               // CONSTANT
        d = 0.0;                            // unlit: emission is added later
    } else {
        d = hal_diffuse_lambert(ndl);
    }

    sp = hal_soften(sp, ndl, s.soften);
    return vec4(d, tint * sp);
}
"""
