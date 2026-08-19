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
    float toon_size2;
    float toon_smooth2;
    float bi_fresnel;
    float bi_fresnel_fac;
    float bi_slope;
    float bi_transp_fresnel;
    float bi_transp_blend;
    float bi_spectra;
    float bi_cubic;
    float bi_tangent;
    float shadow_receive;
    float cast_only;
    float shadows_only;
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
    // verbatim 2.79 (R155): nv clamps at 0 (View_A caps at pi/2) and
    // the smaller angle scales by 0.95 before tan -- the C's guard
    float s2 = roughness * roughness;
    float A = 1.0 - 0.5 * s2 / (s2 + 0.33);
    float B = 0.45 * s2 / (s2 + 0.09);
    float nv = max(clamp(ndv, -1.0, 1.0), 0.0);
    vec3 lp = normalize(l - n * ndl);
    vec3 vp = normalize(v - n * nv);
    float cosphi = max(dot(lp, vp), 0.0);
    float ti = acos(clamp(ndl, -1.0, 1.0));
    float tr = acos(nv);
    float alpha = max(ti, tr);
    float beta = min(ti, tr) * 0.95;
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

float hal_bi_spec_pow(float inp, float gloss)
{
    // 2.79's spec(): the integer-bit square-multiply power, verbatim
    // (R155). b1 floors at 0.01 on the first square, zeroes below
    // 0.001 twice up the ladder, an even hardness drops the x^1
    // factor, bit 256 squares once more. Hardness is shi->har -- a
    // SHORT -- so the float chain truncates here, exactly where the
    // C's assignment did.
    if (inp >= 1.0) { return 1.0; }
    if (inp <= 0.0) { return 0.0; }
    int hard = int(floor(gloss));
    float x = inp;
    float outv = ((hard & 1) == 0) ? 1.0 : x;
    float b1 = max(x * x, 0.01);
    if ((hard & 2) != 0)   { outv *= b1; }
    b1 *= b1;
    if ((hard & 4) != 0)   { outv *= b1; }
    b1 *= b1;
    if ((hard & 8) != 0)   { outv *= b1; }
    b1 *= b1;
    if ((hard & 16) != 0)  { outv *= b1; }
    b1 *= b1;
    if (b1 < 0.001) { b1 = 0.0; }
    if ((hard & 32) != 0)  { outv *= b1; }
    b1 *= b1;
    if ((hard & 64) != 0)  { outv *= b1; }
    b1 *= b1;
    if ((hard & 128) != 0) { outv *= b1; }
    if (b1 < 0.001) { b1 = 0.0; }
    if ((hard & 256) != 0) { b1 *= b1; outv *= b1; }
    return outv;
}

float hal_spec_bi_cooktorr(float ndl, float ndv, float ndh, float gloss)
{
    // Blender Internal's CookTorr_Spec, verbatim (R155):
    // spec(N.H, hard) / (0.1 + N.V); NO N.L gate, no upper clamp
    if (ndh < 0.0) { return 0.0; }
    float nv = max(ndv, 0.0);
    return hal_bi_spec_pow(ndh, gloss) / (0.1 + nv);
}

float hal_spec_bi_phong(float ndl, float ndh, float gloss)
{
    // Blender Internal's Phong_Spec, verbatim: the HALF-vector lobe
    // through spec(); no N.L gate in the C
    if (ndh <= 0.0) { return 0.0; }
    return hal_bi_spec_pow(ndh, gloss);
}

float hal_spec_bi_blinn(float ndl, float ndv, float ndh, float vdh,
                        float gloss, float ior)
{
    // Blender Internal's Blinn_Spec, verbatim (R155): spec_power is
    // (float)shi->har (int-truncated); the geometry pick is the C's
    // STRICT-compare chain (g stays 0 on ties); no upper clamp
    float refrac = max(ior, 0.0);
    if (refrac < 1.0 || ndl <= 0.01 || ndh < 0.0) { return 0.0; }
    float sp = max(floor(gloss), 1.0);
    float spow = (sp < 100.0) ? sqrt(1.0 / sp) : (10.0 / sp);
    float nh = max(ndh, 0.0);
    float nv = max(ndv, 0.01);
    float nl = max(ndl, 0.0);
    float vh = max(vdh, 0.01);
    float a = 1.0;
    float b = 2.0 * nh * nv / vh;
    float c = 2.0 * nh * nl / vh;
    float g = 0.0;
    if (a < b && a < c)      { g = a; }
    else if (b < a && b < c) { g = b; }
    else if (c < a && c < b) { g = c; }
    float p = sqrt(max(refrac * refrac + vh * vh - 1.0, 0.0));
    float f = ((p - vh) * (p - vh)) / ((p + vh) * (p + vh))
              * (1.0 + ((vh * (p + vh) - 1.0) * (vh * (p + vh) - 1.0))
                        / ((vh * (p - vh) + 1.0) * (vh * (p - vh) + 1.0)));
    float ang = acos(clamp(nh, -1.0, 1.0));
    float outv = f * g * exp(-(ang * ang)
                             / max(2.0 * spow * spow, 1e-8));
    return max(outv, 0.0);
}

float hal_bi_fresnel_fac(float t1, float grad, float fac)
{
    // Blender Internal's fresnel_fac(), transcribed
    if (fac == 0.0) { return 1.0; }
    float t2 = (t1 > 0.0) ? (1.0 + t1) : (1.0 - t1);
    t2 = grad + (1.0 - grad) * pow(max(t2, 0.0), fac);
    return clamp(t2, 0.0, 1.0);
}

float hal_diffuse_bi_fresnel(float ndl, float grad, float fac)
{
    // BI's Fresnel diffuse: fresnel_fac with lv pointing LAMP to
    // surface, so the argument is -N.L. Replaces the cosine outright
    return hal_bi_fresnel_fac(-ndl, grad, fac);
}

float hal_bi_cubic(float dif)
{
    // BI's Cubic Interpolation, with 2.79's own strictly-inside guard
    if (dif > 0.0 && dif < 1.0) {
        return 3.0 * dif * dif - 2.0 * dif * dif * dif;
    }
    return dif;
}

vec3 hal_bi_tangent_normal(vec3 t, vec3 l)
{
    // Tangent Shading's per-light fake normal: cross(tang,
    // cross(lv, tang)) = L - T*(T.L), normalized (falls back to L
    // where the light runs exactly along the tangent)
    vec3 n_eff = l - t * dot(t, l);
    float len = sqrt(max(dot(n_eff, n_eff), 1e-18));
    return n_eff / len;
}

vec3 hal_ramp_blend(int mode, vec3 col, float fac, vec3 rc)
{
    // 2.79's ramp_blend() (blenkernel material.c), mode in MA_RAMP_*
    // DNA order. The CPU twin is shading.bi_ramp_blend; every branch
    // keeps the C's per-channel conditionals and achromatic guards.
    float facm = 1.0 - fac;
    if (mode == 0) { return facm * col + fac * rc; }            // MIX
    if (mode == 1) { return col + fac * rc; }                   // ADD
    if (mode == 2) { return col * (facm + fac * rc); }          // MULT
    if (mode == 3) { return col - fac * rc; }                   // SUB
    if (mode == 4) {                                            // SCREEN
        return vec3(1.0) - (vec3(facm) + fac * (vec3(1.0) - rc))
               * (vec3(1.0) - col);
    }
    if (mode == 5) {                                            // DIV
        vec3 safe = facm * col + fac * col /
                    vec3(rc.x != 0.0 ? rc.x : 1.0,
                         rc.y != 0.0 ? rc.y : 1.0,
                         rc.z != 0.0 ? rc.z : 1.0);
        return vec3(rc.x != 0.0 ? safe.x : col.x,
                    rc.y != 0.0 ? safe.y : col.y,
                    rc.z != 0.0 ? safe.z : col.z);
    }
    if (mode == 6) { return facm * col + fac * abs(col - rc); } // DIFF
    if (mode == 7) {                                            // DARK
        return min(col, rc + (vec3(1.0) - rc) * facm);
    }
    if (mode == 8) { return max(col, fac * rc); }               // LIGHT
    if (mode == 9) {                                            // OVERLAY
        vec3 low = col * (facm + 2.0 * fac * rc);
        vec3 high = vec3(1.0) - (vec3(facm) + 2.0 * fac
                    * (vec3(1.0) - rc)) * (vec3(1.0) - col);
        return vec3(col.x < 0.5 ? low.x : high.x,
                    col.y < 0.5 ? low.y : high.y,
                    col.z < 0.5 ? low.z : high.z);
    }
    if (mode == 10) {                                           // DODGE
        vec3 tmp = vec3(1.0) - fac * rc;
        vec3 lifted = vec3(
            tmp.x <= 0.0 ? 1.0 : min(col.x / tmp.x, 1.0),
            tmp.y <= 0.0 ? 1.0 : min(col.y / tmp.y, 1.0),
            tmp.z <= 0.0 ? 1.0 : min(col.z / tmp.z, 1.0));
        return vec3(col.x != 0.0 ? lifted.x : col.x,
                    col.y != 0.0 ? lifted.y : col.y,
                    col.z != 0.0 ? lifted.z : col.z);
    }
    if (mode == 11) {                                           // BURN
        vec3 tmp = vec3(facm) + fac * rc;
        return vec3(
            tmp.x <= 0.0 ? 0.0 : clamp(1.0 - (1.0 - col.x) / tmp.x,
                                       0.0, 1.0),
            tmp.y <= 0.0 ? 0.0 : clamp(1.0 - (1.0 - col.y) / tmp.y,
                                       0.0, 1.0),
            tmp.z <= 0.0 ? 0.0 : clamp(1.0 - (1.0 - col.z) / tmp.z,
                                       0.0, 1.0));
    }
    if (mode == 12) {                                           // HUE
        vec3 chsv = hal_rgb2hsv(rc);
        if (chsv.y == 0.0) { return col; }
        vec3 rhsv = hal_rgb2hsv(col);
        vec3 t = hal_hsv2rgb(vec3(chsv.x, rhsv.y, rhsv.z));
        return facm * col + fac * t;
    }
    if (mode == 13) {                                           // SAT
        vec3 rhsv = hal_rgb2hsv(col);
        if (rhsv.y == 0.0) { return col; }
        vec3 chsv = hal_rgb2hsv(rc);
        return hal_hsv2rgb(vec3(rhsv.x, facm * rhsv.y + fac * chsv.y,
                                rhsv.z));
    }
    if (mode == 14) {                                           // VAL
        vec3 rhsv = hal_rgb2hsv(col);
        vec3 chsv = hal_rgb2hsv(rc);
        return hal_hsv2rgb(vec3(rhsv.x, rhsv.y,
                                facm * rhsv.z + fac * chsv.z));
    }
    if (mode == 15) {                                           // COLOR
        vec3 chsv = hal_rgb2hsv(rc);
        if (chsv.y == 0.0) { return col; }
        vec3 rhsv = hal_rgb2hsv(col);
        vec3 t = hal_hsv2rgb(vec3(chsv.x, chsv.y, rhsv.z));
        return facm * col + fac * t;
    }
    if (mode == 16) {                                           // SOFT
        vec3 scr = vec3(1.0) - (vec3(1.0) - rc) * (vec3(1.0) - col);
        return facm * col + fac * ((vec3(1.0) - col) * rc * col
                                   + col * scr);
    }
    if (mode == 17) {                                           // LINEAR
        return col + fac * vec3(
            rc.x > 0.5 ? 2.0 * (rc.x - 0.5) : 2.0 * rc.x - 1.0,
            rc.y > 0.5 ? 2.0 * (rc.y - 0.5) : 2.0 * rc.y - 1.0,
            rc.z > 0.5 ? 2.0 * (rc.z - 0.5) : 2.0 * rc.z - 1.0);
    }
    return col;
}

float hal_diffuse_bi_toon(float ndl, float size, float smoothv)
{
    // BI's Toon_Diff: a hard angular band on acos(N.L)
    float ang = acos(clamp(ndl, -1.0, 1.0));
    if (ang < size) { return 1.0; }
    if (smoothv <= 0.0 || ang >= size + smoothv) { return 0.0; }
    return clamp(1.0 - (ang - size) / max(smoothv, 1e-9), 0.0, 1.0);
}

float hal_diffuse_bi_minnaert(float ndl, float ndv, float darkness)
{
    // BI's Minnaert_Diff, both branches
    float nl = max(ndl, 0.0);
    float nv = max(ndv, 0.0);
    if (darkness <= 1.0) {
        return nl * pow(max(nv * nl, 0.1), darkness - 1.0);
    }
    return nl * pow(max(1.001 - nv, 1e-6), darkness - 1.0);  // 1.001: the C's constant
}

float hal_spec_bi_toon(float ndl, float ndh, float size, float smoothv)
{
    // BI's Toon_Spec, verbatim: the band on acos(N.H); no N.L gate
    float ang = acos(clamp(ndh, -1.0, 1.0));
    if (ang < size) { return 1.0; }
    if (smoothv <= 0.0 || ang >= size + smoothv) { return 0.0; }
    return clamp(1.0 - (ang - size) / max(smoothv, 1e-9), 0.0, 1.0);
}

float hal_spec_bi_wardiso(float ndl, float ndv, float ndh, float rms)
{
    // BI's WardIso_Spec, verbatim (R155): nl/nv/nh CLAMP to 0.001 --
    // the C never gates -- and there is no upper clamp
    float nh = max(ndh, 0.001);
    float nv = max(ndv, 0.001);
    float nl = max(ndl, 0.001);
    float alpha = max(rms, 0.001);
    float angle = tan(acos(clamp(nh, -1.0, 1.0)));
    return nl * (1.0 / (4.0 * 3.14159265 * alpha * alpha))
             * (exp(-(angle * angle) / (alpha * alpha))
                / sqrt(max(nv * nl, 1e-8)));
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
float hal_bi_matrix_diffuse(int di, HalcyonSurface s, vec3 n, vec3 l,
                            vec3 v, float ndl, float ndv)
{
    // the BI node's diffuse ladder, callable twice: once forward, once
    // through the flipped normal for BI's translucency
    if (di == 1) {
        return hal_diffuse_oren_nayar(ndl, ndv, l, v, n, s.roughness);
    }
    if (di == 2) {
        return hal_diffuse_bi_toon(ndl, s.toon_size, s.toon_smooth);
    }
    if (di == 3) {
        return hal_diffuse_bi_minnaert(ndl, ndv, s.roughness);
    }
    if (di == 4) {
        return hal_diffuse_bi_fresnel(ndl, s.bi_fresnel,
                                      s.bi_fresnel_fac);
    }
    return hal_diffuse_lambert(ndl);
}

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
    // The CPU's evaluate() folds the specular COLOUR into its specular term
    // (spec_col), and the light loop then multiplies only specular_level. So
    // the tint starts as s.specular here and Metal overrides it with the
    // diffuse, exactly as the CPU does -- it used to start as white and be
    // multiplied by s.specular in the loop afterwards, which double-tinted
    // Metal and Strauss. Invisible while every test used a white specular.
    vec3 tint = s.specular;

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
    } else if (model == 18) {               // BI_COOKTORR
        d = hal_diffuse_lambert(ndl);
        sp = hal_spec_bi_cooktorr(ndl, ndv, ndh, s.glossiness);
    } else if (model == 19) {               // BI_PHONG
        d = hal_diffuse_lambert(ndl);
        sp = hal_spec_bi_phong(ndl, ndh, s.glossiness);
    } else if (model == 20) {               // BI_BLINN
        d = hal_diffuse_lambert(ndl);
        sp = hal_spec_bi_blinn(ndl, ndv, ndh, vdh, s.glossiness, s.ior);
    } else if (model >= 100) {              // the BI MATERIAL NODE:
        // 100 + diffuse*10 + spec, both menus in DNA order. Every
        // branch is the transcribed 2.79 formula; the CPU twin is
        // shading.bi_matrix_terms. The decode is written floor-style
        // rather than as (model-100)/10: real drivers divide ints,
        // the parity simulator divides floats, and this form gives
        // the same digits under BOTH semantics.
        float fm = float(model) - 100.0;
        int di = int(floor(fm * 0.1 + 0.001));
        int si = int(fm - float(di) * 10.0 + 0.5);
        vec3 n_use = n;
        if (s.bi_tangent > 0.5) {
            // Tangent Shading swaps the normal per light, both lobes
            n_use = hal_bi_tangent_normal(s.tangent, l);
            ndl = dot(n_use, l);
            ndv = dot(n_use, v);
            ndh = dot(n_use, h);
        }
        d = hal_bi_matrix_diffuse(di, s, n_use, l, v, ndl, ndv);
        if (s.translucency > 0.0) {
            // BI's translucency: the same diffuse shader through the
            // flipped normal, scaled by the slider
            d += clamp(s.translucency, 0.0, 1.0)
                 * hal_bi_matrix_diffuse(di, s, -n_use, l, v,
                                         -ndl, -ndv);
        }
        if (s.bi_cubic > 0.5) { d = hal_bi_cubic(d); }
        if (si == 1) {
            sp = hal_spec_bi_phong(ndl, ndh, s.glossiness);
        } else if (si == 2) {
            sp = hal_spec_bi_blinn(ndl, ndv, ndh, vdh, s.glossiness,
                                   s.ior);
        } else if (si == 3) {
            sp = hal_spec_bi_toon(ndl, ndh, s.toon_size2,
                                  s.toon_smooth2);
        } else if (si == 4) {
            sp = hal_spec_bi_wardiso(ndl, ndv, ndh, s.bi_slope);
        } else {
            sp = hal_spec_bi_cooktorr(ndl, ndv, ndh, s.glossiness);
        }
    } else {
        d = hal_diffuse_lambert(ndl);
    }

    sp = hal_soften(sp, ndl, s.soften);
    return vec4(d, tint * sp);
}
"""


# ---- Blender Internal subsurface scattering: the octree gather ----------
#
# The scatter tree rides a data texture (core/sss.ScatterTree.pack_gpu's
# layout) and the traversal is STACKLESS: nodes linearised in preorder --
# the C's own 0..7 child visit order -- with hit/miss links, exactly the
# threading gpu/rtrace.py uses for the BVH, because the kernels support
# divergent while-loops but not dynamic array writes. The 'self' chain
# (SUBNODE_INDEX matches forcing descent past the error criterion) is a
# per-node box of the ancestors' split half-spaces, >= on the upper side
# exactly like SUBNODE_INDEX; collapsed cells contributed no split and
# therefore no half-space, matching the C where the collapse is invisible
# to the traversal. The Rd lookup keeps the C's three branches: the fine
# squared-distance table, the far distance table, and the true dipole
# past both.
SSS_GLSL = """
uniform sampler2D hal_sss_tree;

vec4 hal_sss_fetch(int i)
{
    return texelFetch(hal_sss_tree, ivec2(i % 2048, i / 2048), 0);
}

vec3 hal_sss_rd_exact(float rr)
{
    vec3 outv = vec3(0.0);
    vec4 p0 = hal_sss_fetch(3);
    vec4 p1 = hal_sss_fetch(4);
    vec4 p2 = hal_sss_fetch(5);
    float sr = sqrt(rr + p0.x * p0.x);
    float sv = sqrt(rr + p0.y * p0.y);
    outv.x = (1.0 / 12.566370614359172) *
        (p0.x * (1.0 + p0.z * sr) * exp(-p0.z * sr) / (sr * sr * sr)
       + p0.y * (1.0 + p0.z * sv) * exp(-p0.z * sv) / (sv * sv * sv));
    sr = sqrt(rr + p1.x * p1.x);
    sv = sqrt(rr + p1.y * p1.y);
    outv.y = (1.0 / 12.566370614359172) *
        (p1.x * (1.0 + p1.z * sr) * exp(-p1.z * sr) / (sr * sr * sr)
       + p1.y * (1.0 + p1.z * sv) * exp(-p1.z * sv) / (sv * sv * sv));
    sr = sqrt(rr + p2.x * p2.x);
    sv = sqrt(rr + p2.y * p2.y);
    outv.z = (1.0 / 12.566370614359172) *
        (p2.x * (1.0 + p2.z * sr) * exp(-p2.z * sr) / (sr * sr * sr)
       + p2.y * (1.0 + p2.z * sv) * exp(-p2.z * sv) / (sv * sv * sv));
    return outv;
}

vec3 hal_sss_rd(float rr)
{
    vec4 offs = hal_sss_fetch(6);
    if (rr > 100000000.0) {
        return hal_sss_rd_exact(rr);
    }
    if (rr > 100.0) {
        float r = sqrt(rr);
        float indexf = r * 1.0;
        int index = int(indexf);
        if (index >= 10000) {
            return hal_sss_rd_exact(rr);
        }
        float t = indexf - float(index);
        int b = int(offs.w) + index;
        return hal_sss_fetch(b).rgb * (1.0 - t)
             + hal_sss_fetch(b + 1).rgb * t;
    }
    float indexf = rr * 100.0;
    int index = int(indexf);
    float t = indexf - float(index);
    int b = int(offs.z) + index;
    return hal_sss_fetch(b).rgb * (1.0 - t)
         + hal_sss_fetch(b + 1).rgb * t;
}

vec3 hal_sss_sample(vec3 co)
{
    vec4 hdr = hal_sss_fetch(0);
    if (hdr.x < 0.5) { return vec3(0.0); }
    vec4 offs = hal_sss_fetch(6);
    vec4 vr0 = hal_sss_fetch(7);
    vec4 vr1 = hal_sss_fetch(8);
    vec4 vr2 = hal_sss_fetch(9);
    vec3 cam = vec3(dot(vr0.xyz, co) + vr0.w,
                    dot(vr1.xyz, co) + vr1.w,
                    dot(vr2.xyz, co) + vr2.w);
    vec3 sco = cam / hdr.z;
    float error = hdr.w;
    vec3 rad = vec3(0.0);
    vec3 brad = vec3(0.0);
    vec3 rdsum = vec3(0.0);
    vec3 brdsum = vec3(0.0);
    float node = 0.0;
    float guard = 0.0;
    while (node > -0.5 && guard < 400000.0) {
        guard += 1.0;
        int nb = int(offs.x) + int(node) * 6;
        vec4 t0 = hal_sss_fetch(nb);
        vec4 t1 = hal_sss_fetch(nb + 1);
        vec4 t2 = hal_sss_fetch(nb + 2);
        vec4 t3 = hal_sss_fetch(nb + 3);
        vec4 blo = hal_sss_fetch(nb + 4);
        vec4 bhi = hal_sss_fetch(nb + 5);
        bool selfp = (sco.x >= blo.x && sco.y >= blo.y && sco.z >= blo.z
                   && sco.x < bhi.x && sco.y < bhi.y && sco.z < bhi.z);
        vec3 dvec = sco - t0.xyz;
        float rr = dot(dvec, dvec);
        if (!selfp && (t1.w + t2.w) <= error * rr) {
            vec3 rd = hal_sss_rd(rr);
            if (t1.w > 0.0) {
                vec3 frd = rd * t1.w;
                rad += t1.xyz * frd;
                rdsum += frd;
            }
            if (t2.w > 0.0) {
                vec3 brd2 = rd * t2.w;
                brad += t2.xyz * brd2;
                brdsum += brd2;
            }
            node = t3.y;
        } else if (t0.w > 0.5) {
            int ps = int(offs.y) + int(t3.z) * 2;
            int pc = int(t3.w);
            for (int i = 0; i < pc; i++) {
                vec4 p0 = hal_sss_fetch(ps + i * 2);
                vec4 p1 = hal_sss_fetch(ps + i * 2 + 1);
                vec3 dp = sco - p0.xyz;
                float prr = dot(dp, dp);
                vec3 prd = hal_sss_rd(prr) * abs(p0.w);
                if (p0.w < 0.0) {
                    brad += p1.xyz * prd;
                    brdsum += prd;
                } else {
                    rad += p1.xyz * prd;
                    rdsum += prd;
                }
            }
            node = t3.y;
        } else {
            node = t3.x;
        }
    }
    vec4 wts = hal_sss_fetch(1);
    vec4 colc = hal_sss_fetch(2);
    rad = rad * wts.x;
    brad = brad * wts.y;
    vec3 backrad = rad + brad;
    vec3 backrdsum = rdsum + brdsum;
    vec3 outv = rad;
    if (rdsum.x > 1e-16) { outv.x = colc.x * rad.x / rdsum.x; }
    if (rdsum.y > 1e-16) { outv.y = colc.y * rad.y / rdsum.y; }
    if (rdsum.z > 1e-16) { outv.z = colc.z * rad.z / rdsum.z; }
    vec3 bb = backrad;
    if (backrdsum.x > 1e-16) { bb.x = colc.x * backrad.x / backrdsum.x; }
    if (backrdsum.y > 1e-16) { bb.y = colc.y * backrad.y / backrdsum.y; }
    if (backrdsum.z > 1e-16) { bb.z = colc.z * backrad.z / backrdsum.z; }
    return max(outv, bb);
}
"""
