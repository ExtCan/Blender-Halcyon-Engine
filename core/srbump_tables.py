"""R251 C023: the PowerVR2 (S,R) bump-mapping angle tables (Dreamcast
CLX2, Sega Naomi, 1998-2001).

A PVR2 BUMP texel is two 8-bit ANGLES -- S the elevation (0..pi/2), R the
azimuth (0..2pi) -- and the polygon's offset colour packs K1 (ambient),
K2 = sin T * H, K3 = cos T * H and Q (the light azimuth / 256); the
hardware evaluates `I = clamp(K1 + K2 sin S + K3 cos S cos(R - Q), 0, 1)`
per pixel (flycast's `pp_BumpMap`, the DCEmulation bump sample). Halcyon
quantises a tangent-space normal-map texel to those two angles through
tables built ONCE here in float64 and stored uint8 / float32, so both
roads fetch the same numbers:

    SIDX[k]      = round(asin(clip(2k/255 - 1, 0, 1)) * 2/pi * 255)   (a negative z clamps to S = 0)
    RTAB[ny][nx] = round((atan2(2ny/255 - 1, 2nx/255 - 1) + pi) / 2pi * 255) & 255
    SINS[k], COSS[k] = sin, cos(pi/2 * k/255);  COS256[k] = cos(2pi * k/256)

The azimuth origin is Halcyon's stated one -- R = 0 at azimuth -pi, R =
128 along +t -- pinned by test (the 8-bit encoding has no exact zero:
2*128/255 - 1 = +0.0039). The intensity reads `(R - Q) & 255` only, so an
origin that differs from KOS's by a constant offset in both R and Q
cancels; only a mirrored azimuth would move pixels (disclosed).
"""
import numpy as np

_k = np.arange(256, dtype=np.float64)
_z = np.clip(2.0 * _k / 255.0 - 1.0, 0.0, 1.0)
SIDX = np.round(np.arcsin(_z) * 2.0 / np.pi * 255.0).astype(np.uint8)
_ax = 2.0 * _k / 255.0 - 1.0
RTAB = (np.round((np.arctan2(_ax[:, None], _ax[None, :]) + np.pi)
                 / (2.0 * np.pi) * 255.0).astype(np.int64) & 255).astype(np.uint8)   # [ny][nx]
SINS = np.sin(np.pi / 2.0 * _k / 255.0).astype(np.float32)
COSS = np.cos(np.pi / 2.0 * _k / 255.0).astype(np.float32)
COS256 = np.cos(2.0 * np.pi * _k / 256.0).astype(np.float32)


def table_pixels():
    """`__sr_tables__`: a (1, 1024, 4) float32 image, red = SIDX[0..255] as
    floats, SINS[256..511], COSS[512..767], COS256[768..1023]."""
    px = np.zeros((1, 1024, 4), np.float32)
    px[0, 0:256, 0] = SIDX.astype(np.float32)
    px[0, 256:512, 0] = SINS
    px[0, 512:768, 0] = COSS
    px[0, 768:1024, 0] = COS256
    px[0, :, 3] = 1.0
    return px


def atan_pixels():
    """`__sr_atan__`: a (256, 256, 4) float32 image, red = RTAB[ny][nx]
    (row = ny, column = nx; 1 MiB, built once)."""
    px = np.zeros((256, 256, 4), np.float32)
    px[:, :, 0] = RTAB.astype(np.float32)
    px[:, :, 3] = 1.0
    return px


def light_constants(light_vec, strength):
    """(K1, K2, K3, Q) for ONE light direction in tangent space (t, b, n)
    and the bump strength H -- shared by both roads (the emitter bakes the
    same numbers the CPU node uses). float32 except `Q`, an int 0..255
    from a float64 atan2 (the CPU only; the GPU never computes it)."""
    L = np.asarray(light_vec, np.float64).reshape(3)
    ln = float(np.sqrt(max(float((L * L).sum()), 0.0)))
    L = (L / max(ln, 1e-12)).astype(np.float32)
    sinT = np.float32(np.clip(L[2], 0.0, 1.0))
    cosT = np.float32(np.sqrt(np.float32(max(float(np.float32(1.0) - sinT * sinT), 0.0))))
    Q = int(np.round((np.arctan2(float(L[1]), float(L[0])) + np.pi) / (2.0 * np.pi) * 255.0)) & 255
    H = np.float32(np.clip(float(strength), 0.0, 1.0))
    K1 = np.float32(np.float32(1.0) - H)
    K2 = np.float32(sinT * H)
    K3 = np.float32(cosT * H)
    return K1, K2, K3, Q


def intensity(n8_r, n8_g, n8_b, K1, K2, K3, Q):
    """The per-pixel PVR2 intensity on integer texel channels (arrays) with
    per-node OR per-pixel float32 K1/K2/K3 and int Q: six statements in
    the stated order, the GLSL twin's."""
    S = SIDX[np.asarray(n8_b, np.int64)]
    R = RTAB[np.asarray(n8_g, np.int64), np.asarray(n8_r, np.int64)]
    p1 = (np.float32(K2) * SINS[S]).astype(np.float32)
    a = (np.float32(K1) + p1).astype(np.float32)
    m = (R.astype(np.int64) - np.asarray(Q, np.int64)) & 255
    c1 = (COSS[S] * COS256[m]).astype(np.float32)
    p2 = (np.float32(K3) * c1).astype(np.float32)
    I = (a + p2).astype(np.float32)
    return np.clip(I, np.float32(0.0), np.float32(1.0)).astype(np.float32)
