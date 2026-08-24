"""R203: the Bryce3D 1995 .mat preset library, properly parsed.

The field supplied the complete preset collection (34 category files,
MetaTools CCmF containers, December 1995) and asked for a real parse.
This module is it, cracked from the bytes up:

**The container** ("CCmF - Composite File Management System", Andrea
Pessino, vers. 5): after the 512-byte signature block and the
'Resource Composite File' header sits a DIRECTORY of {u32 size}
{u32 offset} big-endian pairs, each followed by the constant
0x01000000; a {0,0} pair separates object groups and a class registry
(name -> tag id pairs) sits wherever the directory says. Every record
begins {u32 size+12}{u32 size}{u32 0} followed by its payload.

**The previews** (matprevcimage records, codec name 'rle2' in the
payload head): 96x96, FOUR PLANES of 9216 bytes -- coverage, R, dG,
dB -- where green and blue are DELTA-CODED from the previous channel
modulo 256 (G = R + dG, B = G + dB), the whole thing compressed with
byte RLE: a count byte under 0x80 repeats the next byte that many
times; 0x80+n copies n literal bytes. Confirmed by decoding Polished
Silver to its grey chrome sphere reflecting the preview stage's blue
sky and green ground.

**The names and notes** (CPresetDescr records): the preset name sits
after an {0x14D}{0x141} size frame (CPresetDescr is always 321 bytes);
its description text follows in the same record.

**The material channels** (R204 -- "1 to 1, not 'eh close enough'"):
every preset's 976-byte C3dBaseMaterial record decodes now. The
layout was pinned by BYTE-DIFFING the teaching presets against their
own 1995 manual text -- Specularity Lesson #1's "coefficients of 12,
12, and 12" is exactly 0.047*255 at offset 468, Lesson #2's "Red
coefficient pushed up to 200" is 0.784*255 there, Warm Gold's "the
object itself is red, but the ambient orange" names offsets 300 and
356, Metallic Chrome's "deep blue diffuse ... less reflectivity"
names 300 and 740, the Light->Heavy Glass ladder walks offset 884
through 1.12/1.2/1.42/1.52/1.68 (the refractive index; Water is
1.33 and Diamond 2.55 on the nose). Values are float32 LITTLE-endian,
colours as r,g,b triples in n/255 steps. A channel whose slot is
preceded by the 'brtx' magic ('xtrb' in the file) is texture-driven.

**The textures** (3dTxt_TxtData, 1588 bytes): name at +8; a global
f64 scale triple at +276; TWO component blocks at +500 and +596
(stride 96), each with an int frequency triple (+24), an
amplitude/offset float pair (+56), an octave count (+64), THREE
palette colours stored B,G,R,pad (+72) and a colour count (+84);
rotation f64s live near +684 (the Classic Checkerboard's 45-degree
diamond spin sits there in plain double precision).

bpy-free, like everything else under core/.
"""

import re
import struct

import numpy as np

_NAME_FRAME = re.compile(
    rb'\x00\x00\x01M\x00\x00\x01A\x00\x00\x00\x00([\x20-\x7e]{3,40})')


def _rle2(src, want=36864):
    """The 'rle2' byte RLE: count<0x80 repeats, 0x80+n copies n."""
    out = bytearray()
    i = 64                      # the 64-byte codec header ('rle2' + pad)
    n = len(src)
    while i < n and len(out) < want:
        c = src[i]
        i += 1
        if c < 0x80:
            if i >= n:
                break
            out += bytes([src[i]]) * c
            i += 1
        else:
            out += src[i:i + c - 0x80]
            i += c - 0x80
    return bytes(out)


def decode_preview(payload):
    """One matprevcimage payload -> (96, 96, 3) float32 RGB in 0..1."""
    dec = _rle2(payload)
    if len(dec) < 36864:
        dec = dec.ljust(36864, b'\x00')
    p = np.frombuffer(dec[:36864], np.uint8).reshape(4, 96, 96)
    r = p[1].astype(np.int32)
    g = (r + p[2]) % 256
    b = (g + p[3]) % 256
    rgb = np.stack([r, g, b], axis=-1).astype(np.float32) / 255.0
    return rgb


def parse_mat(data):
    """Parse one .mat file -> list of preset dicts.

    Each dict: {'name', 'description', 'preview' ((96,96,3) float32
    or None)}. Presets are matched to previews by stream order (the
    thumbnail record precedes its descriptor in every file checked).
    """
    if isinstance(data, (str, bytes)) and not isinstance(data, bytes):
        data = open(data, 'rb').read()
    d = data
    names = [(m.start(), m.group(1).decode('ascii', 'ignore').strip())
             for m in _NAME_FRAME.finditer(d)]
    # descriptions: printable text later in the same descriptor record
    presets = []
    for k, (pos, nm) in enumerate(names):
        end = names[k + 1][0] if k + 1 < len(names) else len(d)
        seg = d[pos + len(nm):min(end, pos + 340)]
        desc = ''
        for m in re.finditer(rb'[\x20-\x7e\t]{18,}', seg):
            t = m.group().decode('ascii', 'ignore').strip()
            if not re.match(r'^[\W\d]', t):
                desc = t
                break
        presets.append({'name': nm, 'description': desc,
                        'preview': None, '_pos': pos})
    # previews: every rle2 record, assigned to the nearest FOLLOWING
    # descriptor (thumb precedes descr within a preset group)
    for m in re.finditer(rb'rle2\x00', d):
        b = m.start()
        try:
            size = struct.unpack('>I', d[b - 12:b - 8])[0] - 12
        except struct.error:
            continue
        if size <= 64 or size > 4_000_000:
            continue
        after = [p for p in presets if p['_pos'] > b]
        if not after:
            continue
        target = min(after, key=lambda p: p['_pos'])
        if target['preview'] is None:
            target['preview'] = decode_preview(d[b:b + size])
    for p in presets:
        p.pop('_pos', None)
    return presets


# ------------------------------------------------- R204: the deep decode
#
# The record walk. Every CCmF record is framed {u32 size+12}{u32 size}
# {u32 0} big-endian; records are contiguous, so once one frame is found
# the rest follow by jumping payloads. Preset clusters read straight off
# the stream order: [textures..., material(976), preview(rle2),
# imgeiv32, descriptor(321)] -- everything since the previous descriptor
# belongs to the preset the next descriptor names.

#: slot offsets inside the 976-byte C3dBaseMaterial record.
#: Colours are 3 x f32 LE; scalars 1 x f32 LE; all in n/255 steps
#: except ior (a real index) and metallic (a 0/1 flag).
MAT_COLORS = {'diffuse_color': 300, 'ambient_color': 356,
              'specular_color': 412, 'specular_coef': 468,
              'transparent_color': 524, 'volume_color': 872}
MAT_SCALARS = {'diffuse': 580, 'ambient': 620, 'specular': 660,
               'transparency': 700, 'reflection': 740, 'bump': 780,
               'metallic': 820, 'volume_density': 860, 'ior': 884}


def _f32(m, off, end='<'):
    try:
        return float(struct.unpack(end + 'f', m[off:off + 4])[0])
    except struct.error:
        return 0.0


def _endian(blob, offsets):
    """'<' or '>' -- whichever byte order reads sanely at `offsets`.

    Most of the library was saved on x86 (little-endian floats), but
    Extra Metal.mat came off a 68k Mac and stores every float BIG
    endian. The right order is simply the one under which the known
    slots hold 0..1-ish numbers.
    """
    def score(end):
        s = 0
        for off in offsets:
            v = abs(_f32(blob, off, end))
            # a CLEAN number: exactly zero, or a magnitude a 1995 dial
            # could hold. Denormals (byte-swapped floats decode to
            # ~1e-41) and huge values both fail, which is the tell.
            if v == 0.0 or 1e-3 <= v <= 512.0:
                s += 1
        return s
    return '<' if score('<') >= score('>') else '>'


def decode_material(blob):
    """One 976-byte C3dBaseMaterial record -> channel dict.

    Returns {'<channel>': value...} plus 'textured': the set of channel
    names whose slot carries the 'brtx' texture binding (the channel's
    value comes from a texture rather than the flat number). Slots
    holding garbage (a handful of light-gel presets serialized live
    pointers) fall back to sane defaults rather than poisoning the
    preset.
    """
    if not blob or len(blob) != 976:
        return None
    end = _endian(blob, [300, 304, 308, 580, 620, 660, 884])
    out = {'endian': end}
    textured = set()
    for nm, off in MAT_COLORS.items():
        c = [_f32(blob, off + 4 * i, end) for i in range(3)]
        if all(-0.002 <= v <= 1.002 for v in c):
            out[nm] = tuple(min(max(v, 0.0), 1.0) for v in c)
        else:
            out[nm] = (0.0, 0.0, 0.0)
        if blob[off - 28:off - 24] in (b'xtrb', b'brtx'):
            textured.add(nm)
    for nm, off in MAT_SCALARS.items():
        v = _f32(blob, off, end)
        if nm == 'ior':
            out[nm] = min(max(v, 1.0), 3.0) if 0.9 <= v <= 3.2 else 1.0
        elif nm == 'metallic':
            out[nm] = 1.0 if abs(v - 1.0) < 0.01 else 0.0
        else:
            out[nm] = min(max(v, 0.0), 1.0) if -0.002 <= v <= 1.002 \
                else 0.0
        if blob[off - 28:off - 24] in (b'xtrb', b'brtx'):
            textured.add(nm)
    out['textured'] = textured
    return out


def decode_texture(blob):
    """One 1588-byte 3dTxt_TxtData record -> texture dict.

    {'name', 'scale' (f64 triple at +276), 'rotation' (degrees, when a
    plausible one sits at +684), 'components': up to two dicts with
    'freq' (int triple), 'amplitude', 'offset', 'octaves', 'colors'
    (list of (r,g,b) 0..1, count-limited), and every colour the whole
    texture names under 'palette'.
    """
    if not blob or len(blob) < 700:
        return None
    nm_end = blob.find(b'\x00', 8)
    name = blob[8:nm_end if 8 < nm_end <= 40 else 8].decode(
        'ascii', 'ignore').strip()
    end = _endian(blob, [556, 560, 652, 656])
    try:
        scale = struct.unpack(end + '3d', blob[276:300])
    except struct.error:
        scale = (1.0, 1.0, 1.0)
    if not all(1e-6 < abs(s) < 1e4 for s in scale):
        scale = (1.0, 1.0, 1.0)
    comps = []
    palette = []
    for base in (500, 596):
        if base + 96 > len(blob):
            break
        kind = struct.unpack(end + '6i', blob[base:base + 24])
        freq = struct.unpack(end + '3i', blob[base + 36:base + 48])
        amp = _f32(blob, base + 56, end)
        off = _f32(blob, base + 60, end)
        octaves = struct.unpack(end + 'i', blob[base + 64:base + 68])[0]
        ncol = struct.unpack(end + 'i', blob[base + 84:base + 88])[0]
        cols = []
        for k in range(3):
            q = blob[base + 72 + 4 * k:base + 76 + 4 * k]
            # one u32 0x00RRGGBB: pad-last b,g,r on x86 saves,
            # pad-first r,g,b on 68k saves
            b, g, r = (q[0], q[1], q[2]) if end == '<' else \
                (q[3], q[2], q[1])
            cols.append((r / 255.0, g / 255.0, b / 255.0))
        ncol = ncol if 0 <= ncol <= 3 else 3
        comps.append({'kind': tuple(kind), 'freq': tuple(freq),
                      'amplitude': amp, 'offset': off,
                      'octaves': octaves if 0 <= octaves <= 12 else 0,
                      'ncol': ncol, 'colors': cols})
        for c in cols[:max(ncol, 1)]:
            if c not in palette:
                palette.append(c)
    rot = 0.0
    try:
        r0 = struct.unpack(end + 'd', blob[680:688])[0]
        if 1e-6 < abs(r0) <= 360.0 and abs(r0) == abs(r0):    # not NaN
            rot = float(r0)
    except (struct.error, ValueError):
        pass
    return {'name': name, 'scale': tuple(float(s) for s in scale),
            'rotation': rot, 'components': comps, 'palette': palette}


def walk_records(data):
    """Every framed record in a .mat file -> [(payload_offset, size)]."""
    recs = []
    pos = 0
    n = len(data)
    while pos + 12 <= n:
        try:
            a, b, c = struct.unpack('>III', data[pos:pos + 12])
        except struct.error:
            break
        if c == 0 and a == b + 12 and 0 < b <= n - pos - 12:
            recs.append((pos, b))
            pos = pos + 12 + b
        else:
            pos += 1
    return recs


def parse_mat_full(data):
    """The complete parse: names, notes, previews AND decoded data.

    Returns the parse_mat() list with two more keys per preset:
    'material' (the decode_material dict or None) and 'textures'
    (decoded 3dTxt dicts, file order). Cluster rule: all records since
    the previous 321-byte descriptor belong to the preset the next
    descriptor names -- the layout every one of the 34 files follows.
    """
    if isinstance(data, str):
        data = open(data, 'rb').read()
    d = data
    presets = []
    acc = []
    for pos, size in walk_records(d):
        pay = d[pos + 12:pos + 12 + size]
        if size == 321 and re.match(rb'[\x20-\x7e]{3,}', pay):
            nm = re.match(rb'([\x20-\x7e]{3,40})',
                          pay).group(1).decode('ascii', 'ignore').strip()
            desc = ''
            for m in re.finditer(rb'[\x20-\x7e\t]{18,}', pay[len(nm):]):
                t = m.group().decode('ascii', 'ignore').strip('\t ')
                if not re.match(r'^[\W\d]', t):
                    desc = t
                    break
            mat = None
            texs = []
            prev = None
            for psize, ppay in acc:
                if psize == 976 and mat is None:
                    mat = decode_material(ppay)
                elif ppay[:4] == b'rle2':
                    prev = decode_preview(ppay)
                elif psize == 1588:
                    t = decode_texture(ppay)
                    if t is not None:
                        texs.append(t)
            presets.append({'name': nm, 'description': desc,
                            'preview': prev, 'material': mat,
                            'textures': texs})
            acc = []
        else:
            acc.append((size, pay))
    return presets


def analyze_preview(rgb):
    """Material character measured off the 1995 preview render.

    The preview stage is constant across the library -- a sphere over
    a ground plane against sky -- so the sphere's pixels are the
    material under known light. Returns a dict of estimated channels:
    diffuse (r,g,b), specular_level, glossiness, reflectivity,
    transparency, emissive -- each 0..1-ish, from measurements named
    in comments. Estimates, and labelled as such: the honest source
    is the picture, not the undecoded floats.
    """
    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    # the sphere sits centred, slightly high; a conservative disc
    disc = (yy - 44.0) ** 2 + (xx - 48.0) ** 2 < 30.0 ** 2
    sph = rgb[disc]
    if sph.size == 0:
        return None
    lum = sph @ np.array([0.299, 0.587, 0.114], np.float32)
    order = np.argsort(lum)
    # diffuse: the mid-tones -- highlight and core shadow excluded
    mid = sph[order[int(0.25 * len(order)):int(0.75 * len(order))]]
    diffuse = mid.mean(axis=0)
    # specular: how far the top percentile outruns the mid tone
    hi = float(lum[order[int(0.99 * (len(order) - 1))]])
    midl = float(np.median(lum))
    spec = float(np.clip((hi - midl) * 1.6, 0.0, 1.0))
    # glossiness: highlight concentration -- fraction of pixels
    # within 90% of the peak; tight highlight = high gloss
    tight = float((lum > hi * 0.92).mean())
    gloss = float(np.clip(1.0 - tight * 12.0, 0.05, 1.0))
    # reflectivity: sky blue leaking into the sphere's upper half
    top = rgb[np.logical_and(disc, yy < 40)]
    refl = 0.0
    if top.size:
        blue_excess = float(np.clip(top[:, 2].mean()
                                    - top[:, 0].mean(), 0, 1))
        refl = float(np.clip(blue_excess * 2.2, 0.0, 1.0))
    # transparency: ground colours continuing THROUGH the lower half
    bot = rgb[np.logical_and(disc, yy > 52)]
    ground = rgb[80:92, 4:24].reshape(-1, 3).mean(axis=0)
    trans = 0.0
    if bot.size:
        near = np.abs(bot - ground[None, :]).mean()
        trans = float(np.clip(0.5 - near * 1.8, 0.0, 1.0))
    sat = float(np.clip((diffuse.max() - diffuse.min()) * 2.0, 0, 1))
    return {'diffuse': tuple(round(float(c), 3) for c in diffuse),
            'specular_level': round(spec, 3),
            'glossiness': round(gloss, 3),
            'reflectivity': round(refl, 3),
            'transparency': round(trans, 3),
            'saturation': round(sat, 3)}
