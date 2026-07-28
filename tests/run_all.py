"""Run everything without Blender:

    python3 -m halcyon.tests.run_all            # tests only
    python3 -m halcyon.tests.run_all --images   # tests, then write demo PNGs

The image pass needs Pillow, which Blender does not ship; it is only used to
write the demo files, never by the engine itself.
"""

import sys
import time

import numpy as np


def run_tests():
    from . import test_render, test_shaders
    rc = 0
    print('=' * 66)
    print('SHADER COMPILER')
    print('=' * 66)
    rc |= test_shaders.main()
    print()
    print('=' * 66)
    print('RENDERER')
    print('=' * 66)
    rc |= test_render.main()
    return rc


def write_images(outdir='.'):
    """Render the demo scene through several presets and save PNGs."""
    try:
        from PIL import Image
    except ImportError:
        print('\nPillow not installed; skipping the image pass.')
        return 0
    import os

    from ..core import post
    from ..core import render as R
    from ..core.settings import RenderSettings
    from ..presets.library import PRESETS, apply_preset
    from .scenebuild import demo_scene

    picks = ['INFINID_4', 'STUDIO_R4', 'PSX', 'N64', 'VOODOO', 'EGA',
             'IMAGINE_3', 'QUAKE_SW', 'TOASTER']
    os.makedirs(outdir, exist_ok=True)
    tiles = []
    print()
    print('=' * 66)
    print('DEMO IMAGES')
    print('=' * 66)
    for key in picks:
        st = RenderSettings()
        apply_preset(st, key)
        st.resolution_x, st.resolution_y = 240, 180
        st.output_scale = 'NONE'
        st.pixel_aspect_x = st.pixel_aspect_y = 1.0
        st.aa_samples = min(st.aa_samples, 4)
        t0 = time.time()
        img = post.process(R.render(demo_scene(st), st), st)[..., :3]
        # our row 0 is the bottom of the picture; PIL wants the top first
        arr = (np.clip(img[::-1], 0, 1) * 255).astype(np.uint8)
        im = Image.fromarray(arr).resize((240, 180), Image.NEAREST)
        path = os.path.join(outdir, f'halcyon_{key.lower()}.png')
        im.save(path)
        tiles.append(np.asarray(im, np.float32) / 255.0)
        print(f'  {PRESETS[key]["label"]:28s} -> {path}   {time.time() - t0:5.2f}s')

    W, H, gap = 240, 180, 4
    sheet = np.zeros((H * 3 + gap * 2, W * 3 + gap * 2, 3), np.float32)
    for i, t in enumerate(tiles):
        r, c = divmod(i, 3)
        sheet[r * (H + gap):r * (H + gap) + H,
              c * (W + gap):c * (W + gap) + W] = t
    contact = f'{outdir}/halcyon_contact_sheet.png'
    Image.fromarray((sheet * 255).astype(np.uint8)).save(contact)
    print(f'  contact sheet               -> {contact}')
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    started = time.time()
    rc = run_tests()
    if '--images' in argv:
        idx = argv.index('--images')
        outdir = argv[idx + 1] if len(argv) > idx + 1 else '.'
        write_images(outdir)
    print()
    print(f'total {time.time() - started:.1f}s')
    print('FAILURES ABOVE' if rc else 'everything passed')
    return rc


if __name__ == '__main__':
    sys.exit(main())
