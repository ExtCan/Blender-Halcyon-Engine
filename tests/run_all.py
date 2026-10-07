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
    from . import test_legacy, test_render, test_shaders, utf8_console
    # R251 (1.90.0): the eleven wave-1 modules (pass 1) and the four
    # wave-2 modules (pass 2), after the legacy import
    from . import (test_r251_shadow, test_r251_lighting_fog, test_r251_lighting_lamps, test_r251_lighting_models,
                   test_r251_raster, test_r251_raster_wire, test_r251_texture, test_r251_transparency,
                   test_r251_sky_camera, test_r251_post_palette, test_r251_post_signal)
    from . import (test_r251_material, test_r251_material_nodes, test_r251_post_tape,
                   test_r251_post_codec)
    # R252 (1.91.0): the Console Emulation Shader
    from . import test_r252_console
    # R253: the Halo node
    from . import test_r253_halo_node
    utf8_console()
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
    print()
    print('=' * 66)
    print('LEGACY IMPORT')
    print('=' * 66)
    rc |= test_legacy.main()
    print()
    print('=' * 66)
    print('R251 SHADOW')
    print('=' * 66)
    rc |= test_r251_shadow.main()
    print()
    print('=' * 66)
    print('R251 LIGHTING -- FOG (LIGHT-A1 + LIGHT-A2)')
    print('=' * 66)
    rc |= test_r251_lighting_fog.main()
    print()
    print('=' * 66)
    print('R251 LIGHTING -- LAMPS (LIGHT-B1)')
    print('=' * 66)
    rc |= test_r251_lighting_lamps.main()
    print()
    print('=' * 66)
    print('R251 LIGHTING -- MODELS (LIGHT-B2)')
    print('=' * 66)
    rc |= test_r251_lighting_models.main()
    print()
    print('=' * 66)
    print('R251 RASTER (RAST-A1 + RAST-A2)')
    print('=' * 66)
    rc |= test_r251_raster.main()
    print()
    print('=' * 66)
    print('R251 RASTER -- AA RESOLVE / WIRE (RAST-B)')
    print('=' * 66)
    rc |= test_r251_raster_wire.main()
    print()
    print('=' * 66)
    print('R251 TEXTURE (TEX-1 + TEX-2)')
    print('=' * 66)
    rc |= test_r251_texture.main()
    print()
    print('=' * 66)
    print('R251 TRANSPARENCY (TRANS-1 + TRANS-2)')
    print('=' * 66)
    rc |= test_r251_transparency.main()
    print()
    print('=' * 66)
    print('R251 SKY-CAMERA (SKY + CAM)')
    print('=' * 66)
    rc |= test_r251_sky_camera.main()
    print()
    print('=' * 66)
    print('R251 POST-PALETTE (PAL-1 + PAL-2)')
    print('=' * 66)
    rc |= test_r251_post_palette.main()
    print()
    print('=' * 66)
    print('R251 POST-SIGNAL (SIG-1)')
    print('=' * 66)
    rc |= test_r251_post_signal.main()
    print()
    print('=' * 66)
    print('R251 MATERIAL (MAT-A)')
    print('=' * 66)
    rc |= test_r251_material.main()
    print()
    print('=' * 66)
    print('R251 MATERIAL -- NODES / REYES (MAT-B)')
    print('=' * 66)
    rc |= test_r251_material_nodes.main()
    print()
    print('=' * 66)
    print('R251 POST-SIGNAL -- TAPE / CABLE / PAL / CHROMA (SIG-2)')
    print('=' * 66)
    rc |= test_r251_post_tape.main()
    print()
    print('=' * 66)
    print('R251 POST-CODEC (SIG-3)')
    print('=' * 66)
    rc |= test_r251_post_codec.main()
    print()
    print('=' * 66)
    print('R252 CONSOLE EMULATION SHADER')
    print('=' * 66)
    rc |= test_r252_console.main()
    print()
    print('=' * 66)
    print('R253 HALO NODE')
    print('=' * 66)
    rc |= test_r253_halo_node.main()
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
