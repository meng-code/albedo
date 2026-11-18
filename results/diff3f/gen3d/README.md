# Full meshes generated from the single photos (Hunyuan3D-2.1 and TRELLIS.2)

The "generate the full mesh first, then match" route: two single-image-to-3D generators turn each
of the 14 photos that have a relief mesh (`outputs/mesh_depth/<name>/relief.ply`) into a full
textured mesh, which can become the source of a full-to-full correspondence. Run on the GPU server
(NVIDIA A40-24Q, a 24 GB vGPU slice; 8 vCPUs; 16 GB RAM) on 2026-10-23/24.

Inputs: the RGBA photos with the background removed by `prepare_images.py`
(`~/IDArb/mydata/my_single/<name>.png` on the server): bird, cat, cat5, chair, chair1, chair10, cow,
fire_dragon, ipadapter_result_00_chair, labubu, okapi, okapi1, panda, superman.

## Files

Per model and photo, `<model>/<name>/`:

| File | Content |
|---|---|
| `textured.glb` | the generated mesh with its texture |
| `textured_vc.ply` | the same mesh with vertex colours sampled from the base-colour texture, for our pipeline |
| `renders.png` | the photo, the mesh from the fitted input camera, and from the opposite side |
| `renders_view.json` | the fitted input camera (azimuth, elevation, silhouette IoU) |
| `shape.json`, `paint.json` (Hunyuan3D), `trellis.json` (TRELLIS.2) | seconds, peak GPU memory, counts, commit, command |

`summary.csv` has one row per mesh. Hunyuan3D's untextured shape (`shape.glb`, 0.38 to 1.0 M faces)
and the painted OBJ with its texture maps stay on the server under `~/semtex/gen3d/`.

## Commands

```
# Hunyuan3D-2.1, environment hunyuan3d (Python 3.10, torch 2.5.1+cu124)
export HY3DGEN_MODELS=~/semtex/gen3d/hy3dgen_models LD_LIBRARY_PATH=~/miniconda3/envs/hunyuan3d/lib
python scripts/idarb/gen3d_hunyuan.py --stage shape --inputs <14 photos> --out_root ~/semtex/gen3d/hunyuan3d
python scripts/idarb/gen3d_hunyuan.py --stage paint --inputs <14 photos> --out_root ~/semtex/gen3d/hunyuan3d
# TRELLIS.2, environment trellis2 (Python 3.10, torch 2.6.0+cu124)
python scripts/idarb/gen3d_trellis.py --inputs <14 photos> --out_root ~/semtex/gen3d/trellis2
# renders and view fit, environment diff3f
python scripts/idarb/gen3d_render.py --mesh <dir>/textured_vc.ply --image <photo> --out <dir>/renders.png \
    [--azim_range -90 90 for Hunyuan3D]
```

Settings: Hunyuan3D shape 50 steps, octree resolution 384, seed 0; paint 6 views at 512 px (the
paint model remeshes to 40 k faces first). TRELLIS.2 512 setting, seed 42 (the official default),
GLB with 200 k faces and a 2048 px texture. Code: gen3d-tmp (see `git log`); Hunyuan3D-2.1 commit
`82920d6`, TRELLIS.2 commit `75fbf01`. Versions and licenses of the code, weights and compiled
extensions are in `NOTICE.md`.

## Timing and memory

| | per mesh | peak GPU memory | faces |
|---|---|---|---|
| Hunyuan3D shape | 56 to 57 s | 7.8 GB | 0.38 to 1.0 M (untextured) |
| Hunyuan3D paint | 52 to 71 s | 13.7 to 14.2 GB | 40 k |
| TRELLIS.2 (512) | 22 to 35 s (79 s for the first two, warm-up) | 11.7 to 12.4 GB | 185 to 200 k |

Both run on our 24 GB slice. Loading: TRELLIS.2 84 s (10.5 GB on the GPU), Hunyuan3D shape about
20 s with about 10 GB of host RAM.

## Frames

- **Hunyuan3D-2.1** keeps the photo's frame: camera on +z looking towards -z, +x right, +y up (the
  relief meshes' convention), mesh scaled into [-1, 1]. The alignment is approximate for oblique
  photos: the fitted camera is within 20 degrees of azimuth 0 for 12 of the 14, at 40 degrees
  for okapi and 60 degrees for fire_dragon (whose flat flame skews the silhouette).
- **TRELLIS.2** puts the object in its own canonical pose: the object's front faces +z, +y up,
  inside [-0.5, 0.5]^3, whatever the photo's viewpoint. A side-view cat comes out facing +z and
  the photo's camera is at azimuth 270 (cat), 180 (cat5), 120 (okapi), 70 (cow).
- So the input camera is fitted per mesh (`gen3d_render.py`): silhouettes from cameras every 10
  degrees of azimuth at elevations -10 to 30, cropped and compared with the photo's alpha by IoU;
  when front and back silhouettes tie within 0.05 IoU (left-right symmetric objects), the render
  closer in colour to the photo wins; for Hunyuan3D the search is limited to the front half. The
  result is a coarse camera (10-degree steps, IoU 0.61 to 0.93) to be refined, for example with
  the keypoints, before projecting the photo's keypoints and albedo onto the mesh.

## Problems met and how they were solved

1. Hunyuan3D's shape loader builds the 3.3 B-parameter model in float32 and reads the 7.4 GB
   checkpoint into RAM, which got the process killed on the 16 GB server; the runner memory-maps
   the checkpoint and builds the modules in float16.
2. Hunyuan3D keeps its own model folder (`~/.cache/hy3dgen`) and downloaded the shape model a
   second time; `HY3DGEN_MODELS` now points at the Hugging Face snapshot.
3. `bpy==4.0` (Blender, used only for the OBJ to GLB conversion) is no longer on PyPI and later
   versions need Python 3.11; the import is stubbed and the conversion done with trimesh.
4. The paint upscaler's basicsr needs `torchvision.transforms.functional_tensor`, removed in
   torchvision 0.20; the repository's own `torchvision_fix.apply_fix()` is called, as its demo does.
5. pytorch_lightning 1.9.5 needs `pkg_resources`; setuptools pinned below 81 in the hunyuan3d env.
6. TRELLIS.2: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (set by its example) makes every
   CUDA allocation fail on the vGPU slice; not set.
7. TRELLIS.2 reads `DINOv3ViTModel.layer`, renamed in transformers 5; transformers pinned to 4.57.1.
8. TRELLIS.2: only the 512 setting's models are loaded and each is moved to the GPU as soon as it
   is loaded (building all of them on the CPU first would need about 14 GB of RAM); its gated
   background-removal model (RMBG-2.0) is stubbed, since our inputs are RGBA.

## Quality notes (from the renders)

Both generators give plausible full meshes for all 14 photos, with backs that fit the object
(the superman cape, the panda's back, the chairs' backrests). Visible problems: Hunyuan3D's
fire_dragon flame is a flat card; TRELLIS.2's fire_dragon has a green patch on the belly, its
`chair` is golden where the photo is silver-white, and its `chair10` is darker than the photo.

## Licence restriction to keep in mind

Hunyuan3D-2.1's licence does not allow its outputs to be used to improve any other AI model, so
the Hunyuan3D meshes may be used for evaluation but not as training data for our own models; it
also does not apply in the EU, the UK or South Korea. TRELLIS.2 is MIT; its nvdiffrast and
nvdiffrec extensions are for non-commercial (research) use. See `NOTICE.md`.
