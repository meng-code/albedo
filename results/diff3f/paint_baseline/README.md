# Paint baselines: a texture generator paints the target from the source photo

No correspondence is used: for each of the 15 annotated pairs, the TARGET full mesh is painted by
a texture generator with the SOURCE photo as reference image (the "Hunyuan3D paint" and
"TRELLIS.2 texturing" rows of the baseline table). Code: `scripts/idarb/gen3d_paint_baseline.py`
on branch paint-baseline-tmp; run on the GPU server (A40-24Q) on 2026-10-25.

Folders:

| Folder | Painter | Target pose given to the painter |
|---|---|---|
| `results/diff3f/paint_baseline/` | Hunyuan3D-2.1 paint model (6 views at 512 px, seed 0) | photo frame |
| `results/diff3f/paint_baseline_trellis2/` | TRELLIS.2 texturing (1024 texture flow, 2048 px texture, seed 0) | photo frame |
| `results/diff3f/paint_baseline_trellis2_canonical/` | TRELLIS.2 texturing | TRELLIS.2's own pose for the photo |

Per pair: `renders.png` (the photo, the painted target from the photo's camera, from behind) and
`paint.json` (seconds, peak GPU memory, face count, the surface check, command, commits). The
painted meshes in the target's own frame are in `paint_baseline_meshes/<painter>/<pair>/`
(`textured.glb`, and `textured_vc.ply` with vertex colours sampled from the albedo texture); they
are remeshed by the painter, so their vertices are not the target's.

Frames: each target is turned into the photo's camera frame with the pair's registration rotation
(`results/diff3f/<pair>/transform.json`; labubu_camel and labubu_lion borrow okapi_camel and
cat_lion, as `eval_features.py` does, so for them the target is seen from the side while the labubu
photo is frontal). For TRELLIS.2 the canonical run additionally rotates about +y by the azimuth at
which the photo's camera was found for TRELLIS.2's own generated mesh of that photo
(`results/diff3f/gen3d/trellis2/<photo>/renders_view.json`). After painting, the mesh is turned back;
the check `surface_distance_*` (distance of the painted vertices to the target surface, as a
fraction of the bounding-box diagonal) is 0 to 1e-6 for every pair.

Commands (repository root of paint-baseline-tmp):

```
LD_LIBRARY_PATH=~/miniconda3/envs/hunyuan3d/lib ~/miniconda3/envs/hunyuan3d/bin/python scripts/idarb/gen3d_paint_baseline.py
~/miniconda3/envs/diff3f/bin/python scripts/idarb/gen3d_paint_baseline.py --renders_only
~/miniconda3/envs/trellis2/bin/python scripts/idarb/gen3d_paint_baseline.py --painter trellis2 --frame photo \
    --out_dir results/diff3f/paint_baseline_trellis2 --mesh_dir ~/semtex/paint_baseline_trellis2
~/miniconda3/envs/trellis2/bin/python scripts/idarb/gen3d_paint_baseline.py --painter trellis2 --frame canonical \
    --out_dir results/diff3f/paint_baseline_trellis2_canonical --mesh_dir ~/semtex/paint_baseline_trellis2_canonical
(each followed by the --renders_only call with the same --out_dir and --mesh_dir)
```

Hunyuan3D-2.1: 38-47 s per pair, peak GPU 13.7-14.2 GB. Licences: Hunyuan3D-2.1's outputs must not
be used to improve other AI models (evaluation only); see `NOTICE.md`.
