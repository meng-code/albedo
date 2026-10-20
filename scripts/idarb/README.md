# scripts/idarb

Scripts we added around IDArb (whose code is in `third_party/IDArb/`).

## Where things run

- **Server (with an NVIDIA GPU):** run IDArb to decompose images into albedo, normal, metallic and
  roughness; generate depth maps with a depth model; run the training smoke test.
- **Laptop (no GPU and no PyTorch needed):** rebuild 3D meshes from the decomposed normal maps.

| Script | Where it runs | What it does |
|---|---|---|
| `prepare_images.py` | server | Turns ordinary photos into the four-channel RGBA PNGs that IDArb expects |
| `fetch_arbobjaverse_subset.py` | server | Downloads only a few chosen objects from the Arb-Objaverse dataset, for the training smoke test |
| `estimate_depth.py` | server | Generates relative depth maps with Depth Anything V2, used as a depth prior for mesh reconstruction |
| `normal_to_mesh.py` | laptop or server | Rebuilds a 3D mesh from an IDArb normal map, optionally with the depth prior |
| `plot_failure_cases.py` | laptop | Draws failure-case figures (input, IDArb output, side views of both meshes), e.g. `results/analysis/failure_cats.png` |
| `eval_samples.py` | laptop | Per-image metrics for all sample images, with and without the depth prior → `results/analysis/sample_metrics.csv` |
| `eval_occlusion.py` | laptop | Two-sphere occlusion experiment at three gaps, plus a single-sphere control → `results/analysis/occlusion_gaps.csv` |
| `eval_method_b.py` | laptop | Distance between the method-B and method-A surfaces → `results/analysis/method_b_vs_a.csv` |
| `compare_depth_models.py` | laptop or server | Correlation between two sets of depth maps, e.g. Depth Anything V2 Small vs Large → `results/analysis/depth_small_vs_large.csv` |
| `poisson_on_depth_points.py` | laptop | Experiment: screened Poisson on points taken straight from the depth map, compared with the other variants (not adopted, see `docs/decisions.md`) |
| `diff3f_features.py` | server | Diff3F semantic feature vector for every (sampled) vertex of a mesh → `.npz` (section 5) |
| `register_partial.py` | laptop or server | Registers a partial mesh to a full mesh with those features: feature matches → RANSAC similarity transform → ICP, with a feature-free ICP baseline (section 5) |
| `make_partial_mesh.py` | laptop | Cuts the camera-visible part of a full mesh into a synthetic partial mesh with known ground truth (section 5) |
| `eval_registration.py` | laptop | Runs the registration settings on every pair with feature files → `results/diff3f/ablation.csv` |
| `render_registration.py` | laptop | Software renders of a registration result folder → `render.png` |

## 1. Server: image decomposition

Run the following commands in `third_party/IDArb/`, in the IDArb conda environment (Python 3.8,
torch 2.2.1). `requirements-server.txt` lists how to install this environment and the versions known to
work. The official `third_party/IDArb/requirements.txt` contains several packages that are not needed
and are hard to install, so we do not recommend using it directly.

`prepare_images.py` removes the background with rembg and writes the foreground into the alpha channel.
If an image already has a usable alpha channel, that channel is kept and the background is not removed
again.

```bash
python ../../scripts/idarb/prepare_images.py --input mydata/raw --output mydata/my_single
python main.py --data_dir mydata/my_single --output_dir output/my_single --input_type single
```

The results are saved as `output/my_single/<name>_00.png`. Each one is four images side by side:
albedo, normal, metallic and roughness.

`main.py` uses the official pretrained weights by default. Add `--pretrained output/train_smoke/pipeckpts`
to use weights you trained yourself.

For the depth prior, also generate depth maps for the same input images (run from the repository root;
Depth-Anything-V2-Small is the default):

```bash
python scripts/idarb/estimate_depth.py --input_dir results/idarb_single/inputs --out_dir results/idarb_single/depth
```

The depth maps are 16-bit greyscale PNGs, pixel-aligned with the IDArb output; brighter means closer to
the camera. Push them to GitHub together with the decomposition results, so that building meshes on the
laptop does not need PyTorch.

## 2. Server: training smoke test

The aim is to check that training IDArb from the original Stable Diffusion 2.1 weights on a public dataset
works end to end. Result quality is not the goal.

```bash
cd third_party/IDArb
python ../../scripts/idarb/fetch_arbobjaverse_subset.py --datalist datalist/my_train.json
accelerate launch --config_file configs/acc/1gpu.yaml train.py --config configs/my_train_smoke.yaml
```

The first command extracts only 3 objects (about 77 MB in total) from an archive of about 10 GB and
places them in `data/Arbobjaverse/`. The config `configs/my_train_smoke.yaml` is a single-GPU, scaled-down
version of the official `train.yaml`: batch size 1, EMA off, 300 training steps, validation every 100
steps. On one A40 (24 GB) it takes about 7.5 minutes and uses about 19 GB of GPU memory.

## 3. Laptop: normal map to mesh

### Installation (once)

```bash
conda create -n mesh python=3.10 -y
conda activate mesh
pip install -r scripts/idarb/requirements-local.txt
```

### Running (from the repository root)

`results/idarb_single/` in the repository contains a set of sample data: `decomposition/` holds the
IDArb results, and `inputs/` holds the matching cut-out inputs (used to find which pixels belong to the
object).

```bash
# One object
python scripts/idarb/normal_to_mesh.py \
    --result results/idarb_single/decomposition/panda_00.png \
    --input results/idarb_single/inputs/panda.png

# A whole folder
python scripts/idarb/normal_to_mesh.py \
    --result_dir results/idarb_single/decomposition --input_dir results/idarb_single/inputs

# Stretch along the depth direction by a factor of 2
python scripts/idarb/normal_to_mesh.py \
    --result_dir results/idarb_single/decomposition --input_dir results/idarb_single/inputs \
    --out_dir outputs/mesh_z2 --z_scale 2

# With the depth prior (recommended): restores the distance between front and back parts and
# corrects the overall tilt
python scripts/idarb/normal_to_mesh.py \
    --result_dir results/idarb_single/decomposition --input_dir results/idarb_single/inputs \
    --depth_dir results/idarb_single/depth --out_dir outputs/mesh_depth

# Check that the algorithm is correct (no input files needed): an ideal hemisphere, and two spheres
# with an occlusion gap
python scripts/idarb/normal_to_mesh.py --selftest
```

`--out_dir` is the parent folder; each object is written to its own subfolder `<out_dir>/<name>/`.

### Output

By default the results go to `outputs/mesh/<name>/` (this folder is not tracked by git):

| File | Contents |
|---|---|
| `relief.ply` / `relief.obj` | Mesh from method A (Poisson normal integration). Only the side facing the camera, coloured with the albedo |
| `poisson.ply` | Mesh from method B (screened Poisson surface reconstruction). On a single image it only has the front side, is almost the same as `relief.ply`, and is not closed |
| `depth.png` | The recovered depth map; brighter means closer to the camera |
| `preview.png` | Preview renders from several angles |

All meshes are moved so that the mean of their vertices is at the origin. We recommend viewing the
`.ply` files: the colours are stored on the vertices, and `.ply` supports vertex colours in a standard
way.

Method B calls Open3D's Poisson reconstruction with a single thread. With the default multithreading it
often aborts with "Failed to close loop" on a Mac with an M-series chip, and the same input gives a
slightly different number of triangles on each run. With a single thread it always succeeds and the
result is identical every time.

### Methods

- **Method A: Poisson normal integration.** The normals give the slope of the surface at every point,
  that is, the gradient of the depth. We look for a depth map whose gradient is as close as possible to
  the gradient given by the normals. This least-squares problem simplifies to a Poisson equation. Once the
  depth map is solved, the pixel grid is connected into triangles.
- **Method B: screened Poisson surface reconstruction** (Kazhdan & Hoppe 2013). The 3D points from
  method A and the normals predicted by the network are passed to Open3D, which reconstructs a surface.
  When the points cover the object from all sides, the output is a closed surface. A single image only
  gives points on the front, so it does not fill in the back. In practice the output is the front of the
  object plus a large flat sheet that extends from the object's outline to the edge of the bounding box.
  After this sheet is trimmed by point density, the remaining front surface is less than half a pixel from
  the method-A surface (median distance about 0.45 px). The 180-degree view in the preview shows the
  reverse side of the same front surface, not a back side.
- **Method A with the depth prior (`--depth_dir`).** With normals alone, depth jumps at occlusions between
  front and back parts are lost (see the known limitations below). With the relative depth map d from
  `estimate_depth.py`, we solve
  `min Σ(neighbouring depth difference − difference given by the normals)² + w·Σ(h − s·d)²`:
  the normals take care of surface detail, and the depth map takes care of how far apart the parts are.
  Between neighbouring pixels where the depth map sees a jump, the surface is no longer required to be
  smooth, and the triangles that span such a jump (the "wall") are removed when the mesh is built.
  The depth model is not accurate on thin structures such as chair legs and cuts them into many small
  floating fragments. Fragments smaller than `--min_part` pixels are therefore merged back into the one
  neighbouring part they share the longest border with (only one, so that parts that should stay apart,
  such as a cape and a body, are not glued together).
  Parameters: `--depth_weight` (default 0.01), `--jump_px` (default 3; how many pixels of depth
  difference count as a jump) and `--min_part` (default 300; fragments smaller than this many pixels are
  merged back).

### Known limitations and results

- A single image can only recover the side facing the camera.
- **With normals only,** depth jumps where one part is in front of another (for example an arm in front
  of the body) are smoothed away, and the front part is "glued" onto the part behind it, so the whole
  object looks flattened. Tested on synthetic scenes with known depth: true gaps of 49, 99 and 149 px were
  all recovered as about 5 px. Without occlusion (a single sphere) the depth is recovered exactly.
- **With the depth prior,** the gap in the same synthetic scene (true 99.4 px) is recovered as 99.4 px.
  On the sample data:
  - chair1's depth-to-width ratio goes from 0.28 to 0.82, and the seat and back are now in the right
    front-to-back order;
  - labubu was tilted as a whole and is now upright;
  - objects without clear occlusions, such as bird, panda and superman, change little;
  - thin structures no longer break into floating fragments: chair1 was first cut into 67 parts (64 of
    them under 300 px) and now has 3 parts, the main body and the two back legs. The back legs are still
    separate from the chair because the top of each back leg is hidden behind the seat in the photo, so
    there is no way to see where it joins the chair; the legs are placed behind it at the depth that the
    depth model estimates;
  - the cost is a larger deviation between the reconstructed surface and the IDArb normals (for example
    chair1 goes from 4.7° to about 11°). Raising the weight from 0.01 to 0.2 barely changes the depth
    ratio but increases the deviation further, so the default is 0.01.
  - Comparison figure: `results/idarb_single/depth_prior_compare.png`.
- **When the normals fail completely, the depth prior fails too.** The depth-prior scale s is fitted to the
  normal slopes. On cat2, cat3 and cat4, IDArb's normals almost all face the camera, so s is only 3 to 16 px
  (about 100 to 250 px on other objects), and the meshes are still a line when seen from the side.
  See `results/analysis/failure_cats.md`.
- The normals predicted by the network have random errors. On 16 samples from the dataset they differ from
  the ground truth by 17.2° on average; the integrated depth is 0.43 to 1.55 times the true depth, with a
  median of 1.00, so there is no systematic flattening.

## 4. Reproducing the numbers in the report

Run from the repository root in the `mesh` environment unless noted. Each script prints its results and
writes them to `results/analysis/`.

| Number in the report | Script | Output |
|---|---|---|
| Hemisphere self-test: mean error 0.01 px, max 0.04 px | `normal_to_mesh.py --selftest` (also `pytest tests/ -q`) | printed |
| Two spheres: true gaps 49.4 / 99.4 / 149.4 px recovered as 5.4 px; depth ratio 0.71 / 0.49 / 0.38; single sphere 1.00 | `eval_occlusion.py` | `occlusion_gaps.csv` |
| 16 of 19 images usable; median depth/width 0.35 → 0.38; mean angle to the normals 5.5° → 11.2°; chair1 0.28 → 0.82; part counts | `eval_samples.py` | `sample_metrics.csv` |
| Method B vs method A: median distance about 0.45 px | `eval_method_b.py` | `method_b_vs_a.csv` |
| Normals only / joint / depth only / Poisson on depth points: 4.8° / 11.0° / 17.0° / 16.9° (chair1) | `poisson_on_depth_points.py` | printed |
| Failed cats: mean n_z, scale s, depth ranges | `plot_failure_cases.py` | printed, `failure_cats.png` |
| IDArb normals vs ground truth (8 Arb-Objaverse training objects × 2 views): mean angle 17.2° (mean of per-image means); outline n_z 0.59 GT vs 0.59 predicted (mean of per-image medians on the outline ring); integrated depth ratio median 1.00 (0.43 to 1.55) | `eval_idarb_gt.py --data_dir <Arb-Objaverse folder>` (server, idarb environment) | `idarb_normals_vs_gt.csv` |
| Depth Anything V2 Small vs Large on chair1, superman, labubu, panda, bird: correlation 0.93 to 0.996 (inside the full object mask) | `estimate_depth.py --model depth-anything/Depth-Anything-V2-Large-hf` (server), then `compare_depth_models.py` | `depth_small_vs_large.csv` |
| Partial-to-full registration with Diff3F features: cow median 0.0201 / cat 0.0110 of the target diagonal, agreement 0.564 / 0.690, feature-free ICP baseline 0.0217 / 0.049; 16 real pairs in all (section 5) | `diff3f_features.py` (server, diff3f environment), then `register_partial.py --piecewise` (section 5) | `results/diff3f/<pair>/metrics.csv` |
| Painting with ground truth: covered vertices within 3.6 / 4.4 RGB, feature-only 7.7 / 8.0, mirrored 2.6 / 4.2 | `eval_painting.py` | `results/diff3f/painting_eval.csv` |
| Synthetic partial meshes with ground truth: rotation error 0°, correspondences 100 % within 1 %, feature-only 30.5 % / 38.7 % | `make_partial_mesh.py`, `diff3f_features.py --front_only` (server), `register_partial.py --gt` | `results/diff3f/{cow,cat}_synth/metrics.csv` |
| Registration ablation: 12 settings on 8 pairs (section 5, findings) | `eval_registration.py` | `results/diff3f/ablation.csv` |

## 5. Registering a partial mesh to a full mesh with Diff3F features

The mesh from a single image is only the visible front of the object (a *partial mesh*). The next step
(supervisor, 2026-10-20) is to *register* it to a full mesh of the same kind of object, that is, to find
the scale, rotation and translation that put the partial mesh onto the full one, and the point-to-point
correspondence between them. Later this correspondence is what carries the albedo from the image onto
the full mesh.

Shape alone cannot do this: the two meshes differ in pose, proportions and coordinate system, and the
partial mesh has no back. We therefore use **Diff3F** (Dutt, Muralikrishnan and Mitra, CVPR 2024, code in
`third_party/Diff3F/`): it gives every vertex a 2048-number *semantic* feature vector, so that "left ear"
on one mesh is close to "left ear" on the other, regardless of shape. Diff3F renders the untextured mesh
from many views as depth and normal maps, has Stable Diffusion 1.5 with two ControlNets paint a textured
image from them, takes the diffusion UNet features and DINOv2 features of that image, projects them back
onto the vertices and averages over the views. No training is needed.

### Server: feature extraction (`diff3f_features.py`)

Install the `diff3f` environment as described in `requirements-diff3f.txt` (Python 3.10, torch 2.1, pytorch3d
0.7.5, diffusers 0.21.4). The first run downloads about 4 GB of weights. Then, from the repository root:

```bash
# the partial mesh: cameras only in front of it (+z, the camera side of relief.ply)
python scripts/idarb/diff3f_features.py --mesh outputs/mesh_depth/cow/relief.ply --prompt cow \
    --front_only --out outputs/diff3f/cow_partial.npz
# the full mesh: cameras all around
python scripts/idarb/diff3f_features.py --mesh third_party/Diff3F/meshes/cow.obj --prompt cow \
    --out outputs/diff3f/cow_full.npz
```

Options: `--num_views` (default 100, a square number; `--front_only` uses a grid over elevations ±45°
and azimuths ±75°), `--max_vertices` (default 15000; the whole mesh is rendered but features are stored
for a random subset of this size, which keeps a 90k-vertex relief mesh tractable), `--tolerance`
(ball-query radius as a fraction of the mesh extent, 0.004 as in the Diff3F notebook), `--seed`.
The `.npz` holds `idx` (vertex indices), `xyz`, `feat` (float16, unit length), `colors` (vertex colours if
the mesh has them) and `meta` (command, git commit, settings, run time). A 15000-vertex file is about 60 MB
and stays out of git. Expect 2 to 4 minutes per mesh on an A40.

### Laptop or server: registration (`register_partial.py`)

```bash
python scripts/idarb/register_partial.py --source outputs/diff3f/cow_partial.npz \
    --target outputs/diff3f/cow_full.npz --source_mesh outputs/mesh_depth/cow/relief.ply \
    --target_mesh third_party/Diff3F/meshes/cow.obj --out_dir outputs/diff3f/cow_registration
```

Steps, all in NumPy/SciPy (no GPU):

1. **Matches.** Cosine similarity between every source vector and every target vector. `--matches both`
   (default) takes every source point with its best target point and every target point with its best
   source point, and keeps the most similar half of them: thousands of matches with outliers that RANSAC
   has to reject. `--matches mutual` keeps only pairs that are each other's best match: few, cleaner,
   but on the first cow run too few (144) to pin down the right transform.
2. **Coarse alignment.** RANSAC over the matches: take 3 matches, fit a similarity transform (scale,
   rotation, translation; Umeyama 1991), count how many matches it brings within `--inlier_dist` (default
   5 % of the target's bounding-box diagonal). The scale is limited to 0.4 to 2.5 times the ratio of the
   two bounding-box diagonals. Up to `--candidates` (default 8) distinct hypotheses are kept, best first,
   each refitted on its inliers.
3. **Fine alignment and selection.** Every candidate is refined by ICP with the scale fixed (pair every
   source point with its nearest target point, drop pairs farther apart than `--icp_dist`, refit the
   rotation and translation, repeat) and then scored by *feature agreement*: the mean cosine similarity
   between each source point's feature and the feature of the target point it lands on (0 for points
   that land nowhere). The candidate with the best agreement is `selected`. This step exists because the
   hypothesis with the most RANSAC inliers can be wrong: on the cartoon cow the big head matched well, and
   the best-inlier transform shrank the whole cow onto the full cow's head; body points landing on head
   points give that transform a low agreement.
4. **Baseline.** The same ICP started from centroid-and-size alignment only, without features, to show
   what the features add.

Options that are off by default (see the findings below for what they did): `--consensus` drops matches
whose spatial neighbours on the source do not land near the same place on the target; `--normals`
rejects ICP pairs whose normals disagree (needs both mesh files); `--refine joint` replaces the geometric
pairing of the best three candidates by a feature-guided one (each source point goes to the target point
that minimises feature distance plus spatial distance, with the spatial tolerance shrinking over the
iterations; the scale stays fixed) and keeps whichever of the geometric and joint results agrees best;
`--depth_scale` also fits one factor on the source's z, restarting the ICP of the best three candidates
from factors 0.5 to 2 (a relief mesh's depth is only known up to the depth-prior scale); `--feature
diffusion|dino` uses only one half of the Diff3F vector; `--seeds` pools the hypotheses of several
RANSAC runs; `--piecewise` refines the selected alignment part by part (see "Per-part refinement"
below).

`metrics.csv` has one row per stage and candidate: `candidate`, `n_matches`, `n_inliers`, `scale`,
`depth_factor`, `agreement`, `part_agreement` (k-means parts of the target features, `--parts 6`, as in
Diff3F's segmentation: the fraction of source points that land on a target point of the same part),
`rms_to_target` and `median_to_target` (distance from the aligned source points to the nearest target
point), `frac_within_2pct` (fraction closer than 2 % of the diagonal), and `match_median` /
`match_within_5pct` (distance between the two points of each feature match after alignment). All
distances are fractions of the target diagonal. The `selected` row is the result. `transform.json` has the
transform; `correspondences.npz` the target point of every source point, after registration and from the
features alone; `aligned_source.ply` is the partial mesh moved onto the target; `correspondence_*.ply`
colour matched points alike; `target_from_source_colors.ply` paints the target with the source's albedo
through the feature matches alone, `target_from_source_colors_registered.ply` through the registration
(the target points the aligned source covers take the colour of the nearest source point, the rest stay
grey: this is the albedo transfer the project aims at); `preview.png` shows the stages side by side.

**Ground truth.** `make_partial_mesh.py` cuts the part of a full mesh that one camera sees and writes it
in the camera frame with a known transform and vertex map (`results/diff3f/synthetic/`). Registering such
a synthetic partial mesh back to its full mesh (`--gt <name>_gt.npz`) adds the columns `rot_err_deg`,
`scale_ratio`, `pos_err_median` and the correspondence accuracy `corr_acc_1pct` / `corr_acc_5pct`
(fraction of source points whose assigned target point lies within 1 % / 5 % of the diagonal of the true
point), also for the feature-only correspondence (`feature_nn` row). `eval_registration.py` runs the
settings above on every pair whose feature files are present and writes `results/diff3f/ablation.csv`.

`pytest tests/ -q` checks the registration maths on synthetic data (a known transform must be recovered
from matches with 40 % outliers, ICP must remove a small misalignment, the depth factor must be found,
and the whole pipeline must beat the feature-free baseline).

### Findings (2026-10-20, `results/diff3f/ablation.csv`)

- **Synthetic partial meshes (same object, ground truth):** every setting recovers the transform
  (rotation error 0°, scale within 0.5 %) and 100 % of the correspondences within 1 % of the diagonal.
  The features alone are coarser: only 30.5 % (cow) / 38.7 % (cat) of the feature-only correspondences
  are within 1 %, 86 % / 97 % within 5 %; the features find the part, the geometry finds the point.
  The feature-free baseline is 29° off on the cow and only happens to land right on the cat.
- **Real relief meshes (cartoon cow and photo cat against Diff3F's cow and cat):** the default gives cow
  median 0.0201, agreement 0.564, part agreement 0.591; cat 0.0110 / 0.690 / 0.792, 0.0106 / 0.707 /
  0.828 with `--piecewise` (5 to 10 s per pair; the numbers before the scale bands were 0.0199 and 0.0116).
  Mutual matches with the best-inlier hypothesis (the first version) fail on the cow (0.0308, agreement
  0.268). The consensus filter, normals and three seeds change the result by less than 0.0005. The
  registered partial faces the same way as the target surface (normals agree for 90 % / 93 % of the close
  pairs), so it is not mirrored onto the far side.
- **Feature-guided ICP:** with the scale refitted it shrinks the source (cow scale 0.32 instead of 0.43,
  cat 0.31 instead of 0.44; part agreement down to 0.50 / 0.68), because a smaller source is closer to
  any surface. With the scale fixed it equals plain ICP within noise at 5 to 8 times the run time, so it is
  an option, not the default.
- **Depth factor:** on synthetic meshes whose depth was halved (true factor 2) it finds 1.79 (cow) / 1.78
  (cat) and brings the rotation error from 12.5° / 14.4° down to 1.9° / 2.7° and the correspondences within
  1 % from 14 % / 10 % up to 84 % / 74 %; on doubled depth (true 0.5) it finds 0.64 / 0.78 (rotation 16.5°
  / 17.4° → 4.7° / 10.7°). On the real relief meshes the factor is not well determined: the cat gives 0.48
  or 1.76 depending on the start with agreements 0.01 apart (its body is thinner than the 5 % tolerance,
  so the front and back surfaces are both "close"), the cow 1.17 to 1.42. Off by default; a diagnostic for
  the depth-prior scale once a sharper criterion exists.
- **Feature halves:** diffusion-only and DINO-only register about as well as the fused vector (cow 0.0204
  / 0.0221, cat 0.0110 / 0.0117), but the fused vector gives the best feature-only correspondence on the
  synthetic pairs (30.5 % against 29.2 % / 28.0 % within 1 %), so it stays. `agreement` values are not
  comparable between feature types (DINO similarities are higher across the board).

### Painting the target (albedo transfer)

`register_partial.py` paints the full mesh with the source's vertex colours (the albedo of the
photo). The target mesh is first subdivided (`--paint_subdiv`, default once: cow 2759 → 11017
vertices, cat 7207 → 28822), because small parts such as eyes have no vertices of their own on a
coarse mesh. Then every target vertex that the aligned source covers (within `--icp_dist`) takes the
colour of the source point that minimises (1 − feature similarity) + (distance / `--paint_delta`)² +
`--part_penalty` if the two points belong to different k-means parts: the colour comes from nearby,
from the same kind of part, and preferably from the same kind of point. Vertices the source does not
cover stay grey in `target_from_source_colors_registered.ply` and take the best feature match
anywhere in `target_from_source_colors_combined.ply`; `target_from_source_colors.ply` uses features
only everywhere. Before this (one colour per original vertex, nearest point only), stripes and
markings landed roughly; now the cat's stripes follow the body. What still fails is detail smaller
than the registration residual (about 2 % of the diagonal on the cow): the cartoon cow's eyes and
muzzle are far larger than the realistic cow's, so their colours land around the real eyes and
muzzle, not on them.

### Per-part refinement (`--piecewise`)

After the global similarity transform, every k-means part (`--parts`, default 6) is aligned once
more on its own by ICP against the target points of the same part, with its own scale (clipped to
0.8 to 1.25) and rotation (at most 15°). A part keeps the global alignment when its transform does
not bring it closer or uncovers target points it covered before (a shrinking part would). The part
transforms are blended by distance (`blend_sigma`, 5 % of the diagonal), so the mesh does not tear
at part borders. Results (`ablation.csv`, config `v2_piecewise`): on the synthetic pairs it changes
nothing where the alignment was already exact and improves the depth-squashed ones (cow z×0.5:
position error 0.023 → 0.017, correspondences within 1 % 14 % → 33 %); on the real cat the median
distance goes from 0.0116 to 0.0099, the points within 2 % from 78 % to 83 % and the agreement from
0.689 to 0.707 (three of six parts accepted, scales 0.80 to 0.83); on the real cow every part keeps
the global alignment, because the cartoon head would need a scale below 0.8 and a rotation above
15° (allowing 0.6 and 30°, `v2_piecewise_loose`, accepts transforms that lower the feature
agreement, so the limits stay). Off by default; the stored cat result was produced with it
(`aligned_source_piecewise.ply` next to `aligned_source.ply`; for the cow the two are the same).

### More pairs (2026-10-20, server batch 1)

Seven more pairs with the Diff3F sample meshes as targets, run on the server (features with
`--no_cpu_offload`: 4 to 10 minutes per mesh, 10 to 13 GB) and registered with `--piecewise`.
Numbers are the `selected` row and, after the arrow, the `piecewise` row of each `metrics.csv`
(median distance as a fraction of the target diagonal, feature agreement, part agreement).
`render.png` in each folder shows the aligned source on the target and the painted target
(`render_registration.py`).

| Pair | Source image → target mesh | Median | Agreement | Part agreement | Remarks |
|---|---|---|---|---|---|
| superman_man | superman → man.obj | 0.0258 → 0.0187 | 0.41 → 0.48 | 0.75 → 0.81 | suit, belt, boots and cape colours land on the right parts |
| superman_posed | superman → posed_human.off (kneeling) | 0.0165 | 0.47 | 0.62 | a different pose still registers; no part was refined |
| okapi_camel | okapi → camel.obj | 0.0220 → 0.0217 | 0.63 | 0.92 → 0.93 | cross-species: dark body and striped legs transfer |
| okapi1_camel | okapi1 → camel.obj | 0.0190 → 0.0181 | 0.66 → 0.67 | 0.88 → 0.91 | same, second photo |
| cat5_cat | cat5 → cat.off | 0.0119 → 0.0120 | 0.64 → 0.66 | 0.82 → 0.84 | walking pose |
| cat_lion | cat → lion.off | 0.0151 → 0.0133 | 0.64 | 0.80 | cross-species: the lion gets the Bengal cat's spots |
| labubu_man | labubu → man.obj | 0.0114 → 0.0107 | 0.32 → 0.34 | 0.60 → 0.67 | toy with a huge head on a human: shirt and trousers land on the body, the head stays uncovered; the low agreement says the features of a plush toy and a human differ, so the distances alone overstate this one |

(Numbers from the final code, with the scale bands and the coverage discount; the server's first
run of these pairs differed by at most 0.002 except superman_man, which improved from 0.0277.)

The camel scale of 50 to 65 is the unit of `camel.obj` (its extent is about 80), not an error.
`man_full` has 515 vertices without features (surfaces no camera saw) against 2 to 43 elsewhere.

### More pairs (2026-10-21, server batch 2: Objaverse targets)

Seven pairs against the CC-BY Objaverse meshes in `results/diff3f/targets/` (sources and licenses in
`NOTICE.md`), same procedure as batch 1.

| Pair | Source image → target mesh | Median | Agreement | Part agreement | Remarks |
|---|---|---|---|---|---|
| chair1_armchair | chair1 → armchair | 0.0144 → 0.0138 | 0.50 | 0.61 → 0.65 | frame and upholstery colours on the right parts |
| chair_armchair | chair (green, ornate) → armchair | 0.0214 → 0.0211 | 0.51 → 0.52 | 0.45 → 0.46 | green pattern on back and seat |
| chair10_roundchair | chair10 (carved) → roundchair | 0.0303 → 0.0295 | 0.38 → 0.39 | 0.56 → 0.59 | worst pair: the carved chair's relief mesh is cut into 6 pieces by the depth prior and the thin structure gives few consistent matches |
| chairgen_armchair | generated chair → armchair | 0.0117 → 0.0104 | 0.57 | 0.64 | clean source, clean result |
| panda_panda | panda → panda | 0.0107 → 0.0104 | 0.78 → 0.81 | 0.76 → 0.81 | best pair of all: eye patches, ears and arms painted black where they belong |
| bird_bird | bird (stone figurine) → sparrow | 0.0121 | 0.62 | 0.63 | the figurine is a blob with a beak; the first run shrank it (scale 0.46, coverage 0.16) into the sparrow's belly with part agreement 0.46; with the scale bands and the coverage discount it registers to the whole bird at scale 0.98 |
| dragon_lizard | fire_dragon → Charmeleon | 0.0135 | 0.54 | 0.58 | orange body, cream belly and the tail flame land where they belong |

Each result folder keeps `metrics.csv`, `transform.json`, `correspondences.npz`, `render.png` and
`target_from_source_colors_combined.ply` (decimated to 40000 triangles when larger); the aligned meshes
and the other painted variants are regenerated by the command in `transform.json`.

The Objaverse meshes have many vertices no camera sees (panda 4950 of 20002: eyes, mouth and other
parts inside the head; chairs 685 and 966); they get the feature of the nearest seen vertex and do
not seem to disturb the registration.

### Painting with ground truth and symmetric completion (2026-10-21)

`eval_painting.py` gives the synthetic partial meshes a known colour field (position inside the
bounding box, folded across the symmetry plane so the field is mirror-symmetric like a real albedo),
registers them and paints the full mesh back, then compares every painted vertex with the field
(mean absolute RGB error, 0 to 255; `results/diff3f/painting_eval.csv`). With the registration
exact, the covered vertices are painted to within 3.6 (cow) / 4.4 (cat) and the nearest-point,
feature-guided and part-aware variants are indistinguishable; the feature-only transfer is twice as
far off (7.7 / 8.0). So the painting is as good as the alignment, and the feature term only matters
where the alignment is not exact (the real pairs).

`paint_target()` now also fills the far side: the target's symmetry plane is taken among its three
principal planes (the ones the reflected shape lies on within 2 % of the diagonal; among those, the
one whose reflection carries the most uncovered vertices onto covered ones), and every uncovered
vertex whose mirror image lies on a covered vertex takes that vertex's colour
(`target_from_source_colors_mirrored.ply`; `--no_mirror` turns it off). On the synthetic colour field
the mirrored vertices are as accurate as the covered ones (2.6 / 4.2). On the real cow the far flank
takes the near flank's colours (43 % covered, 17 % mirrored); on the panda the plane fits less well
(error 0.0197) and 7 % are mirrored. `combined` uses the registered colours, then the mirrored ones,
then the feature-only colours.

### Candidates at every scale, and a coverage check (2026-10-21)

The stone-bird pair showed a failure the agreement alone cannot see: RANSAC's best hypothesis
shrank the blob-like figurine (scale 0.38 times the bounding-box ratio) into the sparrow's belly,
where every source point is close to the surface and the features agree reasonably, but only 16 % of
the target is covered. Two changes:

- RANSAC runs once per scale band (0.5 to 0.8, 0.8 to 1.3 and 1.3 to 2.5 times the ratio of the
  bounding-box diagonals; `scale_bands`), so the candidate pool always holds hypotheses at every
  plausible scale, and the refit on the inliers can no longer leave the allowed range. On every pair
  that registered, the selected scale was 0.55 to 1.2 times the ratio.
- `metrics.csv` gains `coverage` (fraction of the target's points within `--icp_dist` of the aligned
  source) and `agreement_t` (the agreement seen from the target side). The selection `score` is the
  agreement discounted when the coverage is below 0.2 (`min_coverage`): registered pairs cover 0.24 to
  0.75 of their targets, the collapsed bird 0.16. Averaging the two agreements instead, which was
  tried first, rewards size rather than correctness (it moved the dragon to a worse larger-scale
  solution). With the discount the bird registers at scale 0.98 (part agreement 0.46 → 0.63) and the
  other pairs keep their results.

### Depth factor with features computed on the deformed mesh (2026-10-21)

The earlier depth-factor test squashed the coordinates after the features had been computed.
`results/diff3f/synthetic/cow_synth_d0.5.ply` and `cow_synth_d2.ply` are cut with the depth already
halved and doubled, and their Diff3F features come from the deformed meshes (the feature-only
correspondence drops from 30.5 % to 24 % / 21 % within 1 %, so the squashing does reach the
features). Ground truth: the depth factor should come out as 2 and 0.5.

| Pair (true factor) | Setting | Factor found | Rotation error | Position error | Correspondences within 1 % |
|---|---|---|---|---|---|
| cow_synth_d0.5 (2) | default | 1 | 12.8° | 0.0226 | 15 % |
| | `--depth_scale` | 1.76 | 2.1° | 0.0062 | 94 % |
| | `--depth_scale --piecewise` | 1.76 | 2.1° | 0.0046 | 87 % |
| cow_synth_d2 (0.5) | default | 1 | 16.4° | 0.0288 | 14 % |
| | `--depth_scale` | 0.65 | 5.5° | 0.0229 | 11 % (99.8 % within 5 %) |
| | `--depth_scale --piecewise` | 0.65 | 5.5° | 0.0140 | 38 % |
| cow_synth (1) | `--depth_scale` | 1.00 | 0.1° | 0.0005 | 100 % |

So the depth factor repairs a relief whose depth is off by a factor of two, more completely when the
relief is too shallow (factor found 1.76 of 2) than when it is too deep (0.65 of 0.5), and does no harm
when the depth is right. It stays off by default only because on the real, thin cat the agreement cannot
tell 0.48 from 1.76 (see above); on objects as wide as the cow it is safe to use.

### Summary over all pairs (`results/diff3f/ablation.csv`, final, 2026-10-21)

The table holds six settings on 24 pairs: 16 real, 2 synthetic, 4 synthetic with the depth squashed
after the features were computed (`_z0.5`, `_z2`) and 2 deformed before (`_d0.5`, `_d2`). Averages
over the 16 real pairs (median distance / agreement / part agreement):

| Setting | Median | Agreement | Part agreement |
|---|---|---|---|
| mutual matches, best-inlier hypothesis (first version) | 0.0199 | 0.51 | 0.66 |
| default (both-direction matches, scale bands, agreement with coverage discount) | 0.0167 | 0.56 | 0.69 |
| default + `--piecewise` | 0.0158 | 0.57 | 0.71 |
| default + `--depth_scale` | 0.0161 | 0.57 | 0.70 |
| DINO half of the feature only | 0.0160 | 0.62 | 0.70 |
| diffusion half only | 0.0170 | 0.47 | 0.59 |

The DINO half alone registers the real pairs as well as the fused vector (agreements of different
feature types are not comparable, part agreements are); the fused vector keeps its edge on the
synthetic feature-only correspondence (30.5 % against 28.0 % within 1 %). The consensus filter, the
normal checks, extra seeds, the feature-guided ICP variants and the loose per-part limits were
evaluated on the first 8 and 15 pairs in earlier versions of this table (git history) and changed
nothing, so the final run leaves them out to stay under an hour on the laptop.

### Full-resolution colours (2026-10-21)

Until now the painting took its colours from the 15000 sampled source points that carry features,
while the relief mesh has 36k to 92k vertices with the photo's albedo. The chosen sampled point now
only marks the region (feature and part terms as before), and the colour comes from the
full-resolution source vertex nearest to the target vertex inside that region
(`paint_target(full_src_xyz, full_src_colors)`, on by default, `--no_full_colors` to switch off); the
target is subdivided twice by default (`--paint_subdiv 2`). The cat's stripes and the panda's eye
patches come out sharp; the cartoon cow's face still lands where the registration puts it. Stored
painted meshes are decimated to 100000 triangles.
