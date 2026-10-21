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
| `correspondence_maps.py` | laptop | Semantic correspondence maps from the features alone: joint feature PCA colours on both meshes, the target coloured by its nearest neighbour in feature space, and the relief's albedo carried over by the same table → `correspondence_maps.png`, coloured PLYs |
| `transfer_texture.py` | laptop | Texture transfer through the correspondence: the target gets texture coordinates into the relief's albedo image (OBJ + MTL + PNG), with a per-triangle consistency measure → `texture_transfer.png`, `texture_transfer.json` |

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
    --front_only --out outputs/diff3f/features/cow_partial.npz
# the full mesh: cameras all around
python scripts/idarb/diff3f_features.py --mesh third_party/Diff3F/meshes/cow.obj --prompt cow \
    --out outputs/diff3f/features/cow_full.npz
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
python scripts/idarb/register_partial.py --source outputs/diff3f/features/cow_partial.npz \
    --target outputs/diff3f/features/cow_full.npz --source_mesh outputs/mesh_depth/cow/relief.ply \
    --target_mesh third_party/Diff3F/meshes/cow.obj --out_dir outputs/diff3f/00_archive_registration_painting/registration_results/cow_registration
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

### Functional maps: tried, not adopted (2026-10-21)

Diff3F reports that its descriptors make functional maps (a smooth, global surface-to-surface
correspondence in the Laplace-Beltrami eigenbasis) work on non-isometric full shapes.
`eval_functional_map.py` tries them for our partial-to-full case, with the vendored pyFM, 50 or 100
eigenfunctions, with and without cutting the target to the part the registration says the source
covers, and with ZoomOut refinement (`results/diff3f/functional_map_eval.csv`). On the synthetic
pairs with ground truth the functional map puts 13 to 24 % of the target vertices within 1 % of the
true source vertex (best: cropped target, 100 eigenfunctions), against 40 to 52 % for plain feature
matching and 100 % for the registration; its mapped points also lie 3 to 4 % of the diagonal from
the target vertex. The reason is partiality: a one-sided relief and a full body do not share their
low-frequency structure, which functional maps assume, and cropping repairs that only in part. On
the real relief meshes (36k to 92k vertices, several components) pyFM's pipeline also breaks down
numerically. Not adopted; the registration plus the feature- and part-aware painting stays.
`potpourri3d` and `robust_laplacian` (pip) are needed only for this script.

### Semantic correspondence maps (`correspondence_maps.py`)

After the group meeting the supervisor's priority is the semantic correspondence itself: get it
accurate first, show it as "same part, same colour" maps, and start from plain nearest-neighbour
matching. `correspondence_maps.py` draws two such maps from the features alone, with no
registration:

- **feature PCA**: the two feature sets are projected together on their first three principal
  components, used as RGB. No matching is involved, so a part has the same colour on both meshes
  only if its features agree. On every pair the maps agree part by part (cat: head magenta, body
  orange, hindquarters cyan, tail blue on both meshes; panda: face orange, ears white).
- **nearest neighbour**: the source is coloured by position and every target vertex takes the colour
  of its nearest neighbour in feature space. Coherent colours mean a coherent correspondence; the
  chair targets and the stone bird come out patchy (legs and seat mix, the blob matches everywhere).
- **albedo transfer**: the same nearest-neighbour table carries the relief's real colours to the
  target (`nn_target_albedo.ply`). This is what the matching gives on its own, with no registration
  and no painting rules: on the cat → lion pair the lion gets the cat's fur tone, lighter face,
  white inner ears and black tail tip, but the rosettes are lost, because the lion has only 5 000
  vertices (one colour each) and neighbouring vertices jump to different source points.

With `--gt` (synthetic pairs) a further map shows the nearest-neighbour error per target vertex.
The nearest neighbour puts 40.1 % (cow) / 52.2 % (cat) of the target vertices within 1 % of the
diagonal of the true source vertex and 87.2 % / 94.3 % within 5 %; the errors sit inside parts (ear
tips, paws, belly), not between parts. Every result folder has `correspondence_maps.png`; the
coloured meshes (`pca_*.ply`, `nn_*.ply`) are written to the output folder for MeshLab.
`--transform results/diff3f/<pair>/transform.json` reuses the stored registration for the target's
viewing direction instead of registering again.

### Texture transfer (`transfer_texture.py`)

The supervisor's correction on the albedo transfer: one colour per vertex is not a texture. A
triangle must show the image inside it, so the correspondence has to carry texture coordinates,
not colours. The relief makes this direct: it is a height field over the albedo image, every relief
vertex is one pixel, so its texture coordinates are its pixel position and its texture is the
albedo image (`transfer_texture.py` rebuilds the image from the relief's vertex colours on their
pixel grid and fills the background from the nearest foreground pixel). Every target vertex takes
the texture coordinates of the relief vertex it is matched to, by default its nearest neighbour in
feature space (`--correspondence` plugs in any other table, so a better matching is tested by
running the same script). The result is an OBJ with MTL and PNG that MeshLab opens with the texture
(`target_textured.obj` in the output folder), drawn with a small software rasteriser in
`texture_transfer.png` (matplotlib cannot sample a texture inside a triangle).

A triangle whose three corners are matched to distant image points shows a stretched streak of the
image, so the span of the corners' texture positions measures the consistency of the
correspondence triangle by triangle: `texture_transfer.json` records the fraction of triangles
whose corners lie within 2 % and 5 % of the image diagonal of each other (over all triangles, and
over the ones the registration says the relief covers), and `target_uv_span.ply` shows it as face
colours. The span alone rewards a collapse (if every target vertex matched the same source point,
every span would be zero), so it is read together with **source used**: the fraction of the
relief's image cells (a 32 × 32 lattice over the texture) that at least one target vertex is
matched into. On cat → lion the rosettes and stripes now appear on the lion, with streaks on the
inconsistent triangles. All 16 real pairs, nearest-neighbour correspondence:

| pair | triangles | within 2 % | within 5 % | source used |
|---|---|---|---|---|
| cow | 5 500 | 39.8 % | 77.0 % | 64 % |
| cat | 14 410 | 68.5 % | 87.6 % | 68 % |
| superman_man | 13 776 | 68.8 % | 86.9 % | 59 % |
| superman_posed | 10 410 | 67.9 % | 85.6 % | 55 % |
| okapi_camel | 3 576 | 34.2 % | 67.6 % | 62 % |
| okapi1_camel | 3 576 | 37.1 % | 74.9 % | 69 % |
| cat5_cat | 14 410 | 76.3 % | 90.5 % | 60 % |
| cat_lion | 9 996 | 66.9 % | 85.5 % | 66 % |
| labubu_man | 13 776 | 64.5 % | 82.7 % | 58 % |
| chair1_armchair | 10 860 | 50.7 % | 71.1 % | 50 % |
| chair_armchair | 10 860 | 49.6 % | 76.3 % | 54 % |
| chair10_roundchair | 16 028 | 56.8 % | 69.0 % | 43 % |
| chairgen_armchair | 10 860 | 48.8 % | 69.8 % | 57 % |
| panda_panda | 40 000 | 74.5 % | 92.5 % | 80 % |
| bird_bird | 40 000 | 47.1 % | 71.0 % | 64 % |
| dragon_lizard | 40 000 | 66.9 % | 81.2 % | 59 % |
| mean | | 57.4 % | 79.3 % | 61 % |

The coarse targets (cow, camel: a few thousand triangles, so each triangle covers a large patch of
the image) and the chairs sit lowest. These are the numbers to raise when the correspondence
improves.

### Cross-category pairs

The supervisor asked for very different source and target, for example the Labubu toy on the
camel. Ten such pairs were run from the features alone, with no registration of the pair itself
(the target's viewing direction is borrowed from a same-target registration):
`results/diff3f/cross/<source>_<target>/` holds `correspondence_maps.png`, `texture_transfer.png`
and `texture_transfer.json`. The last pair, a chair on the cow, is a control with no semantic
relation at all.

| pair | triangles | within 2 % | within 5 % | source used |
|---|---|---|---|---|
| labubu → camel | 3 576 | 31.7 % | 55.4 % | 38 % |
| labubu → cow | 5 500 | 37.7 % | 58.1 % | 48 % |
| labubu → lion | 9 996 | 51.5 % | 70.2 % | 45 % |
| superman → camel | 3 576 | 37.1 % | 60.4 % | 37 % |
| panda → man | 13 776 | 56.7 % | 77.0 % | 70 % |
| cat → man | 13 776 | 53.9 % | 70.1 % | 66 % |
| cow → camel | 3 576 | 33.1 % | 64.1 % | 53 % |
| fire_dragon → man | 13 776 | 54.8 % | 69.8 % | 46 % |
| bird → cat | 14 410 | 67.7 % | 79.6 % | 28 % |
| chair1 → cow (control) | 5 500 | 36.5 % | 48.4 % | 35 % |
| mean | | 46.1 % | 65.3 % | 46 % |

The part level holds across categories: the Labubu's head lands on the head of the camel, the cow
and the lion (its pink eyes at their eyes), the shirt on the body, the jeans on the legs and the
white shoes on the hooves and paws; Superman's suit covers the camel with the red boots at the
bottom of the legs; the panda gives the man a white face with black eye patches, a white belly and
black limbs; the cow's hooves and pink muzzle land on the camel's feet and mouth; the dragon's cream
belly lands on the man's chest and belly. What the source lacks is filled with the nearest thing:
the camel's hump takes the Labubu's head and Superman's emblem, the cape scatters over the legs.
The numbers are lower than within a category (mean 46 % / 65 % against 57 % / 79 %). The control
pair also produces a picture (the seat on the body, the wooden legs on the legs), which is why
the numbers matter: it sits at the bottom together with the stone bird, whose relief is a ball and
whose matching collapses onto a few source points (source used 28 %), something the span alone
would have scored as consistent.

### One side only

The supervisor's next instruction: texture only the side the relief shows and leave the back
blank. `transfer_texture.py` now textures only the faces visible from the target's viewing
direction (a depth-buffer test at 1024 px, plus the front-facing faces whose centroid is not behind
the buffer) and points every other face at a grey strip of the texture image; `--both_sides`
restores the old behaviour. The consistency numbers are computed on the textured faces
(`textured_faces` in `texture_transfer.json`; the `all_*` fields keep the whole mesh for
comparison).

### Refining the correspondence (`refine_correspondence.py`)

The supervisor's verdict on the nearest-neighbour texture: fine from afar, poor in the details,
improve it, and try manual landmarks. Nearest neighbour matches every target vertex on its own, so
neighbours jump between distant source points. `refine_correspondence.py` keeps the features and
adds three things, each optional:

- **anchors**: reliable pairs. Manual landmarks (see below), or automatically the mutual nearest
  neighbours (the target point's best source point has that target point as its best match),
  thinned to a spread subset by farthest-point sampling (`--max_anchors`, default 400).
- **prior**: a smooth map from target positions to source positions fitted on the anchors (an
  affine map plus Gaussian radial basis functions, ridge-regularised; a similarity transform
  instead of the affine map when there are fewer than 8 anchors). Every target vertex then chooses
  among its `--k` most similar source points (default 64) the one minimising
  `(1 - similarity) + w_prior * (distance to its predicted position / radius)^2`, with the radius
  5 % of the source diagonal.
- **smoothing**: `--iters` passes (default 2) in which every vertex re-chooses among its candidates
  with a penalty on the distance to the mean matched position of its one-ring neighbours.

The output is the same `t2s` table the other scripts read with `--correspondence`. On the synthetic
pairs (ground truth; fraction of target vertices within 1 % / 5 % of the diagonal of the true
source vertex):

| correspondence | cat | cow |
|---|---|---|
| nearest neighbour | 52.2 % / 94.3 % | 40.1 % / 87.2 % |
| + prior from 150 mutual anchors (k 32, radius 10 %) | 69.0 % / 97.5 % | 51.5 % / 98.7 % |
| + 400 anchors, k 64, radius 5 % | 73.0 % / 99.1 % | 55.3 % / 99.0 % |
| + 2 smoothing passes (the defaults) | 73.0 % / 99.2 % | 56.6 % / 99.0 % |
| smoothing alone, no prior | 52.3 % / 93.7 % | |
| 8 exact landmarks (from the ground truth), no smoothing | 92.3 % / 98.0 % | 91.6 % / 99.6 % |
| 16 exact landmarks | 92.7 % / 98.0 % | 91.8 % / 99.6 % |
| 32 exact landmarks | 92.8 % / 98.0 % | 92.0 % / 99.6 % |

The prior is what helps; smoothing on its own does nothing. Exact landmarks nearly solve the
synthetic pairs, but those pairs are rigidly related (the partial mesh is cut from the full one),
so a handful of exact pairs pins the map down; a real pair has different shapes.

**Manual landmarks.** Two ways in: MeshLab's PickPoints tool on both meshes (`.pp` files, same
features in the same order, `--landmarks_source` / `--landmarks_target`), or
`pick_landmarks.py`, which needs no 3D viewer: it renders a mesh with a pixel grid, you read the
pixel coordinates of the features off the image, and `--pick x,y ...` turns them into vertices
(a pick on the background snaps to the nearest surface pixel, for ear tips and paws, and a pick
returns the visible vertex nearest to it, so on a coarse mesh an ear tip gives the tip vertex),
writes them to a JSON and draws them numbered on the image for checking; `--zoom x0 y0 x1 y1`
with a finer `--grid` shows one region of the render for small features such as a nose or an eye,
and the picked image is the check that the same feature was picked on both meshes. `--merge`
joins the source and target files into the `--landmarks` JSON of `refine_correspondence.py`. Landmark vertices keep their
pair exactly; `--add_mutual` uses the mutual anchors as well. Eleven landmarks were picked this
way on the cat relief and the lion (nose, ear tip, eye, chin, a front and a hind paw, tail tip,
rump, back, belly, chest; `results/diff3f/landmarks/`). Cat → lion, textured side only:

| correspondence | within 2 % | within 5 % | source used |
|---|---|---|---|
| nearest neighbour | 69.0 % | 90.9 % | 58 % |
| automatic: 131 mutual anchors + prior + smoothing | 73.7 % | 96.1 % | 67 % |
| 11 manual landmarks | 70.3 % | 94.8 % | 63 % |
| 11 landmarks + the mutual anchors | 73.0 % | 96.2 % | 69 % |

On this real pair eleven landmarks alone sit between nearest neighbour and the automatic anchors;
with the mutual anchors added they edge past the automatic result. The landmarks guarantee the
picked features themselves (nose on nose, toes on toes) and change little elsewhere. Picking them
consistently matters more than their number: a throat paired with a chin, a cheek with a nose and
a heel with a toe went unnoticed until the owner checked the picked images.

**All pairs, automatic anchors**, textured side only (`texture_transfer_refined.{png,json}` next
to the nearest-neighbour files in every result folder):

| pair | textured triangles | nearest neighbour: 2 % / 5 % / used | refined: 2 % / 5 % / used |
|---|---|---|---|
| cow | 2 525 | 41.8 % / 84.1 % / 59 % | 49.4 % / 90.2 % / 70 % |
| cat | 4 343 | 66.7 % / 92.6 % / 61 % | 75.3 % / 97.8 % / 70 % |
| superman_man | 6 453 | 75.2 % / 92.8 % / 59 % | 88.4 % / 96.5 % / 59 % |
| superman_posed | 4 783 | 71.4 % / 88.4 % / 49 % | 77.2 % / 93.3 % / 52 % |
| okapi_camel | 1 629 | 35.8 % / 74.9 % / 52 % | 39.4 % / 84.7 % / 58 % |
| okapi1_camel | 1 398 | 32.4 % / 78.2 % / 61 % | 40.6 % / 91.2 % / 67 % |
| cat5_cat | 4 491 | 76.0 % / 92.0 % / 52 % | 79.1 % / 95.6 % / 66 % |
| cat_lion | 3 666 | 69.0 % / 90.9 % / 58 % | 73.7 % / 96.1 % / 67 % |
| labubu_man | 5 482 | 60.7 % / 76.9 % / 39 % | 67.9 % / 83.0 % / 50 % |
| chair1_armchair | 4 302 | 50.3 % / 75.8 % / 45 % | 53.3 % / 83.2 % / 66 % |
| chair_armchair | 4 120 | 48.3 % / 80.1 % / 49 % | 62.5 % / 87.0 % / 57 % |
| chair10_roundchair | 6 849 | 63.3 % / 76.2 % / 35 % | 67.1 % / 82.6 % / 51 % |
| chairgen_armchair | 4 566 | 48.1 % / 68.8 % / 52 % | 50.7 % / 75.6 % / 72 % |
| panda_panda | 16 153 | 78.3 % / 95.5 % / 73 % | 86.4 % / 97.8 % / 78 % |
| bird_bird | 16 661 | 49.1 % / 74.3 % / 51 % | 58.2 % / 83.8 % / 62 % |
| dragon_lizard | 15 622 | 64.8 % / 82.9 % / 50 % | 76.3 % / 87.5 % / 54 % |
| mean | | 58.2 % / 82.8 % / 53 % | 65.4 % / 89.1 % / 62 % |

Every real pair improves on all three numbers. The cross-category pairs gain less (mean
46.0 % / 67.1 % / 37 % → 48.5 % / 72.8 % / 45 %), and the four weakest of them (Labubu on the camel
and the lion, the stone bird on the cat, the chair on the cow) do not improve at 2 %: their mutual
anchors are as unreliable as the rest of the matching, so the prior fitted on them has nothing to
add.

### Smooth correspondence field (`refine_correspondence.py --field`)

Even after the prior, every target vertex still ends on one of the sampled source points, so
neighbouring vertices land a sample apart and a triangle's three corners rarely form the small
patch they should. `--field` fits a continuous source-position field to the matches instead: it
minimises the weighted distance of every vertex to its matched position plus a Laplacian
smoothness term along the target surface (a sparse linear solve; `--field_order 2` uses the
bi-Laplacian). The weights are the match similarities scaled between their 10th and 90th
percentiles, raised at the anchors and fixed high at landmarks. The texture coordinates come
straight from the field (`transfer_texture.py` reads `src_pos` from the npz), so nothing jumps
inside a triangle. On the synthetic ground truth a weak field (`--field 0.1 --field_order 2`, the
setting used below) keeps the accuracy of the matches (cat 73.3 % within 1 %, cow 57.5 %);
stronger smoothing trades it for continuity (`--field 1`: cat 64.8 %). Nearest neighbour,
refined matches and the field on every pair, textured side only (triangles within 2 % / within
5 % / relief used):

| pair | nearest neighbour | refined | smooth field |
|---|---|---|---|
| cow | 41.8 % / 84.1 % / 59 % | 49.4 % / 90.2 % / 70 % | 48.8 % / 90.6 % / 72 % |
| cat | 66.7 % / 92.6 % / 61 % | 75.3 % / 97.8 % / 70 % | 80.0 % / 99.7 % / 73 % |
| superman_man | 75.2 % / 92.8 % / 59 % | 88.4 % / 96.5 % / 59 % | 97.6 % / 98.6 % / 62 % |
| superman_posed | 71.4 % / 88.4 % / 49 % | 77.2 % / 93.3 % / 52 % | 75.8 % / 93.0 % / 56 % |
| okapi_camel | 35.8 % / 74.9 % / 52 % | 39.4 % / 84.7 % / 58 % | 34.4 % / 93.4 % / 60 % |
| okapi1_camel | 32.4 % / 78.2 % / 61 % | 40.6 % / 91.2 % / 67 % | 38.8 % / 97.4 % / 71 % |
| cat5_cat | 76.0 % / 92.0 % / 52 % | 79.1 % / 95.6 % / 66 % | 80.6 % / 99.2 % / 68 % |
| cat_lion | 69.0 % / 90.9 % / 58 % | 73.7 % / 96.1 % / 67 % | 76.2 % / 98.3 % / 70 % |
| labubu_man | 60.7 % / 76.9 % / 39 % | 67.9 % / 83.0 % / 50 % | 87.3 % / 91.2 % / 63 % |
| chair1_armchair | 50.3 % / 75.8 % / 45 % | 53.3 % / 83.2 % / 66 % | 51.8 % / 80.7 % / 82 % |
| chair_armchair | 48.3 % / 80.1 % / 49 % | 62.5 % / 87.0 % / 57 % | 59.7 % / 82.7 % / 68 % |
| chair10_roundchair | 63.3 % / 76.2 % / 35 % | 67.1 % / 82.6 % / 51 % | 57.1 % / 76.9 % / 78 % |
| chairgen_armchair | 48.1 % / 68.8 % / 52 % | 50.7 % / 75.6 % / 72 % | 44.5 % / 77.2 % / 93 % |
| panda_panda | 78.3 % / 95.5 % / 73 % | 86.4 % / 97.8 % / 78 % | 87.5 % / 96.3 % / 82 % |
| bird_bird | 49.1 % / 74.3 % / 51 % | 58.2 % / 83.8 % / 62 % | 72.8 % / 92.9 % / 61 % |
| dragon_lizard | 64.8 % / 82.9 % / 50 % | 76.3 % / 87.5 % / 54 % | 70.4 % / 88.9 % / 59 % |
| mean, 16 same-category pairs | 58.2 % / 82.8 % / 53 % | 65.4 % / 89.1 % / 62 % | 66.5 % / 91.1 % / 70 % |
| labubu → camel | 31.3 % / 52.5 % / 29 % | 30.3 % / 52.5 % / 35 % | 16.8 % / 56.5 % / 42 % |
| labubu → cow | 40.8 % / 62.6 % / 35 % | 41.6 % / 68.2 % / 43 % | 31.4 % / 72.2 % / 55 % |
| labubu → lion | 52.4 % / 71.9 % / 30 % | 46.6 % / 69.4 % / 43 % | 41.0 % / 73.8 % / 56 % |
| superman → camel | 33.7 % / 56.0 % / 27 % | 38.8 % / 64.8 % / 31 % | 37.1 % / 73.0 % / 39 % |
| panda → man | 58.9 % / 83.6 % / 63 % | 71.0 % / 93.3 % / 76 % | 96.5 % / 97.6 % / 84 % |
| cat → man | 55.5 % / 73.7 % / 54 % | 62.0 % / 76.9 % / 59 % | 82.8 % / 89.1 % / 67 % |
| cow → camel | 34.5 % / 71.5 % / 45 % | 34.8 % / 85.5 % / 51 % | 33.8 % / 97.2 % / 54 % |
| fire_dragon → man | 56.4 % / 74.4 % / 42 % | 69.0 % / 85.2 % / 51 % | 91.6 % / 96.6 % / 56 % |
| bird → cat | 58.4 % / 73.9 % / 22 % | 54.2 % / 74.4 % / 29 % | 50.1 % / 70.4 % / 62 % |
| chair1 → cow | 37.8 % / 50.4 % / 26 % | 36.6 % / 57.6 % / 35 % | 15.4 % / 49.3 % / 56 % |
| mean, 10 cross-category pairs | 46.0 % / 67.1 % / 37 % | 48.5 % / 72.8 % / 45 % | 49.6 % / 77.6 % / 57 % |

The 2 % column can fall while the texture gets better (the chairs, Labubu on the camel): a
collapsed triangle whose three corners sit on one source point spans nothing and counts as
consistent, and the field undoes exactly those collapses. The next section measures the map
itself.

### Texture-map quality (`uv_metrics.py`)

The transferred texture coordinates form a map from the mesh to the image, and a map is judged
the way mesh parametrisations are: per textured triangle, the linear map from the triangle's own
shape to its image has two singular values, their ratio is the **conformal distortion** (1: the
patch keeps its shape; large: a streak), and a **flip** is a triangle that lands on the image
mirrored (a reversed patch), tested against the triangle's winding on screen seen from the relief's
camera. A **collapsed** triangle has its three corners on one point. These are the numbers that
image-to-3D attribute-transfer papers report (for example RealSkin, arXiv 2607.12495, mCD and
Flip%), so they put our results on the same footing. `uv_metrics.py` reads the OBJ written by
`transfer_texture.py`; `results/diff3f/uv_quality.csv` has every pair and variant. Means over the
pairs (median distortion is the median over the pairs of each pair's median):

| 16 same-category pairs | median distortion | over tenfold | collapsed | flipped |
|---|---|---|---|---|
| nearest neighbour | 674 | 65.3 % | 15.2 % | 48.3 % |
| refined matches | 8.3 | 46.9 % | 7.5 % | 39.1 % |
| smooth field | 4.1 | 31.6 % | 1.7 % | 29.1 % |

| 10 cross-category pairs | median distortion | over tenfold | collapsed | flipped |
|---|---|---|---|---|
| nearest neighbour | 11 819 | 72.0 % | 15.3 % | 52.1 % |
| refined matches | 21.6 | 59.1 % | 6.4 % | 47.5 % |
| smooth field | 9.1 | 45.5 % | 3.5 % | 43.9 % |

Nearest neighbour alone maps half of the triangles mirrored and two thirds stretched more than
tenfold; the field brings the median distortion of a same-category pair to 4 and the flips to
under a third. A flip map (`outputs/diff3f/99_experiments/arap/cat_lion_flip_map.png`) shows
most of the remaining flips on silhouette slivers seen edge-on, whose winding on screen is not
reliable, so `flipped_area` (flips weighted by screen area) is the fairer figure.

### As-rigid-as-possible field (`--arap`)

The smooth field allows any smooth deformation, including local stretching and mirroring.
`--arap N` refines it so that every vertex's one-ring moves by a rotation and a scale: it
alternates a Procrustes fit of that similarity per vertex (reflections rejected, the scale kept
within 0.3 to 3 times the global size ratio) with a sparse solve for the positions, N times
(`--arap 10 --arap_weight 3` below). On the synthetic ground truth it keeps or improves the
accuracy (cat 71.5 %, cow 60.1 % within 1 %, against 73.3 % and 55.7 % for the smooth field), and
on the real pairs it is the step that makes the texture crisp: the rosettes on the lion stay round
and the stripes on the camel's legs run straight (`outputs/diff3f/07_texture_arap/overview/`,
`cat_lion_progress.png` shows the four steps on one pair). Means over the pairs:

| 16 same-category pairs | within 2 % / 5 % / relief used | median distortion | over tenfold | collapsed | flipped (area) |
|---|---|---|---|---|---|
| nearest neighbour | 58.2 % / 82.8 % / 53 % | 674 | 65.3 % | 15.2 % | 38.4 % |
| refined matches | 65.4 % / 89.1 % / 62 % | 8.3 | 46.9 % | 7.5 % | 27.6 % |
| smooth field | 66.5 % / 91.1 % / 70 % | 4.1 | 31.6 % | 1.7 % | 21.4 % |
| ARAP field | 68.0 % / 96.8 % / 66 % | 2.0 | 11.0 % | 0.0 % | 23.0 % |

| 10 cross-category pairs | within 2 % / 5 % / relief used | median distortion | over tenfold | collapsed | flipped (area) |
|---|---|---|---|---|---|
| nearest neighbour | 46.0 % / 67.1 % / 37 % | 11 819 | 72.0 % | 15.3 % | 46.4 % |
| refined matches | 48.5 % / 72.8 % / 45 % | 21.6 | 59.1 % | 6.4 % | 40.7 % |
| smooth field | 49.6 % / 77.6 % / 57 % | 9.1 | 45.5 % | 3.5 % | 38.5 % |
| ARAP field | 52.4 % / 90.2 % / 53 % | 2.5 | 12.3 % | 0.0 % | 38.3 % |

Per pair, ARAP field (within 2 % / 5 % / relief used): cow 48.2 / 97.0 / 66, cat 79.0 / 100 / 76,
superman_man 97.0 / 99.8 / 62, superman_posed 78.3 / 97.9 / 49, okapi_camel 37.0 / 95.8 / 61,
okapi1_camel 42.2 / 96.4 / 65, cat5_cat 80.8 / 100 / 72, cat_lion 76.0 / 98.2 / 65, labubu_man
84.0 / 97.2 / 67, chair1_armchair 49.5 / 90.0 / 76, chair_armchair 60.6 / 93.2 / 66,
chair10_roundchair 71.2 / 92.5 / 65, chairgen_armchair 42.1 / 92.5 / 78, panda_panda
84.6 / 99.8 / 73, bird_bird 78.3 / 99.1 / 62, dragon_lizard 78.5 / 99.8 / 56; cross pairs labubu →
camel 27.0 / 76.4 / 36, labubu → cow 33.1 / 88.0 / 51, labubu → lion 42.9 / 92.4 / 52, superman →
camel 41.7 / 84.3 / 34, panda → man 95.5 / 99.0 / 86, cat → man 81.7 / 94.4 / 69, cow → camel
37.4 / 99.4 / 48, fire_dragon → man 90.0 / 98.2 / 57, bird → cat 53.6 / 87.5 / 57, chair1 → cow
20.9 / 82.3 / 44 (`texture_transfer_arap.json` in every result folder).

The flips do not move with ARAP: they are not triangle-scale noise but regions where the features
place a patch mirrored, and a rigidity term cannot know which way round a patch should go. Two
things that did not help, kept under `outputs/diff3f/99_experiments/`: subdividing the target
twice before matching (finer triangles, same features: distortion and flips per triangle got
slightly worse), and the rigid registration of `register_partial.py` as the position prior
instead of the mutual anchors (no better on four pairs).

### Landmarks on the cross-category pairs

The supervisor's landmark question matters most where the automatic anchors fail, so ten
landmarks (nose, head top, ear tip, eye, chin, neck base, chest, belly, front toe, hind toe) were
picked on the Labubu relief and on the camel and the lion. Two annotation agents picked them with
`pick_landmarks.py`, two independent checking agents verified every pair on zoomed renders and
against the meshes' extreme points, and their corrections (the Labubu chin had been put on the fur
under the face by both annotators; one camel toe was on the far foot) were applied before use;
the contact sheets are in `results/diff3f/landmarks/`. Textured side only:

| pair | correspondence | within 2 % | within 5 % | relief used | median distortion | flipped |
|---|---|---|---|---|---|---|
| labubu → camel | nearest neighbour | 31.3 % | 52.5 % | 29 % | 56.4 | 47.0 % |
| labubu → camel | automatic anchors | 30.3 % | 52.5 % | 35 % | 17.5 | 43.5 % |
| labubu → camel | automatic anchors + field | 16.8 % | 56.5 % | 42 % | 9.5 | 49.7 % |
| labubu → camel | 10 landmarks | 30.6 % | 57.0 % | 34 % | 17.5 | 40.7 % |
| labubu → camel | landmarks + automatic anchors | 31.1 % | 55.2 % | 34 % | 16.9 | 42.6 % |
| labubu → camel | landmarks + anchors + field | 18.5 % | 57.3 % | 40 % | 9.9 | 47.1 % |
| labubu → lion | nearest neighbour | 52.4 % | 71.9 % | 30 % | (degenerate) | 55.3 % |
| labubu → lion | automatic anchors | 46.6 % | 69.4 % | 43 % | 25.6 | 51.0 % |
| labubu → lion | automatic anchors + field | 41.0 % | 73.8 % | 56 % | 8.5 | 42.9 % |
| labubu → lion | 10 landmarks | 57.1 % | 75.8 % | 39 % | 20.9 | 50.7 % |
| labubu → lion | landmarks + automatic anchors | 50.0 % | 72.0 % | 41 % | 19.7 | 49.8 % |
| labubu → lion | landmarks + anchors + field | 44.2 % | 75.8 % | 52 % | 8.6 | 41.7 % |

Ten landmarks help these pairs a little (the fewest flips without the field, the best 5 % figure
with it) and do not change the picture: a toy and a camel differ too much for ten points to pin
down, and the field remains the main source of continuity. Where the shapes are closer (the
panda, the dragon and the cat on the man) the automatic pipeline already reaches 83 to 97 %
within 2 %.
