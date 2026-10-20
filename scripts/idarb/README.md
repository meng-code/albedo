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
| Partial-to-full registration with Diff3F features: cow median 0.0199 / cat 0.0116 of the target diagonal, agreement 0.563 / 0.689, feature-free ICP baseline 0.0217 / 0.049 | `diff3f_features.py` (server, diff3f environment, 4 meshes), then `register_partial.py` (section 5) | `results/diff3f/{cow,cat}/metrics.csv` |
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
RANSAC runs.

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
  median 0.0199, agreement 0.563, part agreement 0.585; cat 0.0116 / 0.689 / 0.829 (5 to 10 s per pair).
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
