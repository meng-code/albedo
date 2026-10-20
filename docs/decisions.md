# Technical decisions

## 2026-07-15 — Project reset

- The human changed the capstone topic to **AI-Based Image Albedo Estimation and Relighting** under the supervision of Dr Juncheng Liu.
- Published Git history will be preserved while the current repository content and package naming move from the obsolete LLM security project to `albedo-relight` / `relight`.
- The initial smoke pipeline is dependency-light plumbing only. It is not an albedo estimator or a scientific relighting baseline.
- Model, training, dataset, input-domain, relighting, and evaluation choices remain open pending literature and feasibility review.
- The human confirmed in Codex chat on 2026-07-15 that AI use is unrestricted for this course provided complete records are retained; the human owns the final submissions and audit record.

## 2026-09-25 — Repository set up for long-term use

The repository will be used beyond the original course, so the owner rewrote `AGENTS.md` and `CLAUDE.md` as general project guidance:

- Course-specific rules (deadlines, semester scope, AI-use record-keeping) were removed.
- Codex cross-review is retired. GitHub Issues are optional. Agents work on `claude/<short-description>` branches, never push to `main`, and the owner merges through pull requests.
- Agents may run jobs on the shared GPU cluster when the owner asks, and may commit under the owner's git identity.
- Obsolete branches `claude/6-agents-default-strategy` and `claude/7-validation-rejection-tests` were deleted; their commits remain reachable through pull requests #9 and #10.
- `third_party/` keeps upstream style and Python version and is excluded from ruff.
- **Update 2026-09-30:** at the owner's request, branch names no longer carry the `claude/` prefix; they are a short hyphenated description such as `depth-prior-mesh`, and `AGENTS.md` was updated. Existing branches were renamed on GitHub: `claude/depth-prior-mesh` to `depth-prior-mesh`, and the merged `claude/idarb-integration` to `idarb-integration`. Pull request #11 still shows the old branch name; its commits are unchanged and remain in `main`.

## 2026-09-25 — IDArb as the first albedo model to reproduce

- With the supervisor's agreement, the first model to get running is **IDArb** (Li et al., arXiv:2412.12083), which decomposes an image into albedo, normal, metallic and roughness. Scope agreed with the supervisor: run the official code, get results on the bundled examples and on our own images, and understand each step; matching the paper's metrics is not required.
- The official code is vendored unmodified at `third_party/IDArb/` (upstream commit `df26b90`); local changes are separate commits.
- **Where things run:** IDArb decomposition needs a GPU and runs on the cluster. Everything downstream (for example normal-to-mesh) runs on the owner's laptop. Going forward the work uses single images only; the training data is not needed locally (supervisor, 2026-09-25).
- **Training smoke test** (supervisor request): train IDArb from Stable Diffusion 2.1 weights, not IDArb's released weights, on a few public objects to confirm the training code works. Result: 3 Arb-Objaverse objects, 300 steps on one A40 in 7m22s; validation normal cosine similarity rose from -0.19 to 0.79. This shows the model can fit the training objects, not that it generalises. Inferring at 512 px with weights trained at 256 px lowered normal cosine similarity from 0.93 to 0.77.
- **Upstream bug fixed:** `mv_dataset.py` `load_mr_abo` used `out_imgs` before creating it, so training on ABO or Arb-Objaverse crashed.
- **Normal-to-mesh** (supervisor suggestion: Poisson reconstruction): implemented both Poisson normal integration and screened Poisson surface reconstruction in `scripts/idarb/normal_to_mesh.py`. The supervisor observed that single-view meshes look flattened. Experiments traced most of this to depth discontinuities at occlusion boundaries, which normals cannot encode (true gaps of 49/99/149 px were recovered as about 5 px). Per the supervisor, meshes are now centred at the origin and can be stretched along Z (`--z_scale`).
- **Multi-view reconstruction** is not pursued; the supervisor considers novel-view-synthesis methods sufficient for that.

## 2026-09-30 — Depth prior for normal-to-mesh

- **Problem:** meshes from normal integration look flattened. The supervisor noted that even the visible surface is squashed. The cause is depth discontinuities at occlusion boundaries, which normals cannot encode, plus small normal biases that integrate into a global tilt.
- **Change:** add a monocular depth prior. `scripts/idarb/estimate_depth.py` runs Depth Anything V2 Small on the server; `scripts/idarb/normal_to_mesh.py --depth_dir` solves `min Σ(Δh − g)² + w·Σ(h − s·d)²`, drops gradient constraints across depth jumps detected in the depth map, merges fragments smaller than 300 px back into the neighbouring part they share the most cut edges with, and removes the "wall" triangles that span a cut. Without `--depth_dir` the behaviour is unchanged.
- **Results:** on a synthetic two-sphere scene the recovered occlusion gap is 99.4 px (true 99.4; normals only 5.4). On samples, chair1 depth/width goes from 0.28 to 0.82 and labubu's tilt disappears; objects without occlusions change little. Surface-to-normal deviation grows (chair1 4.7° to about 11°); the weight barely changes the depth ratio between 0.01 and 0.2, so the default is 0.01.
- **Fragment fix:** a first version cut thin parts into many floating pieces (chair1: 67 parts, 64 under 300 px), reported from the Mac run. Merging each small fragment into only one neighbour removes them (chair1: 3 parts) without re-gluing parts that should stay apart (superman's cape stays separate from the body); merging a fragment into all neighbours did glue them.
- **Model choice:** Small (Apache-2.0, 99 MB) instead of Large (CC-BY-NC-4.0, 1.3 GB); on five samples their depth maps correlate at 0.94 to 0.997 and give nearly identical meshes.
- **Workflow:** depth maps are produced on the server and pushed with the decomposition results, so mesh reconstruction on the laptop still needs no PyTorch.
- **Correction on screened Poisson (method B):** on single images it does not produce a closed mesh or a back side, contrary to earlier comments. On panda, chair1 and bird the raw output is the front surface plus a flat sheet extending from the object's outline to the bounding box; after density trimming only the front remains, with a median distance of about 0.45 px to the method-A surface. The preview's 180° view had looked like a back side because the preview shades both faces of a surface alike. Comments and the README were corrected.

## 2026-10-05 to 2026-10-06 — Relighting dropped; new project title

- The supervisor agreed to drop relighting from the project. The title is now
  **Textured mesh generation by single image decomposition**, replacing "AI-Based Image Albedo Estimation
  and Relighting" from 2026-07-15.
- Scope: decompose a single image with IDArb, rebuild the visible surface from the normals
  (`scripts/idarb/normal_to_mesh.py`, optionally with the depth prior), and colour the mesh with the albedo.
- The repository name `albedo-relight` and the placeholder package `src/relight/` are kept, to avoid
  renaming the repository, the package and the tests. `README.md` and `AGENTS.md` now describe relighting
  as dropped.
- A failure case was recorded in `results/analysis/failure_cats.md`: on cat2, cat3 and cat4 IDArb predicts
  normals that almost all face the camera, and because the depth-prior scale is fitted to the normals it
  also goes to nearly zero, so the meshes stay flat. `normal_to_mesh.py` was not changed.

## 2026-10-10 — Tried: Poisson reconstruction on depth-map points (not adopted)

- **Supervisor's suggestion:** build a point cloud directly from the Depth Anything depth map (one point
  per pixel, z = s·d), attach IDArb's normal to every point, and run screened Poisson surface
  reconstruction on that cloud, so that the depth map gives the front-to-back structure and the normals
  give the surface detail.
- **Result:** the normals have almost no effect. Mean angle between the mesh's normals and IDArb's
  normals (smaller is better; the joint solve is the current default):

  | Object | normals only | joint solve (normals + depth) | depth map alone | Poisson on depth points |
  |---|---|---|---|---|
  | chair1 | 4.8° | 11.0° | 17.0° | 16.9° |
  | superman | 7.6° | 13.9° | 17.9° | 18.0° |
  | panda | 3.3° | 7.2° | 11.9° | 11.4° |

  The depth-to-width ratio is the same for all depth-based variants (chair1: 0.82). Octree depths 6 to 9
  give 20.2° to 16.9° on chair1, so a coarser octree does not help either.
- **Why:** screened Poisson pins the surface to the input points. With a point in every octree cell
  (depth 9 is about one pixel), the surface passes through the depth-map points, including their noise and
  blurred occlusion edges, and the normals only shape the surface inside each cell. The joint solve in
  `normal_to_mesh.py` weights the depth term at 0.01, so its surface can leave the depth-map points to
  follow the normals; in effect the Poisson variant is the joint solve with a very large depth weight.
  Open3D 0.18 does not expose PoissonRecon's point weight, which would be the parameter to lower.
- **Decision:** keep the joint solve. Script: `scripts/idarb/poisson_on_depth_points.py`.

## 2026-10-16 — Repository cleaned up to match the current project

- Removed the relighting placeholder package `src/relight/`, its test `tests/test_smoke.py`, the empty
  `configs/` and the unused template `docs/diary.md`. The repository is no longer an installable package;
  `pyproject.toml` only holds the ruff and pytest settings and a `test` dependency group, and ruff now
  targets Python 3.8 because `scripts/idarb/` also runs in the server environment.
- New tests in `tests/test_normal_integration.py` check the hemisphere and two-sphere scenes from
  `normal_to_mesh.py --selftest`. To share the code, the self-test was split into `hemisphere_check()` and
  `occlusion_check()`, which return numbers; `--selftest` prints exactly what it printed before.
  `occlusion_check()` now takes the distance between the spheres, so the 49.4 / 99.4 / 149.4 px cases and
  the single-sphere control in the report can be recomputed.
- Added scripts so that the numbers in the report can be recomputed in the repository:
  `eval_samples.py`, `eval_occlusion.py`, `eval_method_b.py` (laptop) and `compare_depth_models.py`.
  Rerun on 2026-10-16, they reproduce the report: gaps 49.4 / 99.4 / 149.4 px recovered as 5.4 px with
  depth ratios 0.71 / 0.49 / 0.38 and 1.00 for a single sphere; 16 of 19 images usable, median
  depth/width 0.35 to 0.38, mean angle 5.5° to 11.2°; method B to method A median distance 0.456 /
  0.448 / 0.438 px on panda / chair1 / bird. Results are in `results/analysis/*.csv`.
- On the GPU server, `eval_idarb_gt.py` turns the inline code from the 2026-09-25 session into a script
  (same 8 objects, views 00 and 06) and reproduces the IDArb-vs-ground-truth numbers exactly: mean angle
  17.2°, outline n_z 0.59 vs 0.59, depth ratio median 1.00 (0.43 to 1.55). The angle is the mean of
  per-image mean angles, and the outline n_z is the mean of per-image medians on the outline ring.
- The Depth Anything Small-vs-Large check on the same five samples (chair1, superman, labubu, panda, bird)
  gives correlations of 0.93 to 0.996 with `compare_depth_models.py`, which uses the full object mask.
  The 0.94 to 0.997 in the 2026-09-30 entry came from a one-off script that used the mask shrunk by 2 px.
  The report now quotes 0.93 to 0.996.
- `AGENTS.md` was merged into `CLAUDE.md` and removed. `README.md` was rewritten for the current project,
  and `NOTICE.md` now lists rembg (MIT) and the U²-Net model it downloads (Apache-2.0).

## 2026-10-20 — Next step: register partial meshes to full meshes with Diff3F features

- **Direction (supervisor):** after the single-image mesh, do "3D point set registration" and "mesh and
  partial mesh registration", using the features of Diff3F (Dutt, Muralikrishnan and Mitra, CVPR 2024,
  https://diff3f.github.io). Our relief mesh is the partial mesh; a full mesh of the same kind of object
  is the target. The correspondence found this way is what will later carry the albedo from the image
  onto a full mesh.
- **Why Diff3F:** its per-vertex features are semantic, not geometric, so they match across different
  poses, proportions and coordinate systems and need no training; the paper reports state-of-the-art
  correspondence on SHREC'19/'20. The paper does not test partial shapes; whether its features hold up on
  our one-sided relief meshes is the open question of this experiment.
- **Code:** `third_party/Diff3F/` (unmodified upstream, commit 5bda687, MIT; see `NOTICE.md`),
  `scripts/idarb/diff3f_features.py` (server, new `diff3f` environment) and
  `scripts/idarb/register_partial.py` (CPU). Design choices:
  - `runwayml/stable-diffusion-v1-5` no longer exists on Hugging Face; the script loads the same weights
    from the `stable-diffusion-v1-5/stable-diffusion-v1-5` mirror, fp16 files only (about 4 GB in total
    with the two ControlNets and DINOv2).
  - Features are stored for at most 15000 randomly chosen vertices (the whole mesh is still rendered),
    as the paper does for point clouds; a full relief mesh has 36k to 92k vertices.
  - `--front_only` renders the partial mesh only from the hemisphere in front of it (elevation ±45°,
    azimuth ±75°), because a one-sided sheet seen from behind would give meaningless images.
  - Registration: mutual nearest-neighbour feature matches → RANSAC similarity transform (Umeyama) →
    rigid ICP with the scale fixed, plus a feature-free ICP baseline. Distances are reported as fractions
    of the target's bounding-box diagonal.
- **First experiment (planned):** cow and cat. Source: `outputs/mesh_depth/{cow,cat}/relief.ply` (our
  meshes with the depth prior); target: `third_party/Diff3F/meshes/cow.obj` (2759 vertices) and
  `meshes/cat.off` (7207 vertices), the sample meshes that ship with Diff3F.
- **Feature extraction (server, A40):** 100 views per mesh, 15000 sampled vertices for the partial meshes.
  Run times 373 to 947 s per mesh (6 to 13 minutes of GPU work, peak 10 to 12 GB), far slower than the
  2 to 3 minutes the paper reports, because Diff3F enables diffusers' model CPU offload; `--no_cpu_offload`
  was added for the next run. Commands, commit and timings are in the commit message of
  `results/diff3f/` (first run) and in `transform.json`.
- **First registration run (mutual matches only, best RANSAC hypothesis):** cat works, cow fails.
  Cat: median distance to the target 0.0118 of the diagonal, 75.5 % of points within 2 %
  (feature-free ICP baseline: 0.0491 / 23.6 %, and the wrong way round). Cow: only 144 mutual matches;
  RANSAC chose a transform with scale 0.295 (bounding-box ratio 0.51) that put the whole partial cow on
  the full cow's head, median 0.0313 / 33.9 %, worse than the baseline (0.0217 / 45.2 %). The cartoon
  cow's head is much larger than the realistic cow's, so the head matches were the most self-consistent.
- **Second run (same features, registration changed):** matches in both directions (8808 for cow,
  11037 for cat), scale limited to 0.4 to 2.5 times the bounding-box ratio, up to 8 RANSAC hypotheses
  refined by ICP and the one with the best feature agreement selected. Cow: scale 0.429, median 0.020,
  RMS 0.031, 90.7 % of points paired by ICP, agreement 0.565 (baseline 0.0217 / 0.0418 / 0.420); the
  head-only transform (scale 0.29) is still among the candidates with agreement 0.27 and is rejected.
  Cat: median 0.0117, agreement 0.689 (baseline 0.049 / 0.245). Both previews show the partial mesh on
  the right part of the full mesh. Run on the laptop with the server's feature files;
  `results/diff3f/{cow,cat}/` holds this run.
- **What the residual means:** a similarity transform cannot fit a cartoon cow's head and body to a
  realistic cow at the same time; the 2 % median is the compromise. If the final goal (albedo transfer by
  correspondence) needs closer surfaces, the next step is a non-rigid refinement after this alignment,
  or using the feature correspondences directly without a global transform.
- **Registration ablation (same day, `scripts/idarb/eval_registration.py`, `results/diff3f/ablation.csv`;
  details in `scripts/idarb/README.md`, section 5):**
  - Ground truth: `make_partial_mesh.py` cuts the camera-visible part of Diff3F's cow and cat meshes
    (1248 of 2759 and 2821 of 7207 vertices) into the camera frame with a known transform; their features
    are computed like the real relief meshes (`--front_only`, with `--no_cpu_offload`: about 4 minutes
    per mesh, 10.3 GB). Every setting registers them exactly (rotation error 0°, correspondences 100 %
    within 1 % of the diagonal), so the method itself is sound on partial meshes; the 2 % residual on the
    real cow is the cartoon-versus-realistic gap. The features alone place only 30 to 39 % of the points
    within 1 % of the true point (86 to 97 % within 5 %): they find the part, the geometry the point.
  - Kept as default: matches in both directions, 8 candidates, geometric ICP with the scale fixed,
    selection by feature agreement. The consensus filter, normal checks and more seeds change nothing.
  - Not default: feature-guided ICP (with a refitted scale it shrinks the source; with the scale fixed it
    equals plain ICP at 5 to 8 times the time) and the depth factor on z (recovers a halved or doubled
    relief depth to within 10 to 30 % on the synthetic meshes and fixes the pose, but is not determined
    on the real cat, whose body is thinner than the 5 % tolerance: 0.48 or 1.76 depending on the start).
  - The registered partial lies on the near side of the target (normals agree for 90 % / 93 % of close
    pairs), so the albedo transfer `target_from_source_colors_registered.ply` paints the right side.
- **Painting and per-part refinement (same day):** the albedo transfer now paints a subdivided
  target mesh and chooses, for every covered vertex, the source point that is close, of the same
  k-means part and of the most similar feature (`paint_target()`); stripes and markings follow the
  body, but detail smaller than the registration residual (the cartoon cow's eyes and muzzle) still
  lands next to, not on, the real feature. `--piecewise` aligns each k-means part on its own with a
  clipped scale and rotation, blended between parts; safe on the synthetic pairs with ground truth,
  helpful on the depth-squashed ones and on the cat (median 0.0116 → 0.0099, agreement
  0.689 → 0.707), no change on the cow because its head would need more deformation than the limits
  allow. Kept optional; the next step for such cases is a real non-rigid deformation of the head.
