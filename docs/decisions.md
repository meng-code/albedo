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
- **Seven more pairs (server batch 1, same day; `results/diff3f/<pair>/`):** superman → man.obj and
  posed_human.off, okapi and okapi1 → camel.obj, cat5 → cat.off, cat → lion.off, labubu → man.obj.
  All register (medians 0.011 to 0.028 of the diagonal; table in `scripts/idarb/README.md`). The
  cross-species pairs work as the paper claims: the okapi's striped legs land on the camel's legs and
  the cat's spots on the lion. The toy-to-human pair has a low feature agreement (0.32), so its small
  distances mean little. Next: chairs, panda, bird and dragon against CC-BY Objaverse meshes
  (`results/diff3f/targets/`, batch 2).
- **Seven more pairs (server batch 2, 2026-10-21; Objaverse targets):** chairs, panda, bird and
  dragon. Five register well (panda best: median 0.0104, agreement 0.81), chair10 is poor (its relief
  mesh is in 6 pieces), and the stone-bird figurine is a failure of a new kind: the registration
  shrinks the blob onto the sparrow's belly (small distances, part agreement 0.46). Table in
  `scripts/idarb/README.md`.
- **Painting with ground truth, symmetric completion (2026-10-21, night):** `eval_painting.py` paints
  the synthetic partial meshes back onto their full meshes with a known mirror-symmetric colour field:
  covered vertices come out within 3.6 / 4.4 (RGB, 0 to 255), feature-only transfer within 7.7 / 8.0,
  and the nearest-point, feature-guided and part-aware painting variants are indistinguishable when
  the alignment is exact. `paint_target()` now fills the far side of a bilaterally symmetric target
  through its symmetry plane (mirrored vertices as accurate as covered ones on the synthetic field;
  cow 17 % of the vertices, panda 7 %). Also fixed: without `--piecewise` the painted files were
  overwritten by point clouds (an `else` attached to the wrong `if`).
- **Scale bands and coverage discount (2026-10-21, night):** the stone bird collapsed because the
  best-inlier RANSAC solution shrank it into the sparrow's belly (scale 0.38 of the bounding-box
  ratio, coverage 0.16). RANSAC now runs per scale band so candidates exist at every plausible
  scale, and the selection score discounts candidates that cover less than 0.2 of the target
  (registered pairs: 0.24 to 0.75). Averaging the source- and target-side agreements was tried and
  rejected: it rewards larger scales (the dragon moved to a worse solution). Validated on all 22
  pairs before adoption (see `ablation.csv`).
- **Stored results slimmed (2026-10-21, night):** `results/diff3f/<pair>/` now keeps only metrics,
  transform, correspondences, `render.png` and one painted mesh (decimated to 40000 triangles when
  larger): 24 MB for 18 folders instead of 192 MB. The branch history still carries the larger files
  the first result pushes contained (about 180 MB in `diff3f-more-pairs`, `diff3f-more-pairs2` and
  earlier commits); whether to rebuild the branch without them before merging is the owner's call.
- **Depth factor with ground truth on deformed geometry (2026-10-21, night):** synthetic cow partials
  cut with the depth halved / doubled and featured as such: `--depth_scale` finds 1.76 (true 2) and
  0.65 (true 0.5), cutting the rotation error from 12.8° / 16.4° to 2.1° / 5.5° and lifting the
  correspondences within 1 % from 15 % to 94 % (halved case); harmless when the depth is right.
  Still optional, because on thin real objects the agreement cannot pick the factor.
- **Full-resolution colours (2026-10-21):** after reading Foti et al., "Heat Kernel Textures"
  (arXiv 2609.07557; a UV-free texture representation fitted by differentiable rendering, not a
  correspondence method), the cheap part of its lesson was applied: the painting now takes colours
  from every source vertex (36k to 92k) instead of the 15000 feature points, and subdivides the
  target twice. Sharper stripes and markings; placement unchanged. HKTex itself would only improve
  how the transferred texture is stored and rendered; adopting it means Mitsuba 3 + Dr.Jit on the
  GPU, deferred until the supervisor confirms the direction. Functional maps (pyFM inside
  third_party/Diff3F) are the more relevant next step for placement.
- **Functional maps, not adopted (2026-10-21):** with Diff3F descriptors and pyFM, 50 or 100
  eigenfunctions, whole or cropped target, with ZoomOut: 13 to 24 % of the target vertices land
  within 1 % of the true source vertex on the synthetic pairs, against 40 to 52 % for feature
  matching and 100 % for the registration (`results/diff3f/functional_map_eval.csv`). The method
  assumes shared low-frequency structure, which a one-sided relief and a full body lack. The eye
  placement therefore needs either a non-rigid deformation driven by the features or a better
  single-image mesh, not a different correspondence solver.

## 2026-10-21 — Supervisor's direction after the group meeting: the correspondence first

- The priority is the semantic correspondence between the partial mesh and the full mesh: make it
  accurate, show it as "same part, same colour" maps on both meshes, and start from nearest-neighbour
  matching in feature space. Painting is an application of the correspondence, not the goal.
- `scripts/idarb/correspondence_maps.py` draws the maps (joint feature PCA, and the target coloured by
  its nearest neighbour) for every pair; `results/diff3f/<pair>/correspondence_maps.png`. The feature
  maps agree part by part on all 16 real pairs; nearest neighbour alone places 40 to 52 % of the
  target vertices within 1 % of the true vertex on the synthetic pairs (87 to 94 % within 5 %), with
  the errors inside parts. Next: improve the nearest-neighbour correspondence itself (denser feature
  sampling, more views or other UNet layers on the server; mutual-neighbour and surface-consistency
  filters on the laptop), measured on the synthetic pairs; the registration (100 % on them) is the
  upper bound.

## 2026-10-21 — Texture transfer, not vertex colours

- Shown the albedo carried over by the nearest-neighbour table (one colour per vertex, the lion's
  5 000 vertices), the supervisor said this is not what is wanted: the texture itself must be
  transferred, so that the inside of a triangle shows the image and not a blend of three colours.
- `scripts/idarb/transfer_texture.py` transfers texture coordinates instead of colours. The relief is a
  height field over the albedo image, so each relief vertex is one pixel and the albedo image is its
  texture; every target vertex takes the texture coordinates of the relief vertex it is matched to.
  Output: `target_textured.obj` + MTL + PNG per pair (MeshLab shows the texture), a render
  (`results/diff3f/<pair>/texture_transfer.png`) and, as the measure of the correspondence's
  consistency, the fraction of triangles whose corners map within 2 % / 5 % of the image diagonal of
  each other (`texture_transfer.json`). Cat → lion: 66.9 % / 85.6 % of all triangles.
- The correspondence table is an input (`--correspondence`), so any improvement of the matching is
  judged by rerunning this script: more consistent triangles, fewer streaks.

## 2026-10-21 — Cross-category pairs, and a second number next to the triangle span

- The supervisor asked for very different source and target (the Labubu toy on the camel). Ten
  cross-category pairs were run from the features alone, without registering the pair:
  `results/diff3f/cross/`. The part level holds (head on head with the eyes at the eyes, shirt on the
  body, jeans on the legs, shoes on the hooves; Superman's boots at the bottom of the camel's legs;
  the panda's black limbs and white belly on the man); what the source lacks is filled with the
  nearest thing (the camel's hump takes the head). Mean consistency 46 % / 65 % against 57 % / 79 %
  within a category. A control pair with no relation (chair on the cow) also gives a picture, so the
  pictures are judged by the numbers.
- The triangle span alone rewards a collapse of the matching onto a few source points (all spans
  zero), which the stone bird showed (67.7 % within 2 % while using 28 % of its relief).
  `transfer_texture.py` therefore also reports **source used**, the fraction of the relief's image
  cells that receive at least one match; the two are read together.

## 2026-10-21 — Improving the details: one side only, a prior for the matching, manual landmarks

- Supervisor's verdict on the nearest-neighbour texture transfer: fine from afar, poor in the
  details; improve it; texture one side only and leave the back blank; try manual landmarks.
- `transfer_texture.py` textures only the faces visible from the relief's camera (depth-buffer
  visibility); the rest point at a grey strip of the texture. Numbers are on the textured faces.
- `refine_correspondence.py`: anchors (mutual nearest neighbours, or landmarks) → a smooth
  target-to-source position prior (affine + Gaussian RBF) → each target vertex takes the most
  similar of its 64 candidates near its predicted position → two smoothing passes. Synthetic ground
  truth: cat 52 → 73 % within 1 %, cow 40 → 57 %; with 8 exact landmarks 92 % (the synthetic pairs
  are rigidly related, so exact landmarks pin the map down). The prior is what helps; smoothing on
  its own does nothing.
- Landmarks: MeshLab .pp files or `pick_landmarks.py` (grid render, pixel coordinates, snap to the
  silhouette). Eleven picked on cat → lion (nose, ear tip, eye, chin, paws, tail tip, rump, back,
  belly, chest, checked on zoomed renders and silhouette extremes after the owner caught a chin
  matched to a throat, a cheek matched to a nose and a heel matched to a toe): 70 % / 95 % of the
  textured triangles within 2 % / 5 %, against 69 % / 91 % for nearest neighbour and 74 % / 96 %
  for the automatic anchors; the landmarks plus the mutual anchors give 73 % / 96 %, the best of
  the four. Consistent picking matters more than the number of landmarks. On a real pair a few landmarks do not beat the
  automatic anchors; they guarantee the picked features and little else.

## 2026-10-22 — Overnight: smooth field, texture-map quality, landmarks on the hard pairs

- The owner asked for the texture to come out clearer and left the night to it. Three things:
- **Smooth correspondence field** (`refine_correspondence.py --field 0.1 --field_order 2`): a
  continuous source-position field fitted to the matches with Laplacian regularisation, texture
  coordinates taken straight from it. Keeps the synthetic accuracy of the matches, lifts the
  16 same-category pairs to 66.5 % / 91.1 % / 70 % (triangles within 2 % / 5 %, relief used) from
  65.4 % / 89.1 % / 62 %, and visibly removes the streaks (rosettes and stripes stay whole on the
  lion). Stronger smoothing is cleaner still but costs synthetic accuracy; left at 0.1.
- **Texture-map quality** (`uv_metrics.py`, `results/diff3f/uv_quality.csv`): conformal distortion
  and flipped triangles, the measures RealSkin (arXiv 2607.12495) reports for the same
  image-to-3D setting (a partial surface from one photo, the target cut to the visible part). Same
  category: median distortion 674 → 8.3 → 4.1 and flips 48 → 39 → 29 % for nearest neighbour,
  refined matches and the field. Flips are the next target; a re-solve that down-weights the
  vertices of flipped triangles (`--field_iters`) was tried and barely moved them (24 → 23 % on
  cat → lion) while costing synthetic accuracy, so it is not used.
- **Landmarks on Labubu → camel and Labubu → lion**: picked by two Opus annotation agents and
  verified by two Opus checking agents (both annotators had put the Labubu chin on the fur under
  the face; one camel toe was on the far foot; fixed before use). Ten landmarks reduce flips by
  4 to 6 points and lift the 5 % figure by about 4 points on these pairs; they do not make a toy
  and a camel correspond. The automatic pipeline already reaches 83 to 97 % within 2 % on the
  cross pairs whose shapes are closer (panda, dragon and cat on the man).
- Literature checked tonight: SemTex (the group's own paper; code not yet released) works in image
  space with patch matching plus diffusion refinement; DenseMatcher and TriTex need trained
  networks; RealSkin's code is unreleased. Nothing to pull and run on the laptop; the ideas taken
  are the sub-mesh cut to the visible side and the distortion/flip metrics.
- **As-rigid-as-possible field** (`--arap 10 --arap_weight 3`, folder `07_texture_arap/`): every
  vertex's one-ring is asked to move by a rotation and a scale, reflections rejected. The step that
  makes the texture crisp: same-category median conformal distortion 4.1 → 2.0, triangles stretched
  more than tenfold 31.6 → 11.0 %, within 5 % 91.1 → 96.8 %, synthetic accuracy kept. Cross pairs
  9.1 → 2.5 and 45.5 → 12.3 %. Flips stay (23 % of the screen area on same-category pairs): they are
  mirrored regions of the feature matching, not triangle noise.
- Not adopted: subdividing the target before matching (no gain per triangle), the rigid
  registration as the prior (no better than the mutual anchors), re-solving the field with the
  flipped triangles down-weighted (no gain, accuracy lost).

## 2026-10-22 — Correction after a code review (same night)

- An Opus review agent checked `uv_metrics.py`, `refine_correspondence.py` and `transfer_texture.py`
  and found: flips had been counted on triangles with no texture area (so "flips 48 → 29 %" was the
  disappearance of degenerate triangles, not of mirrored ones); the distortion was measured in
  normalised uv instead of pixels; `man.obj` loaded as a triangle soup (41 320 vertices for 6 890
  positions), so no smoothing had acted on the five pairs with the man; field positions could land
  in the texture's blank strip; the rigid-registration prior had been mixed into the anchors in the
  comparison. All fixed; the five man pairs rerun through every step, the field and ARAP textures
  of every pair rebuilt, every number recomputed. Corrected picture, 16 same-category pairs:
  nearest neighbour / refined / smooth field / ARAP within 5 % 82.8 / 88.9 / 89.3 / 96.6 %;
  triangles with no texture patch 65 / 38 / 0 / 1 %; median conformal distortion of the rest
  4.1 / 3.2 / 3.6 / 2.0; mirrored screen area 23 / 19 / 21 / 23 %, i.e. unchanged by any step.
  The numbers in the two entries above that disagree with these are superseded.

## 2026-10-22 — IDArb above 512 px: the normals collapse

- To raise the texture resolution the owner suggested running IDArb at 1024 and 2048 (its
  `max_length` is a constant in the dataset loader). The server ran both on cat, okapi, superman and
  panda with the released weights: no out-of-memory on the 24 GB card (1024: fp32 + xformers,
  17.8 GB peak, 30 s per image; 2048: fp16 + xformers + VAE slicing, 15 GB peak, 230 s per image).
- The outputs are unusable for geometry: the normal map collapses to an almost flat, camera-facing
  field (spread of the unit normals inside the object 0.34 / 0.37 / 0.19 at 512, 0.09 / 0.08 / 0.06
  at 1024, 0.06 / 0.05 / 0.04 at 2048; mean n_z 0.84 → 0.98 → 0.99). Metallic becomes erratic
  (cat 0.59 → 0.99). Inputs that are native 1024 fail the same way, so it is the model outside its
  training size, not upscaling. The albedo stays close to the input (1024 removes less lighting;
  2048 adds a fine generated hatching on fur). Comparison images are kept in
  `outputs/diff3f/99_experiments/idarb_hires/`; the temporary branch was deleted.
- Decision: geometry stays at 512. For a sharper texture, use the 1024 albedo on the 512 relief
  (`transfer_texture.py --albedo_image`), which doubles the texture detail at the cost of slightly
  weaker lighting removal; tiled 512 inference would be the way to raise the normals as well.

## 2026-10-22 — Better semantic features: 3D-native feature models (started)

- The supervisor's reading of the ARAP results: the texture is crisp but the Labubu's face still
  lands on the camel's body, so the semantic features themselves are not good enough; improve the
  features. Diff3F lifts 2D features (Stable Diffusion + DINOv2) from 100 renders; the alternative
  is a model that reads the 3D shape directly.
- Read (PDFs in the owner's Zotero): PartField (Liu et al., CVPR 2025; feed-forward point cloud →
  triplane feature field, 448-d, trained with a part-contrastive loss on Objaverse; code and the
  Objaverse checkpoint released; the paper itself says the features are orientation-sensitive),
  MeshFM (Zhou et al., arXiv 2607.27592; feed-forward, 1536-d, distilled from DINOv2 fields with
  SAM-based anti-aliasing, trained with SO(3) rotation augmentation; code and checkpoint released;
  dense-correspondence AUC on DenseCorr3D 0.52 vs Diff3F 0.46 and PartField 0.35, on TOSCA 0.34 vs
  0.48 / 0.25), HNSR (Du et al., arXiv 2509.17431; features from the TRELLIS 3D generator, global
  → local matching, SHREC'19 accuracy 37.4 vs Diff3F 26.4; no code), TokenMatch (Islam et al., arXiv
  2609.04202; mesh transformer with functional maps trained on BeCoS humans/animals, partial
  matching; code promised), UniMatch (Xiao et al., arXiv 2602.19112; PartField parts + GPT-5 part
  names + a trained functional-map refiner; needs training and an LLM).
- Decision: try the two released feed-forward models as drop-in replacements for the Diff3F
  features, MeshFM first (rotation-robust, built for general use, best correspondence numbers of
  the released ones), PartField second (part-aware; also the natural basis for a part-level step
  in which a target part with no counterpart in the source follows its neighbours, the owner's
  "hump follows the body" idea). The server queries each model's feature field at the same sampled
  points as the Diff3F files, so the feature files keep the same idx / xyz and the whole pipeline
  (anchors, prior, field, ARAP, texture, metrics) runs unchanged. HNSR has no code and TRELLIS is
  heavy; TokenMatch and UniMatch need training data or an LLM: not first.
- `eval_features.py` runs every pair (16 same-category, 10 cross-category, 2 synthetic) for one
  feature folder and writes a CSV, so feature sets are compared with one command each. Risk noted
  before running: both models were trained on closed, whole objects normalised to the unit cube;
  our source is an open, front-only relief.

## 2026-10-22 — MeshFM as a feature replacement: fails on partial shapes

- The server ran MeshFM (checkpoint 502 MB, env cloned from diff3f, 0.1–0.7 s per mesh, 1.4 GB
  GPU) at the Diff3F sample points of all 29 meshes (`scripts/idarb/meshfm_features.py`,
  `results/diff3f/check_meshfm/`). The features are smooth and part-like on the open reliefs, but
  the partial mesh's features are shifted against the full mesh's: synthetic ground truth within
  5 % of the diagonal, nearest neighbour, cow 36 % (Diff3F 86 %), cat 58 % (97 %); mutual nearest
  neighbours 0.5 % vs 1 % on the real cat and cow pairs. Through the whole pipeline (ARAP field,
  `eval_features.py`): cat 10.8 % / cow 7.8 % within 1 % against 72.1 / 58.2 for Diff3F. Fusing
  the two sets (`fuse_features.py`, w 0.75 on Diff3F) gives 63.0 / 55.7, below Diff3F alone.
- Over the 26 real pairs (`results/diff3f/features_meshfm.csv`) the span numbers look no worse
  (same-category ARAP 97.4 % within 5 % vs 96.6 %), but they are the span of a collapsed matching:
  relief used 20 % (nearest neighbour) against 53 % for Diff3F, mutual anchors 38 per pair against
  127, mirrored screen area 34 % against 24 %; cross-category 8 mutual anchors per pair against 47.
  The caveat recorded on 2026-10-21 applies: span without source-used is not a quality measure.
  The fused set (w 0.75) over the same 26 pairs is indistinguishable from Diff3F alone
  (`results/diff3f/features_fused_w0.75.csv`: same-category ARAP 96.5 % within 5 %, relief used
  64 %, 118 anchors; synthetic 59.4 % within 1 % against 65.3 %): MeshFM adds nothing there either.
- Cause, by elimination on the server: not the orientation, the normalisation or the number of
  encoder points (50k–200k; moving the partial into the full mesh's frame with the true transform
  changes nothing). The encoder sees the whole shape, so a front-only relief is a different shape to
  it. Closing the relief with a mirrored back copy (thin shell) gains 2–10 points and is still 30–50
  points behind Diff3F; a thick closure hurts. Diff3F works view by view, so what it sees of the
  front is the same on both meshes: this is what makes it robust to partial input.
- Decision: no feed-forward 3D encoder on the partial relief (PartField uses the same PVCNN +
  triplane encoder and is expected to fail the same way, so its 4–5 GB download is not started).
  Better features have to come from the 2D side: variants of the Diff3F extraction (real albedo
  render of the relief instead of a ControlNet painting, DINOv3 in the DINO branch, MeshFM's
  SAM-based anti-aliasing of the 2D feature maps) are the next server runs. A 3D-native model
  would only become usable with a completed shape (an image-to-3D model such as TRELLIS, whose
  back is a guess); that widens the project's scope and is for the owner to decide.
- Negative side result on the laptop: anchors from neighbourhood-pooled Diff3F features (the
  global-then-local idea of HNSR) are fewer and less precise than the plain mutual nearest
  neighbours (cat anchors within 1 %: 62 → 56 / 48 / 36 % for pooling radii 2 / 5 / 10 % of the
  diagonal; refined accuracy drops accordingly). Not adopted.

## 2026-10-22 — Part-level anchors after SemAnCorr: tested on the Diff3F features

- SemAnCorr (Dong et al., RA-L 2026) matches parts before points: k-means parts on lifted 2D
  features, part similarity after removing each object's mean feature, a joint rigid alignment
  with a soft part assignment, bilateral-margin confidence with each part used once, anchors only
  inside confident part pairs. `scripts/idarb/part_anchors.py` implements this on our feature
  files (the paper uses SigLIP2 features and a functional map afterwards; we feed the anchors to
  the existing prior / field / ARAP chain).
- Same-category synthetic pairs: the part pairs are right (clear diagonal in the relative
  similarity, 0.8–0.9), but anchors taken by nearest neighbour inside the matched regions are
  imprecise (cat 47 %, cow 67 % within 5 % of the diagonal) against 98–100 % for the mutual
  nearest neighbours, so as a source of anchors the part step is worse here (refined accuracy
  cat 19 % vs 73 % within 1 %). As a filter it keeps the mutual anchors' precision (99–100 %) and
  discards a third to three quarters of them.
- Cross-category Labubu → camel: after removing the object means the Diff3F part similarities
  are all below 0.35 and the selected pairs have confidence 0.001: the features do not tell the
  parts of a toy from the parts of a camel. This is the measurement behind the supervisor's
  judgement, and it points at the feature itself, not the matching. SemAnCorr's language-aligned
  SigLIP2 features are the next server variant; `part_anchors.py --out_dir` writes the part
  similarity matrix and part-coloured point clouds, so the same measurement is repeated on every
  new feature set.

## 2026-10-23 — Where the mirrored texture comes from, and three steps against it (night run)

- Measurement first. On the synthetic pairs (the partial cut from the target itself) the mirrored
  screen area after refinement is about 1 %, so the 20–35 % on the real pairs is not a property
  of the pipeline but of real pairs: different individuals, different poses, and features that
  cannot tell the left edge of a limb from its right edge. By facing angle (ARAP, 16 same-category
  pairs): faces almost edge-on to the texture camera mirror 47 % of their area, faces turned
  towards it 6.5 %; the worst pairs (posed human, the chairs, the stone bird, dragon → lizard)
  keep 25–40 % even on front faces. Anchor consistency: on real pairs 14–44 % of the mutual
  anchors disagree with one similarity transform (synthetic: 0–1 %).
- Tried, no effect: a penalty in the candidate choice for mirroring the vertex's camera-facing fan
  (`--w_flip`; the neighbours are mirrored together, so no single change helps).
- Three steps that do help, each a refinement option, measured on the five most mirrored pairs
  (superman_posed, chair10_roundchair, bird_bird, cat_lion, dragon_lizard; ARAP; triangles
  within 2 % / 5 %, relief used, median distortion, mirrored area):
  - `--anchor_filter rigid:0.1`: anchors farther than 10 % of the diagonal from a RANSAC
    similarity transform of all anchors are dropped. All 26 pairs: same-category 64.3 → 66.2 %
    within 2 %, mirrored 23.7 → 21.3 %, distortion 2.18 → 2.12; cross 44.3 → 47.9 %. Small, consistent.
  - `--w_view w`: every vertex's ARAP rotation also has to carry the texture camera's direction
    onto the relief's viewing axis (a virtual edge pair weighted by w times the one-ring), so a
    limb cannot be turned about its own axis to put its hidden side on the relief's visible side.
    w = 1: mirrored area 32.4 → 21.9 %, within 2 % 76.2 → 79.5 %, but distortion 2.53 → 3.75 and
    relief used 59 → 55 %. (A first version let the pair inflate the ARAP scale; the scale is now
    fitted from the edges alone.) Visibly: the posed human's legs lose the cape bleed, the chair's
    backrest becomes one continuous pattern.
  - `--coarse_to_fine 2 --coarse_factor 4`: the field and the choice of matches alternate, the
    smoothness falling from 16× to the final weight, every level predicting each vertex's source
    position from the current field (the non-rigid ICP schedule of Jung et al. 2025). Within 2 %
    76.2 → 79.9 %, distortion 2.53 → 2.21, mirrored area about unchanged (chair 37 → 24 %).
  - Combined (`--coarse_to_fine 2 --coarse_factor 4 --w_view 1 --anchor_filter rigid:0.1`):
    within 2 % 76.2 → 85.5 %, within 5 % 97.4 → 99.3 %, mirrored 32.4 → 17.9 %, distortion
    2.53 → 3.36, relief used 59 → 53 %; with `--w_view 0.3` instead: 81.1 % / 98.8 % / 17.8 % /
    3.03 / 55 %. Synthetic ground truth: the view pair costs about 2 points within 1 % (cat 72.1 →
    69.9), coarse-to-fine is neutral, the rigid filter gains (cat 72.1 → 77.6 at 5 % threshold).
- Decision, after runs over all 28 pairs with 2, 3 and 4 coarse-to-fine levels
  (`results/diff3f/features_diff3f_c2f{2,3,4}.csv`, with `--w_view 0.3 --anchor_filter
  rigid:0.1`): three levels is the recommended setting
  (`outputs/diff3f/09_texture_view_consistent/`). ARAP means, 07 → 09: same-category within 2 %
  64.3 → 80.1 %, within 5 % 96.6 → 99.2 %, mirrored area 23.7 → 9.1 %, distortion 2.18 → 2.23,
  relief used 65 → 56 %; cross-category 44.3 → 72.1 %, 89.3 → 98.2 %, 39.0 → 14.1 %, 2.31 → 2.45,
  49 → 37 %; synthetic within 1 % 65.3 → 60.4 % (within 5 % 99.5 both). Each extra level smooths
  more; four levels reaches 84.6 / 81.5 % within 2 % and 6.8 / 8.8 % mirrored but uses only 52 /
  31 % of the relief and drops the synthetic accuracy to 58.1 %, and on Labubu → camel the face
  vanishes instead of landing anywhere, so the map is collapsing: not adopted. The confidence gate
  on top adds little (`features_diff3f_c2f3_gate.csv`); `--w_view 1` costs distortion and relief
  used (`features_diff3f_view1_c2f.csv`). Visible gains beyond the numbers: the Labubu's, the
  cat's and the panda's faces now land on the man's face, the dragon's on the lizard's head.

## 2026-10-22 — Better semantic features: MeshFM and Diff3F variants (server results)

- **MeshFM** (threedle/MeshFM `e6577b1`, 502 MB checkpoint, no license file): queried at the idx /
  xyz of all 29 Diff3F files (`scripts/idarb/meshfm_features.py`), 0.1 to 0.7 s per mesh, 1.4 GB
  GPU. On the synthetic partials (share matched within 5 % of the full mesh's diagonal) it reaches
  cow 0.360 and cat 0.584 against Diff3F's 0.860 and 0.974. Not caused by orientation,
  normalisation or point count (50k / 200k points, partial moved into the full mesh's frame, the
  full mesh's normalisation: cow 0.34 to 0.37, cat 0.43 to 0.60). The encoder sees the whole shape,
  so a front-only partial gets shifted features. Closing the relief with a mirrored copy
  (`--close_mirror`) gives mid-depth 0.380 / 0.688 and z_min 0.183 / 0.558: not enough. PartField
  (same encoder family, about 4 to 5 GB of downloads) was not run.
- **Diff3F variants** (`scripts/idarb/diff3f_features.py`, new options; without them the script
  still calls the unchanged upstream function), on cow_synth / cat_synth / cow / cat. Same seed:
  a rerun reproduces Diff3F exactly (cow 0.8598 both, cat 0.9741 vs 0.9738), so run-to-run noise
  is negligible. Within 5 % (cow, cat) and mutual nearest neighbours on the real pairs (cat, cow):
  - Diff3F: 0.860, 0.974; 0.0089, 0.0096.
  - C, SAM cleaning (`--sam_tau 1`, MeshFM-style: per SAM ViT-B mask, pixels farther than 1 from
    the mask's median feature take the median): 0.853, 0.956; 0.0071, 0.0101.
  - E, SigLIP2 (`--siglip2`, timm `vit_base_patch16_siglip_512.v2_webli`, mean of the last four
    blocks on the same painted views, 768-d): 0.645, 0.736; 0.0060, 0.0069.
  - Diff3F + SigLIP2 concatenated (each unit length, divided by √2): 0.846, 0.964; 0.0064, 0.0090.
  - A, the relief rendered with its own IDArb albedo, one UNet pass at t = 100 with the positive
    prompt and no ControlNet, plus DINOv2 on the same image (`--appearance vertex_colors`); the
    full meshes keep the painted features: no synthetic number (the synthetic partials have no
    colours); real pairs 0.0052, 0.0089. With SigLIP2 on that image: 0.0031, 0.0032.
  - B (DINOv3) was skipped: gated model, no Hugging Face token on the server. D (other SD layers or
    steps) was skipped: the layer (`up_blocks[1]`) and the step averaging are local variables
    inside `third_party/Diff3F`.
- Runtime and peak GPU (nvidia-smi): Diff3F alone 247 to 632 s, 10 to 13 GB; the combined pass with
  SAM and SigLIP2 resident 623 to 1097 s, 21.1 to 22.4 GB (SAM's automatic masks roughly double the
  time); A 286 to 459 s, 16.7 to 16.9 GB.
- Upstream detail: Diff3F's diffusion feature is the first half of the classifier-free-guidance
  batch, i.e. the negative-prompt branch.
- The per-pixel SigLIP2 features are the outputs of the vision tower's last blocks, before the
  final norm and the attention-pool head. They are not in SigLIP's joint image-text space, so a dot
  product with a text embedding is not meaningful without pooling them through that head.
- Decision: keep Diff3F. No variant beats it at point level. SigLIP2 is computed for all 29 point
  sets (`results/diff3f/features_siglip2/`, on the temporary branch only) so that the cross-category
  part-level similarity can be measured on the laptop. The Diff3F part of those runs equals the
  original Diff3F files exactly (mean cosine 1.0000 on all 29), so the concatenation is simply the
  existing Diff3F features plus SigLIP2 and is formed on the laptop instead of stored. SAM and A
  are not pursued. The text-embedding route waits for that measurement.

## 2026-10-23 — SigLIP2 at the part level, and the part-naming test

- The 29 SigLIP2 feature files (server, `results/diff3f/features_siglip2/` on the temporary branch,
  `outputs/diff3f/features_siglip2/` locally) were put through `part_anchors.py` next to the
  Diff3F files. Relative part similarity (object mean removed), best value and best pair
  confidence: Labubu → camel 0.43 / 0.03 (Diff3F 0.33 / 0.05), Labubu → lion 0.31 / 0.01
  (0.35 / 0.02), cat → man 0.35 / 0.03 (0.32 / 0.04), superman → camel 0.54 / 0.04 (0.39 / 0.03);
  same-category cat → lion 0.55 / 0.06 (0.61 / 0.07), synthetic cat 0.91 / 0.55 (0.92 / 0.32). So
  the lifted per-pixel SigLIP2 features separate parts of the same object about as well as Diff3F
  and cross-category parts no better. The server's note explains why: the per-pixel features are
  the transformer's token outputs before the attention-pool head, which is where SigLIP2's
  image-text alignment is formed, so they are not in the joint space.
- Test in progress on the server: each part's patch tokens pooled through the attention-pool head
  per rendered view (masked pooling), averaged over views, compared with text embeddings of a part
  vocabulary ("a photo of the hump of a camel", ...). Part labels of eight meshes are on the
  temporary branch `part-labels-tmp` (`results/diff3f/part_labels_tmp/`). If "face" and "hump" come
  apart there, the own method's first version (parts named by image-text similarity, matched by
  name, unmatched parts follow the body through `--parts`) is viable; if not, the language route
  is dropped.
- Full pipeline (09 setting, `eval_features.py`) on the SigLIP2 files and on the Diff3F+SigLIP2
  concatenation (`fuse_features.py --w 0.5`, built locally; `results/diff3f/features_siglip2.csv`,
  `features_diff3f_siglip2.csv`): SigLIP2 alone synthetic 34.7 % within 1 % (Diff3F 60.4 %),
  mutual anchors 30 / 5 per same / cross pair (99 / 27), same-category within 2 % 77.0 % (80.1 %);
  the concatenation 62.4 % synthetic, 79.9 / 70.7 % within 2 %, 79 / 16 anchors: equal to Diff3F
  within noise. Neither is adopted; the 1.5 GB of concatenated files are not stored.

## 2026-10-23 — Part naming with SigLIP2 (server) and name-based part matching: partial

- Server (`scripts/idarb/siglip2_part_naming.py`, `results/diff3f/part_naming/`): every part of
  eight meshes pooled through SigLIP2's attention head over Diff3F's painted views, compared with
  a part vocabulary. Whole objects are all named right (8 of 8). Parts: the camel's parts come out
  as hump, front leg, leg, paw, foot, hind leg; the cow's as head, snout, tail, hoof; the Labubu's
  as ear, hood, face, face, shirt, shirt. But the scores are small (0.02–0.11) with margins
  between the best names often below 0.01, and the raw part-vector cosines across objects are
  nearly flat (0.61–0.79); after mean removal the Labubu's face still goes to the camel's
  hump-and-head part.
- Two limits come from the parts themselves: they are k-means clusters of the Diff3F features,
  so the camel's hump and head sit in one part (the feature confusion the whole line is about),
  and six parts on the Labubu leave its legs and shoes inside "shirt". And the names of a dressed
  toy (hood, shirt) and of an animal (leg, paw) do not overlap, so name agreement alone
  (`name_parts.py`, same word or same group over the best three names, plus a little of the
  vector cosine) yields no confident pair on Labubu → camel or Labubu → lion, one on cat → man
  (face ← snout), three on cat → lion and cow → cow (head, snout, tail). With no pair, `--parts`
  would gate every target part, so the step cannot drive the correspondence as it stands.
- Decision: keep the machinery (naming works at the object level and for animal parts) and
  change the two inputs before judging the method: parts from a part model rather than from the
  Diff3F clusters (PartField on the full mesh, SAM masks lifted to the relief), and name
  similarity from text embeddings rather than a hand-written group table. Not pursued tonight;
  the owner reviews the dataset phase A first.

## 2026-10-23 — Supervisor: fine-scale matching ("eye to eye, nose to nose"); keypoint accuracy

- The supervisor's reading of the Labubu → camel result (08:40): what is needed is fine-scale
  matching, eye to eye and nose to nose, and the face features look wrong. New measure:
  `scripts/idarb/eval_keypoints.py` maps every annotated target landmark through a correspondence
  and reports its distance to the annotated source landmark as a fraction of the relief's
  diagonal (mean, median, PCK within 2 / 5 / 10 %). On the three pairs that have landmarks
  (`results/diff3f/landmarks/`, 10–11 each; `results/diff3f/keypoints/landmark_accuracy.csv`):
  cat → lion mean error nearest neighbour 4.2 %, ARAP (07) 2.9 %, the 09 setting 3.5 %;
  Labubu → camel 8.8 / 8.3 / 9.3 %; Labubu → lion 7.8 / 9.0 / 7.9 %. So at the keypoint level the
  09 setting is not better than ARAP, and both are 3–9 % of the diagonal away from the landmarks:
  the smooth-map improvements of the night did not move the fine-scale accuracy, which agrees with
  the review's warning that span and mirror numbers measure smoothness.
- Why the face features are coarse: Diff3F renders every view at 512 px and DINOv2 works on
  14-px patches, so a camel's face in a full-body render covers 3 to 4 patches; the lifted features
  are blurred at the scale of an eye. Planned test on the server (session to be reopened by the
  owner): zoomed views (`--zoom 2`: Z×Z tiles per camera at Z times the focal length) and the
  synthetic 1 % accuracy as the fine-scale measure.
- The landmark set is being extended to the other pairs (eyes, nose, mouth, ear tips, paws) so
  that keypoint accuracy becomes the primary number next to the synthetic ground truth.

## 2026-10-23 — Keypoint benchmark on 15 pairs: the refinement loses fine accuracy; texture cameras were wrong on 6 pairs

- Keypoints: two annotation agents picked 5–7 named keypoints per pair (eye, nose, chin or
  mouth, ear tip, front and hind paw or hoof tip, tail; hands, feet and head top on the humans;
  seat corners and leg bottoms on the chair pair) on 12 more pairs with `pick_landmarks.py`,
  each verified on its contact sheet (`results/diff3f/keypoints/<pair>.json`, sheets in
  `results/diff3f/keypoints/check/`); with the three older landmark sets that is 15 pairs and
  93 keypoints. `keypoint_table.py` evaluates every result folder on all of them.
- Result (`results/diff3f/keypoints/keypoint_accuracy.csv`), 13 same-category pairs, mean error
  as a fraction of the relief diagonal and PCK within 2 / 5 / 10 %: nearest neighbour 5.3 %,
  54 / 76 / 89 %; ARAP (07) 6.0 %, 39 / 68 / 82 %; the 09 setting 5.6 %, 40 / 63 / 82 %. So the
  whole smoothing chain (prior, field, ARAP, coarse-to-fine) makes the map smoother and
  mirror-free but moves the keypoints away from their right place: at 2 % of the diagonal it is
  14 points worse than the plain nearest neighbour. The night's improvements were measured on
  smoothness proxies; on the supervisor's criterion (eye to eye, nose to nose) they are a loss.
- Texture cameras: comparing each pair's stored camera with the camera implied by its own
  mutual anchors (RANSAC similarity transform, relief axis +z pulled back to the target) shows
  the stored one looking at the target from behind on labubu_man (179°, a failed registration:
  the Labubu's face had been painted on the back of the man's head and reported as "on his
  face"), and on the cross pairs that borrowed another pair's camera: superman_camel 174°,
  bird_cat 160°, labubu_lion 102°, labubu_camel 96°; cat_man 62°, labubu_cow 75° and
  chairgen_armchair 47° off. `refine_correspondence.py --view_from_anchors` now derives the
  camera from the anchors and stores it in the npz; `eval_features.py --view anchors` uses it
  for texturing and the metrics. All 28 pairs are being rerun with it
  (`outputs/diff3f/11_anchor_view/`).
- Decision: keypoint accuracy (with the synthetic ground truth) is the primary measure from now
  on; the span, distortion and mirror numbers are diagnostics. The next steps aim at the
  keypoints: finer features (zoomed views, running on the server) and a refinement that keeps
  the nearest-neighbour accuracy where it is right instead of smoothing it away.
- Where the keypoint accuracy is lost (89 keypoints, 13 same-category pairs, PCK within 2 %):
  nearest neighbour 54 %; the discrete choice with the rigid-filtered anchors, prior window and
  two smoothing passes 51 % (k 16 candidates: 56 %; prior-position candidates `--k_prior 16`:
  45 %; RBF width 0.1: 47 %); then the coarse-to-fine rematching 39 % and the field/ARAP 38–40 %.
  So the prior stage is nearly neutral once the anchors are filtered, and the losses are the
  coarse-to-fine levels and the smoothing. `--keep_confident q` keeps the nearest neighbour of
  the vertices whose similarity is above the quantile q and that pass a soft mutual test (on
  the keypoints these matches reach 65 % within 2 %); on the synthetic pairs it costs
  (cat 67.7 → 59.3 % within 1 % with q 0.7), because there the prior improves every match.
  Variants without coarse-to-fine and with kept matches are being run on the server against the
  keypoints.
- Six refinement variants without coarse-to-fine, run on the server against the keypoints
  (`results/diff3f/kp_variants/`, branch kp-variants-tmp; 13 same-category pairs, PCK within
  2 %): anchor filter + view term 37 %; + kept confident matches (q 0.7) 35 %; k 16 candidates
  34 %; k 16 + kept 33 %; kept matches without ARAP 44 %; kept q 0.5 with field weight 10 41 %;
  nearest neighbour 54 %. The variant that keeps the most keypoint accuracy (no ARAP) is the worst
  on the texture-map measures (triangles stretched over tenfold 28 % against 12 %, mirrored area
  18 % against 12 %), so the ARAP smoothing is what moves keypoints off their place, and with the
  present features no setting of this chain keeps both. Conclusion: the lever is the features
  (zoomed views running on the server) and, for the map, a formulation that keeps confident
  matches as hard constraints rather than weights.
- Smoothing-strength sweep on the Windows laptop (`results/diff3f/sweep_smooth/`, branch
  sweep-win-tmp merged; 8 settings × 28 pairs, keypoints on the 13 same-category pairs, texture
  measures on the 16): weakening the smoothing raises keypoint PCK@2 % monotonically from 37 %
  (field 0.1, ARAP weight 3) to 42 % (ARAP 1), 43 % (ARAP 0.3), 45–46 % (no ARAP), while the
  texture quality falls just as monotonically: triangles within 5 % 97.5 → 89.5 %, median
  conformal distortion 2.37 → 4.30, stretched over tenfold 11.6 → 27.0 %. Kept confident matches
  with field weight 10 or 30 are worse on both sides and cost the synthetic pairs 9 points. So with
  the present features the chain only trades fine accuracy against texture coherence; nothing in
  it recovers the nearest neighbour's 54 %. This curve is the argument for changing the features
  (zoomed views) or the input (a generated full mesh), not the smoothing.

## 2026-10-23 — Zoomed views raise the fine-scale accuracy (server, first batch)

- `diff3f_features.py --zoom 2` (server): for every camera, besides the full view, up to 2×2
  tiles rendered with twice the focal length through the camera intrinsics (each tile equals a
  512-px crop of a 1024-px render, same perspective), painted, lifted and averaged like the full
  views; 300–370 views per mesh, 13–38 min per mesh (relief tiles are mostly foreground), 8.5 GB
  GPU. Features in `results/diff3f/features_zoom2/` on the temporary branch.
- Synthetic ground truth, nearest neighbour within 1 % (check_features.py): cat 38.8 → 49.5 %,
  cow 30.4 → 37.2 %; through the pipeline (eval_features, anchor filter + view term) cat
  70.6 → 75.8 %, cow 57.2 → 55.2 %. Keypoints on the four pairs in the batch (cat, cat → lion,
  Labubu → camel, Labubu → lion; 38 keypoints): nearest neighbour PCK@2 % 24 → 34 %, after the
  pipeline 22 → 34 %; the gain is consistent on the cat pairs and Labubu → lion, none on
  Labubu → camel. A seed-1 rerun of the plain extraction (painting is random) and zoom 3 are
  running as controls; zoom 2 for the remaining 15 keypoint meshes follows overnight.
- First evidence that the feature resolution, not the matching, is what limits "eye to eye".
- Zoom 3 and a control (server, `features_zoom3/`, `features_zoom1_seed1/`): the control is not
  a noise estimate, because Diff3F fixes the painting seed (torch.manual_seed(60) per view in its
  diffusion.py), so `--seed` only changes the vertex subsample and the features are the same;
  through the pipeline the numbers still move by up to 1.5 points, which is the pipeline's own
  spread. Synthetic within 1 %, nearest neighbour (check_features) / after the pipeline: cat zoom 1
  38.8 / 70.6 %, zoom 2 49.5 / 75.8 %, zoom 3 53.9 / 72.4 %; cow 30.4 / 57.2, 37.2 / 55.2,
  38.4 / 53.3 %. Zoom 3 adds a little on the raw measure, nothing through the pipeline, at twice
  the cost (22–28 min per mesh): zoom 2 is the setting kept. A true noise estimate needs a
  different paint seed per run (a patch of that line); deferred.

## 2026-10-24 — Zoom 2 on the whole keypoint benchmark: +9 points at the fine scale

- Zoom-2 features now exist for the 23 meshes of the 15 annotated pairs (server,
  `results/diff3f/features_zoom2/` on the temporary branch; reliefs 29–48 min, full meshes 16–20
  min each). Keypoints, 13 same-category pairs, PCK within 2 / 5 / 10 % and mean error:
  nearest neighbour zoom 1 54 / 76 / 89 %, 5.3 % → zoom 2 63 / 77 / 89 %, 4.8 %; after the
  pipeline (anchor filter + view term + ARAP) 37 / 66 / 88 % → 43 / 69 / 87 %. No pair gets worse
  at 2 % with the nearest neighbour, 6 of 13 gain; the two cross pairs 5 → 10 %. The texture
  measures are unchanged (same-category within 2 % 71.1 → 71.6, within 5 % 98.3 → 98.0, mirrored
  area 10.6 → 10.7 %), the synthetic cat gains and the cow is flat. Numbers in
  `results/diff3f/kp_variants/` (server runs, branch kp-variants-tmp merged).
- Decision: zoom 2 is the standard feature extraction from now on; the six meshes still at zoom 1
  only (the chair pairs' meshes and the two depth-scaled synthetic cows) are queued. The pipeline
  still loses 20 points of PCK@2 against the nearest neighbour; that is the next thing to change.

## 2026-10-24 — Generate-then-match, step 1: full meshes from the photos and their alignment

- Supervisor (23 Oct, evening): try single-image-to-3D generation (TRELLIS.2, Hunyuan3D) and
  "generate the full 3D first, then match". Server: Hunyuan3D-2.1 (shape 56 s + paint 52–71 s,
  14 GB; keeps the photo's frame approximately) and TRELLIS.2 at its 512 setting (22–35 s, 12 GB;
  canonical pose, photo camera fitted by silhouette IoU) on the 14 photos that have a relief
  (`results/diff3f/gen3d/`, meshes on the temporary branch gen3d-tmp and in
  `outputs/diff3f/gen3d/`). All 28 meshes are plausible with sensible backs; known flaws: the
  dragon's flame as a flat card (Hunyuan3D), colour shifts on the chairs (TRELLIS.2). Hunyuan3D's
  licence forbids training other models on its outputs; TRELLIS.2 is MIT.
- `align_generated.py` registers the relief (the photo's frame) to the generated mesh: camera
  rotation from the fitted view, similarity in x, y from the silhouettes, depth offset from the
  generated front surface under each relief pixel, ICP with the relief's depth factor. Over the
  28 meshes (`results/diff3f/gen3d/alignment.csv`): relief vertices within 2 % of the diagonal
  of the generated front surface 80–98 % (Hunyuan3D fire_dragon and superman 61–62 %), ICP rms
  1–6 % of the diagonal (TRELLIS.2 1–2 %), rotations mostly below 10° (okapi 40–57°: the fitted
  camera was off and ICP recovered it). The relief's keypoints land on the generated mesh with
  snap distances below 0.11 of the diagonal (median about 0.05), and the photo's albedo is
  projected onto the visible side (58–100 % of the front faces). So the generated mesh can take
  the relief's place as the source with the keypoint benchmark intact; its zoom-2 Diff3F features
  are being computed on the server (28 meshes, 7–9 h) for the full-to-full matching.

## 2026-10-24 — Generate-then-match, step 3: Hunyuan3D sources lose the fine scale

- Setting: source = the Hunyuan3D-2.1 mesh generated from the photo, aligned to the relief
  (`align_generated.py`), with the photo's albedo baked into its atlas (`bake_albedo.py`) and
  the relief's keypoints carried onto it (`gen_keypoints.py`); zoom-2 Diff3F features of the
  welded generated mesh computed on the server (commit c8d58a1: 100 views all around, 320 tiles,
  15 000 points; the relief's features, commit 61b4c84, used `--front_only`: 314 tiles from the
  front hemisphere). Target and keypoints as in the keypoint benchmark. Driver
  `eval_generated.py` (recommended refinement setting, anchor-derived camera, `--source_glb`).
- Result (`results/diff3f/generated_hunyuan3d.csv`), PCK@2 = keypoints within 2 % of the source
  diagonal, 13 same-category pairs: nearest neighbour 17 %, pipeline 16 %; with the relief as
  source (zoom 2, `results/diff3f/kp_variants/keypoint_accuracy_zoom2_all.csv`) 63 % / 43 %.
  The two cross-category pairs with keypoints are near zero with either source. Per pair the
  generated source is worse on 12 of 13 (cat is equal at 71 % after refinement). Texture-space
  measures (mirrored area, conformal distortion) are not comparable for an atlas texture: the
  GLB's uv charts have their own orientation and scale, so `uv_metrics.py` reports 30–70 %
  "mirrored" area that is chart orientation, not a mirrored mapping. Triangle span within 2 %
  is 0.62 on the same-category pairs.
- Why (`keypoint_rank.py`, `results/diff3f/keypoint_rank_{relief,hunyuan3d}.csv`): for every
  target keypoint, the rank of the true source location among all source points by feature
  similarity (0 = the nearest neighbour is right) and whether the nearest neighbour faces the
  photo's camera. Relief: median rank 4–51 on 11 of 13 same-category pairs (labubu_man 294,
  bird 105), nearest neighbours on the photo's side by construction. Generated mesh: median rank
  57–1969, and 0–57 % of the nearest neighbours land on the side the photo never saw (the
  other ear or hoof, the back). The similarity at the true location is as high or higher for
  the generated source on the animals (cat 0.83 against 0.77) but lower on the faces
  (superman_man 0.63 against 0.78): on the animals the generated features are less distinctive
  (many points score as high), on the faces the generator's smooth geometry does not carry the
  photo's facial structure that the relief's normals do. The alignment is not the cause: the
  carried keypoints sit on the right parts (`outputs/diff3f/13_generated/*/check.png`).
- Hybrid source (`hybrid_features.py`: the relief's features on the camera-visible points of
  the generated mesh through the alignment, 33–55 % of the points; the generated features
  elsewhere; `results/diff3f/generated_hunyuan3d_hybrid.csv`): nearest neighbour 18 %,
  pipeline 20 %. The generated-style features score systematically higher similarity to the
  target than the relief's (best similarity to a generated-feature point 0.77–0.90, to a
  relief-feature point 0.65–0.80, on 12 of 15 pairs), so the unreplaced side takes the match
  (nearest neighbour on a replaced point 0–71 % per pair). Restricted to the replaced points
  the nearest neighbour reaches 14–57 % per pair, below the relief's own 63 %, because the
  carried features inherit the alignment error (ICP rms 1–2 % of the diagonal).
- Standing: the generated full mesh does not replace the relief as the matching source at the
  fine scale with Diff3F features; it remains the candidate for the geometry and the unseen
  side. Open: the same evaluation with the TRELLIS.2 sources (features in progress on the
  server), and the view-sampling confound (generated features with `--front_only`, queued
  on the server), which separates fewer front views from the smoother geometry.
- Commands (repository root, `mesh` environment, this commit):
  `python scripts/idarb/eval_generated.py --model hunyuan3d --gen_features outputs/diff3f/features_zoom2_gen --feature_dir outputs/diff3f/features_zoom2 --out_dir outputs/diff3f/14_generated_match/hunyuan3d --csv results/diff3f/generated_hunyuan3d.csv --threads 3`;
  `python scripts/idarb/hybrid_features.py --model hunyuan3d --out outputs/diff3f/features_zoom2_gen_hybrid`
  followed by the same driver with `--gen_features outputs/diff3f/features_zoom2_gen_hybrid`
  and `--csv results/diff3f/generated_hunyuan3d_hybrid.csv`;
  `python scripts/idarb/keypoint_rank.py --source relief --csv results/diff3f/keypoint_rank_relief.csv`
  and `--source hunyuan3d --csv results/diff3f/keypoint_rank_hunyuan3d.csv`.

## 2026-10-24 — Owner: try every open item, cheapest first; two-sided texture; held-out split

- The owner's instruction (evening): try all the open items, ordered by time cost from small to
  large: (1) the two-sided texture (photo through the relief on the seen side, the generated
  mesh's texture on the rest), (2) a held-out split of the keypoint benchmark, (3) a Hunyuan3D-
  Paint baseline (target white mesh + photo), (4) DINOv3 inside Diff3F, (5) GeoAware-SC
  features inside Diff3F, (6) a DenseMatcher baseline, (7) dataset phase B features, (8) a
  locally injective image-plane map with a chirality filter, (9) method 1 redone (SAM2 +
  PartField + VLM naming), (10) method 2. Standing rule from the same evening: no CPU runs on
  the Mac; every run goes to the Windows laptop, GPU work to the server.
- (1) `merge_sides.py` + `eval_generated.py --merge`: the faces the texture camera sees take
  the relief's image through the relief correspondence (keypoint-oriented setting, anchor
  camera), every other face takes the generated mesh's atlas (photo albedo baked into the seen
  side) through the generated-mesh correspondence; both images packed into one texture, one
  OBJ. Cat pair (Mac, before the rule): 4 401 front faces from the photo, 10 009 from the
  generated mesh, no grey face left; the far flank continues the stripes plausibly, the seam
  along the silhouette shows colour and pattern jumps on the face and chest. The 15 pairs run
  on the Windows laptop (`results/diff3f/generated_hunyuan3d_merge.csv`, previews under
  `results/diff3f/generated_merge/`, pending).
- (2) `keypoint_split.py`: a seeded split of the 15 annotated pairs, stored in
  `results/diff3f/keypoints/split.json`: tuning = cat, cat_lion, dragon_lizard, labubu_lion,
  okapi1_camel, okapi_camel, superman_man; held out = bird_bird, cat5_cat, chair1_armchair,
  cow, labubu_camel, labubu_man, panda_panda, superman_posed. From now on setting decisions use
  the tuning half; the held-out half is reported. Per half (`split_accuracy.csv`; mean PCK@2
  over pairs with a 95 % bootstrap interval): zoom 2 nearest neighbour tuning 61 % [46, 80],
  held out 51 % [30, 73]; zoom 2 pipeline 45 % [30, 60] and 32 % [13, 55]; zoom 1 nearest
  neighbour 49 % and 47 %, three-level setting 41 % and 33 %. The held-out half is the harder
  one (bird, chair, cow, Labubu → camel). With 7–8 pairs per half the intervals are about ±15
  to 20 points wide, so differences of a few points between settings (the 37–43 % range of the
  refinement variants) are within noise, while nearest neighbour against the pipeline and the
  relief against the generated source (63 % against 17 %) are not.
- (3) and (4) are specified for the server (Hunyuan3D-Paint on the 15 pairs; `--dino dinov3`
  in `diff3f_features.py`, tested on five meshes); the server session was unreachable when they
  were sent, so they wait for the owner to reopen it.
