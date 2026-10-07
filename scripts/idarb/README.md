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
