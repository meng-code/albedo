# Third-party notices

Add each future dependency with its source URL, version or commit, license, intended use, and any redistribution conditions before it is incorporated into the project.

## IDArb (source code)

- **Path:** `third_party/IDArb/`
- **Source:** https://github.com/Lizb6626/IDArb, commit `df26b90` ("update bibtex")
- **License:** MIT, Copyright (c) 2024 Zhibing Li. The full license text is kept at `third_party/IDArb/LICENSE`.
- **Use:** Intrinsic decomposition of images into albedo, normal, metallic and roughness (inference), plus a small training smoke test.
- **Modifications:** The first commit that adds this directory is an unmodified copy of upstream. Later commits that touch `third_party/IDArb/` are local changes; see `git log -- third_party/IDArb`.
- **Paper:** Li et al., "IDArb: Intrinsic Decomposition for Arbitrary Number of Input Views and Illuminations", arXiv:2412.12083.

## IDArb pretrained weights (not included)

- **Source:** https://huggingface.co/lizb6626/IDArb, revision `1d00fda59f`
- **License:** Not declared on the model card as of 2026-09-25.
- **Use:** Default weights loaded by `third_party/IDArb/main.py` (downloaded at run time, not stored in this repository).

## Stable Diffusion 2.1 base, zero-SNR (not included)

- **Source:** https://huggingface.co/ByteDance/sd2.1-base-zsnr-laionaes5, revision `77c78cc958`
- **License:** CreativeML Open RAIL++-M (`openrail++`)
- **Use:** Initial weights for the IDArb training smoke test (`third_party/IDArb/configs/my_train_smoke.yaml`). Downloaded at run time.

## Arb-Objaverse dataset (not included)

- **Source:** https://huggingface.co/datasets/lizb6626/Arb-Objaverse, revision `0840f781c7`
- **License:** MIT (as declared on the dataset card)
- **Use:** Three objects from `data/000-000.zip`, listed in `third_party/IDArb/datalist/my_train.json`, used for the training smoke test. Fetched with `scripts/idarb/fetch_arbobjaverse_subset.py`; stored under the git-ignored `third_party/IDArb/data/`.

## Depth Anything V2 Small (not included)

- **Source:** https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf, revision `5426e4f0f3`
- **License:** Apache-2.0. (The Base and Large variants are CC-BY-NC-4.0, non-commercial only; they can be selected with `--model` but are not the default.)
- **Paper:** Yang et al., "Depth Anything V2", arXiv:2406.09414.
- **Use:** Relative depth maps used as a prior for normal-to-mesh reconstruction, produced by `scripts/idarb/estimate_depth.py`. The model is downloaded at run time; the resulting depth maps for the samples are stored in `results/idarb_single/depth/`.

## Diff3F (source code)

- **Path:** `third_party/Diff3F/`
- **Source:** https://github.com/niladridutt/Diffusion-3D-Features, commit `5bda687` ("Update README.md",
  2025-02-11)
- **License:** MIT, Copyright (c) 2024 Niladri Shekhar Dutt. The full license text is kept at
  `third_party/Diff3F/LICENSE`. The directory also contains a copy of pyFM (functional maps, by Robin
  Magnet, https://github.com/RobinMagnet/pyFM, MIT) as shipped upstream, and sample meshes in `meshes/`
  for which upstream states "provided as examples from various sources and we do not claim any
  copyright". We use `meshes/cow.obj` and `meshes/cat.off` as full-mesh targets in registration
  experiments; their original sources are not stated upstream, so review them before any publication.
- **Use:** Per-vertex semantic features for meshes (`scripts/idarb/diff3f_features.py`), which
  `scripts/idarb/register_partial.py` uses to register our partial meshes to full meshes.
- **Modifications:** The first commit that adds this directory is an unmodified copy of upstream. Our
  script replaces the Stable Diffusion repository id and the camera set from outside the directory; later
  commits that touch `third_party/Diff3F/` are local changes; see `git log -- third_party/Diff3F`.
- **Paper:** Dutt, Muralikrishnan and Mitra, "Diffusion 3D Features (Diff3F): Decorating Untextured
  Shapes with Distilled Semantic Features", CVPR 2024, arXiv:2311.17024.

## Stable Diffusion v1.5 (not included)

- **Source:** https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5 (community mirror of
  `runwayml/stable-diffusion-v1-5`, which was removed from Hugging Face in 2024; same weights)
- **License:** CreativeML Open RAIL-M
- **Use:** The image generator inside Diff3F (`diff3f_features.py`), fp16 weight files. Downloaded at run
  time.

## ControlNet v1.1 depth and normal (not included)

- **Source:** https://huggingface.co/lllyasviel/control_v11f1p_sd15_depth and
  https://huggingface.co/lllyasviel/control_v11p_sd15_normalbae
- **License:** CreativeML Open RAIL-M (`openrail` on the model cards)
- **Use:** Condition Stable Diffusion on the rendered depth and normal maps inside Diff3F. Downloaded at
  run time (fp16 files).

## DINOv2 ViT-B/14 (not included)

- **Source:** https://github.com/facebookresearch/dinov2 (`dinov2_vitb14` through `torch.hub`)
- **License:** Apache-2.0
- **Paper:** Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision", arXiv:2304.07193.
- **Use:** Image features fused with the diffusion features inside Diff3F. Downloaded at run time.

## rembg (Python package)

- **Source:** https://github.com/danielgatis/rembg, version 2.0.61 (from PyPI)
- **License:** MIT
- **Use:** Background removal in `scripts/idarb/prepare_images.py`, which turns photos into the RGBA
  inputs IDArb expects. Installed in the server environment (`scripts/idarb/requirements-server.txt`).

## U²-Net background-removal model (not included)

- **Source:** https://github.com/xuebinqin/U-2-Net; rembg downloads the `u2net` ONNX weights from its own
  GitHub releases at first use and caches them outside this repository.
- **License:** Apache-2.0
- **Paper:** Qin et al., "U²-Net: Going deeper with nested U-structure for salient object detection",
  Pattern Recognition 106 (2020).
- **Use:** The default model (`--model u2net`) used by `prepare_images.py` through rembg.

## Objaverse sample meshes (included, `results/diff3f/targets/`)

Five full meshes used as registration targets for the sample images that have no counterpart among
Diff3F's sample meshes. They come from Objaverse (Deitke et al., "Objaverse: A Universe of Annotated 3D
Objects", CVPR 2023; https://huggingface.co/datasets/allenai/objaverse), which mirrors Sketchfab models
under Creative Commons licenses. All five are **CC BY 4.0**; the author must be credited when the meshes
or results made from them are shown. `results/diff3f/targets/sources.json` holds the same data. The
meshes are stored without textures, cleaned, decimated to at most 40000 triangles, centred and scaled to
a bounding-box diagonal of 2.

| File | Sketchfab model (uid) | Author | License |
|---|---|---|---|
| `armchair.ply` | "Fri, 21 Sep 2018 19:46:37" (`e9696f0f630949a4b0360804a02d0ce6`), a museum chair scan | newfields-3dprinting | CC BY 4.0 |
| `roundchair.ply` | "Sat, 18 May 2019 20:26:15" (`a911adbf46d14001954e6b41d4f52ff0`), a museum chair scan | newfields-3dprinting | CC BY 4.0 |
| `panda.ply` | "Panda Shake Head" (`229c3e79701b4485b33bc93e2fa6156f`) | Wang.Jun.Jie | CC BY 4.0 |
| `bird.ply` | "The Sparrow 3rd November 2021" (`4809199aa426497daa46aeee398dd387`) | cornsoup | CC BY 4.0 |
| `lizard.ply` | "Charmeleon" (`6fe52b29551e4e27bdf4e2a830fb90db`) | bharat_bk | CC BY 4.0 (fan model of a Pokémon character; the character itself is Nintendo/Game Freak property, so use it for research only, not in publications) |

Each model's page is `https://sketchfab.com/3d-models/<uid>`. Fetched on 2026-10-20 with the `objaverse`
Python package (Apache-2.0), which is not needed to run anything in this repository.
