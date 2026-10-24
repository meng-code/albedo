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

## DINOv3 ViT-B/16 (not included)

- **Source:** https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m (327 MB), gated; the
  owner accepted the DINOv3 License with his Hugging Face account on 2026-10-24. Loaded with
  transformers 4.57.1 in the trellis2 environment (`scripts/idarb/dinov3_worker.py`).
- **License:** DINOv3 License (Meta), as shown on the model page.
- **Paper:** Siméoni et al., "DINOv3", arXiv:2508.10104.
- **Use:** Optional replacement of DINOv2 inside Diff3F (`scripts/idarb/diff3f_features.py --dino
  dinov3`), tested on five meshes on 2026-10-25.

## DistillDIFT (source code and weights, not included)

- **Source:** https://github.com/CompVis/DistillDIFT, commit `b177d83`, cloned outside this repository
  (`~/semtex/DistillDIFT` on the GPU server), with its LoRA checkpoints
  `eval/checkpoints/distilled_{s,ws,us}.ckpt` (86 MB each); the base model is DINOv2 ViT-B/14 with
  registers (`dinov2_vitb14_reg` through `torch.hub`, 331 MB, Apache-2.0).
- **License:** none stated. The repository has no license file and GitHub reports no license, so
  by default all rights are reserved by the authors. We only run it locally for research; neither
  its code nor its weights are copied into this repository. Ask the authors before redistributing
  anything, including features, in a publication or release.
- **Paper:** Fundel, Schusterbauer, Hu and Ommer, "DistillDIFT: Distillation of Diffusion Features
  for Semantic Correspondence", WACV 2025, arXiv:2412.03512.
- **Use:** Optional per-view features inside Diff3F (`scripts/idarb/diff3f_features.py --features
  distilldift`, default checkpoint `distilled_s`), tested on five meshes.

## MeshFM (source code and weights, not included)

- **Source:** https://github.com/threedle/MeshFM, commit `e6577b1`, cloned outside this repository
  (`~/semtex/MeshFM` on the GPU server). Weights `meshfm.pth` (502 MB, 125.5 M parameters) from the
  Google Drive link in its README (file id `1ONUMs3Ji_VlmyflO74No5rk5bAHM6qgc`), downloaded 2026-10-22.
- **License:** none stated. The repository has no license file and GitHub reports no license, so by
  default all rights are reserved by the authors. We only run it locally for research; neither its code
  nor its weights are copied into this repository. Ask the authors before redistributing anything,
  including features, in a publication or release.
- **Paper:** Zhou, Liu, Lang and Hanocka, "MeshFM: 2D Features Are All You Need for 3D Shape
  Understanding", ECCV 2026, arXiv:2607.27592.
- **Use:** Per-point semantic features as an alternative to Diff3F (`scripts/idarb/meshfm_features.py`).

## SigLIP2 base patch16-512 (not included)

- **Source:** https://huggingface.co/timm/vit_base_patch16_siglip_512.v2_webli (the vision tower of
  google/siglip2-base-patch16-512, loaded with timm 1.0.30), 374 MB, downloaded at first use
  2026-10-22.
- **License:** Apache-2.0
- **Paper:** Tschannen et al., "SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic
  Understanding, Localization, and Dense Features", arXiv:2502.14786.
- **Use:** Optional per-pixel features in `scripts/idarb/diff3f_features.py --siglip2`.
- **Also:** the full model with its text tower, https://huggingface.co/timm/ViT-B-16-SigLIP2-512
  (1.5 GB, loaded with open_clip as `ViT-B-16-SigLIP2-512` / `webli`; the vision weights are the same),
  downloaded 2026-10-23, for the part-naming test `scripts/idarb/siglip2_part_naming.py`.

## Segment Anything ViT-B (not included)

- **Source:** https://github.com/facebookresearch/segment-anything, Python package
  `segment-anything` 1.0, weights `sam_vit_b_01ec64.pth` (375 MB) from
  https://dl.fbaipublicfiles.com/segment_anything/, kept on the GPU server
  (`~/semtex/weights/sam/`), downloaded 2026-10-22.
- **License:** Apache-2.0
- **Paper:** Kirillov et al., "Segment Anything", ICCV 2023.
- **Use:** Optional MeshFM-style cleaning of the per-view features in
  `scripts/idarb/diff3f_features.py --sam_tau`. Tried on 2026-10-22 and found slightly worse; the
  option stays in the script.

## open_clip (Python package)

- **Source:** https://github.com/mlfoundations/open_clip, `open_clip_torch` 3.3.0 from PyPI
- **License:** MIT
- **Use:** Loads SigLIP2's image and text towers in `scripts/idarb/siglip2_part_naming.py`. Installed
  in the siglip2oc environment on the GPU server (`scripts/idarb/requirements-meshfm.txt`).

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
