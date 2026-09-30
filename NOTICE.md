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
