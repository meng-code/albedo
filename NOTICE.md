# Third-party notices

Add each future dependency with its source URL, version or commit, license, intended use, and any redistribution conditions before it is incorporated into the project.

## IDArb (source code)

- **Path:** `third_party/IDArb/`
- **Source:** https://github.com/Lizb6626/IDArb, commit `df26b90` ("update bibtex")
- **License:** MIT, Copyright (c) 2024 Zhibing Li. The full license text is kept at `third_party/IDArb/LICENSE`.
- **Use:** Intrinsic decomposition of images into albedo, normal, metallic and roughness (inference), plus a small training smoke test.
- **Modifications:** The first commit that adds this directory is an unmodified copy of upstream. Later commits that touch `third_party/IDArb/` are local changes; see `git log -- third_party/IDArb`.
- **Paper:** Li et al., "IDArb: Intrinsic Decomposition for Arbitrary Number of Input Views and Illuminations", arXiv:2412.12083.
