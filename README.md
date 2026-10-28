# Report images: TRELLIS.2 generate-then-match texture previews

Per-pair texture previews of the TRELLIS.2 source run (eval_generated.py --model trellis2,
2026-10-25, on the Windows laptop; numbers in results/diff3f/generated_trellis2_merge.csv on
diff3f-registration), copied unchanged from outputs/diff3f/14_generated_match/trellis2/same/<pair>/:

- `<pair>_texture_transfer.png`: texture transferred from the generated TRELLIS.2 mesh alone
  (`<pair>/texture_transfer.png`).
- `<pair>_merged.png`: the merged texture (`<pair>/merged/merged.png`).

The run wrote no per-pair keypoint figure; its keypoints are JSON only
(results/diff3f/keypoints_gen/trellis2/). SHA256SUMS lists every png. Custody copy:
D:\albedo-archive\report-tmp\trellis2_match on the laptop.

# Further sets (13 same-category pairs each)

- `chain_before_after/`: `<pair>_old.png` is the texture_transfer.png of the old refinement chain
  at zoom 2 without the new options (server kp_variants run `zoom2_all_v1`, eval_features.py
  --variants arap --view anchors --refine_args "--w_view 0.3 --anchor_filter rigid:0.1"; PCK@2 on
  the 13 pairs 43.2 %, results/diff3f/kp_variants/keypoint_accuracy_zoom2_all.csv), from the
  laptop's archive copy archive-server-kpvar-tmp/kpvar/zoom2_all_v1/same/<pair>/arap/.
  `<pair>_rm5x3.png` is the same image for rm5x3 (laptop batch 7, outputs/diff3f/16_hard2/rm5x3,
  results/diff3f/hard_sweep2/). Unchanged copies.
- `c2f_photo/`: the 2026-10-26 photo-localisation runs on the laptop, named `<variant>_<pair>.png`.
  The set was over 30 MB, so images wider than 1200 px (the 1512 px texture previews) were
  downscaled to 1200 px; the 1000 px previews are unchanged.
  - local840a/b/c: 29_local2d preview.png (840 px; a --window 0.04, b 0.08, c --turn 25 --window 0.06)
  - local1120w025/w040: 30_local2d_1120 preview.png (1120 px, --window 0.025 / 0.04)
  - pervertex, pervertex-texture: 31_local2d_pv preview.png and arap/texture_transfer.png (--per_vertex)
  - field-base, field-a_t50_w3, field-b_t25_w3, field-c_t50_w10: 32_local_field
    arap/texture_transfer.png (rm5x3 with --local_npz soft data from 31; base = control)
  - cell, cell-texture: 33_local2d_cell preview.png and arap/texture_transfer.png (per-cell, 1120 px)
  - field-cell_t100_w3, field-cell_t100_w10: 34_local_field_cell arap/texture_transfer.png
- `lm2d/`: match2d v3 previews (outputs/diff3f/17_match2d_v3/<res1120|res840_5v>/same/<pair>/
  preview.png), named `<variant>_<pair>.png`; they differ from results/diff3f/match2d/previews,
  which are byte-identical to the v1 run (17_match2d).

Every set has its own SHA256SUMS. Custody copies: D:\albedo-archive\report-tmp\<set>.
