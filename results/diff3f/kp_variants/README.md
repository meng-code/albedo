# Refinement variants on the keypoint benchmark (GPU server, 2026-10-23)

Code: diff3f-registration e51bff1. Features: outputs/diff3f (zoom 1). Run 2 at a time with 3 threads each (the server has 8 vCPUs):

```
python scripts/idarb/eval_features.py --feature_dir outputs/diff3f --out_dir ~/semtex/kpvar/v1_filter_view --csv results/diff3f/kp_variants/v1_filter_view.csv --variants arap --threads 3 --view anchors --refine_args "--w_view 0.3 --anchor_filter rigid:0.1"
python scripts/idarb/eval_features.py --feature_dir outputs/diff3f --out_dir ~/semtex/kpvar/v2_keep07 --csv results/diff3f/kp_variants/v2_keep07.csv --variants arap --threads 3 --view anchors --refine_args "--w_view 0.3 --anchor_filter rigid:0.1 --keep_confident 0.7"
python scripts/idarb/eval_features.py --feature_dir outputs/diff3f --out_dir ~/semtex/kpvar/v3_k16 --csv results/diff3f/kp_variants/v3_k16.csv --variants arap --threads 3 --view anchors --refine_args "--w_view 0.3 --anchor_filter rigid:0.1 --k 16"
python scripts/idarb/eval_features.py --feature_dir outputs/diff3f --out_dir ~/semtex/kpvar/v4_k16_keep07 --csv results/diff3f/kp_variants/v4_k16_keep07.csv --variants arap --threads 3 --view anchors --refine_args "--w_view 0.3 --anchor_filter rigid:0.1 --k 16 --keep_confident 0.7"
python scripts/idarb/eval_features.py --feature_dir outputs/diff3f --out_dir ~/semtex/kpvar/v5_keep07_noarap --csv results/diff3f/kp_variants/v5_keep07_noarap.csv --variants arap --threads 3 --view anchors --refine_args "--w_view 0.3 --anchor_filter rigid:0.1 --keep_confident 0.7 --arap 0"
python scripts/idarb/eval_features.py --feature_dir outputs/diff3f --out_dir ~/semtex/kpvar/v6_keep05_w5 --csv results/diff3f/kp_variants/v6_keep05_w5.csv --variants arap --threads 3 --view anchors --refine_args "--w_view 0.3 --anchor_filter rigid:0.1 --keep_confident 0.5 --keep_weight 10"
python scripts/idarb/keypoint_table.py --feature_dir outputs/diff3f --result v1=~/semtex/kpvar/v1_filter_view --result v2=~/semtex/kpvar/v2_keep07 --result v3=~/semtex/kpvar/v3_k16 --result v4=~/semtex/kpvar/v4_k16_keep07 --result v5=~/semtex/kpvar/v5_keep07_noarap --result v6=~/semtex/kpvar/v6_keep05_w5 --csv results/diff3f/kp_variants/keypoint_accuracy_variants.csv
```

Group means printed by keypoint_table.py (mean error, PCK@2 %, PCK@5 %, PCK@10 %):

```
means per group and method (mean error, PCK@2 %, PCK@5 %, PCK@10 %):
  same   nearest_neighbour  pairs 13  mean 0.053  pck2 0.54  pck5 0.76  pck10 0.89
  same   v1                 pairs 13  mean 0.052  pck2 0.37  pck5 0.66  pck10 0.88
  same   v2                 pairs 13  mean 0.051  pck2 0.35  pck5 0.70  pck10 0.85
  same   v3                 pairs 13  mean 0.057  pck2 0.34  pck5 0.67  pck10 0.83
  same   v4                 pairs 13  mean 0.054  pck2 0.33  pck5 0.69  pck10 0.83
  same   v5                 pairs 13  mean 0.048  pck2 0.44  pck5 0.70  pck10 0.87
  same   v6                 pairs 13  mean 0.051  pck2 0.41  pck5 0.68  pck10 0.85
  cross  nearest_neighbour  pairs  2  mean 0.083  pck2 0.05  pck5 0.40  pck10 0.70
  cross  v1                 pairs  2  mean 0.083  pck2 0.05  pck5 0.35  pck10 0.75
  cross  v2                 pairs  2  mean 0.084  pck2 0.05  pck5 0.35  pck10 0.75
  cross  v3                 pairs  2  mean 0.077  pck2 0.10  pck5 0.40  pck10 0.75
  cross  v4                 pairs  2  mean 0.078  pck2 0.05  pck5 0.35  pck10 0.75
  cross  v5                 pairs  2  mean 0.071  pck2 0.05  pck5 0.35  pck10 0.85
  cross  v6                 pairs  2  mean 0.094  pck2 0.05  pck5 0.30  pck10 0.60
```

## Zoom-2 features, batch 1 (2026-10-23)

Features: features-3d-tmp 71ac59a, results/diff3f/features_zoom2/ (cat_full, cat_partial, camel_full,
labubu_partial, lion_full and the synthetic pairs). Code: diff3f-registration 7a83c45.

```
F=~/semtex/wt_f3d/results/diff3f/features_zoom2
python scripts/idarb/eval_features.py --feature_dir $F --out_dir ~/semtex/kpvar/zoom2_v1 --csv results/diff3f/kp_variants/zoom2_v1.csv --groups same cross synthetic --pairs cat_lion labubu_camel labubu_lion cat_synth cow_synth --variants arap --threads 3 --view anchors --refine_args "--w_view 0.3 --anchor_filter rigid:0.1"
python scripts/idarb/eval_features.py --feature_dir $F --out_dir ~/semtex/kpvar/zoom2_v1 --csv results/diff3f/kp_variants/zoom2_v1_cat.csv --groups same --pairs cat --variants arap --threads 3 --view anchors --refine_args "--w_view 0.3 --anchor_filter rigid:0.1"
python scripts/idarb/keypoint_table.py --feature_dir $F --result zoom2_v1=~/semtex/kpvar/zoom2_v1 --csv results/diff3f/kp_variants/keypoint_accuracy_zoom2.csv
```

The zoom-1 rows for the same pairs are the nearest_neighbour and v1 rows of
keypoint_accuracy_variants.csv (same code path, features outputs/diff3f).

## Zoom-2 features, all keypoint pairs (2026-10-24)

```
F=~/semtex/wt_f3d/results/diff3f/features_zoom2
python scripts/idarb/eval_features.py --feature_dir $F --out_dir ~/semtex/kpvar/zoom2_all_v1 --csv results/diff3f/kp_variants/zoom2_all_v1.csv --variants arap --threads 3 --view anchors --refine_args "--w_view 0.3 --anchor_filter rigid:0.1"
python scripts/idarb/keypoint_table.py --feature_dir $F --result zoom2_v1=~/semtex/kpvar/zoom2_all_v1 --csv results/diff3f/kp_variants/keypoint_accuracy_zoom2_all.csv
```

Group means (mean error, PCK@2 / 5 / 10 %), zoom 1 (keypoint_accuracy_variants.csv) -> zoom 2:

```
same  (13)  nearest neighbour  0.053 0.54 0.76 0.89 -> 0.048 0.63 0.77 0.89
same  (13)  v1                 0.052 0.37 0.66 0.88 -> 0.048 0.43 0.69 0.87
cross  (2)  nearest neighbour  0.083 0.05 0.40 0.70 -> 0.079 0.10 0.55 0.70
cross  (2)  v1                 0.083 0.05 0.35 0.75 -> 0.084 0.05 0.40 0.60
```
