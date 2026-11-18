"""
Ablation of register_partial.py: runs several registration settings on every (partial, full) pair
whose Diff3F feature files are present and writes one row per setting and pair.

Pairs (feature files in --feature_dir, meshes in the repository):
  cow, cat            our relief meshes from single images -> Diff3F's sample cow.obj / cat.off
  cow_synth, cat_synth  synthetic partial meshes cut from the same full meshes
                        (make_partial_mesh.py), with ground truth

Reported per row: the "selected" stage of register_partial.register() (agreement, part agreement,
distances, and the ground-truth columns when available), the feature-free baseline's agreement and
median distance, and the run time.

Usage (repository root, mesh environment):
    python scripts/idarb/eval_registration.py --out results/diff3f/ablation.csv
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_partial as rp  # noqa: E402

MESHES = dict(
    cow="third_party/Diff3F/meshes/cow.obj",
    cat="third_party/Diff3F/meshes/cat.off",
    man="third_party/Diff3F/meshes/man.obj",
    posed_human="third_party/Diff3F/meshes/posed_human.off",
    camel="third_party/Diff3F/meshes/camel.obj",
    lion="third_party/Diff3F/meshes/lion.off",
    armchair="results/diff3f/targets/armchair.ply",
    roundchair="results/diff3f/targets/roundchair.ply",
    panda="results/diff3f/targets/panda.ply",
    bird="results/diff3f/targets/bird.ply",
    lizard="results/diff3f/targets/lizard.ply",
)

SYNTH = dict(
    cow=("cow_synth_partial.npz", "cow_full.npz", "results/diff3f/synthetic/cow_synth"),
    cat=("cat_synth_partial.npz", "cat_full.npz", "results/diff3f/synthetic/cat_synth"),
)

# real pairs: name, source image, target mesh key
REAL = [
    ("cow", "cow", "cow"),
    ("cat", "cat", "cat"),
    ("superman_man", "superman", "man"),
    ("superman_posed", "superman", "posed_human"),
    ("okapi_camel", "okapi", "camel"),
    ("okapi1_camel", "okapi1", "camel"),
    ("cat5_cat", "cat5", "cat"),
    ("cat_lion", "cat", "lion"),
    ("labubu_man", "labubu", "man"),
    ("chair1_armchair", "chair1", "armchair"),
    ("chair_armchair", "chair", "armchair"),
    ("chair10_roundchair", "chair10", "roundchair"),
    ("chairgen_armchair", "ipadapter_result_00_chair", "armchair"),
    ("panda_panda", "panda", "panda"),
    ("bird_bird", "bird", "bird"),
    ("dragon_lizard", "fire_dragon", "lizard"),
]

PAIRS = [
    # name, source npz, target npz, source mesh, gt npz, z squash of the source, target mesh key
    (
        _n,
        f"{_s}_partial.npz",
        f"{_t}_full.npz",
        f"outputs/mesh_depth/{_s}/relief.ply",
        None,
        1.0,
        _t,
    )
    for _n, _s, _t in REAL
]
for _n, (_src, _dst, _base) in SYNTH.items():
    for _squash in (1.0, 0.5, 2.0):
        # squash != 1 multiplies the source's z after loading, to imitate a relief mesh whose depth
        # scale is wrong by 1/squash (the features stay those of the correct geometry)
        _name = f"{_n}_synth" if _squash == 1.0 else f"{_n}_synth_z{_squash:g}"
        PAIRS.append((_name, _src, _dst, _base + ".ply", _base + "_gt.npz", _squash, _n))
# the same two deformations built into the mesh before the features were computed
# (make_partial_mesh.py --depth_factor 0.5 / 2): the true depth factor is 2 and 0.5
for _d in ("0.5", "2"):
    PAIRS.append(
        (
            f"cow_synth_d{_d}",
            f"cow_synth_d{_d}_partial.npz",
            "cow_full.npz",
            f"results/diff3f/synthetic/cow_synth_d{_d}.ply",
            f"results/diff3f/synthetic/cow_synth_d{_d}_gt.npz",
            1.0,
            "cow",
        )
    )

CONFIGS = [
    # name, feature, options for register()
    ("v1_mutual_best", "full", dict(match_mode="mutual", n_candidates=1)),
    ("v2_both_select", "full", dict()),
    ("v2_consensus", "full", dict(consensus=True)),
    ("v2_normals", "full", dict(normals=True)),
    ("v2_depth_scale", "full", dict(depth_scale=True)),
    ("v2_seeds3", "full", dict(seeds=(0, 1, 2))),
    ("v3_joint", "full", dict(refine="joint")),
    ("v3_joint_normals", "full", dict(refine="joint", normals=True)),
    ("v3_joint_depth", "full", dict(refine="joint", depth_scale=True)),
    ("v3_joint_consensus", "full", dict(refine="joint", consensus=True)),
    ("v2_diffusion_only", "diffusion", dict()),
    ("v2_dino_only", "dino", dict()),
    ("v2_piecewise", "full", dict(piecewise=True)),
    (
        "v2_piecewise_loose",
        "full",
        dict(piecewise=True, part_scale_range=(0.6, 1.5), part_max_rot=30.0),
    ),
]

FIELDS = [
    "pair", "config", "feature", "seconds", "n_matches", "n_inliers", "candidate", "scale",
    "depth_factor", "agreement", "agreement_t", "score", "part_agreement", "coverage",
    "median_to_target", "frac_within_2pct",
    "rot_err_deg", "scale_ratio", "pos_err_median", "corr_acc_1pct", "corr_acc_5pct",
    "corr_median", "feature_nn_acc_1pct", "feature_nn_acc_5pct",
    "baseline_agreement", "baseline_median", "baseline_rot_err_deg", "baseline_corr_acc_5pct",
]  # fmt: skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_dir", default="outputs/diff3f/features")
    ap.add_argument("--out", default="results/diff3f/ablation.csv")
    ap.add_argument("--pairs", nargs="*", help="subset of pair names")
    ap.add_argument("--configs", nargs="*", help="subset of config names")
    args = ap.parse_args()

    rows = []
    for name, src_file, dst_file, src_mesh, gt_file, squash, target_key in PAIRS:
        if args.pairs and name not in args.pairs:
            continue
        src_path = os.path.join(args.feature_dir, src_file)
        dst_path = os.path.join(args.feature_dir, dst_file)
        if not (os.path.exists(src_path) and os.path.exists(dst_path)):
            print(f"{name}: feature files missing, skipped")
            continue
        gt = dict(np.load(gt_file, allow_pickle=False)) if gt_file else None
        dst_mesh = MESHES[target_key]
        for cfg, feature, options in CONFIGS:
            if args.configs and cfg not in args.configs:
                continue
            src = rp.load_features(src_path, feature, src_mesh)
            dst = rp.load_features(dst_path, feature, dst_mesh)
            if squash != 1.0:
                src["xyz"] = rp.apply_depth(src["xyz"], squash)
            t0 = time.time()
            res = rp.register(src, dst, gt, **options)
            stages = {r["stage"]: r for r in res["rows"]}
            sel = stages["selected"]
            base = stages["baseline_icp_no_features"]
            row = dict(pair=name, config=cfg, feature=feature, seconds=round(time.time() - t0, 1))
            row.update({k: sel.get(k) for k in FIELDS if k in sel})
            row["baseline_agreement"] = base["agreement"]
            row["baseline_median"] = base["median_to_target"]
            row["baseline_rot_err_deg"] = base.get("rot_err_deg")
            row["baseline_corr_acc_5pct"] = base.get("corr_acc_5pct")
            if "feature_nn" in stages:
                row["feature_nn_acc_1pct"] = stages["feature_nn"]["corr_acc_1pct"]
                row["feature_nn_acc_5pct"] = stages["feature_nn"]["corr_acc_5pct"]
            rows.append(row)
            print({k: v for k, v in row.items() if v is not None})
            if "piecewise" in stages:  # the per-part refinement of the selected result
                pw = stages["piecewise"]
                row2 = dict(row, config=cfg + "[piecewise]")
                row2.update({k: pw.get(k) for k in FIELDS if k in pw})
                rows.append(row2)
                print({k: v for k, v in row2.items() if v is not None})

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    import csv

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("saved", args.out)


if __name__ == "__main__":
    main()
