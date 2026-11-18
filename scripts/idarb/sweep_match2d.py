"""
The 2D route (match2d.py: the photo against renders of the target) over the annotated
same-category pairs for several extractor settings, scored at the keypoints.

A configuration is <extractor>:<resolution>:<views>, for example marco:770:3, marco:840:5 or
dinov2:840:5 (views 3: the relief camera and two turned views; 5: two tilted views more). Every
pair's match2d.py result goes to <out_root>/<tag>/same/<pair>/refined.npz, the layout
keypoint_table.py reads; the relief camera's direction is taken from an existing chain result
(--view_root, the refined.npz of the same pair). Then keypoint_table.py --plane over all
configurations and one summary row per method: hits and PCK@2 pooled over the keypoints, the
mean error, and the paired difference against the nearest neighbour of the Diff3F features
(95 % interval from a bootstrap over pairs).

Needs torch and a GPU for MARCO (about 3.6 GB); the matching itself is on the CPU.

Usage (repository root, on the server):
    python scripts/idarb/sweep_match2d.py --configs marco:770:3 marco:840:5 dinov2:840:5 \
        --view_root outputs/diff3f/24_debias/none --feature_dir outputs/diff3f/features_zoom2
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)
import eval_features as ef  # noqa: E402
from sweep_anchor_filter import git_commit, paired_interval, read_csv, run  # noqa: E402

KEYPOINT_DIRS = ["results/diff3f/keypoints", "results/diff3f/landmarks"]


def parse_config(config):
    """'marco:770:5' -> (extractor, resolution, views, tag 'marco_770_5')."""
    extractor, resolution, views = config.split(":")
    if int(views) not in (1, 3, 5):
        raise ValueError("views must be 1, 3 or 5: " + config)
    return extractor, int(resolution), int(views), config.replace(":", "_")


def summarise(csv_dir, tags):
    rows = [r for r in read_csv(os.path.join(csv_dir, "keypoint_accuracy.csv"))
            if r["group"] == "same"]
    methods = ["nearest_neighbour"] + tags
    per = {m: {r["pair"]: r for r in rows if r["method"] == m} for m in methods}
    base = per["nearest_neighbour"]
    out = []
    for m in methods:
        for suffix in ("", "_xy"):
            rs = per[m] if not suffix else {r["pair"]: r for r in rows
                                            if r["method"] == m + suffix}
            pairs = sorted(rs)
            if not pairs:
                continue
            n = np.array([int(rs[p]["n"]) for p in pairs])
            hits = np.array([round(float(rs[p]["pck_2"]) * int(rs[p]["n"])) for p in pairs])
            row = dict(method=m + suffix, pairs=len(pairs), keypoints=int(n.sum()),
                       hits_pck2=int(hits.sum()),
                       pck2_pooled=round(100.0 * hits.sum() / n.sum(), 1),
                       mean_error=round(float(np.mean([float(rs[p]["mean"]) for p in pairs])), 4),
                       diff_vs_nn="", ci_low="", ci_high="")
            if m != "nearest_neighbour" and not suffix:
                common = [p for p in pairs if p in base]
                nb = np.array([int(base[p]["n"]) for p in common])
                ha = np.array([round(float(base[p]["pck_2"]) * int(base[p]["n"]))
                               for p in common])
                hb = np.array([round(float(rs[p]["pck_2"]) * int(rs[p]["n"])) for p in common])
                d, lo, hi = paired_interval(ha, hb, nb)
                row.update(diff_vs_nn=round(d, 1), ci_low=round(lo, 1), ci_high=round(hi, 1))
            out.append(row)
    with open(os.path.join(csv_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    cols = list(out[0].keys())
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in out:
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--configs", nargs="+", default=["marco:770:3", "marco:840:5"])
    ap.add_argument("--view_root", default="outputs/diff3f/24_debias/none",
                    help="eval_features.py out_dir whose refined.npz holds each pair's view")
    ap.add_argument("--feature_dir", default="outputs/diff3f/features_zoom2",
                    help="Diff3F features, for the nearest-neighbour row of the table")
    ap.add_argument("--out_root", default="outputs/diff3f/25_match2d")
    ap.add_argument("--csv_dir", default="results/diff3f/match2d_marco")
    ap.add_argument("--pairs", nargs="*", help="only these pair names (for a quick check)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--match2d_args", default="", help="more arguments for match2d.py")
    ap.add_argument("--summarise_only", action="store_true",
                    help="rebuild the table from the CSV already in --csv_dir")
    args = ap.parse_args()

    parsed = [parse_config(c) for c in args.configs]
    tags = [p[3] for p in parsed]
    os.makedirs(args.csv_dir, exist_ok=True)
    if not args.summarise_only:
        py = sys.executable
        commands, seconds = [], {}
        for extractor, resolution, views, tag in parsed:
            t0 = time.time()
            done = 0
            for _group, name, _s, src_mesh, target, _tr, _ex in ef.pair_table(["same"]):
                if args.pairs and name not in args.pairs:
                    continue
                if not any(os.path.exists(os.path.join(d, name + ".json"))
                           for d in KEYPOINT_DIRS):
                    continue
                view_npz = os.path.join(args.view_root, "same", name, "refined.npz")
                if not os.path.exists(view_npz):
                    print("no view for " + name + ", skipped", flush=True)
                    continue
                cmd = [py, os.path.join(SCRIPTS, "match2d.py"), "--relief", src_mesh,
                       "--target_mesh", ef.TARGET_MESH[target], "--view_from", view_npz,
                       "--extractor", extractor, "--resolution", str(resolution),
                       "--turn", "25" if views >= 3 else "0",
                       "--elev", "25" if views >= 5 else "0", "--device", args.device,
                       "--out", os.path.join(args.out_root, tag, "same", name, "refined.npz")]
                cmd += args.match2d_args.split()
                run(cmd)
                commands.append(" ".join(cmd))
                done += 1
            seconds[tag] = round((time.time() - t0) / max(done, 1), 1)
        cmd = [py, os.path.join(SCRIPTS, "keypoint_table.py"), "--feature_dir", args.feature_dir,
               "--plane", "--csv", os.path.join(args.csv_dir, "keypoint_accuracy.csv")]
        for tag in tags:
            cmd += ["--result", f"{tag}={os.path.join(args.out_root, tag)}"]
        run(cmd)
        commands.append(" ".join(cmd))
        with open(os.path.join(args.csv_dir, "run.json"), "w") as f:
            json.dump(dict(commit=git_commit(), command=" ".join(sys.argv), configs=args.configs,
                           seconds_per_pair=seconds, steps=commands), f, indent=1)
    summarise(args.csv_dir, tags)


if __name__ == "__main__":
    main()
