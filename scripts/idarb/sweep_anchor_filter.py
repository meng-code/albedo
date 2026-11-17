"""
Sensitivity of the chain to the anchor-filter threshold (the 0.1 in --anchor_filter rigid:0.1).

Only the threshold changes; everything else is the standing keypoint-oriented setting on the
zoom-2 Diff3F features (--w_view 0.3 --view_from_anchors --final_rematch 0.05 --rematch_rounds 3,
same-category pairs). For every threshold it runs eval_features.py into its own folder, then
keypoint_table.py over all of them, and writes one summary row per threshold:

  anchors      mutual pairs before the filter and anchors kept (mean over the pairs)
  keypoints    chain PCK@2 pooled over the annotated same-category pairs, the per-pair mean,
               and the mean error (fractions of the partial mesh's bounding-box diagonal)
  texture      triangles within 2 % / 5 % and mirrored area of the chain's texture (pair means)
  paired       pooled PCK@2 minus the control threshold's, with a 95 % interval from a
               bootstrap over pairs (one keypoint is about one point of PCK)

"none" runs the chain without the filter; its anchor count is the number before the filter.
CPU only. About 15 minutes per threshold on one machine (16 pairs, about 50 s each).

Usage (repository root, mesh environment):
    python scripts/idarb/sweep_anchor_filter.py \
        --thresholds none 0.05 0.075 0.1 0.15 0.2 0.3 --jobs 2
    python scripts/idarb/sweep_anchor_filter.py --summarise_only   # rebuild the table only
"""

import argparse
import csv
import json
import os
import subprocess
import sys

import numpy as np

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
FIXED_ARGS = "--w_view 0.3 --final_rematch 0.05 --rematch_rounds 3"


def tag_of(thr):
    return "none" if thr == "none" else "rigid" + thr.replace(".", "p")


def run(cmd):
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True)
        return out.stdout.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def pct(values):
    """Mean of fractions, in per cent with one decimal."""
    return round(100.0 * float(np.mean(values)), 1)


def paired_interval(hits_a, hits_b, n, draws=20000, seed=0):
    """Pooled PCK of b minus a in points, and its 95 % interval from resampling the pairs."""
    rng = np.random.default_rng(seed)
    diff = 100.0 * (hits_b.sum() - hits_a.sum()) / n.sum()
    idx = rng.integers(0, len(n), size=(draws, len(n)))
    d = 100.0 * (hits_b[idx].sum(axis=1) - hits_a[idx].sum(axis=1)) / n[idx].sum(axis=1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return diff, lo, hi


def summarise(args, tags):
    kp = [r for r in read_csv(os.path.join(args.csv_dir, "keypoint_accuracy.csv"))
          if r["group"] == "same"]
    per = {}
    for tag in tags:
        rows = {r["pair"]: r for r in kp if r["method"] == tag}
        tex = {r["pair"]: r for r in read_csv(os.path.join(args.csv_dir, tag + ".csv"))
               if r["group"] == "same" and r["variant"] == "arap"}
        per[tag] = dict(kp=rows, tex=tex)
    before = {p: int(r["mutual_anchors"]) for p, r in per.get("none", {}).get("tex", {}).items()}
    control = tag_of(args.control)
    out, pair_rows = [], []
    for tag in tags:
        rows, tex = per[tag]["kp"], per[tag]["tex"]
        pairs = sorted(rows)
        n = np.array([int(rows[p]["n"]) for p in pairs])
        hits = np.array([round(float(rows[p]["pck_2"]) * int(rows[p]["n"])) for p in pairs])
        row = dict(
            threshold=tag, pairs_texture=len(tex),
            anchors_before=round(float(np.mean([before[p] for p in tex if p in before])), 1)
            if before else "",
            anchors_kept=round(float(np.mean([int(r["mutual_anchors"]) for r in tex.values()])), 1),
            pairs_keypoints=len(pairs), keypoints=int(n.sum()), hits_pck2=int(hits.sum()),
            pck2_pooled=round(100.0 * hits.sum() / n.sum(), 1),
            pck2_pair_mean=pct([float(rows[p]["pck_2"]) for p in pairs]),
            mean_error=round(float(np.mean([float(rows[p]["mean"]) for p in pairs])), 4),
            within_2pct=pct([float(r["within_2pct"]) for r in tex.values()]),
            within_5pct=pct([float(r["within_5pct"]) for r in tex.values()]),
            mirrored_area=pct([float(r["flipped_area"]) for r in tex.values()]),
            diff_vs_control="", ci_low="", ci_high="",
        )
        if control in per and tag != control:
            base = per[control]["kp"]
            common = [p for p in pairs if p in base]
            nb = np.array([int(base[p]["n"]) for p in common])
            ha = np.array([round(float(base[p]["pck_2"]) * int(base[p]["n"])) for p in common])
            hb = np.array([round(float(rows[p]["pck_2"]) * int(rows[p]["n"])) for p in common])
            d, lo, hi = paired_interval(ha, hb, nb)
            row.update(diff_vs_control=round(d, 1), ci_low=round(lo, 1), ci_high=round(hi, 1))
        out.append(row)
        for p, r in tex.items():
            k = rows.get(p)
            pair_rows.append(dict(
                threshold=tag, pair=p, anchors_before=before.get(p, ""),
                anchors_kept=int(r["mutual_anchors"]),
                keypoints=int(k["n"]) if k else "",
                hits_pck2=round(float(k["pck_2"]) * int(k["n"])) if k else "",
                mean_error=k["mean"] if k else "",
                within_2pct=r["within_2pct"], within_5pct=r["within_5pct"],
                mirrored_area=r["flipped_area"],
            ))
    for name, rows in (("summary.csv", out), ("per_pair.csv", pair_rows)):
        with open(os.path.join(args.csv_dir, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    cols = list(out[0].keys())
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in out:
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--thresholds", nargs="+",
                    default=["none", "0.05", "0.075", "0.1", "0.15", "0.2", "0.3"],
                    help="fractions of the source diagonal; 'none' runs without the filter")
    ap.add_argument("--control", default="0.1", help="threshold the paired differences refer to")
    ap.add_argument("--feature_dir", default="outputs/diff3f/features_zoom2")
    ap.add_argument("--out_root", default="outputs/diff3f/23_anchor_filter_sweep")
    ap.add_argument("--csv_dir", default="results/diff3f/anchor_filter_sweep")
    ap.add_argument("--pairs", nargs="*", help="only these pair names (for a quick check)")
    ap.add_argument("--jobs", type=int, default=1, help="pairs evaluated at once")
    ap.add_argument("--threads", type=int, default=3, help="BLAS / OpenMP threads per step")
    ap.add_argument("--summarise_only", action="store_true",
                    help="rebuild the tables from the CSVs already in --csv_dir")
    args = ap.parse_args()

    tags = [tag_of(t) for t in args.thresholds]
    os.makedirs(args.csv_dir, exist_ok=True)
    if not args.summarise_only:
        py = sys.executable
        commands = []
        for thr, tag in zip(args.thresholds, tags):
            refine = FIXED_ARGS if thr == "none" else f"--anchor_filter rigid:{thr} {FIXED_ARGS}"
            cmd = [py, os.path.join(SCRIPTS, "eval_features.py"),
                   "--feature_dir", args.feature_dir, "--groups", "same",
                   "--variants", "nn", "arap", "--view", "anchors",
                   "--refine_args", refine, "--threads", str(args.threads),
                   "--jobs", str(args.jobs),
                   "--out_dir", os.path.join(args.out_root, tag),
                   "--csv", os.path.join(args.csv_dir, tag + ".csv")]
            if args.pairs:
                cmd += ["--pairs"] + args.pairs
            run(cmd)
            commands.append(" ".join(cmd))
        cmd = [py, os.path.join(SCRIPTS, "keypoint_table.py"), "--feature_dir", args.feature_dir,
               "--csv", os.path.join(args.csv_dir, "keypoint_accuracy.csv")]
        for tag in tags:
            cmd += ["--result", f"{tag}={os.path.join(args.out_root, tag)}"]
        run(cmd)
        commands.append(" ".join(cmd))
        with open(os.path.join(args.csv_dir, "run.json"), "w") as f:
            json.dump(dict(commit=git_commit(), command=" ".join(sys.argv), fixed_args=FIXED_ARGS,
                           thresholds=args.thresholds, steps=commands), f, indent=1)
    summarise(args, tags)


if __name__ == "__main__":
    main()
