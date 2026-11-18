"""
The standing chain with debiased features, and the oracle test of its re-match step.

Every configuration runs eval_features.py on the same-category pairs with the standing
keypoint-oriented setting (--anchor_filter rigid:0.1 --w_view 0.3 --view_from_anchors
--final_rematch 0.05 --rematch_rounds 3, zoom-2 Diff3F features) into its own folder:

  none             the control (the standing chain)
  <debias spec>    the same with --debias <spec> (feature_debias.py), for example center,
                   csls:10, center+csls:10, whiten:256:1
  oracle           the annotated keypoints as exact constraints of the field before the
                   re-match, then released (--landmarks_hard --landmarks_release): the map is
                   right at every keypoint when the re-match rounds start, so the keypoint
                   accuracy after them shows what the re-match does to a correct map. Not a
                   method: the evaluation's own keypoints are its input.

Then keypoint_table.py over all of them and one summary row per configuration: chain PCK@2
pooled over the annotated pairs, mean error, the texture numbers, the paired difference against
the control (95 % interval from a bootstrap over pairs) and, for the oracle, the keypoints'
distance from the truth before and after the re-match rounds.

CPU only, about 8 minutes per configuration with --jobs 8.

Usage (repository root, mesh environment):
    python scripts/idarb/sweep_debias.py --configs none center csls:10 center+csls:10 \
        whiten:256:1 oracle --jobs 8 --threads 2
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)
import eval_features as ef  # noqa: E402
from feature_debias import tag_of  # noqa: E402
from sweep_anchor_filter import git_commit, paired_interval, pct, read_csv, run  # noqa: E402

FIXED_ARGS = "--anchor_filter rigid:0.1 --w_view 0.3 --final_rematch 0.05 --rematch_rounds 3"
KEYPOINT_DIRS = ["results/diff3f/keypoints", "results/diff3f/landmarks"]


def release_stats(out_dir, pairs):
    """Mean over the pairs of the released landmarks' drift before and after the re-match."""
    out = {}
    for key in ("release_before", "release_after"):
        vals = []
        for p in pairs:
            path = os.path.join(out_dir, "same", p, "refined.npz")
            if os.path.exists(path):
                st = json.loads(str(np.load(path)["stats"]))
                if key in st:
                    vals.append(st[key])
        if vals:
            out[key] = {c: round(float(np.mean([v[c] for v in vals])), 4) for c in vals[0]}
    return out


def summarise(args, tags):
    kp = [r for r in read_csv(os.path.join(args.csv_dir, "keypoint_accuracy.csv"))
          if r["group"] == "same"]
    per = {t: {r["pair"]: r for r in kp if r["method"] == t} for t in tags}
    base = per.get("none", {})
    out = []
    for tag in tags:
        rows = per[tag]
        tex = [r for r in read_csv(os.path.join(args.csv_dir, tag + ".csv"))
               if r["group"] == "same" and r["variant"] == "arap"]
        pairs = sorted(rows)
        n = np.array([int(rows[p]["n"]) for p in pairs])
        hits = np.array([round(float(rows[p]["pck_2"]) * int(rows[p]["n"])) for p in pairs])
        row = dict(
            config=tag, pairs_keypoints=len(pairs), keypoints=int(n.sum()),
            hits_pck2=int(hits.sum()), pck2_pooled=round(100.0 * hits.sum() / n.sum(), 1),
            mean_error=round(float(np.mean([float(rows[p]["mean"]) for p in pairs])), 4),
            pairs_texture=len(tex),
            anchors=round(float(np.mean([int(r["mutual_anchors"]) for r in tex])), 1),
            within_2pct=pct([float(r["within_2pct"]) for r in tex]),
            within_5pct=pct([float(r["within_5pct"]) for r in tex]),
            mirrored_area=pct([float(r["flipped_area"]) for r in tex]),
            diff_vs_none="", ci_low="", ci_high="", release_before="", release_after="",
        )
        if base and tag != "none":
            common = [p for p in pairs if p in base]
            nb = np.array([int(base[p]["n"]) for p in common])
            ha = np.array([round(float(base[p]["pck_2"]) * int(base[p]["n"])) for p in common])
            hb = np.array([round(float(rows[p]["pck_2"]) * int(rows[p]["n"])) for p in common])
            d, lo, hi = paired_interval(ha, hb, nb)
            row.update(diff_vs_none=round(d, 1), ci_low=round(lo, 1), ci_high=round(hi, 1))
        for key, val in release_stats(os.path.join(args.out_root, tag), pairs).items():
            row[key] = json.dumps(val)
        out.append(row)
    with open(os.path.join(args.csv_dir, "summary.csv"), "w", newline="") as f:
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
    ap.add_argument("--configs", nargs="+",
                    default=["none", "center", "csls:10", "center+csls:10", "whiten:256:1",
                             "oracle"])
    ap.add_argument("--feature_dir", default="outputs/diff3f/features_zoom2")
    ap.add_argument("--out_root", default="outputs/diff3f/24_debias")
    ap.add_argument("--csv_dir", default="results/diff3f/debias_chain")
    ap.add_argument("--pairs", nargs="*", help="only these pair names (for a quick check)")
    ap.add_argument("--jobs", type=int, default=1, help="pairs evaluated at once")
    ap.add_argument("--threads", type=int, default=3, help="BLAS / OpenMP threads per step")
    ap.add_argument("--summarise_only", action="store_true",
                    help="rebuild the table from the CSVs already in --csv_dir")
    args = ap.parse_args()

    tags = [tag_of(c) for c in args.configs]
    os.makedirs(args.csv_dir, exist_ok=True)
    if not args.summarise_only:
        py = sys.executable
        commands = []
        for config, tag in zip(args.configs, tags):
            refine, more, pairs = FIXED_ARGS, [], args.pairs
            if config == "oracle":
                refine += " --landmarks_hard --landmarks_release"
                more = ["--landmarks_from"] + KEYPOINT_DIRS
                annotated = [n for n, _s, _t in ef.SAME
                             if any(os.path.exists(os.path.join(d, n + ".json"))
                                    for d in KEYPOINT_DIRS)]
                pairs = [p for p in annotated if not args.pairs or p in args.pairs]
            elif config != "none":
                refine += " --debias " + config
            cmd = [py, os.path.join(SCRIPTS, "eval_features.py"),
                   "--feature_dir", args.feature_dir, "--groups", "same",
                   "--variants", "nn", "arap", "--view", "anchors",
                   "--refine_args", refine, "--threads", str(args.threads),
                   "--jobs", str(args.jobs),
                   "--out_dir", os.path.join(args.out_root, tag),
                   "--csv", os.path.join(args.csv_dir, tag + ".csv")] + more
            if pairs:
                cmd += ["--pairs"] + pairs
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
                           configs=args.configs, steps=commands), f, indent=1)
    summarise(args, tags)


if __name__ == "__main__":
    main()
