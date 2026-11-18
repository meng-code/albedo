"""
Sensitivity of the chain to the numbers of the candidate-choice stage that were fixed without a
comparison (refine() of refine_correspondence.py), one parameter at a time around the standing
setting (--anchor_filter rigid:0.1 --w_view 0.3 --view_from_anchors --final_rematch 0.05
--rematch_rounds 3 on top of eval_features.py's field and ARAP arguments, same-category pairs):

  prior_sigma   width of the Gaussian radial basis functions of the position prior, of the
                target diagonal (standing 0.25); 'affine' drops them and keeps the affine part
  w_smooth      weight of the neighbour term in the smoothing passes (0.3)
  iters         number of smoothing passes (2)
  radius        scale of the position term in the candidate cost, of the source diagonal (0.05).
                The same number is the unit of the gate's "windows away from the prior", so the
                gate moves with it
  k             candidates per target vertex (64)
  radius_s      scale of the neighbour term (0.05); its effect is that of w_smooth / radius_s^2,
                so it is not in the default plan

--sweep takes <parameter>=<value>,<value>,... per parameter. The standing setting is run once
(tag 'standing') and serves every parameter; a value equal to the standing one refers to that
run. Every other value runs eval_features.py into <out_root>/<tag> and <csv_dir>/<tag>.csv;
runs whose CSV exists are skipped unless --rerun. Then keypoint_table.py over all runs and one
summary row per parameter and value:

  keypoints   hits and PCK@2 pooled over the annotated pairs, the mean error, and the paired
              difference of the pooled PCK@2 against the standing run with a 95 % interval from
              a bootstrap over pairs
  texture     triangles within 2 % / 5 %, triangles distorted more than 10 times and mirrored
              area (pair means, in per cent), and for within 2 % and mirrored area the mean
              per-pair difference against the standing run with its bootstrap interval

CPU only. About 5 minutes per run at --jobs 4 --threads 2.

Usage (repository root):
    python scripts/idarb/sweep_chain_defaults.py --jobs 4 --threads 2
    python scripts/idarb/sweep_chain_defaults.py --sweep k=16,64 --pairs cat_cat   # quick check
    python scripts/idarb/sweep_chain_defaults.py --summarise_only
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)
from compare_chain_runs import mean_interval  # noqa: E402
from sweep_anchor_filter import git_commit, paired_interval, pct, read_csv, run  # noqa: E402

FIXED_ARGS = "--anchor_filter rigid:0.1 --w_view 0.3 --final_rematch 0.05 --rematch_rounds 3"
# parameter -> (option of refine_correspondence.py, standing value)
PARAMS = {
    "prior_sigma": ("--prior_sigma", "0.25"),
    "w_smooth": ("--w_smooth", "0.3"),
    "iters": ("--iters", "2"),
    "radius": ("--radius", "0.05"),
    "k": ("--k", "64"),
    "radius_s": ("--radius_s", "0.05"),
}
DEFAULT_SWEEP = [
    "prior_sigma=0.05,0.1,0.15,0.25,0.4,0.6,1.0,affine",
    "w_smooth=0,0.1,0.3,0.6,1.0,3.0",
    "iters=0,1,2,4",
    "radius=0.025,0.05,0.075,0.1,0.2",
    "k=8,16,32,64,128",
]
STANDING = "standing"


def option_value(param, value):
    """The value as it goes on the command line ('affine' is --prior_sigma 0)."""
    if value == "affine":
        if param != "prior_sigma":
            raise ValueError("'affine' is a value of prior_sigma only")
        return "0"
    float(value)
    return value


def tag_of(param, value):
    """'standing' for the standing value of the parameter, else e.g. 'prior_sigma_0p05'."""
    if value != "affine" and float(value) == float(PARAMS[param][1]):
        return STANDING
    return param + "_" + value.replace(".", "p")


def parse_sweep(items):
    """['k=8,64', ...] -> [(parameter, value, tag), ...] in the order given."""
    plan = []
    for item in items:
        param, _, values = item.partition("=")
        if param not in PARAMS or not values:
            raise ValueError("not <parameter>=<values> with a known parameter: " + item)
        for value in values.split(","):
            option_value(param, value)
            plan.append((param, value, tag_of(param, value)))
    return plan


def refine_args(param, value):
    """The --refine_args string of one run (param None: the standing setting)."""
    if param is None:
        return FIXED_ARGS
    return FIXED_ARGS + " " + PARAMS[param][0] + " " + option_value(param, value)


def hits_of(rows, pairs):
    return np.array([round(float(rows[p]["pck_2"]) * int(rows[p]["n"])) for p in pairs])


def summarise(kp, tex, plan):
    """kp[tag][pair]: a keypoint_table row; tex[tag][pair]: an eval_features row of the chain.
    One row per (parameter, value) of the plan."""
    base_kp, base_tex = kp[STANDING], tex[STANDING]
    out = []
    for param, value, tag in plan:
        rows, t = kp[tag], tex[tag]
        pairs = sorted(rows)
        n = np.array([int(rows[p]["n"]) for p in pairs])
        hits = hits_of(rows, pairs)
        row = dict(
            parameter=param, value=value, run=tag,
            pairs_keypoints=len(pairs), keypoints=int(n.sum()), hits_pck2=int(hits.sum()),
            pck2_pooled=round(100.0 * hits.sum() / n.sum(), 1),
            mean_error=round(float(np.mean([float(rows[p]["mean"]) for p in pairs])), 4),
            pairs_texture=len(t),
            within_2pct=pct([float(r["within_2pct"]) for r in t.values()]),
            within_5pct=pct([float(r["within_5pct"]) for r in t.values()]),
            distortion_over_10=pct([float(r["distortion_over_10"]) for r in t.values()]),
            mirrored_area=pct([float(r["flipped_area"]) for r in t.values()]),
            anchors=round(float(np.mean([int(r["mutual_anchors"]) for r in t.values()])), 1),
            pck2_diff="", pck2_ci_low="", pck2_ci_high="",
            within_2pct_diff="", within_2pct_ci_low="", within_2pct_ci_high="",
            mirrored_diff="", mirrored_ci_low="", mirrored_ci_high="",
        )
        if tag != STANDING:
            common = [p for p in pairs if p in base_kp]
            nb = np.array([int(base_kp[p]["n"]) for p in common])
            d, lo, hi = paired_interval(hits_of(base_kp, common), hits_of(rows, common), nb)
            row.update(pck2_diff=round(d, 1), pck2_ci_low=round(lo, 1), pck2_ci_high=round(hi, 1))
            both = sorted(set(t) & set(base_tex))
            for key, name in (("within_2pct", "within_2pct"), ("flipped_area", "mirrored")):
                diff = 100.0 * np.array([float(t[p][key]) - float(base_tex[p][key])
                                         for p in both])
                m, lo, hi = mean_interval(diff)
                row.update({name + "_diff": round(m, 1), name + "_ci_low": round(lo, 1),
                            name + "_ci_high": round(hi, 1)})
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sweep", nargs="+", default=DEFAULT_SWEEP,
                    help="<parameter>=<value>,<value>,... per parameter")
    ap.add_argument("--feature_dir", default="outputs/diff3f/features_zoom2")
    ap.add_argument("--out_root", default="outputs/diff3f/26_chain_defaults")
    ap.add_argument("--csv_dir", default="results/diff3f/chain_defaults_sweep")
    ap.add_argument("--pairs", nargs="*", help="only these pair names (for a quick check)")
    ap.add_argument("--jobs", type=int, default=4, help="pairs evaluated at once")
    ap.add_argument("--threads", type=int, default=2, help="BLAS / OpenMP threads per step")
    ap.add_argument("--rerun", action="store_true", help="also the runs whose CSV exists")
    ap.add_argument("--summarise_only", action="store_true",
                    help="rebuild the table from the CSVs already in --csv_dir")
    args = ap.parse_args()

    plan = parse_sweep(args.sweep)
    runs = {STANDING: (None, None)}
    for param, value, tag in plan:
        runs.setdefault(tag, (param, value))
    os.makedirs(args.csv_dir, exist_ok=True)
    kp_csv = os.path.join(args.csv_dir, "keypoint_accuracy.csv")
    if not args.summarise_only:
        py = sys.executable
        commands = []
        for tag, (param, value) in runs.items():
            out_csv = os.path.join(args.csv_dir, tag + ".csv")
            if os.path.exists(out_csv) and not args.rerun:
                print("kept " + out_csv, flush=True)
                continue
            cmd = [py, os.path.join(SCRIPTS, "eval_features.py"),
                   "--feature_dir", args.feature_dir, "--groups", "same",
                   "--variants", "nn", "arap", "--view", "anchors",
                   "--refine_args", refine_args(param, value),
                   "--threads", str(args.threads), "--jobs", str(args.jobs),
                   "--out_dir", os.path.join(args.out_root, tag), "--csv", out_csv]
            if args.pairs:
                cmd += ["--pairs"] + args.pairs
            run(cmd)
            commands.append(" ".join(cmd))
        cmd = [py, os.path.join(SCRIPTS, "keypoint_table.py"), "--feature_dir", args.feature_dir,
               "--csv", kp_csv]
        for tag in runs:
            cmd += ["--result", f"{tag}={os.path.join(args.out_root, tag)}"]
        run(cmd)
        commands.append(" ".join(cmd))
        with open(os.path.join(args.csv_dir, "run.json"), "w") as f:
            json.dump(dict(commit=git_commit(), command=" ".join(sys.argv), fixed_args=FIXED_ARGS,
                           standing={p: v[1] for p, v in PARAMS.items()}, sweep=args.sweep,
                           jobs=args.jobs, threads=args.threads, steps=commands), f, indent=1)

    kp_rows = [r for r in read_csv(kp_csv) if r["group"] == "same"]
    kp = {tag: {r["pair"]: r for r in kp_rows if r["method"] == tag} for tag in runs}
    tex = {tag: {r["pair"]: r for r in read_csv(os.path.join(args.csv_dir, tag + ".csv"))
                 if r["group"] == "same" and r["variant"] == "arap"} for tag in runs}
    out = summarise(kp, tex, plan)
    with open(os.path.join(args.csv_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    cols = list(out[0].keys())
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in out:
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")


if __name__ == "__main__":
    main()
