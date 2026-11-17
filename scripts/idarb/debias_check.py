"""
Two keypoint-level checks on the existing features, without running the chain:

1. Do statistical corrections of the features (feature_debias.py: centering, whitening, CSLS
   hub correction) move the nearest neighbour at the annotated keypoints?
2. Does the similarity peak sit at the right place when the search already starts at the right
   place? For every keypoint the search is restricted to the source samples near the TRUE
   source point, and the most similar one is taken:
     rematch_truth   among the keypoint's 64 candidates within 5 % of the true point: exactly
                     what the chain's final re-match would choose from a perfect field (the
                     field position, error 0, is kept when no candidate is that close)
     peak_5, peak_10 among all source samples within 5 % / 10 % of the true point
   If these stay near the nearest neighbour's accuracy, the similarity's local maximum is itself
   displaced from the true point, and no better prior can help while the last step re-matches
   on the same similarity.

Also per variant: bound_64 (a sample within 2 % of the truth is among the 64 candidates), and
hubness: the share of all target samples whose nearest neighbour is one of the 1 % most often
chosen source samples, and the share of the wrong keypoint matches that land on such a hub.

All errors are fractions of the partial mesh's bounding-box diagonal; a hit is an error below
2 %. Pooled over the annotated same-category pairs (cross pairs in the per-pair CSV only). The
paired difference of the nearest neighbour against "none" has a 95 % interval from a bootstrap
over pairs. CPU only, a few minutes.

Usage (repository root, mesh environment):
    python scripts/idarb/debias_check.py --feature_dir outputs/diff3f/features_zoom2 \
        --csv_dir results/diff3f/debias_check
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_features as ef  # noqa: E402
import register_partial as rp  # noqa: E402
from correspondence_maps import load_mesh  # noqa: E402
from feature_debias import debias  # noqa: E402
from sweep_anchor_filter import git_commit, paired_interval  # noqa: E402

VARIANTS = ["none", "center", "whiten:64:1", "whiten:256:1", "whiten:256:0.5", "whiten:1024:0.5",
            "csls:10", "center+csls:10", "whiten:256:1+csls:10"]


def nearest_all(fd, fs, chunk=2048):
    out = np.empty(len(fd), dtype=np.int64)
    for i in range(0, len(fd), chunk):
        out[i : i + chunk] = (fd[i : i + chunk] @ fs.T).argmax(axis=1)
    return out


def best_within(sim, dist, radius):
    """Per row, the error of the most similar sample among those nearer than radius to the
    truth (0 where there is none: the field position is kept), and whether there was one."""
    allowed = dist < radius
    pick = np.where(allowed, sim, -np.inf).argmax(axis=1)
    has = allowed.any(axis=1)
    return np.where(has, dist[np.arange(len(dist)), pick], 0.0), has


def check_pair(fs, fd, xs, t_idx, lm_s, diag, k, thr):
    """The measures of one pair and one feature variant; arrays have one entry per keypoint."""
    sim = fd[t_idx] @ fs.T
    dist = np.linalg.norm(xs[None] - lm_s[:, None], axis=2) / diag
    rows = np.arange(len(t_idx))
    nn = sim.argmax(axis=1)
    err_nn = dist[rows, nn]
    top = np.argpartition(-sim, min(k, sim.shape[1]) - 1, axis=1)[:, :k]
    sim_k, dist_k = np.take_along_axis(sim, top, 1), np.take_along_axis(dist, top, 1)
    err_re, has_re = best_within(sim_k, dist_k, 0.05)
    err_p5, _ = best_within(sim, dist, 0.05)
    err_p10, _ = best_within(sim, dist, 0.10)
    # hubs: the 1 % of the source samples chosen most often by the target samples
    counts = np.bincount(nearest_all(fd, fs), minlength=len(fs))
    hubs = np.zeros(len(fs), dtype=bool)
    hubs[np.argsort(-counts)[: max(len(fs) // 100, 1)]] = True
    wrong = err_nn >= thr
    return dict(
        n=len(t_idx), hits_nn=int((~wrong).sum()), err_nn=err_nn,
        bound=int((dist_k < thr).any(axis=1).sum()),
        hits_rematch=int((err_re < thr).sum()), rematch_none=int((~has_re).sum()),
        err_rematch=err_re, hits_peak5=int((err_p5 < thr).sum()), err_peak5=err_p5,
        hits_peak10=int((err_p10 < thr).sum()), err_peak10=err_p10,
        hub_share=float(counts[hubs].sum() / counts.sum()),
        wrong=int(wrong.sum()), wrong_on_hub=int((wrong & hubs[nn]).sum()),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feature_dir", default="outputs/diff3f/features_zoom2")
    ap.add_argument("--keypoints", default="results/diff3f/keypoints")
    ap.add_argument("--landmarks", default="results/diff3f/landmarks")
    ap.add_argument("--variants", nargs="+", default=VARIANTS)
    ap.add_argument("--k", type=int, default=64, help="candidates, as in the chain")
    ap.add_argument("--threshold", type=float, default=0.02)
    ap.add_argument("--pairs", nargs="*", help="only these pair names")
    ap.add_argument("--csv_dir", default="results/diff3f/debias_check")
    args = ap.parse_args()

    per_pair, res = [], {}
    for group, name, s, src_mesh, t, _tr, _ex in ef.pair_table(["same", "cross"]):
        if args.pairs and name not in args.pairs:
            continue
        kp = os.path.join(args.keypoints, name + ".json")
        if not os.path.exists(kp):
            kp = os.path.join(args.landmarks, name + ".json")
        src_npz = os.path.join(args.feature_dir, s + "_partial.npz")
        dst_npz = os.path.join(args.feature_dir, t + "_full.npz")
        if not (os.path.exists(kp) and os.path.exists(src_npz) and os.path.exists(dst_npz)):
            continue
        with open(kp) as f:
            d = json.load(f)
        lm_s, lm_t = np.asarray(d["source"], float), np.asarray(d["target"], float)
        src, dst = rp.load_features(src_npz), rp.load_features(dst_npz)
        Vs = np.asarray(load_mesh(src_mesh).vertices, dtype=np.float64)
        diag = float(np.linalg.norm(Vs.max(axis=0) - Vs.min(axis=0)))
        t_idx = cKDTree(dst["xyz"]).query(lm_t)[1]
        for v in args.variants:
            fs, fd = debias(src["feat"], dst["feat"], v)
            r = check_pair(fs, fd, src["xyz"], t_idx, lm_s, diag, args.k, args.threshold)
            res.setdefault(v, {})[(group, name)] = r
            row = dict(group=group, pair=name, variant=v)
            row.update({c: (round(float(np.mean(x)), 4) if isinstance(x, np.ndarray)
                            else (round(x, 4) if isinstance(x, float) else x))
                        for c, x in r.items()})
            per_pair.append(row)
            print(json.dumps(row), flush=True)

    summary = []
    base = res.get("none", {})
    for v in args.variants:
        rs = {p: r for (g, p), r in res[v].items() if g == "same"}
        pairs = sorted(rs)
        n = np.array([rs[p]["n"] for p in pairs])
        tot = float(n.sum())

        def share(key, rs=rs, pairs=pairs, tot=tot):
            return round(100.0 * sum(rs[p][key] for p in pairs) / tot, 1)

        def pooled(key, fn, rs=rs, pairs=pairs):
            return round(float(fn(np.concatenate([rs[p][key] for p in pairs]))), 4)

        wrong = sum(rs[p]["wrong"] for p in pairs)
        row = dict(
            variant=v, pairs=len(pairs), keypoints=int(tot),
            hits_nn=sum(rs[p]["hits_nn"] for p in pairs), pck2_nn=share("hits_nn"),
            mean_err_nn=pooled("err_nn", np.mean), diff_vs_none="", ci_low="", ci_high="",
            bound_64=share("bound"), rematch_truth=share("hits_rematch"),
            rematch_none=sum(rs[p]["rematch_none"] for p in pairs),
            rematch_median_err=pooled("err_rematch", np.median),
            peak_5=share("hits_peak5"), peak_5_median_err=pooled("err_peak5", np.median),
            peak_10=share("hits_peak10"), peak_10_median_err=pooled("err_peak10", np.median),
            hub_share=round(100.0 * float(np.mean([rs[p]["hub_share"] for p in pairs])), 1),
            wrong_on_hub=round(100.0 * sum(rs[p]["wrong_on_hub"] for p in pairs) / max(wrong, 1),
                               1),
        )
        if v != "none" and base:
            ha = np.array([base[("same", p)]["hits_nn"] for p in pairs])
            hb = np.array([rs[p]["hits_nn"] for p in pairs])
            diff, lo, hi = paired_interval(ha, hb, n)
            row.update(diff_vs_none=round(diff, 1), ci_low=round(lo, 1), ci_high=round(hi, 1))
        summary.append(row)

    os.makedirs(args.csv_dir, exist_ok=True)
    for fname, rows in (("summary.csv", summary), ("per_pair.csv", per_pair)):
        with open(os.path.join(args.csv_dir, fname), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    with open(os.path.join(args.csv_dir, "run.json"), "w") as f:
        json.dump(dict(commit=git_commit(), command=" ".join(sys.argv), variants=args.variants,
                       k=args.k, threshold=args.threshold), f, indent=1)
    cols = list(summary[0].keys())
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in summary:
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")


if __name__ == "__main__":
    main()
