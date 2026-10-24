"""
One command to judge a set of semantic features on every pair: nearest-neighbour and ARAP-field
texture transfer, the texture-map quality numbers and, on the synthetic pairs, the ground-truth
accuracy. Run it once per feature set (Diff3F, MeshFM, PartField, ...) and compare the CSVs.

The feature files must follow the Diff3F naming (<source>_partial.npz, <target>_full.npz) and
format (idx, xyz, feat, colors) in --feature_dir; the meshes are the ones in the repository and in
outputs/mesh_depth. For every pair the script runs, in the mesh environment,

  refine_correspondence.py   mutual anchors, prior, smoothing, smooth field and ARAP field
                             (the settings of outputs/diff3f/07_texture_arap), with --gt on the
                             synthetic pairs
  transfer_texture.py        the nearest-neighbour map and the ARAP field, textured side only
  uv_metrics.py              conformal distortion and flips of both texture maps

and writes <out_dir>/<group>/<pair>/{nn,arap}/ plus one CSV row per pair and variant (triangles
within 2 % / 5 %, relief used, distortion, flips, number of mutual anchors) and a summary with the
group means. The synthetic pairs have no albedo, so their rows hold only the ground-truth accuracy
(target vertices within 1 % / 5 % of the diagonal of their true source vertex) of the nearest
neighbour and of the ARAP field. The target's viewing direction comes from the stored
registration of the pair (results/diff3f/<pair>/transform.json) or, for a cross-category pair,
from a same-target registration, exactly as in the stored results.

Usage (repository root, mesh environment):
    python scripts/idarb/eval_features.py --feature_dir outputs/diff3f/features \
        --out_dir outputs/diff3f/08_features_diff3f --csv results/diff3f/features_diff3f.csv
    python scripts/idarb/eval_features.py --feature_dir outputs/diff3f/features_meshfm \
        --out_dir outputs/diff3f/08_features_meshfm --csv results/diff3f/features_meshfm.csv
    --groups same cross synthetic (default all), --pairs cat_lion cow (a subset),
    --jobs 8 --threads 2 on a machine with many cores (pairs run side by side)
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS = os.path.join(ROOT, "scripts", "idarb")

TARGET_MESH = {
    "cow": "third_party/Diff3F/meshes/cow.obj",
    "cat": "third_party/Diff3F/meshes/cat.off",
    "man": "third_party/Diff3F/meshes/man.obj",
    "posed_human": "third_party/Diff3F/meshes/posed_human.off",
    "camel": "third_party/Diff3F/meshes/camel.obj",
    "lion": "third_party/Diff3F/meshes/lion.off",
    "armchair": "results/diff3f/targets/armchair.ply",
    "roundchair": "results/diff3f/targets/roundchair.ply",
    "panda": "results/diff3f/targets/panda.ply",
    "bird": "results/diff3f/targets/bird.ply",
    "lizard": "results/diff3f/targets/lizard.ply",
}

# same-category pairs: name, source relief, target; each has results/diff3f/<name>/transform.json
SAME = [
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
# cross-category pairs: no registration of their own; the target view is borrowed from a
# same-category registration of the same target
CROSS = [
    ("labubu_camel", "labubu", "camel"),
    ("labubu_cow", "labubu", "cow"),
    ("labubu_lion", "labubu", "lion"),
    ("superman_camel", "superman", "camel"),
    ("panda_man", "panda", "man"),
    ("cat_man", "cat", "man"),
    ("cow_camel", "cow", "camel"),
    ("fire_dragon_man", "fire_dragon", "man"),
    ("bird_cat", "bird", "cat"),
    ("chair1_cow", "chair1", "cow"),
]
VIEW_FROM = {"camel": "okapi_camel", "cow": "cow", "lion": "cat_lion", "man": "superman_man",
             "cat": "cat"}
# synthetic pairs: the partial mesh is cut from the target itself, so the truth is known
# (name, source stem of the mesh and feature file, target)
SYNTH = [("cat_synth", "cat_synth", "cat"), ("cow_synth", "cow_synth", "cow")]

REFINE_ARGS = ["--field", "0.1", "--field_order", "2", "--arap", "10", "--arap_weight", "3"]
CSV_FIELDS = [
    "group", "pair", "variant", "textured_faces", "within_2pct", "within_5pct", "source_used",
    "conformal_median", "distortion_over_10", "flipped", "flipped_area", "mutual_anchors",
    "gt_acc_1pct", "gt_acc_5pct", "seconds",
]


def view_of(transform_path):
    """The target's viewing direction stored in a registration: the relief's camera axis."""
    with open(transform_path) as f:
        R = np.array(json.load(f)["rotation"], dtype=np.float64)
    return R[:, 2]


def pair_table(groups):
    rows = []
    if "same" in groups:
        rows += [("same", n, s, f"outputs/mesh_depth/{s}/relief.ply", t,
                  f"results/diff3f/{n}/transform.json", None) for n, s, t in SAME]
    if "cross" in groups:
        rows += [("cross", n, s, f"outputs/mesh_depth/{s}/relief.ply", t, None,
                  f"results/diff3f/{VIEW_FROM[t]}/transform.json") for n, s, t in CROSS]
    if "synthetic" in groups:
        rows += [("synthetic", n, s, f"results/diff3f/synthetic/{s}.ply", t,
                  f"results/diff3f/{n}/transform.json", f"results/diff3f/synthetic/{s}_gt.npz")
                 for n, s, t in SYNTH]
    return rows


def run(cmd, env):
    out = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    if out.returncode != 0:
        sys.stderr.write(out.stdout + out.stderr)
        raise RuntimeError("failed: " + " ".join(cmd))
    return out.stdout


def last_json(text):
    for line in reversed(text.strip().splitlines()):
        if line.startswith("{"):
            return json.loads(line.replace("'", '"')) if "'" in line else json.loads(line)
    return {}


def evaluate_pair(args, env, group, name, src, src_mesh, target, transform, extra):
    src_npz = os.path.join(args.feature_dir, f"{src}_partial.npz")
    dst_npz = os.path.join(args.feature_dir, f"{target}_full.npz")
    dst_mesh = TARGET_MESH[target]
    pair_dir = os.path.join(args.out_dir, group, name)
    os.makedirs(pair_dir, exist_ok=True)
    py = sys.executable
    common = ["--source", src_npz, "--source_mesh", src_mesh, "--target", dst_npz,
              "--target_mesh", dst_mesh]
    gt = extra if group == "synthetic" else None
    if transform:
        view_args = ["--transform", transform]
        view = view_of(transform)
    else:
        view = view_of(extra)
        view_args = ["--target_view"] + [f"{v:.6f}" for v in view]
    view_str = [f"{v:.6f}" for v in view]

    t0 = time.time()
    refined = os.path.join(pair_dir, "refined.npz")
    cmd = [py, os.path.join(SCRIPTS, "refine_correspondence.py")] + common + REFINE_ARGS
    if args.view == "anchors":
        cmd += ["--view_from_anchors"]
    else:
        cmd += ["--target_view"] + view_str
    cmd += ["--out", refined] + (["--gt", gt] if gt else [])
    stats = last_json(run(cmd, env))
    if args.view == "anchors" and not gt:
        view = np.load(refined)["view"]
        view_str = [f"{v:.6f}" for v in view]
        view_args = ["--target_view"] + view_str
    anchors = stats.get("anchors", "")
    mutual = int(anchors.split("+")[-1].split()[0]) if "mutual" in anchors else 0
    t_refine = time.time() - t0

    rows = []
    if gt:
        # the synthetic meshes carry no albedo, so there is no texture to transfer: the row holds
        # the ground-truth accuracy of the nearest neighbour and of the ARAP field (snapped back
        # to the sampled source points)
        for variant, key in (("nn", "nearest_neighbour"), ("arap", "field_snapped")):
            acc = stats.get(key, {})
            rows.append(dict(
                {c: None for c in CSV_FIELDS}, group=group, pair=name, variant=variant,
                mutual_anchors=mutual, gt_acc_1pct=acc.get("acc_1pct"),
                gt_acc_5pct=acc.get("acc_5pct"), seconds=round(t_refine, 1),
            ))
            print(json.dumps(rows[-1]), flush=True)
        return rows
    for variant in args.variants:
        t1 = time.time()
        out_dir = os.path.join(pair_dir, variant)
        cmd = [py, os.path.join(SCRIPTS, "transfer_texture.py")] + common + view_args
        cmd += ["--out_dir", out_dir]
        if variant == "arap":
            cmd += ["--correspondence", refined]
        tex = last_json(run(cmd, env))
        cmd = [py, os.path.join(SCRIPTS, "uv_metrics.py"), "--obj",
               os.path.join(out_dir, "target_textured.obj"), "--view"] + view_str
        q = last_json(run(cmd, env))
        rows.append(dict(
            group=group, pair=name, variant=variant,
            textured_faces=tex.get("textured_faces"), within_2pct=tex.get("within_2pct"),
            within_5pct=tex.get("within_5pct"), source_used=tex.get("source_used"),
            conformal_median=q.get("conformal_median"),
            distortion_over_10=q.get("distortion_over_10"), flipped=q.get("flipped"),
            flipped_area=q.get("flipped_area"), mutual_anchors=mutual,
            gt_acc_1pct=None, gt_acc_5pct=None,
            seconds=round(time.time() - t1 + (t_refine if variant == "arap" else 0.0), 1),
        ))
        print(json.dumps(rows[-1]), flush=True)
    return rows


def summarise(rows):
    """Mean of every numeric column per group and variant."""
    out = {}
    for r in rows:
        key = f"{r['group']}/{r['variant']}"
        out.setdefault(key, []).append(r)
    summary = {}
    for key, rs in out.items():
        s = dict(pairs=len(rs))
        for col in CSV_FIELDS[3:]:
            vals = [r[col] for r in rs if r[col] is not None]
            if vals:
                s[col] = round(float(np.mean(vals)), 4)
        summary[key] = s
    return summary


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--feature_dir", default="outputs/diff3f/features")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--csv", required=True, help="one row per pair and variant")
    ap.add_argument("--groups", nargs="+", default=["same", "cross", "synthetic"],
                    choices=["same", "cross", "synthetic"])
    ap.add_argument("--pairs", nargs="*", help="only these pair names")
    ap.add_argument("--variants", nargs="+", default=["nn", "arap"], choices=["nn", "arap"])
    ap.add_argument(
        "--view",
        choices=["stored", "anchors"],
        default="stored",
        help="texture camera: the stored registration (cross pairs borrow one) or each pair's "
        "own, from its anchors' similarity transform",
    )
    ap.add_argument("--threads", type=int, default=3, help="BLAS / OpenMP threads per step")
    ap.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="pairs evaluated at the same time (each with --threads threads); the per-pair "
        "solves are mostly single-threaded, so parallel pairs scale, more threads do not",
    )
    ap.add_argument(
        "--refine_args",
        default="",
        help="extra arguments for refine_correspondence.py, one string, for example "
        "'--anchor_filter rigid:0.1' (the ARAP settings stay)",
    )
    args = ap.parse_args()
    if args.refine_args:
        REFINE_ARGS.extend(args.refine_args.split())

    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[var] = str(args.threads)
    table = pair_table(args.groups)
    if args.pairs:
        table = [row for row in table if row[1] in args.pairs]
    def one(row):
        group, name, src, src_mesh, target, transform, extra = row
        try:
            return evaluate_pair(args, env, group, name, src, src_mesh, target, transform, extra)
        except (RuntimeError, FileNotFoundError, KeyError) as e:
            print(json.dumps(dict(group=group, pair=name, error=str(e)[:300])), flush=True)
            return []

    rows = []
    if args.jobs > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            for result in pool.map(one, table):  # table order is kept
                rows += result
    else:
        for row in table:
            rows += one(row)
    os.makedirs(os.path.dirname(os.path.abspath(args.csv)) or ".", exist_ok=True)
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    summary = dict(feature_dir=args.feature_dir, refine_args=REFINE_ARGS, groups=summarise(rows))
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary["groups"], indent=1))


if __name__ == "__main__":
    main()
