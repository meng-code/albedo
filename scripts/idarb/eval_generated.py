"""
The "generate, then match" pipeline over the annotated pairs: the source is the full mesh
generated from the photo (Hunyuan3D-2.1 or TRELLIS.2) with the photo's albedo baked into its
texture atlas, the target the usual full mesh. Runs refine_correspondence.py (recommended
setting, anchor camera), transfer_texture.py with --source_glb, uv_metrics.py and the keypoint
accuracy (eval_keypoints.py's measure, with the keypoints of gen_keypoints.py), one row per pair
in a CSV, plus the group means.

Needs: zoom-2 Diff3F features of the generated meshes in --gen_features/<model>/<name>.npz
(sample points on textured_vc.ply, in the generated mesh's own frame), the target features in
--feature_dir, align_generated.py and bake_albedo.py outputs under --aligned/<model>/<name>/.

Usage (repository root, mesh environment):
    python scripts/idarb/eval_generated.py --model hunyuan3d \
        --gen_features outputs/diff3f/features_zoom2_gen \
        --feature_dir outputs/diff3f/features_zoom2 \
        --out_dir outputs/diff3f/14_generated_match/hunyuan3d \
        --csv results/diff3f/generated_hunyuan3d.csv
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_features as ef  # noqa: E402
from correspondence_maps import load_mesh  # noqa: E402
from eval_keypoints import landmark_errors, summary  # noqa: E402

REFINE_ARGS = ["--field", "0.1", "--field_order", "2", "--arap", "10", "--arap_weight", "3",
               "--w_view", "0.3", "--anchor_filter", "rigid:0.1", "--view_from_anchors"]
FIELDS = ["group", "pair", "model", "textured_faces", "within_2pct", "within_5pct",
          "source_used", "conformal_median", "distortion_over_10", "flipped", "flipped_area",
          "mutual_anchors", "kp_n", "kp_mean", "kp_pck_2", "kp_pck_5", "kp_pck_10",
          "kp_nn_pck_2", "merged_front_faces", "merged_back_faces", "relief_kp_mean",
          "relief_kp_pck_2", "relief_kp_pck_5", "seconds"]


def merge_pair(args, env, py, group, name, src, target, dst_npz, src_npz, src_mesh, src_glb,
               refined, pair_dir, Vd):
    """The two-sided texture of one pair: the relief correspondence (computed with the same
    setting when missing) for the seen side, the generated-mesh correspondence for the rest.
    Returns the CSV columns: face counts and the relief correspondence's keypoint accuracy."""
    relief_npz = os.path.join(args.feature_dir, f"{src}_partial.npz")
    relief_mesh = f"outputs/mesh_depth/{src}/relief.ply"
    r_dir = os.path.join(args.relief_dir, group, name)
    os.makedirs(r_dir, exist_ok=True)
    r_refined = os.path.join(r_dir, "refined.npz")
    if not os.path.exists(r_refined):
        cmd = [py, os.path.join(ef.SCRIPTS, "refine_correspondence.py"), "--source", relief_npz,
               "--source_mesh", relief_mesh, "--target", dst_npz, "--target_mesh",
               ef.TARGET_MESH[target]] + REFINE_ARGS + ["--out", r_refined]
        ef.run(cmd, env)
    cmd = [py, os.path.join(ef.SCRIPTS, "merge_sides.py"), "--target_mesh",
           ef.TARGET_MESH[target], "--front_source", relief_npz, "--front_mesh", relief_mesh,
           "--front_correspondence", r_refined, "--back_source", src_npz, "--back_mesh",
           src_mesh, "--back_correspondence", refined, "--back_glb", src_glb, "--out_dir",
           os.path.join(pair_dir, "merged")]
    m = ef.last_json(ef.run(cmd, env))
    out = dict(merged_front_faces=m.get("front_faces"), merged_back_faces=m.get("back_faces"))
    kp_path = os.path.join(args.relief_keypoints, name + ".json")
    if not os.path.exists(kp_path):
        kp_path = os.path.join(args.landmarks, name + ".json")
    if os.path.exists(kp_path):
        with open(kp_path) as f:
            lm = json.load(f)
        Vr = np.asarray(load_mesh(relief_mesh).vertices, dtype=np.float64)
        z = np.load(r_refined)
        P = z["src_pos"].astype(np.float64) if "src_pos" in z.files else Vr[z["t2s"]]
        diag_r = float(np.linalg.norm(Vr.max(axis=0) - Vr.min(axis=0)))
        s = summary(landmark_errors(P, Vd, np.asarray(lm["source"], dtype=np.float64),
                                    np.asarray(lm["target"], dtype=np.float64), diag_r))
        out.update(relief_kp_mean=s["mean"], relief_kp_pck_2=s["pck_2"],
                   relief_kp_pck_5=s["pck_5"])
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", required=True, choices=["hunyuan3d", "trellis2"])
    ap.add_argument("--gen_features", default="outputs/diff3f/features_zoom2_gen")
    ap.add_argument("--feature_dir", default="outputs/diff3f/features_zoom2")
    ap.add_argument("--aligned", default="outputs/diff3f/13_generated")
    ap.add_argument("--gen_meshes", default="outputs/diff3f/gen3d")
    ap.add_argument("--keypoints", default="results/diff3f/keypoints_gen")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--pairs", nargs="*")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--refine_args", default="")
    ap.add_argument("--merge", action="store_true",
                    help="also texture the side the photo never saw from the generated mesh: "
                         "the relief correspondence (same setting) for the seen side, "
                         "merge_sides.py for the rest; adds the relief's keypoint accuracy")
    ap.add_argument("--relief_dir", default="outputs/diff3f/15_relief_zoom2",
                    help="the relief correspondences for --merge, computed when missing")
    ap.add_argument("--relief_keypoints", default="results/diff3f/keypoints")
    ap.add_argument("--landmarks", default="results/diff3f/landmarks")
    args = ap.parse_args()
    if args.refine_args:
        REFINE_ARGS.extend(args.refine_args.split())
    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[var] = str(args.threads)
    py = sys.executable
    rows = []
    for group, name, src, _rm, target, _tr, _e in ef.pair_table(["same", "cross"]):
        if args.pairs and name not in args.pairs:
            continue
        kp_path = os.path.join(args.keypoints, args.model, name + ".json")
        src_npz = os.path.join(args.gen_features, args.model, src + ".npz")
        # the features were computed on a welded (and, for TRELLIS.2, decimated) copy pushed
        # next to them; fall back to the GLB-derived mesh
        src_mesh = os.path.join(args.gen_features, args.model, src + "_mesh.ply")
        if not os.path.exists(src_mesh):
            src_mesh = os.path.join(args.gen_meshes, args.model, src, "textured_vc.ply")
        src_glb = os.path.join(args.aligned, args.model, src, "baked.glb")
        dst_npz = os.path.join(args.feature_dir, f"{target}_full.npz")
        if not all(os.path.exists(p) for p in (kp_path, src_npz, src_mesh, src_glb, dst_npz)):
            print(json.dumps(dict(pair=name, skipped="missing input")), flush=True)
            continue
        pair_dir = os.path.join(args.out_dir, group, name)
        os.makedirs(pair_dir, exist_ok=True)
        common = ["--source", src_npz, "--source_mesh", src_mesh, "--target", dst_npz,
                  "--target_mesh", ef.TARGET_MESH[target]]
        t0 = time.time()
        refined = os.path.join(pair_dir, "refined.npz")
        cmd = [py, os.path.join(ef.SCRIPTS, "refine_correspondence.py")] + common + REFINE_ARGS
        cmd += ["--out", refined]
        stats = ef.last_json(ef.run(cmd, env))
        anchors = stats.get("anchors", "")
        mutual = int(anchors.split("+")[-1].split()[0]) if "mutual" in anchors else 0
        view = np.load(refined)["view"]
        view_str = [f"{v:.6f}" for v in view]
        cmd = [py, os.path.join(ef.SCRIPTS, "transfer_texture.py")] + common
        cmd += ["--source_glb", src_glb, "--target_view"] + view_str
        cmd += ["--correspondence", refined, "--out_dir", pair_dir]
        tex = ef.last_json(ef.run(cmd, env))
        cmd = [py, os.path.join(ef.SCRIPTS, "uv_metrics.py"), "--obj",
               os.path.join(pair_dir, "target_textured.obj"), "--view"] + view_str
        q = ef.last_json(ef.run(cmd, env))
        # keypoints: the field positions of the target's keypoint vertices against the source
        # keypoints on the generated mesh (the relief's keypoints carried over)
        with open(kp_path) as f:
            lm = json.load(f)
        lm_s = np.asarray(lm["source"], dtype=np.float64)
        lm_t = np.asarray(lm["target"], dtype=np.float64)
        Vs = np.asarray(load_mesh(src_mesh).vertices, dtype=np.float64)
        Vd = np.asarray(load_mesh(ef.TARGET_MESH[target]).vertices, dtype=np.float64)
        diag_s = float(np.linalg.norm(Vs.max(axis=0) - Vs.min(axis=0)))
        z = np.load(refined)
        P = z["src_pos"].astype(np.float64) if "src_pos" in z.files else Vs[z["t2s"]]
        err = landmark_errors(P, Vd, lm_s, lm_t, diag_s)
        s_kp = summary(err)
        # nearest neighbour on the same keypoints, for reference
        import register_partial as rp
        from correspondence_maps import mesh_features, nearest_in_features

        S = rp.load_features(src_npz)
        _, _, fd, _ = mesh_features(dst_npz, ef.TARGET_MESH[target])
        nn = S["idx"][nearest_in_features(fd, S["feat"])]
        s_nn = summary(landmark_errors(Vs[nn], Vd, lm_s, lm_t, diag_s))
        merged = {}
        if args.merge:
            merged = merge_pair(args, env, py, group, name, src, target, dst_npz, src_npz,
                                src_mesh, src_glb, refined, pair_dir, Vd)
        rows.append(dict(
            group=group, pair=name, model=args.model, textured_faces=tex.get("textured_faces"),
            within_2pct=tex.get("within_2pct"), within_5pct=tex.get("within_5pct"),
            source_used=tex.get("source_used"), conformal_median=q.get("conformal_median"),
            distortion_over_10=q.get("distortion_over_10"), flipped=q.get("flipped"),
            flipped_area=q.get("flipped_area"), mutual_anchors=mutual, kp_n=len(err),
            kp_mean=s_kp["mean"], kp_pck_2=s_kp["pck_2"], kp_pck_5=s_kp["pck_5"],
            kp_pck_10=s_kp["pck_10"], kp_nn_pck_2=s_nn["pck_2"],
            seconds=round(time.time() - t0, 1), **merged,
        ))
        print(json.dumps(rows[-1]), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.csv)) or ".", exist_ok=True)
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    for group in ("same", "cross"):
        rs = [r for r in rows if r["group"] == group]
        if not rs:
            continue
        cols = ["kp_pck_2", "kp_nn_pck_2", "kp_mean", "within_2pct", "within_5pct",
                "source_used", "conformal_median", "flipped_area"]
        if args.merge:
            cols += ["relief_kp_pck_2", "merged_back_faces"]
        mean = {c: round(float(np.mean([r[c] for r in rs if r.get(c) is not None])), 3)
                for c in cols}
        print(f"{group}: pairs {len(rs)}  kp_pck_2 {mean['kp_pck_2']} (nn "
              f"{mean['kp_nn_pck_2']})  kp_mean {mean['kp_mean']}  within_2pct "
              f"{mean['within_2pct']}  within_5pct {mean['within_5pct']}  source_used "
              f"{mean['source_used']}  distortion {mean['conformal_median']}  flipped_area "
              f"{mean['flipped_area']}"
              + (f"  relief_kp_pck_2 {mean['relief_kp_pck_2']}  merged_back_faces "
                 f"{mean['merged_back_faces']}" if args.merge else ""))


if __name__ == "__main__":
    main()
