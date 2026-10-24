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
          "kp_nn_pck_2", "seconds"]


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
        rows.append(dict(
            group=group, pair=name, model=args.model, textured_faces=tex.get("textured_faces"),
            within_2pct=tex.get("within_2pct"), within_5pct=tex.get("within_5pct"),
            source_used=tex.get("source_used"), conformal_median=q.get("conformal_median"),
            distortion_over_10=q.get("distortion_over_10"), flipped=q.get("flipped"),
            flipped_area=q.get("flipped_area"), mutual_anchors=mutual, kp_n=len(err),
            kp_mean=s_kp["mean"], kp_pck_2=s_kp["pck_2"], kp_pck_5=s_kp["pck_5"],
            kp_pck_10=s_kp["pck_10"], kp_nn_pck_2=s_nn["pck_2"],
            seconds=round(time.time() - t0, 1),
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
        mean = {c: round(float(np.mean([r[c] for r in rs if r[c] is not None])), 3)
                for c in ("kp_pck_2", "kp_nn_pck_2", "kp_mean", "within_2pct", "within_5pct",
                          "source_used", "conformal_median", "flipped_area")}
        print(f"{group}: pairs {len(rs)}  kp_pck_2 {mean['kp_pck_2']} (nn "
              f"{mean['kp_nn_pck_2']})  kp_mean {mean['kp_mean']}  within_2pct "
              f"{mean['within_2pct']}  within_5pct {mean['within_5pct']}  source_used "
              f"{mean['source_used']}  distortion {mean['conformal_median']}  flipped_area "
              f"{mean['flipped_area']}")


if __name__ == "__main__":
    main()
