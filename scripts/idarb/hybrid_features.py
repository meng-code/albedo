"""
Hybrid source features for "generate, then match": the generated mesh keeps its geometry and,
on the side the photo never saw, its own Diff3F features; every feature point the photo's camera
sees takes instead the feature of the relief point it lands on (through align_generated.py's
transform). This separates the generator's geometry from its features: if the hybrid source
matches better than the plain generated one, what "generate, then match" loses is the generated
mesh's features, not its shape or the full coverage.

Usage (repository root, mesh environment), one model, every source relief that has an alignment:
    python scripts/idarb/hybrid_features.py --model hunyuan3d \
        --gen_features outputs/diff3f/features_zoom2_gen \
        --feature_dir outputs/diff3f/features_zoom2 \
        --aligned outputs/diff3f/13_generated \
        --out outputs/diff3f/features_zoom2_gen_hybrid
The output folder has the layout eval_generated.py reads (<model>/<name>.npz, <name>_mesh.ply).
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_features as ef  # noqa: E402
from align_generated import front_depth  # noqa: E402
from correspondence_maps import load_mesh  # noqa: E402


def relief_frame(xyz_cam, tr):
    """Points of the generated mesh in the photo frame into the relief's frame: the inverse of
    the fit relief -> generated (scale, rotation, translation, depth factor)."""
    s, a = tr["scale"], tr["depth_factor"]
    R, t = np.array(tr["rotation"]), np.array(tr["translation"])
    q = ((xyz_cam - t) / s) @ R
    q[:, 2] /= a
    return q


def hybrid(S_xyz, S_feat, Vg, Fg, tr, R_xyz, R_feat, max_dist=0.03):
    """The generated features with every camera-visible point's feature replaced by its nearest
    relief feature (within max_dist of the relief diagonal). Returns the features and the mask
    of replaced points."""
    Rc = np.array(tr["camera_rotation"])
    V_cam = Vg @ Rc.T
    x_cam = S_xyz @ Rc.T
    diag = float(np.linalg.norm(V_cam.max(axis=0) - V_cam.min(axis=0)))
    seen = front_depth(V_cam, Fg, x_cam[:, :2])
    visible = np.isfinite(seen) & (np.abs(seen - x_cam[:, 2]) < 0.01 * diag)
    q = relief_frame(x_cam, tr)
    d, j = cKDTree(R_xyz).query(q)
    diag_r = float(np.linalg.norm(R_xyz.max(axis=0) - R_xyz.min(axis=0)))
    take = visible & (d < max_dist * diag_r)
    feat = S_feat.copy()
    feat[take] = R_feat[j[take]]
    return feat, take


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", required=True, choices=["hunyuan3d", "trellis2"])
    ap.add_argument("--gen_features", default="outputs/diff3f/features_zoom2_gen")
    ap.add_argument("--feature_dir", default="outputs/diff3f/features_zoom2")
    ap.add_argument("--aligned", default="outputs/diff3f/13_generated")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_dist", type=float, default=0.03,
                    help="largest relief distance for a replacement, fraction of its diagonal")
    ap.add_argument("--names", nargs="*", help="source stems; default every pair's source")
    args = ap.parse_args()
    names = args.names or sorted({r[2] for r in ef.pair_table(["same", "cross"])})
    out_dir = os.path.join(args.out, args.model)
    os.makedirs(out_dir, exist_ok=True)
    for name in names:
        gen_npz = os.path.join(args.gen_features, args.model, name + ".npz")
        gen_mesh = os.path.join(args.gen_features, args.model, name + "_mesh.ply")
        rel_npz = os.path.join(args.feature_dir, name + "_partial.npz")
        tr_path = os.path.join(args.aligned, args.model, name, "transform.json")
        if not all(os.path.exists(p) for p in (gen_npz, gen_mesh, rel_npz, tr_path)):
            print(json.dumps(dict(name=name, skipped="missing input")), flush=True)
            continue
        zg = np.load(gen_npz, allow_pickle=False)
        zr = np.load(rel_npz, allow_pickle=False)
        with open(tr_path) as f:
            tr = json.load(f)
        m = load_mesh(gen_mesh)
        feat, take = hybrid(
            zg["xyz"].astype(np.float64), zg["feat"], np.asarray(m.vertices, dtype=np.float64),
            np.asarray(m.faces), tr, zr["xyz"].astype(np.float64), zr["feat"], args.max_dist,
        )
        meta = json.loads(str(zg["meta"])) if "meta" in zg.files else {}
        meta["hybrid"] = dict(relief_features=rel_npz, transform=tr_path,
                              max_dist=args.max_dist, replaced=int(take.sum()),
                              replaced_share=round(float(take.mean()), 4))
        np.savez(os.path.join(out_dir, name + ".npz"), idx=zg["idx"], xyz=zg["xyz"],
                 feat=feat.astype(zg["feat"].dtype), colors=zg["colors"],
                 meta=np.array(json.dumps(meta)))
        shutil.copyfile(gen_mesh, os.path.join(out_dir, name + "_mesh.ply"))
        print(json.dumps(dict(name=name, points=int(len(take)), replaced=int(take.sum()),
                              replaced_share=round(float(take.mean()), 4))), flush=True)


if __name__ == "__main__":
    main()
