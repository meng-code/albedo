"""
Baselines without any correspondence: paint the TARGET full mesh of every annotated pair with a
texture generator, using the SOURCE photo as the reference image (the "Hunyuan3D paint" and
"TRELLIS.2 texturing" rows of the baseline table). Our method instead transfers the photo's
albedo through a correspondence; this shows what a texture generator does with the same photo
and mesh.

--painter hunyuan3d: Hunyuan3D-2.1's paint model (multiview diffusion from the reference image
and the mesh's normal and position maps, baked into a texture). It treats its reference image as
the front view of the mesh (camera on +z, +y up, as for its own generated meshes).
--painter trellis2: TRELLIS.2's shape-conditioned texturing (microsoft/TRELLIS.2-4B,
texturing_pipeline.json): the mesh is encoded into TRELLIS.2's shape latent and the 1.3 B texture
flow model generates a PBR texture latent from the image (DINOv3 features) and that shape. Its
training objects stand in a canonical pose (front towards +z, +y up) and the image may be taken
from any side.

The target meshes have their own frames. --frame photo (the default) turns each target into the
photo's camera frame with the pair's registration (results/diff3f/<pair>/transform.json maps the
relief, which lives in the photo's camera frame, onto the target: target = s R relief + t, so
the target centred and multiplied by R^T is in the photo's frame; the cross-category pairs
borrow the registration of a same-category pair with the same target, as eval_features.py does).
--frame canonical (TRELLIS.2 only) rotates that further about +y by the azimuth at which the
photo's camera was found for TRELLIS.2's own mesh of the same photo
(~/semtex/gen3d/trellis2/<photo>/renders_view.json, gen3d_render.py), i.e. into the pose
TRELLIS.2 itself chose for that photo.
Both painters return a new mesh (Hunyuan3D remeshes to at most 40 k faces; both unwrap UVs), which
is turned back into the target's frame, and its distance to the original target surface is
recorded as a check.

Per pair, under --out_dir/<pair>/ (results, small): renders.png (the photo, the painted target
from the photo's camera, from behind) and paint.json (seconds, peak GPU memory, counts, the
check). Under --mesh_dir/<pair>/ (meshes): textured.glb and textured_vc.ply (vertex colours
sampled from the albedo texture) in the target's own frame, and the painted mesh in the painting
frame.

Runs on the GPU server: painting in the hunyuan3d or trellis2 environment (see gen3d_hunyuan.py,
gen3d_trellis.py), then the renders in the diff3f environment. Usage (repository root):
    LD_LIBRARY_PATH=~/miniconda3/envs/hunyuan3d/lib python scripts/idarb/gen3d_paint_baseline.py
    python scripts/idarb/gen3d_paint_baseline.py --renders_only
    python scripts/idarb/gen3d_paint_baseline.py --painter trellis2 \
        --out_dir results/diff3f/paint_baseline_trellis2 --mesh_dir ~/semtex/paint_baseline_trellis2
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gen3d_hunyuan import git_commit, texture_vertex_colors  # noqa: E402

# pair: (source photo, target, registration whose rotation gives the photo's frame)
PAIRS = {
    "cow": ("cow", "cow", "cow"),
    "cat": ("cat", "cat", "cat"),
    "superman_man": ("superman", "man", "superman_man"),
    "superman_posed": ("superman", "posed_human", "superman_posed"),
    "okapi_camel": ("okapi", "camel", "okapi_camel"),
    "okapi1_camel": ("okapi1", "camel", "okapi1_camel"),
    "cat5_cat": ("cat5", "cat", "cat5_cat"),
    "cat_lion": ("cat", "lion", "cat_lion"),
    "labubu_man": ("labubu", "man", "labubu_man"),
    "chair1_armchair": ("chair1", "armchair", "chair1_armchair"),
    "panda_panda": ("panda", "panda", "panda_panda"),
    "bird_bird": ("bird", "bird", "bird_bird"),
    "dragon_lizard": ("fire_dragon", "lizard", "dragon_lizard"),
    "labubu_camel": ("labubu", "camel", "okapi_camel"),
    "labubu_lion": ("labubu", "lion", "cat_lion"),
}


def target_mesh_path(target):
    from eval_features import TARGET_MESH

    return os.path.join(ROOT, TARGET_MESH[target])


def rot_y(deg):
    """Rotation about +y that takes +z to (sin a, 0, cos a)."""
    a = math.radians(deg)
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])


def hunyuan_painter(args):
    from gen3d_hunyuan import paint_pipeline

    os.chdir(args.repo)  # the paint code reads some of its files relative to the repository
    pipe = paint_pipeline(args.repo, args.max_num_view, args.resolution)

    def paint(mesh, photo, mdir):
        import trimesh

        path = os.path.join(mdir, "target_frame.obj")
        mesh.export(path)
        obj = pipe(mesh_path=path, image_path=photo,
                   output_mesh_path=os.path.join(mdir, "textured_frame.obj"), save_glb=False)
        return trimesh.load(obj, force="mesh")

    return paint


def trellis_painter(args):
    import torch
    from PIL import Image

    sys.path.insert(0, args.trellis_repo)
    import trellis2.models as tmodels
    from trellis2.pipelines import Trellis2TexturingPipeline, rembg

    from gen3d_trellis import NoRembg

    rembg.BiRefNet = NoRembg  # RGBA inputs skip background removal; the model is gated
    flow = f"tex_slat_flow_model_{args.trellis_resolution}"
    Trellis2TexturingPipeline.model_names_to_load = ["shape_slat_encoder", "tex_slat_decoder", flow]
    load_model = tmodels.from_pretrained

    def load_to_gpu(*a, **k):  # see gen3d_trellis.py: keeps the 16 GB of host RAM sufficient
        model = load_model(*a, **k).cuda()
        torch.cuda.empty_cache()
        return model

    tmodels.from_pretrained = load_to_gpu
    pipe = Trellis2TexturingPipeline.from_pretrained("microsoft/TRELLIS.2-4B",
                                                     config_file="texturing_pipeline.json")
    tmodels.from_pretrained = load_model
    pipe.low_vram = False
    pipe.cuda()

    def paint(mesh, photo, mdir):
        v = np.asarray(mesh.vertices, dtype=np.float64)
        # the pipeline centres the mesh and scales its longest side to 0.99999; undone below
        centre = (v.min(0) + v.max(0)) / 2
        scale = 0.99999 / (v.max(0) - v.min(0)).max()
        out = pipe.run(mesh, Image.open(photo).convert("RGBA"), seed=args.seed,
                       resolution=args.trellis_resolution, texture_size=args.texture_size)
        out.vertices = np.asarray(out.vertices) / scale + centre
        out.export(os.path.join(mdir, "textured_frame.glb"))
        return out

    return paint


def paint_all(args):
    import torch
    import trimesh

    painter = (hunyuan_painter if args.painter == "hunyuan3d" else trellis_painter)(args)
    for pair in args.pairs:
        src, target, reg = PAIRS[pair]
        mdir = os.path.join(args.mesh_dir, pair)
        rdir = os.path.join(args.out_dir, pair)
        os.makedirs(mdir, exist_ok=True)
        os.makedirs(rdir, exist_ok=True)
        with open(os.path.join(ROOT, "results", "diff3f", reg, "transform.json")) as f:
            R = np.array(json.load(f)["rotation"], dtype=np.float64)
        azimuth = 0.0
        if args.frame == "canonical":
            with open(os.path.join(args.gen3d_dir, "trellis2", src, "renders_view.json")) as f:
                azimuth = float(json.load(f)["azimuth"])
            R = R @ rot_y(azimuth).T  # photo frame -> TRELLIS.2's pose for this photo
        tm = trimesh.load(target_mesh_path(target), process=False, force="mesh")
        v = np.asarray(tm.vertices, dtype=np.float64)
        centre = (v.min(0) + v.max(0)) / 2
        framed = trimesh.Trimesh(vertices=(v - centre) @ R, faces=tm.faces, process=False)
        photo = os.path.join(args.photos, src + ".png")
        torch.manual_seed(args.seed)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        painted = painter(framed, photo, mdir)
        seconds = time.time() - t0
        trimesh.Trimesh(vertices=painted.vertices, faces=painted.faces,
                        vertex_colors=texture_vertex_colors(painted),
                        process=False).export(os.path.join(mdir, "textured_vc_frame.ply"))
        # back into the target's frame: target = R frame + centre
        back = trimesh.Trimesh(vertices=np.asarray(painted.vertices) @ R.T + centre,
                               faces=painted.faces, visual=painted.visual, process=False)
        back.export(os.path.join(mdir, "textured.glb"))
        trimesh.Trimesh(vertices=back.vertices, faces=back.faces,
                        vertex_colors=texture_vertex_colors(painted),
                        process=False).export(os.path.join(mdir, "textured_vc.ply"))
        # check: the painted mesh should lie on the target surface (the painters only remesh)
        step = max(1, len(back.vertices) // 5000)
        _, dist, _ = trimesh.proximity.closest_point(tm, back.vertices[::step])
        diag = float(np.linalg.norm(v.max(0) - v.min(0)))
        info = dict(pair=pair, painter=args.painter, frame=args.frame, frame_azimuth=azimuth,
                    source_photo=photo, target=target,
                    target_mesh=os.path.relpath(target_mesh_path(target), ROOT),
                    frame_from=f"results/diff3f/{reg}/transform.json", seed=args.seed,
                    seconds=round(seconds, 1),
                    peak_gpu_mib=round(torch.cuda.max_memory_allocated() / 2 ** 20),
                    n_vertices=int(len(back.vertices)), n_faces=int(len(back.faces)),
                    surface_distance_median=round(float(np.median(dist)) / diag, 6),
                    surface_distance_max=round(float(dist.max()) / diag, 6),
                    git_commit=git_commit(ROOT), command=" ".join(sys.argv))
        if args.painter == "hunyuan3d":
            info.update(max_num_view=args.max_num_view, resolution=args.resolution,
                        repo_commit=git_commit(args.repo))
        else:
            info.update(trellis_resolution=args.trellis_resolution,
                        texture_size=args.texture_size, repo_commit=git_commit(args.trellis_repo))
        with open(os.path.join(rdir, "paint.json"), "w") as f:
            json.dump(info, f, indent=1)
        print(json.dumps(info), flush=True)


def renders(args):
    """photo | painted target from the photo's camera | from behind. The painted mesh is put in
    the photo's frame (the painting frame, undoing --frame canonical's extra rotation), where the
    photo's camera is azimuth 0 (gen3d_render.py --no_fit_view)."""
    import trimesh

    for pair in args.pairs:
        src, _, _ = PAIRS[pair]
        mdir = os.path.join(args.mesh_dir, pair)
        with open(os.path.join(args.out_dir, pair, "paint.json")) as f:
            azimuth = float(json.load(f).get("frame_azimuth", 0.0))
        m = trimesh.load(os.path.join(mdir, "textured_vc_frame.ply"), process=False)
        ply = os.path.join(mdir, "textured_vc_photo_frame.ply")
        trimesh.Trimesh(vertices=np.asarray(m.vertices) @ rot_y(azimuth), faces=m.faces,
                        vertex_colors=m.visual.vertex_colors, process=False).export(ply)
        subprocess.run([args.render_python, os.path.join(ROOT, "scripts", "idarb",
                                                          "gen3d_render.py"),
                        "--mesh", ply, "--image", os.path.join(args.photos, src + ".png"),
                        "--out", os.path.join(args.out_dir, pair, "renders.png"),
                        "--no_fit_view"], check=True, capture_output=True)
        print("rendered", pair, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--painter", default="hunyuan3d", choices=["hunyuan3d", "trellis2"])
    ap.add_argument("--frame", default="photo", choices=["photo", "canonical"])
    ap.add_argument("--pairs", nargs="+", default=list(PAIRS))
    ap.add_argument("--photos", default=os.path.expanduser("~/IDArb/mydata/my_single"))
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "results", "diff3f", "paint_baseline"))
    ap.add_argument("--mesh_dir", default=os.path.expanduser("~/semtex/paint_baseline"))
    ap.add_argument("--gen3d_dir", default=os.path.expanduser("~/semtex/gen3d"))
    ap.add_argument("--repo", default=os.path.expanduser("~/semtex/Hunyuan3D-2.1"))
    ap.add_argument("--trellis_repo", default=os.path.expanduser("~/semtex/TRELLIS.2"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_num_view", type=int, default=6, help="hunyuan3d paint views")
    ap.add_argument("--resolution", type=int, default=512, help="hunyuan3d paint view size")
    ap.add_argument("--trellis_resolution", type=int, default=1024, choices=[512, 1024])
    ap.add_argument("--texture_size", type=int, default=2048, help="trellis2 texture")
    ap.add_argument("--renders_only", action="store_true",
                    help="only the renders (diff3f environment), after the painting")
    ap.add_argument("--render_python", default=sys.executable)
    args = ap.parse_args()
    if args.frame == "canonical" and args.painter != "trellis2":
        ap.error("--frame canonical is for --painter trellis2")
    args.out_dir, args.mesh_dir = os.path.abspath(args.out_dir), os.path.abspath(args.mesh_dir)
    (renders if args.renders_only else paint_all)(args)


if __name__ == "__main__":
    main()
