"""
Diff3F semantic features for the vertices of a mesh. Runs on the GPU server in the diff3f
environment (scripts/idarb/requirements-diff3f.txt); the method code is third_party/Diff3F.

Diff3F (Dutt, Muralikrishnan and Mitra, CVPR 2024) renders the untextured mesh from many camera
views as a depth map and a normal map, lets Stable Diffusion 1.5 with two ControlNets paint a
textured image from them, takes the diffusion UNet features (1280 numbers per pixel) and DINOv2
features (768 numbers per pixel) of that image, projects them back onto the vertices and averages
over the views. The result is one 2048-number vector per vertex that describes what the vertex is
(an ear, a front leg, a chair seat) rather than its shape, so two different meshes of the same kind
of object can be matched by comparing vectors.

Features are computed for at most --max_vertices vertices, chosen at random with --seed, so a
90k-vertex relief mesh from normal_to_mesh.py stays tractable (the whole mesh is still rendered).
--front_only limits the cameras to the hemisphere in front of the mesh (+z, the camera side of a
relief mesh), so a one-sided partial mesh is never rendered from behind.

Output: an .npz with idx (vertex indices), xyz (their positions in mesh units), feat (float16,
N x 2048, unit length), colors (uint8 vertex colours if the mesh has them, else empty) and meta
(a JSON string with the command, git commit, prompt and view set).

Usage (repository root, diff3f environment):
    python scripts/idarb/diff3f_features.py --mesh outputs/mesh_depth/cow/relief.ply --prompt cow \
        --front_only --out outputs/diff3f/features/cow_partial.npz
    python scripts/idarb/diff3f_features.py --mesh third_party/Diff3F/meshes/cow.obj --prompt cow \
        --out outputs/diff3f/features/cow_full.npz
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
DIFF3F_DIR = os.path.join(ROOT, "third_party", "Diff3F")
sys.path.insert(0, DIFF3F_DIR)

import torch  # noqa: E402
import trimesh  # noqa: E402

# runwayml/stable-diffusion-v1-5 (the repository Diff3F uses) was removed from Hugging Face in
# 2024; this is the community mirror of the same weights.
SD_REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CONTROLNET_DEPTH = "lllyasviel/control_v11f1p_sd15_depth"
CONTROLNET_NORMAL = "lllyasviel/control_v11p_sd15_normalbae"


def git_commit():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def load_mesh(path):
    """Vertices (float32), faces (int64) and uint8 vertex colours (or None) of a mesh file."""
    m = trimesh.load(path, process=False, force="mesh")
    verts = np.asarray(m.vertices, dtype=np.float32)
    faces = np.asarray(m.faces, dtype=np.int64)
    colors = None
    if m.visual.kind == "vertex":
        colors = np.asarray(m.visual.vertex_colors[:, :3], dtype=np.uint8)
    return verts, faces, colors


def to_pytorch3d(verts, faces, device):
    from pytorch3d.renderer import TexturesVertex
    from pytorch3d.structures import Meshes

    v = torch.as_tensor(verts)
    f = torch.as_tensor(faces)
    # a flat light grey, as in Diff3F's convert_trimesh_to_torch_mesh
    tex = TexturesVertex(verts_features=torch.ones_like(v)[None] * 0.8)
    return Meshes(verts=[v], faces=[f], textures=tex).to(device)


def init_pipe(device, sd_repo, fp16_variant=True, cpu_offload=True):
    """Diff3F's init_pipe, with the SD repository replaced and fp16 weight files where available
    (downloads about 4 GB instead of 8 GB). cpu_offload=False keeps every model on the GPU; the
    run needs more memory but skips the CPU-GPU copies of every denoising step."""
    from diffusers import ControlNetModel, DDIMScheduler
    from huggingface_hub import hf_hub_download
    from pipeline_controlnet_img2img import StableDiffusionControlNetImg2ImgPipeline
    from safetensors.torch import load_file
    from unet_2d_condition import UNet2DConditionModel

    variant = "fp16" if fp16_variant else None
    controlnet = [
        ControlNetModel.from_pretrained(repo, torch_dtype=torch.float16, variant=variant)
        for repo in (CONTROLNET_DEPTH, CONTROLNET_NORMAL)
    ]
    # Diff3F's own UNet class returns the intermediate decoder features; load the SD 1.5 weights
    # into it exactly as upstream does.
    unet = UNet2DConditionModel.from_config(sd_repo, subfolder="unet").to(device, torch.float16)
    unet.load_state_dict(
        load_file(
            hf_hub_download(
                repo_id=sd_repo,
                subfolder="unet",
                filename="diffusion_pytorch_model.fp16.safetensors",
            )
        )
    )
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        sd_repo,
        unet=unet,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        variant=variant,
        safety_checker=None,
    )
    pipe.set_progress_bar_config(disable=True)
    pipe = pipe.to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    if cpu_offload:
        pipe.enable_model_cpu_offload()
    return pipe


@torch.no_grad()
def render_views(device, mesh, elevation, azimuth, H, W):
    """Diff3F's run_rendering with the camera angles given explicitly (degrees). Returns the shaded
    images, the normal-map images, the cameras and the depth buffers, like Diff3F's batch_render."""
    from normal_shading import HardPhongNormalShader
    from pytorch3d.renderer import (
        MeshRenderer,
        PerspectiveCameras,
        PointLights,
        look_at_view_transform,
    )
    from pytorch3d.renderer.mesh.rasterizer import MeshRasterizer, RasterizationSettings
    from pytorch3d.renderer.mesh.shader import HardPhongShader

    bbox = mesh.get_bounding_boxes()
    bmin = bbox.min(dim=-1).values[0]
    bmax = bbox.max(dim=-1).values[0]
    center = ((bmin + bmax) / 2.0)[None]
    distance = torch.sqrt(((bmax - bmin) ** 2).sum()) * 0.65
    rotation, translation = look_at_view_transform(
        dist=distance, elev=elevation, azim=azimuth, device=device, at=center
    )
    camera = PerspectiveCameras(R=rotation, T=translation, device=device)
    settings = RasterizationSettings(
        image_size=(H, W), blur_radius=0.0, faces_per_pixel=1, bin_size=0
    )
    rasterizer = MeshRasterizer(cameras=camera, raster_settings=settings)
    lights = PointLights(
        diffuse_color=((0.4, 0.4, 0.5),),
        ambient_color=((0.6, 0.6, 0.6),),
        specular_color=((0.01, 0.01, 0.01),),
        location=camera.get_camera_center(),
        device=device,
    )
    batch = mesh.extend(len(elevation))
    shader = HardPhongShader(device=device, cameras=camera, lights=lights)
    images = MeshRenderer(rasterizer=rasterizer, shader=shader)(batch)
    normal_shader = HardPhongNormalShader(device=device, cameras=camera, lights=lights)
    normals = MeshRenderer(rasterizer=rasterizer, shader=normal_shader)(batch)
    depth = rasterizer(batch).zbuf
    return images, normals, camera, depth


def front_hemisphere_angles(num_views, max_elev=45.0, max_azim=75.0):
    """A steps x steps grid of (elevation, azimuth) in front of the mesh. azim=0, elev=0 is the
    camera on the +z axis, which is where normal_to_mesh.py's camera is."""
    steps = int(math.sqrt(num_views))
    elev = torch.linspace(-max_elev, max_elev, steps).repeat(steps)
    azim = torch.repeat_interleave(torch.linspace(-max_azim, max_azim, steps), steps)
    return elev, azim


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--mesh", required=True, help="mesh file (ply/obj/off)")
    ap.add_argument("--prompt", required=True, help="what the object is, for example 'cow'")
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--num_views", type=int, default=100, help="rendered views (a square number)")
    ap.add_argument(
        "--max_vertices", type=int, default=15000, help="features for at most this many vertices"
    )
    ap.add_argument(
        "--front_only", action="store_true", help="cameras only in front of the mesh (+z)"
    )
    ap.add_argument(
        "--tolerance",
        type=float,
        default=0.004,
        help="ball-query radius as a fraction of the mesh extent (Diff3F notebook: 0.004)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sd_repo", default=SD_REPO)
    ap.add_argument(
        "--no_fp16_variant",
        action="store_true",
        help="download the full-precision weight files instead",
    )
    ap.add_argument(
        "--no_cpu_offload",
        action="store_true",
        help="keep all models on the GPU (faster, needs about 10 GB more memory)",
    )
    args = ap.parse_args()

    import diff3f as d3f
    from dino import init_dino

    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")
    t0 = time.time()

    verts, faces, colors = load_mesh(args.mesh)
    rng = np.random.default_rng(args.seed)
    if len(verts) > args.max_vertices:
        idx = np.sort(rng.choice(len(verts), args.max_vertices, replace=False))
    else:
        idx = np.arange(len(verts))
    print(f"{args.mesh}: {len(verts)} vertices, {len(faces)} faces, features for {len(idx)}")

    mesh = to_pytorch3d(verts, faces, device)
    if args.front_only:
        elev, azim = front_hemisphere_angles(args.num_views)

        def batch_render(device, mesh, mesh_vertices, num_views, H, W, use_normal_map=False):
            return render_views(device, mesh, elev, azim, H, W)

        d3f.batch_render = batch_render  # replaces the full-sphere cameras for this run only
        num_views = len(elev)
    else:
        num_views = args.num_views

    pipe = init_pipe(
        device,
        args.sd_repo,
        fp16_variant=not args.no_fp16_variant,
        cpu_offload=not args.no_cpu_offload,
    )
    dino = init_dino(device)
    print(f"models ready after {time.time() - t0:.0f} s")

    mesh_vertices = mesh.verts_list()[0][torch.as_tensor(idx, device=device)]
    feat = d3f.get_features_per_vertex(
        device=device,
        pipe=pipe,
        dino_model=dino,
        mesh=mesh,
        prompt=args.prompt,
        mesh_vertices=mesh_vertices,
        num_views=num_views,
        H=512,
        W=512,
        tolerance=args.tolerance,
        num_images_per_prompt=1,
        use_normal_map=True,
    )
    feat = feat.float().cpu().numpy()
    feat /= np.maximum(np.linalg.norm(feat, axis=1, keepdims=True), 1e-8)

    meta = dict(
        command=" ".join(sys.argv),
        commit=git_commit(),
        mesh=args.mesh,
        prompt=args.prompt,
        num_views=num_views,
        front_only=args.front_only,
        tolerance=args.tolerance,
        seed=args.seed,
        sd_repo=args.sd_repo,
        n_vertices=int(len(verts)),
        n_features=int(len(idx)),
        seconds=round(time.time() - t0, 1),
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(
        args.out,
        idx=idx.astype(np.int64),
        xyz=verts[idx],
        feat=feat.astype(np.float16),
        colors=colors[idx] if colors is not None else np.zeros((0, 3), np.uint8),
        meta=json.dumps(meta),
    )
    print("saved", args.out, json.dumps(meta))


if __name__ == "__main__":
    main()
