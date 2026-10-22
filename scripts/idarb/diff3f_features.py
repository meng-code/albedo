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

Variants tried on 2026-10-22 (docs/decisions.md; none beat the default). They need the diff3fvar
environment (scripts/idarb/requirements-meshfm.txt). Without them the script calls Diff3F's own
get_features_per_vertex unchanged; with any of them a step-by-step copy of it runs:
  --appearance vertex_colors  render the mesh with its own vertex colours (for a relief, the IDArb
                              albedo) instead of letting ControlNet paint a grey render, and take
                              the UNet features from one pass at --timestep (SD-DINO style).
  --sam_tau T                 clean each view's per-pixel features with SAM masks (MeshFM style)
                              and write them to <out>_sam.npz.
  --siglip2                   also SigLIP2 features of the same views, written to <out>_siglip2.npz
                              (768-d; pre-head block outputs, not in the image-text joint space).

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


def to_pytorch3d(verts, faces, device, colors=None):
    from pytorch3d.renderer import TexturesVertex
    from pytorch3d.structures import Meshes

    v = torch.as_tensor(verts)
    f = torch.as_tensor(faces)
    if colors is None:
        # a flat light grey, as in Diff3F's convert_trimesh_to_torch_mesh
        tex = TexturesVertex(verts_features=torch.ones_like(v)[None] * 0.8)
    else:  # --appearance vertex_colors
        rgb = torch.as_tensor(colors, dtype=torch.float32)[None] / 255
        tex = TexturesVertex(verts_features=rgb)
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


@torch.no_grad()
def real_image_unet_features(pipe, image, prompt, timestep, generator):
    """--appearance vertex_colors: the decoder features of one UNet pass over a real rendered image,
    as in SD-DINO (Zhang et al. 2023): VAE-encode the image, add noise at a small timestep, run the
    UNet once with the prompt and no ControlNet, and keep the same decoder output Diff3F keeps
    (up_blocks[1], 1280 channels at 1/16 resolution), L2-normalised per pixel."""
    from PIL import Image

    device = pipe.unet.device
    pos_prompt = f"{prompt},best quality,highly detailed,photorealistic,photo"  # as run_diffusion
    emb, _ = pipe.encode_prompt(pos_prompt, device, 1, False)
    img = pipe.image_processor.preprocess(Image.fromarray(image)).to(device, torch.float16)
    lat = pipe.vae.encode(img).latent_dist.mean * pipe.vae.config.scaling_factor
    noise = torch.randn(lat.shape, generator=generator, device=device, dtype=lat.dtype)
    t = torch.tensor([timestep], device=device)
    noisy = pipe.scheduler.add_noise(lat, noise, t)
    _, f_map = pipe.unet(noisy, t, encoder_hidden_states=emb, return_dict=False, extract_layer=True)
    return torch.nn.functional.normalize(f_map, dim=1)[0]


@torch.no_grad()
def sam_clean(features, image, sam_generator, tau):
    """--sam_tau: MeshFM-style cleaning of one view's per-pixel features before they are projected
    onto the vertices. SAM's automatic masks split the image into regions; every pixel belongs to
    the smallest mask that covers it. Per mask, the channel-wise median feature is computed, and
    pixels whose unit-normalised feature lies farther than tau from the unit-normalised median are
    replaced by the median. features: [1, C, H*W]; returns the cleaned copy and the share of
    pixels that were replaced."""
    masks = sorted(sam_generator.generate(image), key=lambda m: -m["area"])
    label = np.full(image.shape[:2], -1, np.int64)
    for k, m in enumerate(masks):  # smaller masks are written later and win
        label[m["segmentation"]] = k
    label = torch.as_tensor(label.reshape(-1), device=features.device)
    f = features[0].T.clone()  # [H*W, C]
    replaced = 0
    for k in range(len(masks)):
        sel = torch.nonzero(label == k)[:, 0]
        if len(sel) < 10:
            continue
        fk = f[sel].float()
        med = fk.median(dim=0).values
        dist = (torch.nn.functional.normalize(fk, dim=1)
                - torch.nn.functional.normalize(med, dim=0)).norm(dim=1)
        bad = sel[dist > tau]
        f[bad] = med.to(f.dtype)
        replaced += len(bad)
    return f.T[None], replaced / max(1, int((label >= 0).sum()))


SIGLIP2 = "vit_base_patch16_siglip_512.v2_webli"  # timm name of google/siglip2-base-patch16-512


def init_siglip2(device):
    import timm

    return timm.create_model(SIGLIP2, pretrained=True, num_classes=0).eval().to(device).half()


@torch.no_grad()
def siglip2_features(model, image, grid):
    """--siglip2 (as in SemAnCorr): SigLIP2 patch embeddings of one view, the mean of the vision
    tower's last four blocks, sampled at every pixel like Diff3F's DINO branch and L2-normalised.
    image: PIL RGB; returns [1, 768, H*W]."""
    x = torch.as_tensor(np.asarray(image.convert("RGB").resize((512, 512)), np.float32) / 255)
    x = ((x.permute(2, 0, 1)[None] - 0.5) / 0.5).to(grid.device, torch.float16)
    maps = model.forward_intermediates(x, indices=4, output_fmt="NCHW", intermediates_only=True)
    f = torch.stack(maps).mean(0)
    f = torch.nn.functional.grid_sample(f, grid, align_corners=False).reshape(1, f.shape[1], -1)
    return torch.nn.functional.normalize(f, dim=1)


def features_per_vertex(device, pipe, dino_model, mesh, prompt, mesh_vertices, num_views, H, W,
                        tolerance, appearance="painted", timestep=100, sam_generator=None,
                        sam_tau=0.0, siglip_model=None, seed=0):
    """Diff3F's get_features_per_vertex (third_party/Diff3F/diff3f.py), copied step by step with
    hooks: the image and UNet features per view (--appearance), SAM cleaning of the fused per-pixel
    features (--sam_tau) and SigLIP2 features of the same views (--siglip2). Only used when one of
    those options is set. Returns a dict of per-vertex features: "diff3f" always, "diff3f_sam" with
    --sam_tau, "siglip2" with --siglip2, and the mean share of pixels SAM cleaning replaced."""
    import diff3f as d3f
    from diffusion import add_texture_to_render
    from dino import get_dino_features
    from PIL import Image
    from pytorch3d.ops import ball_query

    maximal_distance = torch.cdist(mesh_vertices, mesh_vertices).max()
    ball_drop_radius = maximal_distance * tolerance
    images, normals, camera, depth = d3f.batch_render(
        device, mesh, mesh.verts_list()[0], num_views, H, W, True
    )
    normals, images, camera, depth = normals.cpu(), images.cpu(), camera.cpu(), depth.cpu()
    pixel_coords = d3f.arange_pixels((H, W), invert_y_axis=True)[0]
    pixel_coords[:, 0] = torch.flip(pixel_coords[:, 0], dims=[0])
    grid = d3f.arange_pixels((H, W), invert_y_axis=False)[0].to(device).reshape(1, H, W, 2).half()
    dims = {"diff3f": d3f.FEATURE_DIMS}
    if sam_generator is not None and sam_tau > 0:
        dims["diff3f_sam"] = d3f.FEATURE_DIMS
    if siglip_model is not None:
        dims["siglip2"] = siglip_model.num_features
    acc = {k: torch.zeros((len(mesh_vertices), d)).half() for k, d in dims.items()}
    count = torch.zeros((len(mesh_vertices), 1)).half()
    gen = torch.Generator(device=device).manual_seed(seed)
    replaced = []
    for idx in range(len(images)):
        dp = depth[idx].flatten().unsqueeze(1)
        xy_depth = torch.cat((pixel_coords, dp), dim=1)
        indices = xy_depth[:, 2] != -1
        xy_depth = xy_depth[indices]
        world_coords = camera[idx].unproject_points(
            xy_depth, world_coordinates=True, from_ndc=True
        ).to(device)
        render_img = (images[idx, :, :, :3].numpy() * 255).astype(np.uint8)
        if appearance == "painted":
            out = add_texture_to_render(
                pipe, render_img, depth[idx, :, :, 0].unsqueeze(0).to(device), prompt,
                normal_map_input=normals[idx], use_latent=False, num_images_per_prompt=1,
                return_image=True,
            )
            unet_ft, view_img = out[0], out[1][0]
        else:  # vertex_colors: the rendered real appearance itself
            view_img = Image.fromarray(render_img)
            unet_ft = real_image_unet_features(pipe, render_img, prompt, timestep, gen)
        dino_ft = get_dino_features(device, dino_model, view_img, grid)
        up = torch.nn.Upsample(size=(H, W), mode="bilinear")(unet_ft.unsqueeze(0)).to(device)
        aligned = torch.nn.functional.grid_sample(up, grid, align_corners=False)
        aligned = torch.nn.functional.normalize(aligned.reshape(1, up.size(1), -1), dim=1)
        aligned = torch.hstack([aligned * 0.5, dino_ft * 0.5])
        per_view = {"diff3f": aligned}
        if "diff3f_sam" in acc:
            per_view["diff3f_sam"], share = sam_clean(
                aligned, np.asarray(view_img.convert("RGB")), sam_generator, sam_tau
            )
            replaced.append(share)
        if "siglip2" in acc:
            per_view["siglip2"] = siglip2_features(siglip_model, view_img, grid)
        q = ball_query(world_coords.unsqueeze(0), mesh_vertices.unsqueeze(0), K=100,
                       radius=ball_drop_radius, return_nn=False).idx[0].cpu()
        mask = q != -1
        count[q[mask]] += 1
        for k, f in per_view.items():
            per_pixel = f[0, :, indices].cpu()
            acc[k][q[mask]] += per_pixel.repeat_interleave(mask.sum(dim=1), dim=1).T
    filled = (count != 0)[:, 0]
    missing = ~filled
    print("Number of missing features: ", int(missing.sum()))
    nearest = None
    if missing.any():
        nearest = torch.cdist(mesh_vertices[missing], mesh_vertices[filled]).argmin(dim=1).cpu()
    for ft in acc.values():
        ft[filled] = ft[filled] / count[filled]
        if nearest is not None:
            ft[missing] = ft[filled][nearest]
    acc["sam_replaced_share"] = float(np.mean(replaced)) if replaced else None
    return acc


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
    ap.add_argument(
        "--appearance",
        default="painted",
        choices=["painted", "vertex_colors"],
        help="painted: Diff3F (ControlNet paints each grey render); vertex_colors: render the "
        "mesh's own vertex colours and take one-pass UNet features at --timestep (SD-DINO style)",
    )
    ap.add_argument("--timestep", type=int, default=100, help="noise step for vertex_colors")
    ap.add_argument(
        "--sam_tau",
        type=float,
        default=0.0,
        help="> 0: MeshFM-style cleaning of each view's features with SAM masks (MeshFM: 1)",
    )
    ap.add_argument(
        "--sam_checkpoint",
        default=os.path.expanduser("~/semtex/weights/sam/sam_vit_b_01ec64.pth"),
    )
    ap.add_argument(
        "--siglip2",
        action="store_true",
        help="also SigLIP2 features of the same views (timm " + SIGLIP2 + ")",
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

    if args.appearance == "vertex_colors" and colors is None:
        raise SystemExit(f"{args.mesh} has no vertex colours, needed by --appearance vertex_colors")
    render_colors = colors if args.appearance == "vertex_colors" else None
    mesh = to_pytorch3d(verts, faces, device, render_colors)
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
    sam_share = None
    extra = {}
    if args.appearance == "painted" and args.sam_tau <= 0 and not args.siglip2:  # upstream path
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
    else:
        sam_gen = None
        if args.sam_tau > 0:
            from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

            sam = sam_model_registry["vit_b"](checkpoint=args.sam_checkpoint).to(device)
            sam_gen = SamAutomaticMaskGenerator(sam)
        siglip = init_siglip2(device) if args.siglip2 else None
        acc = features_per_vertex(
            device, pipe, dino, mesh, args.prompt, mesh_vertices, num_views, 512, 512,
            args.tolerance, appearance=args.appearance, timestep=args.timestep,
            sam_generator=sam_gen, sam_tau=args.sam_tau, siglip_model=siglip, seed=args.seed,
        )
        feat, sam_share = acc["diff3f"], acc["sam_replaced_share"]
        extra = {k: v for k, v in acc.items() if k in ("diff3f_sam", "siglip2")}

    def unit(f):
        f = f.float().cpu().numpy()
        return f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-8)

    feat = unit(feat)

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
        appearance=args.appearance,
        timestep=args.timestep if args.appearance == "vertex_colors" else None,
        sam_tau=args.sam_tau,
        sam_replaced_share=sam_share,
        siglip2=SIGLIP2 if args.siglip2 else None,
        n_vertices=int(len(verts)),
        n_features=int(len(idx)),
        seconds=round(time.time() - t0, 1),
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    # --out holds the Diff3F features (with --sam_tau or --siglip2 they are still the uncleaned
    # Diff3F features of the same views); extra sets go next to it as <out>_<set>.npz
    outputs = {args.out: (feat, "diff3f")}
    for k, f in extra.items():
        outputs[args.out[: -len(".npz")] + "_" + k.replace("diff3f_", "") + ".npz"] = (unit(f), k)
    for path, (f, kind) in outputs.items():
        np.savez_compressed(
            path,
            idx=idx.astype(np.int64),
            xyz=verts[idx],
            feat=f.astype(np.float16),
            colors=colors[idx] if colors is not None else np.zeros((0, 3), np.uint8),
            meta=json.dumps(dict(meta, feature_set=kind, feature_dim=int(f.shape[1]))),
        )
        print("saved", path, kind, f.shape)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
