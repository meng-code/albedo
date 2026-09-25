import argparse
import os
import json
import torch
from PIL import Image
import numpy as np
from packaging import version

from accelerate import Accelerator
from diffusers.utils.import_utils import is_xformers_available
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor

from idarbdiffusion.models.unet_dr2d_condition import UNetDR2DConditionModel
from idarbdiffusion.pipelines.pipeline_idarbdiffusion import IDArbDiffusionPipeline
from idarbdiffusion.data.custom_dataset import CustomDataset
from idarbdiffusion.data.custom_mv_dataset import CustomMVDataset


def reform_image(img_nd):
    albedo, normal = img_nd[0, ...], img_nd[1, ...]
    mro = img_nd[2, ...]
    mtl, rgh = mro[:1, ...], mro[1:2, ...]
    mtl, rgh = np.repeat(mtl, 3, axis=0), np.repeat(rgh, 3, axis=0)
    img_reform = np.concatenate([albedo, normal, mtl, rgh], axis=-1)
    return img_reform.transpose(1, 2, 0)


def save_image(out, name):
    Nv = out.shape[0]
    for i in range(Nv):
        img = reform_image(out[i])
        img = (img * 255).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(output_dir, f'{name}_{(i):02d}.png'))


def load_pipeline():
    """
    load pipeline from hub
    or load from local ckpts: pipeline = IDArbDiffusionPipeline.from_pretrained("./pipeckpts")
    """
    text_encoder = CLIPTextModel.from_pretrained("lizb6626/IDArb", subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained("lizb6626/IDArb", subfolder="tokenizer")
    feature_extractor = CLIPImageProcessor.from_pretrained("lizb6626/IDArb", subfolder="feature_extractor")
    vae = AutoencoderKL.from_pretrained("lizb6626/IDArb", subfolder="vae")
    scheduler = DDIMScheduler.from_pretrained("lizb6626/IDArb", subfolder="scheduler")
    unet = UNetDR2DConditionModel.from_pretrained("lizb6626/IDArb", subfolder="unet")
    pipeline = IDArbDiffusionPipeline(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        vae=vae,
        unet=unet,
        safety_checker=None,
        scheduler=scheduler,
    )
    return pipeline


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='example/single', help='Input data directory')
parser.add_argument('--input_type', type=str, default='single', choices=['single', 'multi'], help='Specify input type (single or multi)')
parser.add_argument('--num_views', type=int, default=4, help='Number of views')
parser.add_argument('--cam', default=False, action='store_true', help='Enable to use camera pose')
parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
args = parser.parse_args()


weight_dtype = torch.float16

pipeline = load_pipeline()

if is_xformers_available():
    import xformers
    xformers_version = version.parse(xformers.__version__)
    pipeline.unet.enable_xformers_memory_efficient_attention()
    print(f'Use xformers version: {xformers_version}')

pipeline.to("cuda")
print("Pipeline loaded successfully")

if args.input_type == 'single':
    dataset = CustomDataset(
        root_dir=args.data_dir,
    )
else:
    dataset = CustomMVDataset(
        root_dir=args.data_dir,
        num_views=args.num_views,
        use_cam=args.cam,
    )

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

Nd = 3

for i, batch in enumerate(dataloader):

    imgs_in, imgs_mask, task_ids = batch['imgs_in'], batch['imgs_mask'], batch['task_ids']
    cam_pose = batch['pose']
    imgs_name = batch['data_name']

    imgs_in = imgs_in.to(weight_dtype).to("cuda")
    cam_pose = cam_pose.to(weight_dtype).to("cuda")

    B, Nv, _, H, W = imgs_in.shape

    imgs_in, imgs_mask, task_ids = imgs_in.flatten(0,1), imgs_mask.flatten(0,1), task_ids.flatten(0,2)

    with torch.autocast("cuda"):
        out = pipeline(
                imgs_in,
                task_ids,
                num_views=Nv,
                cam_pose=cam_pose,
                height=H, width=W,
                # generator=generator,
                guidance_scale=1.0,
                output_type='pt',
                num_images_per_prompt=1,
                eta=1.0,
            ).images

        out = out.view(B, Nv, Nd, *out.shape[1:])
        out = out.detach().cpu().numpy()

        for i in range(B):
            save_image(out[i], imgs_name[i])
