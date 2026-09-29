# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  IDArb 推理 pipeline —— 学习注释版                                              ║
# ║  (以下所有中文注释为学习用途添加;代码本身一行未改。原始版本见 git)                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# 【这个文件在整个系统里的位置】
#   main.py 第 114 行 pipeline(imgs_in, task_ids, ...) 执行的就是本文件的 __call__。
#   它负责"扩散采样":从纯噪声出发,让 UNet 反复去噪 50 次,得到 3 个分量
#   (albedo / normal / material)的 latent,再用 VAE 解码成图,交还给 main.py。
#
# 【标签说明】
#   🔴 算法 —— 真正的扩散 / 条件注入逻辑,必须逐行理解
#   ⚪ 管道 —— 格式转换、检查、搬运;知道"把什么变成什么"即可
#   ⚫ 废弃 —— 死代码 / 本次运行不会走到的分支 / 作者遗留物;可以跳过
#
# 【本次运行(main.py 的调用方式)固定下来的事实】
#   guidance_scale = 1.0   → do_classifier_free_guidance = False → 所有 CFG 分支都是 ⚫
#   num_inference_steps=50 → 去噪循环跑 50 圈(main.py 没传,用默认)
#   eta = 1.0              → DDIM 带随机性(等价 DDPM)
#   output_type = 'pt'     → 返回 0~1 的 torch tensor
#   safety_checker = None  → 安全检查是空操作
#   num_domains = 3        → 每张输入图产生 3 个分量
#
# 【全流程 9 站,本文件负责 ③④⑤⑧】
#   ① 读图 [custom_dataset]        ② 装车 [main.py]
#   ③ 备料:三样提示 [本文件 _encode_image / prepare_text_embeddings / 原 383 行]
#   ④ 起点:造噪声 [本文件 prepare_latents]
#   ⑤ 去噪循环 [本文件 原 420-453 行]  ← 心脏
#   ⑥ UNet [unet_dr2d_condition.py]  ⑦ 注意力 [transformer_dr2d.py]
#   ⑧ 还原:VAE 解码 [本文件 原 459 行]   ⑨ 存盘 [main.py]
#
# 【🔴 算法行速查】(原始行号;加注释后编辑器行号会变,注释里统一写"原 N 行")
#   原 148-149, 152   输入图 → -1~1 → VAE latent [N,4,64,64],复制 3 份 → [3N,4,64,64]
#   原 248-251, 383   5 个任务词 → CLIP 向量表 [5,77,1024];用 task_ids 查表取 3 条
#   原 238            造纯噪声起点 [3N,4,64,64]
#   原 426-428        带噪 latent(4ch)+ 输入图 latent(4ch)→ 8ch —— "输入图"喂给 UNet 的方式
#   原 432-439        调 UNet 预测(输入:8ch latent、时间步 t、任务向量、位姿、视角数)
#   原 447            调度器按 DDIM 公式走一步:x_t → x_{t-1}
#   原 459            50 圈后的干净 latent → VAE 解码成图
#
# 【形状约定】(单图 B=1,Nv=1;括号内为四视角 Nv=4)
#   image              [1,3,512,512]   (4,...)    白底输入图
#   image_latents      [3,4,64,64]     (12,...)   输入图 latent ×3 份
#   text_embeddings    [3,77,1024]     (12,...)   三个任务词的向量
#   latents            [3,4,64,64]     (12,...)   起点是噪声,50 圈后是结果
#   latent_model_input [3,8,64,64]     (12,...)   噪声 + 输入图 latent 通道拼接
#   noise_pred         [3,4,64,64]     (12,...)   UNet 输出
#   image(输出)        [3,3,512,512]   (12,...)   解码后 0~1 的图 → main.py 的 .images
#
# 【排列顺序(非常重要,第 5 部分要用)】
#   3N 条 latent 的顺序 = [图0-albedo, 图0-normal, 图0-material, 图1-albedo, 图1-normal, 图1-material, ...]
#   即"每张图连续 3 条,分量顺序固定 0/1/2"。image_latents、text_embeddings、latents 三者顺序一致。
# ──────────────────────────────────────────────────────────────────────────────
# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ──────────────────────────────────────────────────────────────────────────────
# ⚪ 管道 · import 区。大多是 diffusers / transformers 的零件,值得认识的几个:
#   DiffusionPipeline      所有 diffusers pipeline 的基类;本类继承它,白拿 .to()、进度条等通用功能
#   randn_tensor           造高斯噪声(站④ 用)
#   TF                     torchvision.transforms.functional,tensor ↔ PIL 图片互转的小工具
#   AutoencoderKL          VAE 的类型;UNet2DConditionModel 是标准 UNet 类型(实际传入的是作者魔改的 UNetDR2DConditionModel)
#   StableDiffusionPipelineOutput  返回值的包装盒,main.py 用 .images 从里面取图
#   einops 的 rearrange / repeat   本文件其实没用到,遗留 import
# ──────────────────────────────────────────────────────────────────────────────
import inspect
import warnings
from typing import Callable, List, Optional, Union

import PIL
import torch
import torchvision.transforms.functional as TF
from packaging import version
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import deprecate, logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from einops import rearrange, repeat

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# ══════════════════════════════════════════════════════════════════════════════
# 🔴 主类 IDArbDiffusionPipeline
#   把 VAE、UNet、调度器、CLIP 文本编码器等零件装在一起,提供 __call__ 做推理。
#   main.py 的 load_pipeline() 构造它;pipeline(...) 就是调它的 __call__。
# ══════════════════════════════════════════════════════════════════════════════
class IDArbDiffusionPipeline(DiffusionPipeline):
    r"""
    Pipeline to generate image variations from an input image using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        image_encoder ([`~transformers.CLIPVisionModelWithProjection`]):
            Frozen CLIP image-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        text_encoder ([`~transformers.CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for more details
            about a model's potential harms.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images; used as inputs to the `safety_checker`.
    """
    # TODO: feature_extractor is required to encode images (if they are in PIL format),
    # we should give a descriptive message if the pipeline doesn't have one.
    _optional_components = ["safety_checker"]

    # ──────────────────────────────────────────────────────────────────────────────
    # ⚪ 管道 · __init__:组装零件(main.py 的 load_pipeline() 调用它)
    #   两个 IDArb 特有的参数:
    #     camera_embedding_type='e_de_da_sincos'  ⚫ 死配置:存进 self 后再没被用过,模型侧零引用(Wonder3D 遗留)。位姿真正的用法见 transformer_dr2d.py 的 CaPE(第 5 部分)
    #     num_domains=3                           分量数(albedo / normal / material)
    # ──────────────────────────────────────────────────────────────────────────────
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True,
        camera_embedding_type: str = 'e_de_da_sincos',
        num_domains: int = 3,
    ):
        super().__init__()

        # ⚫ 废弃 · 原 86-121 行:safety_checker 的警告 + 老版本 diffusers 的兼容补丁。
        #   我们 safety_checker=None、UNet 配置是新版:这些 if 要么只打印一条警告,要么根本不触发。跳过。
        if safety_checker is None and requires_safety_checker:
            logger.warn(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely .If you're checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        # ⚪ 管道 · register_modules:把 7 个零件登记到 self 上,之后可用 self.vae / self.unet / self.scheduler 访问。
        #   登记过的模块,pipeline.to("cuda") 时会自动一起搬上 GPU。
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        # ⚪ 管道 · VAE 压缩倍率:VAE 有 4 级 block → 2^(4-1) = 8。512×512 的图压成 64×64 的 latent。
        #   image_processor 负责最后把 -1~1 的解码结果转回 0~1(原 470 行用)。
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

        # 记下 camera_embedding_type(⚫ 之后没用到)和分量数 3(🔴 后面按这个数复制 latent)。
        self.camera_embedding_type: str = camera_embedding_type
        self.num_domains: int = num_domains

        # 🔴 算法 · 站③ 备料(提前做一次):把 5 个任务词过一遍 CLIP 文本编码器,做成一张"查询表"存起来。
        #   推理时 __call__ 原 383 行用 task_ids 查这张表。详见下面的 prepare_text_embeddings。
        self.text_embeddings_dict = self.prepare_text_embeddings()

    # ══════════════════════════════════════════════════════════════════════════════
    # 🔴 算法 · 站③ 备料之一:输入图 → VAE latent,并复制成 3 份
    #   输入:image_pil —— PIL 图片列表,长度 N = B·Nv(单图 1,四视角 4)
    #   输出:image_latents —— [3N, 4, 64, 64]
    #   这是"输入图"进入模型的第一步:先压成 latent;之后每一圈去噪都把它和噪声拼在一起喂 UNet(原 426 行)。
    # ══════════════════════════════════════════════════════════════════════════════
    def _encode_image(self, image_pil, device, num_images_per_prompt, do_classifier_free_guidance):
        # ⚪ 取 VAE 权重的数据类型,保证后面新造的 tensor 类型一致
        dtype = self.vae.dtype

        # ⚫ 废弃 · 原 144-145 行:用 CLIP 的 feature_extractor 处理图片,但结果马上被原 147 行覆盖,等于没做。
        image_pt = self.feature_extractor(images=image_pil, return_tensors="pt").pixel_values
        image_pt = image_pt.to(device=device, dtype=dtype)

        # ⚪ 管道 · PIL 列表 → tensor [N,3,512,512],数值 0~1(TF.to_tensor 自动 /255)
        #   (__call__ 原 380 行刚把 tensor 转成 PIL,这里又转回来 —— 作者抄代码留下的绕圈,结果不变)
        image_pt = torch.stack([TF.to_tensor(img) for img in image_pil], dim=0).to(device).to(dtype)
        # 🔴 算法 · 0~1 → -1~1。VAE 训练时输入范围是 [-1, 1],必须对齐,否则 latent 全错。
        image_pt = image_pt * 2.0 - 1.0
        # 🔴 算法 · VAE 编码:[N,3,512,512] → [N,4,64,64]
        #   .latent_dist       VAE 编码器输出的是一个高斯分布(均值+方差),不是一个确定值
        #   .mode()            取分布的众数(= 均值),不做随机采样 —— 推理要稳定
        #   * scaling_factor   (0.18215)SD 的固定套路:把 latent 数值缩放到方差≈1 的量级,UNet 是在这个量级上训练的。
        #                      解码前会除回来(原 459 行),一进一出对称。
        image_latents = self.vae.encode(image_pt).latent_dist.mode() * self.vae.config.scaling_factor
  
        # ⚪ 管道 · num_images_per_prompt=1 → repeat 1 次 = 不变
        image_latents = image_latents.repeat(num_images_per_prompt, 1, 1, 1)
        # 🔴 算法 · 关键一行:沿第 0 维把每张图的 latent"就地复制 3 份"(num_domains=3)
        #   [N,4,64,64] → [3N,4,64,64]
        #   为什么:一张图要出 3 个分量,每个分量是一次独立的去噪任务,每个任务都要看同一张输入图。
        #   排列顺序(repeat_interleave 决定):[图0-albedo, 图0-normal, 图0-material, 图1-albedo, 图1-normal, ...]
        #   即"每张图连续 3 条,分量顺序 0/1/2"。这个顺序和原 383 行的 text_embeddings、以及第 5 部分
        #   跨分量注意力的分组方式一一对应,不能乱。
        image_latents = image_latents.repeat_interleave(self.num_domains, dim=0)

        # ⚫ 废弃 · CFG 关闭,不走。(开着的话会拼一份全零 latent 当"无条件"输入,和有条件的一起算)
        if do_classifier_free_guidance:
            image_latents = torch.cat([torch.zeros_like(image_latents), image_latents])

        return image_latents

    # ⚫ 废弃 · run_safety_checker:safety_checker=None → 直接返回 (image, None),什么都不做。
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker
    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        return image, has_nsfw_concept

    # ⚫ 废弃 · decode_latents:本文件没有任何地方调用它(收尾用的是原 459 行直接 vae.decode)。
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents
    def decode_latents(self, latents):
        warnings.warn(
            "The decode_latents method is deprecated and will be removed in a future version. Please"
            " use VaeImageProcessor instead",
            FutureWarning,
        )
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    # ⚪ 管道 · prepare_extra_step_kwargs:把 eta(=1.0)和 generator 打包成字典,原 447 行传给 scheduler.step。
    #   用 inspect 检查调度器的 step 函数接不接受这些参数(不同调度器签名不同)。
    #   eta 是 DDIM 的"随机性旋钮":0 = 完全确定(同样起点每次结果一样),1 = 每步再加随机(等价 DDPM)。作者选 1.0。
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    # ⚪ 管道 · check_inputs:三项检查,不满足就 raise 报错
    #   1. image 类型必须是 tensor / PIL / 列表
    #   2. height、width 必须是 8 的倍数(VAE 压 8 倍,不整除会出小数尺寸)
    #   3. callback_steps 必须是正整数(我们没用 callback,这条无实际意义)
    def check_inputs(self, image, height, width, callback_steps):
        if (
            not isinstance(image, torch.Tensor)
            and not isinstance(image, PIL.Image.Image)
            and not isinstance(image, list)
        ):
            raise ValueError(
                "`image` has to be of type `torch.FloatTensor` or `PIL.Image.Image` or `List[PIL.Image.Image]` but is"
                f" {type(image)}"
            )

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    # ══════════════════════════════════════════════════════════════════════════════
    # 🔴 算法 · 站④ 起点:造纯噪声 latent
    #   扩散的反向过程从"纯高斯噪声"开始,这里就是造那团噪声。
    #   调用时 batch_size = B·Nv·3(每个分量一团),num_channels_latents = 4
    # ══════════════════════════════════════════════════════════════════════════════
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        # 🔴 形状 = (B·Nv·3, 4, 512//8, 512//8) = (3, 4, 64, 64)  ← 形状表里 64 的来源
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        # ⚪ generator 是可选的随机种子对象;我们没传(None),这个检查不触发
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # 🔴 算法 · 没传 latents(我们没传)→ randn_tensor 生成标准正态分布(均值 0、方差 1)的随机数 = 纯噪声
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # ⚪ 管道 · 乘调度器要求的初始噪声标准差。DDIM 的 init_noise_sigma = 1.0,乘了等于没乘。
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    # ══════════════════════════════════════════════════════════════════════════════
    # 🔴 算法 · 站③ 备料之二:任务词 → CLIP 向量"查询表"(只在 __init__ 时算一次)
    #   这是 IDArb"一个模型出多种分量"的机制:换一句文字提示,就换一个分量。
    #   和 Stable Diffusion 换 prompt 生不同图是同一个原理,只是这里 prompt 固定为 5 个任务词。
    # ══════════════════════════════════════════════════════════════════════════════
    @torch.no_grad()
    def prepare_text_embeddings(self):
        # 🔴 5 个任务词。列表下标就是 task_id:0=albedo,1=normal,2=metallic and roughness(即 material)
        #   3、4 是训练时用的单独任务,推理只用 0/1/2。
        task_prompts = ['albedo', 'normal', 'metallic and roughness', 'metallic', 'roughness']
        # ⚪ tokenizer:把每句话切成整数编号,补齐到 77 个(CLIP 固定长度)→ input_ids [5,77]
        input_ids = self.tokenizer(task_prompts, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids
        # ⚪ 搬到文本编码器所在的设备
        input_ids = input_ids.to(self.text_encoder.device)
        # 🔴 text_encoder(CLIP 文本编码器):编号 → 语义向量。输出 [5, 77, 1024]
        #   每个任务词得到一个 [77,1024] 的矩阵;1024 = UNet 配置里的 cross_attention_dim。
        #   UNet 的 cross-attention 层就是用这个向量"读懂"当前这团噪声要生成哪个分量。
        text_embeddings_dict = self.text_encoder(input_ids, return_dict=False)[0]
        return text_embeddings_dict
    
    # ⚪ 管道 · 重写 .to():pipeline.to("cuda") 时顺便把查询表也搬上 GPU(它不是登记的模块,不会自动搬)。
    def to(self, *args, **kwargs):
        self.text_embeddings_dict = self.text_embeddings_dict.to(*args, **kwargs)
        super().to(*args, **kwargs)

    # ══════════════════════════════════════════════════════════════════════════════
    # 🔴 主函数 __call__ —— main.py 原 114 行 pipeline(...) 执行的就是它
    #   @torch.no_grad():整个函数不记梯度(推理不学习,省显存;和 main.py 的 .detach() 同一目的)
    #   参数 ↔ main.py 实际传的值:
    #     image          = imgs_in   [B·Nv,3,512,512]  白底图 tensor
    #     task_ids       = [0,1,2, 0,1,2, ...]  长度 B·Nv·3
    #     num_views      = Nv
    #     cam_pose       = [B,Nv,4]
    #     height / width = 512
    #     guidance_scale = 1.0(CFG 关)  eta = 1.0  output_type = 'pt'  num_images_per_prompt = 1
    #     其余用默认:num_inference_steps=50, generator=None, latents=None, callback=None, normal_cond=None
    #   作者自己用 "# 0." ~ "# 7." 标了 8 步,下面沿用这个编号。
    # ══════════════════════════════════════════════════════════════════════════════
    @torch.no_grad()
    def __call__(
        self,
        image: Union[List[PIL.Image.Image], torch.FloatTensor],
        task_ids: torch.LongTensor,
        num_views: int = 1,
        cam_pose: Optional[torch.FloatTensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        normal_cond: Optional[Union[List[PIL.Image.Image], torch.FloatTensor]] = None,
    ):
        # ⚫ 废弃 · 原 279-348 行的 docstring 是从 SD image-variation pipeline 抄的,参数说明和示例都不完全适用,跳过。
        r"""
        The call function to the pipeline for generation.

        Args:
            image (`PIL.Image.Image` or `List[PIL.Image.Image]` or `torch.FloatTensor`):
                Image or images to guide image generation. If you provide a tensor, it needs to be compatible with
                [`CLIPImageProcessor`](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/blob/main/feature_extractor/preprocessor_config.json).
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. This parameter is modulated by `strength`.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.

        Examples:

        ```py
        from diffusers import StableDiffusionImageVariationPipeline
        from PIL import Image
        from io import BytesIO
        import requests

        pipe = StableDiffusionImageVariationPipeline.from_pretrained(
            "lambdalabs/sd-image-variations-diffusers", revision="v2.0"
        )
        pipe = pipe.to("cuda")

        url = "https://lh3.googleusercontent.com/y-iFOHfLTwkuQSUegpwDdgKmOjRSTvPxat63dQLB25xkTs4lhIbRUFeNBWZzYf370g=s1200"

        response = requests.get(url)
        image = Image.open(BytesIO(response.content)).convert("RGB")

        out = pipe(image, num_images_per_prompt=3, guidance_scale=15)
        out["images"][0].save("result.jpg")
        ```
        """
        # ⚪ 管道 · 第 0 步:默认高宽 = sample_size(32) × vae_scale_factor(8) = 256。
        #   `a or b`:a 有值就用 a。main.py 传了 512,所以用 512,这个默认值没用上。
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # ⚪ 管道 · 第 1 步:检查输入(见 check_inputs);num_images_per_prompt 只允许 1,否则 raise
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(image, height, width, callback_steps)
        if num_images_per_prompt > 1:
            raise NotImplementedError("num_images_per_prompt > 1 is not supported yet.")

        # ⚪ 管道 · 第 2 步:数"这次几张图"。我们传的是 tensor → batch_size = image.shape[0] = B·Nv(单图 1,四视角 4)
        # 2. Define call parameters
        if isinstance(image, list):
            batch_size = len(image)
        elif isinstance(image, torch.Tensor):
            batch_size = image.shape[0]
            # assert batch_size >= self.num_views and batch_size % self.num_views == 0
        # ⚫ 废弃 · 单张 PIL 输入不支持;raise 之后那两行永远执行不到(旧版遗留)
        elif isinstance(image, PIL.Image.Image):
            raise NotImplementedError
            image = [image]*self.num_views*2
            batch_size = self.num_views*2
    
        # ⚪ 管道 · device = "cuda"(模型在 GPU 上,后面造的 tensor 都要放同一设备,否则报错)
        #   dtype 这个局部变量后面没被用到(遗留;需要类型时作者写的是 text_embeddings.dtype)
        device = self._execution_device
        dtype = self.vae.dtype
        # ⚪ 管道 · CFG 开关:guidance_scale=1.0 → False。
        #   CFG(classifier-free guidance,无分类器引导):SD 常用技巧,每步猜两次(有提示 / 无提示)再把差距放大 w 倍,
        #   让图更贴提示,代价是每步算两遍。IDArb 关掉它:输入图 + 任务词已经是很强的条件,不需要放大。
        #   → 本文件所有 `if do_classifier_free_guidance:` 分支都不会执行。
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale != 1.0

        # ══════════════ 站③ 备料:三样提示(输入图 / 任务词 / 位姿)══════════════
        # 提示①:输入图 → latent(原 376-381)
        # ⚪ 管道 · 原 377-380:把 image 统一成 PIL 列表。我们传的是 tensor → 原 380 行逐张 tensor→PIL(列表推导式)。
        #   纯格式转换;_encode_image 内部又会转回 tensor。
        # 3. Encode input image
        if isinstance(image, list):
            image_pil = image
        elif isinstance(image, torch.Tensor):
            image_pil = [TF.to_pil_image(image[i]) for i in range(image.shape[0])]
        # 🔴 算法 · 调 _encode_image:VAE 编码 + 复制 3 份 → image_latents [B·Nv·3, 4, 64, 64]
        image_latents = self._encode_image(image_pil, device, num_images_per_prompt, do_classifier_free_guidance)

        # 提示②:任务词
        # 🔴 算法 · 用 task_ids 查表:text_embeddings_dict [5,77,1024] 用 tensor 当下标 [0,1,2,...] 取行
        #   → text_embeddings [B·Nv·3, 77, 1024]
        #   task_ids 来自 Dataset 的 [0,1,2](每张图一份),经 main.py flatten 后为 [0,1,2, 0,1,2, ...],
        #   和原 152 行 image_latents 的排列 [图0-a, 图0-n, 图0-m, 图1-a, ...] 逐条对应:
        #   第 k 条 latent 配第 k 条任务向量 —— "这团噪声要生成哪个分量"就是这样指定的。
        text_embeddings = self.text_embeddings_dict[task_ids]
        # ⚫ 废弃 · CFG 关,不走
        if do_classifier_free_guidance:
            text_embeddings = torch.cat([text_embeddings, text_embeddings])

        # ⚫ 废弃 · normal_cond=None(main.py 没传),整段不走。(作者实验过"额外给法线图当条件",推理未启用)
        if normal_cond is not None:
            if isinstance(normal_cond, list):
                normal_cond_pil = normal_cond
            elif isinstance(normal_cond, torch.Tensor):
                normal_cond_pil = [TF.to_pil_image(normal_cond[i]) for i in range(normal_cond.shape[0])]
            _, image_latents = self._encode_image(normal_cond_pil, device, num_images_per_prompt, do_classifier_free_guidance)

        # 提示③:相机位姿
        # ⚪ 管道 · pose 原样透传给 UNet,[B,Nv,4];怎么编码、怎么用是第 4 部分 UNet 的事。class_labels 没用。
        # prepare camera embedding
        pose = cam_pose
        class_labels = None
        # ⚫ 废弃 · CFG 关,不走
        if do_classifier_free_guidance:
            pose = torch.cat([pose] * 2)
        
        # ⚪ 管道 · 第 4 步:排时间表。训练时噪声分 1000 级(t=999 纯噪声 … t=0 干净图),
        #   set_timesteps(50) 从中等距挑 50 个、降序:timesteps = [999, 979, 958, ..., 20, 0](每隔 20 级取一个)。去噪循环沿它走。
        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # ══════════════ 站④ 起点 ══════════════
        # 🔴 算法 · 第 5 步:造噪声。num_channels_latents = UNet 输出通道 = 4;
        #   条数 = batch_size × 1 × 3 = B·Nv·3。详见 prepare_latents → latents [B·Nv·3, 4, 64, 64] 纯噪声
        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.out_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt * self.num_domains,
            num_channels_latents,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )
        
        # ⚪ 管道 · 第 6 步:把 eta=1.0 打包给调度器
        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # ══════════════ 站⑤ 去噪循环 —— 本文件的心脏 ══════════════
        # 🔴 算法 · 第 7 步:50 圈,每圈做三件事:拼输入 → UNet 猜 → 调度器去一点噪
        #   num_warmup_steps 和 progress_bar 只是进度条相关(⚪);循环体真正的算法只有
        #   原 426-428(拼)、原 432-439(UNet)、原 447(调度器)三处。
        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            # 🔴 i = 第几圈(0..49),t = 当前噪声等级(999 → … → 0)
            for i, t in enumerate(timesteps):
                # ⚫ 废弃 · CFG 关 → 三元表达式走 else:latent_model_input = latents(不复制)
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                # 🔴 算法 · 通道拼接:[带噪 latent 4ch] + [输入图 latent 4ch] → 8ch
                #   torch.cat(..., dim=1) 沿通道维叠。[3,4,64,64] + [3,4,64,64] → [3,8,64,64]
                #   这就是 UNet 配置 in_channels=8 的由来,也是"输入图"喂给 UNet 的方式:
                #   不是当 prompt,而是和噪声并排叠在一起,让 UNet 每一层都能直接看到原图。
                #   (对比:SD img2img 是把图加噪当起点;IDArb 是把图当条件拼通道,做法类似 InstructPix2Pix / Marigold)
                latent_model_input = torch.cat([
                    latent_model_input, image_latents
                ], dim=1)
                # ⚪ 管道 · DDIM 的 scale_model_input 是恒等变换(Euler 等调度器需要缩放,DDIM 不需要)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # 🔴 算法 · 调 UNet 猜 —— 每圈最贵的一步,内部结构见第 4、5 部分。参数逐个看:
                #   latent_model_input     [B·Nv·3, 8, 64, 64]  带噪 latent + 输入图 latent
                #   t                      当前噪声等级;UNet 内部转成时间步向量,告诉它"现在噪声有多重"
                #   encoder_hidden_states  = text_embeddings [B·Nv·3, 77, 1024],任务词向量,走 cross-attention
                #   num_views              = Nv;第 5 部分跨视角注意力用它把 B·Nv·3 条按视角分组
                #   pose                   [B,Nv,4] 相机位姿;⚠ 不进时间步向量(class_embedding 为 None,配置里的 projection_class_embeddings_input_dim=6 是死配置),
#                          真正用法:transformer_dr2d.py 里用 CaPE 把它当角度旋转 attn_mv 的 Q/K(第 5 部分);单图或不加 --cam 时所有视角 pose 相同,CaPE 等于空转
                #   class_labels           None
                #   .sample                取出输出 tensor → noise_pred [B·Nv·3, 4, 64, 64]
                #   注意:配置 prediction_type = v_prediction,所以它预测的是 DDIM 论文里的 "v",不是噪声本身;
                #   变量名 noise_pred 是沿用 SD 的叫法。调度器知道怎么换算,不影响理解。
                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                    num_views=num_views,
                    pose=pose,
                    class_labels=class_labels,
                ).sample

                # ⚫ 废弃 · CFG 关,不走(开着的话把"无提示"和"有提示"两次预测按 guidance_scale 加权合并)
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # 🔴 算法 · 调度器走一步:x_t → x_{t-1}
                #   scheduler.step(预测, 当前 t, 当前 latents, eta=1.0) 按 DDIM 公式算出"噪声少一级"的 latent,
                #   .prev_sample 取出它,覆盖 latents。这一行就是论文里反向扩散更新公式的代码化,公式封装在 diffusers 里。
                #   50 圈后,latents 从纯噪声变成 3 个分量的干净 latent。
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # ⚪ 管道 · 更新进度条;callback=None,不触发
                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # ══════════════ 站⑧ 还原:latent → 图 ══════════════
        # ⚪ 管道 · output_type='pt' ≠ 'latent' → 走解码分支
        if not output_type == "latent":
            # ⚫ 废弃 · num_channels_latents = 4 ≠ 8,不走
            if num_channels_latents == 8:
                latents = torch.cat([latents[:, :4], latents[:, 4:]], dim=0)

            # 🔴 算法 · VAE 解码:先除回 scaling_factor(和原 149 行对称),再 [B·Nv·3,4,64,64] → [B·Nv·3,3,512,512],数值 -1~1
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            # ⚫ 废弃 · safety_checker=None,原样返回
            image, has_nsfw_concept = self.run_safety_checker(image, device, text_embeddings.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        # ⚪ 管道 · has_nsfw_concept=None → 所有图都要 denormalize(-1~1 → 0~1)
        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        # ⚪ 管道 · postprocess:(x/2 + 0.5).clamp(0,1) 把 -1~1 变 0~1;output_type='pt' 保持 tensor
        #   → image [B·Nv·3, 3, 512, 512],这就是 main.py 拿到的 .images
        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # ⚪ 管道 · 打包返回。main.py 用 .images 取图,然后 view(B, Nv, 3, ...) 还原分组 → 站⑨ 存盘
        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
    
