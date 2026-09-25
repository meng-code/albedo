# ==============================================================================
#  transformer_dr2d.py —— 学习注释版(注释是后加的,代码一行没改;原始版本在 git 里)
# ==============================================================================
#
# 一、这个文件是干什么的
#
#   UNet 里一共有 16 个 transformer block(下面统一叫"注意力模块")。IDArb 论文的创新点——
#   三个分量互相看、多个视角互相看、用相机角度调整注意力——全部写在这个文件里。
#   另外两个文件(unet_dr2d_condition.py 和 unet_dr2d_blocks.py)只负责把参数传到这里。
#
#   这个文件分三个部分:
#     第一部分  原 64-372 行    类 TransformerDR2DModel。它是注意力模块的外层,负责把特征图整理成序列,
#                              交给下面的 block 处理,再整理回特征图。这部分基本是 diffusers 的原版代码。
#     第二部分  原 376-626 行   类 BasicDRTransformerBlock。这是注意力模块的内部结构。IDArb 在这里
#                              改了第一个注意力层,并且新增了一个注意力层。
#     第三部分  原 628-770 行   这 143 行是作者全新写的,diffusers 里没有对应的代码。包括相机角度编码
#                              (CaPE)和真正执行"互相看"的类 XFormersJointAttnProcessor。
#
# 二、标签的意思
#   🔴 算法  这是创新点本身,需要理解。
#   ⚪ 管道  这是从 diffusers 复制来的常规代码,知道它的输入和输出是什么就够了。
#   ⚫ 跳过  没有被调用的代码,或者这次推理不会执行到的分支。
#
# 三、需要先知道的几个词
#   token         注意力机制处理的基本单位。在这里,特征图上的每一个像素位置就是一个 token。
#                 64×64 的特征图有 4096 个 token。
#   序列          把一组 token 排成一排。一张 64×64 的特征图整理之后就是一条长度为 4096 的序列。
#   通道数 C      每个 token 是一个向量,向量的长度叫通道数。在这个文件里 C 是 320、640 或 1280。
#   注意力        让一条序列里的每个 token 去看同一条序列里的其他 token,并从它们那里收集信息。
#                 关键在于:只有在同一条序列里的 token 才能互相看。IDArb 就是靠"重新分组序列"
#                 来控制谁能看谁。
#   Q、K、V       注意力计算里的三个中间量。你只需要知道:Q 和 K 的点积决定"看谁看得多",
#                 V 是被收集的信息。
#   rearrange     einops 库的函数,按照一个字符串描述的规则重新排列张量的维度。
#                 例如 '(b t) l c -> b (t l) c' 的意思是:把第一维拆成 b 和 t 两层,再把 t 合并进第二维。
#   processor     在 diffusers 里,注意力层(Attention 类)只保存权重,真正的计算步骤写在一个叫
#                 processor 的对象里。换一个 processor 就换了一种计算方式,权重不变。
#
# 四、进入 block 的张量长什么样(这是理解后面所有 rearrange 的基础)
#
#   进入 block 的张量形状是 [B·Nv·Nd, L, C]。其中:
#     B   一批有几个物体。推理时是 1。
#     Nv  每个物体有几个视角。单图是 1,多视角默认是 4。
#     Nd  分量数,固定是 3(albedo、normal、material)。
#     L   token 数。64×64 的层是 4096。
#     C   通道数。
#
#   第一维的排列顺序是在 pipeline 里由 repeat_interleave(3) 决定的:
#     第 0 条  图0 的 albedo
#     第 1 条  图0 的 normal
#     第 2 条  图0 的 material
#     第 3 条  图1 的 albedo
#     第 4 条  图1 的 normal
#     ......
#   也就是说,同一张图的 3 个分量是连续排在一起的。
#
#   以 64×64 那一层为例:单图时进入 block 的张量是 [3, 4096, 320];四视角时是 [12, 4096, 320]。
#
# 五、算法行速查(原始行号。插入注释之后编辑器里的行号会变,所以注释里一律写"原 N 行")
#   原 441-460   attn1 改用 CustomJointAttention
#   原 462-479   新增 attn_mv,并把它的输出层初始化为 0
#   原 545-555   调用 attn1,打开"跨分量"开关
#   原 569-584   调用 attn_mv,打开"跨视角"开关,并传入相机位姿
#   原 629-665   相机角度编码 CaPE 的三个函数
#   原 668-674   CustomJointAttention:把计算方式换成 XFormersJointAttnProcessor
#   原 706-710   计算之前,按"跨分量"或"跨视角"重新分组
#   原 741-743   用相机角度旋转 Q 和 K
#   原 757-760   计算之后,恢复原来的分组
# ------------------------------------------------------------------------------
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
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.embeddings import ImagePositionalEmbeddings
from diffusers.utils import BaseOutput, deprecate
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import FeedForward, AdaLayerNorm, AdaLayerNormZero, Attention
from diffusers.models.embeddings import PatchEmbed
from diffusers.models.lora import LoRACompatibleConv, LoRACompatibleLinear
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils.import_utils import is_xformers_available

# 🔴 einops 库。后面所有"重新分组"的操作都是用它的 rearrange 和 repeat 完成的。
from einops import rearrange, repeat


# 🔴 检测 xformers 是否安装。跨分量和跨视角注意力只写了 xformers 版本,所以 xformers 是必需的。
#   如果没有安装,程序不会报错,但结果是错的。原因见原 668 行的注释。
if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None

# ⚫ 这个函数在整个仓库里没有被调用过,可以跳过。
def my_repeat(tensor, num_repeats):
    """
    Repeat a tensor along a given dimension
    """
    if len(tensor.shape) == 3:
        return repeat(tensor,  "b d c -> (b v) d c", v=num_repeats)
    elif len(tensor.shape) == 4:
        return repeat(tensor,  "a b d c -> (a v) b d c", v=num_repeats)


@dataclass
class TransformerDR2DModelOutput(BaseOutput):
    """
    The output of [`Transformer2DModel`].

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` or `(batch size, num_vector_embeds - 1, num_latent_pixels)` if [`Transformer2DModel`] is discrete):
            The hidden states output conditioned on the `encoder_hidden_states` input. If discrete, returns probability
            distributions for the unnoised latent pixels.
    """

    sample: torch.FloatTensor


class TransformerDR2DModel(ModelMixin, ConfigMixin):
    """
    A 2D Transformer model for image-like data.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        in_channels (`int`, *optional*):
            The number of channels in the input and output (specify if the input is **continuous**).
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        sample_size (`int`, *optional*): The width of the latent images (specify if the input is **discrete**).
            This is fixed during training since it is used to learn a number of position embeddings.
        num_vector_embeds (`int`, *optional*):
            The number of classes of the vector embeddings of the latent pixels (specify if the input is **discrete**).
            Includes the class for the masked latent pixel.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to use in feed-forward.
        num_embeds_ada_norm ( `int`, *optional*):
            The number of diffusion steps used during training. Pass if at least one of the norm_layers is
            `AdaLayerNorm`. This is fixed during training since it is used to learn a number of embeddings that are
            added to the hidden states.

            During inference, you can denoise for up to but not more steps than `num_embeds_ada_norm`.
        attention_bias (`bool`, *optional*):
            Configure if the `TransformerBlocks` attention should contain a bias parameter.
    """

    # ==============================================================================
    # 第一部分:类 TransformerDR2DModel(原 64-372 行)
    #
    #   ⚪ 这个类基本上是 diffusers 的 Transformer2DModel 原样复制。它做三件事:
    #     1. 把输入的特征图 [N, C, h, w] 整理成序列 [N, h·w, C](原 299-316 行)。
    #     2. 依次经过 num_layers 个 BasicDRTransformerBlock(原 318-330 行)。这个模型里 num_layers 是 1。
    #     3. 把序列整理回特征图 [N, C, h, w](原 333-341 行)。
    #   IDArb 只改了三处:__init__ 多了三个参数(原 95、115、116 行);构造 block 时用了自己的类
    #   (原 196 行);forward 多接了 pose 和 num_views 两个参数并传下去(原 237-238 和 324-325 行)。
    # ==============================================================================
    @register_to_config
    def __init__(
        self,
        # 🔴 num_domains 是分量数,推理时是 3。它是第一个位置参数,所以创建这个类时第一个传入的值就是它。
        num_domains: int,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        sample_size: Optional[int] = None,
        num_vector_embeds: Optional[int] = None,
        patch_size: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        norm_type: str = "layer_norm",
        norm_elementwise_affine: bool = True,
        # 🔴 两个开关。attn1_cd 打开跨分量注意力,attn_mid_cv 打开跨视角注意力。训练好的模型里两个都是 True。
        #   cd 是 cross-domain 的缩写,cv 是 cross-view 的缩写。
        #   attn_mid_cv 里的 mid 指的是"插在 block 的中间",和 UNet 的 mid_block 没有关系。
        attn1_cd: bool = False,
        attn_mid_cv: bool = False,
    ):
        super().__init__()
        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        # 1. Transformer2DModel can process both standard continuous images of shape `(batch_size, num_channels, width, height)` as well as quantized image embeddings of shape `(batch_size, num_image_vectors)`
        # Define whether input is continuous or discrete depending on configuration
        self.is_input_continuous = (in_channels is not None) and (patch_size is None)
        self.is_input_vectorized = num_vector_embeds is not None
        self.is_input_patches = in_channels is not None and patch_size is not None

        if norm_type == "layer_norm" and num_embeds_ada_norm is not None:
            deprecation_message = (
                f"The configuration file of this model: {self.__class__} is outdated. `norm_type` is either not set or"
                " incorrectly set to `'layer_norm'`.Make sure to set `norm_type` to `'ada_norm'` in the config."
                " Please make sure to update the config accordingly as leaving `norm_type` might led to incorrect"
                " results in future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it"
                " would be very nice if you could open a Pull request for the `transformer/config.json` file"
            )
            deprecate("norm_type!=num_embeds_ada_norm", "1.0.0", deprecation_message, standard_warn=False)
            norm_type = "ada_norm"

        if self.is_input_continuous and self.is_input_vectorized:
            raise ValueError(
                f"Cannot define both `in_channels`: {in_channels} and `num_vector_embeds`: {num_vector_embeds}. Make"
                " sure that either `in_channels` or `num_vector_embeds` is None."
            )
        elif self.is_input_vectorized and self.is_input_patches:
            raise ValueError(
                f"Cannot define both `num_vector_embeds`: {num_vector_embeds} and `patch_size`: {patch_size}. Make"
                " sure that either `num_vector_embeds` or `num_patches` is None."
            )
        elif not self.is_input_continuous and not self.is_input_vectorized and not self.is_input_patches:
            raise ValueError(
                f"Has to define `in_channels`: {in_channels}, `num_vector_embeds`: {num_vector_embeds}, or patch_size:"
                f" {patch_size}. Make sure that `in_channels`, `num_vector_embeds` or `num_patches` is not None."
            )

        # 2. Define input layers
        if self.is_input_continuous:
            self.in_channels = in_channels

            self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
            if use_linear_projection:
                self.proj_in = LoRACompatibleLinear(in_channels, inner_dim)
            else:
                self.proj_in = LoRACompatibleConv(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        elif self.is_input_vectorized:
            assert sample_size is not None, "Transformer2DModel over discrete input must provide sample_size"
            assert num_vector_embeds is not None, "Transformer2DModel over discrete input must provide num_embed"

            self.height = sample_size
            self.width = sample_size
            self.num_vector_embeds = num_vector_embeds
            self.num_latent_pixels = self.height * self.width

            self.latent_image_embedding = ImagePositionalEmbeddings(
                num_embed=num_vector_embeds, embed_dim=inner_dim, height=self.height, width=self.width
            )
        elif self.is_input_patches:
            assert sample_size is not None, "Transformer2DModel over patched input must provide sample_size"

            self.height = sample_size
            self.width = sample_size

            self.patch_size = patch_size
            self.pos_embed = PatchEmbed(
                height=sample_size,
                width=sample_size,
                patch_size=patch_size,
                in_channels=in_channels,
                embed_dim=inner_dim,
            )

        # 3. Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                # 🔴 这里用 IDArb 自己写的 BasicDRTransformerBlock 替换了 diffusers 原版的 BasicTransformerBlock,并把 num_domains 和两个开关传进去。
                BasicDRTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    num_domains,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=norm_elementwise_affine,
                    attn1_cd=attn1_cd,
                    attn_mid_cv=attn_mid_cv,
                )
                for d in range(num_layers)
            ]
        )

        # 4. Define output layers
        self.out_channels = in_channels if out_channels is None else out_channels
        if self.is_input_continuous:
            # TODO: should use out_channels for continuous projections
            if use_linear_projection:
                self.proj_out = LoRACompatibleLinear(inner_dim, in_channels)
            else:
                self.proj_out = LoRACompatibleConv(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
        elif self.is_input_vectorized:
            self.norm_out = nn.LayerNorm(inner_dim)
            self.out = nn.Linear(inner_dim, self.num_vector_embeds - 1)
        elif self.is_input_patches:
            self.norm_out = nn.LayerNorm(inner_dim, elementwise_affine=False, eps=1e-6)
            self.proj_out_1 = nn.Linear(inner_dim, 2 * inner_dim)
            self.proj_out_2 = nn.Linear(inner_dim, patch_size * patch_size * self.out_channels)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        # 🔴 forward 新增的两个参数。pose 是相机位姿,形状 [B, Nv, 4];num_views 是视角数。它们是从 UNet 一路传到这里的。
        pose: Optional[torch.FloatTensor] = None,
        num_views: int = None,
        timestep: Optional[torch.LongTensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        """
        The [`Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.LongTensor` of shape `(batch size, num latent pixels)` if discrete, `torch.FloatTensor` of shape `(batch size, channel, height, width)` if continuous):
                Input `hidden_states`.
            encoder_hidden_states ( `torch.FloatTensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `torch.LongTensor`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            class_labels ( `torch.LongTensor` of shape `(batch size, num classes)`, *optional*):
                Used to indicate class labels conditioning. Optional class labels to be applied as an embedding in
                `AdaLayerZeroNorm`.
            encoder_attention_mask ( `torch.Tensor`, *optional*):
                Cross-attention mask applied to `encoder_hidden_states`. Two formats supported:

                    * Mask `(batch, sequence_length)` True = keep, False = discard.
                    * Bias `(batch, 1, sequence_length)` 0 = keep, -10000 = discard.

                If `ndim == 2`: will be interpreted as a mask, then converted into a bias consistent with the format
                above. This bias will be added to the cross-attention scores.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None and attention_mask.ndim == 2:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 1. Input
        if self.is_input_continuous:
            batch, _, height, width = hidden_states.shape
            residual = hidden_states

            hidden_states = self.norm(hidden_states)
            if not self.use_linear_projection:
                hidden_states = self.proj_in(hidden_states)
                inner_dim = hidden_states.shape[1]
                hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
            else:
                inner_dim = hidden_states.shape[1]
                hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
                hidden_states = self.proj_in(hidden_states)
        elif self.is_input_vectorized:
            hidden_states = self.latent_image_embedding(hidden_states)
        elif self.is_input_patches:
            hidden_states = self.pos_embed(hidden_states)

        # 2. Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                # 🔴 把 num_views 和 pose 继续传给 block。注意从这里开始 pose 改名叫 posemb,后面搜代码时两个名字都要搜。
                num_views=num_views,
                posemb=pose,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=class_labels,
            )

        # 3. Output
        if self.is_input_continuous:
            if not self.use_linear_projection:
                hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
                hidden_states = self.proj_out(hidden_states)
            else:
                hidden_states = self.proj_out(hidden_states)
                hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()

            output = hidden_states + residual
        elif self.is_input_vectorized:
            hidden_states = self.norm_out(hidden_states)
            logits = self.out(hidden_states)
            # (batch, self.num_vector_embeds - 1, self.num_latent_pixels)
            logits = logits.permute(0, 2, 1)

            # log(p(x_0))
            output = F.log_softmax(logits.double(), dim=1).float()
        elif self.is_input_patches:
            # TODO: cleanup!
            conditioning = self.transformer_blocks[0].norm1.emb(
                timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
            shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
            hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
            hidden_states = self.proj_out_2(hidden_states)

            # unpatchify
            height = width = int(hidden_states.shape[1] ** 0.5)
            hidden_states = hidden_states.reshape(
                shape=(-1, height, width, self.patch_size, self.patch_size, self.out_channels)
            )
            hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
            output = hidden_states.reshape(
                shape=(-1, self.out_channels, height * self.patch_size, width * self.patch_size)
            )

        if not return_dict:
            return (output,)

        return TransformerDR2DModelOutput(sample=output)


# ==============================================================================
# 第二部分:类 BasicDRTransformerBlock(原 376-626 行)—— 一个注意力模块的内部结构
#
#   diffusers 原版的 block 有三段,按顺序执行:
#     第 1 段  attn1   自注意力:一条序列里的 token 互相看。
#     第 2 段  attn2   交叉注意力:token 去看文字向量。在 IDArb 里,文字向量就是任务词的向量。
#     第 3 段  ff      前馈层:对每个 token 单独做一次变换。
#   每一段的输出都会加回到输入上(写法是 hidden_states = attn_output + hidden_states)。
#   这叫残差连接,作用是让每一段只需要学"在原来的基础上改多少",训练更稳定。
#
#   IDArb 把它改成了四段:
#     第 1 段    attn1     改成跨分量注意力:同一张图的 3 个分量拼成一条序列,互相看。
#     新增一段  attn_mv   跨视角注意力:同一个分量的 Nv 个视角拼成一条序列,互相看,并用相机角度调整。
#     第 2 段    attn2     没有改动,仍然是看任务词的向量。
#     第 3 段    ff        没有改动。
# ==============================================================================
@maybe_allow_in_graph
class BasicDRTransformerBlock(nn.Module):
    r"""
    A basic Transformer block.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        only_cross_attention (`bool`, *optional*):
            Whether to use only cross-attention layers. In this case two cross attention layers are used.
        double_self_attention (`bool`, *optional*):
            Whether to use two self-attention layers. In this case no cross attention layers are used.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm (:
            obj: `int`, *optional*): The number of diffusion steps used during training. See `Transformer2DModel`.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        # 🔴 num_domains 和下面 414-415 行的两个开关,从 TransformerDR2DModel 传到了这里。
        num_domains: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        final_dropout: bool = False,
        attn1_cd: bool = False,
        attn_mid_cv: bool = False,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention

        self.attn1_cd = attn1_cd
        self.attn_mid_cv = attn_mid_cv

        self.use_ada_layer_norm_zero = (num_embeds_ada_norm is not None) and norm_type == "ada_norm_zero"
        self.use_ada_layer_norm = (num_embeds_ada_norm is not None) and norm_type == "ada_norm"

        if norm_type in ("ada_norm", "ada_norm_zero") and num_embeds_ada_norm is None:
            raise ValueError(
                f"`norm_type` is set to {norm_type}, but `num_embeds_ada_norm` is not defined. Please make sure to"
                f" define `num_embeds_ada_norm` if setting `norm_type` to {norm_type}."
            )

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn --> Joint-Domain-Attn
        if self.use_ada_layer_norm:
            self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm)
        elif self.use_ada_layer_norm_zero:
            self.norm1 = AdaLayerNormZero(dim, num_embeds_ada_norm)
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)

        # 🔴 第 1 段 attn1 的构造。
        #   如果开关 attn1_cd 是 True,就用 CustomJointAttention;否则用 diffusers 原版的 Attention。
        #   请对比两个分支传的参数:它们完全一样。也就是说,两种情况下这一层的权重形状完全相同。
        #   CustomJointAttention 和原版唯一的区别是计算方式不同,见原 668 行。
        #   因为权重形状没变,所以这一层可以直接使用 Stable Diffusion 预训练好的权重,不需要从头训练。
        if self.attn1_cd:
            self.attn1 = CustomJointAttention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                cross_attention_dim=cross_attention_dim if only_cross_attention else None,
                upcast_attention=upcast_attention,
            )
        else:
            self.attn1 = Attention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                cross_attention_dim=cross_attention_dim if only_cross_attention else None,
                upcast_attention=upcast_attention
            )
        
        # 🔴 新增的一段:attn_mv,跨视角注意力。
        #   这是 Stable Diffusion 里没有的、全新的一层,它的权重是训练时从头学出来的。
        #   训练好的模型里,UNet 的 16 个注意力模块每个都有一个 attn_mv。
        #   原 478-479 行把它的输出层 to_out[0] 的权重和偏置全部设为 0。这样做的原因是:
        #   刚开始训练时,这一层的输出恒等于 0,加回到输入上等于什么都没做,模型的行为和原版 SD 完全一样;
        #   之后训练过程再慢慢让它学出有用的东西。这是给预训练模型加新层时的常用做法,可以避免新层一开始就把模型搞乱。
        # Add multi-view attention
        if self.attn_mid_cv:
            self.attn_mv = CustomJointAttention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                cross_attention_dim=cross_attention_dim if only_cross_attention else None,
                upcast_attention=upcast_attention,
            )
            self.norm_mv = (
                AdaLayerNorm(dim, num_embeds_ada_norm)
                if self.use_ada_layer_norm
                else nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
            )
            nn.init.zeros_(self.attn_mv.to_out[0].weight.data)
            nn.init.zeros_(self.attn_mv.to_out[0].bias.data)

        # 2. Cross-Attn
        if cross_attention_dim is not None or double_self_attention:
            # We currently only use AdaLayerNormZero for self attention where there will only be one attention block.
            # I.e. the number of returned modulation chunks from AdaLayerZero would not make sense if returned during
            # the second cross attention block.
            self.norm2 = (
                AdaLayerNorm(dim, num_embeds_ada_norm)
                if self.use_ada_layer_norm
                else nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
            )
            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim if not double_self_attention else None,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )  # is self-attn if encoder_hidden_states is none
        else:
            self.norm2 = None
            self.attn2 = None

        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn, final_dropout=final_dropout)

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

        self.num_domains = num_domains

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    # ------------------------------------------------------------------------------
    # 🔴 block 的 forward:数据实际流过四段的地方。
    #   参数里 posemb 是相机位姿,num_views 是视角数。其余参数是 diffusers 原版就有的。
    # ------------------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
        posemb: Optional[torch.FloatTensor] = None,
        num_views: Optional[int] = None,
    ):
        # ⚪ 作者自己注明:新的注意力不支持 attention_mask。推理时它确实是 None。
        assert attention_mask is None # not supported yet
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 1. Self-Attention
        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        else:
            norm_hidden_states = self.norm1(hidden_states)

        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}

        # 🔴 第 1 段:调用 attn1。
        #   开关 attn1_cd 是 True 时,比原版多传了四个参数:num_d=3(分量数)、num_v(视角数)、cd_attn=True、cv_attn=False。
        #   这四个参数会一路传到 XFormersJointAttnProcessor(原 682 行)。cd_attn=True 的意思是按"跨分量"的方式分组。
        #   这里的 encoder_hidden_states 传的是 None(因为 only_cross_attention 是 False),所以这一层不看任务词。
        if self.attn1_cd:
            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                attention_mask=attention_mask,
                num_d=self.num_domains,
                num_v=num_views,
                cd_attn=True,
                cv_attn=False,
                **cross_attention_kwargs,
            )
        else:
            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )

        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        # 残差连接:把注意力的输出加回到输入上。
        hidden_states = attn_output + hidden_states

        # 🔴 新增的一段:调用 attn_mv。
        #   和上面的区别是 cd_attn=False、cv_attn=True(按"跨视角"分组),并且多传了 posemb(相机位姿)。
        #   这是整个模型里相机位姿唯一被使用的地方。
        # Multi-view attention
        if self.attn_mid_cv:
            norm_hidden_states = (
                self.norm_mv(hidden_states, timestep) if self.use_ada_layer_norm else self.norm_mv(hidden_states)
            )
            attn_output = self.attn_mv(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                attention_mask=attention_mask,
                num_d=self.num_domains,
                num_v=num_views,
                cd_attn=False,
                cv_attn=True,
                posemb=posemb,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states 

        # ⚪ 第 2 段:attn2,交叉注意力。这段代码没有改动。
        #   这里的 encoder_hidden_states 就是任务词的向量,形状 [B·Nv·3, 77, 1024]。
        #   每条序列去看自己对应的那个任务词,从而知道自己应该生成 albedo、normal 还是 material。
        # 2. Cross-Attention
        if self.attn2 is not None:
            norm_hidden_states = (
                self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
            )

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # ⚪ 第 3 段:前馈层。没有改动。
        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)

        if self.use_ada_layer_norm_zero:
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            if norm_hidden_states.shape[self._chunk_dim] % self._chunk_size != 0:
                raise ValueError(
                    f"`hidden_states` dimension to be chunked: {norm_hidden_states.shape[self._chunk_dim]} has to be divisible by chunk size: {self._chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
                )

            num_chunks = norm_hidden_states.shape[self._chunk_dim] // self._chunk_size
            ff_output = torch.cat(
                [self.ff(hid_slice) for hid_slice in norm_hidden_states.chunk(num_chunks, dim=self._chunk_dim)],
                dim=self._chunk_dim,
            )
        else:
            ff_output = self.ff(norm_hidden_states)

        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = ff_output + hidden_states

        return hidden_states

# ==============================================================================
# 第三部分:作者全新写的 143 行(原 628-770 行)
#
#   包括两组代码:
#     (1) 原 629-665 行  相机角度编码 CaPE。CaPE 是 Camera Pose Encoding 的缩写。
#     (2) 原 668-770 行  类 CustomJointAttention 和类 XFormersJointAttnProcessor。
#                        后者是真正执行"跨分量"和"跨视角"注意力的地方。
#
#   先说 CaPE 是什么,以及一个需要纠正的说法。
#
#   pipeline 和 config.json 里有 'e_de_da_sincos'、'4DoF'、projection_class_embeddings_input_dim 这些字段,
#   看起来像是"相机位姿被编码之后加进了时间步向量"。实际上不是。这些字段是没有被使用的配置,模型代码不读它们。
#   证据是:config.json 里 class_embed_type 是 null,所以 UNet 里的 class_embedding 是 None,对应的代码不会执行;
#   训练好的权重里也没有任何 class_embedding 的参数。
#
#   相机位姿真正的用法是 CaPE:把注意力里的 Q 和 K 各自旋转一个角度,角度由相机位姿决定。
#   这样做的效果是:两个 token 之间的注意力分数只和它们所属视角的相机角度之差有关。
#   换句话说,模型看到的是"两个视角之间的相对角度",而不是每个视角的绝对角度。
#   这和大语言模型里的 RoPE(旋转位置编码)是同一个原理,只是把"第几个词"换成了"哪个相机"。
#
#   一个重要的推论:如果所有视角的位姿相同,那么旋转的角度也相同,旋转对注意力分数就没有任何影响。
#   单图推理时 Nv 是 1,位姿是一个写死的常量;多视角不加 --cam 时,所有视角用的也是同一个默认位姿。
#   所以这两种情况下 CaPE 实际上不起作用。只有加了 --cam、并且 transforms.json 里有真实的相机矩阵时,它才有效果。
# ==============================================================================
# 4DoF CaPE
# 🔴 CaPE 的第一个函数。
#   把向量最后一维的数两个两个分组,每一组看成平面上的一个点 (a, b),然后把它变成 (-b, a)。
#   在坐标纸上画一下就能看到,(a, b) 变成 (-b, a) 正好是逆时针旋转 90 度。
#   有了"转 90 度"这个操作,就可以用公式 v·cos(m) + rot90(v)·sin(m) 把向量 v 旋转任意角度 m。
def rotate_every_two(x):
    x = rearrange(x, '... (d j) -> ... d j', j=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d j -> ... (d j)')

# 🔴 CaPE 的第二个函数:把 4 个位姿数字扩展到和特征向量一样长。
#   x 是特征,形状 [N, L, C];p 是位姿,形状 [N, L, 4]。输出形状 [N, L, C],做法是把 4 个数每个重复 C/4 次。
#   所以输出向量的前 1/4 全是仰角,接下来 1/4 全是方位角,再 1/4 全是距离编码,最后 1/4 全是 0。
#   最后那 1/4 的角度是 0,意味着特征向量的最后 1/4 不会被旋转,保持原样。
#   assert 那一行要求 C 能被 8 整除。C 是 320、640 或 1280,都满足。
def cape(x, p):
    d, l, n = x.shape[-1], p.shape[-2], p.shape[-1]
    assert d % (2 * n) == 0
    m = repeat(p, 'b l n -> b l (n k)', k=d // n)
    return m

# 🔴 CaPE 的主函数:用位姿旋转 Q 和 K。
#   p1 和 p2 分别是 Q 和 K 对应的位姿(这里传进来的是同一个);qq 和 kk 是 Q 和 K。
#   原 659-660 行先用 cape 把位姿扩展成角度向量 m1 和 m2。
#   原 662-663 行用公式 v·cos(m) + rot90(v)·sin(m) 把 Q 旋转 m1、把 K 旋转 m2。
#   旋转之后,Q 和 K 的点积只和 m1 减 m2 有关。这就是"只看相对角度"的来源。
def cape_embed(p1, p2, qq, kk):
    """
    Embed camera position encoding into attention map
    Args:
        p1: query pose  b, l_q, pose_dim
        p2: key pose    b, l_k, pose_dim
        qq: query feature map   b, l_q, feature_dim
        kk: key feature map    b, l_k, feature_dim

    Returns: cape embedded attention map    b, l_q, l_k

    """
    assert p1.shape[-1] == p2.shape[-1]
    assert qq.shape[-1] == kk.shape[-1]
    assert p1.shape[0] == p2.shape[0] == qq.shape[0] == kk.shape[0]
    assert p1.shape[1] == qq.shape[1]
    assert p2.shape[1] == kk.shape[1]

    m1 = cape(qq, p1)
    m2 = cape(kk, p2)

    q = (qq * m1.cos()) + (rotate_every_two(qq) * m1.sin())
    k = (kk * m2.cos()) + (rotate_every_two(kk) * m2.sin())

    return q, k


# 🔴 类 CustomJointAttention:继承 diffusers 的 Attention,只改写了一个方法。
#   main.py 调用 pipeline.unet.enable_xformers_memory_efficient_attention() 时,diffusers 会对模型里的每一个
#   注意力层调用 set_use_memory_efficient_attention_xformers()。这个类把该方法改成:不管传进来什么值,
#   一律把计算方式换成下面的 XFormersJointAttnProcessor。
#
#   这就是 xformers 必须安装的真正原因。如果没有 xformers,main.py 会跳过上面那个调用,注意力层就还是
#   diffusers 默认的计算方式。默认的计算方式不认识 num_d、num_v、cd_attn、cv_attn、posemb 这几个参数,
#   diffusers 会把它们直接丢掉,只打印一条警告。程序照样运行、照样输出图片,但是跨分量和跨视角注意力
#   都没有发生,结果是错的。所以 xformers 不是为了让程序更快,而是为了让程序算对。
class CustomJointAttention(Attention):
    def set_use_memory_efficient_attention_xformers(
        self, use_memory_efficient_attention_xformers: bool, *args, **kwargs
    ):
        processor = XFormersJointAttnProcessor()
        self.set_processor(processor)
        # print("using xformers attention processor")


# 🔴 类 XFormersJointAttnProcessor:真正执行"跨分量"和"跨视角"注意力的地方。
#   它是把 diffusers 的 XFormersAttnProcessor 复制过来改的。改动只有三处:
#     原 706-710 行  计算之前重新分组;
#     原 741-743 行  用相机位姿旋转 Q 和 K;
#     原 757-760 行  计算之后恢复分组。
#   其余代码(原 695-705、712-740、745-755、762-770 行)是标准的注意力流程,和原版一样,不需要细看。
#   下面只解释这三处。
class XFormersJointAttnProcessor:
    r"""
    Default processor for performing attention-related computations.
    """

    # 🔴 __call__ 比原版多了五个参数:num_d(分量数)、num_v(视角数)、cd_attn 和 cv_attn(两个开关)、posemb(相机位姿)。
    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        num_d=3,
        num_v=1,
        cd_attn=False,
        cv_attn=False,
        posemb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        # 🔴 计算之前的重新分组。
        #   此时 hidden_states 的形状是 [B·Nv·Nd, L, C],第一维的排列是"同一张图的 3 个分量连续"。
        #
        #   跨分量(cd_attn 是 True 时,原 707 行):
        #     '(b t) l c -> b (t l) c',t=num_d
        #     意思是:把第一维拆成 b 和 t 两层,t 等于 3。因为同一张图的 3 个分量本来就是连续的,
        #     所以拆开之后 b 对应"第几张图",t 对应"第几个分量"。然后把 t 合并进第二维,
        #     也就是把 3 个分量的序列首尾相接,拼成一条长度为 3L 的序列。
        #     单图时:[3, 4096, 320] 变成 [1, 12288, 320]。四视角时:[12, 4096, 320] 变成 [4, 12288, 320]。
        #     之后做注意力时,albedo 的每个 token 既能看自己序列里的 token,也能看 normal 和 material 的 token。
        #     这就是论文里的 cross-component attention。
        #
        #   跨视角(cv_attn 是 True 时,原 709 行):
        #     '(b m t) l c -> (b t) (m l) c',t=num_d,m=num_v
        #     意思是:把第一维拆成 b、m、t 三层,t 是分量数,m 是视角数,b 是物体数。
        #     输出的第一维按 (b t) 分组,也就是"同一个物体、同一个分量"归为一组;
        #     第二维按 (m l) 合并,也就是把这个分量的 Nv 个视角的序列首尾相接。
        #     四视角时:[12, 4096, 320] 变成 [3, 16384, 320]。单图时 Nv 是 1,形状不变。
        #     之后做注意力时,同一个分量的 4 个视角的 token 互相看。这就是论文里的 cross-view attention。
        #
        #   两种分组是互补的:跨分量时不同视角互相看不见,跨视角时不同分量互相看不见。
        #   两层一前一后,信息就能在"视角"和"分量"两个方向上都传播。
        if cd_attn:
            hidden_states = rearrange(hidden_states, '(b t) l c -> b (t l) c', t=num_d).contiguous()
        if cv_attn:
            hidden_states = rearrange(hidden_states, '(b m t) l c -> (b t) (m l) c', t=num_d, m=num_v).contiguous()
            # assert posemb is None

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        # from yuancheng; here attention_mask is None
        if attention_mask is not None:
            # expand our mask's singleton query_tokens dimension:
            #   [batch*heads,            1, key_tokens] ->
            #   [batch*heads, query_tokens, key_tokens]
            # so that it can be added as a bias onto the attention scores that xformers computes:
            #   [batch*heads, query_tokens, key_tokens]
            # we do this explicitly because xformers doesn't broadcast the singleton dimension for us.
            _, query_tokens, _ = hidden_states.shape
            attention_mask = attention_mask.expand(-1, query_tokens, -1)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # 🔴 用相机位姿旋转 Q 和 K。只在 attn_mv 里执行,因为只有它传了 posemb。
        #   原 742 行先把 posemb 从 [B, Nv, 4] 扩展成 [B·Nd, Nv·L, 4],让每个 token 都对应上自己所属视角的位姿。
        #   注意:这一行字符串里的字母和原 709 行的含义不一样。这里 posemb 的形状是 [B, Nv, 4],
        #   所以 't' 对应的是视角数 Nv(不是分量数),'m' 对应的是分量数 Nd(因为写了 m=num_d)。
        #   结果的第一维按 (b m) 排,和此时 query 的第一维 (b t) 一致;第二维按 (t l) 排,前 L 个 token 用视角 0 的位姿,
        #   接下来 L 个用视角 1 的位姿,依此类推,和原 709 行里 (m l) 的顺序一致。
        #   原 743 行调用 cape_embed 完成旋转。
        if posemb is not None:
            posemb = repeat(posemb, 'b t d -> (b m) (t l) d', l=query.shape[1]//num_v, m=num_d)
            query, key = cape_embed(posemb, posemb, query, key)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        # 🔴 计算完成之后,用相反的 rearrange 把分组恢复成原来的 [B·Nv·Nd, L, C],交回给 block。
        if cd_attn:
            hidden_states = rearrange(hidden_states, 'b (t l) c -> (b t) l c', t=num_d).contiguous()
        if cv_attn:
            hidden_states = rearrange(hidden_states, '(b t) (m l) c -> (b m t) l c', t=num_d, m=num_v).contiguous()

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        
        return hidden_states
