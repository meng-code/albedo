# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  main.py —— IDArb 推理入口(学习注释版;代码一行未改,原始版本见 git)               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# 【这个文件在干什么】
#   你敲的 `python main.py --data_dir example/single --output_dir output/single --input_type single`
#   执行的就是它。它是"总指挥":定参数 → 装模型 → 读数据 → 循环调 pipeline 做扩散 → 把 3 个分量拼图存盘。
#   算法本体不在这里(在 pipeline / UNet / transformer 里),这里 95% 是 ⚪ 管道。
#
# 【标签说明】
#   🔴 关键 —— 后面算法要用的量在这里产生 / 形状在这里变,必须搞清
#   ⚪ 管道 —— 知道"把什么变成什么"即可
#   ⚫ 废弃 —— 没用到的 import、遗留写法,跳过
#
# 【文件结构:5 个阶段】
#   注意:原 60 行起是"脚本顶层代码",python 一运行就从上到下顺序执行,没有 main 函数、没有 if __name__。
#   阶段① 原 60-66    argparse 解析命令行参数
#   阶段② 原 69-80    load_pipeline() 装 6 个零件 + 开 xformers + 搬上 GPU
#   阶段③ 原 82-96    建 Dataset / DataLoader,建输出目录
#   阶段④ 原 100-128  推理主循环:取 batch → 搬 GPU → 压扁 → 调 pipeline → 还原形状
#   阶段⑤ 原 20-34    save_image / reform_image:3 个分量拼成四连图存 PNG(函数定义在前、真正调用在原 131 行)
#
# 【全流程 9 站,本文件负责 ② 装车 和 ⑨ 存盘】
#   ① 读图 [custom_dataset]  ② 装车 [本文件阶段③④]  ③④⑤⑧ [pipeline]  ⑥ UNet  ⑦ 注意力  ⑨ 存盘 [本文件阶段⑤]
#
# 【形状链】(单图 B=1,Nv=1;括号内为四视角 Nv=4)
#   batch['imgs_in']    [B,Nv,3,512,512]              DataLoader 给的
#   flatten(0,1) 后     [B·Nv,3,512,512]    (4,...)   交给 pipeline
#   pipeline 返回       [B·Nv·3,3,512,512]  (12,...)  3 = Nd 个分量
#   view 后             [B,Nv,Nd,3,512,512]           还原分组
#   save_image 收到     [Nv,Nd,3,512,512]             一个物体的全部视角
#   reform_image 收到   [Nd,3,512,512]                一个视角的 3 个分量 → 拼成 [512, 4×512, 3] 四连图
#
# 【两个容易混的 3】
#   Nd=3 是"分量个数"(albedo / normal / material);后面那个 3 是"每个分量图自己的 RGB 通道数"。
# ──────────────────────────────────────────────────────────────────────────────
# ⚪ 标准库:argparse 解析命令行;os 处理路径/建目录;torch;PIL.Image 存图;numpy 拼图;version 比版本号
#   json 没用到(⚫)
import argparse
import os
import json
import torch
from PIL import Image
import numpy as np
from packaging import version

# ⚫ accelerate 的 Accelerator 没用到;DDPMScheduler 没用到(实际用的是 DDIM)。其余是 HuggingFace 的零件类型。
from accelerate import Accelerator
from diffusers.utils.import_utils import is_xformers_available
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor

# 🔴 IDArb 自己写的 4 个模块:魔改的 UNet(第 4 部分)、pipeline(第 3 部分)、单图 Dataset、多视角 Dataset(第 2 部分)
from idarbdiffusion.models.unet_dr2d_condition import UNetDR2DConditionModel
from idarbdiffusion.pipelines.pipeline_idarbdiffusion import IDArbDiffusionPipeline
from idarbdiffusion.data.custom_dataset import CustomDataset
from idarbdiffusion.data.custom_mv_dataset import CustomMVDataset


# ══════════════════════════════════════════════════════════════════════════════
# ⚪ 管道 · 阶段⑤ 之一:reform_image —— 把一个视角的 3 个分量拼成一张横向四连图
#   输入 img_nd:[Nd=3, 3, H, W] = "3 张照片,每张 3 层"
#     第 0 张 = albedo(3 层 = R/G/B)
#     第 1 张 = normal(3 层 = 法线的 X/Y/Z)
#     第 2 张 = material(3 层 = metallic / roughness / 全 0 占位)—— 不是真彩色图,是硬凑成 3 层的壳
#   输出:[H, 4W, 3] —— 从左到右 [albedo | normal | metallic | roughness],通道放最后(PIL 要的顺序)
#   为什么 material 只有 2 个有效层:底座是 SD,VAE 只会输出 3 通道图;金属度、粗糙度各只有 1 个数,
#   于是塞进前 2 层,第 3 层填 0(见训练代码 mv_dataset.py 原 338-340 行 "orm -> mro")。论文里没有"遮蔽"这个输出。
# ══════════════════════════════════════════════════════════════════════════════
def reform_image(img_nd):
    # 取第 0 张(albedo)和第 1 张(normal)。[0, ...] = "第一维取 0 号,其余维度全要" → 各 [3,H,W]
    albedo, normal = img_nd[0, ...], img_nd[1, ...]
    # 取第 2 张(material),[3,H,W]。变量名 mro 是训练素材 ORM 贴图格式的遗留叫法,o 那一层实际是 0
    mro = img_nd[2, ...]
    # 从 material 里切出第 0 层 metallic、第 1 层 roughness,各 [1,H,W]
    #   用切片 [:1] 而不是 [0]:切片保留"层"这一维(得 [1,H,W]),[0] 会压掉它(得 [H,W]),下一行 repeat 需要这一维
    mtl, rgh = mro[:1, ...], mro[1:2, ...]
    # 沿第 0 维(层)各复制成 3 层 → [3,H,W]。R=G=B 三层相同 = 灰度图;这样才能和 3 层的 albedo/normal 并排拼
    mtl, rgh = np.repeat(mtl, 3, axis=0), np.repeat(rgh, 3, axis=0)
    # 沿最后一维(宽 W)横向拼接 4 张 [3,H,W] → [3,H,4W]。这就是 output/ 里"四连图"的来源
    img_reform = np.concatenate([albedo, normal, mtl, rgh], axis=-1)
    # 维度顺序 [3,H,4W] → [H,4W,3]:PyTorch 习惯通道在前,图片库(PIL)要通道在后。只换排列,数值不变
    return img_reform.transpose(1, 2, 0)


# ══════════════════════════════════════════════════════════════════════════════
# ⚪ 管道 · 阶段⑤ 之二:save_image —— 一个物体的所有视角,逐个视角拼图存盘
#   输入 out:[Nv, Nd=3, 3, H, W](一个物体的全部视角);name:物体名(如 'bouquet')
#   输出文件:{name}_00.png, {name}_01.png ...(单图只有 _00)
# ══════════════════════════════════════════════════════════════════════════════
def save_image(out, name):
    # 第 0 维 = 视角数
    Nv = out.shape[0]
    # 逐个视角
    for i in range(Nv):
        # out[i] 是第 i 个视角 [Nd,3,H,W] → reform_image 拼成 [H,4W,3],数值 0~1
        img = reform_image(out[i])
        # 0~1 小数 → 0~255 整数(uint8 = 8 位无符号整数,图片文件的标准像素类型)。和 Dataset 里的 /255 对称
        img = (img * 255).astype(np.uint8)
        # numpy 数组 → PIL 图片对象 → 存 PNG。os.path.join 拼路径;f-string 的 {i:02d} 把整数补零到 2 位(00、01…)
        #   注意:这里用的 output_dir 是原 95 行定义的"全局变量"。函数定义时它还不存在,但真正调用(原 131 行)时已经有了,所以能用
        Image.fromarray(img).save(os.path.join(output_dir, f'{name}_{(i):02d}.png'))


# ══════════════════════════════════════════════════════════════════════════════
# ⚪ 管道 · 阶段②:load_pipeline —— 把 6 个零件从 HuggingFace 仓库 lizb6626/IDArb 下载/加载,拼成完整模型
#   .from_pretrained("lizb6626/IDArb", subfolder="xxx"):从该仓库的 xxx 子文件夹加载权重。
#   首次运行下载 5.4GB,之后缓存在 ~/.cache/huggingface/,不再重复下载。
#   6 个零件里真正要关心的只有 UNet(大脑)和 VAE(压缩/解压机);其余是 SD 架构带来的"陪跑零件"。
#     零件               作用
#     text_encoder      CLIP 文本编码器:把 5 个任务词编成向量(pipeline 原 251 行),这是它唯一的用处
#     tokenizer         把文字切成整数编号,给 text_encoder 用
#     feature_extractor CLIP 图像预处理器:pipeline 原 144 行用了但结果被覆盖,实际无用
#     vae               图像压缩机:512×512×3 ↔ 64×64×4 latent
#     scheduler         DDIM 去噪节奏表:决定每一步加/减多少噪,无可学习参数
#     unet              🔴 大脑:自定义的 UNetDR2DConditionModel(不是 diffusers 原版),第 4、5 部分讲
# ══════════════════════════════════════════════════════════════════════════════
def load_pipeline(path="lizb6626/IDArb"):
    """
    load pipeline from hub
    or load from local ckpts: pipeline = IDArbDiffusionPipeline.from_pretrained("./pipeckpts")
    """
    text_encoder = CLIPTextModel.from_pretrained(path, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(path, subfolder="tokenizer")
    feature_extractor = CLIPImageProcessor.from_pretrained(path, subfolder="feature_extractor")
    vae = AutoencoderKL.from_pretrained(path, subfolder="vae")
    scheduler = DDIMScheduler.from_pretrained(path, subfolder="scheduler")
    # 🔴 注意这里用的是 IDArb 自己的 UNetDR2DConditionModel(原 14 行 import 的),不是 diffusers 的标准 UNet —— 魔改就在这个类里
    unet = UNetDR2DConditionModel.from_pretrained(path, subfolder="unet")
    # 把 6 个零件装进 IDArbDiffusionPipeline(pipeline 文件的 __init__);safety_checker=None 关掉 SD 的内容安全检查
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


# ══════════════════════════════════════════════════════════════════════════════
# ⚪ 管道 · 阶段①:argparse —— 把命令行里的 --xxx 解析成 args.xxx
#   从这行起是脚本顶层代码,python main.py 一运行就顺序执行到文件尾。
#   type=str/int 规定类型;default 是不写时的默认值(所以裸跑 python main.py 也能跑);choices 限定可选值
# ══════════════════════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='example/single', help='Input data directory')
parser.add_argument('--input_type', type=str, default='single', choices=['single', 'multi'], help='Specify input type (single or multi)')
parser.add_argument('--num_views', type=int, default=4, help='Number of views')
# 开关型参数(action='store_true'):命令里写了 --cam 就是 True,不写就是 False。
#   它唯一的作用是 custom_mv_dataset.py 原 112 行那个 if:决定用 json 里的相机矩阵算位姿,还是用默认位姿
parser.add_argument('--cam', default=False, action='store_true', help='Enable to use camera pose')
parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
# 新增参数:权重的位置。默认是官方训练好的 HuggingFace 仓库;也可以填自己训练出来的本地目录,
#   例如 output/train_smoke/pipeckpts(train.py 训练结束时保存的完整模型,子文件夹结构和官方仓库一样)
parser.add_argument('--pretrained', type=str, default='lizb6626/IDArb', help='HF repo id or local pipeckpts dir')
# 解析完成。之后 args.data_dir、args.input_type、args.num_views、args.cam、args.output_dir 就是你敲的值
args = parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# ⚪ 管道 · 阶段②:装模型、开加速、搬 GPU
# ══════════════════════════════════════════════════════════════════════════════
# 半精度浮点数。只用来转换输入数据(原 106-107 行);模型权重加载时仍是 float32,靠原 113 行的 autocast 做混合精度
weight_dtype = torch.float16

# 调上面的 load_pipeline():下载/加载 6 个零件 → 完整 pipeline 对象
pipeline = load_pipeline(args.pretrained)

# xformers:加速 attention、省显存的库。IDArb 的跨视角/跨分量注意力只实现了 xformers 版本(第 5 部分),
#   所以它是硬依赖 —— 这就是你搭环境时必须验证 import xformers 成功的原因
if is_xformers_available():
    import xformers
    xformers_version = version.parse(xformers.__version__)
    pipeline.unet.enable_xformers_memory_efficient_attention()
    print(f'Use xformers version: {xformers_version}')

# 把整个模型(6 个零件 + 任务词查询表)从 CPU 内存搬到 GPU 显存。"cuda" = NVIDIA GPU。模型在 GPU 上才能快速算矩阵
pipeline.to("cuda")
print("Pipeline loaded successfully")

# ══════════════════════════════════════════════════════════════════════════════
# ⚪ 管道 · 阶段③:建数据集
#   Dataset:回答"一共几个样本 / 第 i 个长什么样"(实现见 custom_dataset.py / custom_mv_dataset.py)
#   单图版:一个样本 = 一张 PNG;多视角版:一个样本 = 一个文件夹(含多张图 + transforms.json)
# ══════════════════════════════════════════════════════════════════════════════
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

# DataLoader:把 Dataset 包一层,负责一批一批地吐样本(背后调 dataset.__getitem__(i)),并在最前面套一个 batch 维 B。
#   batch_size=1 → 一次处理 1 个样本(单图 = 1 张图;多视角 = 1 个文件夹的 4 张图);shuffle=False 不打乱(推理不需要)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

# 输出目录;makedirs 建目录,exist_ok=True 表示已存在也不报错
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

# 🔴 Nd = number of domains = 分量个数 = 3(albedo / normal / material)。原 127 行还原形状时用
Nd = 3

# ══════════════════════════════════════════════════════════════════════════════
# 🔴 阶段④:推理主循环 —— 每圈处理一个 batch
#   每圈 5 步:取出字典里的东西 → 搬 GPU → 拆形状 → 压扁 → 调 pipeline → 还原形状 → 存图
# ══════════════════════════════════════════════════════════════════════════════
# batch 是 Dataset.__getitem__ 返回的字典再套一个 B 维;enumerate 同时给出序号 i 和内容
for i, batch in enumerate(dataloader):

    # 按键从字典里取。这些键名是 Dataset 的 __getitem__ 定的:
    #   imgs_in    [B,Nv,3,512,512]  白底输入图(真正进网络的东西)
    #   imgs_mask  [B,Nv,1,512,512]  0/1 蒙版(物体=1 背景=0)。⚫ 推理时取出来但没传给 pipeline,是训练遗留物
    #   task_ids   [B,Nv,3]          每张图的"订单" [0,1,2] = 三个分量都要
    imgs_in, imgs_mask, task_ids = batch['imgs_in'], batch['imgs_mask'], batch['task_ids']
    # [B,Nv,4] 每张图的相机位姿 [仰角, 方位角, 距离编码, 0];单图 / 不加 --cam 时是默认值
    cam_pose = batch['pose']
    # 文件名(如 'bouquet'),存图命名用
    imgs_name = batch['data_name']

    # 转半精度 + 搬上 GPU。数据必须和模型在同一设备上才能一起算
    imgs_in = imgs_in.to(weight_dtype).to("cuda")
    cam_pose = cam_pose.to(weight_dtype).to("cuda")

    # 🔴 拆开 5 维形状:B=几个样本, Nv=几个视角, _=通道数(不关心), H/W=高宽。这几个数后面还原形状要用
    B, Nv, _, H, W = imgs_in.shape

    # 🔴 压扁:把"样本"和"视角"两个维度合并成一个"图片总数"维
    #   imgs_in  [B,Nv,3,H,W] → [B·Nv,3,H,W]     四视角:[1,4,...] → [4,...]
    #   task_ids [B,Nv,3]     → [B·Nv·3]         四视角:[1,4,3] → [12] = [0,1,2,0,1,2,0,1,2,0,1,2]
    #   为什么:网络不在乎"哪几张是一组",只要"一摞图";分组信息记在 B、Nv 两个数里,算完再用 view 还原(原 127 行)
    #   task_ids 用 flatten(0,2) 是因为它有 3 维要全压平,imgs_in 只压前两维
    imgs_in, imgs_mask, task_ids = imgs_in.flatten(0,1), imgs_mask.flatten(0,1), task_ids.flatten(0,2)

    # 自动混合精度(AMP):在这个 with 块里,卷积/矩阵乘自动用 float16 算,更快更省显存
    with torch.autocast("cuda"):
        # 🔴 整个推理最耗时的一行:调 pipeline 的 __call__(pipeline 文件原 258 行),内部跑 50 圈 UNet 去噪。参数:
        #   imgs_in        [B·Nv,3,512,512]  白底图
        #   task_ids       [B·Nv·3]          每团噪声要生成哪个分量(0/1/2)
        #   num_views=Nv                     第 5 部分跨视角注意力按它分组
        #   cam_pose       [B,Nv,4]          相机位姿
        #   guidance_scale=1.0               关掉 CFG(pipeline 里所有 CFG 分支都不走)
        #   output_type='pt'                 返回 torch tensor 而不是 PIL
        #   eta=1.0                          DDIM 随机性旋钮拧到最大(等价 DDPM)
        #   .images                          从返回的包装盒里取图 → [B·Nv·3, 3, 512, 512],数值 0~1
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

        # 🔴 还原分组:[B·Nv·3, 3,512,512] → [B, Nv, Nd=3, 3,512,512]。*out.shape[1:] 是"后面几维原样保留"
        #   这和原 111 行的 flatten 是一对:压扁 → 算 → 展开
        out = out.view(B, Nv, Nd, *out.shape[1:])
        # .detach() 切断梯度记录(推理不学习,省内存);.cpu() 从 GPU 搬回 CPU;.numpy() 转成 numpy 数组给 PIL 存图
        out = out.detach().cpu().numpy()

        # 逐个样本存图。out[i] 是第 i 个物体 [Nv,Nd,3,H,W],imgs_name[i] 是它的名字
        #   (⚫ 小瑕疵:这里的 i 覆盖了外层循环的 i,因为外层的 i 后面没再用,所以无害)
        for i in range(B):
            save_image(out[i], imgs_name[i])
