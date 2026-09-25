# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  custom_dataset.py —— 单图模式的数据集(学习注释版;代码一行未改,原始版本见 git)   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# 【这个文件在干什么】全流程"站① 读图":一张 PNG → 网络能吃的 tensor,并配上两张"便签"(task_ids、pose)。
#   main.py 阶段③ 用 CustomDataset(root_dir=...) 构造它;DataLoader 每圈在背后调 __getitem__(i) 取第 i 个样本。
#
# 【标签说明】 🔴 关键(必须搞清)  ⚪ 管道(知道把什么变成什么)  ⚫ 废弃(跳过)
#
# 【PyTorch Dataset 的"三件套"规矩】
#   __init__(...)       准备工作:数据在哪、有哪些文件
#   __len__()           一共几个样本?(DataLoader 靠它知道循环几圈)
#   __getitem__(index)  第 index 个样本长什么样?返回一个字典(键名和 main.py 里 batch['xxx'] 的 xxx 一一对应)
#   你自己不会手动调这三个,是 DataLoader 在背后调。
#
# 【一张图的旅程(load_image 6 步 + __getitem__ 打包)】以 bouquet.png(943×1024 RGBA)为例
#   ① 打开、长边缩到 512(短边等比)          943×1024 → 472×512
#   ② np.array、/255 变 0~1、assert 必须 RGBA  [512,472,4]
#   ③ 补 0 成正方形                          [512,512,4]
#   ④ 第 4 通道 alpha > 0.5 → 0/1 蒙版         alpha [512,512,1]
#   ⑤ 背景涂白:img*alpha + 白*(1-alpha)      img [512,512,3]
#   ⑥ numpy → torch
#   __getitem__:permute 通道到最前 [3,512,512] → 配 task_ids=[0,1,2]、默认 pose → [None,...] 加 Nv=1 维 → 装字典
#
# 【交出去的字典】(DataLoader 再在最前面套一个 B 维)
#   'imgs_in'   [1,3,512,512]  白底 RGB 图 —— 真正进网络的东西
#   'imgs_mask' [1,1,512,512]  0/1 蒙版 —— ⚫ 推理时没用到(训练算 loss 用)
#   'task_ids'  [1,3] = [[0,1,2]]  三个分量都要
#   'pose'      [1,4]          默认相机位姿
#   'data_name' 'bouquet'      文件名去掉 .png
# ──────────────────────────────────────────────────────────────────────────────
# ⚪ import:numpy、torch、Dataset 基类、PIL 读图、os 路径。Path 和 json 没用到(⚫)
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from PIL import Image
import os
# ⚫ 给 OpenCV 开 EXR 格式支持的环境变量,本文件不用 OpenCV,无作用(训练代码遗留)
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"


# ══════════════════════════════════════════════════════════════════════════════
# 🔴 CustomDataset(Dataset):括号里的 Dataset 表示"继承 PyTorch 的 Dataset 类",即按它的三件套规矩写
# ══════════════════════════════════════════════════════════════════════════════
class CustomDataset(Dataset):
    # ⚪ __init__:main.py 里 CustomDataset(root_dir=args.data_dir) 就是调它。object_list 我们没传(None)
    def __init__(self,
        root_dir: str,
        object_list: list = None,
        ) -> None:
        # self.xxx = 把值存在对象身上,别的方法(load_image / __getitem__)也能用
        self.root_dir = root_dir
        # 背景色 = 白色 (R=1,G=1,B=1)。训练时物体都在白底上,推理必须一致
        self.bg_color = np.array([1., 1., 1.], dtype=np.float32) # white

        # 列出 root_dir 里所有 .png 文件名 → all_objects = ['bouquet.png', 'breakfast.png', ...]
        #   [i for i in ... if ...] 是列表推导式 = "遍历 + 筛选 + 收集"压成一行
        if object_list is None:
            self.all_objects = [i for i in os.listdir(root_dir) if i.endswith('.png') ]
        else:
            self.all_objects = object_list

        # 打印样本数(你跑的时候看到的 "loading 9 objects" 就是它)
        print("loading ", len(self.all_objects), " objects in the dataset")

    # ══════════════════════════════════════════════════════════════════════════════
    # 🔴 load_image —— 本文件核心:PNG → 干净的 0~1 数字 + 0/1 蒙版
    #   参数 alpha 从头到尾没用(函数自己从 PNG 第 4 通道算),调用时传 None 占位(作者遗留)
    #   返回 (img [512,512,3] 白底 RGB, alpha [512,512,1] 蒙版)
    # ══════════════════════════════════════════════════════════════════════════════
    def load_image(self, img_path, bg_color, alpha, return_type='np'):
        # ① 打开 PNG → PIL 图片对象(以"图片身份"存在,可 .resize / .save / 查 .size .mode)
        img = Image.open(img_path)

        # ① 等比缩放:长边缩到 512,短边跟着按比例缩,不拉变形。BICUBIC 是插值算法(画质较好),不用管
        #   ⚠ 变量名坑:PIL 的 img.size 返回 (宽, 高),所以这里 h 装的其实是宽、w 是高 —— 名字起反了;
        #     但 resize() 也按 (宽, 高) 收参数,反着用两次刚好抵消,结果正确:长边 → 512
        # resize, max length, keep aspect ratio
        max_length = 512 # hard coded
        h, w = img.size
        if h > w:
            new_h = max_length
            new_w = int(w / h * max_length)
        else:
            new_w = max_length
            new_h = int(h / w * max_length)
        img = img.resize((new_h, new_w), Image.BICUBIC)

        # ② PIL 图片 → numpy 数组 [高, 宽, 4]。numpy 里图片是"通道在最后"
        img = np.array(img)
        # ② 0~255 整数 → 0~1 小数。神经网络喜欢小数字,且必须和训练时一致;main.py 存图时 *255 乘回去
        img = img.astype(np.float32) / 255.
        # 🔴 ② 硬规定:必须是 4 通道(RGBA)。assert 条件不成立就报错并打印提示。
        #   第 4 通道 alpha 是后面抠背景的唯一依据 —— 这就是"自己的图要先用 rembg 处理成 RGBA"的出处。
        #   alpha 不是测量的,是 rembg 这类去背景 AI 猜出来的(物体处 255,背景 0,边缘中间值)
        assert img.shape[-1] == 4, f"please input RGBA images, {img_path}"

        # ③ 补 0 成 512×512 正方形(模型要正方形输入)
        # pad to [max_length, max_length]
        h, w = img.shape[:2]
        # 上下 / 左右各补多少:差值的一半(居中)。bouquet 现在 512 高 × 472 宽,宽差 40 → 左右各补 20 列
        pad_h, pad_w = (max_length - h)//2, (max_length - w)//2
        # np.pad 的三层元组 = ((上补, 下补), (左补, 右补), (通道不补)),补的值是 0。
        #   补进去的 0 在 alpha 通道上也是 0 = 透明,所以下一步会被当成背景涂白,不会留黑边
        img = np.pad(img, ((pad_h, max_length - h - pad_h), (pad_w, max_length - w - pad_w), (0, 0)), mode='constant', constant_values=0)

        # ⚫ 原 49-53 行:被注释掉的"裁成 8 的倍数"逻辑,已由上面的 pad 到 512 取代
        # crop image so that it can be diveded by 8
        # h, w = img.shape[:2]
        # nh, nw = h//8*8, w//8*8
        # dh, dw = int((h-nh)//2), int((w-nw)//2)
        # img = img[dh:dh+nh, dw:dw+nw]
        
        # 🔴 ④ 从 alpha 造 0/1 蒙版
        #   img.shape[-1]==3 那支走不到(原 42 行已保证是 4 通道);实际走 else:
        if img.shape[-1] == 3:
            alpha = np.ones_like(img[..., :1])
        else:
            # 取第 3 通道(alpha)。用 3: 而不是 3,是为了保留那一维 → [512,512,1]
            #   > 0.5 得 True/False,再转 1.0/0.0:物体处 1、背景处 0。rembg 猜的模糊边缘在这里被一刀切
            alpha = (img[:, :, 3:] > 0.5).astype(np.float32)
        # 只留前 3 通道 RGB,扔掉 alpha → [512,512,3]
        img = img[:, :, :3]

        # 🔴 ⑤ 背景涂白(合成到白底),逐像素看:
        #   物体处 alpha=1 → img*1 + 白*0 = 保留原像素
        #   背景处 alpha=0 → img*0 + 白*1 = 纯白
        #   形状 img [512,512,3]、alpha [512,512,1]、bg_color [3] 不同,numpy 自动"广播"拉成一样大再逐元素算。
        #   为什么蒙版之外还要白底图:网络只认 3 通道 RGB 图,背景像素总得有个颜色,而训练时全是白底;
        #   蒙版本身不能代替图片喂给网络。推理时进网络的只有这张白底图,蒙版没用到
        img = img[...,:3] * alpha + bg_color * (1 - alpha)

        # ⚪ ⑥ 按调用者要求决定返回 numpy 还是 torch。__getitem__ 传的是 'pt' → 走 elif 转 torch。
        #   注意只转了 img,没转 alpha(alpha 还是 numpy),所以 __getitem__ 原 85 行要自己再 from_numpy 一次
        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
        else:
            raise NotImplementedError

        # 一次返回两个值(Python 允许 return a, b),接收时 a, b = ... 对应拆开
        return img, alpha

    # ⚪ 三件套之二:样本总数 = png 文件数(9)
    def __len__(self):
        return len(self.all_objects)
    
    # ══════════════════════════════════════════════════════════════════════════════
    # 🔴 三件套之三:__getitem__(index) —— "给我第几个,我就把第几个样本准备好交给你"
    #   DataLoader 每圈调一次:index=0 取 bouquet.png,index=1 取 breakfast.png ...(顺序 = all_objects 列表顺序)
    #   你自己不会手动调它;dataset[0] 这种方括号写法就等于 dataset.__getitem__(0)
    # ══════════════════════════════════════════════════════════════════════════════
    def __getitem__(self, index):
        # 从文件名列表取第 index 个 → 'bouquet.png'(只是文件名,不是完整路径)
        data_path = self.all_objects[index]

        # get the bg color
        # 把白色复制到局部变量,纯粹为了下面传参少打几个字
        bg_color = self.bg_color

        # 拼完整路径:'example/single' + 'bouquet.png' → 'example/single/bouquet.png'
        img_path = os.path.join(self.root_dir, data_path)
        # 调上面的 load_image → img_tensor [512,512,3] torch(白底 RGB),mask [512,512,1] numpy(0/1)
        img_tensor, mask = self.load_image(img_path, bg_color, None, return_type='pt')
        # 🔴 permute(2,0,1):维度顺序 [512,512,3] → [3,512,512],通道从最后挪到最前(PyTorch 卷积要求通道在前)。
        #   数值不变只换排列;和 main.py reform_image 里的 transpose(1,2,0) 方向相反。.float() 确保 float32
        img_tensors_in = img_tensor.permute(2, 0, 1).float()

        # mask 还是 numpy → 先 from_numpy 转 torch,再同样 permute → [1,512,512]
        img_tensors_mask = torch.from_numpy(mask).permute(2, 0, 1).float()

        # 🔴 task_ids = 这张图的"订单":[0,1,2] = albedo、normal、material 三样都要。
        #   下标就是 pipeline 里 5 个任务词的序号(pipeline 原 248 行)。.long() 转 64 位整数 —— 它后面要当下标查表,必须是整数
        task_ids = torch.tensor([0, 1, 2]).long()

        # 🔴 pose = 这张图的"拍摄位置",4 个数 [仰角, 方位角, 距离编码, 占位0]。
        #   单图没有相机信息,给一个固定的"默认站位":仰角 0、方位角 0、距离 1.5。
        #   中间那串 log 公式 = 把距离 1.5 在训练范围 [1.2, 2.2] 内做 log 归一化再乘 π(模型训练时的编码约定),不用会算。
        #   多视角加 --cam 时这 4 个数才是从相机矩阵真算出来的(见 custom_mv_dataset.py get_pose)
        mv_pose = torch.tensor([torch.tensor(0), torch.tensor(0), (np.log(1.5) - np.log(1.2))/(np.log(2.2)-np.log(1.2)) * torch.pi, torch.tensor(0)])
        
        # 🔴 装字典。键名就是 main.py 里 batch['imgs_in'] 等的键。
        #   [None, ...] = 在最前面凭空加一个长度为 1 的维度 —— 这一维就是"视角维 Nv"(单图 Nv=1)。
        #   这样单图和多视角的数据格式统一为 [Nv, ...],后面代码不用分两套写
        item = {
            'imgs_in': img_tensors_in[None,...], # (1, 3, H, W)
            # 'imgs_out': img_tensors_out, # (Nd, 3, H, W) (albedo, normal, mr)
            'imgs_mask': img_tensors_mask[None,...], # (1, 1, H, W)
            'task_ids': task_ids[None,...], # (ND, ND)
            'pose': mv_pose[None,...],
        }

        # 文件名去掉最后 4 个字符 '.png' → 'bouquet',存图时命名用
        item['data_name'] = os.path.basename(data_path)[:-4] # filename
        
        # 交给 DataLoader;它再在最前面套一个 B 维 → main.py 拆出的 [B, Nv, 3, H, W]
        return item
