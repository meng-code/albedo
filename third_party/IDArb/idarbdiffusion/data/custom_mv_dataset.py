# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  custom_mv_dataset.py —— 多视角模式的数据集(学习注释版;代码一行未改,原始版本见 git)  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# 【这个文件在干什么】全流程"站① 读图"的多视角版:一个文件夹(多张图 + transforms.json)→ 摞成 [Nv, ...] 的 tensor。
#   main.py --input_type multi 时用 CustomMVDataset(root_dir, num_views, use_cam) 构造它。
#
# 【标签说明】 🔴 关键(必须搞清)  ⚪ 管道(知道把什么变成什么)  ⚫ 废弃(跳过)
#
# 【和单图版 custom_dataset.py 的关系】
#   90% 代码是复制粘贴的(load_image 一字不差;__fetch_data 就是单图版 __getitem__ 的翻版)。
#   两个文件互不 import。只有 3 处不同:
#     ① 一个样本 = 一个子文件夹(原 24 行),不是一个 png
#     ② __getitem__ 按 transforms.json 循环读前 num_views 张,再 torch.stack 摞出 Nv 维(原 120-148 行)
#     ③ 加了 --cam 时,用 json 里的 4×4 相机矩阵真算 pose(原 112-116 行 + get_pose)
#
# 【transforms.json 长什么样】json.load 后变成 Python 字典:
#   cfg = {'frames': [ {'file_path': '0061', 'transform_matrix': 4×4列表 或 null},   ← 一个 frame = 一张照片的"档案"
#                      {'file_path': '0062', 'transform_matrix': ...}, ... ]}         (示例里每个文件夹 10 帧,默认只用前 4)
#   这个格式是 NeRF 系工作的通用约定;COLMAP 等工具能导出。没有位姿就写 null(moldGoldCape 就是这样)。
#   换自己的多视角图时这个文件要自己准备。
#
# 【为什么默认 num_views=4】模型是把 N 张图一起送进网络让它们互相"看"(第 5 部分跨视角注意力);
#   训练配置 configs/train.yaml 里视角数是 3 和 4,默认取训练上限 4。可 --num_views 改,但显存随之增长、超出训练范围效果没保证。
#
# 【交出去的字典】(Nv=4;DataLoader 再套一个 B 维)
#   'imgs_in' [4,3,512,512]  'imgs_mask' [4,1,512,512]  'task_ids' [4,3]  'pose' [4,4]  'data_name' 'car_scene002'
# ──────────────────────────────────────────────────────────────────────────────
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from PIL import Image
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"


# ══════════════════════════════════════════════════════════════════════════════
# 🔴 CustomMVDataset(Dataset):继承 PyTorch Dataset,同样的三件套(__init__ / __len__ / __getitem__)
# ══════════════════════════════════════════════════════════════════════════════
class CustomMVDataset(Dataset):
    # ⚪ __init__:比单图版多两个参数 —— num_views(取几张)、use_cam(--cam 开关)
    def __init__(self,
        root_dir: str,
        num_views: int = 4,
        object_list: list = None,
        use_cam: bool = False,
        ) -> None:
        self.root_dir = root_dir
        self.num_views = num_views
        self.use_cam = use_cam
        self.bg_color = np.array([1., 1., 1.], dtype=np.float32) # white

        # 不同点①:挑的是 root_dir 下的"子文件夹"(os.path.isdir 判断),不是 png。
        #   example/multi 下 car_scene002/、moldGoldCape/ 各是一个样本
        if object_list is None:
            self.all_objects = [i for i in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, i))]
        else:
            self.all_objects = object_list

        print("loading ", len(self.all_objects), " objects in the dataset")
    
    # ⚪ load_image:和 custom_dataset.py 的 load_image 一字不差(复制粘贴来的),详细注释见那边。
    #   6 步:打开缩放 → /255 + assert RGBA → 补成 512² → alpha 造蒙版 → 背景涂白 → 转 torch
    def load_image(self, img_path, bg_color, alpha, return_type='np'):
        img = Image.open(img_path)

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

        img = np.array(img)
        img = img.astype(np.float32) / 255.
        # 🔴 同样的硬规定:必须 RGBA。多视角的每张图也要先 rembg
        assert img.shape[-1] == 4 # RGBA # XXX

        # pad to [max_length, max_length]
        h, w = img.shape[:2]
        pad_h, pad_w = (max_length - h)//2, (max_length - w)//2
        img = np.pad(img, ((pad_h, max_length - h - pad_h), (pad_w, max_length - w - pad_w), (0, 0)), mode='constant', constant_values=0)

        # crop image so that it can be diveded by 8
        # h, w = img.shape[:2]
        # nh, nw = h//8*8, w//8*8
        # dh, dw = int((h-nh)//2), int((w-nw)//2)
        # img = img[dh:dh+nh, dw:dw+nw]
        
        if img.shape[-1] == 3:
            alpha = np.ones_like(img[..., :1])
        else:
            alpha = (img[:, :, 3:] > 0.5).astype(np.float32)
        img = img[:, :, :3]

        img = img[...,:3] * alpha + bg_color * (1 - alpha)

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
        else:
            raise NotImplementedError

        return img, alpha

    # ══════════════════════════════════════════════════════════════════════════════
    # ⚪ cartesian_to_spherical:直角坐标 (x,y,z) → 球坐标 (仰角 theta, 方位角 azimuth, 距离 z)
    #   标准三角公式,知道"它在做坐标转换"即可,不用抠。get_pose 调它
    # ══════════════════════════════════════════════════════════════════════════════
    def cartesian_to_spherical(self, xyz):
        # ⚫ ptsnew 算了没用(遗留)
        ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
        xy = xyz[:,0]**2 + xyz[:,1]**2
        z = np.sqrt(xy + xyz[:,2]**2)
        # theta:从 z 轴(正上方)往下数的角度(仰角)
        theta = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
        #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
        # azimuth:绕物体一圈的方位角
        azimuth = np.arctan2(xyz[:,1], xyz[:,0])
        return np.array([theta, azimuth, z])

    # ══════════════════════════════════════════════════════════════════════════════
    # 🔴 get_pose:4×4 相机矩阵 → 模型要的 4 个数 [仰角, 方位角, 距离编码, 0](只在加了 --cam 时被调用)
    #   4×4 矩阵 = 相机的"身份证",记两件事:
    #     左上 3×3       相机朝哪看(三根轴的方向)—— 代码没用它(训练时相机默认对着物体中心,朝向冗余)
    #     最后一列前 3 个 相机站在哪 (x,y,z),以物体中心为原点 —— 代码只用这个
    #     最后一行       固定 0 0 0 1,凑格式,无信息
    #   例:car_scene002 第 0 帧位置 (0.664, -3.548, 2.186) → 仰角 58.8°、方位角 -79.4°、距离 4.22
    #      距离 4.22 超出训练范围 [1.2, 2.2] 被截成 2.2 → log 归一化后恰好 = π
    # ══════════════════════════════════════════════════════════════════════════════
    def get_pose(self, c2w):
        # 切出最后一列的前 3 个数 = 相机位置 (x,y,z)。[:3, -1] = 前 3 行、最后一列
        origin = c2w[:3, -1]
        # (x,y,z) → (仰角, 方位角, 距离)。origin[None,:] 加一维是因为 cartesian_to_spherical 按"一批点"写的
        theta_target, azimuth_target, z_target = self.cartesian_to_spherical(origin[None, :])
        # 距离超出训练范围 [1.2, 2.2] 就截断到边界(模型只把距离当粗略信号)
        if z_target.item() < 1.2 or z_target.item() > 2.2:
            z_target = np.clip(z_target.item(), 1.2, 2.2)
        # 拼成 4 个数:[仰角, 方位角, 距离的 log 归一化 × π, 占位 0]。和单图版默认 pose 的格式完全一样,只是数是真算的
        target_T = torch.tensor([theta_target.item(), azimuth_target.item(), (np.log(z_target.item()) - np.log(1.2))/(np.log(2.2)-np.log(1.2)) * torch.pi, torch.tensor(0)])
        # 保险:4 个数都要在 [-π, π] 内
        assert torch.all(target_T <= torch.pi) and torch.all(target_T >= -torch.pi)
        return target_T
    
    # ⚪ 三件套之二:样本数 = 子文件夹数(example/multi 下是 5 个)
    def __len__(self):
        return len(self.all_objects)
    
    # ══════════════════════════════════════════════════════════════════════════════
    # 🔴 __fetch_data:读"一张"图并配好便签 —— 就是单图版 __getitem__ 的翻版,定义在本文件(不是调单图文件)
    #   参数 frame_info = 一个 frame 小字典 {'file_path':..., 'transform_matrix':...};data_dir = 子文件夹名
    #   返回 4 样东西:img [3,512,512]、mask [1,512,512]、task_ids [3]、pose [4]
    #   函数名前两个下划线 = "内部小工具"的命名习惯
    # ══════════════════════════════════════════════════════════════════════════════
    def __fetch_data(self, frame_info, data_dir):

        bg_color = self.bg_color
        # 从档案取文件名 → '0061'(json 里不带后缀)
        fn = frame_info['file_path']
        # 手动补 '.png' 再拼路径 → 'example/multi/car_scene002/0061.png'
        img_path = os.path.join(self.root_dir, data_dir, fn+'.png')

        # 原 103-106 行和单图版原 82-85 行一字不差:读图 → 白底 + 蒙版 → permute 通道到最前
        img_tensor, mask = self.load_image(img_path, bg_color, None, return_type='pt')
        img_tensors_in = img_tensor.permute(2, 0, 1).float()

        img_tensors_mask = torch.from_numpy(mask).permute(2, 0, 1).float()

        # ⚫ ND 定义了但没用(遗留)
        ND = 3
        
        # 🔴 订单 [0,1,2]:三个分量都要(每张图各一份)
        task_ids = torch.tensor([0, 1, 2]).long()

        # 🔴 不同点③ —— 命令行 --cam 开关的全部作用就是这一个 if:
        #   json 里有矩阵(不是 null)且加了 --cam → 走上面,用 get_pose 真算 4 个数
        #   否则 → 走 else,和单图版一样的默认站位(所有视角同一组默认数 = 告诉模型"角度不知道,你自己看图猜")
        if frame_info['transform_matrix'] is not None and self.use_cam:
            # json 里的 4×4 列表 → numpy 矩阵
            c2w = np.array(frame_info['transform_matrix'])
            pose = self.get_pose(c2w)
        else:
            pose = torch.tensor([torch.tensor(0), torch.tensor(0), (np.log(1.5) - np.log(1.2))/(np.log(2.2)-np.log(1.2)) * torch.pi, torch.tensor(0)])
        
        # 4 样东西一起交回 __getitem__ 原 128 行,被 img_in, img_mask, task_ids, pose = ... 依次接住
        return img_tensors_in, img_tensors_mask, task_ids, pose

    # ══════════════════════════════════════════════════════════════════════════════
    # 🔴 三件套之三:__getitem__(index) —— 不同点②:一次交出"一个文件夹"的 num_views 张图
    #   套路:读 json → 取前 4 帧 → 开 4 个空篮子 → 循环 4 次每次读一张、4 样东西分别入篮 → 每篮 torch.stack 摞成带 Nv 维的 tensor
    #   "空列表 + 循环 append + stack"是 PyTorch 代码里极常见的套路
    # ══════════════════════════════════════════════════════════════════════════════
    def __getitem__(self, index):
        # 从子文件夹名列表取第 index 个 → 'car_scene002'
        obj_dir = self.all_objects[index]
        # 拼路径 → open 打开 → json.load 把文本翻译成 Python 字典({}→字典,[]→列表,null→None)
        cfg = json.load(open(os.path.join(self.root_dir, obj_dir, f'transforms.json')))
        # 按键取出 frames 列表(10 个小字典)
        frames = cfg['frames']
        # 列表切片,只留前 num_views(4)项;后 6 帧根本不读
        frames = frames[:self.num_views]

        # 🔴 开 4 个空篮子(一行同时赋值:左边 4 个变量对应右边 4 个空列表)。mv = multi-view。
        #   4 个篮子分别装:图、蒙版、订单、拍摄位置 —— 这 4 样都是"每张图一份",所以要 4 个篮子
        mv_img_in, mv_img_mask, mv_task_ids, mv_pose = [], [], [], []
        # 循环 4 次,每次 frame 是一个小字典(一张照片的档案),如 {'file_path': '0061', 'transform_matrix': [...]}
        for frame in frames:
            # 调本类的 __fetch_data 读这一张 → 一次拿到 4 样东西
            img_in, img_mask, task_ids, pose = self.__fetch_data(frame, obj_dir)
            # 4 样东西各自 append 进对应的篮子。循环完每个篮子里有 4 项(还是 Python 列表,不是 tensor)
            mv_img_in.append(img_in)
            mv_img_mask.append(img_mask)
            mv_task_ids.append(task_ids)
            mv_pose.append(pose)
        
        # 🔴 torch.stack(dim=0):把篮子里 4 个形状相同的 tensor 摞成一个,最前面多出"第几张"的维度 = Nv
        #   [3,512,512] ×4 → [4,3,512,512];[1,512,512] ×4 → [4,1,512,512];[3] ×4 → [4,3];[4] ×4 → [4,4]
        #   单图版用 [None,...] 凭空加 Nv=1,多视角用 stack 摞出 Nv=4 —— 殊途同归,都是造 Nv 维
        mv_img_in = torch.stack(mv_img_in, dim=0)
        mv_img_mask = torch.stack(mv_img_mask, dim=0)
        mv_task_ids = torch.stack(mv_task_ids, dim=0)
        mv_pose = torch.stack(mv_pose, dim=0)

        # 🔴 装字典。键名和单图版一模一样,所以 main.py 不用管来的是单图还是多视角,统一按 batch['imgs_in'] 取
        item = {
            'imgs_in': mv_img_in,
            'imgs_mask': mv_img_mask,
            'task_ids': mv_task_ids,
            'pose': mv_pose,
        }

        # 名字 = 子文件夹名 'car_scene002',存图时命名用
        item['data_name'] = self.all_objects[index]
        
        # 交给 DataLoader;它再套一个 B 维 → main.py 拆出 B=1, Nv=4
        return item
            