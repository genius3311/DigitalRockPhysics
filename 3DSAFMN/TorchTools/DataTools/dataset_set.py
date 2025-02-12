import torch.utils.data as data
import os
import os.path
from .FileTools import _all_images
from .Loaders import pil_loader
from .Prepro import random_pre_process_pair
from ..Functions import functional as Func
import torchvision.transforms as transforms

class RealPairDataset(data.Dataset):# 真实数据对
    """
    Dataset for Real HL Pair # 真实高分辨率图像数据对
    Add Param For Image Enhance: No Change in Resolution 分辨率不改变
    :param rgb_range: 255. for RCAN and EDSR, 1. for others # RCAN和EDSR用0-255灰度范围，其他模型用0-1灰度范围
    :param need_hr_down: For Degrade Train, From HR_down to LR;# 是否需要下采样
    :param need_lr_up: For Upgrade Train, From LR_up to HR;# 是否需要上采样
    :param need_edge_mask: a mask based on edge detect algorithm, to augment edge loss;# 是否需要边缘掩码块
    :param multiHR: for HGSR, which need multi HR for intermediate supervision;# 是否需要多尺度高分辨率图像
    """
    def __init__(self,
                 pair_folder_path,# 数据根路径(包含了train_LR，和train_HR两个文件夹)
                 lr_patch_size,# 低分辨图像的大小,opt.size=48
                 # mode='RGB',# 颜色通道类型，RGB三通道
                 mode='Y',# in_ch=1
                 scala=4,# 放大因子：4
                 prepro=random_pre_process_pair,
                 train=True,# 训练模式
                 rgb_range=1.,# 灰度范围
                 data_augu=False# 是否需要数据扩展
                 ):
        # 低分辨率图像绝对路径(分两种情况，训练和测试)
        lr_file_path = os.path.join(pair_folder_path, 'train_LR') if train else os.path.join(pair_folder_path, 'test_LR')
        # 高分辨率图像绝对路径(分两种情况，训练和测试)
        hr_file_path = os.path.join(pair_folder_path, 'train_HR') if train else os.path.join(pair_folder_path, 'test_HR')
        self.lr_file_list = _all_images(lr_file_path)# 按顺序取出该路径下的所有LR图像文件路径
        self.hr_file_list = _all_images(hr_file_path)# 按顺序取出该路径下的所有HR图像文件路径
        print('Initializing DataSet, image list: %s ...' % pair_folder_path)# 输出初始化数据集
        print('Found %d HR %d LR ...' % (len(self.hr_file_list), len(self.lr_file_list)))# 输出找到()张HR,()张LR,
        self.lr_size = lr_patch_size# 低分辨图像的大小,opt.size=48
        self.mode = mode# # 颜色通道类型，RGB三通道
        self.hr_size = lr_patch_size * scala # HR图像的大小
        self.prepro = prepro# 随机预处理
        self.current_image = None
        self.train = train# 训练模式
        self.scale = scala# 放大因子
        self.rgb_range = rgb_range# RGB灰度范围
        self.data_augu = data_augu# 需要数据加速

    def __len__(self):
        return len(self.lr_file_list)# 返回LR图像数据集的长度

    def __getitem__(self, index):
        data = {}# 定义数据字典
        # For Color Mode
        if self.mode == 'Y':# 如果颜色模型为'Y'
            hr_img = pil_loader(self.hr_file_list[index].strip('\n'), mode='YCbCr')# 以'YCbCr'颜色模式加载HR图像
            lr_img = pil_loader(self.lr_file_list[index].strip('\n'), mode='YCbCr')# 以'YCbCr'颜色模式加载LR图像
        else:# # 如果颜色模型为'RGB'
            hr_img = pil_loader(self.hr_file_list[index].strip('\n'), mode='RGB')# 以'RGB'颜色模式加载HR图像
            lr_img = pil_loader(self.lr_file_list[index].strip('\n'), mode='RGB')# 以'RGB'颜色模式加载LR图像
        # For Train or Test, Whether Crop/Rotate Image
        if self.train:# 如果是训练模式
            # 随机裁剪
            hr_patch, lr_patch= self.prepro(hr_img, lr_img, self.lr_size, self.scale, self.data_augu)
        else:# 如果是测试模式
            # 不裁剪，使用原图
            hr_patch, lr_patch = hr_img, lr_img
        # Image To Tensor 将图像转换成张量形式
        if self.mode == 'Y':# 如果颜色模式为‘Y’
            tt=transforms.ToTensor()
            lr_patch=tt(lr_patch)
            hr_patch=tt(hr_patch)
            # data['LR'] = Func.to_tensor(lr_patch)[:1] * self.rgb_range# 取LR图像第一个通道，乘上RGB范围(1)
            data['LR']=lr_patch[:1]
            data['HR']=hr_patch[:1]
            # data['HR'] = Func.to_tensor(hr_patch)[:1] * self.rgb_range# 取HR图像第一个通道，乘上RGB范围(1)
        else:# 如果颜色模式为RGB
            data['LR'] = Func.to_tensor(lr_patch) * self.rgb_range# 取LR图像所有通道，乘上RGB范围
            data['HR'] = Func.to_tensor(hr_patch) * self.rgb_range# 取HR图像所有通道，乘上RGB范围
        return data # 返回data字典



