# -*- encoding: utf-8 -*-
import torch.utils.data as data
from torch.utils.data import Dataset
import json
import os
import torch
import numpy as np
import cv2
import tifffile

class SRDataset(Dataset):# 数据集生成函数
    """
    数据集加载器
    """
    def __init__(self, data_folder, split,):

        self.data_folder = data_folder# Json数据文件所在文件夹路径
        self.split = split.lower()# 名称全部转换为小写,'train','val','test'

        # 读取图像路径
        if self.split == 'train':# 读取训练图像json路径文件中，将图片路径加载到images列表中
            with open(os.path.join(data_folder, 'train_sandstone_images.json'), 'r') as j1:
                self.images = json.load(j1)
            with open(os.path.join(data_folder, 'train_sandstone_images_lr.json'), 'r') as j2:
                self.images_lr = json.load(j2)
        elif self.split == 'val':# 读取验证图像json路径文件中，将图片路径加载到images列表中
            with open(os.path.join(data_folder, 'val_sandstone_images.json'), 'r') as j1:
                self.images = json.load(j1)
            with open(os.path.join(data_folder, 'val_sandstone_images_lr.json'), 'r') as j2:
                self.images_lr = json.load(j2)
        else:# self.split == 'test',读取测试图像json路径文件中，将图片路径加载到images列表中
            with open(os.path.join(data_folder, 'test_sandstone_images.json'), 'r') as j1:
                self.images = json.load(j1)
            with open(os.path.join(data_folder, 'test_sandstone_images_lr.json'), 'r') as j2:
                self.images_lr = json.load(j2)

    def __getitem__(self, i):
        """
        为了使用PyTorch的DataLoader，必须提供该方法.
        :参数 i: 图像检索号
        :返回: 返回第i个低分辨率和高分辨率的图像对
        """
        # 读取图像
        hr_img=tifffile.imread(self.images[i], mode='r')
        lr_img=tifffile.imread(self.images_lr[i],mode='r')
        return lr_img, hr_img  # 返回低分辨和高分辨图像的tensor形式

    def __len__(self):
        """
        为了使用PyTorch的DataLoader，必须提供该方法.

        :返回: 加载的图像总数
        """
        return len(self.images)  # 返回图像总数

# class DatasetFromHdf5(data.Dataset):
#     def __init__(self, file_path):
#         super(DatasetFromHdf5, self).__init__()
#         hf = h5py.File(file_path, 'r+')
#         self.data = hf.get('data')
#         self.target = hf.get('label')
#         # self.data = self.data.reshape((-1, 25, 25))
#         # self.target = self.target.reshape((-1, 25, 25))
#     def __getitem__(self, index):
# #        print ('data size:',self.data.shape)
#         return torch.from_numpy(self.data[index,:,:,:]).float(), torch.from_numpy(self.target[index,:,:,:]).float()
#     def __len__(self):
#
#         return len(self.data[0])

def generate_2Dimage(array_like,save_mode='3D_VDSR_/',image_format='bmp'):
    if not isinstance(array_like,np.ndarray):
        array_like=np.asarray(array_like)
#    shape=array_like.shape()
    if not os.path.exists('3D_VDSR_/'):
        os.mkdir(save_mode)
    for count,every_image in enumerate(array_like):
        cv2.imwrite(save_mode+str(count+1)+'.'+image_format,every_image)
    print ('Successfully save'+str(count)+image_format+'image!')





