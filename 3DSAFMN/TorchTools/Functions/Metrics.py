import numpy as np
try:
    from math import log10
except ImportError:
    from math import log
    def log10(x):
        return log(x) / log(10.)

import torch
import math
from .functional import to_tensor
from PIL import Image

def mse(x, y):# mse函数
    """
    MSE Error
    :param x: tensor / numpy.ndarray
    :param y: tensor / numpy.ndarray
    :return: float
    """
    diff = x - y # 计算x,y的差值
    diff = diff * diff # 计算差值的平方
    if isinstance(diff, np.ndarray):# 如果diff是np.ndarray类型
        diff = torch.FloatTensor(diff)# 将diff转换成tensor形式
    # return torch.mean(diff)
    # 按dim=1,2,3求平均值，即求单个高分辨和超分辨图像对的psnr(psnr_single),diff(b,c,h,w):b:batch_size=1,测试集取1
    return torch.mean(diff, dim=[1,2,3])# 返回结果,得到batch_size个值，batch_size=1时，得到1个值


def psnr(x, y, peak=1.):# 定义计算psnr函数
    """
    psnr from tensor
    :param x: tensor x图像向量 SR
    :param y: tensor y图像向量 HR
    :peak: peak=opt.rgb_range RGB范围
    :return: float (mse, psnr)
    """
    _mse = mse(x, y)# 输入HR图像，和对应的SR图像
    # return _mse, 10 * log10((peak ** 2) / _mse)
    return (10 * np.log10((peak ** 2) / _mse)).sum()# 计算psnr

def PSNR(x, y, c=-1):
    """
    PSNR from PIL.Image / tensor
    :param x: PIL.Image
    :param y: PIL.Image
    :return: float (mse, psnr)
    """
    if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
        return psnr(x, y, peak=1.)
    else:
        if c == -1:
            return psnr(to_tensor(x), to_tensor(y), peak=1.)
        else:
            return psnr(to_tensor(x)[c], to_tensor(y)[c], peak=1.)


def YCbCr_psnr(sr, hr, scale, peak=1.):
    """
    Caculate PSNR in YCbCr`s Y Channel
    :param sr:
    :param hr:
    :param scale:
    :return:
    """
    diff = (sr - hr) / peak
    shave = scale
    if diff.size(1) > 1:
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        diff = diff.mul(convert).sum(dim=1)

    valid = diff[..., shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)

