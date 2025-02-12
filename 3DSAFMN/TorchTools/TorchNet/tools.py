from math import sqrt, ceil
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init

from collections import OrderedDict
import PIL
from PIL import Image

from ..Functions.functional import to_pil_image
from torch.nn.init import xavier_uniform as xavier
import functools
import sys
import pdb

class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return Variable(images)
        return_images = []
        for image in images:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = Variable(torch.cat(return_images, 0))
        return return_images


def calculate_parameters(model):
    parameters = 0
    for weight in model.parameters():
        p = 1
        for dim in weight.size():
            p *= dim
        parameters += p
    return parameters


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def FeatureMapsVisualization(feature_maps, instan_norm=False):
    """
    visualize feature maps
    :param feature_maps: must be 4D tensor with B equals to 1 or 3D tensor N * H * W
    :return: PIL.Image of feature maps
    """
    if len(feature_maps.size()) == 4:
        feature_maps = feature_maps.view(feature_maps.size()[1:])
    if not instan_norm:
        feature_maps = (feature_maps - feature_maps.min()) / (feature_maps.max() - feature_maps.min())
    maps_number = feature_maps.size()[0]
    feature_H = feature_maps.size()[1]
    feature_W = feature_maps.size()[2]
    W_n = ceil(sqrt(maps_number))
    H_n = ceil(maps_number / W_n)
    map_W = W_n * feature_W
    map_H = H_n * feature_H
    MAP = Image.new('L', (map_W, map_H))
    for i in range(maps_number):
        map_t = feature_maps[i]
        if instan_norm:
            map_t = (map_t - map_t.min()) / (map_t.max() - map_t.min())
        map_t = map_t.view((1, ) + map_t.size())
        map_pil = to_pil_image(map_t)
        n_row = i % W_n
        n_col = i // W_n
        MAP.paste(map_pil, (n_row * feature_W, n_col * feature_H))
    return MAP


def ModelToSequential(model, seq_output=True):
    Sequential_list = list()
    for sub in model.children():
        if isinstance(sub, torch.nn.modules.container.Sequential):
            Sequential_list.extend(ModelToSequential(sub, seq_output=False))
        else:
            Sequential_list.append(sub)
    if seq_output:
        return nn.Sequential(*Sequential_list)
    else:
        return Sequential_list


def weights_init_xavier(m):# xavier初始化方法
    if isinstance(m, nn.Conv2d):
        if not isinstance(m.weight, torch.HalfTensor):
            xavier(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0)

def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):# kaiming初始化方法
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


## model tools
# 定义权重初始化函数
def load_weights(model, weights='', gpus=1, init_method='kaiming', strict=True, scale=0.1, resume=False, just_weight=False):
    """
    load model from weights, remove "module" if weights is dataparallel
    :param model:模型名称
    :param weights:预训练路径
    :param gpus:gpu数量
    :param init_method:初始化方法
    :return:
    """
    # Initiate Weights
    if weights == '':# 如果预训练路径为空
        print('Training from scratch......')# 输出开始训练
        if init_method == 'xavier':# 如果初始化方法为xavier
            model.apply(weights_init_xavier)# 应用weights_init_xavier方法初始化
        elif init_method == 'kaiming':# 如果初始化方法为kaiming
            weights_init_kaiming_ = functools.partial(weights_init_kaiming, scale=scale)# 定义weights_init_kaiming_方法
            model.apply(weights_init_kaiming_)# 应用weights_init_kaiming方法初始化
        else:# 如果都不是
            model.apply(weights_init_xavier)# 应用weights_init_xavier方法初始化
        print('Init weights with %s' % init_method)# 输出用...方法训练
    # Load Pre-train Model
    else:# 如果预训练路径不为空
        model_weights = torch.load(weights)# 加载路径
        if not just_weight:# 如果just_weight为假
            model_weights = model_weights['optim'] if resume else model_weights['state_dict']# 得到预训练模型的优化器部分
        try:# 抛出异常
            model.load_state_dict(model_weights, strict=strict)
        except:
            print('Loading from DataParallel module......')# 输出从...模块加载
            model = _rm_module(model, model_weights)# 重新生成权重初始化后的模型
        print('Loading %s success.....' % weights)# 输出加载权重成功
    if gpus > 1:# 如果可用gpu数量大于1
        model = nn.DataParallel(model, device_ids=[i for i in range(gpus)])# 数据并行
    sys.stdout.flush()# 将str写到流，原封不动
    return model# 返回模型


def _rm_module(model, weights):
    new_state_dict = OrderedDict()
    for k, v in weights.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    return model


def remove_module(net, load_path, save_path):
    """
    Remove key words module caused by Dataparallel
    :param net: model
    :param load_path:
    :param save_path:
    :return:
    """
    model = torch.load(load_path)
    print('load %s success.....' % load_path)
    new_state_dict = OrderedDict()
    for k, v in model.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    torch.save(net.state_dict(), save_path)
    print('save to %s .....' % save_path)


# def KernelsVisualization(kernels, instan_norm=False):
#     """
#     visualize feature maps
#     :param feature_maps: must be 4D tensor
#     :return: PIL.Image of feature maps
#     """
#     if not instan_norm:
#         feature_maps = (kernels - kernels.min()) / (kernels.max() - kernels.min())
#     kernels_out = kernels.size()[0]
#     kernels_in = kernels.size()[1]
#     feature_H = kernels.size()[2]
#     feature_W = kernels.size()[3]
#     W_n = ceil(sqrt(kernels_in))
#     H_n = ceil(kernels_in / W_n)
#     big_W_n = ceil(sqrt(kernels_out))
#     big_H_n = ceil(kernels_out / W_n)
#     map_W = W_n * feature_W
#     map_H = H_n * feature_H
#     MAP = Image.new('L', (map_W, map_H))
#     for i in range(maps_number):
#         map_t = feature_maps[i]
#         if instan_norm:
#             map_t = (map_t - map_t.min()) / (map_t.max() - map_t.min())
#         map_t = map_t.view((1, ) + map_t.size())
#         map_pil = to_pil_image(map_t)
#         n_row = i % W_n
#         n_col = i // W_n
#         MAP.paste(map_pil, (n_row * feature_W, n_col * feature_H))
#     return MAP





