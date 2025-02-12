# -*- encoding: utf-8 -*-
import os
import json # json字符串
import torch
import math
import numpy as np
import torchvision.transforms.functional as FT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_data_lists(train_folders,train_et_folders,train_lr_folders,train_lr_et_folders,val_folders,val_lr_folders,test_folders,test_lr_folders,output_folder):
    """
    创建训练集和测试集列表文件.
        参数 train_folders: 训练文件夹集合; 各文件夹中的图像将被合并到一个图片列表文件里面
        参数 test_folders: 测试文件夹集合; 每个文件夹将形成一个图片列表文件
    """
    print("\n正在创建文件列表... 请耐心等待.\n")
    train_images=list()# 训练图像列表
    train_images_lr=list()
    for d in train_folders:# 遍历训练集文件夹
        for i in os.listdir(d):# 遍历文件夹中的所有图片
            img_path = os.path.join(d, i)# 获取图片路径
            train_images.append(img_path)# 添加图片路径至列表
    with open(os.path.join(output_folder, 'train_sandstone_images.json'), 'w') as j:
        json.dump(train_images,j)
    for d in train_et_folders:# 遍历训练集文件夹
        for i in os.listdir(d):# 遍历文件夹中的所有图片
            img_path = os.path.join(d, i)# 获取图片路径
            train_images.append(img_path)# 添加图片路径至列表
    with open(os.path.join(output_folder, 'train_sandstone_images.json'), 'w') as j:
        json.dump(train_images,j)

    for d in train_lr_folders:
        for i in os.listdir(d):
            img_path = os.path.join(d,i)
            train_images_lr.append(img_path)
    with open(os.path.join(output_folder, 'train_sandstone_images_lr.json'), 'w') as j:
        json.dump(train_images_lr,j)
    for d in train_lr_et_folders:
        for i in os.listdir(d):
            img_path = os.path.join(d,i)
            train_images_lr.append(img_path)
    with open(os.path.join(output_folder, 'train_sandstone_images_lr.json'), 'w') as j:
        json.dump(train_images_lr,j)
    print("训练集中共有 %d 张图像\n" % len(train_images_lr))  # 输出训练集的图片数0

    val_images=list()
    val_images_lr=list()
    for d in val_folders:
        for i in os.listdir(d):
            img_path = os.path.join(d, i)
            val_images.append(img_path)
    with open(os.path.join(output_folder, 'val_sandstone_images.json'), 'w') as j:
        json.dump(val_images, j)
    for d in val_lr_folders:
        for i in os.listdir(d):
            img_path = os.path.join(d,i)
            val_images_lr.append(img_path)
    with open(os.path.join(output_folder, 'val_sandstone_images_lr.json'), 'w') as j:
        json.dump(val_images_lr,j)
    print("验证集中共有 %d 张图像\n" % len(val_images_lr))

    test_images=list()
    test_images_lr=list()
    for d in test_folders:
        for i in os.listdir(d):
            img_path = os.path.join(d, i)
            test_images.append(img_path)
    with open(os.path.join(output_folder, 'test_sandstone_images.json'), 'w') as j:
        json.dump(test_images, j)
    for d in test_lr_folders:
        for i in os.listdir(d):
            img_path = os.path.join(d, i)
            test_images_lr.append(img_path)
    with open(os.path.join(output_folder, 'test_sandstone_images_lr.json'), 'w') as j:
        json.dump(test_images_lr, j)
    print("测试中共有 %d 张图像\n" % len(test_images_lr))
    print("生成完毕。训练集和测试集文件列表已保存在 %s 下\n" % output_folder)

def convert_image(img, source, target):
    """
    转换图像格式.

    :参数 img: 输入图像
    :参数 source: 数据源格式, 共有3种
                   (1) 'pil' (PIL图像)
                   (2) '[0, 1]'
                   (3) '[-1, 1]'
    :参数 target: 数据目标格式, 共5种
                   (1) 'pil' (PIL图像)
                   (2) '[0, 1]'
                   (3) '[-1, 1]'
                   (4) 'imagenet-norm' (由imagenet数据集的平均值和方差进行标准化)
                   (5) 'y-channel' (亮度通道Y，采用YCbCr颜色空间, 用于计算PSNR 和 SSIM)
    :返回: 转换后的图像
    """
    # 图像源格式
    assert source in {'pil', '[0, 1]', '[-1, 1]'
                      }, "无法转换图像源格式 %s!" % source
    assert target in {
        'pil', '[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm', 'y-channel'
    }, "无法转换图像目标格式t %s!" % target

    # pil(source)->tensor(target):1.[0 ,1],2.[-1, 1],3.[0, 255]
    # 或 pil->tensor->pil,保存图片
    # 转换成tensor:[0,1]
    # 把一个取值范围是[0,255]的PIL.Image 转换成形状为[C,H,W]的Tensor，取值范围是[0,1.0]
    if source == 'pil':# 输入是PIL.Image格式，输出tensor形式
        img = FT.to_tensor(img)

    elif source == '[0, 1]':# 输入是tensor形式，输出tensor形式
        pass  # 已经在tensor:[0, 1]范围内无需处理

    elif source == '[-1, 1]':# 输入是tensor形式，输出tensor形式
        img = (img + 1.) / 2.# 在tensor:[-1, 1]之间,需要(img+1)/2,输出tensor:[0, 1]

    # 从 tensor:[0, 1] 转换至目标图片格式
    if target == 'pil':# 输入是tensor:[0,1]，输出是PIL.Image格式
        img = FT.to_pil_image(img)

    elif target == '[0, 255]':# 输入是tensor:[0,1]，输出是tensor:[0,255]
        img = 255. * img # source='pil',target='[0, 255]'

    elif target == '[0, 1]':# 输入是tensor:[0,1]，输出是tensor格式:[0,1]
        pass  # 无需处理，source='pil',target='[0, 1]',img=FT.to_tensor(img)->tensor:[0,1]

    elif target == '[-1, 1]':# 输入是tensor:[0,1]，输出是tensor:[-1,1]
        img = 2. * img - 1. # source='pil',target='[-1, 1]'

    return img# 返回图片tensor形式

class AverageMeter(object):
    """
    跟踪记录类，用于统计一组数据的平均值、累加和、数据个数.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0 # 本身
        self.avg = 0 # 平均
        self.sum = 0 # 总和
        self.count = 0 # 数量

    def update(self, val, n=1):
        self.val = val# 每个batch的平均参数
        self.sum += val * n# n=batch_size,累加得到该epoch的参数总和
        self.count += n# 累加得到一个epoch的长度
        self.avg = self.sum / self.count# 每一个epoch的平均参数
# var得到每个epoch最后一个batch的参数
# avg得到每个epoch的平均参数

def clip_gradient(optimizer, grad_clip):
    """
    丢弃梯度防止计算过程中梯度爆炸.

    :参数 optimizer: 优化器，其梯度将被截断
    :参数 grad_clip: 截断值
    """
    for group in optimizer.param_groups:
        for param in group['params']:# params:w,b
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def save_checkpoint(state, filename):
    """
    保存训练结果.

    :参数 state: 逐项预保存内容
    """

    torch.save(state, filename)


def adjust_learning_rate(optimizer, shrink_factor):
    """
    调整学习率.

    :参数 optimizer: 需要调整的优化器
    :参数 shrink_factor: 调整因子，范围在 (0, 1) 之间，用于乘上原学习率.
    """

    print("\n调整学习率.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("新的学习率为 %f\n" % (optimizer.param_groups[0]['lr'], ))

# 计算PSNR，2d和3d都适用
def PSNR(pred, gt):#this function(tested) can be used in 2D or 3D
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(1.0 / rmse)


def Interpolation3D(src_img, dst_size):

    # src_img:原3d图像，dst_size:插值后图像的大小
    srcZ, srcY, srcX = src_img.shape  # 原图图像大小
    dst_img = np.zeros(shape=dst_size, dtype=np.int8)  # 插值后的图像

    new_Z, new_Y, new_X = dst_img.shape  # 插值后图像大小
    print("插值后图像的大小", dst_img.shape)

    factor_z = srcZ / new_Z # 缩放因子
    factor_y = srcY / new_Y # 缩放因子
    factor_x = srcX / new_X # 缩放因子

    for z in range(new_Z):
        for y in range(new_Y):
            for x in range(new_X):

                src_z = z * factor_z
                src_y = y * factor_y
                src_x = x * factor_x

                src_z_int = math.floor(z * factor_z)
                src_y_int = math.floor(y * factor_y)
                src_x_int = math.floor(x * factor_x)

                w = src_z - src_z_int
                u = src_y - src_y_int
                v = src_x - src_x_int

                dst_img[z, y, x] = src_img[src_z_int, src_y_int, src_x_int]

    return dst_img

def save_checkpoint(model, epoch, saved_path="model/"):
    model_out_path = os.path.join(saved_path, "sandstone_HAMSR3D.pth".format(epoch))
    state = {"epoch": epoch, "model": model.state_dict()}
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)

    torch.save(state, model_out_path)

    # print("=========Checkpoint saved to {}".format(model_out_path))