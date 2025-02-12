import torch
import torch.nn as nn
from MBHASR.model import common
from MBHASR.model.common import *
from paper.TorchTools.TorchNet.tools import calculate_parameters

# growRate,growRate0,nConvLayers
# RDN 残差深度网络 超分方法

class RDB_Conv(nn.Module):  # RDB卷积模块
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels  # 输入通道,3
        G = growRate  # 输出通道
        self.conv = nn.Sequential(*[
            nn.Conv3d(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.ReLU()
        ])  # 3x3卷积核，输出图像大小不变

    def forward(self, x):  # 前向传播
        out = self.conv(x)  # 经过卷积层输出
        return torch.cat((x, out), 1)  # 把x和out在通道维concat起来,最后输出通道数是Cin+G


class RDB(nn.Module):  # RDB模块
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0  # 初始输入通道
        G = growRate  # 中间输出通道
        C = nConvLayers  # 卷积层数

        convs = []
        for c in range(C):  # 一系列RDB卷积层，输入通道随着卷积层数增加
            convs.append(RDB_Conv(G0 + c * G, G))
        # c=0：输入通道G0,传入G，输出通道G0+G;c=1：输入通道G0+G,传入G，输出通道G0+2G...
        self.convs = nn.Sequential(*convs)  # 解包

        # Local Feature Fusion #1x1卷积改变通道数，使输出通道还原为初始输入通道
        self.LFF = nn.Conv3d(G0 + C * G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x  # 最后加上初始输入x并输出


class RDN3D(nn.Module):  # RDN通道
    def __init__(self, r=4, G0=32, kSize=3, config='A',conv=common.default_conv):
        super(RDN3D, self).__init__()
        # r = 4 # 取scale参数中的第一个值，缩放因子,4
        # G0 = 64# 初始输入通道
        # kSize = 3 # 卷积核大小,3

        # number of RDB blocks, conv layers, out channels
        self.D, C, G = {  # (RDB模块数量,卷积层数量,输出通道数G)，分为A,B两类
            'A': (14, 6, 16),
            'B': (16, 8, 64),
        }[config]  # RDNconfig的值(A,B),由输入的args决定,'A','B'

        # Shallow feature extraction net 浅层特征提取网络，经过两个卷积层，输出通道为G0
        self.SFENet1 = nn.Conv3d(1, G0, kSize, padding=(kSize - 1) // 2, stride=1)  # 3
        self.SFENet2 = nn.Conv3d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)

        # Redidual dense blocks and dense feature fusion 残差深度模块和深度特征提取网络
        self.RDBs = nn.ModuleList()  # 可以把任意nn.Module的子类添加到list中
        for i in range(self.D):  # D：RDB数量
            self.RDBs.append(
                RDB(growRate0=G0, growRate=G, nConvLayers=C)
            )  # RDB继承自nn.Module，可以添加

        # Global Feature Fusion 全局特征融合，两个卷积层
        self.GFF = nn.Sequential(*[
            nn.Conv3d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv3d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        ])  # 输入通道D*G0，输出通道G0

        # Up-sampling net # 上采样模块
        if r == 2 or r == 3:  # 如果放大因子等于2或3
            self.UPNet = nn.Sequential(*[
                nn.Conv3d(G0, G * r * r, kSize, padding=(kSize - 1) // 2, stride=1),  # 通道数变为原来的r*2倍
                nn.PixelShuffle(r),  # 像素清洗后通道数为变G，高和宽变为原来的r倍
                nn.Conv3d(G, 3, kSize, padding=(kSize - 1) // 2, stride=1)  # 再通过一个卷积层使通道数变为RGB通道数，还原为图像
            ])
        elif r == 4:  # 如果放大因子等于4
            self.UPNet = nn.Sequential(*[
                nn.Conv3d(G0, G * 4, kSize, padding=(kSize - 1) // 2, stride=1),
                Upsampler3d(conv,scale=2,n_feats=G*4,act=False),  # 先经过一次像素清洗，使高宽放大2倍
                nn.Conv3d(G*4, G * 4, kSize, padding=(kSize - 1) // 2, stride=1),
                Upsampler3d(conv,scale=2,n_feats=G*4,act=False),  # 再经过一次像素清洗，使高宽放大2倍，最终得到高宽宽放大2*2=4倍的图像
                nn.Conv3d(G*4, 1, kSize, padding=(kSize - 1) // 2, stride=1)
            ])
        else:
            raise ValueError("scale must be 2 or 3 or 4.")  # 如果放大因子不是2，3，4，则抛出异常

    def forward(self, x):  # 前向传播
        f__1 = self.SFENet1(x)  # 浅层特征提取网络1
        x = self.SFENet2(f__1)  # 浅层特征提取网络2

        RDBs_out = []
        for i in range(self.D):  # D个RDB模块
            x = self.RDBs[i](x)
            RDBs_out.append(x)  # 添加x经过每个RDB模块得到的结果RDBs_out

        x = self.GFF(torch.cat(RDBs_out, 1))  # 把RDBs_out在通道维上concat起来得到x
        x += f__1  # 将x和经过浅层特征提取网络1相加

        return self.UPNet(x)  # 放入上采样网络中得到输出


# if __name__ == '__main__':
#     x = torch.rand(1, 3, 12, 12, 12)
#     model = RDN3D()
#     y = model(x)
#     print(y.shape)
#     print('%s Created, Parameters: %d' % (model.__class__.__name__, calculate_parameters(model)))