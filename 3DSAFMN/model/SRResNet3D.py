import torch.nn as nn
import torch
from MBHASR.model import common
from MBHASR.model.common import *
from paper.TorchTools.TorchNet.tools import calculate_parameters

class ResBlock(nn.Module):
    """残差模块"""

    def __init__(self, inChannals, outChannals):
        """初始化残差模块"""
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv3d(inChannals, outChannals, kernel_size=3,stride=1,padding=1,bias=False)
        self.bn1 = nn.BatchNorm3d(outChannals)
        self.relu = nn.PReLU()
        self.conv2 = nn.Conv3d(outChannals, outChannals, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(outChannals)

    def forward(self, x):
        """前向传播过程"""
        resudial = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += resudial

        return out


class SRResNet(nn.Module):
    """SRResNet模型(4x)"""

    def __init__(self,conv=common.default_conv):
        """初始化模型配置"""
        super(SRResNet, self).__init__()

        # 卷积模块1
        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, padding=1, padding_mode='reflect', stride=1)
        self.relu = nn.PReLU()
        # 残差模块
        self.resBlock = self._makeLayer_(ResBlock, 64, 64, 10)
        # 卷积模块2
        self.conv2 = nn.Conv3d(64, 64, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.relu2 = nn.PReLU()

        # 子像素卷积
        self.convPos1 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.pixelShuffler1 = Upsampler3d(conv,scale=2,n_feats=64,act=False)
        self.reluPos1 = nn.PReLU()

        self.convPos2 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.pixelShuffler2 = Upsampler3d(conv, scale=2, n_feats=64, act=False)
        self.reluPos2 = nn.PReLU()

        self.finConv = nn.Conv3d(64, 1, kernel_size=3, stride=1, padding=1)

    def _makeLayer_(self, block, inChannals, outChannals, blocks):
        """构建残差层"""
        layers = []
        layers.append(block(inChannals, outChannals))

        for i in range(1, blocks):
            layers.append(block(outChannals, outChannals))

        return nn.Sequential(*layers)

    def forward(self, x):
        """前向传播过程"""
        x = self.conv1(x)
        x = self.relu(x)
        residual = x

        out = self.resBlock(x)

        out = self.conv2(out)
        out = self.bn2(out)
        out += residual


        out = self.convPos1(out)
        out = self.pixelShuffler1(out)
        out = self.reluPos1(out)
        out = self.convPos2(out)
        out = self.pixelShuffler2(out)
        out = self.reluPos2(out)
        out = self.finConv(out)

        return out

if __name__ == '__main__':

    model=SRResNet()
    x=torch.rand(1,1,10,10,10)
    print(model(x).shape)
    print('%s Created, Parameters: %d' % (model.__class__.__name__, calculate_parameters(model)))