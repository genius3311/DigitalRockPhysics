import torch
import torch.nn as nn
from math import sqrt
from MBHASR.model import common

"""
SRCNN:先将低分辨率图像插值成与高分辨率图像相同大小，再进行训练
"""
# 3D卷积块+激活函数 channel=64->64

class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.conv(x))

conv=common.default_conv
m_tail11 = [
            common.Upsampler3d(conv, 4, 64, act=False),
            conv(64, 64, 3)
        ]
# 整体网络结构，输入卷积层(1->64)+残差卷积层(64->64)+输出卷积层(64->1)
class Net(nn.Module):
    def __init__(self, layers_num):
        super(Net, self).__init__()
        self.residual_layer = self.make_layer(Conv_ReLU_Block, num_of_layer=layers_num)#add layers up to 22
        self.input = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.upsampler = nn.Sequential(*m_tail11)
        self.output = nn.Conv3d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
    
        for m in self.modules():# 如果是3d卷积层，进行权重初始化
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] *m.kernel_size[2]* m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                
    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):# 共num_of_layer层
            layers.append(block())# 每一层添加一个block()
        return nn.Sequential(*layers)

    def forward(self, x):# 前向传播

        residual = x  # 初始输入x作为残差块
        out = self.relu(self.input(x))# 输入卷积层(1->64)
        out = self.residual_layer(out)# 残差卷积层(64->64),22层
        out = torch.add(out, residual)  # 加入残差
        out = self.upsampler(out)
        out = self.output(out)# 输出卷积层(64->1)

        return out # 返回输出
    
# if __name__=='__main__':
#     vdsr=Net(layers_num=32)
#     x=torch.randn(1,1,12,12,12)
#     y=vdsr(x)
#     print(y.shape)
