import os
import sys
sys.path.append(os.path.abspath('.'))
import torch.nn as nn
from MBHASR.model import common
import torch.nn.functional as F
import torch
from MBHASR.model.common import BAM


# 1.2d:channel=64->3d:channel=32,
# 2.4个resblock，2个ResidualGroup保持不变
class HAMSR3D(nn.Module):
    def __init__(self, n_resblocks=2, n_feats=64, kernel_size=3, n_colors=1,
                 reduction=16, act=nn.ReLU(), conv=common.default_conv):
        super(HAMSR3D, self).__init__()
        scale = 4

        m_head = [conv(n_colors, n_feats, kernel_size)]

        m_body1 = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=0.1, n_resblocks=n_resblocks) \
            for _ in range(1)
        ]
        m_body1.append(conv(n_feats, n_feats, kernel_size))

        m_body11 = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=0.1, n_resblocks=n_resblocks) \
            for _ in range(1)
        ]
        m_body11.append(conv(n_feats, n_feats, kernel_size))

        m_body12 = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=0.1, n_resblocks=n_resblocks) \
            for _ in range(1)
        ]
        m_body12.append(conv(n_feats, n_feats, kernel_size))

        m_body2 = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=0.1, n_resblocks=n_resblocks) \
            for _ in range(1)
        ]
        m_body2.append(conv(n_feats, n_feats, kernel_size))

        m_body21 = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=0.1, n_resblocks=n_resblocks) \
            for _ in range(1)
        ]
        m_body21.append(conv(n_feats, n_feats, kernel_size))

        m_body22 = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=0.1, n_resblocks=n_resblocks) \
            for _ in range(1)
        ]
        m_body22.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail11 = [
            common.Upsampler3d(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size)
        ]
        m_tail12 = [
            common.Upsampler3d(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size)
        ]
        m_tail21 = [
            common.Upsampler3d(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size)
        ]
        m_tail22 = [
            common.Upsampler3d(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size)
        ]

        self.res_conv1 = conv(n_feats, n_feats, kernel_size)
        self.res_conv2 = conv(n_feats, n_feats, kernel_size)
        self.res_conv3 = conv(n_feats, n_feats, kernel_size)
        self.res_conv4 = conv(n_feats, n_feats, kernel_size)
        self.res_conv5 = conv(n_feats, n_feats, kernel_size)
        self.res_conv6 = conv(n_feats, n_feats, kernel_size)
        self.res_conv7 = conv(n_feats, n_feats, kernel_size)
        self.res_conv8 = conv(n_feats, n_feats, kernel_size)

        self.head = nn.Sequential(*m_head)

        self.body1 = nn.Sequential(*m_body1)
        self.body11 = nn.Sequential(*m_body11)
        self.body12 = nn.Sequential(*m_body12)

        self.body2 = nn.Sequential(*m_body2)
        self.body21 = nn.Sequential(*m_body21)
        self.body22 = nn.Sequential(*m_body22)

        self.tail11 = nn.Sequential(*m_tail11)
        self.tail12 = nn.Sequential(*m_tail12)
        self.tail21 = nn.Sequential(*m_tail21)
        self.tail22 = nn.Sequential(*m_tail22)

        # 3.dropout策略
        self.dropout = nn.Dropout(0.2)

        self.fusion_final = nn.Sequential(
            common.BasicConv(4, 64, 3, stride=1, padding=1, relu=True),
            # BAM(channel=32),
            common.BasicConv(64, 64, 3, stride=1, padding=1, relu=True),
            # BAM(channel=32),
            common.BasicConv(64, 4, 3, stride=1, padding=1, relu=True))

    def forward(self, x):

        x_head = self.head(x)

        res1 = self.body1(x_head)
        res2 = self.body2(x_head)

        res11 = self.body11(res1)
        res12 = self.body12(res1)

        res21 = self.body21(res2)
        res22 = self.body22(res2)

        res11=self.res_conv1(res11)
        res11 += res1
        res11=self.res_conv2(res11)
        res11 += x_head
        res11 = self.dropout(res11)
        res11 = self.tail11(res11)

        res12=self.res_conv3(res12)
        res12 += res1
        res12=self.res_conv4(res12)
        res12 += x_head
        res12 = self.dropout(res12)
        res12 = self.tail12(res12)

        res21=self.res_conv5(res21)
        res21 += res2
        res21=self.res_conv6(res21)
        res21 += x_head
        res21 = self.dropout(res21)
        res21 = self.tail21(res21)

        res22=self.res_conv7(res22)
        res22 += res2
        res22=self.res_conv8(res22)
        res22 += x_head
        res22 = self.dropout(res22)
        res22 = self.tail22(res22)

        cat_out = torch.concat((res11, res12, res21, res22), dim=1)
        mask = self.fusion_final(cat_out)
        mask = F.softmax(mask, dim=1)
        mask1 = mask[:, 0, ...].unsqueeze(1)
        mask2 = mask[:, 1, ...].unsqueeze(1)
        mask3 = mask[:, 2, ...].unsqueeze(1)
        mask4 = mask[:, 3, ...].unsqueeze(1)
        fusion = res11 * mask1 + res12 * mask2 + res21 * mask3 + res22 * mask4
        return fusion

class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, n_resblocks,act,res_scale=0.1):
        super(ResidualGroup, self).__init__()
        modules_body = [
            MIRCAB(
                in_channels=64, out_channels=64, expansion=4) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class MIRCAB(nn.Module):
    def __init__(
        self, in_channels, out_channels, expansion,stride=1):
        super(MIRCAB, self).__init__()
        channels=expansion*in_channels

        self.basic_block = nn.Sequential(
            nn.Conv3d(in_channels, channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.ReLU6(),
            # DepthWiseConv(in_channel=256, out_channel=256),
            nn.Conv3d(channels, channels, kernel_size=3, stride=stride, padding=1, groups=channels, bias=False),
            # nn.BatchNorm3d(channels),
            nn.ReLU6(),
            nn.Conv3d(channels, out_channels, kernel_size=1, stride=1, bias=False),
            # nn.BatchNorm3d(out_channels),
        )
        self.CA=CALayer(in_channels,reduction=16)

    def forward(self, x):
        out = self.basic_block(x)
        out = self.CA(out)
        out = out + x

        return out

class RCAB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size, reduction,
            bias=True, bn=False, act=nn.ReLU(), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res += x
        return res

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv3d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv3d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class ConvolutionalBlock(nn.Module):
    """
    卷积模块,由卷积层, BN归一化层, 激活层构成.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, batch_norm=False, activation=None):
        """
        :参数 in_channels: 输入通道数
        :参数 out_channels: 输出通道数
        :参数 kernel_size: 核大小
        :参数 stride: 步长
        :参数 batch_norm: 是否包含BN层
        :参数 activation: 激活层类型; 如果没有则为None
        """
        super(ConvolutionalBlock, self).__init__()

        if activation is not None:
            activation = activation.lower()
            assert activation in {'prelu', 'leakyrelu', 'tanh'}

        # 层列表
        layers = list()

        # 1个卷积层
        layers.append(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2))

        # 1个BN归一化层
        if batch_norm is True:
            layers.append(nn.BatchNorm3d(num_features=out_channels))

        # 1个激活层
        if activation == 'prelu':
            layers.append(nn.PReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.2))
        elif activation == 'tanh':
            layers.append(nn.Tanh())

        # 合并层
        self.conv_block = nn.Sequential(*layers)

    def forward(self, input):
        """
        前向传播

        :参数 input: 输入图像集，张量表示，大小为 (N, in_channels, w, h)
        :返回: 输出图像集，张量表示，大小为(N, out_channels, w, h)
        """
        output = self.conv_block(input)

        return output

class Discriminator(nn.Module):
    """
    SRGAN判别器
    """

    def __init__(self, kernel_size=3, n_channels=32, n_blocks=7, fc_size=1024):
        """
        参数 kernel_size: 所有卷积层的核大小
        参数 n_channels: 初始卷积层输出通道数, 后面每隔一个卷积层通道数翻倍
        参数 n_blocks: 卷积块数量
        参数 fc_size: 全连接层连接数
        """
        super(Discriminator, self).__init__()

        in_channels = 1

        # 卷积系列，参照论文SRGAN进行设计
        conv_blocks = list()
        for i in range(n_blocks):
            out_channels = (n_channels if i == 0 else in_channels * 2) if i % 2 == 0 else in_channels
            conv_blocks.append(
                ConvolutionalBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=1 if i % 2 == 0 else 2, batch_norm=i != 0, activation='LeakyReLu'))
            in_channels = out_channels
        self.conv_blocks = nn.Sequential(*conv_blocks)

        # 固定输出大小
        self.adaptive_pool = nn.AdaptiveAvgPool3d((6, 6, 6))

        self.fc1 = nn.Linear(out_channels * 6 * 6 * 6, fc_size)

        self.leaky_relu = nn.LeakyReLU(0.2)

        self.fc2 = nn.Linear(1024, 1)

        # 最后不需要添加sigmoid层，因为PyTorch的nn.BCEWithLogitsLoss()已经包含了这个步骤

    def forward(self, imgs):

        """
        前向传播.

        参数 imgs: 用于作判别的原始高清图或超分重建图，张量表示，大小为(N, 3, w * scaling factor, h * scaling factor)
        返回: 一个评分值， 用于判断一副图像是否是高清图, 张量表示，大小为 (N)
        """
        batch_size = imgs.size(0)
        output = self.conv_blocks(imgs)
        output = self.adaptive_pool(output)
        output = self.fc1(output.view(batch_size, -1))
        output = self.leaky_relu(output)
        logit = self.fc2(output)

        return logit
