import torch
import math
import torch.nn as nn
import torch.nn.functional as F

# 1.
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv3d(in_channels, out_channels, kernel_size,
                     padding=(kernel_size//2), bias=bias)

class Scale(nn.Module):

    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4516, 0.4516, 0.4516), rgb_std=(1.0, 1.0, 1.0), sign=-1):  # 0.4488, 0.4371, 0.4040

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act = nn.LeakyReLU(negative_slope=0.01, inplace=True)):   # act=nn.ReLU(True)

        m = [conv(in_channels, out_channels, kernel_size, stride=stride,bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

# 2.
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.LeakyReLU(negative_slope=0.01, inplace=True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class LuConv(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.LeakyReLU(0.05), res_scale=1):
        super(LuConv, self).__init__()
        # self.scale1 = Scale(1)
        # self.scale2 = Scale(1)
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feats))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

# 3.
class PixelShuffle3d(nn.Module):
    '''
    This class is a 3d version of pixelshuffle.
    '''
    def __init__(self, scale):
        '''
        :param scale: upsample scale
        '''
        super().__init__()
        self.scale = scale

    def forward(self, input):
        batch_size, channels, in_depth, in_height, in_width = input.size()
        nOut = channels // self.scale ** 3

        out_depth = in_depth * self.scale
        out_height = in_height * self.scale
        out_width = in_width * self.scale

        input_view = input.contiguous().view(batch_size, nOut, self.scale, self.scale, self.scale, in_depth, in_height, in_width)

        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return output.view(batch_size, nOut, out_depth, out_height, out_width)

# 4.
class Upsampler3d(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 8 * n_feats, 3, bias))
                m.append(PixelShuffle3d(2))
                if bn:
                    m.append(nn.BatchNorm3d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm3d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler3d, self).__init__(*m)

# 5.通道注意力
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):# reduction=16，通道压缩因子
        super(ChannelAttention, self).__init__()
        mid_channel = channel // reduction # 中间通道数

        self.avg_pool = nn.AdaptiveAvgPool3d(1)# 全局平均池化,(b,c,1,1)
        self.shared_MLP = nn.Sequential(
            nn.Linear(in_features=channel, out_features=mid_channel),# 第一个线性层使特征通道数减少
            nn.ReLU(inplace=True),# 激活函数
            nn.Linear(in_features=mid_channel, out_features=channel)# 第二个线性层使特征通道数还原
        )

    def forward(self, x):
        avg = self.avg_pool(x).view(x.size(0), -1)# avg:(b,c)
        out = self.shared_MLP(avg).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x)# out:(b,c,1,1,1)
        return out # (b,c,h,w,d)


# 6.空间注意力
class SpatialAttention(nn.Module):
    # dilation_conv_num：空洞卷积个数，dilation_rate：膨胀率
    def __init__(self, channel, reduction=16, dilation_conv_num=2, dilation_rate=4):
        super(SpatialAttention, self).__init__()
        mid_channel = channel // reduction# 中间通道数
        self.reduce_conv = nn.Sequential(
            nn.Conv3d(channel, mid_channel, kernel_size=1),# 1x1卷积减少通道数
            nn.BatchNorm3d(mid_channel),# 批归一化
            nn.ReLU(inplace=True)# 激活函数
        )
        dilation_convs_list = []# 空洞卷积列表
        for i in range(dilation_conv_num):# 共有dilation_conv_num个空洞卷积
            dilation_convs_list.append(
                nn.Conv3d(mid_channel, mid_channel, kernel_size=3,
                        padding=dilation_rate, dilation=dilation_rate))# 添加一个3x3空洞卷积层
            dilation_convs_list.append(nn.BatchNorm3d(mid_channel))# 添加一个批归一化层
            dilation_convs_list.append(nn.ReLU(inplace=True))# 添加激活函数
        self.dilation_convs = nn.Sequential(*dilation_convs_list)# 解包
        self.final_conv = nn.Conv3d(mid_channel, channel, kernel_size=1)# 最后添加一个1x1卷积使通道数还原

    def forward(self, x):
        y = self.reduce_conv(x)# 第一个1x1卷积
        y = self.dilation_convs(y)# 空洞卷积
        out = self.final_conv(y).expand_as(x)# 第二个1x1卷积
        return out

class ECAttention(nn.Module):# ECA注意力机制模块，不降维
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, num_feat,k_size=3):# k_size=5
        super(ECAttention, self).__init__()
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(num_feat, num_feat, kernel_size=k_size, padding=int(k_size//2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y=self.avg_pool(x).squeeze(-1)
        y=self.conv(y).unsqueeze(-1)
        y=self.sigmoid(y)
        return x * y

# 7.
class BAM(nn.Module):
    """
        BAM: Bottleneck Attention Module
        https://arxiv.org/pdf/1807.06514.pdf
    """

    def __init__(self, channel):
        super(BAM, self).__init__()
        self.channel_attention = ChannelAttention(channel)# 通道注意力函数
        self.spatial_attention = SpatialAttention(channel)# 空间注意力函数
        self.sigmoid = nn.Sigmoid()# sigmoid激活函数

    def forward(self, x):
        att = 1 + self.sigmoid(self.channel_attention(x) * self.spatial_attention(x))# 得到BAM混合注意力
        return att * x # 将注意力与输入x相乘得到输出

#  深度残差模块
class DepthWiseConv(nn.Module):
    def __init__(self, in_channel=3, out_channel=24):
        # 这一行千万不要忘记
        super(DepthWiseConv, self).__init__()

        # 逐通道卷积
        self.depth_conv = nn.Conv3d(in_channels=in_channel,
                                    out_channels=in_channel,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_channel)
        # groups是一个数，当groups=in_channel时,表示做逐通道卷积

        # 逐点卷积
        self.point_conv = nn.Conv3d(in_channels=in_channel,
                                    out_channels=out_channel,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)


    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

## Multi-path Adaptive Modulation(MAM) Layer
class MAMLayer(nn.Module):# 多路径自适应调制层
    def __init__(self, channels, reduction=16):
        super(MAMLayer, self).__init__()
        # feature channel downscale and upscale --> channel weight 特征通道上采样和下采样
        self.conv_du = nn.Sequential(
                nn.Conv2d(channels, channels // reduction, 1, padding=0, bias=True),# 添加第一个1x1卷积层，通道数/通道缩放因子
                nn.ReLU(inplace=True),# 添加激活函数
                # nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(channels // reduction, channels, 1, padding=0, bias=True),# 添加第二个1x1卷积层，通道数还原
        )
        # depthwise convolution 深度卷积
        self.csd = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=True)# 添加第三个3x3卷积,通道数不变
        self.act = nn.Sigmoid()# 添加激活函数

    def forward(self, x):# 前向传播
        var = torch.var(x, dim=(2, 3), keepdim=True)# 计算方差
        var = F.normalize(var, p=2, dim=1)  # var normalization 方差归一化
        ca_out = self.conv_du(var)# 经过上下采样卷积层
        csd_out = self.csd(x)# 经过3x3卷积层
        y = var + ca_out + csd_out# 加入方差
        y = self.act(y)# 通过激活函数
        return x * y # 返回x*y

# 空间注意力
class SpatialAttention_(nn.Module):
    def __init__(self, kernel_size=3,n_feat=64):
        super(SpatialAttention_, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'

        self.avg_atten = nn.AvgPool3d(3, stride=1, padding=1)  # 定义平均池化
        self.max_atten = nn.MaxPool3d(3, stride=1, padding=1)  # 定义最大池化
        self.conv1 = nn.Conv3d(2*n_feat, n_feat, kernel_size=3, padding=1, bias=False)  # 输入两个通道，一个是maxpool 一个是avgpool的
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_atten(x)
        max_out = self.max_atten(x)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)  # 对池化完的数据cat 然后进行卷积
        return self.sigmoid(x)


