import torch
import torch.nn as nn
import torch.nn.functional as F
from MBHASR.model import common
from MBHASR.model.common import *
from torchvision import ops
# from basicsr.utils.registry import ARCH_REGISTRY

# Layer Norm
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None,None]
            return x


# SE
class SqueezeExcitation(nn.Module):
    def __init__(self, dim, shrinkage_rate=0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(dim, hidden_dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv3d(hidden_dim, dim, 1, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x)


# Channel MLP: Conv1*1 -> Conv1*1
class ChannelMLP(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.mlp = nn.Sequential(
            nn.Conv3d(dim, hidden_dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv3d(hidden_dim, dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.mlp(x)


# MBConv: Conv1*1 -> DW Conv3*3 -> [SE] -> Conv1*1
class MBConv(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.mbconv = nn.Sequential(
            nn.Conv3d(dim, hidden_dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv3d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim),
            nn.GELU(),
            SqueezeExcitation(hidden_dim),
            nn.Conv3d(hidden_dim, dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.mbconv(x)


# CCM
class CCM(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.ccm = nn.Sequential(
            nn.Conv3d(dim, hidden_dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv3d(hidden_dim, dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.ccm(x)

class GDFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(GDFN, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv3d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv3d(hidden_features*2, hidden_features*2,kernel_size=3,
                                stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Sequential(
            # SqueezeExcitation(hidden_features),
            nn.Conv3d(hidden_features, dim, kernel_size=1, bias=bias),
        )

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

# SAFM
class SAFM(nn.Module):
    def __init__(self, dim, n_levels=4):
        super().__init__()
        self.n_levels = n_levels
        chunk_dim = dim // n_levels

        # Spatial Weighting
        self.mfr = nn.ModuleList(
            [nn.Conv3d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim) for i in range(self.n_levels)])

        # # Feature Aggregation
        self.aggr = nn.Conv3d(dim, dim, 1, 1, 0)

        # Activation
        self.act = nn.GELU()

    def forward(self, x):
        l, h, w = x.size()[-3:]

        xc = x.chunk(self.n_levels, dim=1)
        out = []
        for i in range(self.n_levels):
            if i > 0:
                p_size = (l // 2 ** i, h // 2 ** i, w // 2 ** i)
                s = F.adaptive_max_pool3d(xc[i], p_size)
                s = self.mfr[i](s)
                s = F.interpolate(s, size=(l, h, w), mode='nearest')
            else:
                s = self.mfr[i](xc[i])
            out.append(s)

        out = self.aggr(torch.cat(out, dim=1))
        out = self.act(out) * x
        return out


class AttBlock(nn.Module):
    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()

        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)

        # Multiscale Block
        self.safm = SAFM(dim)
        # Feedforward layer
        # self.ccm = CCM(dim, ffn_scale)
        # self.mbc = MBConv(dim=64)
        self.gdfn = GDFN(dim=64,ffn_expansion_factor=2,bias=False)

    def forward(self, x):
        x = self.safm(self.norm1(x)) + x
        # x = self.ccm(self.norm2(x)) + x
        x = self.gdfn(self.norm2(x)) + x
        # x = self.mbc(self.norm2(x)) + x
        return x


# @ARCH_REGISTRY.register()
class SAFMN(nn.Module):
    def __init__(self, dim, n_blocks=8, ffn_scale=2.0,conv=common.default_conv):
        super().__init__()
        self.to_feat = nn.Conv3d(1, dim, 3, 1, 1)

        self.feats = nn.Sequential(*[AttBlock(dim, ffn_scale) for _ in range(n_blocks)])


        self.to_img = nn.Sequential(
            nn.Conv3d(dim, dim, 3, 1, 1),
            Upsampler3d(conv, scale=2, n_feats=dim, act=False),
            nn.Conv3d(dim, dim, 3, 1, 1),
            Upsampler3d(conv, scale=2, n_feats=dim, act=False),
            nn.Conv3d(dim, 1, 3, 1, 1),
        )

    def forward(self, x):
        x = self.to_feat(x)
        x = self.feats(x) + x
        x = self.to_img(x)
        return x

if __name__ == '__main__':
    #############Test Model Complexity #############
    # from torch.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis

    # x = torch.randn(1, 3, 640, 360)
    # x = torch.randn(1, 3, 427, 240)
    x = torch.randn(1, 1, 10, 10, 10)
    # x = torch.randn(1, 3, 256, 256)
    # model=SAFMN(dim=64)
    # print(model(x).shape)

    model = SAFMN(dim=64, n_blocks=12, ffn_scale=2.0)
    # model = SAFMN(dim=36, n_blocks=12, ffn_scale=2.0, upscaling_factor=2)
    print(model)
    print(model(x).shape)
    print(f'params: {sum(map(lambda x: x.numel(), model.parameters()))}')
    # print(flop_count_table(FlopCountAnalysis(model, x), activations=ActivationCountAnalysis(model, x)))
    # output = model(x)
    # print(output.shape)