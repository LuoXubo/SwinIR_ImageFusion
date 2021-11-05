import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
EPSILON = 1e-10


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size/2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, X):
        out = self.reflection_pad(X)
        out = self.conv2d(out)
        if self.is_last is False:
            out = F.relu(out, inplace=True)

        return out


class FusionBlock_res(torch.nn.Module):
    def __init__(self, channels):
        super(FusionBlock_res, self).__init__()
        ws = 3
        self.conv_fusion = ConvLayer(2*channels, channels, ws, 1)

        self.conv_ir = ConvLayer(channels, channels, ws, 1)
        self.conv_vi = ConvLayer(channels, channels, ws, 1)

        block = []
        block += [ConvLayer(2*channels, channels, 1, 1),
                  ConvLayer(channels, channels, ws, 1),
                  ConvLayer(channels, channels, ws, 1)]
        self.bottleblock = nn.Sequential(*block)

    def forward(self, x_ir, x_vi):
        f_cat = torch.cat([x_ir, x_vi], 1)
        f_init = self.conv_fusion(f_cat)

        out_ir = self.conv_ir(x_ir)
        out_vi = self.conv_vi(x_vi)
        out = torch.cat([out_ir, out_vi], 1)
        out = self.bottleblock(out)
        out = f_init + out
        return out


if __name__ == '__main__':
    x1 = torch.randn([1,60,256,256])
    x2 = torch.randn([1,60,256,256])

    model = FusionBlock_res(60)
    f = model(x1, x2)
    print(f.shape)
