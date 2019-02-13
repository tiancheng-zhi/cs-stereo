import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class EDNet(nn.Module):

    def __init__(self, in_shape, in_channels, out_channels, out_scale):
        super(EDNet, self).__init__()
        shape = [(in_shape[0] // (2 ** i), in_shape[1] // (2 ** i)) for i in range(8)]
        nf = [16 * (2 ** x) for x in range(6)]
        self.e1 = Encoder(shape[0], shape[1], in_channels, nf[1], 7)
        self.e2 = Encoder(shape[1], shape[2], nf[1], nf[2], 5)
        self.e3 = Encoder(shape[2], shape[3], nf[2], nf[3])
        self.e4 = Encoder(shape[3], shape[4], nf[3], nf[4])
        self.e5 = Encoder(shape[4], shape[5], nf[4], nf[5])
        self.e6 = Encoder(shape[5], shape[6], nf[5], nf[5])
        self.e7 = Encoder(shape[6], shape[7], nf[5], nf[5])
        self.d7 = Decoder(shape[6], nf[5], nf[5], out_channels, nf[5])
        self.d6 = Decoder(shape[5], nf[5], nf[5], out_channels, nf[5])
        self.d5 = Decoder(shape[4], nf[5], nf[4], out_channels, nf[4])
        self.d4 = Decoder(shape[3], nf[4], nf[3], out_channels, nf[3], out_scale)
        self.d3 = Decoder(shape[2], nf[3], nf[2], out_channels, nf[2] + out_channels, out_scale)
        self.d2 = Decoder(shape[1], nf[2], nf[1], out_channels, nf[1] + out_channels, out_scale)
        self.d1 = Decoder(shape[0], nf[1], nf[0], out_channels, out_channels, out_scale)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        d7 = self.d7(e7, e6)
        d6 = self.d6(d7, e5)
        d5 = self.d5(d6, e4)
        d4, out4 = self.d4(d5, e3)
        out4up = F.interpolate(out4, e2.size()[2:], mode='bilinear')
        d3, out3 = self.d3(d4, torch.cat((e2, out4up), 1))
        out3up = F.interpolate(out3, e1.size()[2:], mode='bilinear')
        d2, out2 = self.d2(d3, torch.cat((e1, out3up), 1))
        out2up = F.interpolate(out2, x.size()[2:], mode='bilinear')
        d1, out1 = self.d1(d2, out2up)
        outs = [out1, out2, out3, out4]
        return outs


class Encoder(nn.Module):

    def __init__(self, in_shape, out_shape, in_channels, out_channels, ksize=3):
        super(Encoder, self).__init__()
        self.conv = Conv2dAP(in_shape, out_shape, in_channels, out_channels, ksize, 2, False)
        self.convbn = nn.BatchNorm2d(out_channels)
        self.convb = Conv2dAP(out_shape, out_shape, out_channels, out_channels, ksize, 1, False)
        self.convbbn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        conv = F.elu(self.convbn(self.conv(x)))
        convb = F.elu(self.convbbn(self.convb(conv)))
        return convb


class Decoder(nn.Module):

    def __init__(self, out_shape, in_channels, out_channels, final_out_channels, skip_channels, out_scale=0, ksize=3):
        super(Decoder, self).__init__()
        self.out_shape = out_shape
        self.conv = Conv2dAP(out_shape, out_shape, in_channels, out_channels, ksize, 1, False)
        self.convbn = nn.BatchNorm2d(out_channels)
        self.iconv = Conv2dAP(out_shape, out_shape, out_channels + skip_channels, out_channels, ksize, 1, False)
        self.iconvbn = nn.BatchNorm2d(out_channels)
        if out_scale > 0:
            self.out = Conv2dAP(out_shape, out_shape, out_channels, final_out_channels, ksize, 1, True)
            self.out_scale = out_scale
        else:
            self.out = None

    def forward(self, x, skip):
        upconv = F.elu(self.convbn(self.conv(F.interpolate(x, self.out_shape, mode='bilinear'))))
        iconv = F.elu(self.iconvbn(self.iconv(torch.cat((upconv, skip), 1))))
        if self.out is not None:
            out = self.out_scale * F.elu(self.out(iconv))
            return iconv, out
        else:
            return iconv


class Conv2dAP(nn.Module):
    def __init__(self, in_shape, out_shape, in_c, out_c, ksize, stride,
                 bias):
        super(Conv2dAP, self).__init__()
        if isinstance(ksize, int):
            ksize = (ksize, ksize)
        if isinstance(stride, int):
            stride = (stride, stride)
        pad = [0, 0]
        pad[0] = int(
            math.ceil((stride[0] * (out_shape[0] - 1) - in_shape[0] + ksize[0])
                      / 2))
        pad[1] = int(
            math.ceil((stride[1] * (out_shape[1] - 1) - in_shape[1] + ksize[1])
                      / 2))
        assert(pad[0] >= 0 and pad[1] >= 0)
        assert(pad[0] < ksize[0] and pad[1] < ksize[1])
        pad = tuple(pad)
        self.layer = nn.Conv2d(in_c, out_c, ksize, stride, pad, bias=bias)
        self.in_s = in_shape
        self.out_s = out_shape
        self.in_c = in_c
        self.out_c = out_c
        self.ksize = ksize
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        xs = x.size()
        assert(xs[1] == self.in_c and
               xs[2] == self.in_s[0] and xs[3] == self.in_s[1])
        y = self.layer(x)
        ys = y.size()
        assert(ys[2] == self.out_s[0] and ys[3] == self.out_s[1])
        return y
