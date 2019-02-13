import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from ednet import EDNet


def dpn_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 1.0)


class DPN(nn.Module):

    def __init__(self, in_shape):
        super(DPN, self).__init__()
        self.ednet = EDNet(in_shape, 4, 2, 0.008)

    def forward(self, rgb, nir):
        disps = self.ednet(torch.cat((rgb, nir), 1))
        ldisps = [disps[0][:, :1, :, :], disps[1][:, :1, :, :], disps[2][:, :1, :, :], disps[3][:, :1, :, :]]
        rdisps = [disps[0][:, 1:, :, :], disps[1][:, 1:, :, :], disps[2][:, 1:, :, :], disps[3][:, 1:, :, :]]
        return ldisps, rdisps
