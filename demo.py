import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels = 96, out_channels = 96, kernel = (3, 3), pad = (1, 1)):
        super(ConvBlock, self).__init__()
        self.relu = nn.ReLU(inplace = True)
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel,
                               padding = pad)
        self.bn1 = nn.BatchNorm2d(num_features = in_channels)
        self._initialize_weights_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x

    def _initialize_weights_(self):
        makeDeltaOrthogonal(self.conv1.weight, init.calculate_gain('relu'))


def genOrthgonal(dim):
    a = torch.zeros((dim, dim)).normal_(0, 1)
    q, r = torch.qr(a)
    d = torch.diag(r, 0).sign()
    diag_size = d.size(0)
    d_exp = d.view(1, diag_size).expand(diag_size, diag_size)
    q.mul_(d_exp)
    return q


def makeDeltaOrthogonal(weights, gain):
    rows = weights.size(0)
    cols = weights.size(1)
    if rows > cols:
        print("In_filters should not be greater than out_filters.")
    weights.data.fill_(0)
    dim = max(rows, cols)
    q = genOrthgonal(dim)
    mid1 = weights.size(2) // 2
    mid2 = weights.size(3) // 2
    weights[:, :, mid1, mid2] = q[:weights.size(0), :weights.size(1)]
    weights.mul_(gain)


if __name__ == '__main__':
    conv_block = ConvBlock()