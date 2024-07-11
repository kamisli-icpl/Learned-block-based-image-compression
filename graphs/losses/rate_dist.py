import logging
import json
import os
from statistics import mean

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from pytorch_msssim import MS_SSIM, ms_ssim, SSIM, ssim
# from visdom import Visdom

# https://github.com/VainF/pytorch-msssim
# https://github.com/VainF/pytorch-msssim/blob/master/pytorch_msssim/ssim.py
# https://github.com/VainF/pytorch-msssim/blob/master/tests/ae_example/train.py


class MSSSIMLoss(MS_SSIM):
    def forward(self, img1, img2):
        return 1*( 1 - super(MSSSIMLoss, self).forward(img1, img2) )


class SSIMLoss(SSIM):
    def forward(self, img1, img2):
        return 1*( 1 - super(SSIMLoss, self).forward(img1, img2) )


class TrainRDLoss(nn.Module):
    def __init__(self, lambda_, distortion="mse", blocksize=1):
        super(TrainRDLoss, self).__init__()
        self.distortion = distortion
        self.B = blocksize
        if distortion == "mse":
            self.mse_loss = nn.MSELoss(reduction='mean')
        elif distortion == "ssim":
            self.mse_loss = SSIMLoss(data_range=1.0, size_average=True, channel=3)  # dont forget to convert img to 3 channels
        elif distortion == "ms_ssim":
            self.mse_loss = MSSSIMLoss(data_range=1.0, size_average=True, channel=3)
        self.lambda_ = lambda_

    def forward(self, x, x_hat, rate):
        if self.distortion == "ssim" or self.distortion == "ms_ssim":
            if self.B > 1:
                x = arrange_channel_dim_to_block_pixels(x + 0.5, self.B, x.device)
                x_hat = arrange_channel_dim_to_block_pixels(x_hat + 0.5, self.B, x_hat.device)
        self.mse  = self.mse_loss(x, x_hat)
        self.rate = torch.sum(rate) / torch.numel(x) * 3
        # self.loss = self.mse + self.lambda_ * self.rate
        self.loss = self.rate + self.lambda_ * self.mse
        return self.loss, self.mse, self.rate

    def forward2(self, x, x_hat, rate1, rate2):
        self.mse  = self.mse_loss(x, x_hat)
        self.rate1 = torch.sum(rate1) / torch.numel(x) * 3
        self.rate2 = torch.sum(rate2) / torch.numel(x) * 3
        # self.loss = self.mse + self.lambda_ * self.rate
        self.loss = self.rate1 + self.rate2 + self.lambda_ * self.mse
        return self.loss, self.mse, self.rate1, self.rate2

    def forward3(self, x, x_hat, rate1, rate2list):
        self.mse  = self.mse_loss(x, x_hat)
        self.rate1 = torch.sum(rate1) / torch.numel(x) * 3
        self.rate2 = 0
        for i in range(len(rate2list)):
            self.rate2 += torch.sum(rate2list[i]) / torch.numel(x) * 3
        self.loss = self.rate1 + self.rate2 + self.lambda_ * self.mse
        return self.loss, self.mse, self.rate1, self.rate2


class TrainDLoss(TrainRDLoss):
    def __init__(self, lambda_, distortion="mse", blocksize=1):
        super(TrainDLoss, self).__init__(lambda_, distortion, blocksize)

    def forward(self, x, x_hat, rate):
        if self.distortion == "ssim" or self.distortion == "ms_ssim":
            if self.B > 1:
                x = arrange_channel_dim_to_block_pixels(x + 0.5, self.B, x.device)
                x_hat = arrange_channel_dim_to_block_pixels(x_hat + 0.5, self.B, x_hat.device)
        self.mse  = self.mse_loss(x, x_hat)
        self.rate = torch.sum(rate) / torch.numel(x) * 3
        # self.loss = self.mse + self.lambda_ * self.rate
        self.loss = 0         + self.lambda_ * self.mse
        return self.loss, self.mse, self.rate

    def forward2(self, x, x_hat, rate1, rate2):
        self.mse  = self.mse_loss(x, x_hat)
        self.rate1 = torch.sum(rate1) / torch.numel(x) * 3
        self.rate2 = torch.sum(rate2) / torch.numel(x) * 3
        # self.loss = self.rate1 + self.rate2 + self.lambda_ * self.mse
        self.loss = 0 + 0 + self.lambda_ * self.mse
        return self.loss, self.mse, self.rate1, self.rate2

    def forward3(self, x, x_hat, rate1, rate2list):
        self.mse  = self.mse_loss(x, x_hat)
        self.rate1 = torch.sum(rate1) / torch.numel(x) * 3
        self.rate2 = 0
        for i in range(len(rate2list)):
            self.rate2 += torch.sum(rate2list[i]) / torch.numel(x) * 3
        self.loss = 0 + 0 + self.lambda_ * self.mse
        return self.loss, self.mse, self.rate1, self.rate2


class ValidRDLoss(nn.Module):
    def __init__(self, lambda_):
        super(ValidRDLoss, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x, x_hat, rate):
        self.mse  = self.psnr(x, x_hat)
        if type(rate) == int:
            rate = torch.tensor([float(rate)])
            # rate = float(rate)
            # self.rate = rate  / torch.numel(x) * 3
        self.rate = torch.sum(rate, dtype=torch.float) / torch.numel(x) * 3
        self.loss = self.mse + self.rate*self.lambda_
        return self.loss, self.mse, self.rate

    def psnr(self, x, x_hat):
        mse  = F.mse_loss(x_hat, x, reduction='none')
        mse  = torch.mean(mse.view(mse.shape[0], -1), 1)
        psnr = -10*torch.log10(mse)
        psnr = torch.mean(psnr)
        return psnr


def arrange_block_pixels_to_channel_dim(x, B, dev):
    C, H, W = x.shape[1], x.shape[2], x.shape[3]
    y = torch.empty(x.shape[0], C*(B**2), H//B, W//B, device=dev)
    for v in range(0, B, 1):
        for h in range(0, B, 1):
            indd = (v*B+h)*C
            y[:, indd:indd+C, :, :] = x[:, :, v::B, h::B]
    return y


def arrange_channel_dim_to_block_pixels(y, B, dev):
    C, H, W = y.shape[1], y.shape[2], y.shape[3]
    C = C//(B**2)
    H = H*B
    W = W*B
    x = torch.empty(y.shape[0], C, H, W, device=dev)
    for v in range(0, B, 1):
        for h in range(0, B, 1):
            indd = (v*B+h)*C
            x[:, :, v::B, h::B] = y[:, indd:indd+C, :, :]
    return x
