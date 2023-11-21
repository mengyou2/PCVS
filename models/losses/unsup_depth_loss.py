from __future__ import division
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from models.losses.inverse_warp import inverse_warp
import math

class photometric_reconstruction_loss(nn.Module):
    def __init__(self):
        super(photometric_reconstruction_loss, self).__init__()
        self.ssim = SSIM()
    def forward(self,tgt_img, ref_imgs, depth,poses,intrinsics):                                   
        warped_results,mask_results = [], []

        if type(depth) not in [list, tuple]:
            depth = [depth]

        total_ploss = 0
        total_sloss = 0
        total_smloss = 0
        # total_semloss = 0

        for ref_img, pose in zip(ref_imgs, poses):
            warped_result,mask_result = [], []
            for d in depth:

                b,_, h, w = d.size()
                if tgt_img.size(2)!=h:
                    downscale = tgt_img.size(2)/h
                    
                    tgt_img_scaled = F.interpolate(tgt_img, (h, w), mode='area')
                    ref_img_scaled = F.interpolate(ref_img, (h, w), mode='area') 
                    intrinsics_scaled = torch.clone(intrinsics)
                    intrinsics_scaled[:,0,0] = intrinsics_scaled[:,0,0] / downscale
                    intrinsics_scaled[:,1,1] = intrinsics_scaled[:,1,1] / downscale
                    intrinsics_scaled[:,0,2] = intrinsics_scaled[:,0,2] / downscale
                    intrinsics_scaled[:,1,2] = intrinsics_scaled[:,1,2] / downscale
                else:
                    ref_img_scaled = torch.clone(ref_img)
                    intrinsics_scaled = torch.clone(intrinsics)
                    tgt_img_scaled = torch.clone(tgt_img)

                ref_img_warped, mask= inverse_warp(ref_img_scaled, d,pose,intrinsics_scaled)
                reconstruction_loss = compute_reconstr_loss(ref_img_warped, tgt_img_scaled, mask, simple=False)

                ssim_loss = torch.mean(self.ssim(tgt_img_scaled, ref_img_warped, mask))
                smoothness_loss = depth_smoothness(d, tgt_img_scaled)



                total_ploss += reconstruction_loss
                total_sloss += ssim_loss
                total_smloss += smoothness_loss
                warped_result.append(ref_img_warped)
                mask_result.append(mask)
            warped_results.append(warped_result)
            mask_results.append(mask_result)
        return total_ploss,total_sloss,total_smloss,warped_results,mask_results

class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)
        self.mask_pool = nn.AvgPool2d(3, 1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y, mask):

        mask = mask.float()


        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)
        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y
        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        SSIM_mask = self.mask_pool(mask)
        output = SSIM_mask * torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
        return output 



def gradient_x(img):
    return img[:, :, :, :-1] - img[:, :, :, 1:]

def gradient_y(img):
    return img[:, :, :-1, :] - img[:, :, 1:, :]

def gradient(pred):
    D_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    return D_dx, D_dy

def depth_smoothness(depth, img,lambda_wt=1):
    """Computes image-aware depth smoothness loss."""
    depth_dx = gradient_x(depth)
    depth_dy = gradient_y(depth)
    image_dx = gradient_x(img)
    image_dy = gradient_y(img)
    weights_x = torch.exp(-(lambda_wt * torch.mean(torch.abs(image_dx), 3, keepdim=True)))
    weights_y = torch.exp(-(lambda_wt * torch.mean(torch.abs(image_dy), 3, keepdim=True)))
    smoothness_x = depth_dx * weights_x
    smoothness_y = depth_dy * weights_y
    return torch.mean(torch.abs(smoothness_x)) + torch.mean(torch.abs(smoothness_y))
def depth_semantic_smoothness(depth, semantic,lambda_wt=1):
    """Computes image-aware depth smoothness loss."""
    depth_dx = gradient_x(depth)
    depth_dy = gradient_y(depth)
    image_dx = gradient_x(semantic)
    image_dy = gradient_y(semantic)
    weights_x = torch.exp(-(lambda_wt * torch.mean(torch.abs(image_dx), 3, keepdim=True)))
    weights_y = torch.exp(-(lambda_wt * torch.mean(torch.abs(image_dy), 3, keepdim=True)))
    smoothness_x = depth_dx * weights_x
    smoothness_y = depth_dy * weights_y
    return torch.mean(torch.abs(smoothness_x)) + torch.mean(torch.abs(smoothness_y))

def compute_reconstr_loss(warped, ref, mask, simple=True):
    if simple:
        return F.smooth_l1_loss(warped*mask, ref*mask, reduction='mean')
    else:
        alpha = 0.5
        ref_dx, ref_dy = gradient(ref * mask)
        warped_dx, warped_dy = gradient(warped * mask)
        photo_loss = F.smooth_l1_loss(warped*mask, ref*mask, reduction='mean')
        grad_loss = F.smooth_l1_loss(warped_dx, ref_dx, reduction='mean') + \
                    F.smooth_l1_loss(warped_dy, ref_dy, reduction='mean')
        return (1 - alpha) * photo_loss + alpha * grad_loss

