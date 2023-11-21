import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import scipy.io as io
import cv2
import random
from models.losses.synthesis import SynthesisLoss
from models.networks.architectures import ResUNet,ResNet
import time
from models.projection.z_buffer_manipulator import PtsManipulator
from models.losses.inverse_warp import inverse_warp
from models.losses.unsup_depth_loss import photometric_reconstruction_loss
from models.networks.mvsnet import MVSNet
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
class ZbufferModelPts(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt
        self.photometric_reconstruction_loss = photometric_reconstruction_loss()
        self.pencoder = ResUNet(depth= 3,out_channels = 32)
        self.encoder = ResUNet(depth= 3,out_channels = 32)
        self.MVSNet = MVSNet()

        self.pts_transformer = PtsManipulator(256, C=67,conf=True, opt=opt)
        self.refine_resnet = ResNet(in_channels=32,out_channels=3,nf=64)
        self.resnet = ResNet(in_channels=9,out_channels=3,nf=64)

        self.loss_function = SynthesisLoss(opt=opt)
    
    def forward(self, batch,isval):
        """ Forward pass of a view synthesis model with a voxel latent field.
        """
        # Input values
        input_img_1 = batch["images"][1]
        input_img_2 = batch["images"][2]
        output_img = batch["images"][0]
        if "depths" in batch.keys():
            depth_img_1 = batch["depths"][1]
            depth_img_2 = batch["depths"][2]
            depth_img_tar = batch["depths"][0]
        # Camera parameters

        K = batch["cameras"][1]["K"]
        K_inv = batch["cameras"][1]["Kinv"]
        input_RT_1 = batch["cameras"][1]["P"]
        input_RTinv_1 = batch["cameras"][1]["Pinv"]
        input_RT_2 = batch["cameras"][2]["P"]
        input_RTinv_2 = batch["cameras"][2]["Pinv"]
        output_RT = batch["cameras"][0]["P"]
        output_RTinv = batch["cameras"][0]["Pinv"]


        pose_1 = batch["cameras"][1]["warp"]
        pose_2 = batch["cameras"][2]["warp"]

        pose_t1 = batch["cameras"][1]["proj"]
        pose_t2 = batch["cameras"][2]["proj"]

        self.opt.num_depth = 128
        depth_values = torch.arange(start=self.opt.min_z,end=self.opt.max_z,step = (self.opt.max_z-self.opt.min_z)/self.opt.num_depth)

        if torch.cuda.is_available():
            input_img_1 = input_img_1.cuda()
            input_img_2 = input_img_2.cuda()
            output_img = output_img.cuda()
            
            if "depths" in batch.keys():
                depth_img_1 = depth_img_1.cuda()
                depth_img_2 = depth_img_2.cuda()
                depth_img_tar = depth_img_tar.cuda()
            K = K.cuda()
            K_inv = K_inv.cuda()
            input_RT_1 = input_RT_1.cuda()
            input_RTinv_1 = input_RTinv_1.cuda()
            input_RT_2 = input_RT_2.cuda()
            input_RTinv_2 = input_RTinv_2.cuda()

            output_RT = output_RT.cuda()
            output_RTinv = output_RTinv.cuda()

            pose_1 = [p1.cuda() for p1 in pose_1 ]
            pose_2 = [p2.cuda() for p2 in pose_2 ]

            pose_t1 = pose_t1.cuda()
            pose_t2 = pose_t2.cuda()
            depth_values = depth_values.cuda()
      

        #code for self-supervised depth estimation
        fs_1 = self.encoder(input_img_1)
        fs_2 = self.encoder(input_img_2) 

        bs,c,w,_ = input_img_1.shape
        depth_values = depth_values.unsqueeze(0).repeat(bs,1)
        depth_1,depth_conf1 = self.MVSNet([fs_1,fs_2],[input_RT_1,input_RT_2],depth_values,K)
        depth_2,depth_conf2 = self.MVSNet([fs_2,fs_1],[input_RT_2,input_RT_1],depth_values,K)

        photoloss_1,ssimloss_1,smoothloss_1,warped1,mask1= self.photometric_reconstruction_loss(input_img_1, [output_img,input_img_2], depth_1,pose_1,K)
        photoloss_2,ssimloss_2,smoothloss_2,warped2,mask2 = self.photometric_reconstruction_loss(input_img_2, [output_img,input_img_1], depth_2,pose_2,K)

        pfs_1 = self.pencoder(input_img_1)
        pfs_2 = self.pencoder(input_img_2) 

        cfs_1 = torch.cat((input_img_1,pfs_1),1)
        cfs_2 = torch.cat((input_img_2,pfs_2),1)
        gen_img,gen_fs,gen_depth= self.pts_transformer.forward_justpts(
            cfs_1,cfs_2,
            depth_1, depth_2,
            K,
            K_inv,
            input_RT_1, input_RT_2,
            input_RTinv_1, input_RTinv_2,
            output_RT,
            output_RTinv,
            depth_conf1,depth_conf2,


        )
        
        res_img = self.refine_resnet(gen_fs)
        refine_img = (gen_img + res_img)

        loss = self.loss_function(gen_img, output_img)
        loss_refine = self.loss_function(refine_img,output_img)
        loss["Total Loss"] += loss_refine["Total Loss"]

      

        
        photoloss = (photoloss_2 +photoloss_1)/2
        ssimloss = (ssimloss_1 +ssimloss_2)/2
        smoothloss = (smoothloss_1+smoothloss_2)/2

        loss['photo_loss'] = photoloss
        loss['smoothloss'] = smoothloss
        loss['ssim_loss'] = ssimloss
        
        loss["Depth Loss"] = 12*photoloss + 0.18*smoothloss + 6*ssimloss  
        loss["Total Loss"] +=loss["Depth Loss"]

        
    
  

        return (
            loss,
            {
                "InputImg_1": input_img_1,
                "InputImg_2": input_img_2,
                "OutputImg": output_img,
                "PredImg": gen_img,
                "Depth_tar": gen_depth,
  
                "RefineImg": refine_img,
            },
        )