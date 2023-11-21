
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
class ZbufferModelPts(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt
        self.pencoder = ResUNet(depth= 3,out_channels = 32)

        self.pts_transformer = PtsManipulator(256, C=67, opt=opt)
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
      


        with torch.no_grad():
            pfs_1 = self.pencoder(input_img_1)
            pfs_2 = self.pencoder(input_img_2) 

            cfs_1 = torch.cat((input_img_1,pfs_1),1)
            cfs_2 = torch.cat((input_img_2,pfs_2),1)
            gen_img,gen_fs,gen_depth= self.pts_transformer.forward_justpts(
                cfs_1,cfs_2,
                depth_img_1,depth_img_2,
                K,
                K_inv,
                input_RT_1, input_RT_2,
                input_RTinv_1, input_RTinv_2,
                output_RT,
                output_RTinv,
            )
        
            res_img = self.refine_resnet(gen_fs)
            refine_img = (gen_img + res_img)

#high frequency refine for image restoration module
#need to load trained model
    # ####################################
        warped_1,_ = inverse_warp(input_img_1, gen_depth,pose_t1,K)
        warped_2,_ = inverse_warp(input_img_2, gen_depth,pose_t2,K)
        hfres_img = self.resnet(torch.cat((refine_img,warped_1,warped_2),1))
        hf_img = hfres_img + refine_img
        loss = self.loss_function(hf_img,output_img)
# ####################################
        
  

        return (
            loss,
            {
                "InputImg_1": input_img_1,
                "InputImg_2": input_img_2,
                "OutputImg": output_img,
                "PredImg": gen_img,
                "Depth_tar": gen_depth,
  
                "RefineImg": refine_img,
                "HFRefineImg": hf_img,
            },
        )