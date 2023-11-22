import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import scipy.io as io
import cv2
# from numpy import *
import random
from models.losses.synthesis import SynthesisLoss
from models.networks.architectures import ResUNet
import time
# from models.networks.utilities import get_decoder, get_encoder
from models.projection.z_buffer_manipulator_posed import PtsManipulator
# from models.depth_model.mvsnet import MVSNet
# from models.losses.photometric_loss import photometric_reconstruction_loss
from models.losses.inverse_warp import inverse_warp
# from models.depth_sfm_learner import DispNetS
from models.losses.unsup_depth_loss import photometric_reconstruction_loss

from models.losses.depth_consistency_loss import depth_consistency_loss,depth_consistency_loss1
from models.networks.semantic_mvsnet import SEMVSNet
from models.losses.inverse_warp import inverse_warp
from models.networks.architectures import ResNet
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from graphs.losses.dist_chamfer import ChamferDist
class ZbufferModelPts(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt
        self.photometric_reconstruction_loss = photometric_reconstruction_loss()
        # ENCODER
        # Encode features to a given resolution
        self.encoder = ResUNet(depth= 3,out_channels = 32)
        self.pencoder = ResUNet(depth= 3,out_channels = 32)
        self.MVSNet = SEMVSNet()
        # POINT CLOUD TRANSFORMER
        # REGRESS 3D POINTS
        # self.pts_regressor = Unet(channels_in=4, channels_out=1, opt=opt)

        # self.pts_regressor = Unet(channels_in=11, channels_out=1, opt=opt)
        # self.raft = RAFT(self.opt)
        # self.f2d = Flow2Depth(H = 256, W = 256)
        # self.refine_net = PatchMatchingRefine()
        # 3D Points transformer
        self.pts_transformer = PtsManipulator(256, C=67, opt=opt)
        self.refine_resnet = ResNet(in_channels=32,out_channels=4,nf=64)
        self.resnet = ResNet(in_channels=9,out_channels=3,nf=64)
        # LOSS FUNCTION
        # Module to abstract away the loss function complexity
        self.loss_function = SynthesisLoss(opt=opt)
    
    def forward(self, batch,isval):
        """ Forward pass of a view synthesis model with a voxel latent field.
        """
        # Input values
        input_img_1 = batch["images"][1]
        input_img_2 = batch["images"][2]
        # input_img_3 = batch["images"][3]
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
        # input_RT_3 = batch["cameras"][3]["P"]
        # input_RTinv_3 = batch["cameras"][3]["Pinv"]
        output_RT = batch["cameras"][0]["P"]
        output_RTinv = batch["cameras"][0]["Pinv"]

        # img_proj_1 = batch["cameras"][1]["Proj"]
        # img_proj_2 = batch["cameras"][2]["Proj"]

        pose_1 = batch["cameras"][1]["warp"]
        pose_2 = batch["cameras"][2]["warp"]
        # pose_3 = batch["cameras"][3]["warp"]
        pose_t1 = batch["cameras"][1]["proj"]
        pose_t2 = batch["cameras"][2]["proj"]
        # pose_t3 = batch["cameras"][3]["proj"]
        self.opt.num_depth = 128
        depth_values = torch.arange(start=self.opt.min_z,end=self.opt.max_z,step = (self.opt.max_z-self.opt.min_z)/self.opt.num_depth)
        # print(depth_values)
        # depth_values = torch.arange(start=425,end=800,step = (800-425)/self.opt.num_depth)
        # depth_values = torch.cat((torch.arange(start=self.opt.min_z,end=self.opt.max_z/2,step = (self.opt.max_z/2-self.opt.min_z)/(self.opt.num_depth*3/4)),
        #                 torch.arange(start=self.opt.max_z/2,end=self.opt.max_z,step = (self.opt.max_z/2)/(self.opt.num_depth/4))),0)
        # print(depth_values)

        if torch.cuda.is_available():
            input_img_1 = input_img_1.cuda()
            input_img_2 = input_img_2.cuda()
            # input_img_3 = input_img_3.cuda()
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
            # input_RT_3 = input_RT_3.cuda()
            # input_RTinv_3 = input_RTinv_3.cuda()
            output_RT = output_RT.cuda()
            output_RTinv = output_RTinv.cuda()
            # img_proj_1 = img_proj_1.cuda()
            # img_proj_2 = img_proj_2.cuda()
            pose_1 = [p1.cuda() for p1 in pose_1 ]
            pose_2 = [p2.cuda() for p2 in pose_2 ]
            # pose_3 = [p3.cuda() for p3 in pose_3 ]

            pose_t1 = pose_t1.cuda()
            pose_t2 = pose_t2.cuda()
            # pose_t3 = pose_t3.cuda()
            depth_values = depth_values.cuda()
      
        # with torch.no_grad():
        fs_1 = self.encoder(input_img_1)
        fs_2 = self.encoder(input_img_2) 



        bs,c,w,_ = input_img_1.shape
        depth_values = depth_values.unsqueeze(0).repeat(bs,1)
        # # # with torch.no_grad
        # # depth_1 = self.SEMVSNet([fs_1,fs_2],[input_RT_1,input_RT_2],depth_values,K)
        # # depth_2 = self.SEMVSNet([fs_2,fs_1],[input_RT_2,input_RT_1],depth_values,K)
        depth_1,depth_conf1 = self.MVSNet([fs_1,fs_2],[input_RT_1,input_RT_2],depth_values,K)
        depth_2,depth_conf2 = self.MVSNet([fs_2,fs_1],[input_RT_2,input_RT_1],depth_values,K)

        photoloss_1,ssimloss_1,smoothloss_1,warped1,mask1= self.photometric_reconstruction_loss(input_img_1, [output_img,input_img_2], depth_1,pose_1,K)
        photoloss_2,ssimloss_2,smoothloss_2,warped2,mask2 = self.photometric_reconstruction_loss(input_img_2, [output_img,input_img_1], depth_2,pose_2,K)

        
        # with torch.no_grad():
        pfs_1 = self.pencoder(input_img_1)
        pfs_2 = self.pencoder(input_img_2) 
        # # print(pfs_1[:,:3]+0.5)
        # # print(pfs_2)
        # # torchvision.utils.save_image((pfs_1[:,12:15]+0.2)*2,'pfs_1.jpg')
        # # torchvision.utils.save_image((pfs_2[:,12:15]+0.5)*2,'pfs_2.jpg')
        # # print(tt)


        cfs_1 = torch.cat((input_img_1,pfs_1),1)
        cfs_2 = torch.cat((input_img_2,pfs_2),1)
        # gen_img,gen_fs= self.pts_transformer.forward_justpts(
        proj_img_1,proj_img_2,gen_fs_1,gen_fs_2,gen_depth_1,gen_depth_2= self.pts_transformer.forward_justpts(
            cfs_1,cfs_2,
            # input_img_1,input_img_2,
            depth_1, depth_2,
            # depth_conf1,depth_conf2,
            # depth_img_1,depth_img_2,
            K,
            K_inv,
            input_RT_1, input_RT_2,
            input_RTinv_1, input_RTinv_2,
            output_RT,
            output_RTinv,
        )
        # gen_img,gen_fs,gen_depth= self.pts_transformer.forward_justpts(
        #     cfs_1,cfs_2,
        #     depth_1, depth_2,
        #     depth_conf1,depth_conf2,
        #     # depth_img_1,depth_img_2,
        #     K,
        #     K_inv,
        #     input_RT_1, input_RT_2,
        #     input_RTinv_1, input_RTinv_2,
        #     output_RT,
        #     output_RTinv,
        # )
        # res_img = self.refine_resnet(gen_fs)
        # refine_img = (gen_img + res_img)
        # loss = self.loss_function(gen_img, output_img)
        # loss_refine = self.loss_function(refine_img,output_img)

        gen_1 = self.refine_resnet(gen_fs_1)
        gen_img_1=proj_img_1 + gen_1[:,:3]
        conf_1=gen_1[:,3].unsqueeze(1)

        gen_2 = self.refine_resnet(gen_fs_2)
        gen_img_2=proj_img_2 + gen_2[:,:3]
        conf_2=gen_2[:,3].unsqueeze(1)

        gen_img_t = torch.cat((gen_img_1.unsqueeze(1),gen_img_2.unsqueeze(1)),1)
        conf_t = torch.cat((conf_1.unsqueeze(1),conf_2.unsqueeze(1)),1)
        # print(conf_t.shape)
        conf = F.softmax(conf_t, dim=1)
        conf = conf.repeat(1,1,3,1,1)
        # print(gen_img_t.shape)
            

        gen_img = torch.sum(gen_img_t * conf, dim=1)

        photoloss = (photoloss_2 +photoloss_1)/2
        ssimloss = (ssimloss_1 +ssimloss_2)/2
        smoothloss = (smoothloss_1+smoothloss_2)/2
        
        
        loss = self.loss_function(gen_img, output_img)
        # # print(loss['psnr'])
        loss["Depth Loss"] = 12*photoloss + 0.18*smoothloss + 6*ssimloss  
        loss_1 = self.loss_function(gen_img_1, output_img)
        loss_2 = self.loss_function(gen_img_2, output_img)
        loss_3 = self.loss_function(proj_img_1, output_img)
        loss_4 = self.loss_function(proj_img_2, output_img)
        loss["Total Loss"] += loss_1["Total Loss"] +loss_2["Total Loss"]+loss["Depth Loss"]+loss_3["Total Loss"] +loss_4["Total Loss"]
        loss["psnr_1"] = loss_1["psnr"] 
        loss["psnr_2"] = loss_2["psnr"]
        loss["psnr_3"] = loss_3["psnr"] 
        loss["psnr_4"] = loss_4["psnr"]

    # # ####################################
        # warped_1,_ = inverse_warp(input_img_1, gen_depth_1,pose_t1,K)
        # warped_2,_ = inverse_warp(input_img_2, gen_depth_2,pose_t2,K)


        # hfres_img = self.resnet(torch.cat((gen_img,warped_1,warped_2),1))
        # hf_img =hfres_img + gen_img
        # loss = self.loss_function(hf_img,output_img)
# ####################################
      

        
        
        # loss["Total Loss"] += loss_refine["Total Loss"]
        # # # loss["Depth Loss"] +
        # # # # # loss["Total Loss"] += loss_refine["Total Loss"]

        # # # loss['photo_loss'] = photoloss
        # # # loss['smoothloss'] = smoothloss
        # # # loss['ssim_loss'] = ssimloss
        # loss['refine_loss_l1'] = loss_refine["L1"]
        # loss['refine_loss_Perceptual'] = loss_refine["Perceptual"]
        # loss['refine_psnr'] = loss_refine["psnr"]      
  

        return (
            loss,
            {
                "InputImg_1": input_img_1,
                "InputImg_2": input_img_2,
                # # # "InputImg_3": input_img_3,
                "OutputImg": output_img,
                "PredImg_1": gen_img_1,
                "PredImg_2": gen_img_2,
                "PredImg": gen_img,
  
                # "RefineImg": refine_img,
                # "HFRefineImg": hf_img,
                # "Depth_1": depth_img_1,
                # "Depth_2": depth_img_2,
                # # "Depth_3": depth_img_3,
                # "Depth_tar":depth_img_tar,
                
                # "PredDepth_1": depth_1,
                # "PredDepth_2": depth_2,
                # "PredDepth_3": depth_3,
                # "Depth_1": depth_img_1,
                # "Depth_2": depth_img_2,
                # "PredDepth":gen_depth,

            },
        )
