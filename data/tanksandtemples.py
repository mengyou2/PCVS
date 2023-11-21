# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# Based on https://github.com/xuchen-ethz/continuous_view_synthesis/blob/master/data/kitti_data_loader.py

import torch
import numpy as np
from scipy.spatial.transform import Rotation as ROT
import torch.utils.data as data
import os
import cv2
import csv
import random
from PIL import Image
import torchvision.transforms as transforms
import torchvision
import matplotlib.image as img

class TanksandTemplesDataLoader(data.Dataset):
    """ Dataset for loading the RealEstate10K. In this case, images are 
    chosen within a video.
    """

    def __init__(self, dataset, opts=None, num_views=2, seed=0, vectorize=False):
        super(TanksandTemplesDataLoader, self).__init__()
        self.mode = dataset
        if dataset == "train":
            self.sceneroot = opts.train_data_path
        else:
            self.sceneroot = opts.eval_data_path
        self.initialize(opts)

    def initialize(self, opt):
        self.opt = opt
        self.scenelist = os.listdir(self.sceneroot) 

        self.dataroot = [self.sceneroot + s +'/dense/ibr3d_pw_0.25' for s in self.scenelist]

        self.fiel_dict = [os.listdir(dr) for dr in self.dataroot]
        self.img_dict = [sorted([s for s in fd if 'im' in s]) for fd in self.fiel_dict]
        self.Ks= [np.load(os.path.join(dr, 'Ks.npy')) for dr in self.dataroot]
        self.Rs= [np.load(os.path.join(dr, 'Rs.npy')) for dr in self.dataroot]
        self.Ts= [np.load(os.path.join(dr, 'ts.npy')) for dr in self.dataroot]
        self.dataset_size = [int(len(imd)) for imd in self.img_dict]
    def __getitem__(self, index):
        scene = random.randint(0,len(self.scenelist)-1)
        if self.mode == 'train':
            index = random.randint(0,self.dataset_size[scene]-1)
        else:
            if self.scenelist[scene] =='Playground':
                index = random.randint(221,253)
            elif self.scenelist[scene] =='Train':
                index = random.randint(174,249)
            elif self.scenelist[scene] =='M60':
                index = random.randint(94,130)
            elif self.scenelist[scene] =='Truck':
                index = random.randint(172,197)

        id_target = self.img_dict[scene][index]
        id_target_ = int(id_target.split('_')[-1].split('.')[0])
        if id_target_ == 0:
            id_1_ = id_target_ +1 
            id_2_ = id_target_+2
        elif id_target_ == self.dataset_size[scene]-1:
            id_1_ = id_target_ -1 
            id_2_ = id_target_-2
        else:
            id_1_ = id_target_ -1 
            id_2_ = id_target_+1
        id_1 = 'im_'+str(id_1_).zfill(8)+'.jpg'
        id_2 = 'im_'+str(id_2_).zfill(8)+'.jpg'
        id_1_depth = 'dm_'+str(id_1_).zfill(8)+'.npy'
        id_2_depth = 'dm_'+str(id_2_).zfill(8)+'.npy'
        id_target_depth = 'dm_'+str(id_target_).zfill(8)+'.npy'
       
        B,hi,wi = self.load_image(id_1,scene)
        B = B / 255. 
        C,_,_ = self.load_image(id_2,scene)
        C = C / 255. 
        A,_,_ = self.load_image(id_target,scene)
        A = A / 255. 

        B_depth = self.load_depth_image(id_1_depth,scene)
        C_depth = self.load_depth_image(id_2_depth,scene)
        A_depth = self.load_depth_image(id_target_depth,scene)
        RB = self.Rs[scene][id_1_]; RC = self.Rs[scene][id_2_]
        RA = self.Rs[scene][id_target_]
        TB = self.Ts[scene][id_1_].reshape(3, 1); TC = self.Ts[scene][id_2_].reshape(3, 1); 
        TA = self.Ts[scene][id_target_].reshape(3, 1)
        


        Ki = self.Ks[scene][id_1_]
        K = np.block([ 
            [Ki,              np.zeros((3,1))],
            [np.zeros((1,3)), 1] 
            ] )
        K[0,0] = K[0,0]/wi*256.; K[1,1] = K[1,1]/hi*256.; K[0,2] = K[0,2]/wi*256.; K[1,2] = K[1,2]/hi*256.
        K = K.astype(np.float32)
        
        Kinv = np.linalg.inv(K).astype(np.float32)
        mat_A = np.block(
            [ [RA, TA],
            [np.zeros((1,3)), 1] ] )
        mat_B = np.block(
            [ [RB, TB],
            [np.zeros((1,3)), 1] ] )
        mat_C = np.block(
            [ [RC, TC],
            [np.zeros((1,3)), 1] ] )

        RT_A = mat_A.astype(np.float32)
        RT_B = mat_B.astype(np.float32)
        RT_C = mat_C.astype(np.float32)
        RTinv_A = np.linalg.inv(mat_A).astype(np.float32)
        RTinv_B = np.linalg.inv(mat_B).astype(np.float32)
        RTinv_C = np.linalg.inv(mat_C).astype(np.float32)
        
        RT_BA = np.matmul(RT_A,np.linalg.inv(RT_B)).astype(np.float32)
        RT_BC = np.matmul(RT_C,np.linalg.inv(RT_B)).astype(np.float32)
        RT_CA = np.matmul(RT_A,np.linalg.inv(RT_C)).astype(np.float32)
        RT_CB = np.matmul(RT_B,np.linalg.inv(RT_C)).astype(np.float32)


        
        RT_AB = np.matmul(RT_B,np.linalg.inv(RT_A)).astype(np.float32)
        RT_AC = np.matmul(RT_C,np.linalg.inv(RT_A)).astype(np.float32)

        return {'images' : [A, B, C], 'cameras' : [{'Pinv' : RTinv_A, 'P' : RT_A},
                                                {'Pinv' : RTinv_B, 'P' : RT_B, 'K' : K,'Kinv' : Kinv,'warp':[RT_BA,RT_BC],'proj':RT_AB},
                                                   {'Pinv' : RTinv_C, 'P' : RT_C, 'warp':[RT_CA,RT_CB],'proj':RT_AC},],
                                                   "depths":[A_depth,B_depth,C_depth]
        }
    
    def load_image(self, id,scene):
        image_path = os.path.join(self.dataroot[scene], id )
        img = cv2.imread(image_path)
        h,w,_ = img.shape
        img = cv2.resize(img, (256,256), interpolation=cv2.INTER_AREA)
        img = torch.from_numpy(img).permute(2,0,1)
        img = img[[2, 1, 0],:, :]
        return img,h,w
    def load_depth_image(self, id,scene):
        image_path = os.path.join(self.dataroot[scene], id )
        img = np.load(image_path)
        
        img = cv2.resize(img, (256,256), interpolation=cv2.INTER_AREA)
        img = torch.from_numpy(img).unsqueeze(0)
        mask = (img.abs() == 0)
        img[mask] = 1e2
        return img
    def __len__(self): 
        return sum(self.dataset_size) * 50

    def toval(self, epoch):
        pass

    def totrain(self, epoch):
        pass