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

class RealEstate10KDataLoader(data.Dataset):
    """ Dataset for loading the RealEstate10K. In this case, images are 
    chosen within a video.
    """

    def __init__(self, dataset, opts=None, num_views=2, seed=0, vectorize=False):
        super(RealEstate10KDataLoader, self).__init__()
        self.dataset = dataset
        self.initialize(opts)

    def initialize(self, opt):
        self.opt = opt
        self.dataroot = opt.train_data_path
        self.frameroot = self.dataroot +'frames/'
        self.cameraroot = self.dataroot + 'cameras/'
        self.scene_all = os.listdir(self.frameroot)  
        if self.dataset == "train":
            self.scene = ['00a5a09a0c68b59b', '00d80d0e284dac72', '00d3066d9cf11672', '00d269908ab6bb3c', '000e285b03f3fddf', '001c43a59a9e0b16', '001cf02b7625a4cc', '001d2c4c4b356c43', '001d7dab76c43f91', '002a58afaeaf24c9', '002bdb7ee61c492b', '003a243e0d37b315', '003c67ed6be72aeb', '003d2563b3c1023e', '004a40da569ee807', '004afb07c2352370', '004bd1e952dc1df8', '004c65679ce85f03', '004d2899ce6cfd7e', '004d9274e286b4df', '004daabedd090309', '004ed278c2b168f1', '004f2522f3d42f94', '004f7056eddd68b8', '005a62ed640d829f', '005b68f45317a0b5', '005c440a72e93b03', '005ee91e6c694114', '005ffc1aae84bdbf', '006b2728057d8236', '006ce7c81b5027e7', '006d459e7f65953b', '006f24f8fa62a3b7', '006f41ee9800fecf', '007ac6cef80a692c', '007bd32591b43f68', '007d91ee3338229d', '007facf53bb75921', '008b073e9b710630', '008caa6df1da0cd1', '008d5bc870d12c22', '008e27f0941b96ed', '008ec1473e4ce029', '008f94cc8929340d', '008f83302bfa22b5', '009a939c6fef1912', '009b0cac4a16bbaf', '009c0252ca19d76b', '009d5ce3e73298df', '009ded7e45b6d15f', '009e7b2d81e10364', '009e573e59c8c393', '009e714d9a952807', '009ef0283740e7b3', '009f6e6c5aea0441', '009f73815f3e3b42', '009f5248262f70e0', '0013a39f3a02b5c3', '0016d89da2d78196', '0020e1f0ec424295', '0021e3be0043c3b3', '0024e835f897f4f2', '0027e46f749ac7ec', '0029e9169a167c4f', '0032c663bb6d838e', '0040ff89cd583609', '0048fce686bb184d', '0049c83bad21bbdf', '0051ae563152ac76', '0051d3a3ada580af', '0051dfdcdc3050dc', '0056df7a0d0739e1', '0059ff3bc0a9b4c6', '0065a058603dfca4', '0066de2687c63b09', '0066ed3711a7240d', '0068e9bba9742bde', '0069fbbde9f77bf8', '0071f6d0b342c75f', '0076aea772f3acc7', '0076e0a94466efb1', '0081cfd790d7ad74', '0081dc6b188fefe7', '0083e7de502c19b2', '0087cf27aa9e6167', '0092c01a563523b7', '0095b1bb592abf6e', '0095d64160cfe952', '0099afa1d37142f4', '00168ec12dda389c', '00182dd4c74ab052', '00210fcc783d3fcf', '00236fcca80c95a0', '00259da07b0dba33', '00336b9590b313a2', '00336da5ecce4cb3', '00404b326b9afd2a', '00407b3f1bad1493', '00415ec2f505e8c3', '00419dfa0b973e11', '00438c154e81cbd1', '00454b8bf9209de8', '00513a2e0e404c0e', '00552c0c18b00e98', '00559a2a16636897', '00560d9f99a6f200', '00579a56a5d9cf86', '00664b826ec082e8', '00703cbf7531ef11', '00761c6dcec91853', '00793a8a3b268d7c', '00830d5bb7464fab', '001511b4a282e504', '002554f1f92ac083', '003908a8fa818d50', '003977c5456b561e', '005935bbd1c0cf38', '006477bb9bada538', '007203bffc9af03d', '007525d507d7a6c1', '007802ba570b907f', '008565fd84fac8bc', '008911a205850050', '009048f2c8444b4f', '009664cb1b8d351a', '009700fcc49fe368', '0010533d5f176d7b', '0014287f39e4686f', '0045483c1473d6a7', '0074915cdbef7836', '00104677f98ca3c9', '00399697cc95c81e', '00469966d50c2b97', '00611063bacaa167', '00792147e1d4d89f', '00821987dfce6103', '00980329a3221f1c', '00983896bd7169fd', '004673337d29eb87', '0059080010e0a279', '0082221897dbdc56', '001573921974bc70', '005758861029cf83', '006363925858461c', '00a5cfccd6508e96', '00a7a9c0ea61670d', '00a50bfbce75d465', '00a87dd567cdc292', '00a97cfa89150952', '00a861d699fb7797', '00a907a68edf05c7', '00a61821997f3fac', '00aa25d111d218cc', '00aad2a54c447c8f', '00cfaaa4ba7a16b2', '00cdf8750f5182cc', '00cc980718fe37cf', '00caa2b14cfc2603', '00c8250efd605554', '00c7627b596c041a', '00c5167e29241893', '00c3799b538b84e7', '00c26c8b16e5a435', '00c14c53270e07a6', '00c14be5659b7fe5', '00c10c2280022998', '00c7f4d292a58bd3', '00c5fe571a003deb', '00c05c92f16e3a7f', '00c4e6f642692c12']
        else:
            self.scene = ['00bbf676b3378549', '00bb31ba2cf05be0', '00b647226f5d6904', '00b8297bf8e2a9ba', '00b180c077d3e3d4', '00b52b21e0d54a42', '00b40cafaa3a389a', '00ab0ac739885029', '00ac578dde876be6', '00ad47b927f0c851', '00adc59ebcbe00f7', '00b5cecbfd7f9a51', '00b6a786f2c21c17', '00b9a7963f9bd9c6', '00b9fa905d6c0830', '00b31f903ceb11a8', '00b37a5222bb6dca']
        self.imgroot = [self.frameroot + s for s in self.scene]
        self.poseroot = [self.cameraroot + s +'.txt' for s in self.scene]
        self.opt.bound = 1

        self.img_dict = [os.listdir(im) for im in self.imgroot ]
        self.idx = [];self.Ks = [];self.pose = []
        for pr in self.poseroot:
            idxi = [];intrinsicsi = [];posei = []
            with open(pr, "r") as f:
                next(f)
                for line in f.readlines():
                    numbers = [float(i) for i in line.split()]
                    idxi.append(int(numbers[0])) 
                    intrinsicsi.append(numbers[1:5])
                    posei.append(numbers[7:])
            self.idx.append(idxi)
            self.Ks.append(intrinsicsi)
            self.pose.append(posei)
        self.dataset_size = [int(len(imd)) for imd in self.img_dict]
    def __getitem__(self, index):
        
        scene = random.randint(0,len(self.dataset_size)-1)
        id_1 = random.randint(0,self.dataset_size[scene]-1)
        frame_1 = self.idx[scene][id_1]
        delta = 1
        delta_2 = 1

        id_target = id_1 + delta  
        id_2 = id_1 + delta + delta_2
        if (id_2 < self.dataset_size[scene]) & (id_target < self.dataset_size[scene]):
            frame_target = self.idx[scene][id_target]
            frame_2 = self.idx[scene][id_2]
        else:
            id_target = id_1 - delta
            frame_target = self.idx[scene][id_target]
            id_2 = id_1 - delta - delta_2
            frame_2 = self.idx[scene][id_2]

        B = self.load_image(frame_1,scene)
        B = B / 255. 
        C = self.load_image(frame_2,scene)
        C = C / 255. 

        A = self.load_image(frame_target,scene)
        A = A / 255. 
    
        pose_B = np.array(self.pose[scene][id_1]).reshape(3,4)
        pose_C = np.array(self.pose[scene][id_2]).reshape(3,4)
        pose_A = np.array(self.pose[scene][id_target]).reshape(3,4)

        
        Ki = self.Ks[scene][id_1]
        K = np.block([ 
            [Ki[0], 0.,Ki[2],0.],
            [0.,Ki[1],Ki[3],0.],
            [0.,0.,1.,0.] ,
            [0.,0.,0.,1.]
            ] )
        K[0,0] = K[0,0]*256.; K[1,1] = K[1,1]*256.; K[0,2] = K[0,2]*256; K[1,2] = K[1,2]*256
        K = K.astype(np.float32)
        Kinv = np.linalg.inv(K).astype(np.float32)

        ones = np.zeros((1,4)); ones[0,3] = 1
        RT_A = np.concatenate((pose_A,ones),axis = 0).astype(np.float32)
        RT_B = np.concatenate((pose_B,ones),axis = 0).astype(np.float32)
        RT_C = np.concatenate((pose_C,ones),axis = 0).astype(np.float32)
        RTinv_A = np.linalg.inv(RT_A).astype(np.float32)
        RTinv_B = np.linalg.inv(RT_B).astype(np.float32)
        RTinv_C = np.linalg.inv(RT_C).astype(np.float32)

        RT_BA = np.matmul(RT_A,np.linalg.inv(RT_B)).astype(np.float32)
        RT_BC = np.matmul(RT_C,np.linalg.inv(RT_B)).astype(np.float32)
        RT_CA = np.matmul(RT_A,np.linalg.inv(RT_C)).astype(np.float32)
        RT_CB = np.matmul(RT_B,np.linalg.inv(RT_C)).astype(np.float32)


        
        RT_AB = np.matmul(RT_B,np.linalg.inv(RT_A)).astype(np.float32)
        RT_AC = np.matmul(RT_C,np.linalg.inv(RT_A)).astype(np.float32)
        return {'images' : [A, B, C], 'cameras' : [{'Pinv' : RTinv_A, 'P' : RT_A},
                                                {'Pinv' : RTinv_B, 'P' : RT_B, 'K' : K,'Kinv' : Kinv,'warp':[RT_BA,RT_BC],'proj':RT_AB},
                                                   {'Pinv' : RTinv_C, 'P' : RT_C, 'warp':[RT_CA,RT_CB],'proj':RT_AC},],
                                              
        }
        

    def load_image(self, id,scene):
        
        image_path = self.imgroot[scene]+'/'+ str(id) +'.png'
        img = cv2.imread(image_path)
        h,w,_ = img.shape
        img = cv2.resize(img, (256,256), interpolation=cv2.INTER_AREA)
        img = torch.from_numpy(img).permute(2,0,1)
        img = img[[2, 1, 0],:, :]
        return img
    def load_depth_image(self, id,scene):
        image_path = os.path.join(self.dataroot[scene], id )
        img = np.load(image_path)
        img = cv2.resize(img, (64,64), interpolation=cv2.INTER_AREA)
        img = torch.from_numpy(img).unsqueeze(0)
        return img
    def __len__(self): 
        return sum(self.dataset_size) * 50

    def toval(self, epoch):
        pass

    def totrain(self, epoch):
        pass