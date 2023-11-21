import torch
import numpy as np
from scipy.spatial.transform import Rotation as ROT
import torch.utils.data as data
import os
import cv2
import csv
import re
import sys
import random
from PIL import Image
import torchvision.transforms as transforms
import torchvision
import matplotlib.image as img

class DTUDataLoader(data.Dataset):
    """ Dataset for loading the RealEstate10K. In this case, images are 
    chosen within a video.
    """

    def __init__(self, dataset, opts=None, num_views=2, seed=0, vectorize=False):
        super(DTUDataLoader, self).__init__()
        if dataset == "train":
            self.scene = [2, 6, 7, 8, 14, 16, 18, 19, 20, 22, 30, 31, 36, 39, 41, 42, 44,
                    45, 46, 47, 50, 51, 52, 53, 55, 57, 58, 60, 61, 63, 64, 65, 68, 69, 70, 71, 72,
                    74, 76, 83, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                    101, 102, 103, 104, 105, 107, 108, 109, 111, 112, 113, 115, 116, 119, 120,
                    121, 122, 123, 124, 125, 126, 127, 128]
        else:
            self.scene = [3, 5, 17, 21, 28, 35, 37, 38, 40, 43, 56, 59, 66, 67, 82, 86, 106, 117]

        self.initialize(opts)

    def initialize(self, opt):
        self.opt = opt
        self.dataroot = opt.train_data_path
        self.scene_size = int(len(self.scene))
        self.opt.bound = 1
        pair_path = self.dataroot + '/Cameras/pair.txt'
        f = open(pair_path)
        self.pair=[]
        for line in f:
            if len(line)>50:
                line_c = line.split()
                pair_c = [int(line_c[1]),int(line_c[3]),int(line_c[5])]
                self.pair.append(pair_c)
        f.close()
    def __getitem__(self, index):
        scene_idx = random.randint(0,self.scene_size-1)
        scene = self.scene[scene_idx]
        id_target = random.randint(1,49)
        id_1 = self.pair[id_target-1][0]+1
        id_2 = self.pair[id_target-1][1]+1
        id_3 = self.pair[id_target-1][2]+1
        B,hi,wi = self.load_image(id_1,scene)
        B = B / 255. 
        C,_,_ = self.load_image(id_2,scene)
        C = C / 255. 
        A,_,_ = self.load_image(id_target,scene)
        A = A / 255. 
        B_depth = self.load_depth_image(id_1,scene)
        C_depth = self.load_depth_image(id_2,scene)
        A_depth = self.load_depth_image(id_target,scene)
        RB,TB = self.load_pose(id_1)
        RC,TC = self.load_pose(id_2)
        RA,TA = self.load_pose(id_target)

        
        
        K1 = np.loadtxt(os.path.join(self.dataroot,'Cameras/train','{:08d}_cam.txt'.format(id_1-1)), skiprows=7, max_rows=3)
        K2 = np.loadtxt(os.path.join(self.dataroot,'Cameras/train','{:08d}_cam.txt'.format(id_target-1)), skiprows=7, max_rows=3)
        K3 = np.loadtxt(os.path.join(self.dataroot,'Cameras/train','{:08d}_cam.txt'.format(id_2-1)), skiprows=7, max_rows=3)
        Ki = (K1+K2+K3)/3

        K = np.block([ 
            [Ki,              np.zeros((3,1))],
            [np.zeros((1,3)), 1] 
            ] )
        K[0,0] = K[0,0]/wi*256.*4; K[1,1] = K[1,1]/hi*256.*4; K[0,2] = K[0,2]/wi*256*4; K[1,2] = K[1,2]/hi*256*4
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
        return {'images' : [A, B, C],  'cameras' : [{'Pinv' : RTinv_A, 'P' : RT_A},
                                                {'Pinv' : RTinv_B, 'P' : RT_B, 'K' : K,'Kinv' : Kinv,'warp':[RT_BA,RT_BC],'proj':RT_AB},
                                                   {'Pinv' : RTinv_C, 'P' : RT_C, 'warp':[RT_CA,RT_CB],'proj':RT_AC},],
                                                #    {'Pinv' : RTinv_D, 'P' : RT_D, 'warp':[RT_DA,RT_DB,RT_DC],'proj':RT_AD}   ],
                                                   "depths":[A_depth,B_depth,C_depth]
        }

    def load_pose(self,id):
        camera_extrinsics = np.loadtxt(os.path.join(self.dataroot,'Cameras/train/','{:08d}_cam.txt'.format(id-1)), skiprows=1, max_rows=3)
        R = camera_extrinsics[:,0:3]
        T = camera_extrinsics[0:3,3].reshape(3, 1)
        return R,T
    def load_image(self, id,scene):
        image_path = os.path.join(self.dataroot,'Rectified','scan{}_train'.format(scene),'rect_{:03d}_3_r5000.png'.format(id))
        img = cv2.imread(image_path)
        h,w,_ = img.shape
        img = cv2.resize(img, (256,256), interpolation=cv2.INTER_AREA)
        img = torch.from_numpy(img).permute(2,0,1)
        img = img[[2, 1, 0],:, :]
        return img,h,w
    def load_depth_image(self, id,scene):
        image_path = os.path.join(self.dataroot,'Depths','scan{}_train'.format(scene),'depth_map_{:04d}.pfm'.format(id-1))
        img,s = read_pfm(image_path)
        img = cv2.resize(img, (256,256), interpolation=cv2.INTER_AREA)
        img = torch.from_numpy(img).unsqueeze(0)
        return img
    def __len__(self): 
        return 5000

    def toval(self, epoch):
        pass

    def totrain(self, epoch):
        pass




def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale