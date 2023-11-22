import os
import random
import numpy as np
import torch
import csv
import cv2

import torch.nn as nn
import torch.utils.data as data
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation as ROT
from models.base_model import BaseModel
from models.networks.sync_batchnorm import convert_model
from options.options import get_dataset, get_model
from options.test_options import ArgumentParser
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
torch.backends.cudnn.benchmark = True
# torch.manual_seed(1)
os.environ["CUDA_VISIBLE_DEVICES"]="1"

class Dataset(data.Dataset):
    def __init__(self):
        super(Dataset, self).__init__()

        self.dataroot = '/home/youmeng/data/RealEstate10K/'

        self.frameroot = self.dataroot +'frames/'
        self.cameraroot = self.dataroot + 'cameras/'
        self.scene_all = os.listdir(self.frameroot)  
        self.scene = ['00bbf676b3378549', '00bb31ba2cf05be0', '00b647226f5d6904', '00b8297bf8e2a9ba', '00b180c077d3e3d4', '00b52b21e0d54a42', '00b40cafaa3a389a', '00ab0ac739885029', '00ac578dde876be6', '00ad47b927f0c851', '00adc59ebcbe00f7', '00b5cecbfd7f9a51', '00b6a786f2c21c17', '00b9a7963f9bd9c6', '00b9fa905d6c0830', '00b31f903ceb11a8', '00b37a5222bb6dca']
        self.imgroot = [self.frameroot + s for s in self.scene]
        self.poseroot = [self.cameraroot + s +'.txt' for s in self.scene]

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
        if index < self.dataset_size[0]:
            scene = 0
        else:
            for i in range(len(self.scene)):
                index = index-self.dataset_size[i]
                scene = i+1
                if index < self.dataset_size[i+1]:
                    break
        frame_target = self.idx[scene][index]
        id_target = index
        if index == 0:
            id_1 = index +1 
            id_2 = index+2
        elif index == self.dataset_size[scene]-1:
            id_1 = index -1 
            id_2 = index-2
        else:
            id_1 = index -1 
            id_2 = index+1
        frame_1 = self.idx[scene][id_1]
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
    def __len__(self):
        return sum(self.dataset_size)



if __name__ == "__main__":
    test_ops, _ = ArgumentParser().parse()

    # Load model to be tested
    MODEL_PATH = test_ops.old_model
    BATCH_SIZE = test_ops.batch_size

    opts = torch.load(MODEL_PATH)["opts"]

    model = get_model(opts)

    opts.render_ids = test_ops.render_ids
    opts.gpu_ids = test_ops.gpu_ids

    torch_devices = [int(gpu_id.strip()) for gpu_id in opts.gpu_ids.split(",")]
    print(torch_devices)
    device = "cuda:" + str(torch_devices[0])

    
    model = nn.DataParallel(model)
    model = model.to(device)

    #  Load the original model to be tested
    model_to_test = BaseModel(model, opts)
    model_to_test.eval()

    # Allow for different image sizes
    state_dict = model_to_test.state_dict()
    pretrained_dict = {
        k: v
        for k, v in torch.load(MODEL_PATH)["state_dict"].items()
    }

    state_dict.update(pretrained_dict)
    model_to_test.load_state_dict(state_dict)

    # Update parameters
    opts.render_ids = test_ops.render_ids
    opts.gpu_ids = test_ops.gpu_ids


    print("Loaded models...")

    # Load the dataset which is the set of images that came
    # from running the baselines' result scripts
    data = Dataset()

    model_to_test.eval()

    # Iterate through the dataset, predicting new views
    data_loader = DataLoader(data, batch_size=1, shuffle=False)
    iter_data_loader = iter(data_loader)
    with torch.no_grad():
        for i in range(0, len(data_loader)):
            _, pred_imgs, batch = model_to_test(
                iter_data_loader, isval=True, return_batch=True
            )
            if not os.path.exists(
                test_ops.result_folder
                + "/%d/" % (i)
            ):
                os.makedirs(
                    test_ops.result_folder
                    + "/%d/" % (i)
                )

            torchvision.utils.save_image(
                pred_imgs["PredImg"],
                test_ops.result_folder
                + "/%d/im_res.png" % (i),
            )
            torchvision.utils.save_image(
                pred_imgs["OutputImg"],
                test_ops.result_folder
                + "/%d/im_tar.png" % (i),
            )
            torchvision.utils.save_image(
                pred_imgs["RefineImg"],
                test_ops.result_folder
                + "/%d/im_refine.png" % (i),
            )
            # torchvision.utils.save_image(
            #     pred_imgs["HFRefineImg"],
            #     test_ops.result_folder
            #     + "/%d/im_hfrefine.png" % (i),
            # )
            torchvision.utils.save_image(
                pred_imgs["InputImg_1"],
                test_ops.result_folder
                + "/%d/im_in1.png" % (i),
            )
            torchvision.utils.save_image(
                pred_imgs["InputImg_2"],
                test_ops.result_folder
                + "/%d/im_in2.png" % (i),
            )
           


