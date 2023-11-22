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
os.environ["CUDA_VISIBLE_DEVICES"]="2"
SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
class Dataset(data.Dataset):
    """ Dataset for loading the RealEstate10K. In this case, images are 
    chosen within a video.
    """

    def __init__(self):
        super(Dataset, self).__init__()

        self.dataroot = os.environ["TANKS"]

        self.bound = 1
        self.fiel_dict = os.listdir(self.dataroot) 
        self.img_dict = sorted([s for s in self.fiel_dict if 'im' in s])
        if os.environ["scene"] == 'Playground':
            self.img_dict = self.img_dict[220:252]#playground
        elif os.environ["scene"] == 'Train':
            self.img_dict = self.img_dict[173:194]+self.img_dict[226:248]#train173
        elif os.environ["scene"] == 'M60':
            self.img_dict = self.img_dict[93:129]#m60
        elif os.environ["scene"] == 'Truck':
            self.img_dict = self.img_dict[171:196]#truck
        self.Ks= np.load(os.path.join(self.dataroot, 'Ks.npy'))
        self.Rs= np.load(os.path.join(self.dataroot, 'Rs.npy'))
        self.Ts= np.load(os.path.join(self.dataroot, 'ts.npy'))
        self.dataset_size = int(len(self.img_dict))
        print(self.dataset_size)
    def __getitem__(self, index):

        id_1 = self.img_dict[index]
        id_1_ = int(id_1.split('_')[-1].split('.')[0])
        id_1_depth = 'dm_'+id_1.split('_')[-1].split('.')[0]+'.npy'
        id_target_ = id_1_ + 1
        id_target = id_1.split('_')[0] +'_' + str(id_target_).zfill(len(id_1.split('_')[-1].split('.')[0]))+'.jpg'
        id_2_ = id_1_ + 2
        id_2 = id_target.split('_')[0] +'_' + str(id_2_).zfill(len(id_1.split('_')[-1].split('.')[0]))+'.jpg'
        id_2_path = os.path.join(self.dataroot, id_2)
        
        B,hi,wi = self.load_image(id_1)
        B = B / 255. 
        C,_,_ = self.load_image(id_2)
        C = C / 255. 
       
        A,_,_ = self.load_image(id_target)
        A = A / 255. 
       
        RB = self.Rs[id_1_]; RC = self.Rs[id_2_]; 
        RA = self.Rs[id_target_]
        TB = self.Ts[id_1_].reshape(3, 1); TC = self.Ts[id_2_].reshape(3, 1); 
        TA = self.Ts[id_target_].reshape(3, 1)
        
       

        Ki = self.Ks[id_1_]
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
                                                #    {'Pinv' : RTinv_D, 'P' : RT_D, 'warp':[RT_DA,RT_DB,RT_DC],'proj':RT_AD}   ],
                                                #    "depths":[A_depth,B_depth,C_depth]
        }


    def load_image(self, id):
        image_path = os.path.join(self.dataroot, id )
        img = cv2.imread(image_path)
       
        h,w,_ = img.shape
        img = cv2.resize(img, (256,256), interpolation=cv2.INTER_AREA)
       
        img = torch.from_numpy(img).permute(2,0,1)
        img = img[[2, 1, 0],:, :]

        
        return img,h,w
    def load_depth_image(self, id):
        image_path = os.path.join(self.dataroot, id )
        img = np.load(image_path)
        
        img = cv2.resize(img, (256,256), interpolation=cv2.INTER_AREA)
        img = torch.from_numpy(img).unsqueeze(0)
       
        return img
    def __len__(self):
        return self.dataset_size 



if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"]="7"
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

    print(opts)
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
            torchvision.utils.save_image(
                pred_imgs["HFRefineImg"],
                test_ops.result_folder
                + "/%d/im_hfrefine.png" % (i),
            )
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
            