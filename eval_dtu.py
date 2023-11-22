import os
import random
import numpy as np
import torch
import csv
import cv2
import re
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

        self.scene = [3, 5, 17, 21, 28, 35, 37, 38, 40, 43, 56, 59, 66, 67, 82, 86, 106, 117]

        self.dataroot = '/home/youmeng/data/DTU/'
        self.imgroot = [os.path.join(self.dataroot,'Rectified','scan{}_train'.format(s)) for s in self.scene]
        self.img_dict = [os.listdir(im) for im in self.imgroot]
        self.imgs =[]
        for s in self.img_dict:
            imgs = sorted([ss for ss in s if '_3_r5000.png' in ss ])
            self.imgs.append(imgs)
        self.dataset_size = [int(len(imd)) for imd in self.imgs]
    def __getitem__(self, index):
        if index < self.dataset_size[0]:
            scene = 0
        else:
            for i in range(len(self.scene)):
                index = index-self.dataset_size[i]
                scene = i+1
                if index < self.dataset_size[i+1]:
                    break
        print(scene,index)

        scene = 0
        index = 32
        index = 17

        id_target = index +1
        if id_target == 1:
            id_1 = id_target +1 
            id_2 = id_target+2
        elif id_target == self.dataset_size[scene]:
            id_1 = id_target -1 
            id_2 = id_target-2
        else:
            id_1 = id_target -1 
            id_2 = id_target+1




        scene = self.scene[scene]
        B,hi,wi = self.load_image(id_1,scene)
        B = B / 255. 
        C,_,_ = self.load_image(id_2,scene)
        C = C / 255. 
        A,_,_ = self.load_image(id_target,scene)
        A = A / 255. 

        B_seg =B
        A_seg =A
        C_seg = C
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
     
        return {'images' : [A, B, C], 'segmentation':[A_seg,B_seg,C_seg], 'cameras' : [{'Pinv' : RTinv_A, 'P' : RT_A},
                                                {'Pinv' : RTinv_B, 'P' : RT_B, 'K' : K,'Kinv' : Kinv,'warp':[RT_BA,RT_BC],'proj':RT_AB},
                                                   {'Pinv' : RTinv_C, 'P' : RT_C, 'warp':[RT_CA,RT_CB],'proj':RT_AC},],
                                                   "depths":[A_depth,B_depth,C_depth]
        }

    def load_depth_image(self, id,scene):
        image_path = os.path.join(self.dataroot,'Depths','scan{}_train'.format(scene),'depth_map_{:04d}.pfm'.format(id-1))
        img,s = read_pfm(image_path)
        
        img = cv2.resize(img, (256,256), interpolation=cv2.INTER_AREA)
        img = torch.from_numpy(img).unsqueeze(0)
        return img
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
    
    def __len__(self):
        return sum(self.dataset_size)



if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"]="7"
    test_ops, _ = ArgumentParser().parse()

    # Load model to be tested
    MODEL_PATH = test_ops.old_model
    BATCH_SIZE = test_ops.batch_size

    opts = torch.load(MODEL_PATH)["opts"]

    model = get_model(opts)
    opts.min_z = 425
    opts.max_z = 900
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
            mask = (pred_imgs["Depth_tar"]!=0)
            torchvision.utils.save_image(
                pred_imgs["RefineImg"],
                test_ops.result_folder
                + "/%d/im_refine.png" % (i),
            )
            torchvision.utils.save_image(
                pred_imgs["OutputImg"],
                test_ops.result_folder
                + "/%d/im_tar.png" % (i),
            )
            torchvision.utils.save_image(
                pred_imgs["HFRefineImg"],
                test_ops.result_folder
                + "/%d/im_hfrefine.png" % (i),
            )
            torchvision.utils.save_image(
                pred_imgs["RefineImg"]*mask,
                test_ops.result_folder
                + "/%d/im_refine_mask.png" % (i),
            )
            torchvision.utils.save_image(
                pred_imgs["OutputImg"]*mask,
                test_ops.result_folder
                + "/%d/im_tar_mask.png" % (i),
            )
            torchvision.utils.save_image(
                pred_imgs["HFRefineImg"]*mask,
                test_ops.result_folder
                + "/%d/im_hfrefine_mask.png" % (i),
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
            