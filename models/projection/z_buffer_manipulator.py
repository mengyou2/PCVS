import torchvision
import torch
import torch.nn as nn
from pytorch3d.structures import Pointclouds
import torch.nn.functional as F
EPS = 1e-2
import numpy as np
from models.networks.pointsfusion import PointsFusion

def get_splatter(
    name, depth_values, opt=None, size=256, C=64, points_per_pixel=8,proj_depth = False
):
    if name == "xyblending":
        from models.projection.z_buffer_layers import RasterizePointsXYsBlending
        if proj_depth == False:
            return RasterizePointsXYsBlending(
                C,
                learn_feature=opt.learn_default_feature,
                radius=opt.radius,
                size=size,
                points_per_pixel=points_per_pixel,
                opts=opt,
            )
        else:
            return RasterizePointsXYsBlending(
                C,
                learn_feature=opt.learn_default_feature,
                radius=opt.depth_radius,
                size=size,
                points_per_pixel=points_per_pixel,
                opts=opt,
            )
    else:
        raise NotImplementedError()

class PtsManipulator(nn.Module):
    def __init__(self, W, C=64,conf=False, opt=None):
        super().__init__()
        self.opt = opt
        self.k = opt.k

        self.splatter = get_splatter(
            opt.splatter, None, opt, size=W, C=C, points_per_pixel=opt.pp_pixel,proj_depth=False
        )
        self.splatter_d = get_splatter(
            opt.splatter, None, opt, size=W, C=1, points_per_pixel=opt.pp_pixel,proj_depth = True
        )
        
        xs = torch.linspace(0, W - 1, W) 
        ys = torch.linspace(0, W - 1, W) 

        xs = xs.view(1, 1, 1, W).repeat(1, 1, W, 1)
        ys = ys.view(1, 1, W, 1).repeat(1, 1, 1, W)

        coord = torch.cat(
            (xs, ys, torch.ones(xs.size()), torch.ones(xs.size())), 1
        ).view(1, 4, -1)
        self.register_buffer("coord", coord)
        if conf == True:
            self.points_fusion = PointsFusion(6, [64, 64,128,256])
        else:
            self.points_fusion = PointsFusion(5, [64, 64,128,256])
  


    def source_to_world(
        self, pts3D, K, K_inv, RT_cam1, RTinv_cam1
    ):

        projected_coors = self.coord * pts3D   #bs,4,wxw
        projected_coors[:, -1, :] = 1
        cam1_X = K_inv.bmm(projected_coors)   #kinv 1,4,4

        world = RTinv_cam1.bmm(cam1_X)
        return world
    def world_to_target(
        self, world, K, K_inv, RT_cam2, RTinv_cam2
    ):
        
        ones = torch.ones([world.size(0),1,world.size(2)]).cuda()
        world_ = torch.cat((world,ones),dim = 1)
        wrld_X = RT_cam2.bmm(world_)

        xy_proj = K.bmm(wrld_X) 
        mask = (xy_proj[:, 2:3, :].abs() < EPS).detach()

        # Remove invalid zs that cause nans
        zs = xy_proj[:, 2:3, :]
        zs[mask] = EPS
        xy = xy_proj[:, 0:2, :]*2/zs/(256-1)-1
        sampler = torch.cat((-xy, xy_proj[:, 2:3, :]), 1)
        sampler[mask.repeat(1, 3, 1)] = 10
        return sampler

    def forward_justpts(
        self, fea_1, fea_2,
        pred_pts_1, pred_pts_2,
        K, K_inv,
        input_RT_1, input_RT_2,
        input_RTinv_1, input_RTinv_2,
        output_RT,
        output_RTinv,
        depth_conf1=None ,depth_conf2=None,
    ):
        # Now project these points into a new view
        bs, c, w, h = fea_1.size()
        N = w*w
        
        if len(pred_pts_1.size()) > 3:
            # reshape into the right positioning
            pred_pts_1 = pred_pts_1.view(bs, 1, -1)  # bs，1，wxw
            pred_pts_2 = pred_pts_2.view(bs, 1, -1)  # bs，1，wxw
            src_1 = fea_1.view(bs, c, -1)   # bs,c,wxw
            src_2 = fea_2.view(bs, c, -1)   # bs,c,wxw

        
        pts3D_1 = self.source_to_world(
            pred_pts_1, K, K_inv, input_RT_1, input_RTinv_1
        )
        pts3D_2 = self.source_to_world(
            pred_pts_2, K, K_inv, input_RT_2, input_RTinv_2
        )


        pointcloud_1 = pts3D_1[:,:3]   
        pointcloud_2 = pts3D_2[:,:3]

     
        k=self.opt.k
        n_ratio = self.opt.n_ratio
        fused_points = self.points_fusion(pointcloud_1,pointcloud_2,src_1,src_2,k,n_ratio,depth_conf1.view(bs, 1, -1),depth_conf2.view(bs, 1, -1))
        
        pts_ = fused_points[:,:3]
        src_ = fused_points[:,3:]
        pts = self.world_to_target(pts_, K, K_inv, output_RT, output_RTinv)

        target_depth = pts[:,2,:].unsqueeze(1)
        result = self.splatter(pts.permute(0, 2, 1).contiguous(), src_.permute(0, 2, 1).contiguous()  )
        result_img = result[:,:3]
        result_fea = result[:,3:]

        result_depth = self.splatter_d(pts.permute(0, 2, 1).contiguous(), target_depth.permute(0, 2, 1).contiguous()  )
        return result_img,result_fea,result_depth

        
        