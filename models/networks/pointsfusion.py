import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch3d.ops import knn_points, knn_gather
import torchvision

class PointsFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PointsFusion, self).__init__()

        layers = []
        out_channels = [in_channels, *out_channels]
        for i in range(1, len(out_channels)):
            layers += [nn.Conv2d(out_channels[i - 1], out_channels[i], 1, bias=True), nn.BatchNorm2d(out_channels[i], track_running_stats=False),nn.ReLU()]
        
        layers_1 = []
        for i in range(1, len(out_channels)):
            layers_1 += [nn.Conv2d(out_channels[i - 1], out_channels[i], 1, bias=True), nn.BatchNorm2d(out_channels[i], track_running_stats=False),nn.ReLU()]
        self.conv = nn.Sequential(*layers)
        self.conv_1 = nn.Sequential(*layers_1)
    
    def knn_group(self, points1, points2,features1, features2,k,confs=None):
        # For each point in points1, query kNN points/features in points2/features2

        points1 = points1.permute(0,2,1).contiguous()
        points2 = points2.permute(0,2,1).contiguous()
        features1 = features1.permute(0,2,1).contiguous()
        features1 = features1.unsqueeze(2).repeat(1,1,k,1)
        _, nn_idx, nn = knn_points(points1, points2, K=k, return_nn=True)
        points_resi = nn - points1.unsqueeze(2).repeat(1,1,k,1)
        grouped_dist = torch.norm(points_resi, dim=-1, keepdim=True)
        grouped_features = knn_gather(features2.permute(0,2,1), nn_idx)
        
        features_resi = torch.mul(features1,grouped_features)
        features_resi = torch.sum(features_resi,-1).unsqueeze(3)

        if confs == None:
            new_features = torch.cat([points_resi, grouped_dist,features_resi], dim=-1)
        else:
            grouped_confs = knn_gather(confs.permute(0,2,1), nn_idx)
            new_features = torch.cat([points_resi, grouped_dist,features_resi,grouped_confs], dim=-1)

        return new_features.permute(0,3,1,2).contiguous(),\
            nn.permute(0,3,1,2).contiguous(),\
            grouped_features.permute(0,3,1,2).contiguous()

    
    def forward(self, points1, points2,features1, features2,k,n_ratio,depth_conf1=None,depth_conf2=None):
        N = points1.shape[-1]
        B = points1.shape[0]
        C = features1.shape[1]

        points = torch.cat((points1,points2),dim = -1)
        features = torch.cat((features1,features2),dim = -1)
        



        # random sampling
        nn = points.shape[-1]
        randidx = torch.randperm(nn)[:int(nn*n_ratio/2)]
        basepoints = points[:,:,randidx]
        basefeatures = features[:,:,randidx]

        if depth_conf1 == None:
            new_features, new_grouped_points, new_grouped_features = self.knn_group(basepoints, points, basefeatures,features, k)
        else:
            confs = torch.cat((depth_conf1,depth_conf2),dim = -1)
            new_features, new_grouped_points, new_grouped_features = self.knn_group(basepoints, points, basefeatures,features, k,confs)
        
        new_features_1 = torch.clone(new_features)
        
        #lean wights for point positions and descriptos
        ##############
        new_features = self.conv(new_features)
        new_features = torch.max(new_features, dim=1, keepdim=False)[0]
        weights = F.softmax(new_features, dim=-1)

        weights = weights.unsqueeze(1).repeat(1,3,1,1)

        ##############
        new_features_1 = self.conv_1(new_features_1)
        new_features_1 = torch.max(new_features_1, dim=1, keepdim=False)[0]
        weights_1 = F.softmax(new_features_1, dim=-1)

        weights_1 = weights_1.unsqueeze(1).repeat(1,C,1,1)

        ##############
        # fuse points with descriptors
        fused_points = torch.sum(torch.mul(weights, new_grouped_points), dim=-1, keepdim=False)
        fused_features = torch.sum(torch.mul(weights_1, new_grouped_features), dim=-1, keepdim=False)
        
        points = torch.cat([fused_points, fused_features], dim=1)

        return points

