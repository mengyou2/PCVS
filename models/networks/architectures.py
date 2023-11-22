

import torch
import torch.nn as nn
import torchvision

import torch.nn.functional as F
import numpy as np
import itertools
import torch.nn.init as init





# VGG architecture, used for the perceptual loss using a pretrained VGG network
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(
            pretrained=True
        ).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        # Normalize the image so that it is in the appropriate range
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out




class ResUNet(nn.Module): 
    def __init__(
        self, out_channels_0=64, out_channels=-1, depth=5, resnet="resnet18"
    ):
        super().__init__()

        if resnet == "resnet18":
            resnet = torchvision.models.resnet18(pretrained=True)
        else:
            raise Exception("invalid resnet model")

        self.normalizer = ImNormalizer()

        if depth < 1 or depth > 5:
            raise Exception("invalid depth of UNet")

        encs = nn.ModuleList()
        enc_translates = nn.ModuleList()
        decs = nn.ModuleList()
        enc_channels = 0
        if depth == 5:
            encs.append(resnet.layer4)
            enc_translates.append(self.convrelu(512, 512, 1))
            enc_channels = 512
        if depth >= 4:
            encs.append(resnet.layer3)
            enc_translates.append(self.convrelu(256, 256, 1))
            if enc_channels > 0:
                decs.append(self.convrelu(enc_channels + 256, 256))
            enc_channels = 256
        if depth >= 3:
            encs.append(resnet.layer2)
            enc_translates.append(self.convrelu(128, 128, 1))
            if enc_channels > 0:
                decs.append(self.convrelu(enc_channels + 128, 128))
            enc_channels = 128
        if depth >= 2:
            encs.append(nn.Sequential(resnet.maxpool, resnet.layer1))
            enc_translates.append(self.convrelu(64, 64, 1))
            if enc_channels > 0:
                decs.append(self.convrelu(enc_channels + 64, 64))
            enc_channels = 64
        if depth >= 1:
            encs.append(nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu))
            enc_translates.append(self.convrelu(64, 64, 1))
            if enc_channels > 0:
                decs.append(self.convrelu(enc_channels + 64, 64))
            enc_channels = 64
        enc_translates.append(
            nn.Sequential(self.convrelu(3, 64), self.convrelu(64, 64))
        )
        decs.append(self.convrelu(enc_channels + 64, out_channels_0))

        self.encs = nn.ModuleList(reversed(encs))
        self.enc_translates = nn.ModuleList(reversed(enc_translates))
        self.decs = nn.ModuleList(decs)

        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        if out_channels <= 0:
            self.out_conv = None
        else:
            self.out_conv = nn.Conv2d(
                out_channels_0, out_channels, kernel_size=1, padding=0
            )

    def convrelu(self, in_channels, out_channels, kernel_size=3, padding=None):
        if padding is None:
            padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
        )

    # disable batchnorm learning in self.encs
    def train(self, mode=True):
        super().train(mode=mode)
        if not mode:
            return
        for mod in self.encs.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.eval()
                for param in mod.parameters():
                    param.requires_grad_(False)

    def forward(self, x):
        x = self.normalizer.apply(x)

        outs = [self.enc_translates[0](x)]
        for enc, enc_translates in zip(self.encs, self.enc_translates[1:]):
            x = enc(x)
            outs.append(enc_translates(x))

        for dec in self.decs:
            x0, x1 = outs.pop(), outs.pop()
            x = torch.cat((self.upsample(x0), x1), dim=1)
            x = dec(x)
            outs.append(x)
        x = outs.pop()

        if self.out_conv:
            x = self.out_conv(x)
        return x



class ImNormalizer(object):
    def __init__(self, in_fmt="-11"):
        self.in_fmt = in_fmt

    def apply(self, x):
        mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).to(x.device)
        std = torch.Tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).to(x.device)


        if self.in_fmt == "-11":
            x = (x + 1) / 2
        elif self.in_fmt != "01":
            raise Exception("invalid input format")

        return (x - mean) /std

class ResidualBlock(nn.Module):
    def __init__(self, nf=64):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.batch_norm1 = nn.BatchNorm2d(nf, track_running_stats=False)
        # self.batch_norm1 = nn.BatchNorm2d(nf)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.batch_norm2 = nn.BatchNorm2d(nf, track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)
        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        out = self.relu(self.batch_norm1(self.conv1(x)))
        out = self.batch_norm2(self.conv2(out))

        return x + out

class ResNet(nn.Module):
    def __init__(self, in_channels=9,out_channels=3,nf=64):
        super(ResNet, self).__init__()
        self.conv0 = nn.Conv2d(in_channels, nf, 3, 1, 1, bias=True)
        self.res1 = ResidualBlock(nf)
        self.res2 = ResidualBlock(nf)
        self.res3 = ResidualBlock(nf)
        self.res4 = ResidualBlock(nf)
        self.res5 = ResidualBlock(nf)
        self.res6 = ResidualBlock(nf)
        self.conv1 = nn.Conv2d(nf, out_channels, 3, 1, 1, bias=True)
        self.norm = nn.Sigmoid()
        self.relu = nn.ReLU()


    def forward(self, x):
        x0 = self.relu(self.conv0(x))
        x1 = self.res1(x0)
        x2 = self.res2(x1) 
        x3 = self.res3(x2)
        x4 = self.res4(x3)
        x5 = self.res5(x4)
        x6 = self.res6(x5)
        x7 = self.conv1(x6)
        return x7



def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()


