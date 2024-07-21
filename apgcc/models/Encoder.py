# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
import torch.nn.functional as F

# VGG backbone
class Base_VGG(nn.Module):
    def __init__(self, name: str, last_pool=False , num_channels=256, **kwargs):
        super().__init__()
        print("### VGG16: last_pool=", last_pool)
        # loading backbone features
        from .backbones import vgg as models
        if name == 'vgg16_bn':
            backbone = models.vgg16_bn(pretrained=True)
        elif name == 'vgg16':
            backbone = models.vgg16(pretrained=True)
        
        features = list(backbone.features.children())

        # setting base module.
        if name == 'vgg16_bn':
            self.body1 = nn.Sequential(*features[:13])
            self.body2 = nn.Sequential(*features[13:23])
            self.body3 = nn.Sequential(*features[23:33])
            if last_pool:
                self.body4 = nn.Sequential(*features[33:44])  # 32x down-sample
            else:
                self.body4 = nn.Sequential(*features[33:43])  # 16x down-sample
        else:
            self.body1 = nn.Sequential(*features[:9])
            self.body2 = nn.Sequential(*features[9:16])
            self.body3 = nn.Sequential(*features[16:23])
            if last_pool:
                self.body4 = nn.Sequential(*features[23:31])  # 32x down-sample
            else:
                self.body4 = nn.Sequential(*features[23:30])  # 16x down-sample
        self.num_channels = num_channels
        self.last_pool = last_pool
        
    def get_outplanes(self):
        outplanes = []
        for i in range(4):
            last_dims = 0
            for param_tensor in self.__getattr__('body'+str(i+1)).state_dict():
                if 'weight' in param_tensor:
                    last_dims = list(self.__getattr__('body'+str(i+1)).state_dict()[param_tensor].size())[0]
            outplanes.append(last_dims)
        return outplanes   # get the last layer params of all modules, and trans to the size.

    def forward(self, tensor_list):
        out = []
        xs = tensor_list
        for _, layer in enumerate([self.body1, self.body2, self.body3, self.body4]):
            xs = layer(xs)
            out.append(xs)
        return out

# ResNet backbone
class Base_ResNet(nn.Module):
    def __init__(self, name: str, last_pool=False , num_channels=256, **kwargs):
        super().__init__()
        print("### ResNet: last_pool=", last_pool)
        # loading backbone features
        from .backbones import resnet as models
        if name == 'resnet18':
            self.backbone = models.resnet18_ibn_a(pretrained=True)
        elif name == 'resnet34':
            self.backbone = models.resnet34_ibn_a(pretrained=True)
        elif name == 'resnet50':
            self.backbone = models.resnet50_ibn_a(pretrained=True)
        elif name == 'resnet101':
            self.backbone = models.resnet101_ibn_a(pretrained=True)
        elif name == 'resnet152':
            self.backbone = models.resnet152_ibn_a(pretrained=True)     

        self.num_channels = num_channels
        self.last_pool = last_pool

    def get_outplanes(self):
        outplanes = []
        for Layer in [self.backbone.layer1, self.backbone.layer2, self.backbone.layer3, self.backbone.layer4]:
            last_dims = 0
            for param_tensor in Layer.state_dict():
                if 'weight' in param_tensor:
                    last_dims = list(Layer.state_dict()[param_tensor].size())[0]
            outplanes.append(last_dims)
        return outplanes   # get the last layer params of all modules, and trans to the size.

    def forward(self, tensor_list):
        out = []
        xs = tensor_list
        out = self.backbone(xs)
        return out