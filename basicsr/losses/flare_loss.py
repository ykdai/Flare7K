import cv2
import numpy as np
import torch
from torch import abs_, nn
from torch import optim
from PIL import Image
from typing import Mapping,Sequence,Tuple,Union
from torchvision.models import vgg19
import torchvision.models.vgg as vgg
from basicsr.utils.registry import LOSS_REGISTRY

class L_Abs_sideout(nn.Module):
    def __init__(self):
        super(L_Abs_sideout, self).__init__()
        self.resolution_weight=[1.,1.,1.,1.]

    def forward(self,x,flare_gt):
        #[256,256],[128,128],[64,64],[32,32]
        Abs_loss=0
        for i in range(4):
            flare_loss=torch.abs(x[i]-flare_gt[i])
            Abs_loss+=torch.mean(flare_loss)*self.resolution_weight[i]
        return Abs_loss
    

class L_Abs(nn.Module):
    def __init__(self):
        super(L_Abs, self).__init__()

    def forward(self,x,flare_gt,base_gt,mask_gt,merge_gt):
        base_predicted=base_gt*mask_gt+(1-mask_gt)*x
        flare_predicted=merge_gt-(1-mask_gt)*x
        base_loss=torch.abs(base_predicted-base_gt)
        flare_loss=torch.abs(flare_predicted-flare_gt)
        Abs_loss=torch.mean(base_loss+flare_loss)
        return Abs_loss

@LOSS_REGISTRY.register()
class L_Abs_pure(nn.Module):
    def __init__(self,loss_weight=1.0):
        super(L_Abs_pure, self).__init__()
        self.loss_weight=loss_weight

    def forward(self,x,flare_gt):
        flare_loss=torch.abs(x-flare_gt)
        Abs_loss=torch.mean(flare_loss)
        return self.loss_weight*Abs_loss

@LOSS_REGISTRY.register()
class L_Abs_weighted(nn.Module):
    def __init__(self,loss_weight=1.0):
        super(L_Abs_weighted, self).__init__()
        self.loss_weight=loss_weight

    def forward(self,x,flare_gt,weight):
        flare_loss=torch.abs(x-flare_gt)
        Abs_loss=torch.mean(flare_loss*weight)
        '''
        mask_area=torch.mean(torch.abs(weight))
        if mask_area>0:
            return self.loss_weight*Abs_loss/mask_area
        else:
        '''
        return self.loss_weight*Abs_loss

@LOSS_REGISTRY.register()
class L_percepture(nn.Module):
    def __init__(self,loss_weight=1.0):
        super(L_percepture, self).__init__()
        self.loss_weight=loss_weight
        vgg = vgg19(pretrained=True)
        model = nn.Sequential(*list(vgg.features)[:31])
        model=model.cuda()
        model = model.eval()
        # Freeze VGG19 #
        for param in model.parameters():
            param.requires_grad = False

        self.vgg = model
        self.mae_loss = nn.L1Loss()
        self.selected_feature_index=[2,7,12,21,30]
        self.layer_weight=[1/2.6,1/4.8,1/3.7,1/5.6,10/1.5]
    
    def extract_feature(self,x):
        selected_features = []
        for i,model in enumerate(self.vgg):
            x = model(x)
            if i in self.selected_feature_index:
                selected_features.append(x.clone())
        return selected_features

    def forward(self, source, target):
        source_feature = self.extract_feature(source)
        target_feature = self.extract_feature(target)
        len_feature=len(source_feature)
        perceptual_loss=0
        for i in range(len_feature):
            perceptual_loss+=self.mae_loss(source_feature[i],target_feature[i])*self.layer_weight[i]
        return self.loss_weight*perceptual_loss

@LOSS_REGISTRY.register()
class CorssEntropy(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(CorssEntropy, self).__init__()
        self.loss_weight=loss_weight
        self.loss = nn.BCELoss()

    def forward(self, source, target):
        cross_entropy_loss = self.loss(source, target)
        return self.loss_weight*cross_entropy_loss

@LOSS_REGISTRY.register()
class WeightedBCE(nn.Module):
    def __init__(self, loss_weight=1.0,class_weight=[1.0,1.0]):
        super(WeightedBCE, self).__init__()
        self.loss_weight=loss_weight
        self.class_weight = class_weight

    def forward(self, input, target):
        input = torch.clamp(input,min=1e-7,max=1-1e-7)
        bce = - self.class_weight[1] * target * torch.log(input) - (1 - target) * self.class_weight[0] * torch.log(1 - input)
        return torch.mean(bce)
