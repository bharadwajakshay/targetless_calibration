import torch
import torch.nn as nn

from model.resnet import *
from model import regressor
from model.swin import SWIN

class onlineCalibration(nn.Module):
    def __init__(self,backbone="RESNET"):
        super(onlineCalibration,self).__init__()

        self.backbone = backbone

        if backbone == 'RESNET':
            self.modelClrImg = resnet50(pretrained=True).to('cuda')
            self.modelDepthImg = resnet50(pretrained=False).to('cuda')
        elif backbone == 'SWIN':
            self.modelClrImg = SWIN(upscale=None, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                    mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')
            self.modelDepthImg = SWIN(upscale=None, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                    mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')

        self.regressor_model = regressor.regressor().to('cuda')

        # define max pooling layer
        self.maxPool = torch.nn.MaxPool2d(5, stride=1)


    def forward(self, clrImg, depthImg):
        with torch.no_grad():
            clrFeatureMap = self.modelClrImg(clrImg)

        maxPooledDepthImg = self.maxPool(depthImg)
        depthFeatureMap = self.modelDepthImg(maxPooledDepthImg)
        aggClrDepthFeatureMap = torch.cat([depthFeatureMap,clrFeatureMap],dim=1)
        
        # Move the regressor model to Cuda 0 and pass the concatinated Feature Vector
        predTransform  = self.regressor_model(aggClrDepthFeatureMap)

        return(predTransform)