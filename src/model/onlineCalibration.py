import torch
import torch.nn as nn

from model.resnet import *
from model import regressor

class onlineCalibration(nn.Module):
    def __init__(self):
        super(onlineCalibration,self).__init__()

        self.resnetClrImg = resnet50(pretrained=True).to('cuda')
        self.resnetDepthImg = resnet50(pretrained=False).to('cuda')
        self.regressor_model = regressor.regressor().to('cuda')

        # define max pooling layer
        self.maxPool = torch.nn.MaxPool2d(5, stride=1)


    def forward(self, clrImg, depthImg):
        with torch.no_grad():
            clrFeatureMap = self.resnetClrImg(clrImg)

        maxPooledDepthImg = self.maxPool(depthImg)
        depthFeatureMap = self.resnetDepthImg(maxPooledDepthImg)
        aggClrDepthFeatureMap = torch.cat([depthFeatureMap,clrFeatureMap],dim=1)
        
        # Move the regressor model to Cuda 0 and pass the concatinated Feature Vector
        predTransform  = self.regressor_model(aggClrDepthFeatureMap)

        return(predTransform)