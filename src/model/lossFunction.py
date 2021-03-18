import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation as rot
import numpy as np

from data_prep.filereaders import *
from data_prep.helperfunctions import *
from data_prep.transforms import *
from common.tensorTools import *
from common.pytorch3D import *
from model.NCC import NCC
from model.utils import *
from model.transformsTensor import *


class get_loss(torch.nn.Module):
    def __init__(self):
        super(get_loss,self).__init__()
        self.criterion = nn.MSELoss()
        #self.criterion = nn.KLDivLoss()


    def forward(self,predT, srcDepthT, targetDepthT, srcIntensityT, targetIntensityT, ptCldT, targetTransformT,device):

        targetTransformT = moveToDevice(targetTransformT, device)
        predT = moveToDevice(predT, device)
        srcDepthT = moveToDevice(srcDepthT, device)
        targetDepthT = moveToDevice(targetDepthT, device)
        srcIntensityT = moveToDevice(srcIntensityT, device)
        targetIntensityT = moveToDevice(targetIntensityT, device)
        ptCldT = moveToDevice(ptCldT, device)

        
        # Get Predicted Transform in terms of Rotation matrix 
        # Using Taylor Exponential Map
        predRot = exponentialMap(predT).type(torch.float64)

        # Calculate euclidean norm to get the measure of deviation of transform with ground truth
        predR = predRot[:,:3,:3]
        predT = predRot[:,:3,3]
        gtR = targetTransformT[:,:3,:3]
        gtT = targetTransformT[:,:3,3]
        
        rotationLoss = torch.norm( torch.matmul(predR.transpose(2,1), gtR) - moveToDevice(torch.eye(3),device),'fro')
        translationLoss = torch.norm(predT - gtT,'fro')
        totalLoss = rotationLoss + translationLoss
        
        
        return(totalLoss)


