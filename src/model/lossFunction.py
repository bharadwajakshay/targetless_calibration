import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation as rot
import numpy as np
from torchvision import transforms

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
        self.criterion = nn.MSELoss(reduction='sum')
        #self.criterion = nn.KLDivLoss()


    def forward(self,predT, srcClrT, srcDepthT,ptCldT, ptCldSize, targetTransformT, calibFileRootDir, device, mode='rotation'):

        targetTransformT = moveToDevice(targetTransformT, device)
        predT = moveToDevice(predT, device)
        srcDepthT = moveToDevice(srcDepthT, device)
        targetDepthT = moveToDevice(srcClrT, device)
        ptCldT = moveToDevice(ptCldT, device)
        
        # Get Predicted Transform in terms of Rotation matrix
        # predicted transform Quaternion ---> Rotation matrix
        predRot = quaternion_to_matrix(predT[:,:4])
        predTrans = predT[:,4:]

        gtR = targetTransformT[:,:3,:3].type(torch.float32)
        gtT = targetTransformT[:,:3,3].type(torch.float32)

        # Calculate euclidean norm to get the measure of deviation of transform with ground truth        
        rotationLoss = torch.empty(gtR.shape[0])
        for batchno in range(0,gtR.shape[0]):
            rotationLoss[batchno] = torch.norm(torch.matmul(predRot[batchno,:,:].transpose(1,0), gtR[batchno,:,:]) - moveToDevice(torch.eye(3),device),'fro')


        translationLoss = torch.norm(predTrans - gtT,'fro')
        translationLoss = torch.empty(gtT.shape[0])
        for batchno in range(0,gtT.shape[0]):
            translationLoss[batchno] = torch.norm(predTrans[batchno,:] - gtT[batchno,:],'fro')
        

        # Read the calibration parameters
        #calibFileRootDir = "/mnt/291d3084-ca91-4f28-8f33-ed0b64be0a8c/akshay/kitti/raw/2011_09_26/"
        [P_rect, R_rect, RT] = findCalibParameterTensor(calibFileRootDir)

        # Extract Ground Truth point cloud
        ptCloudBase = getGroundTruthPointCloud(ptCldT, P_rect, R_rect, RT)

        # Create the transformation function 
        predTransform = convSO3NTToSE3(predRot, predTrans.unsqueeze(1)).type(torch.float64)

        # Create cross correlation
        # Setp 0: Project the points that are rectified by inv of target transform
        # Step 1: Multiply the point clouds by the predicted value
        # Step 2: Caluclate the image tensor for target and predicted point cloud
        # Step 3: Caluclate the cross correlation of the target image and the predicted image
        # Step 4: Caluclate the maximum likelyhood summation value for a predefined radius of correlated value

        # Step 0
        # extract only the XYZ from the point cloud 

        ptCloudBaseHomo = convertToHomogenousCoordTensor(ptCloudBase[:,:,:3])

        # Use this point cloud as the base for all the future caluclations 
        ptCloudTarget = torch.matmul(targetTransformT, torch.transpose(ptCloudBaseHomo,2,1))

        ptCloudPred = torch.matmul(predTransform, torch.transpose(ptCloudBaseHomo,2,1))

      
        euclideanDistancePtCld = calculateManhattanDistOfPointClouds(torch.transpose(ptCloudTarget,2,1)[:,:,:3], torch.transpose(ptCloudPred,2,1)[:,:,:3], ptCldSize)
        totalLoss = torch.mean(euclideanDistancePtCld) + (1.5*torch.mean(translationLoss)) + (2*torch.mean(rotationLoss))
        return(totalLoss.type(torch.float32), torch.mean(euclideanDistancePtCld))