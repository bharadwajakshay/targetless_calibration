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

        if mode == 'rotation':
            # Calculate Euclidean Distance between the 
            manhattanDistancePtCld = calculateManhattanDistOfPointClouds(torch.transpose(ptCloudTarget,2,1)[:,:,:3], torch.transpose(ptCloudPred,2,1)[:,:,:3], ptCldSize)

            # Get inv pred Transformm
            invPredTransform = calculateInvRTTensor(predTransform[:,:3,:3],predTransform[:,:3,3])

            # Since reverting only the translation, predTransform[:,:3,:3] = eye 
            predTransform[:,:3,:3] = torch.eye(3)
            rotationCorrectedPtCld = torch.matmul(predTransform,torch.transpose(ptCloudBaseHomo,2,1))

            # Step 2
            # Get the image tensor by projecting the point cloud back to image plane
            # These points are in the image coordinate frame
            PredPtCldImgCord = getImageTensorFrmPtCloud(P_rect, rotationCorrectedPtCld)

            # Transpose the vectors to create a mask
            PredPtCldImgCord = torch.transpose(PredPtCldImgCord,2,1)

            # Now filter the points that are not in front of the camera 
            imgHeight = 375
            imgWidth = 1242

            # Replace the 4th coloum of pt by intensities
            rotationCorrectedPtCld = torch.cat((torch.transpose(rotationCorrectedPtCld,2,1)[:,:,:3], torch.unsqueeze(ptCldT[:,:,3],dim=2)),dim=2)
        
            predImgCoord, rotationCorrectedPtCld = filterPtClds(PredPtCldImgCord, rotationCorrectedPtCld, imgHeight, imgWidth)
        
            # create Depth Image tensor
            predDepthTensor = createImage(predImgCoord, rotationCorrectedPtCld[:,:,2],imgWidth, imgHeight).transpose(1,3).transpose(2,3)
            predIntensityTensor = createImage(predImgCoord, rotationCorrectedPtCld[:,:,3],imgWidth, imgHeight).transpose(1,3).transpose(2,3)

            # sanity check the images 
            #sanityCheckDepthMaps(predDepthTensor,predIntensityTensor)

            # Caluclate the depth map for next stage filtering
            # create a depthTensor such that channel 1 and 2 = depth maps and channel 3 is intensity map
            rotationCorrectedDepthMap = torch.empty_like(srcDepthT.cpu())
            rotationCorrectedDepthMap[:,0] = predDepthTensor[:,0]
            rotationCorrectedDepthMap[:,1] = predDepthTensor[:,0]
            rotationCorrectedDepthMap[:,2] = predIntensityTensor[:,0]
        
            imgTensorPreProc = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            rotationCorrectedDepthMap = imgTensorPreProc(rotationCorrectedDepthMap)

            manhattanDistanceLoss = torch.mean(manhattanDistancePtCld) # +torch.mean(euclideanDistanceIntensityPtCld),2)

            totalLoss = manhattanDistanceLoss + rotationLoss  #+ photometricLoss
        
            return(totalLoss.type(torch.float32),manhattanDistanceLoss,rotationCorrectedDepthMap)

        else:
            euclideanDistancePtCld = calculateManhattanDistOfPointClouds(torch.transpose(ptCloudTarget,2,1)[:,:,:3], torch.transpose(ptCloudPred,2,1)[:,:,:3], ptCldSize)
            totalLoss = torch.mean(euclideanDistancePtCld) + torch.mean(translationLoss) + (5*torch.mean(rotationLoss))
            return(totalLoss.type(torch.float32), torch.mean(euclideanDistancePtCld))