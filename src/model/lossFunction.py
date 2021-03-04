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


class get_loss(torch.nn.Module):
    def __init__(self):
        super(get_loss,self).__init__()
        self.criterion = nn.MSELoss()
        #self.criterion = nn.KLDivLoss()

    def forward(self, predDepth, predIntensity, ptCloud, grayImage, targetTransform, device):

        """
        Make sure that all the variables are in the same device
        """
        predDepthRot = predDepth[1]
        predDepthT   = predDepth[0]

        """
        predIntensityRot = predIntensity[1]
        predIntensityT = predIntensity[0]
        """

        if predDepthRot.get_device() != device:
            predDepthRot = predDepthRot.to('cuda:'+str(device))
            predDepthT = predDepth[1].to('cuda:'+str(device))
        
        """
        if predIntensity[0].get_device() != device:
            predIntensityRot = predIntensityRot.to('cuda:'+str(device))
            predIntensityT = predIntensityT.to('cuda:'+str(device))
        """

        if ptCloud.get_device() != device:
            ptCloud = ptCloud.to('cuda:'+str(device))

        if targetTransform.get_device() != device:
            targetTransform = targetTransform.to('cuda:'+str(device))

        if grayImage.get_device() != device:
            grayImage = grayImage.to('cuda:'+str(device))
 
        # Read the calibration parameters
        calibFileRootDir = "/mnt/291d3084-ca91-4f28-8f33-ed0b64be0a8c/akshay/kitti/raw/train/2011_09_26"
        [P_rect, R_rect, RT] = findCalibParameterTensor(calibFileRootDir)

        # Extract Ground Truth point cloud
        ptCloudTarget = getGroundTruthPointCloud(ptCloud, P_rect, R_rect, RT)

        # Create the transformation function 
        predDepthTransform = createRTMatTensor(predDepthRot, predDepthT)
        #predIntensityTransform = createRTMatTensor(predIntensityRot, predIntensityT)


        # Inverse of the target transform
        invTargetRT = calculateInvRTTensor(targetTransform)

        # Extract the translation from target transform
        targetT = targetTransform[:,:3,3].unsqueeze(1)

        # Calculate the distance between the target and the predicted
        #euclideanDistanceIntensity = calculateEucledianDistTensor(predIntensityT,targetT)
        euclideanDistanceDepth = calculateEucledianDistTensor(predDepthT, targetT)

        # Calculate the angular distance between the target and predicted
        targetR = matrix_to_euler_angles(targetTransform[:,:3,:3],"ZXY").unsqueeze(1)

        # print(predDepthRot)

        euclideanAngularDistanceDepth = calculateEucledianDistTensor(torch.rad2deg(predDepthRot), torch.rad2deg(targetR))
        #euclideanAngularDistanceIntensity = calculateEucledianDistTensor(torch.rad2deg(predIntensityRot), torch.rad2deg(targetR))

        # One component of the loss function 
        # Eucliedian depth loss
        #lossEDD = (0.7*euclideanAngularDistanceDepth) + (0.3*euclideanDistanceDepth*100)
        # Eucliedian Intensity loss
        #lossEDI = (0.7*euclideanAngularDistanceIntensity) + (0.3*euclideanDistanceIntensity)

        #lossEuclideanDistanceBtwTransform = lossEDD #+ lossEDI

        # Create cross correlation
        # Setp 0: Project the points that are rectified by inv of target transform
        # Step 1: Multiply the point clouds by the predicted value
        # Step 2: Caluclate the image tensor for target and predicted point cloud
        # Step 3: Caluclate the cross correlation of the target image and the predicted image
        # Step 4: Caluclate the maximum likelyhood summation value for a predefined radius of correlated value

        # Step 0
        # extract only the XYZ from the point cloud 

        ptCloudTargetHomo = convertToHomogenousCoordTensor(torch.transpose(ptCloudTarget, 2, 1)[:,:,:3])

        # Use this point cloud as the base for all the future caluclations 
        ptCloudBase = torch.matmul(invTargetRT, torch.transpose(ptCloud,2,1))

        ptCloudBase = convertToHomogenousCoordTensor(torch.transpose(ptCloudBase,2,1)[:,:,:3])

        # Step 1
        # Multiply the base point cloud with the predicted transponse 
        finalDepthPredPtCld = torch.matmul(predDepthTransform, torch.transpose(ptCloudBase, 2,1).type(torch.float))
        #finalIntensityPredCld = torch.matmul(predIntensityTransform, ptCloudBase.type(torch.float))

        # Calculate Euclidean Distance between the 
        euclideanDistanceDepthPtCld = calculateEucledianDistOfPointClouds(torch.transpose(finalDepthPredPtCld,2,1)[:,:,:3], torch.transpose(ptCloudTarget,2,1)[:,:,:3])
        #euclideanDistanceIntensityPtCld = calculateEucledianDistOfPointClouds(torch.transpose(finalIntensityPredCld,2,1)[:,:,:3], torch.transpose(finalIntensityPredCld,2,1)[:,:,:3])


        """

        # Step 2
        # Get the image tensor by projecting the point cloud back to image plane
        # These points are in the image coordinate frame
        targetPredPtCldImgCord = getImageTensorFrmPtCloud(P_rect, ptCloudTarget)
        finalDepthPredPtCldImgCord = getImageTensorFrmPtCloud(P_rect.type(torch.float), finalDepthPredPtCld)
        finalIntensityPredCldImgCord = getImageTensorFrmPtCloud(P_rect.type(torch.float), finalIntensityPredCld)

        # Transpose the vectors to create a mask
        targetPredPtCldImgCord = torch.transpose(targetPredPtCldImgCord,2,1)
        finalDepthPredPtCldImgCord = torch.transpose(finalDepthPredPtCldImgCord,2,1)
        finalIntensityPredCldImgCord = torch.transpose(finalIntensityPredCldImgCord,2,1)

        # Now filter the points that are not in front of the camera 
        imgHeight = 375
        imgWidth = 1242

        # Replace the 4th coloum of pt by intensities
        ptCloudTarget = torch.cat((torch.transpose(ptCloudTarget,2,1)[:,:,:3], torch.unsqueeze(intensity,dim=2)),dim=2)
        finalDepthPredPtCld = torch.cat((torch.transpose(finalDepthPredPtCld,2,1)[:,:,:3], torch.unsqueeze(intensity,dim=2)),dim=2)
        finalIntensityPredCld = torch.cat((torch.transpose(finalIntensityPredCld,2,1)[:,:,:3], torch.unsqueeze(intensity,dim=2)),dim=2)

        targetImgCoord, targetPtCld = filterPtClds(targetPredPtCldImgCord, ptCloudTarget, imgHeight, imgWidth)
        finalDepthImgCoord, finalDepthPredPtCld = filterPtClds(finalDepthPredPtCldImgCord, finalDepthPredPtCld, imgHeight, imgWidth)
        finalIntImgCoord, finalIntensityPredCld = filterPtClds(finalIntensityPredCldImgCord, finalIntensityPredCld, imgHeight, imgWidth)
        
        # create Depth Image tensor
        targetDepthTensor = createImage(targetImgCoord, targetPtCld[:,:,2],imgWidth, imgHeight)
        targetIntensityTensor = createImage(targetImgCoord, targetPtCld[:,:,3],imgWidth, imgHeight)
        depthTensor = createImage(finalDepthImgCoord,finalDepthPredPtCld[:,:,2], imgWidth, imgHeight)
        IntensityTensor = createImage(finalIntImgCoord,finalIntensityPredCld[:,:,3], imgWidth, imgHeight)


        
        # Create a sobel Kernel to run thru the image
        edgeDepthTensor = applySobelOperator(depthTensor)
        edgeintensityTensor = applySobelOperator(IntensityTensor)
        grayImage = applySobelOperator(grayImage.type(torch.float))


        # Cross-Correlation
        nccDepth = NCC(torch.transpose(targetDepthTensor,3,1))
        nccIntensity = NCC(targetIntensityTensor)

        # Move it cuda 
        nccDepth = nccDepth.to('cuda:'+str(device))
        nccIntensity = nccIntensity.to('cuda:'+str(device))

        crossCorrelationDepth = nccDepth(torch.transpose(depthTensor[None,...],3,1))
        crossCorrelationIntensity = nccIntensity(IntensityTensor[None,...])

        # Get MaxLikely hood sum
        """

        euclideanDistanceLoss = torch.max(euclideanDistanceDepthPtCld) # +torch.mean(euclideanDistanceIntensityPtCld),2)

        totalLOSS = euclideanDistanceLoss + euclideanDistanceDepth + euclideanAngularDistanceDepth
        

        return(torch.mean(totalLOSS))


