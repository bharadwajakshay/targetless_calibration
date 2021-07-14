import torch
from model.resnet import *
import numpy as np
from tqdm import tqdm
import os
import sys
import importlib
import shutil
import json
from data_prep.dataLoader import *
import importlib
from pathlib import Path
import provider
from model import regressor
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2'
from model.lossFunction import get_loss
import concurrent.futures
from common.utilities import *
from common.pytorch3D import *
from common.tensorTools import convertImageTensorToCV, overlayPtCldOnImg, saveModelParams

from torchvision import transforms
from model.transformsTensor import *
from common.tensorTools import calculateInvRTTensor, exponentialMap, moveToDevice

import config
from torchsummary import summary
from pytictoc import TicToc

from common.tensorTools import findCalibParameterTensor, getGroundTruthPointCloud, getImageTensorFrmPtCloud

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'model'))
modelPath = '/home/akshay/targetless_calibration/src/model/trained/bestTargetCalibrationModel.pth'


os.environ["CUDA_VISIBLE_DEVICES"] = '0'

_Debug = False


def evaluate(colorImgModel, depthImgModel, regressorModel, maxPool, dataLoader, calibFileRootDir):

    simpleDistanceSE3 = np.empty(0)
    errorTranslationVec = np.empty((len(dataLoader),3))
    errorEulerAngleVec = np.empty((len(dataLoader),3))

    for j, data in tqdm(enumerate(dataLoader,0), total=len(dataLoader)):

       # Expand the data into its respective components
        srcClrT, srcDepthT, ptCldT, __, targetTransformT, optional = data

        # Transpose the tensors such that the no of channels are the 2nd term
        srcClrT = srcClrT.to('cuda')
        srcDepthT = srcDepthT.to('cuda')

        # Cuda 0
        featureMapClrImg = colorImgModel(srcClrT)
        # Cuda 1
        maxPooledDepthImg = maxPool(srcDepthT)
        featureMapDepthImg = depthImgModel(maxPooledDepthImg)

        # Cuda 0
        aggClrDepthFeatureMap = torch.cat([featureMapDepthImg.to('cuda'),featureMapClrImg],dim=1)

        # Cuda 0
        predTransform  = regressorModel(aggClrDepthFeatureMap)

        # Simple SE3 distance
        predRot = quaternion_to_matrix(predTransform[:,:4])
        predT = predTransform[:,4:]


        targetTransformT = moveToDevice(targetTransformT, predTransform.get_device())

        gtRot = targetTransformT[:,:3,:3].type(torch.float32)
        gtT = targetTransformT[:,:3,3].type(torch.float32)

        RtR = torch.matmul(calculateInvRTTensor(predRot, predT), targetTransformT.type(torch.float32))
        #RtR = torch.matmul(calculateInvRTTensor(targetTransformT.type(torch.float32)[:,:3,:3], targetTransformT.type(torch.float32)[:,:3,3]), targetTransformT.type(torch.float32))

        
        I = moveToDevice(torch.eye(4,dtype=torch.float32), predTransform.get_device())
        simpleDistanceSE3 = np.append(simpleDistanceSE3, torch.norm( RtR - I,'fro').to('cpu').numpy())


        # Caluclate the euler angles from rotation matrix
        predEulerAngles = matrix_to_euler_angles(predRot, "ZXY")
        targetEulerAngles = matrix_to_euler_angles(gtRot, "ZXY")

        
        errorEulerAngle = torch.square(targetEulerAngles - predEulerAngles)
        errorEulerAngle = torch.rad2deg(torch.sqrt(torch.mean(errorEulerAngle,dim=0)))
        errorEulerAngleVec[j,:] = errorEulerAngle.to('cpu').numpy()
        
        errorTranslation = torch.square(gtT - predT)
        errorTranslation = torch.sqrt(torch.mean(errorTranslation,dim=0))
        errorTranslationVec[j,:] = errorTranslation.to('cpu').numpy()
        

        '''
        errorEulerAngle = torch.abs(targetEulerAngles - predEulerAngles)
        errorEulerAngle = torch.rad2deg(torch.mean(errorEulerAngle,dim=0))
        errorEulerAngleVec[j,:] = errorEulerAngle.to('cpu').numpy()
        
        errorTranslation = torch.abs(gtT - predT)
        errorTranslation = torch.mean(errorTranslation,dim=0)
        errorTranslationVec[j,:] = errorTranslation.to('cpu').numpy()
        '''
        
        
        
        
        '''
        # Visualization Testing
        # Conert the image tensor to image 
        clrImage = convertImageTensorToCV(optional[0])
        # Step1: Get Ground Truth Data
        # Get calibration data
        [P, rectR, rT] = findCalibParameterTensor(calibFileRootDir)
        gTData = getGroundTruthPointCloud(ptCldT, P, rectR, rT)
        # Visualization
        image = overlayPtCldOnImg(clrImage, ptCldT, rT, P, rectR)
        for batch in range(0,image.shape[0]):
            cv2.imwrite('testing/images/Evaluation/groundTruth_'+str(j)+'_'+str(batch)+'.png', image[batch])

        # Step2: Multiply the pointcloud with Target transform
        targetTransformedPtCld = getGroundTruthPointCloud(gTData,__,torch.eye(4,dtype=torch.float64), targetTransformT)
        imageTargetTransform = overlayPtCldOnImg(clrImage, targetTransformedPtCld, torch.eye(4,dtype=torch.float64), P, torch.eye(4,dtype=torch.float64))

        for batch in range(0,image.shape[0]):
            cv2.imwrite('testing/images/Evaluation/TargetTransform'+str(j)+'_'+str(batch)+'.png', imageTargetTransform[batch])

        # Step3: Multiply the pointcloud with Target transform with inverse of predicted transform
        invPredRT = calculateInvRTTensor(predRot, predT).type(torch.float64)
        invPredTransformedPtCld = getGroundTruthPointCloud(gTData,__,torch.eye(4,dtype=torch.float64), invPredRT)
        imageInvPredTransform = overlayPtCldOnImg(clrImage, invPredTransformedPtCld, torch.eye(4,dtype=torch.float64), P, torch.eye(4,dtype=torch.float64))

        for batch in range(0,image.shape[0]):
            cv2.imwrite('testing/images/Evaluation/InvPredTransform'+str(j)+'_'+str(batch)+'.png', imageInvPredTransform[batch])
        '''

        
    return(np.mean(simpleDistanceSE3), errorEulerAngleVec, errorTranslationVec)


def main():

    # Default parameters 
    epochs = config.training['epoch']

    # Path to Pretrained models
    modelPath = config.pathToPretrainedModel

    # Time instance
    timeInstance = TicToc()

    """
    +---------------------------+--------+
    |      Model/Variable       |  GPU   |
    +---------------------------+--------+
    | RESNet50: Color Image     | CUDA 0 |
    | RESNet50: Depth Image     | CUDA 1 |
    | RESNet50: Intensity Image | CUDA 2 |
    | Regressor NW              | CUDA 0 |
    | LossFunction              | CUDA 1 |
    | Color Image Tensor        | CUDA 0 |
    | Depth Image Tensor        | CUDA 1 |
    | Intensity Image Tensor    | CUDA 2 |
    +---------------------------+--------+
    """

    # empty the CUDA memory
    torch.cuda.empty_cache()

    # Choose the RESNet network used to get features from the images 
    resnetClrImg = resnet50(pretrained=True).to('cuda')
    resnetDepthImg = resnet50(pretrained=False).to('cuda')
    regressor_model = regressor.regressor().to('cuda')

    # define max pooling layer
    maxPool = torch.nn.MaxPool2d(5, stride=1)
    #resnetIntensityImg = resnet50(pretrained=True)

    # Get the max point cloud size
    file = open(config.maxPtCldSizeFile,'r')
    maxPtCldSize = int(file.read())
    file.close()

    # Hyper Parameters 

    EVALUATE_DATASET = dataLoader(config.trainingDataFile, maxPtCldSize,mode='evaluate')

    evaluateDataLoader = torch.utils.data.DataLoader(EVALUATE_DATASET, batch_size=1, shuffle=True, num_workers=0)

    # Error 
    simpleDistanceSE3Err = 100
    distanceErr = np.empty(0)
    angularErr = np.empty(0)
    translationErr = np.empty(0)

    # Check if there are existing models, If so, Load them
    # Check if the file exists
    if os.path.isfile(modelPath):
        try:
            model_weights = torch.load(modelPath)
            resnetDepthImg.load_state_dict(model_weights['resNetModelStateDict'])
            regressor_model.load_state_dict(model_weights['modelStateDict'])
        except:
            print("Failed to load the model. Continuting without loading weights")

    # Check if the logs folder exitst, if not make the dir
    if not os.path.isdir(config.logsDirs):
        os.makedirs(config.logsDirs)
        Path(os.path.join(config.logsDirs,'DistanceErr.npy')).touch()
        Path(os.path.join(config.logsDirs,'ErrorInAngles.npy')).touch()
        Path(os.path.join(config.logsDirs,'ErrorInTranslation.npy')).touch()

    manhattanDistArray = np.empty(0)

    with torch.no_grad():
        simpleDistanceSE3, errorInAngles, errorInTranslation = evaluate(resnetClrImg, resnetDepthImg, regressor_model, maxPool, evaluateDataLoader, config.calibrationDir)
        meanErrorsInAngles = np.mean(errorInAngles,axis=0)
        meanErrorsInTranslation = np.mean(errorInTranslation,axis=0)

        print("Calculated mean Errors:" +  str(simpleDistanceSE3))
        print("Mean Angular Error: "+str(meanErrorsInAngles))
        print("Mean Translation Error: "+str(meanErrorsInTranslation))

    
    np.save(os.path.join(config.logsDirs,'ErrorInAngles.npy'), errorInAngles)
    np.save(os.path.join(config.logsDirs,'ErrorInTranslation.npy'), errorInTranslation)

    
    print("something")
        
            

if __name__ == "__main__":
    main()

