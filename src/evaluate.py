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
import time

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


os.environ["CUDA_VISIBLE_DEVICES"] = '1'

_Debug = False


def evaluate(colorImgModel, depthImgModel, regressorModel, maxPool, dataLoader, calibFileRootDir):

    simpleDistanceSE3 = np.empty(0)
    RMSETranslationVec = np.empty((len(dataLoader),3))
    RMSEEulerAngleVec = np.empty((len(dataLoader),3))
    MAETranslationVec = np.empty((len(dataLoader),3))
    MAEEulerAngleVec = np.empty((len(dataLoader),3))

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

        '''
        #####################################################################################
        Root Mean Squared Error
        #####################################################################################
        '''
        RMSEEulerAngleVec = torch.square(torch.rad2deg(predEulerAngles - targetEulerAngles)).to('cpu').numpy()
        RMSETranslationVec[j,:] = torch.square(predT - gtT).to('cpu').numpy()

    
        '''
        #####################################################################################
        Mean Absolute Error
        #####################################################################################
        '''
        MAEEulerAngleVec[j,:] = torch.abs(torch.rad2deg(predEulerAngles - targetEulerAngles)).to('cpu').numpy()        
        MAETranslationVec[j,:] = torch.abs(predT - gtT).to('cpu').numpy()

        

    RMSEEulerAngleVec = np.sqrt(np.mean(RMSEEulerAngleVec, axis=0))
    RMSETranslationVec = np.sqrt(np.mean(RMSETranslationVec, axis=0))

    MAEEulerAngleVec = np.sqrt(np.mean(MAEEulerAngleVec, axis=0))
    MAETranslationVec = np.sqrt(np.mean(MAETranslationVec, axis=0))
        

        
    return(np.mean(simpleDistanceSE3), RMSEEulerAngleVec, RMSETranslationVec, MAEEulerAngleVec, MAETranslationVec)


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

    # Check if log directories are available
    logsDirsEvaluation = os.path.join(config.logsDirs,'evaluate')
    if not os.path.exists(logsDirsEvaluation):
        os.makedirs(logsDirsEvaluation)

    # open a log file
    timeStamp = str(time.time())
    logFile = open(os.path.join(logsDirsEvaluation,timeStamp+'.txt'),'a')

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

    manhattanDistArray = np.empty(0)

    with torch.no_grad():
        simpleDistanceSE3, RMSEInAngles, RMSEInTranslation, MAEInAngles, MAEInTranslation = evaluate(resnetClrImg, resnetDepthImg, regressor_model, maxPool, evaluateDataLoader, config.calibrationDir)
        

        print("Calculated mean Errors:" +  str(simpleDistanceSE3))
        print("RMSE Angular Error: "+str(RMSEInAngles))
        print("RMSE Translation Error: "+str(RMSEInTranslation))
        print("MAE Angular Error: "+str(MAEInAngles))
        print("MAE Translation Error: "+str(MAEInTranslation))


    logFile.write('Root Mean Square Error\nAngle: '+str(RMSEInAngles)+'\nTranslation: '+str(RMSEInTranslation)+'\n')
    logFile.write('Mean Absolute Error\nAngle: '+str(MAEInAngles)+'\nTranslation: '+str(MAEInTranslation))

    
    '''
    np.save(os.path.join(config.logsDirs,'RMSEInAngles.npy'), RMSEInAngles)
    np.save(os.path.join(config.logsDirs,'RMSEInTranslation.npy'), RMSEInTranslation)
    np.save(os.path.join(config.logsDirs,'MAEInAngles.npy'), MAEInAngles)
    np.save(os.path.join(config.logsDirs,'MAEInTranslation.npy'), MAEInTranslation)
    '''
    
    logFile.close()
        
            

if __name__ == "__main__":
    main()

