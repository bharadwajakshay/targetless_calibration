import torch
from model.onlineCalibration import onlineCalibration
from model.resnet import *
import numpy as np
from tqdm import tqdm
import os
import sys
import importlib
import shutil
import json
import time
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
from common.tensorTools import saveModelParams, saveCheckPoint

from torchvision import transforms
from model.transformsTensor import *
from common.tensorTools import calculateInvRTTensor, exponentialMap, moveToDevice

import config
from torchsummary import summary
from pytictoc import TicToc
from datetime import datetime
from model.onlineCalibration import onlineCalibration

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'model'))
modelPath = '/home/akshay/targetless_calibration/src/model/trained/bestTargetCalibrationModel.pth'


os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2'

def inference(model, data):
    print(breakpoint)

    simpleDistanceSE3 = np.empty(0)

    # Setup arrays to report the errors
    eulerAngleErrors = np.empty((3,len(dataLoader)), dtype=float)
    translationError = np.empty((3,len(dataLoader)), dtype=float)
    s3DistanceError = np.empty((1,len(dataLoader)), dtype=float)
    gteulerAngle = np.empty((3,len(dataLoader)), dtype=float)
    gttranslation = np.empty((3,len(dataLoader)), dtype=float)

    for j, data in tqdm(enumerate(dataLoader,0), total=len(dataLoader)):
        # Expand the data into its respective components
        srcClrT, srcDepthT, targetTransformT, ptCldT , options = data
        # Transpose the tensors such that the no of channels are the 2nd term
        srcClrT = srcClrT.to('cuda')
        srcDepthT = srcDepthT.to('cuda')

        predTransform  = model(srcClrT, srcDepthT)

        # Simple SE3 distance
        predRot = quaternion_to_matrix(predTransform[:,:4])
        predT = predTransform[:,4:]

        targetTransformT = moveToDevice(targetTransformT, predTransform.get_device())

        gtRot = targetTransformT[:,:3,:3].type(torch.float32)
        gtT = targetTransformT[:,:3,3].type(torch.float32)

        RtR = torch.matmul(calculateInvRTTensor(predRot, predT), targetTransformT.type(torch.float32))
        
        I = moveToDevice(torch.eye(4,dtype=torch.float32), predTransform.get_device())

        simpleDistanceSE3 = np.empty(RtR.shape[0],dtype=float)
        for batchno in range(0,RtR.shape[0]):
            simpleDistanceSE3[batchno] = torch.norm( RtR[batchno] - I,'fro').to('cpu').numpy()

        s3DistanceError[0,j] = np.mean(simpleDistanceSE3)

        # Caluclate the euler angles from rotation matrix
        predEulerAngles = matrix_to_euler_angles(predRot, "ZXY")
        targetEulerAngles = matrix_to_euler_angles(gtRot, "ZXY")
        errorEulerAngle = torch.abs(targetEulerAngles - predEulerAngles)
        eulerAngleErrors[:,j] = torch.rad2deg(torch.mean(errorEulerAngle,dim=0)).to('cpu').numpy()
        errorTranslation = torch.abs(gtT - predT)
        translationError[:,j] = torch.mean(errorTranslation,dim=0).to('cpu').numpy()

        gteulerAngle[:,j] = torch.rad2deg(torch.mean(targetEulerAngles,dim=0)).to('cpu').numpy()
        gttranslation[:,j] = torch.mean(gtT,dim=0).to('cpu').numpy()

    return(s3DistanceError, eulerAngleErrors, translationError, gteulerAngle, gttranslation)
    


def main():
    print("Running inference")

    modelPath = config.pathToPretrainedModel
    if not os.path.exists(modelPath):
        print("No trained model found")
        exit(-1)

    if not os.path.exists(config.evaluationDataFile):
        print("No dataset file found")
        exit(-1)
    
    # empty the CUDA memory
    torch.cuda.empty_cache()

    model = onlineCalibration(config.backbone, depth=config.networkDepth)
    if torch.cuda.is_available():
        device = 'cuda'
        if torch.cuda.device_count() > 1:
            print('Multiple GPUs found. Moving to Dataparallel approach')
            model = torch.nn.DataParallel(model)
    else: 
        device = 'cpu'

    model = model.to(device)

    EVALUATION_DATASET = dataLoader(config.evaluationDataFile, mode='eval')

    evalDataLoader = torch.utils.data.DataLoader(EVALUATION_DATASET, batch_size=1,
                                                 shuffle=True, num_workers=0,drop_last=False)
    
    # Check if log directories are available
    logsDirsEvaluation = os.path.join(config.logsDirs,'evaluation')
    if not os.path.exists(logsDirsEvaluation):
        os.makedirs(logsDirsEvaluation)

    # logging the errors
    simpleDistanceSE3Err = float(config.previousBestSE3Dist)
    distanceErr = np.empty((1,0))
    angularErr = np.empty((3,0))
    translationErr = np.empty((3,0))
    angularGT = np.empty((3,0))
    translationGT = np.empty((3,0))

    try:
        print("Loading the model weight")
        modelWeights = torch.load(config.pathToPretrainedModel)
        model.load_state_dict(modelWeights['modelStateDict'])
    except:
        print("Failed to load the model.")
        exit(-1)

    if not os.path.isdir(os.path.join(config.logsDirs)):
        os.makedirs(config.logsDirs)
        Path(os.path.join(config.logsDirs,'DistanceErr.npy')).touch()
        Path(os.path.join(config.logsDirs,'ErrorInAngles.npy')).touch()
        Path(os.path.join(config.logsDirs,'ErrorInTranslation.npy')).touch()
        Path(os.path.join(config.logsDirs,'GroundTruthAngles.npy')).touch()
        Path(os.path.join(config.logsDirs,'GroundTruthTranslation.npy')).touch()

    # Get timestamp 
    currentTimeStamp = datetime.timestamp(currentTimeStamp)
    
    # Open Inference file 
    logFileInference = open(os.path.join(logsDirsEvaluation,str(currentTimeStamp)+'.txt'),'w')

    model = model.eval()

    with torch.no_grad():
        simpleDistanceSE3, errorInAngles, errorInTranslation, gtAngles, gtTranslation = inference(model, evalDataLoader)






   


if __name__ == "__main__":
    main()