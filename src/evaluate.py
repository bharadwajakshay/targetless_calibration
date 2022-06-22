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
from common.tensorTools import saveModelParams

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


os.environ["CUDA_VISIBLE_DEVICES"] = '0'

_Debug = False


def evaluate(model, dataLoader):

    simpleDistanceSE3 = np.empty(0)

    # Setup arrays to report the errors
    eulerAngleErrors = np.empty((3,len(dataLoader)),dtype=float)
    translationError = np.empty((3,len(dataLoader)),dtype=float)
    s3DistanceError = np.empty((1,len(dataLoader)),dtype=float)
    timeConsumption = np.empty((1,len(dataLoader)),dtype=float)
  

    for j, data in tqdm(enumerate(dataLoader,0), total=len(dataLoader)):

       # Expand the data into its respective components
        srcClrT, srcDepthT, __, ptCldT, ptCldSize, targetTransformT, options = data

        # Transpose the tensors such that the no of channels are the 2nd term
        srcClrT = srcClrT.to('cuda')
        srcDepthT = srcDepthT.to('cuda')

        startTimeStamp = datetime.now()
    
        predTransform  = model(srcClrT, srcDepthT)

        endTimeStamp = datetime.now()
        timeConsumption[:,j] = datetime.timestamp(endTimeStamp)-datetime.timestamp(startTimeStamp)

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
        
    return(s3DistanceError, eulerAngleErrors, translationError, timeConsumption)


def main():

    # Default parameters 

    # Path to Pretrained models
    modelPath = config.pathToPretrainedModel

    # empty the CUDA memory
    torch.cuda.empty_cache()

    # Call the main model which includes all the other models
    model = onlineCalibration()
    if torch.cuda.is_available():
        device = 'cuda'
#        if torch.cuda.device_count() > 1:
#            print('Multiple GPUs found. Moving to Dataparallel approach')
#            model = torch.nn.DataParallel(model)
    else: 
        device = 'cpu'

    model = model.to(device)

    # Get the max point cloud size
    file = open(config.maxPtCldSizeFile,'r')
    maxPtCldSize = int(file.read())
    file.close()

    # Hyper Parameters 
    EVALUATE_DATASET = dataLoader(config.trainingDataFile, maxPtCldSize,mode='evaluate')

    evaluateDataLoader = torch.utils.data.DataLoader(EVALUATE_DATASET, batch_size=1, shuffle=True, num_workers=4,drop_last=True)
    
    # Check if log directories are available
    logsDirsEvaluate = os.path.join(config.logsDirs,'evaluate')
    if not os.path.exists(logsDirsEvaluate):
        os.makedirs(logsDirsEvaluate)

    # Error 
    simpleDistanceSE3Err = float(config.previousBestSE3Dist)
    distanceErr = np.empty((1,0))
    angularErr = np.empty((3,0))
    translationErr = np.empty((3,0))

    # Check if there are existing models, If so, Load themconfig
    # Check if the file exists
    if os.path.isfile(modelPath):
        try:
            model_weights = torch.load(modelPath)
            model.load_state_dict(model_weights['modelStateDict'])
        except:
            print("Failed to load the model. Continuting without loading weights")
            exit(-1)

    # Check if the logs folder exitst, if not make the dir
    if not os.path.isdir(config.logsDirs):
        os.makedirs(config.logsDirs)
        Path(os.path.join(config.logsDirs,'DistanceErr.npy')).touch()
        Path(os.path.join(config.logsDirs,'ErrorInAngles.npy')).touch()
        Path(os.path.join(config.logsDirs,'ErrorInTranslation.npy')).touch()

    manhattanDistArray = np.empty(0)

    # Get timestamp 
    currentTimeStamp = datetime.now()
    currentTimeStamp = datetime.timestamp(currentTimeStamp)
    
    # Open Evaluation file 
    logFileEvaluation = open(os.path.join(logsDirsEvaluate,str(currentTimeStamp)+'.txt'),'w')


    with torch.no_grad():
        model=model.eval()
        simpleDistanceSE3, errorInAngles, errorInTranslation, executionTime = evaluate(model, evaluateDataLoader)

        print("Calculated mean Errors:" +  str(np.mean(simpleDistanceSE3))+"\t Standard Deviation:"+str(np.std(simpleDistanceSE3)))
        print("Mean Angular Error: "+str(np.mean(errorInAngles,axis=1))+"\t Standard Deviation:"+str(np.std(errorInAngles,axis=1)))
        print("Mean Translation Error: "+str(np.mean(errorInTranslation,axis=1))+"\t Standard Deviation:"+str(np.std(errorInTranslation,axis=1)))
        print("Mean Execution Time: "+str(np.mean(executionTime,axis=1))+"\t Standard Deviation:"+str(np.std(executionTime)))

        logFileEvaluation.write("Calculated mean SE3 Errors: "+  str(np.mean(simpleDistanceSE3)))
        logFileEvaluation.write("Mean Angular Error: "+str(np.mean(errorInAngles,axis=1)))
        logFileEvaluation.write("Mean Translation Error: "+str(np.mean(errorInTranslation,axis=1)))


        distanceErr = np.append(distanceErr, simpleDistanceSE3)
        angularErr = np.append(angularErr, errorInAngles,axis=1)
        translationErr = np.append(translationErr, errorInTranslation,axis=1)
        

    np.save(os.path.join(logsDirsEvaluate,'DistanceErr.npy'), distanceErr)
    np.save(os.path.join(logsDirsEvaluate,'ErrorInAngles.npy'), angularErr)
    np.save(os.path.join(logsDirsEvaluate,'ErrorInTranslation.npy'), translationErr)

    logsDirsEvaluate.close()
            

if __name__ == "__main__":
    main()


