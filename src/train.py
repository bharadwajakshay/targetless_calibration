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

_Debug = False


def test(model, dataLoader):

    simpleDistanceSE3 = np.empty(0)

    # Setup arrays to report the errors
    eulerAngleErrors = np.empty((3,len(dataLoader)),dtype=float)
    translationError = np.empty((3,len(dataLoader)),dtype=float)
    s3DistanceError = np.empty((1,len(dataLoader)),dtype=float)
    gteulerAngle = np.empty((3,len(dataLoader)),dtype=float)
    gttranslation = np.empty((3,len(dataLoader)),dtype=float)
  

    for j, data in tqdm(enumerate(dataLoader,0), total=len(dataLoader)):

       # Expand the data into its respective components
        srcClrT, srcDepthT, targetTransformT, ptCldT , options = data = data

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

    # Default parameters 
    epochs = config.training['epoch']

    # Path to Pretrained models
    modelPath = config.pathToPretrainedModel
    if not os.path.exists('/'.join(modelPath.split('/')[:-1])):
        os.makedirs('/'.join(modelPath.split('/')[:-1]))

    # Path to checkpoint
    currentTimeStamp = datetime.now()
    checkpointDir = os.path.join(config.pathToCheckpoint, str(currentTimeStamp))
    if not os.path.exists(checkpointDir):
        os.makedirs(checkpointDir)


    # Time instance
    timeInstance = TicToc()


    # empty the CUDA memory
    torch.cuda.empty_cache()

    # Call the main model which includes all the other models
    model = onlineCalibration(config.backbone, depth=config.networkDepth)
    if torch.cuda.is_available():
        device = 'cuda'
        if torch.cuda.device_count() > 1:
            print('Multiple GPUs found. Moving to Dataparallel approach')
            model = torch.nn.DataParallel(model)
    else: 
        device = 'cpu'

    model = model.to(device)

    # get th eloss fucntion
    loss_function = get_loss().to(device)
    
    # Hyper Parameters 
    TRAIN_DATASET = dataLoader(config.trainingDataFile, mode='train')
    TEST_DATASET = dataLoader(config.trainingDataFile, mode='test')

    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=config.training['batchSize'], shuffle=True, num_workers=10,drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=config.training['batchSize'], shuffle=True, num_workers=10,drop_last=True)
    #MODEL = importlib.import_module(pointcloudnet)

    
    # Check if log directories are available
    logsDirsTraining = os.path.join(config.logsDirs,'training')
    if not os.path.exists(logsDirsTraining):
        os.makedirs(logsDirsTraining)

    logsDirsTesting = os.path.join(config.logsDirs,'testing')
    if not os.path.exists(logsDirsTesting):
        os.makedirs(logsDirsTesting)

    

    optimizermodel = torch.optim.Adam(
        model.parameters(),
        lr = config.training['learningRate'],
        betas = (config.training['beta0'], config.training['beta1']),
        eps = config.training['eps'],
        weight_decay = config.training['decayRate'],
    )


    #schedulerModel = torch.optim.lr_scheduler.MultiStepLR(optimizermodel, milestones=[24,30], gamma=0.1)
    schedulerModel = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizermodel, mode='min', patience=4)
    

    start_epoch = 0
    global_epoch = 0
    global_step = 0
    besteulerdistance = 100
    # Error 
    simpleDistanceSE3Err = float(config.previousBestSE3Dist)
    distanceErr = np.empty((1,0))
    angularErr = np.empty((3,0))
    translationErr = np.empty((3,0))
    angularGT = np.empty((3,0))
    translationGT = np.empty((3,0))


    # Check if there are existing models, If so, Load themconfig
    # Check if the file exists
    if config.loadCheckPoint:
        try:
            model_weights = torch.load(config.checkpointFilename)
            model.load_state_dict(model_weights['modelStateDict'])
            global_epoch = model_weights['epoch']
        except:
            print("Failed to load the model. Continuting without loading weights")

    # Check if the logs folder exitst, if not make the dir

    if not os.path.isdir(os.path.join(config.logsDirs)):
        os.makedirs(config.logsDirs)
        Path(os.path.join(config.logsDirs,'DistanceErr.npy')).touch()
        Path(os.path.join(config.logsDirs,'ErrorInAngles.npy')).touch()
        Path(os.path.join(config.logsDirs,'ErrorInTranslation.npy')).touch()
        Path(os.path.join(config.logsDirs,'GroundTruthAngles.npy')).touch()
        Path(os.path.join(config.logsDirs,'GroundTruthTranslation.npy')).touch()
        

    manhattanDistArray = np.empty(0)

    # Get timestamp 
    currentTimeStamp = datetime.timestamp(currentTimeStamp)
    
    # Open Training file 
    logFileTraining = open(os.path.join(logsDirsTraining,str(currentTimeStamp)+'.txt'),'w')

    # Open Testing file 
    logFileTesting = open(os.path.join(logsDirsTesting,str(currentTimeStamp)+'.txt'),'w')

    # Training
    train = True

    modelNotChangingCount = 0


    while train:
        model = model.train()

        timeInstance.tic()
        for batch_no, data in tqdm(enumerate(trainDataLoader,0), total=len(trainDataLoader), smoothing=0.9):
            
            optimizermodel.zero_grad()
            # Expand the data into its respective components
            srcClrT, srcDepthT, targetTransformT, ptCldT , options = data
            #print(f'time taken to read the data is {timeInstance.toc()}')

            # Color Image - Cuda 0
            srcClrT = srcClrT.to(device)

            # Depth Image - Cuda 0
            srcDepthT = srcDepthT.to(device)
             
            predTransform  = model(srcClrT, srcDepthT)

            # Move the loss Function to Cuda 0    
            #timeInstance.tic()      
            loss, manhattanDist = loss_function(predTransform, srcClrT, srcDepthT, ptCldT, targetTransformT, config.calibrationDir, 1, None )
            #print(f'time taken to calculate loss is {timeInstance.toc()}')

            loss.backward()
            optimizermodel.step()
            manhattanDistArray = np.append(manhattanDistArray,manhattanDist.to('cpu').detach().numpy()) 

        global_step += 1

        
        logFileTraining.write('Global Epoch: '+str(global_epoch)+'\n')
        logFileTraining.write('Mean Manhattan Distance: '+str(manhattanDistArray.mean())+'\n')
        
        print('Global Epoch: '+str(global_epoch))
        print("Mean Manhattan Distance for the epoch: "+str(manhattanDistArray.mean()))
        manhattanDistArray = np.delete(manhattanDistArray.reshape(manhattanDistArray.shape[0],1),0,1)


        with torch.no_grad():
            simpleDistanceSE3, errorInAngles, errorInTranslation, gtAngles, gtTranslation = test(model, testDataLoader)

            print("Calculated mean Errors:" +  str(np.mean(simpleDistanceSE3)))
            print("Mean Angular Error: "+str(np.mean(errorInAngles,axis=1)))
            print("Mean Translation Error: "+str(np.mean(errorInTranslation,axis=1)))

            logFileTesting.write('Global Epoch: '+str(global_epoch)+'\n')
            logFileTesting.write("Calculated mean SE3 Errors: "+  str(np.mean(simpleDistanceSE3)))
            logFileTesting.write("Mean Angular Error: "+str(np.mean(errorInAngles,axis=1)))
            logFileTesting.write("Mean Translation Error: "+str(np.mean(errorInTranslation,axis=1)))

            distanceErr = np.append(distanceErr, simpleDistanceSE3)
            angularErr = np.append(angularErr, errorInAngles,axis=1)
            translationErr = np.append(translationErr, errorInTranslation,axis=1)
            angularGT = np.append(angularGT, gtAngles,axis=1)
            translationGT = np.append(translationGT, gtTranslation)

            # Increment the model not bettering count
            modelNotChangingCount += 1

            
            if (np.mean(simpleDistanceSE3) <  simpleDistanceSE3Err):

                simpleDistanceSE3Err = np.mean(simpleDistanceSE3)
                saveModelParams(model, modelPath)
                modelNotChangingCount = 0

        
        #schedulerModel.step()
        schedulerModel.step(simpleDistanceSE3Err)
        checkPOintPath = os.path.join(checkpointDir,f'checkpoint_epoch_{global_epoch}')
        saveCheckPoint(model,optimizermodel,global_epoch, loss,schedulerModel,checkPOintPath)
        global_epoch += 1

        


        if global_epoch == config.training['epoch']:
            train = False

    np.save(os.path.join(logsDirsTesting,'DistanceErr.npy'), distanceErr)
    np.save(os.path.join(logsDirsTesting,'ErrorInAngles.npy'), angularErr)
    np.save(os.path.join(logsDirsTesting,'ErrorInTranslation.npy'), translationErr)
    np.save(os.path.join(logsDirsTesting,'GroundTruthAngles.npy'), angularGT)
    np.save(os.path.join(logsDirsTesting,'GroundTruthTranslation.npy'), translationGT)

    logFileTraining.close()
    logFileTesting.close()
       
            

if __name__ == "__main__":
    main()


