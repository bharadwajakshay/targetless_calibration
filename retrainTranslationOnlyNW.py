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
from common.tensorTools import saveModelParams

from torchvision import transforms
from model.transformsTensor import *
from common.tensorTools import calculateInvRTTensor, exponentialMap, moveToDevice

import config
from torchsummary import summary
from pytictoc import TicToc

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'model'))
modelPath = '/home/akshay/targetless_calibration/src/model/trained/bestTargetCalibrationModel.pth'


os.environ["CUDA_VISIBLE_DEVICES"] = '1'

_Debug = False


def test(colorImgModel, depthImgModel, regressorModel, maxPool, dataLoader):

    simpleDistanceSE3 = np.empty(0)
    # Check if log directories are available
    if not os.path.exists(config.logsDirsTesting):
        os.makedirs(config.logsDirsTesting)

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
        
        I = moveToDevice(torch.eye(4,dtype=torch.float32), predTransform.get_device())
        simpleDistanceSE3 = np.append(simpleDistanceSE3, torch.norm( RtR - I,'fro').to('cpu').numpy())

        # Caluclate the euler angles from rotation matrix
        predEulerAngles = matrix_to_euler_angles(predRot, "ZXY")
        targetEulerAngles = matrix_to_euler_angles(gtRot, "ZXY")
        errorEulerAngle = torch.abs(targetEulerAngles - predEulerAngles)
        errorEulerAngle = torch.rad2deg(torch.mean(errorEulerAngle,dim=0))
        errorTranslation = torch.abs(gtT - predT)
        errorTranslation = torch.mean(errorTranslation,dim=0)

        """
        print(errorEulerAngle)
        print(errorTranslation)
        
        print("Breakpoint")
        """    
        
    return(np.mean(simpleDistanceSE3), errorEulerAngle.to('cpu').numpy(), errorTranslation.to('cpu').numpy())


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
    loss_function = get_loss().to('cuda')

    # define max pooling layer
    maxPool = torch.nn.MaxPool2d(5, stride=1)
    #resnetIntensityImg = resnet50(pretrained=True)

    # Get the max point cloud size
    file = open(config.maxPtCldSizeFile,'r')
    maxPtCldSize = int(file.read())
    file.close()

    # Hyper Parameters 
    TRAIN_DATASET = dataLoader(config.trainingDataFile, maxPtCldSize,mode='train')
    TEST_DATASET = dataLoader(config.trainingDataFile, maxPtCldSize,mode='test')

    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=config.training['batchSize'], shuffle=True, num_workers=0)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=config.training['batchSize'], shuffle=True, num_workers=0)
    #MODEL = importlib.import_module(pointcloudnet)

    
    # Check if log directories are available
    if not os.path.exists(config.logsDirsTraining):
        os.makedirs(config.logsDirsTraining)

    

    optimizerRegression = torch.optim.Adam(
        regressor_model.parameters(),
        lr = config.training['learningRate'],
        betas = (config.training['beta0'], config.training['beta1']),
        eps = config.training['eps'],
        weight_decay = config.training['decayRate'],
    )

    optimizerResNET = torch.torch.optim.Adam(
        resnetDepthImg.parameters(),
        lr = config.training['learningRate'],
        betas = (config.training['beta0'], config.training['beta1']),
        eps = config.training['eps'],
        weight_decay = config.training['decayRate'],
    )


    schedulerRegressor = torch.optim.lr_scheduler.MultiStepLR(optimizerRegression, milestones=[19,24], gamma=0.1)
    schedulerResNet = torch.optim.lr_scheduler.MultiStepLR(optimizerResNET, milestones=[19,24], gamma=0.1)

    start_epoch = 0
    global_epoch = 0
    global_step = 0
    besteulerdistance = 100
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

    # Training 
    for epoch in range(start_epoch, epochs):

        timeInstance.tic()
        for batch_no, data in tqdm(enumerate(trainDataLoader,0), total=len(trainDataLoader), smoothing=0.9):

        
            # Expand the data into its respective components
            srcClrT, srcDepthT, ptCldT, ptCldSize, targetTransformT, options = data
            
            optimizerResNET.zero_grad()
            optimizerRegression.zero_grad()
            resnetClrImg = resnetClrImg.eval()
            maxPool = maxPool.eval()
            resnetDepthImg = resnetDepthImg.eval()
            regressor_model = regressor_model.train()

            # Transpose the tensors such that the no of channels are the 2nd term

            # Color Image - Cuda 0
            srcClrT = srcClrT.to('cuda')
            featureMapClrImg = resnetClrImg(srcClrT)


            # Depth Image - Cuda 0
            srcDepthT = srcDepthT.to('cuda')
            maxPool = maxPool.to('cuda')
            maxPooledDepthImg = maxPool(srcDepthT)
            featureMapDepthImg = resnetDepthImg(maxPooledDepthImg)
 


            # Concatinate the feature Maps # Still in Cuda 0
            aggClrDepthFeatureMap = torch.cat([featureMapDepthImg,featureMapClrImg],dim=1)

            # Move the regressor model to Cuda 0 and pass the concatinated Feature Vector
            predTransform  = regressor_model(aggClrDepthFeatureMap,True)
            # Move the regressor back to CPU
            #regressor_model = regressor_model.to('cpu')

            # Move the loss Function to Cuda 0          
            loss, manhattanDist = loss_function(predTransform, srcClrT, srcDepthT, ptCldT, ptCldSize, targetTransformT, config.calibrationDir, 1)
            # Move the model back to CPU
            #loss_function = loss_function.to('cpu')

                  
            loss.backward()

            manhattanDistArray = np.append(manhattanDistArray,manhattanDist.to('cpu').detach().numpy()) 

            # Debug
            if _Debug:
                summary(resnetClrImg)
                summary(resnetDepthImg)
                summary(regressor_model)   


                f = open("prestep.txt",'w')
                for name, param in regressor_model.named_parameters():
                    if param.requires_grad:
                        f.write(name)
                        f.write(str(param.data.to('cpu').numpy()))
                        f.write('\n')
                f.close()

            optimizerRegression.step()
            optimizerResNET.step()

            # Debug
            if _Debug:
                f = open("postStep.txt",'w')
                for name, param in regressor_model.named_parameters():
                    if param.requires_grad:
                        f.write(name)
                        f.write(str(param.data.to('cpu').numpy()))
                        f.write('\n')
                f.close()

            global_step += 1

        print("Mean Manhattan Distance for the epoch: "+str(manhattanDistArray.mean()))
        manhattanDistArray = np.delete(manhattanDistArray.reshape(manhattanDistArray.shape[0],1),0,1)


        with torch.no_grad():
            simpleDistanceSE3, errorInAngles, errorInTranslation = test(resnetClrImg, resnetDepthImg, regressor_model, maxPool, testDataLoader)

            print("Calculated mean Errors:" +  str(simpleDistanceSE3))
            print("Mean Angular Error: "+str(errorInAngles))
            print("Mean Translation Error: "+str(errorInTranslation))

            distanceErr = np.append(distanceErr,simpleDistanceSE3)
            angularErr = np.append(angularErr, errorInAngles)
            translationErr = np.append(translationErr, errorInTranslation)

            if (simpleDistanceSE3 <  simpleDistanceSE3Err):

                simpleDistanceSE3Err = simpleDistanceSE3
                saveModelParams(resnetDepthImg, regressor_model, optimizerResNET, optimizerRegression, global_epoch, modelPath)

        
        schedulerRegressor.step()
        schedulerResNet.step()
        global_epoch += 1


    np.save(os.path.join(config.logsDirs,'DistanceErr.npy'), distanceErr)
    np.save(os.path.join(config.logsDirs,'ErrorInAngles.npy'), errorInAngles)
    np.save(os.path.join(config.logsDirs,'ErrorInTranslation.npy'), errorInTranslation)

    
    print("something")
        
            

if __name__ == "__main__":
    main()


