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



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'model'))
modelPath = '/home/akshay/targetless_calibration/src/model/trained/bestTargetCalibrationModel.pth'


os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2'

batch_size = 10


def test(colorImgModel, depthImgModel, IntensiyImgModel, regressorModel, dataLoader):

    x = np.empty((0,1))
    y = np.empty((0,1))
    z = np.empty((0,1))
    roll = np.empty((0,1))
    pitch = np.empty((0,1))
    yaw = np.empty((0,1))

    for j, data in tqdm(enumerate(dataLoader,0), total=len(dataLoader)):

        ptCldTensor, clrImgTensor, grayImgTensor, depthImgTensor, __, targetTransform = data
        grayImgTensor = torch.unsqueeze(grayImgTensor,3)

        # Transpose the tensors such that the no of channels are the 2nd term
        clrImgTensor = clrImgTensor.transpose(3,1).to('cuda:0')
        depthImgTensor = depthImgTensor.transpose(3,1).to('cuda:1')
        #intensityImgTensor = intensityImgTensor.transpose(3,1).to('cuda:2')

        # Cuda 0
        featureMapClrImg = colorImgModel(clrImgTensor)
        # Cuda 1
        featureMapDepthImg = depthImgModel(depthImgTensor)
        # Cuda 2
        #featureMapIntensityImg = IntensiyImgModel(intensityImgTensor)

        # Cuda 0
        aggClrDepthFeatureMap = torch.cat([featureMapDepthImg.to('cuda:0'),featureMapClrImg],dim=1)

        # Cuda 0
        #aggClrIntensityFeatureMap = torch.cat([featureMapIntensityImg.to('cuda:0'),featureMapClrImg],dim=1)

        aggClrDepthFeatureMap = aggClrDepthFeatureMap.unsqueeze(dim=2)
        [predDepthT, predDepthR]  = regressorModel(aggClrDepthFeatureMap.transpose(2,1))
        
        """
        aggClrIntensityFeatureMap = aggClrIntensityFeatureMap.unsqueeze(dim=2)
        [predIntensityT, predIntensityR]  = regressorModel(aggClrIntensityFeatureMap.transpose(2,1))
        """

        # Calculate the diff between predicted transform and target transform 
        targetT = targetTransform[:,:3,3].unsqueeze(1)
        targetR = matrix_to_euler_angles(targetTransform[:,:3,:3],"ZXY").unsqueeze(1)

        diffDepthT = (predDepthT.to('cpu') - targetT).numpy()
        diffDepthR = torch.rad2deg((predDepthR.to('cpu')  - targetR)).numpy()

        """
        diffIntensityT = (predIntensityT.to('cpu')  - targetT).numpy()
        diffIntensityR = (predIntensityR.to('cpu')  - targetR).numpy()
        """
        
        if(x.size == 0):
            x = np.append(x,diffDepthT[:,0,0].flatten())
            #x = np.concatenate((x,diffIntensityT[:,0,0].flatten()))
            y = np.append(y,diffDepthT[:,0,1].flatten())
            #y = np.concatenate((y,diffIntensityT[:,0,1].flatten()))
            z = np.append(z,diffDepthT[:,0,2].flatten())
            #z = np.concatenate((z,diffIntensityT[:,0,2].flatten()))

            roll = np.append(roll,diffDepthR[:,0,0].flatten())
            #roll = np.concatenate((roll,diffIntensityT[:,0,0].flatten()))
            pitch = np.append(pitch,diffDepthR[:,0,1].flatten())
            #pitch = np.concatenate((pitch,diffIntensityT[:,0,1].flatten()))
            yaw = np.append(yaw,diffDepthR[:,0,2].flatten())
            #yaw = np.concatenate((yaw,diffIntensityT[:,0,2].flatten()))


        else:
            x = np.concatenate((x,diffDepthT[:,0,0].flatten()))
            #x = np.concatenate((x,diffIntensityT[:,0,0].flatten()))
            y = np.concatenate((y,diffDepthT[:,0,1].flatten()))
            #y = np.concatenate((y,diffIntensityT[:,0,1].flatten()))
            z = np.concatenate((z,diffDepthT[:,0,1].flatten()))
            #z = np.concatenate((z,diffIntensityT[:,0,2].flatten()))

            roll = np.concatenate((roll,diffDepthR[:,0,0].flatten()))
            #roll = np.concatenate((roll,diffIntensityT[:,0,0].flatten()))
            pitch = np.concatenate((pitch,diffDepthR[:,0,0].flatten()))
            #pitch = np.concatenate((pitch,diffIntensityT[:,0,1].flatten()))
            yaw = np.concatenate((yaw,diffDepthR[:,0,0].flatten()))
            #yaw = np.concatenate((yaw,diffIntensityT[:,0,2].flatten()))
            
        
    return([np.mean(x), np.std(x)], [np.mean(y), np.std(y)], [np.mean(z), np.std(z)],
            [np.mean(roll), np.std(roll)], [np.mean(pitch), np.std(pitch)], [np.mean(yaw), np.std(yaw)])


def main():

    # Default parameters 
    epochs = 30
    learning_rate = 0.0001 # 10^-5
    decay_rate = 1e-4


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



    # Choose the RESNet network used to get features from the images 
    resnetClrImg = resnet50(pretrained=True).to('cuda:0')
    resnetDepthImg = resnet50(pretrained=False).to('cuda:1')
    #resnetIntensityImg = resnet50(pretrained=True).to('cuda:2')

    # Hyper Parameters 
    TRAIN_DATASET = dataLoader('/mnt/291d3084-ca91-4f28-8f33-ed0b64be0a8c/akshay/kitti/processed/train/trainingdata.json')
    TEST_DATASET = dataLoader('/mnt/291d3084-ca91-4f28-8f33-ed0b64be0a8c/akshay/kitti/processed/test/testingdata.json')

    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=batch_size, shuffle=False, num_workers=0)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=batch_size, shuffle=False, num_workers=0)
    #MODEL = importlib.import_module(pointcloudnet)

    # empty the CUDA memory
    torch.cuda.empty_cache()

    regressor_model = regressor.regressor().to('cuda:0')
    loss_function = get_loss().to('cuda:1')

    

    optimizer = torch.optim.Adam(
        regressor_model.parameters(),
        lr = learning_rate,
        betas = (0.9, 0.999),
        eps = 1e-08,
        weight_decay = decay_rate
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    start_epoch = 0
    global_epoch = 0
    global_step = 0
    besteulerdistance = 100
    # Error [mean, variance, std]
    xErr   = 100
    yErr   = 100
    zErr   = 100
    rErr   = 100
    pErr   = 100
    yawErr = 100

    savedModel = False


    eulerdistances = np.empty(0)
    loss_function_vec = np.empty(0)
    errorX = np.empty(0)
    errorY = np.empty(0)
    errorZ = np.empty(0)
    errorR = np.empty(0)
    errorP = np.empty(0)
    errorYa = np.empty(0)

    # Check if there are existing models, If so, Load them
    # Check if the file exists
    if os.path.isfile(modelPath):
        model_weights = torch.load(modelPath)
        regressor_model.load_state_dict(model_weights['modelStateDict'])

    # Training 
    for epoch in range(start_epoch, epochs):
        scheduler.step()

        for batch_no, data in tqdm(enumerate(trainDataLoader,0), total=len(trainDataLoader), smoothing=0.9):
        
            # Expand the data into its respective components
            ptCldTensor, clrImgTensor, grayImgTensor, depthImgTensor, __, transformTensor = data

            grayImgTensor = torch.unsqueeze(grayImgTensor,3)
            
            optimizer.zero_grad()
            resnetClrImg = resnetClrImg.eval()
            resnetDepthImg = resnetDepthImg.train()
            #resnetIntensityImg = resnetIntensityImg.eval()
            regressor_model = regressor_model.train()


            # Transpose the tensors such that the no of channels are the 2nd term
            clrImgTensor = clrImgTensor.transpose(3,1).to('cuda:0')
            depthImgTensor = depthImgTensor.transpose(3,1).to('cuda:1')
            #intensityImgTensor = intensityImgTensor.transpose(3,1).to('cuda:2')

            # Cuda 0
            featureMapClrImg = resnetClrImg(clrImgTensor)
            # Cuda 1
            featureMapDepthImg = resnetDepthImg(depthImgTensor)
            # Cuda 2
            #featureMapIntensityImg = resnetIntensityImg(intensityImgTensor)

            # Cuda 0
            aggClrDepthFeatureMap = torch.cat([featureMapDepthImg.to('cuda:0'),featureMapClrImg],dim=1)

            # Cuda 0
            #aggClrIntensityFeatureMap = torch.cat([featureMapIntensityImg.to('cuda:0'),featureMapClrImg],dim=1)

            aggClrDepthFeatureMap = aggClrDepthFeatureMap.unsqueeze(dim=2)
            [predDepthT, predDepthR]  = regressor_model(aggClrDepthFeatureMap.transpose(2,1))

            """
            aggClrIntensityFeatureMap = aggClrIntensityFeatureMap.unsqueeze(dim=2)
            [predIntensityT, predIntensityR]  = regressor_model(aggClrIntensityFeatureMap.transpose(2,1))
            """

            # Cuda 1
            loss = loss_function([predDepthT, predDepthT], [__, __], ptCldTensor, grayImgTensor, transformTensor, 1)
      
            loss.backward()
            optimizer.step()
            global_step += 1

        with torch.no_grad():
            meanXError, meanYError, meanZError, meanRollError, meanPitchError, meanYawError = test(resnetClrImg.eval(), resnetDepthImg.eval(), __, regressor_model.eval(), testDataLoader)

            errorX = np.append(errorX, meanXError[0])
            errorY = np.append(errorY, meanYError[0])
            errorZ = np.append(errorZ, meanZError[0])
            errorR = np.append(errorR, meanRollError[0])
            errorP = np.append(errorP, meanPitchError[0])
            errorYa = np.append(errorYa, meanYawError[0])

            print("Calculated mean Errors: X = "+str(meanXError[0])+" Y = "+str(meanYError[0])+" Z = "+str(meanZError[0])+" Roll = "+str(meanRollError[0])+" Pitch = "+str(meanPitchError[0])+" Yaw = "+str(meanYawError[0]))
            if (xErr>meanXError[0]) and (not savedModel):
                savedModel = True
                xErr = meanXError[0]
                saveModelParams(regressor_model, optimizer, global_epoch, modelPath)

            if (yErr>meanYError[0]) and (not savedModel):
                savedModel = True
                yErr = meanYError[0]
                saveModelParams(regressor_model, optimizer, global_epoch, modelPath)

            if (zErr>meanZError[0]) and (not savedModel):
                savedModel = True
                zErr = meanZError[0]
                saveModelParams(regressor_model, optimizer, global_epoch, modelPath)

            if (rErr>meanRollError[0]) and (not savedModel):
                savedModel = True
                rErr = meanRollError[0]
                saveModelParams(regressor_model, optimizer, global_epoch, modelPath)

            if (pErr>meanPitchError[0]) and (not savedModel):
                savedModel = True
                pErr = meanPitchError[0]
                saveModelParams(regressor_model, optimizer, global_epoch, modelPath)

            if (yawErr>meanYawError[0]) and (not savedModel):
                savedModel = True
                yawErr = meanYawError[0]
                saveModelParams(regressor_model, optimizer, global_epoch, modelPath)


        global_epoch += 1
        savedModel = False


    np.save('/tmp/predicted_R.npy', errorX)
    np.save('/tmp/predicted_T.npy', errorY)
    np.save('/tmp/predicted_P.npy', errorZ)
    np.save('/tmp/target_R.npy', errorR)
    np.save('/tmp/target_T.npy', errorP)
    np.save('/tmp/target_P.npy', errorYa)

    
    print("something")
        
            

if __name__ == "__main__":
    main()


