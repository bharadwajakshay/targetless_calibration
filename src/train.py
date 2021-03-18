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




BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'model'))
modelPath = '/home/akshay/targetless_calibration/src/model/trained/bestTargetCalibrationModel.pth'


os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2'

batch_size = 10


def test(colorImgModel, depthImgModel, intensityImgModel, maxPool, regressorModel, dataLoader):

    simpleDistanceSE3 = []

    for j, data in tqdm(enumerate(dataLoader,0), total=len(dataLoader)):

       # Expand the data into its respective components
        srcDepthT, targetDepthT, srcIntensityT, targetIntensityT, srcClrT, ptCldT, targetTransformT = data

        # Transpose the tensors such that the no of channels are the 2nd term
        srcClrT = srcClrT.to('cuda:0')
        srcDepthT = srcDepthT.to('cuda:1')
        srcIntensityT = srcIntensityT.to('cuda:1')

        # Cuda 0
        featureMapClrImg = colorImgModel(srcClrT)
        # Cuda 1
        maxPooledDepthImg = maxPool(srcDepthT)
        featureMapDepthImg = depthImgModel(maxPooledDepthImg)
        # Cuda 2
        maxPooledDepthImg = maxPool(srcIntensityT)
        featureMapIntensityImg = intensityImgModel(maxPooledDepthImg.to('cuda:2'))

        # Cuda 0
        aggClrDepthFeatureMap = torch.cat([featureMapDepthImg.unsqueeze(1).to('cuda:0'),featureMapIntensityImg.unsqueeze(1).to('cuda:0'),featureMapClrImg.unsqueeze(1)],dim=1)

        # Cuda 0
        predTransform  = regressorModel(aggClrDepthFeatureMap)

        # Simple SE3 distance
        predTransform = exponentialMap(predTransform).type(torch.float64)

        targetTransformT = moveToDevice(targetTransformT, predTransform.get_device())

        predRot = predTransform[:,:3,:3]
        predT = predTransform[:,:3,3]

        gtRot = targetTransformT[:,:3,:3]
        gtT = targetTransformT[:,:3,3]

        simpleDistanceSE3.append(torch.norm(torch.matmul(calculateInvRTTensor(predTransform), targetTransformT) - moveToDevice(torch.eye(4), predTransform.get_device()),'fro'))
        #simpleDistanceSE3.append(torch.norm(torch.matmul(calculateInvRTTensor(targetTransformT), targetTransformT) - moveToDevice(torch.eye(4), predTransform.get_device()),'fro'))
            
        
    return(max(simpleDistanceSE3))


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
    # define max pooling layer
    maxPool = torch.nn.MaxPool2d(5, stride=1).to('cuda:1')
    resnetIntensityImg = resnet50(pretrained=True).to('cuda:2')

    # Get the max point cloud size
    file = open('maxPtCldSize.txt','r')
    maxPtCldSize = int(file.read())
    file.close()

    # Hyper Parameters 
    TRAIN_DATASET = dataLoader('/home/akshay/targetless_calibration/parsed_set.txt', maxPtCldSize,mode='train')
    TEST_DATASET = dataLoader('/home/akshay/targetless_calibration/parsed_set.txt', maxPtCldSize,mode='test')

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
    # Error 
    simpleDistanceSE3Err = 10000
    distanceErr = np.empty(0)



    # Check if there are existing models, If so, Load them
    # Check if the file exists
    if os.path.isfile(modelPath):
        try:
            model_weights = torch.load(modelPath)
            regressor_model.load_state_dict(model_weights['modelStateDict'])
        except:
            print("Failed to load the model. Continuting without loading weights")


    # Training 
    for epoch in range(start_epoch, epochs):
        scheduler.step()

        for batch_no, data in tqdm(enumerate(trainDataLoader,0), total=len(trainDataLoader), smoothing=0.9):
        
            # Expand the data into its respective components
            srcDepthT, targetDepthT, srcIntensityT, targetIntensityT, srcClrT, ptCldT, targetTransformT = data
            
            optimizer.zero_grad()
            resnetClrImg = resnetClrImg.eval()
            maxPool = maxPool.train()
            resnetDepthImg = resnetDepthImg.train()
            resnetIntensityImg = resnetIntensityImg.train()
            regressor_model = regressor_model.train()


            # Transpose the tensors such that the no of channels are the 2nd term
            srcClrT = srcClrT.to('cuda:0')
            srcDepthT = srcDepthT.to('cuda:1')
            srcIntensityT = srcIntensityT.to('cuda:1')

            # Cuda 0
            featureMapClrImg = resnetClrImg(srcClrT)
            # Cuda 1
            maxPooledDepthImg = maxPool(srcDepthT)
            featureMapDepthImg = resnetDepthImg(maxPooledDepthImg)
            # Cuda 2
            maxPooledDepthImg = maxPool(srcIntensityT)
            featureMapIntensityImg = resnetIntensityImg(maxPooledDepthImg.to('cuda:2'))

            # Cuda 0
            aggClrDepthFeatureMap = torch.cat([featureMapDepthImg.unsqueeze(1).to('cuda:0'),featureMapIntensityImg.unsqueeze(1).to('cuda:0'),featureMapClrImg.unsqueeze(1)],dim=1)

            # Cuda 0
            predTransform  = regressor_model(aggClrDepthFeatureMap)

            # Cuda 1
            loss = loss_function(predTransform, srcDepthT, targetDepthT, srcIntensityT, targetIntensityT, ptCldT, targetTransformT, 1)
                  
            loss.backward()
            optimizer.step()
            global_step += 1

        with torch.no_grad():
            simpleDistanceSE3 = test(resnetClrImg.eval(), resnetDepthImg.eval(), resnetIntensityImg.eval(), maxPool.eval(), regressor_model.eval(), testDataLoader)

            print("Calculated mean Errors:" +  str(simpleDistanceSE3))

            distanceErr = np.append(distanceErr,simpleDistanceSE3.to('cpu').numpy())

            if (simpleDistanceSE3 <  simpleDistanceSE3Err ):

                simpleDistanceSE3Err = simpleDistanceSE3
                saveModelParams(regressor_model, optimizer, global_epoch, modelPath)


        global_epoch += 1



    np.save('/tmp/DistanceErr.npy', distanceErr)

    
    print("something")
        
            

if __name__ == "__main__":
    main()


