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
from model import pointcloudnet
import concurrent.futures
from common.utilities import *


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'model'))

def test(model_pointnet, model_resnet, model_regressor, dataLoader):
    eucledian_dist = []
    predr = []
    predp = []
    predy = []
    targetr = []
    targetp = []
    targety = []

    for j, data in tqdm(enumerate(dataLoader,0), total=len(dataLoader)):
        inputPtTensor, imgTensor, transformTensor, targetTensor = data
        
        inputPtTensor = targetTensor.transpose(2,1)
        inputPtTensor = inputPtTensor

        poinrFeatureMap = model_pointnet(inputPtTensor)
        imgTensor = imgTensor.transpose(3,1)
        imgFeatureMap = model_resnet(imgTensor).unsqueeze(dim=2)
        aggregatedTensor = torch.cat([poinrFeatureMap,imgFeatureMap.to('cuda:2')],dim=2).transpose(2,1)
        pred = model_regressor(aggregatedTensor)
        [eucledeanDist, predTransform, targetTransform] = calculateEucledianDist(pred, transformTensor, inputPtTensor, targetTensor)
        eucledian_dist.append(eucledeanDist)
        [r,p,y] = extractRotnTranslation(predTransform)
        predr.append(r)
        predp.append(p)
        predy.append(y)
        [r,p,y] = extractRotnTranslation(targetTransform)
        targetr.append(r)
        targetp.append(p)
        targety.append(y)
        
    return(np.mean(eucledian_dist),np.mean(predr),np.mean(predp),np.mean(predy),np.mean(targetr),np.mean(targetp),np.mean(targety))


def main():

    # Default parameters 
    batch_size = 10
    epochs = 50
    learning_rate = 0.0001 # 10^-5
    decay_rate = 1e-4

    # Choose the RESNet network used to get features from the images 
    resnetClrImg = resnet50(pretrained=True)
    resnetDepthImg = resnet50(pretrained=True)
    resnetIntensityImg = resnet50(pretrained=True)

    # Hyper Parameters 
    TRAIN_DATASET = dataLoader('/mnt/291d3084-ca91-4f28-8f33-ed0b64be0a8c/akshay/kitti/processed/train/trainingdata.json')
    TEST_DATASET = dataLoader('/mnt/291d3084-ca91-4f28-8f33-ed0b64be0a8c/akshay/kitti/processed/test/testingdata.json')

    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=batch_size, shuffle=False, num_workers=0)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=1, shuffle=False, num_workers=0)
    #MODEL = importlib.import_module(pointcloudnet)

    # empty the CUDA memory
    torch.cuda.empty_cache()

    regressor_model = regressor.regressor()
    loss_function = pointcloudnet.get_loss()

    

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


    eulerdistances = np.empty(0)
    loss_function_vec = np.empty(0)
    pred_R = np.empty(0)
    pred_P = np.empty(0)
    pred_Y = np.empty(0)
    target_R = np.empty(0)
    target_P = np.empty(0)
    target_Y = np.empty(0)

    # Training 
    for epoch in range(start_epoch, epochs):
        scheduler.step()

        for batch_no, data in tqdm(enumerate(trainDataLoader,0), total=len(trainDataLoader), smoothing=0.9):
            
            # Expand the data into its respective components
            ptCldTensor, clrImgTensor, depthImgTensor, intensityImgTensor, transformTensor = data
            
            optimizer.zero_grad()
            resnetClrImg = resnetClrImg.eval()
            resnetDepthImg = resnetDepthImg.eval()
            resnetIntensityImg = resnetIntensityImg.eval()

            # Transpose the tensors such that the no of channels are the 2nd term
            clrImgTensor = clrImgTensor.transpose(3,1)
            depthImgTensor = depthImgTensor.transpose(3,1)
            intensityImgTensor = intensityImgTensor.transpose(3,1)

            featureMapClrImg = resnetClrImg(clrImgTensor)
            featureMapDepthImg = resnetDepthImg(depthImgTensor)
            featureMapIntensityImg = resnetIntensityImg(intensityImgTensor)

            aggClrDepthFeatureMap = torch.cat([featureMapDepthImg,featureMapClrImg],dim=1)
            aggClrIntensityFeatureMap = torch.cat([featureMapIntensityImg,featureMapClrImg],dim=1)

            aggClrDepthFeatureMap = aggClrDepthFeatureMap.unsqueeze(dim=2)
            [predDepthT, predDepthR]  = regressor_model(aggClrDepthFeatureMap.transpose(2,1))

            aggClrIntensityFeatureMap = aggClrIntensityFeatureMap.unsqueeze(dim=2)
            [predIntensityT, predIntensityR]  = regressor_model(aggClrIntensityFeatureMap.transpose(2,1))

            loss = loss_function([predDepthT, predDepthT], [predIntensityT, predIntensityR], ptCldTensor, transformTensor)
      
            loss.backward()
            optimizer.step()
            global_step += 1

        with torch.no_grad():
            eulerDist, predr, predp, predy, targetr, targetp, targety = test(network_model.eval(), resnet, regressor_model.eval(), testDataLoader)
            eulerdistances = np.append(eulerdistances,eulerDist)
            pred_R = np.append(pred_R, predr)
            pred_P = np.append(pred_P, predp)
            pred_Y = np.append(pred_Y, predy)
            target_R = np.append(target_R, targetr)
            target_P = np.append(target_P, targetp)
            target_Y = np.append(target_Y, targety)
            print("Calculated mean Euler Distance: "+str(eulerDist)+" and the loss: "+str(loss_function_vec[global_epoch])+" for Global Epoch: "+str(global_epoch))
            if(eulerDist<besteulerdistance):
                besteulerdistance = eulerDist

                # make sure you save the model as checkpoint
                print("saving the model")
                savepath = "/tmp/bestmodel_targetlesscalibration.pth"
                state = {
                    'epoch': global_epoch,
                    'bestEulerDist': besteulerdistance,
                    'model_state_dict_1':network_model.state_dict(),
                    'model_state_dict_2':regressor_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state,savepath)

        global_epoch += 1
    np.save('/tmp/eulerdistances.npy',eulerdistances)
    np.save('/tmp/loss.npy', loss_function_vec)
    np.save('/tmp/predicted_R.npy', pred_R)
    np.save('/tmp/predicted_T.npy', pred_P)
    np.save('/tmp/predicted_P.npy', pred_Y)
    np.save('/tmp/target_R.npy', target_R)
    np.save('/tmp/target_T.npy', target_P)
    np.save('/tmp/target_P.npy', target_Y)

    
    print("something")
        
            

if __name__ == "__main__":
    main()


