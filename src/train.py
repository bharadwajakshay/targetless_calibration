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


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'model'))

def getTranslationRot(prediction):
    xyz = xyz = prediction[0:3,:]
    rot = prediction[3:,:].flatten()
    return(xyz, rot)

def getRotMatFromQuat(quat):
    rObj = rot.from_quat(quat)
    return(rObj.as_matrix())

def getInvTransformMat(R,T):
    # https://math.stackexchange.com/questions/152462/inverse-of-transformation-matrix
    invR = R.transpose()
    invT = np.matmul(-invR,T)
    R_T = np.vstack((np.hstack((invR,invT)),[0, 0, 0, 1.]))
    return(R_T)

def extractEulerAngles(R):
    rObj = rot.from_matrix(R)
    euler = rObj.as_euler(seq='zyx', degrees=True).reshape(3,1)
    return(euler[0],euler[1],euler[2])


def extractRotnTranslation(transformation):
    R = transformation[0:3,0:3]
    T = transformation[0:3,3]
    return(extractEulerAngles(R))


def calculateEucledianDist(pred,target,inputTensor, targetTensor):
    pred = pred.squeeze(dim=0).cpu()
    target = target.squeeze(dim=0).cpu()
    inPutCldTensor = inputTensor.transpose(2,1).cpu()
    targetCldTensor = targetTensor.cpu()

    pred = pred.data.numpy()
    target = target.data.numpy()
    inPutCld = inPutCldTensor.data.numpy()
    targetCld = targetCldTensor.data.numpy()

    '''
    Im estimating the decalibration applied.
    So th etransofrmation is inverse of the decalibration
    = decalibration ^-1 
    '''

    [predT, predQuat] = getTranslationRot(pred)
    R = getRotMatFromQuat(predQuat)
    invRt = getInvTransformMat(R, predT)

    ones = np.ones(inPutCld.shape[1]).reshape(inPutCld.shape[1],1)
    paddedinPutCld = np.hstack((inPutCld[0,:,:], ones))
    transformedptCld = np.matmul(invRt, paddedinPutCld.T).T[:,:3]


    [targetT, targetQuat] = getTranslationRot(target)
    targetR = getRotMatFromQuat(targetQuat)
    targetRT = np.vstack((np.hstack((targetR,targetT)),[0, 0, 0, 1.]))

    ones = np.ones(targetCld.shape[1]).reshape(targetCld.shape[1],1)
    paddedTargetCld = np.hstack((targetCld[0,:,:], ones))
    transformedTargetCld = np.matmul(targetRT, paddedTargetCld.T).T[:,:3]

    # calculate the eucledean distance between the the transformed and target point cloud
    eucledeanDist = np.linalg.norm(transformedTargetCld[:] - transformedptCld[:],axis=1)
  
    return(np.average(eucledeanDist), invRt, targetRT)




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
    batch_size = 5
    epochs = 50
    learning_rate = 0.0001 # 10^-5
    decay_rate = 1e-4
    resnet = resnet18(pretrained=True)

    # Hyper Parameters 
    TRAIN_DATASET = dataLoader()
    TEST_DATASET = dataLoader('/mnt/291d3084-ca91-4f28-8f33-ed0b64be0a8c/akshay/targetless_calibration/2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data/agumenteddata/test_data.json')

    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=batch_size, shuffle=False, num_workers=0)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=1, shuffle=False, num_workers=0)
    #MODEL = importlib.import_module(pointcloudnet)

    # empty the CUDA memory
    torch.cuda.empty_cache()

    network_model = pointcloudnet.pointcloudnet(layers=[1, 1, 1, 1, 1, 1])
    regressor_model = regressor.regressor().to('cuda:2')
    loss_function = pointcloudnet.get_loss().to('cuda:1')

    

    optimizer = torch.optim.Adam(
        network_model.parameters(),
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
            inputPtTensor, imgTensor, transformTensor, targetTensor = data
            # Extract point clouds
            inputCld = inputPtTensor.data.numpy()
            targetCld = targetTensor.data.numpy()

            # Preprocessing the input cloud
            #inputCldPtsDropped = provider.random_point_dropout(inputCld)
            '''inputCldPtsDropped[:,:,0:3] = provider.random_scale_point_cloud(inputCldPtsDropped[:,:, 0:3])'''
            # Convert it back to tensor
            #inputPtsDroppedTensor = torch.Tensor(inputCldPtsDropped)

            # Move the data to cuda
            # inputPtsDroppedTensor = inputPtsDroppedTensor.cuda()
            inputPtTensor = inputPtTensor.transpose(2,1)
            inputPtTensor = inputPtTensor
            transformTensor = transformTensor

            optimizer.zero_grad()

            network_model = network_model.train()
            feature_map = network_model(inputPtTensor)
            imgTensor = imgTensor.transpose(3,1)
            img_featuremap = resnet(imgTensor)
            img_featuremap = img_featuremap.unsqueeze(dim=2)
            aggTensor = torch.cat([feature_map,img_featuremap.to('cuda:2')],dim=2)
            pred = regressor_model(aggTensor.transpose(2,1))
            loss = loss_function(pred.to('cuda:1'), transformTensor.to('cuda:1'), inputPtTensor, targetTensor)
            loss_function_vec = np.append(loss_function_vec,loss.cpu().data.numpy())
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


