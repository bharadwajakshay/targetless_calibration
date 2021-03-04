import torch
from common.pytorch3D import *
import numpy as np
from data_prep.helperfunctions import *

def invTransformation():
    return(invRT)

def createRTMatTensor(R, T):
    # The input Tensor is composed of two components R, T
    # T = inputTensor[0]
    # R = inputTensor[1]
    """
    # Consider R[0] = Alpha
    #          R[1] = Beta
    #          R[2] = Gamma

    cosAlpha = torch.cos(R[:,:,0])
    sinAlpha = torch.sin(R[:,:,0])
    cosBeta = torch.cos(R[:,:,1])
    sinBeta = torch.sin(R[:,:,1])
    cosGamma = torch.cos(R[:,:,2])
    sinGamma = torch.sin(R[:,:,2])

    zeroTensor = torch.zeros(cosAlpha.shape)
    oneTensor  = torch.ones(cosAlpha.shape)

    #   |  cosG  sinG  0 |
    #   | -sinG  cosG  0 |
    #   |   0     0    1 |

    rZ0 = torch.unsqueeze(torch.hstack([cosGamma[:,], sinGamma[:,], zeroTensor[:,]]),dim=1)
    rZ1 = torch.unsqueeze(torch.hstack([torch.neg(sinGamma[:,]), cosGamma[:,], zeroTensor[:,]]),dim=1)
    rZ2 = torch.unsqueeze(torch.hstack([zeroTensor[:,], zeroTensor[:,], oneTensor[:,]]),dim=1)
    rZ = torch.cat((rZ0,rZ1,rZ2),dim=1)

    #   |  cosB   0  -sinB |
    #   |   0     1    0   |
    #   |  sinB   0   cosB |

    rY0 = torch.unsqueeze(torch.hstack([cosBeta[:,], zeroTensor[:,], torch.neg(sinBeta[:,]) ]),dim=1)
    rY1 = torch.unsqueeze(torch.hstack([zeroTensor[:,], oneTensor[:,], zeroTensor[:,]]),dim=1)
    rY2 = torch.unsqueeze(torch.hstack([sinBeta[:,], zeroTensor[:,], cosBeta[:,]]),dim=1)
    rY  = torch.cat((rY0, rY1, rY2),dim=1) 


    #   |  1    0     0   |
    #   |  0   cosA  sinA |
    #   |  0  -sinA  cosA |

    rX0 = torch.unsqueeze(torch.hstack([oneTensor[:,], zeroTensor[:,], zeroTensor[:,]]),dim=1)
    rX1 = torch.unsqueeze(torch.hstack([zeroTensor[:,], cosAlpha[:,], sinAlpha[:,]]),dim=1)
    rX2 = torch.unsqueeze(torch.hstack([zeroTensor[:,], torch.neg(sinAlpha[:,]), cosAlpha[:,]]),dim=1)
    rX = torch.cat((rX0, rX1, rX2),dim=1)

    # Multiply  ZXY to get Rotation Matrix 
    rXY = torch.matmul(rX, rY)
    rZXY = torch.matmul(rZ, rXY)
    """
    rZXY = euler_angles_to_matrix(R, "ZXY")
    rZXY = torch.squeeze(rZXY,dim = 1)

    T = torch.transpose(T,2,1)
    RT = torch.cat((rZXY, T),dim=2)

    # Zeros vector 
    zeros = torch.zeros((RT.shape[0],1, RT.shape[2])) 
    ones = torch.ones((RT.shape[0],1))
    zeros[:,:,3] = ones[:] 


    # Move the vector zeros to device in which RT is present
    RT = torch.cat((RT,zeros.to('cuda:'+str(RT.get_device()))),dim=1)

    return(RT)


def calculateInvRTTensor(RT):

    # Extract the rotation matrix from 4x4 matrix 
    R = RT[:,:3,:3]

    # Extract the translation matrix from 4x4 matrix
    T = RT[:,:3,3]
    T = torch.unsqueeze(T,dim=2)

    # Calculate the inverse of the matrix 
    invR = torch.inverse(R)
    invT = torch.negative(torch.matmul(invR,T))
    invRT = torch.cat((invR,invT),dim=2)

    zeros = torch.zeros((RT.shape[0],1, RT.shape[2])) 
    ones = torch.ones((RT.shape[0],1))
    zeros[:,:,3] = ones[:]

    invRT = torch.cat((invRT,zeros.to('cuda:'+str(invRT.get_device()))),dim=1)

    return(invRT)

def getImageTensorFrmPtCloud(projectionMat, ptCld):

    projPts = torch.matmul(projectionMat.to('cuda:'+str(ptCld.get_device())), ptCld)

    projPts[:,0,:] = torch.div(projPts[:,0,:], projPts[:,2,:])
    projPts[:,1,:] = torch.div(projPts[:,1,:], projPts[:,2,:])

    return(projPts)

def filterPtClds(ptCld2D, ptCld3D, imgHeight, imgWidth):
    
    mask = (ptCld2D[:,:,0] < imgWidth) & (ptCld2D[:,:,0] >= 0) &\
           (ptCld2D[:,:,1] < imgHeight) & (ptCld2D[:,:,1] >= 0) &\
           (ptCld3D[:,:,2] > 0)


    """
    Can this be replaced by by select index
    """


    maxSize2D = 0
    maxSize3D = 0
    for eachChannel in range(0,mask.shape[0]):
        tempTensor = ptCld2D[eachChannel, mask[eachChannel,:]]
        tempTensor3D = ptCld3D[eachChannel, mask[eachChannel,:]]
        if tempTensor.shape[0] > maxSize2D:
            maxSize2D = tempTensor.shape[0]

        if tempTensor3D.shape[0] > maxSize3D:
            maxSize3D = tempTensor3D.shape[0]


    tensorImgCoord = torch.empty((ptCld2D.shape[0],maxSize2D,ptCld2D.shape[2]))
    tensor3DCoord = torch.empty((ptCld3D.shape[0],maxSize3D,ptCld3D.shape[2]))

    for eachChannel in range(0,mask.shape[0]):
        tempTensor2D = ptCld2D[eachChannel, mask[eachChannel,:]]
        tempTensor3D = ptCld3D[eachChannel, mask[eachChannel,:]]

        if tempTensor2D.shape[0] < maxSize2D:
            zeroTensor = torch.zeros((maxSize2D - tempTensor2D.shape[0], tempTensor2D.shape[1])).to('cuda:'+str(tempTensor2D.get_device()))
            tempTensor2D = torch.vstack((tempTensor2D,zeroTensor))

        if tempTensor3D.shape[0] < maxSize3D:
            zeroTensor = torch.zeros((maxSize3D - tempTensor3D.shape[0], tempTensor3D.shape[1])).to('cuda:'+str(tempTensor3D.get_device()))
            tempTensor3D = torch.vstack((tempTensor3D,zeroTensor))
        
        tensorImgCoord[eachChannel,:,:] = tempTensor2D
        tensor3DCoord[eachChannel,:,:] = tempTensor3D

    return(tensorImgCoord, tensor3DCoord)

    

def createImage(ptCld2D,addChannel, imgWidth, imgHeight):

    # no of channel
    noChannel = 1
    addChannelMax = torch.max(addChannel[:])
    addChannelMin = torch.zeros(addChannelMax.shape)

    maxValue = torch.ones(addChannelMax.shape)*255
    minValue = torch.zeros(addChannelMax.shape)

    # filter points
    addChannel = (((addChannel[:] - addChannelMin)/(addChannelMax-addChannelMin))*(maxValue-minValue)) + minValue
    addChannel = torch.unsqueeze(addChannel,dim=2)

    # Convert points to int 
    ptCld2D = ptCld2D.type(torch.int)

    # Convert points back to long
    ptCld2D = ptCld2D.type(torch.long)


    # create a new image
    newImg = torch.zeros((ptCld2D.shape[0], imgHeight, imgWidth, noChannel))

    # For each channel of batch size
    #for batchId in range(newImg.shape[0]):
    newImg[:, ptCld2D[:,:,1], ptCld2D[:,:,0]] =255 - addChannel[:,:]

    return(newImg)


def applySobelOperator(inputTensor):
    """
        |1 0 -1|    | 1  2  1| 
        |2 0 -2|    | 0  0  0|
        |1 0 -1|    |-1 -2 -1|
    """

    device = inputTensor.get_device()

    sobel0 = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    sobel90 = np.array([[1, 0 ,-1],[2,0,-2],[1,0,-1]])

    outChannel = 1
    inChannel = 1

    inputTensor = torch.transpose(inputTensor,3,1)
    channel = inputTensor.shape[1]
    sobel0Tensor = torch.from_numpy(sobel0).float().unsqueeze(0).unsqueeze(0)
    sobel90Tensor = torch.from_numpy(sobel90).float().unsqueeze(0).unsqueeze(0)

    sobel0ImgConv = torch.nn.Conv2d(inChannel, outChannel, kernel_size=3, stride=1, padding=1, bias=False)

    if device >=0:
        sobel0Tensor = sobel0Tensor.to('cuda:'+str(device))
        sobel90Tensor = sobel90Tensor.to('cuda:'+str(device))
        sobel0ImgConv = sobel0ImgConv.to('cuda:'+str(device))

    
    sobel0ImgConv.weight = torch.nn.Parameter(sobel0Tensor)
    sobel0Img = sobel0ImgConv(inputTensor)

    sobel90ImgConv = torch.nn.Conv2d(inChannel, outChannel, kernel_size=3, stride=1, padding=1, bias=False)
    sobel90ImgConv.weight = torch.nn.Parameter(sobel90Tensor)
    sobel90Img = sobel90ImgConv(inputTensor)
    
    edgeImg = torch.sqrt(torch.pow(sobel0Img,2)+torch.pow(sobel90Img,2))


    return(edgeImg)

def performCrossCorrelation(imgTensor0, imgTensor1):
    print()


def calculateEucledianDistTensor(tensor1, tensor2):
    euclideanDistance =  torch.empty(tensor1.shape[0],tensor1.shape[1])
    for channel in range(0,tensor1.shape[0]):
        euclideanDistance[channel] = torch.dist(tensor1[channel,0,:], tensor2[channel,0,:], p=2)

    return(euclideanDistance)

def calculateEucledianDistOfPointClouds(PtCld0, PtCld1):

    # Claculate the mean euclidean distance between each point point cloud
    meanEucledianDist = torch.empty(PtCld0.shape[0],1)
    for channel in range(0,PtCld0.shape[0]):
        D = torch.sqrt(torch.pow(PtCld1[channel,:,0] - PtCld0[channel,:,0],2) 
            + torch.pow(PtCld1[channel,:,1] - PtCld0[channel,:,1],2) 
            + torch.pow(PtCld1[channel,:,2] - PtCld0[channel,:,2],2))

        meanEucledianDist[channel,:] = torch.mean(D)
   

    return(meanEucledianDist)

def getGroundTruthPointCloud(ptCloud, P_rect, R_rect, RT):

    # Correct the point cloud 
    # Detach the intensities and attach the unit coloumn 
    intensity = ptCloud[:,:,3]
    ptCloud = ptCloud[:,:,:3]
    ones = torch.ones((ptCloud.shape[0],ptCloud.shape[1],1)).to('cuda:'+str(ptCloud.get_device()))

    ptCloud = torch.cat([ptCloud,ones],dim=2)
    ptCloud = torch.transpose(ptCloud, 2,1)
        
    # Corecting for RT
    ptCloud = torch.matmul(RT.to('cuda:'+str(ptCloud.get_device())),ptCloud[:])

    # Correcting for rotation cam R00  
    ptCloud = torch.matmul(R_rect.to('cuda:'+str(ptCloud.get_device())), ptCloud)

    return(ptCloud)



def findCalibParameterTensor(rootDir):

    [P_rect, R_rect, R, T] = findtheCalibparameters(rootDir)
    # Convert the NumPy array to Tensor
    P_rect = torch.from_numpy(P_rect)
    R_rect = torch.from_numpy(R_rect)
    RT = createRTMat(R,T)
    RT = torch.from_numpy(RT)

    return([P_rect, R_rect, RT])

def convertToHomogenousCoordTensor(ptCld):
    ones = torch.ones((ptCld.shape[0],ptCld.shape[1],1)).to('cuda:'+str(ptCld.get_device()))
    ptCld = torch.cat([ptCld,ones],dim=2)

    return(ptCld)

def saveModelParams(model, optimizer, epoch, path):

    print("saving the model")
    state = {'epoch': epoch,
            'modelStateDict':model.state_dict(),
            'optimizerStateDict': optimizer.state_dict(),}
    torch.save(state, path)