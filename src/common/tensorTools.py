import torch
from common.pytorch3D import *


def invTransformation():
    return(invRT)

def createRTMatTensor(inputTensor):
    # The input Tensor is composed of two components R, T
    T = inputTensor[0]
    R = inputTensor[1]
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


    RT = torch.cat((RT,zeros),dim=1)

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

    invRT = torch.cat((invRT,zeros),dim=1)

    return(invRT)

def getImageTensorFrmPtCloud(projectionMat, ptCld):

    projPts = torch.matmul(projectionMat, ptCld)

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
            zeroTensor = torch.zeros((maxSize2D - tempTensor2D.shape[0], tempTensor2D.shape[1]))
            tempTensor2D = torch.vstack((tempTensor2D,zeroTensor))

        if tempTensor3D.shape[0] < maxSize3D:
            zeroTensor = torch.zeros((maxSize3D - tempTensor3D.shape[0], tempTensor3D.shape[1]))
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
    addChannel = (((addChannel[:] - addChannelMin[:])/(addChannelMax[:]-addChannelMin[:]))*(maxValue[:]-minValue[:])) + minValue[:]

    # Convert points to int 
    ptCld2D = ptCld2D.type(torch.int)


    # create a new image
    newImg = torch.zeros((ptCld2D.shape[0], imgHeight, imgWidth, noChannel))

    newImg[:, ptCld2D[:,1], ptCld2D[:,0]] =255 - addChannel[:]

    return(newImg)



