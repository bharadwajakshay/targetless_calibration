import numpy as np
import cv2
import random
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as rot
import tqdm
import json
import glob
import os
from data_prep.filereaders import *
from scipy.spatial.transform import Rotation as rot

def projectptstoimgplane(ptcloud, intensityData, P_rect, R_rect, R, T, randomTF = False):

    tfRand = np.eye(4)
    
    # Correct the points with calibration parameters 
    # Convert points into Homogeneous coordinates
    ptcloud = np.hstack((ptcloud,np.ones((ptcloud.shape[0],1))))

    # Create a Transformation matrix
    R_T = createRTMat(R, T)

    ptcloud = np.matmul(R_T,np.transpose(ptcloud))

    #convert data into homogeneous points 
    ptcloud_proj = np.hstack((np.transpose(ptcloud)[:,:3],np.ones((np.transpose(ptcloud).shape[0],1))))

    # Now project points on the image 
    ptcloud_proj = np.matmul(R_rect,np.transpose(ptcloud_proj))

    """
    # Convert data into homogeneous points 
    ptcloud = np.hstack((np.transpose(ptcloud)[:,:3],np.ones((np.transpose(ptcloud).shape[0],1))))

    """
    # Tranform the points by the Randomly genereated transform data
    if randomTF:
        # Generate a random transform
        rRand, tRand = generaterandomRT(1)
        rRand = convertEuler2Rot(rRand)
        tfRand = createRTMat(rRand, tRand)

        # Calulate the inv of RT to transform the point cloud by
        invtfRand = calculateInvTF(tfRand)

        #multiply the points by the transform
        ptcloud_proj = np.matmul(invtfRand,ptcloud_proj)

    # Project the points on to 2D image space [u,v,1]
    imgPts = np.matmul(P_rect,ptcloud_proj)

    # Reintroduce intensity data into ptcloud
    ptcloud = np.transpose(ptcloud)[:,:3]
    ptcloud = np.hstack((ptcloud,intensityData.reshape(len(intensityData),1)))

    # Divide the u,v by the resedual to obtain the right access
    imgPts = np.transpose(imgPts)
    imgPts[:,0] /= imgPts[:,2]
    imgPts[:,1] /= imgPts[:,2]

    return[imgPts[:,:2],ptcloud, tfRand]

def filterPts(points2D, points3D, imgWidth, imgHeight, img = None):
    condition = (points2D[:,0] < imgWidth) & (points2D[:,0] >= 0) &\
                (points2D[:,1] < imgHeight) & (points2D[:,1] >= 0) &\
                (points3D[:,2] > 0)
    
    pointsImg = points2D[condition]
    pointsCld = points3D[condition]

    return [pointsImg, pointsCld]

def generateDepthImage(pointsImg, pointsCld, imgWidth, imgHeight, img=None):

    minDist = 0
    maxDist = np.amax(pointsCld[:,2])
    intensityMax = 255
    intensityMin = 0

    # scaletointensity (value - minDist)/(maxDist-minDist) x (intensityMax-intensityMin) + intensityMin

    scaletointensity = (((pointsCld[:,2] - minDist)/(maxDist-minDist))*(intensityMax-intensityMin)) + intensityMin
    
    # assert depth map as intensty values
    pointsImg = np.hstack((pointsImg,scaletointensity.reshape(scaletointensity.shape[0],1)))
    
    # convert points to int
    pointsImg = pointsImg.astype(int)

    # Create a new image
    depthimg = np.zeros((imgHeight,imgWidth),dtype=np.uint8)
    depthimg[pointsImg[:,1],pointsImg[:,0]] = 255 - pointsImg[:,2]
    
    """    
    cv2.imwrite("Test_depth_img.png",depthimg)

    # Create a overlap image
    if (img.any()):
        #convert the image into HSV
        img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        # PLOT points of the image over the image
        for idx in range(1,pointsImg.shape[0]):
            img[pointsImg[idx,1],pointsImg[idx,0]] = (pointsImg[idx,2],255,255)
            cv2.circle(img, (pointsImg[idx,0],pointsImg[idx,1]),1, (int(pointsImg[idx,2]),255,255),-1)
        img = cv2.cvtColor(img,cv2.COLOR_HSV2BGR)
        cv2.imwrite("TDepth_img_Overlap.png",img)
    """

    return depthimg
    
def generateIntensityImage(pointsImg, pointsCld, imgWidth, imgHeight, img=None):
    
    minIntensity = np.amin(pointsCld[:,3])
    maxIntensity = np.amax(pointsCld[:,3])
    intensityMax = 255
    intensityMin = 0

    scaletheintensity = (((pointsCld[:,3] - minIntensity)/(maxIntensity-minIntensity))*(intensityMax-intensityMin)) + intensityMin

    # assert depth map as intensty values
    pointsImg = np.hstack((pointsImg,scaletheintensity.reshape(scaletheintensity.shape[0],1)))
    
    # convert points to int
    pointsImg = pointsImg.astype(int)

    # Create a new image
    intensityImg = np.zeros((imgHeight,imgWidth),dtype=np.uint8)
    intensityImg[pointsImg[:,1],pointsImg[:,0]] = 255 - pointsImg[:,2]

    """
    cv2.imwrite("Test_intensity_img.png",intensityImg)
    """
    
    return intensityImg

def createRTMat(R, T):
    RT = np.hstack((R, T))
    RT = np.vstack((RT,np.zeros((1,RT.shape[1]))))
    RT[3][3] = 1
    return RT

def calculateInvTF(rT):
    R = rT[:3,:3]
    T = rT[:3,3]

    invR = np.linalg.inv(R)
    invT = -np.matmul(invR,T)
    invRT = createRTMat(invR,invT.reshape(invT.shape[0],1))
    return invRT

def generaterandomRT(sample_size):
    '''
    generate random rotation and translation 
    [pitch, roll, yaw] range between [-30, 30] 
    and t between [-1, 1]
    '''
    R = np.random.randint(-3,3,size=(3,sample_size))
    t =  np.random.normal(-0.3,0.3,[3,sample_size])
    return[R,t]

def convertEuler2Rot(E,degree=True):
    Robj = rot.from_euler('zyx',E.T,degrees=degree)
    return(Robj.as_matrix()[0,:,:])
