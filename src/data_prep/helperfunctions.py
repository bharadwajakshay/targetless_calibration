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
from data_prep.transforms import *



def findtheCalibparameters(rootDir):
    veloCamCalib = "calib_velo_to_cam.txt"
    camCamCalib = "calib_cam_to_cam.txt"
    P_camCam, R_camCam = readcamtocamcalibrationdata(os.path.join(rootDir,camCamCalib))
    R_veloCam, T_veloCam = readveltocamcalibrationdata(os.path.join(rootDir,veloCamCalib))

    return(P_camCam, R_camCam, R_veloCam, T_veloCam)

def processRunDir(runDir, procDataDir, P_rect, R_rect, R, T):
    # Get the images name 
    img_02 = 'image_02'
    veloPts = 'velodyne_points'
    data = 'data'
    imgsDir = os.path.join(runDir,img_02,data)
    veloDir = os.path.join(runDir,veloPts,data)

    dir = runDir.split('/')[(len(runDir.split('/')) - 2):]
    procDataDir = os.path.join(procDataDir,dir[0],dir[1])
    procDataDirDepth = os.path.join(procDataDir,"depthImg")
    procDataDirIntensity = os.path.join(procDataDir,"intensityImg")

    # Check if the folders are present, if not create the folders
    if not os.path.exists(procDataDirDepth):
        os.makedirs(procDataDirDepth)
    
    if not os.path.exists(procDataDirIntensity):
        os.makedirs(procDataDirIntensity)


    recordkeeper = []
    maxSizePtCld = 0

    """
    for every image file in imgs Dir, Find the corresponding points
    1. Get the filename of the point 
    2. Get the corresponding filename of the image
    3. Read the point cloud
    4. generate random transforms 
    5. Caluclate depth map  
    """

    with tqdm.tqdm(total=len(glob.glob(veloDir+'/*.bin'))) as processBar:

        for points in glob.glob(veloDir+'/*.bin'):
            imgFileId = points.split('/')[len(points.split('/'))-1].split('.')[0]
            imgFileName = os.path.join(imgsDir,imgFileId+'.png')

            # Read the point cloud  
            ptcloud,intensityData = readvelodynepointcloud(points)
            sizePtCld = ptcloud.shape[0]

            # Project points over imageplane
            imgPts, rectPtCld, tfRand = projectptstoimgplane(ptcloud, intensityData, P_rect, R_rect, R, T)

            # Filter the points
            [imgW, imgH, img] = readimgfromfile(imgFileName)
            filteredPts2D, filteredPts3D = filterPts(imgPts,rectPtCld, imgW, imgH)
            depthImage = generateDepthImage(filteredPts2D, filteredPts3D, imgW, imgH, img)
            intensityImage = generateIntensityImage(filteredPts2D, filteredPts3D, imgW, imgH,img)

            depthImageFileName = os.path.join(procDataDirDepth,imgFileId+'.png')
            intensityImageFileName = os.path.join(procDataDirIntensity,imgFileId+'.png')

            # Save Images
            cv2.imwrite(depthImageFileName,depthImage)
            cv2.imwrite(intensityImageFileName,intensityImage)

            if sizePtCld > maxSizePtCld:
                maxSizePtCld = sizePtCld

            rundetails={
                'colorImgFileName': imgFileName,
                'pointCldFileName': points,
                'depthImgFileName': depthImageFileName,
                'intensityImgFileName': intensityImageFileName,
                'transform2Estimate': tfRand.flatten().tolist()            
            }

            recordkeeper.append(rundetails)
            processBar.update(1)
    
    return(recordkeeper, maxSizePtCld)


