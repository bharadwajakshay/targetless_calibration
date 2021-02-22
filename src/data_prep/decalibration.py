#!/usr/bin/env python
import os
import sys
import glob
import cv2
import numpy as np
from helperfunctions import *
import tqdm
import json
import glob


'''
Project the Lidar points on to the image to make sure that the calibration
between the Point cloud and the image are good. 
Steps to achieve it
1. Read the input image
2. Read the input pontcloud
3. Read the calibration file
4. Convert the 3D points on to 2D points similar to the Homogeneous camera calibration 
'''

intensity_img_height = 64
intesnity_img_width = 4500

def generateData(dataDirRaw, dataDirProc, jsonFile):
    recordCnt = 0
    testSubDirs = os.listdir(dataDirRaw)

    globalrecodkeper = []
    globalMaxSizePtCld = 0
    
    print("Total of "+str(len(testSubDirs))+"dated recordings found")
    for dirs in testSubDirs:
        print("Processing direcoty "+dirs )
        subDir = os.path.join(dataDirRaw,dirs)
        subSubDirs = os.listdir(subDir)
        [P_rect, R_rect, R, T] = findtheCalibparameters(os.path.join(dataDirRaw,dirs))
        
        print("Total of "+str(len(subSubDirs))+" recordings found")
        #For each run folder do the following
        for subdirs in subSubDirs:
            print("Processing directory "+subdirs )
            # Make sure the entry is a sub dir and not a file
            runDir = os.path.join(subDir,subdirs)
            if os.path.isdir(runDir):
                #Process the directory
                record, maxSizePtCld  = processRunDir(runDir,dataDirProc, P_rect, R_rect, R, T)
                globalrecodkeper = globalrecodkeper + record

                if maxSizePtCld > globalMaxSizePtCld:
                    globalMaxSizePtCld = maxSizePtCld

    globalrecodkeper.append(globalMaxSizePtCld)
    
    # Dump all the testing data to a json file
    fpJson = open(jsonFile,'w')
    json.dump(globalrecodkeper,fpJson)

def main():
    
    """
    Later move it to config file
    """
    # Check if the no of sys args are correct
    if(len(sys.argv)!= 2):
        print("Error in usage")

    rawDataDir = sys.argv[1]+'raw/'
    trainingDataDirRaw = rawDataDir+'train/'
    testDataDirRaw = rawDataDir+'test/'

    procDataDir = os.path.join(sys.argv[1],'processed')

    trainingDataDirProc = os.path.join(procDataDir,'train')
    trainingJsonFile = os.path.join(trainingDataDirProc,'trainingdata.json')

    testDataDirProc = os.path.join(procDataDir, 'test')
    testJsonFile = os.path.join(testDataDirProc,'testingdata.json')


    """
    Test Data
    """
    print("Generating test data")
    generateData(testDataDirRaw, testDataDirProc, testJsonFile)
                     
                           
    """
    Train Data
    """
    print("Generating training data")
    generateData(trainingDataDirRaw, trainingDataDirProc, trainingJsonFile)
    

    return(0)

if __name__ == '__main__':
    main()