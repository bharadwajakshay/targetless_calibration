import numpy as np
import cv2
import random
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as rot
import tqdm
import json
import glob


def readimgfromfile(path_to_file):
    image = cv2.imread(path_to_file)
    return([image.shape[1],image.shape[0],image])

def readvelodynepointcloud(path_to_file):
    '''
    The velodyne data that is presented is in the form of a np array written as binary data.
    So to read the file, we use the inbuilt fuction form the np array to rad from the file
    '''

    pointcloud = np.fromfile(path_to_file, dtype=np.float32).reshape(-1, 4)
    intensity_data = pointcloud[:,3]
    
    # Return points ignoring the reflectivity 
    return(pointcloud[:,:3], intensity_data)

def readveltocamcalibrationdata(path_to_file):
    ''' 
    get Rotation(R : 3x3), Translation(T : 3x1) matrix info 
    using R,T matrix, we can convert velodyne coordinates to camera coordinates
    '''
    with open(path_to_file, "r") as f:
        file = f.readlines()    
        
        for line in file:
            (key, val) = line.split(':',1)
            if key == 'R':
                R = np.fromstring(val, sep=' ')
                R = R.reshape(3, 3)
            if key == 'T':
                T = np.fromstring(val, sep=' ')
                T = T.reshape(3, 1)

    return R, T

def readcamtocamcalibrationdata(path_to_file, mode='02'):
    """
    If your image is 'rectified image' :
        get only Projection(P : 3x4) matrix is enough
    but if your image is 'distorted image'(not rectified image) :
        you need undistortion step using distortion coefficients(5 : D)
        
    in this code, I'll get P matrix since I'm using rectified image
    """
    with open(path_to_file, "r") as f:
        file = f.readlines()
        
        for line in file:
            (key, val) = line.split(':',1)
            if key == ('P_rect_' + mode):
                P_ = np.fromstring(val, sep=' ')
                P_ = P_.reshape(3, 4)

            if key == ('R_rect_00'):
                R_ = np.fromstring(val,sep=' ')
                R_ = R_.reshape(3, 3)
                # append zero coloumn and zero row to make it 4x4 and set R[3,3] = 1
                R_ = np.vstack((R_,np.zeros((1,R_.shape[1]))))
                R_ = np.hstack((R_,np.zeros((R_.shape[0],1))))
                R_[3][3] = 1

    return [P_, R_]
