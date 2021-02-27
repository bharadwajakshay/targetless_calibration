import shutil
import json  
import numpy as np
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as rot
from PIL import Image
import cv2
import math



def normalizePtCld(pointcloud):
    centroid = np.mean(pointcloud, axis=0)
    pointcloud = pointcloud - centroid
    m = np.max(np.sqrt(np.sum(pointcloud**2, axis=1)))
    pointcloud = pointcloud / m
    return pointcloud

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

def normalizeImg(image):
    # Convert the image into float
    image = image.astype('float32')
    image = np.divide(image,255)
    return (image)

class dataLoader(Dataset):

    def __init__ (self, filename='/mnt/291d3084-ca91-4f28-8f33-ed0b64be0a8c/akshay/targetless_calibration/data/2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data/agumenteddata/angles_summary.json'):
        self.datafile = filename
        file_descriptor = open(self.datafile,'r')
        self.data = json.load(file_descriptor)

        # Read the longest point cloud 
        self.maxPtCldSize = self.data[-1]

        # remove the last entry as It is not needed any more
        self.data = self.data[:-1]
        self.data = self.data[:2000]

    def __len__(self):
        return(len(self.data))

    def __getitem__(self, key):
        return(self.getItem(key))

    def getItem(self, key):
        # read the point cloud 
        ptCldFileName = self.data[key]["pointCldFileName"]
        ptCld, intensityValues = readvelodynepointcloud(ptCldFileName)
        intensityValues = intensityValues.reshape(intensityValues.shape[0],1)
        
        if ptCld.shape[0] < self.maxPtCldSize :
            # Pad the point cloud with 0
            paddingValuesPtCld = np.zeros((self.maxPtCldSize - ptCld.shape[0], ptCld.shape[1]))
            ptCld = np.vstack((ptCld,paddingValuesPtCld))

            # Padding values for intensity 
            paddingValuesIntesnity = np.zeros((self.maxPtCldSize - intensityValues.shape[0], 1))
            intensityValues = np.vstack((intensityValues,paddingValuesIntesnity))
        
        # Combine the point Cloud with intensity data
        ptCld = np.hstack((ptCld,intensityValues))
        
        # Read the color image
        colorImgFileName = self.data[key]["colorImgFileName"]
        colorImgW, colorImgH, colorImg = readimgfromfile(colorImgFileName)
        
        # Convert the image to grayscale
        grayImg = cv2.cvtColor(colorImg,cv2.COLOR_BGR2GRAY)

        # normalize the image
        colorImg = normalizeImg(colorImg)


        # Read the depth image
        depthImgFileName = self.data[key]["depthImgFileName"]
        depthImgW, depthImgH, depthImg = readimgfromfile(depthImgFileName)
        #normalize the image
        depthImg = normalizeImg(depthImg)

        # Read the intensity image
        intesnityImgFileName = self.data[key]["intensityImgFileName"]
        intesnityImgW, intesnityImgH, intesnityImg = readimgfromfile(intesnityImgFileName)
        # normalize  the image
        intesnityImg = normalizeImg(intesnityImg)


        # Read the transform 
        transform = self.data[key]["transform2Estimate"]
        transform = np.asarray(transform)
        transform = np.reshape(transform,(4,4))

        return(ptCld, colorImg, grayImg, depthImg, intesnityImg, transform)

if __name__ == "__main__":
    obj = dataLoader('/mnt/291d3084-ca91-4f28-8f33-ed0b64be0a8c/akshay/kitti/processed/train/trainingdata.json')
    x0, x1, x2, x3, x4, x5 = obj.__getitem__(int(15))
    print("Tested")