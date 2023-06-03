import shutil
import json  
import numpy as np
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as rot
from PIL import Image, ImageOps
import cv2
import math
import open3d as o3d
import json

from torchvision import transforms


def getNormals(ptCld):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(ptCld)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    o3d.visualization.draw_geometries([pcd])
    normals = np.asarray(pcd.normals)
    print('BreakPoint')
    return(normals)

def normalizePtCld(pointcloud):
    centroid = np.mean(pointcloud, axis=0)
    pointcloud = pointcloud - centroid
    m = np.max(np.sqrt(np.sum(pointcloud**2, axis=1)))
    pointcloud = pointcloud / m
    return pointcloud

def readimgfromfileCV2(path_to_file):
    image = cv2.imread(path_to_file)
    return([image.shape[1],image.shape[0],image])

def readimgfromfilePIL(path_to_file):
    image = Image.open(path_to_file)
    image = image.convert('RGB')
    return(image.width,image.height,image)

def readvelodynepointcloud(path_to_file):
    '''
    The velodyne data that is presented is in the form of a np array written as binary data.
    So to read the file, we use the inbuilt fuction form the np array to rad from the file
    '''

    pointcloud = np.fromfile(path_to_file).reshape(4,-1)
    intensity_data = pointcloud[3,:]
    
    # Return points ignoring the reflectivity 
    return(pointcloud[:3,:], intensity_data)

def resizeImgForResNet(image):
    # Since resnet is trainted to take in 224x224, we want resize the image of the shortest size to be 224
    imageScaleRatio = 224/375
    newImW = int(image.shape[1]*imageScaleRatio)
    newImH = int(image.shape[0]*imageScaleRatio)
    image = cv2.resize(image,(newImW,newImH))
    return (image)
     

def normalizePILGrayImg(image):
    # Convert the image into float
    image = np.array(image)
    image = image.astype('float32')
    for idx in range(0,image.shape[2]):
        max  = image[:,:,idx].max().max()
        image[:,:,idx] = np.divide(image[:,:,idx],max)
    
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    return (image)

def normalizePILImg(image):
    # Convert the image into float
    image = np.array(image)
    image = image.astype('float32')
    image = np.divide(image,255)
    
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    return (image)

class dataLoader(Dataset):

    def __init__ (self, filename,  mode='train'):
        self.datafile = filename
        with open(self.datafile,'r') as jsonFile:
            self.data = json.load(jsonFile)

        self.dataList = []

        for keys in list(self.data.keys()):
            for scenes in  list(self.data[keys].keys()):
                self.dataList.append(self.data[keys][scenes])

        trainIdx = int(len(self.dataList)*0.8)
        testIdx = int(len(self.dataList)*0.9)

        if mode =='train':
            self.data = self.dataList[:trainIdx]
           
        if mode =='test':
            self.data = self.dataList[trainIdx:testIdx]
       
        if mode =='evaluate':
            self.data = self.dataList[testIdx:]

    def __len__(self):
        return(len(self.data))

    def __getitem__(self, key):
        return(self.getItem(key))

    def getItem(self, key):
        """
        [0]  Source depth map
        [1]  Target depth map
        [2]  Source intensity map
        [3]  Target intensity map
        [4]  Color image source
        [5]  Color image target
        [6]  Point Cloud Filename
        [7:] Transforms 
        """

        """
        """
        # Define preprocessing Pipeline
        imgTensorPreProc = transforms.Compose([
        #transforms.Resize(188),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


        sample = self.dataList[key]

        # Read from the file
        srcDepthImageFileName = sample['depth image filename']

        srcIntensityImageFileName = sample['intensity image filename']

        srcPointImageFileName = sample['point image filename']

        srcColorImageFileName = sample['image filename']

        gtPointCldFileName = sample['points in camera frame']

        transform = np.asarray(sample['transfromation'])

        __, __, srcDepthImg = readimgfromfilePIL(srcDepthImageFileName)
        srcDepthImg = imgTensorPreProc(srcDepthImg)

        __, __, srcIntensityImg = readimgfromfilePIL(srcIntensityImageFileName)
        srcIntensityImg = imgTensorPreProc(srcIntensityImg)

        __, __, srcPointImg = readimgfromfilePIL(srcPointImageFileName)
        srcPointImg = imgTensorPreProc(srcPointImg)
        

        __, __, srcClrImg = readimgfromfilePIL(srcColorImageFileName)
        colorImage = np.array(srcClrImg)
        srcClrImg = imgTensorPreProc(srcClrImg)

        # read the point cloud 
        gtPtCld, gtIntensityValues = readvelodynepointcloud(gtPointCldFileName)

        gtIntensityValues = gtIntensityValues.reshape(gtIntensityValues.shape[0],1)
        srcDepthImg[2,:,:] = srcIntensityImg[0,:,:]
        srcDepthImg[0,:,:] = srcPointImg[0,:,:]
        
        return (srcClrImg, srcDepthImg, transform, gtPtCld.T, [colorImage])


if __name__ == "__main__":
    obj = dataLoader('/mnt/291d3084-ca91-4f28-8f33-ed0b64be0a8c/akshay/kitti/processed/train/trainingdata.json')
    x0, x1, x2, x3, x4, x5 = obj.__getitem__(int(15))
    print("Tested")