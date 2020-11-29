import shutil
import json  
import numpy as np
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as rot



def normalizePtCld(pointcloud):
    centroid = np.mean(pointcloud, axis=0)
    pointcloud = pointcloud - centroid
    m = np.max(np.sqrt(np.sum(pointcloud**2, axis=1)))
    pointcloud = pointcloud / m
    return pointcloud

def readvelodynepointcloud(path_to_file):
    '''
    The velodyne data that is presented is in the form of a np array written as binary data.
    So to read the file, we use the inbuilt fuction form the np array to rad from the file
    '''

    pointcloud = np.fromfile(path_to_file, dtype=np.float32).reshape(-1, 4)
    intensity_data = pointcloud[:,3]
    
    # Return points ignoring the reflectivity 
    return(pointcloud[:,:3], intensity_data)

def readtransformedpointcloud(path_to_file):
    '''
    The velodyne data that is presented is in the form of a np array written as binary data.
    So to read the file, we use the inbuilt fuction form the np array to rad from the file
    '''

    pointcloud = np.fromfile(path_to_file, dtype=np.float64).reshape(-1,3)
    
    # Return points ignoring the reflectivity 
    return(pointcloud[:,:3])

class dataLoader(Dataset):

    def __init__ (self, filename="/home/akshay/targetless_calibration/data/2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data/agumenteddata/angles_summary.json"):
        self.datafile = filename
        file_descriptor = open(self.datafile,'r')
        self.data = json.load(file_descriptor)

    def __len__(self):
        return(len(self.data))

    def __getitem__(self, key):
        return(self.getItem(str(key)))

    def getItem(self, key):
        # read the point cloud 
        ptCldFileName = self.data[key]["point_filename"]
        targetCldFileName = self.data[key]["target_filename"]
        ptCld = readtransformedpointcloud(ptCldFileName)
        targetCld, intensity = readvelodynepointcloud(targetCldFileName)

        if(ptCld.shape[0] != targetCld.shape[0]):
            print("The size of the input cloud and the target cloud is not the same")
            print("Input Point Cloud Size = "+str(ptCld.shape[0])+"\tTransformed Point Cloud Size = "+str(targetCld.shape[0]))
            raise

        transform = self.data[key]["transform"].strip("[[]]").split(" ")
        R_t = []
        for i in range(len(transform)):
            try:
                R_t.append(float(transform[i]))
            except ValueError:
                pass
        transform = np.array(R_t, dtype=float).reshape(7,1)
        '''
        translation = transform[:3]
        rotation = transform[4:]
        '''
        # Pending normalization of point cloud
        # ptCld[:,0:3] = normalizePtCld(ptCld[:,0:3])
        return (ptCld, transform, targetCld)
        
    

if __name__ == "__main__":
    obj = dataLoader()
    x1, x2, x3 = obj.__getitem__('15')
    print("Tested")