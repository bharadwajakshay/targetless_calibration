import shutil
import json  
import numpy as np
from torch.utils.data import Dataset
from decalibration import readvelodynepointcloud
from scipy.spatial.transform import Rotation as rot



def normalizePtCld(pointcloud):
    centroid = np.mean(pointcloud, axis=0)
    pointcloud = pointcloud - centroid
    m = np.max(np.sqrt(np.sum(pointcloud**2, axis=1)))
    pointcloud = pointcloud / m
    return pointcloud

class dataLoader(Dataset):

    def __init__ (self, filename="/home/akshay/targetless_calibration/data/2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data/agumenteddata/angles_summary.json"):
        self.datafile = filename
        file_descriptor = open(self.datafile,'r')
        self.data = json.load(file_descriptor)

    def __len__(self):
        return(len(self.data))

    def getItem(self, key):
        # read the point cloud 
        ptCldFileName = self.data[key]["point_filename"]
        ptCld, intensity = readvelodynepointcloud(ptCldFileName)
        transform = self.data[key]["transform"].strip("[[]]").split(" ")
        R_t = []
        for i in range(len(transform)):
            try:
                R_t.append(float(transform[i]))
            except ValueError:
                pass
        transform = np.array(R_t, dtype=float).reshape(7,1)
        translation = transform[:3]
        rotation = transform[4:]

        # Pending normalization of point cloud
        ptCld[:,0:3] = normalizePtCld(ptCld[:,0:3])
        return ptCld, transform
        
        
    def __getitem__(self, key):
        return(getItem(key))
    

if __name__ == "__main__":
    obj = dataLoader()
    x1, x2 = obj.getItem("0000000082_2")
    print("Tested")