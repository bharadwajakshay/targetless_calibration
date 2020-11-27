#!/usr/bin/env python
import os
import sys
import glob
import cv2
import numpy as np
from helperfunctions import *
from scipy.spatial.transform import Rotation as rot
import tqdm
import json


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
                # erase 4th column ([0,0,0])
                P_ = P_[:3, :3]

            if key == ('R_rect_'+mode):
                R_ = np.fromstring(val,sep=' ')
                R_ = R_.reshape(3, 3)
    return [P_, R_] 


def projpointcloud2imgplane(points, v_fov, h_fov, velcamR, velcamT, camProj, mode='02'):

    xyz_v, c_ = velo_points_filter(points, v_fov, h_fov)
    
    RT_ = np.concatenate((velcamR, velcamT),axis = 1)
    
    # convert velodyne coordinates(X_v, Y_v, Z_v) to camera coordinates(X_c, Y_c, Z_c) 
    for i in range(xyz_v.shape[1]):
        xyz_v[:3,i] = np.matmul(RT_, xyz_v[:,i])
        
    xyz_c = np.delete(xyz_v, 3, axis=0)

    # convert camera coordinates(X_c, Y_c, Z_c) image(pixel) coordinates(x,y) 
    for i in range(xyz_c.shape[1]):
        xyz_c[:,i] = np.matmul(camProj, xyz_c[:,i])    

    xy_i = xyz_c[::]/xyz_c[::][2]
    ans = np.delete(xy_i, 2, axis=0)
    
    return ans, c_


def main():
    '''
    The main function where the images are read, point clouds are read and calibration data
    '''
    if len(sys.argv) != 5:
        print("Incorrect usage")
        exit(-1)
    camvelocalibfilename = sys.argv[1]
    cameracalibfilename = sys.argv[2]
    imagefiledir = sys.argv[3]
    pointcloudfiledir = sys.argv[4]

    agumenteddatapath = pointcloudfiledir+'agumenteddata/'

    data = {}

    # Move into image fdirectory
    os.chdir(imagefiledir)
    
    with tqdm.tqdm(total=len(glob.glob('*.png'))) as img_bar:

        for img_file in glob.glob('*.png'):
            filename = img_file.split('.')[0]
            
            pointcloudfilename = pointcloudfiledir+filename+'.bin'
            if(not os.path.exists(agumenteddatapath)):
                os.makedirs(agumenteddatapath)

            [width,height,img] = readimgfromfile(img_file)
            [pointcloud,intensity_val] = readvelodynepointcloud(pointcloudfilename)

            # read cam projection matrix
            [P,R_02] = readcamtocamcalibrationdata(cameracalibfilename)
            # read vel - cam Roation and Translation matrix
            [R,T] = readveltocamcalibrationdata(camvelocalibfilename)

            veltocam_Transform = np.vstack((np.hstack((R,T)),[0, 0, 0, 1.0]))

        
            # project the velodyne points on to the image
            # H-FOV = 360deg
            # V-FOV = 26.9deg

            samplesize = 30
            [R_euler,t_decalib] = generaterandomRT(samplesize)

            [points,color] = projpointcloud2imgplane(pointcloud,(-24.9,2),(-45,45), R, T, P)
            ptsnimg = displayprojectedptsonimg(points,color,img)
            cv2.imwrite(agumenteddatapath+filename+'_original.png',ptsnimg)

            with tqdm.tqdm(total=samplesize) as rotation_bar:
                for idx in range(samplesize):
                    R_obj = rot.from_euler('zyx',R_euler[:,idx].T,degrees=True)
                    R_rand = R_obj.as_matrix()
                    quat_rand = R_obj.as_quat().reshape(4,1)
                    T_rand = t_decalib[:,idx].reshape(3,1)

                    transform = np.vstack((np.hstack((R_rand, T_rand)),[0,0,0,1.0]))

                    # add another coloumn to point clouds
                    one_col = np.ones(pointcloud.shape[0]).reshape(pointcloud.shape[0],1)

                    pointcloud_padded = np.hstack((pointcloud, one_col))

                    transformed_pts = np.matmul(transform,np.matmul(veltocam_Transform, pointcloud_padded.T))
                    transformed_pts = transformed_pts.T[:,:3]
 
                    transformed_pts_filename = agumenteddatapath+filename+'_'+str(idx)+'.bin'
                    transformed_pts_imagename = agumenteddatapath+filename+'_'+str(idx)+'.png'

                    transformed_pts.astype(transformed_pts.dtype).tofile(transformed_pts_filename)

                    pt_Cld_vis = np.matmul(transform,pointcloud_padded.T).T[:,:3]

                    [points_decal,color] = projpointcloud2imgplane(pt_Cld_vis,(-24.9,2),(-45,45), R, T, P)
                    ptsnimg_decal = displayprojectedptsonimg(points_decal,color,img)
                    cv2.imwrite(transformed_pts_imagename, ptsnimg_decal)

                    # Write the R|t to the file
                    transform_quat = np.vstack((T_rand,quat_rand))
                    data[filename+'_'+str(idx)] = {'transform' : str(np.expand_dims(np.ndarray.flatten(transform_quat), 0)), 'point_filename' : transformed_pts_filename} 
                    rotation_bar.update(1)
            img_bar.update(1)

        jsonFilename = open(agumenteddatapath+'angles_summary.json','x')
        json.dump(data,jsonFilename)
        jsonFilename.close()


    return(0)

if __name__ == '__main__':
    main()