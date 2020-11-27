import numpy as np
import cv2
import random


def getdepthcolor(val, min_d=0, max_d=120):
    '''
    print Color(HSV's H value) corresponding to distance(m) 
    close distance = red , far distance = blue
    '''
    np.clip(val, 0, max_d, out=val) # max distance is 120m but usually not usual
    return (((val - min_d) / (max_d - min_d)) * 120).astype(np.uint8) 

def getinrangeptsH(points, m, n, fov):
    # extract horizontal in-range points
    return np.logical_and(np.arctan2(n,m) > (-fov[1] * np.pi / 180), \
                          np.arctan2(n,m) < (-fov[0] * np.pi / 180))

def getinrangeptsV(points, m, n, fov):
    # extract vertical in-range points
    return np.logical_and(np.arctan2(n,m) < (fov[1] * np.pi / 180), \
                          np.arctan2(n,m) > (fov[0] * np.pi / 180))

def fov_setting(points, x, y, z, dist, h_fov, v_fov):
    # filter points based on h,v FOV  
    
    if h_fov[1] == 180 and h_fov[0] == -180 and v_fov[1] == 2.0 and v_fov[0] == -24.9:
        return points
    
    if h_fov[1] == 180 and h_fov[0] == -180:
        return points[getinrangeptsV(points, dist, z, v_fov)]
    elif v_fov[1] == 2.0 and v_fov[0] == -24.9:        
        return points[getinrangeptsH(points, x, y, h_fov)]
    else:
        h_points = getinrangeptsH(points, x, y, h_fov)
        v_points = getinrangeptsV(points, dist, z, v_fov)
        return points[np.logical_and(h_points, v_points)]

def in_range_points(points, size):
    # extract in-range points ""
    return np.logical_and(points > 0, points < size)    

def velo_points_filter(points, v_fov, h_fov):
    # extract points corresponding to FOV setting
    
    # Projecting to 2D
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    if h_fov[0] < -90:
        h_fov = (-90,) + h_fov[1:]
    if h_fov[1] > 90:
        h_fov = h_fov[:1] + (90,)
    
    x_lim = fov_setting(x, x, y, z, dist, h_fov, v_fov)[:,None]
    y_lim = fov_setting(y, x, y, z, dist, h_fov, v_fov)[:,None]
    z_lim = fov_setting(z, x, y, z, dist, h_fov, v_fov)[:,None]

    # Stack arrays in sequence horizontally
    xyz_ = np.hstack((x_lim, y_lim, z_lim))
    xyz_ = xyz_.T

    # stack (1,n) arrays filled with the number 1
    one_mat = np.full((1, xyz_.shape[1]), 1)
    xyz_ = np.concatenate((xyz_, one_mat),axis = 0)

    # need dist info for points color
    dist_lim = fov_setting(dist, x, y, z, dist, h_fov, v_fov)
    color = getdepthcolor(dist_lim, 0, 70)
    
    return xyz_, color


def displayprojectedptsonimg(points, color, image):
    """ project converted velodyne points into camera image """
    
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    for i in range(points.shape[1]):
        cv2.circle(hsv_image, (np.int32(points[0][i]),np.int32(points[1][i])),1, (int(color[i]),255,255),-1)

    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

def generaterandomRT(sample_size):
    '''
    generate random rotation and translation 
    [pitch, roll, yaw] range between [-30, 30] 
    and t between [-1, 1]
    '''
    R = np.random.randint(-15,15,size=(3,sample_size))
    t =  np.random.normal(-1,1,[3,sample_size])
    return[R,t]
    




