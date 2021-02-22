import torch
import numpy as np
import os
import sys
import importlib
import shutil
import json
import importlib
from pathlib import Path
import provider

def getTranslationRot(prediction):
    xyz = xyz = prediction[0:3,:]
    rot = prediction[3:,:].flatten()
    return(xyz, rot)

def getRotMatFromQuat(quat):
    rObj = rot.from_quat(quat)
    return(rObj.as_matrix())

def getInvTransformMat(R,T):
    # https://math.stackexchange.com/questions/152462/inverse-of-transformation-matrix
    invR = R.transpose()
    invT = np.matmul(-invR,T)
    R_T = np.vstack((np.hstack((invR,invT)),[0, 0, 0, 1.]))
    return(R_T)

def extractEulerAngles(R):
    rObj = rot.from_matrix(R)
    euler = rObj.as_euler(seq='zyx', degrees=True).reshape(3,1)
    return(euler[0],euler[1],euler[2])


def extractRotnTranslation(transformation):
    R = transformation[0:3,0:3]
    T = transformation[0:3,3]
    return(extractEulerAngles(R))


def calculateEucledianDist(predPtCld, targetPtCld):
    euclideanDist = 1000
  
    return(euclideanDist)