import torch
from data_prep.filereaders import readvelodynepointcloud


def openPtCld(path):
    batchSize = len(path)

    min = 10000000 
    for idx in range(0, batchSize):
        ptCld, __ = readvelodynepointcloud(path[idx])
        if min > ptCld.shape[0]:
            min = ptCld.shape[0]       

    ptCldTensor = torch.empty(batchSize) 
    for idx in range(0, batchSize):
        ptCld, intensity = readvelodynepointcloud(path[idx])
        if min > ptCld.shape[0]:
            print("Currently does nothing")
            

def calculateEuclideanNorm(mat):
    print("Nothing Done")