import numpy as np

image = dict(height = 375,
             width = 1242,
             channel = 3) 

trainingDataFile = "/mnt/data/akshay/kitti/processed/2011_09_26/datasetdetails.json"
camToCamCalibFile = "/mnt/291d3084-ca91-4f28-8f33-ed0b64be0a8c/akshay/kitti/raw/2011_09_26/calib_cam_to_cam.txt"
camToVeloCalibFile = "/mnt/291d3084-ca91-4f28-8f33-ed0b64be0a8c/akshay/kitti/raw/2011_09_26/calib_velo_to_cam.txt"

pathToPretrainedModel = "/home/akshay/pyTorchEnv/targetless_calibration/src/model/trained/bestModelEucledian.pth"
pathToCheckpoint = "/home/akshay/pyTorchEnv/targetless_calibration/checkpoints"

logsDirs = "/home/akshay/pyTorchEnv/targetless_calibration/logs/"

backbone = 'RESNET' # only possible options are 'RESNET' and 'SWIN'
networkDepth = 'shallow' # only possible options are 'shallow' and 'deep'
useAttention = True

calibrationDir = "/mnt/data/akshay/kitti/raw/2011_09_26/"

loadCheckPoint = False
checkpointFilename = None



training = dict(
    batchSize = 30,
    epoch = 25,
    learningRate = 0.005,
    beta0 = 0.9,
    beta1 = 0.999,
    eps = 1e-08,
    decayRate = 1e-4
)

previousBestSE3Dist = 100
