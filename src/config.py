import numpy as np

image = dict(height = 375,
             width = 1242,
             channel = 3) 

trainingDataFile = "/home/akshay/pytorchEnv/targetless_calibration/parsed_set.txt"
maxPtCldSizeFile = "/home/akshay/pytorchEnv/targetless_calibration/maxPtCldSize.txt"
camToCamCalibFile = "/mnt/291d3084-ca91-4f28-8f33-ed0b64be0a8c/akshay/kitti/raw/2011_09_26/calib_cam_to_cam.txt"
camToVeloCalibFile = "/mnt/291d3084-ca91-4f28-8f33-ed0b64be0a8c/akshay/kitti/raw/2011_09_26/calib_velo_to_cam.txt"

pathToPretrainedModel = "/home/akshay/pytorchEnv/targetless_calibration/src/model/trained/bestTargetCalibrationModel.pth"

logsDirs = "/home/akshay/pytorchEnv/targetless_calibration/testing/logs"

calibrationDir = "/mnt/291d3084-ca91-4f28-8f33-ed0b64be0a8c/akshay/kitti/raw/2011_09_26/"

training = dict(
    batchSize = 5,
    epoch = 30,
    learningRate = 0.0001,
    beta0 = 0.9,
    beta1 = 0.999,
    eps = 1e-08,
    decayRate = 1e-4
)

