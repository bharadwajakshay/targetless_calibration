import numpy as np

image = dict(height = 375,
             width = 1242,
             channel = 3) 

trainingDataFile = "/home/akshay/programming/pytorchvenv/targetless_calibration/parsed_set.txt"
maxPtCldSizeFile = "/home/akshay/programming/pytorchvenv/targetless_calibration/maxPtCldSize.txt"
camToCamCalibFile = "/home/akshay/5TB-HDD/Datasets/kitti/raw/2011_09_26/calib_cam_to_cam.txt"
camToVeloCalibFile = "/home/akshay/5TB-HDD/Datasets/kitti/raw/2011_09_26/calib_velo_to_cam.txt"

pathToPretrainedModel = "/home/akshay/programming/pytorchvenv/targetless_calibration/src/model/trained/bestTargetCalibrationModel.pth"

logsDirs = "/home/akshay/programming/pytorchvenv/targetless_calibration/testing/logs"

calibrationDir = "/home/akshay/5TB-HDD/Datasets/kitti/raw/2011_09_26/"

training = dict(
    batchSize = 5,
    epoch = 30,
    learningRate = 0.0001,
    beta0 = 0.9,
    beta1 = 0.999,
    eps = 1e-08,
    decayRate = 1e-4
)

