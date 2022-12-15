import numpy as np

image = dict(height = 375,
             width = 1242,
             channel = 3) 

trainingDataFile = "/home/akshay/pyTorchEnv/targetless_calibration/parsed_set_20_10.txt"
maxPtCldSizeFile = "/home/akshay/pyTorchEnv/targetless_calibration/maxPtCldSize.txt"
camToCamCalibFile = "/mnt/291d3084-ca91-4f28-8f33-ed0b64be0a8c/akshay/kitti/raw/2011_09_26/calib_cam_to_cam.txt"
camToVeloCalibFile = "/mnt/291d3084-ca91-4f28-8f33-ed0b64be0a8c/akshay/kitti/raw/2011_09_26/calib_velo_to_cam.txt"

pathToPretrainedModel = "/home/akshay/pytorchEnv/targetless_calibration/src/model/trained/bestTargetCalibrationModel_20_1m5_35Epocs_shallow.pth"
pathToSWINPreTrainedModel = "/home/akshay/pytorchEnv/targetless_calibration/src/model/trained/SWINPath.pth"

logsDirs = "/home/akshay/pytorchEnv/targetless_calibration/logs/"

backbone = 'RESNET' # only possible options are 'RESNET' and 'SWIN'
networkDepth = 'shallow' # only possible options are 'shallow' and 'deep'

calibrationDir = "/mnt/data/akshay/kitti/raw/2011_09_26/"

training = dict(
    batchSize = 8,
    epoch = 35,
    # Batchsize for larger GPUs (A6000)
    #batchSize = 40,
    #epoch = 50,
    learningRate = 0.00003,
    beta0 = 0.9,
    beta1 = 0.999,
    eps = 1e-08,
    decayRate = 1e-4
)

previousBestSE3Dist = 100
