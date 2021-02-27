import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation as rot
import numpy as np

from data_prep.filereaders import *
from data_prep.helperfunctions import *
from data_prep.transforms import *
from common.tensorTools import *
from common.pytorch3D import *
from model.NCC import NCC

def conv(in_channels, out_channels, kernel_size=3, stride_len = 1, dilation = 1):
    return nn.Conv1d(in_channels, out_channels, kernel_size = kernel_size, 
                stride = stride_len, padding=dilation, groups=1, bias= False,
                dilation=dilation)


def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def _index_from_letter(letter: str):
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2

def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
):
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.
    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.
    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


def matrix_to_euler_angles(matrix, convention: str):
    """
    Convert rotations given as rotation matrices to Euler angles in radians.
    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.
    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)




'''
# This acts as the blasic block of the neural network.
# @ in_channels : No of channels in the input image [int]   
# @ out_channels: No of channels the convolution produces [int]
# @ stride      : Controls the stride for the cross-correlation [int/tuple]
# @ downsample  : 
# @ dialation   : Controls the spacing between the kernel points [int/tuple]
'''
class BasicBlock(nn.Module):
    def __init__(self,in_channels, out_channels, stride, downsample=None, dialation=1):
        super(BasicBlock, self).__init__()
        #Batch normalization
        normalization_layer = nn.BatchNorm1d
        self.conv_block1 = conv(in_channels,out_channels,stride_len=stride)
        self.batchnormal_1 = normalization_layer(out_channels)
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.conv_block2 = conv(out_channels,out_channels)
        self.batchnormal_2 = normalization_layer(out_channels)
        self.downsample = downsample 
        self.stride = stride

    def forward(self,x):
        identiy = x
        x=x.float()

        out = self.conv_block1(x)
        out = self.batchnormal_1(out)
        out = self.lrelu(out)

        out = self.conv_block2(out)
        out = self.batchnormal_2(out)

        if self.downsample is not None:
            identiy = self.downsample(x)

        # out += identiy
        out = self.lrelu(out)

        return (out)

class pointcloudnet(nn.Module):
    def __init__(self,layers):
        super(pointcloudnet,self).__init__()

        # standard parameters initialization
        self.inplanes = 64
        self.dialation = 1
        self.groups = 1
        self.base_wodth = 64

        #  As there are x,y,z channels
        input_channel  = 3

        # Using Batch normalization: Define normalization layer
        self.BNL0 = nn.BatchNorm1d(64).to('cuda:1')
        self.BNL1 = nn.BatchNorm1d(128).to('cuda:1')
        self.BNL2 = nn.BatchNorm1d(256).to('cuda:1')
        self.BNL3 = nn.BatchNorm1d(512).to('cuda:2')
        self.BNL4 = nn.BatchNorm1d(1024)
        self.BNL5 = nn.BatchNorm1d(2048)
        self.lReLu = nn.ReLU(inplace=True)

        '''
        # Need to start with a conv layer
        self.layer0 = self.create_layers(BasicBlock, 3, 64, layers[0])
        self.layer1 = self.create_layers(BasicBlock, 64, 128, layers[1])
        self.layer2 = self.create_layers(BasicBlock, 128, 256, layers[2]) 
        self.layer3 = self.create_layers(BasicBlock, 256, 512, layers[3]) 
        self.layer4 = self.create_layers(BasicBlock, 512, 1024, layers[4]) 
        self.layer5 = self.create_layers(BasicBlock, 1024, 2048, layers[5])

        self.leakyReLU_layer = nn.LeakyReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((64,1)
        self.avgpool2 = nn.AdaptiveAvgPool1d(256)
        self.avgpool3 = nn.AdaptiveAvgPool1d(512)
        ''' 
        self.convB1 = nn.Conv1d(input_channel, 64, 1).to('cuda:1')
        nn.init.xavier_uniform_(self.convB1.weight)
        self.convB2 = nn.Conv1d(64, 128, 1).to('cuda:1')
        nn.init.xavier_uniform_(self.convB2.weight)
        self.convB3 = nn.Conv1d(128, 256, 1).to('cuda:1')
        nn.init.xavier_uniform_(self.convB3.weight)
        self.convB4 = nn.Conv1d(256, 512, 1).to('cuda:2')
        nn.init.xavier_uniform_(self.convB4.weight)
        self.convB5 = nn.Conv1d(512, 1024, 1)
        self.convB6 = nn.Conv1d(1024, 2048, 1)

        self.avgpool = nn.AdaptiveAvgPool1d(1).to('cuda:2')
    '''
    # Makes/ builds each layer of the network  
    '''
    def create_layers(self, block,inChannel, planes, blocks, stride=1, downsample=None):
        
        batchNormalLayer = self.batch_normal_layer
        dialtion = self.dialation

        # multiply dialation by stride
        dialtion *= stride

        layers = []
        layers.append(block(inChannel, planes, stride, downsample,dialtion))

        for _ in range(1,blocks):
            layers.append(block(self.inplanes, planes, stride, downsample,dialtion))

        return nn.Sequential(*layers)


    def forward(self,x):
        '''
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.avgpool(x)

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool2(x)

        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool3(x)
        '''
        x=x.float()

        x = self.convB1(x.to('cuda:1'))
        x = self.lReLu(self.BNL0(x))
        x = self.convB2(x.to('cuda:1'))
        x = self.lReLu(self.BNL1(x))
        x = self.convB3(x.to('cuda:1'))
        x = self.lReLu(self.BNL2(x))
        x = self.convB4(x.to('cuda:2'))
        x = self.lReLu(self.BNL3(x))
        
        '''
        x = self.convB5(x)
        x = self.lReLu(self.BNL4(x))
        x = self.convB6(x)
        x = self.lReLu(self.BNL5(x))
        '''
        x = self.avgpool(x)



        return(x)

class get_loss(torch.nn.Module):
    def __init__(self):
        super(get_loss,self).__init__()
        self.criterion = nn.MSELoss()
        #self.criterion = nn.KLDivLoss()

    def forward(self, predDepth, predIntensity, ptCloud, grayImage, targetTransform, device):

        """
        Make sure that all the variables are in the same device
        """
        predDepthRot = predDepth[1]
        predDepthT   = predDepth[0]

        predIntensityRot = predIntensity[1]
        predIntensityT = predIntensity[0]

        if predDepthRot.get_device() != device:
            predDepthRot = predDepthRot.to('cuda:'+str(device))
            predDepthT = predDepth[1].to('cuda:'+str(device))
        
        if predIntensity[0].get_device() != device:
            predIntensityRot = predIntensityRot.to('cuda:'+str(device))
            predIntensityT = predIntensityT.to('cuda:'+str(device))

        if ptCloud.get_device() != device:
            ptCloud = ptCloud.to('cuda:'+str(device))

        if targetTransform.get_device() != device:
            targetTransform = targetTransform.to('cuda:'+str(device))

        if grayImage.get_device() != device:
            grayImage = grayImage.to('cuda:'+str(device))
 
        # Read the calibration parameters
        calibFileRootDir = "/mnt/291d3084-ca91-4f28-8f33-ed0b64be0a8c/akshay/kitti/raw/train/2011_09_26"
        [P_rect, R_rect, R, T] = findtheCalibparameters(calibFileRootDir)

        # Convert the NumPy array to Tensor
        P_rect = torch.from_numpy(P_rect)
        R_rect = torch.from_numpy(R_rect)
        RT = createRTMat(R,T)
        RT = torch.from_numpy(RT)

        # Correct the point cloud 
        # Detach the intensities and attach the unit coloumn 
        intensity = ptCloud[:,:,3]
        ptCloud = ptCloud[:,:,:3]
        ones = torch.ones((ptCloud.shape[0],ptCloud.shape[1],1)).to('cuda:'+str(ptCloud.get_device()))

        ptCloud = torch.cat([ptCloud,ones],dim=2)
        ptCloud = torch.transpose(ptCloud, 2,1)
        
        # Corecting for RT

        ptCloud = torch.matmul(RT.to('cuda:'+str(ptCloud.get_device())),ptCloud[:])

        # Correcting for rotation cam R00  
        ptCloud = torch.matmul(R_rect.to('cuda:'+str(ptCloud.get_device())), ptCloud)

        # Use this Point cloud as the target point cloud to be achieved after multiplying the point cloud by
        # predicted transforms
        ptCloudTarget = ptCloud

        # Create the transformation function 
        predDepthTransform = createRTMatTensor(predDepthRot, predDepthT)
        predIntensityTransform = createRTMatTensor(predIntensityRot, predIntensityT)

        """

        # Test to check if the implementation of the rotationmatrix is correct
        predDepthTransformRef = euler_angles_to_matrix(predDepth[1],"ZXY")
        predIntensityTransformRef = euler_angles_to_matrix(predIntensity[1],"ZXY")

        if torch.eq(predDepthTransformRef,predDepthTransform[:,:3,:3]):
            print("The calculated Rotation matrix for the depth prediction is correct")
        else:
            print("The calculated Rotation matrix for the depth prediction is incorrect")

        if torch.eq(predIntensityTransform[:,:3,:3],predIntensityTransformRef):
            print("The calculated Rotation matrix for the intensity prediction is correct")
        else:
            print("The calculated Rotation matrix for the intensity prediction is incorrect")

        """


        # Inverse of the target transform
        invTargetRT = calculateInvRTTensor(targetTransform)

        # Extract the translation from target transform
        targetT = targetTransform[:,:3,3].unsqueeze(1)

        # Calculate the distance between the target and the predicted
        euclideanDistanceIntensity = calculateEucledianDistTensor(predIntensityT,targetT)
        euclideanDistanceDepth = calculateEucledianDistTensor(predDepthT, targetT)

        # Calculate the angular distance between the target and predicted
        targetR = matrix_to_euler_angles(targetTransform[:,:3,:3],"ZXY").unsqueeze(1)

        euclideanAngularDistanceDepth = calculateEucledianDistTensor(torch.rad2deg(predDepthRot), torch.rad2deg(targetR))
        euclideanAngularDistanceIntensity = calculateEucledianDistTensor(torch.rad2deg(predIntensityRot), torch.rad2deg(targetR))

        # One component of the loss function 
        # Eucliedian depth loss
        lossEDD = (0.7*euclideanAngularDistanceDepth) + (0.3*euclideanDistanceDepth)
        # Eucliedian Intensity loss
        lossEDI = (0.7*euclideanAngularDistanceIntensity) + (0.3*euclideanDistanceIntensity)

        lossEuclideanDistanceBtwTransform = lossEDD + lossEDI

        # Create cross correlation
        # Setp 0: Project the points that are rectified by inv of target transform
        # Step 1: Multiply the point clouds by the predicted value
        # Step 2: Caluclate the image tensor for target and predicted point cloud
        # Step 3: Caluclate the cross correlation of the target image and the predicted image
        # Step 4: Caluclate the maximum likelyhood summation value for a predefined radius of correlated value

        # Step 0
        # extract only the XYZ from the point cloud 
        ptCloud = torch.transpose(ptCloud,2,1)[:,:,:3]
        ptCloud = torch.cat([ptCloud,ones],dim=2)
        ptCloud = torch.transpose(ptCloud, 2,1)

        ptCloudTarget = ptCloud

        # Use this point cloud as the base for all the future caluclations 
        ptCloudBase = torch.matmul(invTargetRT, ptCloud)

        ptCloudBase = torch.transpose(ptCloudBase,2,1)[:,:,:3]
        ptCloudBase = torch.cat([ptCloudBase,ones],dim=2)
        ptCloudBase = torch.transpose(ptCloudBase, 2,1)

        # Step 1
        # Multiply the base point cloud with the predicted transponse 
        finalDepthPredPtCld = torch.matmul(predDepthTransform, ptCloudBase.type(torch.float))
        finalIntensityPredCld = torch.matmul(predIntensityTransform, ptCloudBase.type(torch.float))

        # Calculate Euclidean Distance between the 
        euclideanDistanceDepthPtCld = calculateEucledianDistOfPointClouds(torch.transpose(finalDepthPredPtCld,2,1)[:,:,:3], torch.transpose(ptCloudTarget,2,1)[:,:,:3])
        euclideanDistanceIntensityPtCld = calculateEucledianDistOfPointClouds(torch.transpose(finalIntensityPredCld,2,1)[:,:,:3], torch.transpose(finalIntensityPredCld,2,1)[:,:,:3])


        """

        # Step 2
        # Get the image tensor by projecting the point cloud back to image plane
        # These points are in the image coordinate frame
        targetPredPtCldImgCord = getImageTensorFrmPtCloud(P_rect, ptCloudTarget)
        finalDepthPredPtCldImgCord = getImageTensorFrmPtCloud(P_rect.type(torch.float), finalDepthPredPtCld)
        finalIntensityPredCldImgCord = getImageTensorFrmPtCloud(P_rect.type(torch.float), finalIntensityPredCld)

        # Transpose the vectors to create a mask
        targetPredPtCldImgCord = torch.transpose(targetPredPtCldImgCord,2,1)
        finalDepthPredPtCldImgCord = torch.transpose(finalDepthPredPtCldImgCord,2,1)
        finalIntensityPredCldImgCord = torch.transpose(finalIntensityPredCldImgCord,2,1)

        # Now filter the points that are not in front of the camera 
        imgHeight = 375
        imgWidth = 1242

        # Replace the 4th coloum of pt by intensities
        ptCloudTarget = torch.cat((torch.transpose(ptCloudTarget,2,1)[:,:,:3], torch.unsqueeze(intensity,dim=2)),dim=2)
        finalDepthPredPtCld = torch.cat((torch.transpose(finalDepthPredPtCld,2,1)[:,:,:3], torch.unsqueeze(intensity,dim=2)),dim=2)
        finalIntensityPredCld = torch.cat((torch.transpose(finalIntensityPredCld,2,1)[:,:,:3], torch.unsqueeze(intensity,dim=2)),dim=2)

        targetImgCoord, targetPtCld = filterPtClds(targetPredPtCldImgCord, ptCloudTarget, imgHeight, imgWidth)
        finalDepthImgCoord, finalDepthPredPtCld = filterPtClds(finalDepthPredPtCldImgCord, finalDepthPredPtCld, imgHeight, imgWidth)
        finalIntImgCoord, finalIntensityPredCld = filterPtClds(finalIntensityPredCldImgCord, finalIntensityPredCld, imgHeight, imgWidth)
        
        # create Depth Image tensor
        targetDepthTensor = createImage(targetImgCoord, targetPtCld[:,:,2],imgWidth, imgHeight)
        targetIntensityTensor = createImage(targetImgCoord, targetPtCld[:,:,3],imgWidth, imgHeight)
        depthTensor = createImage(finalDepthImgCoord,finalDepthPredPtCld[:,:,2], imgWidth, imgHeight)
        IntensityTensor = createImage(finalIntImgCoord,finalIntensityPredCld[:,:,3], imgWidth, imgHeight)


        
        # Create a sobel Kernel to run thru the image
        edgeDepthTensor = applySobelOperator(depthTensor)
        edgeintensityTensor = applySobelOperator(IntensityTensor)
        grayImage = applySobelOperator(grayImage.type(torch.float))


        # Cross-Correlation
        nccDepth = NCC(torch.transpose(targetDepthTensor,3,1))
        nccIntensity = NCC(targetIntensityTensor)

        # Move it cuda 
        nccDepth = nccDepth.to('cuda:'+str(device))
        nccIntensity = nccIntensity.to('cuda:'+str(device))

        crossCorrelationDepth = nccDepth(torch.transpose(depthTensor[None,...],3,1))
        crossCorrelationIntensity = nccIntensity(IntensityTensor[None,...])

        # Get MaxLikely hood sum
        """

        euclideanDistanceLoss = torch.div(torch.mean(euclideanDistanceDepthPtCld)+torch.mean(euclideanDistanceIntensityPtCld),2)

        totalLOSS = euclideanDistanceLoss + lossEuclideanDistanceBtwTransform
        

        return(torch.mean(totalLOSS))


