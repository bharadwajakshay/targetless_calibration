import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation as rot
import numpy as np


def conv(in_channels, out_channels, kernel_size=3, stride_len = 1, dilation = 1):
    return nn.Conv1d(in_channels, out_channels, kernel_size = kernel_size, 
                stride = stride_len, padding=dilation, groups=1, bias= False,
                dilation=dilation)


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
        self.BNL0 = nn.BatchNorm1d(64)
        self.BNL1 = nn.BatchNorm1d(128)
        self.BNL2 = nn.BatchNorm1d(256)
        self.BNL3 = nn.BatchNorm1d(512)
        self.BNL4 = nn.BatchNorm1d(1024)
        self.BNL5 = nn.BatchNorm1d(2048)
        self.lReLu = nn.LeakyReLU(inplace=True)

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
        self.convB1 = nn.Conv1d(input_channel, 64, 1)
        self.convB2 = nn.Conv1d(64, 128, 1)
        self.convB3 = nn.Conv1d(128, 256, 1)
        self.convB4 = nn.Conv1d(256, 512, 1)
        self.convB5 = nn.Conv1d(512, 1024, 1)
        self.convB6 = nn.Conv1d(1024, 2048, 1)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
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

        x = self.convB1(x)
        x = self.lReLu(self.BNL0(x))
        x = self.convB2(x)
        x = self.lReLu(self.BNL1(x))
        x = self.convB3(x)
        x = self.lReLu(self.BNL2(x))
        x = self.convB4(x)
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

    def forward(self, prediction, target, inPutCldTensor, targetCldTensor):
        '''
        Euclidean distance as loss functions
        Calculate Euclidean distance of each point and calculate the mean euclidean distance

        Assuming prediction is a 7x1 vector 
        '''
        '''
        Calculate Euler distace between point clouds
        '''
        '''
        # copy variables from CUDA to cpu
        prediction = prediction.squeeze(dim=0).cpu()
        inPutCldTensor = inPutCldTensor.transpose(2,1).cpu()
        targetCldTensor = targetCldTensor.cpu()

        prediction = prediction.data.numpy()
        inPutCld = inPutCldTensor.data.numpy()
        targetCld = targetCldTensor.data.numpy()
        
        meanEucledeanDist =[]

        for idx in range(0,prediction.shape[0]):
            xyz = prediction[idx,0:3,:]
            quat = prediction[idx,3:,:].flatten()

            rObj = rot.from_quat(quat)
            R = rObj.as_matrix()

            # Since we need to caluclate the inverese of teh transformation 
            # The transformation can be found here 
            # https://math.stackexchange.com/questions/152462/inverse-of-transformation-matrix
        
            invR = R.transpose()
            invT = np.matmul(-invR,xyz)

            R_T = np.vstack((np.hstack((invR,invT)),[0, 0, 0, 1.]))

            # pad the pointcloud
            ones = np.ones(inPutCld.shape[1]).reshape(inPutCld.shape[1],1)
            paddedinPutCld = np.hstack((inPutCld[idx,:,:], ones))
            transformedPtCld = np.matmul(R_T, paddedinPutCld.T)
            transformedPtCld = transformedPtCld.T[:,:3]
            targetCloudIdx = targetCld[idx,:,:]

            # calculate the eucledean distance between the the transformed and target point cloud
            eucledeanDist = np.linalg.norm(transformedPtCld[:] - targetCloudIdx[:],axis=1)
            meanEucledeanDist.append(np.average(eucledeanDist))
        '''
        target = target.float()

        loss = self.criterion(input=prediction, target=target)

        #return(torch.tensor(np.average(meanEucledeanDist)))
        return(loss)


