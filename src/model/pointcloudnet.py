import torch
import torch.nn as nn


def conv(in_channels, out_channels, kernel_size=3, stride_len = 1, dilation = 1):
    return nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, 
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
        normalization_layer = nn.BatchNorm2d
        self.conv_block1 = conv(in_channels,out_channels,stride_len=stride)
        self.batchnormal_1 = normalization_layer(out_channels)
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.conv_block2 = conv(out_channels,out_channels)
        self.batchnormal_2 = normalization_layer(out_channels)
        self.downsample = downsample 
        self.stride = stride

    def forward(self,x):
        identiy = x

        out = self.conv_block1(x)
        out = self.batchnormal_1(out)
        out = self.lrelu(out)

        out = self.conv_block2(out)
        out = self.batchnormal_2(out)

        if self.downsample is not None:
            identiy = self.downsample(x)

        out += identiy
        out = lrelu(out)

        return (out)

class pointcloudnet(nn.Module):
    def __init__(self,block,layers):
        super(pointcloudnet,self).__init__()

        # standard parameters initialization
        self.inplanes = 64
        self.dialation = 1
        self.groups = 1
        self.base_wodth = 64

        # Need to start with a conv layer
        self.layer0 = self.create_layers(BasicBlock,64,layers[0])
        self.layer1 = self.create_layers(BasicBlock,128,layers[1])
        self.layer2 = self.create_layers(BasicBlock,256,layers[2]) 
        self.layer3 = self.create_layers(BasicBlock,512,layers[3]) 
        self.layer4 = self.create_layers(BasicBlock,1024,layers[4]) 
        self.layer5 = self.create_layers(BasicBlock,2018,layers[5])

        # Using Batch normalization: Define normalization layer
        self.batch_normal_layer = nn.BatchNorm2d

        self.leakyReLU_layer = nn.LeakyReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

    '''
    # Makes/ builds each layer of the network  
    '''
    def create_layers(self, block, planes, blocks, stride=1, downsample=None):
        
        batch_normal_layer = self.batch_normal_layer
        dialtion = self.dialation

        # multiply dialation by stride
        dialtion *= stride

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,dialtion))

        for _ in range(1,blocks):
            layers.append(block(self.inplanes, planes, stride, downsample,dialtion))

        return nn.Sequential(*layers)


    def forward(self,x):
        
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.avgpool(x)

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)

        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)

        return(x)






