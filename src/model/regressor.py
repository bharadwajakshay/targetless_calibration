import torch
import torch.nn as nn


class featureMatchingNW(nn.Module):
    def __init__(self, backbone='RESNET'):
        super(featureMatchingNW,self).__init__()

        self.backbone = backbone

        kernelSize = 3
        padding = (1,1)
        stride = (1,1)

        if self.backbone == 'RESNET':
            # The input to the feature matching network will be [batch size, 1, 4096]
            # Setting the kernel size to 3
            channels = 4096

        elif self.backbone == 'SWIN':
            print('Using regressor for SWIN')
            channels = 128
        

        ## Feature Matching Network 
        # With weight initilization 
        self.conv1x1B0 = nn.Conv2d(channels, 2048, kernelSize, padding=padding, stride=stride)
        nn.init.xavier_uniform_(self.conv1x1B0.weight)
        self.conv1x1B1 = nn.Conv2d(2048, 1024,kernelSize, padding=padding, stride=stride)
        nn.init.xavier_uniform_(self.conv1x1B1.weight)
        self.conv1x1B2 = nn.Conv2d(1024, 512,kernelSize, padding=padding, stride=stride)
        nn.init.xavier_uniform_(self.conv1x1B2.weight)
        self.conv1x1B3 = nn.Conv2d(512, 256, kernelSize, padding=padding, stride=stride)
        nn.init.xavier_uniform_(self.conv1x1B3.weight)
        self.conv1x1B4 = nn.Conv2d(256, 256,kernelSize, padding=padding, stride=stride)
        nn.init.xavier_uniform_(self.conv1x1B4.weight)
        self.conv1x1B5 = nn.Conv2d(256, 256,kernelSize, padding=padding, stride=stride)
        nn.init.xavier_uniform_(self.conv1x1B5.weight)

        # Batch Normalization 
        self.bn0 = nn.BatchNorm2d(2048)
        self.bn1 = nn.BatchNorm2d(1024)
        self.bn2 = nn.BatchNorm2d(512)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(256)

        self.Relu = nn.ReLU(inplace=True)

    def forward(self, x):

        """
        Feature matching network
        """
        x = self.conv1x1B0(x)
        x = self.bn0(x)
        x = self.Relu(x)

        x = self.conv1x1B1(x)
        x = self.bn1(x)
        x = self.Relu(x)
        
        x = self.conv1x1B2(x)
        x = self.bn2(x)
        x = self.Relu(x)

        x = self.conv1x1B3(x)
        x = self.bn3(x)
        x = self.Relu(x)

        x = self.conv1x1B4(x)
        x = self.bn4(x)
        x = self.Relu(x)
        
        x = self.conv1x1B5(x)
        x = self.bn5(x)
        x = self.Relu(x)
        
        return(x)


class regressionTransNw(nn.Module):
    def __init__(self):
        super(regressionTransNw,self).__init__()

        channels = 256
        kernelSize = 5
        padding = (2,2)
        stride = (1,2)

        self.conv1x1B0 = nn.Conv2d(channels, 512, kernelSize, padding=padding, stride=stride)
        nn.init.xavier_uniform_(self.conv1x1B0.weight)

        self.conv1x1B1 = nn.Conv2d(512, 256, kernelSize, padding=padding, stride=stride)
        nn.init.xavier_uniform_(self.conv1x1B0.weight)

        self.conv1x1B2 = nn.Conv2d(256, 128, kernelSize, padding=padding, stride=stride)
        nn.init.xavier_uniform_(self.conv1x1B2.weight)

        # BatchNormalization
        self.bn0 = nn.BatchNorm2d(512)
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(128)

        self.Relu = nn.ReLU(inplace=True)

        # Linear FC network Layers
        self.FC0 = nn.Linear(7680,3840)
        nn.init.xavier_uniform_(self.FC0.weight)
        
        self.FC1 = nn.Linear(3840,3)
        nn.init.xavier_uniform_(self.FC1.weight)

        # Dropout layer
        self.dropoutLayer = torch.nn.Dropout(p=0.7)


    def forward(self,x):

        x = self.conv1x1B0(x)
        x = self.bn0(x)
        x = self.Relu(x)
        x = self.conv1x1B1(x)
        x = self.bn1(x)
        x = self.Relu(x)
        x = self.conv1x1B2(x)
        x = self.bn2(x)
        x = self.Relu(x)

        x = torch.flatten(x,start_dim=1)

        x = self.FC0(x)

        x = self.dropoutLayer(x)

        x = self.FC1(x)

        return(x)

class regressionRotNw(nn.Module):
    def __init__(self):
        super(regressionRotNw,self).__init__()

        channels = 256
        kernelSize = 5
        padding = (2,2)
        stride = (1,2)

        self.conv1x1B0 = nn.Conv2d(channels, 512, kernelSize, padding=padding, stride=stride)
        nn.init.xavier_uniform_(self.conv1x1B0.weight)

        self.conv1x1B1 = nn.Conv2d(512, 256, kernelSize, padding=padding, stride=stride)
        nn.init.xavier_uniform_(self.conv1x1B1.weight)

        self.conv1x1B2 = nn.Conv2d(256, 128, kernelSize, padding=padding, stride=stride)
        nn.init.xavier_uniform_(self.conv1x1B2.weight)

        # BatchNormalization
        self.bn0 = nn.BatchNorm2d(512)
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(128)

        self.Relu = nn.ReLU(inplace=True)

        # Linear FC network Layers
        self.FC0 = nn.Linear(7680,3840)
        nn.init.xavier_uniform_(self.FC0.weight)
        self.FC1 = nn.Linear(3840,4)
        nn.init.xavier_uniform_(self.FC1.weight)

        # Dropout layer
        self.dropoutLayer = torch.nn.Dropout(p=0.7)


    def forward(self,x):

        x = self.conv1x1B0(x)
        x = self.bn0(x)
        x = self.Relu(x)
 
        x = self.conv1x1B1(x)
        x = self.bn1(x)
        x = self.Relu(x)

        x = self.conv1x1B2(x)
        x = self.bn2(x)
        x = self.Relu(x)

        x = torch.flatten(x,start_dim=1)

        x = self.FC0(x)
        x = self.dropoutLayer(x)
        x = self.FC1(x)


        return(x)        

class regressor(nn.Module):
    def __init__(self):
        super(regressor,self).__init__()

        self.featurematching = featureMatchingNW()
        self.regressionRot = regressionRotNw()
        self.regressionTrans = regressionTransNw()


    def forward(self, x):
        x = self.featurematching(x)
        xR = self.regressionRot(x)
        xT = self.regressionTrans(x)

        tR = torch.cat([xR,xT],dim=1)
        
        return(tR)





         

         