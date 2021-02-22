import torch
import torch.nn as nn



class regressor(nn.Module):
    def __init__(self):
        super(regressor,self).__init__()

        # The input to the feature matching network will be [batch size, 1, 4096]
        # Setting the kernel size to 3
        channels = 1
        kernelSize = 3

        ## Feature Matching Network 
        # With weight initilization 
        self.conv1x1B0 = nn.Conv1d(channels, 8, kernelSize)
        nn.init.xavier_uniform_(self.conv1x1B0.weight)
        self.conv1x1B1 = nn.Conv1d(8, 16,kernelSize)
        nn.init.xavier_uniform_(self.conv1x1B1.weight)
        self.conv1x1B2 = nn.Conv1d(16, 32,kernelSize)
        nn.init.xavier_uniform_(self.conv1x1B2.weight)
        self.FC0 = nn.Linear(32,32)
        nn.init.xavier_uniform_(self.FC0.weight)

        # Batch Normalization 
        self.bn0 = nn.BatchNorm1d(8)
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(32)
        
        # Linear FC network Layers
        self.FC1 = nn.Linear(4090,2048)
        nn.init.xavier_uniform_(self.FC1.weight)
        self.FC2 = nn.Linear(1024,512)
        nn.init.xavier_uniform_(self.FC2.weight)
        self.FC3 = nn.Linear(256,128)

        self.Relu = nn.ReLU(inplace=True)
        self.lRelu = nn.LeakyReLU(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.avgpoolRegress0 = nn.AdaptiveAvgPool1d(1024)
        self.avgpoolRegress1 = nn.AdaptiveAvgPool1d(256)
        self.avgpoolRegress2 = nn.AdaptiveAvgPool1d(3)



    def forward(self, x):
        """
        Feature matching network
        """
        x = self.conv1x1B0(x)
        x = self.Relu(self.bn0(x))
        x = self.conv1x1B1(x)
        x = self.Relu(self.bn1(x))
        x = self.conv1x1B2(x)
        x = self.Relu(self.bn2(x))
        x = self.avgpool(x.transpose(1,2))

        """
        Regression network translation
        """
        xT = self.FC1(x.transpose(2,1))
        xT = self.avgpoolRegress0(xT)
        xT = self.FC2(xT)
        xT = self.avgpoolRegress1(xT)
        xT = self.FC3(xT)
        xT = self.avgpoolRegress2(xT)

        """
        Regression network Rotation
        """
        # x = x.squeeze(dim=2)
        xR = self.FC1(x.transpose(2,1))
        xR = self.avgpoolRegress0(xR)
        xR = self.FC2(xR)
        xR = self.avgpoolRegress1(xR)
        xR = self.FC3(xR)
        xR = self.avgpoolRegress2(xR)

        return(xT, xR)





         

         