import torch
import torch.nn as nn



class regressor(nn.Module):
    def __init__(self):
        super(regressor,self).__init__()

        # The input to the feature matching network will be [batch size, 1, 4096]
        # Setting the kernel size to 3
        channels = 3
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
        self.FC1 = nn.Linear(2042,2042)
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
        print(x.shape)
        x = self.bn0(x)
        print(x.shape)
        x = self.Relu(x)
        print(x.shape)
        x = self.conv1x1B1(x)
        print(x.shape)
        x = self.bn1(x)
        print(x.shape)
        x = self.Relu(x)
        print(x.shape)
        x = self.conv1x1B2(x)
        print(x.shape)
        x = self.bn2(x)
        print(x.shape)
        x = self.Relu(x)
        print(x.shape)
        x = self.avgpool(x.transpose(1,2))
        print(x.shape)

        """
        Regression network translation
        """
        xT = self.FC1(x.transpose(2,1))
        print(xT.shape)
        xT = self.avgpoolRegress0(xT)
        print(xT.shape)
        xT = self.FC2(xT)
        print(xT.shape)
        xT = self.avgpoolRegress1(xT)
        print(xT.shape)
        xT = self.FC3(xT)
        print(xT.shape)
        xT = self.avgpoolRegress2(xT)
        print(xT.shape)

        """
        Regression network Rotation
        """
        # x = x.squeeze(dim=2)
        xR = self.FC1(x.transpose(2,1))
        print(xR.shape)
        xR = self.avgpoolRegress0(xR)
        print(xR.shape)
        xR = self.FC2(xR)
        print(xR.shape)
        xR = self.avgpoolRegress1(xR)
        print(xR.shape)
        xR = self.FC3(xR)
        print(xR.shape)
        xR = self.avgpoolRegress2(xR)
        print(xR.shape)

        tR = torch.cat((xT, xR),dim=2)

        return(tR)





         

         