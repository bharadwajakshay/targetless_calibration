import torch
import torch.nn as nn



class regressor(nn.Module):
    def __init__(self):
        super(regressor,self).__init__()

        # The input to the feature matching network will be [24 2 512]
        # Setting the kernel size to 1
        channels = 2
        self.conv1x1B0 = nn.Conv1d(2, 64,1)
        nn.init.xavier_uniform(self.conv1x1B0.weight)
        self.conv1x1B1 = nn.Conv1d(64, 128,1)
        nn.init.xavier_uniform(self.conv1x1B1.weight)
        self.conv1x1B2 = nn.Conv1d(128, 256,1)
        nn.init.xavier_uniform(self.conv1x1B2.weight)

        self.FC0 = nn.Linear(256,128)
        nn.init.xavier_uniform(self.FC0.weight)
        self.FC1 = nn.Linear(128,64)
        nn.init.xavier_uniform(self.FC1.weight)
        self.FC2 = nn.Linear(64,7)
        nn.init.xavier_uniform(self.FC2.weight)
        self.bn0 = nn.BatchNorm1d(64)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.lRelu = nn.ReLU(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool1d(256)
        self.avgpool_regress = nn.AdaptiveAvgPool1d(1)


    def forward(self, x):
        """
        Feature matching
        """
        x = self.conv1x1B0(x)
        x = self.lRelu(self.bn0(x))
        x = self.conv1x1B1(x)
        x = self.lRelu(self.bn1(x))
        x = self.conv1x1B2(x)
        x = self.lRelu(self.bn2(x))
        x = self.avgpool(x)

        """
        Regression
        """

        x = self.FC0(x)
        x = self.FC1(x)
        x = self.FC2(x)

        x = self.avgpool_regress(x.transpose(2,1))

        return(x)





         

         