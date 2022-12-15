import torch
import torch.nn as nn
from YourOwnNetxx import YourOwnNetxx
from customNetwork import customNetworkObjDetection


class mymodel(nn.Module):
    def __init__(self):
        super(mymodel,self).__init__()

        # Backbone Network
        self.YourOwnNetxx = YourOwnNetxx()
        # Custom Network for object detection
        self.customNetworkObjDetection = customNetworkObjDetection()

    def forward(self,x):
        with torch.no_grad():
            self.YourOwnNetxx = self.YourOwnNetxx.eval()
            x = self.YourOwnNetxx(x)
        self.customNetworkObjDetection = self.customNetworkObjDetection.weight.require_grad(False)
        x = self.customNetworkObjDetection(x)
        return(x)


