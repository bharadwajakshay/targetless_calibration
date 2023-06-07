import torch
import torch.nn as nn
import math

kernelSize = 3
padding = (0,0)
stride = (1,1)

class crossAttention(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.q = nn.Conv2d(2048,1024,kernelSize,padding=padding,stride=stride)
        self.k = nn.Conv2d(2048,1024,kernelSize,padding=padding,stride=stride)
        self.v = nn.Conv2d(2048,1024,kernelSize,padding=padding,stride=stride)
        self.qkv = nn.Conv2d(1024,1024,kernelSize,padding=padding,stride=stride)

        self.softmax = nn.Softmax2d()


    def forward(self,x,y):
        q = self.q(x)
        k = self.k(x)
        v = self.v(y)

        QK = torch.matmul(q,k.transpose(3,2))
        
        # need to add scalling
        QK = torch.div(QK,math.sqrt(2048))

        # Apply softmax
        QK = self.softmax(QK)

        QKV = torch.matmul(QK, v)

        QKV = self.qkv(QKV)

        return(QKV)





        