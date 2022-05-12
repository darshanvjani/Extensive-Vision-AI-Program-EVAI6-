from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms




class Net(nn.Module):
    def __init__(self,normalization_tech):
        super(Net, self).__init__()
        self.norm_tech = normalization_tech
        #################################   BN   #################################  
        self.conv1block_b = nn.Sequential(    # 28 -> 26 | RF:3
            nn.Conv2d(1,10,3,padding=0,bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout2d(0.03)
        )
        self.conv2block_b = nn.Sequential(    # 26 -> 24 | RF:5
            nn.Conv2d(10,18,3,padding=0,bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(18),
            nn.Dropout2d(0.03)
        )
        # self.conv3block = nn.Sequential(    # 24 -> 22 | RF:7
        #     nn.Conv2d(16,32,3,padding=0,bias=False),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(32),
        #     nn.Dropout2d(0.07)
        # )
        
        self.pool1 = nn.MaxPool2d(2,2)    # 24 -> 12 | RF:10
        
        self.trans1block = nn.Sequential(    # 12 -> 12 | RF:12
            nn.Conv2d(18,10,1,padding=0,bias=False),
            nn.ReLU()
        )

        self.conv4block_b = nn.Sequential(    # 12 -> 10 | RF:14
            nn.Conv2d(10,18,3,padding=0,bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(18),
            nn.Dropout2d(0.03)
        )
        self.conv5block_b = nn.Sequential(    # 10 -> 8 | RF:16
            nn.Conv2d(18,18,3,padding=0,bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(18),
            nn.Dropout2d(0.03)
        )
        self.conv6block_b = nn.Sequential(    # 8 -> 8 | RF:18
            nn.Conv2d(18,18,3,padding=1,bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(18),
            nn.Dropout2d(0.03)
        )

        self.outputblock = nn.Sequential(    # 8 -> 1 | RF:18
            nn.AvgPool2d(8),
            nn.Conv2d(18,20,1,padding=0,bias=False),
            nn.ReLU(),
            nn.Conv2d(20,10,1,padding=0,bias=False)
        )

        #################################   GN   #################################

        self.conv1block_g = nn.Sequential(    # 28 -> 26 | RF:3
            nn.Conv2d(1,10,3,padding=0,bias=False),
            nn.ReLU(),
            nn.GroupNorm(5,10),
            nn.Dropout2d(0.03)
        )
        self.conv2block_g = nn.Sequential(    # 26 -> 24 | RF:5
            nn.Conv2d(10,18,3,padding=0,bias=False),
            nn.ReLU(),
            nn.GroupNorm(9,18),
            nn.Dropout2d(0.03)
        )
        # self.conv3block = nn.Sequential(    # 24 -> 22 | RF:7
        #     nn.Conv2d(16,32,3,padding=0,bias=False),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(32),
        #     nn.Dropout2d(0.07)
        # )
        
        # self.pool1 = nn.MaxPool2d(2,2)    # 24 -> 12 | RF:10
        
        # self.trans1block = nn.Sequential(    # 12 -> 12 | RF:12
        #     nn.Conv2d(17,10,1,padding=0,bias=False),
        #     nn.ReLU()
        # )

        self.conv4block_g = nn.Sequential(    # 12 -> 10 | RF:14
            nn.Conv2d(10,18,3,padding=0,bias=False),
            nn.ReLU(),
            nn.GroupNorm(9,18),
            nn.Dropout2d(0.03)
        )
        self.conv5block_g = nn.Sequential(    # 10 -> 8 | RF:16
            nn.Conv2d(18,18,3,padding=0,bias=False),
            nn.ReLU(),
            nn.GroupNorm(9,18),
            nn.Dropout2d(0.03)
        )
        self.conv6block_g = nn.Sequential(    # 8 -> 8 | RF:18
            nn.Conv2d(18,18,3,padding=1,bias=False),
            nn.ReLU(),
            nn.GroupNorm(9,18),
            nn.Dropout2d(0.03)
        )

        # self.outputblock = nn.Sequential(    # 8 -> 1 | RF:18
        #     nn.AvgPool2d(6),
        #     nn.Conv2d(18,20,1,padding=0,bias=False),
        #     nn.ReLU(),
        #     nn.Conv2d(20,10,1,padding=0,bias=False)
        # )
        #################################   LN   #################################
        
        self.conv1block_l = nn.Sequential(    # 28 -> 26 | RF:3
            nn.Conv2d(1,10,3,padding=0,bias=False),
            nn.ReLU(),
            nn.LayerNorm([10,26,26]),
            nn.Dropout2d(0.03)
        )
        self.conv2block_l = nn.Sequential(    # 26 -> 24 | RF:5
            nn.Conv2d(10,18,3,padding=0,bias=False),
            nn.ReLU(),
            nn.LayerNorm([18,24,24]),
            nn.Dropout2d(0.03)
        )
        # self.conv3block = nn.Sequential(    # 24 -> 22 | RF:7
        #     nn.Conv2d(16,32,3,padding=0,bias=False),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(32),
        #     nn.Dropout2d(0.07)
        # )
        
        # self.pool1 = nn.MaxPool2d(2,2)    # 24 -> 12 | RF:10
        
        # self.trans1block = nn.Sequential(    # 12 -> 10 | RF:12
        #     nn.Conv2d(17,10,1,padding=0,bias=False),
        #     nn.ReLU()
        # )

        self.conv4block_l = nn.Sequential(    # 12 -> 10 | RF:14
            nn.Conv2d(10,18,3,padding=0,bias=False),
            nn.ReLU(),
            nn.LayerNorm([18,10,10]),
            nn.Dropout2d(0.03)
        )
        self.conv5block_l = nn.Sequential(    # 10 -> 8 | RF:16
            nn.Conv2d(18,18,3,padding=0,bias=False),
            nn.ReLU(),
            nn.LayerNorm([18,8,8]),
            nn.Dropout2d(0.03)
        )
        self.conv6block_l = nn.Sequential(    # 8 -> 8 | RF:18
            nn.Conv2d(18,18,3,padding=1,bias=False),
            nn.ReLU(),
            nn.LayerNorm([18,8,8]),
            nn.Dropout2d(0.03)
        )

        print("Normalization Technique: ",self.norm_tech)
        
        # self.outputblock = nn.Sequential(    # 4 -> 1 | RF:18
        #     nn.AvgPool2d(6),
        #     nn.Conv2d(18,20,1,padding=0,bias=False),
        #     nn.ReLU(),
        #     nn.Conv2d(20,10,1,padding=0,bias=False)
        # )

    def forward(self, x):
        if self.norm_tech == "BN":
            x = self.conv1block_b(x)
            x = self.conv2block_b(x)
            x = self.pool1(x)
            x = self.trans1block(x)
            x = self.conv4block_b(x)
            x = self.conv5block_b(x)
            x = self.conv6block_b(x)
            x = self.outputblock(x)

            x = x.view(-1, 10)
            return F.log_softmax(x, dim=-1)
        elif self.norm_tech == "GN":
            x = self.conv1block_g(x)
            x = self.conv2block_g(x)
            x = self.pool1(x)
            x = self.trans1block(x)
            x = self.conv4block_g(x)
            x = self.conv5block_g(x)
            x = self.conv6block_g(x)
            x = self.outputblock(x)

            x = x.view(-1, 10)
            return F.log_softmax(x, dim=-1)
        elif self.norm_tech == "LN":
            x = self.conv1block_l(x)
            x = self.conv2block_l(x)
            x = self.pool1(x)
            x = self.trans1block(x)
            x = self.conv4block_l(x)
            x = self.conv5block_l(x)
            x = self.conv6block_l(x)
            x = self.outputblock(x)

            x = x.view(-1, 10)
            return F.log_softmax(x, dim=-1)
        