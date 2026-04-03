import os 
import numpy as np
import pandas as pd
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(project_root)

class GranulatedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_granules=4):
        super(GranulatedBlock, self).__init__()
        self.num_granules = num_granules
        assert out_channels % num_granules == 0, "out_channels must be divisible by num_granules"
        
        self.granule_ch = out_channels // num_granules
    
        self.granule_convs = nn.ModuleList([
            nn.Conv2d(in_channels, self.granule_ch, kernel_size=3, padding=1, groups=1)
            for _ in range(num_granules)
        ])
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        granule_outputs = []
        for conv in self.granule_convs:
            granule_outputs.append(conv(x))
        
        out = torch.cat(granule_outputs, dim=1)
        
        out = self.bn(out)
        return self.relu(out)

class GranulatedCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(GranulatedCNN, self).__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.layer1 = GranulatedBlock(64, 128, num_granules=4)
        self.layer2 = GranulatedBlock(128, 256, num_granules=8)
        self.layer3 = GranulatedBlock(256, 512, num_granules=8)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
