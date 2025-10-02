import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNetModern(nn.Module):
    def __init__(self):
        super(LeNetModern, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)

        self.fc1 = nn.Linear(1600, 256)
        self.dropout = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(256, 128)

        self.output = nn.Linear(128, 11)
        
    def forward(self, x):
        # x: (B, 3, 64, 64)
        x = self.conv1(x)      # (B, 16, 62, 62)
        x = self.relu(x)
        x = self.pool(x)       # (B, 16, 31, 31)

        x = self.conv2(x)      # (B, 32, 27, 27)
        x = self.relu(x)
        x = self.pool(x)       # (B, 32, 13, 13)

        x = self.conv3(x)      # (B, 64, 11, 11)
        x = self.relu(x)
        x = self.pool(x)       # (B, 64, 5, 5)

        x = x.view(x.size(0), -1) 

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)  
        x = self.relu(x)
        x = self.dropout(x)

        x = self.output(x)
        
        return x