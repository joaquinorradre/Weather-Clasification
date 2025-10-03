import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_V1(nn.Module):
    def __init__(self, input_dim, num_classes=11):
        super(CNN_V1, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 5),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Compute flattened feature size after convs
        # For 64x64 input:
        # Conv/Pooling dims: ((64-4)/1)->60 -> 60-4=56 -> Pool/2 =28 etc.
        self.flattened_size = 128*12*12  # check formula below

        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x
    
class CNN_V1_reg(nn.Module):
    def __init__(self, input_dim, num_classes=11):
        super(CNN_V1_reg, self).__init__()
        self.p = 0.4
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 5),
            nn.ReLU(),
            nn.Dropout(self.p),
            nn.Conv2d(16, 32, 5),
            nn.ReLU(),
            nn.Dropout(self.p),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.Dropout(self.p),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.Dropout(self.p),
            nn.MaxPool2d(2, 2)
        )

        # Compute flattened feature size after convs
        # For 64x64 input:
        # Conv/Pooling dims: ((64-4)/1)->60 -> 60-4=56 -> Pool/2 =28 etc.
        self.flattened_size = 128*12*12  

        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(),
            nn.Dropout(self.p),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(self.p),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.p),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x
    

class CNN_V2(nn.Module):
    def __init__(self, input_dim, num_classes=11):
        super(CNN_V2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),

            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 256, 5),
            nn.ReLU(),

            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, 3),
            nn.ReLU(),

            nn.MaxPool2d(3, 3)
        )

        # Compute flattened feature size after convs
        self.flattened_size = 512*8*8 

        # Compute flattened feature size after convs
        self.flattened_size = 512*7*7  

        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x
    
class CNN_V2_reg(nn.Module):
    def __init__(self, input_dim, num_classes=11):
        super(CNN_V2_reg, self).__init__()
        self.p = 0.4
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(),
            nn.Dropout(self.p),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.Dropout(self.p),

            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.Dropout(self.p),
            nn.Conv2d(128, 256, 5),
            nn.ReLU(),
            nn.Dropout(self.p),

            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, 3),
            nn.ReLU(),
            nn.Dropout(self.p),

            nn.MaxPool2d(3, 3)
        )

        # Compute flattened feature size after convs
        self.flattened_size = 512*8*8 

        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(),
            nn.Dropout(self.p),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(self.p),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.p),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x