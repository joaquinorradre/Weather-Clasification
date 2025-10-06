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
        # Dropout probabilities
        self.p_conv = 0.15   # Convolutional layers
        self.p_fc = 0.4      # Fully connected layers

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 5),
            nn.ReLU(),
            nn.Dropout(self.p_conv),

            nn.Conv2d(16, 32, 5),
            nn.ReLU(),
            nn.Dropout(self.p_conv),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.Dropout(self.p_conv),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.Dropout(self.p_conv),
            nn.MaxPool2d(2, 2)
        )

        # Compute flattened feature size after convs
        # For 64x64 input:
        # Conv/Pooling dims: ((64-4)/1)->60 -> 60-4=56 -> Pool/2 =28 etc.
        self.flattened_size = 128*12*12  

        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(),
            nn.Dropout(self.p_fc),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(self.p_fc),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.p_fc),
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
        self.p_conv = 0.15   # Convolutional layers
        self.p_fc = 0.4      # Fully connected layers

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(),
            nn.Dropout(self.p_conv),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.Dropout(self.p_conv),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.Dropout(self.p_conv),
            nn.Conv2d(128, 256, 5),
            nn.ReLU(),
            nn.Dropout(self.p_conv),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, 3),
            nn.ReLU(),
            nn.Dropout(self.p_conv),

            nn.MaxPool2d(3, 3)
        )

        # Compute flattened feature size after convs
        self.flattened_size = 512*8*8 

        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(),
            nn.Dropout(self.p_fc),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(self.p_fc),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.p_fc),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x
    
class CNN_V3(nn.Module):
    def __init__(self, input_dim, num_classes=11):
        super(CNN_V3, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # Compute flattened feature size after convs
        self.flattened_size = 128*10*10


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

class CNN_V3_reg(nn.Module):
    def __init__(self, input_dim, num_classes=11):
        super(CNN_V3_reg, self).__init__()
        self.p_conv = 0.15   # Convolutional layers
        self.p_fc = 0.4      # Fully connected layers

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.Dropout(self.p_conv),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.Dropout(self.p_conv),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.Dropout(self.p_conv),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 5),
            nn.ReLU(),
            nn.Dropout(self.p_conv),
            nn.MaxPool2d(2, 2),
        )

        # Compute flattened feature size after convs
        self.flattened_size = 128*10*10


        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(),
            nn.Dropout(self.p_fc),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(self.p_fc),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.p_fc),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x
    
# Improved version of your existing CNN_V3
class CNN_V3_Improved(nn.Module):
    """
    Direct improvement of your current CNN_V3
    Keeps similar structure but adds:
    - More channels
    - Better dropout placement
    """
    def __init__(self, input_dim, num_classes=11):
        super(CNN_V3_Improved, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.15),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 5, padding=2),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 5, padding=2),  # 256 instead of 128
            nn.ReLU(),
            nn.Dropout2d(0.25),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(256, 512, 3, padding=1),  # Extra layer
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(2, 2),
        )

        # For 224x224 input
        self.flattened_size = 512 * 7 * 7

        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, 1024),  # Bigger
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, num_classes)
        )
        
        #self._initialize_weights()
    """
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    """
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x