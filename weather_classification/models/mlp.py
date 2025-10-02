import torch
import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=11):
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, num_classes)  # logits for softmax
        )
        
    def forward(self, x):
        return self.model(x)
