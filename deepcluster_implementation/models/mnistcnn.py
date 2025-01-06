import torch
import torch.nn as nn
import torch.nn.functional as F

_all_ = ['mnistmodel']

class MNISTmodel(nn.Module):
    def _init_(self, sobel=False):
        super(MNISTmodel, self)._init_()
        self.sobel = sobel

        # Feature extractor (convolutional layers)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier (set to a sequential container to allow extensions)
        self.classifier = nn.Sequential()

        # Top layer (linear layer for classification)
        self.top_layer = nn.Linear(64, 10)

    def forward(self, x):
        # Apply Sobel filter if enabled
        if self.sobel:
            x = self.apply_sobel(x)

        # Feature extraction
        x = self.features(x)

        # Global Average Pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)  # Flatten for the linear layer

        # Forward through classifier (empty by default)
        x = self.classifier(x)

        # Top layer classification
        if self.top_layer:
            x = self.top_layer(x)
        return x

    def apply_sobel(self, x):
        """Apply Sobel filter to the input."""
        sobel_filter = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        if x.is_cuda:
            sobel_filter = sobel_filter.to(x.device)
        sobel_x = F.conv2d(x, sobel_filter, padding=1)
        sobel_y = F.conv2d(x, sobel_filter.transpose(2, 3), padding=1)
        return torch.sqrt(sobel_x ** 2 + sobel_y ** 2)

# Function to return the MNIST model
def mnistmodel(sobel=False):
    return MNISTmodel(sobel=sobel)