import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['EMNISTcnn', 'emnistcnn']

class EMNISTcnn(nn.Module):
    def __init__(self, sobel):
        super(EMNISTcnn, self).__init__()
        self.sobel = sobel

        # Feature extractor (convolutional layers)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier (set to a sequential container to allow extensions)
        self.classifier = nn.Sequential()

        # Top layer (linear layer for classification)
        self.top_layer = nn.Linear(32, 26)

    def forward(self, x):
        x = x.to(next(self.parameters()).device)
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

# Function to return the EMNIST model
def emnistcnn(sobel=False, bn=False, out=26):
    model = EMNISTcnn(sobel)
    return model