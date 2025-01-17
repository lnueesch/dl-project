import math
import torch
import torch.nn as nn

__all__ = ['SimpleCNN', 'simplecnn']

# Configuration for layers: (number of filters, kernel size, stride, padding)
CFG = {
    'mnist': [(8, 3, 2, 1), 'M', (16, 2, 1, 1), 'M', (32, 2, 1, 1), 'M']  # Simplified for MNIST
}


class SimpleCNN(nn.Module):
    def __init__(self, features, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = features  # Feature extractor
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 64),  # Adjusted for MNIST
            nn.ReLU(inplace=True),
        )
        self.top_layer = nn.Linear(64, num_classes)  # Final output layer
        self._initialize_weights()

    def forward(self, x):
        x = x.to(next(self.parameters()).device)
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        if self.top_layer:
            x = self.top_layer(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers_features(cfg, input_dim, bn):
    """Dynamically create layers from configuration."""
    layers = []
    in_channels = input_dim
    for v in cfg:
        if v == 'M':  # MaxPooling
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:  # Convolutional Layer
            conv2d = nn.Conv2d(in_channels, v[0], kernel_size=v[1], stride=v[2], padding=v[3])
            if bn:
                layers += [conv2d, nn.BatchNorm2d(v[0]), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v[0]
    return nn.Sequential(*layers)


def simplecnn(bn=True, out=10):
    """Constructor function for SimpleCNN."""
    model = SimpleCNN(make_layers_features(CFG['mnist'], 1, bn=bn), out)
    return model