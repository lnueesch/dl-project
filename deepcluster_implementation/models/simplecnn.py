import math
import torch
import torch.nn as nn

__all__ = ['SimpleCNN', 'simplecnn']

# Configuration for layers: (number of filters, kernel size, stride, padding)
CFG = {
    'mnist': [(32, 3, 2, 1), 'M', (64, 2, 1, 1), 'M', (128, 2, 1, 1), 'M']  # Simplified for MNIST
    # 'mnist': [(32, 3, 2, 1), 'M', (64, 3, 2, 1), 'M']
    # 'mnist': [(32, 3, 2, 1), 'M', (64, 3, 1, 1), 'M']
}


class SimpleCNN(nn.Module):
    def __init__(self, features, num_classes, sobel):
        super(SimpleCNN, self).__init__()
        self.features = features  # Feature extractor
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            # nn.Linear(576, 256),  # Adjusted for MNIST
            nn.Linear(512, 256),  # Adjusted for MNIST
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            # nn.Linear(512, 256),
            # nn.ReLU(inplace=True),
        )
        self.top_layer = nn.Linear(256, num_classes)  # Final output layer
        self._initialize_weights()

        if sobel:
            # Sobel filter for edge detection
            grayscale = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0)
            grayscale.weight.data.fill_(1.0)
            grayscale.bias.data.zero_()
            sobel_filter = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1)
            sobel_filter.weight.data[0, 0].copy_(
                torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
            )
            sobel_filter.weight.data[1, 0].copy_(
                torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
            )
            sobel_filter.bias.data.zero_()
            self.sobel = nn.Sequential(grayscale, sobel_filter)
            for p in self.sobel.parameters():
                p.requires_grad = False
        else:
            self.sobel = None

    def forward(self, x):
        x = x.to(next(self.parameters()).device)
        if self.sobel:
            x = self.sobel(x)
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
    print(input_dim)
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


def simplecnn(sobel=False, bn=True, out=10):
    """Constructor function for SimpleCNN."""
    dim = int(not sobel)  # Adjust input channels for Sobel
    model = SimpleCNN(make_layers_features(CFG['mnist'], dim, bn=bn), out, sobel)
    return model