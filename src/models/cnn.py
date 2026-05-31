import torch
import torch.nn as nn


class CNNModel(nn.Module):
    """Simple PyTorch CNN used by tests. Keeps a small, deterministic architecture."""
    def __init__(self, num_classes=10, input_channels=3):
        super(CNNModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Use adaptive pooling to support multiple input sizes while keeping the classifier small
        self.adaptive_pool = nn.AdaptiveAvgPool2d((28, 28))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # Quick smoke test
    model = CNNModel(num_classes=2, input_channels=3)
    sample = torch.randn(1, 3, 224, 224)
    out = model(sample)
    print('Output shape:', out.shape)
