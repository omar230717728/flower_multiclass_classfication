
# This file defines the custom CNN model for flower classification using PyTorch.
import torch.nn as nn
import torch.nn.functional as F

class CustomCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(CustomCNN, self).__init__()

        # First convolutional block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Third convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Fourth convolutional block
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)

        # Fifth convolutional block (deepest)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.pool5 = nn.AdaptiveAvgPool2d((1, 1))

        # Dropout for regularization before the fully connected layer
        self.dropout = nn.Dropout(0.5)

        # Fully connected layer
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # Apply conv + ReLU + pooling step by step
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.pool5(F.relu(self.conv5(x)))

        x = x.view(x.size(0), -1)         # Flatten
        x = self.dropout(x)              # Apply dropout to reduce overfitting
        x = self.fc(x)                   # Final classification layer
        return x
