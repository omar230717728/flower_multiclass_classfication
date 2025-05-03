# This script defines the VGG16 model for fine-tuning, freezing early layers and training deeper ones.

import torch.nn as nn
from torchvision import models
from torchvision.models import VGG16_Weights

def build_vgg16_fine_tuned(num_classes=5):
    # Load VGG16 with pretrained ImageNet weights
    vgg16 = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

    # Freeze the first convolutional block (conv1_1 and conv1_2)
    for name, param in vgg16.features.named_parameters():
        if "0" in name or "2" in name:  # conv1_1 (0), conv1_2 (2)
            param.requires_grad = False

    # Replace the classifier to adapt to the flower dataset
    vgg16.classifier = nn.Sequential(
        nn.Linear(25088, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, num_classes)
    )

    return vgg16
