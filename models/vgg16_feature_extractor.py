# File: vgg16_feature_extractor.py
# This script defines a model using VGG16 as a frozen feature extractor and replaces the classifier for flower classification.

import torch.nn as nn
from torchvision import models
from torchvision.models import VGG16_Weights

def build_vgg16_feature_extractor(num_classes=5):
    # Load the pretrained VGG16 model
    vgg16 = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

    # Freeze all VGG16 parameters to prevent updates during training
    for param in vgg16.features.parameters():
        param.requires_grad = False

    # Replace the classifier (FC layers)
    vgg16.classifier = nn.Sequential(
        nn.Linear(25088, 1024),  # First FC layer
        nn.ReLU(),
        nn.Dropout(0.5),         # ✅ Dropout for regularization
        nn.Linear(1024, num_classes)  # Output layer for 5 flower classes
    )

    return vgg16
