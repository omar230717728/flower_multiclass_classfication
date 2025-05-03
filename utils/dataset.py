# This script handles dataset loading, augmentation, and preprocessing for the flower classification task.

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_data_loaders(data_dir, batch_size=32, val_split=0.2):
    # Define transforms with augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),  # Horizontal flip augmentation
        transforms.RandomRotation(15),     # Slight rotation
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Validation and test transforms (no augmentation)
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load full dataset
    full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    val_size = int(val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size

    # Split into training and validation
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Replace transform for validation set (no augmentation)
    val_dataset.dataset.transform = test_transform

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader, full_dataset.classes
