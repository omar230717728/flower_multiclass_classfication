# This script trains and evaluates the VGG16 feature extractor model for flower classification.

import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from models.vgg16_feature_extractor import build_vgg16_feature_extractor
from utils.dataset import get_data_loaders
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training configuration
num_epochs = 10
batch_size = 32
learning_rate = 0.001

# Hide base paths inside main block
if __name__ == "__main__":
    # ✅ Define absolute project root path
    ABS_PATH = r"D:\progects\flower_multiclass classfication"
    data_dir = os.path.join(ABS_PATH, "data", "flowers")
    model_dir = os.path.join(ABS_PATH, "trained_models")
    results_dir = os.path.join(ABS_PATH, "results", "vgg16_feature_extractor")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Load data
    train_loader, val_loader, class_names = get_data_loaders(data_dir, batch_size=batch_size)

    # Initialize model
    model = build_vgg16_feature_extractor(num_classes=len(class_names)).to(device)

    # Loss function and optimizer (only classifier parameters are trainable)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate, weight_decay=1e-4)  # ✅ L2 regularization

    # Stats tracking
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    # ✅ Early stopping setup
    best_val_loss = float('inf')
    patience = 3
    counter = 0

    # Training loop
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss_avg = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        val_losses.append(val_loss_avg)
        val_accuracies.append(val_acc)

        print(f"Epoch [{epoch + 1}/{num_epochs}] - "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss_avg:.4f}, "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        # ✅ Early stopping logic
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"⏹️ Early stopping triggered at epoch {epoch + 1}.")
                break

    end_time = time.time()
    print(f"\nTraining completed in {end_time - start_time:.2f} seconds.")

    # Save model
    model_path = os.path.join(model_dir, "vgg16_feature_extractor.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Evaluation report
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print("\nClassification Report:\n")
    print(report)

    # Save report
    report_path = os.path.join(results_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    # Plot loss and accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Acc")
    plt.plot(val_accuracies, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Epochs")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "training_plots.png"))
    plt.show()
