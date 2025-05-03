# This script trains and evaluates the fine-tuned VGG16 model on the flower dataset with visualization.

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from models.vgg16_fine_tuned import build_vgg16_fine_tuned
from utils.dataset import get_data_loaders

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training config
num_epochs = 10
batch_size = 32
learning_rate = 0.0001

if __name__ == "__main__":
    # ✅ Define base project folder
    ABS_PATH = r"D:\progects\flower_multiclass classfication"

    data_dir = os.path.join(ABS_PATH, "data", "flowers")
    model_dir = os.path.join(ABS_PATH, "trained_models")
    results_dir = os.path.join(ABS_PATH, "results", "vgg16_fine_tuned")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Load data
    train_loader, val_loader, class_names = get_data_loaders(data_dir, batch_size=batch_size)

    # Build model
    model = build_vgg16_fine_tuned(num_classes=len(class_names)).to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate,
                           weight_decay=1e-4)  # ✅ Added L2 regularization

    # Store history
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    # ✅ Early stopping setup
    best_val_loss = float('inf')
    patience = 3
    counter = 0

    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_losses.append(running_loss / len(train_loader))
        train_accuracies.append(correct / total)

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss_avg = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        val_losses.append(val_loss_avg)
        val_accuracies.append(val_acc)

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {train_losses[-1]:.4f}, "
              f"Val Loss: {val_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.4f}, "
              f"Val Acc: {val_accuracies[-1]:.4f}")

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
    model_path = os.path.join(model_dir, "vgg16_fine_tuned.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Evaluate model
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Classification report
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print("\nClassification Report:\n")
    print(report)

    # Save report
    report_path = os.path.join(results_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    # Plotting
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
    plt.close()

    print("Training plot and report saved successfully.")