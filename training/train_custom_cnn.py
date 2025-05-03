# This script handles the training, evaluation, and visualization for the Custom CNN model.

import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from models.CustomCNN import CustomCNN
from utils.dataset import get_data_loaders
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configurations

# Define absolute path to your project root (only change this line if you move folders)
ABS_PATH = r"D:\progects\flower_multiclass classfication"
data_dir = os.path.join(ABS_PATH, "data", "flowers") # Update to your dataset path
num_epochs = 15
batch_size = 32
learning_rate = 0.001

# Load data
train_loader, val_loader, class_names = get_data_loaders(data_dir, batch_size=batch_size)

# Initialize model
model = CustomCNN(num_classes=len(class_names)).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Define standard subfolders
results_dir = os.path.join(ABS_PATH, "results", "custom_cnn")
model_dir = os.path.join(ABS_PATH, "trained_models")
os.makedirs(results_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Train the model
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # Evaluate on validation set
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_losses.append(val_loss / len(val_loader))
    val_accuracies.append(val_correct / val_total)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_losses[-1]:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_accuracies[-1]:.4f}")

end_time = time.time()
training_time = end_time - start_time
print(f"\nTraining completed in {training_time:.2f} seconds.\n")

# Evaluate on validation set and print classification report
all_preds = []
all_labels = []
model.eval()

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Generate classification report
report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
print("Classification Report:\n")
print(report)

# Save report
report_path = os.path.join(results_dir, "classification_report.txt")
with open(report_path, "w") as f:
    f.write(report)

# Plot training/validation loss and accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Train Acc")
plt.plot(val_accuracies, label="Validation Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy over Epochs")
plt.legend()

plt.tight_layout()

# Save plot
plot_path = os.path.join(results_dir, "training_plots.png")
plt.savefig(plot_path)
plt.show(block=False)

# Save the trained model
model_path = os.path.join(model_dir, "custom_cnn.pth")
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")
