# This script visualizes feature maps from the fine-tuned VGG16 model layers: conv1, conv3, conv5.

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from models.vgg16_fine_tuned import build_vgg16_fine_tuned
from utils.dataset import get_data_loaders

# Convert tensor to a denormalized image
def tensor_to_image(tensor):
    image = tensor.cpu().clone().detach().numpy()
    image = image.transpose(1, 2, 0)
    image = image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    return np.clip(image, 0, 1)

# Visualize feature maps from a VGG layer
def visualize_layer(model, layer_idx, data_loader, device, tag, output_dir):
    model.eval()
    inputs, _ = next(iter(data_loader))
    inputs = inputs[:1].to(device)

    activation = {}

    def hook_fn(module, input, output):
        activation[tag] = output.detach()

    handle = model.features[layer_idx].register_forward_hook(hook_fn)

    with torch.no_grad():
        _ = model(inputs)

    handle.remove()

    fmap = activation[tag].squeeze(0).cpu()
    num_filters = min(fmap.shape[0], 6)
    fig, axs = plt.subplots(1, num_filters, figsize=(15, 5))
    for i in range(num_filters):
        axs[i].imshow(fmap[i], cmap='viridis')
        axs[i].axis('off')
        axs[i].set_title(f'{tag} - F{i}')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{tag}_features_finetuned.png"))
    print(f"Saved: {os.path.join(output_dir, f'{tag}_features_finetuned.png')}")
    plt.show()

# Main
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ Base path config
    ABS_PATH = r"D:\progects\flower_multiclass classfication"
    model_path = os.path.join(ABS_PATH, "trained_models", "vgg16_fine_tuned.pth")
    data_dir = os.path.join(ABS_PATH, "data", "flowers")
    output_dir = os.path.join(ABS_PATH, "results", "vgg16_fine_tuned")

    # Load model
    model = build_vgg16_fine_tuned().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load data
    train_loader, _, _ = get_data_loaders(data_dir)

    # Visualize conv1_1 (0), conv3_1 (10), conv5_1 (24)
    visualize_layer(model, 0, train_loader, device, "conv1", output_dir)
    visualize_layer(model, 10, train_loader, device, "conv3", output_dir)
    visualize_layer(model, 24, train_loader, device, "conv5", output_dir)
