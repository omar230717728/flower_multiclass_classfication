import torch
import matplotlib.pyplot as plt
import numpy as np
from models.CustomCNN import CustomCNN
from utils.dataset import get_data_loaders
import os

# Convert a tensor to a numpy image (denormalized)
def tensor_to_image(tensor):
    image = tensor.cpu().clone().detach().numpy()
    image = image.transpose(1, 2, 0)
    image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    image = np.clip(image, 0, 1)
    return image

# Visualize feature maps from a specific layer
def visualize_layer(model, layer_name, data_loader, device, output_dir):
    model.eval()
    inputs, _ = next(iter(data_loader))  # Take one batch
    inputs = inputs[:1].to(device)       # Use a single image

    # Store activation
    activations = {}

    # Hook function
    def hook_fn(module, input, output):
        activations[layer_name] = output.detach()

    # Attach correct hook
    if layer_name == "conv1":
        hook = model.conv1.register_forward_hook(hook_fn)
    elif layer_name == "conv3":
        hook = model.conv3.register_forward_hook(hook_fn)
    elif layer_name == "conv5":
        hook = model.conv5.register_forward_hook(hook_fn)
    else:
        raise ValueError("Invalid layer name")

    # Forward pass
    with torch.no_grad():
        _ = model(inputs)

    hook.remove()  # Remove hook after use

    # Visualize filters
    fmap = activations[layer_name].squeeze(0).cpu()
    num_filters = min(fmap.shape[0], 6)
    fig, axs = plt.subplots(1, num_filters, figsize=(15, 5))
    for i in range(num_filters):
        axs[i].imshow(fmap[i], cmap='viridis')
        axs[i].axis('off')
        axs[i].set_title(f'{layer_name} - Filter {i}')

    plt.tight_layout()

    # Save to the correct folder
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{layer_name}_features.png")
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.show()

# Main
if __name__ == "__main__":
    import torch
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from models.CustomCNN import CustomCNN
    from utils.dataset import get_data_loaders


    # Convert a tensor to a numpy image (denormalized)
    def tensor_to_image(tensor):
        image = tensor.cpu().clone().detach().numpy()
        image = image.transpose(1, 2, 0)
        image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        image = np.clip(image, 0, 1)
        return image


    # Visualize feature maps from a specific layer
    def visualize_layer(model, layer_name, data_loader, device, output_dir):
        model.eval()
        inputs, _ = next(iter(data_loader))
        inputs = inputs[:1].to(device)

        activations = {}

        def hook_fn(module, input, output):
            activations[layer_name] = output.detach()

        if layer_name == "conv1":
            hook = model.conv1.register_forward_hook(hook_fn)
        elif layer_name == "conv3":
            hook = model.conv3.register_forward_hook(hook_fn)
        elif layer_name == "conv5":
            hook = model.conv5.register_forward_hook(hook_fn)
        else:
            raise ValueError("Invalid layer name")

        with torch.no_grad():
            _ = model(inputs)

        hook.remove()

        fmap = activations[layer_name].squeeze(0).cpu()
        num_filters = min(fmap.shape[0], 6)
        fig, axs = plt.subplots(1, num_filters, figsize=(15, 5))
        for i in range(num_filters):
            axs[i].imshow(fmap[i], cmap='viridis')
            axs[i].axis('off')
            axs[i].set_title(f'{layer_name} - Filter {i}')

        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{layer_name}_features.png")
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
        plt.show()


    # Main execution
    if __name__ == "__main__":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hide path definitions inside main block
        ABS_PATH = r"D:\progects\flower_multiclass classfication"
        model_path = os.path.join(ABS_PATH, "trained_models", "custom_cnn.pth")
        data_dir = os.path.join(ABS_PATH, "data", "flowers")
        output_dir = os.path.join(ABS_PATH, "results", "custom_cnn")

        # Load model and dataset
        model = CustomCNN().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        train_loader, _, _ = get_data_loaders(data_dir)

        # Visualize key convolutional layers
        for layer in ["conv1", "conv3", "conv5"]:
            visualize_layer(model, layer, train_loader, device, output_dir)
