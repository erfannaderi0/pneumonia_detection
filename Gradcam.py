"""
gradcam.py
==========
Grad-CAM heatmap visualisation for both the custom CNN and ResNet152.

Usage (from your project root):
    python gradcam.py --model resnet152 --image path/to/xray.jpeg
    python gradcam.py --model custom   --image path/to/xray.jpeg
    python gradcam.py --model resnet152 --batch  # runs on 4 test samples and saves a grid

What it does:
    Highlights the regions of the X-ray the model focuses on when making
    its prediction. Red = high attention, blue = low attention.
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from module import improved_cnn
from module import CropBorders


# ─────────────────────────────────────────────
#  Grad-CAM core
# ─────────────────────────────────────────────

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.

    Registers forward and backward hooks on a target conv layer.
    After a forward pass, call .generate() to get the heatmap.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None

        # Hook: save feature maps during forward pass
        self._fwd_hook = target_layer.register_forward_hook(self._save_activations)
        # Hook: save gradients during backward pass
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None):
        """
        Args:
            input_tensor : shape (1, C, H, W) — already on the right device
            class_idx    : which class to explain (None = predicted class)
        Returns:
            heatmap : np.ndarray shape (H, W), values in [0, 1]
            pred_class : int
            pred_prob  : float
        """
        self.model.eval()
        output = self.model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred_class = int(torch.argmax(probs, dim=1).item())
        pred_prob = float(probs[0, pred_class].item())

        if class_idx is None:
            class_idx = pred_class

        # Backward pass for target class
        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward()

        # Global average pool the gradients → importance weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = torch.relu(cam).squeeze().cpu().numpy()

        # Resize to input size and normalise
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam, pred_class, pred_prob

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


# ─────────────────────────────────────────────
#  Model loaders
# ─────────────────────────────────────────────

def load_resnet152(device):
    model = models.resnet152(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(256, 2)
    )
    path = 'models/model2.5final_resnet152.pth'
    #path = 'models/model2_improved_cnn.pth'
    if not os.path.exists(path):
        print(f"❌ Model not found: {path}")
        return None, None
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    # Best layer: last conv block before the global pool
    target_layer = model.layer4[-1].conv3
    return model, target_layer


def load_custom_cnn(device):
    model = improved_cnn(input_shape=1, hidden_units=32, output_shape=2)
    path = 'models/model2_improved_cnn.pth'
    if not os.path.exists(path):
        print(f"❌ Model not found: {path}")
        return None, None
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    # Last conv in block4 (deepest feature extractor)
    target_layer = model.block4[-3]  # second Conv2d inside block4
    return model, target_layer


# ─────────────────────────────────────────────
#  Image preprocessing
# ─────────────────────────────────────────────

def preprocess(image_path: str, model_type: str):
    img = Image.open(image_path).convert('RGB')
    xray_mean = [0.482, 0.482, 0.482]
    xray_std  = [0.234, 0.234, 0.234]
    
    if model_type == 'resnet152':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            #CropBorders(threshold=10, crop_percent=0.2, output_size=(224, 224)),
            CropBorders(),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(xray_mean, xray_std)])

    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    tensor = transform(img).unsqueeze(0)
    # Keep original for display
    display = img.resize((224, 224))
    return tensor, np.array(display)


# ─────────────────────────────────────────────
#  Overlay helper
# ─────────────────────────────────────────────

def overlay_heatmap(image_np: np.ndarray, cam: np.ndarray, alpha: float = 0.45):
    """
    Blend a Grad-CAM heatmap onto the original image.
    Returns an RGB np.ndarray (224, 224, 3).
    """
    # Resize cam to image size
    from PIL import Image as PILImage
    cam_resized = np.array(
        PILImage.fromarray((cam * 255).astype(np.uint8)).resize(
            (image_np.shape[1], image_np.shape[0]), PILImage.BILINEAR
        )
    ) / 255.0

    # Apply jet colormap
    heatmap = cm.jet(cam_resized)[:, :, :3]  # drop alpha channel

    # Convert original to float [0,1]
    if image_np.dtype == np.uint8:
        img_float = image_np.astype(float) / 255.0
    else:
        img_float = image_np.copy()

    # Blend
    blended = alpha * heatmap + (1 - alpha) * img_float
    blended = np.clip(blended, 0, 1)
    return (blended * 255).astype(np.uint8)


# ─────────────────────────────────────────────
#  Single-image visualisation
# ─────────────────────────────────────────────

def visualise_single(image_path: str, model_type: str = 'resnet152', save_dir: str = 'results_log/gradcam'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(save_dir, exist_ok=True)

    if model_type == 'resnet152':
        model, target_layer = load_resnet152(device)
    else:
        model, target_layer = load_custom_cnn(device)

    if model is None:
        return

    tensor, display_img = preprocess(image_path, model_type)
    tensor = tensor.to(device)

    grad_cam = GradCAM(model, target_layer)
    cam, pred_class, pred_prob = grad_cam.generate(tensor)
    grad_cam.remove_hooks()

    class_names = ['NORMAL', 'PNEUMONIA']
    label = class_names[pred_class]
    colour = '#D85A30' if pred_class == 1 else '#1D9E75'

    blended = overlay_heatmap(display_img, cam)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle(
        f"Grad-CAM — {Path(image_path).name}",
        fontsize=13, y=1.01
    )

    axes[0].imshow(display_img)
    axes[0].set_title('Original X-ray', fontsize=11)
    axes[0].axis('off')

    im = axes[1].imshow(cam, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title('Attention heatmap', fontsize=11)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(blended)
    axes[2].set_title(
        f'Prediction: {label}\nConfidence: {pred_prob:.1%}',
        fontsize=11, color=colour
    )
    axes[2].axis('off')

    plt.tight_layout()

    slug = Path(image_path).stem
    save_path = os.path.join(save_dir, f'gradcam_{model_type}_{slug}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved → {save_path}")
    plt.show()


# ─────────────────────────────────────────────
#  Batch grid: show 4 samples (2 normal, 2 pneumonia)
# ─────────────────────────────────────────────

def visualise_batch(model_type: str = 'resnet152',
                    data_dir: str = 'data/reorganized_chest_xray/test',
                    n_per_class: int = 2,
                    save_dir: str = 'results_log/gradcam'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(save_dir, exist_ok=True)

    if model_type == 'resnet152':
        model, target_layer = load_resnet152(device)
    else:
        model, target_layer = load_custom_cnn(device)

    if model is None:
        return

    # Collect images
    samples = []
    for cls in ['NORMAL', 'PNEUMONIA']:
        cls_dir = Path(data_dir) / cls
        imgs = sorted(cls_dir.glob('*.jpeg'))[:n_per_class]
        if not imgs:
            imgs = sorted(cls_dir.glob('*.jpg'))[:n_per_class]
        for p in imgs:
            samples.append((str(p), cls))

    grad_cam = GradCAM(model, target_layer)
    class_names = ['NORMAL', 'PNEUMONIA']

    n = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(13, 4.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, (img_path, true_label) in enumerate(samples):
        tensor, display_img = preprocess(img_path, model_type)
        tensor = tensor.to(device)
        cam, pred_class, pred_prob = grad_cam.generate(tensor)
        blended = overlay_heatmap(display_img, cam)

        pred_label = class_names[pred_class]
        correct = pred_label == true_label
        status = '✓' if correct else '✗'
        colour = '#1D9E75' if correct else '#D85A30'

        axes[row, 0].imshow(display_img)
        axes[row, 0].set_title(f'True: {true_label}', fontsize=10)
        axes[row, 0].axis('off')

        axes[row, 1].imshow(cam, cmap='jet', vmin=0, vmax=1)
        axes[row, 1].set_title('Attention', fontsize=10)
        axes[row, 1].axis('off')

        axes[row, 2].imshow(blended)
        axes[row, 2].set_title(
            f'{status} Pred: {pred_label} ({pred_prob:.1%})',
            fontsize=10, color=colour
        )
        axes[row, 2].axis('off')

    grad_cam.remove_hooks()

    fig.suptitle(f'Grad-CAM batch — {model_type}', fontsize=13, y=1.01)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f'gradcam_{model_type}_batch.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved → {save_path}")
    plt.show()


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Grad-CAM for pneumonia detection')
    parser.add_argument('--model', choices=['resnet152', 'custom'], default='resnet152')
    parser.add_argument('--image', type=str, default=None, help='Path to a single X-ray image')
    parser.add_argument('--batch', action='store_true', help='Run on 4 test samples and save grid')
    args = parser.parse_args()

    if args.batch:
        visualise_batch(model_type=args.model)
    elif args.image:
        visualise_single(image_path=args.image, model_type=args.model)
    else:
        # Default: run batch on test set
        print("No --image provided, running batch mode on test set...")
        visualise_batch(model_type=args.model)