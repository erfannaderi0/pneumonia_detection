"""
demo.py
--------
Standalone inference demo for the pneumonia detection project.

Loads a trained checkpoint (Custom CNN or ResNet152) and runs inference on a
single image or a folder of images, with an optional GradCAM overlay and the
tuned clinical threshold (0.98) applied to the raw probability.

Decoupled from results_tracker.py / runs.json on purpose — no experiment
logging here, just fast direct inference.

Usage:
    python demo.py --model resnet152 --image path/to/xray.jpeg
    python demo.py --model custom --image path/to/xray.jpeg
    python demo.py --model resnet152 --folder path/to/images/ --output results.csv
    python demo.py --model resnet152 --image path/to/xray.jpeg --gradcam

"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from module import improved_cnn, CropBorders
from Gradcam import GradCAM, overlay_heatmap  # reuse the actual GradCAM implementation

# ─────────────────────────────────────────────
#  Config — keep in sync with resnet152.py / results_tracker.py
# ─────────────────────────────────────────────
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
RESNET152_CHECKPOINT = "models/model2_resnet152.pth"
CUSTOM_CNN_CHECKPOINT = "models/model2_improved_cnn.pth"
CLINICAL_THRESHOLD = 0.98  # tuned against the original (non-redistributed) test split
XRAY_MEAN = [0.482, 0.482, 0.482]
XRAY_STD = [0.234, 0.234, 0.234]


# ─────────────────────────────────────────────
#  Model loading
# ─────────────────────────────────────────────
def load_model(model_type: str, device: torch.device):
    if model_type == "resnet152":
        ckpt_path = Path(RESNET152_CHECKPOINT)
        if not ckpt_path.exists():
            sys.exit(f"❌ Checkpoint not found: {ckpt_path}. Train ResNet152 first.")

        model = models.resnet152(weights=None)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 2),
        )
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device).eval()
        target_layer = model.layer4[-1].conv3
        return model, target_layer

    ckpt_path = Path(CUSTOM_CNN_CHECKPOINT)
    if not ckpt_path.exists():
        sys.exit(f"❌ Checkpoint not found: {ckpt_path}. Train the custom CNN first.")

    model = improved_cnn(input_shape=1, hidden_units=32, output_shape=2)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()
    target_layer = model.block4[-3]
    return model, target_layer


# ─────────────────────────────────────────────
#  Preprocessing — mirrors resnet152.py's 'test' transform exactly
# ─────────────────────────────────────────────
def get_transform(model_type: str, legacy_crop: bool):
    if model_type == "resnet152":
        if legacy_crop:
            # Matches Gradcam.py's preprocess() — NOT what the model was evaluated with.
            crop = CropBorders()
        else:
            # Matches resnet152.py's real 'test' transform.
            crop = CropBorders(threshold=10, crop_percent=0.1, output_size=(224, 224))
        return transforms.Compose([
            transforms.Resize((224, 224)),
            crop,
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(XRAY_MEAN, XRAY_STD),
        ])

    # Custom CNN: grayscale 1-channel (matches Gradcam.py's fallback path).
    # NOTE: I don't have the custom CNN's actual training-time transform file,
    # so this may not exactly match what improved_cnn was trained/evaluated with.
    # Worth double-checking against your custom-CNN training script.
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])


def preprocess(image_path: Path, model_type: str, legacy_crop: bool):
    image = Image.open(image_path).convert("RGB")
    transform = get_transform(model_type, legacy_crop)
    tensor = transform(image).unsqueeze(0)
    display = image.resize((224, 224))
    return tensor, np.array(display)


# ─────────────────────────────────────────────
#  Inference
# ─────────────────────────────────────────────
def predict(model, tensor, device):
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        pneumonia_prob = probs[0, 1].item()

    decision = "PNEUMONIA" if pneumonia_prob >= CLINICAL_THRESHOLD else "NORMAL"
    return pneumonia_prob, decision


def process_image(image_path: Path, model, target_layer, model_type: str, device,
                   legacy_crop: bool, gradcam_dir: Path | None):
    tensor, display_img = preprocess(image_path, model_type, legacy_crop)
    prob, decision = predict(model, tensor, device)

    print(f"\n{image_path.name}")
    print(f"  Raw probability (PNEUMONIA): {prob:.4f}")
    print(f"  Threshold applied: {CLINICAL_THRESHOLD}")
    print(f"  Decision: {decision}")

    cam_path = None
    if gradcam_dir is not None:
        gradcam_dir.mkdir(parents=True, exist_ok=True)
        grad_cam = GradCAM(model, target_layer)
        cam, pred_class, pred_prob = grad_cam.generate(tensor.to(device))
        grad_cam.remove_hooks()

        blended = overlay_heatmap(display_img, cam)
        cam_path = gradcam_dir / f"gradcam_{model_type}_{image_path.stem}.png"
        Image.fromarray(blended).save(cam_path)
        print(f"  GradCAM saved: {cam_path}")

    return {"image": image_path.name, "probability": prob, "decision": decision,
            "gradcam": str(cam_path) if cam_path else ""}


def main():
    parser = argparse.ArgumentParser(description="Pneumonia detection demo")
    parser.add_argument("--model", choices=["custom", "resnet152"], required=True)
    parser.add_argument("--image", type=str, help="Path to a single X-ray image")
    parser.add_argument("--folder", type=str, help="Path to a folder of images (batch mode)")
    parser.add_argument("--output", type=str, help="CSV path to save batch results")
    parser.add_argument("--gradcam", action="store_true", help="Save GradCAM overlays")
    parser.add_argument("--legacy-crop", action="store_true",
                         help="Use Gradcam.py's CropBorders() defaults instead of the "
                              "real resnet152.py eval-time crop (crop_percent=0.1, resize-after-crop)")
    args = parser.parse_args()

    if not args.image and not args.folder:
        parser.error("Provide either --image or --folder")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, target_layer = load_model(args.model, device)
    gradcam_dir = Path("results_log/gradcam_demo") if args.gradcam else None

    if args.image:
        process_image(Path(args.image), model, target_layer, args.model, device,
                       args.legacy_crop, gradcam_dir)
        return

    folder = Path(args.folder)
    image_paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not image_paths:
        sys.exit(f"❌ No images found in {folder}")

    results = [
        process_image(p, model, target_layer, args.model, device, args.legacy_crop, gradcam_dir)
        for p in image_paths
    ]

    if args.output:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image", "probability", "decision", "gradcam"])
            writer.writeheader()
            writer.writerows(results)
        print(f"\n✅ Results saved to {args.output}")


if __name__ == "__main__":
    main()
