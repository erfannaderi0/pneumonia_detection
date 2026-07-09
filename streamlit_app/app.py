"""
app.py — Streamlit Cloud entry point for the pneumonia detection demo.

Downloads the ResNet152 checkpoint from the public Hugging Face model repo
(erfanna/pneumonia-resnet152) at startup, then runs single-image inference
with a GradCAM overlay in the browser.

Reuses the real project code (CropBorders from module.py, GradCAM +
overlay_heatmap from Gradcam.py) rather than reimplementing it, so this
stays in sync with the actual training/eval pipeline.
"""

import os
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from huggingface_hub import hf_hub_download

# ─────────────────────────────────────────────
# Make the repo root importable (module.py / Gradcam.py live one level up
# from this streamlit_app/ folder)
# ─────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from module import CropBorders          # noqa: E402
from Gradcam import GradCAM, overlay_heatmap  # noqa: E402

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
HF_REPO_ID = "erfanna/pneumonia-resnet152"
HF_CHECKPOINT_FILENAME = "model2_resnet152.pth"
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
CLINICAL_THRESHOLD = 0.98  # matches the tuned test-set threshold
XRAY_MEAN = [0.482, 0.482, 0.482]
XRAY_STD = [0.234, 0.234, 0.234]


# ─────────────────────────────────────────────
# Model loading (cached — only runs once per app instance)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Downloading model from Hugging Face…")
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # HF_TOKEN is only needed if the repo is gated/private. Reads from
    # Streamlit Cloud's Secrets manager — never hardcode a token here.
    token = st.secrets.get("HF_TOKEN", None) if hasattr(st, "secrets") else None

    ckpt_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_CHECKPOINT_FILENAME,
        token=token,
    )

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
    return model, target_layer, device


# ─────────────────────────────────────────────
# Preprocessing — mirrors resnet152.py's real 'test' transform
# ─────────────────────────────────────────────
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        CropBorders(threshold=10, crop_percent=0.1, output_size=(224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(XRAY_MEAN, XRAY_STD),
    ])


def preprocess(image: Image.Image):
    transform = get_transform()
    tensor = transform(image.convert("RGB")).unsqueeze(0)
    display = image.convert("RGB").resize((224, 224))
    return tensor, np.array(display)


# ─────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────
st.set_page_config(page_title="Pneumonia Detection Demo", page_icon="🫁", layout="centered")

st.title("🫁 Pneumonia Detection from Chest X-rays")
st.markdown(
    "Upload a chest X-ray image to get a prediction from a ResNet152 model "
    "fine-tuned on the Kermany/Mooney chest X-ray dataset, along with a "
    "GradCAM overlay showing which regions the model focused on."
)
st.caption(
    f"Clinical decision threshold: **{CLINICAL_THRESHOLD}** "
    "(tuned for high precision on the original, non-redistributed test split)."
)

uploaded_file = st.file_uploader("Upload a chest X-ray (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    with st.spinner("Loading model…"):
        model, target_layer, device = load_model()

    tensor, display_img = preprocess(image)
    tensor = tensor.to(device)

    with st.spinner("Running inference…"):
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)
            pneumonia_prob = probs[0, 1].item()
        decision = "PNEUMONIA" if pneumonia_prob >= CLINICAL_THRESHOLD else "NORMAL"

    with st.spinner("Generating GradCAM overlay…"):
        grad_cam = GradCAM(model, target_layer)
        cam, pred_class, pred_prob = grad_cam.generate(tensor)
        grad_cam.remove_hooks()
        blended = overlay_heatmap(display_img, cam)

    col1, col2 = st.columns(2)
    with col1:
        st.image(display_img, caption="Uploaded X-ray", use_container_width=True)
    with col2:
        st.image(blended, caption="GradCAM overlay", use_container_width=True)

    st.divider()

    if decision == "PNEUMONIA":
        st.error(f"**Prediction: {decision}**")
    else:
        st.success(f"**Prediction: {decision}**")

    st.metric("Raw probability (PNEUMONIA)", f"{pneumonia_prob:.4f}")
    st.caption(f"Decision uses threshold {CLINICAL_THRESHOLD} — raw probability is shown above the cutoff.")

    st.warning(
        "⚠️ This is a research/portfolio demo, not a diagnostic tool. "
        "Do not use this to make real medical decisions."
    )
else:
    st.info("Upload an X-ray image above to get started.")
