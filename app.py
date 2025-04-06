# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 15:18:28 2025

@author: Dell
"""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from PIL import Image
from monai.networks.nets import UNet
from monai.transforms import ScaleIntensity, Transpose, ToTensor, Compose

# Set device
device = torch.device("cpu")

# Load the trained model
@st.cache_resource
def load_model():
    model = UNet(
        spatial_dims=3,
        in_channels=3,
        out_channels=4,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2
    )
    model.load_state_dict(torch.load("brats_3d_updated.pth", map_location=device))
    model.eval()
    return model

model = load_model()

# Define MONAI transforms
transforms = Compose([
    ScaleIntensity(),
    ToTensor()
])

# Streamlit UI
st.title("Brain Tumor Detection & Segmentation")
st.write("Upload a **TIFF, NIfTI (.nii), or NumPy (.npy) file** to check for brain tumors.")

uploaded_file = st.file_uploader("Upload an image file", type=["npy", "nii", "tiff", "tif"])

def load_image(file):
    """Loads and preprocesses the uploaded image file."""
    ext = file.name.split(".")[-1]

    if ext in ["npy"]:
        img = np.load(file)
    elif ext in [".nii", ".nii.gz"]:
        nii_img = nib.load(file)
        img = nii_img.get_fdata()
    elif ext in [".tiff", ".tif"]:
        img = Image.open(file)
        img = np.array(img)
    else:
        raise ValueError("Unsupported file format")

    return img

if uploaded_file:
    try:
        # Load and preprocess the image
        input_img = load_image(uploaded_file)
        
        # Ensure correct shape (H, W, D, C) → MONAI format (C, D, H, W)
        if len(input_img.shape) == 3:  # If shape (H, W, D), assume single-channel
            input_img = np.expand_dims(input_img, axis=0)  # Convert to (1, H, W, D)

        elif len(input_img.shape) == 4 and input_img.shape[-1] in [1, 3]:  
    # If shape (H, W, D, C), move C to the first position
            input_img = np.transpose(input_img, (3, 0, 1, 2))  # (C, H, W, D)

        elif len(input_img.shape) == 4 and input_img.shape[0] in [1, 3]:  
    # If already in (C, H, W, D), do nothing
            pass

        else:
            raise ValueError(f"Unexpected image shape: {input_img.shape}. Expected (H, W, D) or (H, W, D, C).")

# Convert to PyTorch tensor and add batch dimension
        input_img = transforms(input_img).unsqueeze(0).to(device)  # (B, C, D, H, W)

        # Make prediction
        with torch.no_grad():
            output = model(input_img)
            pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

        # Check if tumor is present
        if np.any(pred_mask > 0):  # If there are nonzero values, tumor is detected
            st.error("🔴 Tumor Detected!")
            tumor_present = True
        else:
            st.success("🟢 No Tumor Detected!")
            tumor_present = False

        # Select a middle slice for visualization
        slice_idx = pred_mask.shape[-1] // 2

        # Fix input image shape for plotting
        input_img_np = input_img.cpu().numpy().squeeze()
        if len(input_img_np.shape) == 4:  # (C, H, W, D)
            input_img_np = input_img_np[0]  # Use only one channel

        # Plot the results
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # Original Image (FLAIR)
        axes[0].imshow(input_img_np[:, :, slice_idx], cmap="gray")
        axes[0].set_title("Original MRI Slice")
        axes[0].axis("off")

        # Predicted Mask
        axes[1].imshow(pred_mask[:, :, slice_idx], cmap="jet", alpha=0.6)
        axes[1].set_title("Predicted Tumor Mask")
        axes[1].axis("off")

        # Overlay Mask on Original Image
        axes[2].imshow(input_img_np[:, :, slice_idx], cmap="gray")
        if tumor_present:
            axes[2].imshow(pred_mask[:, :, slice_idx], cmap="jet", alpha=0.5)
        axes[2].set_title("Overlay (Image + Mask)")
        axes[2].axis("off")

        st.pyplot(fig)
        st.success("✅ Prediction Complete! Check the tumor segmentation above.")
    
    except Exception as e:
        st.error(f"Error processing file: {e}")
