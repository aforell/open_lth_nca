import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from ..models import med_nca, initializers

# -----------------------------
# User settings
# -----------------------------
MODEL_PATH = "/local/scratch/aforell-thesis/open_lth_data/train_e185cd9eefb2e3593cb95de6cea0c635/replicate_1/main/model_ep2001_it0.pth"                # your NCA model .pth file
IMAGE_PATH = "/local/scratch/aforell-thesis/open_lth_datasets/prostate2d/slices/imagesTs/prostate_42_slice_13.npy"                # input image path
OUTPUT_PLOT_PATH = "segmentation_plot.png"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Load Model
# -----------------------------
def load_model(model_path):
    # 1. Create your model architecture
    model = med_nca.Model.get_model_from_name('med_nca_2d_segmentation', initializers.kaiming_normal)   # <-- IMPORTANT: define this to match your training model

    # 2. Load state_dict from .pth file
    state_dict = torch.load(model_path, map_location=DEVICE)

    # 3. Load weights into model
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()
    return model

# -----------------------------
# Load & preprocess image
# -----------------------------
def load_image(path):
    # Load numpy array
    arr = np.load(path)      # shape could be (H,W), (H,W,C), or (C,H,W)
    print(f"Loaded image shape: {arr.shape}")
    assert False
    
    # Convert to float32 and normalize to [0,1] if necessary
    arr = arr.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0

    # Ensure channel order is (C, H, W)
    if arr.ndim == 2:  # grayscale (H, W)
        arr = arr[np.newaxis, ...]  # -> (1, H, W)
    elif arr.ndim == 3:
        if arr.shape[0] in [1,3]:  
            # already (C,H,W)
            pass
        else:
            # probably (H,W,C) → convert to (C,H,W)
            arr = np.transpose(arr, (2,0,1))

    tensor = torch.tensor(arr).unsqueeze(0).to(DEVICE)  # add batch dim

    # For plotting, convert to HWC format
    if arr.shape[0] == 1:
        plot_img = arr[0]  # grayscale
    else:
        plot_img = np.transpose(arr, (1,2,0))

    return plot_img, tensor

# -----------------------------
# Run inference
# -----------------------------
def infer(model, input_tensor):
    with torch.no_grad():
        output = model(input_tensor)

    # If segmentation, assume output shape is [B, C, H, W]
    if output.shape[1] > 1:
        # multi-class: take argmax
        mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    else:
        # binary: apply sigmoid and threshold
        mask = torch.sigmoid(output).squeeze().cpu().numpy()
        mask = (mask > 0.5).astype(np.uint8)

    return mask

# -----------------------------
# Plot input + mask
# -----------------------------
def save_plot(input_img, mask, save_path):
    plt.figure(figsize=(10,5))

    # Input image
    plt.subplot(1,2,1)
    plt.imshow(input_img)
    plt.title("Input Image")
    plt.axis("off")

    # Segmentation mask (color map)
    plt.subplot(1,2,2)
    plt.imshow(mask, cmap="jet", alpha=0.9)
    plt.title("Segmentation Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    print("Loading model...")
    model = load_model(MODEL_PATH)

    print("Loading image...")
    input_img, input_tensor = load_image(IMAGE_PATH)

    print("Running inference...")
    mask = infer(model, input_tensor)

    print(f"Saving plot to: {OUTPUT_PLOT_PATH}")
    save_plot(input_img, mask, OUTPUT_PLOT_PATH)

    print("Done!")
