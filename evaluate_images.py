from generator import *

import os
import glob
import time
import csv

import torch
from torchvision import transforms
from PIL import Image
import numpy as np

#from skimage.metrics import peak_signal_noise_ratio
#from skimage.metrics import structural_similarity


# ============================================================
# CONFIGURATION
# ============================================================

# Folder containing UNSEEN blurred images
INPUT_FOLDER = r"deblur\blurred"

# Optional:
# Folder containing corresponding sharp/original images.
#
# Filenames MUST match.
#
# Example:
# blurred/
#     image1.jpg
#     image2.jpg
#
# sharp/
#     image1.jpg
#     image2.jpg
#
# Set to None if you only want visual testing.
GROUND_TRUTH_FOLDER = None

# Example:
# GROUND_TRUTH_FOLDER = r"sharp"


# Best trained generator
MODEL_PATH = r"deblur\new training\epoch 151 to 200\best_generator.pth"


# Folder where deblurred images will be saved
OUTPUT_FOLDER = r"deblur\deblurred"


# Device
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# IMAGENET NORMALIZATION
# ============================================================
#
# These MUST match the normalization used during training.
#
# Your training transform was:
#
# transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
#
# Therefore we explicitly define them here.
# ============================================================

IMAGENET_MEAN = (
    0.485,
    0.456,
    0.406
)

IMAGENET_STD = (
    0.229,
    0.224,
    0.225
)


# ============================================================
# IMAGE TRANSFORM
# ============================================================
#
# IMPORTANT:
#
# Do NOT use Resize
# Do NOT use RandomRotation
# Do NOT use RandomHorizontalFlip
# Do NOT use ColorJitter
#
# during inference.
#
# We only apply the SAME normalization used during training.
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])


# ============================================================
# PRINT CONFIGURATION
# ============================================================

print("=" * 70)
print("UNSEEN IMAGE DEBLURRING TEST")
print("=" * 70)

print(f"Device        : {DEVICE}")
print(f"Model         : {MODEL_PATH}")
print(f"Input folder  : {INPUT_FOLDER}")
print(f"Output folder : {OUTPUT_FOLDER}")

print("\nNormalization:")
print(f"Mean = {IMAGENET_MEAN}")
print(f"Std  = {IMAGENET_STD}")


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ============================================================
# LOAD GENERATOR
# ============================================================

generator = DeblurGenerator().to(DEVICE)


print("\nLoading generator...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


# ============================================================
# HANDLE DIFFERENT CHECKPOINT FORMATS
# ============================================================

if isinstance(checkpoint, dict):

    if "generator_state_dict" in checkpoint:

        state_dict = checkpoint[
            "generator_state_dict"
        ]

    elif "state_dict" in checkpoint:

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        # Check whether this is a raw state_dict
        state_dict = checkpoint

else:

    state_dict = checkpoint


# ============================================================
# LOAD WEIGHTS
# ============================================================

generator.load_state_dict(
    state_dict
)

generator.eval()


print("Generator loaded successfully.")


# ============================================================
# HELPER: LOAD IMAGE
# ============================================================

def load_image(path):

    image = Image.open(
        path
    ).convert("RGB")

    # Apply ImageNet normalization
    tensor = transform(image)

    # Add batch dimension
    tensor = tensor.unsqueeze(0)

    return image, tensor


# ============================================================
# HELPER: CONVERT MODEL OUTPUT BACK TO IMAGE
# ============================================================

def tensor_to_image(tensor):

    tensor = tensor.squeeze(0).detach().cpu()

    # --------------------------------------------------------
    # Reverse ImageNet normalization
    #
    # normalized = (image - mean) / std
    #
    # therefore:
    #
    # image = normalized * std + mean
    # --------------------------------------------------------

    mean = torch.tensor(
        IMAGENET_MEAN,
        dtype=tensor.dtype
    ).view(
        3, 1, 1
    )

    std = torch.tensor(
        IMAGENET_STD,
        dtype=tensor.dtype
    ).view(
        3, 1, 1
    )

    tensor = (
        tensor * std
        + mean
    )

    # Keep valid image range
    tensor = tensor.clamp(
        0.0,
        1.0
    )

    image = transforms.ToPILImage()(
        tensor
    )

    return image


# ============================================================
# FIND INPUT IMAGES
# ============================================================

extensions = [
    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.bmp",
    "*.webp",

    "*.JPG",
    "*.JPEG",
    "*.PNG",
    "*.BMP",
    "*.WEBP"
]


image_paths = []

for ext in extensions:

    image_paths.extend(
        glob.glob(
            os.path.join(
                INPUT_FOLDER,
                ext
            )
        )
    )


image_paths = sorted(
    image_paths
)


# ============================================================
# CHECK INPUT
# ============================================================

if len(image_paths) == 0:

    raise RuntimeError(
        f"\nNo images found in:\n"
        f"{os.path.abspath(INPUT_FOLDER)}"
    )


print(
    f"\nFound {len(image_paths)} images."
)


# ============================================================
# CSV REPORT
# ============================================================

csv_path = os.path.join(
    OUTPUT_FOLDER,
    "deblur_results.csv"
)


csv_file = open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
)


writer = csv.writer(
    csv_file
)


writer.writerow([
    "Image",
    "Width",
    "Height",
    "Processing_Time_sec",

    "PSNR_Input",
    "PSNR_Output",
    "PSNR_Gain",

    "SSIM_Input",
    "SSIM_Output",
    "SSIM_Gain"
])


# ============================================================
# METRIC STORAGE
# ============================================================

psnr_input_values = []
psnr_output_values = []

ssim_input_values = []
ssim_output_values = []


total_time = 0.0


# ============================================================
# PROCESS IMAGES
# ============================================================

for index, image_path in enumerate(
    image_paths,
    start=1
):

    filename = os.path.basename(
        image_path
    )


    print("\n" + "-" * 70)

    print(
        f"[{index}/{len(image_paths)}] "
        f"{filename}"
    )


    # ========================================================
    # LOAD IMAGE
    # ========================================================

    blurred_pil, input_tensor = load_image(
        image_path
    )


    width, height = blurred_pil.size


    print(
        f"Original size : "
        f"{width} x {height}"
    )


    input_tensor = input_tensor.to(
        DEVICE
    )


    # ========================================================
    # INFERENCE
    # ========================================================

    start_time = time.time()


    with torch.no_grad():

        output_tensor = generator(
            input_tensor
        )


    elapsed = (
        time.time()
        - start_time
    )


    total_time += elapsed


    # ========================================================
    # CONVERT OUTPUT
    # ========================================================

    output_pil = tensor_to_image(
        output_tensor
    )


    # ========================================================
    # SAVE OUTPUT
    # ========================================================

    output_path = os.path.join(
        OUTPUT_FOLDER,
        filename
    )


    output_pil.save(
        output_path,
        quality=95
    )


    print(
        f"Processing time: "
        f"{elapsed:.3f} sec"
    )

    print(
        f"Saved to       : "
        f"{output_path}"
    )


