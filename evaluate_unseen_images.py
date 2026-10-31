import csv
from pathlib import Path

from generator import *
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image, to_tensor


# ============================================================
# Configuration — change these paths
# ============================================================

MODEL_PATH = Path(r"resume_training\two-stage training\day 5\best_generator.pth")
IMAGE_FOLDER = Path(r"test images")
OUTPUT_FOLDER = Path(r"tested images")

MAX_IMAGES = None      # Use None to process every image in IMAGE_FOLDER.
SCALE = 2
SAVE_DPI = 200

SUPPORTED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"
}

# Must match the architecture used to train best_generator.pth.
GENERATOR_KWARGS = {
    "in_channels": 3,
    "num_channels": 64,
    "num_residual_blocks": 16,
    "upscale_factor": 2,
}


# ============================================================
# Helpers
# ============================================================

def pad_to_multiple(image_tensor, multiple=2):
    """
    Pads only the right and bottom edges using reflection padding.

    Input shape:  (1, C, H, W)
    Output:       padded_tensor, original_height, original_width
    """
    _, _, height, width = image_tensor.shape

    pad_height = (-height) % multiple
    pad_width = (-width) % multiple

    if pad_height == 0 and pad_width == 0:
        return image_tensor, height, width

    pad_mode = "reflect" if height > 1 and width > 1 else "replicate"

    padded_tensor = F.pad(
        image_tensor,
        (0, pad_width, 0, pad_height),
        mode=pad_mode
    )

    return padded_tensor, height, width


def crop_to_original(image_tensor, original_height, original_width):
    """Removes the added right/bottom padding."""
    return image_tensor[..., :original_height, :original_width]


def load_generator(device):
    generator = Generator(**GENERATOR_KWARGS).to(device)

    try:
        checkpoint = torch.load(
            MODEL_PATH,
            map_location=device,
            weights_only=True
        )
    except TypeError:
        checkpoint = torch.load(MODEL_PATH, map_location=device)

    # Supports both a plain state_dict and a checkpoint dictionary.
    if isinstance(checkpoint, dict) and "generator_state_dict" in checkpoint:
        state_dict = checkpoint["generator_state_dict"]
    else:
        state_dict = checkpoint

    # Supports DataParallel checkpoints.
    if all(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {
            key.replace("module.", "", 1): value
            for key, value in state_dict.items()
        }

    generator.load_state_dict(state_dict, strict=True)
    generator.eval()

    return generator


def calculate_metrics(prediction, target):
    """
    prediction and target must be in [0, 1]
    and have shape (1, 3, H, W).
    """
    psnr = peak_signal_noise_ratio(
        prediction,
        target,
        data_range=1.0
    ).item()

    ssim = structural_similarity_index_measure(
        prediction,
        target,
        data_range=1.0
    ).item()

    return psnr, ssim


def save_comparison(
    output_path,
    image_name,
    hr_image,
    lr_image,
    bicubic_image,
    srgan_image,
    bicubic_psnr,
    bicubic_ssim,
    srgan_psnr,
    srgan_ssim,
    error_map,
):
    """Save Ground Truth, LR, Bicubic, SRGAN, and error heatmap."""

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(25, 6),
        constrained_layout=True
    )

    fig.suptitle(image_name, fontsize=16)

    panels = [
        (hr_image, "Ground Truth (HR)"),
        (lr_image, "Low Resolution"),
        (
            bicubic_image,
            f"Bicubic\nPSNR: {bicubic_psnr:.2f} dB\nSSIM: {bicubic_ssim:.4f}"
        ),
        (
            srgan_image,
            f"SRGAN\nPSNR: {srgan_psnr:.2f} dB\nSSIM: {srgan_ssim:.4f}"
        ),
    ]

    for axis, (image, title) in zip(axes[:4], panels):
        axis.imshow(image)
        axis.set_title(title, fontsize=11)
        axis.axis("off")

    heatmap = axes[4].imshow(
        error_map,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0
    )

    axes[4].set_title(
        "SRGAN Absolute Error\n(mean RGB difference from HR)",
        fontsize=11
    )
    axes[4].axis("off")

    fig.colorbar(
        heatmap,
        ax=axes[4],
        fraction=0.046,
        pad=0.04,
        label="Absolute Error"
    )

    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main evaluation
# ============================================================

def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not IMAGE_FOLDER.is_dir():
        raise FileNotFoundError(f"Image folder not found: {IMAGE_FOLDER}")

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path
        for path in IMAGE_FOLDER.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if MAX_IMAGES is not None:
        image_paths = image_paths[:MAX_IMAGES]

    if not image_paths:
        raise RuntimeError("No supported images were found in IMAGE_FOLDER.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = load_generator(device)

    normalize = transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )

    bicubic_mode = getattr(Image, "Resampling", Image).BICUBIC

    total_bicubic_psnr = 0.0
    total_bicubic_ssim = 0.0
    total_srgan_psnr = 0.0
    total_srgan_ssim = 0.0

    results = []

    print(f"Using device: {device}")
    print(f"Evaluating {len(image_paths)} image(s)")
    print("=" * 65)

    with torch.inference_mode():
        for index, image_path in enumerate(image_paths, start=1):

            # Load original image as HR ground truth.
            with Image.open(image_path) as image_file:
                hr_image = image_file.convert("RGB")

            hr_original_tensor = to_tensor(hr_image).unsqueeze(0)

            # Pad HR first if its dimensions are odd.
            # This avoids shape mismatches while retaining the full image.
            hr_padded_tensor, original_height, original_width = pad_to_multiple(
                hr_original_tensor,
                multiple=SCALE
            )

            hr_padded_image = to_pil_image(hr_padded_tensor.squeeze(0))
            padded_width, padded_height = hr_padded_image.size

            # Create LR image using 2x bicubic downsampling.
            lr_image = hr_padded_image.resize(
                (padded_width // SCALE, padded_height // SCALE),
                bicubic_mode
            )

            # Bicubic baseline.
            bicubic_padded_image = lr_image.resize(
                (padded_width, padded_height),
                bicubic_mode
            )

            bicubic_tensor = to_tensor(
                bicubic_padded_image
            ).unsqueeze(0)

            bicubic_tensor = crop_to_original(
                bicubic_tensor,
                original_height,
                original_width
            )

            # SRGAN inference.
            lr_tensor = to_tensor(lr_image)
            lr_tensor = normalize(lr_tensor).unsqueeze(0).to(device)

            sr_padded_tensor = generator(lr_tensor)

            expected_size = (padded_height, padded_width)
            if sr_padded_tensor.shape[-2:] != expected_size:
                raise RuntimeError(
                    f"{image_path.name}: Expected output size {expected_size}, "
                    f"but got {tuple(sr_padded_tensor.shape[-2:])}. "
                    "Check that you loaded the correct 2x generator."
                )

            # Generator output: [-1, 1] → [0, 1].
            sr_padded_tensor = torch.clamp(
                (sr_padded_tensor + 1.0) / 2.0,
                0.0,
                1.0
            )

            sr_tensor = crop_to_original(
                sr_padded_tensor,
                original_height,
                original_width
            )

            # Compute metrics using original, unpadded HR dimensions.
            hr_target = hr_original_tensor.to(device)
            bicubic_device = bicubic_tensor.to(device)

            bicubic_psnr, bicubic_ssim = calculate_metrics(
                bicubic_device,
                hr_target
            )

            srgan_psnr, srgan_ssim = calculate_metrics(
                sr_tensor,
                hr_target
            )

            # Convert tensors for figure saving.
            bicubic_image = to_pil_image(bicubic_tensor.squeeze(0))
            srgan_image = to_pil_image(sr_tensor.squeeze(0).cpu())

            # Mean absolute RGB difference: H × W heatmap.
            error_map = torch.abs(
                sr_tensor.cpu() - hr_original_tensor
            ).squeeze(0).mean(dim=0).numpy()

            figure_path = OUTPUT_FOLDER / (
                f"{image_path.stem}_comparison.png"
            )

            save_comparison(
                figure_path,
                image_path.name,
                hr_image,
                lr_image,
                bicubic_image,
                srgan_image,
                bicubic_psnr,
                bicubic_ssim,
                srgan_psnr,
                srgan_ssim,
                error_map,
            )

            psnr_gain = srgan_psnr - bicubic_psnr
            ssim_gain = srgan_ssim - bicubic_ssim

            results.append({
                "image": image_path.name,
                "bicubic_psnr": bicubic_psnr,
                "bicubic_ssim": bicubic_ssim,
                "srgan_psnr": srgan_psnr,
                "srgan_ssim": srgan_ssim,
                "psnr_gain": psnr_gain,
                "ssim_gain": ssim_gain,
            })

            total_bicubic_psnr += bicubic_psnr
            total_bicubic_ssim += bicubic_ssim
            total_srgan_psnr += srgan_psnr
            total_srgan_ssim += srgan_ssim

            print(f"[{index}/{len(image_paths)}] {image_path.name}")
            print(
                f"  Bicubic — PSNR: {bicubic_psnr:.4f} dB | "
                f"SSIM: {bicubic_ssim:.4f}"
            )
            print(
                f"  SRGAN   — PSNR: {srgan_psnr:.4f} dB | "
                f"SSIM: {srgan_ssim:.4f}"
            )
            print(
                f"  Gain    — PSNR: {psnr_gain:+.4f} dB | "
                f"SSIM: {ssim_gain:+.4f}"
            )

    count = len(results)

    average_bicubic_psnr = total_bicubic_psnr / count
    average_bicubic_ssim = total_bicubic_ssim / count
    average_srgan_psnr = total_srgan_psnr / count
    average_srgan_ssim = total_srgan_ssim / count

    # Save detailed results.
    csv_path = OUTPUT_FOLDER / "metrics.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "image",
            "bicubic_psnr",
            "bicubic_ssim",
            "srgan_psnr",
            "srgan_ssim",
            "psnr_gain",
            "ssim_gain",
        ]

        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

        writer.writerow({
            "image": "AVERAGE",
            "bicubic_psnr": average_bicubic_psnr,
            "bicubic_ssim": average_bicubic_ssim,
            "srgan_psnr": average_srgan_psnr,
            "srgan_ssim": average_srgan_ssim,
            "psnr_gain": average_srgan_psnr - average_bicubic_psnr,
            "ssim_gain": average_srgan_ssim - average_bicubic_ssim,
        })

    print("=" * 65)
    print("Average Results\n")
    print(f"Bicubic PSNR : {average_bicubic_psnr:.4f} dB")
    print(f"Bicubic SSIM : {average_bicubic_ssim:.4f}")
    print()
    print(f"SRGAN PSNR   : {average_srgan_psnr:.4f} dB")
    print(f"SRGAN SSIM   : {average_srgan_ssim:.4f}")
    print()
    print(
        f"PSNR Gain    : "
        f"{average_srgan_psnr - average_bicubic_psnr:+.4f} dB"
    )
    print(
        f"SSIM Gain    : "
        f"{average_srgan_ssim - average_bicubic_ssim:+.4f}"
    )
    print("=" * 65)
    print(f"Figures and metrics.csv saved to:\n{OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()