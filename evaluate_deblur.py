import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_tensor, to_pil_image

from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)

from generator import *


# ============================================================
# CONFIGURATION
# ============================================================

# Folder containing blurred input images
INPUT_FOLDER = Path(r"deblur/test/blur")

# Folder containing corresponding sharp images.
# Set to None if ground-truth images are unavailable.
#
# Example:
# GROUND_TRUTH_FOLDER = Path(r"test/sharp")
#
# Filenames must match those in INPUT_FOLDER.

GROUND_TRUTH_FOLDER = Path(r"deblur/test/sharp")

# Best trained deblurring generator
MODEL_PATH = Path(
    r"deblur\new training\epoch 151 to 200\best_generator.pth"
)

# Folder for deblurred images, comparisons and metrics
OUTPUT_FOLDER = Path(r"deblur/test/deblur")

# Set to None to process all images
MAX_IMAGES = None

# Resolution of saved comparison figures
SAVE_DPI = 200

# Supported image formats
SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


# ============================================================
# NORMALIZATION
# ============================================================
#
# These values MUST match your training transforms.
#
# Your original inference script uses ImageNet normalization.
# Therefore, this evaluation script uses the same normalization.
# ============================================================

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

normalize = transforms.Normalize(
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# IMAGE HELPERS
# ============================================================

def load_image(image_path):
    """
    Load an RGB image and return:
        PIL image
        Normalized image tensor with shape (1, 3, H, W)
    """

    with Image.open(image_path) as image_file:
        image = image_file.convert("RGB")

    tensor = normalize(to_tensor(image)).unsqueeze(0)

    return image, tensor


def tensor_to_image(tensor):
    """
    Convert the generator's ImageNet-normalized output
    back into an RGB PIL image.
    """

    tensor = tensor.detach().cpu().squeeze(0)

    mean = torch.tensor(
        IMAGENET_MEAN,
        dtype=tensor.dtype,
    ).view(3, 1, 1)

    std = torch.tensor(
        IMAGENET_STD,
        dtype=tensor.dtype,
    ).view(3, 1, 1)

    # Reverse ImageNet normalization
    tensor = tensor * std + mean

    tensor = tensor.clamp(0.0, 1.0)

    return to_pil_image(tensor)


# ============================================================
# MODEL LOADING
# ============================================================

def load_generator(device):
    """
    Load the trained DeblurGenerator.

    Supports:
        1. Raw state_dict
        2. Checkpoint containing generator_state_dict
        3. Checkpoint containing state_dict
        4. DataParallel checkpoints
    """

    generator = DeblurGenerator().to(device)

    try:
        checkpoint = torch.load(
            MODEL_PATH,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(
            MODEL_PATH,
            map_location=device,
        )

    if isinstance(checkpoint, dict):

        if "generator_state_dict" in checkpoint:
            state_dict = checkpoint["generator_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    # Remove DataParallel prefixes if present
    if (
        isinstance(state_dict, dict)
        and state_dict
        and all(
            key.startswith("module.")
            for key in state_dict.keys()
        )
    ):
        state_dict = {
            key.replace("module.", "", 1): value
            for key, value in state_dict.items()
        }

    generator.load_state_dict(
        state_dict,
        strict=True,
    )

    generator.eval()

    return generator


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(prediction, target):
    """
    Calculate PSNR and SSIM.

    Both tensors must:
        - Have shape (1, 3, H, W)
        - Contain values in [0, 1]
        - Have matching dimensions
    """

    psnr = peak_signal_noise_ratio(
        prediction,
        target,
        data_range=1.0,
    ).item()

    ssim = structural_similarity_index_measure(
        prediction,
        target,
        data_range=1.0,
    ).item()

    return psnr, ssim


# ============================================================
# VISUAL COMPARISON
# ============================================================

def save_comparison(
    output_path,
    image_name,
    blurred_image,
    deblurred_image,
    ground_truth_image=None,
    input_psnr=None,
    input_ssim=None,
    output_psnr=None,
    output_ssim=None,
    error_map=None,
):
    """
    Save a visual comparison.

    Without ground truth:
        Blurred Input | Deblurred Output

    With ground truth:
        Blurred Input | Deblurred Output |
        Ground Truth | Absolute Error Map
    """

    has_ground_truth = ground_truth_image is not None

    if has_ground_truth:
        fig, axes = plt.subplots(
            1,
            4,
            figsize=(22, 6),
            constrained_layout=True,
        )

    else:
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(12, 6),
            constrained_layout=True,
        )

    fig.suptitle(
        image_name,
        fontsize=16,
    )

    # --------------------------------------------------------
    # Blurred input
    # --------------------------------------------------------

    axes[0].imshow(blurred_image)

    axes[0].set_title(
        "Blurred Input",
        fontsize=12,
    )

    axes[0].axis("off")

    # --------------------------------------------------------
    # Deblurred output
    # --------------------------------------------------------

    if output_psnr is not None:

        output_title = (
            f"Deblurred Output\n"
            f"PSNR: {output_psnr:.2f} dB\n"
            f"SSIM: {output_ssim:.4f}"
        )

    else:

        output_title = "Deblurred Output"

    axes[1].imshow(deblurred_image)

    axes[1].set_title(
        output_title,
        fontsize=12,
    )

    axes[1].axis("off")

    # --------------------------------------------------------
    # Ground truth and error map
    # --------------------------------------------------------

    if has_ground_truth:

        axes[2].imshow(ground_truth_image)

        axes[2].set_title(
            "Ground Truth (Sharp)",
            fontsize=12,
        )

        axes[2].axis("off")

        heatmap = axes[3].imshow(
            error_map,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )

        axes[3].set_title(
            "Deblurred Output Absolute Error\n"
            "Mean RGB Difference from Ground Truth",
            fontsize=11,
        )

        axes[3].axis("off")

        fig.colorbar(
            heatmap,
            ax=axes[3],
            fraction=0.046,
            pad=0.04,
            label="Absolute Error",
        )

    fig.savefig(
        output_path,
        dpi=SAVE_DPI,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    # --------------------------------------------------------
    # Validate paths
    # --------------------------------------------------------

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not INPUT_FOLDER.is_dir():
        raise FileNotFoundError(
            f"Input folder not found: {INPUT_FOLDER}"
        )

    if GROUND_TRUTH_FOLDER is not None:

        if not GROUND_TRUTH_FOLDER.is_dir():
            raise FileNotFoundError(
                f"Ground-truth folder not found: "
                f"{GROUND_TRUTH_FOLDER}"
            )

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Find input images
    # --------------------------------------------------------

    image_paths = sorted(
        path
        for path in INPUT_FOLDER.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if MAX_IMAGES is not None:
        image_paths = image_paths[:MAX_IMAGES]

    if not image_paths:
        raise RuntimeError(
            f"No supported images found in {INPUT_FOLDER}"
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    generator = load_generator(DEVICE)

    # --------------------------------------------------------
    # Initialize metric storage
    # --------------------------------------------------------

    results = []

    total_input_psnr = 0.0
    total_input_ssim = 0.0

    total_output_psnr = 0.0
    total_output_ssim = 0.0

    total_time = 0.0
    metric_count = 0

    print("=" * 75)
    print("DEBLURRING EVALUATION")
    print("=" * 75)

    print(f"Device           : {DEVICE}")
    print(f"Model            : {MODEL_PATH}")
    print(f"Input folder     : {INPUT_FOLDER}")
    print(f"Ground truth     : {GROUND_TRUTH_FOLDER}")
    print(f"Output folder    : {OUTPUT_FOLDER}")
    print(f"Images to process: {len(image_paths)}")

    if GROUND_TRUTH_FOLDER is None:
        print(
            "\nGround truth unavailable. "
            "PSNR and SSIM will not be calculated."
        )

    print("=" * 75)

    # --------------------------------------------------------
    # Process images
    # --------------------------------------------------------

    with torch.inference_mode():

        for index, image_path in enumerate(
            image_paths,
            start=1,
        ):

            print(
                f"\n[{index}/{len(image_paths)}] "
                f"{image_path.name}"
            )

            # -----------------------------------------------
            # Load blurred input
            # -----------------------------------------------

            blurred_image, input_tensor = load_image(
                image_path
            )

            width, height = blurred_image.size

            input_tensor = input_tensor.to(DEVICE)

            # -----------------------------------------------
            # Load corresponding ground truth if available
            # -----------------------------------------------

            ground_truth_image = None
            target_tensor = None

            if GROUND_TRUTH_FOLDER is not None:

                ground_truth_path = (
                    GROUND_TRUTH_FOLDER / image_path.name
                )

                if ground_truth_path.is_file():

                    with Image.open(
                        ground_truth_path
                    ) as image_file:

                        ground_truth_image = (
                            image_file.convert("RGB")
                        )

                    # PSNR and SSIM require identical dimensions.
                    if ground_truth_image.size != blurred_image.size:

                        raise ValueError(
                            f"Dimension mismatch for "
                            f"{image_path.name}: "
                            f"input={blurred_image.size}, "
                            f"ground truth="
                            f"{ground_truth_image.size}"
                        )

                    target_tensor = to_tensor(
                        ground_truth_image
                    ).unsqueeze(0).to(DEVICE)

                else:

                    print(
                        "Warning: Matching ground-truth "
                        "image not found. Metrics unavailable."
                    )

            # -----------------------------------------------
            # Deblur inference
            # -----------------------------------------------

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            start_time = time.perf_counter()

            output_tensor = generator(input_tensor)

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start_time

            total_time += elapsed

            # -----------------------------------------------
            # Validate output dimensions
            # -----------------------------------------------

            if output_tensor.shape != input_tensor.shape:

                raise RuntimeError(
                    f"{image_path.name}: Input shape "
                    f"{tuple(input_tensor.shape)} does not "
                    f"match output shape "
                    f"{tuple(output_tensor.shape)}. "
                    f"Check the deblurring generator."
                )

            # -----------------------------------------------
            # Convert output to image
            # -----------------------------------------------

            deblurred_image = tensor_to_image(
                output_tensor
            )

            # -----------------------------------------------
            # Save deblurred output
            # -----------------------------------------------

            output_path = OUTPUT_FOLDER / image_path.name

            save_kwargs = {}

            if output_path.suffix.lower() in {
                ".jpg",
                ".jpeg",
            }:
                save_kwargs["quality"] = 95

            deblurred_image.save(
                output_path,
                **save_kwargs,
            )

            # -----------------------------------------------
            # Initialize per-image metrics
            # -----------------------------------------------

            input_psnr = None
            input_ssim = None

            output_psnr = None
            output_ssim = None

            psnr_gain = None
            ssim_gain = None

            error_map = None

            # -----------------------------------------------
            # Calculate metrics if ground truth exists
            # -----------------------------------------------

            if target_tensor is not None:

                # Convert input back to [0, 1]
                blurred_tensor = to_tensor(
                    blurred_image
                ).unsqueeze(0).to(DEVICE)

                # Convert prediction back to [0, 1]
                deblurred_tensor = to_tensor(
                    deblurred_image
                ).unsqueeze(0).to(DEVICE)

                # Input metrics
                input_psnr, input_ssim = calculate_metrics(
                    blurred_tensor,
                    target_tensor,
                )

                # Output metrics
                output_psnr, output_ssim = calculate_metrics(
                    deblurred_tensor,
                    target_tensor,
                )

                # Improvements over blurred input
                psnr_gain = output_psnr - input_psnr
                ssim_gain = output_ssim - input_ssim

                # Pixel-wise absolute error map
                error_map = torch.abs(
                    deblurred_tensor - target_tensor
                ).squeeze(0).mean(dim=0).cpu().numpy()

                # Accumulate dataset metrics
                total_input_psnr += input_psnr
                total_input_ssim += input_ssim

                total_output_psnr += output_psnr
                total_output_ssim += output_ssim

                metric_count += 1

                print(
                    f"  Input PSNR  : {input_psnr:.4f} dB"
                )

                print(
                    f"  Output PSNR : {output_psnr:.4f} dB"
                )

                print(
                    f"  PSNR Gain   : {psnr_gain:+.4f} dB"
                )

                print(
                    f"  Input SSIM  : {input_ssim:.4f}"
                )

                print(
                    f"  Output SSIM : {output_ssim:.4f}"
                )

                print(
                    f"  SSIM Gain   : {ssim_gain:+.4f}"
                )

            # -----------------------------------------------
            # Save visual comparison
            # -----------------------------------------------

            figure_path = OUTPUT_FOLDER / (
                f"{image_path.stem}_comparison.png"
            )

            save_comparison(
                output_path=figure_path,
                image_name=image_path.name,
                blurred_image=blurred_image,
                deblurred_image=deblurred_image,
                ground_truth_image=ground_truth_image,
                input_psnr=input_psnr,
                input_ssim=input_ssim,
                output_psnr=output_psnr,
                output_ssim=output_ssim,
                error_map=error_map,
            )

            print(
                f"  Image size   : {width} x {height}"
            )

            print(
                f"  Inference    : {elapsed:.4f} sec"
            )

            print(
                f"  Output       : {output_path}"
            )

            print(
                f"  Comparison   : {figure_path}"
            )

            # -----------------------------------------------
            # Store results for CSV
            # -----------------------------------------------

            results.append({
                "image": image_path.name,
                "width": width,
                "height": height,
                "processing_time_sec": elapsed,
                "input_psnr": input_psnr,
                "output_psnr": output_psnr,
                "psnr_gain": psnr_gain,
                "input_ssim": input_ssim,
                "output_ssim": output_ssim,
                "ssim_gain": ssim_gain,
            })

    # ========================================================
    # SAVE METRICS CSV
    # ========================================================

    csv_path = OUTPUT_FOLDER / "metrics.csv"

    fieldnames = [
        "image",
        "width",
        "height",
        "processing_time_sec",
        "input_psnr",
        "output_psnr",
        "psnr_gain",
        "input_ssim",
        "output_ssim",
        "ssim_gain",
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(results)

        # ----------------------------------------------------
        # Average metrics
        # ----------------------------------------------------

        if metric_count > 0:

            average_input_psnr = (
                total_input_psnr / metric_count
            )

            average_input_ssim = (
                total_input_ssim / metric_count
            )

            average_output_psnr = (
                total_output_psnr / metric_count
            )

            average_output_ssim = (
                total_output_ssim / metric_count
            )

            writer.writerow({
                "image": "AVERAGE",
                "input_psnr": average_input_psnr,
                "output_psnr": average_output_psnr,
                "psnr_gain": (
                    average_output_psnr - average_input_psnr
                ),
                "input_ssim": average_input_ssim,
                "output_ssim": average_output_ssim,
                "ssim_gain": (
                    average_output_ssim - average_input_ssim
                ),
            })

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print("\n" + "=" * 75)
    print("DEBLURRING EVALUATION SUMMARY")
    print("=" * 75)

    print(f"Images processed     : {len(results)}")

    print(
        f"Total inference time : {total_time:.4f} sec"
    )

    print(
        f"Average inference    : "
        f"{total_time / len(results):.4f} sec/image"
    )

    if metric_count > 0:

        print(
            f"\nImages with ground truth: {metric_count}"
        )

        print("\nAverage Results")
        print("-" * 45)

        print(
            f"Input PSNR  : "
            f"{average_input_psnr:.4f} dB"
        )

        print(
            f"Output PSNR : "
            f"{average_output_psnr:.4f} dB"
        )

        print(
            f"PSNR Gain   : "
            f"{average_output_psnr - average_input_psnr:+.4f} dB"
        )

        print()

        print(
            f"Input SSIM  : "
            f"{average_input_ssim:.4f}"
        )

        print(
            f"Output SSIM : "
            f"{average_output_ssim:.4f}"
        )

        print(
            f"SSIM Gain   : "
            f"{average_output_ssim - average_input_ssim:+.4f}"
        )

    else:

        print(
            "\nPSNR and SSIM were not calculated because "
            "no matching ground-truth images were available."
        )

    print("\nSaved results:")
    print(f"  Deblurred images : {OUTPUT_FOLDER}")
    print(f"  Metrics CSV      : {csv_path}")
    print("  Comparison plots : *_comparison.png")

    print("=" * 75)


if __name__ == "__main__":
    main()