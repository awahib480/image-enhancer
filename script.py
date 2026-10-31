import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
from generator import *
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure

# ======================================================
# Configuration
# ======================================================

MODEL_PATH = r"resume_training\two-stage training\day 5\best_generator.pth"

VALID_FOLDER = r"DIV2K_valid_HR"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================================
# Load Generator
# ======================================================

generator = Generator().to(device)

generator.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

generator.eval()

# ======================================================
# Metrics
# ======================================================

psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

# ======================================================
# Transform (same as training)
# ======================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# ======================================================
# Running totals
# ======================================================

bicubic_psnr = 0
bicubic_ssim = 0

srgan_psnr = 0
srgan_ssim = 0

count = 0

# ======================================================
# Evaluation
# ======================================================

with torch.no_grad():

    for filename in sorted(os.listdir(VALID_FOLDER)):

        if not filename.lower().endswith(extensions):
            continue

        path = os.path.join(VALID_FOLDER, filename)

        # ------------------------------------------
        # Ground Truth
        # ------------------------------------------

        hr = Image.open(path).convert("RGB")

        w, h = hr.size

        # ------------------------------------------
        # Create LR exactly like training
        # ------------------------------------------

        lr = hr.resize(
            (w // 2, h // 2),
            Image.BICUBIC
        )

        # ------------------------------------------
        # Bicubic Upscale
        # ------------------------------------------

        bicubic = lr.resize(
            (w, h),
            Image.BICUBIC
        )

        # ------------------------------------------
        # Generator Input
        # ------------------------------------------

        lr_tensor = transform(lr).unsqueeze(0).to(device)

        # ------------------------------------------
        # SRGAN
        # ------------------------------------------

        sr = generator(lr_tensor)

        sr = sr.squeeze(0).cpu()

        # Undo normalization
        sr = sr * 0.5 + 0.5
        sr = torch.clamp(sr, 0, 1)

        sr = sr.unsqueeze(0).to(device)

        # ------------------------------------------
        # Ground Truth Tensor
        # ------------------------------------------

        hr_tensor = to_tensor(hr).unsqueeze(0).to(device)

        # ------------------------------------------
        # Bicubic Tensor
        # ------------------------------------------

        bicubic_tensor = to_tensor(bicubic).unsqueeze(0).to(device)

        # ------------------------------------------
        # Metrics
        # ------------------------------------------

        bicubic_psnr += psnr(bicubic_tensor, hr_tensor).item()
        bicubic_ssim += ssim(bicubic_tensor, hr_tensor).item()

        srgan_psnr += psnr(sr, hr_tensor).item()
        srgan_ssim += ssim(sr, hr_tensor).item()

        count += 1

        print(f"{filename}")

# ======================================================
# Average
# ======================================================

bicubic_psnr /= count
bicubic_ssim /= count

srgan_psnr /= count
srgan_ssim /= count

# ======================================================
# Results
# ======================================================

print("\n" + "=" * 60)

print("Average Results\n")

print(f"Bicubic PSNR : {bicubic_psnr:.4f}")
print(f"Bicubic SSIM : {bicubic_ssim:.4f}")

print()

print(f"SRGAN PSNR   : {srgan_psnr:.4f}")
print(f"SRGAN SSIM   : {srgan_ssim:.4f}")

print()

print(f"PSNR Gain    : {srgan_psnr - bicubic_psnr:.4f} dB")
print(f"SSIM Gain    : {srgan_ssim - bicubic_ssim:.4f}")

print("=" * 60)