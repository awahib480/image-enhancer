# Image Resolution Enhancement and Deblurring

A deep learning project for improving image quality through super-resolution and image deblurring. The repository contains two trained models, evaluation scripts and instructions for downloading model checkpoints and reproducing evaluations.

The project focuses on two image restoration tasks:

- **Image Super-Resolution:** Enhances image resolution using a Super-Resolution Generative Adversarial Network (SRGAN).
- **Image Deblurring:** Restores sharpness and image details in blurred images using a trained deep learning model.

Model checkpoints are distributed through GitHub Releases to keep large binary files separate from the source code.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Models](#models)
  - [Image Resolution Enhancer](#1-image-resolution-enhancer)
  - [Image Deblurring](#2-image-deblurring)
- [Datasets](#datasets)
- [Training Environment](#training-environment)
- [Installation](#installation)
- [Downloading Model Checkpoints](#downloading-model-checkpoints)
- [Evaluation](#evaluation)
- [Evaluation Metrics](#evaluation-metrics)
- [Training and Reproducibility](#training-and-reproducibility)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Project Overview

Images captured under different conditions can suffer from low resolution, motion blur, defocus blur and loss of fine details. These degradations can affect image quality and limit the usefulness of images in downstream applications.

This project explores deep learning-based image restoration through two complementary approaches.

| Component | Image Resolution Enhancer | Image Deblurring |
|---|---|---|
| Task | Single-image super-resolution | Image deblurring |
| Model | SRGAN-based generator | Trained deblurring network |
| Input | Low-resolution image | Blurred image |
| Output | Higher-resolution image | Restored image |
| Primary objective | Recover perceptually convincing high-frequency details | Recover image structures and details lost through blur |
| Training platform | Kaggle | Kaggle |
| Model distribution | GitHub Releases | GitHub Releases |
| Evaluation | PSNR, SSIM and visual comparison | PSNR, SSIM and visual comparison |

The two models are trained and evaluated independently. They address different types of image degradation and should not be considered interchangeable.

## Repository Structure

The repository is organized to separate model implementations, evaluation utilities, datasets documentation and trained checkpoints.

```text
image-enhancer/
│
├── README.md
├── requirements.txt
│
├── resolution_enhancer/
│   ├── generator.py
│   ├── infer_and_evaluate_images.py
│   └── test_script.py
|   |__ training_log.txt
│
├── deblur/
│   ├── generator.py
│   ├── infer_and_evaluate_images.py
│   └── training_log.txt
│
├── notebooks/
│   ├── image-res.ipynb
│   └── deblur.ipynb
│
├── results/
│   ├── resolution_enhancer/
│   └── deblurring/
│
└── releases/
    └── README.md
```

---

## Models

### 1. Image Resolution Enhancer

The resolution enhancement component uses a Super-Resolution Generative Adversarial Network (SRGAN) to reconstruct higher-resolution images from low-resolution inputs.

#### Architecture

SRGAN consists of two primary networks:

**Generator**

The generator learns to map low-resolution images to higher-resolution images. Its objective is to reconstruct image details while producing perceptually convincing results.

**Discriminator**

The discriminator learns to distinguish generated super-resolution images from real high-resolution images. Its adversarial feedback encourages the generator to produce realistic textures and details.

The generator is the primary network required during inference.

#### Supported Enhancement Factors

The enhancement pipeline supports 2×, 4× and 8× output scaling by repeatedly applying the trained 2× super-resolution model.

| Enhancement Factor | Processing Pipeline |
|---|---|
| 2× | One SRGAN inference pass |
| 4× | Two successive SRGAN inference passes |
| 8× | Three successive SRGAN inference passes |

An optional sharpening pass can be applied after the final super-resolution stage to enhance perceived edge definition.

**Important:** Repeated application of a 2× model is a sequential enhancement strategy rather than a model specifically trained for direct 4× or 8× super-resolution. Repeated inference can amplify artifacts or introduce additional distortions. Results should therefore be evaluated separately for each scaling factor.

#### Training Data

The resolution enhancement model was trained using images from the following datasets:

- DIV2K
- Flickr2K

High-resolution images are used to construct corresponding low-resolution and high-resolution training pairs.

The low-resolution images are generated through downsampling. The resulting pairs allow the generator to learn a mapping from degraded inputs to their high-resolution counterparts.

A separate validation set containing 100 samples was retained for model evaluation.

#### Training Objective

SRGAN training combines adversarial learning with reconstruction-oriented objectives. The exact loss functions and weights should be documented in the training configuration.

Typical SRGAN implementations may use:

- Pixel or reconstruction loss
- Perceptual loss
- Adversarial loss

Refer to the training notebook and configuration files for the exact implementation used in this project.

#### Model Checkpoint

Download the resolution enhancement checkpoint from:

**[Resolution Enhancer Model Release]([https://github.com/awahib480/image-enhancer/releases/tag/v1.0])**

The release should contain the trained generator weights and, where applicable, the associated configuration or checkpoint metadata.

---

### 2. Image Deblurring

The deblurring component uses a trained deep learning model to restore images degraded by blur.

The model learns a mapping between blurred input images and their corresponding sharp reference images.

#### Processing Pipeline

The general inference pipeline consists of:

1. Load a blurred image.
2. Apply the preprocessing operations expected by the trained model.
3. Run inference using the trained deblurring network.
4. Convert the model output into an image.
5. Save the restored image for visual inspection or quantitative evaluation.

The exact architecture, input dimensions, normalization scheme and output processing should match the implementation associated with the released checkpoint.

#### Training Data

The deblurring model was trained using the RealBlur-J dataset.

RealBlur provides real-world blurred and sharp image pairs for image deblurring research. The J variant is used for this project.

The dataset split used in the training workflow is:

| Split | Number of Samples |
|---|---:|
| Training | 686 |
| Validation | 147 |
| Testing | 147 |
| **Total** | **980** |

The training and validation splits are used for model optimization and checkpoint selection. The test split should be reserved for final evaluation and should not be used for training or model selection.

Refer to the dataset documentation and the training notebook for the exact preprocessing and data-loading procedure.

#### Model Checkpoint

Download the deblurring checkpoint from:

**[Deblurring Model Release]([https://github.com/awahib480/image-enhancer/releases/tag/v1.0])**

The release should include the trained weights and any configuration required to reconstruct the model for inference.

---

## Datasets

The models use different datasets because super-resolution and deblurring involve different image degradation processes.

### Super-Resolution Datasets

| Dataset | Purpose | Official Resource |
|---|---|---|
| DIV2K | High-resolution images for super-resolution training | [DIV2K Dataset](https://www.kaggle.com/datasets/joe1995/div2k-dataset) |
| Flickr2K | Additional high-resolution training images | [Flickr2K Dataset](https://www.kaggle.com/datasets/daehoyang/flickr2k) |

**Dataset preparation:** High-resolution images are downsampled to generate low-resolution counterparts. These paired images are then used to train the super-resolution network.

### Deblurring Dataset

| Dataset | Purpose | Official Resource |
|---|---|---|
| RealBlur-J and GoPro | Real-world and GoPro blurred and sharp image pairs | [RealBlur and GoPro Dataset](https://www.kaggle.com/datasets/jishnuparayilshibu/a-curated-list-of-image-deblurring-datasets) |

**Dataset preparation:** Blurred images are paired with their corresponding sharp reference images. The model is trained to restore the sharp image from the blurred input.

### Dataset Availability

The datasets are not included in this repository.

Users must obtain the datasets from their respective sources and comply with the applicable dataset licenses and usage conditions.

**Training dataset links and references:**

- DIV2K: `https://www.kaggle.com/datasets/joe1995/div2k-dataset`
- Flickr2K: `https://www.kaggle.com/datasets/daehoyang/flickr2k`
- RealBlur-J: `https://www.kaggle.com/datasets/jishnuparayilshibu/a-curated-list-of-image-deblurring-datasets`

---

## Training Environment

Both models were trained using Kaggle notebooks and GPU-backed compute resources.

| Component | Details |
|---|---|
| Training platform | Kaggle |
| Framework | PyTorch |
| Hardware | Kaggle GPU environment |
| Resolution training data | DIV2K and Flickr2K |
| Deblurring training data | RealBlur-J and GoPro|
| Model distribution | GitHub Releases |
| Evaluation | Python-based evaluation scripts |

The exact GPU model, CUDA version, PyTorch version, training duration and hyperparameters should be recorded from the corresponding Kaggle notebook.

### Training References

- **Resolution enhancer notebook:** `https://www.kaggle.com/code/abdulwahib480/image-res`
- **Deblurring notebook:** `https://www.kaggle.com/code/abdulwahib480/deblur`

Training notebooks contain dataset preparation, model initialization, training loops, validation procedures and checkpoint selection.

The trained checkpoints distributed in Releases allow users to evaluate the models without repeating the full training process.

---

## Installation

### Prerequisites

Refer to requirements.txt

### Clone the Repository

```bash
git clone https://www.github.com/awahib480/image-enhancer
cd image-enhancer
```

### Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Downloading Model Checkpoints

Trained weights are provided through GitHub Releases rather than being stored directly in the source repository.

### Available Model Checkpoints

| Model | Release Asset | Purpose |
|---|---|---|
| Resolution Enhancer | `https://github.com/awahib480/image-enhancer/releases/download/v1.0/srgan_resume_checkpoint.pth` | Super-resolution inference |
| Deblurring Model | `https://github.com/awahib480/image-enhancer/releases/download/v1.0/deblur_resume_checkpoint.pth` | Image deblurring inference |

Download the appropriate checkpoint from the repository's Releases page:

**[View All Model Releases](https://github.com/awahib480/image-enhancer/releases)**

After downloading the files, place them in the locations expected by the corresponding inference and evaluation scripts.

### Checkpoint Compatibility

Before loading a checkpoint, ensure that:

- The model architecture matches the checkpoint.
- The checkpoint corresponds to the expected model and training configuration.
- The preprocessing and normalization steps are compatible.
- The inference script uses the correct device and tensor dimensions.

For PyTorch checkpoints, load trusted model files only. If the checkpoint contains a model state dictionary, initialize the corresponding model architecture and load its weights rather than assuming the file contains a complete serialized model.

---

## Evaluation

Evaluation scripts are provided to measure model performance on image pairs.

### 1. Evaluate the Resolution Enhancer

The evaluation procedure should:

1. Load the trained generator.
2. Load the low-resolution inputs and corresponding high-resolution references.
3. Apply the selected enhancement pipeline.
4. Compute the supported image quality metrics.
5. Report aggregate results across the evaluation samples.

Use the same reference images, image range, color space and metric implementation when comparing different enhancement factors.

### 2. Evaluate the Deblurring Model

The deblurring evaluation procedure should:

1. Load the trained deblurring model.
2. Read the blurred inputs and corresponding sharp references.
3. Apply the model's required preprocessing.
4. Generate restored images.
5. Calculate the supported evaluation metrics.
6. Report aggregate results over the test set.

Ensure that the test set is used only for final evaluation and that each prediction is compared with its corresponding sharp reference.

---

## Evaluation Metrics

The models can be evaluated using Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index Measure (SSIM).

### Peak Signal-to-Noise Ratio (PSNR)

PSNR measures the difference between a reconstructed image and its reference image using mean squared error.

Higher PSNR generally indicates lower pixel-level reconstruction error.

PSNR is expressed in decibels (dB).

### Structural Similarity Index Measure (SSIM)

SSIM measures structural similarity between two images by considering aspects such as luminance, contrast and structure.

Values closer to 1 generally indicate greater structural similarity.

### Metric Interpretation

| Metric | Higher Is Better? | What It Measures |
|---|---|---|
| PSNR | Generally, yes | Pixel-level reconstruction fidelity |
| SSIM | Generally, yes | Structural similarity |

PSNR and SSIM do not fully capture perceptual quality. A super-resolution model may produce visually convincing details without achieving the highest PSNR. Conversely, a model may achieve good numerical scores while producing perceptually less appealing images.

For this reason, quantitative metrics should be complemented by visual comparisons.

### Reporting Results

#### Resolution Enhancer

Results are available here:

https://github.com/awahib480/image-enhancer/tree/main/results/resolution_enhancer

#### Deblurring

Results are available here:

https://github.com/awahib480/image-enhancer/tree/main/results/deblurring

---

## Training and Reproducibility

The repository provides trained checkpoints for users who want to evaluate the models without repeating training.

For users who want to reproduce the training process, document the following information alongside the relevant notebook or configuration.

### Resolution Enhancer

- Training dataset versions and sources
- Training and validation split
- High-resolution image dimensions
- Low-resolution generation and downsampling method
- Generator and discriminator architecture
- Loss functions and their weights
- Optimizer and learning rate
- Batch size and number of epochs
- Random seeds and checkpoint selection criteria

### Deblurring Model

- RealBlur-J dataset version and split
- Image preprocessing and normalization
- Model architecture
- Loss function
- Optimizer and learning rate
- Batch size and number of epochs
- Random seeds and checkpoint selection criteria

### Experimental Considerations

For meaningful comparisons, keep the following consistent:

- Evaluation images and reference images
- Image dimensions and color space
- Metric implementation and image value range
- Dataset split and preprocessing
- Model checkpoint selection criteria

Exact reproduction may depend on the original software environment, hardware and training configuration.

---

## Limitations

- **Super-resolution is not information recovery:** Upscaling can reconstruct plausible details but cannot guarantee recovery of the original high-resolution image.
- **Repeated upscaling can amplify artifacts:** Applying the 2× model multiple times may introduce distortions or accumulate errors at higher output scales.
- **Deblurring is degradation-dependent:** Performance may vary with motion blur, defocus blur, noise and other distortions not adequately represented in the training data.
- **Dataset generalization is not guaranteed:** Results on benchmark datasets may differ from results on real-world images.
- **Quantitative metrics are incomplete:** PSNR and SSIM alone do not fully represent perceptual quality.
- **Hardware affects inference:** Processing time and memory consumption depend on image dimensions, model architecture and available compute resources.

---

## Future Improvements

Potential extensions include:

- More extensive evaluation on unseen datasets
- Comparison with bicubic interpolation and other super-resolution baselines
- Comparison with established deblurring methods
- Improved high-scale super-resolution strategies
- Perceptual quality assessment alongside PSNR and SSIM
- Inference speed and GPU memory benchmarking
- An interactive interface for uploading, processing and comparing images
- Automated generation of evaluation reports and visual comparisons

---

## License

Specify the license governing the source code and model weights.

**Repository license:** `MIT License`

**Dataset licenses:** Refer to the original terms of use for DIV2K, Flickr2K and RealBlur-J.

The repository license does not automatically override the licenses or usage restrictions associated with the datasets used to train the models.

---

## Acknowledgements

This project builds on research in deep learning-based image restoration and uses publicly available image datasets.

- **SRGAN:** Ledig et al., *Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network*. CVPR 2017.
- **DIV2K:** A high-quality image dataset introduced for image restoration and super-resolution research.
- **Flickr2K:** A high-resolution image dataset used in image super-resolution research.
- **RealBlur:** A real-world dataset for learning-based image deblurring.

Please consult the original papers and dataset documentation for full citations and usage conditions.

---

## Author

**Abdul Wahib**

GitHub: `www.github.com/awahib480`

Repository: `www.github.com/awahib480/image-enhancer`
