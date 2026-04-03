# Enhanced Deep Super-Resolution (EDSR) — CNN Project

Implementation and architectural analysis of **Enhanced Deep Super-Resolution (EDSR)** for single-image super-resolution using **PyTorch**.

---

## Overview

This project tackles the problem of **image super-resolution**, where a high-resolution (HR) image is reconstructed from a low-resolution (LR) input.

We implement the **EDSR architecture**, a ResNet-based deep CNN optimized specifically for super-resolution tasks. Unlike traditional interpolation methods, EDSR learns a **data-driven mapping from LR → HR images**, enabling recovery of high-frequency details such as edges and textures.

The complete theoretical and experimental analysis is documented in the report:
`report/Architectural Analysis of EDSR Models for SuperResolution.pdf`

---

## Objectives

* Implement EDSR from scratch using PyTorch
* Understand residual learning in image reconstruction
* Evaluate performance using perceptual and pixel-based metrics
* Analyze how architectural choices affect reconstruction quality

---

## Key Concepts

* Residual Learning (ResBlocks with skip connections)
* PixelShuffle (Sub-pixel convolution for upsampling)
* Patch-based training for memory efficiency
* LR–HR supervised learning pipeline
* Perceptual evaluation (LPIPS)

---

## Model Architecture

The implemented EDSR model follows a **three-stage pipeline**:

1. **Feature Extraction (Head)**
2. **Residual Learning (Body)**
3. **Upsampling + Reconstruction**

### Architecture Details

* 32 Residual Blocks
* 192 Feature Channels
* Residual Scaling: 0.1
* Upsampling: PixelShuffle (×4)
* Loss Function: L1 Loss
* Optimizer: Adam

As explained in the report, **batch normalization is removed** because it negatively affects pixel-level reconstruction quality. 

---

## Training Pipeline

The training workflow implemented in the notebook:

### 1. Data Preparation

* Dataset: DIV2K (800 train / 100 validation images)
* Images are converted from BGR → RGB
* Normalized to [0, 1]
* RGB mean subtraction applied

### 2. Patch Extraction

* HR patches: 96×96 → 160×160
* Corresponding LR patches generated using bicubic downsampling
* Scaling factor: ×4

Example:

* HR: 160×160
* LR: 40×40

### 3. Data Augmentation

* Horizontal flip
* Vertical flip
* Random rotations (90° multiples)

---

## Implementation (Code)

All implementation is provided in:

```
notebooks/EDSR_SuperResolution.ipynb
```

### Includes:

* EDSR model definition
* Residual block implementation
* Training loop
* Evaluation pipeline
* Visualization tools

---

## Experiments

We trained multiple configurations to analyze architectural impact:

| Model          | Blocks | Channels | Patch   | Epochs  |
| -------------- | ------ | -------- | ------- | ------- |
| EDSR-Small     | 16     | 64       | 96      | 60      |
| EDSR-Medium    | 32     | 128      | 96      | 100     |
| EDSR-Mid-Long  | 32     | 128      | 128     | 300     |
| EDSR-40-Blocks | 40     | 128      | 128     | 300     |
| **EDSR-Best**  | **32** | **192**  | **160** | **235** |

---

## Results

### Best Model Performance

* **PSNR:** 27.07 dB
* **SSIM:** 0.733
* **LPIPS:** 0.278

These results demonstrate strong performance across:

* Pixel accuracy (PSNR)
* Structural similarity (SSIM)
* Perceptual realism (LPIPS)

---

## Training Behavior

From the report:

* Rapid initial learning of low-frequency structures
* Stable convergence after ~150 epochs
* Learning rate scheduler improves final performance

Loss stabilizes around ~0.03, indicating convergence. 

---

## Visual Results

The model is compared against:

* Low-resolution input
* Bicubic interpolation
* Ground truth

### Observations:

* Bicubic → smooth but blurry
* EDSR → sharper edges + better textures
* Improved structural continuity

As shown in the report (Patch-level comparison section), EDSR reconstructs high-frequency details significantly better than interpolation methods. 

---

## Evaluation Metrics

### 1. PSNR (Peak Signal-to-Noise Ratio)

* Measures pixel-level similarity
* Higher is better

### 2. SSIM (Structural Similarity Index)

* Measures structural and perceptual similarity
* Range: 0–1

### 3. LPIPS (Perceptual Metric)

* Measures similarity in feature space
* Lower is better

The report highlights that **PSNR alone is insufficient**, and perceptual metrics like LPIPS are essential for evaluating realism. 

---

## Analysis & Insights

### What Worked

* Increasing channels → better feature representation
* Larger patch size → better global context
* Longer training → improved perceptual quality

### Limitations

* High-frequency textures (leaf veins, fine edges) remain challenging
* Errors concentrated around edges
* Model struggles with micro-structures

Error map analysis in the report shows structured failure patterns in high-frequency regions. 

---

## Key Findings

* Residual learning enables deep architectures without instability
* Removing batch normalization improves SR performance
* LPIPS improves significantly with:

  * Larger patch size
  * Longer training
* Performance saturates beyond certain depth

---

## Future Improvements

* Implement **EDSR+ (256 channels)**
* Convert to modular production code (`src/`)
* Add inference script for real-world usage

---

## Contributions

* Implemented EDSR architecture and training pipeline
* Designed evaluation metrics and visualization system
* Conducted architectural experiments
* Performed detailed quantitative and qualitative analysis

---

## References

* Bee Lim et al. (2017) — EDSR
* Wenzhe Shi et al. (2016) — PixelShuffle
* Zhao et al. (2015) — Loss functions

(Full references available in report)

---

## Takeaways

This project demonstrates:

* Deep understanding of CNN architectures
* Ability to design and evaluate ML systems
* Strong experimentation and analysis skills
* Balance between theory and implementation

---

