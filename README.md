# Frequency Domain Super Resolution 🚀

This repository contains the official implementation of our novel **Frequency Domain Super Resolution** approach. Instead of directly upsampling pixels in the spatial domain like conventional SR models (e.g., VDSR, SRCNN), our model operates exclusively in the **Wavelet Frequency Domain**.

By separating the image into low-frequency (LL) and high-frequency (LH, HL, HH) sub-bands using the 2D Haar Wavelet Transform, our model (`FreqUNet`) is trained to predict missing high-frequency details (sharp edges and textures) that are typically lost during standard bicubic upsampling.

## 🌟 Key Features

1. **Pure Frequency Prediction**: Predicts missing high-frequency sub-bands directly from LR inputs.
2. **Dual-Model Architecture (Ablation Study)**:
   - `LRFreqUNet` (4-channel): Purely predicts edges from LR (Low-Resolution) images.
   - `FreqUNet` (8-channel): Uses standard SR output (e.g., VDSR) as a structural hint to push performance beyond the baseline.
3. **Advanced Attention Mechanisms**: Integrates **CBAM** (Channel and Spatial Attention) within deeper Residual Blocks to capture complex texture patterns.
4. **Weighted Wavelet Loss**: A customized L1 loss function that applies a 2x penalty to the HH (diagonal edge) band, forcing the model to recover the sharpest, most difficult details.

## 📊 Performance (DIV2K Validation Set - 100 images)

We rigorously tested our models on the entire DIV2K validation dataset. By simply predicting the high-frequency 5% and adding it to the low-frequency base, we achieved significant PSNR/SSIM gains:

| Model / Approach | PSNR (dB) | SSIM |
| :--- | :---: | :---: |
| LR (Bicubic Baseline) | 26.59 | 0.7641 |
| VDSR (SR Baseline) | 27.86 | 0.8060 |
| **Recon (LR_LL + Pred)** | **26.83** | **0.7702** | 
| **Recon (SR_LL + Pred)** | **27.89** | **0.8063** |

*Note: Our 8-Channel model successfully surpasses the VDSR baseline (27.89dB vs 27.86dB).*

## 🚀 How to Run

### 1. Data Preparation
Prepare the DIV2K patches (LR, SR, HR) and extract their wavelet transforms:
```bash
python prepare_dataset.py
```

### 2. Training
We provide two separate training pipelines. The model dynamically reduces the learning rate when the loss plateaus.
```bash
# Train the 8-Channel Model (LR + SR Hint)
python train.py

# Train the 4-Channel Independent Model (LR Only)
python train_lr.py
```

### 3. Evaluation & Verification
Run the evaluation scripts to compute PSNR/SSIM across the entire DIV2K validation set:
```bash
# Evaluate 8-Channel Model
python evaluate.py

# Evaluate 4-Channel Model
python evaluate_lr.py
```

To visually verify and generate a comparison plot for a single image:
```bash
python verify.py
python verify_lr.py
```

## 🧠 Insights from Ablation Studies
During our research, we discovered that **L1 Loss** strongly outperforms Smooth L1 (Huber) or L2 loss in the frequency domain. Because wavelet high-frequency maps are highly sparse (mostly zeroes with sharp spikes at edges), L2-based losses encourage "safe" blurry edges to minimize overall penalty, which mathematically lowers the loss but ruins real-world PSNR and SSIM. Therefore, we firmly rely on pure L1 Loss without phase-inverting data augmentations (like random flips) to achieve optimal edge recovery.
