import os
import sys
import importlib.util
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import pywt
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# 1. Load LRFreqUNet
from model_lr import LRFreqUNet

# 2. Load VDSR safely to avoid model.py collision
spec = importlib.util.spec_from_file_location("vdsr_model", "../VDSR/model.py")
vdsr_module = importlib.util.module_from_spec(spec)
sys.modules["vdsr_model"] = vdsr_module
spec.loader.exec_module(vdsr_module)
VDSR = vdsr_module.VDSR

def extract_wavelet_numpy(img_gray):
    coeffs2 = pywt.dwt2(img_gray, 'haar')
    LL, (LH, HL, HH) = coeffs2
    return LL, LH, HL, HH

def verify():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Models
    vdsr_model = VDSR(in_channels=3).to(device)
    vdsr_model.load_state_dict(torch.load("../VDSR/weights/vdsr_epoch_100.pth", map_location=device))
    vdsr_model.eval()

    freq_model = LRFreqUNet(in_channels=4, out_channels=3).to(device)
    freq_model.load_state_dict(torch.load("./weights_lr_only/lrfrequnet_epoch_50.pth", map_location=device))
    freq_model.eval()

    # Load an unseen HR Image (Validation set)
    hr_path = "../VDSR/DIV2K/DIV2K_valid_HR/0801.png"
    hr_img = Image.open(hr_path).convert("RGB")
    
    # 전처리 (가로, 세로 길이를 256의 배수로 맞춰 비교)
    w, h = hr_img.size
    new_w = w - (w % 256)
    new_h = h - (h % 256)
    hr_img = hr_img.crop((0, 0, new_w, new_h))
    
    # Create LR by Downsampling (Bicubic, 1/4 size)
    lr_w, lr_h = new_w // 4, new_h // 4
    lr_img_small = hr_img.resize((lr_w, lr_h), Image.BICUBIC)
    
    # 1. Bicubic Upsampling
    lr_img_upsampled = lr_img_small.resize((new_w, new_h), Image.BICUBIC)
    
    # 2. VDSR Super Resolution
    lr_tensor = transforms.ToTensor()(lr_img_upsampled).unsqueeze(0).to(device)
    with torch.no_grad():
        _, sr_tensor = vdsr_model(lr_tensor)
    sr_img = transforms.ToPILImage()(sr_tensor.squeeze(0).cpu().clamp(0, 1))
    
    # 변환 (Grayscale for Frequency Analysis)
    lr_gray = np.array(lr_img_upsampled.convert("L")).astype(np.float32) / 255.0
    sr_gray = np.array(sr_img.convert("L")).astype(np.float32) / 255.0
    hr_gray = np.array(hr_img.convert("L")).astype(np.float32) / 255.0
    
    # 3. Wavelet Transform
    lr_ll, lr_lh, lr_hl, lr_hh = extract_wavelet_numpy(lr_gray)
    sr_ll, sr_lh, sr_hl, sr_hh = extract_wavelet_numpy(sr_gray)
    
    # 모델에 넣기 위해 텐서로 변환 (오직 LR 정보만: Batch=1, Channels=4)
    input_tensor = torch.cat([
        torch.from_numpy(lr_ll).unsqueeze(0), torch.from_numpy(lr_lh).unsqueeze(0),
        torch.from_numpy(lr_hl).unsqueeze(0), torch.from_numpy(lr_hh).unsqueeze(0)
    ], dim=0).unsqueeze(0).to(device)
    
    # 고주파수 예측
    with torch.no_grad():
        pred_high_freq = freq_model(input_tensor).squeeze(0).cpu() # [3, W/2, H/2]
        
    pred_lh = pred_high_freq[0].numpy()
    pred_hl = pred_high_freq[1].numpy()
    pred_hh = pred_high_freq[2].numpy()
    
    # IDWT (역 웨이블릿 변환) 1: SR(VDSR)의 저주파수(LL) + 예측 고주파수
    recon_coeffs_sr = (sr_ll, (pred_lh, pred_hl, pred_hh))
    recon_gray_sr = pywt.idwt2(recon_coeffs_sr, 'haar')
    recon_gray_sr = np.clip(recon_gray_sr, 0, 1)
    
    # IDWT (역 웨이블릿 변환) 2: 오직 LR의 저주파수(LL) + 예측 고주파수 (진짜 순수 100% LR 기반)
    recon_coeffs_lr = (lr_ll, (pred_lh, pred_hl, pred_hh))
    recon_gray_lr = pywt.idwt2(recon_coeffs_lr, 'haar')
    recon_gray_lr = np.clip(recon_gray_lr, 0, 1)
    
    # 정량적 평가 (PSNR, SSIM 계산)
    lr_psnr = psnr(hr_gray, lr_gray, data_range=1.0)
    lr_ssim = ssim(hr_gray, lr_gray, data_range=1.0)
    
    vdsr_psnr = psnr(hr_gray, sr_gray, data_range=1.0)
    vdsr_ssim = ssim(hr_gray, sr_gray, data_range=1.0)
    
    recon_sr_psnr = psnr(hr_gray, recon_gray_sr, data_range=1.0)
    recon_sr_ssim = ssim(hr_gray, recon_gray_sr, data_range=1.0)
    
    recon_lr_psnr = psnr(hr_gray, recon_gray_lr, data_range=1.0)
    recon_lr_ssim = ssim(hr_gray, recon_gray_lr, data_range=1.0)
    
    print(f"LR (Bicubic)      - PSNR: {lr_psnr:.2f}dB, SSIM: {lr_ssim:.4f}")
    print(f"VDSR SR           - PSNR: {vdsr_psnr:.2f}dB, SSIM: {vdsr_ssim:.4f}")
    print(f"Recon (SR_LL+Pred) - PSNR: {recon_sr_psnr:.2f}dB, SSIM: {recon_sr_ssim:.4f}")
    print(f"Recon (LR_LL+Pred) - PSNR: {recon_lr_psnr:.2f}dB, SSIM: {recon_lr_ssim:.4f}")
    
    # 시각화 (Visualization)
    fig, axes = plt.subplots(2, 5, figsize=(26, 10))
    fig.suptitle('LR-Only Frequency Predictor Verification (Unseen Image: 0801.png)', fontsize=20, fontweight='bold')
    
    # 윗줄: 공간 도메인 (실제 이미지 비교)
    axes[0, 0].imshow(lr_gray, cmap='gray')
    axes[0, 0].set_title(f'LR (Bicubic)\nPSNR: {lr_psnr:.2f}dB / SSIM: {lr_ssim:.4f}')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(sr_gray, cmap='gray')
    axes[0, 1].set_title(f'SR (VDSR)\nPSNR: {vdsr_psnr:.2f}dB / SSIM: {vdsr_ssim:.4f}')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(recon_gray_lr, cmap='gray')
    axes[0, 2].set_title(f'Recon (LR LL + Pred)\nPSNR: {recon_lr_psnr:.2f}dB / SSIM: {recon_lr_ssim:.4f}')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(recon_gray_sr, cmap='gray')
    axes[0, 3].set_title(f'Recon (SR LL + Pred)\nPSNR: {recon_sr_psnr:.2f}dB / SSIM: {recon_sr_ssim:.4f}')
    axes[0, 3].axis('off')
    
    axes[0, 4].imshow(hr_gray, cmap='gray')
    axes[0, 4].set_title('GT (HR Original)')
    axes[0, 4].axis('off')
    
    # 아랫줄: 고주파수 (엣지 맵 비교) - 복원력 확인
    axes[1, 0].imshow(np.abs(lr_lh) + np.abs(lr_hl) + np.abs(lr_hh), cmap='gray')
    axes[1, 0].set_title('LR High Frequencies (Blurry)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(np.abs(sr_lh) + np.abs(sr_hl) + np.abs(sr_hh), cmap='gray')
    axes[1, 1].set_title('SR High Frequencies (Smoothed)')
    axes[1, 1].axis('off')
    
    # 빈 칸 (줄 맞춤용)
    axes[1, 2].axis('off')
    
    axes[1, 3].imshow(np.abs(pred_lh) + np.abs(pred_hl) + np.abs(pred_hh), cmap='gray')
    axes[1, 3].set_title('Predicted High Freqs (LR-Only)')
    axes[1, 3].axis('off')
    
    # GT의 고주파수 (정답)
    hr_ll, hr_lh, hr_hl, hr_hh = extract_wavelet_numpy(hr_gray)
    axes[1, 4].imshow(np.abs(hr_lh) + np.abs(hr_hl) + np.abs(hr_hh), cmap='gray')
    axes[1, 4].set_title('GT High Frequencies')
    axes[1, 4].axis('off')
    
    plt.tight_layout()
    plt.savefig('verification_result_lr.png')
    print("Verification complete. Result saved to 'verification_result_lr.png'")

if __name__ == "__main__":
    verify()
