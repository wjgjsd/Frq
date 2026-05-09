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

# 1. Load FreqUNet
from model import FreqUNet

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

    freq_model = FreqUNet(in_channels=8, out_channels=3).to(device)
    freq_model.load_state_dict(torch.load("./weights/frequnet_epoch_100.pth", map_location=device))
    freq_model.eval()

    # Load an unseen HR Image (Validation set)
    hr_path = "../VDSR/DIV2K/DIV2K_valid_HR/0801.png"
    hr_img = Image.open(hr_path).convert("RGB")
    
    # 디테일을 자세히 보기 위해 이미지 중앙을 512x512 크기로 자릅니다.
    transform_crop = transforms.CenterCrop(512)
    hr_img = transform_crop(hr_img)
    
    # Create LR bicubic
    lr_img = hr_img.resize((128, 128), Image.BICUBIC).resize((512, 512), Image.BICUBIC)
    
    # Generate SR (VDSR)
    to_tensor = transforms.ToTensor()
    lr_tensor = to_tensor(lr_img).unsqueeze(0).to(device)
    with torch.no_grad():
        _, sr_tensor = vdsr_model(lr_tensor)
        sr_tensor = torch.clamp(sr_tensor, 0, 1)
    
    # Convert to PIL
    to_pil = transforms.ToPILImage()
    sr_img = to_pil(sr_tensor.squeeze(0).cpu())
    
    # Convert to grayscale for frequency analysis
    hr_gray = np.array(hr_img.convert("L"), dtype=np.float32) / 255.0
    lr_gray = np.array(lr_img.convert("L"), dtype=np.float32) / 255.0
    sr_gray = np.array(sr_img.convert("L"), dtype=np.float32) / 255.0
    
    # Extract Wavelets
    lr_ll, lr_lh, lr_hl, lr_hh = extract_wavelet_numpy(lr_gray)
    sr_ll, sr_lh, sr_hl, sr_hh = extract_wavelet_numpy(sr_gray)
    hr_ll, hr_lh, hr_hl, hr_hh = extract_wavelet_numpy(hr_gray)
    
    # 모델에 넣기 위해 텐서로 변환 (Batch=1, Channels=8)
    input_tensor = torch.cat([
        torch.from_numpy(lr_ll).unsqueeze(0), torch.from_numpy(lr_lh).unsqueeze(0),
        torch.from_numpy(lr_hl).unsqueeze(0), torch.from_numpy(lr_hh).unsqueeze(0),
        torch.from_numpy(sr_ll).unsqueeze(0), torch.from_numpy(sr_lh).unsqueeze(0),
        torch.from_numpy(sr_hl).unsqueeze(0), torch.from_numpy(sr_hh).unsqueeze(0)
    ], dim=0).unsqueeze(0).to(device)
    
    # 고주파수 예측
    with torch.no_grad():
        pred_high_freq = freq_model(input_tensor).squeeze(0).cpu() # [3, 256, 256]
        
    pred_lh = pred_high_freq[0].numpy()
    pred_hl = pred_high_freq[1].numpy()
    pred_hh = pred_high_freq[2].numpy()
    
    # IDWT (역 웨이블릿 변환)을 이용한 이미지 재구성 (Reconstruction)
    # 잔차 학습을 제거했으므로, 예측된 고주파수(pred_lh, pred_hl, pred_hh)를 직접 사용합니다.
    # 뼈대는 여전히 안정적인 SR(VDSR)의 저주파수(LL)를 사용합니다.
    recon_coeffs = (sr_ll, (pred_lh, pred_hl, pred_hh))
    recon_gray = pywt.idwt2(recon_coeffs, 'haar')
    recon_gray = np.clip(recon_gray, 0, 1)
    
    # 정량적 평가 (PSNR, SSIM 계산)
    vdsr_psnr = psnr(hr_gray, sr_gray, data_range=1.0)
    vdsr_ssim = ssim(hr_gray, sr_gray, data_range=1.0)
    
    recon_psnr = psnr(hr_gray, recon_gray, data_range=1.0)
    recon_ssim = ssim(hr_gray, recon_gray, data_range=1.0)
    
    print(f"VDSR SR - PSNR: {vdsr_psnr:.2f}dB, SSIM: {vdsr_ssim:.4f}")
    print(f"FreqUNet Recon - PSNR: {recon_psnr:.2f}dB, SSIM: {recon_ssim:.4f}")
    
    # 시각화 (Visualization)
    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    fig.suptitle('Frequency Predictor Verification (Unseen Image: 0801.png)', fontsize=20, fontweight='bold')
    
    # 윗줄: 공간 도메인 (실제 이미지 비교)
    axes[0, 0].imshow(lr_gray, cmap='gray')
    axes[0, 0].set_title('LR (Bicubic)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(sr_gray, cmap='gray')
    axes[0, 1].set_title(f'SR (VDSR)\nPSNR: {vdsr_psnr:.2f}dB / SSIM: {vdsr_ssim:.4f}')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(recon_gray, cmap='gray')
    axes[0, 2].set_title(f'Reconstructed (FreqUNet IDWT)\nPSNR: {recon_psnr:.2f}dB / SSIM: {recon_ssim:.4f}')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(hr_gray, cmap='gray')
    axes[0, 3].set_title('Original HR (GT)')
    axes[0, 3].axis('off')
    
    # 아랫줄: 주파수 도메인 (고주파수 총합 시각화)
    sr_high = np.abs(sr_lh) + np.abs(sr_hl) + np.abs(sr_hh)
    pred_high = np.abs(pred_lh) + np.abs(pred_hl) + np.abs(pred_hh)
    hr_high = np.abs(hr_lh) + np.abs(hr_hl) + np.abs(hr_hh)
    lr_high = np.abs(lr_lh) + np.abs(lr_hl) + np.abs(lr_hh)
    
    axes[1, 0].imshow(np.log(lr_high + 1), cmap='gray')
    axes[1, 0].set_title('LR High Frequencies')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(np.log(sr_high + 1), cmap='gray')
    axes[1, 1].set_title('SR High Frequencies')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(np.log(pred_high + 1), cmap='gray')
    axes[1, 2].set_title('Predicted High Frequencies')
    axes[1, 2].axis('off')
    
    axes[1, 3].imshow(np.log(hr_high + 1), cmap='gray')
    axes[1, 3].set_title('True HR High Frequencies')
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('verification_result.png', dpi=300)
    print("Verification complete. Result saved to 'verification_result.png'")

if __name__ == "__main__":
    verify()
