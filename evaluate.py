import os
import sys
import glob
import importlib.util
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import pywt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

# 1. Load Original 8-channel FreqUNet
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

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Models
    print("Loading models...")
    vdsr_model = VDSR(in_channels=3).to(device)
    vdsr_model.load_state_dict(torch.load("../VDSR/weights/vdsr_epoch_100.pth", map_location=device))
    vdsr_model.eval()

    # 기존의 LR + SR 기반 8채널 예측 모델 로드
    freq_model = FreqUNet(in_channels=8, out_channels=3).to(device)
    freq_model.load_state_dict(torch.load("./weights/frequnet_epoch_50.pth", map_location=device))
    freq_model.eval()

    # Get all validation HR images
    valid_dir = "../VDSR/DIV2K/DIV2K_valid_HR/"
    image_paths = sorted(glob.glob(os.path.join(valid_dir, "*.png")))
    
    if not image_paths:
        print(f"No validation images found in {valid_dir}")
        return
        
    print(f"Found {len(image_paths)} validation images. Starting evaluation (LR+SR 8-channel model)...")
    
    # 누적 점수 저장용 딕셔너리
    metrics = {
        'lr_psnr': 0.0, 'lr_ssim': 0.0,
        'vdsr_psnr': 0.0, 'vdsr_ssim': 0.0,
        'recon_lr_psnr': 0.0, 'recon_lr_ssim': 0.0,
        'recon_sr_psnr': 0.0, 'recon_sr_ssim': 0.0
    }
    
    for img_path in tqdm(image_paths, desc="Evaluating"):
        hr_img = Image.open(img_path).convert("RGB")
        
        # 전처리 (가로, 세로 길이를 256의 배수로 맞춰 비교)
        w, h = hr_img.size
        new_w = w - (w % 256)
        new_h = h - (h % 256)
        
        # 너무 작은 이미지는 패스
        if new_w < 256 or new_h < 256:
            continue
            
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
        
        # 모델에 넣기 위해 텐서로 변환 (LR + SR 정보 모두: Batch=1, Channels=8)
        input_tensor = torch.cat([
            torch.from_numpy(lr_ll).unsqueeze(0), torch.from_numpy(lr_lh).unsqueeze(0),
            torch.from_numpy(lr_hl).unsqueeze(0), torch.from_numpy(lr_hh).unsqueeze(0),
            torch.from_numpy(sr_ll).unsqueeze(0), torch.from_numpy(sr_lh).unsqueeze(0),
            torch.from_numpy(sr_hl).unsqueeze(0), torch.from_numpy(sr_hh).unsqueeze(0)
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
        
        # IDWT (역 웨이블릿 변환) 2: 오직 LR의 저주파수(LL) + 예측 고주파수
        recon_coeffs_lr = (lr_ll, (pred_lh, pred_hl, pred_hh))
        recon_gray_lr = pywt.idwt2(recon_coeffs_lr, 'haar')
        recon_gray_lr = np.clip(recon_gray_lr, 0, 1)
        
        # 정량적 평가 (PSNR, SSIM 계산) 누적
        metrics['lr_psnr'] += psnr(hr_gray, lr_gray, data_range=1.0)
        metrics['lr_ssim'] += ssim(hr_gray, lr_gray, data_range=1.0)
        
        metrics['vdsr_psnr'] += psnr(hr_gray, sr_gray, data_range=1.0)
        metrics['vdsr_ssim'] += ssim(hr_gray, sr_gray, data_range=1.0)
        
        metrics['recon_lr_psnr'] += psnr(hr_gray, recon_gray_lr, data_range=1.0)
        metrics['recon_lr_ssim'] += ssim(hr_gray, recon_gray_lr, data_range=1.0)
        
        metrics['recon_sr_psnr'] += psnr(hr_gray, recon_gray_sr, data_range=1.0)
        metrics['recon_sr_ssim'] += ssim(hr_gray, recon_gray_sr, data_range=1.0)

    # 평균 계산
    num_images = len(image_paths)
    for key in metrics:
        metrics[key] /= num_images

    # 결과 출력
    print("\n" + "="*60)
    print(f"🏆 DIV2K Validation Dataset Evaluation (LR+SR 8-Ch Model) 🏆")
    print("="*60)
    print(f"[1] LR (Bicubic)      - PSNR: {metrics['lr_psnr']:.2f}dB, SSIM: {metrics['lr_ssim']:.4f}")
    print(f"[2] VDSR SR           - PSNR: {metrics['vdsr_psnr']:.2f}dB, SSIM: {metrics['vdsr_ssim']:.4f}")
    print(f"[3] Recon (LR_LL+Pred)- PSNR: {metrics['recon_lr_psnr']:.2f}dB, SSIM: {metrics['recon_lr_ssim']:.4f}")
    print(f"[4] Recon (SR_LL+Pred)- PSNR: {metrics['recon_sr_psnr']:.2f}dB, SSIM: {metrics['recon_sr_ssim']:.4f}")
    print("="*60)

if __name__ == "__main__":
    evaluate()
