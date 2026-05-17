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

# 1. Load FreqUNet
from model import FreqUNet

# 2. Load VDSR safely
spec = importlib.util.spec_from_file_location("vdsr_model", "../VDSR/model.py")
vdsr_module = importlib.util.module_from_spec(spec)
sys.modules["vdsr_model"] = vdsr_module
vdsr_module_spec = importlib.util.module_from_spec(spec)
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
    # Ensure this path exists or handle error
    try:
        vdsr_model.load_state_dict(torch.load("../VDSR/weights/vdsr_epoch_100.pth", map_location=device))
    except:
        print("Warning: VDSR weights not found. Using random weights for baseline (Metrics will be low).")
    vdsr_model.eval()

    # Residual Refinement 모델 로드
    freq_model = FreqUNet(in_channels=8, out_channels=3).to(device)
    # 최신 가중치 로드 (없을 경우를 대비해 에러 처리)
    weight_path = "./weights/refine_epoch_100.pth"
    if not os.path.exists(weight_path):
        # 최신 에포크 파일 찾기
        pth_files = glob.glob("./weights/refine_epoch_*.pth")
        if pth_files:
            weight_path = sorted(pth_files)[-1]
            print(f"Loading latest weights: {weight_path}")
        else:
            print("No refinement weights found. Please train the model first.")
            return

    freq_model.load_state_dict(torch.load(weight_path, map_location=device))
    freq_model.eval()

    # DIV2K Validation Dataset
    valid_dir = "./DIV2K/DIV2K_valid_HR/" # Updated path
    if not os.path.exists(valid_dir):
        # fallback to the path used in prepare_dataset if needed
        valid_dir = "../VDSR/DIV2K/DIV2K_valid_HR/"
        
    image_paths = sorted(glob.glob(os.path.join(valid_dir, "*.png")))
    
    if not image_paths:
        print(f"No validation images found in {valid_dir}. Please wait for download or check path.")
        return
        
    print(f"Evaluating {len(image_paths)} images with Residual Refinement logic...")
    
    metrics = {
        'vdsr_psnr': 0.0, 'vdsr_ssim': 0.0,
        'refine_psnr': 0.0, 'refine_ssim': 0.0
    }
    
    for img_path in tqdm(image_paths, desc="Evaluating"):
        hr_img = Image.open(img_path).convert("RGB")
        w, h = hr_img.size
        new_w, new_h = w - (w % 256), h - (h % 256)
        if new_w < 256 or new_h < 256: continue
        hr_img = hr_img.crop((0, 0, new_w, new_h))
        
        # LR creation
        lr_img_small = hr_img.resize((new_w // 4, new_h // 4), Image.BICUBIC)
        lr_img_up = lr_img_small.resize((new_w, new_h), Image.BICUBIC)
        
        # VDSR Baseline
        lr_tensor = transforms.ToTensor()(lr_img_up).unsqueeze(0).to(device)
        with torch.no_grad():
            _, sr_tensor = vdsr_model(lr_tensor)
        sr_img = transforms.ToPILImage()(sr_tensor.squeeze(0).cpu().clamp(0, 1))
        
        # Frequency Preparation
        lr_gray = np.array(lr_img_up.convert("L")).astype(np.float32) / 255.0
        sr_gray = np.array(sr_img.convert("L")).astype(np.float32) / 255.0
        hr_gray = np.array(hr_img.convert("L")).astype(np.float32) / 255.0
        
        lr_ll, lr_lh, lr_hl, lr_hh = extract_wavelet_numpy(lr_gray)
        sr_ll, sr_lh, sr_hl, sr_hh = extract_wavelet_numpy(sr_gray)
        
        # Input: [LR_WT, SR_WT]
        input_tensor = torch.cat([
            torch.from_numpy(lr_ll).unsqueeze(0), torch.from_numpy(lr_lh).unsqueeze(0),
            torch.from_numpy(lr_hl).unsqueeze(0), torch.from_numpy(lr_hh).unsqueeze(0),
            torch.from_numpy(sr_ll).unsqueeze(0), torch.from_numpy(sr_lh).unsqueeze(0),
            torch.from_numpy(sr_hl).unsqueeze(0), torch.from_numpy(sr_hh).unsqueeze(0)
        ], dim=0).unsqueeze(0).to(device)
        
        # Predict Residual
        with torch.no_grad():
            pred_res = freq_model(input_tensor).squeeze(0).cpu().numpy()
            
        # Refine: SR_HF + Pred_Residual
        refine_lh = sr_lh + pred_res[0]
        refine_hl = sr_hl + pred_res[1]
        refine_hh = sr_hh + pred_res[2]
        
        # Final Reconstruction (Using LR_LL for maximum fidelity)
        recon_coeffs = (lr_ll, (refine_lh, refine_hl, refine_hh))
        recon_gray = pywt.idwt2(recon_coeffs, 'haar')
        recon_gray = np.clip(recon_gray, 0, 1)
        
        # Accumulate Metrics
        metrics['vdsr_psnr'] += psnr(hr_gray, sr_gray, data_range=1.0)
        metrics['vdsr_ssim'] += ssim(hr_gray, sr_gray, data_range=1.0)
        metrics['refine_psnr'] += psnr(hr_gray, recon_gray, data_range=1.0)
        metrics['refine_ssim'] += ssim(hr_gray, recon_gray, data_range=1.0)

    num_images = len(image_paths)
    print("\n" + "="*50)
    print(f"✨ Residual Refinement Evaluation Result ✨")
    print("="*50)
    print(f"Baseline (VDSR) - PSNR: {metrics['vdsr_psnr']/num_images:.2f}dB, SSIM: {metrics['vdsr_ssim']/num_images:.4f}")
    print(f"Refined Result  - PSNR: {metrics['refine_psnr']/num_images:.2f}dB, SSIM: {metrics['refine_ssim']/num_images:.4f}")
    improvement = (metrics['refine_psnr'] - metrics['vdsr_psnr']) / num_images
    print(f"PSNR Improvement: {improvement:+.4f} dB")
    print("="*50)

if __name__ == "__main__":
    evaluate()
