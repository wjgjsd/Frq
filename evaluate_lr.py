import os
import glob
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import pywt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

from model_lr import LRFreqUNet

def extract_wavelet_numpy(img_gray):
    coeffs2 = pywt.dwt2(img_gray, 'haar')
    LL, (LH, HL, HH) = coeffs2
    return LL, LH, HL, HH

def evaluate_lr():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Model
    model = LRFreqUNet(in_channels=4, out_channels=3).to(device)
    weight_path = "./weights_lr_only/lr_only_epoch_100.pth"
    if not os.path.exists(weight_path):
        pth_files = glob.glob("./weights_lr_only/lr_only_epoch_*.pth")
        if pth_files:
            weight_path = sorted(pth_files)[-1]
            print(f"Loading latest weights: {weight_path}")
        else:
            print("No weights found for Scenario 1.")
            return
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    # Valid images
    valid_hr_dir = "./DIV2K/DIV2K_valid_HR/"
    valid_lr_dir = "./DIV2K/DIV2K_valid_LR_bicubic/X4/"
    
    hr_paths = sorted(glob.glob(os.path.join(valid_hr_dir, "*.png")))
    lr_paths = sorted(glob.glob(os.path.join(valid_lr_dir, "*.png")))
    
    print(f"Evaluating Scenario 1 with {len(hr_paths)} images...")
    
    metrics = {
        'bicubic_psnr': 0.0, 'bicubic_ssim': 0.0,
        'our_psnr': 0.0, 'our_ssim': 0.0
    }
    
    for hr_path, lr_path in tqdm(zip(hr_paths, lr_paths), total=len(hr_paths)):
        hr_img = Image.open(hr_path).convert("L")
        lr_img = Image.open(lr_path).convert("L")
        
        w, h = hr_img.size
        # First, upscale LR to full HR size to maintain alignment
        lr_img_up_full = lr_img.resize((w, h), Image.BICUBIC)
        
        # Then, crop both using same coordinates
        new_w, new_h = w - (w % 256), h - (h % 256)
        if new_w < 256 or new_h < 256: continue
        
        hr_img_cropped = hr_img.crop((0, 0, new_w, new_h))
        lr_img_cropped = lr_img_up_full.crop((0, 0, new_w, new_h))
        
        hr_gray = np.array(hr_img_cropped).astype(np.float32) / 255.0
        lr_gray = np.array(lr_img_cropped).astype(np.float32) / 255.0
        
        # Wavelet for Input
        ll, lh, hl, hh = extract_wavelet_numpy(lr_gray)
        input_tensor = torch.cat([
            torch.from_numpy(ll).unsqueeze(0), torch.from_numpy(lh).unsqueeze(0),
            torch.from_numpy(hl).unsqueeze(0), torch.from_numpy(hh).unsqueeze(0)
        ], dim=0).unsqueeze(0).to(device)
        
        # Predict HF
        with torch.no_grad():
            pred_hf = model(input_tensor).squeeze(0).cpu().numpy()
            
        # Recon
        recon_coeffs = (ll, (pred_hf[0], pred_hf[1], pred_hf[2]))
        recon_gray = pywt.idwt2(recon_coeffs, 'haar')
        recon_gray = np.clip(recon_gray, 0, 1)
        
        # Metrics
        metrics['bicubic_psnr'] += psnr(hr_gray, lr_gray, data_range=1.0)
        metrics['bicubic_ssim'] += ssim(hr_gray, lr_gray, data_range=1.0)
        metrics['our_psnr'] += psnr(hr_gray, recon_gray, data_range=1.0)
        metrics['our_ssim'] += ssim(hr_gray, recon_gray, data_range=1.0)
        
    num = len(hr_paths)
    print("\n" + "="*50)
    print(f"📊 Scenario 1 (LR-Only) Evaluation Result 📊")
    print("="*50)
    print(f"Bicubic Baseline - PSNR: {metrics['bicubic_psnr']/num:.2f}dB, SSIM: {metrics['bicubic_ssim']/num:.4f}")
    print(f"Our Model (Freq) - PSNR: {metrics['our_psnr']/num:.2f}dB, SSIM: {metrics['our_ssim']/num:.4f}")
    print(f"Improvement over Bicubic: { (metrics['our_psnr'] - metrics['bicubic_psnr'])/num:+.4f} dB")
    print("="*50)

if __name__ == "__main__":
    evaluate_lr()
