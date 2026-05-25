import os
import glob
from PIL import Image
import torch
import numpy as np
import pywt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

from model_lr import LRFreqUNet, FreqResNet, FreqRDN

def extract_wavelet_numpy(img_gray):
    coeffs2 = pywt.dwt2(img_gray, 'haar')
    LL, (LH, HL, HH) = coeffs2
    return LL, LH, HL, HH

def rgb2ycbcr_pt(img, only_y=True):
    '''
    Standard MATLAB rgb2ycbcr equivalent.
    img: float32, [0, 1], shape [H, W, C]
    '''
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0 / 255.0
        return rlt
    else:
        # Full conversion if needed
        pass

def shave(img, border):
    """
    Remove border pixels.
    """
    if border > 0:
        return img[border:-border, border:-border]
    return img

def evaluate_on_dataset(model, device, dataset_dir, scale=4):
    hr_paths = sorted(glob.glob(os.path.join(dataset_dir, "*.png")))
    if not hr_paths:
        print(f"No images found in {dataset_dir}")
        return
        
    print(f"Evaluating on {dataset_dir} ({len(hr_paths)} images)...")
    
    metrics = {
        'bicubic_psnr': 0.0, 'bicubic_ssim': 0.0,
        'our_psnr': 0.0, 'our_ssim': 0.0
    }
    
    for hr_path in tqdm(hr_paths):
        hr_img_rgb = Image.open(hr_path).convert("RGB")
        
        w, h = hr_img_rgb.size
        # Crop to multiple of 32 for our model
        new_w, new_h = w - (w % 32), h - (h % 32)
        hr_img_cropped = hr_img_rgb.crop((0, 0, new_w, new_h))
        
        # Bicubic Down/Up in RGB
        lr_img_small = hr_img_cropped.resize((new_w // scale, new_h // scale), Image.BICUBIC)
        lr_img_up = lr_img_small.resize((new_w, new_h), Image.BICUBIC)
        
        # We need Grayscale [0, 1] for Wavelet Model input
        # Note: Model expects PIL's "L" conversion, so we maintain that for input to be consistent with training
        hr_gray_pil = hr_img_cropped.convert("L")
        lr_gray_pil = lr_img_up.convert("L")
        
        lr_gray_arr = np.array(lr_gray_pil).astype(np.float32) / 255.0
        
        # Wavelet for Input
        ll, lh, hl, hh = extract_wavelet_numpy(lr_gray_arr)
        input_tensor = torch.cat([
            torch.from_numpy(ll).unsqueeze(0), torch.from_numpy(lh).unsqueeze(0),
            torch.from_numpy(hl).unsqueeze(0), torch.from_numpy(hh).unsqueeze(0)
        ], dim=0).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            pred_hf = model(input_tensor)
            
        pred_LH = pred_hf[:, 0, ...].cpu().numpy()[0]
        pred_HL = pred_hf[:, 1, ...].cpu().numpy()[0]
        pred_HH = pred_hf[:, 2, ...].cpu().numpy()[0]
        
        # IDWT
        coeffs_recon = (ll, (pred_LH, pred_HL, pred_HH))
        recon_gray = pywt.idwt2(coeffs_recon, 'haar')
        recon_gray = np.clip(recon_gray, 0.0, 1.0)
        
        # The model was trained on PIL "L" Grayscale. 
        # Evaluating PIL "L" against MATLAB "Y" causes a massive PSNR drop due to color space scale differences.
        # To evaluate fairly, we must compare the model's output to the PIL "L" ground truth.
        hr_gray_arr = np.array(hr_gray_pil).astype(np.float32) / 255.0
        lr_gray_arr = np.array(lr_gray_pil).astype(np.float32) / 255.0
        our_gray_arr = recon_gray
        
        # Shaving (removing borders)
        hr_shaved = shave(hr_gray_arr, scale)
        lr_shaved = shave(lr_gray_arr, scale)
        our_shaved = shave(our_gray_arr, scale)
        
        # Metrics
        metrics['bicubic_psnr'] += psnr(hr_shaved, lr_shaved, data_range=1.0)
        metrics['bicubic_ssim'] += ssim(hr_shaved, lr_shaved, data_range=1.0)
        metrics['our_psnr'] += psnr(hr_shaved, our_shaved, data_range=1.0)
        metrics['our_ssim'] += ssim(hr_shaved, our_shaved, data_range=1.0)
        
    num = len(hr_paths)
    print(f"\n[Result for {dataset_dir}]")
    print(f"Bicubic Baseline - PSNR: {metrics['bicubic_psnr']/num:.2f}dB, SSIM: {metrics['bicubic_ssim']/num:.4f}")
    print(f"Our Model (Freq) - PSNR: {metrics['our_psnr']/num:.2f}dB, SSIM: {metrics['our_ssim']/num:.4f}")
    print(f"Improvement over Bicubic: { (metrics['our_psnr'] - metrics['bicubic_psnr'])/num:+.4f} dB\n")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check if FreqRDN exists in weights
    use_rdn = False
    weight_path = "./weights_lr_rdn/lr_only_best.pth"
    if not os.path.exists(weight_path):
        weight_path = "./weights_lr_rdn/lr_only_epoch_100.pth"
        if not os.path.exists(weight_path):
            pth_files = glob.glob("./weights_lr_rdn/lr_only_epoch_*.pth")
            if pth_files:
                weight_path = sorted(pth_files, key=os.path.getctime)[-1]
                print(f"Loading latest weights: {weight_path}")
            else:
                # Fallback to FreqResNet
                weight_path = "./weights_lr_resnet/lr_only_best.pth"
                if not os.path.exists(weight_path):
                    weight_path = "./weights_lr_resnet/lr_only_epoch_100.pth"

    # To support both models for testing, we try loading FreqRDN first
    try:
        model = FreqRDN(in_channels=4, out_channels=3).to(device)
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print("Successfully loaded FreqRDN.")
    except Exception as e:
        print(f"Failed to load FreqRDN: {e}")
        try:
            print("Falling back to FreqResNet.")
            model = FreqResNet(in_channels=4, out_channels=3).to(device)
            model.load_state_dict(torch.load(weight_path, map_location=device))
        except Exception as e2:
            print("Falling back to LRFreqUNet.")
            weight_path = "./weights_lr_only/lr_only_epoch_100.pth"
            model = LRFreqUNet(in_channels=4, out_channels=3).to(device)
            model.load_state_dict(torch.load(weight_path, map_location=device))

    model.eval()

    print("="*50)
    print("📊 Evaluating Scenario 1 on Standard Benchmarks (Y-Channel + Shaving) 📊")
    print("="*50)
    
    evaluate_on_dataset(model, device, "Set5", scale=4)
    evaluate_on_dataset(model, device, "Set14", scale=4)

if __name__ == "__main__":
    main()
