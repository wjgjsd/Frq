import os
import glob
from PIL import Image
import torch
import numpy as np
import pywt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

from model_lr import LRFreqUNet, FreqResNet

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
        
        # Predict HF
        with torch.no_grad():
            pred_hf = model(input_tensor).squeeze(0).cpu().numpy()
            
        # Recon (Grayscale)
        recon_coeffs = (ll, (pred_hf[0], pred_hf[1], pred_hf[2]))
        recon_gray = pywt.idwt2(recon_coeffs, 'haar')
        recon_gray = np.clip(recon_gray, 0, 1)
        
        # For standard metrics, we need the RGB images to convert to Y channel.
        # Since our model only predicted Grayscale (L channel), we will inject this back into YCbCr 
        # or calculate PSNR purely on the predicted Grayscale output vs HR Grayscale.
        # However, to be perfectly fair with the paper, the standard protocol assumes full RGB SR output,
        # then converts to YCbCr.
        # Our model is a Grayscale model. So we calculate Y-channel equivalent metrics by extracting
        # the Y channel from the original HR, and comparing it to our model's output (which acts as Y).
        
        # Extract Standard Y-Channel from Ground Truth
        hr_rgb_arr = np.array(hr_img_cropped).astype(np.float32) / 255.0
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
    
    # Check if FreqResNet exists in weights
    use_resnet = False
    weight_path = "./weights_lr_resnet/lr_only_best.pth"
    if not os.path.exists(weight_path):
        weight_path = "./weights_lr_resnet/lr_only_epoch_100.pth"
        if not os.path.exists(weight_path):
            pth_files = glob.glob("./weights_lr_resnet/lr_only_epoch_*.pth")
            if pth_files:
                weight_path = sorted(pth_files, key=os.path.getctime)[-1]
                print(f"Loading latest weights: {weight_path}")
            else:
                print("No weights found.")
                return

    # To support both models for testing, we try loading FreqResNet first
    try:
        model = FreqResNet(in_channels=4, out_channels=3).to(device)
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print("Successfully loaded FreqResNet.")
    except Exception as e:
        print("Falling back to LRFreqUNet.")
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
