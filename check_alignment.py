import os
import glob
from PIL import Image
import numpy as np

def check_alignment():
    hr_dir = "./DIV2K/DIV2K_valid_HR/"
    lr_dir = "./DIV2K/DIV2K_valid_LR_bicubic/X4/"
    
    hr_paths = sorted(glob.glob(os.path.join(hr_dir, "*.png")))
    lr_paths = sorted(glob.glob(os.path.join(lr_dir, "*.png")))
    
    print(f"Total HR: {len(hr_paths)}, Total LR: {len(lr_paths)}")
    
    for i in range(5):
        hr_name = os.path.basename(hr_paths[i])
        lr_name = os.path.basename(lr_paths[i])
        print(f"Matching: {hr_name} <-> {lr_name}")
        
        hr_img = Image.open(hr_paths[i]).convert("L")
        lr_img = Image.open(lr_paths[i]).convert("L")
        
        # Resize LR to match HR size
        lr_up = lr_img.resize(hr_img.size, Image.BICUBIC)
        
        hr_arr = np.array(hr_img).astype(np.float32)
        lr_arr = np.array(lr_up).astype(np.float32)
        
        mse = np.mean((hr_arr - lr_arr) ** 2)
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        print(f"  Sample {i} Bicubic PSNR: {psnr:.2f} dB")

if __name__ == "__main__":
    check_alignment()
