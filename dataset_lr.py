import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import pywt
import glob

def extract_wavelet_numpy(img_gray):
    # Haar Wavelet Level 1
    coeffs2 = pywt.dwt2(img_gray, 'haar')
    LL, (LH, HL, HH) = coeffs2
    
    return {
        'LL': torch.from_numpy(LL).unsqueeze(0), # [1, H/2, W/2]
        'LH': torch.from_numpy(LH).unsqueeze(0),
        'HL': torch.from_numpy(HL).unsqueeze(0),
        'HH': torch.from_numpy(HH).unsqueeze(0)
    }

class LRFreqDataset(Dataset):
    def __init__(self, data_dir):
        """
        data_dir: 'data_lr/train' 또는 'data_lr/valid'
        """
        super().__init__()
        self.lr_dir = os.path.join(data_dir, "LR")
        self.hr_dir = os.path.join(data_dir, "HR")
        self.image_files = sorted([os.path.basename(f) for f in glob.glob(os.path.join(self.lr_dir, "*.png"))])
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        file_name = self.image_files[idx]
        
        # Load as Grayscale for frequency analysis
        lr_img = Image.open(os.path.join(self.lr_dir, file_name)).convert("L")
        hr_img = Image.open(os.path.join(self.hr_dir, file_name)).convert("L")
        
        lr_gray = np.array(lr_img).astype(np.float32) / 255.0
        hr_gray = np.array(hr_img).astype(np.float32) / 255.0
        
        lr_wt = extract_wavelet_numpy(lr_gray)
        hr_wt = extract_wavelet_numpy(hr_gray)
        
        # Input: LR (LL, LH, HL, HH) - 4 Channels
        input_tensor = torch.cat([
            lr_wt['LL'], lr_wt['LH'], lr_wt['HL'], lr_wt['HH']
        ], dim=0)
        
        # Target: HR HF (LH, HL, HH) - 3 Channels
        target_tensor = torch.cat([
            hr_wt['LH'], 
            hr_wt['HL'], 
            hr_wt['HH']
        ], dim=0)
        
        return input_tensor, target_tensor

if __name__ == "__main__":
    ds = LRFreqDataset("./data_lr/train")
    if len(ds) > 0:
        inp, tar = ds[0]
        print(f"Input: {inp.shape}, Target: {tar.shape}")
