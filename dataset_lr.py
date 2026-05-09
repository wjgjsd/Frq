import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import pywt
import glob

def extract_wavelet_numpy(img_gray):
    """
    PIL Image(Grayscale) 또는 Numpy 배열을 받아 1-level Haar Wavelet 변환을 수행하고,
    [LL, LH, HL, HH] 형태의 주파수 대역 리스트를 반환합니다.
    """
    if isinstance(img_gray, Image.Image):
        img_array = np.array(img_gray).astype(np.float32) / 255.0
    else:
        img_array = img_gray
        
    coeffs2 = pywt.dwt2(img_array, 'haar')
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
        data_dir: LR, SR, HR 이미지가 저장된 최상위 디렉토리 (예: './data/train')
        여기서는 LR 이미지 파일들만 추적하여 사용합니다. (이름 규칙은 일치한다고 가정)
        """
        super().__init__()
        self.data_dir = data_dir
        self.lr_dir = os.path.join(data_dir, "LR")
        self.hr_dir = os.path.join(data_dir, "HR")
        
        self.image_files = sorted([os.path.basename(f) for f in glob.glob(os.path.join(self.lr_dir, "*.png"))])
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        file_name = self.image_files[idx]
        
        # 이미지 로드 (Grayscale 변환 - 주파수는 밝기 성분이 핵심이므로)
        lr_img = Image.open(os.path.join(self.lr_dir, file_name)).convert("L")
        hr_img = Image.open(os.path.join(self.hr_dir, file_name)).convert("L")
        
        # On-the-fly 웨이블릿 변환 수행
        lr_wt = extract_wavelet_numpy(lr_img)
        hr_wt = extract_wavelet_numpy(hr_img)
        
        # 모델 입력 (Input): 오직 LR의 주파수 대역 (4채널)
        input_tensor = torch.cat([
            lr_wt['LL'], lr_wt['LH'], lr_wt['HL'], lr_wt['HH']
        ], dim=0)
        
        # 예측 타겟 (Target): HR 고주파수 직접 예측 (잔차 학습 미사용)
        target_tensor = torch.cat([
            hr_wt['LH'], 
            hr_wt['HL'], 
            hr_wt['HH']
        ], dim=0)
        
        return input_tensor, target_tensor
