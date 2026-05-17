import os
import glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pywt
from torchvision import transforms

class FreqDataset(Dataset):
    def __init__(self, data_dir):
        # data_dir는 'data/train' 또는 'data/valid' 경로입니다.
        self.lr_dir = os.path.join(data_dir, "LR")
        self.sr_dir = os.path.join(data_dir, "SR")
        self.hr_dir = os.path.join(data_dir, "HR")
        
        # 파일 목록 가져오기 (정렬하여 짝맞춤)
        self.filenames = sorted([os.path.basename(p) for p in glob.glob(os.path.join(self.lr_dir, "*.png"))])

    def __len__(self):
        return len(self.filenames)

    def extract_wavelet(self, image):
        # 이미지를 그레이스케일로 변환 후 Numpy 배열로 변환 (0.0 ~ 1.0)
        img_gray = np.array(image.convert("L"), dtype=np.float32) / 255.0
        
        # Haar 웨이블릿 변환 적용
        coeffs2 = pywt.dwt2(img_gray, 'haar')
        LL, (LH, HL, HH) = coeffs2
        
        # 텐서로 변환 (채널 차원 추가)
        return {
            'LL': torch.from_numpy(LL).unsqueeze(0),
            'LH': torch.from_numpy(LH).unsqueeze(0),
            'HL': torch.from_numpy(HL).unsqueeze(0),
            'HH': torch.from_numpy(HH).unsqueeze(0),
        }

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        
        # 이미지 불러오기
        lr_img = Image.open(os.path.join(self.lr_dir, filename))
        sr_img = Image.open(os.path.join(self.sr_dir, filename))
        hr_img = Image.open(os.path.join(self.hr_dir, filename))
        
        # 웨이블릿 추출
        lr_wt = self.extract_wavelet(lr_img)
        sr_wt = self.extract_wavelet(sr_img)
        hr_wt = self.extract_wavelet(hr_img)
        
        # 모델 입력 (Input): LR 웨이블릿(4채널) + SR 웨이블릿(4채널) = 8채널
        input_tensor = torch.cat([
            lr_wt['LL'], lr_wt['LH'], lr_wt['HL'], lr_wt['HH'],
            sr_wt['LL'], sr_wt['LH'], sr_wt['HL'], sr_wt['HH']
        ], dim=0)
        
        # 예측 타겟 (Target): HR과 SR 고주파수의 차이 (Residual Learning)
        # SR이 놓친 부분을 학습하도록 설정
        target_tensor = torch.cat([
            hr_wt['LH'] - sr_wt['LH'], 
            hr_wt['HL'] - sr_wt['HL'], 
            hr_wt['HH'] - sr_wt['HH']
        ], dim=0)
        
        return input_tensor, target_tensor

if __name__ == "__main__":
    # 데이터로더 테스트
    dataset = FreqDataset(data_dir="./data/train")
    if len(dataset) > 0:
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        inputs, targets = next(iter(loader))
        print(f"Dataset Size: {len(dataset)}")
        print(f"Input Shape (Batch, Channels, H, W): {inputs.shape}")   # (4, 8, 128, 128) - 256의 절반
        print(f"Target Shape (Batch, Channels, H, W): {targets.shape}") # (4, 3, 128, 128)
    else:
        print("No data found! Please run prepare_dataset.py first.")
