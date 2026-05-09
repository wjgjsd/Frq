import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import FreqDataset
from model import FreqUNet

def train():
    # 1. 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. 하이퍼파라미터 설정
    batch_size = 32
    learning_rate = 1e-4
    num_epochs = 100
    data_dir = "./data/train"
    weights_dir = "./weights"
    
    os.makedirs(weights_dir, exist_ok=True)

    # 3. 데이터셋 로드
    print("Loading dataset...")
    try:
        dataset = FreqDataset(data_dir=data_dir)
        print(f"Total training patches: {len(dataset)}")
        if len(dataset) == 0:
            raise ValueError("Dataset is empty.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please run 'python prepare_dataset.py' first to generate the data.")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    # 4. 모델 및 Loss 초기화
    model = FreqUNet(in_channels=8, out_channels=3).to(device)
    
    # 가중치 주파수 손실 (Weighted Frequency Loss)
    # HH 밴드(대각선 초고주파수)가 가장 맞추기 어려우므로 틀렸을 때 벌점을 2배로 부여합니다.
    class WeightedWaveletLoss(nn.Module):
        def __init__(self, w_lh=1.0, w_hl=1.0, w_hh=2.0):
            super().__init__()
            self.w_lh = w_lh
            self.w_hl = w_hl
            self.w_hh = w_hh
            self.l1 = nn.L1Loss()
            
        def forward(self, pred, target):
            # Channels: 0=LH, 1=HL, 2=HH
            loss_lh = self.l1(pred[:, 0, ...], target[:, 0, ...])
            loss_hl = self.l1(pred[:, 1, ...], target[:, 1, ...])
            loss_hh = self.l1(pred[:, 2, ...], target[:, 2, ...])
            return self.w_lh * loss_lh + self.w_hl * loss_hl + self.w_hh * loss_hh

    criterion = WeightedWaveletLoss(w_lh=1.0, w_hl=1.0, w_hh=2.0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 5. 학습 루프
    print("Starting training...")
    loss_history = []  # 에포크별 평균 Loss를 저장할 리스트
    lr_reduced = False # 학습률이 절반으로 줄었는지 확인하는 플래그
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Loss 계산
            loss = criterion(outputs, targets)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx+1}/{len(dataloader)}] Loss: {loss.item():.6f}")
                
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"===> Epoch {epoch+1} Complete. Average Loss: {avg_loss:.6f} | Current LR: {current_lr}")
        
        # 사용자가 제안한 맞춤형 학습률 스케줄링 (Average Loss가 0.075 이하로 떨어지면 LR 절반 감소)
        if not lr_reduced and avg_loss <= 0.075:
            print("\n[Scheduler] Average loss reached 0.075! Reducing learning rate by half to fine-tune the model.\n")
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
            lr_reduced = True
        
        # 주기적으로 모델 가중치 저장
        if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs:
            save_path = os.path.join(weights_dir, f"frequnet_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved model to {save_path}\n")

    # 6. 학습 완료 후 Loss 그래프 저장
    print("Training finished. Saving loss history and plot...")
    import json
    import matplotlib.pyplot as plt
    
    # Loss 기록 텍스트 파일로 저장
    history_path = os.path.join(weights_dir, "loss_history.json")
    with open(history_path, 'w') as f:
        json.dump(loss_history, f)
        
    # Loss 그래프 이미지로 저장
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), loss_history, marker='o', linestyle='-', color='b')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(weights_dir, "loss_curve.png")
    plt.savefig(plot_path)
    print(f"Loss history saved to {history_path}")
    print(f"Loss curve plot saved to {plot_path}")

if __name__ == "__main__":
    train()
