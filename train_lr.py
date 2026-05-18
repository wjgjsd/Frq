import os
import json
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset_lr import LRFreqDataset
from model_lr import LRFreqUNet, FreqResNet

# Differentiable Haar IDWT
class HaarIDWT(nn.Module):
    def __init__(self):
        super(HaarIDWT, self).__init__()
        kernel = torch.tensor([
            [[[1, 1], [1, 1]]],
            [[[1, -1], [1, -1]]],
            [[[1, 1], [-1, -1]]],
            [[[1, -1], [-1, 1]]]
        ]).float() / 2.0
        self.register_buffer('kernel', kernel)

    def forward(self, LL, LH, HL, HH):
        x = torch.cat([LL, LH, HL, HH], dim=1)
        return F.conv_transpose2d(x, self.kernel, stride=2, groups=1)

def compute_ssim(img1, img2, window_size=11):
    C1, C2 = 0.01**2, 0.03**2
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=window_size//2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size//2) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

class Scenario1Loss(nn.Module):
    def __init__(self, w_freq=1.0, w_img=1.0, w_ssim=0.5):
        super().__init__()
        self.w_freq, self.w_img, self.w_ssim = w_freq, w_img, w_ssim
        self.l1 = nn.L1Loss()
        self.idwt = HaarIDWT()

    def forward(self, pred_hf, target_hf, lr_ll):
        loss_freq = self.l1(pred_hf, target_hf)
        recon_img = self.idwt(lr_ll, pred_hf[:, 0:1, ...], pred_hf[:, 1:2, ...], pred_hf[:, 2:3, ...])
        recon_img = torch.clamp(recon_img, 0, 1)
        target_img = self.idwt(lr_ll, target_hf[:, 0:1, ...], target_hf[:, 1:2, ...], target_hf[:, 2:3, ...])
        target_img = torch.clamp(target_img, 0, 1)
        loss_img = self.l1(recon_img, target_img)
        loss_ssim = 1 - compute_ssim(recon_img, target_img)
        return self.w_freq * loss_freq + self.w_img * loss_img + self.w_ssim * loss_ssim, loss_freq, loss_img, loss_ssim

def train_lr():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters for FreqResNet
    batch_size = 32
    learning_rate = 1e-4
    num_epochs = 100
    data_dir = "./data_lr/train"
    weights_dir = "./weights_lr_resnet"
    os.makedirs(weights_dir, exist_ok=True)

    print("Loading Scenario 1 dataset...")
    dataset = LRFreqDataset(data_dir=data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    model = FreqResNet(in_channels=4, out_channels=3).to(device)
    criterion = Scenario1Loss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # Resume Logic
    start_epoch = 0
    checkpoint_list = glob.glob(os.path.join(weights_dir, "lr_only_epoch_*.pth"))
    if checkpoint_list:
        latest_checkpoint = max(checkpoint_list, key=os.path.getctime)
        print(f"Resuming from checkpoint: {latest_checkpoint}")
        model.load_state_dict(torch.load(latest_checkpoint))
        start_epoch = int(os.path.basename(latest_checkpoint).split('_')[-1].split('.')[0])
        # Fast-forward scheduler
        for _ in range(start_epoch):
            scheduler.step()
        print(f"Starting from epoch {start_epoch + 1}")

    # Load previous history if exists
    history_path = os.path.join(weights_dir, "loss_history.json")
    loss_history = []
    if os.path.exists(history_path) and start_epoch > 0:
        with open(history_path, 'r') as f:
            loss_history = json.load(f)
            
    best_loss = min(loss_history) if loss_history else float('inf')

    print(f"Starting training (FreqResNet, Batch Size: {batch_size})...")
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            lr_ll = inputs[:, 0:1, ...]
            
            optimizer.zero_grad()
            pred_hf = model(inputs)
            
            loss, l_f, l_i, l_s = criterion(pred_hf, targets, lr_ll)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if (batch_idx + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx+1}/{len(dataloader)}] "
                      f"LR: {scheduler.get_last_lr()[0]:.2e} Loss: {loss.item():.4f} (F:{l_f.item():.4f}, I:{l_i.item():.4f}, S:{l_s.item():.4f})")
                
        scheduler.step()
        
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"===> Epoch {epoch+1} Complete. Average Loss: {avg_loss:.6f}")
        
        # Save Best Model
        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"🌟 New Best Loss: {best_loss:.6f}! Saving model...")
            torch.save(model.state_dict(), os.path.join(weights_dir, "lr_only_best.pth"))
        
        # Save Checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(weights_dir, f"lr_only_epoch_{epoch+1}.pth"))

        # Save history every epoch
        with open(history_path, 'w') as f:
            json.dump(loss_history, f)
            
    print("Scenario 1 training finished.")

if __name__ == "__main__":
    train_lr()
