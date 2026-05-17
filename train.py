import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset import FreqDataset
from model import FreqUNet

# Differentiable Haar IDWT implementation for PyTorch
class HaarIDWT(nn.Module):
    def __init__(self):
        super(HaarIDWT, self).__init__()
        # Synthesis kernels
        kernel = torch.tensor([
            [[[1, 1], [1, 1]]],
            [[[1, -1], [1, -1]]],
            [[[1, 1], [-1, -1]]],
            [[[1, -1], [-1, 1]]]
        ]).float() / 2.0
        self.register_buffer('kernel', kernel)

    def forward(self, LL, LH, HL, HH):
        # LL, LH, HL, HH: [B, 1, H/2, W/2]
        B, C, H, W = LL.shape
        # Concat to [B, 4, H/2, W/2]
        x = torch.cat([LL, LH, HL, HH], dim=1)
        # Using grouped convolution for IDWT
        # We process each channel (R, G, B) separately if C > 1, 
        # but here the dataset provides them as single channels or concatenated.
        # Given dataset.py structure, we handle 1 channel at a time or loop.
        
        # Helper for single channel IDWT
        def idwt_single(coeffs):
            return F.conv_transpose2d(coeffs, self.kernel, stride=2, groups=1)

        # Dataset provides components for Grayscale (1-ch) in this specific project
        out = idwt_single(x)
        return out

def compute_ssim(img1, img2, window_size=11):
    # Simplified SSIM implementation
    C1 = 0.01**2
    C2 = 0.03**2
    
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=window_size//2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size//2) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

class RefinementLoss(nn.Module):
    def __init__(self, w_freq=1.0, w_img=1.0, w_ssim=0.5):
        super().__init__()
        self.w_freq = w_freq
        self.w_img = w_img
        self.w_ssim = w_ssim
        self.l1 = nn.L1Loss()
        self.idwt = HaarIDWT()

    def forward(self, pred_res, target_res, sr_wt, hr_img, lr_ll):
        # 1. Frequency Domain Loss (Residual)
        loss_freq = self.l1(pred_res, target_res)
        
        # 2. Image Domain Reconstruction
        # Final HF = SR_HF + Pred_Residual
        final_lh = sr_wt[:, 1:2, ...] + pred_res[:, 0:1, ...]
        final_hl = sr_wt[:, 2:3, ...] + pred_res[:, 1:2, ...]
        final_hh = sr_wt[:, 3:4, ...] + pred_res[:, 2:3, ...]
        
        # Reconstruction using LR's LL (the truth at low scale)
        recon_img = self.idwt(lr_ll, final_lh, final_hl, final_hh)
        recon_img = torch.clamp(recon_img, 0, 1)
        
        loss_img = self.l1(recon_img, hr_img)
        
        # 3. SSIM Loss
        loss_ssim = 1 - compute_ssim(recon_img, hr_img)
        
        total_loss = self.w_freq * loss_freq + self.w_img * loss_img + self.w_ssim * loss_ssim
        return total_loss, loss_freq, loss_img, loss_ssim

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 32
    learning_rate = 2e-4 # Slightly higher for residual learning
    num_epochs = 100
    data_dir = "./data/train"
    weights_dir = "./weights"
    os.makedirs(weights_dir, exist_ok=True)

    print("Loading dataset...")
    dataset = FreqDataset(data_dir=data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    model = FreqUNet(in_channels=8, out_channels=3).to(device)
    criterion = RefinementLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # To calculate image domain loss, we need the original HR image.
    # Current FreqDataset returns (input_tensor, target_tensor).
    # We need to modify dataset.py to return original HR gray image or compute it via IDWT of target.
    # For now, let's assume we can reconstruct HR from its wavelet coefficients in the loss.
    
    print("Starting training (Residual Refinement)...")
    loss_history = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Extract components for the loss function
            # inputs: [LR_LL, LR_LH, LR_HL, LR_HH, SR_LL, SR_LH, SR_HL, SR_HH]
            lr_ll = inputs[:, 0:1, ...]
            sr_wt = inputs[:, 4:8, ...]
            
            optimizer.zero_grad()
            pred_res = model(inputs)
            
            # Reconstruct HR image from targets for the Image-domain loss
            # Since target_res = hr_wt_hf - sr_wt_hf, then hr_wt_hf = sr_wt_hf + target_res
            hr_lh = sr_wt[:, 1:2, ...] + targets[:, 0:1, ...]
            hr_hl = sr_wt[:, 2:3, ...] + targets[:, 1:2, ...]
            hr_hh = sr_wt[:, 3:4, ...] + targets[:, 2:3, ...]
            
            with torch.no_grad():
                # We use LR_LL as the target LL because that's what we use for reconstruction
                hr_img = criterion.idwt(lr_ll, hr_lh, hr_hl, hr_hh).clamp(0, 1)
            
            loss, l_f, l_i, l_s = criterion(pred_res, targets, sr_wt, hr_img, lr_ll)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx+1}/{len(dataloader)}] "
                      f"Loss: {loss.item():.4f} (F:{l_f.item():.4f}, I:{l_i.item():.4f}, S:{l_s.item():.4f})")
                
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"===> Epoch {epoch+1} Complete. Average Loss: {avg_loss:.6f}")
        
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(weights_dir, f"refine_epoch_{epoch+1}.pth"))

    # Save results
    history_path = os.path.join(weights_dir, "refine_loss_history.json")
    with open(history_path, 'w') as f:
        json.dump(loss_history, f)
        
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), loss_history)
    plt.title('Refinement Training Loss')
    plt.savefig(os.path.join(weights_dir, "refine_loss_curve.png"))
    print("Training finished.")

if __name__ == "__main__":
    train()
