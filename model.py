import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv(x))
        return torch.cat([x, out], dim=1)

class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate=32, num_layers=4):
        super(ResidualDenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        current_channels = in_channels
        for _ in range(num_layers):
            self.layers.append(DenseLayer(current_channels, growth_rate))
            current_channels += growth_rate
            
        # Local Feature Fusion
        self.lff = nn.Conv2d(current_channels, in_channels, kernel_size=1)
        self.cbam = CBAM(in_channels)

    def forward(self, x):
        res = x
        for layer in self.layers:
            res = layer(res)
        res = self.lff(res)
        res = self.cbam(res)
        return x + res

class FreqRDN_Refine(nn.Module):
    """
    Scenario 2 Model: Residual Refinement
    Input: 8 Channels (LR_DWT + SR_Baseline_DWT)
    Output: 3 Channels (Pred_Residual for LH, HL, HH)
    """
    def __init__(self, in_channels=8, out_channels=3, num_blocks=8, base_channels=64, growth_rate=32):
        super(FreqRDN_Refine, self).__init__()
        
        # Initial Feature Extraction
        self.head = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # Residual Dense Blocks (RDB)
        self.body = nn.Sequential(*[ResidualDenseBlock(base_channels, growth_rate) for _ in range(num_blocks)])
        
        # Global Feature Fusion (GFF)
        self.gff = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Frequency-Aware Branches (Tail)
        self.tail_LH = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 2, 1, kernel_size=3, padding=1)
        )
        self.tail_HL = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 2, 1, kernel_size=3, padding=1)
        )
        self.tail_HH = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 2, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # Extract features
        feat = self.head(x)
        
        # Deep mapping with dense connections
        res = self.body(feat)
        res = self.gff(res)
        res = res + feat # Global feature skip connection
        
        # Frequency-aware reconstruction
        out_LH = self.tail_LH(res)
        out_HL = self.tail_HL(res)
        out_HH = self.tail_HH(res)
        
        # For Scenario 2, the target itself is the residual difference (HR_HF - SR_HF).
        # Therefore, the model's direct output is the predicted residual.
        # We do not add the input SR_HF here; that addition is handled during inference/reconstruction.
        out = torch.cat([out_LH, out_HL, out_HH], dim=1)
        
        return out

if __name__ == "__main__":
    # 구조 테스트
    model = FreqRDN_Refine(in_channels=8, out_channels=3)
    dummy_input = torch.randn(2, 8, 128, 128)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("FreqRDN_Refine 구조 정상!")
