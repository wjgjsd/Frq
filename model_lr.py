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

class DoubleConv(nn.Module):
    """(Conv => BatchNorm => ReLU) * 2 + CBAM"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.cbam = CBAM(out_channels)

    def forward(self, x):
        x = self.double_conv(x)
        x = self.cbam(x)
        return x

class LRFreqUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=3):
        super(LRFreqUNet, self).__init__()
        
        # 깊이(Depth) 4 + CBAM Attention
        
        # 인코더 (Downsampling)
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))
        
        # 디코더 (Upsampling)
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024, 512) # skip connection 512 + 512
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512, 256)  # skip connection 256 + 256
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)  # skip connection 128 + 128
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)   # skip connection 64 + 64
        
        # 마지막 출력층
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # 인코딩
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # 디코딩
        x = self.up1(x5)
        x = torch.cat([x4, x], dim=1) # Skip connection
        x = self.conv1(x)
        
        x = self.up2(x)
        x = torch.cat([x3, x], dim=1) # Skip connection
        x = self.conv2(x)
        
        x = self.up3(x)
        x = torch.cat([x2, x], dim=1) # Skip connection
        x = self.conv3(x)
        
        x = self.up4(x)
        x = torch.cat([x1, x], dim=1) # Skip connection
        x = self.conv4(x)
        
        output = self.outc(x)
        return output

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.cbam = CBAM(channels)

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        res = self.cbam(res)
        return x + res

class FreqResNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, num_blocks=16, base_channels=64):
        super(FreqResNet, self).__init__()
        
        # Initial Feature Extraction
        self.head = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # Residual Blocks
        self.body = nn.Sequential(*[ResBlock(base_channels) for _ in range(num_blocks)])
        
        # Final Convolution
        self.tail = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # x is [B, 4, H/2, W/2] : (LL, LH, HL, HH)
        # We want to predict HR HF (LH, HL, HH)
        
        # Extract features
        feat = self.head(x)
        
        # Deep mapping
        res = self.body(feat)
        res = res + feat # Global residual connection for features
        
        # Final output
        out = self.tail(res)
        
        # Global Skip Connection (Add input HF to predicted HF)
        # x[:, 1:4] corresponds to LR's LH, HL, HH
        return out + x[:, 1:4, :, :]

if __name__ == "__main__":
    # 구조 테스트
    model = LRFreqUNet(in_channels=4, out_channels=3)
    dummy_input = torch.randn(2, 4, 128, 128)
    output = model(dummy_input)
    print(f"LRFreqUNet Output shape: {output.shape}")
    
    model2 = FreqResNet()
    output2 = model2(dummy_input)
    print(f"FreqResNet Output shape: {output2.shape}")
    print("Models 구조 정상!")
