import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class ValueCNN(nn.Module):
    def __init__(self, in_channels=5, base_ch=4):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, base_ch)         # 5  -> 32
        self.pool1 = nn.MaxPool2d(2)                        # 64x64 -> 32x32

        self.enc2 = ConvBlock(base_ch, base_ch * 2)         # 32 -> 64
        self.pool2 = nn.MaxPool2d(2)                        # 32x32 -> 16x16

        self.bottleneck = ConvBlock(base_ch * 2, base_ch * 4)  # 64 -> 128

        # Decoder
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = ConvBlock(base_ch * 4 + base_ch * 2, base_ch * 2)  # 128+64 -> 64

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = ConvBlock(base_ch * 2 + base_ch, base_ch)          # 64+32 -> 32

        self.out_conv = nn.Conv2d(base_ch, 1, kernel_size=1)  # 32 -> 1

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)          # (B, 32, H, W)
        p1 = self.pool1(x1)        # (B, 32, H/2, W/2)

        x2 = self.enc2(p1)         # (B, 64, H/2, W/2)
        p2 = self.pool2(x2)        # (B, 64, H/4, W/4)

        x_bn = self.bottleneck(p2) # (B, 128, H/4, W/4)

        # Decoder
        u2 = self.up2(x_bn)        # (B, 128, H/2, W/2)
        # concat skip from enc2
        u2 = torch.cat([u2, x2], dim=1)  # (B, 128+64, H/2, W/2)
        d2 = self.dec2(u2)               # (B, 64, H/2, W/2)

        u1 = self.up1(d2)                # (B, 64, H, W)
        # concat skip from enc1
        u1 = torch.cat([u1, x1], dim=1)  # (B, 64+32, H, W)
        d1 = self.dec1(u1)               # (B, 32, H, W)

        out = self.out_conv(d1)          # (B, 1, H, W)
        return out

