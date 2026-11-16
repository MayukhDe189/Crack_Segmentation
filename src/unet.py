import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    Two consecutive convolutional layers, each followed by BatchNorm and ReLU.
    [Conv2d -> BatchNorm -> ReLU -> Conv2d -> BatchNorm -> ReLU]
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownBlock(nn.Module):
    """
    Downscaling block: MaxPool2d(2) followed by DoubleConv.
    """
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """
    Upscaling block: ConvTranspose2d (or bilinear upsample) followed by DoubleConv.
    Concatenates skip connection feature map from encoder.
    """
    def __init__(self, in_channels, out_channels, bilinear=False):
        super(UpBlock, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1: input from previous decoder stage
        # x2: skip connection from corresponding encoder stage
        x1 = self.up(x1)
        # Handle small dimension mismatches if any
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        if diff_y != 0 or diff_x != 0:
            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                            diff_y // 2, diff_y - diff_y // 2])
        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard U-Net Architecture for Binary Crack Segmentation.
    Encoder-Decoder with skip connections preserving fine crack boundaries.
    Input: (B, 3, 256, 256)
    Output: (B, 1, 256, 256) raw logits (pre-sigmoid).
    """
    def __init__(self, in_channels=3, out_channels=1, base_features=32, bilinear=False):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_features = base_features

        f = base_features
        # Encoder
        self.inc = DoubleConv(in_channels, f)        # 3 -> 32
        self.down1 = DownBlock(f, f * 2)             # 32 -> 64
        self.down2 = DownBlock(f * 2, f * 4)         # 64 -> 128
        self.down3 = DownBlock(f * 4, f * 8)         # 128 -> 256
        self.bottleneck = DownBlock(f * 8, f * 16)   # 256 -> 512

        # Decoder with skip connections
        self.up1 = UpBlock(f * 16, f * 8, bilinear)  # 512 + 256 -> 256
        self.up2 = UpBlock(f * 8, f * 4, bilinear)   # 256 + 128 -> 128
        self.up3 = UpBlock(f * 4, f * 2, bilinear)   # 128 + 64 -> 64
        self.up4 = UpBlock(f * 2, f, bilinear)       # 64 + 32 -> 32

        # 1x1 convolution mapping to 1 channel (raw logits)
        self.outc = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder passes
        x1 = self.inc(x)         # (B, 32, 256, 256)
        x2 = self.down1(x1)      # (B, 64, 128, 128)
        x3 = self.down2(x2)      # (B, 128, 64, 64)
        x4 = self.down3(x3)      # (B, 256, 32, 32)
        x5 = self.bottleneck(x4) # (B, 512, 16, 16)

        # Decoder passes with skip connections
        x = self.up1(x5, x4)     # (B, 256, 32, 32)
        x = self.up2(x, x3)      # (B, 128, 64, 64)
        x = self.up3(x, x2)      # (B, 64, 128, 128)
        x = self.up4(x, x1)      # (B, 32, 256, 256)
        logits = self.outc(x)    # (B, 1, 256, 256)
        return logits
