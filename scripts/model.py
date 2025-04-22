# scripts/model.py
import torch
import torch.nn as nn

class UNet3D(nn.Module):
    """
    Minimal 3D UNet without spatial resolution changes:
    Input and output both at the same resolution.
    """
    def __init__(self, in_ch=1, base_ch=32):
        super().__init__()
        # Two conv layers preserving spatial dims
        self.conv1 = nn.Conv3d(in_ch, base_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(base_ch, base_ch, kernel_size=3, padding=1)
        # Final 1×1×1 conv back to in_ch channels
        self.out = nn.Conv3d(base_ch, in_ch, kernel_size=1)
        self.act = nn.ReLU()

    def forward(self, x, t=None):
        # No downsampling or upsampling; shape stays (B, C, D, H, W)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return self.out(x)

