import torch
import torch.nn as nn


class MiniUNet(nn.Module):
    """
    간소화된 U-Net (채널 수를 줄일 경우 CPU 학습 가능)
    9개 클래스(건물,주차장,도로,가로수,논,밭,산림,나지,비대상지)
    """

    def __init__(self, num_classes=9):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(2)

        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(48, 16, 3, padding=1), nn.ReLU())

        self.up2 = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, num_classes, 1)
        )

    def forward(self, x):
        x1 = self.enc1(x)  # (16,H,W)
        p1 = self.pool1(x1)  # (16,H/2,W/2)

        x2 = self.enc2(p1)  # (32,H/2,W/2)
        p2 = self.pool2(x2)  # (32,H/4,W/4)

        up1 = self.up1(p2)  # (16,H/2,W/2)
        cat1 = torch.cat([up1, x2], dim=1)  # skip connection (16+32=48,H/2,W/2)
        d1 = self.dec1(cat1)  # (16,H/2,W/2)

        up2 = self.up2(d1)  # (16,H,W)
        cat2 = torch.cat([up2, x1], dim=1)
        out = self.dec2(cat2)  # (num_classes,H,W)
        return out
