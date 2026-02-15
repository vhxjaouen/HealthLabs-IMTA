import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Classification
# -----------------------------
class SimpleCNN(nn.Module):
    """
    CNN minimaliste pour classification.
    Entrée: (B,1,H,W)
    Sortie: logits (B,num_classes)
    """
    def __init__(self, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32*32*32, 128)  # suppose input=128x128
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))     # -> (B,16,H,W)
        x = self.pool(x)              # -> (B,16,H/2,W/2)
        x = F.relu(self.conv2(x))     # -> (B,32,H/2,W/2)
        x = self.pool(x)              # -> (B,32,H/4,W/4)
        x = x.view(x.size(0), -1)     # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# -----------------------------
# Segmentation
# -----------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class MiniUNet2D(nn.Module):
    """
    UNet minimaliste 2D pour segmentation binaire.
    """
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        # encoder
        self.enc1 = ConvBlock(in_channels, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        # bottleneck
        self.bottleneck = ConvBlock(32, 64)
        # decoder
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(32, 16)
        # output
        self.out_conv = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        out = self.out_conv(d1)
        return out

# -----------------------------
# Image-to-Image
# -----------------------------
class TinyAutoencoder(nn.Module):
    """
    Autoencodeur simple pour traduction de modalité (ex. T1ce -> T2).
    """
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        # encoder
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU()
        )
        # bottleneck
        self.bottleneck = nn.Conv2d(32, 64, 3, padding=1)
        # decoder
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, out_channels, 3, padding=1)
        )

    def forward(self, x):
        x = self.enc(x)
        x = self.bottleneck(x)
        x = self.dec(x)
        return x

# -----------------------------
# Factory
# -----------------------------
def build_simple_model(cfg):
    """
    Factory pour choisir un modèle jouet en fonction du YAML.
    """
    model_type = cfg["type"]
    if model_type == "SimpleCNN":
        return SimpleCNN(num_classes=cfg["out_channels"])
    elif model_type == "MiniUNet2D":
        return MiniUNet2D(in_channels=cfg["in_channels"], out_channels=cfg["out_channels"])
    elif model_type == "TinyAutoencoder":
        return TinyAutoencoder(in_channels=cfg["in_channels"], out_channels=cfg["out_channels"])
    else:
        raise ValueError(f"Unknown simple model type: {model_type}")
