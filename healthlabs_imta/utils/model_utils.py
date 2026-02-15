import torch.nn as nn
from monai.networks.nets import DenseNet121, resnet18, UNet, EfficientNetBN
from healthlabs_imta.models.simple_models import SimpleCNN, MiniUNet2D, TinyAutoencoder



def model_factory(cfg):
    """
    Factory unifiée qui construit un modèle selon cfg["type"].
    Gère à la fois classification (DenseNet, ResNet, SmallCNN),
    segmentation (UNet, MiniUNet2D), et I2I (TinyAutoencoder).
    """
    mtype = cfg["type"]

    # --- Classification (MONAI + custom) ---
    if mtype == "DenseNet121":
        return DenseNet121(
            spatial_dims=cfg["spatial_dims"],
            in_channels=cfg["in_channels"],
            out_channels=cfg["out_channels"],
            pretrained=cfg.get("pretrained", False)
        )

    elif mtype == "ResNet18":
        return resnet18(
            spatial_dims=cfg["spatial_dims"],
            n_input_channels=cfg["in_channels"],
            num_classes=cfg["out_channels"]
        )

    elif mtype == "SmallCNN":
        return SmallCNN(
            in_channels=cfg["in_channels"],
            num_classes=cfg["out_channels"],
            width=cfg.get("width", 32)
        )

    # --- Segmentation ---
    elif mtype == "UNet":
        return UNet(
            spatial_dims=cfg["spatial_dims"],
            in_channels=cfg["in_channels"],
            out_channels=cfg["out_channels"],
            channels=tuple(cfg["channels"]),
            strides=tuple(cfg["strides"]),
            num_res_units=cfg.get("num_res_units", 2),
        )

    elif mtype == "MiniUNet2D":
        return MiniUNet2D(
            in_channels=cfg["in_channels"],
            out_channels=cfg["out_channels"]
        )

    # --- I2I ---
    elif mtype == "TinyAutoencoder":
        return TinyAutoencoder(
            in_channels=cfg["in_channels"],
            out_channels=cfg["out_channels"]
        )

    # --- EfficientNet (optionnel) ---
    elif mtype.startswith("EfficientNet"):
        return EfficientNetBN(
            model_name=mtype.lower(),
            spatial_dims=cfg["spatial_dims"],
            in_channels=cfg["in_channels"],
            num_classes=cfg["out_channels"]
        )

    else:
        raise ValueError(f"Unknown model type: {mtype}")
