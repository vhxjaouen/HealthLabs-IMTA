import os
import sys
import torch
import nibabel as nib
import numpy as np

sys.path.append(os.path.abspath(".."))

from monai.transforms import (
    LoadImage,
    EnsureChannelFirstd,
    SpatialPadd,
    CenterSpatialCropd,
    ScaleIntensityRangePercentilesd,
    EnsureTyped,
    Compose,
)
from utils.model_utils import model_factory


# --------------------
# Prétraitement
# --------------------
def load_and_preprocess(image_path, target_size=(240,240)):
    loader = LoadImage(image_only=True)
    vol = loader(image_path)  # (H,W,D)
    slices = []
    for d in range(vol.shape[-1]):
        slice_2d = vol[..., d]
        data = {"image": slice_2d}
        transforms = Compose([
            EnsureChannelFirstd(keys="image"),
            SpatialPadd(keys="image", spatial_size=target_size),
            CenterSpatialCropd(keys="image", roi_size=target_size),
            ScaleIntensityRangePercentilesd(
                keys="image", lower=1, upper=99, b_min=-1.0, b_max=1.0, clip=True
            ),
            EnsureTyped(keys="image"),
        ])
        proc = transforms(data)["image"]  # (1,H,W)
        slices.append(proc)
    img = torch.stack(slices, dim=-1)      # (1,H,W,D)
    img = img.unsqueeze(0).float()         # (B=1,C,H,W,D)
    print("Final preprocessed shape:", img.shape)
    return img, vol

# --------------------
# Inférence slice par slice
# --------------------
def infer_i2i_volume(model, img, device="cuda"):
    """
    Inférence slice par slice sur un volume 3D avec un modèle 2D.
    img : torch.Tensor (1, 1, H, W, D)
    Retour : torch.Tensor (1, 1, H, W, D)
    """
    model.eval()
    with torch.no_grad():
        img = img.to(device)
        _, _, H, W, D = img.shape
        preds = []

        for d in range(D):
            slice_ = img[:, :, :, :, d]        # (1,1,H,W)
            out = model(slice_)                # (1,1,H,W)
            preds.append(out.cpu())

        preds = torch.stack(preds, dim=-1)     # (1,1,H,W,D)
    return preds

def apply_brain_mask(pred, orig_img, min_val=-1.0):
    """
    pred     : torch.Tensor (1,1,H,W,D) output of model
    orig_img : torch.Tensor (1,1,H,W,D) preprocessed input
    min_val  : float, value to assign outside the brain mask
    """
    mask = (orig_img != 0)  # True inside brain
    out = pred.clone()
    out[~mask] = min_val
    return out

# --------------------
# Restauration manuelle dans l’espace original
# --------------------
def restore_to_original(pred, reference_path):
    """
    pred : torch.Tensor (1,1,Hc,Wc,D)
    reference_path : chemin vers le NIfTI original
    Retour : numpy array aligné à la taille originale
    """
    ref_img = nib.load(reference_path)
    orig_shape = ref_img.shape  # ex (240,240,155)

    data = np.zeros(orig_shape, dtype=np.float32)
    pred_np = pred.squeeze().cpu().numpy()  # (Hc,Wc,D)

    # On calcule les offsets pour centrer le crop
    z_pad = (orig_shape[0] - pred_np.shape[0]) // 2
    y_pad = (orig_shape[1] - pred_np.shape[1]) // 2

    data[z_pad:z_pad+pred_np.shape[0], y_pad:y_pad+pred_np.shape[1], :] = pred_np
    return data, ref_img


def save_restored(pred, reference_path, output_path):
    data, ref_img = restore_to_original(pred, reference_path)
    nib.save(nib.Nifti1Image(data, ref_img.affine, ref_img.header), output_path)
    print(f"[INFO] Saved restored NIfTI: {output_path}")


# --------------------
# Main
# --------------------
if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="I2I inference on 3D volume")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to config yaml (synthesis.yaml)")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to input T1ce NIfTI")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to save synthetic T2")
    parser.add_argument("-w", "--weights", type=str, required=True, help="Path to trained model weights")
    args = parser.parse_args()

    # Charger la config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Charger le modèle
    model = model_factory(cfg["model"]).to(device)
    state_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(state_dict)
    print("[INFO] Model loaded")

    # Prétraitement
    img, orig = load_and_preprocess(args.input, target_size=tuple(cfg["data"]["target_size"]))

    # Inférence
    pred = infer_i2i_volume(model, img, device=device)
# Mask outside-brain regions
    pred = apply_brain_mask(pred, orig.unsqueeze(0).unsqueeze(0), min_val=-1.0)
    pred[pred > 1.5] = 1.5
    pred[pred < -1.5] = -1.5
    # Restauration et sauvegarde
    save_restored(pred, args.input, args.output)
