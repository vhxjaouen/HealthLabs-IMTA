import matplotlib.pyplot as plt
from skimage import measure

def show_overlay(img, label, pred, epoch=None, channel=0):
    """
    img   : torch.Tensor (3,H,W) input image
    label : torch.Tensor (1,H,W) ground truth mask (binary)
    pred  : torch.Tensor (1,H,W) predicted mask (binary)
    """
    img_np   = img.detach().cpu().permute(1,2,0).numpy()
    label_np = label.detach().cpu().squeeze().numpy()
    pred_np  = pred.detach().cpu().squeeze().numpy()

    plt.figure(figsize=(12,4))

    # --- Input image ---
    plt.subplot(1,3,1)
    plt.imshow(img_np[...,channel], cmap="gray", vmin=-1, vmax=1)
    plt.title(f"Input (channel {channel})")
    plt.axis("off")

    # --- Ground truth contours ---
    plt.subplot(1,3,2)
    plt.imshow(img_np[...,channel], cmap="gray", vmin=-1, vmax=1)
    for contour in measure.find_contours(label_np, 0.5):
        plt.plot(contour[:,1], contour[:,0], color="red", linewidth=2)
    plt.title("Ground truth")
    plt.axis("off")

    # --- Prediction contours ---
    plt.subplot(1,3,3)
    plt.imshow(img_np[...,channel], cmap="gray", vmin=-1, vmax=1)
    for contour in measure.find_contours(pred_np, 0.5):
        plt.plot(contour[:,1], contour[:,0], color="blue", linewidth=2)
    plt.title("Prediction")
    plt.axis("off")

    if epoch is not None:
        plt.suptitle(f"Epoch {epoch}")
    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import numpy as np

def show_i2i_triplet(inp, target, pred, epoch=None, cmap="gray"):
    """
    Affiche un triplet input / target / pred côte à côte.
    - inp, target, pred: torch.Tensor (H,W) ou (1,H,W)
    """
    inp_np = inp.detach().cpu().squeeze().numpy()
    tgt_np = target.detach().cpu().squeeze().numpy()
    pred_np = pred.detach().cpu().squeeze().numpy()

    plt.figure(figsize=(12,4))
    
    plt.subplot(1,3,1)
    plt.imshow(inp_np, cmap=cmap, vmin=-1, vmax=1)
    plt.title("Input (T1ce)")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(tgt_np, cmap=cmap, vmin=-1, vmax=1)
    plt.title("Target (T2)")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(pred_np, cmap=cmap, vmin=-1, vmax=1)
    plt.title("Prediction")
    plt.axis("off")

    if epoch is not None:
        plt.suptitle(f"Epoch {epoch}")
    plt.tight_layout()
    plt.show()
