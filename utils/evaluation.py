# utils/evaluation.py
import torch
import matplotlib.pyplot as plt
import numpy as np
from monai.metrics import DiceMetric
from skimage.metrics import peak_signal_noise_ratio as psnr


# ------------------------
# Classification
# ------------------------
import torch
import matplotlib.pyplot as plt
import math

def evaluate_classification(model, val_loader, device="cuda", max_examples=12, class_names=None):
    """
    Evaluate classification model and show a mosaic of examples.
    
    Args:
        model: trained model
        val_loader: validation DataLoader
        device: "cuda" or "cpu"
        max_examples: number of samples to visualize
        class_names: optional dict or list mapping label indices to names
    """
    model.eval()
    correct, total = 0, 0
    examples = []

    with torch.no_grad():
        for batch in val_loader:
            images, labels = batch["image"].to(device), batch["label"].to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for i in range(images.size(0)):
                if len(examples) < max_examples:
                    examples.append((
                        images[i].cpu(), 
                        int(labels[i].cpu().item()), 
                        int(preds[i].cpu().item())
                    ))

    acc = correct / total
    print(f"[Classification] Final accuracy: {acc:.4f}")

    # --- mosaic display ---
    n = len(examples)
    cols = min(6, n)  # max 6 columns
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))

    if rows == 1:
        axes = [axes]  # ensure iterable

    for idx, (img, label, pred) in enumerate(examples):
        r, c = divmod(idx, cols)
        ax = axes[r][c] if rows > 1 else axes[c]

        img = img[0].numpy() if img.ndim == 3 else img.numpy()
        ax.imshow(img, cmap="gray")
        gt_name = class_names[label] if class_names else label
        pred_name = class_names[pred] if class_names else pred
        ax.set_title(f"GT={gt_name} / Pred={pred_name}", fontsize=10)
        ax.axis("off")

    # hide unused subplots
    for j in range(n, rows*cols):
        r, c = divmod(j, cols)
        ax = axes[r][c] if rows > 1 else axes[c]
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    return acc




# ------------------------
# Segmentation
# ------------------------
import matplotlib.pyplot as plt
import torch

def evaluate_segmentation(model, val_loader, device="cuda", max_examples=6):
    """
    Run inference on val_loader and show a mosaic of input / GT / Pred.
    """
    model.eval()
    examples = []
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch["image"].to(device), batch["label"].to(device)
            outputs = torch.sigmoid(model(inputs))
            preds = (outputs > 0.5).float()

            for i in range(inputs.shape[0]):
                examples.append((inputs[i].cpu(), labels[i].cpu(), preds[i].cpu()))
                if len(examples) >= max_examples:
                    break
            if len(examples) >= max_examples:
                break

    # --- Plot mosaic ---
    n = len(examples)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3*n))
    if n == 1:
        axes = [axes]  # make iterable if only one row

    for idx, (img, gt, pred) in enumerate(examples):
        # assume (C,H,W), pick channel 0 for grayscale background
        img_np = img[0].numpy() if img.ndim == 3 else img.numpy()
        gt_np  = gt.squeeze().numpy()
        pred_np = pred.squeeze().numpy()

        axes[idx][0].imshow(img_np, cmap="gray")
        axes[idx][0].set_title("Input")
        axes[idx][0].axis("off")

        axes[idx][1].imshow(img_np, cmap="gray")
        axes[idx][1].imshow(gt_np, cmap="Reds", alpha=0.4)
        axes[idx][1].set_title("GT")
        axes[idx][1].axis("off")

        axes[idx][2].imshow(img_np, cmap="gray")
        axes[idx][2].imshow(pred_np, cmap="Blues", alpha=0.4)
        axes[idx][2].set_title("Prediction")
        axes[idx][2].axis("off")

    plt.tight_layout()
    plt.show()


# ------------------------
# Image-to-Image
# ------------------------
def evaluate_i2i(model, val_loader, device="cuda", max_examples=3):
    model.eval()
    total_psnr, total_samples = 0, 0
    examples = []

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            inputs, targets = batch["image"].to(device), batch["label"].to(device)
            outputs = model(inputs)

            # compute PSNR
            psnr_val = psnr(
                targets.cpu().numpy().squeeze(),
                outputs.cpu().numpy().squeeze(),
                data_range=2.0  # since scaled to [-1,1]
            )
            total_psnr += psnr_val
            total_samples += 1

            if i < max_examples:
                examples.append((
                    images.cpu()[0], 
                    labels.cpu()[0].item(),   # scalar
                    preds.cpu()[0].item()     # scalar
                ))

    mean_psnr = total_psnr / total_samples
    print(f"[I2I] Final mean PSNR: {mean_psnr:.2f} dB")

    # visualize triplets
    for idx, (img, label, pred) in enumerate(examples):
        img = img[0].numpy() if img.ndim == 3 else img.numpy()
        axes[idx,0].imshow(img, cmap="gray"); axes[idx,0].set_title("Input"); axes[idx,0].axis("off")
        axes[idx,1].text(0.5,0.5,f"GT: {label}",ha="center",va="center",fontsize=12); axes[idx,1].axis("off")
        axes[idx,2].text(0.5,0.5,f"Pred: {pred}",ha="center",va="center",fontsize=12); axes[idx,2].axis("off")


    return mean_psnr
