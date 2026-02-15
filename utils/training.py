import torch
import tqdm

def train_segmentation(model, train_loader, val_loader, loss_fn, dice_metric, optimizer,
                device="cuda", max_epochs=10, overlay_fn=None):
    train_losses, val_dices = [], []
    best_dice, best_weights = -1, None

    for epoch in range(max_epochs):
        print(f"Epoch {epoch+1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        for batch in tqdm.tqdm(train_loader, desc=f"Train {epoch+1}", leave=False):
            inputs, labels = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)

        # --- Validation ---
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                inputs, labels = batch["image"].to(device), batch["label"].to(device)
                outputs = torch.sigmoid(model(inputs))
                preds = (outputs > 0.5).float()
                dice_metric(y_pred=preds, y=labels)

                if overlay_fn and i == 0:
                    overlay_fn(inputs[0], labels[0], preds[0], epoch+1)

            mean_dice = dice_metric.aggregate().item()
            dice_metric.reset()
            val_dices.append(mean_dice)

        print(f"  Loss: {epoch_loss:.4f}, Val Dice: {mean_dice:.4f}")

        if mean_dice > best_dice:
            best_dice, best_weights = mean_dice, model.state_dict().copy()

    return train_losses, val_dices, best_dice, best_weights

def compute_accuracy(y_pred, y_true):
    preds = torch.argmax(y_pred, dim=1)
    correct = (preds == y_true).float().sum()
    return correct / y_true.numel()

def train_classification(model, train_loader, val_loader, loss_fn, optimizer,
                         device="cuda", max_epochs=10, save_path=None):
    train_losses, val_accs = [], []
    best_acc, best_weights = -1, None

    for epoch in range(max_epochs):
        print(f"   Epoch {epoch+1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        for batch in tqdm.tqdm(train_loader, desc=f"Train {epoch+1}", leave=False):
            inputs, labels = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)

        # --- Validation ---
        model.eval()
        total_correct, total_samples = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch["image"].to(device), batch["label"].to(device)
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.numel()
        mean_acc = total_correct / total_samples if total_samples > 0 else 0
        val_accs.append(mean_acc)

        print(f"Loss: {epoch_loss:.4f}, Val Acc: {mean_acc:.4f}")

        import copy
        if mean_acc > best_acc:
            best_acc, best_weights = mean_acc, model.state_dict().copy()
            if save_path is not None:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                best_weights = copy.deepcopy(model.state_dict())
                torch.save(best_weights, save_path)
                print(f"Best model saved to {save_path}")

    return train_losses, val_accs, best_acc, best_weights

import tqdm
import numpy as np
import os
def train_i2i(
    model, train_loader, val_loader,
    loss_fn, optimizer,
    device="cuda", max_epochs=10,
    overlay_fn=None,
    val_metric="mae"
):
    train_losses, val_metrics = [], []
    best_val, best_weights = float(0.), None
    maximize = val_metric == "psnr"

    for epoch in range(max_epochs):
        print(f"Epoch {epoch+1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            inputs, targets = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)

        # --- Validation ---
        model.eval()
        total, n = 0, 0
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                inputs, targets = batch["image"].to(device), batch["label"].to(device)
                outputs = model(inputs)

                # visualisation (1er batch uniquement)
                if overlay_fn and i == 0:
                    overlay_fn(inputs[0], targets[0], outputs[0], epoch+1)

                if val_metric == "mae":
                    metric_val = torch.abs(outputs - targets).mean().item()
                elif val_metric == "psnr":
                    mse = torch.mean((outputs - targets) ** 2).item()
                    metric_val = 20 * np.log10(2.0 / np.sqrt(mse)) if mse > 0 else 100.0
                else:
                    raise ValueError(f"Unknown val_metric: {val_metric}")

                total += metric_val
                n += 1

        val_score = total / n
        val_metrics.append(val_score)

        improved = val_score > best_val if maximize else val_score < best_val
        if improved:
            print(("  New best model found!" if epoch > 0 else "  First model saved!"))
            best_val, best_weights = val_score, model.state_dict().copy()

        print(f"  Train Loss: {epoch_loss:.4f}, Val {val_metric.upper()}: {val_score:.4f}")

    return train_losses, val_metrics, best_val, best_weights


