"""
Step 4: Apply magnitude-based unstructured pruning (50% sparsity) using PyTorch.

Pruning zeroes out the smallest weights in each Linear layer.
After pruning, we fine-tune for a few epochs to recover accuracy,
then make the sparsity permanent by removing the pruning masks.

Produces: gesture_model_pruned.pt
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from model import GestureNet, load_checkpoint

MODEL_PT   = "gesture_model.pt"
PRUNED_PT  = "gesture_model_pruned.pt"

SPARSITY        = 0.5
FINETUNE_EPOCHS = 10
BATCH_SIZE      = 32
LR              = 1e-3


def apply_pruning(model, sparsity):
    """Zero out `sparsity` fraction of weights in every Linear layer."""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=sparsity)
    return model


def remove_pruning_masks(model):
    """Make pruning permanent — bakes the zero mask into the actual weights."""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            try:
                prune.remove(module, "weight")
            except ValueError:
                pass
    return model


def run_epoch(model, loader, optimizer, criterion, device, train=True):
    model.train() if train else model.eval()
    total_loss = correct = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if train:
                optimizer.zero_grad()
            out  = model(xb)
            loss = criterion(out, yb)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(xb)
            correct    += (out.argmax(1) == yb).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


def main():
    print("=== Stage 4: Pruning ===\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reload training data for fine-tuning
    df = pd.read_csv("gesture_data.csv")
    X  = df.drop("label", axis=1).values.astype("float32")
    y  = LabelEncoder().fit_transform(df["label"].values).astype("int64")
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    X_test = np.load("X_test.npy")
    y_test = np.load("y_test.npy").astype("int64")

    train_dl = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=BATCH_SIZE, shuffle=True)
    test_dl  = DataLoader(
        TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
        batch_size=BATCH_SIZE)

    model, input_dim, num_classes = load_checkpoint(MODEL_PT, map_location=device)
    model = model.to(device)

    model = apply_pruning(model, SPARSITY)
    print(f"Applied {SPARSITY*100:.0f}% pruning. Fine-tuning for {FINETUNE_EPOCHS} epochs...\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, FINETUNE_EPOCHS + 1):
        tr_loss, tr_acc   = run_epoch(model, train_dl, optimizer, criterion, device, train=True)
        val_loss, val_acc = run_epoch(model, test_dl,  optimizer, criterion, device, train=False)
        print(f"Epoch {epoch:2d}/{FINETUNE_EPOCHS}  train={tr_acc*100:.1f}%  val={val_acc*100:.1f}%")

    # Make pruning permanent before saving
    model = remove_pruning_masks(model)
    _, final_acc = run_epoch(model, test_dl, optimizer, criterion, device, train=False)
    print(f"\nPruned model accuracy: {final_acc*100:.2f}%")

    torch.save({
        "model_state": model.state_dict(),
        "input_dim":   input_dim,
        "num_classes": num_classes,
    }, PRUNED_PT)
    print(f"Saved {PRUNED_PT}")
    print("Run 5_benchmark_all.py next.")


if __name__ == "__main__":
    main()
