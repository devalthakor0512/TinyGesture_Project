"""
Step 2: Train the baseline gesture classifier using PyTorch.

Produces:
  gesture_model.pt          — saved model weights + metadata
  label_encoder_classes.npy — class name mapping
  X_test.npy / y_test.npy   — held-out test set for later steps
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from model import GestureNet

DATA_FILE     = "gesture_data.csv"
MODEL_PT      = "gesture_model.pt"
LABEL_CLASSES = "label_encoder_classes.npy"
EPOCHS        = 30
BATCH_SIZE    = 32
LR            = 1e-3


def load_data():
    df = pd.read_csv(DATA_FILE)
    X  = df.drop("label", axis=1).values.astype("float32")
    le = LabelEncoder()
    y  = le.fit_transform(df["label"].values).astype("int64")
    np.save(LABEL_CLASSES, le.classes_)
    print(f"Classes: {le.classes_}")
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y), le


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
    print("=== Stage 2: Training Baseline Model ===\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    (X_train, X_test, y_train, y_test), le = load_data()
    print(f"Train: {len(X_train)}  Test: {len(X_test)}\n")

    np.save("X_test.npy", X_test)
    np.save("y_test.npy", y_test)

    train_dl = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=BATCH_SIZE, shuffle=True)
    test_dl  = DataLoader(
        TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
        batch_size=BATCH_SIZE)

    model     = GestureNet(X_train.shape[1], len(le.classes_)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc   = run_epoch(model, train_dl, optimizer, criterion, device, train=True)
        val_loss, val_acc = run_epoch(model, test_dl,  optimizer, criterion, device, train=False)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS}  "
                  f"train={tr_acc*100:.1f}%  val={val_acc*100:.1f}%")

    _, final_acc = run_epoch(model, test_dl, optimizer, criterion, device, train=False)
    print(f"\nBaseline accuracy: {final_acc*100:.2f}%")

    torch.save({
        "model_state": model.state_dict(),
        "input_dim":   X_train.shape[1],
        "num_classes": len(le.classes_),
    }, MODEL_PT)
    print(f"Saved {MODEL_PT}")


if __name__ == "__main__":
    main()
