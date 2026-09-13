"""
Step 5: Benchmark all three models and produce the comparison table.

Measures for each model:
  - File size (KB)
  - Classification accuracy on the held-out test set
  - Average inference latency (ms) over 100 runs

Produces: compression_comparison_results.csv
"""

import os
import time
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import GestureNet

MODELS = {
    "Baseline (float32)":  "gesture_model.pt",
    "Quantized (int8)":    "gesture_model_quant.pt",
    "Pruned (50% sparse)": "gesture_model_pruned.pt",
}
OUTPUT_CSV   = "compression_comparison_results.csv"
LATENCY_RUNS = 100


def load_model(path):
    ckpt = torch.load(path, map_location="cpu")
    model = GestureNet(ckpt["input_dim"], ckpt["num_classes"])

    if ckpt.get("quantized"):
        # Re-apply dynamic quantization wrapper before loading quantized state dict
        model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def benchmark(path, X_test, y_test):
    size_kb = os.path.getsize(path) / 1024
    model   = load_model(path)

    X_t = torch.from_numpy(X_test)
    y_t = torch.from_numpy(y_test.astype("int64"))

    # Accuracy
    with torch.no_grad():
        logits  = model(X_t)
        correct = (logits.argmax(1) == y_t).sum().item()
    accuracy = correct / len(y_t) * 100

    # Latency — single-sample inference, averaged over LATENCY_RUNS
    times = []
    with torch.no_grad():
        for i in range(LATENCY_RUNS):
            sample = X_t[i % len(X_t)].unsqueeze(0)
            t0 = time.perf_counter()
            model(sample)
            times.append((time.perf_counter() - t0) * 1000)

    return size_kb, accuracy, float(np.mean(times))


def main():
    print("=== Stage 5: Benchmarking All Models ===\n")

    X_test = np.load("X_test.npy")
    y_test = np.load("y_test.npy")
    print(f"Test samples: {len(X_test)}\n")

    results = []
    for name, path in MODELS.items():
        if not os.path.exists(path):
            print(f"  SKIP — {path} not found")
            continue
        print(f"Benchmarking: {name} ...")
        size_kb, acc, latency = benchmark(path, X_test, y_test)
        results.append({
            "Technique":    name,
            "Size (KB)":    f"{size_kb:.1f}",
            "Accuracy (%)": f"{acc:.2f}",
            "Latency (ms)": f"{latency:.3f}",
        })
        print(f"  {size_kb:.1f} KB | {acc:.2f}% | {latency:.3f} ms")

    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Technique","Size (KB)","Accuracy (%)","Latency (ms)"])
        w.writeheader()
        w.writerows(results)

    print(f"\nResults saved to {OUTPUT_CSV}")
    print("\n" + "="*62)
    print(f"{'Technique':<25} {'Size (KB)':>10} {'Accuracy (%)':>13} {'Latency (ms)':>13}")
    print("-"*62)
    for r in results:
        print(f"{r['Technique']:<25} {r['Size (KB)']:>10} {r['Accuracy (%)']:>13} {r['Latency (ms)']:>13}")
    print("="*62)


if __name__ == "__main__":
    main()
