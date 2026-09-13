"""
Step 3: Apply dynamic int8 quantization via PyTorch.

Dynamic quantization converts Linear layer weights to int8.
No calibration data needed — activations are quantized on-the-fly.

Produces: gesture_model_quant.pt
"""

import os
import torch
import torch.nn as nn
from model import GestureNet, load_checkpoint

MODEL_PT = "gesture_model.pt"
QUANT_PT = "gesture_model_quant.pt"


def main():
    print("=== Stage 3: Dynamic Quantization ===\n")

    model, input_dim, num_classes = load_checkpoint(MODEL_PT)

    quantized = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8,
    )
    quantized.eval()

    # Save with metadata so the benchmark can reload it
    torch.save({
        "model_state": quantized.state_dict(),
        "input_dim":   input_dim,
        "num_classes": num_classes,
        "quantized":   True,
    }, QUANT_PT)

    size_kb = os.path.getsize(QUANT_PT) / 1024
    print(f"Quantized model saved: {QUANT_PT}  ({size_kb:.1f} KB)")
    print("Run 4_prune.py next.")


if __name__ == "__main__":
    main()
