"""Shared model definition used by all pipeline scripts."""

import torch.nn as nn


class GestureNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),        nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_checkpoint(path, map_location="cpu"):
    """Load a saved checkpoint and return (model, input_dim, num_classes)."""
    import torch
    ckpt  = torch.load(path, map_location=map_location)
    model = GestureNet(ckpt["input_dim"], ckpt["num_classes"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["input_dim"], ckpt["num_classes"]
