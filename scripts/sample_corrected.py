#!/usr/bin/env python3
"""
scripts/sample_corrected.py

Generate N toy‐sphere volumes using the trained UNet, and save as a 4D array
(samples, depth, height, width).  Drops all singleton dims so evaluation lines up.
"""

import os, glob
import torch
import numpy as np
from model import UNet3D

# ─── CONFIG ────────────────────────────────────────────────────────────
T        = 10                   # diffusion timesteps
N        = 1000                 # number of samples
GT_GLOB  = "data/synthetic/vox_*.npy"
CKPT     = "outputs/baseline_model.pth"
OUT_PATH = "outputs/samples_synth.npy"
DEVICE   = "cuda"

# ─── Infer resolution from your ground‐truth set ────────────────────────
first_gt = sorted(glob.glob(GT_GLOB))[0]
D, H, W  = np.load(first_gt).shape
print(f"Sampling {N} volumes at resolution ({D},{H},{W})")

# ─── Load model ────────────────────────────────────────────────────────
model = UNet3D(in_ch=1, base_ch=32).to(DEVICE)
if not os.path.exists(CKPT):
    raise FileNotFoundError(f"Checkpoint not found: {CKPT}")
model.load_state_dict(torch.load(CKPT))
model.eval()

# ─── Sampling loop ─────────────────────────────────────────────────────
samples = []
for i in range(N):
    # start from pure noise
    x = torch.randn(1,1,D,H,W, device=DEVICE)
    # simple reverse loop: each step we feed the model’s x0‐estimate back in
    for t_val in reversed(range(T)):
        t = torch.full((1,), t_val, device=DEVICE, dtype=torch.long)
        with torch.no_grad():
            x = model(x, t)
    # x is shape (1,1,D,H,W) → drop both first dims to get (D,H,W)
    samples.append(x[0,0].cpu().numpy())

# ─── Save ──────────────────────────────────────────────────────────────
samples = np.stack(samples, 0)   # → (N, D, H, W)
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
np.save(OUT_PATH, samples)
print(f"✅ Saved {N} samples of shape {samples.shape} to {OUT_PATH}")

