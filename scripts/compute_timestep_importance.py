#!/usr/bin/env python3
"""
Compute gradient-norm importance for each diffusion timestep T, offline.
Outputs a Python file with the list of top-K timesteps to keep.
"""
import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from model import UNet3D
from scheduler import q_sample

# Configuration
T = 100               # total timesteps
pilot_batches = 2    # number of batches to use for importance estimation
batch_size = 4
K = 10              # number of timesteps to keep (top-K)

# Dataset loader
class VoxDataset(Dataset):
    def __init__(self, folder):
        self.files = sorted(os.listdir(folder))
        self.folder = folder
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        v = np.load(os.path.join(self.folder, self.files[idx]))
        return torch.from_numpy(v).unsqueeze(0).float()

# Main
def main():
    data_folder = os.path.join(os.path.dirname(__file__), '../data/synthetic')
    ds = VoxDataset(data_folder)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = UNet3D(in_ch=1, base_ch=32).cuda()
    model.eval()

    grad_sums = torch.zeros(T, device='cuda')
    counts = torch.zeros(T, device='cuda')

    # Disable grad checkpointing for importance sweep
    with torch.no_grad():
        # Actually we need grads, so enable:
        torch.set_grad_enabled(True)
        for i, x0 in enumerate(loader):
            if i >= pilot_batches: break
            x0 = x0.cuda()
            for t_val in range(T):
                t = torch.full((x0.size(0),), t_val, device='cuda', dtype=torch.long)
                x_noisy = q_sample(x0, t)

                # Compute loss and gradients
                pred = model(x_noisy, t)
                loss = ((pred - x0)**2).mean()
                model.zero_grad()
                loss.backward()

                # Sum gradient norms
                total_norm = torch.norm(torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])).item()
                grad_sums[t_val] += total_norm
                counts[t_val] += 1

        # average
        grad_means = grad_sums / counts
        # pick top-K timesteps
        topk = torch.topk(grad_means, K).indices.cpu().tolist()
        topk_sorted = sorted(topk)

        # Save to file
        out = {'important_timesteps': topk_sorted}
        with open(os.path.join(os.path.dirname(__file__), 'important_timesteps.json'), 'w') as f:
            json.dump(out, f, indent=2)
        print(f"Top-{K} timesteps: {topk_sorted}")
        print("Saved to scripts/important_timesteps.json")

if __name__ == '__main__':
    main()
