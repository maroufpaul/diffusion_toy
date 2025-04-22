#!/usr/bin/env python3
"""
Training loop that uses a fixed subset of timesteps (adaptive schedule) per iteration.
Loads important_timesteps.json and trains only on those timesteps.
"""
import os
import json
import datetime
import torch
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.data import Dataset, DataLoader
from model import UNet3D
from scheduler import q_sample

# Hyperparameters
batch_size = 4
num_steps  = 6
learning_rate = 1e-4

# Load important timesteps
script_dir = os.path.dirname(__file__)
with open(os.path.join(script_dir, 'important_timesteps.json')) as f:
    important_ts = sorted(json.load(f)['important_timesteps'])
T = max(important_ts) + 1

# Dataset
class VoxDataset(Dataset):
    def __init__(self, folder):
        self.files = sorted(os.listdir(folder))
        self.folder = folder
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        v = np.load(os.path.join(self.folder, self.files[idx]))
        return torch.from_numpy(v).unsqueeze(0).float()

def main():
    # Data
    dataset = VoxDataset(os.path.join(script_dir, '../data/synthetic'))
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model & optimizer
    model = UNet3D(in_ch=1, base_ch=16).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Profiler
    logdir = '../profiling/adaptive_fixed'
    os.makedirs(logdir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f"{logdir}/{stamp}"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for step, x0 in enumerate(loader):
            x0 = x0.cuda()

            for t_val in important_ts:
                t = torch.full((x0.size(0),), t_val, device='cuda', dtype=torch.long)
                with record_function(f"qsamp_t{t_val}"):
                    x_noisy = q_sample(x0, t)
                with record_function(f"fwd_t{t_val}"):
                    pred = model(x_noisy, t)
                    loss = ((pred - x0)**2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            prof.step()
            if step + 1 >= num_steps:
                break

    print(f"Adaptive-fixed trace written to {logdir}/{stamp}")
    print("\n\n=========== Adaptive‑Fixed Top‑10 Ops by CUDA time ===========")
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=10))
    print("\n\n=========== Adaptive‑Fixed Top‑10 Ops by CUDA memory ===========")
    print(prof.key_averages().table(
        sort_by="cuda_memory_usage", row_limit=10))

    # Export Chrome trace
    json_path = os.path.join(logdir, f"{stamp}.adaptive_fixed.chrome_trace.json")
    prof.export_chrome_trace(json_path)
    print(f"Chrome trace exported to {json_path}")

if __name__ == '__main__':
    main()
