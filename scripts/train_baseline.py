# scripts/train_baseline.py
import os
import datetime
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.data import DataLoader, Dataset
import numpy as np
from model import UNet3D
from scheduler import q_sample


# Setup: Dataset & DataLoader

class VoxDataset(Dataset):
    def __init__(self, folder):
        self.files = sorted(os.listdir(folder))
        self.folder = folder

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        v = np.load(f"{self.folder}/{self.files[i]}")
        return torch.from_numpy(v).unsqueeze(0).float()

# point to our synthetic spheres
ds = VoxDataset("data/synthetic")
loader = DataLoader(ds, batch_size=4, shuffle=True)


# Model & Optimizer

# tiny 3D UNet in fp16 for memory savings
model = UNet3D().cuda()
optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

# diffusion timesteps
T = 100

#
# Profiler config
# 
logdir = "profiling/baseline"
os.makedirs(logdir, exist_ok=True)
stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
num_steps = 6
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(f"{logdir}/{stamp}")
) as prof:
    # run a few iterations for baseline
    for step, x0 in enumerate(loader):
        # move data to GPU and cast to fp16
        x0 = x0.cuda()

        # thought: gotta train on every timestep for a fair baseline
        for t_val in range(T):
            # build a tensor of identical timesteps
            t = torch.full((x0.size(0),), t_val, device="cuda", dtype=torch.long)

            # forward diffusion: add noise
            with record_function(f"q_sample_t{t_val}"):
                x_noisy = q_sample(x0, t)

            # model forward: predict denoised volume
            with record_function(f"model_fwd_t{t_val}"):
                pred = model(x_noisy, t)

            # backward pass
            with record_function(f"loss_bwd_t{t_val}"):
                loss = ((pred - x0) ** 2).mean()
                loss.backward()
                optim.step()
                optim.zero_grad()

        # step the profiler (iter = 1 random batch)
        prof.step()
        if step + 1 >= num_steps :
            break

print("Baseline trace:", f"{logdir}/{stamp}")

# Summarize profiler results
# show the ops hogging GPU time
print("\n\n=========== Profiler Top-10 Ops by CUDA time ===========")
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))

# show the ops hogging memory
print("\n\n=========== Profiler Top-10 Ops by CUDA memory ===========")
print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))

# export a Chrome trace for deep-dive
json_path = os.path.join(logdir, f"{stamp}.chrome_trace.json")
try:
    prof.export_chrome_trace(json_path)
    print(f"Chrome trace exported to {json_path}")
except RuntimeError:
    print("Chrome trace already exists, skipping export.")
# save the final weights for sampling later
os.makedirs("outputs", exist_ok=True)
torch.save(model.state_dict(), "outputs/baseline_model.pth")
 
