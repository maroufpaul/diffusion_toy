# 🌀 3D Diffusion Profiling & Optimization 

> Memory-Efficient Diffusion Models for 3D Images 
> Author(s): Marouf Paul & Matthew Scanlon 
> Environment: SSH-accessed GPU machine (4GB VRAM), PyTorch, Micromamba  

---

## 🧠 Project Overview

This project implements and analyzes a lightweight **3D diffusion model** for voxel-based shape generation, with a special focus on **profiling GPU usage** and implementing **memory-efficient training techniques** like **adaptive timestep selection**.

We simulate a baseline and optimized setup using toy data (synthetic spheres) voxelized in a `64×64×64` grid to test memory and compute patterns.

---

## 📦 Repository Structure

```
diffusion_toy/
├── data/                  ← synthetic training data (voxelized spheres)
├── outputs/               ← generated voxel outputs
├── profiling/             ← PyTorch profiler traces (for TensorBoard)
├── results/               ← evaluation metrics (PSNR, SSIM, etc.)
├── scripts/               ← all training, sampling, and eval scripts
│   ├── train_baseline.py          ← baseline full-timestep training
│   ├── train_adaptive.py          ← adaptive timestep selection (SpeeD-style)
│   ├── train_adaptive2.py         ← fixed top-K timestep training (precomputed)
│   ├── compute_timestep_importance.py  ← computes which timesteps are most important
│   ├── gen_synthetic.py           ← generates toy voxel sphere data
│   ├── evaluate_metrics.py        ← computes PSNR, SSIM on generated samples
│   ├── sample_corrected.py        ← denoises and generates voxel samples
│   ├── scheduler.py               ← diffusion noise schedule (q_sample)
│   ├── model.py                   ← 3D UNet architecture
│   └── view_voxel.py              ← optional visualizer for voxel samples
├── requirements.txt        ← dependencies (for pip or Micromamba)
├── README.md               ← this file
└── .gitignore              ← ignores data, .npy, and logs
```

---

## 🧪 Goal & Tasks

### 📌 Baseline Profiling
- Train a simple 3D diffusion model on voxelized spheres.
- Use `torch.profiler` to measure memory usage and runtime bottlenecks.
- Visualize the results in **TensorBoard**.

### 📌 Adaptive Timestep Allocation (Inspired by SpeeD)
- Compute **gradient norms** for each diffusion timestep.
- Select top-K timesteps with the **largest training signal**.
- Train only on those timesteps (faster + less memory).

### 📌 Evaluation & Benchmarking
- Compare baseline vs adaptive:
  - Memory usage (CUDA)
  - Training speed (iterations/sec)
  - Image quality (PSNR, SSIM)

---

## 🛠️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/maroufpaul1/diffusion_toy.git
cd diffusion_toy
```

### 2. Setup environment

We recommend using [Micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) for isolated environments:

```bash
micromamba create -n sd3d python=3.10 -y
micromamba activate sd3d
pip install -r requirements.txt
```

**If using pip directly:**

```bash
pip install -r requirements.txt
```

---

## 🧪 Data Generation

Create synthetic 3D voxel sphere data:

```bash
python scripts/gen_synthetic.py
```

This generates `1000` random binary voxel spheres in:

```
data/synthetic/
├── vox_0000.npy
├── vox_0001.npy
...
```

Each is a `(64, 64, 64)` voxel grid.

---

## 🚀 Training

### ▶️ Baseline Training (all timesteps)

```bash
python scripts/train_baseline.py
```

- Uses all timesteps from 0 to T (default 10).
- Logs a trace to `profiling/baseline/`.
- Use TensorBoard to view memory and time per operation.

### ▶️ Adaptive Timestep Training (per-batch top-K)

```bash
python scripts/train_adaptive.py
```

- Computes gradient norms for all timesteps **on-the-fly**.
- Selects top-K for each batch and only trains on those.
- Logs to `profiling/adaptive/`.

### ▶️ Fixed Adaptive Training (precomputed top-K)

```bash
# First compute top-K timesteps
python scripts/compute_timestep_importance.py

# Then train using fixed timesteps
python scripts/train_adaptive2.py
```

---

## 📊 Profiling & TensorBoard

To visualize memory and performance traces:

```bash
tensorboard --logdir profiling --port 6006 --bind_all
```

Access via your browser (e.g. `http://YOUR_SSH_HOST:6006`).

To fix visibility of traces:

```bash
mkdir -p profiling/baseline/<timestamp>/plugins/profile
mv *.pt.trace.json profiling/baseline/<timestamp>/plugins/profile/
```

Repeat similarly for `adaptive/`.

---

## 🧪 Generating Samples

After training:

```bash
python scripts/sample_corrected.py
```

This denoises 1000 samples and saves them to:

```
outputs/samples_synth.npy
```

---

## 📈 Evaluation (PSNR & SSIM)

Compare generated samples against ground-truth:

```bash
python scripts/evaluate_metrics.py \
  --gen outputs/samples_synth.npy \
  --gt-dir data/synthetic \
  --output results/metrics.txt
```

Metrics include:

- **PSNR (Peak Signal-to-Noise Ratio)**: voxel-wise fidelity
- **SSIM (Structural Similarity Index)**: perceptual similarity

---

## 🧠 Key Concepts

### 🔄 Diffusion Timesteps

At each timestep `t`, noise is added using:

```python
x_noisy = sqrt(α_t) * x0 + sqrt(1 - α_t) * noise
```

The model is trained to **denoise** this `x_noisy` back to `x0`.

### ⚡ Adaptive Timestep Scheduling

- Not all timesteps contribute equally.
- We measure **gradient norm** per `t`.
- Top-K timesteps are used for faster and more memory-efficient training.

---

## 🔍 What You Can Explore Next

- Try different base shapes (e.g., cubes, combined primitives).
- Use more realistic voxelized objects (e.g., from ShapeNet).
- Implement full reverse sampling loop (e.g., DDPM).
- Add `FID`, `LPIPS`, or `MMD/COV` evaluation if needed.

---

## 📜 Citation

This project adapts ideas from:
- [SpeeD: Fast and Accurate Image Generation with Denoising Diffusion GANs (2023)](https://arxiv.org/abs/2307.05072)
- [High-Resolution Image Synthesis with Latent Diffusion Models (CVPR 2022)](https://arxiv.org/abs/2112.10752)

---

## 🙏 Acknowledgments

Special thanks to the course staff, Stability AI (original SD), and community examples from GitHub and papers.

---

