    #!/usr/bin/env python3
"""
scripts/evaluate_metrics.py

Compute volume‐level PSNR and slice‐averaged SSIM between generated diffusion samples
and ground‐truth synthetic data.

Usage:
    python scripts/evaluate_metrics.py \
        --gen outputs/samples_synth.npy \
        --gt-dir data/synthetic \
        --output results/metrics.txt
"""
import os
import argparse
import numpy as np
import math
from skimage.metrics import structural_similarity as ssim

# -----------------------------------------------------------------------------------
# Compute PSNR & SSIM for 3D volumes
# -----------------------------------------------------------------------------------
def compute_metrics(gen_path, gt_dir):
    # Load generated volumes and drop any singleton dims
    raw = np.load(gen_path)                  # e.g. (N,1,1,D,H,W) or (N,1,D,H,W) or (N,D,H,W)
    gen = np.squeeze(raw)                    # now (N, D, H, W)

    # Load ground-truth volumes
    N = gen.shape[0]
    gt_list = []
    for i in range(N):
        fname = os.path.join(gt_dir, f"vox_{i:04d}.npy")
        if not os.path.exists(fname):
            raise FileNotFoundError(f"Missing ground-truth file: {fname}")
        gt = np.load(fname)
        gt_list.append(gt)
    orig = np.stack(gt_list, 0)

    # Sanity check
    if orig.shape != gen.shape:
        raise ValueError(f"Shape mismatch: gen {gen.shape} vs orig {orig.shape}")
    print(f"Comparing {N} volumes of shape {gen.shape[1:]}...")

    # Compute metrics
    psnrs = []
    ssims = []
    for i in range(N):
        gt = orig[i].astype(np.float32)
        im = gen[i].astype(np.float32)

        # PSNR over the whole volume
        mse = np.mean((gt - im)**2)
        psnr = 20 * math.log10(gt.max() / math.sqrt(mse + 1e-10))
        psnrs.append(psnr)

        # SSIM: average over non-empty slices
        slice_scores = []
        for z in range(gt.shape[0]):
            slice_gt = gt[z]
            if slice_gt.max() != slice_gt.min():
                slice_scores.append(
                    ssim(slice_gt, im[z], data_range=slice_gt.max() - slice_gt.min())
                )
        ssims.append(np.mean(slice_scores) if slice_scores else 1.0)

    return np.mean(psnrs), np.mean(ssims)

# -----------------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate PSNR and SSIM for 3D volumes")
    parser.add_argument('--gen',    type=str, required=True,
                        help="Path to generated samples .npy")
    parser.add_argument('--gt-dir', type=str, required=True,
                        help="Directory containing ground-truth .npy files")
    parser.add_argument('--output', type=str,
                        help="Optional output file for metrics summary")
    args = parser.parse_args()

    psnr_avg, ssim_avg = compute_metrics(args.gen, args.gt_dir)
    summary = f"Avg PSNR: {psnr_avg:.3f} dB\nAvg SSIM: {ssim_avg:.4f}\n"
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            f.write(summary)
        print(f"Results written to {args.output}")
    else:
        print(summary)

