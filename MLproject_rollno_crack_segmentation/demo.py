import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.io
import torch

from src.unet import UNet

def run_demo(sample_id="001", checkpoint_path="checkpoints/proposed_unet_weighted.pth", baseline_path="checkpoints/baseline_unet_w1.pth", output_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Normalize sample_id to 3 digits (e.g. "2" -> "002")
    if sample_id.isdigit():
        sample_id = f"{int(sample_id):03d}"
        
    print(f"Running inference demo on sample: {sample_id} using {device}")
    
    img_path = os.path.join("CrackForest-dataset", "image", f"{sample_id}.jpg")
    gt_path = os.path.join("CrackForest-dataset", "groundTruth", f"{sample_id}.mat")
    
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found at {img_path}. Please choose a sample ID between 001 and 118.")
        
    # Load and preprocess
    raw_img = Image.open(img_path).convert("RGB").resize((256, 256), Image.BILINEAR)
    img_np = np.array(raw_img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm_img = (img_np - mean) / std
    t_img = torch.from_numpy(norm_img.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    
    # Ground truth
    has_gt = os.path.exists(gt_path)
    if has_gt:
        mat = scipy.io.loadmat(gt_path)
        raw_seg = mat["groundTruth"][0, 0]["Segmentation"]
        gt_mask = (raw_seg == 2).astype(np.uint8) * 255
        gt_mask = np.array(Image.fromarray(gt_mask).resize((256, 256), Image.NEAREST)) > 127
        
    # Load Models
    prop_model = UNet(base_features=16).to(device)
    prop_model.load_state_dict(torch.load(checkpoint_path, map_location=device)["model_state_dict"])
    prop_model.eval()
    
    base_model = UNet(base_features=16).to(device)
    base_model.load_state_dict(torch.load(baseline_path, map_location=device)["model_state_dict"])
    base_model.eval()
    
    with torch.no_grad():
        p_prob = torch.sigmoid(prop_model(t_img)).squeeze().cpu().numpy()
        b_prob = torch.sigmoid(base_model(t_img)).squeeze().cpu().numpy()
        
    p_mask = (p_prob > 0.5).astype(np.float32)
    b_mask = (b_prob > 0.5).astype(np.float32)
    
    # Visualization
    cols = 5 if has_gt else 4
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))
    
    axes[0].imshow(raw_img)
    axes[0].set_title(f"Input Pavement ({sample_id}.jpg)", fontsize=11, fontweight="bold")
    axes[0].axis("off")
    
    if has_gt:
        axes[1].imshow(gt_mask, cmap="gray")
        axes[1].set_title("Ground Truth Mask", fontsize=11, fontweight="bold")
        axes[1].axis("off")
        
    ax_b = axes[2] if has_gt else axes[1]
    ax_b.imshow(b_mask, cmap="magma")
    ax_b.set_title("Baseline (w=1.0 BCE)\n[Fragmented / Broken]", fontsize=10, color="#1f77b4", fontweight="bold")
    ax_b.axis("off")
    
    ax_p = axes[3] if has_gt else axes[2]
    ax_p.imshow(p_mask, cmap="viridis")
    ax_p.set_title("Proposed (w=5.0 WBCE)\n[Continuous Crack]", fontsize=10, color="#d62728", fontweight="bold")
    ax_p.axis("off")
    
    ax_o = axes[4] if has_gt else axes[3]
    overlay = np.array(raw_img).copy()
    overlay[p_mask == 1] = [255, 40, 40]
    ax_o.imshow(overlay)
    ax_o.set_title("Proposed Overlay on Road", fontsize=11, fontweight="bold")
    ax_o.axis("off")
    
    plt.tight_layout()
    
    # Determine output path: unique per sample_id to prevent file collisions
    if output_path is None:
        os.makedirs("results", exist_ok=True)
        target_path = os.path.join("results", f"demo_{sample_id}.png")
    else:
        target_path = output_path
        
    target_dir = os.path.dirname(target_path)
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)
        
    # Robust save: handles file locking by Windows Photos viewer or OneDrive sync
    try:
        plt.savefig(target_path, dpi=300)
        saved_path = target_path
    except OSError as e:
        # File is likely open in Windows Photos or locked by OneDrive sync
        ts = int(time.time())
        fallback_path = os.path.join("results", f"demo_{sample_id}_{ts}.png")
        print(f"\n[Note] Could not overwrite '{target_path}' because it is open in an image viewer or locked by OneDrive.")
        print(f"       Saving to fallback location: '{fallback_path}' instead.")
        plt.savefig(fallback_path, dpi=300)
        saved_path = fallback_path
        
    plt.close()
    print(f"\n[SUCCESS] Inference output saved to: {saved_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pavement Crack Segmentation Inference Demo")
    parser.add_argument("--sample_id", type=str, default="002", help="Sample ID from CrackForest dataset (e.g. 001, 002, 010, 085)")
    parser.add_argument("--output", type=str, default=None, help="Custom output image path (default: results/demo_{sample_id}.png)")
    args = parser.parse_args()
    run_demo(args.sample_id, output_path=args.output)
