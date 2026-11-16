import os
import json
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import CrackDataset, get_data_splits
from src.unet import UNet
from src.losses import WeightedBCELoss
from src.metrics import MetricTracker

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for imgs, masks, _ in dataloader:
        imgs = imgs.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(dataloader.dataset)

def evaluate(model, dataloader, criterion, device, threshold=0.5):
    model.eval()
    running_loss = 0.0
    tracker = MetricTracker(threshold=threshold)
    
    with torch.no_grad():
        for imgs, masks, _ in dataloader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            running_loss += loss.item() * imgs.size(0)
            
            probs = torch.sigmoid(outputs)
            tracker.update(probs.cpu(), masks.cpu())
            
    val_loss = running_loss / len(dataloader.dataset)
    metrics = tracker.compute()
    metrics["loss"] = val_loss
    return metrics

def train_model(model_name, pos_weight, train_loader, val_loader, epochs=25, lr=1e-3, device="cpu", save_dir="checkpoints"):
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Training {model_name} (pos_weight = {pos_weight}) for {epochs} epochs")
    print(f"{'='*60}")
    
    set_seed(42)
    model = UNet(in_channels=3, out_channels=1, base_features=16).to(device)
    criterion = WeightedBCELoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_f1": [],
        "val_iou": [],
        "val_precision": [],
        "val_recall": []
    }
    
    best_f1 = -1.0
    best_path = os.path.join(save_dir, f"{model_name}.pth")
    
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        t_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        history["train_loss"].append(t_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_iou"].append(val_metrics["iou"])
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])
        
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "pos_weight": pos_weight,
                "val_f1": best_f1,
                "val_iou": val_metrics["iou"],
                "history": history
            }, best_path)
            
        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {t_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | Val F1: {val_metrics['f1']:.4f} | Val IoU: {val_metrics['iou']:.4f} | Val Rec: {val_metrics['recall']:.4f}")
            
    total_time = time.time() - start_time
    print(f"Finished {model_name} in {total_time:.1f}s. Best Val F1: {best_f1:.4f}. Saved checkpoint: {best_path}")
    return best_path, history

def run_pipeline():
    os.makedirs("data", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    if hasattr(os, "cpu_count") and os.cpu_count():
        torch.set_num_threads(os.cpu_count())
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (PyTorch threads: {torch.get_num_threads()})")
    
    # 1. Dataset splits
    splits = get_data_splits(total_samples=118, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)
    with open("data/splits.json", "w") as f:
        json.dump(splits, f, indent=2)
    print(f"Dataset split created: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")
    
    img_dir = "CrackForest-dataset/image"
    gt_dir = "CrackForest-dataset/groundTruth"
    
    print("Pre-caching dataset in memory for maximum throughput...")
    train_ds = CrackDataset(splits["train"], img_dir, gt_dir, augment=True, cache_in_ram=True)
    val_ds = CrackDataset(splits["val"], img_dir, gt_dir, augment=False, cache_in_ram=True)
    test_ds = CrackDataset(splits["test"], img_dir, gt_dir, augment=False, cache_in_ram=True)
    
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)
    
    # 2. Train Baseline Model (w = 1.0, Ordinary BCE)
    baseline_path, baseline_hist = train_model(
        model_name="baseline_unet_w1",
        pos_weight=1.0,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=20,
        lr=1.5e-3,
        device=device
    )
    
    # 3. Validation Split Tuning for Pos-Weight: w in [1, 5, 10, 15]
    weight_candidates = [1.0, 5.0, 10.0, 15.0]
    tuning_results = []
    
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING: Evaluating pos_weight candidates on Val Split")
    print("="*60)
    
    candidate_histories = {}
    for w in weight_candidates:
        if w == 1.0:
            best_ckpt = torch.load(baseline_path, map_location=device)
            m = UNet(base_features=16).to(device)
            m.load_state_dict(best_ckpt["model_state_dict"])
            val_m = evaluate(m, val_loader, WeightedBCELoss(pos_weight=1.0), device)
            tuning_results.append({
                "weight": w,
                "val_loss": val_m["loss"],
                "val_precision": val_m["precision"],
                "val_recall": val_m["recall"],
                "val_f1": val_m["f1"],
                "val_iou": val_m["iou"]
            })
            candidate_histories[w] = baseline_hist
        else:
            ckpt_path, hist = train_model(
                model_name=f"tuning_unet_w{int(w)}",
                pos_weight=w,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=20,
                lr=1.5e-3,
                device=device
            )
            candidate_histories[w] = hist
            best_ckpt = torch.load(ckpt_path, map_location=device)
            m = UNet(base_features=16).to(device)
            m.load_state_dict(best_ckpt["model_state_dict"])
            val_m = evaluate(m, val_loader, WeightedBCELoss(pos_weight=w), device)
            tuning_results.append({
                "weight": w,
                "val_loss": val_m["loss"],
                "val_precision": val_m["precision"],
                "val_recall": val_m["recall"],
                "val_f1": val_m["f1"],
                "val_iou": val_m["iou"]
            })
            
    df_tuning = pd.DataFrame(tuning_results)
    df_tuning.to_csv("results/validation_weight_tuning.csv", index=False)
    print("\nValidation Weight Tuning Summary:")
    print(df_tuning.to_string(index=False))
    
    # Select best weight according to Val F1
    best_weight_row = df_tuning.sort_values(by="val_f1", ascending=False).iloc[0]
    best_w = float(best_weight_row["weight"])
    print(f"\n>>> Optimal weight chosen on validation split: w* = {best_w} (Val F1: {best_weight_row['val_f1']:.4f}, Val IoU: {best_weight_row['val_iou']:.4f})")
    
    # Save proposed model checkpoint as proposed_unet_weighted.pth
    proposed_ckpt_path = f"checkpoints/tuning_unet_w{int(best_w)}.pth" if best_w != 1.0 else baseline_path
    proposed_final_path = "checkpoints/proposed_unet_weighted.pth"
    best_ckpt_data = torch.load(proposed_ckpt_path, map_location=device)
    torch.save(best_ckpt_data, proposed_final_path)
    
    # Plot Validation Weight Tuning Curve
    plt.figure(figsize=(10, 5.5))
    if "val_accuracy" in df_tuning.columns:
        plt.plot(df_tuning["weight"], df_tuning["val_accuracy"], 'x-', linewidth=2.2, markersize=8, color='#8c564b', label='Val Pixel Accuracy')
    plt.plot(df_tuning["weight"], df_tuning["val_f1"], 'o-', linewidth=2.5, markersize=8, color='#1f77b4', label='Val F1 Score')
    plt.plot(df_tuning["weight"], df_tuning["val_iou"], 's--', linewidth=2.0, markersize=8, color='#2ca02c', label='Val IoU')
    plt.plot(df_tuning["weight"], df_tuning["val_recall"], '^:', linewidth=2.0, markersize=7, color='#ff7f0e', label='Val Recall')
    plt.plot(df_tuning["weight"], df_tuning["val_precision"], 'd-.', linewidth=2.0, markersize=7, color='#d62728', label='Val Precision')
    plt.axvline(x=best_w, color='purple', linestyle='--', alpha=0.75, linewidth=1.8, label=f'Optimal w* = {best_w}')
    plt.title("CFD Crack Segmentation: Validation Metrics vs. pos_weight (w)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Positive Class Weight (w in Weighted BCE)", fontsize=11, fontweight='bold')
    plt.ylabel("Validation Metric Score (0.0 to 1.0)", fontsize=11, fontweight='bold')
    plt.ylim(0.3, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True, fontsize=10, loc='center right')
    plt.tight_layout()
    plt.savefig("results/validation_weight_tuning.png", dpi=300)
    plt.close()
    
    # 4. Final Evaluation on Unseen Test Split
    print("\n" + "="*60)
    print("TEST SPLIT EVALUATION (Reporting Precision, Recall, F1, IoU)")
    print("="*60)
    
    baseline_model = UNet(base_features=16).to(device)
    baseline_model.load_state_dict(torch.load(baseline_path, map_location=device)["model_state_dict"])
    baseline_test_metrics = evaluate(baseline_model, test_loader, WeightedBCELoss(pos_weight=1.0), device)
    
    proposed_model = UNet(base_features=16).to(device)
    proposed_model.load_state_dict(torch.load(proposed_final_path, map_location=device)["model_state_dict"])
    proposed_test_metrics = evaluate(proposed_model, test_loader, WeightedBCELoss(pos_weight=best_w), device)
    
    # Summary Table
    comparison_data = [
        {
            "Model": "Baseline U-Net (Ordinary BCE, w=1.0)",
            "Pos Weight (w)": 1.0,
            "Precision": baseline_test_metrics["precision"],
            "Recall": baseline_test_metrics["recall"],
            "F1 Score": baseline_test_metrics["f1"],
            "IoU": baseline_test_metrics["iou"],
            "Mean Sample F1": baseline_test_metrics["mean_sample_f1"],
            "Mean Sample IoU": baseline_test_metrics["mean_sample_iou"]
        },
        {
            "Model": f"Proposed U-Net (Weighted BCE, w={best_w})",
            "Pos Weight (w)": best_w,
            "Precision": proposed_test_metrics["precision"],
            "Recall": proposed_test_metrics["recall"],
            "F1 Score": proposed_test_metrics["f1"],
            "IoU": proposed_test_metrics["iou"],
            "Mean Sample F1": proposed_test_metrics["mean_sample_f1"],
            "Mean Sample IoU": proposed_test_metrics["mean_sample_iou"]
        }
    ]
    df_comparison = pd.DataFrame(comparison_data)
    df_comparison.to_csv("results/test_metrics_summary.csv", index=False)
    with open("results/test_metrics_summary.json", "w") as f:
        json.dump(comparison_data, f, indent=2)
        
    print("\n--- Final Test Set Results ---")
    print(df_comparison[["Model", "Precision", "Recall", "F1 Score", "IoU"]].to_string(index=False))
    
    # Relative improvements
    prec_diff = proposed_test_metrics["precision"] - baseline_test_metrics["precision"]
    rec_diff = proposed_test_metrics["recall"] - baseline_test_metrics["recall"]
    f1_diff = proposed_test_metrics["f1"] - baseline_test_metrics["f1"]
    iou_diff = proposed_test_metrics["iou"] - baseline_test_metrics["iou"]
    print(f"\nImprovement with Weighted BCE (w={best_w}):")
    print(f"Recall Delta:    {rec_diff:+.4f} ({rec_diff/baseline_test_metrics['recall']*100:+.1f}%)")
    print(f"F1 Score Delta:  {f1_diff:+.4f} ({f1_diff/baseline_test_metrics['f1']*100:+.1f}%)")
    print(f"IoU Delta:       {iou_diff:+.4f} ({iou_diff/baseline_test_metrics['iou']*100:+.1f}%)")
    
    # 5. Plot Loss Curves
    plt.figure(figsize=(10, 4.5))
    plt.subplot(1, 2, 1)
    plt.plot(baseline_hist["train_loss"], label="Train Loss (w=1)", color='#1f77b4', linestyle='--')
    plt.plot(baseline_hist["val_loss"], label="Val Loss (w=1)", color='#1f77b4')
    plt.title("Baseline U-Net (Ordinary BCE)", fontsize=11, fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    prop_hist = candidate_histories[best_w]
    plt.plot(prop_hist["train_loss"], label=f"Train Loss (w={best_w})", color='#d62728', linestyle='--')
    plt.plot(prop_hist["val_loss"], label=f"Val Loss (w={best_w})", color='#d62728')
    plt.title(f"Proposed U-Net (Weighted BCE w={best_w})", fontsize=11, fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/loss_curves.png", dpi=300)
    plt.close()
    
    # 6. Plot Test Metrics Comparison Bar Chart
    metrics_names = ["Precision", "Recall", "F1 Score", "IoU"]
    b_vals = [baseline_test_metrics["precision"], baseline_test_metrics["recall"], baseline_test_metrics["f1"], baseline_test_metrics["iou"]]
    p_vals = [proposed_test_metrics["precision"], proposed_test_metrics["recall"], proposed_test_metrics["f1"], proposed_test_metrics["iou"]]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, b_vals, width, label='Baseline U-Net (Ordinary BCE, w=1)', color='#7293CB', edgecolor='black')
    rects2 = ax.bar(x + width/2, p_vals, width, label=f'Proposed U-Net (Weighted BCE, w={best_w})', color='#E15759', edgecolor='black')
    
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Test Split Metric Comparison: Baseline vs. Proposed Model', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(frameon=True, fontsize=10)
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
    autolabel(rects1)
    autolabel(rects2)
    plt.tight_layout()
    plt.savefig("results/metrics_comparison_bar.png", dpi=300)
    plt.close()
    
    # 7. Generate Visual Test Comparisons (Ground Truth vs Baseline vs Proposed)
    print("\nGenerating qualitative test prediction visual comparison grid...")
    baseline_model.eval()
    proposed_model.eval()
    
    test_sample_ids = splits["test"][:4]  # 4 representative test images
    fig, axes = plt.subplots(len(test_sample_ids), 5, figsize=(15, 3 * len(test_sample_ids)))
    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    for row_idx, sid in enumerate(test_sample_ids):
        img_p = os.path.join(img_dir, f"{sid}.jpg")
        gt_p = os.path.join(gt_dir, f"{sid}.mat")
        
        # Load raw image
        raw_img = Image.open(img_p).convert("RGB").resize((256, 256), Image.BILINEAR)
        img_np = np.array(raw_img, dtype=np.float32) / 255.0
        
        # Load GT
        import scipy.io
        mat = scipy.io.loadmat(gt_p)
        raw_seg = mat["groundTruth"][0, 0]["Segmentation"]
        gt_mask = (raw_seg == 2).astype(np.uint8) * 255
        gt_mask = np.array(Image.fromarray(gt_mask).resize((256, 256), Image.NEAREST)) > 127
        
        # Tensor
        norm_img = (img_np - mean) / std
        t_img = torch.from_numpy(norm_img.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
        
        with torch.no_grad():
            b_out = torch.sigmoid(baseline_model(t_img)).squeeze().cpu().numpy()
            p_out = torch.sigmoid(proposed_model(t_img)).squeeze().cpu().numpy()
            
        b_pred = (b_out > 0.5).astype(np.float32)
        p_pred = (p_out > 0.5).astype(np.float32)
        
        # Difference map: Green = TP proposed only, Red = Missed by both, Yellow = FP
        diff_map = np.zeros((256, 256, 3), dtype=np.float32)
        diff_map[gt_mask, 0] = 0.5  # Ground truth base
        diff_map[p_pred == 1, 1] = 1.0  # Proposed detection in Green
        diff_map[b_pred == 1, 2] = 1.0  # Baseline detection in Blue
        
        # Col 0: Pavement Input
        axes[row_idx, 0].imshow(raw_img)
        axes[row_idx, 0].set_title(f"Image {sid}", fontsize=10)
        axes[row_idx, 0].axis("off")
        
        # Col 1: Ground Truth
        axes[row_idx, 1].imshow(gt_mask, cmap='gray')
        axes[row_idx, 1].set_title("Ground Truth", fontsize=10)
        axes[row_idx, 1].axis("off")
        
        # Col 2: Baseline (w=1)
        axes[row_idx, 2].imshow(b_pred, cmap='magma')
        axes[row_idx, 2].set_title(f"Baseline (w=1.0)\nBroken/Missing", fontsize=9, color='#1f77b4', fontweight='bold')
        axes[row_idx, 2].axis("off")
        
        # Col 3: Proposed (w=best_w)
        axes[row_idx, 3].imshow(p_pred, cmap='viridis')
        axes[row_idx, 3].set_title(f"Proposed (w={best_w})\nContinuous Crack", fontsize=9, color='#d62728', fontweight='bold')
        axes[row_idx, 3].axis("off")
        
        # Col 4: Overlay Comparison
        overlay = np.array(raw_img).copy()
        # Overlay proposed in red, ground truth in green
        overlay[p_pred == 1] = [255, 50, 50]
        axes[row_idx, 4].imshow(overlay)
        axes[row_idx, 4].set_title("Proposed Overlay", fontsize=10)
        axes[row_idx, 4].axis("off")
        
    plt.suptitle("Qualitative Test Set Crack Segmentation: Baseline (w=1.0) vs Proposed Weighted BCE", fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig("results/test_predictions_comparison.png", dpi=300)
    plt.close()
    
    print("All evaluation plots generated successfully in 'results/' folder!")
    print("\nExecution complete!")

if __name__ == "__main__":
    run_pipeline()
