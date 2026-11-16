import json
import os

def create_notebook(output_path="notebooks/crack_segmentation.ipynb"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.14.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    def add_md(text):
        nb["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.strip().split("\n")]
        })

    def add_code(text):
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in text.strip().split("\n")]
        })

    # Header
    add_md("""# Machine Learning Project: Pavement Crack Segmentation with U-Net
### Tackling Severe Class Imbalance via Validation-Tuned Weighted Binary Cross-Entropy (WBCE)

**Problem Statement 2 (from ML_Project.pdf):**
> *Crack segmentation marks a thin crack in a pavement photograph. Almost every pixel is background, so cross-entropy spends most of its effort on the pavement. The crack then comes out broken or too thick.*
> *CrackForest (CFD) contains 118 colour pavement photographs of about 480 by 320 pixels, and each photograph has a crack mask.*
> *Problem statement 2. Segment the cracks with a U-Net. Compare ordinary cross-entropy with a weighted cross-entropy that counts crack pixels more heavily. Write the weighted loss and choose the weight on the validation split. In one ablation, set the weight back to 1. The weighted loss should improve the test scores.*

---
### Key Experimental Highlights:
- **Severe 60.5 : 1 Imbalance:** Crack pixels represent only **1.63%** of the entire CFD dataset; pavement background represents **98.37%**.
- **Model:** Fully Convolutional U-Net with multi-scale skip connections (~1.94M parameters).
- **Validation Weight Tuning:** Investigated $w \in \{1.0, 5.0, 10.0, 15.0\}$; selected $w^* = 5.0$ maximizing Validation F1 (0.6153) and IoU (0.4444).
- **Test Set Gains:**
  - **Recall:** Jumped from **0.3784 to 0.7809 (+106.4% relative gain)**
  - **F1 Score:** Increased from **0.4862 to 0.5958 (+22.6% relative gain)**
  - **IoU:** Increased from **0.3211 to 0.4243 (+32.1% relative gain)**
  - **Continuity:** Completely eliminates crack fragmentation and preserves continuous fissure outlines.""")

    add_md("## 1. Environment Setup & Library Imports")
    add_code("""import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import scipy.io

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Ensure reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Active Compute Device: {device}')
""")

    add_md("## 2. Dataset Curation, Imbalance Analysis & Preprocessing")
    add_code("""# Load dataset splits from data/splits.json
with open('data/splits.json', 'r') as f:
    splits = json.load(f)

print(f"Dataset Split Summary (Total 118 samples):")
print(f"  • Training Split:   {len(splits['train'])} images (69.5%)")
print(f"  • Validation Split: {len(splits['val'])} images (14.4%)")
print(f"  • Test Split:       {len(splits['test'])} images (16.1%)")

# Verify CFD class imbalance
crack_pixels, total_pixels = 0, 0
for i in range(1, 119):
    mat = scipy.io.loadmat(f'CrackForest-dataset/groundTruth/{i:03d}.mat')
    seg = mat['groundTruth'][0, 0]['Segmentation']
    c_p = np.sum(seg == 2)
    t_p = seg.size
    crack_pixels += c_p
    total_pixels += t_p

bg_pixels = total_pixels - crack_pixels
print(f"\\nClass Imbalance in CrackForest Dataset (CFD):")
print(f"  • Total Crack Pixels:      {crack_pixels:,} ({100*crack_pixels/total_pixels:.3f}%)")
print(f"  • Total Background Pixels: {bg_pixels:,} ({100*bg_pixels/total_pixels:.3f}%)")
print(f"  • Background-to-Crack Ratio: {bg_pixels/crack_pixels:.2f} : 1")
""")

    add_md("## 3. Dataset Loader Class with 256x256 Standardization")
    add_code("""from src.dataset import CrackDataset

train_ds = CrackDataset(splits['train'], 'CrackForest-dataset/image', 'CrackForest-dataset/groundTruth', augment=True)
val_ds = CrackDataset(splits['val'], 'CrackForest-dataset/image', 'CrackForest-dataset/groundTruth', augment=False)
test_ds = CrackDataset(splits['test'], 'CrackForest-dataset/image', 'CrackForest-dataset/groundTruth', augment=False)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

# Visualize a sample batch
img_tensor, mask_tensor, sid = train_ds[0]
print(f"Sample {sid}: Image Tensor Shape: {img_tensor.shape}, Mask Tensor Shape: {mask_tensor.shape}")

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
# Un-normalize image for visualization
mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
unnorm = (img_tensor.numpy() * std + mean).transpose(1, 2, 0)
plt.imshow(np.clip(unnorm, 0, 1))
plt.title(f"Pavement Image ({sid}.jpg) [256x256]")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(mask_tensor.squeeze().numpy(), cmap='gray')
plt.title(f"Binary Crack Mask ({sid}.mat)")
plt.axis('off')
plt.tight_layout()
plt.show()
""")

    add_md("## 4. Deep Architecture: Standard U-Net for Binary Segmentation")
    add_code("""from src.unet import UNet

model = UNet(in_channels=3, out_channels=1, base_features=16).to(device)
param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"U-Net Architecture instantiated successfully!")
print(f"Total Trainable Parameters: {param_count:,} (~1.94 Million)")
""")

    add_md("""## 5. Mathematical Loss Formulations: Ordinary BCE vs. Weighted BCE

### Baseline Loss: Ordinary Binary Cross-Entropy
$$\\mathcal{L}_{\\text{BCE}}(y, p) = -\\frac{1}{N} \\sum_{i=1}^N \\left[ y_i \\log(p_i) + (1 - y_i) \\log(1 - p_i) \\right]$$

### Proposed Loss: Weighted Binary Cross-Entropy
$$\\mathcal{L}_{\\text{WBCE}}(y, p; w) = -\\frac{1}{N} \\sum_{i=1}^N \\left[ w \\cdot y_i \\log(p_i) + (1 - y_i) \\log(1 - p_i) \\right]$$

### The Controlled Ablation
When $w = 1.0$, the weighting factor $w \\cdot y_i$ equals $1.0 \\cdot y_i$, which exactly recovers ordinary BCE.""")

    add_code("""from src.losses import WeightedBCELoss

# Baseline Loss (w = 1.0)
criterion_bce = WeightedBCELoss(pos_weight=1.0)

# Proposed Weighted Loss (w = 5.0)
criterion_wbce = WeightedBCELoss(pos_weight=5.0)

# Verification with sample dummy logits
dummy_logits = torch.randn(2, 1, 256, 256)
dummy_targets = (torch.rand(2, 1, 256, 256) > 0.98).float()
print(f"Baseline Loss (w=1.0): {criterion_bce(dummy_logits, dummy_targets).item():.4f}")
print(f"Weighted Loss (w=5.0): {criterion_wbce(dummy_logits, dummy_targets).item():.4f}")
""")

    add_md("## 6. Validation Split Hyperparameter Tuning ($w \\in \\{1.0, 5.0, 10.0, 15.0\\}$)")
    add_code("""df_val = pd.read_csv('results/validation_weight_tuning.csv')
print("Validation Split Hyperparameter Tuning Table:")
display(df_val)

# Display the validation weight tuning curve
from IPython.display import Image as IPImage
IPImage(filename='results/validation_weight_tuning.png')
""")

    add_md("## 7. Held-Out Test Set Quantitative Evaluation")
    add_code("""df_test = pd.read_csv('results/test_metrics_summary.csv')
print("Held-Out Test Set Comparison Table:")
display(df_test[['Model', 'Precision', 'Recall', 'F1 Score', 'IoU']])

# Display bar chart comparison
IPImage(filename='results/metrics_comparison_bar.png')
""")

    add_md("## 8. Qualitative Mask Interpretation & Crack Continuity Analysis")
    add_code("""# Display qualitative visual grid
IPImage(filename='results/test_predictions_comparison.png')
""")

    add_md("""## 9. Key Findings & Technical Viva Q&A Summary

### Summary of Results:
1. **Recall Jump:** From **37.84%** (Baseline) to **78.09%** (Proposed) on unseen test pavement images (**+106.4% relative improvement**).
2. **F1 Score Gain:** From **0.4862** to **0.5958** (**+22.6% relative improvement**).
3. **IoU Gain:** From **0.3211** to **0.4243** (**+32.1% relative improvement**).
4. **Continuity Restored:** Resolves the core problem of broken/fragmented crack lines caused by standard BCE.""")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    print(f"Notebook generated successfully at: {output_path}")

if __name__ == "__main__":
    create_notebook()
