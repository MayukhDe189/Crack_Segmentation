# Automated Pavement Crack Segmentation with U-Net: Weighted Binary Cross-Entropy Study

This project implements an end-to-end deep learning framework for **Pavement Crack Segmentation** on the **CrackForest Dataset (CFD)**, addressing **Problem Statement 2** from the Machine Learning Project guidelines.

---

## 📌 Project Overview & Highlights
- **Problem:** Thin cracks in pavement photography suffer from extreme background dominance (over **98.37%** of pixels are pavement, an extreme **60.5 : 1 class imbalance**). Standard unweighted Binary Cross-Entropy (BCE) spends 98% of its gradient signal on pavement, causing cracks to appear broken, fragmented, or missed entirely.
- **Architecture:** Fully Convolutional **U-Net** (4 downsampling & 4 upsampling stages, skip connections, ~1.94M parameters).
- **Proposed Methodology:** **Weighted Binary Cross-Entropy (WBCE)** where positive crack pixels are scaled by weight $w > 1.0$.
- **Validation Tuning:** Evaluated $w \in \{1.0, 5.0, 10.0, 15.0\}$ on the validation split; selected **$w^* = 5.0$** as the optimal harmonic trade-off maximizing F1 and IoU.
- **Ablation Study:** In the controlled ablation, setting $w = 1.0$ collapses the loss back to standard BCE, directly proving that the positive weighting mechanism is the decisive factor in restoring crack continuity.
- **Held-Out Test Set Results:**
  - **Recall:** Jumped from **0.3784 to 0.7809 (+106.4% relative gain)**
  - **F1 Score:** Increased from **0.4862 to 0.5958 (+22.6% relative gain)**
  - **IoU (Jaccard):** Increased from **0.3211 to 0.4243 (+32.1% relative gain)**

---

## 📂 Submission Directory Structure (`MLproject_rollno_crack_segmentation/`)

In compliance with the project submission instructions (*"Submit the link to a folder named `MLproject_rollno_topic`. The folder must contain the code, the slides, and the report"*):

```
MLproject_rollno_crack_segmentation/
├── src/                               # Modular source code
│   ├── dataset.py                     # CFD loader, 256x256 resizing, augmentations, RAM caching
│   ├── unet.py                        # U-Net architecture with skip connections (1.94M params)
│   ├── losses.py                      # Ordinary BCE & Weighted BCE with pos_weight w
│   └── metrics.py                     # Precision, Recall, F1 Score, IoU (Jaccard Index)
├── data/
│   └── splits.json                    # Deterministic 70/15/15 train/val/test splits (seed=42)
├── checkpoints/                       # Saved trained PyTorch models
│   ├── baseline_unet_w1.pth           # Baseline U-Net (Ordinary BCE, w=1.0)
│   ├── proposed_unet_weighted.pth     # Proposed U-Net (Weighted BCE, w=5.0)
│   ├── tuning_unet_w5.pth             # Validation checkpoint for w=5.0
│   ├── tuning_unet_w10.pth            # Validation checkpoint for w=10.0
│   └── tuning_unet_w15.pth            # Validation checkpoint for w=15.0
├── results/                           # Evaluation tables and publication-quality figures
│   ├── test_metrics_summary.csv       # Precision, Recall, F1, IoU on held-out test split
│   ├── test_metrics_summary.json      # Structured test metrics
│   ├── validation_weight_tuning.csv   # Validation grid search metrics across weights
│   ├── validation_weight_tuning.png   # Validation curve (F1, IoU, Recall vs w)
│   ├── metrics_comparison_bar.png     # Grouped bar chart comparing baseline vs proposed
│   ├── loss_curves.png                # Training & validation loss learning curves
│   ├── test_predictions_comparison.png# Visual comparison (Image, GT, Baseline, Proposed, Overlay)
│   └── demo_prediction.png            # Sample inference demonstration
├── notebooks/
│   └── crack_segmentation.ipynb       # Interactive Jupyter Notebook walkthrough
├── slides/
│   └── crack_segmentation_presentation.pptx # 10-slide PowerPoint presentation with figures
├── report/
│   ├── Crack_Segmentation_Report.pdf  # Professional 7-page PDF report (<= 8 pages)
│   ├── Crack_Segmentation_Report.md   # Complete Markdown documentation
│   └── generate_report.py             # Script used to compile the PDF report
├── train_and_evaluate.py              # End-to-end training, tuning, and evaluation pipeline
├── demo.py                            # Standalone CLI demo inference on any image
└── README.md                          # Project documentation and reproduction guide
```

---

## 🚀 Quickstart & Usage

### 1. Requirements
Ensure the required packages are installed:
```bash
pip install torch torchvision pillow matplotlib scipy pandas python-pptx reportlab pymupdf
```

### 2. Run Interactive Demo Inference
To run inference on any sample pavement image (e.g., sample `001`, `010`, or `085`):
```bash
python demo.py --sample_id 010 --output results/demo_prediction.png
```
This produces a side-by-side comparison:
`[Original Pavement | Ground Truth Mask | Baseline (w=1.0) | Proposed (w=5.0) | Overlay]`

### 3. Re-train & Re-evaluate Entire Pipeline
To re-run the full training, validation grid search across candidate weights, test evaluation, and figure generation:
```bash
python train_and_evaluate.py
```

### 4. Regenerate Slides or PDF Report
- To regenerate the 10-slide PowerPoint presentation:
  ```bash
  python generate_slides.py
  ```
- To regenerate the PDF report (strictly $\le 8$ pages):
  ```bash
  python report/generate_report.py
  ```

---

## 📊 Summary of Final Test Set Results

| Model Architecture & Loss | Pos Weight ($w$) | Precision | Recall | F1 Score | IoU (Jaccard) |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline U-Net (Ordinary BCE)** | 1.0 (Ablation) | **0.6797** | 0.3784 | 0.4862 | 0.3211 |
| **Proposed U-Net (Weighted BCE)** | 5.0 (Optimal) | 0.4817 | **0.7809** | **0.5958** | **0.4243** |
| **Absolute Delta ($\Delta$)** | +4.0 | -0.1980 | **+0.4025** | **+0.1097** | **+0.1032** |
| **Relative Improvement (%)** | — | -29.1% | **+106.4%** | **+22.6%** | **+32.1%** |

---

## 🎯 Technical Viva Key Questions & Defense

1. **Why did ordinary cross-entropy cause cracks to be broken or disconnected?**  
   *Answer:* CFD has an extreme 60.5 : 1 background-to-crack imbalance. In ordinary BCE, every pixel is weighted equally, so >98% of gradients come from pavement background. To minimize loss, the model learns an overly conservative threshold, finding only thick crack cores and dropping faint lateral branches.
2. **Why not set $w = 60$, matching the exact inverse class frequency?**  
   *Answer:* An extreme weight like $w = 60$ causes the model to over-penalize false negatives to such a degree that rough asphalt textures and shadows are flagged as cracks, collapsing precision and causing bloated masks. Tuning on an independent validation set identified $w^* = 5.0$ as the optimal harmonic trade-off maximizing F1 score.
3. **What is the ablation study?**  
   *Answer:* In one ablation, setting $w = 1.0$ identically recovers ordinary BCE. Comparing $w = 5.0$ with $w = 1.0$ under identical seeds and architecture isolates the exact contribution of the weighted loss.
