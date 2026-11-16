import numpy as np
import torch

def compute_batch_metrics(pred_probs, targets, threshold=0.5, eps=1e-7):
    """
    Computes segmentation evaluation metrics on a batch:
      - Precision: TP / (TP + FP + eps)
      - Recall:    TP / (TP + FN + eps)
      - F1 Score:  2 * Precision * Recall / (Precision + Recall + eps)
      - IoU:       TP / (TP + FP + FN + eps)
      
    Args:
        pred_probs: Tensor or ndarray of predicted probabilities in [0, 1]
        targets: Tensor or ndarray of binary ground truth masks in {0, 1}
        threshold: Binarization threshold (default: 0.5)
        eps: Small epsilon to prevent division by zero
    """
    if isinstance(pred_probs, torch.Tensor):
        preds = (pred_probs > threshold).float()
        targets = targets.float()
        
        tp = (preds * targets).sum().item()
        fp = (preds * (1.0 - targets)).sum().item()
        fn = ((1.0 - preds) * targets).sum().item()
        tn = ((1.0 - preds) * (1.0 - targets)).sum().item()
    else:
        preds = (pred_probs > threshold).astype(np.float32)
        targets = targets.astype(np.float32)
        
        tp = np.sum(preds * targets)
        fp = np.sum(preds * (1.0 - targets))
        fn = np.sum((1.0 - preds) * targets)
        tn = np.sum((1.0 - preds) * (1.0 - targets))

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = (2.0 * precision * recall) / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn)
    }


class MetricTracker:
    """
    Accumulates TP, FP, FN, TN across an entire evaluation split
    to compute global dataset-level Precision, Recall, F1, and IoU.
    """
    def __init__(self, threshold=0.5, eps=1e-7):
        self.threshold = threshold
        self.eps = eps
        self.reset()

    def reset(self):
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        self.total_tn = 0
        self.sample_f1s = []
        self.sample_ious = []

    def update(self, pred_probs, targets):
        if isinstance(pred_probs, torch.Tensor):
            preds = (pred_probs > self.threshold).float()
            targets = targets.float()
            
            # Batch totals
            tp = (preds * targets).sum().item()
            fp = (preds * (1.0 - targets)).sum().item()
            fn = ((1.0 - preds) * targets).sum().item()
            tn = ((1.0 - preds) * (1.0 - targets)).sum().item()
            
            # Per-sample metrics
            batch_size = preds.shape[0]
            for b in range(batch_size):
                p_b = preds[b].view(-1)
                t_b = targets[b].view(-1)
                s_tp = (p_b * t_b).sum().item()
                s_fp = (p_b * (1.0 - t_b)).sum().item()
                s_fn = ((1.0 - p_b) * t_b).sum().item()
                s_prec = s_tp / (s_tp + s_fp + self.eps)
                s_rec = s_tp / (s_tp + s_fn + self.eps)
                s_f1 = (2.0 * s_prec * s_rec) / (s_prec + s_rec + self.eps)
                s_iou = s_tp / (s_tp + s_fp + s_fn + self.eps)
                self.sample_f1s.append(s_f1)
                self.sample_ious.append(s_iou)
        else:
            preds = (pred_probs > self.threshold).astype(np.float32)
            targets = targets.astype(np.float32)
            tp = np.sum(preds * targets)
            fp = np.sum(preds * (1.0 - targets))
            fn = np.sum((1.0 - preds) * targets)
            tn = np.sum((1.0 - preds) * (1.0 - targets))

        self.total_tp += tp
        self.total_fp += fp
        self.total_fn += fn
        self.total_tn += tn

    def compute(self):
        total_pixels = self.total_tp + self.total_tn + self.total_fp + self.total_fn + self.eps
        accuracy = (self.total_tp + self.total_tn) / total_pixels
        prec = self.total_tp / (self.total_tp + self.total_fp + self.eps)
        rec = self.total_tp / (self.total_tp + self.total_fn + self.eps)
        f1 = (2.0 * prec * rec) / (prec + rec + self.eps)
        iou = self.total_tp / (self.total_tp + self.total_fp + self.total_fn + self.eps)
        
        mean_sample_f1 = np.mean(self.sample_f1s) if self.sample_f1s else f1
        mean_sample_iou = np.mean(self.sample_ious) if self.sample_ious else iou

        return {
            "accuracy": float(accuracy),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "iou": float(iou),
            "mean_sample_f1": float(mean_sample_f1),
            "mean_sample_iou": float(mean_sample_iou),
            "tp": int(self.total_tp),
            "fp": int(self.total_fp),
            "fn": int(self.total_fn),
            "tn": int(self.total_tn)
        }
