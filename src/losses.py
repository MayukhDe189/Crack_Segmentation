import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross-Entropy Loss for imbalanced segmentation.
    
    Formula:
        L_WBCE(y, p; w) = - (1/N) * sum_{i=1}^N [ w * y_i * log(p_i) + (1 - y_i) * log(1 - p_i) ]
        
    where:
        y_i in {0, 1} is the ground truth mask (1 = crack, 0 = pavement background)
        p_i = sigmoid(z_i) is the predicted probability of pixel i
        w >= 1 is the positive class weight (pos_weight).
        
    When w = 1.0, this exactly reduces to standard Binary Cross-Entropy (BCE).
    When w > 1.0, false negatives (missed crack pixels) incur w times higher penalty,
    preventing the gradient from being overwhelmed by the dominant background class.
    """
    def __init__(self, pos_weight=1.0):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = float(pos_weight)

    def forward(self, logits, targets):
        """
        logits: (B, 1, H, W) raw outputs from U-Net
        targets: (B, 1, H, W) ground truth binary masks
        """
        if self.pos_weight == 1.0:
            return F.binary_cross_entropy_with_logits(logits, targets)
        else:
            weight_tensor = torch.tensor([self.pos_weight], device=logits.device, dtype=logits.dtype)
            return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=weight_tensor)


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for overlap comparison.
    Dice = 2 * |P cap G| / (|P| + |G| + eps)
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        return 1.0 - dice
