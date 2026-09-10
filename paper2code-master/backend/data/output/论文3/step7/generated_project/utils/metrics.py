import torch

__all__ = ["compute_metrics"]


def compute_metrics(
    preds=None,
    targets=None,
    threshold=0.5,
    batch_size=32,
    config=None,
    smooth=1e-6,
):
    """Compute binary segmentation Dice coefficient and IoU.

    Args:
        preds: Tensor or array-like with shape (N, H, W) or (N, 1, H, W).
            Expected to contain predicted probabilities in [0, 1].
        targets: Tensor or array-like with the same spatial shape as preds.
        threshold: Binarization threshold for predicted probabilities.
        batch_size: Kept for API compatibility; not used inside this function.
        config: Kept for API compatibility; not used inside this function.
        smooth: Smoothing term to avoid division by zero.

    Returns:
        dict with keys ``dice_coefficient`` and ``iou``.
    """
    _ = batch_size
    _ = config

    if preds is None or targets is None:
        return {"dice_coefficient": 0.0, "iou": 0.0}

    if not isinstance(preds, torch.Tensor):
        preds = torch.as_tensor(preds, dtype=torch.float32)
    else:
        preds = preds.float()

    if not isinstance(targets, torch.Tensor):
        targets = torch.as_tensor(targets, dtype=torch.float32)
    else:
        targets = targets.float()

    if preds.device != targets.device:
        preds = preds.to(targets.device)

    if preds.dim() == 4 and preds.size(1) == 1:
        preds = preds.squeeze(1)
    if targets.dim() == 4 and targets.size(1) == 1:
        targets = targets.squeeze(1)

    if preds.dim() == 2 and targets.dim() == 2:
        preds = preds.unsqueeze(0)
        targets = targets.unsqueeze(0)

    if preds.shape != targets.shape:
        raise ValueError(
            f"preds shape {tuple(preds.shape)} must match "
            f"targets shape {tuple(targets.shape)}"
        )

    if preds.dim() < 3 or preds.dim() > 4:
        raise ValueError(
            "Expected preds to have shape (N, H, W) or (N, 1, H, W) "
            f"after squeezing channel dim, got {preds.dim()}D."
        )

    preds_bin = (preds > threshold).float()

    dice_scores = []
    iou_scores = []

    with torch.no_grad():
        for pred, target in zip(preds_bin, targets):
            pred_flat = pred.reshape(-1)
            target_flat = target.reshape(-1)

            intersection = float((pred_flat * target_flat).sum().item())
            pred_sum = float(pred_flat.sum().item())
            target_sum = float(target_flat.sum().item())
            union = pred_sum + target_sum - intersection

            if pred_sum == 0.0 and target_sum == 0.0:
                dice_scores.append(1.0)
                iou_scores.append(1.0)
            else:
                dice = (2.0 * intersection + smooth) / (
                    pred_sum + target_sum + smooth
                )
                iou = (intersection + smooth) / (union + smooth)
                dice_scores.append(dice)
                iou_scores.append(iou)

    if dice_scores:
        dice = sum(dice_scores) / len(dice_scores)
        iou = sum(iou_scores) / len(iou_scores)
    else:
        dice = 0.0
        iou = 0.0

    return {
        "dice_coefficient": float(dice),
        "iou": float(iou),
    }