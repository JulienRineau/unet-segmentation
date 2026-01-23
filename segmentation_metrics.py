import torch


def compute_per_image_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int,
) -> float:
    if pred.ndim == 3:
        pred = pred.squeeze(0)
    if target.ndim == 3:
        target = target.squeeze(0)
    pred = pred.to(torch.int64).view(-1)
    target = target.to(torch.int64).view(-1)

    if ignore_index is not None:
        valid = target != ignore_index
        pred = pred[valid]
        target = target[valid]

    if pred.numel() == 0:
        return 0.0

    ious = []
    for cls in range(num_classes):
        pred_mask = pred == cls
        target_mask = target == cls
        union = pred_mask | target_mask
        if union.any():
            intersection = pred_mask & target_mask
            ious.append(intersection.sum().float() / union.sum().float())

    if not ious:
        return 0.0
    return torch.stack(ious).mean().item()
