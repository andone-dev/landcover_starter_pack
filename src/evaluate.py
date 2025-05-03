import numpy as np


def compute_iou(pred, gt, num_classes=9):
    iou_list = []
    for c in range(num_classes):
        intersection = ((pred == c) & (gt == c)).sum()
        union = ((pred == c) | (gt == c)).sum()
        if union == 0:
            iou_list.append(1.0)  # or 0.0
        else:
            iou_list.append(intersection / union)
    return np.mean(iou_list)


def evaluate_segmentation(preds, gts, num_classes=9):
    ious = []
    for p, g in zip(preds, gts):
        ious.append(compute_iou(p, g, num_classes))
    return np.mean(ious)
