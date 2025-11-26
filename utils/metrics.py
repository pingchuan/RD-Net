import torch
import numpy as np

def evaluate(pred, gt):
    if isinstance(pred, (list, tuple)):
        pred = pred[0]

    pred_binary = pred.round().float()
    pred_binary_inverse = (pred_binary == 0).float()

    gt_binary = gt.round().float()
    gt_binary_inverse = (gt_binary == 0).float()

    TP = pred_binary.mul(gt_binary).sum()
    FP = pred_binary.mul(gt_binary_inverse).sum()
    TN = pred_binary_inverse.mul(gt_binary_inverse).sum()
    FN = pred_binary_inverse.mul(gt_binary).sum()

    if TP.item() == 0:
        TP = torch.tensor(1.0, device=pred.device)
    # Acc
    ACC_overall = (TP + TN) / (TP + FP + FN + TN + 1e-8)


    # IoU
    IoU_poly = TP / (TP + FP + FN + 1e-8)

    # Dice
    size = pred.size(0)
    pred_flat = pred.view(size, -1)
    target_flat = gt.view(size, -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    dice_score = torch.mean((2 * intersection + 1e-8) / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + 1e-8))

    return {
        'ACC_overall': ACC_overall.item(),
        'Dice': dice_score.item(),
        'IoU': IoU_poly.item(),
    }



class Metrics(object):
    def __init__(self, metrics_list):
        self.metrics = dict()
        for metric in metrics_list:
            self.metrics[metric] = list()

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.metrics.keys():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                self.metrics[k].append(v)

    def mean(self):
        mean_metrics = dict()
        for k, v in self.metrics.items():
            mean_metrics[k] = np.mean(v)
        return mean_metrics

    def clean(self):
        for k in self.metrics.keys():
            self.metrics[k].clear()
