import torch.nn as nn

import torch

import random


###


def normalize_features(features):
    norm = features.norm(dim=1, keepdim=True)
    return features / (norm + 1e-8)

def dpa(depth_map, images_s, pred_u, beta, t, T):



    def calculate_variance_hardness(depth_map):

        N, _, H, W = depth_map.size()
        h, w = random.choice([40, 10, 20]), random.choice([40, 10, 20])



        patches = depth_map.unfold(2, h, h).unfold(3, w, w)
        patch_variance = patches.var(dim=(-2, -1))
        hardness_scores = patch_variance.flatten(1)

        return hardness_scores, h, w



    hardness_scores, h, w = calculate_variance_hardness(depth_map)

    N, num_patches = hardness_scores.size()

    if t<T:
       k = int(beta * (t / T) * num_patches)
    else:
        k = int(beta * num_patches)


    _, indices = torch.topk(hardness_scores, k, dim=1, largest=True)

    augmented_imgs = []
    augmented_labels = []

    for i in range(N):
        mask = torch.zeros(num_patches, dtype=torch.float32)
        mask[indices[i]] = 1.0
        available_indices = [j for j in range(num_patches) if mask[j] == 0]
        augmented_img = images_s[i].clone()
        augmented_label = pred_u[i].clone()


        if available_indices:
            chosen_patch = random.choice(available_indices)


            patch_h = chosen_patch // (images_s.size(3) // w)
            patch_w = chosen_patch % (images_s.size(3) // w)
            x_start = patch_w * w
            x_end = x_start + w
            y_start = patch_h * h
            y_end = y_start + h


            next_img_idx = (i + 1) % N
            augmented_img[:, y_start:y_end, x_start:x_end] = images_s[next_img_idx][:, y_start:y_end, x_start:x_end]
            augmented_label[:, y_start:y_end, x_start:x_end] = pred_u[next_img_idx][:, y_start:y_end, x_start:x_end]

        augmented_imgs.append(augmented_img)
        augmented_labels.append(augmented_label)


    augmented_imgs = torch.stack(augmented_imgs)
    augmented_labels = torch.stack(augmented_labels)

    return augmented_imgs, augmented_labels


def dice_coefficient(pred, target, eps=1e-6):
    intersection = (pred * target).sum()
    return 2 * intersection / (pred.sum() + target.sum() + eps)


def build_pairs(pred_u, pred_l, pred_u_d, pred_l_d):

    pred_img = torch.cat((pred_l, pred_u))
    pred_depth = torch.cat((pred_l_d, pred_u_d))

    N = len(pred_img)
    preds_positive = []
    preds_negative = []


    for i in range(N):
        preds_positive.append((pred_img[i], pred_depth[i]))


    for i in range(N):
        neg_samples = []


        for j in range(N):
            if j != i:
                neg_samples.append(pred_img[j])
                neg_samples.append(pred_depth[j])


        preds_negative.extend(neg_samples)

    return preds_positive, preds_negative


def cont_loss(pred_u, pred_l, pred_u_d, pred_l_d):

    preds_positive, preds_negative = build_pairs(pred_u, pred_l, pred_u_d, pred_l_d)

    N = len(preds_positive)
    loss = 0.0

    for pos_pair in preds_positive:

        s_pos = dice_coefficient(pos_pair[0], pos_pair[1])


        neg_dice = sum(torch.exp(dice_coefficient(pos_pair[0], neg)) for neg in preds_negative)


        loss += -torch.log(torch.exp(s_pos) / (torch.exp(s_pos) + neg_dice))

    return loss / N


def mse_loss(pred_1, pred_2):
    loss = torch.mean((pred_1 - pred_2) ** 2)
    return loss






def update_pseudo_labels(ema_pred_u, pred_u_d, gamma):

    d_k = torch.where(ema_pred_u > 0.5, 1 - ema_pred_u, ema_pred_u)
    d_k_prime = torch.where(pred_u_d > 0.5, 1 - pred_u_d, pred_u_d)


    updated_labels = torch.clone(ema_pred_u)
    condition1 = (pred_u_d > gamma) & (d_k_prime < d_k)
    condition2 = (pred_u_d < 1 - gamma) & (d_k_prime < d_k)


    updated_labels[condition1] = pred_u_d[condition1]
    updated_labels[condition2] = pred_u_d[condition2]

    return updated_labels






class BCELoss1(nn.Module):
    def __init__(self, weight=None, reduction='none'):
        super(BCELoss1, self).__init__()
        self.bce_loss = nn.BCELoss(weight=weight, reduction=reduction)

    def forward(self, pred, target):
        size = pred.size(0)
        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)

        loss = self.bce_loss(pred_flat, target_flat)
        return loss

class DiceLoss1(nn.Module):
    def __init__(self):
        super(DiceLoss1, self).__init__()

    def forward(self, pred, target, mask):
        smooth = 1

        size = pred.size(0)

        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)

        intersection = pred_flat * target_flat
        dice_score = (2 * intersection.sum(1) + smooth) / (pred_flat.sum(1) + target_flat.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum() / size

        # 应用掩码
        masked_dice_loss = dice_loss * mask.sum() / (mask.sum() + 1e-8)  # 避免除以零

        return masked_dice_loss


class BCELoss_d(nn.Module):
    def __init__(self, reduction='mean'):
        super(BCELoss_d, self).__init__()
        self.reduction = reduction

    def forward(self, pred, gt):

        pred = torch.clamp(pred, min=1e-4, max=1 - 1e-4)


        loss = - (gt * torch.log(pred) + (1 - gt) * torch.log(1 - pred))

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class BCELoss1_d(nn.Module):
    def __init__(self, reduction='none'):
        super(BCELoss1_d, self).__init__()
        self.bce_loss = BCELoss_d(reduction=reduction)

    def forward(self, pred, target):
        size = pred.size(0)
        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)

        loss = self.bce_loss(pred_flat, target_flat)
        return loss

class BceDiceLoss1_D(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0, threshold=0.7):
        super(BceDiceLoss1_D, self).__init__()
        self.bce = BCELoss1_d(reduction='none')
        self.dice = DiceLoss1()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.threshold = threshold

    def forward(self, pred, target, threshold):

        threshold = threshold

        target = torch.where(target <= 0.0, torch.tensor(0.0+1e-3, device=target.device), target)
        target = torch.where(target >= 1.0, torch.tensor(1.0-1e-3, device=target.device), target)

        size = pred.size(0)

        mask = (target > threshold) | (target < 1 - threshold)
        mask = mask.float()


        bce_loss = self.bce(pred, target)
        mask = mask.view_as(bce_loss)


        bce_loss = bce_loss * mask

        masked_bce_loss = bce_loss.sum() / (mask.sum() + 1e-8)


        dice_loss = self.dice(pred, target, mask)


        loss = self.bce_weight * masked_bce_loss + self.dice_weight * dice_loss

        return loss


class BceDiceLoss1(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0, threshold=0.7):
        super(BceDiceLoss1, self).__init__()
        self.bce = BCELoss1(reduction='none')
        self.dice = DiceLoss1()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.threshold = threshold

    def forward(self, pred, target, threshold):
        size = pred.size(0)

        threshold = threshold

        mask = (target > threshold) | (target < 1-threshold)

        mask = mask.float()


        bce_loss = self.bce(pred, target)
        mask = mask.view_as(bce_loss)


        bce_loss = bce_loss * mask

        masked_bce_loss = bce_loss.sum() / (mask.sum() + 1e-8)


        dice_loss = self.dice(pred, target, mask)


        loss = self.bce_weight * masked_bce_loss + self.dice_weight * dice_loss

        return loss


def L2_loss(shallow_features, deep_features, mask=None):

    loss = (shallow_features - deep_features).pow(2)

    if mask is not None:

        loss = loss * mask


    return loss.mean()


def mse_consistency_loss(pred_F, pred_G):
    loss = torch.mean((pred_F - pred_G) ** 2)
    return loss
