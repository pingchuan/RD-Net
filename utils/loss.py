import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import random

def normalize_features(features):
    norm = features.norm(dim=1, keepdim=True)
    return features / (norm + 1e-8)  # 添加一个小常数避免除以零


def info_nce_loss(features, temperature=0.07):
    """
    InfoNCE loss for contrastive learning.

    Parameters:
    - features: Tensor of shape [batch_size, feature_dim] containing features from modality A.
    - temperature: scaling factor to control sharpness of the distribution.

    Returns:
    - Loss value as a scalar tensor.
    """
    # 获取批次大小和特征维度
    batch_size = features.size(0)
    feature_dim = features.size(1)
    spatial_dim = features.size(2) * features.size(3)  # 高度 * 宽度

    # 展开空间维度
    features = features.view(batch_size, feature_dim, -1)  # [batch_size, feature_dim, H*W]
    features = features.view(batch_size, -1)  # [batch_size, feature_dim*H*W]

    # 特征归一化（可选但通常推荐）
    features = F.normalize(features, dim=1)
    # 计算特征之间的相似度 (点积)
    similarity_matrix = torch.matmul(features, features.T)  # [batch_size, batch_size]

    # 计算对角线上的掩码 (正样本的相似度)
    labels = torch.arange(batch_size, device=features.device)  # [batch_size]

    # 归一化相似度矩阵
    similarity_matrix = similarity_matrix / temperature

    # 创建一个掩码，标记负样本的相似度 (不是对角线元素)
    mask = torch.eye(batch_size, device=features.device, dtype=torch.bool)

    # 用负无穷大填充掩码中的对角线元素
    similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))

    # 计算损失
    loss = F.cross_entropy(similarity_matrix, labels)

    return loss


def adaptive_patch_cutmix(depth_map, images_s, pred_u, beta, t, T):
    """
    自适应 Patch Cutmix，基于深度图计算困难分数并进行增强。

    参数:
    - depth_map: [N, 1, H, W] 深度图
    - images_s: [N, C, H, W] 第一张图像
    - pred_u: [N, C, H, W] 第二张图像
    - beta: 控制 patch 遮挡比例的超参数
    - t: 当前训练 epoch
    - T: 总 epoch

    返回:
    - augmented_imgs: 增强后的图像，形状为 [N-1, C, H, W]
    - augmented_labels: 增强后的标签，形状为 [N-1, C, H, W]
    """

    def calculate_variance_hardness(depth_map):
        """
        计算每个 patch 的方差作为困难分数。
        """
        N, _, H, W = depth_map.size()
        h, w = random.choice([40, 10, 20]), random.choice([40, 10, 20])
        num_patches = (H // h) * (W // w)

        # 计算每个 patch 的方差
        patches = depth_map.unfold(2, h, h).unfold(3, w, w)  # [N, 1, num_patches_h, num_patches_w, h, w]
        patch_variance = patches.var(dim=(-2, -1))  # [N, num_patches_h, num_patches_w]
        hardness_scores = patch_variance.flatten(1)  # [N, num_patches]

        return hardness_scores, h, w

    def calculate_hardness_score(pred):
        """
        计算基于信息熵的困难分数。APA
        """
        N, _, H, W = pred.size()
        h, w = random.choice([40, 10, 20]), random.choice([40, 10, 20])

        # 创建 [batch_size, 2, H, W] 的概率分布
        probs = torch.cat((1 - pred, pred), dim=1)  # [N, 2, H, W]

        # 将图像划分为 patches
        patches = probs.unfold(2, h, h).unfold(3, w, w)  # [N, 2, num_patches_h, num_patches_w, h, w]

        # 计算每个 patch 的信息熵
        entropy = -torch.sum(patches * torch.log(patches + 1e-10), dim=1)  # [N, num_patches_h, num_patches_w, h, w]
        hardness_scores = entropy.mean(dim=(-2, -1))  # [N, num_patches_h, num_patches_w]

        return hardness_scores, h, w

    def calculate_threshold_hardness(depth_map, threshold=0.5):
        """
        计算基于阈值的方法的困难分数。
        """
        N, _, H, W = depth_map.size()
        h, w = random.choice([5, 10, 20]), random.choice([5, 10, 20])
        num_patches = (H // h) * (W // w)

        # 计算每个 patch 是否超过阈值
        patches = depth_map.unfold(2, h, h).unfold(3, w, w)  # [N, 1, num_patches_h, num_patches_w, h, w]
        patch_threshold = (patches.mean(dim=(-2, -1)) < threshold).float()  # [N, num_patches_h, num_patches_w]
        hardness_scores = patch_threshold.flatten(1)  # [N, num_patches]

        return hardness_scores, h, w

    def calculate_edge_complexity_hardness(depth_map):
        """
        计算每个 patch 的边缘复杂度困难分数。
        """
        import random
        N, _, H, W = depth_map.size()
        h, w = random.choice([40, 10, 20]), random.choice([40, 10, 20])
        num_patches = (H // h) * (W // w)
        depth_map = depth_map.float()
        # 计算深度图的梯度（边缘），使用适当的填充
        kernel_x = torch.tensor([[[[-1.0, 1.0, 0.0]]]], device=depth_map.device)  # 水平梯度
        kernel_y = torch.tensor([[[[-1.0], [1.0], [0.0]]]], device=depth_map.device)  # 垂直梯度

        # 使用 padding 确保输出形状与输入形状一致
        depth_dx = F.conv2d(depth_map, kernel_x, stride=1, padding=(0, 1))  # 对于 3x3 卷积，padding=(0,1)
        depth_dy = F.conv2d(depth_map, kernel_y, stride=1, padding=(1, 0))  # 对于 3x3 卷积，padding=(1,0)


        edge_magnitude = torch.sqrt(depth_dx ** 2 + depth_dy ** 2)  # [N, 1, H, W]

        # 计算每个 patch 的边缘强度平均值作为困难分数
        patches = edge_magnitude.unfold(2, h, h).unfold(3, w, w)  # [N, 1, num_patches_h, num_patches_w, h, w]
        patch_edge_strength = patches.mean(dim=(-2, -1))  # [N, num_patches_h, num_patches_w]
        hardness_scores = patch_edge_strength.flatten(1)  # [N, num_patches]

        return hardness_scores, h, w
    # 计算困难分数
    #hardness_scores, h, w = calculate_edge_complexity_hardness(depth_map)
    hardness_scores, h, w = calculate_variance_hardness(depth_map)
    # 计算 shielded patch 数量
    N, num_patches = hardness_scores.size()

    if t<T:
       k = int(beta * (t / T) * num_patches)
    else:
        k = int(beta * num_patches)

    # 按困难分数排序并选择需要遮挡的 patch
    _, indices = torch.topk(hardness_scores, k, dim=1, largest=True)

    augmented_imgs = []
    augmented_labels = []

    for i in range(N):  # 生成 N 对新图像
        mask = torch.zeros(num_patches, dtype=torch.float32)
        mask[indices[i]] = 1.0  # 将高困难 patch 标记为 1，表示遮挡
        available_indices = [j for j in range(num_patches) if mask[j] == 0]
        augmented_img = images_s[i].clone()  # 保留原始图像
        augmented_label = pred_u[i].clone()  # 保留原始伪标签

        # 随机选择一个未遮挡的 patch
        if available_indices:
            chosen_patch = random.choice(available_indices)

            # 计算当前选择的 patch 的位置
            patch_h = chosen_patch // (images_s.size(3) // w)
            patch_w = chosen_patch % (images_s.size(3) // w)
            x_start = patch_w * w
            x_end = x_start + w
            y_start = patch_h * h
            y_end = y_start + h

            # 替换为下一张图像的对应部分
            next_img_idx = (i + 1) % N  # 循环获取图像索引
            augmented_img[:, y_start:y_end, x_start:x_end] = images_s[next_img_idx][:, y_start:y_end, x_start:x_end]
            augmented_label[:, y_start:y_end, x_start:x_end] = pred_u[next_img_idx][:, y_start:y_end, x_start:x_end]

        augmented_imgs.append(augmented_img)
        augmented_labels.append(augmented_label)

        # 将列表转换为张量
    augmented_imgs = torch.stack(augmented_imgs)
    augmented_labels = torch.stack(augmented_labels)

    return augmented_imgs, augmented_labels


def dice_coefficient(pred, target, eps=1e-6):
    intersection = (pred * target).sum()
    return 2 * intersection / (pred.sum() + target.sum() + eps)


def build_pairs(pred_u, pred_l, pred_u_d, pred_l_d):
    # 拼接预测结果
    pred_img = torch.cat((pred_l, pred_u))  # RGB 网络的预测
    pred_depth = torch.cat((pred_l_d, pred_u_d))  # 深度网络的预测

    N = len(pred_img)  # 假设 pred_l 和 pred_u 的数量相同
    preds_positive = []
    preds_negative = []

    # 构建正样本对
    for i in range(N):
        preds_positive.append((pred_img[i], pred_depth[i]))  # RGB 和 Depth 网络的正样本对

    # 生成负样本集合
    for i in range(N):
        neg_samples = []

        # 选择当前正样本以外的所有样本
        for j in range(N):
            if j != i:
                neg_samples.append(pred_img[j])
                neg_samples.append(pred_depth[j])


        preds_negative.extend(neg_samples)  # 添加到负样本集合

    return preds_positive, preds_negative


def asc_loss(pred_u, pred_l, pred_u_d, pred_l_d):
    # 构建正负样本对
    preds_positive, preds_negative = build_pairs(pred_u, pred_l, pred_u_d, pred_l_d)

    N = len(preds_positive)  # 正样本对的数量
    loss = 0.0

    for pos_pair in preds_positive:
        # 计算 Dice 相似度
        s_pos = dice_coefficient(pos_pair[0], pos_pair[1])

        # 计算负样本的 Dice 相似度
        neg_dice = sum(torch.exp(dice_coefficient(pos_pair[0], neg)) for neg in preds_negative)

        # ASC 损失
        loss += -torch.log(torch.exp(s_pos) / (torch.exp(s_pos) + neg_dice))

    return loss / N


def mse_consistency_loss(pred_F, pred_G):
    """
    计算 MSE 一致性损失。

    参数:
    pred_F: 第一个模型的预测结果
    pred_G: 第二个模型的预测结果

    返回:
    损失值
    """
    # 计算均方误差
    loss = torch.mean((pred_F - pred_G) ** 2)
    return loss

class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target):
        valid_mask = (target > 0).detach()
        diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
                          self.lambd * torch.pow(diff_log.mean(), 2))

        return loss
class FinalConLoss(torch.nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        """
        Contrastive Learning for Unpaired Image-to-Image Translation
        models/patchnce.py
        """
        super(FinalConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.nce_includes_all_negatives_from_minibatch = False
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.mask_dtype = torch.bool

    def forward(self, feat_q, feat_k):
        assert feat_q.size() == feat_k.size(), (feat_q.size(), feat_k.size())
        batch_size = feat_q.shape[0]
        dim = feat_q.shape[1]
        width = feat_q.shape[2]
        feat_q = feat_q.view(batch_size, dim, -1).permute(0, 2, 1)
        feat_k = feat_k.view(batch_size, dim, -1).permute(0, 2, 1)
        feat_q = F.normalize(feat_q, dim=-1, p=1)
        feat_k = F.normalize(feat_k, dim=-1, p=1)
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(feat_q.reshape(-1, 1, dim), feat_k.reshape(-1, dim, 1))
        l_pos = l_pos.view(-1, 1)

        # neg logit
        if self.nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = batch_size

        # reshape features to batch size
        feat_q = feat_q.reshape(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.reshape(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]

        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.temperature

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        return loss


class FinalConsistLoss(nn.Module):
    def __init__(self):
        super(FinalConsistLoss, self).__init__()

    def forward(self, patch_outputs, output):
        bs = output.shape[0]
        cls = output.shape[1]
        psz = patch_outputs.shape[-1]
        cn = output.shape[-1] // psz

        patch_outputs = patch_outputs.reshape(bs, cn, cn, cls, psz, psz)
        output = output.reshape(bs, cls, cn, psz, cn, psz).permute(0, 2, 4, 1, 3, 5)

        p_output_soft = torch.sigmoid(patch_outputs)
        outputs_soft = torch.sigmoid(output)

        loss = torch.mean((p_output_soft - outputs_soft) ** 2, dim=(0, 3, 4, 5)).sum()

        return loss


"""BCE loss"""
def update_pseudo_labels(ema_pred_u, pred_u_d, gamma):
    # 计算 d_k 和 d_k'
    d_k = torch.where(ema_pred_u > 0.5, 1 - ema_pred_u, ema_pred_u)  # 对应 d_k 计算
    d_k_prime = torch.where(pred_u_d > 0.5, 1 - pred_u_d, pred_u_d)  # 对应 d_k' 计算

    # 更新伪标签
    updated_labels = torch.clone(ema_pred_u)  # 创建一个副本以更新伪标签
    condition1 = (pred_u_d > gamma) & (d_k_prime < d_k)  # 第一条件
    condition2 = (pred_u_d < 1 - gamma) & (d_k_prime < d_k)  # 第二条件

    # 应用条件更新
    updated_labels[condition1] = pred_u_d[condition1]
    updated_labels[condition2] = pred_u_d[condition2]

    return updated_labels

class BCELoss(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super(BCELoss, self).__init__()
        self.bce_loss = nn.BCELoss(weight=weight, reduction=reduction)

    def forward(self, pred, target):
        size = pred.size(0)
        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)

        loss = self.bce_loss(pred_flat, target_flat)

        return loss


"""Dice loss"""


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1

        size = pred.size(0)

        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)

        intersection = pred_flat * target_flat
        dice_score = (2 * intersection.sum(1) + smooth) / (pred_flat.sum(1) + target_flat.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum() / size

        return dice_loss


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
        # 确保输入值在 [0, 1] 之间
        pred = torch.clamp(pred, min=1e-4, max=1 - 1e-4)  # 避免log(0)的情况

        # 计算二分类交叉熵损失
        loss = - (gt * torch.log(pred) + (1 - gt) * torch.log(1 - pred))

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class BCELoss1_d(nn.Module):
    def __init__(self, weight=None, reduction='none'):
        super(BCELoss1_d, self).__init__()
        self.bce_loss =BCELoss_d(reduction=reduction)

    def forward(self, pred, target):
        size = pred.size(0)
        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)

        loss = self.bce_loss(pred_flat, target_flat)
        return loss

class BceDiceLoss1_D(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0, threshold=0.7):
        super(BceDiceLoss1_D, self).__init__()
        self.bce = BCELoss1_d(reduction='none')  # 使用 'none' 以便获取每个像素的损失
        self.dice = DiceLoss1()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.threshold = threshold

    def forward(self, pred, target,threshold):
        # 将 target 的值限制在 [0, 1] 的范围内
        threshold = threshold
        #target = torch.clamp(target, min=0.0, max=1.0)
        target = torch.where(target <= 0.0, torch.tensor(0.0+1e-3, device=target.device), target)
        target = torch.where(target >= 1.0, torch.tensor(1.0-1e-3, device=target.device), target)

        size = pred.size(0)
        # 应用阈值掩码
        mask = (target > threshold) | (target < 1 - threshold)
        mask = mask.float()

        # 计算 BCE 损失
        bce_loss = self.bce(pred, target)
        mask = mask.view_as(bce_loss)

        # 使用掩码选择要计算损失的像素点
        bce_loss = bce_loss * mask

        masked_bce_loss = bce_loss.sum() / (mask.sum() + 1e-8)  # 避免除以零

        # 计算 Dice 损失
        dice_loss = self.dice(pred, target, mask)

        # 计算总损失
        loss = self.bce_weight * masked_bce_loss + self.dice_weight * dice_loss

        return loss


class BceDiceLoss1(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0, threshold=0.7):
        super(BceDiceLoss1, self).__init__()
        self.bce = BCELoss1(reduction='none')  # 使用 'none' 以便获取每个像素的损失
        self.dice = DiceLoss1()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.threshold = threshold

    def forward(self, pred, target, threshold):
        size = pred.size(0)
        # 应用阈值掩码
        threshold = threshold
        #mask = target > threshold
        mask = (target > threshold) | (target < 1-threshold)
        #mask = (target > threshold) | (target < 0.5)
        mask = mask.float()

        # 计算 BCE 损失
        bce_loss = self.bce(pred, target)
        mask = mask.view_as(bce_loss)

        # 应用阈值掩码

        # 使用掩码选择要计算损失的像素点
        bce_loss = bce_loss * mask

        masked_bce_loss = bce_loss.sum() / (mask.sum() + 1e-8)  # 避免除以零

        # 计算 Dice 损失
        dice_loss = self.dice(pred, target, mask)

        # 计算总损失
        loss = self.bce_weight * masked_bce_loss + self.dice_weight * dice_loss

        return loss

class BceDiceLoss2(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0, threshold=0.7):
        super(BceDiceLoss2, self).__init__()
        self.bce = BCELoss1(reduction='none')  # 使用 'none' 以便获取每个像素的损失
        self.dice = DiceLoss1()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.threshold = threshold

    def forward(self, pred, target, mask):
        size = pred.size(0)
        # 应用阈值掩码

        mask = mask
        #mask = (target > threshold) | (target < 0.5)
        mask = mask.float()

        # 计算 BCE 损失
        bce_loss = self.bce(pred, target)
        mask = mask.view_as(bce_loss)

        # 应用阈值掩码

        # 使用掩码选择要计算损失的像素点
        bce_loss = bce_loss * mask

        masked_bce_loss = bce_loss.sum() / (mask.sum() + 1e-8)  # 避免除以零

        # 计算 Dice 损失
        dice_loss = self.dice(pred, target, mask)

        # 计算总损失
        loss = self.bce_weight * masked_bce_loss + self.dice_weight * dice_loss

        return loss

def generate_pseudo_label(ema_pred_u, pred_u_d, threshold):
    # 条件1: ema_pred_u > 阈值 且 ema_pred_u > pred_u_d
    #mask1 = (ema_pred_u > threshold) & (ema_pred_u > pred_u_d)

    # 条件2: pred_u_d > 阈值 且 pred_u_d > ema_pred_u
    mask2 = (pred_u_d > threshold) & (pred_u_d > ema_pred_u) & (pred_u_d < 1)

    # 条件3: ema_pred_u < 1 - 阈值 且 ema_pred_u < pred_u_d
    #mask3 = (ema_pred_u < (1 - threshold)) & (ema_pred_u < pred_u_d)

    # 条件4: pred_u_d < 1 - 阈值 且 pred_u_d < ema_pred_u
    mask4 = (pred_u_d < (1 - threshold)) & (pred_u_d < ema_pred_u) & (pred_u_d > 0)
    #mask_u_rgbd = mask1 | mask2 | mask3 | mask4
    # 生成伪标签 pred_u_rgbd, 根据保留的掩码赋值
    pred_u_rgbd = ema_pred_u

    # 应用 mask1 和 mask2

    pred_u_rgbd[mask2] = pred_u_d[mask2]

    # 应用 mask3 和 mask4

    pred_u_rgbd[mask4] = pred_u_d[mask4]
    pred_u_rgbd = torch.clamp(pred_u_rgbd, min=1e-6, max=1 - 1e-6)  # 限制输出在 (1e-6, 1-1e-6) 范围内
    # 打印 pred_u_d 的最大值和最小值
    #print("Max value of pred_u_d:", torch.max(pred_u_rgbd).item(), torch.min(pred_u_rgbd).item())

    return pred_u_rgbd.float()


def L2_loss(shallow_features, deep_features, mask=None):
    """
    shallow_features: 浅层特征 (top-half channel feature)
    deep_features: 深层特征 (bottom-half channel feature)
    mask: 可选的mask，用于选择某些位置的特征进行计算
    """
    # 计算两个特征的L2范数差异
    loss = (shallow_features - deep_features).pow(2)

    if mask is not None:
        # 如果有mask，应用mask只在指定位置计算loss
        loss = loss * mask

    # 计算损失均值
    return loss.mean()
"""BCE + DICE Loss"""


class BceDiceLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss(weight, reduction=reduction)
        self.dice = DiceLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)

        loss = bce_loss + dice_loss

        return loss


""" Entropy Loss """


def entropy_loss(p, C=2):
    # p N*C*W*H*D
    y1 = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1) / \
         torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent


""" Deep Supervision Loss"""


def DeepSupervisionLoss(pred, gt, labeled_bs):
    d0, d1, d2, d3, d4 = pred[0:]

    criterion = BceDiceLoss()

    loss0 = criterion(d0[:labeled_bs], gt[:labeled_bs])
    gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True)
    loss1 = criterion(d1[:labeled_bs], gt[:labeled_bs])
    gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True)
    loss2 = criterion(d2[:labeled_bs], gt[:labeled_bs])
    gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True)
    loss3 = criterion(d3[:labeled_bs], gt[:labeled_bs])
    gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True)
    loss4 = criterion(d4[:labeled_bs], gt[:labeled_bs])

    return loss0 + loss1 + loss2 + loss3 + loss4


def DeepSupervisionLoss_(pred, gt, labeled_bs):
    d0, d1, d2, d3, d4 = pred[0:]

    criterion = BceDiceLoss()

    loss0 = criterion(d0[:labeled_bs], gt[:labeled_bs])
    d1 = F.interpolate(d1[:labeled_bs], scale_factor=2, mode='bilinear', align_corners=True)
    loss1 = criterion(d1, gt[:labeled_bs])
    d2 = F.interpolate(d2[:labeled_bs], scale_factor=4, mode='bilinear', align_corners=True)
    loss2 = criterion(d2, gt[:labeled_bs])
    d3 = F.interpolate(d3[:labeled_bs], scale_factor=8, mode='bilinear', align_corners=True)
    loss3 = criterion(d3, gt[:labeled_bs])
    d4 = F.interpolate(d4[:labeled_bs], scale_factor=16, mode='bilinear', align_corners=True)
    loss4 = criterion(d4, gt[:labeled_bs])

    return loss0 + loss1 + loss2 + loss3 + loss4
