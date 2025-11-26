#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

###
def tv_loss(feature_map):
    """
    Input:
    - feature_map: (C, H, W)
    Return:
    - total variation loss
    """
    tv_loss = ((feature_map[:, :, :-1] - feature_map[:, :, 1:])**2).sum() + ((feature_map[:, :-1, :] - feature_map[:, 1:, :])**2).sum()

    return tv_loss


def calculate_accuracy(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    total_pixels = np.prod(y_true.shape)
    return correct_predictions / total_pixels
    
    
def calculate_iou(y_true, y_pred, num_classes):
    iou = []
    for i in range(num_classes):
        true_labels = y_true == i
        predicted_labels = y_pred == i
        intersection = np.logical_and(true_labels, predicted_labels)
        union = np.logical_or(true_labels, predicted_labels)
        iou_score = np.sum(intersection) / np.sum(union)
        iou.append(iou_score)
    return np.nanmean(iou)  


import torch.nn.functional as F

def separation_loss_cosine_batch(F, neighbors, distances, alpha=1.0, reduction='mean'):
    """
    Batched cosine similarity-based separation loss.

    F:         (N, C)       - N 个目标点的特征
    neighbors: (N, 16, C)   - 每个点的 16 个邻近点的特征
    distances: (N, 16)      - 每个目标点到邻居的距离
    alpha: float            - 衰减系数
    reduction: str          - 'mean' or 'sum'

    Returns:
        Scalar loss
    """

    # L2 normalize target features and neighbor features
    F_norm = F / F.norm(dim=1, keepdim=True)                  # (N, C)
    neighbors_norm = neighbors / neighbors.norm(dim=2, keepdim=True)  # (N, 16, C)

    # Compute cosine similarity: (N, 16)
    cossim = torch.sum(neighbors_norm * F_norm.unsqueeze(1), dim=2)

    # Compute weights based on spatial distance: (N, 16)
    weights = torch.exp(-alpha * distances)

    # Weighted cosine similarity loss
    loss_matrix = weights * cossim  # (N, 16)

    # Total loss: we want large similarity → large loss
    # Optionally invert sign if you prefer minimization
    if reduction == 'mean':
        loss = loss_matrix.mean()
    elif reduction == 'sum':
        loss = loss_matrix.sum()
    else:
        return loss_matrix  # shape: (N, 16)

    loss = torch.clamp(loss, min=0.0)
    return loss