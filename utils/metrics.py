# -*- coding: utf-8 -*-
# @File : metrics.py
# @Author: Runist
# @Time : 2021/9/26 10:56
# @Software: PyCharm
# @Brief: Model metrics function

import numpy as np
import torch
from collections import defaultdict, deque
import datetime
import time
import torch
import torch.distributed as dist

import errno
import os



def CD_metric_torch(pred, target):
    # shape: batch_size x w x h
    TP = torch.sum(pred & target, (1, 2)).float()
    FP = torch.sum(pred & ~target, (1, 2)).float()
    FN = torch.sum(~pred & target, (1, 2)).float()
    TN = torch.sum(~pred & ~target, (1, 2)).float()
    union = torch.sum(pred | target, (1, 2)).float()

    precision = TP / (TP + FP + 1e-7)
    recall = TP / (TP + FN + 1e-7)
    accuracy = (TP + TN) / (TP + FP + FN + TN + 1e-7)
    f1_score = 2 * recall * precision / (precision + recall + 1e-7)
    IoU = TP / (union + 1e-7)
    return (precision.cpu(), recall.cpu(), accuracy.cpu(), f1_score.cpu())






def generate_matrix(gt_image, pre_image, num_classes=2):
    mask = (gt_image >= 0) & (gt_image < num_classes)
    label = num_classes * gt_image[mask].astype('int') + pre_image[mask]
    count = np.bincount(label, minlength=num_classes**2)
    confusion_matrix = count.reshape(num_classes, num_classes)

    return confusion_matrix


def get_mean_iou(cm):
    sum_over_col = np.sum(cm, axis=0).astype(float)
    sum_over_row = np.sum(cm, axis=1).astype(float)
    cm_diag = np.diagonal(cm).astype(float)
    denominator = sum_over_row + sum_over_col - cm_diag

    num_valid_entries = np.sum((denominator != 0).astype(float))

    # If the value of the denominator is 0, set it to 1 to avoid
    # zero division.
    denominator = np.where(
        denominator > 0,
        denominator,
        np.ones_like(denominator))

    ious = cm_diag / denominator

    # If the number of valid entries is 0 (no classes) we return 0.
    mean_iou = np.where(
        num_valid_entries > 0,
        np.sum(ious) / num_valid_entries,
        0)

    return mean_iou, ious


def get_recall(cm):
    sum_over_row = np.sum(cm, axis=1).astype(float)
    cm_diag = np.diagonal(cm).astype(float)

    sum_over_row = np.where(
        sum_over_row > 0,
        sum_over_row,
        np.ones_like(sum_over_row))

    recalls = cm_diag / sum_over_row
    mean_recall = np.mean(recalls)

    return mean_recall, recalls


def get_precision(cm):
    sum_over_col = np.sum(cm, axis=0).astype(float)
    cm_diag = np.diagonal(cm).astype(float)

    sum_over_col = np.where(
        sum_over_col > 0,
        sum_over_col,
        np.ones_like(sum_over_col))

    precisions = cm_diag / sum_over_col
    mean_precision = np.mean(precisions)

    return mean_precision, precisions

