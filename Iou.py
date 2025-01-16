import os

import cv2
from PIL import Image
import numpy as np


def compute_iou(pred, label, num_classes):
    """
    计算每个类的 IoU（Intersection over Union）

    :param pred: 预测的分割图 (H x W)
    :param label: 真实的标签图 (H x W)
    :param num_classes: 类别数
    :return: 每个类别的 IoU
    """
    iou_list = []
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        label_inds = (label == cls)

        intersection = np.logical_and(pred_inds, label_inds).sum()
        union = np.logical_or(pred_inds, label_inds).sum()

        if union == 0:
            iou = float('nan')  # 忽略没有出现的类别
        else:
            iou = intersection / union
        iou_list.append(iou)

    return iou_list


def compute_miou(preds, labels, num_classes):
    """
    计算 mIoU（mean Intersection over Union）
    :param preds: 预测的分割图集合 (N x H x W)
    :param labels: 真实的标签图集合 (N x H x W)
    :param num_classes: 类别数
    :return: mIoU
    """
    iou_list = []
    for pred, label in zip(preds, labels):
        iou = compute_iou(pred, label, num_classes)
        iou_list.append(iou)
    iou_list = np.array(iou_list)
    mean_iou = np.nanmean(iou_list, axis=0)  # 忽略 nan 值
    return mean_iou


def compute_meanTotal(pred,mask):
    num_classes = 1
    predlist = os.listdir(pred)
    masklist = os.listdir(mask)
    mean_total = []
    for pred_img,mask_img in zip(predlist,masklist):
        pp = os.path.join(pred,pred_img)
        mp = os.path.join(mask,mask_img)
        picp = cv2.imread(pp)
        picm = cv2.imread(mp)
        miou = compute_miou(picp, picm, num_classes)
        mean_total.append(miou)
    mean = sum(mean_total)/len(mean_total)
    return mean


# 示例用法
if __name__ == "__main__":
    pred = 'output_img1'
    mask = 'output_img2'
    mean = compute_meanTotal(pred,mask)
    print(mean)


