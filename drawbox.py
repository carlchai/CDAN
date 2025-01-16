import os

import cv2
import numpy as np
import torch




def drawRec(samplepath,maskpath):
    store = []
    info = []
    masks = os.listdir(maskpath)
    pic = os.listdir(samplepath)
    for pictures,mp in zip(pic,masks):
        boxes = []
        mppath = os.path.join(maskpath, mp)
        mask = cv2.imread(mppath)
        hsv = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)
        low_hsv = np.array([0,0, 221])
        high_hsv = np.array([180,30,255])
        gray = cv2.inRange(hsv, lowerb=low_hsv, upperb=high_hsv)
        contours, hierarchy = cv2.findContours(gray,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])

            rect = cv2.boundingRect(contours[i])

            [x , y, w, h] = rect
            box = [x,y,x+w,y+h]
            boxes.append(box)
        info.append(boxes)
    return info


def calculate_IOU(box1,box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area_box1 + area_box2 - intersection
    iou = intersection / union
    return iou


def calculate_precision_recall(detected_boxes, true_boxes, iou_threshold):
    num_true_boxes = len(true_boxes)
    true_positives = 0
    mean_iou=[]
    for true_box in true_boxes:
        max_iou = 0
        for detected_box in detected_boxes:
            iou = calculate_IOU(detected_box, true_box)
            if iou > max_iou:
                max_iou = iou
            if max_iou >= iou_threshold:
                true_positives += 1
                break
        mean_iou.append(max_iou)
    if len(mean_iou) == 0:
        mean = 0
    else:
        mean = sum(mean_iou)/len(mean_iou)
    false_positives = len(detected_boxes) - true_positives
    false_negatives = num_true_boxes - true_positives
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    return precision, recall,mean


def compute_precisionandrecall(pred_box,gt_box):
    precision_all = []
    recall_all = []
    mean_all = []
    for i in range(len(pred_box)):
        precision, recall,mean = calculate_precision_recall(pred_box[i], gt_box[i], 0.5)
        precision_all.append(precision)
        recall_all.append(recall)
        mean_all.append(mean)
    return precision_all,recall_all,mean_all



if __name__ == '__main__':
    samplepath = r'./output_img3'
    pred_path = r'./output_img2'
    gt_path = r'./output_img1'

    pred_box = drawRec(samplepath, pred_path)
    gt_box = drawRec(samplepath, gt_path)

    pre,rec,mean_all = compute_precisionandrecall(pred_box,gt_box)
    print(sum(mean_all)/len(mean_all))
    sum = 0
    for i in range(len(pre)):
        score=pre[i]*rec[i]
        sum = sum+pre[i]
    ap = sum/len(pre)
    print(ap)








