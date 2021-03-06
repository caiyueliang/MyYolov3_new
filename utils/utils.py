# encoding: utf-8
from __future__ import division
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# import matplotlib.pyplot as plt
# import matplotlib.patches as patches


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def bbox_iou_numpy(box1, box2):
    """Computes IoU between bounding boxes.
    Parameters
    ----------
    box1 : ndarray
        (N, 4) shaped array with bboxes
    box2 : ndarray
        (M, 4) shaped array with bboxes
    Returns
    -------
    : ndarray
        (N, M) shaped array with IoUs
    """
    area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    iw = np.minimum(np.expand_dims(box1[:, 2], axis=1), box2[:, 2]) - np.maximum(
        np.expand_dims(box1[:, 0], 1), box2[:, 0]
    )
    ih = np.minimum(np.expand_dims(box1[:, 3], axis=1), box2[:, 3]) - np.maximum(
        np.expand_dims(box1[:, 1], 1), box2[:, 1]
    )

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def non_max_suppression(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()
        image_pred = image_pred[conf_mask]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5 : 5 + num_classes], 1, keepdim=True)
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
        # Iterate through all predicted classes
        unique_labels = detections[:, -1].cpu().unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
        for c in unique_labels:
            # Get the detections with the particular class
            detections_class = detections[detections[:, -1] == c]
            # Sort the detections by maximum objectness confidence
            _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
            detections_class = detections_class[conf_sort_index]
            # Perform non-maximum suppression
            max_detections = []
            while detections_class.size(0):
                # Get detection with highest confidence and save as max detection
                max_detections.append(detections_class[0].unsqueeze(0))
                # Stop if we're at the last detection
                if len(detections_class) == 1:
                    break
                # Get the IOUs for all boxes with lower confidence
                ious = bbox_iou(max_detections[-1], detections_class[1:])
                # Remove detections with IoU >= NMS threshold
                detections_class = detections_class[1:][ious < nms_thres]

            max_detections = torch.cat(max_detections).data
            # Add max detections to outputs
            output[image_i] = (
                max_detections if output[image_i] is None else torch.cat((output[image_i], max_detections))
            )

    return output


def build_targets(pred_boxes, pred_conf, pred_cls, target, anchors, num_anchors, num_classes, grid_size, ignore_thres,
                  conf_thres, iou_match_thres, img_dim):
    # print('pred_boxes', pred_boxes.size())        # (-1, 3, 13, 13, 4)
    # print('pred_conf', pred_conf.size())          # (-1, 3, 13, 13)
    # print('pred_cls', pred_cls.size())            # (-1, 3, 13, 13, 2)
    # print('target', target.size())                # (-1, 50, 5)           最多50个框：类别,x,y,w,h
    # print('anchors', anchors.size())              # (3, 2)                每层3各锚框：w,h
    # print('num_anchors', num_anchors)             # 3
    # print('num_classes', num_classes)             # 2
    # print('grid_size', grid_size)                 # 13
    # print('ignore_thres', ignore_thres)           # 忽略的阈值0.5，0.8
    # print('img_dim', img_dim)                     # 416

    nB = target.size(0)                                     # batch size： -1
    nA = num_anchors                                        # 3
    nC = num_classes                                        # 2
    nG = grid_size                                          # 13,步长？

    # mask和conf_mask相反的东西？一个表示是背景，一个表示是目标？
    mask = torch.zeros(nB, nA, nG, nG)                      # (-1, 3, 13, 13)       mask全为0,预测对的目标为1，用来预测目标
    conf_mask = torch.ones(nB, nA, nG, nG)                  # (-1, 3, 13, 13)       conf_mask全为1,预测对的目标为1(后面会减掉)，以及与锚框IOU重叠度小的值为0，预测背景

    tx = torch.zeros(nB, nA, nG, nG)                        # (-1, 3, 13, 13)
    ty = torch.zeros(nB, nA, nG, nG)                        # (-1, 3, 13, 13)
    tw = torch.zeros(nB, nA, nG, nG)                        # (-1, 3, 13, 13)
    th = torch.zeros(nB, nA, nG, nG)                        # (-1, 3, 13, 13)
    tconf = torch.ByteTensor(nB, nA, nG, nG).fill_(0)       # (-1, 3, 13, 13)       填充0
    tcls = torch.ByteTensor(nB, nA, nG, nG, nC).fill_(0)    # (-1, 3, 13, 13, 2)    填充0

    nGT = 0
    nCorrect = 0
    for b in range(nB):
        for t in range(target.shape[1]):            # 遍历真实的每个标签（50个,有值的才继续）
            if target[b, t].sum() == 0:
                continue
            nGT += 1                                # 实际标签个数

            # Convert to position relative to box   # 真实标签框转换为相对于方框的位置
            gx = target[b, t, 1] * nG               # 真实标签框的x * 13,步长 ？
            gy = target[b, t, 2] * nG               # 真实标签框的y * 13,步长 ？
            gw = target[b, t, 3] * nG               # 真实标签框的w * 13,步长 ？  TODO
            gh = target[b, t, 4] * nG               # 真实标签框的h * 13,步长 ？
            # print('gx', gx.size(), gx)            # 3.4881|6.7398
            # print('gy', gh.size(), gy)            # 3.7652|6.7463
            # print('gw', gw.size(), gw)            # 0.5290|1.1356
            # print('gh', gh.size(), gh)            # 0.5314|1.0758

            # Get grid box indices                  # 获取网格框索引（绑定具体某个网格）
            gi = int(gx)                            # 绑定具体某个网格
            gj = int(gy)                            # 绑定具体某个网格

            # Get shape of gt box                   # 获取gt框的形状
            gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
            # print('gt_box', gt_box.size(), gt_box)                                # (1, 4), tensor([[ 0.0000,  0.0000,  1.4098,  0.7390]])
            # Get shape of anchor box
            anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((len(anchors), 2)), np.array(anchors)), 1))
            # print('anchor_shapes', anchor_shapes.size(), anchor_shapes)           # size (3, 4)

            # Calculate iou between gt and anchor shapes
            anch_ious = bbox_iou(gt_box, anchor_shapes)                             # 计算gt框和锚框的iou值
            # print('anch_ious', anch_ious.size(), anch_ious)                       # (3,), tensor([ 0.2281,  0.0875,  0.0223]))
            # Where the overlap is larger than threshold set mask to zero (ignore)
            conf_mask[b, anch_ious > ignore_thres, gj, gi] = 0                      # 如果重叠大于阈值，则将mask设置为零（忽略）,后面只让最好的一个框来预测？
            # print('conf_mask', conf_mask.size(), conf_mask)                       # (-1, 3, 13, 13)， 大都是 1 ？？？？ TODO

            # Find the best matching anchor box
            best_n = np.argmax(anch_ious)                                           # iou值最大的锚框标记为最好的预测锚框best_n
            # Get ground truth box
            gt_box = torch.FloatTensor(np.array([gx, gy, gw, gh])).unsqueeze(0)     # 真实标签框
            # Get the best prediction
            pred_box = pred_boxes[b, best_n, gj, gi].unsqueeze(0)                   # 最好的预测框
            # Masks
            mask[b, best_n, gj, gi] = 1                                             # mask全为0,预测对的目标为1，用来预测目标
            conf_mask[b, best_n, gj, gi] = 1                                        # conf_mask全为1,预测对的目标为1(后面会减掉)，以及与锚框IOU重叠度小的值为0，预测背景
            # Coordinates
            tx[b, best_n, gj, gi] = gx - gi
            ty[b, best_n, gj, gi] = gy - gj
            # Width and height
            tw[b, best_n, gj, gi] = math.log(gw / anchors[best_n][0] + 1e-16)
            th[b, best_n, gj, gi] = math.log(gh / anchors[best_n][1] + 1e-16)
            # One-hot encoding of label
            target_label = int(target[b, t, 0])
            tcls[b, best_n, gj, gi, target_label] = 1
            tconf[b, best_n, gj, gi] = 1

            # Calculate iou between ground truth and best matching prediction
            iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)                        # 真实标签框gt_box和最好的预测框pred_box计算iou
            pred_label = torch.argmax(pred_cls[b, best_n, gj, gi])
            score = pred_conf[b, best_n, gj, gi]
            if iou > iou_match_thres and pred_label == target_label and score > conf_thres:
                nCorrect += 1                                                       # 对的个数

    return nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return torch.from_numpy(np.eye(num_classes, dtype="uint8")[y])
