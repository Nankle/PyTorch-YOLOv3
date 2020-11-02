from __future__ import division
import math
import time
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import shapely
from shapely.geometry import Polygon,MultiPoint #多边形

def to_cpu(tensor):
    return tensor.detach().cpu()


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


# 用于detect模块
def rescale_boxes(boxes, current_dim, original_shape):
    """ Rescales bounding boxes to the original shape """
    orig_h, orig_w = original_shape
    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes


def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm.tqdm(unique_classes, desc="Computing AP"):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")


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


def get_batch_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, -1]

        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics


def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    
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

    """

    # 已知4个角点求IOU-scores
    # b1xc = box1[:, 0]
    # b1yc = box1[:, 1]
    # b1x1 = b1xc + box1[:, 2]
    # b1y1 = b1yc + box1[:, 3]
    # b1x2 = b1xc + box1[:, 4]
    # b1y2 = b1yc + box1[:, 5]
    # b1x3 = b1xc + box1[:, 6]
    # b1y3 = b1yc + box1[:, 7]
    # b1x4 = b1xc + box1[:, 8]
    # b1y4 = b1yc + box1[:, 9]

    # b2xc = box2[:, 0]
    # b2yc = box2[:, 1]
    # b2x1 = b2xc + box2[:, 2]
    # b2y1 = b2yc + box2[:, 3]
    # b2x2 = b2xc + box2[:, 4]
    # b2y2 = b2yc + box2[:, 5]
    # b2x3 = b2xc + box2[:, 6]
    # b2y3 = b2yc + box2[:, 7]
    # b2x4 = b2xc + box2[:, 8]
    # b2y4 = b2yc + box2[:, 9]

    ious = []
    for i in range(box1.size(0)):
        b1xc = box1[i, 0]
        b1yc = box1[i, 1]
        b1x1 = b1xc + box1[i, 2]
        b1y1 = b1yc + box1[i, 3]
        b1x2 = b1xc + box1[i, 4]
        b1y2 = b1yc + box1[i, 5]
        b1x3 = b1xc + box1[i, 6]
        b1y3 = b1yc + box1[i, 7]
        b1x4 = b1xc + box1[i, 8]
        b1y4 = b1yc + box1[i, 9]

        b2xc = box2[i, 0]
        b2yc = box2[i, 1]
        b2x1 = b2xc + box2[i, 2]
        b2y1 = b2yc + box2[i, 3]
        b2x2 = b2xc + box2[i, 4]
        b2y2 = b2yc + box2[i, 5]
        b2x3 = b2xc + box2[i, 6]
        b2y3 = b2yc + box2[i, 7]
        b2x4 = b2xc + box2[i, 8]
        b2y4 = b2yc + box2[i, 9]

        line1 = np.concatenate(b1x1.numpy(), b1y1.numpy(), b1x2.numpy(), b1y2.numpy(),\
            b1x3.numpy(), b1y3.numpy(), b1x4.numpy(), b1y4.numpy()).reshape(4,2)
        poly1 = Polygon(line1).convex_hull

        line2 = np.concatenate(b2x1.numpy(), b2y1.numpy(), b2x2.numpy(), b2y2.numpy(),\
            b2x3.numpy(), b2y3.numpy(), b2x4.numpy(), b2y4.numpy()).reshape(4,2)
        poly2 = Polygon(line2).convex_hull

        union_poly = np.concatenate((line1,line2))

        inter_area = poly1.intersection(poly2).area  #相交面积

        #union_area = poly1.area + poly2.area - inter_area
        union_area = MultiPoint(union_poly).convex_hull.area
        
        if union_area == 0:
            iou= 0
        else:
            iou=float(inter_area) / union_area
        ious += iou
        
    return torch.Tensor(iou)


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Object confidence times class confidence
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            weights = detections[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output


def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):

    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    nB = pred_boxes.size(0)    # Batch Size
    nA = pred_boxes.size(1)    # Anchors Number
    nC = pred_cls.size(-1)     # Prediction Class Number
    nG = pred_boxes.size(2)    # Grid Size

    # Output tensors
    obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)
    noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
    iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)

    txc = FloatTensor(nB, nA, nG, nG).fill_(0)
    tyc = FloatTensor(nB, nA, nG, nG).fill_(0)
    tx1b = FloatTensor(nB, nA, nG, nG).fill_(0)
    ty1b = FloatTensor(nB, nA, nG, nG).fill_(0)
    tx2b = FloatTensor(nB, nA, nG, nG).fill_(0)
    ty2b = FloatTensor(nB, nA, nG, nG).fill_(0)
    tx3b = FloatTensor(nB, nA, nG, nG).fill_(0)
    ty3b = FloatTensor(nB, nA, nG, nG).fill_(0)
    tx4b = FloatTensor(nB, nA, nG, nG).fill_(0)
    ty4b = FloatTensor(nB, nA, nG, nG).fill_(0)

    # tx = FloatTensor(nB, nA, nG, nG).fill_(0)
    # ty = FloatTensor(nB, nA, nG, nG).fill_(0)
    # tw = FloatTensor(nB, nA, nG, nG).fill_(0)
    # th = FloatTensor(nB, nA, nG, nG).fill_(0)

    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)

    # Convert to position relative to box
    target_boxes = target[:, 2:] * nG    #归一化后的box在当前层的实际位置
    gxy = target_boxes[:, :2]
    # gwh = target_boxes[:, 2:]
    gbias = target_boxes[:, 2:]


    # Get anchors with best iou
    # ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    # best_ious, best_n = ious.max(0)

    # 因为目前只有一个anchors，因此暂时把anchors纬的筛选省略掉
    # 只能对第0维进行操作
    best_n = torch.zeros(gxy.size(0), dtype=torch.int)

    # Separate target values
    b, target_labels = target[:, :2].long().t()

    gx, gy = gxy.t()
    # gw, gh = gwh.t()
    gx1, gy1, gx2, gy2, gx3, gy3, gx4, gy4 = gbias.t()
    gi, gj = gxy.long().t()  #取整😍
    # Set masks
    obj_mask[b, best_n, gj, gi] = 1
    noobj_mask[b, best_n, gj, gi] = 0

    # Set noobj mask to zero where iou exceeds ignore threshold
    # for i, anchor_ious in enumerate(ious.t()):
    #     noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # Coordinates
    # tx[b, best_n, gj, gi] = gx - gx.floor()   #向下取整
    # ty[b, best_n, gj, gi] = gy - gy.floor()

    txc[b, best_n, gj, gi] = gx - gx.floor()
    tyc[b, best_n, gj, gi] = gy - gy.floor()
    tx1b[b, best_n, gj, gi] = gx1
    ty1b[b, best_n, gj, gi] = gy1
    tx2b[b, best_n, gj, gi] = gx2
    ty2b[b, best_n, gj, gi] = gy2
    tx3b[b, best_n, gj, gi] = gx3
    ty3b[b, best_n, gj, gi] = gy3
    tx4b[b, best_n, gj, gi] = gx4
    ty4b[b, best_n, gj, gi] = gy4
    

    # Width and height
    # tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    # th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)

    # One-hot encoding of label
    tcls[b, best_n, gj, gi, target_labels] = 1

    # Compute label correctness and iou at best anchor
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

    tconf = obj_mask.float()
    return iou_scores, class_mask, obj_mask, noobj_mask, txc, tyc, tx1b, ty1b,\
        tx2b, ty2b, tx3b, ty3b, tx4b, ty4b, tcls, tconf
