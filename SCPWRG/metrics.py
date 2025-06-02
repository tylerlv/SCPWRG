import numpy as np
from sklearn.metrics import f1_score
import torch
def calJI(binary_GT,binary_R):
    row, col = binary_GT.shape
    DSI_s,DSI_t = 0,0
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 1 and binary_R[i][j] == 1:
                DSI_s += 1
            if binary_GT[i][j] == 1:
                DSI_t += 1
            if binary_R[i][j]  == 1:
                DSI_t += 1
    JI = DSI_s/(DSI_t-DSI_s+1e-10)
    return JI

def calPrecision(binary_GT,binary_R):
    row, col = binary_GT.shape
    P_s,P_t = 0,0
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 1 and binary_R[i][j] == 1:
                P_s += 1
            if binary_R[i][j]   == 1:
                P_t += 1

    Precision = P_s/(P_t+1e-10)
    return Precision

def calRecall(binary_GT,binary_R):
    row, col = binary_GT.shape
    R_s,R_t = 0,0
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 1 and binary_R[i][j] == 1:
                R_s += 1
            if binary_GT[i][j]   == 1:
                R_t += 1

    Recall = R_s/(R_t+1e-10)
    return Recall

def mean_iou(input, target, classes = 1):
    """  compute the value of mean iou
    :param input:  2d array, int, prediction
    :param target: 2d array, int, ground truth
    :param classes: int, the number of class
    :return:
        miou: float, the value of miou
    """
    miou = 0
    for i in range(classes):
        intersection = np.logical_and(target == i, input == i)
        union = np.logical_or(target == i, input == i)
        temp = np.sum(intersection) / (np.sum(union)+1e-10)
        miou += temp
    return  miou/classes



def iou(input, target, classes=1):
    """  compute the value of iou
    :param input:  2d array, int, prediction
    :param target: 2d array, int, ground truth
    :param classes: int, the number of class
    :return:
        iou: float, the value of iou
    """
    intersection = np.logical_and(target == classes, input == classes)
    union = np.logical_or(target == classes, input == classes)
    iou = np.sum(intersection) / (np.sum(union)+1e-10)

    return iou

def get_pixel_area(segm):
    return segm.shape[0] * segm.shape[1]

def extract_both_masks(eval_segm, gt_segm, cl, n_cl):
    eval_mask = extract_masks(eval_segm, cl, n_cl)
    gt_mask   = extract_masks(gt_segm, cl, n_cl)

    return eval_mask, gt_mask

def extract_classes(segm):
    cl = np.unique(segm)
    n_cl = len(cl)

    return cl, n_cl

def union_classes(eval_segm, gt_segm):
    eval_cl, _ = extract_classes(eval_segm)
    gt_cl, _   = extract_classes(gt_segm)

    cl = np.union1d(eval_cl, gt_cl)
    n_cl = len(cl)

    return cl, n_cl

def extract_masks(segm, cl, n_cl):
    h, w  = segm_size(segm)
    masks = np.zeros((n_cl, h, w))

    for i, c in enumerate(cl):
        masks[i, :, :] = segm == c

    return masks

def compute_f1(prediction, target):
    """
    :param prediction: 2d array, int,
            estimated targets as returned by a classifier
    :param target: 2d array, int,
            ground truth
    :return:
        f1: float
    """
    prediction.tolist(), target.tolist()
    img, target = np.array(prediction).flatten(), np.array(target).flatten()
    f1 = f1_score(y_true=target, y_pred=img)
    return  f1

def segm_size(segm):
    try:
        height = segm.shape[0]
        width  = segm.shape[1]
    except IndexError:
        raise

    return height, width

def check_size(eval_segm, gt_segm):
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)


def pixel_accuracy(eval_segm, gt_segm):
    '''
    sum_i(n_ii) / sum_i(t_i)
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    sum_n_ii = 0
    sum_t_i = 0

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        sum_n_ii += np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        sum_t_i += np.sum(curr_gt_mask)

    if (sum_t_i == 0):
        pixel_accuracy_ = 0
    else:
        pixel_accuracy_ = sum_n_ii / sum_t_i

    return pixel_accuracy_

def calDSI(predict, target):
    epsilon = 1e-5
    assert predict.size() == target.size(), "the size of predict and target must be equal."
    num = predict.size(0)

    pre = predict.view(num, -1)
    tar = target.view(num, -1)

    intersection = (pre * tar).sum(-1).sum()
    union = (pre + tar).sum(-1).sum()

    score  =  2 * (intersection + epsilon) / (union + epsilon)
    return score

def calVOE(binary_GT,binary_R):
    row, col = binary_GT.shape
    VOE_s,VOE_t = 0,0
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 1:
                VOE_s += 1
            if binary_R[i][j]  == 1:
                VOE_t += 1
    VOE = 2*(VOE_t - VOE_s)/(VOE_t + VOE_s)
    return VOE

def calRVD(binary_GT,binary_R):
    row, col = binary_GT.shape
    RVD_s,RVD_t = 0,0
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 1:
                RVD_s += 1
            if binary_R[i][j]  == 1:
                RVD_t += 1
    RVD = RVD_t/(RVD_s - 1+1e-10)
    return RVD