import numpy as np
from typing import List, Dict
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# Copied from busProjectTest
def IOU(boxAList, boxBList):
    Th = 0.7
    iou = []
    matches = {}
    tp = 0
    fp = len(boxBList)
    missed = len(boxAList)
    for i in range(len(boxAList)):
        boxA = boxAList[i][:4]
        iou_ = []
        for j in range(len(boxBList)):
            boxB = boxBList[j][:4]
            if(not ((boxB[0] <= boxA[0] <= boxB[0] + boxB[2]) or (boxA[0] <= boxB[0] <= boxA[0] + boxA[2]))):
                iou_.append(0.0)
                continue
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
            yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
            interArea = (xB - xA + 1) * (yB - yA + 1)
            boxAArea = (boxA[2] + 1)*(boxA[3] + 1)
            boxBArea = (boxB[2] + 1)*(boxB[3] + 1)
            iou_.append(interArea / float(boxAArea + boxBArea - interArea))
        # maxIou = max(iou_)
        maxIou = [t for t in iou_ if t >= Th]
        maxIouIndex = [iou_.index(t) for t in maxIou]
        # maxIouIndex = iou_.index(max(iou_))
        if len(maxIou) > 1: # Two possible IoUs, choose the one with the correct color, and than the biggest
            poss = []
            for ind in range(len(maxIouIndex)):
                if (boxAList[i][4] == boxBList[maxIouIndex[ind]][4]):
                    poss.append(ind)
            if len(poss) > 1:
                maxIou = max(maxIou[poss])
                maxIouIndex = maxIouIndex.index(maxIou)
            else:
                poss = poss[0]
                maxIou = maxIou[poss]
                maxIouIndex = maxIouIndex[poss]
        elif len(maxIou) == 1:
            maxIou = maxIou[0]
            maxIouIndex = maxIouIndex[0]
        else:
            continue
        iou.append(maxIou)
        if (maxIouIndex in matches and maxIou > iou[matches[maxIouIndex]]): # If a match is found with bigger IOU
            if (iou[matches[maxIouIndex]] >= Th and boxAList[matches[maxIouIndex]][4] == boxBList[maxIouIndex][4]):
                pass
            elif(maxIou >= Th and boxAList[i][4] == boxBList[maxIouIndex][4]):
                tp += 1
                missed -= 1
                fp -= 1
            matches[maxIouIndex] = i
        if(not maxIouIndex in matches):
            matches[maxIouIndex] = i
            if(maxIou > Th and boxAList[i][4] == boxBList[maxIouIndex][4]):
                tp += 1
                missed -= 1
                fp -= 1
    return tp, fp, missed, iou

def load_single_image(path:str, pre_proc: List=None) -> np.ndarray:
    img = mpimg.imread(path)
    if pre_proc is not None:
        # run pre-processing according to input dict
        if 'normalize' in pre_proc:
            img = img / 255
        if 'c_first' in pre_proc:
            img = img.transpose(2, 0, 1)

    return img


def save_single_image_detections(file_name: str, detections_dict: Dict, image_name: str):
    """

    :param image_name:
    :param file_name: annotation file name
    :param detections_dict: dictionary of the sort {boxes, labels}. Labels is a list labels. boxes is N x 4 nd array
                            where N is the number of detections.
    :return:
    """
    # Format: <image name>:[x, y, w, h, class]....
    num_detections = detections_dict['boxes'].shape[0]
    with open(file_name, mode='a+') as f:
        f.write(image_name + ':')
        for detection_indx in range(num_detections):
            bbox = detections_dict['boxes'][detection_indx, :]
            f.write(f'[{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}, {detections_dict["labels"][detection_indx]}]')
            if detection_indx + 1 < num_detections:
                # If this isn't the last detection, add ","
                f.write(',')
        f.write('\n')
        f.close()


def draw_boxes(img_np, boxes, labels):
    label2clr_dict = {1: 'g', 2: 'y', 3: 'w', 4: 'tab:gray', 5: 'b', 6: 'r'}
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    for indx, box in enumerate(boxes):
        # Create a Rectangle patch
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=3,
                                 edgecolor=label2clr_dict[labels[indx]], facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)

    ax.set_axis_off()
    ax.imshow(img_np)
    plt.show()


def calc_f1_score(gt_detections: List[np.array], prd_detections: List[np.array]) -> float:
    """
    :param gt_detections: List in length M. Every entry is an N X 5. Where: M = num of images, N = num of detections for specific image.
    for each (m,n) there's an array of [x, y, w, h, label]
    :param prd_detections: same as above.
    :return: the F1 score based on the predictions in prd_detections
    """
    TP = FP = MISS = 0
    # some checks
    if len(gt_detections) != len(prd_detections):
        assert "Number of images don't match"

    for im_num in range(len(gt_detections)):
        curr_im_detections = prd_detections[im_num]
        curr_im_gt = gt_detections[im_num]
        if [] == curr_im_detections:
            # no detections
            tp = 0
            fp = 0
            numGT = 0
            missed = len(curr_im_gt)
        else:
            tp, fp, missed, iou = IOU(curr_im_gt, curr_im_detections)

        TP += tp
        FP += fp
        MISS += missed

    # Done looping through all the images - calc F1
    if(TP == 0):
        F1Score = 0
    else:
        precision = TP / (TP + FP)
        recall = TP / (TP + MISS)
        F1Score = 2 * (precision * recall) / (precision + recall)

    return F1Score