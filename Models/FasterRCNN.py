import sys

import numpy as np
import torch
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from utils import load_single_image
from python_scripts.Main import my_time
from utils import save_single_image_detections
from enum import Enum

from typing import Dict


class BackBone(Enum):
    MOBILE_NET_V2 = 1
    RESNET_50 = 2


verbose = True


def main():
    score_th = 0.5
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'------Using device : {device}---------')
    # Create Faster RCNN model and run it on single image

    model_resnet_bb = get_model_instance(BackBone.RESNET_50)
    # load image and preprocess
    pre_proc = ['normalize']
    im = load_single_image('C:\\Users\\yoavk\\Documents\\GitHub\\ObjDetectChallenge\\data\\busesTrain\\DSCF1110.JPG',
                           pre_proc)
    im_torch = torch.from_numpy(im).to(device=device)
    im_torch = im_torch.permute(2, 0, 1).unsqueeze(0)
    print(im_torch.shape, im_torch.dtype)

    model_resnet_bb.eval()
    model_resnet_bb = model_resnet_bb.double()  # TODO does this damage accuracy?

    t = my_time()
    t.tic()

    with torch.no_grad():
        prediction = model_resnet_bb(im_torch)

    t.toc()
    detection_dict = filter_detections(score_th, prediction[0])
    save_single_image_detections('C:\\Users\\yoavk\\Documents\\GitHub\\ObjDetectChallenge\\testAnnotations.txt',
                                 detection_dict, 'DSCF1110.JPG')


def get_model_instance(backbone_type: BackBone, min_size=800) -> torch.nn.Module:
    if backbone_type == BackBone.MOBILE_NET_V2:
        # based on pytorch tutorial
        # (https://colab.research.google.com/github/pytorch/vision/blob/temp-tutorial/tutorials/torchvision_finetuning_instance_segmentation.ipynb#scrollTo=RoAEkUgn4uEq)
        backbone = torchvision.models.mobilenet_v2(pretrained=True).features  # TODO set to false after getting weights
        backbone.out_channels = 1280
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                           aspect_ratios=((0.5, 1.0, 2.0),))
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                        output_size=7,
                                                        sampling_ratio=2)

        model = torchvision.models.detection.FasterRCNN(backbone, num_classes=91, rpn_anchor_generator=anchor_generator,
                                                        box_roi_pool=roi_pooler, min_size=min_size)
    elif backbone_type == BackBone.RESNET_50:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    else:

        sys.exit('Bad backBone type')

    return model


def filter_detections(score_thresh: float, detections_dict: Dict) -> Dict:
    out_dict = {}
    # convert from tensor to numpy
    scores, boxes, labels = detections_dict['scores'].numpy(), detections_dict['boxes'].numpy(), detections_dict['labels'].numpy()

    num_detections_pre_filt = scores.shape[0]
    if verbose:
        print(f'Num detections before filtering = {num_detections_pre_filt}, threshold = {score_thresh}')

    filt_indices = np.where(scores > score_thresh)
    out_dict['boxes'] = boxes[filt_indices]
    out_dict['labels'] = labels[filt_indices]

    print(f'Num detections before filtering = {len(out_dict["labels"])}')

    return out_dict


# TODO implement this method. will be called from runMe.py
def run_frcnn(annFileName: str, im_dir: str):
    # create model, load weights, runs the model on each the images in the dir, saves annotation to annFileName
    pass



if __name__ == '__main__':
    main()