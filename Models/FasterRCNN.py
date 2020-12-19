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


class MyFasterRCNNModel:
    def __init__(self, backbone_type: BackBone, max_num_predictions, verbose=False,
                 score_thresh=0.5, min_size=800, **kwargs):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.verbose = verbose
        self.backbone = backbone_type
        if self.verbose:
            print(f'Creating Faster-RCNN model. Backbone = {self.backbone}, max number of predicitons = '
                  f'{max_num_predictions}')
        self.model = self._get_model_instance(score_thresh, max_num_predictions, min_size, **kwargs)
        self.model.double().to(self.device)

    def _get_model_instance(self, score_thresh, max_num_predictions, min_size=800, **kwargs) -> torch.nn.Module:
        if self.backbone == BackBone.MOBILE_NET_V2:
            # based on pytorch tutorial
            # (https://colab.research.google.com/github/pytorch/vision/blob/temp-tutorial/tutorials/torchvision_finetuning_instance_segmentation.ipynb#scrollTo=RoAEkUgn4uEq)
            backbone = torchvision.models.mobilenet_v2(
                pretrained=True).features  # TODO set to false after getting weights??
            backbone.out_channels = 1280
            anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                               aspect_ratios=((0.5, 1.0, 2.0),))
            roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                            output_size=7,
                                                            sampling_ratio=2)

            model = torchvision.models.detection.FasterRCNN(backbone, num_classes=91,
                                                            rpn_anchor_generator=anchor_generator,
                                                            box_roi_pool=roi_pooler,
                                                            min_size=min_size,
                                                            box_score_thresh=score_thresh,
                                                            box_detections_per_img=max_num_predictions,
                                                            **kwargs)
        elif self.backbone == BackBone.RESNET_50:
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, min_size=min_size,
                                                                         box_score_thresh=score_thresh,
                                                                         box_detections_per_img=max_num_predictions,
                                                                         **kwargs)
        else:
            sys.exit('Bad backBone type')

        return model

    def train(self):
        pass

    def __call__(self, im: np.ndarray) -> Dict:
        """
        Run model inference. Also converts the output back to numpy. returns a dictionary of labels, scores and boxes.
        :param im:
        :return:
        """
        image = im.copy()
        # prepare image
        if np.max(image) > 1:
            # Normalize image
            image = image / 255
        if image.shape[2] == 3:
            if self.verbose:
                print(f'image shape is {image.shape}, transposing so channel is first dim')
            image = np.transpose(image, (2, 0, 1))

        im_torch = torch.from_numpy(image).unsqueeze(0).to(self.device)  # also add batch dim

        self.model.eval()
        with torch.no_grad():
            prediction = self.model(im_torch)[0]

        # convert from tensor to numpy
        scores, boxes, labels = prediction['scores'].numpy(), prediction['boxes'].numpy(), prediction['labels'].numpy()
        out_dict = {'scores': scores, 'boxes': boxes, 'labels': labels}

        return out_dict


def main():
    score_th = 0.5
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'------Using device : {device}---------')
    # Create Faster RCNN model and run it on single image

    fasterRCNN = MyFasterRCNNModel(backbone_type=BackBone.RESNET_50, max_num_predictions=5, verbose=True)
    # load image and preprocess
    im = load_single_image('C:\\Users\\yoavk\\Documents\\GitHub\\ObjDetectChallenge\\data\\busesTrain\\DSCF1110.JPG')

    t = my_time()
    t.tic()

    prediction = fasterRCNN(im)

    t.toc()
    save_single_image_detections('C:\\Users\\yoavk\\Documents\\GitHub\\ObjDetectChallenge\\testAnnotations.txt',
                                 {'boxes': prediction['boxes'], 'labels': prediction['labels']}, 'DSCF1110.JPG')


# TODO implement this method. will be called from runMe.py
def run_frcnn(annFileName: str, im_dir: str):
    # create model, load weights, runs the model on each the images in the dir, saves annotation to annFileName
    pass



if __name__ == '__main__':
    main()