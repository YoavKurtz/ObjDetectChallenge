import sys
import copy

import numpy as np
import torch
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from ..utils import load_single_image
# from python_scripts.Main import my_time
from ..utils import save_single_image_detections
from enum import Enum

from typing import Dict

from ..utils.TorchTrainUtils.engine import train_one_epoch, evaluate


class BackBone(Enum):
    MOBILE_NET_V2 = 1
    RESNET_50 = 2


class MyFasterRCNNModel:
    def __init__(self, num_classes, backbone_type: BackBone, max_num_predictions=100, verbose=False,
                 score_thresh=0.05, min_size=800, **kwargs):
        """

        :param num_classes: including 'background'
        :param backbone_type:
        :param max_num_predictions:
        :param verbose:
        :param score_thresh:
        :param min_size:
        :param kwargs:
        """
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.verbose = verbose
        self.backbone = backbone_type
        self.num_classes = num_classes
        if self.verbose:
            print(f'Creating Faster-RCNN model. #classes (including background) = {self.num_classes} ,'
                  f'Backbone = {self.backbone}, max number of predicitons = '
                  f'{max_num_predictions}')
        self.model = self._get_model_instance(score_thresh, max_num_predictions, min_size, **kwargs)
        self.model.to(self.device)
        self.best_val_loss = np.inf

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
            # get the number of input features for the classifier
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            # replace the pre-trained head with a new one
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        else:
            sys.exit('Bad backBone type')

        return model

    def _get_val_loss(self, date_val_loader):
        # go over the validation set and calculate average loss
        train_loss = []
        cls_loss = []
        # Not changing to eval because torchvision's fasterRCNN has different output for eval and train modes.
        # model.eval()
        with torch.no_grad():
            for images, targets in date_val_loader:
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                loss_dict = self.model(images, targets)

                losses = sum(loss for loss in loss_dict.values())
                loss_value = losses.item()
                train_loss.append(loss_value)

                loss_classifier = loss_dict['loss_classifier'].item()
                cls_loss.append(loss_classifier)

        epoch_train_loss = np.mean(train_loss)
        epoch_train_cls_loss = np.mean(cls_loss)

        #self.model.train()

        return epoch_train_loss, epoch_train_cls_loss

    def train(self, num_epochs, optimizer, train_loader, test_loader, lr_scheduler, weights_path=None, starting_epoch=0,
              tb_writer=None, fancy_eval=False):
        best_model_wts = copy.deepcopy(self.model.state_dict())

        for epoch in range(num_epochs):
            epoch_num = starting_epoch + epoch
            # train for one epoch, printing every 10 iterations
            _, train_loss_iter = train_one_epoch(self.model, optimizer, train_loader, self.device, epoch_num, print_freq=10)
            # update the learning rate
            lr_scheduler.step()
            # evaluate on the test dataset
            if fancy_eval:
                evaluate(self.model, test_loader, device=self.device)

            epoch_val_loss, _ = self._get_val_loss(test_loader)
            epoch_train_loss = np.mean(train_loss_iter)  # mean of the loss values during the epoch.

            print(f'Epoch #{epoch_num} loss(sum of losses): train = {epoch_train_loss}, val = {epoch_val_loss}')
            # Add results to tensor board
            if tb_writer is not None:
                with tb_writer:
                    tb_writer.add_scalars('Training convergence/',
                                          {'train_loss': epoch_train_loss,
                                           'val_loss': epoch_val_loss}, epoch_num)
                    # writer.add_scalars('loss metrics/',
                    #                    {'train_cls_loss': epoch_train_cls_loss,
                    #                     'val_cls_loss': epoch_val_cls_loss}, epoch_num)

            if epoch_val_loss < self.best_val_loss:
                # Save best model params
                print(f'current epoch val loss {epoch_val_loss} < best so far {self.best_val_loss} keeping weights')
                self.best_val_loss = epoch_val_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())

            # load best model weights
            self.model.load_state_dict(best_model_wts)
            if tb_writer is not None:
                with tb_writer:
                    tb_writer.flush()

    def train_simple(self, num_epochs, optimizer, lr_scheduler, writer, data_train_loader,
                     starting_epoch=0, data_val_loader=None, use_fancy_eval=False, print_every=10):
        itr = 1
        for epoch in range(num_epochs):
            epoch_num = starting_epoch + epoch
            self.model.train()
            train_loss = []
            cls_loss = []
            for images, targets in data_train_loader:

                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                loss_dict = self.model(images, targets)

                loss_classifier = loss_dict['loss_classifier'].item()
                losses = sum(loss for loss in loss_dict.values())
                loss_value = losses.item()
                train_loss.append(loss_value)
                cls_loss.append(loss_classifier)

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                if itr % print_every == 0:
                    print(f"Iteration #{itr} loss: {loss_value}")

                itr += 1

            # update the learning rate
            if lr_scheduler is not None:
                lr_scheduler.step()

            epoch_train_loss = np.mean(train_loss)
            epoch_train_cls_loss = np.mean(cls_loss)
            if data_val_loader is not None:
                epoch_val_loss, epoch_val_cls_loss = self._get_val_loss(data_val_loader)

            # Record to tensorBoard
            with writer:
                if data_val_loader is not None:
                    writer.add_scalars('Training convergence/',
                                       {'train_loss': epoch_train_loss,
                                        'val_loss': epoch_val_loss}, epoch_num)
                    writer.add_scalars('loss metrics/',
                                       {'train_cls_loss': epoch_train_cls_loss,
                                        'val_cls_loss': epoch_val_cls_loss}, epoch_num)
                else:
                    writer.add_scalars('Training convergence/', epoch_train_loss, epoch_num)
                    writer.add_scalars('loss metrics/', epoch_train_cls_loss, epoch_num)

            print(f"Epoch #{epoch_num + 1}/{starting_epoch + num_epochs} train_loss: {epoch_train_loss}, val_loss = {epoch_val_loss}")
            print('-' * 10)

            if use_fancy_eval:
                evaluate(self.model, data_val_loader, device=self.device)
        writer.flush()

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
        scores, boxes, labels = prediction['scores'].cpu().numpy(), prediction['boxes'].cpu().numpy(), prediction['labels'].cpu().numpy()
        out_dict = {'scores': scores, 'boxes': boxes, 'labels': labels}

        return out_dict


def main():
    score_th = 0.5
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'------Using device : {device}---------')
    # Create Faster RCNN model and run it on single image

    fasterRCNN = MyFasterRCNNModel(num_classes=7, backbone_type=BackBone.RESNET_50, max_num_predictions=5, verbose=True)
    # load image and preprocess
    im = load_single_image('C:\\Users\\yoavk\\Documents\\GitHub\\ObjDetectChallenge\\data\\busesTrain\\DSCF1110.JPG')

    prediction = fasterRCNN(im)

    save_single_image_detections('C:\\Users\\yoavk\\Documents\\GitHub\\ObjDetectChallenge\\testAnnotations.txt',
                                 {'boxes': prediction['boxes'], 'labels': prediction['labels']}, 'DSCF1110.JPG')


# TODO implement this method. will be called from runMe.py
def run_frcnn(annFileName: str, im_dir: str):
    print('TODO stuff here')
    # create model, load weights, runs the model on each the images in the dir, saves annotation to annFileName



if __name__ == '__main__':
    main()