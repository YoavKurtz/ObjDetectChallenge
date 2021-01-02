import sys
import copy

import numpy as np
import torch
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from ..utils import load_single_image
# from python_scripts.Main import my_time
from ..utils import save_single_image_detections, calc_f1_score
from enum import Enum

from typing import Dict, List

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
        self.num_epochs_trained = 0
        self.model = self._get_model_instance(score_thresh, max_num_predictions, min_size, **kwargs)
        self.model.to(self.device)
        self.best_score = 0
        self.trainable_backbone_layers = 0
        for layer in self.model.backbone.body.parameters():
            if layer.requires_grad:
                self.trainable_backbone_layers += 1

        self.initial_lr = 0  # filled when train is called
        if self.verbose:
            print(f'Creating Faster-RCNN model. #classes (including background) = {self.num_classes} ,'
                  f'Backbone = {self.backbone}, max number of predicitons = '
                  f'{max_num_predictions}, trainable_backbone_layers = {self.trainable_backbone_layers}')

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
        self.model.train()
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

        return epoch_train_loss, epoch_train_cls_loss

    def _get_f1_score(self, data_val_loader):
        ground_truth_list = []
        prdct_list = []
        self.model.eval()
        with torch.no_grad():
            for images, gt_targets in data_val_loader:
                if len(images) > 1:
                    assert "method does not support data loader with batch size > 1"

                boxes = gt_targets[0]['boxes'].detach().cpu().numpy()
                labels = gt_targets[0]['labels'].detach().cpu().numpy()
                boxes_w_labels = np.concatenate((boxes, labels.reshape(-1, 1)), axis=1)
                ground_truth_list.append(boxes_w_labels)

                images = list(image.to(self.device) for image in images)

                prediction = self.model(images)[0]  # Assuming only one image was processed

                boxes = prediction['boxes'].detach().cpu().numpy()
                labels = prediction['labels'].detach().cpu().numpy()
                boxes_w_labels = np.concatenate((boxes, labels.reshape(-1, 1)), axis=1)
                prdct_list.append(boxes_w_labels)

        self.model.train()
        return calc_f1_score(ground_truth_list, prdct_list)

    def _save_checkpoint(self, chkpnt_dir_path: str, optimizer, lr_scheduler, loss_history:List, ap_history:List):
        file_name = f'FasterRCNN_{self.backbone}_{self.trainable_backbone_layers}'
        if self.verbose:
            print(f'Storing checkpoint at {chkpnt_dir_path + file_name}. best mAP = {ap_history[-1]}')
        # Save model weights
        torch.save({
            'epoch': self.num_epochs_trained,
            'model_state_dict': self.model.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'loss_history': loss_history,
            'ap_history': ap_history
        }, chkpnt_dir_path + file_name + '.pt')
        # Save info text file
        with open(chkpnt_dir_path + file_name + '.txt', 'w') as f:
            f.write(f'Num of epochs trained = {self.num_epochs_trained}\n')
            f.write(f'Best mAP score = {ap_history[-1]}\n')
            f.write(f'Initial config : lr = {self.initial_lr}, weight_decay = {optimizer.param_groups[0]["weight_decay"]}, '
                    f'lr_scheduler gamma = {lr_scheduler.gamma} step_size = {lr_scheduler.step_size}')

            f.close()

    def _load_from_checkpoint(self, chkpnt_path: str, optimizer, lr_scheduler):
        """
        :param chkpnt_path: full path to the checkpoint dictionary
        """
        checkpoint_dict = torch.load(chkpnt_path)
        self.model.load_state_dict(checkpoint_dict['model_state_dict'])
        optimizer.load_state_dict(checkpoint_dict['optim_state_dict'])
        lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler_state_dict'])

        if self.verbose:
            print(f'Successfully loaded checkpoint from {chkpnt_path}')

    def train(self, num_epochs, optimizer, train_loader, test_loader, lr_scheduler, chkpnt_dir_path=None,
              chkpnt_path = None, tb_writer=None, print_val_loss=False, print_f1_every=None):

        if chkpnt_path is not None:
            # Load model from checkpoint
            self._load_from_checkpoint(chkpnt_path, optimizer, lr_scheduler)

        best_model_wts = copy.deepcopy(self.model.state_dict())

        self.initial_lr = lr_scheduler.get_last_lr()[0]
        loss_history = []
        ap_history = []
        for epoch in range(num_epochs):
            # train for one epoch, printing every 10 iterations
            _, train_loss_iteration = train_one_epoch(self.model, optimizer, train_loader, self.device,
                                                      self.num_epochs_trained, print_freq=10)
            # update the learning rate
            lr_scheduler.step()
            # evaluate on the test dataset
            epoch_train_loss = np.mean(train_loss_iteration)  # mean of the loss values during the epoch.
            mAP_score = 0
            if epoch_train_loss < 5451740.5: # just a big number i saw
                coco_eval = evaluate(self.model, test_loader, device=self.device)
                mAP_score = coco_eval.map_score

            loss_history.append(epoch_train_loss)
            ap_history.append(mAP_score)

            if tb_writer is not None or print_val_loss:
                epoch_val_loss, _ = self._get_val_loss(test_loader)
                if self.verbose:
                    print(f'Epoch #{self.num_epochs_trained} loss(sum of losses): train = {epoch_train_loss}, val = {epoch_val_loss}')
                # Add results to tensor board
                if tb_writer is not None:
                    with tb_writer:
                        tb_writer.add_scalars('Training convergence/',
                                              {'train_loss': epoch_train_loss,
                                               'val_loss': epoch_val_loss}, self.num_epochs_trained)
                        tb_writer.add_scalar('0.75% IOU mAP score/test',
                                             mAP_score, self.num_epochs_trained)

            if print_f1_every is not None and self.num_epochs_trained % print_f1_every == 0:
                val_f1_score = self._get_f1_score(test_loader)
                print(f'Epoch #{self.num_epochs_trained}/{num_epochs} F1 score = {val_f1_score}')

            if mAP_score > self.best_score:
                # Save best model params
                if self.verbose:
                    print(f'current epoch score {mAP_score} > best so far {self.best_score} keeping weights')
                self.best_score = mAP_score
                best_model_wts = copy.deepcopy(self.model.state_dict())
                if chkpnt_dir_path is not None:
                    self._save_checkpoint(chkpnt_dir_path, optimizer, lr_scheduler, loss_history, ap_history)

            self.num_epochs_trained += 1

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        if tb_writer is not None:
            with tb_writer:
                tb_writer.flush()

    def train_simple(self, num_epochs, optimizer, lr_scheduler, writer, data_train_loader,
                     data_val_loader=None, use_fancy_eval=False, print_every=10):
        itr = 1
        for epoch in range(num_epochs):
            epoch_num = self.num_epochs_trained + epoch
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

            print(f"Epoch #{epoch_num + 1}/{self.num_epochs_trained + num_epochs} train_loss: {epoch_train_loss}, val_loss = {epoch_val_loss}")
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