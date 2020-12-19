import numpy as np
from typing import List, Dict
import matplotlib.image as mpimg
import cv2

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


def draw_boxes(boxes, classes, labels, image):
    # read the image with OpenCV
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
    for i, box in enumerate(boxes):
        color = [0, 255, 0]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )
        cv2.putText(image, classes[i], (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                    lineType=cv2.LINE_AA)

    return image
