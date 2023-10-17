# test.py xyh
import os
import cv2
import torch
from pathlib import Path
import sys
# Get the current file path
FILE = Path(__file__).resolve()
# Get the root directory of YOLOv5
ROOT = FILE.parents[0]
# Add ROOT to system path if it's not already included
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
# Convert ROOT path to relative path
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# Import related modules
from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (non_max_suppression, scale_coords)
from utils.torch_utils import select_device
import numpy as np

# Function to load the model
def load_model(weights='./best.pt',
               data=Path(__file__).resolve().parents[0] / 'data/coco128.yaml',
               device='',
               half=False,
               dnn=False):
    # Select the device, e.g., GPU or CPU
    device = select_device(device)
    # Load the model
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride = model.stride
    names = model.names
    pt = model.pt
    return model, stride, names, pt

# Function to run the model and return detection results
def run(model, img, stride, pt, imgsz=(640, 640), conf_thres=0.05, iou_thres=0.10, max_det=1000, device='', classes=None, agnostic_nms=False, augment=False, half=False):
    cal_detect = []
    device = select_device(device)
    # Get class names
    names = model.module.names if hasattr(model, 'module') else model.names

    # Image preprocessing: resize
    im = letterbox(img, imgsz, stride, pt)[0]

    # Convert image format
    im = im.transpose((2, 0, 1))[::-1]
    im = np.ascontiguousarray(im)

    # Image preprocessing: normalization, etc.
    im = torch.from_numpy(im).to(device)
    im = im.half() if half else im.float()
    im /= 255.0
    if len(im.shape) == 3:
        im = im[None]

    # Perform object detection on the image
    pred = model(im, augment=augment)
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

    # Process detection results, e.g., adjust coordinates
    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], img.shape).round()
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)
                label = f'{names[c]}'
                cal_detect.append([label, xyxy, str(float(conf))[:5]])

    return cal_detect

# Main function
if __name__ == "__main__":
    # Load the model
    model, stride, names, pt = load_model()

    test_path = './test'  # Define the path of the test folder
    # Iterate through each image in the test folder
    for img_path in os.listdir(test_path):
        full_path = os.path.join(test_path, img_path)
        if os.path.isfile(full_path):
            img = cv2.imread(full_path)
            # Get detection results for each image
            detections = run(model, img, stride, pt)

            # Draw detection results on the image
            for detection in detections:
                label, (x1, y1, x2, y2), conf = detection
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(img, f'{label} {conf}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Display the image with detection results using OpenCV
            img = cv2.resize(img,(0,0),fx=0.3,fy=0.3)
            cv2.imshow('Detection', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
