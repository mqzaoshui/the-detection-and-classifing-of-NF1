import os
import cv2
import torch
from pathlib import Path
import sys
# 获取当前文件的路径
FILE = Path(__file__).resolve()
# 获取YOLOv5的根目录
ROOT = FILE.parents[0]
# 如果ROOT不在系统路径中，则添加到系统路径
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
# 将ROOT路径转为相对路径
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# 导入相关模块
from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (non_max_suppression, scale_coords)
from utils.torch_utils import select_device
import numpy as np

# 加载模型的函数
def load_model(weights='./best.pt',
               data=Path(__file__).resolve().parents[0] / 'data/coco128.yaml',
               device='',
               half=False,
               dnn=False):
    # 选择设备，如GPU或CPU
    device = select_device(device)
    # 加载模型
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride = model.stride
    names = model.names
    pt = model.pt
    return model, stride, names, pt

# 运行模型并返回检测结果的函数
def run(model, img, stride, pt, imgsz=(640, 640), conf_thres=0.05, iou_thres=0.10, max_det=1000, device='', classes=None, agnostic_nms=False, augment=False, half=False):
    cal_detect = []
    device = select_device(device)
    # 获取类别名称
    names = model.module.names if hasattr(model, 'module') else model.names

    # 图像预处理：调整尺寸
    im = letterbox(img, imgsz, stride, pt)[0]

    # 图像格式转换
    im = im.transpose((2, 0, 1))[::-1]
    im = np.ascontiguousarray(im)

    # 图像预处理：归一化等操作
    im = torch.from_numpy(im).to(device)
    im = im.half() if half else im.float()
    im /= 255.0
    if len(im.shape) == 3:
        im = im[None]

    # 对图像进行目标检测
    pred = model(im, augment=augment)
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

    # 处理检测结果，如调整坐标
    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], img.shape).round()
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)
                label = f'{names[c]}'
                cal_detect.append([label, xyxy, str(float(conf))[:5]])

    return cal_detect

# 主函数
if __name__ == "__main__":
    # 加载模型
    model, stride, names, pt = load_model()

    test_path = './test'  # 定义test文件夹的路径
    # 遍历test文件夹中的每一张图片
    for img_path in os.listdir(test_path):
        full_path = os.path.join(test_path, img_path)
        if os.path.isfile(full_path):
            img = cv2.imread(full_path)
            # 获取每张图片的检测结果
            detections = run(model, img, stride, pt)

            # 将检测结果绘制到图片上
            for detection in detections:
                label, (x1, y1, x2, y2), conf = detection
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(img, f'{label} {conf}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # 使用OpenCV展示带有检测结果的图片
            img = cv2.resize(img,(0,0),fx=0.3,fy=0.3)
            cv2.imshow('Detection', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
