import logging
import os
import sys
import time
import logging
import os
import sys
import time
from argparse import ArgumentParser

import cv2
import numpy as np
import onnxruntime
from insightface.app import FaceAnalysis

from YOLOv8 import YOLOv8
import cv2
import numpy as np
import onnxruntime
import psutil

from YOLOv8 import YOLOv8
from utils import xywh2xyxy, draw_detections, multiclass_nms

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("detect_performance.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("YOLOv8-Performance")

# 获取应用程序的基本路径
def resource_path(relative_path):
    """ 获取资源的绝对路径 """
    try:
        # PyInstaller 创建临时文件夹并将路径存储在 _MEIPASS 中
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

class AiDetect:

    def __init__(self):
        model_path = resource_path("models/yolo/yolov8n.onnx")
        logger.info(f"yolo使用模型: {model_path}")
        print(f"model_path文件是否存在: {os.path.exists(model_path)}")
        self.object_detector = YOLOv8(model_path, conf_thres=0.3, iou_thres=0.5)

        face_model_path = resource_path('')
        print('face_model_path=', face_model_path)
        # 初始化人脸分析应用
        self.face_analysis = FaceAnalysis('buffalo_sc', root=face_model_path, providers=onnxruntime.get_available_providers())
        self.face_analysis.prepare(ctx_id=0, det_size=(640, 640))

    def __call__(self, image):
        logger.info(f"处理图像 - 尺寸: {image.shape}")
        start_time = time.perf_counter()
        boxes, scores, class_ids = self.object_detector(image)
        face_result = self.face_analysis.get(image)
        total_time = (time.perf_counter() - start_time) * 1000
        logger.info(f"总检测时间: {total_time:.2f} ms yolo检出{len(class_ids)}个 face检出{len(face_result)}个")
        return {boxes, scores, class_ids}, face_result
