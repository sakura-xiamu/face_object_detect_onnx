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

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("yolov8_performance.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("YOLOv8-Performance")

# Parse arguments from user input.
parser = ArgumentParser()
parser.add_argument("--video", type=str, default=None,
                    help="Video file to be processed.")
parser.add_argument("--cam", type=int, default=0,
                    help="The webcam index.")
args = parser.parse_args()

print(__doc__)
print("OpenCV version: {}".format(cv2.__version__))


# 获取应用程序的基本路径
def resource_path(relative_path):
    """ 获取资源的绝对路径 """
    try:
        # PyInstaller 创建临时文件夹并将路径存储在 _MEIPASS 中
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


def cosine_similarity(x, y):
    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return abs(num / denom)


# 获取应用程序的基本路径
def resource_path(relative_path):
    """ 获取资源的绝对路径 """
    try:
        # PyInstaller 创建临时文件夹并将路径存储在 _MEIPASS 中
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


def run():
    # Before estimation started, there are some startup works to do.

    # Initialize the video source from webcam or video file.
    video_src = args.cam if args.video is None else args.video
    cap = cv2.VideoCapture(video_src)
    print(f"Video source: {video_src}")

    # Get the frame size. This will be used by the following detectors.
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Measure the performance with a tick meter.
    tm = cv2.TickMeter()

    model_path = resource_path("models/yolo/yolov8n.onnx")
    logger.info(f"yolo使用模型: {model_path}")
    print(f"model_path文件是否存在: {os.path.exists(model_path)}")
    yolov8_detector = YOLOv8(model_path, conf_thres=0.3, iou_thres=0.5)

    face_model_path = resource_path('')
    print('face_model_path=', face_model_path)
    # 初始化人脸分析应用
    app = FaceAnalysis('buffalo_sc', root=face_model_path, providers=onnxruntime.get_available_providers())
    app.prepare(ctx_id=0, det_size=(640, 640))

    # Now, let the frames flow.
    while True:
        start_time = time.perf_counter()

        # Read a frame.
        frame_got, frame = cap.read()
        if frame_got is False:
            break

        # If the frame comes from webcam, flip it so it looks like a mirror.
        if video_src == 0:
            frame = cv2.flip(frame, 2)

        # Detect Objects
        boxes, scores, class_ids = yolov8_detector(frame)
        print(f'yolo_result识别出了{len(class_ids)}个物体')

        # Step 1: Get faces from current frame.
        faces = app.get(frame)

        print(f'识别出了{len(faces)}个脸')
        load_time = (time.perf_counter() - start_time) * 1000
        print(f"模型执行单次耗时: {load_time:.2f} ms")
        time.sleep(0.4)


if __name__ == '__main__':
    run()
