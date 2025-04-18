import logging
import os
import time

import cv2
import numpy as np
import onnxruntime
import psutil

from utils import xywh2xyxy, draw_detections, multiclass_nms

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


class YOLOv8:

    def __init__(self, path, conf_thres=0.7, iou_thres=0.5):
        logger.info(f"初始化YOLOv8模型 - 配置: conf_thres={conf_thres}, iou_thres={iou_thres}")
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres

        # 记录初始内存使用情况
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        logger.info(f"初始化前内存使用: {mem_before:.2f} MB")

        start_time = time.perf_counter()
        # Initialize model
        self.initialize_model(path)
        init_time = (time.perf_counter() - start_time) * 1000

        # 记录初始化后内存使用情况
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        logger.info(f"初始化后内存使用: {mem_after:.2f} MB (增加: {mem_after - mem_before:.2f} MB)")
        logger.info(f"模型初始化耗时: {init_time:.2f} ms")

    def __call__(self, image):
        logger.info(f"处理图像 - 尺寸: {image.shape}")
        start_time = time.perf_counter()
        result = self.detect_objects(image)
        total_time = (time.perf_counter() - start_time) * 1000
        logger.info(f"总检测时间: {total_time:.2f} ms")
        return result

    def initialize_model(self, path):
        logger.info(f"加载模型: {path}")
        logger.info(f"可用提供程序: {onnxruntime.get_available_providers()}")

        start_time = time.perf_counter()
        self.session = onnxruntime.InferenceSession(path,
                                                    providers=onnxruntime.get_available_providers())
        load_time = (time.perf_counter() - start_time) * 1000
        logger.info(f"模型加载耗时: {load_time:.2f} ms")

        # Get model info
        self.get_input_details()
        self.get_output_details()
        logger.info(f"模型输入尺寸: {self.input_width}x{self.input_height}")

    def detect_objects(self, image):
        # 记录各阶段时间
        timings = {}

        # 准备输入
        start = time.perf_counter()
        input_tensor = self.prepare_input(image)
        timings['prepare_input'] = (time.perf_counter() - start) * 1000

        # 执行推理
        start = time.perf_counter()
        outputs = self.inference(input_tensor)
        timings['inference'] = (time.perf_counter() - start) * 1000

        # 处理输出
        start = time.perf_counter()
        self.boxes, self.scores, self.class_ids = self.process_output(outputs)
        timings['process_output'] = (time.perf_counter() - start) * 1000

        # 记录性能数据
        logger.info(f"性能统计:")
        logger.info(f"  - 准备输入: {timings['prepare_input']:.2f} ms")
        logger.info(f"  - 推理执行: {timings['inference']:.2f} ms")
        logger.info(f"  - 处理输出: {timings['process_output']:.2f} ms")
        logger.info(f"  - 检测到的对象: {len(self.boxes)}")

        return self.boxes, self.scores, self.class_ids

    def prepare_input(self, image):
        start = time.perf_counter()
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 记录颜色转换时间
        color_time = (time.perf_counter() - start) * 1000
        start = time.perf_counter()

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # 记录调整大小时间
        resize_time = (time.perf_counter() - start) * 1000
        start = time.perf_counter()

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        # 记录归一化和转置时间
        norm_time = (time.perf_counter() - start) * 1000

        logger.debug(f"准备输入细节:")
        logger.debug(f"  - 颜色转换: {color_time:.2f} ms")
        logger.debug(f"  - 调整大小: {resize_time:.2f} ms")
        logger.debug(f"  - 归一化和转置: {norm_time:.2f} ms")

        return input_tensor

    def inference(self, input_tensor):
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        inference_time = (time.perf_counter() - start) * 1000

        logger.info(f"推理时间: {inference_time:.2f} ms")
        return outputs

    def process_output(self, output):
        start_total = time.perf_counter()

        # 记录各个处理步骤的时间
        start = time.perf_counter()
        predictions = np.squeeze(output[0]).T
        squeeze_time = (time.perf_counter() - start) * 1000

        # Filter out object confidence scores below threshold
        start = time.perf_counter()
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]
        filter_time = (time.perf_counter() - start) * 1000

        if len(scores) == 0:
            logger.info("没有检测到对象")
            return [], [], []

        # Get the class with the highest confidence
        start = time.perf_counter()
        class_ids = np.argmax(predictions[:, 4:], axis=1)
        class_time = (time.perf_counter() - start) * 1000

        # Get bounding boxes for each object
        start = time.perf_counter()
        boxes = self.extract_boxes(predictions)
        box_time = (time.perf_counter() - start) * 1000

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        start = time.perf_counter()
        indices = multiclass_nms(boxes, scores, class_ids, self.iou_threshold)
        nms_time = (time.perf_counter() - start) * 1000

        total_time = (time.perf_counter() - start_total) * 1000

        logger.debug(f"处理输出细节:")
        logger.debug(f"  - 预处理: {squeeze_time:.2f} ms")
        logger.debug(f"  - 过滤阈值: {filter_time:.2f} ms")
        logger.debug(f"  - 类别识别: {class_time:.2f} ms")
        logger.debug(f"  - 边界框提取: {box_time:.2f} ms")
        logger.debug(f"  - NMS处理: {nms_time:.2f} ms")
        logger.debug(f"  - 总处理时间: {total_time:.2f} ms")
        logger.info(f"检测结果: 找到 {len(indices)} 个对象 (过滤前: {len(scores)})")

        return boxes[indices], scores[indices], class_ids[indices]

    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        start = time.perf_counter()
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)

        extract_time = (time.perf_counter() - start) * 1000
        logger.debug(f"边界框提取时间: {extract_time:.2f} ms")

        return boxes

    def rescale_boxes(self, boxes):
        start = time.perf_counter()
        # Rescale boxes to original image dimensions
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])

        rescale_time = (time.perf_counter() - start) * 1000
        logger.debug(f"边界框缩放时间: {rescale_time:.2f} ms")
        return boxes

    def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):
        start = time.perf_counter()
        result = draw_detections(image, self.boxes, self.scores,
                                 self.class_ids, mask_alpha)
        draw_time = (time.perf_counter() - start) * 1000
        logger.info(f"绘制检测结果耗时: {draw_time:.2f} ms")
        return result

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

        logger.info(f"模型输入详情: names={self.input_names}, shape={self.input_shape}")

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]
        logger.info(f"模型输出详情: names={self.output_names}")
