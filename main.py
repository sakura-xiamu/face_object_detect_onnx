import os
from io import BytesIO

import cv2
import numpy as np
import uvicorn
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException

from ai_detect import AiDetect

# 创建 FastAPI 应用实例
app = FastAPI()

# 创建上传目录
UPLOAD_DIR = "uploads"
IMAGE_DIR = os.path.join(UPLOAD_DIR, "images")
VIDEO_DIR = os.path.join(UPLOAD_DIR, "videos")

# 确保目录存在
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

# 允许的文件类型
ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/gif", "image/webp"]
ALLOWED_VIDEO_TYPES = ["video/mp4", "video/mpeg", "video/avi", "video/quicktime", "video/x-matroska"]

# 全局变量存储对象
ObjectDetect: AiDetect


# 将PIL图像转换为OpenCV格式
def pil_to_cv2(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


# 定义根路由
@app.get("/")
def read_root():
    return {"Hello": "World"}


# 定义带参数的路由
@app.get("/init")
def init():
    global ObjectDetect
    ObjectDetect = AiDetect()


# 定义 POST 请求路由
@app.post("/face/analysis")
async def face_analysis(file: UploadFile = File(...)):
    try:
        """
            上传图片并返回 base64 编码
            """
        # 检查文件类型
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="文件必须是图片格式")

        # 读取图片内容
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        # 转换为OpenCV格式
        frame = pil_to_cv2(image)
        for i in 100:
            detect_result, face_result = ObjectDetect(frame)

        return {
            "detect_result": detect_result,
            "face_result": face_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理图片时出错: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
