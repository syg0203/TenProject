from fastapi import APIRouter, UploadFile
import cv2
import io
from PIL import Image
import numpy as np
import os
import base64
from tensorflow.keras.models import load_model
import sys

root_path = os.getcwd()
sys.path.insert(1, root_path)

from modules import common

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class faceage:
    '''
    1. 5살 간격으로 라벨 세분화 필요
    2. 모델 고도화 필요
    3. 코드 정리 필요

    '''

    def __init__(self):
        self.label_li = [
            '당신의 얼굴 나이는 : 0세~6세',
            '당신의 얼굴 나이는 : 7세~12세',
            '당신의 얼굴 나이는 : 13세~19세',
            '당신의 얼굴 나이는 : 20세~30세',
            '당신의 얼굴 나이는 : 31세~45세',
            '당신의 얼굴 나이는 : 46세~55세',
            '당신의 얼굴 나이는 : 56세~66세',
            '당신의 얼굴 나이는 : 67세~80세'
        ]
        self.model = load_model(
            "./asset/faceage_model.hdf5")
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  # 얼굴 검출기 모델

    async def imagedown_async(self, face):
        # 이미지 정규화 및 예측 함수
        face = (face / 255.0)
        face = np.expand_dims(face, axis=0)

        y_pred = self.model.predict(face)
        return y_pred[0]

    async def predict_batch(self, image):
        results = await self.imagedown_async(image)

        predict_arr = np.argmax(results, axis=0)
        recommend = self.label_li[predict_arr]
        return recommend

    async def predict(self, image):
        # byte numpy 변환 및 검출기 실행
        img_byte = np.array(Image.open(io.BytesIO(image)))
        faces = self.face_cascade.detectMultiScale(img_byte, 1.3, 5)
        img_byte = cv2.cvtColor(img_byte, cv2.COLOR_BGR2RGB)
        # 검출기 예외처리
        if faces == ():
            _, img_with_box = cv2.imencode('.jpg', img_byte)
            img_with_box = base64.b64encode(img_with_box).decode('utf-8')
            return {"classification_age": "얼굴이 검출되지 않았습니다.", "face_byte": img_with_box}

        x_from, _, _, _ = np.min(faces, axis=0)
        _, y_from, x_length, y_length = np.max(faces, axis=0)
        face = img_byte[y_from: y_from + y_length, x_from: x_from + x_length]
        face = cv2.resize(face, (128, 128), interpolation=cv2.INTER_CUBIC)

        # 검출된 얼굴 나이 예측
        get_pb = await self.predict_batch(face)

        # 원본이미지 검출된 얼굴 사각형표시
        img_with_box = img_byte.copy()
        img_with_box = cv2.rectangle(img_with_box, (x_from, y_from), (x_from +
                                                                      x_length, y_from + y_length), (0, 0, 255), 3)
        _, img_with_box = cv2.imencode('.jpg', img_with_box)
        img_with_box = base64.b64encode(img_with_box).decode('utf-8')
        result = {"classification_age": get_pb, "face_byte": img_with_box}
        return result


route = APIRouter()

@route.post("/faceage_router")
async def upload_photo(file: UploadFile):
    try:
        faceagecls = faceage()
        content = await file.read()

        json_string = await faceagecls.predict(content)
        return json_string
    
    except Exception as e:
        common.exception_func(e, "faceage", yg=False)

    finally:
        del faceagecls