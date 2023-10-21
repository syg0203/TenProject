from fastapi import APIRouter, UploadFile
import cv2
import io
from PIL import Image
import numpy as np
import os
from tensorflow.keras.models import load_model

class faceage:
    def __init__(self):
        self.label_li=[
            '0-6 years old',
            '7-12 years old',
            '13-19 years old',
            '20-30 years old',
            '31-45 years old',
            '46-55 years old',
            '56-66 years old',
            '67-80 years old'
        ]
        self.model = load_model("./asset/FaceClassification_CP128_2023-10-16.hdf5")
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
    async def imagedown_async(self, img_byte):
        # 이미지 정규화 및 예측 함수
        img_byte=np.array(Image.open(io.BytesIO(img_byte)))[:, :, ::-1]
        faces = self.face_cascade.detectMultiScale(img_byte, 1.3, 5)

        img = Image.open(faces).resize((128, 128))
        img = img.convert("RGB")

        x = np.array(img) / 255.0
        x = np.expand_dims(x, axis=0)

        y_pred = self.model.predict(x, verbose=0)
        return y_pred[0]

    async def predict_batch(self, image):
        results = await self.imagedown_async(image)

        predict_arr = np.argmax(results, axis=0)
        recommend = self.label_li[predict_arr]
        return recommend
    
    async def predict(self, image):
        get_pb = await self.predict_batch(image)
        result = {"classification_age": get_pb}
        return result

route = APIRouter()


@route.post("/faceage")
async def upload_photo(file: UploadFile):
    try:
        # Doraemong 클래스의 인스턴스 생성
        faceagecls = faceage()

        # 업로드된 파일의 내용 읽기
        content = await file.read()

        # DoraemonG 모델을 사용하여 예측 수행
        json_string = await faceagecls.predict(content)
        result = {
            "recommend": json_string,
        }
        return result

    finally:
        del faceagecls