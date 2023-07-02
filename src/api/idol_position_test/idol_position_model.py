# Face detection
import base64
import uuid
from fastapi import APIRouter, UploadFile
from api.idol_position_test.resnet_model import ResNet
from albumentations.pytorch.transforms import ToTensor
import albumentations as A  # ver 0.5.2
import json
import os
import numpy as np  # 1.24.3
import torch  # ver 2.0.1+cu118
import cv2
seed_num = 1004

# importation
# cv2

# Torch

# PIL & numpy

# Utils

# Transform tool kits

# Resnet model


class idol_position:
    def __init__(self):
        base_path = './src/api/idol_position_test/'
        self.label_map = {'label_1': {'1': '남돌', '2': '여돌'}, 'label_2': {'1': '메인', '2': '서브'}, 'label_3': {
            '1': '보컬', '2': '댄서', '3': '래퍼'}, 'label_4_M': {'1': '비쥬얼', '2': '피지컬', '3': '예능'}, 'label_4_W': {'1': '큐티', '2': '섹시', '3': '청순'}}
        self.idx2label = []
        with open(f"{base_path}label_info.json", "r") as read_file:
            class_idx = json.load(read_file)
            self.idx2label = [class_idx[str(k)][1]
                              for k in range(len(class_idx))]
        self.model = ResNet(num_classes=36)
        self.model.load_state_dict(torch.load(
            './asset/model_weights(ResNet_IDOL).pt', map_location=torch.device('cpu')))
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    async def face_detector(self, image_dir, img_size=(224, 224), scaleFactor=1.3, minNeighbors=5):
        image_type = image_dir.split('.')[-1]
        if image_type not in ('jpg', 'jpeg', 'png'):
            return image_type
        if image_type == 'png':
            img = cv2.imread(image_dir, cv2.IMREAD_UNCHANGED)
            img = img[:, :, :3]
        else:
            img = cv2.imread(image_dir)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor, minNeighbors)
        if faces == ():
            return 0
        x_from, _, _, _ = np.min(faces, axis=0)
        _, y_from, x_length, y_length = np.max(faces, axis=0)
        img = img[y_from: y_from + y_length, x_from: x_from + x_length]
        img = cv2.resize(
            img, (img_size[0], img_size[1]), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    async def predict_label(self, img_dir):
        img = await self.face_detector(img_dir)
        if type(img) == int:
            return 0
        img = cv2.cvtColor(np.array(img), cv2.COLOR_GRAY2RGB)
        img_transform = A.Compose([A.HorizontalFlip(), ToTensor()])
        img = img_transform(image=img)['image']
        img = img.reshape(-1, 3, 224, 224)
        output = self.model(img)
        pred_label = np.argmax(output[0].detach().numpy())
        return pred_label

    async def label_mapping(self, img_dir):
        pred_label = await self.predict_label(img_dir)
        if pred_label == 0:
            return 0
        return self.idx2label[pred_label]

    async def label_converter(self, img_dir):
        label_num = await self.label_mapping(img_dir)
        if label_num == 0:
            return 0, 0, 0, 0, 0
        la_1, la_2, la_3, la_4 = list(label_num)
        gender = self.label_map['label_1'][la_1]
        title = self.label_map['label_2'][la_2]
        main_position = self.label_map['label_3'][la_3]
        if la_1 == '1':
            sub_position = self.label_map['label_4_M'][la_4]
        else:
            sub_position = self.label_map['label_4_W'][la_4]
        return gender, title, main_position, sub_position, label_num


class response_generator:
    def __init__(self):
        self.mapper = {}
        self.result_image_path = './static/Idol_position_test/Matching_img'

    async def __call__(self, g, t, m, s, label_num):
        if g == 0:
            matching_image = cv2.imread(f'{self.result_image_path}/0.jpg')
            msg = '이게 얼굴이라고? ㅈㄹ ㄴㄴ'
        else:
            #matching_image_path =f'{self.result_image_path}/{label_num}.jpg'
            matching_image = cv2.imread(f'{self.result_image_path}/1111.jpg')
            msg = f'당신은 {g} {t} {m} {s}'
        matching_image = await self.img_cnv_bite(matching_image)
        return {'message': msg, 'image': matching_image}

    async def img_cnv_bite(self, img):
        _, encoded_image = cv2.imencode('.jpg', np.array(img))
        b64_string = base64.b64encode(encoded_image).decode('utf-8')
        return b64_string


router = APIRouter()


@router.post("/get_idol_position")
async def upload_result(file: UploadFile):
    get_position = idol_position()
    UPLOAD_DIR = './temp'
    content = await file.read()
    file_path = os.path.join(UPLOAD_DIR, str(
        uuid.uuid4()) + '.' + file.filename.split('.')[-1])
    with open(file_path, 'wb') as fp:
        fp.write(content)
    g, t, m, s, label_num = await get_position.label_converter(file_path)
    rg = response_generator()
    result = await rg(g, t, m, s, label_num)
    return result


# http://127.0.0.1:8000/docs#/default
