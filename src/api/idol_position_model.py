import base64
import uuid
from fastapi import APIRouter, UploadFile
from albumentations.pytorch.transforms import ToTensor
import albumentations as A  # ver 0.5.2
import json
import os
import numpy as np  # 1.24.3
import torch  # ver 2.0.1+cu118
import torch.nn as nn
import cv2
seed_num = 1004

# Resnet Model
def conv_start():
    return nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=4),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
    )

def bottleneck_block(in_dim, mid_dim, out_dim, down=False):
    layers = []
    if down:
        layers.append(nn.Conv2d(in_dim, mid_dim, kernel_size=1, stride=2, padding=0))
    else:
        layers.append(nn.Conv2d(in_dim, mid_dim, kernel_size=1, stride=1, padding=0))
    layers.extend([
        nn.BatchNorm2d(mid_dim),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_dim, mid_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(mid_dim),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_dim, out_dim, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(out_dim),
    ])
    return nn.Sequential(*layers)

class Bottleneck(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim, down:bool = False, starting:bool=False) -> None:
        super(Bottleneck, self).__init__()
        if starting:
            down = False
        self.block = bottleneck_block(in_dim, mid_dim, out_dim, down=down)
        self.relu = nn.ReLU(inplace=True)
        if down:
            conn_layer = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=2, padding=0)
        else:
            conn_layer = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0)
        self.changedim = nn.Sequential(conn_layer, nn.BatchNorm2d(out_dim))

    def forward(self, x):
        identity = self.changedim(x)
        x = self.block(x)
        x += identity
        x = self.relu(x)
        return x

def make_layer(in_dim, mid_dim, out_dim, repeats, starting=False):
        layers = []
        layers.append(Bottleneck(in_dim, mid_dim, out_dim, down=True, starting=starting))
        for _ in range(1, repeats):
            layers.append(Bottleneck(out_dim, mid_dim, out_dim, down=False))
        return nn.Sequential(*layers)

class ResNet(nn.Module):
    def __init__(self, repeats:list = [3,4,6,3], num_classes=36):
        super(ResNet, self).__init__()
        self.num_classes = num_classes
        self.conv1 = conv_start()
        
        base_dim = 64
        self.conv2 = make_layer(base_dim, base_dim, base_dim*4, repeats[0], starting=True)
        self.conv3 = make_layer(base_dim*4, base_dim*2, base_dim*8, repeats[1])
        self.conv4 = make_layer(base_dim*8, base_dim*4, base_dim*16, repeats[2])
        self.conv5 = make_layer(base_dim*16, base_dim*8, base_dim*32, repeats[3])
        
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.classifer1 = nn.Linear(2048, 1024)
        self.dropuot_1 = nn.Dropout(p=0.5)
        self.classifer2 = nn.Linear(1024, 512)
        self.dropuot_2 = nn.Dropout(p=0.4)
        self.classifer3 = nn.Linear(512, 256)
        self.dropuot_3 = nn.Dropout(p=0.3)
        self.classifer4 = nn.Linear(256, 128)
        self.dropuot_4 = nn.Dropout(p=0.2)        
        self.classifer5 = nn.Linear(128, 64)
        self.classifer_out = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifer1(x)
        x = self.dropuot_1(x)
        x = self.classifer2(x)
        x = self.dropuot_2(x)
        x = self.classifer3(x)
        x = self.dropuot_3(x)
        x = self.classifer4(x)
        x = self.dropuot_4(x)
        x = self.classifer5(x)
        x = self.classifer_out(x)
        return x
    



# Pipe line
class idol_position:
    def __init__(self):
        self.label_map = {'label_1': {'1': '남돌', '2': '여돌'}, 'label_2': {'1': '메인', '2': '서브'}, 'label_3': {
            '1': '보컬', '2': '댄서', '3': '래퍼'}, 'label_4_M': {'1': '비쥬얼', '2': '피지컬', '3': '예능'}, 'label_4_W': {'1': '큐티', '2': '섹시', '3': '청순'}}
        self.idx2label = ['1111', '1112', '1113', '1121', 
                          '1122', '1123', '1131', '1132', 
                          '1133', '1211', '1212', '1213', 
                          '1221', '1222', '1223', '1231', 
                          '1232', '1233', '2111', '2112', 
                          '2113', '2121', '2122', '2123', 
                          '2131', '2132', '2133', '2211', 
                          '2212', '2213', '2221', '2222', 
                          '2223', '2231', '2232', '2233']
        self.model = ResNet(num_classes=36)
        self.model.load_state_dict(torch.load(
            './asset/model_weights(ResNet_IDOL).pt', map_location=torch.device('cpu')))
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    async def face_detector(self, img, img_size=(224, 224), scaleFactor=1.3, minNeighbors=5):
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

    async def __call__(self, g, t, m, s, label_num, image):
        if g == 0:
            matching_image = cv2.imread(f'{self.result_image_path}/0.jpg')
            msg = '이게 얼굴이라고? ㅈㄹ ㄴㄴ'
        else:
            matching_image = image
            msg = f'당신은 {g} {t} {m} {s}'
        matching_image = await self.img_cnv_bite(matching_image)
        return {'message': msg, 'image': matching_image}

    async def img_cnv_bite(self, img):
        _, encoded_image = cv2.imencode('.jpg', np.array(img))
        b64_string = base64.b64encode(encoded_image).decode('utf-8')
        return b64_string




# API 
router = APIRouter()

from PIL import Image
from io import BytesIO

@router.post("/get_idol_position")
async def upload_result(file: UploadFile):
    get_position = idol_position()
    content = await file.read()
    content = np.array(Image.open(BytesIO(content)))[:,:,::-1]
    g, t, m, s, label_num = await get_position.label_converter(content)
    rg = response_generator()
    result = await rg(g, t, m, s, label_num, content)
    return result


# http://127.0.0.1:8000/docs#/default













##### Stamp processing
import cv2

# 스탬프 이미지 불러오기
stamp = cv2.imread('C:/ALL/TEN_PTOJECT_V2/Stamp/stamp_0.jpg')
# 베이스 이미지 불러오기
img = cv2.imread('C:/ALL/TEN_PTOJECT_V2/TEST_IMAGES/KakaoTalk_20230611_211203709_04.jpg')

# 마진 설정
h_margin_t, w_margin_l = 10,10

# 스탬프 크기 정의
rows, cols, channels = stamp.shape

# 배경이미지에서 스탬프가 들어갈 위치 추출
roi = img[h_margin_t : rows + h_margin_t, w_margin_l : cols + w_margin_l]

# 스탬프 흑백으로 바꿔주기
img2gray = cv2.cvtColor(stamp, cv2.COLOR_BGR2GRAY)
# 임계치 정의(알지비값> 100,255 사이)및 임계치 사이의 값만 남김
ret, mask = cv2.threshold(img2gray, 100, 255, cv2.THRESH_BINARY)
# 색상 반전
mask_inv = cv2.bitwise_not(mask)

## 아래 두작엄은 공통적으로 마스크의 검정색 부분이 날라감
# 원본 이미지에 빈공간 만들기 (이미지와 마스크가 겹치는 부분 공백으로)
img_bg = cv2.bitwise_and(roi,roi, mask = mask)
# 스탬프이미지 누끼따기
stamp_fg = cv2.bitwise_and(stamp, stamp, mask = mask_inv)

# 누끼 따진 스탬프와 스탬프 자리 비워진 원본 이미지(위에 원본에서 추출된 일부분) 합치기
dst = cv2.add(img_bg, stamp_fg)
# 합쳐진 이미지 일부분 원본이미지에 바꿔 넣기
img[h_margin_t:rows+h_margin_t, w_margin_l:cols+w_margin_l] = dst


cv2.imshow('result', img_bg)
cv2.waitKey(0)
cv2.destroyAllWindows()