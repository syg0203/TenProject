from io import BytesIO
from PIL import Image
import base64
from fastapi import APIRouter, UploadFile
from albumentations.pytorch.transforms import ToTensor
import albumentations as A  # ver 0.5.2
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
        layers.append(nn.Conv2d(in_dim, mid_dim,
                      kernel_size=1, stride=2, padding=0))
    else:
        layers.append(nn.Conv2d(in_dim, mid_dim,
                      kernel_size=1, stride=1, padding=0))
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
    def __init__(self, in_dim, mid_dim, out_dim, down: bool = False, starting: bool = False) -> None:
        super(Bottleneck, self).__init__()
        if starting:
            down = False
        self.block = bottleneck_block(in_dim, mid_dim, out_dim, down=down)
        self.relu = nn.ReLU(inplace=True)
        if down:
            conn_layer = nn.Conv2d(
                in_dim, out_dim, kernel_size=1, stride=2, padding=0)
        else:
            conn_layer = nn.Conv2d(
                in_dim, out_dim, kernel_size=1, stride=1, padding=0)
        self.changedim = nn.Sequential(conn_layer, nn.BatchNorm2d(out_dim))

    def forward(self, x):
        identity = self.changedim(x)
        x = self.block(x)
        x += identity
        x = self.relu(x)
        return x


def make_layer(in_dim, mid_dim, out_dim, repeats, starting=False):
    layers = []
    layers.append(Bottleneck(in_dim, mid_dim, out_dim,
                  down=True, starting=starting))
    for _ in range(1, repeats):
        layers.append(Bottleneck(out_dim, mid_dim, out_dim, down=False))
    return nn.Sequential(*layers)


class ResNet(nn.Module):
    def __init__(self, repeats: list = [3, 4, 6, 3], num_classes=36):
        super(ResNet, self).__init__()
        self.num_classes = num_classes
        self.conv1 = conv_start()

        base_dim = 64
        self.conv2 = make_layer(
            base_dim, base_dim, base_dim*4, repeats[0], starting=True)
        self.conv3 = make_layer(base_dim*4, base_dim*2, base_dim*8, repeats[1])
        self.conv4 = make_layer(base_dim*8, base_dim*4,
                                base_dim*16, repeats[2])
        self.conv5 = make_layer(base_dim*16, base_dim*8,
                                base_dim*32, repeats[3])

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
        self.label_map = {'label_1': {'1': 'main', '2': 'sub'},
                          'label_2': {'1': 'vocal', '2': 'dancer', '3': 'rapper'},
                          'label_3': {'1': 'cute', '2': 'sexy', '3': 'pure'}}

        self.idx2label = ['221', '122', '233', '113', '211', '131',
                          '132', '213', '232', '212', '133', '112',
                          '222', '223', '121', '231', '123', '111']

        self.model = ResNet(repeats=[3, 4, 23, 3], num_classes=18)
        self.model.load_state_dict(torch.load(
            './asset/idolposition_model.pt', map_location=torch.device('cpu')))
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
            return 0, 0, 0, 0
        la_1, la_2, la_3 = list(label_num)
        label_1 = self.label_map['label_1'][la_1]
        label_2 = self.label_map['label_2'][la_2]
        label_3 = self.label_map['label_3'][la_3]
        return label_1, label_2, label_3, label_num


class response_generator:
    def __init__(self):
        self.mapper = {'label_1': {'main': '메인', 'sub': '서브'},
                       'label_2': {'vocal': '보컬', 'dancer': '댄서', 'rapper': '래퍼'},
                       'label_3': {'cute': '큐티', 'sexy': '섹시', 'pure': '청순'}}

    async def __call__(self, la_1, la_2, la_3, label_num, image):
        if la_1 == 0:
            matching_image = image
            msg = '당신은 데뷔가 불가능 합니다~!!! 현생에 충실하세요~! >.<'
        else:
            matching_image = image
            matching_image = await self.stamp_image(la_1, matching_image, pos=(2, 1))
            matching_image = await self.stamp_image(la_2, matching_image, pos=(1, 1))
            matching_image = await self.stamp_image(la_3, matching_image, pos=(1, 2))
            la_1 = self.mapper['label_1'][la_1]
            la_2 = self.mapper['label_2'][la_2]
            la_3 = self.mapper['label_3'][la_3]
            msg = f'당신은 {la_1} {la_2} {la_3}'
        matching_image = await self.img_cnv_bite(matching_image)
        return {'message': msg, 'image': matching_image}

    async def img_cnv_bite(self, img):
        _, encoded_image = cv2.imencode('.jpg', np.array(img))
        b64_string = base64.b64encode(encoded_image).decode('utf-8')
        return b64_string

    async def stamp_image(self, stamp, img, pos):
        pos_x, pos_y = pos
        stamp = cv2.imread(f'./asset/stamps/idolposition/{stamp}.jpg')
        img_length, img_hight, _ = img.shape
        stamp_ratio = round(img_length/5)
        stamp = cv2.resize(stamp, dsize=[stamp_ratio, stamp_ratio])
        rows, cols, _ = stamp.shape
        h_margin, w_margin = (pos_y-1)*stamp_ratio, img_hight - pos_x*(cols)
        roi = img[h_margin: rows + h_margin, w_margin: cols + w_margin]
        img2gray = cv2.cvtColor(stamp, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(img2gray, 170, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        img_bg = cv2.bitwise_and(roi, roi, mask=mask)
        stamp_fg = cv2.bitwise_and(stamp, stamp, mask=mask_inv)
        dst = cv2.add(img_bg, stamp_fg)
        img[h_margin:rows+h_margin, w_margin:cols+w_margin] = dst
        return img


# API
router = APIRouter()


@router.post("/get_idol_position")
async def upload_result(file: UploadFile):
    get_position = idol_position()
    content = await file.read()
    content = np.array(Image.open(BytesIO(content)))[:, :, ::-1]
    la_1, la_2, la_3, label_num = await get_position.label_converter(content)
    rg = response_generator()
    result = await rg(la_1, la_2, la_3, label_num, content)
    return result

# http://127.0.0.1:8000/docs#/default
