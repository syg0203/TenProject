from io import BytesIO
from PIL import Image
import base64
from fastapi import APIRouter, UploadFile
import numpy as np  # 1.24.3
import torch  # ver 2.0.1+cu118
import torch.nn as nn
import cv2
import torch.nn.functional as F
import mediapipe as mp
import math
import asyncio

# whos the king model
class wtk_model(nn.Module):
    def __init__(self):
        super().__init__()        # input shape (478,478)
        self.conv_1st = nn.Conv2d(1, 3, 2**8, padding = 10)   # out shape (243,243)
        self.pool_1st = nn.MaxPool2d((9,9),stride = 2) # out shape (118,118)
        self.conv_2nd = nn.Conv2d(3,3,2**6, padding = 8) # out shape (65,65)
        self.pool_2nd = nn.MaxPool2d((5,5), stride = 5) # out shape (14,14)
        self.ffn_1st = nn.Linear(3*14*14, 2**7)
        self.ffn_2nd = nn.Linear(2**7, 2**5)
        self.ffn_fin = nn.Linear(2**5, 4)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = self.pool_1st(F.relu(self.conv_1st(x)))
        x = self.pool_2nd(F.relu(self.conv_2nd(x)))
        x = torch.flatten(x,1)
        x = self.dropout(F.relu(self.ffn_1st(x)))
        x = self.dropout(F.relu(self.ffn_2nd(x)))
        x = self.ffn_fin(x)
        return x 



# data parsing
class data_parser :
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks = True, static_image_mode = True, max_num_faces = 4)
    
    async def __call__(self, img):
        result = await self.generate_face_mesh(img)
        if result == '-1':
            return '-1', '-1'
        position_holder = await self.get_data_for_stamping(result, (img.shape[0], img.shape[1]))
        result = await self.get_landmarks_by_face(result)
        data_holder = [await self.generate_matrix(result[key]) for key in result.keys()]
        data_holder = np.array(data_holder)
        return np.expand_dims(data_holder, axis = 1), position_holder

    async def get_length(self, x,y,z):
        return math.sqrt((x[1]-x[0])**2+(y[1]-y[0])**2+(z[1]-z[0])**2)

    async def generate_face_mesh(self, img):
        mesh_data = self.face_mesh.process(img)
        return '-1' if mesh_data.multi_face_landmarks == None else mesh_data

    async def get_landmarks_by_face(self, mesh_data):
        landmark_holder = dict()
        for i, j in enumerate(mesh_data.multi_face_landmarks):
            landmark_holder[str(i+1)] = j.landmark
        return landmark_holder
    
    async def generate_matrix(self, landmark_info):
        cal_arr = list()
        for i in range(len(landmark_info)):
            one, temp_arr = landmark_info[i], list()
            for j in range(len(landmark_info)):
                two = landmark_info[j]
                temp_arr.append(await self.get_length([one.x, two.x], [one.y, two.y], [one.z, two.z]))
            cal_arr.append(temp_arr)
        return cal_arr
    
    async def get_data_for_stamping(self,mesh_data, act_img_shape):
        position_holder = dict()
        act_y, act_x =  act_img_shape
        for person_idx in range(len(mesh_data.multi_face_landmarks)):
            middle_x = mesh_data.multi_face_landmarks[person_idx].landmark[151].x
            middle_y = mesh_data.multi_face_landmarks[person_idx].landmark[151].y
            left_x = mesh_data.multi_face_landmarks[person_idx].landmark[162].x
            right_x = mesh_data.multi_face_landmarks[person_idx].landmark[389].x
            bottom_x = mesh_data.multi_face_landmarks[person_idx].landmark[199].x
            bottom_y = mesh_data.multi_face_landmarks[person_idx].landmark[199].y
            middle_x, middle_y, left_x, right_x, bottom_x, bottom_y = int(middle_x * act_x), int(middle_y * act_y), math.floor(left_x * act_x), math.ceil(right_x * act_x), int(bottom_x * act_x), int(bottom_y * act_y)
            deg = (math.atan2(bottom_x - middle_x, bottom_y - middle_y)*180)/math.pi
            position_holder[person_idx] = [middle_y, left_x, right_x, round(deg,2)]
        return position_holder


  
class proba_generator:
    def __init__(self):
        self.model = wtk_model()
        self.model.load_state_dict(torch.load('./asset/wtk_model_weights.pt', map_location=torch.device('cpu')))

    async def predict_all(self, whole_data):
        proba_holder = [await self.predict(single_data) for single_data in whole_data]
        return proba_holder

    async def predict(self, single_data):
        single_data = np.expand_dims(single_data, axis=0)
        single_data = torch.tensor(single_data, dtype=torch.float32)

        with torch.no_grad(): 
            proba = nn.Softmax(dim=1)(self.model(single_data))
        return proba.numpy()
    

class response_generator:
    def __init__(self, original_img):
        self.img = original_img
        self.imgmapper = { 
            0:'./static/whostheking/STAMPS/king.jpeg',
            1:'./static/whostheking/STAMPS/noble.jpeg',
            2:'./static/whostheking/STAMPS/commoner.jpeg',
            3:'./static/whostheking/STAMPS/slave.jpeg'
            }

    async def rank_sorter(self, proba_list):
        rank_dict = {0:None,1:None,2:None,3:None}
        for label_num in range(0,4):    
            ordered_list = [proba_list[person][0][label_num] for person in range(len(proba_list))]
            temp_list = sorted(ordered_list)
            while True:
                if len(temp_list) == 0:
                    break
                cur_high = temp_list.pop()
                cur_person = ordered_list.index(cur_high)
                if rank_dict[cur_person] == None:
                    rank_dict[cur_person] = label_num
                    break
        return rank_dict

    async def stamp_img(self, rank_dict, position_info):
        for person_idx in position_info.keys():
            psh = position_info[person_idx]
            stamp = cv2.imread(self.imgmapper[rank_dict[person_idx]])
            middle_y, lx, rx, deg = psh
            target_length = rx - lx
            ratio = target_length/stamp.shape[1]
            target_y = int(ratio*stamp.shape[0])
            if (middle_y - target_y) < 0:
                stamp = cv2.resize(stamp, dsize = [target_length, middle_y])
            else:
                stamp = cv2.resize(stamp, dsize= [target_length, int(ratio*stamp.shape[0])])

            (h, w) = stamp.shape[:2]
            (cX, cY) = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D((cX, cY), deg, 1.0)
            stamp = cv2.warpAffine(stamp, M, (w, h), borderValue=(255,255,255))
            
            stamp_y, _, _ = stamp.shape
            roi = self.img[middle_y-stamp_y: middle_y, lx :rx]
            img2gray = cv2.cvtColor(stamp, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(img2gray, 250, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            img_bg = cv2.bitwise_and(roi, roi, mask=mask)
            stamp_fg = cv2.bitwise_and(stamp, stamp, mask=mask_inv)
            dst = cv2.add(img_bg, stamp_fg)
            self.img[middle_y-stamp_y: middle_y, lx : rx] = dst

        self.img = await self.img_cnv_bite(self.img)
        return self.img

    async def img_cnv_bite(self, img):
        _, encoded_image = cv2.imencode('.jpg', np.array(img))
        b64_string = base64.b64encode(encoded_image).decode('utf-8')
        return b64_string



# API
route = APIRouter()

@route.post("/whos_the_king")
async def upload_result(file: UploadFile):
    content = await file.read()
    content_out = np.array(Image.open(BytesIO(content)))[:, :, ::-1]

    proba_gen, parser, response_gen = proba_generator(), data_parser(), response_generator(content_out)

    input_data, pos_holder = await parser(content_out)
    if input_data == '-1':
        content=base64.b64encode(content).decode('utf-8')
        return {'message': '얼굴을 인식할수 없습니다 ㅜㅜ' ,'image':content}

    proba_list = await proba_gen.predict_all(input_data)
    rank_dict = await response_gen.rank_sorter(proba_list)
    out_img = await response_gen.stamp_img(rank_dict, pos_holder)
    
    return {'message': '' ,'image':out_img}
