# import psutil
from fastapi import APIRouter, UploadFile
import numpy as np
import asyncio
import base64
import cv2
import io
import torch
from PIL import Image
from torchvision.ops import box_iou
import os
import yolov5
import gc

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class doraemong:
    def __init__(self):
        self.label_li = np.array(['fat', 'thin'])
        self.model = yolov5.load('./asset/epoch77-l.pt')
        self.model.names[0] = '최고 도라에몽'

    async def remove_overlapping_boxes(self, boxes, scores, labels, threshold):
        num_boxes = len(boxes)
        if num_boxes == 1:
            return None
        # 중첩된 객체 검출을 위한 IoU 계산
        iou = box_iou(boxes, boxes)

        # IoU 값의 임계치를 기준으로 중첩 여부 확인 보편적으로 0.5지만, 라벨링 직접해서 0.8로 설정
        overlapping_boxes = iou > threshold

        # 중첩된 박스를 제거하기 위한 마스크 생성
        mask = torch.ones(num_boxes, dtype=torch.bool)

        for i in range(num_boxes):
            if mask[i]:
                # 중첩된 객체의 인덱스 목록
                overlapping_indices = torch.nonzero(
                    overlapping_boxes[i]).squeeze(1)

                # 중첩된 객체 중 가장 높은 신뢰도를 가진 객체 선택
                max_score_idx = torch.argmax(scores[overlapping_indices])
                max_score_box_idx = overlapping_indices[max_score_idx]

                # 선택한 객체 이외의 중첩된 객체들의 마스크를 False로 설정
                mask[overlapping_indices] = False
                # 선택한 객체의 마스크를 True로 설정
                mask[max_score_box_idx] = True

                # 라벨이 다른 경우에도 중첩 여부 확인하여 처리
                for idx in overlapping_indices:
                    if labels[idx] != labels[max_score_box_idx]:
                        mask[idx] = False
        return mask

    async def process_predictions(self, predictions, threshold):
        boxes = predictions.xyxy[0][:, :4]  # 바운딩박스 좌표
        scores = predictions.xyxy[0][:, 4]  # 신뢰도
        labels = predictions.xyxy[0][:, 5]  # 예측된 라벨

        # 중첩된 박스 제거
        mask = await self.remove_overlapping_boxes(boxes, scores, labels, threshold)
        return mask

    async def imagedown_async(self, img_path):
        image = Image.open(io.BytesIO(img_path))
        self.results = self.model(image)
        # mask true인덱스만 추출 -> 중첩 박스 제거
        mask = await self.process_predictions(self.results, threshold=0.8)
        if mask == None:
            return -1
        self.results.xyxy[0] = self.results.xyxy[0][torch.nonzero(
            mask).squeeze(1)]

        # 라벨이 노말인 경우 도라에몽으로 변환 normalization
        mask_tmp = self.results.xyxy[0][:, 5] == 1
        self.results.xyxy[0][mask_tmp, 4] = (
            1-self.results.xyxy[0][mask_tmp, 4])/2

        # 라벨 도라에몽일 경우 normalization
        mask_tmp = self.results.xyxy[0][:, 5] == 0
        self.results.xyxy[0][mask_tmp, 4] = (
            self.results.xyxy[0][mask_tmp, 4]+1)/2

        # 라벨 전체 도라에몽으로
        self.results.xyxy[0][:, 5] = self.results.xyxy[0][:, 5]*0

        # 최고 도라에몽만 추출
        try:
            self.results.xyxy[0] = self.results.xyxy[0][self.results.xyxy[0]
                                                        [:, 4] == self.results.xyxy[0][:, 4].max()]
        except RuntimeError:
            pass

        data = None  # data 변수 초기화
        for detection in self.results.pandas().xyxy[0].iterrows():
            _, data = detection
        try:
            return data['confidence']
        except:
            return 0

    async def predict_batch(self, img_paths):
        results = await asyncio.gather(*(self.imagedown_async(i) for i in img_paths))
        predict_arr = np.max(results)

        recommend = self.label_li[np.argmax(results)]
        return recommend, predict_arr

    async def predict(self, image):
        get_pb = await self.predict_batch([image])
        recommend = get_pb[0]
        predict_arr = get_pb[1]
        return recommend, predict_arr


route = APIRouter()


@route.post("/photo")
async def upload_photo(file: UploadFile):
    # ram_usage = psutil.virtual_memory()
    # # RAM 사용량을 메가바이트(MB)로 변환
    # ram_usage_mb = ram_usage.used / (1024 * 1024)
    # # RAM 사용량 출력
    # print(f"현재 RAM 사용량: {ram_usage_mb:.2f} MB")
    try:
        # Doraemong 클래스의 인스턴스 생성
        doraecls = doraemong()

        # 업로드된 파일의 내용 읽기
        content = await file.read()

        # DoraemonG 모델을 사용하여 예측 수행
        json_string = await doraecls.predict(content)

        # 결과를 이미지 바이트로 변환
        image_bytes = np.array(doraecls.results.render()[0])
        # 이미지 바이트를 RGB 형식으로 변환
        image_rgb = cv2.cvtColor(image_bytes, cv2.COLOR_BGR2RGB)

        # RGB 이미지를 JPEG 형식으로 인코딩
        _, encoded_image = cv2.imencode('.jpg', image_rgb)

        # JPEG 이미지를 base64 문자열로 인코딩
        b64_string = base64.b64encode(encoded_image).decode('utf-8')

        # 아싸이미지 base64 문자열로 인코딩
        AssaImg = cv2.imread('./asset/assa.jpg', cv2.IMREAD_COLOR)
        AssaImg = np.array(AssaImg)
        _, AssaImg = cv2.imencode('.jpg', AssaImg)
        AssaImg = base64.b64encode(AssaImg).decode('utf-8')

        # 예측된 클래스에 따라 결과 딕셔너리 생성
        if json_string[1] == 0:
            result = {
                "recommend": json_string[0],
                'predict_arr': str(0),
                'filename': b64_string
            }
        elif json_string[1] == -1:
            result = {
                "recommend": json_string[0],
                'predict_arr': str(-1),
                'filename': AssaImg
            }
        else:
            result = {
                "recommend": json_string[0],
                'predict_arr': str(round((json_string[1] * 100), 1)),
                'filename': b64_string
            }

        # 결과 딕셔너리 반환
        return result

    finally:
        del doraecls
        del content
        del AssaImg
        del json_string
        del image_bytes
        del image_rgb
        del encoded_image
        del b64_string
        del _
        del result
        gc.collect()
