import uvicorn
from fastapi import FastAPI, UploadFile
import numpy as np
import asyncio
import nest_asyncio
from fastapi.staticfiles import StaticFiles
import cv2
import base64
import uuid
import os
nest_asyncio.apply()
import torch
import os
from PIL import Image

label_li=np.array(['fat','thin'])
# DeepLearning=load_model('./asset/dorae-mobile_4_2023-05-13.hdf5')
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./asset/epoch73.pt')
output_folder = './temp_output'

async def imagedown_async(img_path):
    image = Image.open(img_path)
    results = model(image)
    results.save(save_dir=output_folder,exist_ok=True)
    
    data = None  # data 변수 초기화
    for detection in results.pandas().xyxy[0].iterrows():
        _, data = detection
    try:
        return data['confidence']
    except:
        return 0

async def predict_batch(img_paths):
    loop = asyncio.get_event_loop()
    results=loop.run_until_complete(asyncio.gather(*(imagedown_async(i) for i in img_paths)))
    # results=asyncio.run((imagedown_async(i) for i in img_paths))
    predict_arr=np.max(results)

    recommend=label_li[np.argmax(results)]
    return recommend,predict_arr

async def predict(image):
    get_pb=asyncio.run(predict_batch(['./temp/'+image]))
    recommend=get_pb[0]
    predict_arr=get_pb[1]
    return recommend,predict_arr

app = FastAPI()

@app.post("/photo")
async def upload_photo(file: UploadFile):
    UPLOAD_DIR = "./temp"
    PREDICT_DIR = "./temp_output"
    content = await file.read()
    filename = f"{str(uuid.uuid4())}.jpg"
    with open(os.path.join(UPLOAD_DIR, filename), "wb") as fp:
        fp.write(content)
    json_string=asyncio.run(predict(filename))
    
    # print(os.path.join(UPLOAD_DIR, filename))
    os.remove(os.path.join(UPLOAD_DIR, filename))
    img = cv2.imread('E:/WD/Doraemon_fist/temp_output/'+filename)
    jpg_img = cv2.imencode('.jpg', img)
    b64_string = base64.b64encode(jpg_img[1]).decode('utf-8')
    os.remove(os.path.join(PREDICT_DIR, filename))
    if json_string[0]=='fat':
        result={"recommend":json_string[0],'predict_arr':str(round(json_string[1]*100,1)),'filename':b64_string}
    else:
        result={"recommend":json_string[0],'predict_arr':str(round((1-json_string[1])*100,1)),'filename':b64_string}
    return result

app.mount("/", StaticFiles(directory="static",html = True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="192.168.0.14", port=2030)