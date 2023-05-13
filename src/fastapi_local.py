import uvicorn
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse,JSONResponse
from ast import literal_eval
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from json import dumps
import asyncio
import nest_asyncio

import uuid
import os
nest_asyncio.apply()
import tensorflow as tf

label_li=np.array(['fat','thin'])
DeepLearning=load_model('./asset/dorae-mobile_4_2023-05-13.hdf5')

async def imagedown_async(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = tf.image.rgb_to_grayscale(img)
    img = tf.image.grayscale_to_rgb(img)

    x = img / 255
    x = np.expand_dims(x, axis=0)

    y_pred = DeepLearning.predict(x)
    return y_pred[0]

async def predict_batch(img_paths):
    loop = asyncio.get_event_loop()
    results=loop.run_until_complete(asyncio.gather(*(imagedown_async(i) for i in img_paths)))

    predict_arr=np.max(results)

    recommend=label_li[np.argmax(results)]
    return recommend,predict_arr

async def predict(image):
    get_pb=asyncio.run(predict_batch(['./temp/'+image]))
    recommend=get_pb[0]
    predict_arr=get_pb[1]
    return recommend,predict_arr

app = FastAPI()

@app.get("/photo")
async def upload_photo(file: UploadFile):
    UPLOAD_DIR = "./temp"
    content = await file.read()
    filename = f"{str(uuid.uuid4())}.jpg"
    with open(os.path.join(UPLOAD_DIR, filename), "wb") as fp:
        fp.write(content)
    json_string=asyncio.run(predict(filename))
    result={"recommend":json_string[0],'predict_arr':str(round(json_string[1]*100,1))+'%'}
    return result

@app.get("/")
def root():
  # return JSONResponse(content={'message':'200','server':'yg-api'})
  return {'message':'200','server':'yg-api'}

@app.get('/photodown')
def get_image():
    return FileResponse("test.jpg")

if __name__ == "__main__":
    uvicorn.run(app, host="192.168.0.14", port=2030)