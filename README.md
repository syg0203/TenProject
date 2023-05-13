### 도라에몽 주먹 판별 모델을 실행하는 API 호출

#### python 활용

img_path = 로컬에 저장된 이미지 경로 입력


```python
img_path="../test6.jpg"


import requests

url = "http://syg0203.iptime.org:2030/photo/"

with open(img_path, "rb") as f:
    files = {"file": (f.read())}

response = requests.get(url, files=files)

print(response.json())
```

    {'recommend': 'thin', 'predict_arr': '65.2%'}
    

#### POSTMAN 활용

![image.png](attachment:image.png)
