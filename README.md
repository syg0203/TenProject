### 도라에몽 주먹 테스트

#### python api

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


```python
import cv2
import matplotlib.pyplot as plt
```


```python
img = cv2.imread('./readme_1.png')
plt.axis('off') # 창에있는 x축 y축 제거
plt.imshow(img)
plt.show()
```


    
![png](README_files/README_6_0.png)
    


### http://syg0203.iptime.org:2030/


```python
img = cv2.imread('./readme_2.jpg')
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax1.axis('off')
ax1.imshow(img)
img = cv2.imread('./readme_3.jpg')
ax2 = fig.add_subplot(1,2,2)
ax2.axis('off')
ax2.imshow(img)
plt.show()
```


    
![png](README_files/README_8_0.png)
    

