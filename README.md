---
jupyter:
  kernelspec:
    display_name: base
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.9.13
  nbformat: 4
  nbformat_minor: 2
  vscode:
    interpreter:
      hash: 6b360fa53dd7ad7afb8921e8f9c25b79969c3ffda36a8028be04e23afd5d378c
---

::: {.cell .markdown}
### 도라에몽 주먹 판별 모델을 실행하는 API 호출
:::

::: {.cell .markdown}
#### python 활용
:::

::: {.cell .markdown}
img_path = 로컬에 저장된 이미지 경로 입력
:::

::: {.cell .code execution_count="2"}
``` python
img_path="../test6.jpg"


import requests

url = "http://syg0203.iptime.org:2030/photo/"

with open(img_path, "rb") as f:
    files = {"file": (f.read())}

response = requests.get(url, files=files)

print(response.json())
```

::: {.output .stream .stdout}
    {'recommend': 'thin', 'predict_arr': '65.2%'}
:::
:::

::: {.cell .markdown}
#### POSTMAN 활용
:::

::: {.cell .markdown}
![image.png](src/image.png)
:::
