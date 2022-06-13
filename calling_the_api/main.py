import requests
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from io import BytesIO

f = "C:/Users/imsub/Desktop/New folder/39.jpg"

with open(f, 'rb') as ff:
    data = ff.read()

img = np.array(Image.open(BytesIO(data)))
img_batch = np.expand_dims(img, 0)

json_data = {
    "instances": img_batch.tolist()
}


response = requests.post("http://localhost:8000/predict-image", json=json_data)

##response = requests.post("http://localhost:8000/predictd", json=data)

if response:
    print(response.text)
else:
    print("not done")
