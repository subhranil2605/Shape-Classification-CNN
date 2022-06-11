from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests


app = FastAPI()


MODEL = tf.keras.models.load_model("../saved_models/1")


CLASS_NAMES = ['circle', 'rectangle']


def read_file_as_image(data: bytes) -> np.ndarray:
    img = np.array(Image.open(BytesIO(data)))
    return img
    

@app.get("/ping")
async def ping():
    return "hello, I am alive"



@app.post("/predict")
async def predict(file: UploadFile=File(...)):
    image = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(image_batch)
    
    class_name = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    
    return {"class": class_name, "confidence": float(confidence)}


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
