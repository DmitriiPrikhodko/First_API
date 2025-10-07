import logging
from contextlib import asynccontextmanager

import PIL
from PIL import Image
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel, field_validator
from utils.funcs import *
from io import BytesIO

logger = logging.getLogger("uvicorn.info")



class ImageResponse(BaseModel):
    mask: str
    masked_image: str
    model_config = {
        "arbitrary_types_allowed": True  # 👈 вот эта строка решает твою ошибку
    }


class TextInput(BaseModel):
    text: str


class TextResponse(BaseModel):
    label: str
    prob: float


unet_model = None
rubert_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Контекстный менеджер для инициализации и завершения работы FastAPI приложения.
    Загружает модели машинного обучения при запуске приложения и удаляет их после завершения.
    """
    global unet_model
    global rubert_model
    # Загрузка UNet модели
    unet_model = load_unet()
    logger.info("UNet model loaded")
    # Загрузка RuBert модели
    rubert_model = load_rubert_cls()
    logger.info("RuBert model loaded")
    yield
    # Удаление моделей и освобождение ресурсов
    del unet_model, rubert_model


app = FastAPI(lifespan=lifespan)


@app.get("/")
def return_info():
    """
    Возвращает привествие
    """
    return "Привет, пользователь!"


@app.post("/clf_post")
def clf_post(input_json: TextInput):
    """
    Эндпоинт для классификации постов в телеграмме.
    Принимает текст, обрабатывает его, делает предсказание и возвращает название класса и вероятность.
    """
    pred, prob = rubert_prediction(input_json.text, rubert_model)
    response = TextResponse(label=pred, prob=prob)
    return response


@app.post("/segment_image")
async def segment_image(file: UploadFile):
    # async def segment_image(file: ImageInput):
    """
    Эндпоинт для сегментации изображений (поиск леса на спутниковых фотографиях)
    Принимает изображение, обрабатывает, сегментирует, возвращает маску сегментации и изображение с наложенной маской
    """

    img = Image.open(BytesIO(await file.read()))
    mask, combined_img = predict_image(unet_model, img)
    response = ImageResponse(
        mask=pil_to_base64(mask), masked_image=pil_to_base64(combined_img)
    )
    return response


if __name__ == "__main__":
    # Запуск приложения на localhost с использованием Uvicorn
    # производится из командной строки: python your/path/api/main.py
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
