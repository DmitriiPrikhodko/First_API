from transformers import AutoTokenizer, AutoModel
import torch
from utils.models import MyPersonalTinyBert, UNet
import torchvision.transforms as T
from PIL import Image
import numpy as np
import json
import torch.nn.functional as F
import os
from io import BytesIO
import base64


# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"
MAIN_DIR = os.path.dirname("main.py")
WEIGHTS_DIR = os.path.join(MAIN_DIR, "api", "weights")


def pil_to_base64(img: Image.Image) -> str:
    """Конвертирует PIL.Image в base64 строку"""
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def text_preproc(text):
    """
    Функция предобработки текста
    Возвращает long тензоры
    """
    tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
    encoded_input = tokenizer(
        text, max_length=64, truncation=True, padding="max_length"
    )

    input_ids = torch.tensor(encoded_input["input_ids"]).long()
    attention_masks = torch.tensor(encoded_input["attention_mask"]).long()

    return input_ids, attention_masks


def load_rubert_cls():
    """
    Функция загрузки RuBert классификатора
    """
    model = MyPersonalTinyBert()
    weights = torch.load(
        os.path.join("..", WEIGHTS_DIR, "rubert_weights.pth"), map_location="cpu"
    )
    model.load_state_dict(weights)
    model.to(DEVICE)
    model.eval()
    return model


def rubert_prediction(text, model):
    """
    Функция принмает текст и возвращает предсказанный класс
    """

    input_tensor, mask_tensor = text_preproc(text)
    input_tensor.to(DEVICE)
    mask_tensor.to(DEVICE)
    model.to(DEVICE)
    model.eval()
    with torch.inference_mode():
        output = model(input_tensor.unsqueeze(0), mask_tensor.unsqueeze(0))
        prob = F.softmax(output, dim=1).max()
        prediction = torch.argmax(output, dim=1).item()

    labels_rubert = {
        "1": "искусство",
        "2": "маркетинг",
        "3": "образование_познавательное",
        "4": "технологии",
    }
    label = labels_rubert.get(str(prediction), "Неизвестно")
    return label, prob


def load_unet():
    """
    Функция загрузки Unet модели для сегментации
    """

    model = UNet(n_class=2)
    print(WEIGHTS_DIR)
    weights = torch.load(
        os.path.join("..", WEIGHTS_DIR, "unet_weights.pth"), map_location="cpu"
    )
    model.load_state_dict(weights)
    model.to(DEVICE)
    model.eval()
    return model


def predict_image(
    model,
    img,
    device=DEVICE,
    img_size=(256, 256),
    overlay_color=(0, 0, 255),
    alpha=0.5,
):
    """
    Принимает изображение, делает предсказание маски и накладывает маску на изображение.
    overlay_color: RGB цвет маски (0-255)
    alpha: прозрачность маски
    Возвращает маску и изображение с наложенной маской
    """

    model.eval()
    with torch.no_grad():
        # --- 1. Загружаем изображение

        orig_size = img.size  # (W,H)

        # --- 2. Преобразуем в тензор и ресайзим
        transform = T.Compose(
            [
                T.Resize(img_size),
                T.ToTensor(),
            ]
        )
        img_tensor = transform(img).unsqueeze(0).to(device)

        # --- 3. Предсказание
        output = model(img_tensor)
        if output.shape[1] == 1:
            mask = torch.sigmoid(output)
            mask = (mask > 0.5).float()
        else:
            mask = torch.argmax(output, dim=1, keepdim=True).float()
        mask = mask.squeeze(0).squeeze(0).cpu().numpy()  # [H,W]

        # --- 4. Resize маски обратно к оригинальному размеру
        mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize(orig_size)
        mask_np = np.array(mask_img) / 255.0  # [0,1]

        # --- 5. Создаем цветную маску
        overlay = np.zeros((orig_size[1], orig_size[0], 3), dtype=np.uint8)
        overlay[..., 0] = overlay_color[0]  # R
        overlay[..., 1] = overlay_color[1]  # G
        overlay[..., 2] = overlay_color[2]  # B

        # --- 6. Накладываем маску на изображение
        img_np = np.array(img).astype(np.uint8)
        combined = img_np.copy()
        combined = (
            combined * (1 - alpha * mask_np[..., None])
            + overlay * (alpha * mask_np[..., None])
        ).astype(np.uint8)
        combined_img = Image.fromarray(combined)

        return mask_img, combined_img
