import os
import warnings
import logging
import argparse
import uvicorn

from PIL import Image, UnidentifiedImageError

from fastapi import FastAPI, UploadFile, Request
from fastapi.templating import Jinja2Templates

import torch
import torchvision.transforms.v2 as transforms
from torchvision.models import resnet50

from fastapi.staticfiles import StaticFiles

os.chdir(os.path.split(__file__)[0])  # REDUNDANCY

import sys
sys.dont_write_bytecode = True
from transforms_custom import transformer_size, IMG_SIZE  # 224
sys.dont_write_bytecode = False



warnings.filterwarnings("ignore")


m_logger = logging.getLogger(__name__)
m_logger.setLevel(logging.DEBUG)
handler_m = logging.StreamHandler()
formatter_m = logging.Formatter(
    "%(name)s %(asctime)s %(levelname)s %(message)s")
handler_m.setFormatter(formatter_m)
m_logger.addHandler(handler_m)


DEVICE = "cpu"

CLASSES = {0: '0',
           1: '90',
           2: '180',
           3: '270'
           }


def pretrained_resnet(params_path: str, device: str):
    """load model and weights"""
    model = resnet50(weights=None)
    in_features = 2048
    out_features = 4
    model.fc = torch.nn.Linear(in_features, out_features)
    model = model.to(device)
    model.load_state_dict(torch.load(
        params_path, map_location=torch.device(device)))
    return model


def predict_rotate(path: str, inp_size: int):
    """detecting function"""
    res_path = path
    try:
        image = Image.open(path).convert('RGB')
    except UnidentifiedImageError:
        m_logger.error(f'something wrong with image')
        status = 'Fail'
        return status, path
    transformer = transformer_size(IMG_SIZE)
    tensor = transformer(image)
    model = pretrained_resnet('resnet50.pth', DEVICE)
    m_logger.info(f'model loaded')
    with torch.no_grad():
        x = tensor.to(DEVICE).unsqueeze(0)
        predictions = model.eval()(x)
    result = int(torch.argmax(predictions, 1).cpu())
    m_logger.info(f'classification completed')
    if result != 0:
        folder, file = os.path.split(path)
        res_path = os.path.join(folder, 'res', file)
        image.rotate(-90 * result, expand=True).save(res_path)

    status = 'OK'
    return status, CLASSES[result], res_path


os.makedirs('tmp', exist_ok=True)
os.makedirs('tmp//res', exist_ok=True)

app = FastAPI()

app.mount("/tmp", StaticFiles(directory="tmp"), name='images')
templates = Jinja2Templates(directory="templates")

app_logger = logging.getLogger(__name__)
app_logger.setLevel(logging.INFO)
app_handler = logging.StreamHandler()
app_formatter = logging.Formatter(
    "%(name)s %(asctime)s %(levelname)s %(message)s")
app_handler.setFormatter(app_formatter)
app_logger.addHandler(app_handler)

INP_SIZE = 224
DEVICE = "cpu"


# image size according to other applications


@app.get("/health")
def health():
    return {"status": "OK"}


@app.get("/")
def main(request: Request):
    return templates.TemplateResponse("start_form.html",
                                      {"request": request})


@app.post("/classify")
def process_request(file: UploadFile, request: Request):
    """save file to the local folder and send the image to the process function"""
    save_pth = os.path.join("tmp", file.filename)
    with open(save_pth, "wb") as fid:
        fid.write(file.file.read())
    app_logger.info(f'processing file - segmentation {save_pth}')
    status, label, res_path = predict_rotate(save_pth, INP_SIZE)
    if status == 'OK' and label != '0':
        app_logger.info(f'classification result {label}')
        message = f"Обнаружен разворот на {label} градусов"
        return templates.TemplateResponse("classify_form.html",
                                          {"request": request,
                                           "original":save_pth,
                                           "path": res_path,
                                           "message": message})
    elif status == 'OK' and label == '0':
        app_logger.info(f'processing status {label}')
        message = f"Изображение не требует коррекции"
        return templates.TemplateResponse("all_clear_form.html",
                                          {"request": request,
                                           "message": message, 
                                           "original": save_pth})
    else:
        app_logger.warning(f'some problems {status}')
        return templates.TemplateResponse("error_form.html",
                                          {"request": request,
                                           "result": status,
                                           "name": file.filename})


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=8010, type=int, dest="port")
    parser.add_argument("--host", default="0.0.0.0", type=str, dest="host")
    args = vars(parser.parse_args())

    uvicorn.run(app, **args)
