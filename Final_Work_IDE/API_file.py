from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

import warnings
import dill
import json
import pandas as pd
from datetime import datetime

from os import PathLike
from pathlib import Path
from sklearn.base import BaseEstimator
from typing import Union

warnings.filterwarnings('ignore')


# Создаем шаблон для присвоения названия моделей
model_name_pattern = 'model_*.pkl'

# Загружаем модели
def load_model(folder: PathLike) -> BaseEstimator:
    """Загружает последнюю модель из папки по времени создания."""

    # Список моделей в папке
    folder = Path(folder)
    model_files = list(folder.glob(model_name_pattern))

    # Загрузим последнюю модель из папки
    if model_files:
        last_model = sorted(model_files)[-1]
        print('Загружаем последнюю обученную модель:', last_model)
        with open(last_model, 'rb') as file:
            model = dill.load(file)
        return model

    else:
        raise FileNotFoundError('Папка с моделями пуста.')


# Загружаем модель

model = load_model('model')

for key, value in model.metadata.items():
    print(key, ":", value)

# Загружаем пример из json-файла для проверки работы модели
with open('data/examples.json', 'rb') as file:
    examples = json.load(file)
    df = pd.DataFrame.from_dict(examples)
    example = df.iloc[[2]]


class Form(BaseModel):
    session_id: str
    client_id: str
    visit_date: str
    visit_time: str
    visit_number: int
    utm_source: str
    utm_medium: str
    utm_campaign: str
    utm_adcontent: str
    utm_keyword: Union[str, None] = None
    device_category: str
    device_os: Union[str, None] = None
    device_brand: str
    device_model: Union[str, None] = None
    device_screen_resolution: str
    device_browser: str
    geo_country: str
    geo_city: str


class Prediction(BaseModel):
    session_id: str
    event_value: float
    time_execution: str


@app.get('/status')
def status():
    return "Все в порядке!"


@app.get('/version')
def version():
    return model.metadata


@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    def time_it():
        elapsed_time = datetime.now() - start_time
        return elapsed_time
    start_time = datetime.now()
    df = pd.DataFrame.from_dict([form.dict()])
    y = model.predict(df)
    t = str(time_it())
    print("хххххххххххххххххххх", example)
    print(model.metadata, t, example)
    return {
        'session_id': form.session_id,
        'event_value': y[0],
        'time_execution': t
    }


#  для запуска в терменале команда uvicorn API_file:app --reload