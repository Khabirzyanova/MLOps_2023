from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib

app = FastAPI()

class Dataset(BaseModel):
    file: UploadFile

class ModelInfo(BaseModel):
    name: str
    description: str
    hyperparameters: dict

class Model:
    def __init__(self, name, description, hyperparameters):
        self.name = name
        self.description = description
        self.hyperparameters = hyperparameters
        self.model = None

models = {}

@app.post("/train_model/{model_name}")
async def train_model(model_name: str, model_info: ModelInfo, dataset: Dataset):
    if model_name in models:
        return HTTPException(status_code=400, detail="Модель с таким именем уже существует")
    
    # Сохраняем загруженный датасет
    dataset_path = f"datasets/{dataset.file.filename}"
    with open(dataset_path, "wb") as f:
        f.write(dataset.file.file.read())
    
    # Загрузка данных и обучение модели
    data = pd.read_csv(dataset_path)
    X = data.drop('target', axis=1)
    y = data['target']
    model = LogisticRegression(**model_info.hyperparameters)
    model.fit(X, y)
    
    # Сохраняем обученную модель
    model_path = f"models/{model_name}.pkl"
    joblib.dump(model, model_path)
    
    new_model = Model(model_name, model_info.description, model_info.hyperparameters)
    new_model.model = model_path
    
    models[model_name] = new_model
    return {"message": f"Модель '{model_name}' успешно обучена"}

@app.get("/models")
async def get_models():
    return list(models.keys())

@app.post("/predict/{model_name}")
async def predict(model_name: str, input_data: dict):
    if model_name not in models:
        return HTTPException(status_code=404, detail="Модель не найдена")
    
    model = models[model_name].model
    if model is None:
        return HTTPException(status_code=400, detail="Модель не обучена")
    
    loaded_model = joblib.load(model)
    # Здесь выполните предсказание с использованием обученной модели
    # prediction = loaded_model.predict(input_data)
    
    return {"prediction": prediction}

@app.delete("/delete_model/{model_name}")
async def delete_model(model_name: str):
    if model_name not in models:
        return HTTPException(status_code=404, detail="Модель не найдена")
    
    del models[model_name]
    return {"message": f"Модель '{model_name}' удалена"}

if __name__ == "__main__":
    import os
    
    if not os.path.exists("datasets"):
        os.makedirs("datasets")
    
    if not os.path.exists("models"):
        os.makedirs("models")
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
