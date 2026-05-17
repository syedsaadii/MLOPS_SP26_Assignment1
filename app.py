from fastapi import FastAPI
from pydantic import BaseModel, field_validator
import pandas as pd
import joblib

app = FastAPI()

model = joblib.load("diabetes_model.pkl")
training_columns = joblib.load("training_columns.pkl")

class PatientData(BaseModel):
    age: float
    urea: float
    cr: float
    hba1c: float
    chol: float
    tg: float
    hdl: float
    ldl: float
    vldl: float
    bmi: float
    gender: str

    @field_validator("gender")
    def validate_gender(cls, value):
        if value not in ["M", "F"]:
            raise ValueError("Gender must be M or F")
        return value


@app.get("/")
def home():
    return {"status": "API is running"}


@app.post("/predict")
def predict(data: PatientData):

    input_data = {
        "age": [data.age],
        "urea": [data.urea],
        "cr": [data.cr],
        "hba1c": [data.hba1c],
        "chol": [data.chol],
        "tg": [data.tg],
        "hdl": [data.hdl],
        "ldl": [data.ldl],
        "vldl": [data.vldl],
        "bmi": [data.bmi],
        "gender_M": [1 if data.gender == "M" else 0],
        "gender_F": [1 if data.gender == "F" else 0],
    }

    df = pd.DataFrame(input_data)

    for col in training_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[training_columns]

    prediction = model.predict(df)

    return {"prediction": int(prediction[0])}