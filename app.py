from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load(
    "house_price_model.pkl"
)

@app.get("/")
def home():

    return {
        "message":
        "House Price Prediction API"
    }

@app.get("/predict")
def predict(size: int):

    prediction = model.predict(
        np.array([[size]])
    )

    return {
        "house_size": size,
        "predicted_price_lakhs":
        round(float(prediction[0]), 2)
    }