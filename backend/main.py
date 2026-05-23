from fastapi import FastAPI
import tensorflow as tf
import numpy as np
import os

from backend.database import SessionLocal, Prediction

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AI_DIR = os.path.join(BASE_DIR, "..", "ai")


def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(4,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])


expiry_model = build_model()
expiry_model.load_weights(
    os.path.join(AI_DIR,"expiry_weights.weights.h5")
)

shortage_model = build_model()
shortage_model.load_weights(
    os.path.join(AI_DIR,"shortage_weights.weights.h5")
)


@app.get("/")
def home():
    return {"message":"running"}


@app.post("/predict")
def predict(data:dict):

    x = np.array([[
        data["quantity"],
        data["usage"],
        data["days_left"],
        data["cost"]
    ]])

    expiry=float(expiry_model.predict(x)[0][0])
    shortage=float(shortage_model.predict(x)[0][0])

    expiry_action=(
        "MOVE / REDISTRIBUTE"
        if expiry>0.8 else
        "MONITOR"
        if expiry>0.5 else
        "SAFE"
    )

    shortage_action=(
        "URGENT RESTOCK"
        if shortage>0.8 else
        "PREPARE RESTOCK"
        if shortage>0.5 else
        "SUFFICIENT"
    )

    db=SessionLocal()

    record=Prediction(
        quantity=data["quantity"],
        usage=data["usage"],
        days_left=data["days_left"],
        cost=data["cost"],
        expiry_risk=expiry,
        shortage_risk=shortage,
        expiry_action=expiry_action,
        shortage_action=shortage_action
    )

    db.add(record)
    db.commit()
    db.close()

    return {
        "expiry_risk":expiry,
        "shortage_risk":shortage,
        "expiry_action":expiry_action,
        "shortage_action":shortage_action
    }