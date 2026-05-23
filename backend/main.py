from fastapi import FastAPI
import numpy as np
import os
import tensorflow as tf

from backend.database import SessionLocal, Prediction

app = FastAPI()

# -----------------------------
# MODEL SETUP
# -----------------------------
model_expiry = None
model_shortage = None

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(4,)),

        tf.keras.layers.Dense(64),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(32),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def load_models():
    global model_expiry, model_shortage

    if model_expiry is None:
        model_expiry = build_model()
        model_expiry.load_weights("expiry_weights.weights.h5")

    if model_shortage is None:
        model_shortage = build_model()
        model_shortage.load_weights("shortage_weights.weights.h5")


# -----------------------------
# ROUTES
# -----------------------------
@app.get("/")
def home():
    return {"message": "AI running"}

@app.post("/predict")
def predict(data: dict):

    load_models()

    x = np.array([[data["quantity"], data["usage"], data["days_left"], data["cost"]]])

    expiry = float(model_expiry.predict(x)[0][0])
    shortage = float(model_shortage.predict(x)[0][0])

    return {
        "expiry_risk": expiry,
        "shortage_risk": shortage,
    }