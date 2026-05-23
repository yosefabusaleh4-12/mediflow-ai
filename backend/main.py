from fastapi import FastAPI
import numpy as np
import os
import tensorflow as tf
import joblib

app = FastAPI()

# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AI_DIR = os.path.join(BASE_DIR, "..", "ai")

expiry_path = os.path.join(AI_DIR, "expiry_weights.weights.h5")
shortage_path = os.path.join(AI_DIR, "shortage_weights.weights.h5")
scaler_path = os.path.join(AI_DIR, "scaler.save")

# =========================
# GLOBALS
# =========================
model_expiry = None
model_shortage = None
scaler = None

# =========================
# MODEL
# =========================
def build_model():
    return tf.keras.Sequential([
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

# =========================
# LOAD MODELS + SCALER
# =========================
def load_models():
    global model_expiry, model_shortage, scaler

    if scaler is None:
        scaler = joblib.load(scaler_path)

    if model_expiry is None:
        model_expiry = build_model()
        model_expiry.load_weights(expiry_path)

    if model_shortage is None:
        model_shortage = build_model()
        model_shortage.load_weights(shortage_path)

# =========================
# ROUTES
# =========================
@app.get("/")
def home():
    return {"message": "AI running"}

@app.post("/predict")
def predict(data: dict):

    load_models()

    x = np.array([[
        data["quantity"],
        data["usage"],
        data["days_left"],
        data["cost"]
    ]])

    # 🔥 IMPORTANT FIX
    x = scaler.transform(x)

    expiry = float(model_expiry.predict(x)[0][0])
    shortage = float(model_shortage.predict(x)[0][0])

    return {
        "expiry_risk": round(expiry, 2),
        "shortage_risk": round(shortage, 2)
    }