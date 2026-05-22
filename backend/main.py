from fastapi import FastAPI
import numpy as np
import os
import tensorflow as tf

from backend.database import SessionLocal, Prediction

app = FastAPI()

# -----------------------------
# MODEL PATHS
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AI_DIR = os.path.join(BASE_DIR, "..", "ai")

expiry_model = None
shortage_model = None


# -----------------------------
# LOAD MODELS (FIXED - IMPORTANT)
# -----------------------------
def load_models():
    global expiry_model, shortage_model

    if expiry_model is None:
        expiry_model = tf.keras.models.load_model(
            os.path.join(AI_DIR, "expiry_model.keras"),
            compile=False
        )

    if shortage_model is None:
        shortage_model = tf.keras.models.load_model(
            os.path.join(AI_DIR, "shortage_model.keras"),
            compile=False
        )


# -----------------------------
# HOME
# -----------------------------
@app.get("/")
def home():
    return {"message": "MediFlow AI is running"}


# -----------------------------
# PREDICT
# -----------------------------
@app.post("/predict")
def predict(data: dict):

    load_models()

    x = np.array([[
        data["quantity"],
        data["usage"],
        data["days_left"],
        data["cost"]
    ]])

    expiry = float(expiry_model.predict(x)[0][0])
    shortage = float(shortage_model.predict(x)[0][0])

    expiry_action = (
        "MOVE / REDISTRIBUTE" if expiry > 0.8 else
        "MONITOR" if expiry > 0.5 else
        "SAFE"
    )

    shortage_action = (
        "URGENT RESTOCK" if shortage > 0.8 else
        "PREPARE RESTOCK" if shortage > 0.5 else
        "SUFFICIENT"
    )

    db = SessionLocal()

    record = Prediction(
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
        "expiry_risk": expiry,
        "shortage_risk": shortage,
        "expiry_action": expiry_action,
        "shortage_action": shortage_action
    }


# -----------------------------
# HISTORY
# -----------------------------
@app.get("/history")
def history():
    db = SessionLocal()
    records = db.query(Prediction).all()

    result = []

    for r in records:
        result.append({
            "quantity": r.quantity,
            "usage": r.usage,
            "days_left": r.days_left,
            "cost": r.cost,
            "expiry_risk": r.expiry_risk,
            "shortage_risk": r.shortage_risk,
            "expiry_action": r.expiry_action,
            "shortage_action": r.shortage_action
        })

    db.close()
    return result


# -----------------------------
# RUN LOCAL
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)