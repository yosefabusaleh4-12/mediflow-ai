from fastapi import FastAPI
import tensorflow as tf
import numpy as np
import os

from backend.database import SessionLocal, Prediction

app = FastAPI()

# -----------------------------
# Load AI models safely
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AI_DIR = os.path.join(BASE_DIR, "..", "ai")

expiry_model = tf.keras.models.load_model(
    os.path.join(AI_DIR, "expiry_model.h5")
)

shortage_model = tf.keras.models.load_model(
    os.path.join(AI_DIR, "shortage_model.h5")
)


@app.get("/")
def home():
    return {"message": "MediFlow AI is running"}


@app.post("/predict")
def predict(data: dict):

    # -----------------------------
    # Prepare input
    # -----------------------------
    x = np.array([[
        data["quantity"],
        data["usage"],
        data["days_left"],
        data["cost"]
    ]])

    # -----------------------------
    # AI predictions
    # -----------------------------
    expiry = float(expiry_model.predict(x)[0][0])
    shortage = float(shortage_model.predict(x)[0][0])

    # -----------------------------
    # Decision Engine
    # -----------------------------
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

    # -----------------------------
    # Save to database
    # -----------------------------
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

    # -----------------------------
    # Return response
    # -----------------------------
    return {
        "expiry_risk": expiry,
        "shortage_risk": shortage,
        "expiry_action": expiry_action,
        "shortage_action": shortage_action
    }
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

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)