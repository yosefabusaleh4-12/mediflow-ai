import tensorflow as tf
import numpy as np

# Load models (CLEAN .keras VERSION)
expiry_model = tf.keras.models.load_model("expiry_model.keras")
shortage_model = tf.keras.models.load_model("shortage_model.keras")


def predict(medicine_data):
    """
    medicine_data = [quantity, usage, days_left, cost]
    """

    data = np.array([medicine_data])

    expiry_risk = float(expiry_model.predict(data)[0][0])
    shortage_risk = float(shortage_model.predict(data)[0][0])

    # =========================
    # DECISION ENGINE
    # =========================

    if expiry_risk > 0.8:
        expiry_action = "MOVE / REDISTRIBUTE"
    elif expiry_risk > 0.5:
        expiry_action = "MONITOR"
    else:
        expiry_action = "SAFE"

    if shortage_risk > 0.8:
        shortage_action = "URGENT RESTOCK"
    elif shortage_risk > 0.5:
        shortage_action = "PREPARE RESTOCK"
    else:
        shortage_action = "SUFFICIENT"

    # IMPORTANT: return JSON (for FastAPI / Flutter)
    return {
        "expiry_risk": expiry_risk,
        "shortage_risk": shortage_risk,
        "expiry_action": expiry_action,
        "shortage_action": shortage_action
    }


# Optional local test
if __name__ == "__main__":
    result = predict([50, 20, 30, 10])
    print(result)