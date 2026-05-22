import tensorflow as tf
import numpy as np

# Load models
expiry_model = tf.keras.models.load_model("expiry_model.h5")
shortage_model = tf.keras.models.load_model("shortage_model.h5")

def predict(medicine_data):
    """
    medicine_data = [quantity, usage, days_left, cost]
    """

    data = np.array([medicine_data])

    expiry_risk = expiry_model.predict(data)[0][0]
    shortage_risk = shortage_model.predict(data)[0][0]

    print("\n--- AI ANALYSIS ---")
    print("Expiry Risk:", round(expiry_risk, 3))
    print("Shortage Risk:", round(shortage_risk, 3))

    # =========================
    # DECISION LOGIC (THE MAGIC)
    # =========================

    if expiry_risk > 0.8:
        action1 = "⚠ MOVE OUT OR REDISTRIBUTE (High expiry risk)"
    elif expiry_risk > 0.5:
        action1 = "📦 Monitor closely"
    else:
        action1 = "✅ Safe stock"

    if shortage_risk > 0.8:
        action2 = "🚨 URGENT RESTOCK REQUIRED"
    elif shortage_risk > 0.5:
        action2 = "⚠ Prepare restocking"
    else:
        action2 = "✅ Stock is sufficient"

    print("\n--- RECOMMENDATIONS ---")
    print(action1)
    print(action2)


# Example test
predict([50, 20, 30, 10])