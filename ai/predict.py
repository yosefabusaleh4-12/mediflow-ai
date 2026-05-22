import tensorflow as tf
import numpy as np

# Load trained model
model = tf.keras.models.load_model("expiry_model.h5")

# Example input (new medicine)
# [quantity, usage, days_left, cost]
sample = np.array([[50, 20, 30, 10]])

# Predict
prediction = model.predict(sample)

print("Risk score:", prediction[0][0])

# Convert to readable result
if prediction[0][0] > 0.5:
    print("HIGH EXPIRY RISK ❌")
else:
    print("SAFE ✅")