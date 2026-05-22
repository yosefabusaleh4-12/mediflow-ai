import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv(r"C:\Users\HP\Desktop\mediflow-ai\datasets\data.csv")

X = data[["quantity", "usage", "days_left", "cost"]]

# =========================
# MODEL 1: EXPIRY RISK
# =========================
y_expiry = data["expiry_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_expiry, test_size=0.2, random_state=42
)

expiry_model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

expiry_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

expiry_model.fit(X_train, y_train, epochs=30, verbose=0)

# SAVE .keras MODEL
expiry_model.save("expiry_model.keras")


# =========================
# MODEL 2: SHORTAGE RISK
# =========================
y_shortage = data["shortage_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_shortage, test_size=0.2, random_state=42
)

shortage_model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

shortage_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

shortage_model.fit(X_train, y_train, epochs=30, verbose=0)

# SAVE .keras MODEL
shortage_model.save("shortage_model.keras")

print("MODELS TRAINED SUCCESSFULLY")