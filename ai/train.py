import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib

# Load dataset
data = pd.read_csv(
    r"C:\Users\HP\Desktop\mediflow-ai\datasets\data.csv"
)

X = data[["quantity", "usage", "days_left", "cost"]]

# =========================
# SCALING (IMPORTANT FIX)
# =========================
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# SAVE SCALER (IMPORTANT)
joblib.dump(scaler, "scaler.save")

# =========================
# MODEL ARCHITECTURE
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
# EXPIRY MODEL
# =========================
y = data["expiry_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, shuffle=True
)

expiry_model = build_model()

expiry_model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

expiry_model.fit(X_train, y_train, epochs=200, batch_size=16, verbose=1)

expiry_model.save_weights("expiry_weights.weights.h5")

# =========================
# SHORTAGE MODEL
# =========================
y = data["shortage_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, shuffle=True
)

shortage_model = build_model()

shortage_model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

shortage_model.fit(X_train, y_train, epochs=200, batch_size=16, verbose=1)

shortage_model.save_weights("shortage_weights.weights.h5")

print("🔥 TRAINING COMPLETE")