import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

data = pd.read_csv(
    r"C:\Users\HP\Desktop\mediflow-ai\datasets\data.csv"
)

X = data[["quantity","usage","days_left","cost"]]

def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(4,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])


# EXPIRY
y = data["expiry_risk"]

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

expiry_model = build_model()

expiry_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

expiry_model.fit(X_train,y_train,epochs=30,verbose=0)

expiry_model.save_weights("expiry_weights.weights.h5")


# SHORTAGE
y = data["shortage_risk"]

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

shortage_model = build_model()

shortage_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

shortage_model.fit(X_train,y_train,epochs=30,verbose=0)

shortage_model.save_weights(
    "shortage_weights.weights.h5"
)

print("done")