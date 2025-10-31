import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from visualization import plot_training_curves, plot_predictions
import joblib

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

df = pd.read_csv(r"C:\Health monitoring\dataset\wearable_health_data.csv")
print("\n📘 Dataset Info:\n")
print(df.info())
print("\n🔹 First 5 Rows:\n", df.head())

X = df.drop(columns=['health_score'])
y = df['health_score']

if y.dtype == 'object':
    le = LabelEncoder()
    y = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

def build_model(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(512, activation='relu',
                     kernel_initializer='he_normal',
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.25),

        layers.Dense(256, activation='relu',
                     kernel_initializer='he_normal',
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.Dense(128, activation='relu',
                     kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Dropout(0.15),

        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    return model

model = build_model(X_train.shape[1])

# Optimizer with gradient clipping
optimizer = optimizers.Adam(learning_rate=0.001, clipnorm=1.0)

model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# ==========================
# Callbacks
# ==========================
early_stop = callbacks.EarlyStopping(
    monitor='val_loss', patience=20, restore_best_weights=True
)

checkpoint = callbacks.ModelCheckpoint(
    "best_health_model.keras", monitor='val_loss',
    save_best_only=True, verbose=1
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1
)


lr_schedule = callbacks.LearningRateScheduler(lambda epoch: 1e-3 * (0.95 ** epoch))


history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=150,
    batch_size=64,
    callbacks=[early_stop, checkpoint, reduce_lr, lr_schedule],
    verbose=1
)

loss, mae = model.evaluate(X_test, y_test)
print(f"\n✅ Final Test MAE: {mae:.2f}")

y_pred = model.predict(X_test).flatten()

mae_manual = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)

print(f"📊 MAE: {mae_manual:.3f}")
print(f"📉 RMSE: {rmse:.3f}")
print(f"📈 R² Score: {r2:.3f}")

plot_training_curves(history)
plot_predictions(model, X_test, y_test)

print("\n🚀 Model training and evaluation completed successfully!")

joblib.dump(scaler, "scaler.save")


