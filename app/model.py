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

# Load dataset
df = pd.read_csv(r"C:\Health monitoring\dataset\wearable_health_data.csv")
print("\n📘 Dataset Info:\n")
print(df.info())
print("\n🔹 First 5 Rows:\n", df.head())

# Features and target
X = df.drop(columns=['health_score'])
y = df['health_score']

# Encode if categorical
if y.dtype == 'object':
    le = LabelEncoder()
    y = le.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Build a simpler neural network
def build_model(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-3)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-3)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    return model

model = build_model(X_train.shape[1])
optimizer = optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# Callbacks
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

checkpoint = callbacks.ModelCheckpoint(
    filepath="app/best_health_model.keras",
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

# Train
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=120,
    batch_size=16,
    callbacks=[early_stop, checkpoint, reduce_lr],
    verbose=1
)

# Evaluate
loss, mae = model.evaluate(X_test, y_test)
print(f"\n✅ Final Test MAE: {mae:.2f}")

# Predictions
y_pred = model.predict(X_test).flatten()
mae_manual = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)

print(f"📊 MAE: {mae_manual:.3f}")
print(f"📉 RMSE: {rmse:.3f}")
print(f"📈 R² Score: {r2:.3f}")

# Visualizations
plot_training_curves(history)
plot_predictions(model, X_test, y_test)

print("\n🚀 Model training and evaluation completed successfully!")

# Save scaler
joblib.dump(scaler, os.path.join("app", "scaler.save"))



