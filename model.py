import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from visualization import plot_training_curves, plot_predictions

# Disable oneDNN log info
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# 1️⃣ Load dataset
df = pd.read_csv(r"C:\Health monitoring\dataset\wearable_health_data.csv")
print("\nDataset Info:\n")
print(df.info())

# 2️⃣ Split features and target
X = df.drop(columns=['health_score'])
y = df['health_score']

# 3️⃣ Encode labels if categorical
if y.dtype == 'object':
    le = LabelEncoder()
    y = le.fit_transform(y)

# 4️⃣ Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5️⃣ Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 6️⃣ Define deep learning model (Regression)
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)  # Regression output
])

# 7️⃣ Compile model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 8️⃣ Train model
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=16,
    verbose=1
)

# 9️⃣ Evaluate model
loss, mae = model.evaluate(X_test, y_test)
print(f"\n✅ Test MAE: {mae:.2f}\n")

# 🔟 Visualizations
plot_training_curves(history)
plot_predictions(model, X_test, y_test)



