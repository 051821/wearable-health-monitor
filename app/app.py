import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------
# 💾 Load resources (model + scaler)
# -------------------------------------
@st.cache_resource
def load_resources():
    model = load_model("app/best_health_model.keras")
    scaler = joblib.load("app/scaler.save")
    return model, scaler

model, scaler = load_resources()

# -------------------------------------
# 🧠 Page setup
# -------------------------------------
st.set_page_config(page_title="Health Score Prediction", page_icon="💪", layout="wide")
st.title("🏥 Wearable Health Monitoring Dashboard")
st.markdown("Use the sliders to input your daily health stats and get your predicted **Health Score**!")

# -------------------------------------
# 📊 Sidebar inputs
# -------------------------------------
st.sidebar.header("🩺 Enter Your Daily Health Data")

user_input = {
    'heart_rate_avg': st.sidebar.slider("Average Heart Rate (bpm)", 50, 120, 80),
    'heart_rate_max': st.sidebar.slider("Max Heart Rate (bpm)", 90, 180, 130),
    'steps': st.sidebar.slider("Steps", 1000, 20000, 7000),
    'calories_burned': st.sidebar.slider("Calories Burned", 1000, 4000, 2000),
    'sleep_duration': st.sidebar.slider("Sleep Duration (hours)", 3.0, 10.0, 7.0),
    'sleep_quality': st.sidebar.slider("Sleep Quality (0-100)", 30, 100, 70),
    'stress_level': st.sidebar.slider("Stress Level (0-100)", 0, 100, 30),
    'activity_minutes': st.sidebar.slider("Activity Minutes", 0, 180, 60),
    'oxygen_saturation': st.sidebar.slider("Oxygen Saturation (%)", 90.0, 100.0, 97.0)
}

input_df = pd.DataFrame([user_input])

# -------------------------------------
# 🧮 Preprocess and predict
# -------------------------------------
try:
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    health_score = float(prediction[0][0])
except Exception as e:
    st.error(f"⚠️ Prediction Error: {e}")
    st.stop()

# -------------------------------------
# ✅ Display prediction
# -------------------------------------
col1, col2 = st.columns(2)
with col1:
    st.metric(label="🩺 Predicted Health Score", value=f"{health_score:.2f}")
with col2:
    st.progress(min(1.0, health_score / 100))

# -------------------------------------
# 📈 Visualization
# -------------------------------------
st.markdown("### 📊 Your Input Summary")
st.dataframe(input_df.style.highlight_max(axis=1, color='lightgreen'))

st.markdown("### 🩸 Correlation Insights")
uploaded_data = pd.read_csv("C:\Health monitoring\dataset\wearable_health_data.csv")

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(uploaded_data.corr(), annot=True, cmap="YlGnBu", ax=ax)
st.pyplot(fig)

# -------------------------------------
# 💡 Health Tips
# -------------------------------------
st.markdown("---")
if health_score < 40:
    st.error("⚠️ Your health score is low. Try improving your sleep and reducing stress.")
elif health_score < 70:
    st.warning("🙂 You’re doing okay! Focus on consistent exercise and sleep.")
else:
    st.success("💪 Excellent health! Keep maintaining your active lifestyle.")

