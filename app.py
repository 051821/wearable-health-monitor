import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib

st.cache_data.clear()
st.cache_resource.clear()


@st.cache_resource
def load_resources():
    with tf.device("/CPU:0"):  # ensure clean load context
        model = load_model("best_health_model.keras", compile=False)
    scaler = joblib.load("scaler.save")
    df = pd.read_csv(r"C:\Health monitoring\dataset\wearable_health_data.csv")
    return model, scaler, df

model, scaler, df = load_resources()
feature_names = list(df.drop(columns=["health_score"]).columns)


st.set_page_config(page_title="🏥 Health Score Predictor", layout="wide")
st.title("🏥 Health Monitoring - Health Score Prediction")
st.markdown("Predict health score from wearable data using your trained ML model.")

st.sidebar.header("Enter Health Data")


user_input = {}
for feature in feature_names:
    user_input[feature] = st.sidebar.number_input(f"{feature}", min_value=0.0, step=0.1)


input_df = pd.DataFrame([user_input])
scaled_input = scaler.transform(input_df)


if st.sidebar.button("🔍 Predict Health Score"):
    with tf.device("/CPU:0"):  
        pred = model.predict(scaled_input, verbose=0)[0][0]
    st.success(f"🩺 Predicted Health Score: **{pred:.2f}**")


with st.expander("📊 View Sample Data"):
    st.dataframe(df.head())


with st.expander("📈 Example Visualization"):
    fig, ax = plt.subplots()
    sample_y = df["health_score"].values[:50]
    ax.plot(sample_y, label="Sample Actual Health Scores", color='blue')
    ax.set_title("Sample Health Score Distribution")
    ax.legend()
    st.pyplot(fig)

st.caption("🚀 Powered by TensorFlow + Streamlit")


