# 🏥 Health Monitoring - Health Score Prediction

This project predicts a **Health Score** using wearable sensor data such as heart rate, sleep hours, step count, calories burned, and more.  
Built with **TensorFlow**, **Streamlit**, and **Scikit-learn**, it combines machine learning and an interactive web dashboard for real-time insights.

---

## 🚀 Features

- 📈 **Health Score Prediction** using a trained TensorFlow model  
- 📊 **Interactive Streamlit Dashboard** for real-time input and visualization  
- 🧠 **Preprocessing & Scaling** with saved Scikit-learn `StandardScaler`  
- 💾 **Trained Model Saved** as `.keras` for reproducible predictions  
- 📉 **Data Visualization** for sample distributions and model insights  

---

## 🧩 Project Structure

📦 Health Monitoring
├── app.py # Streamlit UI for health score prediction
├── model.py # Model training and evaluation script
├── visualization.py # Data visualization and analysis
├── best_health_model.keras # Saved trained model
├── scaler.save # Saved StandardScaler for input normalization
├── dataset/
│ └── wearable_health_data.csv
├── requirements.txt # All Python dependencies
└── README.md # Project documentation


## ⚙️ Setup Instructions

### 1️⃣ Clone this repository

git clone https://github.com/your-username/health-monitoring.git
cd health-monitoring
2️⃣ Create and activate a virtual environment
bash
Copy code
python -m venv venv
venv\Scripts\activate    # on Windows
# or
source venv/bin/activate # on Mac/Linux
3️⃣ Install dependencies
bash
Copy code
pip install -r requirements.txt
🧠 Model Training
Run the training script to retrain or fine-tune the model:


python model.py
This will:

Load the dataset

Preprocess and scale features

Train a neural network

Save the trained model as best_health_model.keras

🌐 Run the Streamlit App
After training, launch the interactive web app:

streamlit run app.py
You’ll see:

A sidebar to input health parameters

A predicted health score on the main page

Expandable visualizations and dataset samples

📈 Example Metrics
Metric	Score
MAE	0.99
RMSE	1.26
R²	0.974

(Lower MAE/RMSE and higher R² indicate strong performance.)

📊 Visualization Preview
Line chart of sample health scores

Feature correlations (optional in visualization.py)

Future scope: trend graphs, feature importances, anomaly detection

💡 Future Enhancements
Add more health parameters (e.g., blood oxygen, stress levels)

Integrate live IoT/wearable device data

Deploy on Streamlit Cloud or AWS
