# Wearable Health Monitoring & Health Score Prediction System

## A Production-Ready ML + DevOps + Automation Project

This project builds an end-to-end machine learning pipeline that predicts an individual's daily health score using sensor data from wearable devices. It integrates TensorFlow, Streamlit, Docker, Jenkins, Terraform, Puppet, and Nagios to deliver a fully automated, monitored, and containerized ML system.

---

## Objective

The goal of this project is to design and deploy a real-world, production-style health prediction system that accomplishes the following:

- Uses deep learning to generate personalized health scores
- Provides an interactive UI for real-time predictions
- Runs inside a Docker container for easy deployment
- Uses Jenkins for automated CI/CD
- Uses Terraform to provision the environment
- Uses Puppet to configure Windows agents
- Uses Nagios to continuously monitor ML app uptime and CPU health

---

## Project Workflow (End-to-End)

### 1. Data Processing & Model Training

Wearable health data is collected, including steps, sleep patterns, heart rate, stress levels, calories burned, and other relevant metrics. The data is then cleaned and scaled using Scikit-learn's StandardScaler. A deep learning regression model is trained using TensorFlow/Keras, and the best-performing model is saved in both Keras (.keras) and ONNX format for deployment.

### 2. Model Deployment Using Docker

The Streamlit app, along with the model, scaler, and visualization scripts, are packaged inside a Docker image. The container exposes the Streamlit UI on port 8501, and the model runs efficiently in ONNX format for faster inference during production.

### 3. CI/CD Pipeline with Jenkins

The Jenkins pipeline automates several critical tasks:

- Pulling the latest code from GitHub
- Training the TensorFlow model
- Building the Docker image
- Running Terraform to deploy the environment
- Performing a basic health check of the deployed service

This ensures continuous testing, continuous deployment, and infrastructure automation throughout the development lifecycle.

### 4. Infrastructure Automation with Terraform

Terraform handles infrastructure provisioning by:

- Uploading Puppet and Nagios configuration files to the Ubuntu VM
- Managing Nagios monitoring configuration
- Deploying the Docker container
- Ensuring reproducible deployment through Infrastructure-as-Code principles

### 5. Configuration Management with Puppet

The Puppet module takes care of configuration management on the Windows agent by:

- Creating monitoring directories on the Windows agent
- Adding a PowerShell health check script
- Scheduling automatic start of Docker Desktop
- Ensuring the Windows node stays ready for monitoring

### 6. Continuous Monitoring with Nagios

Nagios provides continuous monitoring of the system by tracking:

- Whether the ML app is running on port 8501
- CPU load of the Windows agent
- General availability of the host machine

Alerts are triggered if the Streamlit app goes down, the host becomes unreachable, or CPU load crosses the defined threshold.

---

## Directory Structure

```
project-root/
│
├── app/               # Streamlit app, model files, dataset, ONNX model, scaler
├── jenkins/           # Jenkins pipeline file
├── terraform/         # Terraform modules for Puppet + Nagios + Docker
├── puppet-files/      # Puppet manifest for Windows
├── nagios/            # Nagios host/service configuration
├── venv/              # Local virtual environment
└── README.md          # Project documentation
```

---

## Features

### Machine Learning
- Deep learning regression model for health score prediction
- Comprehensive data normalization and preprocessing pipeline
- Model achieves an R² score of 0.95

### Deployment
- Fully containerized using Docker for consistent environments
- Fast ONNX inference for production-ready performance
- Streamlit UI for real-time predictions and user interaction

### DevOps Automation
- CI/CD pipeline with Jenkins for automated builds and deployments
- Infrastructure provisioning with Terraform
- Configuration automation with Puppet
- Monitoring and alerting via Nagios

### Observability
- Host health checks to ensure system availability
- App port monitoring to verify service status
- CPU load monitoring to track resource utilization

---

## How to Run the Project

### 1. Clone the repository

Download or clone the project to your local system.

### 2. Install Python dependencies (optional if using Docker)

Use the provided `requirements.txt` inside the `app/` folder to install necessary Python packages.

### 3. Start the Streamlit app

The app can run locally or inside Docker, depending on your preference and deployment needs.

### 4. Run with Docker

Build and run the container to launch the ML app on port 8501.

### 5. Optional: Run Jenkins Pipeline

Execute the Jenkinsfile stages to automate:
- Model training
- Docker build
- Terraform deployment
- Health check validation

### 6. Deploy Infrastructure Using Terraform

Initialize Terraform and apply the configuration to deploy:
- Puppet module
- Nagios configuration
- Docker container

### 7. Enable Monitoring

Configure Nagios server to monitor:
- ML app on port 8501
- CPU load on the Windows agent
- Windows host availability

---

## System Architecture (Summary)

The system follows this workflow:

1. Developer pushes code to GitHub
2. Jenkins pipeline triggers, which trains the model and builds the Docker image
3. Terraform applies infrastructure changes, provisions resources, and deploys the Docker container
4. Puppet auto-configures the Windows node for monitoring
5. Nagios monitors uptime, CPU load, and service health continuously
6. Streamlit app serves predictions to end users through an intuitive interface

---

## Conclusion

This project demonstrates a complete ML + MLOps + Automation pipeline, combining machine learning, containerization, infrastructure-as-code, configuration management, and continuous monitoring. It reflects how real-world production ML systems are built and maintained in modern organizations, providing a practical example of industry-standard practices for deploying and managing machine learning applications at scale.