🩺 Digital Kidney Twin AI

A Hybrid Intelligence System for Chronic Kidney Disease (CKD) Management.
Combines clinical guidelines with Machine Learning to create a "Digital Twin" for diagnosis, staging, and lifestyle simulation.

🚀 Project Overview

This system bridges the gap between static medical data and dynamic health prediction. It operates on two simultaneous layers:

Functional Twin (The "Now"): Uses mathematical precision (CKD-EPI 2021) to determine current kidney function.

Biological Twin (The "Future"): Uses AI to predict systemic risk based on lifestyle, stress, and diet.

✨ Key Features

Feature

Description

Technology

Dual-Engine AI

Separate models for Detection (Binary) and Prediction (Multi-class).

XGBoost

Precision Staging

Automated calculation of eGFR using the latest clinical formulas.

CKD-EPI 2021

Digital Twin

A virtual replica that evolves based on patient lifestyle inputs.

Python Logic

What-If Simulator

Interactive slider to simulate BP/Stress reduction and visualize outcomes.

Streamlit

🧠 System Architecture

The dashboard processes patient data through a strict 5-Step Clinical Workflow:

graph TD;
    A[Patient Input] -->|Vitals| B(Model A: Detection);
    A -->|Creatinine| C(Math: eGFR Calc);
    B --> D{CKD Detected?};
    C --> E[Determine Stage 1-5];
    E --> F[Digital Twin Analysis];
    A -->|Lifestyle Data| F;
    F --> G(Model B: Risk Prediction);
    G --> H[Simulation & Prognosis];


Input: User enters clinical vitals and lifestyle data.

Detection: ML model flags potential disease presence.

Calculation: Mathematical formula computes exact filtration rate.

Staging: Current disease stage is assigned (1-5).

Simulation: AI predicts future risk trajectory based on habits.

📂 Repository Structure

📦 digital-kidney-twin
 ┣ 📜 app_integrated.py           # 📱 Main Dashboard Application
 ┣ 📜 train_model_a.py            # 🧠 Training Script: Detection Engine
 ┣ 📜 train_model_b.py            # 🧠 Training Script: Digital Twin Engine
 ┣ 📊 kidney_disease.csv          # 💾 Dataset: Clinical Biomarkers
 ┣ 📊 ckd_dataset_with_stages.csv # 💾 Dataset: Lifestyle Factors
 ┗ 📜 README.md                   # 📄 Project Documentation


🛠️ Installation & Setup

1. Clone the Repository

git clone [https://github.com/yourusername/digital-kidney-twin.git](https://github.com/yourusername/digital-kidney-twin.git)
cd digital-kidney-twin


2. Install Dependencies

pip install streamlit pandas numpy xgboost scikit-learn Pillow


3. Initialize the AI Brains 🧠

You must train the models once before running the app.

Step A: Train Detection Model

python train_model_a.py
# Output: model_a_detection.pkl


Step B: Train Prediction Model

python train_model_b.py
# Output: model_b_prediction.pkl


4. Launch the Dashboard 🚀

streamlit run app_integrated.py


🔮 Simulator Guide

The What-If Simulator allows patients to visualize the impact of lifestyle changes:

Analyze: Check the "Future Trajectory" card. If Predicted Stage > Current Stage, risk is high.

Simulate: Scroll to the bottom simulation panel.

Adjust: Lower the Target Blood Pressure slider.

Run: Click Run Simulation and watch the Digital Twin update its prediction in real-time.

⚠️ Medical Disclaimer

Prototype Only: This software is for educational and research purposes. The eGFR calculations and ML predictions should not replace professional medical advice. Always consult a nephrologist.

📜 License

Distributed under the MIT License. See LICENSE for more information.
