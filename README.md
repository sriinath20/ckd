🩺 Digital Kidney Twin: Integrated Diagnostic & Prognostic Dashboard

A comprehensive Hybrid Intelligence System for Chronic Kidney Disease (CKD) management. This project creates a "Digital Twin" of a patient's kidney to detect current disease status, calculate precise eGFR, and predict future risk trajectories based on lifestyle factors.

🚀 Key Features

Hybrid Architecture: Combines clinical guidelines (CKD-EPI 2021) with Machine Learning (XGBoost).

Dual-Model System:

Model A (Detection): Detects CKD presence using clinical biomarkers (UCI Dataset).

Model B (Digital Twin): Predicts risk severity based on lifestyle, diet, and stress (Lifestyle Dataset).

Interactive Digital Twin: Visualizes kidney health changes in real-time.

What-If Simulation: Simulates how changes in Blood Pressure, Stress, or Diet impact the patient's future risk stage.

🧠 System Architecture (The 5-Step Workflow)

The dashboard follows a strict 5-step clinical logic flow:

Input: User enters clinical vitals (Creatinine, Age) and lifestyle data (Diet, Stress).

Detection (Model A): ML model analyzes biomarkers to flag "CKD Detected" or "Healthy".

eGFR Calculation: Uses the CKD-EPI 2021 Formula (Math-based, no ML) for medical accuracy.

Staging: Determines Current Stage (1-5) via standard lookup tables.

Digital Twin (Model B): ML model predicts the Risk Trajectory and runs simulation scenarios.

📂 File Structure

File Name

Description

app_integrated.py

The main Streamlit Dashboard application.

train_model_a.py

Training script for Detection (Binary Classification). Uses kidney_disease.csv.

train_model_b.py

Training script for Digital Twin/Risk (Multi-class Classification). Uses ckd_dataset_with_stages.csv.

kidney_disease.csv

Dataset 1: Clinical biomarkers for CKD detection.

ckd_dataset_with_stages.csv

Dataset 2: Lifestyle and systemic factors for risk prediction.

🛠️ Installation & Setup

1. Clone the Repository

git clone [https://github.com/yourusername/digital-kidney-twin.git](https://github.com/yourusername/digital-kidney-twin.git)
cd digital-kidney-twin


2. Install Dependencies

Ensure you have Python 3.8+ installed. Run:

pip install streamlit pandas numpy xgboost scikit-learn Pillow


3. Prepare the Data

Ensure kidney_disease.csv and ckd_dataset_with_stages.csv are in the root directory.

⚡ Usage Instructions

This system requires a one-time training step before the dashboard can run.

Step 1: Train Model A (Detection)

Run the first training script to create the detection engine.

python train_model_a.py


Output: model_a_detection.pkl

Step 2: Train Model B (Digital Twin)

Run the second training script to create the lifestyle risk engine.

python train_model_b.py


Output: model_b_prediction.pkl

Step 3: Launch the Dashboard

Start the application.

streamlit run app_integrated.py


🔮 How to Use the Simulator

Navigate to the Digital Twin section on the dashboard (Right column).

Observe the Predicted Risk Level vs. your Current Stage.

If Risk Level > Current Stage: Your lifestyle is accelerating disease progression.

Scroll to "What-If Simulation".

Adjust the Simulate Lower BP slider or change Stress Level.

Click "Run Simulation" to see if your predicted risk stage improves.

⚠️ Medical Disclaimer

This software is a prototype designed for educational and research purposes only. The eGFR calculations and ML predictions should not replace professional medical advice, diagnosis, or treatment. Always consult with a nephrologist for kidney health concerns.

📜 License

MIT License
