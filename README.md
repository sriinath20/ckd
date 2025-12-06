```markdown
# 🩺 Digital Kidney Twin (Single-Model Edition)

**A Hybrid Intelligence System for CKD Management.**

This project combines medical mathematics (CKD-EPI 2021) with a robust AI Lifestyle Engine to create a "Digital Twin" of your kidney health.

## 🚀 Project Overview

This dashboard operates on a simple, powerful logic:

- The **"Functional" Twin (Math)**: accurately calculates your current kidney function using standard medical formulas.
- The **"Lifestyle" Twin (AI)**: predicts what your kidney health should be based on your diet, stress, and habits.
- If the Lifestyle Twin predicts a worse stage than the Functional Twin, it serves as an early warning system for hidden damage.

## 📂 File Structure

| File Name | Description |
|-----------|-------------|
| `Improved Dashboard Script.py` | 📱 The Dashboard: The main application interface. |
| `Improved Training script.py` | 🧠 The Brain: Trains the AI model on lifestyle data. |
| `ckd_dataset_with_stages.csv` | 💾 The Data: Dataset containing lifestyle vs. stage correlations. |
| `digital_twin_model.pkl` | 🤖 The Model: (Created after running the training script). |

> ⚠️ **Important Note**: Python files usually should not have spaces in their names. It is highly recommended to rename your files to `train_model.py` and `app.py` to avoid errors in the terminal.

## 🛠️ Installation & Setup

### 1. Requirements

Ensure you have Python installed. Install the necessary libraries:

```bash
pip install streamlit pandas numpy xgboost scikit-learn Pillow
```

### 2. Initialize the AI (Training)

You must train the model once so the dashboard has a "brain" to use. Run the training script:

```bash
# If using your filename with spaces, wrap it in quotes:
python "Improved Training script.py"
```

**Success Message**: `Success! Saved 'digital_twin_model.pkl'`

### 3. Launch the Dashboard

Start the application to interact with your Digital Twin:

```bash
# If using your filename with spaces:
streamlit run "Improved Dashboard Script.py"
```

## 🧠 How It Works

### Step 1: Functional Analysis (The Math)
- **Input**: Age, Sex, Serum Creatinine.
- **Engine**: Uses the CKD-EPI 2021 Formula.
- **Output**: Calculates exact eGFR and determines Current Stage (1-5).
- **Why?**: Math is 100% accurate for diagnostics.

### Step 2: Lifestyle Analysis (The AI)
- **Input**: Blood Pressure, Diet, Water Intake, Stress, BUN, Calcium.
- **Engine**: XGBoost Classifier (Trained on `ckd_dataset_with_stages.csv`).
- **Output**: Predicts the Risk Stage.
- **Why?**: It detects if your habits are creating a biological environment for advanced disease.

### Step 3: Simulation
The dashboard compares Step 1 vs. Step 2.
You can use the Simulator to adjust sliders (e.g., lower BP, better diet).
The AI instantly recalculates to show if your "Risk Stage" improves.

## ⚠️ Troubleshooting

**Q: "Model not found" error?**  
A: Make sure you ran the training script before running the dashboard. The `digital_twin_model.pkl` file must exist in the same folder.

**Q: Precision is low?**  
A: The training script automatically filters out "Stage 0" and balances the classes to ensure the AI focuses on detecting severe cases accurately.

## 📜 Medical Disclaimer

This software is a prototype for research/educational purposes only. It does not constitute medical advice.
```
