# 🩺 Digital Kidney Twin (Single-Model Edition)

**A Hybrid Intelligence System for CKD Management.**

This project combines medical mathematics (CKD-EPI 2021) with a robust AI Lifestyle Engine to create a "Digital Twin" of your kidney health.

## 🚀 Project Overview

This dashboard operates on a simple, powerful logic:

- The **"Functional" Twin (Math)**: accurately calculates your current kidney function using standard medical formulas.
- The **"Lifestyle" Twin (AI)**: predicts what your kidney health should be based on your diet, stress, and habits.
- If the Lifestyle Twin predicts a worse stage than the Functional Twin, it serves as an early warning system for hidden damage.

## 📂 File Structure

| File Name                      | Description                                               |
|--------------------------------|-----------------------------------------------------------|
| `Improved Dashboard Script.py` | 📱 The Dashboard: The main application interface.         |
| `Improved Training script.py`  | 🧠 The Brain: Trains the AI model on lifestyle data.      |
| `ckd_dataset_with_stages.csv`  | 💾 The Data: Dataset containing lifestyle vs. stage correlations. |
| `digital_twin_model.pkl`       | 🤖 The Model: (Created after running the training script). |

> ⚠️ **Important Note**: Python files usually should not have spaces in their names. It is highly recommended to rename your files to `train_model.py` and `app.py` to avoid errors in the terminal.

## 🛠️ Installation & Setup

### 1. Requirements

Ensure you have Python installed (preferably 3.8+). Install the required packages:

```bash
pip install streamlit pandas numpy xgboost scikit-learn Pillow
```
### 2. Initialize the AI (Training)

You must train the model once so the dashboard has a "brain" to use. Run the training script:

```bash
# If using the original filename with spaces:
python "Improved Training script.py"
```

✅ **Success Message**: `Success! Saved 'digital_twin_model.pkl'`

💡 **Tip**: Rename the script to `train_model.py` for cleaner commands:  
```bash
python train_model.py
```

### 3. Launch the Dashboard

Start the Streamlit app to interact with your Digital Kidney Twin:

```bash
# If using the original filename:
streamlit run "Improved Dashboard Script.py"
```

💡 **Tip**: After renaming, use:  
```bash
streamlit run app.py
```

Your browser will open automatically at `http://localhost:8501`.

## 🧠 How It Works

### Step 1: Functional Analysis (The Math)
- **Inputs**: Age, Sex, Serum Creatinine  
- **Engine**: CKD-EPI 2021 equation (gold standard for eGFR estimation)  
- **Output**: Exact eGFR value and Current CKD Stage (1–5)  
- **Purpose**: Provides a clinically validated baseline of kidney function.

### Step 2: Lifestyle Analysis (The AI)
- **Inputs**: Blood Pressure, Diet Quality, Water Intake, Stress Level, BUN, Calcium  
- **Engine**: XGBoost Classifier trained on real-world CKD lifestyle data  
- **Output**: Predicted "Risk Stage" based on modifiable habits  
- **Purpose**: Flags if your lifestyle is pushing you toward worse kidney health than your current lab values suggest.

### Step 3: Interactive Simulation
- Adjust sliders for lifestyle factors (e.g., reduce sodium, increase water)  
- Watch your AI-predicted "Risk Stage" update in real time  
- **Goal**: Align your Lifestyle Twin with your Functional Twin—or make it better!

## ⚠️ Troubleshooting

### ❌ "Model not found" error
Make sure you ran the training script first. The file `digital_twin_model.pkl` must be in the same directory as the dashboard script.

### 📉 Low model accuracy?
The training pipeline:
- Removes ambiguous "Stage 0" entries
- Applies class balancing (SMOTE or class weights)
- Focuses on distinguishing moderate-to-severe CKD (Stages 3–5)

For best results, ensure your dataset has diverse, high-quality lifestyle labels.

### 💥 Filename issues with spaces
Avoid spaces in filenames. Rename:
- `"Improved Training script.py"` → `train_model.py`
- `"Improved Dashboard Script.py"` → `app.py`

Then use:
```bash
python train_model.py
streamlit run app.py
```

## 📜 Medical Disclaimer

> **This software is a research prototype for educational and demonstration purposes only.**  
> It is **not** a medical device and **does not** provide medical advice, diagnosis, or treatment.  
> Always consult a licensed nephrologist or healthcare provider for kidney health concerns.  
> Do not make clinical decisions based solely on this tool’s output.
```
