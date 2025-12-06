## 🩺 Digital Kidney Twin (Clinical Edition)

> **A Hybrid Intelligence System for CKD Risk Profiling**

This project separates **diagnosis** from **prognosis**:  
- ✅ **Diagnosis**: Uses medical mathematics (CKD-EPI 2021) to determine your **current CKD stage** (1–5) based on serum creatinine.  
- 🔮 **Prognosis**: Uses clinical AI to predict your **future risk profile** based on systemic health markers (**BP, Diabetes, Anemia**, etc.).

> **Key Innovation**: The AI **does NOT use creatinine**. It assesses kidney stress through the lens of whole-body physiology.

---

## 🚀 Project Overview

Solves the **"Circular Logic"** problem in chronic kidney disease (CKD) management by decoupling:
- **Functional Engine (Math)**: Calculates exact eGFR and CKD stage.
- **Risk Engine (AI)**: Predicts "High Severity" risk using systemic biomarkers (Blood Pressure, Glucose, Hemoglobin, Albumin, etc.).

This separation enables clinicians to see **both current status and future trajectory**—even before creatinine worsens.

---

## 📂 File Structure

| File Name                         | Description |
|----------------------------------|-------------|
| `app.py`   | 🖥️ **The Application**: Interactive 3-tab Streamlit dashboard |
| `train.py`    | 🧠 **The Training Logic**: Preprocesses data & trains the risk model |
| `kidney_disease.csv`             | 💾 **The Dataset**: Clinical records (BP, Hemo, Albumin, Glucose, etc.) |
| `kidney_risk_model.pkl`          | 🤖 **The Artifact**: Trained AI model (auto-generated after training) |
| `requirements.txt`          | 💾 **Requirements**: Reqired Packages need to be Installed |

> ⚠️ **All files must reside in the same directory.**

---

## 🛠️ Installation & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```bash
streamlit run "app.py"
```
The app will open in your browser at `http://localhost:8501`.

---

## 🧠 How It Works

### **Tab 1: Patient Data Entry**
- **Mandatory**: Age, Sex, Serum Creatinine (for eGFR).
- **Clinical Inputs**:  
  - Systolic & Diastolic Blood Pressure  
  - Random Glucose  
  - Hemoglobin  
  - Albumin  
  - Hypertension Status (Yes/No)  
  - Diabetes Status (Yes/No)

> 💡 All inputs mirror standard hospital lab reports.

---

### **Tab 2: Diagnosis & Risk**
- **Current Status**:  
  - eGFR (mL/min/1.73m²)  
  - CKD Stage (1–5) — *calculated via CKD-EPI 2021 formula*
- **Risk Analysis**:  
  - **Systemic Vulnerability Score** (0–100%)  
    - **Low (<30%)**: Systemic factors well-controlled  
    - **High (>70%)**: Biomarker pattern resembles advanced CKD  

> 🔍 The AI sees risk **before** kidney function declines.

---

### **Tab 3: What-If Simulation**
- Adjust sliders for BP, Glucose, Hemo, etc.
- Watch **real-time risk updates** as you modify inputs.
- **Key Insight**: Risk drops **immediately** with improved systemic control—even if creatinine hasn’t changed yet.

> 🌟 Proves that managing **whole-body health** directly protects kidney outcomes.

---

## ⚠️ Troubleshooting

| Issue | Solution |
|------|--------|
| `Model file not found` | You skipped training! Run `python "Improved Training script.py"` first. |
| `X does not have valid feature names` | Fixed in the **Improved** scripts—ensure you’re using the latest versions. |

---

## 📜 Disclaimer

> ⚠️ **This tool is a research prototype for educational purposes only.**  
> - Trained on the [UCI Chronic Kidney Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Chronic_Kidney_Disease).  
> - eGFR calculations follow CKD-EPI 2021 but **must be validated by a licensed clinician**.  
> - **Not a substitute for professional medical advice, diagnosis, or treatment.**
---

