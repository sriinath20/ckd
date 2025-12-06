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
