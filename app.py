import streamlit as st
import numpy as np
import pandas as pd
import pickle
from PIL import Image, ImageDraw

# ================================
# DIGITAL KIDNEY TWIN v3.0
# 3-Tab Dashboard Implementation
# ================================

# --- 1. MATHEMATICAL LAYER (eGFR & Scoring) ---
def calculate_egfr(creatinine, age, sex):
    """Calculates eGFR using CKD-EPI 2021 Equation."""
    if sex == "Male":
        k, alpha = 0.9, -0.302
    else:
        k, alpha = 0.7, -0.241
    
    factor = 1 if sex == "Male" else 1.012
    
    if creatinine <= k:
        egfr = 142 * ((creatinine / k) ** alpha) * (0.9938 ** age) * factor
    else:
        egfr = 142 * ((creatinine / k) ** -1.200) * (0.9938 ** age) * factor
    return round(egfr, 2)

def get_health_score_and_color(egfr):
    """
    Maps eGFR to Kidney Health Score and Color based on Dashboard Details.docx.
    """
    if egfr >= 90:
        return "Stage 1", 95, "#4CAF50", "Excellent kidney function" # Green
    elif egfr >= 60:
        return "Stage 2", 82, "#AED581", "Good health, mild decline" # Green/Yellow (Light Green)
    elif egfr >= 45:
        return "Stage 3a", 65, "#FFEB3B", "Moderate reduction; caution needed" # Yellow
    elif egfr >= 30:
        return "Stage 3b", 45, "#FF9800", "Poor function, high risk" # Orange
    elif egfr >= 15:
        return "Stage 4", 25, "#F44336", "Severe loss; very high risk" # Red
    else:
        return "Stage 5 / ESRD", 10, "#8B0000", "Kidney failure; critical" # Dark Red

# --- 2. VISUALIZATION LAYER ---
def create_kidney_image(color_hex):
    img = Image.new("RGBA", (200, 200), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    # Bean shape approximation
    draw.ellipse((50, 50, 150, 180), fill=color_hex, outline="black", width=2)
    draw.ellipse((70, 40, 160, 150), fill=color_hex, outline="black", width=2)
    # Indentation
    draw.ellipse((30, 80, 70, 130), fill="white", outline=None) 
    return img

# --- 3. ML LAYER (Systemic Risk) ---
def load_artifacts():
    try:
        with open('kidney_risk_model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

def predict_systemic_risk(artifacts, inputs):
    """Predicts probability of 'Severe Profile' using ML."""
    model = artifacts['model']
    imputer = artifacts['imputer']
    
    feature_names = [
        'age', 'bp', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba',
        'bgr', 'bu', 'pot', 'wc', 'htn', 'dm', 'cad', 'pe', 'ane', 'hemo'
    ]
    
    # Create DataFrame to ensure column order is correct
    input_df = pd.DataFrame([inputs], columns=feature_names)
    
    # FIX: Convert to numpy array to avoid feature name mismatch errors in sklearn
    # Some sklearn versions are strict about feature names if passed a DataFrame.
    # Passing values ensures it treats it as a raw vector (safe since we ordered it above).
    imputed_vector = imputer.transform(input_df.values)
    
    prob = model.predict_proba(imputed_vector)[0][1]
    return prob

# --- MAIN APP ---
def main():
    st.set_page_config(page_title="Digital Kidney Twin", layout="wide")
    st.title("🩺 Digital Kidney Twin Dashboard")

    artifacts = load_artifacts()
    if not artifacts:
        st.error("⚠️ Model file not found. Please run `python train_revised.py` first.")
        return

    # Initialize Session State for Inputs if not present
    if 'inputs' not in st.session_state:
        st.session_state['inputs'] = {}

    # --- TABS CONFIGURATION ---
    tab1, tab2, tab3 = st.tabs(["1. Patient Data Entry", "2. Diagnosis & Health", "3. Digital Twin Simulation"])

    # ==========================================
    # TAB 1: DATA INPUTS
    # ==========================================
    with tab1:
        st.header("📝 Patient Clinical Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Mandatory for eGFR")
            name = st.text_input("Patient Name", "John Doe")
            age = st.number_input("Age (years)", 1, 120, 55)
            sex = st.selectbox("Sex", ["Male", "Female"])
            creatinine = st.number_input("Serum Creatinine (mg/dL)", 0.4, 20.0, 1.2, step=0.1)
        
        with col2:
            st.subheader("Digital Twin Parameters (ML)")
            bp = st.number_input("Blood Pressure (Diastolic mmHg)", 50, 200, 85)
            bgr = st.number_input("Glucose (Random) mg/dL", 70, 500, 140)
            bu = st.number_input("Blood Urea", 10, 300, 45)
            hemo = st.number_input("Hemoglobin (g/dL)", 3.0, 18.0, 13.5)
            pot = st.number_input("Potassium (mmol/L)", 2.0, 8.0, 4.5)
            wc = st.number_input("White Blood Cell Count", 2000, 30000, 9000)
            
        with st.expander("Additional Clinical History"):
            c1, c2, c3 = st.columns(3)
            with c1:
                htn = st.selectbox("Hypertension", ["yes", "no"])
                dm = st.selectbox("Diabetes", ["yes", "no"])
            with c2:
                cad = st.selectbox("Coronary Artery Disease", ["no", "yes"])
                pe = st.selectbox("Pedal Edema", ["no", "yes"])
            with c3:
                ane = st.selectbox("Anemia", ["no", "yes"])
                al = st.slider("Albumin Level (0-5)", 0, 5, 0)
                su = st.slider("Sugar Level (0-5)", 0, 5, 0)

        # Store in session state for other tabs
        st.session_state['inputs'] = {
            'name': name, 'age': age, 'sex': sex, 'creatinine': creatinine,
            'bp': bp, 'bgr': bgr, 'bu': bu, 'hemo': hemo, 'pot': pot, 'wc': wc,
            'htn': htn, 'dm': dm, 'cad': cad, 'pe': pe, 'ane': ane, 'al': al, 'su': su
        }
        
        st.info("👉 After filling details, switch to **'Diagnosis & Health'** tab.")

    # ==========================================
    # TAB 2: DIAGNOSIS
    # ==========================================
    with tab2:
        st.header("🔍 Diagnosis & Risk Assessment")
        
        inputs = st.session_state['inputs']
        
        # 1. Calculations
        egfr = calculate_egfr(inputs['creatinine'], inputs['age'], inputs['sex'])
        stage, score, color, meaning = get_health_score_and_color(egfr)
        
        # ML Feature Prep
        features = [
            inputs['age'], inputs['bp'], inputs['al'], inputs['su'], 0, 0, 0, 0, # rbc, pc etc placeholders
            inputs['bgr'], inputs['bu'], inputs['pot'], inputs['wc'],
            1 if inputs['htn']=='yes' else 0, 1 if inputs['dm']=='yes' else 0,
            1 if inputs['cad']=='yes' else 0, 1 if inputs['pe']=='yes' else 0,
            1 if inputs['ane']=='yes' else 0, inputs['hemo']
        ]
        
        risk_prob = predict_systemic_risk(artifacts, features)
        
        # 2. Display Layout
        d_col1, d_col2, d_col3 = st.columns([1, 1, 1])
        
        with d_col1:
            st.markdown("### eGFR & Stage")
            st.metric("eGFR Value", f"{egfr}", "mL/min/1.73m²")
            st.markdown(f"**Current Status:** {stage}")
            if egfr >= 90: st.success(meaning)
            elif egfr >= 60: st.warning(meaning)
            else: st.error(meaning)

        with d_col2:
            st.markdown("### Kidney Health Score")
            st.image(create_kidney_image(color), width=150, caption=f"Health Score: {score}/100")
            st.metric("Health Score", f"{score}/100")

        with d_col3:
            st.markdown("### Risk Analysis")
            st.metric("Systemic Vulnerability", f"{risk_prob:.1%}")
            if risk_prob < 0.3:
                st.success("Low Systemic Risk")
            elif risk_prob < 0.6:
                st.warning("Moderate Systemic Risk")
            else:
                st.error("High Systemic Risk")
            st.caption("Based on BP, Glucose, Hemoglobin, etc.")

    # ==========================================
    # TAB 3: SIMULATION
    # ==========================================
    with tab3:
        st.header("🔮 Digital Twin 'What-If' Simulation")
        st.write("Adjust parameters to simulate future scenarios.")

        inputs = st.session_state['inputs']
        
        # Simulation Controls
        s_col1, s_col2 = st.columns(2)
        with s_col1:
            sim_creatinine = st.slider("Simulate Creatinine Change", 0.5, 10.0, inputs['creatinine'], 0.1)
        with s_col2:
            sim_bp = st.slider("Simulate BP Control", 50, 180, inputs['bp'])
            
        # Sim Calculations
        sim_egfr = calculate_egfr(sim_creatinine, inputs['age'], inputs['sex'])
        sim_stage, sim_score, sim_color, sim_meaning = get_health_score_and_color(sim_egfr)
        
        # Sim ML Risk
        sim_features = features.copy()
        sim_features[1] = sim_bp # Update BP in feature vector
        sim_risk_prob = predict_systemic_risk(artifacts, sim_features)
        
        st.markdown("---")
        st.subheader("📊 Comparison: Current vs Simulated")
        
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.markdown("**eGFR Status**")
            st.metric("New eGFR", sim_egfr, delta=f"{round(sim_egfr - egfr, 2)}")
            st.write(f"Stage: {stage} ➝ **{sim_stage}**")
            
        with c2:
            st.markdown("**Kidney Health**")
            col1, col2 = st.columns(2)
            with col1:
                st.image(create_kidney_image(color), width=80, caption="Current")
            with col2:
                st.image(create_kidney_image(sim_color), width=80, caption="Simulated")
            st.metric("Score Change", f"{sim_score}", delta=f"{sim_score - score}")
            
        with c3:
            st.markdown("**Risk Profile**")
            risk_delta = sim_risk_prob - risk_prob
            st.metric("New Risk Probability", f"{sim_risk_prob:.1%}", delta=f"{risk_delta:.1%}", delta_color="inverse")
            
            if sim_risk_prob < risk_prob:
                st.success("✅ Risk Reduced!")
            else:
                st.warning("⚠️ Risk Increased")

if __name__ == "__main__":
    main()
