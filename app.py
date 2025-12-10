import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
from PIL import Image, ImageDraw
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

# ================================
# DIGITAL KIDNEY TWIN v3.1 (Self-Healing)
# ================================

# --- 1. TRAINING ENGINE (Self-Healing Logic) ---
# We include the training logic here so the app can rebuild the model 
# if the pickle file is from a different scikit-learn version.
@st.cache_resource
def train_and_save_model():
    """Trains the model from scratch using current environment libraries."""
    try:
        # Load Data
        if not os.path.exists('kidney_disease.csv'):
            return None # Cannot train without data
            
        df = pd.read_csv('kidney_disease.csv')
        
        # Preprocessing (Mirroring train_revised.py)
        if 'id' in df.columns: df = df.drop('id', axis=1)
        
        df['classification'] = df['classification'].astype(str).str.strip().str.lower().replace({'ckd\t': 'ckd'})
        df['classification_numeric'] = df['classification'].map({'ckd': 1, 'notckd': 0, 'no': 0})
        df['sc'] = pd.to_numeric(df['sc'], errors='coerce')
        
        # Target: Severe Risk (CKD + SC > 2.0)
        df['severe_risk'] = ((df['classification_numeric'] == 1) & (df['sc'] > 2.0)).astype(int)
        
        # Cleaning features
        def clean_col(df, col, mapping):
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower().replace(mapping)
                df[col] = pd.to_numeric(df[col], errors='coerce')
            return df

        df = clean_col(df, 'htn', {'yes': 1, 'no': 0})
        df = clean_col(df, 'dm', {'yes': 1, 'no': 0, '\tyes': 1, ' yes': 1})
        df = clean_col(df, 'cad', {'yes': 1, 'no': 0, '\tno': 0})
        df = clean_col(df, 'pe', {'yes': 1, 'no': 0})
        df = clean_col(df, 'ane', {'yes': 1, 'no': 0})
        
        # Other nominals
        df = clean_col(df, 'rbc', {'abnormal': 1, 'normal': 0})
        df = clean_col(df, 'pc', {'abnormal': 1, 'normal': 0})
        df = clean_col(df, 'pcc', {'present': 1, 'notpresent': 0})
        df = clean_col(df, 'ba', {'present': 1, 'notpresent': 0})
        
        # Numerics
        num_cols = ['age', 'bp', 'bgr', 'bu', 'hemo', 'pot', 'wc', 'rc', 'al', 'su']
        for col in num_cols:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')

        feature_cols = [
            'age', 'bp', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba',
            'bgr', 'bu', 'pot', 'wc', 'htn', 'dm', 'cad', 'pe', 'ane', 'hemo'
        ]
        
        # Filter valid columns
        valid_cols = [c for c in feature_cols if c in df.columns]
        X = df[valid_cols]
        y = df['severe_risk']
        
        # Train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        imputer = SimpleImputer(strategy='median')
        X_train_imputed = imputer.fit_transform(X_train)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X_train_imputed, y_train)
        
        artifacts = {
            "model": model,
            "imputer": imputer,
            "features": valid_cols
        }
        
        # Save to disk
        with open('kidney_risk_model.pkl', 'wb') as f:
            pickle.dump(artifacts, f)
            
        return artifacts
        
    except Exception as e:
        st.error(f"Auto-Training Failed: {e}")
        return None

# --- 2. MATHEMATICAL LAYER (eGFR & Scoring) ---
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

# --- 3. VISUALIZATION LAYER ---
def create_kidney_image(color_hex):
    img = Image.new("RGBA", (200, 200), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    # Bean shape approximation
    draw.ellipse((50, 50, 150, 180), fill=color_hex, outline="black", width=2)
    draw.ellipse((70, 40, 160, 150), fill=color_hex, outline="black", width=2)
    # Indentation
    draw.ellipse((30, 80, 70, 130), fill="white", outline=None) 
    return img

# --- 4. ML LAYER (Systemic Risk) ---
def load_artifacts():
    # Attempt to load existing model
    artifacts = None
    if os.path.exists('kidney_risk_model.pkl'):
        try:
            with open('kidney_risk_model.pkl', 'rb') as f:
                artifacts = pickle.load(f)
                # Test the artifacts to ensure version compatibility
                # We try a dummy transform. If it fails (AttributeError), we force retrain.
                dummy_input = np.zeros((1, len(artifacts['features'])))
                artifacts['imputer'].transform(dummy_input)
        except (AttributeError, ModuleNotFoundError, Exception) as e:
            # print(f"Model incompatible or corrupt ({e}), retraining...")
            artifacts = None # Force retrain

    # If load failed or file missing, train now
    if artifacts is None:
        with st.spinner("Initializing AI Model (One-time setup)..."):
            artifacts = train_and_save_model()
            
    return artifacts

def predict_systemic_risk(artifacts, inputs):
    """Predicts probability of 'Severe Profile' using ML."""
    model = artifacts['model']
    imputer = artifacts['imputer']
    features = artifacts['features']
    
    # Create DataFrame to ensure column order is correct
    # We construct a DF with 0s for missing features to be safe
    input_df = pd.DataFrame(0, index=[0], columns=features)
    
    # Fill known values
    feature_names = [
        'age', 'bp', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba',
        'bgr', 'bu', 'pot', 'wc', 'htn', 'dm', 'cad', 'pe', 'ane', 'hemo'
    ]
    
    # Map input list to the DataFrame columns safely
    for i, col in enumerate(feature_names):
        if col in input_df.columns:
            input_df[col] = inputs[i]
    
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
        st.error("⚠️ Model could not be loaded or trained. Please ensure 'kidney_disease.csv' is in the directory.")
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
        sim_features = list(features).copy()
        sim_features[1] = sim_bp # Update BP in feature vector (Index 1)
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
