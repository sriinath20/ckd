import argparse
import logging
import os
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(data_path):
    """Load and validate input data."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded dataset with shape: {df.shape}")
    return df

def clean_categorical(df, col_name, mapping):
    """Clean specific categorical columns with whitespace/tab issues."""
    if col_name in df.columns:
        df[col_name] = df[col_name].astype(str).str.strip().str.lower()
        df[col_name] = df[col_name].replace(mapping)
        # Convert to numeric, errors become NaN
        df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
    return df

def preprocess_data(df):
    """Apply robust preprocessing and create the NEW target variable."""
    df = df.copy()

    # Drop ID if present
    if 'id' in df.columns:
        df = df.drop('id', axis=1)

    # 1. FIX TARGET VARIABLE
    # Clean 'classification' column first
    df['classification'] = df['classification'].astype(str).str.strip().str.lower()
    df['classification'] = df['classification'].replace({'ckd\t': 'ckd'})
    # Map to 1 (CKD) and 0 (Not CKD)
    df['classification_numeric'] = df['classification'].map({'ckd': 1, 'notckd': 0, 'no': 0})
    
    # Ensure 'sc' (Creatinine) is numeric for our logic
    df['sc'] = pd.to_numeric(df['sc'], errors='coerce')
    
    # --- CREATE "SEVERE RISK" TARGET ---
    # Logic: If patient has CKD AND Creatinine > 2.0, they are "High Risk" (Advanced Stage).
    # If they have CKD but Creatinine <= 2.0, they are "Low Risk" (Manageable).
    # If they don't have CKD, they are "Low Risk".
    # This trains the model to find the "Bad" cases based on systemic factors.
    df['severe_risk'] = ((df['classification_numeric'] == 1) & (df['sc'] > 2.0)).astype(int)
    
    logger.info(f"Target Distribution (Severe Risk vs Manageable/Normal):\n{df['severe_risk'].value_counts()}")

    # 2. CLEAN FEATURES
    # Numerical Columns to keep
    num_cols = ['age', 'bp', 'bgr', 'bu', 'hemo', 'pot', 'wc', 'rc', 'al', 'su']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Categorical Columns cleaning
    # HTN (Hypertension): yes/no
    df = clean_categorical(df, 'htn', {'yes': 1, 'no': 0})
    # DM (Diabetes): yes/no
    df = clean_categorical(df, 'dm', {'yes': 1, 'no': 0, '\tyes': 1, ' yes': 1})
    # CAD (Coronary Artery Disease): yes/no
    df = clean_categorical(df, 'cad', {'yes': 1, 'no': 0, '\tno': 0})
    # PE (Pedal Edema): yes/no
    df = clean_categorical(df, 'pe', {'yes': 1, 'no': 0})
    # ANE (Anemia): yes/no
    df = clean_categorical(df, 'ane', {'yes': 1, 'no': 0})
    
    # Other nominals
    df = clean_categorical(df, 'rbc', {'abnormal': 1, 'normal': 0})
    df = clean_categorical(df, 'pc', {'abnormal': 1, 'normal': 0})
    df = clean_categorical(df, 'pcc', {'present': 1, 'notpresent': 0})
    df = clean_categorical(df, 'ba', {'present': 1, 'notpresent': 0})

    return df

def get_training_features():
    """
    Return list of features for the ML model.
    CRITICAL: We EXCLUDE 'sc' (Creatinine) so the model relies on systemic factors.
    """
    return [
        'age', 'bp', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba',
        'bgr', 'bu', 'pot', 'wc', 'htn', 'dm', 'cad', 'pe', 'ane', 'hemo'
    ]

def train_model(X, y):
    """Train RandomForest with Imputation pipeline."""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Impute Missing Values (Median for numerical, Mode for categorical would be ideal, 
    # but strictly numerical imputation works since we encoded everything to numbers above)
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Train Model
    logger.info("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train_imputed, y_train)

    # Evaluate
    y_pred = model.predict(X_test_imputed)
    acc = accuracy_score(y_test, y_pred)
    
    return model, imputer, acc, y_test, y_pred

def main():
    parser = argparse.ArgumentParser(description="Train Kidney Severity Prediction Model")
    parser.add_argument('--data-path', type=str, default='kidney_disease.csv', help='Path to CSV file')
    parser.add_argument('--output-model', type=str, default='kidney_risk_model.pkl', help='Output pickle file')
    args = parser.parse_args()

    # 1. Load
    logger.info("Loading data...")
    try:
        df = load_data(args.data_path)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return

    # 2. Preprocess
    logger.info("Preprocessing and creating 'Severe Risk' target...")
    df_processed = preprocess_data(df)

    # 3. Select Features
    feature_cols = get_training_features()
    
    # Check for missing columns
    missing_cols = [c for c in feature_cols if c not in df_processed.columns]
    if missing_cols:
        logger.error(f"Missing columns in dataset: {missing_cols}")
        return

    X = df_processed[feature_cols]
    y = df_processed['severe_risk']

    # 4. Train
    logger.info(f"Training on {len(feature_cols)} features (Creatinine Excluded).")
    model, imputer, accuracy, y_test, y_pred = train_model(X, y)

    logger.info(f"Model Training Complete.")
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info("\nClassification Report:\n" + classification_report(y_test, y_pred))

    # 5. Save Model AND Imputer (needed for new data)
    output_data = {
        "model": model,
        "imputer": imputer,
        "features": feature_cols
    }
    
    with open(args.output_model, 'wb') as f:
        pickle.dump(output_data, f)
    
    logger.info(f"Model and artifacts saved to {args.output_model}")

if __name__ == "__main__":
    main()
