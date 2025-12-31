import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. Load Resources ---
# We use caching so it doesn't reload the model on every interaction
@st.cache_resource
def load_resources():
    try:
        model = joblib.load('model.pkl')
        model_cols = joblib.load('model_columns.pkl')
        return model, model_cols
    except Exception as e:
        return None, None

model, model_columns = load_resources()

# --- 2. Helper Functions ---
def process_age(age_str):
    age_map = {
        '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
        '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
        '[80-90)': 85, '[90-100)': 95
    }
    return age_map.get(age_str, 55)

def map_diagnosis(code):
    s_code = str(code)
    if s_code.startswith('V') or s_code.startswith('E') or code == '?': return 'Other'
    try: n = float(code)
    except: return 'Other'
    if 390 <= n <= 459 or n == 785: return 'Circulatory'
    if 460 <= n <= 519 or n == 786: return 'Respiratory'
    if 520 <= n <= 579 or n == 787: return 'Digestive'
    if 250 <= n < 251: return 'Diabetes'
    if 800 <= n <= 999: return 'Injury'
    if 710 <= n <= 739: return 'Musculoskeletal'
    if 580 <= n <= 629 or n == 788: return 'Genitourinary'
    if 140 <= n <= 239: return 'Neoplasms'
    return 'Other'

# --- 3. UI Layout ---
st.set_page_config(page_title="Hospital Readmission AI", page_icon="🏥")
st.title("🏥 Hospital Readmission Predictor")
st.markdown("Enter patient details below to estimate the risk of readmission within 30 days.")

# Inputs
col1, col2 = st.columns(2)

with col1:
    age_input = st.selectbox("Age Group", ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'], index=7)
    gender_input = st.selectbox("Gender", ["Female", "Male"])
    race_input = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other"])

with col2:
    time_in_hospital = st.slider("Time in Hospital (Days)", 1, 14, 3)
    num_procedures = st.slider("Number of Procedures", 0, 6, 0)
    num_medications = st.slider("Number of Medications", 1, 81, 12)

st.subheader("Clinical Data")
c1, c2, c3 = st.columns(3)
diag_1 = c1.text_input("Primary Diagnosis (ICD-9)", "428")
diag_2 = c2.text_input("Secondary Diagnosis (ICD-9)", "250.01")
diag_3 = c3.text_input("Tertiary Diagnosis (ICD-9)", "401")

change_input = st.selectbox("Medication Change?", ["No", "Ch"])
diabetes_med_input = st.selectbox("Diabetes Meds Prescribed?", ["No", "Yes"])
med_specialty = st.selectbox("Admitting Specialty", ["InternalMedicine", "Emergency/Trauma", "Family/GeneralPractice", "Cardiology", "Surgery-General", "Other"])

# --- 4. Prediction Logic ---
if st.button("Analyze Risk"):
    if model is None:
        st.error("Model not loaded. Ensure 'model.pkl' and 'model_columns.pkl' are uploaded.")
    else:
        # A. Create Raw DataFrame
        input_data = {
            'race': [race_input], 'gender': [gender_input], 'age': [age_input],
            'time_in_hospital': [time_in_hospital], 'num_procedures': [num_procedures],
            'num_medications': [num_medications], 'number_outpatient': [0],
            'number_emergency': [0], 'number_inpatient': [0],
            'diag_1': [diag_1], 'diag_2': [diag_2], 'diag_3': [diag_3],
            'diag_4': ['?'], 'diag_5': ['?'],
            'medical_specialty': [med_specialty], 'change': [change_input],
            'diabetesMed': [diabetes_med_input]
        }

        # Add dummy X columns
        for i in range(3, 26):
            if i not in [1, 2]: input_data[f'X{i}'] = ['No']

        df = pd.DataFrame(input_data)

        # B. Apply Feature Engineering
        df['age_num'] = df['age'].apply(process_age)
        df.drop('age', axis=1, inplace=True)

        df['total_visits'] = df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']

        diag_cols = ['diag_1', 'diag_2', 'diag_3', 'diag_4', 'diag_5']
        for col in diag_cols:
            df[f'{col}_cat'] = df[col].apply(map_diagnosis)

        cat_cols_exist = [f'{c}_cat' for c in diag_cols]
        df['has_diabetes_diag'] = df[cat_cols_exist].apply(lambda row: 1 if 'Diabetes' in row.values else 0, axis=1)

        med_cols = [f'X{i}' for i in range(3, 26)]
        df['num_med_changes'] = 0
        df['num_meds_active'] = 0

        df['severity_score'] = df['time_in_hospital'] * df['num_procedures']

        top_10 = ['InternalMedicine', 'Emergency/Trauma', 'Family/GeneralPractice', 'Cardiology', 'Surgery-General', 'Nephrology', 'Orthopedics', 'Orthopedics-Reconstructive', 'Radiologist']
        df['med_spec_grouped'] = df['medical_specialty'].apply(lambda x: x if x in top_10 else 'Other')
        df.drop('medical_specialty', axis=1, inplace=True)

        # Drop raw cols
        df.drop(diag_cols + med_cols, axis=1, inplace=True)

        # C. Align with Model
        df_encoded = pd.get_dummies(df)
        df_final = df_encoded.reindex(columns=model_columns, fill_value=0)

        # D. Predict
        prob = model.predict_proba(df_final)[0][1]
        threshold = 0.35
        pred = 1 if prob >= threshold else 0

        st.divider()
        if pred == 1:
            st.error(f"⚠️ **High Risk of Readmission** (Probability: {prob:.1%})")
        else:
            st.success(f"✅ **Low Risk** (Probability: {prob:.1%})")
