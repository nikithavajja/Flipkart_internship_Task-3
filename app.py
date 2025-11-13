import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.pkl")

st.title("Health Risk Prediction System")

st.header("Enter Patient Details")

age = st.number_input("Age", min_value=1, max_value=120)
sex = st.selectbox("Sex (1=Male, 0=Female)", [1, 0])
cp = st.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3)
trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200)
chol = st.number_input("Cholesterol", min_value=100, max_value=600)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)", [1, 0])
restecg = st.number_input("Resting ECG Results (0-2)", min_value=0, max_value=2)
thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220)
exang = st.selectbox("Exercise Induced Angina (1=Yes, 0=No)", [1, 0])
oldpeak = st.number_input("ST Depression Induced by Exercise")
slope = st.number_input("Slope (0-2)", min_value=0, max_value=2)
ca = st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3)
thal = st.number_input("Thal (1=Normal, 2=Fixed Defect, 3=Reversible Defect)", min_value=1, max_value=3)

if st.button("Predict"):
    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                                exang, oldpeak, slope, ca, thal]],
                              columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                                       'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("High Risk of Heart Disease")
    else:
        st.info("Low Risk of Heart Disease")
