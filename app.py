import streamlit as st
import pandas as pd
import joblib

# Load the model

model=joblib.load("diabetes_rf_model.pkl")

st.title("🩺 Diabetic app")
st.write("Please enter the values to check wether the patient has diabete or not")

pregnancies = st.slider("Number of Pregnancies", 0, 20, 0)
glucose = st.slider("Glucose Level", 0, 200, 100)
blood_pressure = st.slider("Blood Pressure", 0, 150, 70)
skin_thickness = st.slider("Skin Thickness", 0, 100, 20)
insulin = st.slider("Insulin Level", 0, 900, 80)
bmi = st.slider("Body Mass Index (BMI)", 0.0, 70.0, 25.0)
dpf = st.slider("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.slider("Age", 0, 120, 30)

# Preparation of the input data
input_data = pd.DataFrame({
    'Pregnancies': [pregnancies],
    'Glucose': [glucose],
    'BloodPressure': [blood_pressure],
    'SkinThickness': [skin_thickness],
    'Insulin': [insulin],
    'BMI': [bmi],
    'DiabetesPedigreeFunction': [dpf],
    'Age': [age]
})

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("⚠️ The patient probably has diabetic")
    else:
        st.success("✅ The patient probably doesn't has diabetic")
    # Display feature importance
    st.subheader("🔍 Feature Importance")

    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    st.dataframe(importance_df)
    st.bar_chart(importance_df.set_index('Feature'))