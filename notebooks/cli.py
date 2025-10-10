import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("random_forest_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title(" Prediction du risque de diabete (Modele Random Forest)")

st.write("Ce modele predit le **risque de diabete** a partir de vos caracteristiques medicales.")

glucose = st.number_input("Glucose")
bmi = st.number_input("BMI (Indice de Masse Corporelle)")
dpf = st.number_input("Diabetes Pedigree Function")
age = st.number_input("Âge")

blood_pressure = st.number_input("Pression arterielle")
insulin = st.number_input("Insuline")

skin_thickness = st.number_input("epaisseur de la peau")
pregnancies = st.number_input("Nombre de grossesses")

if st.button(" Predire le risque"):

    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    
    input_scaled = scaler.transform(input_data)
    
    prediction = model.predict(input_scaled)[0]
    
    if prediction == 1:
        st.error(" Risque eLEVe de diabete")
    else:
        st.success(" Risque FAIBLE de diabete")
