import streamlit as st
import numpy as np
import joblib

# Load models
price_model = joblib.load("price_model.pkl")
mileage_model = joblib.load("mileage_model.pkl")

st.title("Car Price & Mileage Prediction")

# ---------------- INPUT FIELDS ---------------- #

fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])

engine = st.slider("Engine CC", 800, 3000, 1200)

kms = st.number_input("KMs Driven", min_value=0, value=20000)

owner = st.selectbox("Ownership", ["First Owner", "Second Owner", "Third Owner"])

transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

# ---------------- ENCODING ---------------- #

fuel_map = {"Petrol": 0, "Diesel": 1, "CNG": 2}
owner_map = {"First Owner": 0, "Second Owner": 1, "Third Owner": 2}
trans_map = {"Manual": 0, "Automatic": 1}

# ---------------- PREDICTION ---------------- #

if st.button("Predict"):

    input_data = np.array([[
        fuel_map[fuel],
        engine,
        kms,
        owner_map[owner],
        trans_map[transmission]
    ]])

    # Price Prediction
    price = price_model.predict(input_data)

    # Mileage Prediction
    mileage = mileage_model.predict(input_data)

    # Output
    st.success(f"Predicted Price: ₹ {int(price[0])}")
    st.success(f"Predicted Mileage: {mileage[0]:.2f} km/l")