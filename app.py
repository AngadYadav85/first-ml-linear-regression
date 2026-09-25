import numpy as np
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")

MODEL_PATH = "linear_regression_model.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# Label-encoding maps, matching the LabelEncoder fits from the training
# notebook (sklearn's LabelEncoder assigns codes in alphabetical order).
BRAND_MAP = {
    "Audi": 0,
    "BMW": 1,
    "Mercedes-Benz": 2,
    "Mitsubishi": 3,
    "Renault": 4,
    "Toyota": 5,
    "Volkswagen": 6,
}
BODY_MAP = {
    "crossover": 0,
    "hatch": 1,
    "other": 2,
    "sedan": 3,
    "vagon": 4,
    "van": 5,
}
ENGINE_TYPE_MAP = {
    "Diesel": 0,
    "Gas": 1,
    "Other": 2,
    "Petrol": 3,
}
REGISTRATION_MAP = {"no": 0, "yes": 1}

# Column order must match X_train used during training:
# Brand, Body, Mileage, EngineV, Engine Type, Registration, Year
FEATURE_ORDER = ["Brand", "Body", "Mileage", "EngineV", "Engine Type", "Registration", "Year"]

st.title("🚗 Used Car Price Predictor")
st.caption("Linear Regression model trained on used-car sales data (log-price target).")

try:
    model = load_model()
except FileNotFoundError:
    st.error(
        f"Couldn't find `{MODEL_PATH}`. Make sure the trained model file from the "
        "notebook (Step 16 — Save the Model) is committed to this repo, in the same "
        "folder as `app.py`."
    )
    st.stop()

with st.form("car_form"):
    col1, col2 = st.columns(2)

    with col1:
        brand = st.selectbox("Brand", list(BRAND_MAP.keys()))
        body = st.selectbox("Body type", list(BODY_MAP.keys()))
        engine_type = st.selectbox("Engine type", list(ENGINE_TYPE_MAP.keys()))
        registration = st.selectbox("Registered?", list(REGISTRATION_MAP.keys()))

    with col2:
        mileage = st.number_input("Mileage (thousand km)", min_value=0, max_value=1000, value=150, step=1)
        engine_v = st.number_input("Engine volume (litres)", min_value=0.5, max_value=10.0, value=2.0, step=0.1)
        year = st.number_input("Year", min_value=1960, max_value=2026, value=2012, step=1)

    submitted = st.form_submit_button("Predict price")

if submitted:
    row = {
        "Brand": BRAND_MAP[brand],
        "Body": BODY_MAP[body],
        "Mileage": mileage,
        "EngineV": engine_v,
        "Engine Type": ENGINE_TYPE_MAP[engine_type],
        "Registration": REGISTRATION_MAP[registration],
        "Year": year,
    }
    features = pd.DataFrame([row])[FEATURE_ORDER]

    log_price_pred = model.predict(features)[0]
    price_pred = float(np.exp(log_price_pred))

    st.success(f"Estimated price: **${price_pred:,.0f}**")
    st.caption(f"(model predicted log-price: {log_price_pred:.3f})")

st.divider()
st.caption(
    "Note: engine volumes above 10L and missing Price/EngineV rows were dropped "
    "during training, so predictions for extreme inputs may be less reliable."
)
