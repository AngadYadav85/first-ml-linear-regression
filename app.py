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

# A generic accent color per brand (not a logo — just a themed color chip,
# since real brand logos are trademarked and can't be reproduced here).
BRAND_COLOR = {
    "Audi": "#B0B0B0",
    "BMW": "#1C69D4",
    "Mercedes-Benz": "#9EA6AD",
    "Mitsubishi": "#D2001C",
    "Renault": "#FFC800",
    "Toyota": "#EB0A1E",
    "Volkswagen": "#00437A",
}

FEATURE_ORDER = ["Brand", "Body", "Mileage", "EngineV", "Engine Type", "Registration", "Year"]

# ---------- Styling ----------
st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top left, #14181f 0%, #0e1117 55%);
    }
    .car-card {
        background: linear-gradient(135deg, #1b2029 0%, #161a22 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 22px 26px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.35);
        margin-bottom: 18px;
    }
    .price-card {
        background: linear-gradient(135deg, #123524 0%, #0f2b1d 100%);
        border: 1px solid rgba(46, 204, 113, 0.35);
        border-radius: 18px;
        padding: 22px 26px;
        text-align: center;
    }
    .price-value {
        font-size: 2.4rem;
        font-weight: 800;
        color: #4ade80;
        margin: 4px 0 0 0;
    }
    .price-label {
        color: #9ca3af;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def car_svg(body_type: str, color: str, size: int = 120) -> str:
    """Return a small generic car silhouette SVG, shaped roughly by body type
    and tinted with the given accent color. No brand logos are used."""

    # body-shape tweaks: (roofline height, body length, wheel offset)
    shape = {
        "sedan": (28, 190, 30),
        "crossover": (34, 190, 30),
        "van": (42, 200, 26),
        "vagon": (30, 200, 30),
        "hatch": (30, 170, 26),
        "other": (30, 185, 28),
    }.get(body_type, (30, 185, 28))

    roof_h, length, wheel_off = shape

    return f"""
    <svg width="{size}" height="{size * 0.6:.0f}" viewBox="0 0 240 130" xmlns="http://www.w3.org/2000/svg">
        <ellipse cx="120" cy="112" rx="95" ry="8" fill="black" opacity="0.25"/>
        <rect x="{(240-length)/2}" y="70" width="{length}" height="30" rx="14" fill="{color}"/>
        <path d="M {(240-length)/2+30} 70
                 Q {120} {70-roof_h} {(240-length)/2+length-30} 70
                 Z" fill="{color}"/>
        <path d="M {(240-length)/2+45} 68
                 Q {120} {70-roof_h+8} {(240-length)/2+length-45} 68
                 L {(240-length)/2+length-55} 70
                 L {(240-length)/2+55} 70 Z" fill="#cfefff" opacity="0.85"/>
        <circle cx="{(240-length)/2 + wheel_off}" cy="100" r="16" fill="#111318"/>
        <circle cx="{(240-length)/2 + wheel_off}" cy="100" r="7" fill="#666"/>
        <circle cx="{(240-length)/2 + length - wheel_off}" cy="100" r="16" fill="#111318"/>
        <circle cx="{(240-length)/2 + length - wheel_off}" cy="100" r="7" fill="#666"/>
    </svg>
    """


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

st.markdown('<div class="car-card">', unsafe_allow_html=True)

form_col, preview_col = st.columns([2.3, 1])

with form_col:
    brand = st.selectbox("Brand", list(BRAND_MAP.keys()))
    body = st.selectbox("Body type", list(BODY_MAP.keys()))
    c1, c2 = st.columns(2)
    with c1:
        engine_type = st.selectbox("Engine type", list(ENGINE_TYPE_MAP.keys()))
        mileage = st.number_input("Mileage (thousand km)", min_value=0, max_value=1000, value=150, step=1)
        year = st.number_input("Year", min_value=1960, max_value=2026, value=2012, step=1)
    with c2:
        registration = st.selectbox("Registered?", list(REGISTRATION_MAP.keys()))
        engine_v = st.number_input("Engine volume (litres)", min_value=0.5, max_value=10.0, value=2.0, step=0.1)

with preview_col:
    st.markdown(
        f"""
        <div style="text-align:center; padding-top: 10px;">
            {car_svg(body, BRAND_COLOR.get(brand, "#4ade80"), size=140)}
            <div style="color:#9ca3af; font-size:0.85rem; margin-top:6px;">{brand} · {body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

submitted = st.button("Predict price", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

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

    st.markdown(
        f"""
        <div class="price-card">
            <div class="price-label">Estimated Price</div>
            <div class="price-value">${price_pred:,.0f}</div>
            <div style="color:#6b7280; font-size:0.8rem; margin-top:6px;">
                log-price prediction: {log_price_pred:.3f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.caption(
    "Note: engine volumes above 10L and missing Price/EngineV rows were dropped "
    "during training, so predictions for extreme inputs may be less reliable. "
    "Car icon color/shape is a generic visual cue, not the manufacturer's logo."
)
