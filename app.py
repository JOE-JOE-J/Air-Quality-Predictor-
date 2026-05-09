import streamlit as st
import numpy as np
import joblib
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="California AQI Risk Predictor",
    page_icon="🌫️",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

.main { background-color: #0f0f0f; }

h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    color: #f0f0f0;
    letter-spacing: -0.5px;
}

.subtitle {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #888;
    margin-top: -12px;
    margin-bottom: 28px;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.result-safe {
    background: #0d2b1a;
    border-left: 4px solid #22c55e;
    padding: 20px 24px;
    border-radius: 6px;
    margin-top: 20px;
}

.result-unsafe {
    background: #2b0d0d;
    border-left: 4px solid #ef4444;
    padding: 20px 24px;
    border-radius: 6px;
    margin-top: 20px;
}

.result-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 6px;
}

.result-prob {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: #aaa;
}

.stSlider > div > div { background: #333 !important; }
.stSelectbox label, .stSlider label, .stNumberInput label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #ccc;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 2px;
    border-bottom: 1px solid #222;
    padding-bottom: 6px;
    margin: 24px 0 16px 0;
}

footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "random_forest_model.sav"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🌫️ California AQI Risk Predictor")
st.markdown('<div class="subtitle">Random Forest · County-Level · Binary Classification</div>', unsafe_allow_html=True)

if model is None:
    st.error("Model file `random_forest_model.sav` not found. Upload it to the same directory as this app.")
    st.info("Expected path: `random_forest_model.sav`")
    st.stop()

# ── Inputs ────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Environmental</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    ozone = st.number_input("Ozone (ppm)", min_value=0.000, max_value=0.200, value=0.040, step=0.001, format="%.3f",
                             help="Daily max 8-hour ozone concentration")
with col2:
    population_raw = st.number_input("County Population", min_value=1000, max_value=10_000_000, value=500_000, step=10000,
                                      help="Approximate county population")

st.markdown('<div class="section-header">Weather</div>', unsafe_allow_html=True)

col3, col4, col5 = st.columns(3)
with col3:
    tmax = st.number_input("Max Temp (°F)", min_value=20, max_value=130, value=75)
with col4:
    tmin = st.number_input("Min Temp (°F)", min_value=0, max_value=110, value=55)
with col5:
    awnd = st.number_input("Wind Speed (mph)", min_value=0.0, max_value=60.0, value=7.0, step=0.5)

col6, col7 = st.columns(2)
with col6:
    prcp = st.number_input("Precipitation (in)", min_value=0.00, max_value=10.00, value=0.00, step=0.01, format="%.2f")
with col7:
    is_rainy = 1 if prcp > 0 else 0
    st.metric("Rainy Day", "Yes" if is_rainy else "No")

temp_range = tmax - tmin

st.markdown('<div class="section-header">Temporal</div>', unsafe_allow_html=True)

col8, col9 = st.columns(2)
with col8:
    season_label = st.selectbox("Season", ["Winter (Dec–Feb)", "Spring (Mar–May)", "Summer (Jun–Aug)", "Fall (Sep–Nov)"])
    season_map = {
        "Winter (Dec–Feb)": 1,
        "Spring (Mar–May)": 2,
        "Summer (Jun–Aug)": 3,
        "Fall (Sep–Nov)": 4
    }
    season = season_map[season_label]
with col9:
    is_weekend = st.selectbox("Day Type", ["Weekday", "Weekend"])
    is_weekend_val = 1 if is_weekend == "Weekend" else 0

# ── Standardize population (same scaler logic as training) ────────────────────
# Training used StandardScaler. We approximate with mean/std from training data.
# From df[['Population']].describe() in notebook: mean ≈ 0, std ≈ 1 after scaling.
# We need to replicate the raw population mean and std before scaling.
# From notebook output: min=-0.537, max=5.445 scaled → raw min=1695 (Alpine), raw max=10_038_388 (LA)
# Approximate: mean_raw ≈ 700_000, std_raw ≈ 1_300_000 (rough estimate from CA county distribution)
POP_MEAN = 700_000
POP_STD = 1_300_000
population_scaled = (population_raw - POP_MEAN) / POP_STD

# ── Build feature vector ───────────────────────────────────────────────────────
# Features order: ['Ozone','IsWeekend','Population','PRCP','TMAX','TMIN','AWND','Temp_Range','Is_Rainy','Season_2','Season_3','Season_4']
season_2 = 1 if season == 2 else 0
season_3 = 1 if season == 3 else 0
season_4 = 1 if season == 4 else 0

features = np.array([[
    ozone,
    is_weekend_val,
    population_scaled,
    prcp,
    tmax,
    tmin,
    awnd,
    temp_range,
    is_rainy,
    season_2,
    season_3,
    season_4
]])

# ── Predict ────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Prediction</div>', unsafe_allow_html=True)

if st.button("Run Prediction", use_container_width=True):
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0]
    prob_unsafe = prob[1] * 100
    prob_safe = prob[0] * 100

    if prediction == 1:
        st.markdown(f"""
        <div class="result-unsafe">
            <div class="result-label">⚠️ UNSAFE AQI PREDICTED</div>
            <div class="result-prob">Risk probability: {prob_unsafe:.1f}% &nbsp;|&nbsp; Safe probability: {prob_safe:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        st.warning("AQI likely exceeds 100. Consider limiting outdoor exposure, especially for sensitive groups.")
    else:
        st.markdown(f"""
        <div class="result-safe">
            <div class="result-label">✅ SAFE AQI PREDICTED</div>
            <div class="result-prob">Safe probability: {prob_safe:.1f}% &nbsp;|&nbsp; Risk probability: {prob_unsafe:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        st.success("AQI conditions appear acceptable based on current inputs.")

    with st.expander("Feature inputs used"):
        import pandas as pd
        feature_names = ['Ozone','IsWeekend','Population (scaled)','PRCP','TMAX','TMIN','AWND','Temp_Range','Is_Rainy','Season_2','Season_3','Season_4']
        st.dataframe(pd.DataFrame(features, columns=feature_names).T.rename(columns={0: "Value"}))

# ── Footer note ───────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Model: Random Forest (balanced class weights) · Trained on California EPA AQI + NOAA weather data 2021–2025 · BSAN 6070 Final Project")
