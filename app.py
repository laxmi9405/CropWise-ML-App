import pickle
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# App configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="CropWise | Crop Prediction",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "crop_yield_model(1).pkl"
DATA_PATH = BASE_DIR / "app_data(1).pkl"

# ---------------------------------------------------------
# Custom styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .stApp {
        background:
            radial-gradient(circle at 90% 2%, rgba(91, 180, 125, .13), transparent 26%),
            linear-gradient(135deg, #f6faf7 0%, #eef5ef 100%);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #123c2b 0%, #1c563b 100%);
    }
    [data-testid="stSidebar"] * { color: #f3fff6 !important; }
    .hero {
        padding: 28px 32px;
        border-radius: 22px;
        color: white;
        background: linear-gradient(115deg, #123c2b 0%, #23764b 58%, #4b9b61 100%);
        box-shadow: 0 12px 32px rgba(20, 76, 48, .16);
        margin-bottom: 24px;
    }
    .hero h1 {
        font-family: 'Manrope', sans-serif;
        font-size: 2.15rem;
        font-weight: 800;
        margin: 0 0 8px 0;
        color: white;
    }
    .hero p { color: #e1f4e6; margin: 0; font-size: 1rem; }
    .section-title {
        color: #173e2c;
        font-family: 'Manrope', sans-serif;
        font-size: 1.18rem;
        font-weight: 800;
        margin: 12px 0 4px 0;
    }
    .hint { color: #63776a; font-size: .88rem; margin-bottom: 14px; }
    .result-card {
        padding: 24px;
        border-radius: 18px;
        background: white;
        border: 1px solid #dcebe0;
        box-shadow: 0 8px 24px rgba(30, 75, 48, .07);
    }
    .result-label {
        color: #668071;
        text-transform: uppercase;
        letter-spacing: .1em;
        font-size: .75rem;
        font-weight: 700;
    }
    .result-value {
        color: #174b31;
        font-family: 'Manrope', sans-serif;
        font-size: 2.15rem;
        font-weight: 800;
        margin: 5px 0;
    }
    .small-note { color: #687b70; font-size: .84rem; }
    div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(90deg, #1d6b43, #318b55);
        color: white;
        border: 0;
        border-radius: 12px;
        padding: .65rem 1.2rem;
        font-weight: 700;
        min-height: 46px;
    }
    div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
        background: #155534;
        color: white;
        border: 0;
    }
    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #dfebe2;
        border-radius: 14px;
        padding: 14px 16px;
    }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Load the saved artifacts
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}. "
            "Place it in the same folder as app.py."
        )
    # Load only PKL files that you created or otherwise trust.
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_app_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"App data file not found: {DATA_PATH.name}. "
            "Place it in the same folder as app.py."
        )
    # This file contains dropdown options and numerical input statistics.
    with open(DATA_PATH, "rb") as file:
        return pickle.load(file)


try:
    model = load_model()
    app_data = load_app_data()
except Exception as error:
    st.error(f"Could not load the model files: {error}")
    st.info(
        "Keep app.py, crop_yield_model(1).pkl and app_data(1).pkl "
        "together in the same project folder."
    )
    st.stop()

# ---------------------------------------------------------
# Header and sidebar
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🌱 CropWise Prediction</h1>
        <p>Data-driven crop category prediction using soil, weather and farming inputs.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 🌿 CropWise")
    st.caption("Smart agriculture • ML prediction")
    st.markdown("---")
    page = st.radio("NAVIGATION", ["Predict", "About the model"])
    st.markdown("---")
    st.caption("Enter the conditions on the Predict page to estimate the model's crop category.")

if page == "About the model":
    st.markdown("## About this machine learning model")
    st.write(
        "This app uses your saved scikit-learn pipeline. The pipeline performs "
        "categorical encoding and numerical scaling before passing the processed "
        "features to a Random Forest Classifier."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Algorithm", "Random Forest")
    c2.metric("Categorical processing", "OneHotEncoder")
    c3.metric("Numerical processing", "StandardScaler")
    st.markdown("### Input features")
    st.write(
        "Crop, season, state, year, N, P, K, pH, average temperature, "
        "total rainfall, average humidity, fertilizer per hectare and "
        "pesticide per hectare."
    )
    st.markdown("### Output")
    st.write("The classifier returns one of the classes: High, Medium or Low.")
    st.warning(
        "This model predicts a category, not an exact numerical yield. "
        "Use predictions as decision support, not as a guarantee of actual farm results."
    )

else:
    st.markdown('<div class="section-title">Farm and crop details</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hint">Choose the crop, location and growing season, then enter the soil and weather measurements.</div>',
        unsafe_allow_html=True,
    )

    crop_options = app_data["crop_options"]
    state_options = app_data["state_options"]
    season_options = app_data["season_options"]
    stats = app_data["numerical_stats"]

    def bounds(feature, default_min, default_max, default_value):
        feature_stats = stats.get(feature, {})
        minimum = float(feature_stats.get("min", default_min))
        maximum = float(feature_stats.get("max", default_max))
        value = float(feature_stats.get("median", default_value))
        # Streamlit number_input needs a valid interval.
        if minimum >= maximum:
            minimum, maximum = default_min, default_max
        value = min(max(value, minimum), maximum)
        return minimum, maximum, value

    with st.form("crop_prediction_form"):
        st.markdown("#### 🌾 Crop information")
        a, b, c = st.columns(3)
        with a:
            crop = st.selectbox("Crop", crop_options, index=0)
        with b:
            state = st.selectbox(
                "State",
                state_options,
                index=state_options.index("Telangana") if "Telangana" in state_options else 0,
            )
        with c:
            season = st.selectbox("Season", season_options, index=0)

        st.markdown("#### 🌦️ Climate conditions")
        a, b, c = st.columns(3)
        with a:
            lo, hi, val = bounds("year", 1997, 2020, 2010)
            year = st.number_input("Year", min_value=int(lo), max_value=int(hi), value=int(val), step=1)
        with b:
            lo, hi, val = bounds("avg_temp_c", 0, 50, 25)
            avg_temp_c = st.number_input("Average temperature (°C)", min_value=lo, max_value=hi, value=val, step=0.1)
        with c:
            lo, hi, val = bounds("total_rainfall_mm", 0, 10000, 1000)
            total_rainfall_mm = st.number_input("Total rainfall (mm)", min_value=lo, max_value=hi, value=val, step=10.0)

        lo, hi, val = bounds("avg_humidity_percent", 0, 100, 68)
        avg_humidity_percent = st.slider(
            "Average humidity (%)", min_value=float(lo), max_value=float(hi),
            value=float(val), step=0.1
        )

        st.markdown("#### 🧪 Soil and farm inputs")
        a, b, c = st.columns(3)
        with a:
            lo, hi, val = bounds("N", 0, 300, 72)
            N = st.number_input("Nitrogen (N)", min_value=lo, max_value=hi, value=val, step=1.0)
        with b:
            lo, hi, val = bounds("P", 0, 200, 38)
            P = st.number_input("Phosphorus (P)", min_value=lo, max_value=hi, value=val, step=1.0)
        with c:
            lo, hi, val = bounds("K", 0, 200, 33)
            K = st.number_input("Potassium (K)", min_value=lo, max_value=hi, value=val, step=1.0)

        a, b, c, d = st.columns(4)
        with a:
            lo, hi, val = bounds("pH", 0, 14, 6.6)
            pH = st.number_input("Soil pH", min_value=lo, max_value=hi, value=val, step=0.1)
        with b:
            lo, hi, val = bounds("fertilizer_per_ha", 0, 1000, 144)
            fertilizer_per_ha = st.number_input("Fertilizer per ha", min_value=lo, max_value=hi, value=val, step=1.0)
        with c:
            lo, hi, val = bounds("pesticide_per_ha", 0, 10, 0.27)
            pesticide_per_ha = st.number_input("Pesticide per ha", min_value=lo, max_value=hi, value=val, step=0.01, format="%.2f")
        with d:
            st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
            st.caption("Use the same measurement units as your training dataset.")

        submitted = st.form_submit_button("🌱 Predict crop category", use_container_width=True)

    if submitted:
        # Keep the exact feature names and column order used during model training.
        input_df = pd.DataFrame([{
            "crop": crop,
            "season": season,
            "state": state,
            "year": year,
            "N": N,
            "P": P,
            "K": K,
            "pH": pH,
            "avg_temp_c": avg_temp_c,
            "total_rainfall_mm": total_rainfall_mm,
            "avg_humidity_percent": avg_humidity_percent,
            "fertilizer_per_ha": fertilizer_per_ha,
            "pesticide_per_ha": pesticide_per_ha,
        }])

        try:
            prediction = model.predict(input_df)[0]
            probabilities = None
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_df)[0]
                class_names = model.classes_

            st.markdown("---")
            st.markdown("## Prediction result")

            result_col, detail_col = st.columns([1, 1.35])
            with result_col:
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-label">Predicted category</div>
                        <div class="result-value">{str(prediction)}</div>
                        <div class="small-note">For the selected crop and entered conditions</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with detail_col:
                st.markdown("#### Input summary")
                st.write(f"**Crop:** {crop}  ·  **State:** {state}  ·  **Season:** {season}")
                st.write(
                    f"**Year:** {year}  ·  **Temperature:** {avg_temp_c:.1f} °C  ·  "
                    f"**Rainfall:** {total_rainfall_mm:,.1f} mm"
                )
                st.write(
                    f"**N / P / K:** {N:g} / {P:g} / {K:g}  ·  "
                    f"**Soil pH:** {pH:.1f}"
                )

            if probabilities is not None:
                st.markdown("### Class probabilities")
                probability_df = pd.DataFrame({
                    "Category": class_names,
                    "Probability": probabilities,
                }).sort_values("Probability", ascending=False)
                for _, row in probability_df.iterrows():
                    st.write(f"**{row['Category']}** — {row['Probability']:.1%}")
                    st.progress(float(row["Probability"]))

            st.caption(
                "Probabilities are model estimates, not guarantees. Results depend on "
                "the quality and range of the training data."
            )
        except Exception as error:
            st.error(f"Prediction failed: {error}")
            st.write("Check that the input feature names and data types match the model used during training.")
