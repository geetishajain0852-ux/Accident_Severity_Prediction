"""
app.py - Streamlit deployment for the Road Traffic Accident severity model
(Random Forest, class_weight='balanced', no scaling needed).

Files required in the SAME folder as this script:
    accident_model.pkl   - the trained rf_best_model
    features.pkl          - X.columns.tolist() saved after get_dummies
    label_encoder.joblib   - the LabelEncoder fit on Accident_severity
    categories.joblib      - dict of {column: [valid raw values]} for the form

Run locally:
    streamlit run app.py
"""

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Accident Severity Predictor", page_icon="🚦", layout="centered")

st.markdown("""
<style>
    .main { padding-top: 1rem; }
    .stButton>button {
        width: 100%; background-color: #1f6feb; color: white;
        border-radius: 8px; padding: 0.6rem; font-weight: 600; border: none;
    }
    .stButton>button:hover { background-color: #1558c0; }
    div[data-testid="stForm"] {
        border: 1px solid #2d333b; border-radius: 12px; padding: 1.5rem;
        background-color: rgba(31, 111, 235, 0.03);
    }
    .result-card {
        padding: 1.2rem; border-radius: 12px; text-align: center;
        margin-top: 1rem; font-size: 1.3rem; font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    model = joblib.load("accident_model.pkl")
    label_encoder = joblib.load("label_encoder.joblib")
    feature_columns = joblib.load("features.pkl")
    categories = joblib.load("categories.joblib")
    return model, label_encoder, feature_columns, categories


try:
    model, label_encoder, feature_columns, categories = load_artifacts()
except FileNotFoundError as e:
    st.error(
        f"Missing file: {e.filename}. Make sure accident_model.pkl, features.pkl, "
        "label_encoder.joblib and categories.joblib are all in this same folder."
    )
    st.stop()

CATEGORICAL_FEATURES = list(categories.keys())  # Time, Day_of_week, Age_band_of_driver, ...

SEVERITY_ICON = {
    "Slight Injury": "🟢",
    "Serious Injury": "🟠",
    "Fatal injury": "🔴",
}

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🚦 Road Traffic Accident Severity Predictor")
st.write(
    "A machine learning model that estimates accident severity — **Slight**, "
    "**Serious**, or **Fatal** — from road and environmental conditions."
)

with st.expander("ℹ️ About this project"):
    st.markdown("""
    - **Model:** Random Forest Classifier (`class_weight='balanced'`, `max_depth=10`)
    - **Dataset:** Road Traffic Accident records (12,316 rows, 32 raw features)
    - **Pipeline:** categorical encoding (one-hot), time-of-day binning, class-imbalance handling
    - **Goal:** flag high-risk conditions, with extra focus on correctly catching
      rare but critical *Serious* and *Fatal* cases rather than optimizing for
      raw accuracy alone
    """)

st.divider()

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        time_val = st.selectbox("Time of accident", categories["Time"])
        day_of_week = st.selectbox("Day of week", categories["Day_of_week"])
        age_band_driver = st.selectbox("Driver age band", categories["Age_band_of_driver"])
        vehicle_type = st.selectbox("Type of vehicle", categories["Type_of_vehicle"])

    with col2:
        area = st.selectbox("Area accident occurred", categories["Area_accident_occured"])
        junction = st.selectbox("Type of junction", categories["Types_of_Junction"])
        light = st.selectbox("Light conditions", categories["Light_conditions"])
        weather = st.selectbox("Weather conditions", categories["Weather_conditions"])
        age_band_casualty = st.selectbox("Casualty age band", categories["Age_band_of_casualty"])

    num_vehicles = st.number_input("Number of vehicles involved", min_value=1, max_value=10, value=2)
    num_casualties = st.number_input("Number of casualties", min_value=1, max_value=20, value=1)

    submitted = st.form_submit_button("Predict severity")

if submitted:
    raw_input = pd.DataFrame([{
        "Time": time_val,
        "Day_of_week": day_of_week,
        "Age_band_of_driver": age_band_driver,
        "Type_of_vehicle": vehicle_type,
        "Area_accident_occured": area,
        "Types_of_Junction": junction,
        "Light_conditions": light,
        "Weather_conditions": weather,
        "Age_band_of_casualty": age_band_casualty,
        "Number_of_vehicles_involved": num_vehicles,
        "Number_of_casualties": num_casualties,
    }])

    # Same one-hot encoding as training, then align to the saved column order.
    # drop_first doesn't matter here since we reindex to the exact saved columns anyway.
    encoded = pd.get_dummies(raw_input, columns=CATEGORICAL_FEATURES)
    encoded = encoded.reindex(columns=feature_columns, fill_value=0)

    pred_class = model.predict(encoded)[0]
    pred_label = label_encoder.inverse_transform([pred_class])[0]

    st.subheader("Prediction")
    icon = SEVERITY_ICON.get(pred_label, "⚪")
    card_color = {"Slight Injury": "#1a4d2e", "Serious Injury": "#5c3d0e", "Fatal injury": "#5c1a1a"}
    bg = card_color.get(pred_label, "#333")
    st.markdown(
        f'<div class="result-card" style="background-color:{bg}; color:white;">'
        f'{icon} {pred_label}</div>',
        unsafe_allow_html=True,
    )

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(encoded)[0]
        proba_df = pd.DataFrame({
            "Severity": label_encoder.classes_,
            "Probability": proba,
        }).sort_values("Probability", ascending=False)
        st.bar_chart(proba_df.set_index("Severity"))

st.divider()
st.caption(
    "Built as a college internship project · Model: Random Forest (class-balanced, "
    "max_depth=10) · A data-driven estimate, not a substitute for professional assessment."
)
