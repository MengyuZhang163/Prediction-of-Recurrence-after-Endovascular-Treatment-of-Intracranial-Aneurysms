import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import os
import xgboost as xgb

# ==========================================
# 1. Configuration & Model Loading
# ==========================================
st.set_page_config(
    page_title="Thrombosis Risk Prediction System",
    page_icon="🧠",
    layout="wide"
)

# --- Please modify the model path here ---
MODEL_PATH = 'XGB.pkl'


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found. Please check the path: {MODEL_PATH}")
        return None
    model = load(MODEL_PATH)
    return model


model = load_model()

# ==========================================
# 2. Define Mapping Dictionary (CRITICAL: Must match training encoding)
# ==========================================
# WARNING: The values 0, 1, 2 below are examples.
# You MUST check your X_train data before training to confirm the specific number for each text!

mapping_dict = {
    "ThrombolysisAfterTirofiban": {"No": 0, "Yes": 1},

    # Assumption: LVIS=0, Enterprise=1, Solitaire=2... Modify based on actual training data
    "StentType": {
        "LVIS": 0,
        "Enterprise": 1,
        "Solitaire": 2,
        "Flow Diverter": 3,
        "Other": 4
    },

    # Assumption: Saccular=0, Irregular=1, Fusiform=2...
    "Morphology": {
        "Saccular": 0,
        "Irregular": 1,
        "Fusiform": 2
    },

    "Rupture": {"Unruptured": 0, "Ruptured": 1},

    # Assumption: Simple=0, Balloon=1, Stent=2...
    "EmbolizationTechnique": {
        "Simple Coiling": 0,
        "Balloon-Assisted": 1,
        "Stent-Assisted": 2
    },

    # AngioAndTreatment specific classification
    "AngioAndTreatment": {
        "Type A": 0,
        "Type B": 1,
        "Type C": 2
    },

    # Heparin Administration Timing
    "HeparinTiming": {
        "Pre-operative": 0,
        "Intra-operative": 1,
        "Post-operative": 2
    }
}

# ==========================================
# 3. Sidebar: Patient Clinical Parameters
# ==========================================
st.sidebar.header("📝 Patient Clinical Parameters")

# 1. ThrombolysisAfterTirofiban
input_thrombolysis = st.sidebar.selectbox(
    "Thrombolysis After Tirofiban",
    options=list(mapping_dict["ThrombolysisAfterTirofiban"].keys())
)

# 2. StentType
input_stent = st.sidebar.selectbox(
    "Stent Type",
    options=list(mapping_dict["StentType"].keys())
)

# 3. Morphology
input_morphology = st.sidebar.selectbox(
    "Aneurysm Morphology",
    options=list(mapping_dict["Morphology"].keys())
)

# 4. Rupture
input_rupture = st.sidebar.selectbox(
    "Rupture Status",
    options=list(mapping_dict["Rupture"].keys())
)

# 5. EmbolizationTechnique
input_technique = st.sidebar.selectbox(
    "Embolization Technique",
    options=list(mapping_dict["EmbolizationTechnique"].keys())
)

# 6. Width (Numerical)
# Adjust min_value and max_value based on your training data statistics
input_width = st.sidebar.number_input(
    "Aneurysm Width (mm)",
    min_value=0.0, max_value=50.0, value=5.0, step=0.1
)

# 7. Neck (Numerical)
input_neck = st.sidebar.number_input(
    "Aneurysm Neck (mm)",
    min_value=0.0, max_value=30.0, value=3.0, step=0.1
)

# 8. AngioAndTreatment
input_angio = st.sidebar.selectbox(
    "Angiography & Treatment",
    options=list(mapping_dict["AngioAndTreatment"].keys())
)

# 9. HeparinTiming
input_heparin = st.sidebar.selectbox(
    "Heparin Timing",
    options=list(mapping_dict["HeparinTiming"].keys())
)

# ==========================================
# 4. Main Interface: Prediction Logic
# ==========================================
st.title("🧠 Prediction of Recurrence after EVT of Intracranial Aneurysms")
st.markdown("---")

# Convert input to DataFrame format required by the model
# Note: Feature order must match the training data exactly!
input_data = {
    'ThrombolysisAfterTirofiban': mapping_dict["ThrombolysisAfterTirofiban"][input_thrombolysis],
    'StentType': mapping_dict["StentType"][input_stent],
    'Morphology': mapping_dict["Morphology"][input_morphology],
    'Rupture': mapping_dict["Rupture"][input_rupture],
    'EmbolizationTechnique': mapping_dict["EmbolizationTechnique"][input_technique],
    'Width': input_width,
    'Neck': input_neck,
    'AngioAndTreatment': mapping_dict["AngioAndTreatment"][input_angio],
    'HeparinTiming': mapping_dict["HeparinTiming"][input_heparin]
}

df_input = pd.DataFrame([input_data])

# Display current input
with st.expander("View Input Feature Values (Encoded)"):
    st.dataframe(df_input)

# Prediction Button
if st.button("🚀 Predict", type="primary"):
    if model:
        try:
            # Predict Probability
            prob = float(model.predict_proba(df_input)[:, 1][0])

            # Set Threshold (Use the best_threshold calculated from your validation set)
            # You need to change this value to your actual best_threshold
            best_threshold = 0.5

            prediction_class = 1 if prob >= best_threshold else 0

            # Result Display Section
            col1, col2 = st.columns(2)

            with col1:
                st.metric(label="Probability of Recurrence", value=f"{prob:.2%}")

            with col2:
                if prediction_class == 1:
                    st.error(f"⚠️ High Risk \n(> {best_threshold})")
                else:
                    st.success(f"✅ Low Risk \n(< {best_threshold})")

            # Progress Bar Visualization
            st.progress(prob, text="Risk Index")

            # Interpretative Text
            if prediction_class == 1:
                st.warning("Note: The model predicts a high risk of recurrence/thrombosis. Close monitoring or adjustment of the treatment strategy is recommended.")

        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.write("Please check if the input data format matches the training data.")
    else:
        st.error("Model not loaded. Cannot predict.")
