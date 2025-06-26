
import streamlit as st
import pandas as pd
from ml_model import train_ml_model, predict_disease, predict_with_ml_model, find_symptom_matches

st.set_page_config(page_title="🩺 Disease Predictor", layout="centered")
st.title("🩺 Disease Prediction System")
st.markdown("Enter your symptoms below to get possible disease predictions using both rule-based logic and a trained ML model.")

user_input = st.text_area("Enter symptoms (comma separated):", placeholder="e.g. headache, chest pain, high fever")

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter symptoms.")
    else:
        raw_symptoms = [s.strip().lower().replace(" ", "_") for s in user_input.split(',')]
        matched_symptoms = []
        for symptom in raw_symptoms:
            matched = find_symptom_matches(symptom)
            if matched:
                matched_symptoms.append(matched[0])  # best match

        if not matched_symptoms:
            st.error("❌ No symptoms matched. Try using simpler medical terms.")
        else:
            clf, mlb = train_ml_model()
            predictions = predict_disease(matched_symptoms)
            ml_prediction = predict_with_ml_model(clf, mlb, matched_symptoms)

            st.subheader("🔍 Rule-Based Predictions")
            if predictions:
                for i, pred in enumerate(predictions, 1):
                    st.markdown(f"**{i}. {pred['disease'].upper()}** ({pred['match_percent']}% match)")
                    st.markdown(f"🧾 **Description**: {pred['description']}")
                    st.markdown(f"🧩 **Symptoms matched**: {', '.join(pred['matched'])}")
                    st.markdown(f"🔥 **Severity**: {pred['severity']}/10")
                    st.markdown(f"📊 **Confidence**: {pred['confidence']}")
                    if pred['precautions']:
                        st.markdown("💊 **Precautions:**")
                        for p in pred['precautions']:
                            st.markdown(f"- {p}")
            else:
                st.warning("No match found in rule-based system.")

            st.subheader("🤖 ML Model Prediction")
            st.success(f"ML model suggests: **{ml_prediction.upper()}**")
