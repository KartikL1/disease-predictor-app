import streamlit as st
import pandas as pd
import numpy as np
import spacy
from rapidfuzz import fuzz, process
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report

st.set_page_config(page_title="🩺 Disease Predictor", layout="centered")
st.title("🩺 Disease Prediction System")
st.markdown("Enter your symptoms below to get possible disease predictions using both rule-based logic and a trained ML model.")

# -------------------------
# Load Data
# -------------------------
@st.cache_data
def load_data():
    df1 = pd.read_csv('data/dataset.csv')
    df2 = pd.read_csv('data/symptom_precaution.csv')
    df3 = pd.read_csv('data/symptom_Description.csv')
    df4 = pd.read_csv('data/Symptom-severity.csv')
    df4['Symptom'] = df4['Symptom'].str.lower().str.strip().str.replace(' ', '_')
    return df1, df2, df3, df4

df1, df2, df3, df4 = load_data()

symptom_severity = dict(zip(df4['Symptom'], df4['weight']))

symptoms_df = df1.melt(id_vars='Disease', value_name='Symptom').drop('variable', axis=1).dropna().reset_index(drop=True)
symptoms_df['Symptom'] = symptoms_df['Symptom'].str.lower().str.strip().str.replace(' ', '_')
all_symptoms = sorted(set(df4['Symptom']).union(symptoms_df['Symptom']))
symptom_corpus = [s.replace('_', ' ') for s in all_symptoms]
symptom_original_map = {s.replace('_', ' '): s for s in all_symptoms}

@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_md")
    except:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_md"])
        return spacy.load("en_core_web_md")

nlp = load_spacy_model()

symptom_definitions = {s: s.replace('_', ' ') for s in all_symptoms}
symptom_vectors = {s: nlp(d).vector for s, d in symptom_definitions.items() if nlp(d).vector_norm}

disease_symptoms = pd.get_dummies(symptoms_df.set_index('Disease')['Symptom']).groupby(level=0).max()
disease_info = df3.merge(df2, on='Disease', how='left')

# -------------------------
# Synonym group builder
# -------------------------
def build_synonym_dict(synonym_groups):
    synonym_dict = {}
    for group in synonym_groups:
        canonical = group[0].lower().replace(' ', '_')
        for term in group:
            synonym_dict[term.lower()] = canonical
            synonym_dict[term.lower().replace(' ', '_')] = canonical
    return synonym_dict

synonym_groups = [
    ["yellow skin", "yellowish skin", "jaundice", "yellow skin color"],
    ["yellow eyes", "yellowish eyes", "scleral icterus"],
    ["dark urine", "black urine", "cola colored urine"],
    ["fatigue", "tiredness", "exhaustion", "lack of energy"],
    ["itching", "pruritus", "skin itch"],
    ["high fever", "fever", "temperature", "pyrexia"],
    ["vomiting", "throwing up", "emesis"],
    ["shortness of breath", "breathlessness", "dyspnea"],
    ["chest pain", "pain in chest"],
    ["cough", "dry cough", "wet cough"],
    ["palpitations", "heart racing", "fast heartbeat"],
    ["abdominal pain", "stomach pain", "belly pain"],
    # Add more synonym groups as needed
]

SYMPTOM_SYNONYMS = build_synonym_dict(synonym_groups)

def preprocess_input_symptom(symptom):
    symptom = symptom.strip().lower()
    if symptom in SYMPTOM_SYNONYMS:
        return SYMPTOM_SYNONYMS[symptom]
    return symptom.replace(' ', '_')

# -------------------------
# Symptom matching functions
# -------------------------
def find_symptom_matches(user_input):
    doc = nlp(user_input.lower().strip())
    tokens = {t.lemma_ for t in doc if t.pos_ in ['NOUN', 'ADJ', 'VERB']}
    matches = set()
    for token in tokens:
        for s in all_symptoms:
            if token in s:
                matches.add(s)
        for s, defn in symptom_definitions.items():
            if token in defn.lower():
                matches.add(s)
    if not matches:
        fuzzy = process.extract(user_input, symptom_corpus, scorer=fuzz.token_set_ratio, limit=3)
        for match, score, _ in fuzzy:
            if score > 80:
                matches.add(symptom_original_map[match])
    if not matches:
        vec = doc.vector
        sims = []
        for s, v in symptom_vectors.items():
            sim = np.dot(vec, v) / (np.linalg.norm(vec) * np.linalg.norm(v))
            if sim > 0.6:
                sims.append((s, sim))
        sims.sort(key=lambda x: x[1], reverse=True)
        matches.update([s[0] for s in sims[:3]])
    return sorted(matches)

# -------------------------
# Rule-based disease prediction
# -------------------------
def predict_disease(matched_symptoms):
    results = []
    for disease in disease_symptoms.index:
        sym_series = disease_symptoms.loc[disease]
        disease_set = set(sym_series[sym_series == 1].index)
        matched = set(matched_symptoms) & disease_set
        if not matched:
            continue
        weight = sum(symptom_severity.get(s, 1) for s in matched)
        total = sum(symptom_severity.get(s, 1) for s in disease_set)
        match_score = weight / total if total else 0
        coverage = len(matched) / len(disease_set) if disease_set else 0
        combined = round((0.6 * match_score + 0.4 * coverage) * 100, 2)
        if combined < 20:
            continue
        conf = "🔒 High Confidence" if combined >= 70 else "⚠️ Medium Confidence" if combined >= 40 else "⚠️ Low Confidence"
        try:
            info = disease_info[disease_info['Disease'] == disease].iloc[0]
        except IndexError:
            continue
        results.append({
            'disease': disease,
            'description': info.get('Description', 'Not available'),
            'precautions': [info.get(f'Precaution_{i}') for i in range(1, 5) if pd.notna(info.get(f'Precaution_{i}'))],
            'matched': sorted(matched),
            'match_percent': combined,
            'severity': min(10, weight),
            'confidence': conf
        })
    results.sort(key=lambda x: (x['match_percent'], x['severity']), reverse=True)
    return results[:3] if results else None

# -------------------------
# Train ML model with reduced overfitting & cross-val report
# -------------------------
@st.cache_resource
def train_ml_model():
    ml_data = df1.fillna("none")
    ml_data["symptom_list"] = ml_data.iloc[:, 1:].values.tolist()
    ml_data["symptom_list"] = ml_data["symptom_list"].apply(
        lambda x: [i.strip().lower().replace(' ', '_') for i in x if i != "none"]
    )
    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(ml_data["symptom_list"])
    y = ml_data["Disease"]
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=3,
        min_samples_split=10,
        min_samples_leaf=10,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_cv = cross_val_predict(rf, X, y, cv=skf)
    report = classification_report(y, y_pred_cv, output_dict=True)
    rf.fit(X, y)
    return rf, mlb, report

# -------------------------
# ML prediction + wrapper for outputting per-disease metrics
# -------------------------
def predict_with_ml_model(clf, mlb, input_symptoms):
    user_vector = [1 if s in input_symptoms else 0 for s in mlb.classes_]
    proba = clf.predict_proba([user_vector])[0]
    pred_idx = np.argmax(proba)
    pred = clf.classes_[pred_idx]
    top3_idxs = np.argsort(proba)[-3:][::-1]
    top3 = [(clf.classes_[idx], proba[idx]) for idx in top3_idxs]
    return pred, top3

# ----
# Streamlit UI
# ----
user_input = st.text_area("Enter symptoms (comma separated):", placeholder="e.g. headache, chest pain, high fever", key="user_input_area")

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter symptoms.")
    else:
        # Apply synonym mapping here:
        raw_symptoms = [preprocess_input_symptom(s) for s in user_input.split(',')]
        matched_symptoms = []
        for symptom in raw_symptoms:
            matched = find_symptom_matches(symptom)
            if matched:
                matched_symptoms.append(matched[0])  # Best single match per input

        if not matched_symptoms:
            st.error("❌ No symptoms matched. Try using simpler medical terms.")
        else:
            clf, mlb, report = train_ml_model()

            # Rule-based predictions
            predictions = predict_disease(matched_symptoms)
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

            # ML model prediction
            ml_pred, ml_top3 = predict_with_ml_model(clf, mlb, matched_symptoms)
            st.subheader("🤖 ML Model Prediction")
            st.markdown("**Top-3 Suggestions:**")
            for disease, prob in ml_top3:
                st.markdown(f"- **{disease}** ({prob*100:.2f}%)")
            st.success(f"Predicted disease: **{ml_pred.upper()}**")

            disease_metrics = report.get(ml_pred)
            if disease_metrics:
                st.markdown("**Model reliability for this disease (cross-validated):**")
                st.markdown(f"- Precision: `{disease_metrics['precision']:.2f}`")
                st.markdown(f"- Recall: `{disease_metrics['recall']:.2f}`")
                st.markdown(f"- F1 Score: `{disease_metrics['f1-score']:.2f}`")
            else:
                st.markdown("*No evaluation data available for this disease.*")
