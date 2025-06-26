
import pandas as pd
import numpy as np
import spacy
from rapidfuzz import fuzz, process
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load data
df1 = pd.read_csv('dataset.csv')
df2 = pd.read_csv('symptom_precaution.csv')
df3 = pd.read_csv('symptom_Description.csv')
df4 = pd.read_csv('Symptom-severity.csv')

df4['Symptom'] = df4['Symptom'].str.lower().str.strip().str.replace(' ', '_')
symptom_severity = dict(zip(df4['Symptom'], df4['weight']))

symptoms_df = df1.melt(id_vars='Disease', value_name='Symptom')     .drop('variable', axis=1).dropna().reset_index(drop=True)
symptoms_df['Symptom'] = symptoms_df['Symptom'].str.lower().str.strip().str.replace(' ', '_')

all_symptoms = sorted(set(df4['Symptom']).union(symptoms_df['Symptom']))
symptom_corpus = [s.replace('_', ' ') for s in all_symptoms]
symptom_original_map = {s.replace('_', ' '): s for s in all_symptoms}

try:
    nlp = spacy.load("en_core_web_md")
except:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_md"])
    nlp = spacy.load("en_core_web_md")


symptom_definitions = {s: s.replace('_', ' ') for s in all_symptoms}
symptom_vectors = {s: nlp(d).vector for s, d in symptom_definitions.items() if nlp(d).vector_norm}

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

disease_symptoms = pd.get_dummies(symptoms_df.set_index('Disease')['Symptom']).groupby(level=0).max()
disease_info = df3.merge(df2, on='Disease', how='left')

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

def train_ml_model():
    ml_data = df1.fillna("none")
    ml_data["symptom_list"] = ml_data.iloc[:, 1:].values.tolist()
    ml_data["symptom_list"] = ml_data["symptom_list"].apply(
        lambda x: [i.strip().lower().replace(' ', '_') for i in x if i != "none"]
    )
    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(ml_data["symptom_list"])
    y = ml_data["Disease"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    return clf, mlb

def predict_with_ml_model(clf, mlb, input_symptoms):
    user_vector = [1 if s in input_symptoms else 0 for s in mlb.classes_]
    return clf.predict([user_vector])[0]
