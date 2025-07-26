# 🩺 Healthcare Disease Diagnosis Tool


A smart healthcare application that predicts possible diseases based on user-entered symptoms. The system uses a hybrid approach combining:
- 🤖 A **Random Forest machine learning model** trained on a Kaggle symptom-disease dataset, carefully regularized to avoid overfitting.
- 🧠 **Rule-based logic** leveraging symptom severity and coverage scores.
- 💻 A friendly web interface built with **Streamlit** for ease of use.

---

## 🚀 Live Demo

> 🔗 **Not deployed yet**  
> You can run the app locally by following the steps below.

---

## 📂 Project Structure

```
disease-predictor-app/
├── app.py           # Main Streamlit app
├── healthcare.ipynb # Jupyter notebook (EDA or dev)
├── ml_model.py      # ML training logic (if modularized)
├── README.md        # Project overview and instructions
├── requirement.txt  # Python dependencies
├── data/            # Data folder
│ ├── dataset.csv
│ ├── symptom_Description.csv
│ ├── symptom_precaution.csv
│ └── Symptom-severity.csv

---

## 💡 Features

- Accepts comma-separated symptom inputs, including varied phrasing and misspellings.
- Maps symptoms using a synonym dictionary to canonical terms.
- Rule-based prediction displaying matched symptoms, severity, confidence, and precautions.
- ML model prediction showing top probable diseases with probabilities.
- Realistic performance reports displaying precision, recall, and F1-score for predicted diseases.
- Regularized Random Forest with stratified cross-validation to reduce overfitting.
- Descriptions and precautions help users understand suggested diseases.


---

## 💬 Sample Inputs

```
high fever, headache, joint pain
blurred vision, vomiting, fatigue
chest pain, cough, shortness of breath
```

---

## 🛠️ How to Run Locally

### 🔧 1. Clone this repository

```bash
git clone https://github.com/KartikL1/disease-predictor-app.git
cd disease-predictor-app
```

### 📦 2. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_md
```

### 🚀 3. Start the Streamlit app

```bash
streamlit run app.py
```

Then go to: [http://localhost:8501](http://localhost:8501)

---

- Enter symptoms separated by commas (e.g., `yellow skin, fatigue, dark urine`).
- The app will display rule-based and ML-based disease predictions.
- Below predictions, view confidence scores and model reliability metrics.
- Use the precaution advice and disease description for further information.

---

## How It Works

### Input Processing

- User inputs symptoms naturally.
- Inputs are preprocessed via a synonym dictionary mapping user phrases to canonical symptom terms.
- Unmatched terms go through fuzzy matching and semantic similarity to find best symptom matches.

### Rule-Based Logic

- For each disease, symptom matches are weighted by predefined symptom severity.
- Combined match and coverage scores produce a confidence percentage.
- Diseases above a threshold are suggested with descriptions and precautions.

## 📈 Machine Learning Model

- A Random Forest classifier is trained on the dataset with symptom presence as features.
- Model complexity is limited (`max_depth=3`, etc.) and evaluated via stratified 5-fold cross-validation.
- On user input, symptoms are converted to feature vectors and the model outputs top probable diseases with probabilities.
- Cross-validated precision, recall, and F1-scores for the predicted disease are shown.
---

## ⚙️ Tech Stack

- Python 3.8+
- Streamlit for UI
- Pandas and NumPy for data processing
- Scikit-learn for Random Forest ML model
- SpaCy for natural language processing
- RapidFuzz for fuzzy string matching

---

## 🧠 NLP & Matching

- Uses `spacy` for semantic similarity
- Fuzzy string matching with `rapidfuzz`
- Handles misspellings like `vommiting`, `pain in belly`, or `feverish`

---

## 🧪 Future Improvements

- Expand synonym dictionaries for greater natural language recognition.
- Incorporate demographic, lab, or imaging data for personalized predictions.
- Experiment with ensemble and deep learning models.
- Add user feedback mechanisms to continually improve predictions.
- Enable multi-language support and voice input.
- Regularly update and validate datasets with real clinical data.


## ✅ Why This Project Matters

- Supports timely clinical decisions with limited access to doctors.
- Bridges expert knowledge and machine learning for robust diagnosis assistance.
- Provides explainable outputs and actionable precautions.
- Improves healthcare accessibility and education.

---

## 👥 Team

Project by **Kartik & Mayur**  
📫 Contact: [kartiklahare122@gmail.com] or [https://www.linkedin.com/in/kartik-lahare-83a091241]


