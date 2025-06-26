# 🩺 Disease Prediction System

A smart healthcare application that predicts possible diseases based on user-entered symptoms. The system uses a hybrid approach combining:
- 🤖 Machine Learning (Random Forest Classifier)
- 🧠 Rule-based logic with semantic symptom matching
- 💻 Streamlit web interface for user interaction

---

## 🚀 Live Demo

> 🔗 **Not deployed yet**  
> You can run the app locally by following the steps below.

---

## 📂 Project Structure

```
disease-predictor-app/
├── app.py                  # Streamlit web frontend
├── ml_model.py             # Rule-based & ML backend logic
├── requirements.txt        # Required Python packages
├── dataset.csv             # Symptom-Disease mapping
├── symptom_Description.csv # Descriptions of diseases
├── symptom_precaution.csv  # Precautions per disease
└── Symptom-severity.csv    # Symptom severity weights
```

---

## 💡 Features

- ✅ Predicts top 3 diseases based on symptoms
- ✅ Symptom severity–weighted logic
- ✅ Uses Random Forest classifier trained on symptom dataset
- ✅ Fuzzy + semantic symptom matching (handles typos & vague inputs)
- ✅ Disease description, precautions, severity score & confidence level
- ✅ Clean, modular code ready for demo or interview
- ✅ Built with Python and Streamlit

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

## 📈 Machine Learning Model

- **Random Forest Classifier** using `scikit-learn`
- Trained at runtime on `dataset.csv` symptom–disease pairs
- Accuracy printed on training
- Combined with rule-based system for explainability

---

## ⚙️ Tech Stack

- Python
- Streamlit (frontend)
- scikit-learn (ML)
- pandas & numpy (data handling)
- SpaCy + RapidFuzz (NLP & fuzzy logic)

---

## 🧠 NLP & Matching

- Uses `spacy` for semantic similarity
- Fuzzy string matching with `rapidfuzz`
- Handles misspellings like `vommiting`, `pain in belly`, or `feverish`

---

## 🧪 Future Improvements

- PDF report generation
- User feedback collection
- Disease confidence visualization (charts)
- Deploy to Streamlit Cloud / HuggingFace Spaces

---

## 👥 Team

Project by **Kartik & Mayur**  
📫 Contact: [kartiklahare122@gmail.com] or [https://www.linkedin.com/in/kartik-lahare-83a091241]


