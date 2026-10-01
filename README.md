# 🎓 Student Placement & Salary Prediction System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://studentplacementprediction-uaysosrgoqxh2mor8sgnsa.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)

An end-to-end Machine Learning web application designed to predict campus placement eligibility and estimate potential salary packages (LPA) based on academic performance, core aptitude, soft skills, and domain experience.

---

## 🔗 Live Application
Access the live interactive Streamlit application here:  
👉 **[Student Placement Prediction Web App](https://studentplacementprediction-uaysosrgoqxh2mor8sgnsa.streamlit.app)**

---

## 🚀 Key Features

- **Dual-Model ML Architecture**:
  - **Classification**: Evaluates whether a student is likely to be placed using a `GradientBoostingClassifier`.
  - **Regression**: Predicts the expected salary package (in LPA) for placed candidates using a `RandomForestRegressor`.
- **Interactive Web Interface**: User-friendly control panel powered by Streamlit for instant inputs and real-time inference.
- **Custom Feature Engineering**: Evaluates academic GPA, aptitude scores, technical skill proficiency, internships, and soft skills to derive predictive insights.

---

## 🛠️ Tech Stack & Dependencies

- **Language**: Python
- **Frontend / Framework**: Streamlit
- **Machine Learning**: Scikit-Learn
- **Data Manipulation**: Pandas, NumPy
- **Model Serialization**: Joblib

---

## 📁 Repository Structure

```text
├── app.py                         # Streamlit application script
├── placement_model_pipeline.pkl   # Pre-trained dual ML pipeline
├── students_placement_v2.csv      # Dataset used for training & verification
├── requirements.txt               # Deployment dependencies
├── .gitignore                     # Ignored files configuration
└── README.md                      # Documentation
