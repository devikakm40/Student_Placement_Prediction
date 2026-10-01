import streamlit as st
import pandas as pd
import joblib

# Page Configuration
st.set_page_config(
    page_title="Student Placement & Salary Predictor",
    page_icon="🎓",
    layout="wide"
)

# Load trained ML pipeline
@st.cache_resource
def load_pipeline():
    return joblib.load('placement_model_pipeline.pkl')

pipeline = load_pipeline()
preprocessor = pipeline['preprocessor']
classifier = pipeline['classifier']
regressor = pipeline['regressor']

# App Title
st.title("🎓 Student Placement & Salary Prediction System")
st.markdown("Enter student academic performance, skills, and target career goals to predict placement outcome and estimated LPA package.")

st.divider()

# Input Layout
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📚 Academic Background")
    score_10th = st.number_input("10th Class Percentage (%)", min_value=40.0, max_value=100.0, value=75.0, step=0.5)
    puc_score = st.number_input("12th / PUC Percentage (%)", min_value=40.0, max_value=100.0, value=78.0, step=0.5)
    puc_stream = st.selectbox("PUC Stream", ["Science", "Commerce", "Arts"])

    # Dynamic Branch selection based on PUC stream
    if puc_stream == "Science":
        branch_options = ["CSE", "AIML", "AIDS", "ECE", "EEE", "Aerospace", "Mechanical", "Civil"]
    elif puc_stream == "Commerce":
        branch_options = ["B.Com", "BBA", "CSE"]
    else:
        branch_options = ["BA", "BBA", "B.Com"]

    degree_branch = st.selectbox("Degree Branch", branch_options)
    cgpa = st.number_input("Degree CGPA (out of 10)", min_value=5.0, max_value=10.0, value=7.8, step=0.1)

with col2:
    st.subheader("🛠️ Skills & Practical Experience")
    internships = st.slider("Number of Internships", min_value=0, max_value=5, value=1)
    projects = st.slider("Number of Projects", min_value=0, max_value=10, value=2)
    aptitude_score = st.slider("Aptitude Test Score", min_value=40, max_value=100, value=70)
    soft_skills_score = st.slider("Soft Skills Assessment Score", min_value=40, max_value=100, value=72)

with col3:
    st.subheader("🎯 Target Career Goal")
    role_options = [
        "Software Developer", "AI Engineer", "ML Engineer", "Data Scientist", 
        "Data Analyst", "Web Developer", "DevOps Engineer", "Embedded Engineer", 
        "VLSI Design Engineer", "Financial Analyst", "Business Analyst", "HR Executive"
    ]
    target_role = st.selectbox("Target Job Role", role_options)

st.divider()

# Prediction Action
if st.button("🚀 Predict Placement Outcome", use_container_width=True):
    # Calculate engineered features matching train_model.py
    academic_avg = (score_10th * 0.2) + (puc_score * 0.2) + (cgpa * 6)
    skill_index = (aptitude_score * 0.6) + (soft_skills_score * 0.4)
    practical_exp = (internships * 2.5) + projects

    # Input DataFrame construction
    input_data = pd.DataFrame({
        'score_10th': [score_10th],
        'puc_score': [puc_score],
        'puc_stream': [puc_stream],
        'degree_branch': [degree_branch],
        'cgpa': [cgpa],
        'internships': [internships],
        'projects': [projects],
        'aptitude_score': [aptitude_score],
        'soft_skills_score': [soft_skills_score],
        'target_role': [target_role],
        'academic_avg': [academic_avg],
        'skill_index': [skill_index],
        'practical_exp': [practical_exp]
    })

    # Preprocess & Predict
    transformed_input = preprocessor.transform(input_data)
    placed_pred = classifier.predict(transformed_input)[0]
    placed_prob = classifier.predict_proba(transformed_input)[0][1] * 100

    st.subheader("📊 Prediction Results")
    res_col1, res_col2 = st.columns(2)

    with res_col1:
        if placed_pred == 1:
            st.success("🎉 **Status: Likely to be Placed!**")
            st.metric("Placement Confidence", f"{placed_prob:.1f}%")
        else:
            st.error("⚠️ **Status: Placement Unlikely**")
            st.metric("Placement Confidence", f"{placed_prob:.1f}%")
            st.info("💡 **Key Improvement Areas:** Boosting CGPA above 7.0 and scoring higher on aptitude assessments yield the largest probability increases.")

    with res_col2:
        if placed_pred == 1:
            predicted_salary = regressor.predict(transformed_input)[0]
            st.metric("Estimated Package", f"₹ {predicted_salary:.2f} LPA")
        else:
            st.metric("Estimated Package", "N/A (Unplaced)")
