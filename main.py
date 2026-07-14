from pathlib import Path
import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Student Placement Prediction")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "placement_model.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

st.title("🎓 Student Placement Prediction")

cgpa = st.slider("CGPA", 0.0, 10.0, 7.0)

internships = st.number_input(
    "Internships",
    min_value=0,
    max_value=10,
    value=1
)

projects = st.number_input(
    "Projects",
    min_value=0,
    max_value=20,
    value=2
)

coding = st.slider(
    "Coding Skill Score",
    0,
    100,
    70
)

aptitude = st.slider(
    "Aptitude Score",
    0,
    100,
    70
)

communication = st.slider(
    "Communication Skill Score",
    0,
    100,
    70
)

if st.button("Predict"):
    data = pd.DataFrame(
        [[
            cgpa,
            internships,
            projects,
            coding,
            aptitude,
            communication
        ]],
        columns=[
            "cgpa",
            "internships_count",
            "projects_count",
            "coding_skill_score",
            "aptitude_score",
            "communication_skill_score",
        ],
    )

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.success("🎉 Student is likely to be Placed")
    else:
        st.error("❌ Student is likely to be Not Placed")