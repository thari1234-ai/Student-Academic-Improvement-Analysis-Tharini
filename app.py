import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from datetime import datetime

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Student Academic Improvement",
    layout="centered"
)

st.title("📊 Student Academic Improvement Analysis By Tharini Ps")
st.caption("Polynomial Regression based student performance tracking")

# ================= FORM =================
with st.form("student_form"):

    st.subheader("👤 Student Details")
    name = st.text_input("Enter Student Name")
    roll_no = st.text_input("Enter Roll Number")

    st.subheader("📘 Academic Inputs")
    semester_pct = st.number_input("Semester Percentage (%)", 0.0, 100.0, 70.0)
    attendance_pct = st.number_input("Attendance Percentage (%)", 0.0, 100.0, 80.0)
    homework_pct = st.number_input("Homework Completion Percentage (%)", 0.0, 100.0, 75.0)
    study_hours = st.number_input("Average Study Hours per Day", 0.0, 12.0, 2.0)

    st.subheader("📅 Weekly Test Scores")
    weeks = np.array([1, 2, 3, 4, 5])
    scores = []

    for i in range(5):
        scores.append(st.number_input(f"Week {i+1} Test Score", 0, 100, 0))

    submit = st.form_submit_button("Analyze Improvement")

# ================= PROCESS =================
if submit:

    if not name or not roll_no:
        st.error("Please enter both Name and Roll Number")
        st.stop()

    y = np.array(scores)
    X = weeks.reshape(-1, 1)

    # Polynomial Regression
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    # ================= LOGIC FIX =================
    avg_score = np.mean(y)
    score_growth = y[-1] - y[0]

    if avg_score >= 90 and score_growth == 0:
        category = "High Consistent Performance"
        bg = "#007bfe"

    elif score_growth >= 15:
        category = "High Improvement"
        bg = "#abe5a7"

    elif score_growth >= 5:
        category = "Moderate Improvement"
        bg = "#6a5101"

    else:
        category = "Low Improvement"
        bg = "#5d040c"

    # ================= REASONS =================
    reasons = []

    if category == "High Consistent Performance":
        reasons.append("Consistently high scores across all weeks")
        reasons.append("Student has already reached an excellent performance level")

    elif category == "High Improvement":
        reasons.append("Strong upward trend in weekly test scores")

    elif category == "Moderate Improvement":
        reasons.append("Gradual improvement with minor score fluctuations")

    else:
        reasons.append("Minimal score growth over weeks")

    # Academic inputs reasoning
    if semester_pct >= 75:
        reasons.append("Good overall semester performance")
    else:
        reasons.append("Semester performance needs improvement")

    if attendance_pct >= 85:
        reasons.append("Consistent class attendance")
    else:
        reasons.append("Attendance inconsistency affected learning")

    if homework_pct >= 80:
        reasons.append("Regular homework completion")
    else:
        reasons.append("Irregular homework practice")

    if study_hours >= 3:
        reasons.append("Sufficient daily study hours")
    else:
        reasons.append("Insufficient daily study time")

    # ================= OUTPUT =================
    st.markdown(
        f"""
        <div style="background-color:{bg}; padding:15px; border-radius:10px">
        <h4>📌 Improvement Category: {category}</h4>
        <b>Name:</b> {name}<br>
        <b>Roll No:</b> {roll_no}<br>
        <b>Average Score:</b> {avg_score:.2f}<br>
        <b>Score Growth:</b> {score_growth}<br>
        <b>Generated On:</b> {datetime.now().strftime("%d-%m-%Y %H:%M:%S")}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("📋 Reason Analysis")
    for r in reasons:
        st.write("•", r)

    # ================= GRAPH =================
    X_plot = np.linspace(1, 5, 100).reshape(-1, 1)
    y_plot = model.predict(poly.transform(X_plot))

    fig, ax = plt.subplots()
    ax.scatter(X, y, label="Actual Scores", s=60)
    ax.plot(X_plot, y_plot, label="Performance Trend")
    ax.set_xlabel("Week")
    ax.set_ylabel("Test Score")
    ax.set_title("Academic Performance Trend")
    ax.legend()

    st.pyplot(fig)
