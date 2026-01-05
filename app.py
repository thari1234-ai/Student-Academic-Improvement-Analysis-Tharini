import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from datetime import datetime

# ================= Page Config =================
st.set_page_config(page_title="Student Academic Improvement", layout="centered")

# ================= Background Image =================
# Replace this URL with your desired Google image link
bg_image_url = "https://images.unsplash.com/photo-1581090700227-38e6bfb57f30?auto=format&fit=crop&w=1350&q=80"

page_bg_img = f"""
<style>
body {{
background-image: url("{bg_image_url}");
background-size: cover;
background-repeat: no-repeat;
background-attachment: fixed;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# ================= Title =================
st.title("📊 Student Academic Improvement Analysis")
st.caption("Polynomial Regression based student performance tracking")

# ================= FORM =================
with st.form("student_form"):

    st.subheader("👤 Student Details")
    name = st.text_input("Enter Student Name")
    roll_no = st.text_input("Enter Roll Number")

    st.subheader("📘 Academic Inputs (User Entered)")
    semester_pct = st.slider("Semester Percentage (%)", 0, 100, 70)
    attendance_pct = st.slider("Attendance Percentage (%)", 0, 100, 80)
    homework_pct = st.slider("Homework Completion Percentage (%)", 0, 100, 75)
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
        st.error("Please enter Name and Roll Number")
        st.stop()

    # Prepare data
    X = weeks.reshape(-1, 1)
    y = np.array(scores)

    # Polynomial Regression
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    coef = model.coef_
    improvement_rate = coef[1] + 2 * coef[2] * X.max()

    # ================= CATEGORY & Box Color =================
    if improvement_rate > 2:
        category = "High Improvement"
        box_bg = "#05ff3f"
    elif improvement_rate >= 1:
        category = "Moderate Improvement"
        box_bg = "#f4bb00"
    else:
        category = "Low Improvement"
        box_bg = "#52050b"

    # ================= REASONS =================
    reasons = []
    # Score trend reason
    if improvement_rate > 2:
        reasons.append("✔ Noticeable score growth over weeks")
    elif improvement_rate >= 1:
        reasons.append("✔ Some improvement but inconsistent performance")
    else:
        reasons.append("✖ Minimal score growth over weeks")

    # Semester reason
    if semester_pct >= 75:
        reasons.append("✔ Strong semester performance")
    else:
        reasons.append("✖ Low semester percentage")

    # Attendance reason
    if attendance_pct >= 85:
        reasons.append("✔ Good attendance consistency")
    else:
        reasons.append("✖ Irregular attendance")

    # Homework reason
    if homework_pct >= 80:
        reasons.append("✔ Homework completed regularly")
    else:
        reasons.append("✖ Homework completion is low")

    # Study hours reason
    if study_hours >= 3:
        reasons.append("✔ Adequate daily study hours")
    else:
        reasons.append("✖ Insufficient daily study time")

    # ================= OUTPUT BOX =================
    st.markdown(
        f"""
        <div style="background-color:{box_bg}; padding:15px; border-radius:10px; color:white;">
        <h4>📌 Improvement Category: {category}</h4>
        <b>Name:</b> {name}<br>
        <b>Roll No:</b> {roll_no}<br>
        <b>Improvement Rate:</b> {improvement_rate:.2f}<br>
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
    ax.set_facecolor("#ffffff80")  # semi-transparent white background for graph
    ax.scatter(X, y, label="Actual Scores", color="blue")
    ax.plot(X_plot, y_plot, label="Polynomial Regression Curve", color="red")
    ax.set_xlabel("Week")
    ax.set_ylabel("Test Score")
    ax.set_title("Academic Improvement Trend")
    ax.legend()
    st.pyplot(fig)
