import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Student Performance Analysis",
    page_icon="🎓",
    layout="wide",
)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data(file=None):
    if file is not None:
        return pd.read_csv(file)
    return pd.read_csv("data/students.csv")

st.sidebar.title("🎓 Navigation")
uploaded_file = st.sidebar.file_uploader("Upload your own CSV (optional)", type=["csv"])
df = load_data(uploaded_file)

page = st.sidebar.radio("Go to", ["🏠 Home", "📊 EDA", "📈 Visualizations", "🤖 Predict Marks"])

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.markdown("---")
st.sidebar.subheader("Filters")
section_filter = st.sidebar.multiselect(
    "Section", options=sorted(df["section"].unique()), default=list(df["section"].unique())
)
gender_filter = st.sidebar.multiselect(
    "Gender", options=sorted(df["gender"].unique()), default=list(df["gender"].unique())
)

filtered_df = df[df["section"].isin(section_filter) & df["gender"].isin(gender_filter)]

# ---------------- HOME ----------------
if page == "🏠 Home":
    st.title("🎓 Student Performance Analysis App")
    st.markdown("Analyze student marks, attendance, and predict performance — all in one place.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Students", len(filtered_df))
    col2.metric("Average Score", f"{filtered_df['average_score'].mean():.1f}")
    col3.metric("Pass %", f"{(filtered_df['result'] == 'Pass').mean() * 100:.1f}%")
    col4.metric("Avg Attendance", f"{filtered_df['attendance'].mean():.1f}%")

    st.markdown("### 🔍 Dataset Preview")
    st.dataframe(filtered_df, use_container_width=True)

# ---------------- EDA ----------------
elif page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")

    st.markdown("### Summary Statistics")
    st.dataframe(filtered_df.describe(), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Section-wise Average Score")
        st.dataframe(filtered_df.groupby("section")["average_score"].mean().round(1))
    with col2:
        st.markdown("### Parent Education vs Avg Score")
        st.dataframe(filtered_df.groupby("parent_education")["average_score"].mean().round(1))

    st.markdown("### Pass / Fail Count")
    st.dataframe(filtered_df["result"].value_counts())

# ---------------- VISUALIZATIONS ----------------
elif page == "📈 Visualizations":
    st.title("📈 Visualizations")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Subject-wise Average Scores")
        subj_avg = filtered_df[["math_score", "science_score", "english_score"]].mean().reset_index()
        subj_avg.columns = ["Subject", "Average"]
        fig1 = px.bar(subj_avg, x="Subject", y="Average", color="Subject", text_auto=".1f")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.markdown("#### Grade Distribution")
        fig2 = px.histogram(filtered_df, x="average_score", nbins=20, color="result")
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### Study Hours vs Average Score")
        fig3 = px.scatter(filtered_df, x="study_hours", y="average_score", color="result",
                           trendline="ols", hover_data=["name"])
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown("#### Attendance vs Average Score")
        fig4 = px.scatter(filtered_df, x="attendance", y="average_score", color="result",
                           trendline="ols", hover_data=["name"])
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("#### Correlation Heatmap")
    corr_cols = ["study_hours", "attendance", "math_score", "science_score", "english_score", "average_score"]
    corr = filtered_df[corr_cols].corr()
    fig5 = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto")
    st.plotly_chart(fig5, use_container_width=True)

# ---------------- PREDICTION ----------------
elif page == "🤖 Predict Marks":
    st.title("🤖 Predict Average Score")
    st.markdown("Enter study hours and attendance to predict a student's likely average score.")

    X = df[["study_hours", "attendance"]]
    y = df["average_score"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))

    st.info(f"Model R² score on test data: **{r2:.2f}**")

    col1, col2 = st.columns(2)
    with col1:
        study_hours_input = st.slider("Study Hours per Day", 0.0, 10.0, 4.0, 0.5)
    with col2:
        attendance_input = st.slider("Attendance (%)", 40.0, 100.0, 80.0, 1.0)

    if st.button("Predict"):
        pred = model.predict([[study_hours_input, attendance_input]])[0]
        pred = float(np.clip(pred, 0, 100))
        st.success(f"📌 Predicted Average Score: **{pred:.1f}**")
        if pred >= 40:
            st.balloons()
            st.write("✅ Likely to **Pass**")
        else:
            st.write("⚠️ Likely to **Fail** — needs more study hours / attendance")

st.sidebar.markdown("---")
st.sidebar.caption("Built with ❤️ using Streamlit")
