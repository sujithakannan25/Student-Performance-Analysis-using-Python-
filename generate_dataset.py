"""
Generates a synthetic Student Performance dataset.
Run this once to create data/students.csv
"""
import numpy as np
import pandas as pd

np.random.seed(42)

n = 300  # number of students

names_first = ["Arun", "Divya", "Karthik", "Priya", "Suresh", "Meena", "Vikram",
               "Anitha", "Ravi", "Deepa", "Sanjay", "Lakshmi", "Naveen", "Pooja",
               "Manoj", "Swathi", "Ajay", "Nisha", "Ganesh", "Kavya"]
names_last = ["Kumar", "Raj", "Sharma", "Iyer", "Nair", "Reddy", "Pillai",
              "Menon", "Rao", "Das"]

genders = np.random.choice(["Male", "Female"], size=n)
sections = np.random.choice(["A", "B", "C"], size=n)
parent_education = np.random.choice(
    ["High School", "Graduate", "Post Graduate"], size=n, p=[0.3, 0.45, 0.25]
)

study_hours = np.round(np.random.normal(loc=4, scale=1.8, size=n).clip(0.5, 10), 1)
attendance = np.round(np.random.normal(loc=80, scale=12, size=n).clip(40, 100), 1)

# Marks depend on study_hours + attendance + some randomness (so ML model has signal)
def generate_marks(base_hours, base_attendance):
    score = (
        35
        + base_hours * 6.5
        + base_attendance * 0.3
        + np.random.normal(0, 8)
    )
    return int(np.clip(score, 0, 100))

math_score = [generate_marks(h, a) for h, a in zip(study_hours, attendance)]
science_score = [generate_marks(h, a) for h, a in zip(study_hours, attendance)]
english_score = [generate_marks(h * 0.8, a) for h, a in zip(study_hours, attendance)]

df = pd.DataFrame({
    "student_id": range(1, n + 1),
    "name": [f"{np.random.choice(names_first)} {np.random.choice(names_last)}" for _ in range(n)],
    "gender": genders,
    "section": sections,
    "parent_education": parent_education,
    "study_hours": study_hours,
    "attendance": attendance,
    "math_score": math_score,
    "science_score": science_score,
    "english_score": english_score,
})

df["average_score"] = df[["math_score", "science_score", "english_score"]].mean(axis=1).round(1)
df["result"] = np.where(df["average_score"] >= 40, "Pass", "Fail")

df.to_csv("data/students.csv", index=False)
print("Dataset created: data/students.csv")
print(df.head())
