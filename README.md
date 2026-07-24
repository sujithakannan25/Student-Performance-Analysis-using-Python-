🎓 Student Performance Analysis using Python:

A data analysis and machine learning project built with Python & Streamlit to explore, visualize, and predict student academic performance based on study hours, attendance, and other key factors.

📌 Project Overview:

Student academic performance is influenced by multiple factors such as study habits, attendance, and parental education background. This project analyzes a student dataset to uncover patterns and trends, and builds a simple Machine Learning model to predict a student's average score.

The project is delivered as an interactive Streamlit web application, allowing users to:

*Explore the dataset visually
*Filter data by section and gender
*Understand relationships between study hours, attendance, and scores
*Predict average scores using a trained ML model
8Upload their own dataset (CSV) for custom analysis

🛠️ Tools:  

Python – Programming language

Streamlit – Web app framework

Pandas, NumPy – Data handling & analysis
 
Matplotlib / Plotly – Data visualization

Scikit-learn – Machine learning (Linear Regression)

VS Code – Development environment

Git & GitHub – Version control

Streamlit Community Cloud – Deployment

Want to be notified when Claude responds?

📁 Dataset:

*The dataset (students.csv) contains information of 300 students with the following columns:

*A synthetic dataset is included (generate_dataset.py) to regenerate sample data if needed.

⚙️ Project Workflow: 

Data Collection – Load dataset (built-in sample or user-uploaded CSV)

Data Cleaning & Preprocessing – Handle missing values, format columns

Exploratory Data Analysis (EDA) – Summary statistics, section-wise & parent-education breakdowns

Data Visualization – Bar charts, histograms, scatter plots with trendlines, correlation heatmap

Model Building – Train a Linear Regression model to predict average_score

Model Evaluation – Check prediction accuracy using metrics like R² score

Deployment – Build an interactive dashboard using Streamlit and deploy on Streamlit Community Cloud

📊 Dashboard:

The Streamlit dashboard includes:

🎚️ Sidebar Filters – Filter by section and gender
📈 Visual Insights – Score distributions, attendance vs. score trends, correlation heatmap
🤖 Prediction Tool – Input study hours & attendance to predict average score
📁 CSV Upload – Analyze your own student dataset
Run Locally
bash
pip install -r requirements.txt
streamlit run app.py

Then open http://localhost:8501 in your browser.

Live Deployment:

Deployed for free on Streamlit Community Cloud.

👤 Author

[sujitha kannan] 

📧 Email: k.sujithakannane2006@gmail.com 

🔗 LinkedIn: www.linkedin.com/in/sujithakannan25
