# 🏥 Hospital Readmission Prediction  
### *Impact of Medication on Lifestyle Disease Patients*  
An End-to-End Machine Learning Project using **XGBoost + Streamlit**

---

## 📌 Project Overview
Hospital readmission is a major challenge in healthcare, especially for lifestyle disease patients such as diabetes, cardiovascular and respiratory illness patients.  
This project builds a **machine learning-based decision support system** that predicts whether a patient is likely to be readmitted, helping hospitals:

- Implement early interventions  
- Optimize resource allocation  
- Improve patient care outcomes  

Predict hospital readmission risk using real-world healthcare data. This project performs EDA, feature engineering and ML modeling to identify key factors like medications, visits and diagnoses influencing readmission, evaluated using F1-score for accurate and actionable insights. Supports better clinical planning and patient outcomes.

---

## 🎯 Objectives
- Clean & preprocess real-world hospital data  
- Engineer meaningful clinical features  
- Handle **class imbalance using SMOTE**  
- Train & compare multiple ML models  
- Deploy the best model using **Streamlit Web App**  
- Provide actionable hospital decision support insights  

---

## 📂 Dataset
- **Source:** Medical Information Team (MiTH) Hackathon  
- **Training Records:** 66,587  
- **Features:** 48 demographic, clinical & administrative attributes  
- **Target:** Readmitted (0/1)  

Dataset contains:
- Demographics  
- Diagnosis Codes  
- Visit History  
- Medication Patterns  
- Encounter Details  

---

## 🛠️ Tech Stack
**Programming Language:** Python 3.x  

**Libraries Used**
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Imbalanced-Learn
- Matplotlib, Seaborn

**Deployment:** Streamlit

---

## 🧠 Methodology
✔ Data Cleaning & Preparation  
✔ Categorical Encoding  
✔ **Feature Engineering**
- Total visit aggregation
- Diagnosis grouping
- Medication change indicators
- Clinical severity score  

✔ **Class Imbalance Handling** using SMOTE  
✔ Model Training + Hyperparameter Tuning  
✔ Probability Threshold Optimization  

---

## 🏆 Model Performance
Multiple models were trained & evaluated:

| Model | Best F1 Score | Best Threshold |
|-------|--------------|----------------|
| Logistic Regression | 0.6395 | 0.35 |
| Random Forest | 0.6397 | 0.35 |
| Gradient Boosting | 0.6404 | 0.25 |
| **XGBoost (Winner)** | **0.6426** | **0.30** |
| Perpetual Boosting | 0.6380 | 0.25 |

**Final Selected Model:** 🥇 XGBoost  
- High recall → captures majority of high-risk patients  
- Balanced precision & recall  
- Suitable for clinical decision support  

---

## 🔍 Key Insights
- Patients with frequent hospital visits are at higher risk
- Long hospital stays & multiple procedures increase readmission probability
- **Medication changes significantly influence readmission**
- Diabetes & chronic illness patients are more vulnerable
- Some medical specialties show higher readmission likelihood

---

## 🚀 Streamlit Application
An interactive Streamlit web application is developed to:
- Input patient details
- Predict readmission risk
- Support real-time hospital decision-making

### ▶️ Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Project Structure
```
├── data/                     # Dataset files
├── notebooks/                # EDA & Model development
├── models/                   # Trained model files
├── app.py                    # Streamlit web application
├── README.md
└── requirements.txt
```

---

## 👨‍💻 Author
**O Nithin Sai Balaji**  
📌 Passionate about AI, ML & Healthcare Analytics  
🔗 LinkedIn: https://www.linkedin.com/in/o-ns23/  
🐙 GitHub: https://github.com/O-NS23  

---

## 🙌 Contributions
Contributions, suggestions, and improvements are welcome!  
Feel free to fork the repo ⭐, enhance the project, and raise a pull request.

---

## 📜 License
This project is developed for **educational and research purposes**.
