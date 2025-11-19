# InterviewGuard-Interview Attendance Risk Predictor


End-to-end machine learning pipeline to **predict** whether a candidate will attend a scheduled recruitment interview, with a production-style Flask API for real-time scoring. 

---

## 🚀 Project Overview 

Companies lose time and revenue when candidates do not show up for scheduled interviews, making proactive no-show prediction highly valuable for recruiters. 
This project builds a full binary classification pipeline using scikit-learn and Flask to estimate the probability of a candidate missing an interview based on historical and scheduling features. 

---

## 🧩 Problem Statement 

Given candidate and interview-related features, the task is to predict the target variable `no_show`, where `1` indicates a no-show and `0` indicates the candidate attended.   
The model output can be used to trigger reminders, double-confirmations, or rescheduling strategies for high-risk candidates. 

---

## 📊 Dataset Description 

Each row represents a scheduled interview with the following features. 

- `job_role` (object): role for which the candidate is interviewing.   
- `experience` (float64): years of relevant experience. 
- `notice_period` (int64): candidate’s notice period in days.   
- `interview_time` (object): scheduled interview datetime.   
- `past_reschedules` (int64): number of past reschedules for this candidate.   
- `current_company_size` (object): size bucket of current employer.   
- `candidate_location` (object): candidate city or region.   
- `communication_score` (float64): screening or communication rating.   
- `expected_salary` (int64): expected annual salary. 
- `scheduled_day_gap` (int64): days between scheduling and interview date. 
- `no_show` (int64): target label, `1` = no-show, `0` = attended. 

---

## 🛠️ Features & Pipeline 

The project follows a clean, industry-style ML pipeline from raw CSV to deployed REST API.   

- **EDA & Data Understanding**: Target balance, distributions, correlations, and categorical breakdowns by `no_show`.   
- **Cleaning & Preprocessing**: Handling missing values, type conversions, and robust encoders for categorical features. 
- **Feature Engineering**: Time-based and ratio features to capture scheduling and behaviour patterns.  
- **Modeling**: Multiple classifiers evaluated, with Random Forest (or similar tree-based model) inside a unified scikit-learn `Pipeline`. 
- **Hyperparameter Tuning**: `RandomizedSearchCV` on the full pipeline using ROC-AUC as the main optimization metric. 
- **Evaluation**: ROC-AUC, F1-score, precision, recall, and confusion matrix on validation and test splits.  
- **Deployment**: Flask API that loads the serialized pipeline and exposes a `/predict` endpoint for real-time scoring. 

Key engineered features include: 

- `interview_hour` and `time_of_day` derived from `interview_time`. 
- `reschedule_ratio = past_reschedules / (scheduled_day_gap + 1)`. 
- `salary_per_year_exp = expected_salary / (experience + 1)`. 
- `notice_gap_diff = notice_period - scheduled_day_gap`. 

---

## 🧱 Tech Stack [web:21]

- **Language**: Python 3.x. 
- **ML / Data**: pandas, NumPy, scikit-learn, SciPy. 
- **Modeling**: RandomForestClassifier (with option to extend to XGBoost or other GBMs). 
- **API**: Flask for RESTful deployment. 
- **Serialization**: joblib for saving and loading the full pipeline. 

---

## 📁 Project Structure 

A suggested minimal structure for the repository is shown below. 

.
├── interview_no_show.csv
│             # Raw dataset
├── interview_no_show_pipeline.pkl
│   
├── 01_eda_and_modeling.ipynb
│   
├── templates
│   ├── form.html              # Training script to build pipeline
│   └── styles.css                       # Helper functions
└── app.py
                          # Streamlit/FastAPI/Flask app


