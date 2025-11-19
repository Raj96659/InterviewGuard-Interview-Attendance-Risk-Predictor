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


---

## 🧱 Tech Stack 

- **Language**: Python 3.x. 
- **ML / Data**: pandas, NumPy, scikit-learn, SciPy. 
- **Modeling**: RandomForestClassifier (with option to extend to XGBoost or other GBMs). 
- **API**: Flask for RESTful deployment. 
- **Serialization**: joblib for saving and loading the full pipeline. 

---

## 🖼️ Screenshots [web:31]

Below is a quick preview of the Flask web interface for the Recruitment Interview No-Show Prediction system.  

![App Screenshot](https://github.com/Raj96659/InterviewGuard-Interview-Attendance-Risk-Predictor/blob/main/Screenshot%202025-11-19%20155500.png) <!-- adjust path/name -->  


## 📈 Model Performance 

The final tuned model was evaluated on a held-out test set using multiple classification metrics that are better suited than accuracy for potentially imbalanced no-show data.  

- **F1 score**: 95% — harmonic mean of precision and recall, indicating a strong balance between the two.  
- **Recall**: 98% — the model captures almost all true no-shows, minimizing false negatives. 
- **Precision**: 96% — most of the candidates flagged as likely no-shows are indeed no-shows, keeping false positives low.

These results suggest that the model is highly effective for operational use, especially in scenarios where missing a no-show (low recall) would be more costly than occasionally over-flagging a candidate (precision). 

