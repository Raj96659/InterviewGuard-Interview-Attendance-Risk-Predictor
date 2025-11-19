from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("no_show_pipeline.joblib")

@app.route("/")
def home():
    return render_template("form.html")

@app.route("/predict_form", methods=["POST"])
def predict_form():

    data = {
        "experience": float(request.form["experience"]),
        "notice_period": float(request.form["notice_period"]),
        "communication_score": float(request.form["communication_score"]),
        "expected_salary": float(request.form["expected_salary"]),
        "scheduled_day_gap": float(request.form["scheduled_day_gap"]),
        "past_reschedules": float(request.form["past_reschedules"]),
        "job_role": request.form["job_role"],
        "candidate_location": request.form["candidate_location"],
        "current_company_size": request.form["current_company_size"],
        "time_of_day": request.form["time_of_day"]
    }

    X = pd.DataFrame([data])

    proba = model.predict_proba(X)[:, 1][0]
    pred = model.predict(X)[0]

    return render_template(
        "form.html",
        prediction=int(pred),
        probability=round(float(proba), 3)
    )

if __name__ == "__main__":
    app.run(debug=True)
