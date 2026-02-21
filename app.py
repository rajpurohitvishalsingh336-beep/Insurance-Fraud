from flask import Flask, render_template, request, redirect, session
import pandas as pd
import pickle
import os
import matplotlib

# IMPORTANT for Flask + Matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = "advanced_secret"

ADMIN_PASSWORD = "1234"

# ---------------- SAFETY SETUP ---------------- #

# Create static folder automatically
if not os.path.exists("static"):
    os.makedirs("static")

# Create insurance.csv if not exists
if not os.path.exists("insurance.csv"):
    df_init = pd.DataFrame(columns=[
        "age",
        "months_as_customer",
        "policy_annual_premium",
        "total_claim_amount",
        "ML_Prediction"
    ])
    df_init.to_csv("insurance.csv", index=False)

# Load trained model
model = pickle.load(open("fraud_model.pkl", "rb"))

# ---------------- LOGIN ---------------- #
@app.route('/')
def login():
    return render_template("login.html")

@app.route('/login', methods=['POST'])
def check_login():
    role = request.form.get('role')
    password = request.form.get('password')

    if role == "admin":
        if password == ADMIN_PASSWORD:
            session['role'] = "admin"
            return redirect('/admin')
        else:
            return "Wrong Admin Password!"
    else:
        session['role'] = "user"
        return redirect('/user')

# ---------------- USER ---------------- #
@app.route('/user')
def user_dashboard():
    return render_template("user_dashboard.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = {
        "age": int(request.form['age']),
        "months_as_customer": int(request.form['months']),
        "policy_annual_premium": float(request.form['premium']),
        "total_claim_amount": float(request.form['claim'])
    }

    df = pd.DataFrame([data])

    # ML Prediction
    prediction = model.predict(df)
    result = "Fraud" if prediction[0] == 1 else "Genuine"

    df["ML_Prediction"] = result

    # Append to CSV safely
    old = pd.read_csv("insurance.csv")
    final_df = pd.concat([old, df], ignore_index=True)
    final_df.to_csv("insurance.csv", index=False)

    return render_template("user_dashboard.html", result=result)

# ---------------- ADMIN ---------------- #
@app.route('/admin')
def admin_dashboard():
    df = pd.read_csv("insurance.csv")

    if df.empty:
        return "No data submitted yet."

    counts = df['ML_Prediction'].value_counts()

    fraud = counts.get("Fraud", 0)
    genuine = counts.get("Genuine", 0)
    total = len(df)

    # -------- BAR CHART --------
    plt.figure(figsize=(5, 4))
    counts.plot(kind='bar', color=['red', 'green'])
    plt.title("Fraud vs Genuine")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("static/bar.png")
    plt.close()

    # -------- PIE CHART --------
    plt.figure(figsize=(5, 4))
    counts.plot.pie(
        autopct="%1.1f%%",
        colors=['red', 'green'],
        startangle=90
    )
    plt.title("Fraud Distribution")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig("static/pie.png")
    plt.close()

    table = df.to_html(classes="table table-striped", index=False)

    return render_template(
        "admin_dashboard.html",
        table=table,
        fraud=fraud,
        genuine=genuine,
        total=total
    )

if __name__ == "__main__":
    app.run(debug=True)