# diabetes_prediction_nb_gui.py

# === Import Required Libraries ===
import tkinter as tk
from tkinter import messagebox
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import joblib
import os

# === Load Dataset from Kaggle CSV ===
try:
    # Try loading with comma delimiter
    df = pd.read_csv("diabetes.csv")

    # If there's only one column, try tab-delimited
    if df.shape[1] == 1:
        df = pd.read_csv("diabetes.csv", delimiter="\t")

    # Clean column names
    df.columns = df.columns.str.strip()

    expected_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

    # Check and reassign columns only if shape matches
    if list(df.columns) != expected_columns:
        if df.shape[1] == len(expected_columns):
            df.columns = expected_columns
        else:
            raise ValueError(f"Expected {len(expected_columns)} columns, but got {df.shape[1]}. Columns: {df.columns.tolist()}")

    print("Loaded Columns:", df.columns.tolist())

except FileNotFoundError:
    messagebox.showerror("Error", "Missing 'diabetes.csv' file. Please download it from Kaggle and place it in the project folder.")
    exit()
except Exception as e:
    messagebox.showerror("Error", f"Error loading dataset: {e}")
    exit()

# === Data Preprocessing ===
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Train Model ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = GaussianNB()
model.fit(X_train, y_train)

# === Save model and scaler ===
joblib.dump(model, "naive_bayes_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# === GUI Application ===
class DiabetesApp:
    def __init__(self, master):
        self.master = master
        master.title("Diabetes Prediction - Naive Bayes")

        self.labels = [
            ("Pregnancies", "(0-20)"),
            ("Glucose", "(70-200)"),
            ("BloodPressure", "(40-122)"),
            ("SkinThickness", "(10-100)"),
            ("Insulin", "(0-846)"),
            ("BMI", "(10-70)"),
            ("DiabetesPedigreeFunction", "(0.1-2.5)"),
            ("Age", "(10-100)")
        ]
        self.entries = []

        for i, (label_text, hint) in enumerate(self.labels):
            label = tk.Label(master, text=f"{label_text} {hint}")
            label.grid(row=i, column=0, padx=10, pady=5, sticky='e')
            entry = tk.Entry(master)
            entry.grid(row=i, column=1, padx=10, pady=5)
            self.entries.append((entry, label_text))

        self.predict_button = tk.Button(master, text="Predict", command=self.predict)
        self.predict_button.grid(row=len(self.labels), column=0, columnspan=2, pady=10)

    def predict(self):
        try:
            input_data = []
            for entry, label_text in self.entries:
                value = entry.get()
                if not value.strip():
                    raise ValueError(f"{label_text} cannot be empty")
                val = float(value)
                if label_text == "Age" and val <= 0:
                    raise ValueError("Enter a valid age greater than 0")
                input_data.append(val)

            scaler = joblib.load("scaler.pkl")
            model = joblib.load("naive_bayes_model.pkl")
            scaled_input = scaler.transform([input_data])
            prediction = model.predict(scaled_input)
            result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
            messagebox.showinfo("Prediction Result", f"The patient is: {result}")

        except Exception as e:
            messagebox.showerror("Input Error", str(e))

# === Run App ===
if __name__ == "__main__":
    root = tk.Tk()
    app = DiabetesApp(root)
    root.mainloop()
