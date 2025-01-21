from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Initialize FastAPI app
app = FastAPI(title="Heart Disease Prediction Tool",
              description="This tool allows users to input health parameters and predict the likelihood of heart disease using a trained machine learning model.",
              version="2.0")

# Load pre-trained models and encoders
numerical_imputer = joblib.load("numerical_imputer.pkl")
categorical_imputer = joblib.load("categorical_imputer.pkl")
label_encoders = joblib.load("label_encoders.pkl")
one_hot_encoder = joblib.load("one_hot_encoder.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
best_model = joblib.load("best_heart_disease_model.pkl")


@app.get("/", response_class=HTMLResponse)
def read_root():
    return """<!DOCTYPE html>
    <html>
    <head>
        <title>Heart Disease Prediction Tool</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f9f9f9;
                margin: 0;
                padding: 0;
            }
            .container {
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background-color: #ffffff;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                padding-top: 40px;
            }
            h1 {
                text-align: center;
                color: #333333;
            }
            .form-section {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin-bottom: 20px;
            }
            .form-section label {
                font-weight: bold;
            }
            .form-section div {
                padding-bottom: 20px;
            }
            input, select {
                width: 100%;
                padding: 10px;
                margin-top: 5px;
                border-radius: 5px;
                border: 1px solid #ddd;
            }
            button {
                display: block;
                margin: 20px auto;
                padding: 10px 20px;
                background-color: #007BFF;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
            }
            button:hover {
                background-color: #0056b3;
            }
            .footer {
                text-align: center;
                margin-top: 20px;
                color: #666666;
            }
            .info-popup {
                background-color: #f1f1f1;
                border: 1px solid #ccc;
                border-radius: 5px;
                padding: 15px;
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                width: 400px;
                display: none;
            }
            .info-popup button {
                background-color: #FF6347;
                color: white;
                padding: 10px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                display: block;
                margin-top: 20px;
            }
            .info-popup button:hover {
                background-color: #FF4500;
            }
            #infoBtn {
                background-color: #32CD32;
                color: white;
            }
            #infoBtn:hover {
                background-color: #228B22;
            }
            .logo {
                width: 150px;
                display: block;
                margin: 0 auto;
            }
            .image-container {
                text-align: center;
                margin-top: 20px;
            }
            .image-container img {
                width: 50%;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
        </style>
        <script>
            function showPopup() {
                document.getElementById('infoPopup').style.display = 'block';
            }
            function closePopup() {
                document.getElementById('infoPopup').style.display = 'none';
            }
        </script>
    </head>
    <body>
        <div class="container">
            <h1>Heart Disease Prediction Tool</h1>
            <p>Welcome! Please fill in the form below to predict the likelihood of heart disease.</p>

            <!-- Image below the welcome message -->
            <div class="image-container">
                <img src="/static/Heart_01.jpeg" alt="Heart Disease Image" class="logo"> <!-- Image below welcome message -->
            </div>

            <button id="infoBtn" onclick="showPopup()">Click for Detailed Information</button>
            <div id="infoPopup" class="info-popup">
                <h2>About This API</h2>
                <p>This API uses a trained machine learning model to predict the likelihood of heart disease based on various health parameters. Input data is processed, encoded, scaled, and then passed through the model to generate a prediction.</p>
                <button onclick="closePopup()">Close</button>
            </div>

            <form action="/predict" method="post">
                <div class="form-section">
                    <div>
                        <label for="Age">Age:</label><br>
                        <input type="number" id="Age" name="Age" required>
                    </div>
                    <div>
                        <label for="Sex">Sex:</label><br>
                        <select id="Sex" name="Sex" required>
                            <option value="Male">Male</option>
                            <option value="Female">Female</option>
                        </select>
                    </div>
                    <div>
                        <label for="ChestPainType">Chest Pain Type:</label><br>
                        <select id="ChestPainType" name="ChestPainType" required>
                            <option value="TA">Typical Angina (TA)</option>
                            <option value="ATA">Atypical Angina (ATA)</option>
                            <option value="NAP">Non-Anginal Pain (NAP)</option>
                            <option value="ASY">Asymptomatic (ASY)</option>
                        </select>
                    </div>
                    <div>
                        <label for="RestingBP">Resting Blood Pressure (mm Hg):</label><br>
                        <input type="number" id="RestingBP" name="RestingBP" required>
                    </div>
                    <div>
                        <label for="Cholesterol">Cholesterol (mg/dl):</label><br>
                        <input type="number" id="Cholesterol" name="Cholesterol" required>
                    </div>
                    <div>
                        <label for="FastingBS">Fasting Blood Sugar > 120 mg/dl:</label><br>
                        <select id="FastingBS" name="FastingBS" required>
                            <option value="1">Yes</option>
                            <option value="0">No</option>
                        </select>
                    </div>
                    <div>
                        <label for="RestingECG">Resting ECG Results:</label><br>
                        <select id="RestingECG" name="RestingECG" required>
                            <option value="Normal">Normal</option>
                            <option value="ST">ST Abnormality</option>
                            <option value="LVH">Left Ventricular Hypertrophy (LVH)</option>
                        </select>
                    </div>
                    <div>
                        <label for="MaxHR">Maximum Heart Rate:</label><br>
                        <input type="number" id="MaxHR" name="MaxHR" required>
                    </div>
                    <div>
                        <label for="ExerciseAngina">Exercise-Induced Angina:</label><br>
                        <select id="ExerciseAngina" name="ExerciseAngina" required>
                            <option value="Y">Yes</option>
                            <option value="N">No</option>
                        </select>
                    </div>
                    <div>
                        <label for="Oldpeak">ST Depression (Oldpeak):</label><br>
                        <input type="number" step="0.1" id="Oldpeak" name="Oldpeak" required>
                    </div>
                    <div>
                        <label for="ST_Slope">ST Slope:</label><br>
                        <select id="ST_Slope" name="ST_Slope" required>
                            <option value="Up">Upsloping</option>
                            <option value="Flat">Flat</option>
                            <option value="Down">Downsloping</option>
                        </select>
                    </div>
                </div>
                <button type="submit">Predict</button>
                <button type="button" onclick="window.print()">Download PDF</button> <!-- For PDF download -->
            </form>

            <div class="footer">
                <p>Developed by Rod Will</p>
            </div>
        </div>
    </body>
    </html>"""

@app.post("/predict", response_class=HTMLResponse)
def predict(
    Age: float = Form(...),
    Sex: str = Form(...),
    ChestPainType: str = Form(...),
    RestingBP: float = Form(...),
    Cholesterol: float = Form(...),
    FastingBS: int = Form(...),
    RestingECG: str = Form(...),
    MaxHR: float = Form(...),
    ExerciseAngina: str = Form(...),
    Oldpeak: float = Form(...),
    ST_Slope: str = Form(...)
):
    try:
        # Create input data
        input_data = {
            "Age": Age,
            "Sex": Sex,
            "ChestPainType": ChestPainType,
            "RestingBP": RestingBP,
            "Cholesterol": Cholesterol,
            "FastingBS": FastingBS,
            "RestingECG": RestingECG,
            "MaxHR": MaxHR,
            "ExerciseAngina": ExerciseAngina,
            "Oldpeak": Oldpeak,
            "ST_Slope": ST_Slope
        }

        input_df = pd.DataFrame([input_data])

        # Handle missing values
        input_df[numerical_imputer.feature_names_in_] = numerical_imputer.transform(
            input_df[numerical_imputer.feature_names_in_]
        )
        input_df[categorical_imputer.feature_names_in_] = categorical_imputer.transform(
            input_df[categorical_imputer.feature_names_in_]
        )

        # Encode categorical features
        for col, le in label_encoders.items():
            input_df[col] = le.transform(input_df[col])

        # Apply one-hot encoding
        one_hot_encoded = one_hot_encoder.transform(input_df[["ChestPainType", "RestingECG"]]).toarray()
        one_hot_encoded_df = pd.DataFrame(
            one_hot_encoded, columns=one_hot_encoder.get_feature_names_out(["ChestPainType", "RestingECG"])
        )
        input_df = pd.concat([input_df.drop(["ChestPainType", "RestingECG"], axis=1), one_hot_encoded_df], axis=1)

        # Feature scaling
        scaled_features = scaler.transform(input_df)

        # Dimensionality reduction
        reduced_features = pca.transform(scaled_features)
        input_df[["AutoEnc1", "AutoEnc2"]] = reduced_features

        # Prediction
        probability = best_model.predict_proba(input_df)[:, 1][0]
        prediction = int(probability > 0.5)

        result_message = "The patient is at high risk of heart disease." if prediction == 1 else "The patient is not at risk of heart disease."

        return f"""<!DOCTYPE html>
        <html>
        <head>
            <title>Prediction Result</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background-color: #f9f9f9;
                    margin: 0;
                    padding: 0;
                }}
                .container {{
                    max-width: 800px;
                    margin: 50px auto;
                    padding: 20px;
                    background-color: #ffffff;
                    border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                }}
                h1 {{
                    text-align: center;
                    color: #333333;
                }}
                p {{
                    font-size: 16px;
                    line-height: 1.5;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 20px;
                    color: #666666;
                }}
                a {{
                    color: #007BFF;
                    text-decoration: none;
                }}
                a:hover {{
                    text-decoration: underline;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Prediction Result</h1>
                <p style="text-align:center; font-size: 1.2em; color: #333;">
                    <strong>Result:</strong> {{ result_message }}
                </p>
                <p style="text-align:center; font-size: 1em; color: #555;">
                    <strong>Confidence Score:</strong> {{ probability * 100:.2f }}%
                </p>
                <button onclick="window.history.back()" style="display: block; margin: 20px auto;">
                    Go Back
                </button>
            </div>
        </body>
        </html>"""

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during prediction: {str(e)}")

# For local testing (uncomment the below line to run locally)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
