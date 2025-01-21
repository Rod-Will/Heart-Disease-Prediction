from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.scrollview import ScrollView
from kivy.uix.popup import Popup
from kivy.core.window import Window
from kivy.config import Config
import joblib
import pandas as pd
from fpdf import FPDF

# Load your trained models
numerical_imputer = joblib.load("numerical_imputer.pkl")
categorical_imputer = joblib.load("categorical_imputer.pkl")
label_encoders = joblib.load("label_encoders.pkl")
one_hot_encoder = joblib.load("one_hot_encoder.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
best_model = joblib.load("best_heart_disease_model.pkl")

# Change the default Kivy icon
Window.set_icon("logo.png")
Window.size = (800, 600)  # Set default window size

class PredictionApp(App):
    def build(self):
        self.title = "Heart Disease Predictor"

        # Main layout
        self.layout = BoxLayout(orientation="vertical", padding=10, spacing=10)

        # Scrollable section for input fields
        scrollview = ScrollView(size_hint=(1, 0.7))
        form_layout = BoxLayout(orientation="vertical", size_hint_y=None, spacing=10)
        form_layout.bind(minimum_height=form_layout.setter('height'))
        
        self.fields = {}
        inputs = [
            "Age", "Sex (Male/Female)", "ChestPainType (TA/ATA/NAP/ASY)", 
            "RestingBP", "Cholesterol", "FastingBS (1/0)", 
            "RestingECG (Normal/ST/LVH)", "MaxHR", 
            "ExerciseAngina (Y/N)", "Oldpeak", "ST_Slope (Up/Flat/Down)"
        ]
        for i, field in enumerate(inputs):
            self.add_input_field(form_layout, field)

        scrollview.add_widget(form_layout)
        self.layout.add_widget(scrollview)

        # Buttons for predict and save PDF
        button_layout = BoxLayout(orientation="horizontal", size_hint=(1, 0.1), spacing=10)
        predict_btn = Button(text="Predict", background_color=(0.2, 0.6, 0.8, 1))
        predict_btn.bind(on_press=self.predict)
        button_layout.add_widget(predict_btn)

        save_pdf_btn = Button(text="Save Prediction as PDF", background_color=(0.2, 0.8, 0.2, 1))
        save_pdf_btn.bind(on_press=self.save_pdf)
        button_layout.add_widget(save_pdf_btn)

        self.layout.add_widget(button_layout)

        # About and How-To-Use section
        info_layout = BoxLayout(orientation="horizontal", size_hint=(1, 0.1), spacing=10)
        about_btn = Button(text="About", background_color=(0.8, 0.5, 0.2, 1))
        about_btn.bind(on_press=self.show_about)
        info_layout.add_widget(about_btn)

        how_to_btn = Button(text="How to Use", background_color=(0.8, 0.2, 0.2, 1))
        how_to_btn.bind(on_press=self.show_how_to_use)
        info_layout.add_widget(how_to_btn)

        self.layout.add_widget(info_layout)

        return self.layout

    def add_input_field(self, layout, field_name):
        # Add input field in a horizontal box layout
        box = BoxLayout(orientation="horizontal", size_hint_y=None, height=40, spacing=10)
        box.add_widget(Label(text=field_name, size_hint_x=0.4))
        input_box = TextInput(multiline=False, size_hint_x=0.6)
        box.add_widget(input_box)
        layout.add_widget(box)
        self.fields[field_name] = input_box

    def predict(self, instance):
        # Gather inputs
        try:
            inputs = {field: self.fields[field].text for field in self.fields}

            # Prepare data for prediction
            input_data = {
                "Age": float(inputs["Age"]),
                "Sex": inputs["Sex (Male/Female)"],
                "ChestPainType": inputs["ChestPainType (TA/ATA/NAP/ASY)"],
                "RestingBP": float(inputs["RestingBP"]),
                "Cholesterol": float(inputs["Cholesterol"]),
                "FastingBS": int(inputs["FastingBS (1/0)"]),
                "RestingECG": inputs["RestingECG (Normal/ST/LVH)"],
                "MaxHR": float(inputs["MaxHR"]),
                "ExerciseAngina": inputs["ExerciseAngina (Y/N)"],
                "Oldpeak": float(inputs["Oldpeak"]),
                "ST_Slope": inputs["ST_Slope (Up/Flat/Down)"]
            }
            input_df = pd.DataFrame([input_data])

            # Preprocessing pipeline
            input_df[numerical_imputer.feature_names_in_] = numerical_imputer.transform(
                input_df[numerical_imputer.feature_names_in_]
            )
            input_df[categorical_imputer.feature_names_in_] = categorical_imputer.transform(
                input_df[categorical_imputer.feature_names_in_]
            )
            for col, le in label_encoders.items():
                input_df[col] = le.transform(input_df[col])
            one_hot_encoded = one_hot_encoder.transform(input_df[["ChestPainType", "RestingECG"]]).toarray()
            one_hot_encoded_df = pd.DataFrame(
                one_hot_encoded, columns=one_hot_encoder.get_feature_names_out(["ChestPainType", "RestingECG"])
            )
            input_df = pd.concat([input_df.drop(["ChestPainType", "RestingECG"], axis=1), one_hot_encoded_df], axis=1)
            scaled_features = scaler.transform(input_df)
            reduced_features = pca.transform(scaled_features)
            input_df[["AutoEnc1", "AutoEnc2"]] = reduced_features

            # Make prediction
            probability = best_model.predict_proba(input_df)[:, 1][0]
            prediction = "High Risk of Heart Disease" if probability > 0.5 else "Low Risk of Heart Disease"

            # Show result
            popup = Popup(
                title="Prediction Result",
                content=Label(
                    text=f"Prediction: {prediction}\nProbability: {probability:.2f}",
                    size_hint=(None, None),
                    size=(400, 200),
                ),
                size_hint=(None, None),
                size=(400, 200),
            )
            popup.open()

        except Exception as e:
            popup = Popup(
                title="Error",
                content=Label(
                    text=f"Error during prediction: {str(e)}",
                    size_hint=(None, None),
                    size=(400, 200),
                ),
                size_hint=(None, None),
                size=(400, 200),
            )
            popup.open()

    def save_pdf(self, instance):
        # Save inputs and prediction as PDF
        try:
            inputs = {field: self.fields[field].text for field in self.fields}
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Heart Disease Prediction Report", ln=True, align="C")
            pdf.ln(10)
            for key, value in inputs.items():
                pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)
            pdf.output("Prediction_Report.pdf")

            popup = Popup(
                title="PDF Saved",
                content=Label(text="PDF saved as 'Prediction_Report.pdf'", size_hint=(None, None), size=(400, 200)),
                size_hint=(None, None),
                size=(400, 200),
            )
            popup.open()

        except Exception as e:
            popup = Popup(
                title="Error",
                content=Label(text=f"Error saving PDF: {str(e)}", size_hint=(None, None), size=(400, 200)),
                size_hint=(None, None),
                size=(400, 200),
            )
            popup.open()

    def show_about(self, instance):
        popup = Popup(
            title="About",
            content=Label(
                text="Heart Disease Predictor\nVersion 1.0\nDeveloped to assess heart disease risk.",
                size_hint=(None, None),
                size=(400, 200),
            ),
            size_hint=(None, None),
            size=(400, 200),
        )
        popup.open()

    def show_how_to_use(self, instance):
        popup = Popup(
            title="How to Use",
            content=Label(
                text="1. Enter patient details in the fields.\n2. Click 'Predict' to get a risk assessment.\n3. Save the result as a PDF if needed.",
                size_hint=(None, None),
                size=(400, 200),
            ),
            size_hint=(None, None),
            size=(400, 200),
        )
        popup.open()

if __name__ == "__main__":
    PredictionApp().run()
