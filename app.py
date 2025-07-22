from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from preprocess import preprocess_text
import numpy as np
import tensorflow as tf

# Load model
model = load_model("model_klasifikasi_gejala_v3.keras")

# Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return "AI Disease Classifier is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing text field'}), 400

    user_input = data['text']
    clean_input = preprocess_text(user_input)

    # Ubah sesuai input model kamu (contoh jika input model langsung berupa teks)
    result = model.predict([clean_input])
    predicted_label = tf.argmax(result, axis=1).numpy()[0]

    label_map = {
        0: "Dengue Fever",
        1: "Asthma",
        2: "Chickenpox",
        3: "Diabetes",
        4: "Varicose Veins",
        5: "Common Cold",
        6: "Malaria",
        7: "Psoriasis",
        8: "Gastroesophageal Reflux Disease",
        9: "Arthritis",
        10: "Impetigo",
        11: "Hypertension",
        12: "Pneumonia",
        13: "Cervical Spondylosis",
        14: "Typhoid",
        15: "Peptic Ulcer Disease",
        16: "Allergy",
        17: "Drug Reaction",
        18: "Urinary Tract Infection",
        19: "Migraine"
    }

    disease_info = {
        "Dengue Fever": {
            "desc": "A viral infection caused by Aedes mosquito bites, with high fever and muscle pain.",
            "solution": "See a doctor for blood tests and fluid monitoring."
        },
        "Asthma": {
            "desc": "A chronic airway disease causing shortness of breath and coughing.",
            "solution": "Consult a doctor for medication and trigger control."
        },
        "Chickenpox": {
            "desc": "A viral infection causing itchy, fluid-filled rashes.",
            "solution": "Visit a doctor for symptom relief and isolation guidance."
        },
        "Diabetes": {
            "desc": "A metabolic disorder with high blood sugar levels.",
            "solution": "Get medical advice for blood sugar control and diet planning."
        },
        "Varicose Veins": {
            "desc": "Swollen and visible veins usually in the legs.",
            "solution": "Consult a doctor for circulation evaluation and treatment."
        },
        "Common Cold": {
            "desc": "A mild viral infection with cough and runny nose.",
            "solution": "Rest and see a doctor if symptoms worsen."
        },
        "Malaria": {
            "desc": "A parasitic infection from Anopheles mosquito bites.",
            "solution": "Visit a doctor promptly for antimalarial treatment."
        },
        "Psoriasis": {
            "desc": "An autoimmune skin condition with scaly patches.",
            "solution": "See a dermatologist for long-term treatment."
        },
        "Gastroesophageal Reflux Disease": {
            "desc": "Stomach acid flows back into the esophagus, causing chest pain.",
            "solution": "Consult a doctor for acid reducers and lifestyle advice."
        },
        "Arthritis": {
            "desc": "Joint inflammation causing pain and stiffness.",
            "solution": "Seek medical evaluation for diagnosis and physical therapy."
        },
        "Impetigo": {
            "desc": "A contagious skin infection with blisters and crusts.",
            "solution": "Visit a doctor for antibiotics."
        },
        "Hypertension": {
            "desc": "High blood pressure that can lead to serious complications.",
            "solution": "Monitor regularly and consult a doctor for medication."
        },
        "Pneumonia": {
            "desc": "A lung infection with productive cough and fever.",
            "solution": "See a doctor for antibiotics and chest X-ray."
        },
        "Cervical Spondylosis": {
            "desc": "Wear and tear in the neck spine causing pain and stiffness.",
            "solution": "Consult a doctor for assessment and physical therapy."
        },
        "Typhoid": {
            "desc": "A bacterial infection from contaminated food or water.",
            "solution": "Seek medical care for antibiotics and hydration."
        },
        "Peptic Ulcer Disease": {
            "desc": "Sores in the stomach or intestines due to acid.",
            "solution": "Consult a doctor for medication and dietary changes."
        },
        "Allergy": {
            "desc": "Immune reaction to foreign substances.",
            "solution": "See a doctor to identify allergens and receive treatment."
        },
        "Drug Reaction": {
            "desc": "Adverse effects due to medication.",
            "solution": "Stop the drug and immediately consult a healthcare provider."
        },
        "Urinary Tract Infection": {
            "desc": "Infection in the urinary system causing painful urination.",
            "solution": "Visit a doctor for antibiotics and evaluation."
        },
        "Migraine": {
            "desc": "Throbbing headache often with nausea or light sensitivity.",
            "solution": "Consult a doctor for medication and stress management."
        }
    }

    label = label_map.get(predicted_label, "Unknown")
    disease_detail = disease_info.get(label, {
        "desc": "No description available.",
        "solution": "Please consult a healthcare provider."
    })

    return jsonify({
        "disease": label,
        "desc": disease_detail["desc"],
        "solution": disease_detail["solution"]
    })
