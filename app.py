from flask import Flask, render_template, request, jsonify
import numpy as np
import PIL.Image
import tensorflow as tf
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the trained model
MODEL_PATH = "skin_disease_classifier_mobilenetv2_fixed.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Class mapping
Str_to_Int = {
    'Actinic keratosis': 0,
    'Atopic Dermatitis': 1,
    'Benign keratosis': 2,
    'Dermatofibroma': 3,
    'Melanocytic nevus': 4,
    'Melanoma': 5,
    'Squamous cell carcinoma': 6,
    'Tinea Ringworm Candidiasis': 7,
    'Vascular lesion': 8
}

class_labels = list(Str_to_Int.keys())

def preprocess_image(image_path):
    img = PIL.Image.open(image_path).convert('RGB')
    img = img.resize((240, 240))
    img = np.asarray(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    
    temp_path = "temp.jpg"
    file.save(temp_path)

    img = preprocess_image(temp_path)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    os.remove(temp_path)

    return jsonify({
        "prediction": class_labels[predicted_class],
        "confidence": f"{confidence:.2f}"
    })

if __name__ == "__main__":
    app.run(debug=True)
