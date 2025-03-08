import os
import cv2 # type: ignore
import numpy as np # type: ignore
import joblib # type: ignore
from flask import Flask, request, jsonify, render_template, send_from_directory, url_for # type: ignore
from werkzeug.utils import secure_filename # type: ignore
from datetime import datetime  # Import for timestamp fix

# Initialize Flask App
app = Flask(__name__)

# Load trained SVM model and scaler
svm_model = joblib.load("svm_cat_dog_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define Image Processing Function
IMG_SIZE = 64
app.config["UPLOAD_FOLDER"] = "static/uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Read, preprocess, and return a flattened image array."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.flatten()
    img = scaler.transform([img])  # Normalize the image
    return img

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded!"})

    file = request.files["file"]
    
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file format!"})

    # Ensure upload folder exists
    if not os.path.exists(app.config["UPLOAD_FOLDER"]):
        os.makedirs(app.config["UPLOAD_FOLDER"])

    # Save the uploaded file (secure filename)
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    # Convert the image to grayscale
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return jsonify({"error": "Invalid image!"})

    # Save the grayscale image (overwrite original)
    gray_filename = f"gray_{filename}"
    gray_file_path = os.path.join(app.config["UPLOAD_FOLDER"], gray_filename)
    cv2.imwrite(gray_file_path, img)  # Save grayscale version

    # Preprocess the grayscale image for prediction
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.flatten()
    img = scaler.transform([img])  # Normalize for model input

    # Make a prediction
    prediction = svm_model.predict(img)[0]
    label = "Dog 🐶" if prediction == 1 else "Cat 🐱"

    return render_template("result.html", 
                           image_url=url_for('uploaded_file', filename=gray_filename), 
                           prediction=label, 
                           timestamp=datetime.now().timestamp())

@app.route("/static/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory("static/uploads", filename)

if __name__ == "__main__":
    app.run(debug=True)
