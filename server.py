from flask import Flask, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 

# Load model and label encoder
print("Loading model and label encoder...")
model = load_model("hasta_mudra_classifier.h5")
le = joblib.load("label_encoder.pkl")
print("Model and label encoder loaded successfully!")

def predict_mudra(image_path):
    print(f"Starting prediction for: {image_path}")
    img = load_img(image_path, color_mode='grayscale', target_size=(48, 48))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    print("Running model prediction...")
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_label = le.inverse_transform([predicted_class_index])[0]
    confidence = np.max(predictions) * 100
    
    print(f"Prediction complete: {predicted_label} with {confidence:.2f}% confidence")
    
    # Convert numpy types to Python native types for JSON serialization
    return str(predicted_label), float(confidence)

@app.route('/')
def home():
    print("Home route accessed")
    return "Flask server is running!"

@app.route('/predict', methods=['POST'])
def predict():
    print("\n" + "="*50)
    print("PREDICT ENDPOINT CALLED")
    print("="*50)
    
    print(f"Request method: {request.method}")
    print(f"Request files: {request.files}")
    print(f"Request form: {request.form}")
    
    if 'file' not in request.files:
        print("ERROR: No file in request")
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    print(f"File received: {file.filename}")
    print(f"File content type: {file.content_type}")
    
    if file.filename == '':
        print("ERROR: Empty filename")
        return jsonify({'error': 'No file selected'}), 400
    
    # Save file temporarily
    filepath = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    print(f"Saving file to: {filepath}")
    file.save(filepath)
    print(f"File saved successfully. Size: {os.path.getsize(filepath)} bytes")
    
    try:
        # Get prediction
        print("Calling predict_mudra function...")
        label, confidence = predict_mudra(filepath)
        
        # Clean up
        print(f"Removing temporary file: {filepath}")
        os.remove(filepath)
        
        result = {
            'mudra': label,
            'confidence': round(confidence, 2)
        }
        print(f"Returning result: {result}")
        print("="*50 + "\n")
        
        return jsonify(result)
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(filepath):
            os.remove(filepath)
        error_msg = f'Prediction failed: {str(e)}'
        print(f"ERROR: {error_msg}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        print("="*50 + "\n")
        return jsonify({'error': error_msg}), 500

if __name__ == "__main__":
    print("\n" + "="*50)
    print("STARTING FLASK SERVER")
    print("="*50)
    print("Server running on http://127.0.0.1:5000")
    print("Press CTRL+C to quit")
    print("="*50 + "\n")
    app.run(debug=True)