from flask import Flask, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
from flask_cors import CORS
import traceback

app = Flask(__name__)

# SIMPLIFIED CORS - Allow everything (most reliable for debugging)
CORS(app)

# Set max file size to 16MB
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Global variables for model and encoder
model = None
le = None
MODELS_LOADED = False

# Load model and label encoder with error handling
print("\n" + "="*60)
print("INITIALIZING FLASK SERVER")
print("="*60)
print("Loading model and label encoder...")

try:
    model_path = "hasta_mudra_classifier.h5"
    encoder_path = "label_encoder.pkl"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_path} not found in {os.getcwd()}")
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"{encoder_path} not found in {os.getcwd()}")
    
    print(f"Found model file: {model_path}")
    print(f"Found encoder file: {encoder_path}")
    
    model = load_model(model_path)
    le = joblib.load(encoder_path)
    
    print("✓ Model loaded successfully")
    print("✓ Label encoder loaded successfully")
    MODELS_LOADED = True
    
except Exception as e:
    print(f"✗ CRITICAL ERROR loading models: {e}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Files in directory: {os.listdir('.')}")
    traceback.print_exc()
    MODELS_LOADED = False

print("="*60 + "\n")

def predict_mudra(image_path):
    """Predict mudra from image path"""
    print(f"Starting prediction for: {image_path}")
    
    img = load_img(image_path, color_mode='grayscale', target_size=(48, 48))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    print(f"Image array shape: {img_array.shape}")
    print("Running model prediction...")
    
    predictions = model.predict(img_array, verbose=0)
    predicted_class_index = np.argmax(predictions)
    predicted_label = le.inverse_transform([predicted_class_index])[0]
    confidence = np.max(predictions) * 100
    
    print(f"Prediction: {predicted_label} ({confidence:.2f}%)")
    
    # Convert numpy types to Python native types
    return str(predicted_label), float(confidence)

@app.after_request
def after_request(response):
    """Add CORS headers to every response"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route('/')
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'message': 'Bharatnatyam Mudra Classifier API',
        'models_loaded': MODELS_LOADED,
        'version': '1.0',
        'endpoints': {
            'health': '/ (GET)',
            'predict': '/predict (POST)'
        }
    }), 200

@app.route('/health')
def health():
    """Detailed health check"""
    return jsonify({
        'status': 'healthy' if MODELS_LOADED else 'unhealthy',
        'models_loaded': MODELS_LOADED,
        'model_available': model is not None,
        'encoder_available': le is not None
    }), 200 if MODELS_LOADED else 503

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    """Predict mudra from uploaded image"""
    
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        return '', 204
    
    print("\n" + "="*60)
    print("PREDICT ENDPOINT CALLED")
    print("="*60)
    
    # Check if models are loaded
    if not MODELS_LOADED or model is None or le is None:
        print("✗ Models not loaded - server not ready")
        return jsonify({
            'error': 'Server not ready. Models failed to load at startup.'
        }), 503
    
    print(f"Request method: {request.method}")
    print(f"Request files: {list(request.files.keys())}")
    print(f"Content-Type: {request.content_type}")
    
    # Validate file in request
    if 'file' not in request.files:
        print("✗ No file in request")
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        print("✗ Empty filename")
        return jsonify({'error': 'No file selected'}), 400
    
    # Validate file type
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
    
    if file_ext not in allowed_extensions:
        print(f"✗ Invalid file type: {file.filename}")
        return jsonify({
            'error': f'Invalid file type. Allowed: {", ".join(allowed_extensions)}'
        }), 400
    
    print(f"✓ File received: {file.filename}")
    print(f"  Content type: {file.content_type}")
    
    # Create uploads directory
    os.makedirs("uploads", exist_ok=True)
    
    # Save file with unique name to avoid conflicts
    import uuid
    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    filepath = os.path.join("uploads", unique_filename)
    
    try:
        file.save(filepath)
        file_size = os.path.getsize(filepath)
        print(f"✓ File saved: {filepath} ({file_size:,} bytes)")
        
        # Get prediction
        print("→ Analyzing image...")
        label, confidence = predict_mudra(filepath)
        
        result = {
            'mudra': label,
            'label': label,
            'confidence': round(confidence, 2),
            'success': True
        }
        
        print(f"✓ Response: {result}")
        print("="*60 + "\n")
        
        return jsonify(result), 200
        
    except Exception as e:
        error_msg = f'Prediction failed: {str(e)}'
        print(f"✗ ERROR: {error_msg}")
        print(f"Error type: {type(e).__name__}")
        traceback.print_exc()
        print("="*60 + "\n")
        
        return jsonify({
            'error': error_msg,
            'success': False
        }), 500
        
    finally:
        # Clean up temporary file
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                print(f"  → Cleaned up: {filepath}")
            except Exception as cleanup_error:
                print(f"  ⚠ Cleanup failed: {cleanup_error}")

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({
        'error': 'File too large. Maximum size is 16MB.'
    }), 413

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    return jsonify({
        'error': 'Internal server error',
        'message': str(e)
    }), 500

if __name__ == "__main__":
    print("\n" + "="*60)
    print("STARTING FLASK SERVER")
    print("="*60)
    
    # Get port from environment (Render sets this)
    port = int(os.environ.get("PORT", 5000))
    
    print(f"Port: {port}")
    print(f"Host: 0.0.0.0")
    print(f"Models loaded: {MODELS_LOADED}")
    print("="*60 + "\n")
    
    # Run server
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        threaded=True
    )