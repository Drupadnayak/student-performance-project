"""
Student Performance Prediction System - Flask Backend
Author: [Your Name]
Date: April 2026
"""

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Global variables for model and scaler
model = None
scaler = None

# ============================================================================
# LOAD MODEL & SCALER
# ============================================================================

def load_model_and_scaler():
    """
    Load the trained model and scaler from pickle files
    """
    global model, scaler
    
    try:
        model_path = 'models/model.pkl'
        scaler_path = 'models/scaler.pkl'
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        print("✓ Model and scaler loaded successfully")
        return True
    
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")
        return False


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_input(study_hours, attendance, previous_score):
    """
    Validate user input
    Returns: (is_valid, error_message)
    """
    errors = []
    
    try:
        study_hours = float(study_hours)
        attendance = float(attendance)
        previous_score = float(previous_score)
    except ValueError:
        return False, "All inputs must be numbers"
    
    if study_hours < 0 or study_hours > 24:
        errors.append("Study hours must be between 0 and 24")
    
    if attendance < 0 or attendance > 100:
        errors.append("Attendance must be between 0 and 100%")
    
    if previous_score < 0 or previous_score > 100:
        errors.append("Previous score must be between 0 and 100")
    
    if errors:
        return False, " | ".join(errors)
    
    return True, ""


# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_score(study_hours, attendance, previous_score):
    """
    Make prediction using the trained model
    
    Args:
        study_hours (float): Hours studied per week
        attendance (float): Attendance percentage
        previous_score (float): Previous exam score
    
    Returns:
        dict: Contains predicted score, status, and confidence
    """
    try:
        # Create input array and scale
        input_data = np.array([[study_hours, attendance, previous_score]])
        scaled_data = scaler.transform(input_data)
        
        # Make prediction
        predicted_score = model.predict(scaled_data)[0]
        
        # Ensure prediction is within valid range
        predicted_score = max(0, min(100, predicted_score))
        
        # Classify as Pass/Fail
        status = "PASS" if predicted_score >= 40 else "FAIL"
        
        # Calculate confidence level
        confidence = min(100, max(0, (predicted_score / 100) * 100))
        
        return {
            'predicted_score': round(predicted_score, 2),
            'status': status,
            'confidence': round(confidence, 2),
            'feedback': get_feedback(predicted_score, study_hours, attendance)
        }
    
    except Exception as e:
        return {
            'error': f"Prediction failed: {str(e)}"
        }


def get_feedback(score, study_hours, attendance):
    """
    Generate personalized feedback based on prediction
    """
    if score >= 80:
        return "Excellent performance expected! Keep up the great work!"
    elif score >= 60:
        return "Good performance expected. A few more study hours could help improve further."
    elif score >= 40:
        return "Passing expected, but increase study hours or attendance to improve."
    else:
        return "⚠ At risk of failing. Urgent attention needed - increase study hours and attendance."


# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def home():
    """
    Home page with prediction form
    """
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for predictions
    Accepts JSON with: study_hours, attendance, previous_score
    Returns: JSON with prediction result
    """
    try:
        # Get JSON data
        data = request.get_json()
        
        study_hours = data.get('study_hours')
        attendance = data.get('attendance')
        previous_score = data.get('previous_score')
        
        # Validate input
        is_valid, error_msg = validate_input(study_hours, attendance, previous_score)
        if not is_valid:
            return jsonify({'error': error_msg}), 400
        
        # Make prediction
        result = predict_score(float(study_hours), float(attendance), float(previous_score))
        
        if 'error' in result:
            return jsonify(result), 500
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({'error': f"Server error: {str(e)}"}), 500


@app.route('/info')
def info():
    """
    Provide information about the model
    """
    return jsonify({
        'model_name': 'Linear Regression',
        'features': ['Study Hours (0-24)', 'Attendance % (0-100)', 'Previous Score (0-100)'],
        'target': 'Final Score (0-100)',
        'pass_threshold': 40,
        'expected_accuracy': 'R² Score ~0.78'
    })


@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint
    """
    if model is None or scaler is None:
        return jsonify({'status': 'Model not loaded'}), 500
    return jsonify({'status': 'OK', 'model': 'ready'}), 200


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


# ============================================================================
# STARTUP
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("STUDENT PERFORMANCE PREDICTION SYSTEM")
    print("Flask Backend Server")
    print("="*60)
    
    # Load model and scaler
    if load_model_and_scaler():
        print("\n✓ All systems ready!")
        print("🚀 Starting Flask server...\n")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n✗ Failed to load model. Please train the model first:")
        print("   python train_model.py")
